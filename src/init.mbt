///|
/// Initialization scale `sqrt(mean(V) / k)` (sklearn-style).
/// For an all-zero input the mean is 0 and the scale falls back to 1 so the
/// initial factors stay strictly positive.
fn init_scale(v : Matrix, k : Int32) -> Result[Double, NMFError] {
  let mut sum = 0.0
  for x in v.data {
    sum = sum + x
  }
  if sum.is_nan() || sum.is_inf() {
    return Err(NumericalError("sum of V overflowed while computing init scale"))
  }
  let mean = sum / (v.rows * v.cols).to_double()
  let scale = if mean > 0.0 { (mean / k.to_double()).sqrt() } else { 1.0 }
  Ok(scale)
}

///|
/// Fills a buffer with strictly positive uniform entries times `scale`.
fn random_fill(buf : Matrix, scale : Double, rng : SplitMix64) -> Unit {
  for idx in 0..<buf.data.length() {
    buf.data[idx] = rng.next_double() * scale
  }
}

///|
/// Random initialization of both factors: W (n×k), H (k×d), strictly
/// positive, reproducible for a fixed seed.
fn random_init(
  v : Matrix,
  k : Int32,
  seed : UInt64,
) -> Result[(Matrix, Matrix), NMFError] {
  let scale = match init_scale(v, k) {
    Ok(s) => s
    Err(e) => return Err(e)
  }
  let rng = SplitMix64::new(seed)
  let w = zeros_buf(v.rows, k)
  let h = zeros_buf(k, v.cols)
  random_fill(w, scale, rng)
  random_fill(h, scale, rng)
  Ok((w, h))
}

///|
/// Random initialization of W only (m×k), used by `transform` where H is
/// fixed from training.
fn random_init_w(
  x : Matrix,
  k : Int32,
  seed : UInt64,
) -> Result[Matrix, NMFError] {
  let scale = match init_scale(x, k) {
    Ok(s) => s
    Err(e) => return Err(e)
  }
  let rng = SplitMix64::new(seed)
  let w = zeros_buf(x.rows, k)
  random_fill(w, scale, rng)
  Ok(w)
}

///|
/// Standard NNDSVD initialization (Boutsidis & Gallopoulos 2008), using the
/// private power-iteration + deflation SVD in `svd_internal.mbt`.
///
/// First triplet: `W[:,0] = sqrt(sigma0) * abs(u0)`,
/// `H[0,:] = sqrt(sigma0) * abs(v0)`. Later triplets choose the positive or
/// negative part pair with the larger norm product and scale W/H symmetrically
/// by `sqrt(sigma * ||u_part|| * ||v_part||)`, normalizing each part.
/// Degenerate components remain zero; no mean filling (NNDSVDa is T10). Start
/// vectors use a fixed internal seed, so NNDSVD is deterministic and
/// independent of `NMFConfig.seed`.
fn nndsvd_init(v : Matrix, k : Int32) -> Result[(Matrix, Matrix), NMFError] {
  let components = match svd_top_k(v, k) {
    Ok(c) => c
    Err(e) => return Err(e)
  }
  let w = zeros_buf(v.rows, k)
  let h = zeros_buf(k, v.cols)
  if k <= 0 {
    return Ok((w, h))
  }
  let first = components[0]
  if first.sigma > 0.0 {
    let scale = first.sigma.sqrt()
    if scale.is_nan() || scale.is_inf() {
      return Err(NumericalError("NNDSVD first-component scale is non-finite"))
    }
    for i in 0..<v.rows {
      w.data[i * k] = scale * first.u[i].abs()
    }
    for j in 0..<v.cols {
      h.data[j] = scale * first.v[j].abs()
    }
  }
  for r in 1..<k {
    let comp = components[r]
    if comp.sigma <= 0.0 {
      continue
    }
    let mut up_sq = 0.0
    let mut un_sq = 0.0
    for i in 0..<v.rows {
      let x = comp.u[i]
      if x > 0.0 {
        up_sq = up_sq + x * x
      } else {
        let y = -x
        un_sq = un_sq + y * y
      }
    }
    let mut vp_sq = 0.0
    let mut vn_sq = 0.0
    for j in 0..<v.cols {
      let x = comp.v[j]
      if x > 0.0 {
        vp_sq = vp_sq + x * x
      } else {
        let y = -x
        vn_sq = vn_sq + y * y
      }
    }
    let u_norm_pos = up_sq.sqrt()
    let u_norm_neg = un_sq.sqrt()
    let v_norm_pos = vp_sq.sqrt()
    let v_norm_neg = vn_sq.sqrt()
    let m_pos = u_norm_pos * v_norm_pos
    let m_neg = u_norm_neg * v_norm_neg
    let use_pos = m_pos >= m_neg
    let u_norm = if use_pos { u_norm_pos } else { u_norm_neg }
    let v_norm = if use_pos { v_norm_pos } else { v_norm_neg }
    if u_norm <= 0.0 || v_norm <= 0.0 {
      // Both parts of the selected combination are numerically absent; this
      // singular triplet cannot contribute a non-negative rank-one factor.
      continue
    }
    let scale = (comp.sigma * u_norm * v_norm).sqrt()
    if scale.is_nan() || scale.is_inf() {
      return Err(
        NumericalError("NNDSVD component scale is non-finite at index \{r}"),
      )
    }
    let w_scale = scale / u_norm
    let h_scale = scale / v_norm
    if w_scale.is_nan() ||
      w_scale.is_inf() ||
      h_scale.is_nan() ||
      h_scale.is_inf() {
      return Err(
        NumericalError("NNDSVD component factor is non-finite at index \{r}"),
      )
    }
    for i in 0..<v.rows {
      let x = comp.u[i]
      let part = if use_pos {
        if x > 0.0 {
          x
        } else {
          0.0
        }
      } else if x < 0.0 {
        -x
      } else {
        0.0
      }
      w.data[i * k + r] = w_scale * part
    }
    for j in 0..<v.cols {
      let x = comp.v[j]
      let part = if use_pos {
        if x > 0.0 {
          x
        } else {
          0.0
        }
      } else if x < 0.0 {
        -x
      } else {
        0.0
      }
      h.data[r * v.cols + j] = h_scale * part
    }
  }
  if !all_finite(w.data) || !all_finite(h.data) {
    return Err(NumericalError("NNDSVD factors became non-finite"))
  }
  Ok((w, h))
}
