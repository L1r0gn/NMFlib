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
/// Degenerate components remain zero; `nndsvda_init` adds the NNDSVDa zero
/// mean-filling step. Start vectors use a fixed internal seed, so NNDSVD is
/// deterministic and independent of `NMFConfig.seed`.
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

///|
/// Mean of a matrix computed after scaling by its largest absolute entry.
/// Every scaled term is in [-1, 1], so the intermediate sum cannot overflow;
/// the average is rescaled at the end. Used only as an overflow fallback.
fn scaled_mean(v : Matrix, max_abs : Double) -> Result[Double, NMFError] {
  let n = (v.rows * v.cols).to_double()
  let mut sum = 0.0
  for x in v.data {
    sum = sum + x / max_abs
  }
  if sum.is_nan() || sum.is_inf() {
    return Err(NumericalError("mean: scaled sum is non-finite"))
  }
  let mean = sum / n * max_abs
  if mean.is_nan() || mean.is_inf() {
    return Err(NumericalError("mean: scaled result is non-finite"))
  }
  Ok(mean)
}

///|
/// Global arithmetic mean of all V entries: `sum(V) / (rows * cols)`.
///
/// The direct sum is used when `max_abs(V) * count` is finite, which bounds
/// the absolute sum and therefore rules out overflow. For very large finite
/// entries the computation falls back to `scaled_mean`, preserving the
/// mathematical mean without producing Infinity. Empty matrices are rejected;
/// an all-zero matrix returns exactly 0.
fn matrix_mean(v : Matrix) -> Result[Double, NMFError] {
  if v.data.length() == 0 {
    return Err(InvalidMatrixShape("mean of an empty matrix is undefined"))
  }
  let n = (v.rows * v.cols).to_double()
  let mut max_abs = 0.0
  for x in v.data {
    let a = x.abs()
    if a > max_abs {
      max_abs = a
    }
  }
  if max_abs.is_nan() || max_abs.is_inf() {
    return Err(NumericalError("mean: matrix contains a non-finite entry"))
  }
  if max_abs == 0.0 {
    return Ok(0.0)
  }
  let bound = max_abs * n
  if bound.is_nan() || bound.is_inf() {
    return scaled_mean(v, max_abs)
  }
  let mut sum = 0.0
  for x in v.data {
    sum = sum + x
  }
  if sum.is_nan() || sum.is_inf() {
    return scaled_mean(v, max_abs)
  }
  let mean = sum / n
  if mean.is_nan() || mean.is_inf() {
    return Err(NumericalError("mean: result is non-finite"))
  }
  Ok(mean)
}

///|
/// NNDSVDa initialization: standard NNDSVD followed by replacement of every
/// exact zero entry of W and H with the global mean of the *original* V.
///
/// Non-zero NNDSVD entries are left untouched; no random noise or additional
/// SVD is introduced. When `mean(V) == 0` (all-zero V for non-negative input),
/// the standard all-zero factors are preserved. Non-finite means, negative
/// means, or non-finite/negative factors are reported as `NumericalError`
/// instead of being silently written into the factors.
fn nndsvda_init(v : Matrix, k : Int32) -> Result[(Matrix, Matrix), NMFError] {
  if v.data.length() == 0 {
    return Err(InvalidMatrixShape("NNDSVDa: V must be non-empty"))
  }
  let (w, h) = match nndsvd_init(v, k) {
    Ok(pair) => pair
    Err(e) => return Err(e)
  }
  let mean = match matrix_mean(v) {
    Ok(x) => x
    Err(e) => return Err(e)
  }
  if mean.is_nan() || mean.is_inf() {
    return Err(NumericalError("NNDSVDa: mean of V is non-finite"))
  }
  if mean < 0.0 {
    return Err(NumericalError("NNDSVDa: mean of V is negative"))
  }
  if mean > 0.0 {
    for idx in 0..<w.data.length() {
      if w.data[idx] == 0.0 {
        w.data[idx] = mean
      }
    }
    for idx in 0..<h.data.length() {
      if h.data[idx] == 0.0 {
        h.data[idx] = mean
      }
    }
  }
  if !all_finite(w.data) || !all_finite(h.data) {
    return Err(NumericalError("NNDSVDa factors became non-finite"))
  }
  for x in w.data {
    if x < 0.0 {
      return Err(NumericalError("NNDSVDa produced a negative W entry"))
    }
  }
  for x in h.data {
    if x < 0.0 {
      return Err(NumericalError("NNDSVDa produced a negative H entry"))
    }
  }
  Ok((w, h))
}
