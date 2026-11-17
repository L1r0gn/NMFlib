///|
/// Relative zero threshold for internal SVD truncation. Singular values below
/// `1e-12 × ‖V‖_F` are treated as numerically zero: after deflation they are
/// indistinguishable from rounding noise, and normalizing them would amplify
/// noise into meaningless directions.
let svd_zero_rel : Double = 1.0e-12

///|
/// Power-iteration convergence tolerance on the right singular vector
/// (componentwise maximum change).
let svd_power_tol : Double = 1.0e-10

///|
/// Fixed power-iteration cap. If the tolerance is not reached, the best
/// iterate is used; this internal routine is an SVD *approximation* for
/// NNDSVD initialization, not a public exact-SVD claim.
let svd_power_max_iter : Int32 = 500

///|
/// Fixed deterministic seed base for power-iteration start vectors. NNDSVD is
/// seed-independent by design: the user-facing `NMFConfig.seed` only drives
/// Random initialization.
let svd_seed_base : UInt64 = 0x4E4E44535644UL

///|
/// One singular triplet extracted from a residual matrix. `u` has length
/// rows, `v` has length cols, both are unit vectors, and `sigma >= 0`.
priv struct SvdComponent {
  sigma : Double
  u : Array[Double]
  v : Array[Double]
}

///|
/// Zero singular triplet for a `m × n` matrix; used for rank-deficient inputs
/// and after a residual becomes numerically zero.
fn svd_zero_component(m : Int32, n : Int32) -> SvdComponent {
  { sigma: 0.0, u: Array::make(m, 0.0), v: Array::make(n, 0.0), }
}

///|
/// Euclidean norm with overflow reporting. Underflow to 0 is allowed and is
/// interpreted by callers as a numerically zero vector.
fn svd_vector_norm(x : Array[Double]) -> Result[Double, NMFError] {
  let mut s = 0.0
  for value in x {
    s = s + value * value
  }
  if s.is_nan() || s.is_inf() {
    return Err(NumericalError("internal SVD vector norm overflowed"))
  }
  Ok(s.sqrt())
}

///|
/// Deterministic strictly-positive unit start vector from SplitMix64.
fn svd_initial_vector(
  n : Int32,
  seed : UInt64,
) -> Result[Array[Double], NMFError] {
  let rng = SplitMix64::new(seed)
  let vec = Array::make(n, 0.0)
  for i in 0..<n {
    vec[i] = rng.next_double()
  }
  let norm = match svd_vector_norm(vec) {
    Ok(x) => x
    Err(e) => return Err(e)
  }
  if norm <= 0.0 || norm.is_nan() || norm.is_inf() {
    return Err(
      NumericalError("internal SVD could not build a non-zero start vector"),
    )
  }
  for i in 0..<n {
    vec[i] = vec[i] / norm
  }
  Ok(vec)
}

///|
/// Extracts the dominant singular triplet of `residual` by power iteration:
/// `u ← R v / ‖R v‖`, `v ← Rᵀu / ‖Rᵀu‖`, with `sigma = ‖R v‖`. The final
/// `u` is recomputed from the last `v`, so `R v = sigma · u` holds for the
/// returned pair up to rounding.
///
/// A residual whose dominant singular value is at or below `zero_tol` returns
/// a zero triplet; no normalization is attempted and no division by zero can
/// occur. Non-finite intermediates return `NumericalError`. If the fixed
/// iteration cap is exhausted, the last iterate is returned (the SVD is an
/// internal approximation; see `svd_power_max_iter`).
fn svd_top_singular(
  residual : Matrix,
  zero_tol : Double,
  seed : UInt64,
) -> Result[SvdComponent, NMFError] {
  let m = residual.rows
  let n = residual.cols
  if m == 0 || n == 0 {
    return Ok(svd_zero_component(m, n))
  }
  let v = match svd_initial_vector(n, seed) {
    Ok(x) => x
    Err(e) => return Err(e)
  }
  let v_next = Array::make(n, 0.0)
  let u = Array::make(m, 0.0)
  for _ in 0..<svd_power_max_iter {
    // u = R v.
    for i in 0..<m {
      let mut s = 0.0
      let row = i * n
      for j in 0..<n {
        s = s + residual.data[row + j] * v[j]
      }
      u[i] = s
    }
    let sigma = match svd_vector_norm(u) {
      Ok(x) => x
      Err(e) => return Err(e)
    }
    if sigma <= zero_tol || sigma == 0.0 {
      return Ok(svd_zero_component(m, n))
    }
    for i in 0..<m {
      u[i] = u[i] / sigma
    }
    // v_next = Rᵀ u.
    for j in 0..<n {
      let mut s = 0.0
      for i in 0..<m {
        s = s + residual.data[i * n + j] * u[i]
      }
      v_next[j] = s
    }
    let norm_v = match svd_vector_norm(v_next) {
      Ok(x) => x
      Err(e) => return Err(e)
    }
    if norm_v <= zero_tol || norm_v == 0.0 {
      return Ok(svd_zero_component(m, n))
    }
    for j in 0..<n {
      v_next[j] = v_next[j] / norm_v
    }
    let mut diff = 0.0
    for j in 0..<n {
      let d = (v_next[j] - v[j]).abs()
      if d > diff {
        diff = d
      }
    }
    for j in 0..<n {
      v[j] = v_next[j]
    }
    if diff <= svd_power_tol {
      break
    }
  }
  // Recompute u and sigma from the final v for a consistent triplet.
  for i in 0..<m {
    let mut s = 0.0
    let row = i * n
    for j in 0..<n {
      s = s + residual.data[row + j] * v[j]
    }
    u[i] = s
  }
  let sigma = match svd_vector_norm(u) {
    Ok(x) => x
    Err(e) => return Err(e)
  }
  if sigma <= zero_tol || sigma == 0.0 {
    return Ok(svd_zero_component(m, n))
  }
  for i in 0..<m {
    u[i] = u[i] / sigma
  }
  if !all_finite(u) || !all_finite(v) {
    return Err(NumericalError("internal SVD produced a non-finite triplet"))
  }
  Ok({ sigma, u, v, })
}

///|
/// Internal truncated SVD by power iteration plus deflation. Returns exactly
/// `k` triplets in non-increasing dominant order; components after a
/// numerically zero singular value are zero triplets. The residual after
/// deflation is allowed to be negative — this routine is a numerical linear
/// algebra helper, not an NMF input check and not a public SVD API.
///
/// Deflation accumulates rounding error; each deflated residual is checked for
/// finiteness, and singular values below `1e-12 × ‖V‖_F` stop the extraction.
fn svd_top_k(a : Matrix, k : Int32) -> Result[Array[SvdComponent], NMFError] {
  let m = a.rows
  let n = a.cols
  let components = Array::make(k, svd_zero_component(m, n))
  if k <= 0 || m == 0 || n == 0 {
    return Ok(components)
  }
  let frob = match frobenius_norm(a) {
    Ok(x) => x
    Err(e) => return Err(e)
  }
  if frob == 0.0 || frob.is_nan() || frob.is_inf() {
    return Ok(components)
  }
  let zero_tol = frob * svd_zero_rel
  let residual = a.clone()
  for r in 0..<k {
    let comp = match
      svd_top_singular(residual, zero_tol, svd_seed_base + r.to_uint64()) {
      Ok(c) => c
      Err(e) => return Err(e)
    }
    components[r] = comp
    if comp.sigma <= zero_tol || comp.sigma == 0.0 {
      break
    }
    for i in 0..<m {
      let row = i * n
      for j in 0..<n {
        residual.data[row + j] = residual.data[row + j] -
          comp.sigma * comp.u[i] * comp.v[j]
      }
    }
    if !all_finite(residual.data) {
      return Err(
        NumericalError("internal SVD deflation produced a non-finite residual"),
      )
    }
  }
  Ok(components)
}
