///|
/// Denominator floor for multiplicative updates.
///
/// We use `max(den, MU_EPS)` rather than `den + MU_EPS`: whenever the
/// denominator is meaningful (>= 1e-10) the update is *exactly* the
/// Lee–Seung ratio, so the protection never perturbs the optimization
/// objective. The floor engages only for numerically-zero denominators;
/// there the numerator (which shares the same W/H factors) is also ~0, the
/// ratio is meaningless, and flooring prevents 0/0 NaNs and unbounded
/// growth. 1e-10 sits far below any meaningful Gram-matrix magnitude for
/// unit-scale data while staying well above Double rounding noise.
let mu_eps : Double = 1.0e-10

///|
/// Solver output: factors plus honest training metadata.
priv struct MuFit {
  w : Matrix
  h : Matrix
  n_iter : Int32
  loss_value : Double
  converged : Bool
}

///|
/// Relative objective change `|F_t − F_{t−1}| / max(1, |F_{t−1}|)`;
/// the `max(1, ·)` avoids division by zero near F = 0.
fn rel_change(f : Double, prev : Double) -> Double {
  let base = if prev.abs() > 1.0 { prev.abs() } else { 1.0 }
  (f - prev).abs() / base
}

///|
/// MU-Frobenius fit (Lee & Seung 2001), minimizing `F = ½‖V − WH‖²_F`.
///
/// Per outer iteration, H is updated first, then W:
/// ```text
/// H ← H ⊙ (WᵀV) ⊘ max((WᵀW)H, ε)
/// W ← W ⊙ (VHᵀ) ⊘ max(W(HHᵀ), ε)
/// ```
/// Gram matrices WᵀW and HHᵀ are reused so each iteration costs
/// O(ndk + nk² + dk²) instead of recomputing full triple products.
/// The true objective is recomputed from a fresh WH every iteration and
/// drives the convergence test; exhausting max_iter without meeting tol
/// leaves `converged = false`.
fn mu_frobenius_fit(
  v : Matrix,
  k : Int32,
  max_iter : Int32,
  tol : Double,
  seed : UInt64,
) -> Result[MuFit, NMFError] {
  let n = v.rows
  let d = v.cols
  let (w, h) = match random_init(v, k, seed) {
    Ok(pair) => pair
    Err(e) => return Err(e)
  }
  // Scratch buffers allocated once and reused every iteration.
  let wv = zeros_buf(k, d) // WᵀV
  let ww = zeros_buf(k, k) // WᵀW
  let wwh = zeros_buf(k, d) // (WᵀW)H
  let vht = zeros_buf(n, k) // VHᵀ
  let hh = zeros_buf(k, k) // HHᵀ
  let whht = zeros_buf(n, k) // W(HHᵀ)
  let wh = zeros_buf(n, d) // WH for the objective
  let mut prev = match frobenius_objective(v, w, h) {
    Ok(f) => f
    Err(e) => return Err(e)
  }
  let mut loss = prev
  let mut n_iter = 0
  let mut converged = false
  for iter in 1..<=max_iter {
    n_iter = iter
    // H update.
    matmul_at_b_into(w, v, wv)
    matmul_at_b_into(w, w, ww)
    matmul_into(ww, h, wwh)
    for idx in 0..<h.data.length() {
      let den = wwh.data[idx]
      let floored = if den > mu_eps { den } else { mu_eps }
      h.data[idx] = h.data[idx] * wv.data[idx] / floored
    }
    if !all_finite(h.data) {
      return Err(NumericalError("H became non-finite during MU updates"))
    }
    // W update.
    matmul_a_bt_into(v, h, vht)
    matmul_a_bt_into(h, h, hh)
    matmul_into(w, hh, whht)
    for idx in 0..<w.data.length() {
      let den = whht.data[idx]
      let floored = if den > mu_eps { den } else { mu_eps }
      w.data[idx] = w.data[idx] * vht.data[idx] / floored
    }
    if !all_finite(w.data) {
      return Err(NumericalError("W became non-finite during MU updates"))
    }
    // True objective from the freshly updated factors.
    matmul_into(w, h, wh)
    let f = 0.5 * sq_diff_from_wh(v, wh)
    if f.is_nan() || f.is_inf() {
      return Err(
        NumericalError("objective became non-finite during MU updates"),
      )
    }
    loss = f
    if rel_change(f, prev) < tol {
      converged = true
      break
    }
    prev = f
  }
  Ok({ w, h, n_iter, loss_value: loss, converged, })
}

///|
/// MU-Frobenius with H fixed: solves `X ≈ W·H` for W only (used by
/// `NMF::transform`). XHᵀ and HHᵀ are constant across iterations and
/// precomputed once.
fn mu_frobenius_transform(
  x : Matrix,
  h : Matrix,
  max_iter : Int32,
  tol : Double,
  seed : UInt64,
) -> Result[Matrix, NMFError] {
  let m = x.rows
  let k = h.rows
  let d = h.cols
  let w = match random_init_w(x, k, seed) {
    Ok(w) => w
    Err(e) => return Err(e)
  }
  let vht = zeros_buf(m, k)
  matmul_a_bt_into(x, h, vht) // XHᵀ, constant
  let hh = zeros_buf(k, k)
  matmul_a_bt_into(h, h, hh) // HHᵀ, constant
  let whht = zeros_buf(m, k)
  let wh = zeros_buf(m, d)
  matmul_into(w, h, wh)
  let s0 = sq_diff_from_wh(x, wh)
  if s0.is_nan() || s0.is_inf() {
    return Err(NumericalError("initial transform objective overflowed"))
  }
  let mut prev = 0.5 * s0
  for _ in 1..<=max_iter {
    matmul_into(w, hh, whht)
    for idx in 0..<w.data.length() {
      let den = whht.data[idx]
      let floored = if den > mu_eps { den } else { mu_eps }
      w.data[idx] = w.data[idx] * vht.data[idx] / floored
    }
    if !all_finite(w.data) {
      return Err(NumericalError("W became non-finite during transform updates"))
    }
    matmul_into(w, h, wh)
    let f = 0.5 * sq_diff_from_wh(x, wh)
    if f.is_nan() || f.is_inf() {
      return Err(NumericalError("transform objective became non-finite"))
    }
    if rel_change(f, prev) < tol {
      break
    }
    prev = f
  }
  Ok(w)
}
