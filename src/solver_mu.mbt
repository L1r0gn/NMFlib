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
  let (w, h) = match random_init(v, k, seed) {
    Ok(pair) => pair
    Err(e) => return Err(e)
  }
  mu_frobenius_fit_from(v, w, h, max_iter, tol)
}

///|
/// MU-Frobenius fit from caller-supplied initial factors. The update rules,
/// buffers, and convergence behavior are identical to `mu_frobenius_fit`;
/// this entry point lets NNDSVD (T09) supply its own non-negative W/H while
/// keeping Random initialization bit-for-bit unchanged.
fn mu_frobenius_fit_from(
  v : Matrix,
  w : Matrix,
  h : Matrix,
  max_iter : Int32,
  tol : Double,
) -> Result[MuFit, NMFError] {
  let n = v.rows
  let d = v.cols
  let k = w.cols
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

///|
/// Computes the KL update ratio `R = V ⊘ WH` into `ratio`.
///
/// Zero policy: `V=0` gives ratio 0 for both `WH=0` and `WH>0`; `V>0, WH=0`
/// is an infinite KL objective and returns `NumericalError` instead of silently
/// substituting 0. Any non-finite ratio or negative entry is rejected.
fn kl_ratio_into(
  v : Matrix,
  wh : Matrix,
  ratio : Matrix,
) -> Result[Unit, NMFError] {
  let d = v.cols
  for idx in 0..<v.data.length() {
    let x = v.data[idx]
    let y = wh.data[idx]
    let i = idx / d
    let j = idx % d
    if x.is_nan() || x.is_inf() || y.is_nan() || y.is_inf() {
      return Err(NumericalError("KL update: non-finite V/WH at (\{i},\{j})"))
    }
    if x < 0.0 || y < 0.0 {
      return Err(NegativeEntry(i, j))
    }
    if x == 0.0 {
      // V=0: the ratio is 0 by the 0/0 and 0/WH limits; no division occurs.
      ratio.data[idx] = 0.0
    } else if y == 0.0 {
      return Err(
        NumericalError("KL update: infinite ratio, V>0 and WH=0 at (\{i},\{j})"),
      )
    } else {
      let r = x / y
      if r.is_nan() || r.is_inf() {
        return Err(NumericalError("KL update: non-finite ratio at (\{i},\{j})"))
      }
      ratio.data[idx] = r
    }
  }
  Ok(())
}

///|
/// MU-KL H update from a precomputed ratio matrix:
/// `H ← H ⊙ (WᵀR) ⊘ colsum(W)`.
///
/// A zero denominator means the corresponding W column is all zero; that H
/// row has no influence on WH, so it is canonicalized to zero instead of
/// dividing by zero. The same rule is used in the W update. No epsilon floor
/// is used; positive denominators are used exactly.
fn kl_update_h_from_ratio(
  w : Matrix,
  ratio : Matrix,
  h : Matrix,
  numerator : Matrix,
  colsum : Array[Double],
) -> Result[Unit, NMFError] {
  let k = w.cols
  let d = h.cols
  matmul_at_b_into(w, ratio, numerator)
  if !all_finite(numerator.data) {
    return Err(NumericalError("KL H update: WᵀR is non-finite"))
  }
  for p in 0..<k {
    let mut s = 0.0
    for i in 0..<w.rows {
      s = s + w.data[i * k + p]
    }
    if s.is_nan() || s.is_inf() || s < 0.0 {
      return Err(NumericalError("KL H update: invalid colsum(W)"))
    }
    colsum[p] = s
  }
  for p in 0..<k {
    let den = colsum[p]
    if den == 0.0 {
      for j in 0..<d {
        h.data[p * d + j] = 0.0
      }
    } else {
      for j in 0..<d {
        let num = numerator.data[p * d + j]
        if num.is_nan() || num.is_inf() || num < 0.0 {
          return Err(
            NumericalError("KL H update: invalid numerator at component \{p}"),
          )
        }
        let next = h.data[p * d + j] * num / den
        if next.is_nan() || next.is_inf() {
          return Err(
            NumericalError("KL H update: non-finite update at component \{p}"),
          )
        }
        h.data[p * d + j] = next
      }
    }
  }
  if !all_finite(h.data) {
    return Err(NumericalError("H became non-finite during KL updates"))
  }
  for x in h.data {
    if x < 0.0 {
      return Err(NumericalError("H became negative during KL updates"))
    }
  }
  Ok(())
}

///|
/// MU-KL W update from a precomputed ratio matrix:
/// `W ← W ⊙ (R Hᵀ) ⊘ rowsum(H)`. A zero H row makes that W column
/// irrelevant to WH and is canonicalized to zero; no epsilon is used.
fn kl_update_w_from_ratio(
  h : Matrix,
  ratio : Matrix,
  w : Matrix,
  numerator : Matrix,
  rowsum : Array[Double],
) -> Result[Unit, NMFError] {
  let k = h.rows
  matmul_a_bt_into(ratio, h, numerator)
  if !all_finite(numerator.data) {
    return Err(NumericalError("KL W update: RHᵀ is non-finite"))
  }
  for p in 0..<k {
    let mut s = 0.0
    for j in 0..<h.cols {
      s = s + h.data[p * h.cols + j]
    }
    if s.is_nan() || s.is_inf() || s < 0.0 {
      return Err(NumericalError("KL W update: invalid rowsum(H)"))
    }
    rowsum[p] = s
  }
  for p in 0..<k {
    let den = rowsum[p]
    if den == 0.0 {
      for i in 0..<w.rows {
        w.data[i * k + p] = 0.0
      }
    } else {
      for i in 0..<w.rows {
        let num = numerator.data[i * k + p]
        if num.is_nan() || num.is_inf() || num < 0.0 {
          return Err(
            NumericalError("KL W update: invalid numerator at component \{p}"),
          )
        }
        let next = w.data[i * k + p] * num / den
        if next.is_nan() || next.is_inf() {
          return Err(
            NumericalError("KL W update: non-finite update at component \{p}"),
          )
        }
        w.data[i * k + p] = next
      }
    }
  }
  if !all_finite(w.data) {
    return Err(NumericalError("W became non-finite during KL updates"))
  }
  for x in w.data {
    if x < 0.0 {
      return Err(NumericalError("W became negative during KL updates"))
    }
  }
  Ok(())
}

///|
/// MU-KL fit. Each iteration recomputes WH and R before H, recomputes them
/// after H, then updates W; the true generalized KL divergence drives the
/// convergence test. This is a separate branch from Frobenius MU: the
/// Frobenius update rules and their epsilon semantics are unchanged.
fn mu_kl_fit(
  v : Matrix,
  k : Int32,
  max_iter : Int32,
  tol : Double,
  seed : UInt64,
) -> Result[MuFit, NMFError] {
  let (w, h) = match random_init(v, k, seed) {
    Ok(pair) => pair
    Err(e) => return Err(e)
  }
  mu_kl_fit_from(v, w, h, max_iter, tol)
}

///|
/// MU-KL fit from caller-supplied initial factors. NNDSVD/NNDSVDa factors are
/// used as-is; this routine does not alter the public initialization output
/// and does not substitute a different Init. It does not add a positive floor:
/// if `V>0` and `WH=0`, it returns `NumericalError` rather than silently
/// changing the objective.
fn mu_kl_fit_from(
  v : Matrix,
  w : Matrix,
  h : Matrix,
  max_iter : Int32,
  tol : Double,
) -> Result[MuFit, NMFError] {
  let n = v.rows
  let d = v.cols
  let k = w.cols
  let wh = zeros_buf(n, d)
  let ratio = zeros_buf(n, d)
  let wtr = zeros_buf(k, d)
  let rht = zeros_buf(n, k)
  let colsum = Array::make(k, 0.0)
  let rowsum = Array::make(k, 0.0)
  matmul_into(w, h, wh)
  let mut prev = match kl_divergence(v, wh) {
    Ok(x) => x
    Err(e) => return Err(e)
  }
  let mut loss = prev
  let mut n_iter = 0
  let mut converged = false
  for iter in 1..<=max_iter {
    n_iter = iter
    match kl_ratio_into(v, wh, ratio) {
      Ok(_) => ()
      Err(e) => return Err(e)
    }
    match kl_update_h_from_ratio(w, ratio, h, wtr, colsum) {
      Ok(_) => ()
      Err(e) => return Err(e)
    }
    // W uses the ratio from the updated H, never the stale H ratio.
    matmul_into(w, h, wh)
    match kl_ratio_into(v, wh, ratio) {
      Ok(_) => ()
      Err(e) => return Err(e)
    }
    match kl_update_w_from_ratio(h, ratio, w, rht, rowsum) {
      Ok(_) => ()
      Err(e) => return Err(e)
    }
    matmul_into(w, h, wh)
    let f = match kl_divergence(v, wh) {
      Ok(x) => x
      Err(e) => return Err(e)
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
/// MU-KL transform: H is fixed from training and W is optimized for new data
/// `X ≈ WH` with `W ← W ⊙ (R Hᵀ) ⊘ rowsum(H)`, using the true KL divergence
/// for convergence. No H retraining, no solver switch, and no model mutation.
fn mu_kl_transform(
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
  let wh = zeros_buf(m, d)
  let ratio = zeros_buf(m, d)
  let rht = zeros_buf(m, k)
  let rowsum = Array::make(k, 0.0)
  matmul_into(w, h, wh)
  let mut prev = match kl_divergence(x, wh) {
    Ok(x) => x
    Err(e) => return Err(e)
  }
  for _ in 1..<=max_iter {
    match kl_ratio_into(x, wh, ratio) {
      Ok(_) => ()
      Err(e) => return Err(e)
    }
    match kl_update_w_from_ratio(h, ratio, w, rht, rowsum) {
      Ok(_) => ()
      Err(e) => return Err(e)
    }
    matmul_into(w, h, wh)
    let f = match kl_divergence(x, wh) {
      Ok(x) => x
      Err(e) => return Err(e)
    }
    if rel_change(f, prev) < tol {
      break
    }
    prev = f
  }
  Ok(w)
}
