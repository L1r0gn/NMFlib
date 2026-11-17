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
/// MU-Frobenius fit (Lee & Seung 2001), minimizing
/// `F = ½‖V − WH‖²_F + Ω(W,H)` (Ω is zero for `no_reg`).
///
/// Per outer iteration, H is updated first, then W:
/// ```text
/// H ← H ⊙ (WᵀV) ⊘ max((WᵀW)H + l1 + l2·H, ε)
/// W ← W ⊙ (VHᵀ) ⊘ max(W(HHᵀ) + l1 + l2·W, ε)
/// ```
/// The `l1` term is the derivative of `l1·ΣH`, the `l2·H` term the derivative
/// of `(l2/2)‖H‖²_F`; both enter the denominator exactly as the auxiliary
/// (majorize–minimize) function prescribes, so each step never increases the
/// regularized objective. With `no_reg` the denominators are bit-for-bit the
/// pre-T12 values.
/// Gram matrices WᵀW and HHᵀ are reused so each iteration costs
/// O(ndk + nk² + dk²) instead of recomputing full triple products.
/// The true regularized objective is recomputed from a fresh WH every
/// iteration and drives the convergence test; exhausting max_iter without
/// meeting tol leaves `converged = false`.
fn mu_frobenius_fit(
  v : Matrix,
  k : Int32,
  max_iter : Int32,
  tol : Double,
  seed : UInt64,
  reg : Reg,
) -> Result[MuFit, NMFError] {
  let (w, h) = match random_init(v, k, seed) {
    Ok(pair) => pair
    Err(e) => return Err(e)
  }
  mu_frobenius_fit_from(v, w, h, max_iter, tol, reg)
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
  reg : Reg,
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
  matmul_into(w, h, wh)
  let mut prev = match frobenius_reg_from_wh(v, wh, w, h, reg) {
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
      let den = wwh.data[idx] + reg.l1 + reg.l2 * h.data[idx]
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
      let den = whht.data[idx] + reg.l1 + reg.l2 * w.data[idx]
      let floored = if den > mu_eps { den } else { mu_eps }
      w.data[idx] = w.data[idx] * vht.data[idx] / floored
    }
    if !all_finite(w.data) {
      return Err(NumericalError("W became non-finite during MU updates"))
    }
    // True regularized objective from the freshly updated factors.
    matmul_into(w, h, wh)
    let f = match frobenius_reg_from_wh(v, wh, w, h, reg) {
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
/// MU-Frobenius with H fixed: solves `X ≈ W·H` for W only (used by
/// `NMF::transform`). XHᵀ and HHᵀ are constant across iterations and
/// precomputed once.
///
/// Regularization applies to the W being optimized only. H is fixed, so its
/// penalty `l1·ΣH + (l2/2)‖H‖²_F` is a constant of the subproblem: it does
/// not change the W update (`W ⊘ (WHHᵀ + l1 + l2·W)`) but is included in the
/// reported convergence objective so that the monitored quantity is the full
/// `J_F(X, W, H)` from the shared definition.
fn mu_frobenius_transform(
  x : Matrix,
  h : Matrix,
  max_iter : Int32,
  tol : Double,
  seed : UInt64,
  reg : Reg,
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
  let mut prev = match frobenius_reg_from_wh(x, wh, w, h, reg) {
    Ok(f) => f
    Err(e) => return Err(e)
  }
  for _ in 1..<=max_iter {
    matmul_into(w, hh, whht)
    for idx in 0..<w.data.length() {
      let den = whht.data[idx] + reg.l1 + reg.l2 * w.data[idx]
      let floored = if den > mu_eps { den } else { mu_eps }
      w.data[idx] = w.data[idx] * vht.data[idx] / floored
    }
    if !all_finite(w.data) {
      return Err(NumericalError("W became non-finite during transform updates"))
    }
    matmul_into(w, h, wh)
    let f = match frobenius_reg_from_wh(x, wh, w, h, reg) {
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
/// `H ← H ⊙ (WᵀR) ⊘ (colsum(W) + l1 + l2·H)`.
///
/// The `l1`/`l2·H` terms are the derivatives of `l1·ΣH` and `(l2/2)‖H‖²_F`;
/// they enter the denominator as the auxiliary-function (MM) update requires.
/// With `no_reg` the denominator is exactly `colsum(W)`.
///
/// A zero denominator means the corresponding W column is all zero *and*
/// both regularization strengths are zero (or `l2·H` is exactly zero): that H
/// entry has no influence on WH and no penalty gradient, so it is
/// canonicalized to zero instead of dividing by zero. The same rule is used in
/// the W update. No epsilon floor is used; positive denominators are used
/// exactly.
fn kl_update_h_from_ratio(
  w : Matrix,
  ratio : Matrix,
  h : Matrix,
  numerator : Matrix,
  colsum : Array[Double],
  reg : Reg,
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
    for j in 0..<d {
      let idx = p * d + j
      let old = h.data[idx]
      let den = colsum[p] + reg.l1 + reg.l2 * old
      if den == 0.0 {
        h.data[idx] = 0.0
      } else {
        let num = numerator.data[idx]
        if num.is_nan() || num.is_inf() || num < 0.0 {
          return Err(
            NumericalError("KL H update: invalid numerator at component \{p}"),
          )
        }
        let next = old * num / den
        if next.is_nan() || next.is_inf() {
          return Err(
            NumericalError("KL H update: non-finite update at component \{p}"),
          )
        }
        h.data[idx] = next
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
/// `W ← W ⊙ (R Hᵀ) ⊘ (rowsum(H) + l1 + l2·W)`. A zero denominator means the
/// corresponding H row and the regularization gradient are both zero; that W
/// entry is irrelevant to WH and is canonicalized to zero. No epsilon is
/// used, and `no_reg` reproduces the T11 denominator exactly.
fn kl_update_w_from_ratio(
  h : Matrix,
  ratio : Matrix,
  w : Matrix,
  numerator : Matrix,
  rowsum : Array[Double],
  reg : Reg,
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
    for i in 0..<w.rows {
      let idx = i * k + p
      let old = w.data[idx]
      let den = rowsum[p] + reg.l1 + reg.l2 * old
      if den == 0.0 {
        w.data[idx] = 0.0
      } else {
        let num = numerator.data[idx]
        if num.is_nan() || num.is_inf() || num < 0.0 {
          return Err(
            NumericalError("KL W update: invalid numerator at component \{p}"),
          )
        }
        let next = old * num / den
        if next.is_nan() || next.is_inf() {
          return Err(
            NumericalError("KL W update: non-finite update at component \{p}"),
          )
        }
        w.data[idx] = next
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
/// after H, then updates W; the true regularized generalized KL divergence
/// drives the convergence test. This is a separate branch from Frobenius MU:
/// the Frobenius update rules and their epsilon semantics are unchanged.
fn mu_kl_fit(
  v : Matrix,
  k : Int32,
  max_iter : Int32,
  tol : Double,
  seed : UInt64,
  reg : Reg,
) -> Result[MuFit, NMFError] {
  let (w, h) = match random_init(v, k, seed) {
    Ok(pair) => pair
    Err(e) => return Err(e)
  }
  mu_kl_fit_from(v, w, h, max_iter, tol, reg)
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
  reg : Reg,
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
  let mut prev = match kl_reg_from_wh(v, wh, w, h, reg) {
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
    match kl_update_h_from_ratio(w, ratio, h, wtr, colsum, reg) {
      Ok(_) => ()
      Err(e) => return Err(e)
    }
    // W uses the ratio from the updated H, never the stale H ratio.
    matmul_into(w, h, wh)
    match kl_ratio_into(v, wh, ratio) {
      Ok(_) => ()
      Err(e) => return Err(e)
    }
    match kl_update_w_from_ratio(h, ratio, w, rht, rowsum, reg) {
      Ok(_) => ()
      Err(e) => return Err(e)
    }
    matmul_into(w, h, wh)
    let f = match kl_reg_from_wh(v, wh, w, h, reg) {
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
/// `X ≈ WH` with `W ← W ⊙ (R Hᵀ) ⊘ (rowsum(H) + l1 + l2·W)`, using the true
/// regularized KL objective for convergence. No H retraining, no solver
/// switch, and no model mutation. H's penalty is a constant of this
/// subproblem and does not enter the W update, but is included in the
/// monitored objective.
fn mu_kl_transform(
  x : Matrix,
  h : Matrix,
  max_iter : Int32,
  tol : Double,
  seed : UInt64,
  reg : Reg,
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
  let mut prev = match kl_reg_from_wh(x, wh, w, h, reg) {
    Ok(x) => x
    Err(e) => return Err(e)
  }
  for _ in 1..<=max_iter {
    match kl_ratio_into(x, wh, ratio) {
      Ok(_) => ()
      Err(e) => return Err(e)
    }
    match kl_update_w_from_ratio(h, ratio, w, rht, rowsum, reg) {
      Ok(_) => ()
      Err(e) => return Err(e)
    }
    matmul_into(w, h, wh)
    let f = match kl_reg_from_wh(x, wh, w, h, reg) {
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
