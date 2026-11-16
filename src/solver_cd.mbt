///|
/// Solver output: factors plus honest training metadata.
priv struct CdFit {
  w : Matrix
  h : Matrix
  n_iter : Int32
  loss_value : Double
  converged : Bool
}

///|
/// One full cyclic sweep over H with W fixed, minimizing
/// `F = ½‖V − W·H‖²_F + l1·ΣH + (l2/2)‖H‖²_F` one coordinate at a time.
///
/// `gram = WᵀW` (k×k) and `cross = WᵀV` (k×d) must be precomputed for the
/// current W; `grad` is a k-entry scratch buffer whose contents are rebuilt
/// for every column and maintained incrementally as coordinates change.
/// Columns are visited left to right and coordinates top to bottom, giving a
/// deterministic order. Each accepted step updates the cached **data**
/// gradient with `grad[i] += gram[i,p] * delta`, so the cache stays equal to
/// `gram·h_j − cross[:,j]` throughout the column sweep. The regularization
/// derivative `l1 + l2·h_p` is added per coordinate from the current value
/// (there is no cross-coordinate coupling), so the total gradient is exact.
///
/// Regularized coordinate subproblem: curvature `gram[p,p] + l2`, gradient
/// `gram·h_j − cross[:,j] + l1 + l2·h_p`, exact non-negative minimizer
/// `max(0, h_p − g / curvature)`. This is the closed-form minimizer of the
/// regularized quadratic, not a post-hoc shrink.
///
/// Curvature `gram[p,p] + l2` is non-negative. When it is strictly positive
/// the exact line minimizer is used with no epsilon floor: adding an epsilon
/// would change the objective. When it is zero (`l2 = 0` and `W[:,p]` is
/// numerically zero), the coordinate does not affect `WH`; the subproblem is
/// `(g_data + l1)·h_p` on `h_p >= 0`, so a negative slope has no finite
/// minimum (`NumericalError`) and any non-negative slope is minimized at 0
/// (canonicalized). Non-finite gradients, steps, or factors are always
/// reported as `NumericalError`; tiny negative values from rounding are
/// clamped to 0.
fn cd_update_h(
  gram : Matrix,
  cross : Matrix,
  h : Matrix,
  grad : Array[Double],
  reg : Reg,
) -> Result[Unit, NMFError] {
  let k = gram.rows
  let d = h.cols
  for j in 0..<d {
    for p in 0..<k {
      let mut s = 0.0
      let row = p * k
      for q in 0..<k {
        s = s + gram.data[row + q] * h.data[q * d + j]
      }
      grad[p] = s - cross.data[p * d + j]
    }
    for p in 0..<k {
      let curvature = gram.data[p * k + p] + reg.l2
      if curvature.is_nan() || curvature.is_inf() || curvature < 0.0 {
        return Err(
          NumericalError(
            "CD H update: invalid curvature at coordinate \{p}, column \{j}",
          ),
        )
      }
      let old = h.data[p * d + j]
      let g = grad[p] + reg.l1 + reg.l2 * old
      if curvature > 0.0 {
        if g.is_nan() || g.is_inf() {
          return Err(
            NumericalError(
              "CD H update: non-finite gradient at coordinate \{p}, column \{j}",
            ),
          )
        }
        let candidate = old - g / curvature
        if candidate.is_nan() || candidate.is_inf() {
          return Err(
            NumericalError(
              "CD H update: non-finite step at coordinate \{p}, column \{j}",
            ),
          )
        }
        let new = if candidate > 0.0 { candidate } else { 0.0 }
        let delta = new - old
        if delta != 0.0 {
          h.data[p * d + j] = new
          for i in 0..<k {
            grad[i] = grad[i] + gram.data[i * k + p] * delta
          }
        }
      } else {
        // Zero curvature: W[:,p] is numerically zero and l2 = 0, so this
        // coordinate has no effect on WH and the subproblem is linear in
        // (g_data + l1). A negative slope is unbounded below; a non-negative
        // slope is minimized at 0, which we canonicalize.
        if g.is_nan() || g.is_inf() || g < 0.0 {
          return Err(
            NumericalError(
              "CD H update: zero curvature with negative slope at coordinate \{p}, column \{j}",
            ),
          )
        }
        if old != 0.0 {
          h.data[p * d + j] = 0.0
          for i in 0..<k {
            grad[i] = grad[i] + gram.data[i * k + p] * -old
          }
        }
      }
    }
  }
  if !all_finite(h.data) {
    return Err(NumericalError("H became non-finite during CD updates"))
  }
  Ok(())
}

///|
/// One full cyclic sweep over W with H fixed. This is the exact mirror of
/// `cd_update_h`: `gram = HHᵀ` (k×k), `cross = VHᵀ` (n×k), rows of W are
/// independent, and coordinates within a row are visited left to right.
/// Curvature is `gram[p,p] + l2` and the coordinate gradient is
/// `gram·w_i − cross[i,:] + l1 + l2·w_p`; the same zero-curvature,
/// non-finite, and non-negativity rules apply.
fn cd_update_w(
  gram : Matrix,
  cross : Matrix,
  w : Matrix,
  grad : Array[Double],
  reg : Reg,
) -> Result[Unit, NMFError] {
  let n = w.rows
  let k = gram.rows
  for i in 0..<n {
    for p in 0..<k {
      let mut s = 0.0
      let row = p * k
      for q in 0..<k {
        s = s + gram.data[row + q] * w.data[i * k + q]
      }
      grad[p] = s - cross.data[i * k + p]
    }
    for p in 0..<k {
      let curvature = gram.data[p * k + p] + reg.l2
      if curvature.is_nan() || curvature.is_inf() || curvature < 0.0 {
        return Err(
          NumericalError(
            "CD W update: invalid curvature at coordinate \{p}, row \{i}",
          ),
        )
      }
      let old = w.data[i * k + p]
      let g = grad[p] + reg.l1 + reg.l2 * old
      if curvature > 0.0 {
        if g.is_nan() || g.is_inf() {
          return Err(
            NumericalError(
              "CD W update: non-finite gradient at coordinate \{p}, row \{i}",
            ),
          )
        }
        let candidate = old - g / curvature
        if candidate.is_nan() || candidate.is_inf() {
          return Err(
            NumericalError(
              "CD W update: non-finite step at coordinate \{p}, row \{i}",
            ),
          )
        }
        let new = if candidate > 0.0 { candidate } else { 0.0 }
        let delta = new - old
        if delta != 0.0 {
          w.data[i * k + p] = new
          for q in 0..<k {
            grad[q] = grad[q] + gram.data[q * k + p] * delta
          }
        }
      } else {
        if g.is_nan() || g.is_inf() || g < 0.0 {
          return Err(
            NumericalError(
              "CD W update: zero curvature with negative slope at coordinate \{p}, row \{i}",
            ),
          )
        }
        if old != 0.0 {
          w.data[i * k + p] = 0.0
          for q in 0..<k {
            grad[q] = grad[q] + gram.data[q * k + p] * -old
          }
        }
      }
    }
  }
  if !all_finite(w.data) {
    return Err(NumericalError("W became non-finite during CD updates"))
  }
  Ok(())
}

///|
/// CD-Frobenius fit (cyclic coordinate descent for
/// `F = ½‖V − WH‖²_F + l1·(ΣW+ΣH) + (l2/2)(‖W‖²+‖H‖²)`; `no_reg` gives the
/// pre-T12 objective).
///
/// Each outer iteration performs a complete H sweep with W fixed, then a
/// complete W sweep with the updated H fixed, then recomputes the true
/// regularized objective and applies the same relative-change convergence
/// rule as MU. Every coordinate step is the exact minimizer of the
/// regularized one-dimensional subproblem, so the full objective is
/// non-increasing.
/// Gram and cross-product buffers are allocated once and refreshed per
/// iteration (WᵀW, WᵀV, HHᵀ, VHᵀ); the gradient scratch buffer is reused for
/// every column/row. No matrix copies or fresh allocations occur inside the
/// coordinate loops.
///
/// Numerical policy: only exactly-zero curvature is special-cased; positive
/// curvature (including `l2 > 0`) uses the exact line minimizer without an
/// epsilon floor, and a non-finite result is a hard `NumericalError`.
/// Exhausting `max_iter` without meeting `tol` leaves `converged = false`.
fn cd_frobenius_fit(
  v : Matrix,
  k : Int32,
  max_iter : Int32,
  tol : Double,
  seed : UInt64,
  reg : Reg,
) -> Result[CdFit, NMFError] {
  let (w, h) = match random_init(v, k, seed) {
    Ok(pair) => pair
    Err(e) => return Err(e)
  }
  cd_frobenius_fit_from(v, w, h, max_iter, tol, reg)
}

///|
/// CD-Frobenius fit from caller-supplied initial factors. The sweeps, buffers,
/// and convergence behavior are identical to `cd_frobenius_fit`; NNDSVD
/// (T09) supplies its own non-negative W/H.
fn cd_frobenius_fit_from(
  v : Matrix,
  w : Matrix,
  h : Matrix,
  max_iter : Int32,
  tol : Double,
  reg : Reg,
) -> Result[CdFit, NMFError] {
  let n = v.rows
  let d = v.cols
  let k = w.cols
  // Buffers allocated once and reused across iterations.
  let wtw = zeros_buf(k, k) // WᵀW
  let wv = zeros_buf(k, d) // WᵀV
  let hh = zeros_buf(k, k) // HHᵀ
  let vht = zeros_buf(n, k) // VHᵀ
  let grad = Array::make(k, 0.0)
  let wh = zeros_buf(n, d) // WH for the true objective
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
    // H sweep with W fixed.
    matmul_at_b_into(w, w, wtw)
    matmul_at_b_into(w, v, wv)
    if !all_finite(wtw.data) || !all_finite(wv.data) {
      return Err(
        NumericalError("CD: non-finite Gram or cross product for H update"),
      )
    }
    match cd_update_h(wtw, wv, h, grad, reg) {
      Ok(_) => ()
      Err(e) => return Err(e)
    }
    // W sweep with the updated H fixed.
    matmul_a_bt_into(h, h, hh)
    matmul_a_bt_into(v, h, vht)
    if !all_finite(hh.data) || !all_finite(vht.data) {
      return Err(
        NumericalError("CD: non-finite Gram or cross product for W update"),
      )
    }
    match cd_update_w(hh, vht, w, grad, reg) {
      Ok(_) => ()
      Err(e) => return Err(e)
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
/// CD-Frobenius with H fixed: solves `X ≈ W·H` for W only (used by
/// `NMF::transform`). `HHᵀ` and `XHᵀ` are constant across iterations and are
/// precomputed once; each iteration performs a full cyclic W sweep with the
/// regularized coordinate update (`curvature = HHᵀ[p,p] + l2`,
/// `gradient = HHᵀ·w_i − XHᵀ[i,:] + l1 + l2·w_p`) followed by the same
/// relative-objective convergence test. H's penalty is constant here, so it
/// never changes the W step but is included in the monitored objective. The
/// training W is never modified, and this path never falls back to MU.
fn cd_frobenius_transform(
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
  let hh = zeros_buf(k, k)
  matmul_a_bt_into(h, h, hh) // HHᵀ, constant
  let vht = zeros_buf(m, k)
  matmul_a_bt_into(x, h, vht) // XHᵀ, constant
  if !all_finite(hh.data) || !all_finite(vht.data) {
    return Err(NumericalError("CD transform: non-finite Gram or cross product"))
  }
  let grad = Array::make(k, 0.0)
  let wh = zeros_buf(m, d)
  matmul_into(w, h, wh)
  let mut prev = match frobenius_reg_from_wh(x, wh, w, h, reg) {
    Ok(f) => f
    Err(e) => return Err(e)
  }
  for _ in 1..<=max_iter {
    match cd_update_w(hh, vht, w, grad, reg) {
      Ok(_) => ()
      Err(e) => return Err(e)
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
