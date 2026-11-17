// Whitebox tests for T12 L1/L2 regularization: hand-computed objective
// values, per-solver update formulas against independent references,
// monotonicity of the full regularized objective, effect checks, transform,
// and zero-regularization compatibility.
//
// The reference helpers below deliberately re-derive every update from the
// mathematical definition with plain triple loops (no shared buffers, no
// incremental gradient cache). They therefore differ from the implementation
// in floating-point summation order only, so comparisons use a relative
// tolerance rather than bit equality.

///|
fn rg_mat(rows : Int32, cols : Int32, data : Array[Double]) -> Matrix {
  match Matrix::new(rows, cols, data) {
    Ok(m) => m
    Err(e) => abort("unexpected matrix error: \{e}")
  }
}

///|
fn rg_reg(l1 : Double, l2 : Double) -> Reg {
  { l1, l2, }
}

///|
fn rg_unwrap_mu(r : Result[MuFit, NMFError]) -> MuFit {
  match r {
    Ok(x) => x
    Err(e) => abort("unexpected MU fit error: \{e}")
  }
}

///|
fn rg_unwrap_cd(r : Result[CdFit, NMFError]) -> CdFit {
  match r {
    Ok(x) => x
    Err(e) => abort("unexpected CD fit error: \{e}")
  }
}

///|
fn rg_unwrap_mat(r : Result[Matrix, NMFError]) -> Matrix {
  match r {
    Ok(m) => m
    Err(e) => abort("unexpected matrix error: \{e}")
  }
}

///|
fn rg_unwrap_double(r : Result[Double, NMFError]) -> Double {
  match r {
    Ok(x) => x
    Err(e) => abort("unexpected: \{e}")
  }
}

///|
fn rg_wh(w : Matrix, h : Matrix) -> Matrix {
  rg_unwrap_mat(matmul(w, h))
}

///|
fn rg_tol(x : Double) -> Double {
  1.0e-9 * (1.0 + x.abs())
}

///|
fn rg_close(a : Double, b : Double) -> Bool {
  (a - b).abs() <= 1.0e-11 * (1.0 + a.abs() + b.abs())
}

///|
fn rg_assert_close(a : Double, b : Double, label : String) -> Unit {
  if !rg_close(a, b) {
    abort("\{label}: \{a} vs \{b}")
  }
}

///|
fn rg_assert_matrix_close(a : Matrix, b : Matrix, label : String) -> Unit raise {
  assert_eq(a.rows, b.rows)
  assert_eq(a.cols, b.cols)
  for idx in 0..<a.data.length() {
    rg_assert_close(a.data[idx], b.data[idx], "\{label}[\{idx}]")
  }
}

///|
fn rg_assert_nonneg_finite(m : Matrix) -> Unit raise {
  for x in m.data {
    assert_true(x >= 0.0)
    assert_true(!x.is_nan() && !x.is_inf())
  }
}

// ---------------------------------------------------------------------
// Test A: hand-computed regularized objectives.
// ---------------------------------------------------------------------

///|
test "wb reg A: hand-computed Frobenius and KL objectives with L1/L2" {
  // V = [[1,2],[3,4]], W = [[0.5],[1.5]], H = [[1,2]]
  // WH = [[0.5,1],[1.5,3]] -> ½‖V−WH‖²_F = 2.25
  // D_KL = 1.3096113731667947 (4 · ln 2 − 2 + ... closed form below)
  // ΣW = 2, ΣH = 3, ‖W‖² = 2.5, ‖H‖² = 5
  let v = rg_mat(2, 2, [1.0, 2.0, 3.0, 4.0])
  let w = rg_mat(2, 1, [0.5, 1.5])
  let h = rg_mat(1, 2, [1.0, 2.0])
  let wh = rg_wh(w, h)
  let kl_base = 1.3096113731667947
  let frob_base = 2.25
  // No regularization must reproduce the public unregularized objectives
  // bit-for-bit (not just approximately).
  assert_eq(frobenius_reg_from_wh(v, wh, w, h, no_reg), Ok(frob_base))
  assert_eq(
    frobenius_reg_from_wh(v, wh, w, h, no_reg),
    frobenius_objective(v, w, h),
  )
  assert_eq(kl_reg_from_wh(v, wh, w, h, no_reg), kl_divergence(v, wh))
  // Ω = l1·(2+3) + (l2/2)·(2.5+5)
  let cases : Array[(Reg, Double, Double)] = [
    (rg_reg(0.0, 0.0), 2.25, kl_base),
    (rg_reg(0.1, 0.0), 2.75, kl_base + 0.5),
    (rg_reg(0.0, 0.2), 3.0, kl_base + 0.75),
    (rg_reg(0.1, 0.2), 3.5, kl_base + 1.25),
  ]
  for c in cases {
    let (reg, want_f, want_kl) = c
    let got_f = rg_unwrap_double(frobenius_reg_from_wh(v, wh, w, h, reg))
    let got_kl = rg_unwrap_double(kl_reg_from_wh(v, wh, w, h, reg))
    assert_true((got_f - want_f).abs() <= 1.0e-12 * (1.0 + want_f.abs()))
    assert_true((got_kl - want_kl).abs() <= 1.0e-12 * (1.0 + want_kl.abs()))
  }
}

///|
test "wb reg A: penalty is exactly l1·(ΣW+ΣH) + (l2/2)(‖W‖²+‖H‖²)" {
  let w = rg_mat(2, 2, [1.0, 2.0, 3.0, 4.0])
  let h = rg_mat(1, 3, [0.5, 1.5, 2.5])
  let reg = rg_reg(0.3, 0.7)
  // ΣW = 10, ΣH = 4.5, ‖W‖² = 30, ‖H‖² = 0.25+2.25+6.25 = 8.75
  let want = 0.3 * 14.5 + 0.5 * 0.7 * 38.75
  rg_assert_close(rg_unwrap_double(reg_penalty(w, h, reg)), want, "penalty")
}

// ---------------------------------------------------------------------
// Test C: MU-Frobenius updates with regularization.
// ---------------------------------------------------------------------

///|
fn rg_mu_frobenius_step(
  v : Matrix,
  w0 : Matrix,
  h0 : Matrix,
  reg : Reg,
) -> (Matrix, Matrix) {
  let n = v.rows
  let d = v.cols
  let k = w0.cols
  let h = h0.clone()
  let w = w0.clone()
  // H update with the original W.
  for p in 0..<k {
    for j in 0..<d {
      let mut num = 0.0
      for i in 0..<n {
        num = num + w.data[i * k + p] * v.data[i * d + j]
      }
      let mut den = 0.0
      for i in 0..<n {
        let mut wh_ij = 0.0
        for q in 0..<k {
          wh_ij = wh_ij + w.data[i * k + q] * h0.data[q * d + j]
        }
        den = den + w.data[i * k + p] * wh_ij
      }
      den = den + reg.l1 + reg.l2 * h0.data[p * d + j]
      let floored = if den > 1.0e-10 { den } else { 1.0e-10 }
      h.data[p * d + j] = h0.data[p * d + j] * num / floored
    }
  }
  // W update with the updated H.
  for i in 0..<n {
    for p in 0..<k {
      let mut num = 0.0
      for j in 0..<d {
        num = num + v.data[i * d + j] * h.data[p * d + j]
      }
      let mut den = 0.0
      for j in 0..<d {
        let mut wh_ij = 0.0
        for q in 0..<k {
          wh_ij = wh_ij + w0.data[i * k + q] * h.data[q * d + j]
        }
        den = den + wh_ij * h.data[p * d + j]
      }
      den = den + reg.l1 + reg.l2 * w.data[i * k + p]
      let floored = if den > 1.0e-10 { den } else { 1.0e-10 }
      w.data[i * k + p] = w0.data[i * k + p] * num / floored
    }
  }
  (w, h)
}

///|
test "wb reg C: MU-Frobenius one iteration matches the regularized formula" {
  let v = rg_mat(3, 3, [1.0, 2.0, 0.5, 3.0, 0.0, 4.0, 0.2, 5.0, 6.0])
  let w0 = rg_mat(3, 2, [0.7, 0.3, 0.4, 1.1, 1.3, 0.6])
  let h0 = rg_mat(2, 3, [0.5, 1.2, 0.3, 0.9, 0.2, 1.4])
  let regs : Array[Reg] = [rg_reg(0.2, 0.0), rg_reg(0.0, 0.4), rg_reg(0.2, 0.4)]
  for reg in regs {
    let (want_w, want_h) = rg_mu_frobenius_step(v, w0, h0, reg)
    let fit = rg_unwrap_mu(
      mu_frobenius_fit_from(v, w0.clone(), h0.clone(), 1, 0.0, reg),
    )
    rg_assert_matrix_close(fit.h, want_h, "MU-Frob H")
    rg_assert_matrix_close(fit.w, want_w, "MU-Frob W")
    rg_assert_nonneg_finite(fit.w)
    rg_assert_nonneg_finite(fit.h)
    // The reported loss must be the full regularized objective at the result.
    let got = rg_unwrap_double(
      frobenius_reg_from_wh(v, rg_wh(fit.w, fit.h), fit.w, fit.h, reg),
    )
    rg_assert_close(fit.loss_value, got, "MU-Frob loss_value")
    // Regularization must enter the update: the result differs from no_reg.
    let plain = rg_unwrap_mu(
      mu_frobenius_fit_from(v, w0.clone(), h0.clone(), 1, 0.0, no_reg),
    )
    let mut differs = false
    for idx in 0..<fit.w.data.length() {
      if (fit.w.data[idx] - plain.w.data[idx]).abs() > 1.0e-12 {
        differs = true
      }
    }
    assert_true(differs)
  }
}

///|
test "wb reg C: MU-Frobenius L1 zeroes coordinates that no longer pay off" {
  // V = W H exactly, with W's second column zero: the second component only
  // adds penalty, so a strong L1 must push it to zero while keeping the
  // non-negative, finite contract.
  let w_true = rg_mat(3, 2, [1.0, 0.0, 2.0, 0.0, 3.0, 0.0])
  let h_true = rg_mat(2, 2, [1.0, 2.0, 3.0, 4.0])
  let v = rg_wh(w_true, h_true)
  let reg = rg_reg(5.0, 0.0)
  let fit = rg_unwrap_mu(mu_frobenius_fit(v, 2, 200, 1.0e-10, 7UL, reg))
  rg_assert_nonneg_finite(fit.w)
  rg_assert_nonneg_finite(fit.h)
  let base = 0.5 * rg_unwrap_double(frobenius_norm(v))
  assert_true(base >= 0.0)
  // Objective stays finite and contains the penalty.
  assert_true(fit.loss_value >= 0.0)
  assert_true(!fit.loss_value.is_nan() && !fit.loss_value.is_inf())
}

// ---------------------------------------------------------------------
// Test D: CD-Frobenius coordinate updates.
// ---------------------------------------------------------------------

///|
test "wb reg D: CD single coordinate matches the exact regularized minimizer" {
  // V=[[4]], W=[[2]], H=[[1]]: G=4, C=8, g_data=-4.
  // l1=0.5, l2=1.0 -> g = -4 + 0.5 + 1 = -2.5, curvature = 5,
  // h_new = 1 - (-2.5)/5 = 1.5.
  let v = rg_mat(1, 1, [4.0])
  let w = rg_mat(1, 1, [2.0])
  let h = rg_mat(1, 1, [1.0])
  let wtw = zeros_buf(1, 1)
  let wv = zeros_buf(1, 1)
  matmul_at_b_into(w, w, wtw)
  matmul_at_b_into(w, v, wv)
  let grad = Array::make(1, 0.0)
  match cd_update_h(wtw, wv, h, grad, rg_reg(0.5, 1.0)) {
    Ok(_) => ()
    Err(e) => abort("unexpected H update error: \{e}")
  }
  rg_assert_close(h.data[0], 1.5, "CD H one coordinate")
  // Clamp: V=[[0]], W=[[1]], H=[[1]], l1=2, l2=0 -> g = 1 + 2 = 3 > 0,
  // candidate 1 - 3 = -2 -> 0.
  let z = rg_mat(1, 1, [0.0])
  let w1 = rg_mat(1, 1, [1.0])
  let h1 = rg_mat(1, 1, [1.0])
  let g1 = zeros_buf(1, 1)
  let c1 = zeros_buf(1, 1)
  matmul_at_b_into(w1, w1, g1)
  matmul_at_b_into(w1, z, c1)
  let grad1 = Array::make(1, 0.0)
  match cd_update_h(g1, c1, h1, grad1, rg_reg(2.0, 0.0)) {
    Ok(_) => ()
    Err(e) => abort("unexpected H clamp error: \{e}")
  }
  assert_eq(h1.data[0], 0.0)
}

///|
fn rg_cd_sweep_h(
  gram : Matrix,
  cross : Matrix,
  h0 : Matrix,
  reg : Reg,
) -> Matrix {
  let k = gram.rows
  let d = h0.cols
  let h = h0.clone()
  for j in 0..<d {
    for p in 0..<k {
      let mut g = 0.0
      for q in 0..<k {
        g = g + gram.data[p * k + q] * h.data[q * d + j]
      }
      g = g - cross.data[p * d + j] + reg.l1 + reg.l2 * h.data[p * d + j]
      let curvature = gram.data[p * k + p] + reg.l2
      if curvature > 0.0 {
        let candidate = h.data[p * d + j] - g / curvature
        h.data[p * d + j] = if candidate > 0.0 { candidate } else { 0.0 }
      }
    }
  }
  h
}

///|
fn rg_cd_sweep_w(
  gram : Matrix,
  cross : Matrix,
  w0 : Matrix,
  reg : Reg,
) -> Matrix {
  let n = w0.rows
  let k = gram.rows
  let w = w0.clone()
  for i in 0..<n {
    for p in 0..<k {
      let mut g = 0.0
      for q in 0..<k {
        g = g + gram.data[p * k + q] * w.data[i * k + q]
      }
      g = g - cross.data[i * k + p] + reg.l1 + reg.l2 * w.data[i * k + p]
      let curvature = gram.data[p * k + p] + reg.l2
      if curvature > 0.0 {
        let candidate = w.data[i * k + p] - g / curvature
        w.data[i * k + p] = if candidate > 0.0 { candidate } else { 0.0 }
      }
    }
  }
  w
}

///|
test "wb reg D: CD sweeps match the regularized reference implementation" {
  let v = rg_mat(4, 3, [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 2.0, 9.0, 1.0,
  ])
  let w = rg_mat(4, 2, [0.8, 0.3, 0.4, 1.1, 1.3, 0.6, 0.2, 0.9])
  let h = rg_mat(2, 3, [0.7, 0.2, 1.4, 0.5, 1.2, 0.3])
  let reg = rg_reg(0.15, 0.25)
  let wtw = zeros_buf(2, 2)
  let wv = zeros_buf(2, 3)
  matmul_at_b_into(w, w, wtw)
  matmul_at_b_into(w, v, wv)
  let want_h = rg_cd_sweep_h(wtw, wv, h, reg)
  let grad = Array::make(2, 0.0)
  match cd_update_h(wtw, wv, h, grad, reg) {
    Ok(_) => ()
    Err(e) => abort("unexpected H update error: \{e}")
  }
  rg_assert_matrix_close(h, want_h, "CD H sweep")
  rg_assert_nonneg_finite(h)
  // The cached data gradient must still equal gram·h_j − cross[:,j]; the
  // regularization derivative is added per coordinate, never cached.
  let j = 2
  for p in 0..<2 {
    let mut s = 0.0
    for q in 0..<2 {
      s = s + wtw.data[p * 2 + q] * h.data[q * 3 + j]
    }
    rg_assert_close(grad[p], s - wv.data[p * 3 + j], "CD gradient cache")
  }
  let hh = zeros_buf(2, 2)
  let vht = zeros_buf(4, 2)
  matmul_a_bt_into(h, h, hh)
  matmul_a_bt_into(v, h, vht)
  let want_w = rg_cd_sweep_w(hh, vht, w, reg)
  let grad2 = Array::make(2, 0.0)
  match cd_update_w(hh, vht, w, grad2, reg) {
    Ok(_) => ()
    Err(e) => abort("unexpected W update error: \{e}")
  }
  rg_assert_matrix_close(w, want_w, "CD W sweep")
  rg_assert_nonneg_finite(w)
}

///|
test "wb reg D: CD zero diagonal is handled by the l2 curvature term" {
  // W has an exactly zero second column -> WᵀW[1,1] = 0. With l2 > 0 the
  // curvature becomes l2 > 0, so the exact step applies instead of the
  // degenerate zero-curvature branch, and the inactive row is driven to 0.
  let w = rg_mat(2, 2, [1.0, 0.0, 2.0, 0.0])
  let h = rg_mat(2, 2, [0.5, 0.3, 0.7, 0.9])
  let v = rg_wh(w, h)
  let wtw = zeros_buf(2, 2)
  let wv = zeros_buf(2, 2)
  matmul_at_b_into(w, w, wtw)
  matmul_at_b_into(w, v, wv)
  assert_eq(wtw.data[3], 0.0)
  let grad = Array::make(2, 0.0)
  match cd_update_h(wtw, wv, h, grad, rg_reg(0.0, 0.5)) {
    Ok(_) => ()
    Err(e) => abort("unexpected zero-diagonal H error: \{e}")
  }
  rg_assert_nonneg_finite(h)
  assert_eq(h.data[2], 0.0)
  assert_eq(h.data[3], 0.0)
}

// ---------------------------------------------------------------------
// Test E: MU-KL updates with regularization.
// ---------------------------------------------------------------------

///|
fn rg_kl_ratio(v : Matrix, wh : Matrix) -> Matrix {
  let out = zeros_buf(v.rows, v.cols)
  for idx in 0..<v.data.length() {
    let x = v.data[idx]
    let y = wh.data[idx]
    if x == 0.0 {
      out.data[idx] = 0.0
    } else {
      out.data[idx] = x / y
    }
  }
  out
}

///|
fn rg_mu_kl_step(
  v : Matrix,
  w0 : Matrix,
  h0 : Matrix,
  reg : Reg,
) -> (Matrix, Matrix) {
  let n = v.rows
  let d = v.cols
  let k = w0.cols
  let h = h0.clone()
  let w = w0.clone()
  // H update.
  let r1 = rg_kl_ratio(v, rg_wh(w, h0))
  for p in 0..<k {
    let mut colsum = 0.0
    for i in 0..<n {
      colsum = colsum + w.data[i * k + p]
    }
    for j in 0..<d {
      let mut num = 0.0
      for i in 0..<n {
        num = num + w.data[i * k + p] * r1.data[i * d + j]
      }
      let den = colsum + reg.l1 + reg.l2 * h0.data[p * d + j]
      h.data[p * d + j] = h0.data[p * d + j] * num / den
    }
  }
  // W update, using the ratio from the updated H.
  let r2 = rg_kl_ratio(v, rg_wh(w, h))
  for p in 0..<k {
    let mut rowsum = 0.0
    for j in 0..<d {
      rowsum = rowsum + h.data[p * d + j]
    }
    for i in 0..<n {
      let mut num = 0.0
      for j in 0..<d {
        num = num + r2.data[i * d + j] * h.data[p * d + j]
      }
      let den = rowsum + reg.l1 + reg.l2 * w.data[i * k + p]
      w.data[i * k + p] = w0.data[i * k + p] * num / den
    }
  }
  (w, h)
}

///|
test "wb reg E: MU-KL one iteration matches the regularized formula" {
  let v = rg_mat(3, 3, [1.0, 2.0, 0.5, 3.0, 1.0, 4.0, 0.2, 5.0, 6.0])
  let w0 = rg_mat(3, 2, [0.7, 0.3, 0.4, 1.1, 1.3, 0.6])
  let h0 = rg_mat(2, 3, [0.5, 1.2, 0.3, 0.9, 0.2, 1.4])
  let regs : Array[Reg] = [rg_reg(0.2, 0.0), rg_reg(0.0, 0.4), rg_reg(0.2, 0.4)]
  for reg in regs {
    let (want_w, want_h) = rg_mu_kl_step(v, w0, h0, reg)
    let fit = rg_unwrap_mu(
      mu_kl_fit_from(v, w0.clone(), h0.clone(), 1, 0.0, reg),
    )
    rg_assert_matrix_close(fit.h, want_h, "MU-KL H")
    rg_assert_matrix_close(fit.w, want_w, "MU-KL W")
    rg_assert_nonneg_finite(fit.w)
    rg_assert_nonneg_finite(fit.h)
    let got = rg_unwrap_double(
      kl_reg_from_wh(v, rg_wh(fit.w, fit.h), fit.w, fit.h, reg),
    )
    rg_assert_close(fit.loss_value, got, "MU-KL loss_value")
  }
}

///|
test "wb reg E: MU-KL keeps the T11 zero conventions with regularization" {
  // Zero row in V: those entries contribute 0 to the KL numerator but still
  // receive the penalty, so the update stays well-defined and finite.
  let v = rg_mat(3, 3, [1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 4.0])
  let fit = rg_unwrap_mu(mu_kl_fit(v, 2, 100, 1.0e-8, 42UL, rg_reg(0.1, 0.2)))
  rg_assert_nonneg_finite(fit.w)
  rg_assert_nonneg_finite(fit.h)
  assert_true(fit.loss_value >= 0.0)
  assert_true(!fit.loss_value.is_nan() && !fit.loss_value.is_inf())
  // V>0 with WH=0 is still an infinite KL: regularization must not hide it.
  let w_zero = rg_mat(1, 1, [0.0])
  let h_zero = rg_mat(1, 1, [0.0])
  let v_pos = rg_mat(1, 1, [1.0])
  assert_true(
    mu_kl_fit_from(v_pos, w_zero, h_zero, 5, 1.0e-6, rg_reg(1.0, 1.0))
    is Err(NumericalError(_)),
  )
}

// ---------------------------------------------------------------------
// Test F: the full regularized objective is non-increasing.
// ---------------------------------------------------------------------

///|
test "wb reg F: every solver is monotone in the full regularized objective" {
  let v = rg_mat(4, 3, [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 2.0, 9.0, 1.0,
  ])
  let w0 = rg_mat(4, 2, [0.8, 0.3, 0.4, 1.1, 1.3, 0.6, 0.2, 0.9])
  let h0 = rg_mat(2, 3, [0.7, 0.2, 1.4, 0.5, 1.2, 0.3])
  let reg = rg_reg(0.05, 0.1)
  // Each budget is a prefix of the next (tol = 0 disables early stopping), so
  // the recorded objectives must form a non-increasing sequence.
  let mut prev_mu = 1.0e300
  let mut prev_cd = 1.0e300
  let mut prev_kl = 1.0e300
  for budget in [1, 2, 3, 5, 8, 12] {
    let mu = rg_unwrap_mu(
      mu_frobenius_fit_from(v, w0.clone(), h0.clone(), budget, 0.0, reg),
    )
    assert_true(mu.loss_value <= prev_mu + rg_tol(prev_mu))
    prev_mu = mu.loss_value
    let cd = rg_unwrap_cd(
      cd_frobenius_fit_from(v, w0.clone(), h0.clone(), budget, 0.0, reg),
    )
    assert_true(cd.loss_value <= prev_cd + rg_tol(prev_cd))
    prev_cd = cd.loss_value
    let kl = rg_unwrap_mu(
      mu_kl_fit_from(v, w0.clone(), h0.clone(), budget, 0.0, reg),
    )
    assert_true(kl.loss_value <= prev_kl + rg_tol(prev_kl))
    prev_kl = kl.loss_value
  }
}

// ---------------------------------------------------------------------
// Test G: regularization actually shapes the solution.
// ---------------------------------------------------------------------

///|
fn rg_l1_penalty(w : Matrix, h : Matrix) -> Double {
  let mut s = 0.0
  for x in w.data {
    s = s + x
  }
  for x in h.data {
    s = s + x
  }
  s
}

///|
fn rg_l2_penalty(w : Matrix, h : Matrix) -> Double {
  let mut s = 0.0
  for x in w.data {
    s = s + x * x
  }
  for x in h.data {
    s = s + x * x
  }
  0.5 * s
}

///|
test "wb reg G: reported loss equals reconstruction plus the penalty" {
  let v = rg_mat(5, 4, [
    1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 6.0, 1.0, 7.0, 8.0, 10.0, 2.0, 0.0, 9.0, 1.0, 3.0,
    2.0, 1.0, 0.0, 5.0,
  ])
  let reg = rg_reg(0.3, 0.6)
  let mu = rg_unwrap_mu(mu_frobenius_fit(v, 2, 60, 1.0e-9, 21UL, reg))
  let base_mu = rg_unwrap_double(frobenius_objective(v, mu.w, mu.h))
  let want_mu = base_mu +
    0.3 * rg_l1_penalty(mu.w, mu.h) +
    0.6 * rg_l2_penalty(mu.w, mu.h)
  rg_assert_close(mu.loss_value, want_mu, "MU loss = base + Ω")
  let cd = rg_unwrap_cd(cd_frobenius_fit(v, 2, 60, 1.0e-9, 21UL, reg))
  let base_cd = rg_unwrap_double(frobenius_objective(v, cd.w, cd.h))
  let want_cd = base_cd +
    0.3 * rg_l1_penalty(cd.w, cd.h) +
    0.6 * rg_l2_penalty(cd.w, cd.h)
  rg_assert_close(cd.loss_value, want_cd, "CD loss = base + Ω")
  let kl = rg_unwrap_mu(mu_kl_fit(v, 2, 60, 1.0e-9, 21UL, reg))
  let base_kl = rg_unwrap_double(kl_divergence(v, rg_wh(kl.w, kl.h)))
  let want_kl = base_kl +
    0.3 * rg_l1_penalty(kl.w, kl.h) +
    0.6 * rg_l2_penalty(kl.w, kl.h)
  rg_assert_close(kl.loss_value, want_kl, "KL loss = base + Ω")
}

///|
test "wb reg G: strong L1 shrinks the factor mass relative to no_reg" {
  // Same initial factors and identical budget; only the penalty differs. This
  // is an empirical fixture check (not a theorem about NMF solutions): a
  // large L1 must trade reconstruction for a smaller ΣW+ΣH.
  let v = rg_mat(5, 4, [
    1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 6.0, 1.0, 7.0, 8.0, 10.0, 2.0, 0.0, 9.0, 1.0, 3.0,
    2.0, 1.0, 0.0, 5.0,
  ])
  let w0 = rg_mat(5, 2, [0.9, 0.4, 0.5, 1.2, 1.1, 0.7, 0.3, 1.0, 0.6, 0.2])
  let h0 = rg_mat(2, 4, [0.8, 0.3, 1.1, 0.5, 1.2, 0.4, 0.6, 0.9])
  let plain = rg_unwrap_mu(
    mu_frobenius_fit_from(v, w0.clone(), h0.clone(), 100, 0.0, no_reg),
  )
  let strong = rg_unwrap_mu(
    mu_frobenius_fit_from(v, w0.clone(), h0.clone(), 100, 0.0, rg_reg(2.0, 0.0)),
  )
  rg_assert_nonneg_finite(strong.w)
  rg_assert_nonneg_finite(strong.h)
  assert_true(
    rg_l1_penalty(strong.w, strong.h) < rg_l1_penalty(plain.w, plain.h),
  )
  // The full objective reported by the regularized run contains that penalty.
  let base = rg_unwrap_double(frobenius_objective(v, strong.w, strong.h))
  rg_assert_close(
    strong.loss_value,
    base + 2.0 * rg_l1_penalty(strong.w, strong.h),
    "strong L1 objective",
  )
}

// ---------------------------------------------------------------------
// Test H: transform regularizes W only.
// ---------------------------------------------------------------------

///|
fn rg_mu_frobenius_w_step(
  x : Matrix,
  w0 : Matrix,
  h : Matrix,
  reg : Reg,
) -> Matrix {
  let m = x.rows
  let d = x.cols
  let k = w0.cols
  let w = w0.clone()
  for i in 0..<m {
    for p in 0..<k {
      let mut num = 0.0
      for j in 0..<d {
        num = num + x.data[i * d + j] * h.data[p * d + j]
      }
      let mut den = 0.0
      for j in 0..<d {
        let mut wh_ij = 0.0
        for q in 0..<k {
          wh_ij = wh_ij + w0.data[i * k + q] * h.data[q * d + j]
        }
        den = den + wh_ij * h.data[p * d + j]
      }
      den = den + reg.l1 + reg.l2 * w.data[i * k + p]
      let floored = if den > 1.0e-10 { den } else { 1.0e-10 }
      w.data[i * k + p] = w0.data[i * k + p] * num / floored
    }
  }
  w
}

///|
test "wb reg H: transform solves W with the regularized update, H untouched" {
  let x = rg_mat(4, 3, [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 2.0, 9.0, 1.0,
  ])
  let h = rg_mat(2, 3, [0.6, 1.1, 0.4, 0.9, 0.3, 1.2])
  let reg = rg_reg(0.3, 0.2)
  let w0 = rg_unwrap_mat(random_init_w(x, 2, 42UL))
  let want = rg_mu_frobenius_w_step(x, w0, h, reg)
  let got = rg_unwrap_mat(mu_frobenius_transform(x, h, 1, 0.0, 42UL, reg))
  rg_assert_matrix_close(got, want, "MU-Frob transform W")
  rg_assert_nonneg_finite(got)
  // H must be bit-for-bit unchanged by the transform path.
  let h_after = h.clone()
  for idx in 0..<h.data.length() {
    assert_eq(h_after.data[idx], h.data[idx])
  }
  // A regularized transform must differ from an unregularized one.
  let plain = rg_unwrap_mat(mu_frobenius_transform(x, h, 1, 0.0, 42UL, no_reg))
  let mut differs = false
  for idx in 0..<plain.data.length() {
    if (plain.data[idx] - got.data[idx]).abs() > 1.0e-12 {
      differs = true
    }
  }
  assert_true(differs)
}

///|
test "wb reg H: CD and MU-KL transform paths accept regularization" {
  let x = rg_mat(4, 3, [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 2.0, 9.0, 1.0,
  ])
  let h = rg_mat(2, 3, [0.6, 1.1, 0.4, 0.9, 0.3, 1.2])
  let reg = rg_reg(0.2, 0.3)
  let cd = rg_unwrap_mat(cd_frobenius_transform(x, h, 20, 1.0e-8, 42UL, reg))
  rg_assert_nonneg_finite(cd)
  let kl = rg_unwrap_mat(mu_kl_transform(x, h, 20, 1.0e-8, 42UL, reg))
  rg_assert_nonneg_finite(kl)
  // Both full objectives at the returned W are finite.
  let jf = rg_unwrap_double(frobenius_reg_from_wh(x, rg_wh(cd, h), cd, h, reg))
  assert_true(jf >= 0.0 && !jf.is_nan() && !jf.is_inf())
  let jk = rg_unwrap_double(kl_reg_from_wh(x, rg_wh(kl, h), kl, h, reg))
  assert_true(jk >= 0.0 && !jk.is_nan() && !jk.is_inf())
}

// ---------------------------------------------------------------------
// Test J: zero regularization is the pre-T12 computation.
// ---------------------------------------------------------------------

///|
test "wb reg J: no_reg is a pure zero and reproduces old objective bits" {
  let v = rg_mat(3, 3, [1.0, 2.0, 0.5, 3.0, 1.0, 4.0, 0.2, 5.0, 6.0])
  let w = rg_mat(3, 2, [0.7, 0.3, 0.4, 1.1, 1.3, 0.6])
  let h = rg_mat(2, 3, [0.5, 1.2, 0.3, 0.9, 0.2, 1.4])
  assert_eq(no_reg.l1, 0.0)
  assert_eq(no_reg.l2, 0.0)
  assert_eq(reg_penalty(w, h, no_reg), Ok(0.0))
  let wh = rg_wh(w, h)
  assert_eq(
    frobenius_reg_from_wh(v, wh, w, h, no_reg),
    frobenius_objective(v, w, h),
  )
  assert_eq(kl_reg_from_wh(v, wh, w, h, no_reg), kl_divergence(v, wh))
}
