// Whitebox tests for the CD-Frobenius solver: exact coordinate updates,
// objective monotonicity, numerical edge cases, API integration, transform,
// reproducibility, and a fair MU-vs-CD comparison (same seed => same random
// initial factors).

///|
fn cd_mat(rows : Int32, cols : Int32, data : Array[Double]) -> Matrix {
  match Matrix::new(rows, cols, data) {
    Ok(m) => m
    Err(e) => abort("unexpected matrix error: \{e}")
  }
}

///|
fn cd_unwrap_matrix(r : Result[Matrix, NMFError]) -> Matrix {
  match r {
    Ok(m) => m
    Err(e) => abort("unexpected matrix error: \{e}")
  }
}

///|
fn cd_unwrap_fit(r : Result[CdFit, NMFError]) -> CdFit {
  match r {
    Ok(f) => f
    Err(e) => abort("unexpected CD fit error: \{e}")
  }
}

///|
fn cd_unwrap_mu(r : Result[MuFit, NMFError]) -> MuFit {
  match r {
    Ok(f) => f
    Err(e) => abort("unexpected MU fit error: \{e}")
  }
}

///|
fn cd_objective(v : Matrix, w : Matrix, h : Matrix) -> Double {
  match frobenius_objective(v, w, h) {
    Ok(f) => f
    Err(e) => abort("unexpected objective error: \{e}")
  }
}

///|
fn cd_numerical_tol(f : Double) -> Double {
  1.0e-10 * (1.0 + f.abs())
}

///|
fn cd_assert_factor_ok(m : Matrix, rows : Int32, cols : Int32) -> Unit raise {
  assert_eq(m.rows, rows)
  assert_eq(m.cols, cols)
  for x in m.data {
    assert_true(x >= 0.0)
    assert_true(!x.is_nan() && !x.is_inf())
  }
}

///|
test "wb cd: single coordinate matches the exact line minimizer" {
  // H side: V=[[4]], W=[[2]], H=[[1]]. G=4, C=8, grad=-4,
  // h_new = 1 - (-4)/4 = 2.
  let v = cd_mat(1, 1, [4.0])
  let w = cd_mat(1, 1, [2.0])
  let h = cd_mat(1, 1, [1.0])
  let wtw = zeros_buf(1, 1)
  let wv = zeros_buf(1, 1)
  matmul_at_b_into(w, w, wtw)
  matmul_at_b_into(w, v, wv)
  let grad = Array::make(1, 0.0)
  match cd_update_h(wtw, wv, h, grad, no_reg) {
    Ok(_) => ()
    Err(e) => abort("unexpected H update error: \{e}")
  }
  assert_true((h.data[0] - 2.0).abs() < 1.0e-12)
  // W side: V=[[4]], W=[[1]], H=[[2]]. G=4, C=8, grad=-4,
  // w_new = 1 - (-4)/4 = 2.
  let w2 = cd_mat(1, 1, [1.0])
  let h2 = cd_mat(1, 1, [2.0])
  let hh = zeros_buf(1, 1)
  let vht = zeros_buf(1, 1)
  matmul_a_bt_into(h2, h2, hh)
  matmul_a_bt_into(v, h2, vht)
  let grad2 = Array::make(1, 0.0)
  match cd_update_w(hh, vht, w2, grad2, no_reg) {
    Ok(_) => ()
    Err(e) => abort("unexpected W update error: \{e}")
  }
  assert_true((w2.data[0] - 2.0).abs() < 1.0e-12)
}

///|
test "wb cd: H and W sweeps preserve dims, non-negativity, and finiteness" {
  let v = cd_mat(3, 2, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  let w = cd_mat(3, 2, [1.0, 0.5, 0.2, 1.0, 0.3, 0.4])
  let h = cd_mat(2, 2, [0.5, 0.2, 0.1, 0.3])
  let h_before = h.data.copy()
  let wtw = zeros_buf(2, 2)
  let wv = zeros_buf(2, 2)
  matmul_at_b_into(w, w, wtw)
  matmul_at_b_into(w, v, wv)
  let grad = Array::make(2, 0.0)
  match cd_update_h(wtw, wv, h, grad, no_reg) {
    Ok(_) => ()
    Err(e) => abort("unexpected H update error: \{e}")
  }
  cd_assert_factor_ok(h, 2, 2)
  let mut h_changed = false
  for idx in 0..<h.data.length() {
    if h.data[idx] != h_before[idx] {
      h_changed = true
    }
  }
  assert_true(h_changed)
  // The cached gradient must still match gram·h_j − cross[:,j] for the last
  // column after all incremental updates.
  let j = 1
  for p in 0..<2 {
    let mut s = 0.0
    for q in 0..<2 {
      s = s + wtw.data[p * 2 + q] * h.data[q * 2 + j]
    }
    let expected = s - wv.data[p * 2 + j]
    assert_true((grad[p] - expected).abs() < 1.0e-12)
  }
  let w_before = w.data.copy()
  let hh = zeros_buf(2, 2)
  let vht = zeros_buf(3, 2)
  matmul_a_bt_into(h, h, hh)
  matmul_a_bt_into(v, h, vht)
  let grad2 = Array::make(2, 0.0)
  match cd_update_w(hh, vht, w, grad2, no_reg) {
    Ok(_) => ()
    Err(e) => abort("unexpected W update error: \{e}")
  }
  cd_assert_factor_ok(w, 3, 2)
  let mut w_changed = false
  for idx in 0..<w.data.length() {
    if w.data[idx] != w_before[idx] {
      w_changed = true
    }
  }
  assert_true(w_changed)
}

///|
test "wb cd: H and W sweeps do not increase the objective" {
  let v = cd_mat(4, 3, [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 2.0, 9.0, 1.0,
  ])
  let w = cd_mat(4, 2, [0.8, 0.3, 0.4, 1.1, 1.3, 0.6, 0.2, 0.9])
  let h = cd_mat(2, 3, [0.7, 0.2, 1.4, 0.5, 1.2, 0.3])
  let f0 = cd_objective(v, w, h)
  let wtw = zeros_buf(2, 2)
  let wv = zeros_buf(2, 3)
  matmul_at_b_into(w, w, wtw)
  matmul_at_b_into(w, v, wv)
  let grad = Array::make(2, 0.0)
  match cd_update_h(wtw, wv, h, grad, no_reg) {
    Ok(_) => ()
    Err(e) => abort("unexpected H update error: \{e}")
  }
  let f1 = cd_objective(v, w, h)
  assert_true(f1 <= f0 + cd_numerical_tol(f0))
  let hh = zeros_buf(2, 2)
  let vht = zeros_buf(4, 2)
  matmul_a_bt_into(h, h, hh)
  matmul_a_bt_into(v, h, vht)
  let grad2 = Array::make(2, 0.0)
  match cd_update_w(hh, vht, w, grad2, no_reg) {
    Ok(_) => ()
    Err(e) => abort("unexpected W update error: \{e}")
  }
  let f2 = cd_objective(v, w, h)
  assert_true(f2 <= f1 + cd_numerical_tol(f1))
}

///|
test "wb cd: full iterations do not increase the objective across budgets" {
  let v = cd_mat(4, 3, [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 2.0, 9.0, 1.0,
  ])
  // tol=0 forces the full budget, so each run is a prefix of the next.
  let mut prev = 1.0e300
  for budget in [1, 5, 20, 100] {
    let fit = cd_unwrap_fit(cd_frobenius_fit(v, 2, budget, 0.0, 11UL, no_reg))
    assert_true(fit.loss_value <= prev + cd_numerical_tol(prev))
    prev = fit.loss_value
  }
}

///|
test "wb cd: NMF API end-to-end contract" {
  let v = cd_mat(4, 3, [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 2.0, 9.0, 1.0,
  ])
  let config = NMFConfig::new(
    n_components=2,
    solver=Solver::CD,
    loss=Loss::Frobenius,
    max_iter=300,
    tol=1.0e-6,
    seed=42UL,
  )
  let model = NMF::new(config)
  assert_eq(model.fit(v), Ok(()))
  let w = cd_unwrap_matrix(model.w())
  let h = cd_unwrap_matrix(model.h())
  cd_assert_factor_ok(w, 4, 2)
  cd_assert_factor_ok(h, 2, 3)
  let loss = match model.loss_value() {
    Ok(x) => x
    Err(e) => abort("unexpected: \{e}")
  }
  assert_eq(frobenius_objective(v, w, h), Ok(loss))
  let err = match model.reconstruction_error(v) {
    Ok(x) => x
    Err(e) => abort("unexpected: \{e}")
  }
  assert_true((err - (2.0 * loss).sqrt()).abs() < 1.0e-9)
  let n_iter = match model.n_iter() {
    Ok(n) => n
    Err(e) => abort("unexpected: \{e}")
  }
  let converged = match model.converged() {
    Ok(c) => c
    Err(e) => abort("unexpected: \{e}")
  }
  assert_true(n_iter >= 1 && n_iter <= 300)
  if !converged {
    assert_eq(n_iter, 300)
  }
  // fit_transform is a full fit followed by w(); same config and seed must
  // return exactly the same factors.
  let model2 = NMF::new(config)
  let w_ft = cd_unwrap_matrix(model2.fit_transform(v))
  assert_eq(w_ft.rows, w.rows)
  assert_eq(w_ft.cols, w.cols)
  for idx in 0..<w.data.length() {
    assert_eq(w_ft.data[idx], w.data[idx])
  }
  // inverse_transform(W) = W·H.
  let recon = cd_unwrap_matrix(model.inverse_transform(w))
  for i in 0..<4 {
    for j in 0..<3 {
      let expected = w.data[i * 2] * h.data[j] +
        w.data[i * 2 + 1] * h.data[3 + j]
      assert_true((recon.data[i * 3 + j] - expected).abs() < 1.0e-12)
    }
  }
}

///|
test "wb cd: known low-rank product is reconstructed accurately" {
  let w_true = cd_mat(5, 2, [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 0.0, 0.0, 2.0])
  let h_true = cd_mat(2, 4, [1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 3.0, 2.0])
  let v = cd_unwrap_matrix(matmul(w_true, h_true))
  let fit = cd_unwrap_fit(cd_frobenius_fit(v, 2, 2000, 1.0e-10, 42UL, no_reg))
  cd_assert_factor_ok(fit.w, 5, 2)
  cd_assert_factor_ok(fit.h, 2, 4)
  let wh = cd_unwrap_matrix(matmul(fit.w, fit.h))
  let mut s = 0.0
  for idx in 0..<v.data.length() {
    let diff = v.data[idx] - wh.data[idx]
    s = s + diff * diff
  }
  let err = s.sqrt()
  let v_norm = match frobenius_norm(v) {
    Ok(x) => x
    Err(e) => abort("unexpected: \{e}")
  }
  assert_true(err / v_norm < 1.0e-2)
}

///|
test "wb cd: all-zero and degenerate inputs stay finite" {
  // All-zero V: objective is 0 and the result must be exact and honest.
  let z = zeros_buf(3, 4)
  let zfit = cd_unwrap_fit(cd_frobenius_fit(z, 2, 50, 1.0e-4, 42UL, no_reg))
  cd_assert_factor_ok(zfit.w, 3, 2)
  cd_assert_factor_ok(zfit.h, 2, 4)
  assert_eq(zfit.loss_value, 0.0)
  assert_true(zfit.converged)
  // Zero row and zero column in V.
  let v = cd_mat(3, 3, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0])
  let fit = cd_unwrap_fit(cd_frobenius_fit(v, 2, 200, 1.0e-8, 7UL, no_reg))
  cd_assert_factor_ok(fit.w, 3, 2)
  cd_assert_factor_ok(fit.h, 2, 3)
  let err = cd_objective(v, fit.w, fit.h)
  assert_true(err >= 0.0)
  assert_true(!err.is_nan() && !err.is_inf())
  // Highly correlated (exactly redundant) components: k=2 on rank-1 data.
  let w_corr = cd_mat(4, 2, [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0])
  let h_corr = cd_mat(2, 3, [1.0, 2.0, 1.0, 0.0, 1.0, 2.0])
  let v_corr = cd_unwrap_matrix(matmul(w_corr, h_corr))
  let cfit = cd_unwrap_fit(
    cd_frobenius_fit(v_corr, 2, 500, 1.0e-8, 13UL, no_reg),
  )
  cd_assert_factor_ok(cfit.w, 4, 2)
  cd_assert_factor_ok(cfit.h, 2, 3)
  let objective = cd_objective(v_corr, cfit.w, cfit.h)
  assert_true(
    (2.0 * objective).sqrt() / cd_unwrap_double(frobenius_norm(v_corr)) < 1.0e-2,
  )
}

///|
test "wb cd: zero curvature coordinates are handled without division" {
  // W has an exactly zero second column, so WᵀW has a zero diagonal entry.
  let w = cd_mat(2, 2, [1.0, 0.0, 2.0, 0.0])
  let h = cd_mat(2, 2, [0.5, 0.3, 0.7, 0.9])
  let v = cd_unwrap_matrix(matmul(w, h))
  let f0 = cd_objective(v, w, h)
  let wtw = zeros_buf(2, 2)
  let wv = zeros_buf(2, 2)
  matmul_at_b_into(w, w, wtw)
  matmul_at_b_into(w, v, wv)
  let grad = Array::make(2, 0.0)
  match cd_update_h(wtw, wv, h, grad, no_reg) {
    Ok(_) => ()
    Err(e) => abort("unexpected H update error: \{e}")
  }
  cd_assert_factor_ok(h, 2, 2)
  assert_eq(h.data[2], 0.0)
  assert_eq(h.data[3], 0.0)
  assert_true(cd_objective(v, w, h) <= f0 + cd_numerical_tol(f0))
  // H has an exactly zero second row, so HHᵀ has a zero diagonal entry.
  let h2 = cd_mat(2, 2, [1.0, 2.0, 0.0, 0.0])
  let w2 = cd_mat(2, 2, [1.0, 1.0, 1.0, 1.0])
  let v2 = cd_unwrap_matrix(matmul(w2, h2))
  let f2 = cd_objective(v2, w2, h2)
  let hh = zeros_buf(2, 2)
  let vht = zeros_buf(2, 2)
  matmul_a_bt_into(h2, h2, hh)
  matmul_a_bt_into(v2, h2, vht)
  let grad2 = Array::make(2, 0.0)
  match cd_update_w(hh, vht, w2, grad2, no_reg) {
    Ok(_) => ()
    Err(e) => abort("unexpected W update error: \{e}")
  }
  cd_assert_factor_ok(w2, 2, 2)
  assert_eq(w2.data[1], 0.0)
  assert_eq(w2.data[3], 0.0)
  assert_true(cd_objective(v2, w2, h2) <= f2 + cd_numerical_tol(f2))
}

///|
test "wb cd: tiny positive values stay finite" {
  let v = cd_mat(2, 2, [1.0e-100, 2.0e-100, 3.0e-100, 4.0e-100])
  let fit = cd_unwrap_fit(cd_frobenius_fit(v, 1, 20, 1.0e-4, 5UL, no_reg))
  cd_assert_factor_ok(fit.w, 2, 1)
  cd_assert_factor_ok(fit.h, 1, 2)
  assert_true(fit.loss_value >= 0.0)
  assert_true(!fit.loss_value.is_nan() && !fit.loss_value.is_inf())
}

///|
test "wb cd: small budgets and illegal configs are honest errors" {
  let v = cd_mat(4, 3, [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 2.0, 9.0, 1.0,
  ])
  let one = NMF::new(
    NMFConfig::new(
      n_components=2,
      solver=Solver::CD,
      max_iter=1,
      tol=0.0,
      seed=42UL,
    ),
  )
  assert_eq(one.fit(v), Ok(()))
  assert_eq(one.n_iter(), Ok(1))
  assert_eq(one.converged(), Ok(false))
  let loss = match one.loss_value() {
    Ok(x) => x
    Err(e) => abort("unexpected: \{e}")
  }
  assert_true(loss >= 0.0)
  assert_true(!loss.is_nan() && !loss.is_inf())
  // CD + KL is an explicit unsupported combination.
  let cd_kl = NMF::new(
    NMFConfig::new(n_components=1, solver=Solver::CD, loss=Loss::KLDivergence),
  )
  assert_true(cd_kl.fit(v) is Err(InvalidConfig(_)))
  assert_true(cd_kl.w() is Err(NotFitted))
  // Illegal inputs are rejected and leave the model untrained.
  let neg = cd_mat(2, 2, [1.0, 2.0, 3.0, -0.5])
  let neg_model = NMF::new(NMFConfig::new(n_components=1, solver=Solver::CD))
  assert_true(neg_model.fit(neg) is Err(NMFError::NegativeEntry(1, 1)))
  assert_true(neg_model.w() is Err(NotFitted))
  let rank_model = NMF::new(NMFConfig::new(n_components=4, solver=Solver::CD))
  assert_true(rank_model.fit(v) is Err(InvalidRank(4)))
  let empty = cd_unwrap_matrix(Matrix::zeros(0, 3))
  let empty_model = NMF::new(NMFConfig::new(n_components=1, solver=Solver::CD))
  assert_true(empty_model.fit(empty) is Err(InvalidMatrixShape(_)))
}

///|
test "wb cd: same seed reproduces initialization and training exactly" {
  let v = cd_mat(4, 3, [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 2.0, 9.0, 1.0,
  ])
  // Initialization is deterministic for a fixed seed.
  let (w1, h1) = match random_init(v, 2, 2026UL) {
    Ok(pair) => pair
    Err(e) => abort("unexpected init error: \{e}")
  }
  let (w2, h2) = match random_init(v, 2, 2026UL) {
    Ok(pair) => pair
    Err(e) => abort("unexpected init error: \{e}")
  }
  for idx in 0..<w1.data.length() {
    assert_eq(w1.data[idx], w2.data[idx])
  }
  for idx in 0..<h1.data.length() {
    assert_eq(h1.data[idx], h2.data[idx])
  }
  let config = NMFConfig::new(
    n_components=2,
    solver=Solver::CD,
    max_iter=40,
    tol=1.0e-6,
    seed=2026UL,
  )
  let m1 = NMF::new(config)
  let m2 = NMF::new(config)
  assert_eq(m1.fit(v), Ok(()))
  assert_eq(m2.fit(v), Ok(()))
  let wa = cd_unwrap_matrix(m1.w())
  let wb = cd_unwrap_matrix(m2.w())
  let ha = cd_unwrap_matrix(m1.h())
  let hb = cd_unwrap_matrix(m2.h())
  for idx in 0..<wa.data.length() {
    assert_eq(wa.data[idx], wb.data[idx])
  }
  for idx in 0..<ha.data.length() {
    assert_eq(ha.data[idx], hb.data[idx])
  }
  assert_eq(m1.n_iter(), m2.n_iter())
  assert_eq(m1.loss_value(), m2.loss_value())
  assert_eq(m1.converged(), m2.converged())
  let m3 = NMF::new(
    NMFConfig::new(
      n_components=2,
      solver=Solver::CD,
      max_iter=40,
      tol=1.0e-6,
      seed=2027UL,
    ),
  )
  assert_eq(m3.fit(v), Ok(()))
  let wc = cd_unwrap_matrix(m3.w())
  let mut differs = false
  for idx in 0..<wa.data.length() {
    if wa.data[idx] != wc.data[idx] {
      differs = true
    }
  }
  assert_true(differs)
}

///|
test "wb cd: transform fixes H, optimizes W with CD, and never mutates W" {
  let v = cd_mat(3, 2, [2.0, 1.0, 4.0, 2.0, 6.0, 3.0])
  let config = NMFConfig::new(
    n_components=1,
    solver=Solver::CD,
    max_iter=500,
    tol=1.0e-10,
    seed=42UL,
  )
  let model = NMF::new(config)
  assert_eq(model.fit(v), Ok(()))
  let h = cd_unwrap_matrix(model.h())
  let w_before = cd_unwrap_matrix(model.w())
  let x = cd_mat(2, 2, [4.0, 2.0, 8.0, 4.0])
  let w_new = cd_unwrap_matrix(model.transform(x))
  cd_assert_factor_ok(w_new, 2, 1)
  let w_after = cd_unwrap_matrix(model.w())
  for idx in 0..<w_before.data.length() {
    assert_eq(w_before.data[idx], w_after.data[idx])
  }
  // Public dispatch must use the CD transform, not MU.
  let direct = cd_unwrap_matrix(
    cd_frobenius_transform(x, h, 500, 1.0e-10, 42UL, no_reg),
  )
  for idx in 0..<w_new.data.length() {
    assert_eq(w_new.data[idx], direct.data[idx])
  }
  let f = cd_objective(x, w_new, h)
  let x_norm = cd_unwrap_double(frobenius_norm(x))
  assert_true((2.0 * f).sqrt() / x_norm < 1.0e-4)
  // All-zero new data must stay finite.
  let zero_x = cd_mat(2, 2, [0.0, 0.0, 0.0, 0.0])
  let zero_w = cd_unwrap_matrix(model.transform(zero_x))
  cd_assert_factor_ok(zero_w, 2, 1)
  // inverse_transform and reconstruction_error stay consistent.
  let recon = cd_unwrap_matrix(model.inverse_transform(w_new))
  assert_eq(recon.rows, 2)
  assert_eq(recon.cols, 2)
  let err = match model.reconstruction_error(v) {
    Ok(x) => x
    Err(e) => abort("unexpected: \{e}")
  }
  let loss = match model.loss_value() {
    Ok(x) => x
    Err(e) => abort("unexpected: \{e}")
  }
  assert_true((err - (2.0 * loss).sqrt()).abs() < 1.0e-9)
}

///|
test "wb cd: MU and CD start from identical factors on the same seed" {
  let v = cd_mat(5, 4, [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 2.0, 9.0, 1.0, 3.0, 4.0, 6.0, 2.0,
    8.0, 1.0, 2.0, 3.0,
  ])
  let seed = 99UL
  let (w0, h0) = match random_init(v, 2, seed) {
    Ok(pair) => pair
    Err(e) => abort("unexpected init error: \{e}")
  }
  let f0 = cd_objective(v, w0, h0)
  let mu = cd_unwrap_mu(mu_frobenius_fit(v, 2, 200, 1.0e-8, seed, no_reg))
  let cd = cd_unwrap_fit(cd_frobenius_fit(v, 2, 200, 1.0e-8, seed, no_reg))
  cd_assert_factor_ok(mu.w, 5, 2)
  cd_assert_factor_ok(mu.h, 2, 4)
  cd_assert_factor_ok(cd.w, 5, 2)
  cd_assert_factor_ok(cd.h, 2, 4)
  assert_true(mu.loss_value <= f0 + cd_numerical_tol(f0))
  assert_true(cd.loss_value <= f0 + cd_numerical_tol(f0))
  assert_eq(frobenius_objective(v, mu.w, mu.h), Ok(mu.loss_value))
  assert_eq(frobenius_objective(v, cd.w, cd.h), Ok(cd.loss_value))
  println(
    "MU: loss=\{mu.loss_value}, n_iter=\{mu.n_iter}, converged=\{mu.converged}",
  )
  println(
    "CD: loss=\{cd.loss_value}, n_iter=\{cd.n_iter}, converged=\{cd.converged}",
  )
}

///|
fn cd_unwrap_double(x : Result[Double, NMFError]) -> Double {
  match x {
    Ok(v) => v
    Err(e) => abort("unexpected: \{e}")
  }
}
