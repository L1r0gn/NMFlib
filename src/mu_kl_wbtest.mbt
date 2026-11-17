// Whitebox tests for MU-KL: ratio/zero handling, updates, API, transform,
// initialization compatibility, reproducibility, and Frobenius comparison.

///|
fn klw_mat(rows : Int32, cols : Int32, data : Array[Double]) -> Matrix {
  match Matrix::new(rows, cols, data) {
    Ok(m) => m
    Err(e) => abort("unexpected matrix error: \{e}")
  }
}

///|
fn klw_unwrap_fit(r : Result[MuFit, NMFError]) -> MuFit {
  match r {
    Ok(f) => f
    Err(e) => abort("unexpected KL fit error: \{e}")
  }
}

///|
fn klw_unwrap_matrix(r : Result[Matrix, NMFError]) -> Matrix {
  match r {
    Ok(m) => m
    Err(e) => abort("unexpected matrix error: \{e}")
  }
}

///|
fn klw_unwrap_double(r : Result[Double, NMFError]) -> Double {
  match r {
    Ok(x) => x
    Err(e) => abort("unexpected: \{e}")
  }
}

///|
fn klw_wh(w : Matrix, h : Matrix) -> Matrix {
  match matmul(w, h) {
    Ok(m) => m
    Err(e) => abort("unexpected matmul error: \{e}")
  }
}

///|
fn klw_objective(v : Matrix, w : Matrix, h : Matrix) -> Double {
  let wh = klw_wh(w, h)
  match kl_divergence(v, wh) {
    Ok(x) => x
    Err(e) => abort("unexpected KL error: \{e}")
  }
}

///|
fn klw_tol(f : Double) -> Double {
  1.0e-9 * (1.0 + f.abs())
}

///|
fn klw_factor_ok(
  w : Matrix,
  h : Matrix,
  m : Int32,
  k : Int32,
  n : Int32,
) -> Unit raise {
  assert_eq(w.rows, m)
  assert_eq(w.cols, k)
  assert_eq(h.rows, k)
  assert_eq(h.cols, n)
  for x in w.data {
    assert_true(x >= 0.0)
    assert_true(!x.is_nan() && !x.is_inf())
  }
  for x in h.data {
    assert_true(x >= 0.0)
    assert_true(!x.is_nan() && !x.is_inf())
  }
}

///|
test "wb kl: hand-computed one iteration" {
  // WH0=[[1,1],[2,2]], R=[[1,2],[1,2]].
  // H update: W column [1,2] -> colsum 3; numerator [3,6] -> H=[1,2].
  // WH1=V, R=ones; W update: rowsum H=3, RHᵀ=[3,3] -> W=[1,2].
  let v = klw_mat(2, 2, [1.0, 2.0, 2.0, 4.0])
  let w = klw_mat(2, 1, [1.0, 2.0])
  let h = klw_mat(1, 2, [1.0, 1.0])
  let fit = klw_unwrap_fit(mu_kl_fit_from(v, w, h, 1, 0.0, no_reg))
  assert_eq(fit.w.rows, 2)
  assert_eq(fit.w.cols, 1)
  assert_eq(fit.h.rows, 1)
  assert_eq(fit.h.cols, 2)
  assert_eq(fit.w.get(0, 0), Ok(1.0))
  assert_eq(fit.w.get(1, 0), Ok(2.0))
  assert_eq(fit.h.get(0, 0), Ok(1.0))
  assert_eq(fit.h.get(0, 1), Ok(2.0))
  assert_eq(fit.loss_value, 0.0)
  assert_eq(fit.n_iter, 1)
  assert_true(!fit.converged)
}

///|
test "wb kl: H/W updates and full iterations do not increase KL" {
  let v = klw_mat(4, 3, [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 2.0, 9.0, 1.0,
  ])
  let (w, h) = match random_init(v, 2, 11UL) {
    Ok(pair) => pair
    Err(e) => abort("unexpected init error: \{e}")
  }
  let wh = zeros_buf(4, 3)
  let ratio = zeros_buf(4, 3)
  matmul_into(w, h, wh)
  let f0 = klw_unwrap_double(kl_divergence(v, wh))
  match kl_ratio_into(v, wh, ratio) {
    Ok(_) => ()
    Err(e) => abort("unexpected ratio error: \{e}")
  }
  let wtr = zeros_buf(2, 3)
  let colsum = Array::make(2, 0.0)
  match kl_update_h_from_ratio(w, ratio, h, wtr, colsum, no_reg) {
    Ok(_) => ()
    Err(e) => abort("unexpected H update error: \{e}")
  }
  matmul_into(w, h, wh)
  let f1 = klw_unwrap_double(kl_divergence(v, wh))
  assert_true(f1 <= f0 + klw_tol(f0))
  match kl_ratio_into(v, wh, ratio) {
    Ok(_) => ()
    Err(e) => abort("unexpected ratio error: \{e}")
  }
  let rht = zeros_buf(4, 2)
  let rowsum = Array::make(2, 0.0)
  match kl_update_w_from_ratio(h, ratio, w, rht, rowsum, no_reg) {
    Ok(_) => ()
    Err(e) => abort("unexpected W update error: \{e}")
  }
  matmul_into(w, h, wh)
  let f2 = klw_unwrap_double(kl_divergence(v, wh))
  assert_true(f2 <= f1 + klw_tol(f1))
  // Full iterations: same seed gives a common trajectory prefix.
  let mut prev = 1.0e300
  for budget in [1, 5, 20, 100] {
    let (w0, h0) = match random_init(v, 2, 11UL) {
      Ok(pair) => pair
      Err(e) => abort("unexpected init error: \{e}")
    }
    let fit = klw_unwrap_fit(mu_kl_fit_from(v, w0, h0, budget, 0.0, no_reg))
    assert_true(fit.loss_value <= prev + klw_tol(prev))
    prev = fit.loss_value
  }
}

///|
test "wb kl: all-zero V yields zero KL and finite factors" {
  let z = zeros_buf(3, 4)
  let fit = klw_unwrap_fit(mu_kl_fit(z, 2, 50, 1.0e-6, 42UL, no_reg))
  assert_eq(fit.loss_value, 0.0)
  assert_true(fit.converged)
  assert_true(fit.n_iter >= 1 && fit.n_iter <= 50)
  klw_factor_ok(fit.w, fit.h, 3, 2, 4)
  for x in fit.w.data {
    assert_true(x.abs() < 1.0e-12)
  }
  for x in fit.h.data {
    assert_true(x.abs() < 1.0e-12)
  }
}

///|
test "wb kl: zero rows and columns train the remaining region" {
  let v = klw_mat(4, 3, [
    0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 3.0, 4.0, 0.0, 5.0, 6.0, 0.0,
  ])
  let fit = klw_unwrap_fit(mu_kl_fit(v, 2, 100, 1.0e-6, 7UL, no_reg))
  klw_factor_ok(fit.w, fit.h, 4, 2, 3)
  assert_true(fit.loss_value >= 0.0)
  assert_true(!fit.loss_value.is_nan() && !fit.loss_value.is_inf())
  // The zero V row and zero V column canonicalize their factors to zero.
  assert_true(klw_unwrap_double(fit.w.get(0, 0)).abs() < 1.0e-12)
  assert_true(klw_unwrap_double(fit.w.get(0, 1)).abs() < 1.0e-12)
  assert_true(klw_unwrap_double(fit.h.get(0, 2)).abs() < 1.0e-12)
  assert_true(klw_unwrap_double(fit.h.get(1, 2)).abs() < 1.0e-12)
}

///|
test "wb kl: tiny and large valid data stay finite" {
  let tiny = klw_mat(2, 2, [1.0e-100, 2.0e-100, 3.0e-100, 4.0e-100])
  let tfit = klw_unwrap_fit(mu_kl_fit(tiny, 1, 50, 1.0e-6, 5UL, no_reg))
  klw_factor_ok(tfit.w, tfit.h, 2, 1, 2)
  assert_true(tfit.loss_value >= 0.0)
  assert_true(!tfit.loss_value.is_nan() && !tfit.loss_value.is_inf())
  let large = klw_mat(2, 2, [1.0e100, 2.0e100, 3.0e100, 4.0e100])
  let lfit = klw_unwrap_fit(mu_kl_fit(large, 1, 50, 1.0e-6, 5UL, no_reg))
  klw_factor_ok(lfit.w, lfit.h, 2, 1, 2)
  assert_true(lfit.loss_value >= 0.0)
  assert_true(!lfit.loss_value.is_nan() && !lfit.loss_value.is_inf())
}

///|
test "wb kl: Random, NNDSVD, and NNDSVDa are supported without fallback" {
  let v = klw_mat(4, 3, [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 2.0, 9.0, 1.0,
  ])
  let configs = [Init::Random, Init::NNDSVD, Init::NNDSVDa]
  for init in configs {
    let (w0, h0) = match init {
      Random =>
        match random_init(v, 2, 2026UL) {
          Ok(pair) => pair
          Err(e) => abort("unexpected init error: \{e}")
        }
      NNDSVD =>
        match nndsvd_init(v, 2) {
          Ok(pair) => pair
          Err(e) => abort("unexpected NNDSVD error: \{e}")
        }
      NNDSVDa =>
        match nndsvda_init(v, 2) {
          Ok(pair) => pair
          Err(e) => abort("unexpected NNDSVDa error: \{e}")
        }
    }
    let initial = klw_objective(v, w0, h0)
    let direct = klw_unwrap_fit(mu_kl_fit_from(v, w0, h0, 200, 1.0e-6, no_reg))
    klw_factor_ok(direct.w, direct.h, 4, 2, 3)
    assert_true(direct.loss_value <= initial + klw_tol(initial))
    let model = NMF::new(
      NMFConfig::new(
        n_components=2,
        solver=Solver::MU,
        loss=Loss::KLDivergence,
        init~,
        max_iter=200,
        tol=1.0e-6,
        seed=2026UL,
      ),
    )
    assert_eq(model.fit(v), Ok(()))
    let w = klw_unwrap_matrix(model.w())
    let h = klw_unwrap_matrix(model.h())
    klw_factor_ok(w, h, 4, 2, 3)
    for idx in 0..<w.data.length() {
      assert_eq(w.data[idx], direct.w.data[idx])
    }
    for idx in 0..<h.data.length() {
      assert_eq(h.data[idx], direct.h.data[idx])
    }
  }
}

///|
test "wb kl: full API contract and KL transform" {
  let v = klw_mat(4, 3, [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 2.0, 9.0, 1.0,
  ])
  let config = NMFConfig::new(
    n_components=2,
    solver=Solver::MU,
    loss=Loss::KLDivergence,
    init=Init::Random,
    max_iter=200,
    tol=1.0e-6,
    seed=42UL,
  )
  let model = NMF::new(config)
  assert_eq(model.fit(v), Ok(()))
  let w = klw_unwrap_matrix(model.w())
  let h = klw_unwrap_matrix(model.h())
  klw_factor_ok(w, h, 4, 2, 3)
  let loss = klw_unwrap_double(model.loss_value())
  let wh = klw_wh(w, h)
  let expected_kl = klw_unwrap_double(kl_divergence(v, wh))
  assert_true((loss - expected_kl).abs() < 1.0e-9)
  let recon = klw_unwrap_double(model.reconstruction_error(v))
  let mut s = 0.0
  for idx in 0..<v.data.length() {
    let diff = v.data[idx] - wh.data[idx]
    s = s + diff * diff
  }
  assert_true((recon - s.sqrt()).abs() < 1.0e-9)
  let n_iter = match model.n_iter() {
    Ok(x) => x
    Err(e) => abort("unexpected: \{e}")
  }
  let converged = match model.converged() {
    Ok(x) => x
    Err(e) => abort("unexpected: \{e}")
  }
  assert_true(n_iter >= 1 && n_iter <= 200)
  if !converged {
    assert_eq(n_iter, 200)
  }
  // fit_transform is a full deterministic fit followed by w().
  let model2 = NMF::new(config)
  let w_ft = klw_unwrap_matrix(model2.fit_transform(v))
  for idx in 0..<w.data.length() {
    assert_eq(w_ft.data[idx], w.data[idx])
  }
  // inverse_transform(W) = W·H.
  let inverse = klw_unwrap_matrix(model.inverse_transform(w))
  for idx in 0..<inverse.data.length() {
    assert_eq(inverse.data[idx], wh.data[idx])
  }
  // transform uses fixed H and the KL W-update exactly.
  let x = klw_mat(2, 3, [2.0, 1.0, 3.0, 4.0, 2.0, 1.0])
  let w_before = klw_unwrap_matrix(model.w())
  let w_new = klw_unwrap_matrix(model.transform(x))
  klw_factor_ok(w_new, h, 2, 2, 3)
  let w_after = klw_unwrap_matrix(model.w())
  for idx in 0..<w_before.data.length() {
    assert_eq(w_before.data[idx], w_after.data[idx])
  }
  let direct = klw_unwrap_matrix(
    mu_kl_transform(x, h, 200, 1.0e-6, 42UL, no_reg),
  )
  for idx in 0..<w_new.data.length() {
    assert_eq(w_new.data[idx], direct.data[idx])
  }
  // All-zero new data is legal and remains finite.
  let zero_x = zeros_buf(2, 3)
  let zero_w = klw_unwrap_matrix(model.transform(zero_x))
  klw_factor_ok(zero_w, h, 2, 2, 3)
}

///|
test "wb kl: CD+KL is rejected and MU-KL seeds reproduce exactly" {
  let v = klw_mat(4, 3, [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 2.0, 9.0, 1.0,
  ])
  let bad = NMF::new(
    NMFConfig::new(n_components=2, solver=Solver::CD, loss=Loss::KLDivergence),
  )
  assert_true(bad.fit(v) is Err(InvalidConfig(_)))
  assert_true(bad.w() is Err(NotFitted))
  let c1 = NMFConfig::new(
    n_components=2,
    solver=Solver::MU,
    loss=Loss::KLDivergence,
    max_iter=80,
    tol=1.0e-6,
    seed=2026UL,
  )
  let c2 = NMFConfig::new(
    n_components=2,
    solver=Solver::MU,
    loss=Loss::KLDivergence,
    max_iter=80,
    tol=1.0e-6,
    seed=2027UL,
  )
  let m1 = NMF::new(c1)
  let m2 = NMF::new(c1)
  assert_eq(m1.fit(v), Ok(()))
  assert_eq(m2.fit(v), Ok(()))
  let w1 = klw_unwrap_matrix(m1.w())
  let w2 = klw_unwrap_matrix(m2.w())
  for idx in 0..<w1.data.length() {
    assert_eq(w1.data[idx], w2.data[idx])
  }
  assert_eq(m1.n_iter(), m2.n_iter())
  assert_eq(m1.loss_value(), m2.loss_value())
  assert_eq(m1.converged(), m2.converged())
  let m3 = NMF::new(c2)
  assert_eq(m3.fit(v), Ok(()))
  let w3 = klw_unwrap_matrix(m3.w())
  let mut differs = false
  for idx in 0..<w1.data.length() {
    if w1.data[idx] != w3.data[idx] {
      differs = true
    }
  }
  assert_true(differs)
}

///|
test "wb kl: V>0 with a zero reconstruction fails explicitly" {
  let v = klw_mat(1, 1, [1.0])
  let w = klw_mat(1, 1, [0.0])
  let h = klw_mat(1, 1, [0.0])
  assert_true(
    mu_kl_fit_from(v, w, h, 10, 1.0e-6, no_reg) is Err(NumericalError(_)),
  )
  // A KL model trained on all-zero data has H=0; transforming positive data
  // cannot be represented and must not silently return a fabricated result.
  let z = zeros_buf(3, 2)
  let model = NMF::new(
    NMFConfig::new(
      n_components=2,
      solver=Solver::MU,
      loss=Loss::KLDivergence,
      max_iter=20,
      seed=42UL,
    ),
  )
  assert_eq(model.fit(z), Ok(()))
  let x = klw_mat(1, 2, [1.0, 1.0])
  assert_true(model.transform(x) is Err(NumericalError(_)))
}

///|
test "wb kl: max_iter exhaustion is reported honestly" {
  let v = klw_mat(4, 3, [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 2.0, 9.0, 1.0,
  ])
  let model = NMF::new(
    NMFConfig::new(
      n_components=2,
      solver=Solver::MU,
      loss=Loss::KLDivergence,
      max_iter=1,
      tol=0.0,
      seed=42UL,
    ),
  )
  assert_eq(model.fit(v), Ok(()))
  assert_eq(model.n_iter(), Ok(1))
  assert_eq(model.converged(), Ok(false))
  assert_true(model.loss_value() is Ok(_))
}

///|
test "wb kl: all-zero input works with NNDSVD and NNDSVDa" {
  let z = zeros_buf(3, 4)
  for init in [Init::NNDSVD, Init::NNDSVDa] {
    let model = NMF::new(
      NMFConfig::new(
        n_components=2,
        solver=Solver::MU,
        loss=Loss::KLDivergence,
        init~,
        max_iter=20,
        seed=42UL,
      ),
    )
    assert_eq(model.fit(z), Ok(()))
    assert_eq(model.loss_value(), Ok(0.0))
    let w = klw_unwrap_matrix(model.w())
    let h = klw_unwrap_matrix(model.h())
    for x in w.data {
      assert_eq(x, 0.0)
    }
    for x in h.data {
      assert_eq(x, 0.0)
    }
  }
}

///|
test "wb kl: NNDSVD with structural zero support is supported" {
  // Zero first row/column, positive 2x2 block: NNDSVD has structural zeros
  // but every positive V entry has a positive first-component reconstruction.
  let v = klw_mat(3, 3, [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 3.0, 4.0])
  let (w0, h0) = match nndsvd_init(v, 2) {
    Ok(pair) => pair
    Err(e) => abort("unexpected NNDSVD error: \{e}")
  }
  let initial = klw_objective(v, w0, h0)
  let fit = klw_unwrap_fit(mu_kl_fit_from(v, w0, h0, 200, 1.0e-6, no_reg))
  klw_factor_ok(fit.w, fit.h, 3, 2, 3)
  assert_true(fit.loss_value <= initial + klw_tol(initial))
}

///|
test "wb kl: MU-Frobenius and MU-KL comparison is recorded" {
  let v = klw_mat(4, 3, [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 2.0, 9.0, 1.0,
  ])
  let frob = NMF::new(
    NMFConfig::new(
      n_components=2,
      solver=Solver::MU,
      loss=Loss::Frobenius,
      max_iter=200,
      tol=1.0e-6,
      seed=42UL,
    ),
  )
  let kl = NMF::new(
    NMFConfig::new(
      n_components=2,
      solver=Solver::MU,
      loss=Loss::KLDivergence,
      max_iter=200,
      tol=1.0e-6,
      seed=42UL,
    ),
  )
  assert_eq(frob.fit(v), Ok(()))
  assert_eq(kl.fit(v), Ok(()))
  let fw = klw_unwrap_matrix(frob.w())
  let fh = klw_unwrap_matrix(frob.h())
  let kw = klw_unwrap_matrix(kl.w())
  let kh = klw_unwrap_matrix(kl.h())
  let frob_err = klw_unwrap_double(frob.reconstruction_error(v))
  let frob_kl = klw_objective(v, fw, fh)
  let kl_err = klw_unwrap_double(kl.reconstruction_error(v))
  let kl_kl = klw_objective(v, kw, kh)
  let frob_iter = match frob.n_iter() {
    Ok(x) => x
    Err(e) => abort("unexpected: \{e}")
  }
  let kl_iter = match kl.n_iter() {
    Ok(x) => x
    Err(e) => abort("unexpected: \{e}")
  }
  let frob_conv = match frob.converged() {
    Ok(x) => x
    Err(e) => abort("unexpected: \{e}")
  }
  let kl_conv = match kl.converged() {
    Ok(x) => x
    Err(e) => abort("unexpected: \{e}")
  }
  println(
    "MU-Frob: frob_err=\{frob_err}, KL=\{frob_kl}, n_iter=\{frob_iter}, converged=\{frob_conv}",
  )
  println(
    "MU-KL: frob_err=\{kl_err}, KL=\{kl_kl}, n_iter=\{kl_iter}, converged=\{kl_conv}",
  )
  assert_true(frob_err >= 0.0 && !frob_err.is_nan() && !frob_err.is_inf())
  assert_true(frob_kl >= 0.0 && !frob_kl.is_nan() && !frob_kl.is_inf())
  assert_true(kl_err >= 0.0 && !kl_err.is_nan() && !kl_err.is_inf())
  assert_true(kl_kl >= 0.0 && !kl_kl.is_nan() && !kl_kl.is_inf())
}
