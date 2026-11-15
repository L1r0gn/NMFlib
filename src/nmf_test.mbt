// Blackbox tests for the public NMF API: config validation, NotFitted
// contract, fit/transform/inverse_transform/reconstruction_error behavior.

///|
fn api_v() -> @nmf.Matrix {
  mat(3, 2, [2.0, 1.0, 4.0, 2.0, 6.0, 3.0])
}

///|
fn fitted_model() -> @nmf.NMF {
  let model = @nmf.NMF::new(@nmf.NMFConfig::new(n_components=1, seed=42UL))
  match model.fit(api_v()) {
    Ok(_) => model
    Err(e) => abort("unexpected fit error: \{e}")
  }
}

///|
test "api: every model-dependent call returns NotFitted before training" {
  let model = @nmf.NMF::new(@nmf.NMFConfig::new(n_components=1))
  let x = mat(1, 2, [1.0, 1.0])
  let w1 = mat(1, 1, [1.0])
  assert_true(model.w() is Err(@nmf.NotFitted))
  assert_true(model.h() is Err(@nmf.NotFitted))
  assert_true(model.n_iter() is Err(@nmf.NotFitted))
  assert_true(model.loss_value() is Err(@nmf.NotFitted))
  assert_true(model.converged() is Err(@nmf.NotFitted))
  assert_true(model.transform(x) is Err(@nmf.NotFitted))
  assert_true(model.inverse_transform(w1) is Err(@nmf.NotFitted))
  assert_true(model.reconstruction_error(x) is Err(@nmf.NotFitted))
}

///|
test "api: unsupported loss/init/regularization values are explicit errors" {
  let v = api_v()
  let cases : Array[@nmf.NMFConfig] = [
    @nmf.NMFConfig::new(
      n_components=1,
      solver=@nmf.Solver::CD,
      loss=@nmf.Loss::KLDivergence,
    ),
    @nmf.NMFConfig::new(n_components=1, loss=@nmf.Loss::KLDivergence),
    @nmf.NMFConfig::new(n_components=1, init=@nmf.Init::NNDSVDa),
    @nmf.NMFConfig::new(n_components=1, l1_reg=0.1),
    @nmf.NMFConfig::new(n_components=1, l2_reg=0.1),
  ]
  for c in cases {
    let model = @nmf.NMF::new(c)
    assert_true(model.fit(v) is Err(@nmf.InvalidConfig(_)))
    assert_true(model.w() is Err(@nmf.NotFitted))
  }
}

///|
test "api: illegal numeric config values are rejected" {
  let v = api_v()
  let bad_iter = @nmf.NMF::new(@nmf.NMFConfig::new(n_components=1, max_iter=0))
  assert_true(bad_iter.fit(v) is Err(@nmf.InvalidConfig(_)))
  let bad_tol = @nmf.NMF::new(@nmf.NMFConfig::new(n_components=1, tol=-1.0))
  assert_true(bad_tol.fit(v) is Err(@nmf.InvalidConfig(_)))
  let nan_tol = @nmf.NMF::new(@nmf.NMFConfig::new(n_components=1, tol=nan()))
  assert_true(nan_tol.fit(v) is Err(@nmf.InvalidConfig(_)))
  let inf_tol = @nmf.NMF::new(@nmf.NMFConfig::new(n_components=1, tol=inf()))
  assert_true(inf_tol.fit(v) is Err(@nmf.InvalidConfig(_)))
  let zero_rank = @nmf.NMF::new(@nmf.NMFConfig::new(n_components=0))
  assert_true(zero_rank.fit(v) is Err(@nmf.InvalidRank(0)))
  let neg_rank = @nmf.NMF::new(@nmf.NMFConfig::new(n_components=-2))
  assert_true(neg_rank.fit(v) is Err(@nmf.InvalidRank(-2)))
}

///|
test "api: fit rejects negative entries with exact coordinates" {
  let v = mat(2, 2, [1.0, 2.0, 3.0, -0.5])
  let model = @nmf.NMF::new(@nmf.NMFConfig::new(n_components=1))
  assert_eq(model.fit(v), Err(@nmf.NMFError::NegativeEntry(1, 1)))
  assert_true(model.w() is Err(@nmf.NotFitted))
}

///|
test "api: fit rejects rank above min(rows, cols)" {
  let v = api_v() // 3x2 -> max rank 2
  let model = @nmf.NMF::new(@nmf.NMFConfig::new(n_components=3))
  assert_true(model.fit(v) is Err(@nmf.InvalidRank(3)))
}

///|
test "api: fit rejects empty matrices" {
  let empty = unwrap_matrix(@nmf.Matrix::zeros(0, 3))
  let model = @nmf.NMF::new(@nmf.NMFConfig::new(n_components=1))
  assert_true(model.fit(empty) is Err(@nmf.InvalidMatrixShape(_)))
}

///|
test "api: fit_transform returns n x k and matches w()" {
  let model = @nmf.NMF::new(@nmf.NMFConfig::new(n_components=1, seed=42UL))
  let w = unwrap_matrix(model.fit_transform(api_v()))
  assert_eq(w.rows(), 3)
  assert_eq(w.cols(), 1)
  let w_again = unwrap_matrix(model.w())
  for i in 0..<3 {
    assert_eq(w.get(i, 0), w_again.get(i, 0))
  }
}

///|
test "api: accessors return defensive copies" {
  let model = fitted_model()
  let w = unwrap_matrix(model.w())
  match w.set(0, 0, 999.0) {
    Ok(_) => ()
    Err(e) => abort("set failed: \{e}")
  }
  let w_fresh = unwrap_matrix(model.w())
  assert_true(unwrap_double(w_fresh.get(0, 0)) != 999.0)
  let h = unwrap_matrix(model.h())
  match h.set(0, 0, 999.0) {
    Ok(_) => ()
    Err(e) => abort("set failed: \{e}")
  }
  let h_fresh = unwrap_matrix(model.h())
  assert_true(unwrap_double(h_fresh.get(0, 0)) != 999.0)
}

///|
test "api: failed fit resets a previously trained model" {
  let model = fitted_model()
  assert_true(model.w() is Ok(_))
  let bad = mat(2, 2, [1.0, 2.0, 3.0, -1.0])
  assert_true(model.fit(bad) is Err(@nmf.NegativeEntry(_, _)))
  assert_true(model.w() is Err(@nmf.NotFitted))
  assert_true(model.h() is Err(@nmf.NotFitted))
  assert_true(model.n_iter() is Err(@nmf.NotFitted))
  assert_true(model.loss_value() is Err(@nmf.NotFitted))
  assert_true(model.converged() is Err(@nmf.NotFitted))
}

///|
test "api: transform validates shape and non-negativity" {
  let model = fitted_model() // trained on 3x2, rank 1
  let wrong_cols = mat(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  assert_true(model.transform(wrong_cols) is Err(@nmf.DimensionMismatch(_)))
  let negative = mat(1, 2, [1.0, -2.0])
  assert_true(
    model.transform(negative) is Err(@nmf.NMFError::NegativeEntry(0, 1)),
  )
  let empty = unwrap_matrix(@nmf.Matrix::zeros(0, 2))
  assert_true(model.transform(empty) is Err(@nmf.InvalidMatrixShape(_)))
  let good = mat(2, 2, [2.0, 1.0, 4.0, 2.0])
  let w = unwrap_matrix(model.transform(good))
  assert_eq(w.rows(), 2)
  assert_eq(w.cols(), 1)
  for i in 0..<2 {
    let x = unwrap_double(w.get(i, 0))
    assert_true(x >= 0.0)
    assert_true(!x.is_nan() && !x.is_inf())
  }
}

///|
test "api: inverse_transform reconstructs m x d from m x k" {
  let model = fitted_model()
  let w = unwrap_matrix(model.w())
  let recon = unwrap_matrix(model.inverse_transform(w))
  assert_eq(recon.rows(), 3)
  assert_eq(recon.cols(), 2)
  // inverse_transform(w()) must equal W·H elementwise
  let h = unwrap_matrix(model.h())
  let wh = unwrap_matrix(@nmf.matmul(w, h))
  for i in 0..<3 {
    for j in 0..<2 {
      assert_eq(recon.get(i, j), wh.get(i, j))
    }
  }
  let wrong_rank = mat(3, 2, [1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
  assert_true(
    model.inverse_transform(wrong_rank) is Err(@nmf.DimensionMismatch(_)),
  )
}

///|
test "api: reconstruction_error equals sqrt(2 * loss_value)" {
  let model = fitted_model()
  let err = unwrap_double(model.reconstruction_error(api_v()))
  let loss = unwrap_double(model.loss_value())
  assert_true((err - (2.0 * loss).sqrt()).abs() < 1.0e-9)
  assert_true(err >= 0.0)
  assert_true(!err.is_nan() && !err.is_inf())
  let wrong_shape = mat(2, 2, [1.0, 2.0, 3.0, 4.0])
  assert_true(
    model.reconstruction_error(wrong_shape) is Err(@nmf.DimensionMismatch(_)),
  )
}

///|
test "api: transform of training data reconstructs it well" {
  let model = @nmf.NMF::new(
    @nmf.NMFConfig::new(n_components=1, max_iter=500, tol=1.0e-10, seed=42UL),
  )
  let v = api_v()
  match model.fit(v) {
    Ok(_) => ()
    Err(e) => abort("fit failed: \{e}")
  }
  let w2 = unwrap_matrix(model.transform(v))
  let recon = unwrap_matrix(model.inverse_transform(w2))
  let mut s = 0.0
  for i in 0..<3 {
    for j in 0..<2 {
      let diff = unwrap_double(v.get(i, j)) - unwrap_double(recon.get(i, j))
      s = s + diff * diff
    }
  }
  assert_true(s < 1.0e-6)
}
