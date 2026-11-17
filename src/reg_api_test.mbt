// Blackbox tests for T12 regularization at the public API boundary:
// parameter validation, solver/loss combinations, loss_value vs
// reconstruction_error semantics, transform contract, and zero-regularization
// compatibility with pre-T12 behavior.

///|
fn rgapi_penalty(
  w : @nmf.Matrix,
  h : @nmf.Matrix,
  l1 : Double,
  l2 : Double,
) -> Double {
  let mut s1 = 0.0
  let mut s2 = 0.0
  for i in 0..<w.rows() {
    for j in 0..<w.cols() {
      let x = unwrap_double(w.get(i, j))
      s1 = s1 + x
      s2 = s2 + x * x
    }
  }
  for i in 0..<h.rows() {
    for j in 0..<h.cols() {
      let x = unwrap_double(h.get(i, j))
      s1 = s1 + x
      s2 = s2 + x * x
    }
  }
  l1 * s1 + 0.5 * l2 * s2
}

///|
fn rgapi_frob_sq(v : @nmf.Matrix, w : @nmf.Matrix, h : @nmf.Matrix) -> Double {
  let mut s = 0.0
  for i in 0..<v.rows() {
    for j in 0..<v.cols() {
      let mut wh = 0.0
      for p in 0..<w.cols() {
        wh = wh + unwrap_double(w.get(i, p)) * unwrap_double(h.get(p, j))
      }
      let diff = unwrap_double(v.get(i, j)) - wh
      s = s + diff * diff
    }
  }
  s
}

///|
fn rgapi_fit_ok(config : @nmf.NMFConfig, v : @nmf.Matrix) -> @nmf.NMF {
  let model = @nmf.NMF::new(config)
  match model.fit(v) {
    Ok(_) => model
    Err(e) => abort("unexpected fit error: \{e}")
  }
}

///|
test "reg B: l1_reg/l2_reg accept zero, one-sided, and combined values" {
  let v = mat(3, 2, [2.0, 1.0, 4.0, 2.0, 6.0, 3.0])
  let configs : Array[@nmf.NMFConfig] = [
    @nmf.NMFConfig::new(n_components=1),
    @nmf.NMFConfig::new(n_components=1, l1_reg=0.0, l2_reg=0.0),
    @nmf.NMFConfig::new(n_components=1, l1_reg=0.1),
    @nmf.NMFConfig::new(n_components=1, l2_reg=0.1),
    @nmf.NMFConfig::new(n_components=1, l1_reg=0.1, l2_reg=0.2),
  ]
  for c in configs {
    let model = rgapi_fit_ok(c, v)
    let w = unwrap_matrix(model.w())
    let h = unwrap_matrix(model.h())
    assert_factor_ok(w, 3, 1)
    assert_factor_ok(h, 1, 2)
  }
}

///|
test "reg B: negative, NaN, and infinite regularization are rejected" {
  let v = mat(3, 2, [2.0, 1.0, 4.0, 2.0, 6.0, 3.0])
  let bad : Array[@nmf.NMFConfig] = [
    @nmf.NMFConfig::new(n_components=1, l1_reg=-0.1),
    @nmf.NMFConfig::new(n_components=1, l1_reg=-1.0e-300),
    @nmf.NMFConfig::new(n_components=1, l2_reg=-0.1),
    @nmf.NMFConfig::new(n_components=1, l2_reg=-1.0e-300),
    @nmf.NMFConfig::new(n_components=1, l1_reg=nan()),
    @nmf.NMFConfig::new(n_components=1, l2_reg=nan()),
    @nmf.NMFConfig::new(n_components=1, l1_reg=inf()),
    @nmf.NMFConfig::new(n_components=1, l2_reg=inf()),
    @nmf.NMFConfig::new(n_components=1, l1_reg=-inf()),
    @nmf.NMFConfig::new(n_components=1, l1_reg=0.1, l2_reg=nan()),
  ]
  for c in bad {
    let model = @nmf.NMF::new(c)
    // Rejected, never silently replaced by 0: the model stays untrained.
    assert_true(model.fit(v) is Err(@nmf.InvalidConfig(_)))
    assert_true(model.w() is Err(@nmf.NotFitted))
    assert_true(model.h() is Err(@nmf.NotFitted))
    assert_true(model.loss_value() is Err(@nmf.NotFitted))
  }
}

///|
test "reg I: all three supported solver/loss combinations accept regularization" {
  let v = mat(4, 3, [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 2.0, 9.0, 1.0,
  ])
  let combos : Array[@nmf.NMFConfig] = [
    @nmf.NMFConfig::new(
      n_components=2,
      solver=@nmf.Solver::MU,
      loss=@nmf.Loss::Frobenius,
      l1_reg=0.1,
      l2_reg=0.2,
      max_iter=60,
      seed=42UL,
    ),
    @nmf.NMFConfig::new(
      n_components=2,
      solver=@nmf.Solver::CD,
      loss=@nmf.Loss::Frobenius,
      l1_reg=0.1,
      l2_reg=0.2,
      max_iter=60,
      seed=42UL,
    ),
    @nmf.NMFConfig::new(
      n_components=2,
      solver=@nmf.Solver::MU,
      loss=@nmf.Loss::KLDivergence,
      l1_reg=0.1,
      l2_reg=0.2,
      max_iter=60,
      seed=42UL,
    ),
  ]
  for c in combos {
    let model = rgapi_fit_ok(c, v)
    let w = unwrap_matrix(model.w())
    let h = unwrap_matrix(model.h())
    assert_factor_ok(w, 4, 2)
    assert_factor_ok(h, 2, 3)
    let loss = unwrap_double(model.loss_value())
    assert_true(loss >= 0.0 && !loss.is_nan() && !loss.is_inf())
  }
  // CD + KL stays an explicit InvalidConfig, with or without regularization.
  for
    c in [
      @nmf.NMFConfig::new(
        n_components=2,
        solver=@nmf.Solver::CD,
        loss=@nmf.Loss::KLDivergence,
      ),
      @nmf.NMFConfig::new(
        n_components=2,
        solver=@nmf.Solver::CD,
        loss=@nmf.Loss::KLDivergence,
        l1_reg=0.1,
      ),
    ] {
    let model = @nmf.NMF::new(c)
    assert_true(model.fit(v) is Err(@nmf.InvalidConfig(_)))
    assert_true(model.w() is Err(@nmf.NotFitted))
  }
}

///|
test "reg: loss_value contains the penalty, reconstruction_error does not" {
  let v = mat(5, 4, [
    1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 6.0, 1.0, 7.0, 8.0, 10.0, 2.0, 0.0, 9.0, 1.0, 3.0,
    2.0, 1.0, 0.0, 5.0,
  ])
  let l1 = 0.3
  let l2 = 0.6
  let config = @nmf.NMFConfig::new(
    n_components=2,
    l1_reg=l1,
    l2_reg=l2,
    max_iter=80,
    tol=1.0e-9,
    seed=21UL,
  )
  let model = rgapi_fit_ok(config, v)
  let w = unwrap_matrix(model.w())
  let h = unwrap_matrix(model.h())
  let loss = unwrap_double(model.loss_value())
  let base = 0.5 * rgapi_frob_sq(v, w, h)
  let penalty = rgapi_penalty(w, h, l1, l2)
  assert_true((loss - (base + penalty)).abs() < 1.0e-10 * (1.0 + loss.abs()))
  // reconstruction_error is still exactly ‖V − WH‖_F, no penalty added.
  let err = unwrap_double(model.reconstruction_error(v))
  assert_true((err - (2.0 * base).sqrt()).abs() < 1.0e-10 * (1.0 + err))
  // And loss_value is strictly larger than the unregularized objective here.
  assert_true(loss > base)
}

///|
test "reg: MU-KL loss_value is KL plus the penalty" {
  let v = mat(4, 3, [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 2.0, 9.0, 1.0,
  ])
  let config = @nmf.NMFConfig::new(
    n_components=2,
    loss=@nmf.Loss::KLDivergence,
    l1_reg=0.2,
    l2_reg=0.4,
    max_iter=80,
    tol=1.0e-9,
    seed=21UL,
  )
  let model = rgapi_fit_ok(config, v)
  let w = unwrap_matrix(model.w())
  let h = unwrap_matrix(model.h())
  let loss = unwrap_double(model.loss_value())
  let penalty = rgapi_penalty(w, h, 0.2, 0.4)
  // KL part is recovered by subtracting the penalty; it must be a finite,
  // non-negative divergence smaller than the reported objective.
  let kl = loss - penalty
  assert_true(kl >= 0.0 && !kl.is_nan() && !kl.is_inf())
  assert_true(kl < loss)
}

///|
test "reg: transform keeps the trained H and regularizes only W" {
  let v = mat(4, 3, [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 2.0, 9.0, 1.0,
  ])
  let x = mat(2, 3, [2.0, 1.0, 3.0, 5.0, 4.0, 6.0])
  let config = @nmf.NMFConfig::new(
    n_components=2,
    l1_reg=0.4,
    l2_reg=0.3,
    max_iter=50,
    tol=1.0e-8,
    seed=42UL,
  )
  let model = rgapi_fit_ok(config, v)
  let h_before = unwrap_matrix(model.h())
  let w_new = unwrap_matrix(model.transform(x))
  let h_after = unwrap_matrix(model.h())
  assert_factor_ok(w_new, 2, 2)
  // The learned H is never modified by transform.
  assert_eq(h_before.rows(), h_after.rows())
  assert_eq(h_before.cols(), h_after.cols())
  for i in 0..<h_before.rows() {
    for j in 0..<h_before.cols() {
      assert_eq(
        unwrap_double(h_before.get(i, j)),
        unwrap_double(h_after.get(i, j)),
      )
    }
  }
  // Deterministic under the same seed.
  let again = unwrap_matrix(model.transform(x))
  for i in 0..<w_new.rows() {
    for j in 0..<w_new.cols() {
      assert_eq(unwrap_double(w_new.get(i, j)), unwrap_double(again.get(i, j)))
    }
  }
  // Strong regularization changes the transform solution compared with an
  // otherwise identical unregularized model.
  let plain = rgapi_fit_ok(
    @nmf.NMFConfig::new(n_components=2, max_iter=50, tol=1.0e-8, seed=42UL),
    v,
  )
  let plain_w = unwrap_matrix(plain.transform(x))
  let mut differs = false
  for i in 0..<w_new.rows() {
    for j in 0..<w_new.cols() {
      if (unwrap_double(w_new.get(i, j)) - unwrap_double(plain_w.get(i, j))).abs() >
        1.0e-12 {
        differs = true
      }
    }
  }
  assert_true(differs)
}

///|
test "reg J: explicit zero regularization is bit-for-bit the default" {
  let v = mat(4, 3, [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 2.0, 9.0, 1.0,
  ])
  let combos : Array[(@nmf.NMFConfig, @nmf.NMFConfig)] = [
    (
      @nmf.NMFConfig::new(n_components=2, max_iter=80, tol=1.0e-8, seed=2026UL),
      @nmf.NMFConfig::new(
        n_components=2,
        l1_reg=0.0,
        l2_reg=0.0,
        max_iter=80,
        tol=1.0e-8,
        seed=2026UL,
      ),
    ),
    (
      @nmf.NMFConfig::new(
        n_components=2,
        solver=@nmf.Solver::CD,
        max_iter=80,
        tol=1.0e-8,
        seed=2026UL,
      ),
      @nmf.NMFConfig::new(
        n_components=2,
        solver=@nmf.Solver::CD,
        l1_reg=0.0,
        l2_reg=0.0,
        max_iter=80,
        tol=1.0e-8,
        seed=2026UL,
      ),
    ),
    (
      @nmf.NMFConfig::new(
        n_components=2,
        loss=@nmf.Loss::KLDivergence,
        max_iter=80,
        tol=1.0e-8,
        seed=2026UL,
      ),
      @nmf.NMFConfig::new(
        n_components=2,
        loss=@nmf.Loss::KLDivergence,
        l1_reg=0.0,
        l2_reg=0.0,
        max_iter=80,
        tol=1.0e-8,
        seed=2026UL,
      ),
    ),
  ]
  for pair in combos {
    let (implicit, explicit) = pair
    let m1 = rgapi_fit_ok(implicit, v)
    let m2 = rgapi_fit_ok(explicit, v)
    assert_eq(m1.n_iter(), m2.n_iter())
    assert_eq(m1.converged(), m2.converged())
    assert_eq(m1.loss_value(), m2.loss_value())
    let w1 = unwrap_matrix(m1.w())
    let w2 = unwrap_matrix(m2.w())
    let h1 = unwrap_matrix(m1.h())
    let h2 = unwrap_matrix(m2.h())
    for i in 0..<w1.rows() {
      for j in 0..<w1.cols() {
        assert_eq(unwrap_double(w1.get(i, j)), unwrap_double(w2.get(i, j)))
      }
    }
    for i in 0..<h1.rows() {
      for j in 0..<h1.cols() {
        assert_eq(unwrap_double(h1.get(i, j)), unwrap_double(h2.get(i, j)))
      }
    }
    let x = mat(2, 3, [2.0, 1.0, 3.0, 5.0, 4.0, 6.0])
    let t1 = unwrap_matrix(m1.transform(x))
    let t2 = unwrap_matrix(m2.transform(x))
    for i in 0..<t1.rows() {
      for j in 0..<t1.cols() {
        assert_eq(unwrap_double(t1.get(i, j)), unwrap_double(t2.get(i, j)))
      }
    }
  }
}

///|
test "reg: zero regularization still satisfies sqrt(2·loss) = reconstruction_error" {
  let v = mat(4, 3, [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 2.0, 9.0, 1.0,
  ])
  let m = rgapi_fit_ok(@nmf.NMFConfig::new(n_components=2, seed=42UL), v)
  let loss = unwrap_double(m.loss_value())
  let err = unwrap_double(m.reconstruction_error(v))
  assert_true((err - (2.0 * loss).sqrt()).abs() < 1.0e-12 * (1.0 + err))
}
