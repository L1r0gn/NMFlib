// Whitebox tests for the MU-Frobenius solver: math correctness, numerical
// guards, honest convergence metadata.

///|
fn mu_v() -> Matrix {
  match Matrix::new(3, 2, [2.0, 1.0, 4.0, 2.0, 6.0, 3.0]) {
    Ok(m) => m
    Err(e) => abort("unexpected: \{e}")
  }
}

///|
fn unwrap_fit(r : Result[MuFit, NMFError]) -> MuFit {
  match r {
    Ok(f) => f
    Err(e) => abort("unexpected fit error: \{e}")
  }
}

///|
test "wb mu: rel_change handles near-zero objective" {
  assert_eq(rel_change(0.0, 0.0), 0.0)
  assert_eq(rel_change(1.0e-12, 0.0), 1.0e-12)
  assert_eq(rel_change(5.0, 4.0), 0.25)
  // base is floored at 1 for small objectives
  assert_eq(rel_change(0.5, 0.25), 0.25)
}

///|
test "wb mu: rank-1 exact factorization is recovered" {
  // V = [1,2,3]ᵀ · [2,1], rank 1
  let v = mu_v()
  let fit = unwrap_fit(mu_frobenius_fit(v, 1, 500, 1.0e-10, 42UL))
  assert_eq(fit.w.rows, 3)
  assert_eq(fit.w.cols, 1)
  assert_eq(fit.h.rows, 1)
  assert_eq(fit.h.cols, 2)
  assert_true(fit.loss_value < 1.0e-6)
  assert_true(fit.converged)
  assert_true(fit.n_iter >= 1 && fit.n_iter <= 500)
  for x in fit.w.data {
    assert_true(x >= 0.0)
    assert_true(!x.is_nan() && !x.is_inf())
  }
  for x in fit.h.data {
    assert_true(x >= 0.0)
    assert_true(!x.is_nan() && !x.is_inf())
  }
}

///|
test "wb mu: loss_value equals the true objective of returned factors" {
  let v = mu_v()
  let fit = unwrap_fit(mu_frobenius_fit(v, 1, 37, 1.0e-12, 7UL))
  let f = match frobenius_objective(v, fit.w, fit.h) {
    Ok(f) => f
    Err(e) => abort("unexpected: \{e}")
  }
  assert_eq(fit.loss_value, f)
}

///|
test "wb mu: objective is non-increasing along the deterministic trajectory" {
  // Same seed => identical trajectory prefix, so longer runs must not end
  // with a larger objective (MU monotonicity, up to rounding noise).
  let v = match
    Matrix::new(4, 3, [
      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 2.0, 9.0, 1.0,
    ]) {
    Ok(m) => m
    Err(e) => abort("unexpected: \{e}")
  }
  let mut prev = 1.0e300
  for budget in [1, 5, 20, 100] {
    let fit = unwrap_fit(mu_frobenius_fit(v, 2, budget, 1.0e-14, 11UL))
    assert_true(fit.loss_value <= prev + 1.0e-9)
    prev = fit.loss_value
  }
}

///|
test "wb mu: all-zero V gives zero loss and finite non-negative factors" {
  let z = zeros_buf(3, 4)
  let fit = unwrap_fit(mu_frobenius_fit(z, 2, 50, 1.0e-4, 42UL))
  assert_eq(fit.loss_value, 0.0)
  assert_true(fit.converged)
  assert_true(fit.n_iter >= 1 && fit.n_iter <= 50)
  for x in fit.w.data {
    assert_true(x >= 0.0)
    assert_true(!x.is_nan() && !x.is_inf())
  }
  for x in fit.h.data {
    assert_true(x >= 0.0)
    assert_true(!x.is_nan() && !x.is_inf())
  }
}

///|
test "wb mu: 1x1 matrix converges to the exact product" {
  let v = match Matrix::new(1, 1, [5.0]) {
    Ok(m) => m
    Err(e) => abort("unexpected: \{e}")
  }
  let fit = unwrap_fit(mu_frobenius_fit(v, 1, 200, 1.0e-12, 3UL))
  assert_true(fit.loss_value < 1.0e-12)
  let wh = fit.w.data[0] * fit.h.data[0]
  assert_true((wh - 5.0).abs() < 1.0e-6)
}

///|
test "wb mu: max_iter exhaustion is reported honestly" {
  let v = match
    Matrix::new(4, 3, [
      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 2.0, 9.0, 1.0,
    ]) {
    Ok(m) => m
    Err(e) => abort("unexpected: \{e}")
  }
  let fit = unwrap_fit(mu_frobenius_fit(v, 2, 1, 1.0e-12, 42UL))
  assert_eq(fit.n_iter, 1)
  assert_true(!fit.converged)
  assert_true(fit.loss_value >= 0.0)
  assert_true(!fit.loss_value.is_nan() && !fit.loss_value.is_inf())
}

///|
test "wb mu: transform with fixed H reconstructs rank-1 data" {
  let v = mu_v()
  let fit = unwrap_fit(mu_frobenius_fit(v, 1, 500, 1.0e-10, 42UL))
  // Transform the training data itself with H fixed.
  let w2 = match mu_frobenius_transform(v, fit.h, 500, 1.0e-10, 42UL) {
    Ok(w) => w
    Err(e) => abort("unexpected: \{e}")
  }
  assert_eq(w2.rows, 3)
  assert_eq(w2.cols, 1)
  for x in w2.data {
    assert_true(x >= 0.0)
    assert_true(!x.is_nan() && !x.is_inf())
  }
  let f = match frobenius_objective(v, w2, fit.h) {
    Ok(f) => f
    Err(e) => abort("unexpected: \{e}")
  }
  assert_true(f < 1.0e-6)
}

///|
test "wb mu: transform of all-zero rows stays finite" {
  let z = zeros_buf(2, 2)
  let h = match Matrix::new(1, 2, [1.0, 2.0]) {
    Ok(m) => m
    Err(e) => abort("unexpected: \{e}")
  }
  let w = match mu_frobenius_transform(z, h, 20, 1.0e-4, 1UL) {
    Ok(w) => w
    Err(e) => abort("unexpected: \{e}")
  }
  for x in w.data {
    assert_true(x >= 0.0)
    assert_true(!x.is_nan() && !x.is_inf())
  }
}
