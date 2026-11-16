// Whitebox tests for standard NNDSVD initialization and its API integration.

///|
fn nnd_test_mat(rows : Int32, cols : Int32, data : Array[Double]) -> Matrix {
  match Matrix::new(rows, cols, data) {
    Ok(m) => m
    Err(e) => abort("unexpected matrix error: \{e}")
  }
}

///|
fn nnd_test_pair(r : Result[(Matrix, Matrix), NMFError]) -> (Matrix, Matrix) {
  match r {
    Ok(pair) => pair
    Err(e) => abort("unexpected NNDSVD error: \{e}")
  }
}

///|
fn nnd_test_double(r : Result[Double, NMFError]) -> Double {
  match r {
    Ok(x) => x
    Err(e) => abort("unexpected: \{e}")
  }
}

///|
fn nnd_test_objective(v : Matrix, w : Matrix, h : Matrix) -> Double {
  match frobenius_objective(v, w, h) {
    Ok(f) => f
    Err(e) => abort("unexpected objective error: \{e}")
  }
}

///|
fn nnd_test_factor_ok(
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
fn nnd_test_rank2_fixture() -> Matrix {
  let a = nnd_test_mat(5, 2, [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 0.0, 0.0, 2.0])
  let b = nnd_test_mat(2, 4, [1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 3.0, 2.0])
  match matmul(a, b) {
    Ok(v) => v
    Err(e) => abort("unexpected matmul error: \{e}")
  }
}

///|
test "wb nndsvd: factors are non-negative, finite, and correctly shaped" {
  let v = nnd_test_mat(4, 3, [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 2.0, 9.0, 1.0,
  ])
  let (w, h) = nnd_test_pair(nndsvd_init(v, 2))
  nnd_test_factor_ok(w, h, 4, 2, 3)
}

///|
test "wb nndsvd: standard initialization leaves structural zeros" {
  // Distinct singular values make the top singular vectors [1,0] and [0,1];
  // standard NNDSVD must leave W[1,0] and H[0,1] exactly zero (no mean
  // filling; NNDSVDa would fill them).
  let v = nnd_test_mat(2, 2, [2.0, 0.0, 0.0, 1.0])
  let (w, h) = nnd_test_pair(nndsvd_init(v, 1))
  nnd_test_factor_ok(w, h, 2, 1, 2)
  // The exact structural zero is represented by rounding-level leakage only;
  // it is nowhere near a mean-filling replacement.
  assert_true(w.data[1].abs() < 1.0e-6)
  assert_true(h.data[1].abs() < 1.0e-6)
  assert_true(w.data[0] > 0.0)
  assert_true(h.data[0] > 0.0)
}

///|
test "wb nndsvd: rank-1 input zeroes the degenerate second component" {
  let u0 = [1.0, 2.0, 3.0]
  let v0 = [2.0, 1.0]
  let v = nnd_test_mat(3, 2, [
    u0[0] * v0[0],
    u0[0] * v0[1],
    u0[1] * v0[0],
    u0[1] * v0[1],
    u0[2] * v0[0],
    u0[2] * v0[1],
  ])
  let (w, h) = nnd_test_pair(nndsvd_init(v, 2))
  nnd_test_factor_ok(w, h, 3, 2, 2)
  assert_true(w.data[0] > 0.0 && w.data[2] > 0.0 && w.data[4] > 0.0)
  // W[:,1] and H[1,:] are exactly zero.
  for i in 0..<3 {
    assert_eq(w.data[i * 2 + 1], 0.0)
  }
  for j in 0..<2 {
    assert_eq(h.data[2 + j], 0.0)
  }
}

///|
test "wb nndsvd: all-zero and rank-deficient inputs degrade to zeros" {
  let z = zeros_buf(3, 4)
  let (wz, hz) = nnd_test_pair(nndsvd_init(z, 2))
  nnd_test_factor_ok(wz, hz, 3, 2, 4)
  for x in wz.data {
    assert_eq(x, 0.0)
  }
  for x in hz.data {
    assert_eq(x, 0.0)
  }
  // Rank-1 data with k=3: only the first component is nonzero, the extra
  // rows/columns are structural zeros.
  let v = nnd_test_mat(3, 3, [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0])
  let (w, h) = nnd_test_pair(nndsvd_init(v, 3))
  nnd_test_factor_ok(w, h, 3, 3, 3)
  for i in 0..<3 {
    assert_eq(w.data[i * 3 + 1], 0.0)
    assert_eq(w.data[i * 3 + 2], 0.0)
  }
  for j in 0..<3 {
    assert_eq(h.data[3 + j], 0.0)
    assert_eq(h.data[6 + j], 0.0)
  }
}

///|
test "wb nndsvd: tiny values stay finite" {
  let v = nnd_test_mat(2, 2, [1.0e-100, 2.0e-100, 3.0e-100, 4.0e-100])
  let (w, h) = nnd_test_pair(nndsvd_init(v, 2))
  nnd_test_factor_ok(w, h, 2, 2, 2)
}

///|
test "wb nndsvd: deterministic and independent of NMFConfig.seed" {
  let v = nnd_test_mat(4, 3, [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 2.0, 9.0, 1.0,
  ])
  let (w1, h1) = nnd_test_pair(nndsvd_init(v, 2))
  let (w2, h2) = nnd_test_pair(nndsvd_init(v, 2))
  for idx in 0..<w1.data.length() {
    assert_eq(w1.data[idx], w2.data[idx])
  }
  for idx in 0..<h1.data.length() {
    assert_eq(h1.data[idx], h2.data[idx])
  }
  // Full training must also be identical because NNDSVD ignores the seed and
  // both solvers are deterministic once initialized.
  let c1 = NMFConfig::new(
    n_components=2,
    solver=Solver::MU,
    init=Init::NNDSVD,
    max_iter=80,
    tol=1.0e-6,
    seed=42UL,
  )
  let c2 = NMFConfig::new(
    n_components=2,
    solver=Solver::MU,
    init=Init::NNDSVD,
    max_iter=80,
    tol=1.0e-6,
    seed=43UL,
  )
  let m1 = NMF::new(c1)
  let m2 = NMF::new(c2)
  assert_eq(m1.fit(v), Ok(()))
  assert_eq(m2.fit(v), Ok(()))
  let wa = match m1.w() {
    Ok(x) => x
    Err(e) => abort("unexpected: \{e}")
  }
  let wb = match m2.w() {
    Ok(x) => x
    Err(e) => abort("unexpected: \{e}")
  }
  for idx in 0..<wa.data.length() {
    assert_eq(wa.data[idx], wb.data[idx])
  }
  assert_eq(m1.n_iter(), m2.n_iter())
  assert_eq(m1.loss_value(), m2.loss_value())
}

///|
test "wb nndsvd: initial objective beats the Random average on a fixed fixture" {
  let v = nnd_test_rank2_fixture()
  let (wn, hn) = nnd_test_pair(nndsvd_init(v, 2))
  let f_nndsvd = nnd_test_objective(v, wn, hn)
  let mut sum = 0.0
  let seeds = 20
  for s in 0..<seeds {
    let (wr, hr) = match random_init(v, 2, s.to_uint64()) {
      Ok(pair) => pair
      Err(e) => abort("unexpected init error: \{e}")
    }
    sum = sum + nnd_test_objective(v, wr, hr)
  }
  let avg_random = sum / seeds.to_double()
  println(
    "NNDSVD initial F=\{f_nndsvd}, Random avg F=\{avg_random} over \{seeds} seeds",
  )
  assert_true(f_nndsvd <= avg_random + 1.0e-12)
}

///|
test "wb nndsvd: MU and CD both train from NNDSVD initialization" {
  let v = nnd_test_rank2_fixture()
  let (w0, h0) = nnd_test_pair(nndsvd_init(v, 2))
  let initial = nnd_test_objective(v, w0, h0)
  for solver in [Solver::MU, Solver::CD] {
    let model = NMF::new(
      NMFConfig::new(
        n_components=2,
        solver~,
        init=Init::NNDSVD,
        max_iter=300,
        tol=1.0e-6,
        seed=42UL,
      ),
    )
    assert_eq(model.fit(v), Ok(()))
    let w = match model.w() {
      Ok(x) => x
      Err(e) => abort("unexpected: \{e}")
    }
    let h = match model.h() {
      Ok(x) => x
      Err(e) => abort("unexpected: \{e}")
    }
    nnd_test_factor_ok(w, h, 5, 2, 4)
    let loss = nnd_test_double(model.loss_value())
    assert_eq(frobenius_objective(v, w, h), Ok(loss))
    assert_true(loss <= initial + 1.0e-9)
    let err = nnd_test_double(model.reconstruction_error(v))
    assert_true((err - (2.0 * loss).sqrt()).abs() < 1.0e-9)
    let n_iter = match model.n_iter() {
      Ok(x) => x
      Err(e) => abort("unexpected: \{e}")
    }
    let converged = match model.converged() {
      Ok(x) => x
      Err(e) => abort("unexpected: \{e}")
    }
    assert_true(n_iter >= 1 && n_iter <= 300)
    if !converged {
      assert_eq(n_iter, 300)
    }
  }
}

///|
test "wb nndsvd: all-zero input fits through MU and CD" {
  let z = zeros_buf(3, 4)
  for solver in [Solver::MU, Solver::CD] {
    let model = NMF::new(
      NMFConfig::new(
        n_components=2,
        solver~,
        init=Init::NNDSVD,
        max_iter=20,
        seed=42UL,
      ),
    )
    assert_eq(model.fit(z), Ok(()))
    assert_eq(model.loss_value(), Ok(0.0))
    let w = match model.w() {
      Ok(x) => x
      Err(e) => abort("unexpected: \{e}")
    }
    let h = match model.h() {
      Ok(x) => x
      Err(e) => abort("unexpected: \{e}")
    }
    for x in w.data {
      assert_eq(x, 0.0)
    }
    for x in h.data {
      assert_eq(x, 0.0)
    }
  }
}

///|
test "wb nndsvd: rank above the numerical rank and invalid inputs" {
  // Rank-1 data with k=3: the two extra components stay zero through MU.
  let rank1 = nnd_test_mat(3, 3, [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0])
  let model = NMF::new(
    NMFConfig::new(
      n_components=3,
      solver=Solver::MU,
      init=Init::NNDSVD,
      max_iter=100,
      tol=1.0e-6,
      seed=42UL,
    ),
  )
  assert_eq(model.fit(rank1), Ok(()))
  let w = match model.w() {
    Ok(x) => x
    Err(e) => abort("unexpected: \{e}")
  }
  let h = match model.h() {
    Ok(x) => x
    Err(e) => abort("unexpected: \{e}")
  }
  nnd_test_factor_ok(w, h, 3, 3, 3)
  for i in 0..<3 {
    assert_eq(w.data[i * 3 + 1], 0.0)
    assert_eq(w.data[i * 3 + 2], 0.0)
  }
  for j in 0..<3 {
    assert_eq(h.data[3 + j], 0.0)
    assert_eq(h.data[6 + j], 0.0)
  }
  // The existing rank range still applies (k <= min(rows, cols)).
  let too_large = NMF::new(NMFConfig::new(n_components=4, init=Init::NNDSVD))
  assert_true(too_large.fit(rank1) is Err(InvalidRank(4)))
  assert_true(too_large.w() is Err(NotFitted))
  // Invalid V is rejected before initialization.
  let neg = nnd_test_mat(2, 2, [1.0, 2.0, 3.0, -1.0])
  let bad = NMF::new(NMFConfig::new(n_components=1, init=Init::NNDSVD))
  assert_true(bad.fit(neg) is Err(NegativeEntry(1, 1)))
  assert_true(bad.w() is Err(NotFitted))
  let empty = match Matrix::zeros(0, 2) {
    Ok(m) => m
    Err(e) => abort("unexpected: \{e}")
  }
  let empty_model = NMF::new(NMFConfig::new(n_components=1, init=Init::NNDSVD))
  assert_true(empty_model.fit(empty) is Err(InvalidMatrixShape(_)))
}

///|
test "wb nndsvd: duplicate singular values remain finite" {
  let v = nnd_test_mat(3, 3, [2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0])
  let (w, h) = nnd_test_pair(nndsvd_init(v, 3))
  nnd_test_factor_ok(w, h, 3, 3, 3)
}

///|
test "wb nndsvd: fit consumes NNDSVD factors without Random fallback" {
  let v = nnd_test_rank2_fixture()
  let (w0_mu, h0_mu) = nnd_test_pair(nndsvd_init(v, 2))
  // MU path: NMF::fit must match a direct fit seeded with the NNDSVD factors.
  let direct_mu = match
    mu_frobenius_fit_from(v, w0_mu, h0_mu, 50, 1.0e-6, no_reg) {
    Ok(x) => x
    Err(e) => abort("unexpected MU error: \{e}")
  }
  let mu_model = NMF::new(
    NMFConfig::new(
      n_components=2,
      solver=Solver::MU,
      init=Init::NNDSVD,
      max_iter=50,
      tol=1.0e-6,
      seed=123UL,
    ),
  )
  assert_eq(mu_model.fit(v), Ok(()))
  let mu_w = match mu_model.w() {
    Ok(x) => x
    Err(e) => abort("unexpected: \{e}")
  }
  let mu_h = match mu_model.h() {
    Ok(x) => x
    Err(e) => abort("unexpected: \{e}")
  }
  for idx in 0..<mu_w.data.length() {
    assert_eq(mu_w.data[idx], direct_mu.w.data[idx])
  }
  for idx in 0..<mu_h.data.length() {
    assert_eq(mu_h.data[idx], direct_mu.h.data[idx])
  }
  // CD path: same check with a fresh NNDSVD pair (the MU call above mutates
  // the factor matrices in place).
  let (w0_cd, h0_cd) = nnd_test_pair(nndsvd_init(v, 2))
  let direct_cd = match
    cd_frobenius_fit_from(v, w0_cd, h0_cd, 50, 1.0e-6, no_reg) {
    Ok(x) => x
    Err(e) => abort("unexpected CD error: \{e}")
  }
  let cd_model = NMF::new(
    NMFConfig::new(
      n_components=2,
      solver=Solver::CD,
      init=Init::NNDSVD,
      max_iter=50,
      tol=1.0e-6,
      seed=123UL,
    ),
  )
  assert_eq(cd_model.fit(v), Ok(()))
  let cd_w = match cd_model.w() {
    Ok(x) => x
    Err(e) => abort("unexpected: \{e}")
  }
  let cd_h = match cd_model.h() {
    Ok(x) => x
    Err(e) => abort("unexpected: \{e}")
  }
  for idx in 0..<cd_w.data.length() {
    assert_eq(cd_w.data[idx], direct_cd.w.data[idx])
  }
  for idx in 0..<cd_h.data.length() {
    assert_eq(cd_h.data[idx], direct_cd.h.data[idx])
  }
}

///|
test "wb nndsvd: NNDSVDa is a supported initialization variant" {
  let v = nnd_test_mat(2, 2, [1.0, 2.0, 3.0, 4.0])
  let model = NMF::new(NMFConfig::new(n_components=1, init=Init::NNDSVDa))
  assert_eq(model.fit(v), Ok(()))
  let w = match model.w() {
    Ok(x) => x
    Err(e) => abort("unexpected: \{e}")
  }
  let h = match model.h() {
    Ok(x) => x
    Err(e) => abort("unexpected: \{e}")
  }
  nnd_test_factor_ok(w, h, 2, 1, 2)
}
