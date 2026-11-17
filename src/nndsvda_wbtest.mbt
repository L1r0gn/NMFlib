// Whitebox tests for NNDSVDa (standard NNDSVD followed by global-mean zero
// replacement).

///|
fn nndsda_mat(rows : Int32, cols : Int32, data : Array[Double]) -> Matrix {
  match Matrix::new(rows, cols, data) {
    Ok(m) => m
    Err(e) => abort("unexpected matrix error: \{e}")
  }
}

///|
fn nndsda_pair(r : Result[(Matrix, Matrix), NMFError]) -> (Matrix, Matrix) {
  match r {
    Ok(pair) => pair
    Err(e) => abort("unexpected NNDSVDa error: \{e}")
  }
}

///|
fn nndsda_double(r : Result[Double, NMFError]) -> Double {
  match r {
    Ok(x) => x
    Err(e) => abort("unexpected: \{e}")
  }
}

///|
fn nndsda_objective(v : Matrix, w : Matrix, h : Matrix) -> Double {
  match frobenius_objective(v, w, h) {
    Ok(f) => f
    Err(e) => abort("unexpected objective error: \{e}")
  }
}

///|
fn nndsda_factor_ok(
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
/// V with a zero row: standard NNDSVD has exact structural zeros there (and
/// in the degenerate second component), so NNDSVDa has something to fill.
fn nndsda_zero_row_v() -> Matrix {
  nndsda_mat(2, 3, [1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
}

///|
/// Checks that NNDSVDa exactly fills standard NNDSVD's zeros with mean(V) and
/// leaves every non-zero standard entry untouched.
fn nndsda_check_fill(v : Matrix, k : Int32) -> Unit raise {
  let mean = nndsda_double(matrix_mean(v))
  let (w0, h0) = nndsda_pair(nndsvd_init(v, k))
  let (wa, ha) = nndsda_pair(nndsvda_init(v, k))
  nndsda_factor_ok(w0, h0, v.rows, k, v.cols)
  nndsda_factor_ok(wa, ha, v.rows, k, v.cols)
  let mut w_zeros = 0
  let mut h_zeros = 0
  for idx in 0..<w0.data.length() {
    if w0.data[idx] == 0.0 {
      w_zeros = w_zeros + 1
      assert_eq(wa.data[idx], mean)
    } else {
      assert_eq(wa.data[idx], w0.data[idx])
    }
  }
  for idx in 0..<h0.data.length() {
    if h0.data[idx] == 0.0 {
      h_zeros = h_zeros + 1
      assert_eq(ha.data[idx], mean)
    } else {
      assert_eq(ha.data[idx], h0.data[idx])
    }
  }
  assert_true(w_zeros > 0)
  assert_true(h_zeros > 0)
}

///|
test "wb nndsvda: zero entries are replaced by the global V mean" {
  let v = nndsda_zero_row_v()
  let mean = nndsda_double(matrix_mean(v))
  assert_eq(mean, 1.0)
  nndsda_check_fill(v, 2)
}

///|
test "wb nndsvda: mean is global, stable, and rejects empty input" {
  // Hand-computed mean of the T10 example: 6 / 6 = 1.
  let hand = nndsda_mat(2, 3, [1.0, 0.0, 3.0, 0.0, 2.0, 0.0])
  assert_eq(matrix_mean(hand), Ok(1.0))
  // Global mean, not row/column means.
  let rectangular = nndsda_mat(2, 2, [1.0, 2.0, 3.0, 4.0])
  assert_eq(matrix_mean(rectangular), Ok(2.5))
  // All-zero and empty.
  let z = zeros_buf(3, 4)
  assert_eq(matrix_mean(z), Ok(0.0))
  let empty = nndsda_mat(0, 2, [])
  assert_true(matrix_mean(empty) is Err(InvalidMatrixShape(_)))
  // Large finite entries: direct summation would overflow; the scaled
  // fallback must still return the correct finite mean.
  let huge = nndsda_mat(2, 2, [1.0e308, 1.0e308, 0.0, 0.0])
  let mu = nndsda_double(matrix_mean(huge))
  assert_true(!mu.is_nan() && !mu.is_inf())
  assert_true((mu - 5.0e307).abs() <= 5.0e293)
}

///|
test "wb nndsvda: all-zero matrix stays all zero" {
  let z = zeros_buf(3, 4)
  assert_eq(matrix_mean(z), Ok(0.0))
  let (w, h) = nndsda_pair(nndsvda_init(z, 2))
  nndsda_factor_ok(w, h, 3, 2, 4)
  for x in w.data {
    assert_eq(x, 0.0)
  }
  for x in h.data {
    assert_eq(x, 0.0)
  }
}

///|
test "wb nndsvda: rank-deficient input fills degenerate zeros" {
  // Rank 1, k=3: two singular components are zero before NNDSVDa.
  let v = nndsda_mat(3, 3, [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
  assert_eq(matrix_mean(v), Ok(2.0 / 3.0))
  nndsda_check_fill(v, 3)
}

///|
test "wb nndsvda: tiny values keep non-zeros and fill zeros" {
  let v = nndsda_mat(2, 3, [1.0e-100, 2.0e-100, 3.0e-100, 0.0, 0.0, 0.0])
  let mean = nndsda_double(matrix_mean(v))
  assert_true(mean > 0.0)
  assert_true(!mean.is_nan() && !mean.is_inf())
  nndsda_check_fill(v, 2)
}

///|
test "wb nndsvda: standard NNDSVD output is not mutated" {
  let v = nndsda_zero_row_v()
  let (w0, h0) = nndsda_pair(nndsvd_init(v, 2))
  let (wa, ha) = nndsda_pair(nndsvda_init(v, 2))
  let (w1, h1) = nndsda_pair(nndsvd_init(v, 2))
  for idx in 0..<w0.data.length() {
    assert_eq(w0.data[idx], w1.data[idx])
    assert_true(wa.data[idx] >= w0.data[idx])
  }
  for idx in 0..<h0.data.length() {
    assert_eq(h0.data[idx], h1.data[idx])
    assert_true(ha.data[idx] >= h0.data[idx])
  }
}

///|
test "wb nndsvda: deterministic output for repeated calls" {
  let v = nndsda_zero_row_v()
  let (w1, h1) = nndsda_pair(nndsvda_init(v, 2))
  let (w2, h2) = nndsda_pair(nndsvda_init(v, 2))
  for idx in 0..<w1.data.length() {
    assert_eq(w1.data[idx], w2.data[idx])
  }
  for idx in 0..<h1.data.length() {
    assert_eq(h1.data[idx], h2.data[idx])
  }
}

///|
test "wb nndsvda: MU and CD train from NNDSVDa without fallback" {
  let v = nndsda_zero_row_v()
  let (w_initial, h_initial) = nndsda_pair(nndsvda_init(v, 2))
  let initial = nndsda_objective(v, w_initial, h_initial)
  for solver in [Solver::MU, Solver::CD] {
    // Fresh factors for the direct solver call: the from-entry points mutate
    // their inputs in place.
    let (w_direct, h_direct) = nndsda_pair(nndsvda_init(v, 2))
    let (direct_w, direct_h) = match solver {
      MU =>
        match
          mu_frobenius_fit_from(v, w_direct, h_direct, 200, 1.0e-6, no_reg) {
          Ok(x) => (x.w, x.h)
          Err(e) => abort("unexpected MU error: \{e}")
        }
      CD =>
        match
          cd_frobenius_fit_from(v, w_direct, h_direct, 200, 1.0e-6, no_reg) {
          Ok(x) => (x.w, x.h)
          Err(e) => abort("unexpected CD error: \{e}")
        }
    }
    let model = NMF::new(
      NMFConfig::new(
        n_components=2,
        solver~,
        init=Init::NNDSVDa,
        max_iter=200,
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
    nndsda_factor_ok(w, h, 2, 2, 3)
    for idx in 0..<w.data.length() {
      assert_eq(w.data[idx], direct_w.data[idx])
    }
    for idx in 0..<h.data.length() {
      assert_eq(h.data[idx], direct_h.data[idx])
    }
    let loss = nndsda_double(model.loss_value())
    assert_eq(frobenius_objective(v, w, h), Ok(loss))
    assert_true(loss <= initial + 1.0e-9)
    let err = nndsda_double(model.reconstruction_error(v))
    assert_true((err - (2.0 * loss).sqrt()).abs() < 1.0e-9)
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
  }
}

///|
test "wb nndsvda: all-zero input fits through MU and CD" {
  let z = zeros_buf(3, 4)
  for solver in [Solver::MU, Solver::CD] {
    let model = NMF::new(
      NMFConfig::new(
        n_components=2,
        solver~,
        init=Init::NNDSVDa,
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
test "wb nndsvda: invalid inputs and illegal rank are rejected" {
  let neg = nndsda_mat(2, 2, [1.0, 2.0, 3.0, -1.0])
  let neg_model = NMF::new(NMFConfig::new(n_components=1, init=Init::NNDSVDa))
  assert_true(neg_model.fit(neg) is Err(NegativeEntry(1, 1)))
  assert_true(neg_model.w() is Err(NotFitted))
  let v = nndsda_mat(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  let rank_model = NMF::new(NMFConfig::new(n_components=4, init=Init::NNDSVDa))
  assert_true(rank_model.fit(v) is Err(InvalidRank(4)))
  let empty = nndsda_mat(0, 2, [])
  let empty_model = NMF::new(NMFConfig::new(n_components=1, init=Init::NNDSVDa))
  assert_true(empty_model.fit(empty) is Err(InvalidMatrixShape(_)))
  // NaN/Inf cannot enter a Matrix at all.
  assert_true(Matrix::new(1, 1, [0.0 / 0.0]) is Err(NumericalError(_)))
  assert_true(Matrix::new(1, 1, [1.0 / 0.0]) is Err(NumericalError(_)))
  // Unsupported loss/solver combinations still fail before initialization.
  let bad = NMF::new(
    NMFConfig::new(
      n_components=1,
      solver=Solver::CD,
      loss=Loss::KLDivergence,
      init=Init::NNDSVDa,
    ),
  )
  assert_true(bad.fit(v) is Err(InvalidConfig(_)))
  assert_true(bad.w() is Err(NotFitted))
  let bad_iter = NMF::new(
    NMFConfig::new(n_components=1, init=Init::NNDSVDa, max_iter=0),
  )
  assert_true(bad_iter.fit(v) is Err(InvalidConfig(_)))
}

///|
test "wb nndsvda: initialization objective is recorded, not assumed superior" {
  let v = nndsda_zero_row_v()
  let (ws, hs) = nndsda_pair(nndsvd_init(v, 2))
  let (wa, ha) = nndsda_pair(nndsvda_init(v, 2))
  let f_standard = nndsda_objective(v, ws, hs)
  let f_a = nndsda_objective(v, wa, ha)
  let mut sum = 0.0
  let seeds = 20
  for s in 0..<seeds {
    let (wr, hr) = match random_init(v, 2, s.to_uint64()) {
      Ok(pair) => pair
      Err(e) => abort("unexpected init error: \{e}")
    }
    sum = sum + nndsda_objective(v, wr, hr)
  }
  let f_random = sum / seeds.to_double()
  println(
    "NNDSVD F=\{f_standard}, NNDSVDa F=\{f_a}, Random avg F=\{f_random} over \{seeds} seeds",
  )
  assert_true(f_standard >= 0.0 && !f_standard.is_nan() && !f_standard.is_inf())
  assert_true(f_a >= 0.0 && !f_a.is_nan() && !f_a.is_inf())
  assert_true(f_random >= 0.0 && !f_random.is_nan() && !f_random.is_inf())
}

///|
test "wb nndsvda: deterministic across NMF seed values" {
  let v = nndsda_zero_row_v()
  let c1 = NMFConfig::new(
    n_components=2,
    solver=Solver::MU,
    init=Init::NNDSVDa,
    max_iter=80,
    tol=1.0e-6,
    seed=42UL,
  )
  let c2 = NMFConfig::new(
    n_components=2,
    solver=Solver::MU,
    init=Init::NNDSVDa,
    max_iter=80,
    tol=1.0e-6,
    seed=43UL,
  )
  let m1 = NMF::new(c1)
  let m2 = NMF::new(c2)
  assert_eq(m1.fit(v), Ok(()))
  assert_eq(m2.fit(v), Ok(()))
  let w1 = match m1.w() {
    Ok(x) => x
    Err(e) => abort("unexpected: \{e}")
  }
  let w2 = match m2.w() {
    Ok(x) => x
    Err(e) => abort("unexpected: \{e}")
  }
  for idx in 0..<w1.data.length() {
    assert_eq(w1.data[idx], w2.data[idx])
  }
  assert_eq(m1.n_iter(), m2.n_iter())
  assert_eq(m1.loss_value(), m2.loss_value())
}
