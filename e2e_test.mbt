// End-to-end acceptance tests A–F (task spec §九) plus randomized property
// checks. Everything here goes through the public API only.

// Test-only LCG for generating varied inputs (independent of the library's
// SplitMix64, which is an internal detail).

///|
priv struct TestRng {
  mut state : UInt64
}

///|
fn TestRng::new(seed : UInt64) -> TestRng {
  { state: seed, }
}

///|
fn TestRng::next_double(self : TestRng) -> Double {
  self.state = (self.state * 1664525UL + 1013904223UL) & 0xFFFFFFFFUL
  (self.state >> 8).to_double() / 16777216.0
}

///|
fn TestRng::next_int(self : TestRng, bound : Int32) -> Int32 {
  (self.next_double() * bound.to_double()).to_int()
}

///|
fn assert_factor_ok(m : @nmf.Matrix, rows : Int32, cols : Int32) -> Unit raise {
  assert_eq(m.rows(), rows)
  assert_eq(m.cols(), cols)
  for i in 0..<rows {
    for j in 0..<cols {
      let x = unwrap_double(m.get(i, j))
      assert_true(x >= 0.0)
      assert_true(!x.is_nan() && !x.is_inf())
    }
  }
}

// ---------------------------------------------------------------------
// Test A: ordinary non-negative matrix through the full pipeline.
// ---------------------------------------------------------------------

///|
test "e2e A: ordinary non-negative matrix fits end to end" {
  let v = mat(4, 3, [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 2.0, 9.0, 1.0,
  ])
  let config = @nmf.NMFConfig::new(
    n_components=2,
    solver=@nmf.Solver::MU,
    loss=@nmf.Loss::Frobenius,
    seed=42UL,
  )
  let model = @nmf.NMF::new(config)
  assert_eq(model.fit(v), Ok(()))
  let w = unwrap_matrix(model.w())
  let h = unwrap_matrix(model.h())
  assert_factor_ok(w, 4, 2)
  assert_factor_ok(h, 2, 3)
  let wh = unwrap_matrix(@nmf.matmul(w, h))
  assert_eq(wh.rows(), 4)
  assert_eq(wh.cols(), 3)
  let err = unwrap_double(model.reconstruction_error(v))
  assert_true(!err.is_nan() && !err.is_inf())
  assert_true(err >= 0.0)
  let n_iter = match model.n_iter() {
    Ok(n) => n
    Err(e) => abort("unexpected: \{e}")
  }
  assert_true(n_iter >= 1 && n_iter <= 200)
  let converged = match model.converged() {
    Ok(c) => c
    Err(e) => abort("unexpected: \{e}")
  }
  // Honest convergence: giving up early is only allowed at the budget limit.
  if !converged {
    assert_eq(n_iter, 200)
  }
  let loss = unwrap_double(model.loss_value())
  assert_true(!loss.is_nan() && !loss.is_inf())
  assert_true((err - (2.0 * loss).sqrt()).abs() < 1.0e-9)
}

// ---------------------------------------------------------------------
// Test B: known low-rank V = W_true · H_true, recovered up to scale and
// permutation (compare reconstruction quality, never elementwise factors).
// ---------------------------------------------------------------------

///|
test "e2e B: known rank-2 product is reconstructed accurately" {
  let w_true = mat(5, 2, [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 0.0, 0.0, 2.0])
  let h_true = mat(2, 4, [1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 3.0, 2.0])
  let v = unwrap_matrix(@nmf.matmul(w_true, h_true))
  let model = @nmf.NMF::new(
    @nmf.NMFConfig::new(n_components=2, max_iter=2000, tol=1.0e-10, seed=42UL),
  )
  assert_eq(model.fit(v), Ok(()))
  let w = unwrap_matrix(model.w())
  let h = unwrap_matrix(model.h())
  assert_factor_ok(w, 5, 2)
  assert_factor_ok(h, 2, 4)
  let err = unwrap_double(model.reconstruction_error(v))
  let v_norm = unwrap_double(@nmf.frobenius_norm(v))
  // Relative reconstruction error below 1%.
  assert_true(err / v_norm < 1.0e-2)
}

// ---------------------------------------------------------------------
// Test C: all-zero matrix — no panic, no NaN/Infinity, exact zero loss.
// ---------------------------------------------------------------------

///|
test "e2e C: all-zero matrix produces a legal exact result" {
  let v = unwrap_matrix(@nmf.Matrix::zeros(3, 4))
  let model = @nmf.NMF::new(@nmf.NMFConfig::new(n_components=2, seed=42UL))
  assert_eq(model.fit(v), Ok(()))
  let w = unwrap_matrix(model.w())
  let h = unwrap_matrix(model.h())
  assert_factor_ok(w, 3, 2)
  assert_factor_ok(h, 2, 4)
  assert_eq(model.reconstruction_error(v), Ok(0.0))
  assert_eq(model.loss_value(), Ok(0.0))
  let converged = match model.converged() {
    Ok(c) => c
    Err(e) => abort("unexpected: \{e}")
  }
  assert_true(converged)
}

// ---------------------------------------------------------------------
// Test D: illegal inputs return errors, never crash or produce numbers.
// ---------------------------------------------------------------------

///|
test "e2e D: illegal inputs are rejected with typed errors" {
  let config = @nmf.NMFConfig::new(n_components=2)
  // negative entry
  let neg = mat(2, 2, [1.0, 2.0, -3.0, 4.0])
  let m1 = @nmf.NMF::new(config)
  assert_true(m1.fit(neg) is Err(@nmf.NMFError::NegativeEntry(1, 0)))
  // NaN / Infinity cannot even enter a Matrix
  assert_true(@nmf.Matrix::new(1, 1, [nan()]) is Err(@nmf.NumericalError(_)))
  assert_true(@nmf.Matrix::new(1, 1, [inf()]) is Err(@nmf.NumericalError(_)))
  assert_true(@nmf.Matrix::new(1, 1, [-inf()]) is Err(@nmf.NumericalError(_)))
  // illegal rank
  let v = mat(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  let m2 = @nmf.NMF::new(@nmf.NMFConfig::new(n_components=5))
  assert_true(m2.fit(v) is Err(@nmf.InvalidRank(5)))
  let m3 = @nmf.NMF::new(@nmf.NMFConfig::new(n_components=0))
  assert_true(m3.fit(v) is Err(@nmf.InvalidRank(0)))
  // wrong dimensions downstream
  let m4 = @nmf.NMF::new(@nmf.NMFConfig::new(n_components=1))
  assert_eq(m4.fit(v), Ok(()))
  let wrong = mat(1, 5, [1.0, 2.0, 3.0, 4.0, 5.0])
  assert_true(m4.transform(wrong) is Err(@nmf.DimensionMismatch(_)))
  assert_true(m4.reconstruction_error(wrong) is Err(@nmf.DimensionMismatch(_)))
  // illegal configuration
  let m5 = @nmf.NMF::new(@nmf.NMFConfig::new(n_components=1, max_iter=-3))
  assert_true(m5.fit(v) is Err(@nmf.InvalidConfig(_)))
  // every rejected model stays untrained
  assert_true(m1.w() is Err(@nmf.NotFitted))
  assert_true(m2.w() is Err(@nmf.NotFitted))
  assert_true(m3.w() is Err(@nmf.NotFitted))
  assert_true(m5.w() is Err(@nmf.NotFitted))
}

// ---------------------------------------------------------------------
// Test E: fixed seed reproduces the entire training run bit-for-bit.
// ---------------------------------------------------------------------

///|
test "e2e E: same seed reproduces factors, iterations, and loss exactly" {
  let v = mat(4, 3, [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 2.0, 9.0, 1.0,
  ])
  let config = @nmf.NMFConfig::new(
    n_components=2,
    max_iter=80,
    tol=1.0e-6,
    seed=2026UL,
  )
  let m1 = @nmf.NMF::new(config)
  let m2 = @nmf.NMF::new(config)
  assert_eq(m1.fit(v), Ok(()))
  assert_eq(m2.fit(v), Ok(()))
  let w1 = unwrap_matrix(m1.w())
  let w2 = unwrap_matrix(m2.w())
  let h1 = unwrap_matrix(m1.h())
  let h2 = unwrap_matrix(m2.h())
  for i in 0..<4 {
    for j in 0..<2 {
      assert_eq(w1.get(i, j), w2.get(i, j))
    }
  }
  for i in 0..<2 {
    for j in 0..<3 {
      assert_eq(h1.get(i, j), h2.get(i, j))
    }
  }
  assert_eq(m1.n_iter(), m2.n_iter())
  assert_eq(m1.loss_value(), m2.loss_value())
  assert_eq(m1.converged(), m2.converged())
  // A different seed must explore a different trajectory.
  let m3 = @nmf.NMF::new(
    @nmf.NMFConfig::new(n_components=2, max_iter=80, tol=1.0e-6, seed=2027UL),
  )
  assert_eq(m3.fit(v), Ok(()))
  let w3 = unwrap_matrix(m3.w())
  let mut differs = false
  for i in 0..<4 {
    for j in 0..<2 {
      if w1.get(i, j) != w3.get(i, j) {
        differs = true
      }
    }
  }
  assert_true(differs)
}

// ---------------------------------------------------------------------
// Test F: a tiny iteration budget must be reported as NOT converged.
// ---------------------------------------------------------------------

///|
test "e2e F: exhausted budget is honestly marked unconverged" {
  let v = mat(4, 3, [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 2.0, 9.0, 1.0,
  ])
  let model = @nmf.NMF::new(
    @nmf.NMFConfig::new(n_components=2, max_iter=3, tol=1.0e-12, seed=42UL),
  )
  assert_eq(model.fit(v), Ok(()))
  let n_iter = match model.n_iter() {
    Ok(n) => n
    Err(e) => abort("unexpected: \{e}")
  }
  assert_true(n_iter <= 3)
  assert_eq(n_iter, 3)
  assert_eq(model.converged(), Ok(false))
  // The final objective is still computed and truthful.
  let loss = unwrap_double(model.loss_value())
  assert_true(!loss.is_nan() && !loss.is_inf())
  let w = unwrap_matrix(model.w())
  let h = unwrap_matrix(model.h())
  assert_eq(@nmf.frobenius_objective(v, w, h), Ok(loss))
  let err = unwrap_double(model.reconstruction_error(v))
  assert_true((err - (2.0 * loss).sqrt()).abs() < 1.0e-9)
}

// ---------------------------------------------------------------------
// Property sweep: 200 varied legal inputs must never panic, must respect
// dims / non-negativity / finiteness / honest metadata; every 10th case is
// corrupted and must be rejected.
// ---------------------------------------------------------------------

///|
test "e2e property: 200 random legal inputs satisfy all invariants" {
  let rng = TestRng::new(20261112UL)
  for case in 0..<200 {
    let rows = 1 + rng.next_int(6)
    let cols = 1 + rng.next_int(6)
    let max_k = if rows < cols { rows } else { cols }
    let k = 1 + rng.next_int(max_k)
    let data = Array::make(rows * cols, 0.0)
    for idx in 0..<data.length() {
      data[idx] = rng.next_double() * 5.0
    }
    let v = unwrap_matrix(@nmf.Matrix::new(rows, cols, data))
    let model = @nmf.NMF::new(
      @nmf.NMFConfig::new(
        n_components=k,
        max_iter=50,
        tol=1.0e-4,
        seed=case.to_uint64(),
      ),
    )
    assert_eq(model.fit(v), Ok(()))
    let w = unwrap_matrix(model.w())
    let h = unwrap_matrix(model.h())
    assert_factor_ok(w, rows, k)
    assert_factor_ok(h, k, cols)
    let n_iter = match model.n_iter() {
      Ok(n) => n
      Err(e) => abort("unexpected: \{e}")
    }
    assert_true(n_iter >= 1 && n_iter <= 50)
    let converged = match model.converged() {
      Ok(c) => c
      Err(e) => abort("unexpected: \{e}")
    }
    if !converged {
      assert_eq(n_iter, 50)
    }
    let loss = unwrap_double(model.loss_value())
    assert_true(loss >= 0.0)
    assert_true(!loss.is_nan() && !loss.is_inf())
    let err = unwrap_double(model.reconstruction_error(v))
    assert_true(!err.is_nan() && !err.is_inf())
    assert_true((err - (2.0 * loss).sqrt()).abs() < 1.0e-9)
    // Periodically corrupt one entry: must be rejected, model reset.
    if case % 10 == 0 {
      let bad_data = data.copy()
      bad_data[0] = -1.0
      let bad = unwrap_matrix(@nmf.Matrix::new(rows, cols, bad_data))
      assert_true(model.fit(bad) is Err(@nmf.NMFError::NegativeEntry(0, 0)))
      assert_true(model.w() is Err(@nmf.NotFitted))
    }
  }
}

///|
test "e2e property: fixed seeds reproduce across 30 varied shapes" {
  let rng = TestRng::new(777UL)
  for case in 0..<30 {
    let rows = 1 + rng.next_int(5)
    let cols = 1 + rng.next_int(5)
    let max_k = if rows < cols { rows } else { cols }
    let k = 1 + rng.next_int(max_k)
    let data = Array::make(rows * cols, 0.0)
    for idx in 0..<data.length() {
      data[idx] = rng.next_double() * 3.0
    }
    let v = unwrap_matrix(@nmf.Matrix::new(rows, cols, data))
    let config = @nmf.NMFConfig::new(
      n_components=k,
      max_iter=30,
      tol=1.0e-5,
      seed=(case * 13 + 1).to_uint64(),
    )
    let m1 = @nmf.NMF::new(config)
    let m2 = @nmf.NMF::new(config)
    assert_eq(m1.fit(v), Ok(()))
    assert_eq(m2.fit(v), Ok(()))
    assert_eq(m1.n_iter(), m2.n_iter())
    assert_eq(m1.loss_value(), m2.loss_value())
    let w1 = unwrap_matrix(m1.w())
    let w2 = unwrap_matrix(m2.w())
    for i in 0..<rows {
      for j in 0..<k {
        assert_eq(w1.get(i, j), w2.get(i, j))
      }
    }
  }
}
