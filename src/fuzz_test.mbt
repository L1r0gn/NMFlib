// Lightweight deterministic fuzz tests (T14): illegal inputs must be rejected
// with the documented error variant, and must never panic or corrupt model
// state.
//
// Scope notes (no invented error paths):
//
// - `fit` / `transform` / `reconstruction_error` take a `Matrix`, and
//   `Matrix::new` rejects NaN/Infinity with `NumericalError`. Those values can
//   therefore never reach the model APIs; the constructor is where they are
//   fuzzed here. `sparsity` takes a bare `Array[Double]`, so it is fuzzed
//   directly.
// - Sizes are bounded (<= 12 per side) so the fuzz never allocates
//   uncontrolled memory. Oversized shapes are exercised only through the
//   constructor's overflow check, which rejects before allocating.
// - In MoonBit a panic aborts the test, so "did not panic" is demonstrated by
//   reaching the assertions after each call; every case also asserts the exact
//   error shape instead of accepting any `Err`.
// - On an already-trained handle the reachable `fit` failures are negative
//   entries and empty matrices (the rank and configuration are fixed at
//   construction), so atomicity is fuzzed with those two; bad-rank and
//   bad-config failures are covered on fresh handles.
//
// Determinism: one fixed seed, the shared `TestRng`, no clock, no external
// randomness. Failure messages carry the seed and case index.

///|
let fuzz_seed : UInt64 = 20261116UL

///|
fn fuzz_require(cond : Bool, case : Int32, detail : String) -> Unit {
  if !cond {
    abort("fuzz failed: seed=\{fuzz_seed} case=\{case} \{detail}")
  }
}

///|
fn fuzz_mat(rows : Int32, cols : Int32, data : Array[Double]) -> @nmf.Matrix {
  match @nmf.Matrix::new(rows, cols, data) {
    Ok(m) => m
    Err(e) => abort("unexpected matrix error: \{e}")
  }
}

///|
fn fuzz_is_negative_at(
  e : @nmf.NMFError,
  want_i : Int32,
  want_j : Int32,
) -> Bool {
  match e {
    @nmf.NMFError::NegativeEntry(i, j) => i == want_i && j == want_j
    _ => false
  }
}

///|
fn fuzz_is_invalid_rank(e : @nmf.NMFError, want : Int32) -> Bool {
  match e {
    @nmf.InvalidRank(k) => k == want
    _ => false
  }
}

///|
fn fuzz_positive_data(
  rng : TestRng,
  rows : Int32,
  cols : Int32,
) -> Array[Double] {
  let data = Array::make(rows * cols, 0.0)
  for idx in 0..<data.length() {
    data[idx] = 0.1 + rng.next_double() * 2.0
  }
  data
}

///|
fn fuzz_fitted(rng : TestRng) -> (@nmf.NMF, @nmf.Matrix) {
  let rows = 3 + rng.next_int(3)
  let cols = 3 + rng.next_int(3)
  let v = fuzz_mat(rows, cols, fuzz_positive_data(rng, rows, cols))
  let model = @nmf.NMF::new(
    @nmf.NMFConfig::new(n_components=2, max_iter=20, tol=1.0e-6, seed=7UL),
  )
  match model.fit(v) {
    Ok(_) => ()
    Err(e) => abort("unexpected fit error while preparing fuzz case: \{e}")
  }
  (model, v)
}

// ---------------------------------------------------------------------
// 1. Illegal fit inputs (80 cases).
// ---------------------------------------------------------------------

///|
test "fuzz: 80 random illegal fit inputs return typed errors" {
  let rng = TestRng::new(fuzz_seed)
  for case in 0..<80 {
    let rows = 2 + rng.next_int(6)
    let cols = 2 + rng.next_int(6)
    let max_k = if rows < cols { rows } else { cols }
    match case % 5 {
      0 => {
        // Negative entry at a known position.
        let data = fuzz_positive_data(rng, rows, cols)
        let i = rng.next_int(rows)
        let j = rng.next_int(cols)
        data[i * cols + j] = -0.25
        let v = fuzz_mat(rows, cols, data)
        let model = @nmf.NMF::new(@nmf.NMFConfig::new(n_components=1))
        match model.fit(v) {
          Err(e) =>
            fuzz_require(
              fuzz_is_negative_at(e, i, j),
              case,
              "want NegativeEntry(\{i},\{j}), got \{e}",
            )
          Ok(_) =>
            abort("seed=\{fuzz_seed} case=\{case} negative entry accepted")
        }
        // A rejected model must not expose any fitted state.
        fuzz_require(model.w() is Err(@nmf.NotFitted), case, "w after failure")
        fuzz_require(
          model.loss_value() is Err(@nmf.NotFitted),
          case,
          "loss after failure",
        )
      }
      1 => {
        // Rank 0 is rejected by configuration validation.
        let v = fuzz_mat(rows, cols, fuzz_positive_data(rng, rows, cols))
        let model = @nmf.NMF::new(@nmf.NMFConfig::new(n_components=0))
        match model.fit(v) {
          Err(e) =>
            fuzz_require(
              fuzz_is_invalid_rank(e, 0),
              case,
              "want InvalidRank(0), got \{e}",
            )
          Ok(_) => abort("seed=\{fuzz_seed} case=\{case} rank 0 accepted")
        }
      }
      2 => {
        // Rank above min(rows, cols).
        let v = fuzz_mat(rows, cols, fuzz_positive_data(rng, rows, cols))
        let bad_k = max_k + 1
        let model = @nmf.NMF::new(@nmf.NMFConfig::new(n_components=bad_k))
        match model.fit(v) {
          Err(e) =>
            fuzz_require(
              fuzz_is_invalid_rank(e, bad_k),
              case,
              "want InvalidRank(\{bad_k}), got \{e}",
            )
          Ok(_) =>
            abort("seed=\{fuzz_seed} case=\{case} rank \{bad_k} accepted")
        }
      }
      3 => {
        // Empty matrices in either direction.
        let zero_rows = if case % 2 == 0 { 0 } else { rows }
        let zero_cols = if case % 2 == 0 { cols } else { 0 }
        let v = match @nmf.Matrix::zeros(zero_rows, zero_cols) {
          Ok(m) => m
          Err(e) => abort("unexpected zeros error: \{e}")
        }
        let model = @nmf.NMF::new(@nmf.NMFConfig::new(n_components=1))
        match model.fit(v) {
          Err(e) =>
            fuzz_require(
              e is @nmf.InvalidMatrixShape(_),
              case,
              "want InvalidMatrixShape, got \{e}",
            )
          Ok(_) => abort("seed=\{fuzz_seed} case=\{case} empty matrix accepted")
        }
      }
      _ => {
        // Illegal numeric configuration.
        let v = fuzz_mat(rows, cols, fuzz_positive_data(rng, rows, cols))
        let bad = match case % 6 {
          0 => @nmf.NMFConfig::new(n_components=1, max_iter=0)
          1 => @nmf.NMFConfig::new(n_components=1, tol=-1.0)
          2 => @nmf.NMFConfig::new(n_components=1, tol=0.0 / 0.0)
          3 => @nmf.NMFConfig::new(n_components=1, l1_reg=-0.1)
          4 => @nmf.NMFConfig::new(n_components=1, l2_reg=1.0 / 0.0)
          _ => @nmf.NMFConfig::new(n_components=1, l1_reg=0.0 / 0.0)
        }
        let model = @nmf.NMF::new(bad)
        match model.fit(v) {
          Err(e) =>
            fuzz_require(
              e is @nmf.InvalidConfig(_),
              case,
              "want InvalidConfig, got \{e}",
            )
          Ok(_) =>
            abort("seed=\{fuzz_seed} case=\{case} illegal config accepted")
        }
      }
    }
  }
}

// ---------------------------------------------------------------------
// 2. Unsupported solver/loss combinations (16 cases).
// ---------------------------------------------------------------------

///|
test "fuzz: CD with KL is rejected for every generated shape" {
  let rng = TestRng::new(fuzz_seed + 1UL)
  for case in 0..<16 {
    let rows = 2 + rng.next_int(5)
    let cols = 2 + rng.next_int(5)
    let max_k = if rows < cols { rows } else { cols }
    let k = 1 + rng.next_int(max_k)
    let v = fuzz_mat(rows, cols, fuzz_positive_data(rng, rows, cols))
    let model = @nmf.NMF::new(
      @nmf.NMFConfig::new(
        n_components=k,
        solver=@nmf.Solver::CD,
        loss=@nmf.Loss::KLDivergence,
        l1_reg=if case % 2 == 0 { 0.1 } else { 0.0 },
      ),
    )
    match model.fit(v) {
      Err(e) =>
        fuzz_require(
          e is @nmf.InvalidConfig(_),
          case,
          "want InvalidConfig for CD+KL, got \{e}",
        )
      Ok(_) => abort("seed=\{fuzz_seed + 1UL} case=\{case} CD+KL accepted")
    }
    fuzz_require(model.w() is Err(@nmf.NotFitted), case, "CD+KL left state")
  }
}

// ---------------------------------------------------------------------
// 3. Illegal transform inputs (60 cases).
// ---------------------------------------------------------------------

///|
test "fuzz: 60 random illegal transform inputs return typed errors" {
  let rng = TestRng::new(fuzz_seed + 2UL)
  let (model, v) = fuzz_fitted(rng)
  let trained_h = match model.h() {
    Ok(m) => m
    Err(e) => abort("unexpected h error: \{e}")
  }
  let loss_before = model.loss_value()
  for case in 0..<60 {
    let rows = 1 + rng.next_int(4)
    match case % 3 {
      0 => {
        // Wrong feature count (never equal to H's column count).
        let bad_cols = v.cols() + 1 + rng.next_int(3)
        let x = fuzz_mat(
          rows,
          bad_cols,
          fuzz_positive_data(rng, rows, bad_cols),
        )
        match model.transform(x) {
          Err(e) =>
            fuzz_require(
              e is @nmf.DimensionMismatch(_),
              case,
              "want DimensionMismatch, got \{e}",
            )
          Ok(_) =>
            abort("seed=\{fuzz_seed + 2UL} case=\{case} wrong width accepted")
        }
      }
      1 => {
        // Zero rows with the right width.
        let x = match @nmf.Matrix::zeros(0, v.cols()) {
          Ok(m) => m
          Err(e) => abort("unexpected zeros error: \{e}")
        }
        match model.transform(x) {
          Err(e) =>
            fuzz_require(
              e is @nmf.InvalidMatrixShape(_),
              case,
              "want InvalidMatrixShape, got \{e}",
            )
          Ok(_) =>
            abort("seed=\{fuzz_seed + 2UL} case=\{case} empty x accepted")
        }
      }
      _ => {
        // Negative entry at the right width.
        let data = fuzz_positive_data(rng, rows, v.cols())
        let i = rng.next_int(rows)
        let j = rng.next_int(v.cols())
        data[i * v.cols() + j] = -1.0
        let x = fuzz_mat(rows, v.cols(), data)
        match model.transform(x) {
          Err(e) =>
            fuzz_require(
              fuzz_is_negative_at(e, i, j),
              case,
              "want NegativeEntry(\{i},\{j}), got \{e}",
            )
          Ok(_) =>
            abort("seed=\{fuzz_seed + 2UL} case=\{case} negative x accepted")
        }
      }
    }
    // Failed transforms must not touch the trained model.
    fuzz_require(
      model.loss_value() == loss_before,
      case,
      "transform changed loss",
    )
    let h_after = match model.h() {
      Ok(m) => m
      Err(e) => abort("unexpected h error after failed transform: \{e}")
    }
    for i in 0..<trained_h.rows() {
      for j in 0..<trained_h.cols() {
        fuzz_require(
          trained_h.get(i, j) == h_after.get(i, j),
          case,
          "failed transform mutated H",
        )
      }
    }
  }
}

// ---------------------------------------------------------------------
// 4. Illegal calls on an untrained model (40 cases) + oversized shapes.
// ---------------------------------------------------------------------

///|
test "fuzz: untrained model and oversized shapes fail without panic" {
  let rng = TestRng::new(fuzz_seed + 3UL)
  let model = @nmf.NMF::new(@nmf.NMFConfig::new(n_components=2))
  let x = fuzz_mat(2, 3, fuzz_positive_data(rng, 2, 3))
  for case in 0..<40 {
    match case % 8 {
      0 => fuzz_require(model.w() is Err(@nmf.NotFitted), case, "w")
      1 => fuzz_require(model.h() is Err(@nmf.NotFitted), case, "h")
      2 => fuzz_require(model.n_iter() is Err(@nmf.NotFitted), case, "n_iter")
      3 =>
        fuzz_require(
          model.loss_value() is Err(@nmf.NotFitted),
          case,
          "loss_value",
        )
      4 =>
        fuzz_require(
          model.converged() is Err(@nmf.NotFitted),
          case,
          "converged",
        )
      5 =>
        fuzz_require(
          model.transform(x) is Err(@nmf.NotFitted),
          case,
          "transform",
        )
      6 =>
        fuzz_require(
          model.inverse_transform(x) is Err(@nmf.NotFitted),
          case,
          "inverse_transform",
        )
      _ =>
        fuzz_require(
          model.reconstruction_error(x) is Err(@nmf.NotFitted),
          case,
          "reconstruction_error",
        )
    }
  }
  // Oversized shapes are rejected by the overflow check before any allocation.
  match @nmf.Matrix::zeros(50000, 50000) {
    Err(e) =>
      fuzz_require(e is @nmf.InvalidMatrixShape(_), 40, "oversized zeros")
    Ok(_) => abort("seed=\{fuzz_seed + 3UL} oversized matrix constructed")
  }
  match @nmf.Matrix::new(50000, 50000, []) {
    Err(e) => fuzz_require(e is @nmf.InvalidMatrixShape(_), 41, "oversized new")
    Ok(_) => abort("seed=\{fuzz_seed + 3UL} oversized matrix constructed")
  }
  // Negative dimensions and length mismatches stay typed errors.
  fuzz_require(
    @nmf.Matrix::new(-1, 4, []) is Err(@nmf.InvalidMatrixShape(_)),
    42,
    "negative dims",
  )
  fuzz_require(
    @nmf.Matrix::new(2, 2, [1.0]) is Err(@nmf.DimensionMismatch(_)),
    43,
    "length mismatch",
  )
}

// ---------------------------------------------------------------------
// 5. Illegal matrix entries never enter through the constructor (12 cases).
// ---------------------------------------------------------------------

///|
test "fuzz: NaN and Infinity are rejected at the matrix boundary" {
  let specials : Array[Double] = [
    0.0 / 0.0,
    1.0 / 0.0,
    -1.0 / 0.0,
    0.0 / 0.0,
    1.0 / 0.0,
    -1.0 / 0.0,
    0.0 / 0.0,
    1.0 / 0.0,
    -1.0 / 0.0,
    0.0 / 0.0,
    1.0 / 0.0,
    -1.0 / 0.0,
  ]
  for case in 0..<12 {
    let data : Array[Double] = [1.0, specials[case], 2.0, 3.0]
    match @nmf.Matrix::new(2, 2, data) {
      Err(e) =>
        fuzz_require(
          e is @nmf.NumericalError(_),
          case,
          "want NumericalError, got \{e}",
        )
      Ok(_) => abort("seed=\{fuzz_seed} case=\{case} non-finite entry accepted")
    }
  }
}

// ---------------------------------------------------------------------
// 6. Failed refit leaves the model untrained (existing atomicity contract).
// ---------------------------------------------------------------------

///|
test "fuzz: a failed refit leaves no half-fitted state" {
  let rng = TestRng::new(fuzz_seed + 4UL)
  let (model, v) = fuzz_fitted(rng)
  // The trained model works.
  fuzz_require(model.w() is Ok(_), 0, "trained model has no W")
  fuzz_require(model.loss_value() is Ok(_), 0, "trained model has no loss")
  for case in 0..<6 {
    if case % 2 == 0 {
      // Negative entry: rejected after the reset point.
      let data = fuzz_positive_data(rng, v.rows(), v.cols())
      let i = rng.next_int(v.rows())
      let j = rng.next_int(v.cols())
      data[i * v.cols() + j] = -2.0
      let bad = fuzz_mat(v.rows(), v.cols(), data)
      match model.fit(bad) {
        Err(e) =>
          fuzz_require(
            fuzz_is_negative_at(e, i, j),
            case,
            "want NegativeEntry(\{i},\{j}), got \{e}",
          )
        Ok(_) =>
          abort("seed=\{fuzz_seed + 4UL} case=\{case} negative refit accepted")
      }
    } else {
      // Empty matrix.
      let empty = match @nmf.Matrix::zeros(0, v.cols()) {
        Ok(m) => m
        Err(e) => abort("unexpected zeros error: \{e}")
      }
      match model.fit(empty) {
        Err(e) =>
          fuzz_require(
            e is @nmf.InvalidMatrixShape(_),
            case,
            "want InvalidMatrixShape, got \{e}",
          )
        Ok(_) =>
          abort("seed=\{fuzz_seed + 4UL} case=\{case} empty refit accepted")
      }
    }
    // Existing contract: any failed fit resets the model to the untrained
    // state, so nothing half-fitted is ever observable.
    fuzz_require(model.w() is Err(@nmf.NotFitted), case, "w after failed refit")
    fuzz_require(model.h() is Err(@nmf.NotFitted), case, "h after failed refit")
    fuzz_require(
      model.loss_value() is Err(@nmf.NotFitted),
      case,
      "loss after failed refit",
    )
    fuzz_require(
      model.n_iter() is Err(@nmf.NotFitted),
      case,
      "n_iter after failed refit",
    )
    fuzz_require(
      model.transform(v) is Err(@nmf.NotFitted),
      case,
      "transform after failed refit",
    )
    fuzz_require(
      model.reconstruction_error(v) is Err(@nmf.NotFitted),
      case,
      "reconstruction_error after failed refit",
    )
    // A subsequent legal fit must work again on the same handle.
    match model.fit(v) {
      Ok(_) => ()
      Err(e) => abort("seed=\{fuzz_seed + 4UL} case=\{case} refit failed: \{e}")
    }
    fuzz_require(model.w() is Ok(_), case, "refit produced no W")
  }
}

// ---------------------------------------------------------------------
// 7. Illegal sparsity inputs (30 cases) — T13 error precedence.
// ---------------------------------------------------------------------

///|
test "fuzz: sparsity rejects illegal vectors with the T13 precedence" {
  let rng = TestRng::new(fuzz_seed + 5UL)
  for case in 0..<30 {
    match case % 5 {
      0 =>
        // Empty vector.
        match @nmf.sparsity([]) {
          Err(e) =>
            fuzz_require(
              e is @nmf.InvalidConfig(_),
              case,
              "empty vector: want InvalidConfig, got \{e}",
            )
          Ok(_) => abort("seed=\{fuzz_seed + 5UL} case=\{case} empty accepted")
        }
      1 => {
        // Negative entry at a known index.
        let n = 2 + rng.next_int(5)
        let v = Array::make(n, 1.0)
        let idx = rng.next_int(n)
        v[idx] = -0.5
        match @nmf.sparsity(v) {
          Err(e) =>
            fuzz_require(
              fuzz_is_negative_at(e, 0, idx),
              case,
              "want NegativeEntry(0,\{idx}), got \{e}",
            )
          Ok(_) =>
            abort("seed=\{fuzz_seed + 5UL} case=\{case} negative accepted")
        }
      }
      2 => {
        // NaN anywhere is a numerical error.
        let n = 2 + rng.next_int(5)
        let v = Array::make(n, 1.0)
        v[rng.next_int(n)] = 0.0 / 0.0
        match @nmf.sparsity(v) {
          Err(e) =>
            fuzz_require(
              e is @nmf.NumericalError(_),
              case,
              "NaN: want NumericalError, got \{e}",
            )
          Ok(_) => abort("seed=\{fuzz_seed + 5UL} case=\{case} NaN accepted")
        }
      }
      3 => {
        // +Infinity and -Infinity: non-finite is checked before negativity,
        // so -Infinity must be NumericalError, not NegativeEntry.
        let n = 2 + rng.next_int(5)
        let v = Array::make(n, 1.0)
        v[rng.next_int(n)] = if case % 2 == 0 { 1.0 / 0.0 } else { -1.0 / 0.0 }
        match @nmf.sparsity(v) {
          Err(e) =>
            fuzz_require(
              e is @nmf.NumericalError(_),
              case,
              "Inf: want NumericalError, got \{e}",
            )
          Ok(_) => abort("seed=\{fuzz_seed + 5UL} case=\{case} Inf accepted")
        }
      }
      _ =>
        // A single negative element at index 0, exercising the boundary.
        match @nmf.sparsity([-1.0]) {
          Err(e) =>
            fuzz_require(
              fuzz_is_negative_at(e, 0, 0),
              case,
              "want NegativeEntry(0,0), got \{e}",
            )
          Ok(_) =>
            abort("seed=\{fuzz_seed + 5UL} case=\{case} negative accepted")
        }
    }
  }
}
