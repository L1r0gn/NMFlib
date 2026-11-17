// Blackbox tests for the T13 Hoyer sparsity metric:
// hand-computed values, scale/permutation invariance, zero and dimension
// boundaries, illegal input, extreme magnitudes, matrix row/column direction,
// range property, and non-interference with trained models.

///|
fn mtx_sparsity(x : Array[Double]) -> Double {
  unwrap_double(@nmf.sparsity(x))
}

///|
fn mtx_close(a : Double, b : Double) -> Bool {
  (a - b).abs() <= 1.0e-12 * (1.0 + a.abs() + b.abs())
}

///|
fn mtx_assert_close(a : Double, b : Double, label : String) -> Unit {
  if !mtx_close(a, b) {
    abort("\{label}: \{a} vs \{b}")
  }
}

///|
fn mtx_col(m : @nmf.Matrix, j : Int32) -> Array[Double] {
  let out = Array::make(m.rows(), 0.0)
  for i in 0..<m.rows() {
    out[i] = unwrap_double(m.get(i, j))
  }
  out
}

///|
fn mtx_row(m : @nmf.Matrix, i : Int32) -> Array[Double] {
  let out = Array::make(m.cols(), 0.0)
  for j in 0..<m.cols() {
    out[j] = unwrap_double(m.get(i, j))
  }
  out
}

// ---------------------------------------------------------------------
// A. Hand-computed standard examples.
// ---------------------------------------------------------------------

///|
test "hoyer A: hand-computed values for non-negative vectors" {
  // Equal entries are not sparse at all.
  assert_eq(@nmf.sparsity([1.0, 1.0, 1.0, 1.0]), Ok(0.0))
  assert_eq(@nmf.sparsity([2.0, 2.0, 2.0]), Ok(0.0))
  assert_eq(@nmf.sparsity([0.5, 0.5, 0.5]), Ok(0.0))
  // Exactly one non-zero entry is maximally sparse.
  assert_eq(@nmf.sparsity([1.0, 0.0, 0.0, 0.0]), Ok(1.0))
  assert_eq(@nmf.sparsity([0.0, 0.0, 3.0]), Ok(1.0))
  // (√4 - 2/√2) / (√4 - 1) = 0.5857864376269051
  mtx_assert_close(
    mtx_sparsity([1.0, 1.0, 0.0, 0.0]),
    0.5857864376269051,
    "two of four",
  )
  // (√4 - 10/√30) / (√4 - 1) = 0.1742581416494462
  mtx_assert_close(
    mtx_sparsity([1.0, 2.0, 3.0, 4.0]),
    0.1742581416494462,
    "1 2 3 4",
  )
  // (√5 - 3/√3) / (√5 - 1) = 0.40775845593087384
  mtx_assert_close(
    mtx_sparsity([1.0, 0.0, 1.0, 0.0, 1.0]),
    0.40775845593087384,
    "1 0 1 0 1",
  )
  // (√2 - 3/√5) / (√2 - 1) = 0.17520617977219377
  mtx_assert_close(mtx_sparsity([1.0, 2.0]), 0.17520617977219377, "1 2")
  // (√6 - 2/√2) / (√6 - 1) = 0.7643027222504615
  mtx_assert_close(
    mtx_sparsity([1.0, 2.0, 0.0, 0.0, 0.0, 0.0]),
    0.7643027222504615,
    "1 2 zero-padded",
  )
}

// ---------------------------------------------------------------------
// B. Scale invariance.
// ---------------------------------------------------------------------

///|
test "hoyer B: positive scaling does not change the value" {
  let base : Array[Double] = [1.0, 2.0, 0.0, 3.0, 0.0]
  let want = mtx_sparsity(base)
  for factor in [2.0, 1000.0, 1.0e-200, 1.0e200] {
    let scaled = Array::make(base.length(), 0.0)
    for i in 0..<base.length() {
      scaled[i] = base[i] * factor
    }
    mtx_assert_close(mtx_sparsity(scaled), want, "scale \{factor}")
  }
}

// ---------------------------------------------------------------------
// C. Permutation invariance.
// ---------------------------------------------------------------------

///|
test "hoyer C: permuting entries does not change the value" {
  let a : Array[Double] = [1.0, 2.0, 3.0, 4.0, 0.0]
  let b : Array[Double] = [4.0, 0.0, 3.0, 2.0, 1.0]
  let c : Array[Double] = [2.0, 4.0, 1.0, 0.0, 3.0]
  let want = mtx_sparsity(a)
  mtx_assert_close(mtx_sparsity(b), want, "permutation 1")
  mtx_assert_close(mtx_sparsity(c), want, "permutation 2")
}

// ---------------------------------------------------------------------
// D. Zero vectors (PLAN §8.7 conventions).
// ---------------------------------------------------------------------

///|
test "hoyer D: all-zero vectors are maximally sparse" {
  assert_eq(@nmf.sparsity([0.0, 0.0, 0.0]), Ok(1.0))
  assert_eq(@nmf.sparsity([0.0, 0.0]), Ok(1.0))
  assert_eq(@nmf.sparsity([0.0, 0.0, 0.0, 0.0, 0.0]), Ok(1.0))
  // n = 1 takes precedence over the all-zero rule (documented in metrics.mbt).
  assert_eq(@nmf.sparsity([0.0]), Ok(0.0))
  // -0.0 is numerically zero and is accepted.
  assert_eq(@nmf.sparsity([-0.0]), Ok(0.0))
  // Zero rows and zero columns of a matrix.
  let m = mat(2, 3, [0.0, 0.0, 0.0, 0.0, 1.0, 2.0])
  assert_eq(@nmf.sparsity(mtx_row(m, 0)), Ok(1.0))
  mtx_assert_close(mtx_sparsity(mtx_row(m, 1)), 0.5333100066722766, "row 1")
  assert_eq(@nmf.sparsity(mtx_col(m, 0)), Ok(1.0))
  // All-zero matrix: every row and every column is maximally sparse.
  let z = mat(3, 2, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
  for i in 0..<3 {
    assert_eq(@nmf.sparsity(mtx_row(z, i)), Ok(1.0))
  }
  for j in 0..<2 {
    assert_eq(@nmf.sparsity(mtx_col(z, j)), Ok(1.0))
  }
}

// ---------------------------------------------------------------------
// E. Dimension boundaries.
// ---------------------------------------------------------------------

///|
test "hoyer E: empty, single-element, row and column shapes" {
  assert_true(@nmf.sparsity([]) is Err(@nmf.NMFError::InvalidConfig(_)))
  // n = 1: denominator √n − 1 is zero, convention fixes 0.0.
  assert_eq(@nmf.sparsity([5.0]), Ok(0.0))
  assert_eq(@nmf.sparsity([1.0e-300]), Ok(0.0))
  // 1×n matrix: the single row has length n.
  let row_mat = mat(1, 4, [1.0, 1.0, 0.0, 0.0])
  mtx_assert_close(mtx_sparsity(mtx_row(row_mat, 0)), 0.5857864376269051, "1xn")
  // n×1 matrix: the single column has length n.
  let col_mat = mat(4, 1, [1.0, 1.0, 0.0, 0.0])
  mtx_assert_close(mtx_sparsity(mtx_col(col_mat, 0)), 0.5857864376269051, "nx1")
  // Zero-sized dimensions are expressible in Matrix; extracting from them
  // yields the empty vector, which has no defined value.
  let no_rows = unwrap_matrix(@nmf.Matrix::zeros(0, 3))
  assert_eq(no_rows.rows(), 0)
  assert_eq(
    @nmf.sparsity(mtx_col(no_rows, 0)),
    Err(
      @nmf.NMFError::InvalidConfig(
        "sparsity: empty vector has no defined value",
      ),
    ),
  )
  let no_cols = unwrap_matrix(@nmf.Matrix::zeros(3, 0))
  assert_eq(no_cols.cols(), 0)
  assert_true(
    @nmf.sparsity(mtx_row(no_cols, 0)) is Err(@nmf.NMFError::InvalidConfig(_)),
  )
}

// ---------------------------------------------------------------------
// F. Illegal input.
// ---------------------------------------------------------------------

///|
test "hoyer F: negative and non-finite entries are typed errors" {
  assert_true(
    @nmf.sparsity([1.0, -2.0, 3.0]) is Err(@nmf.NMFError::NegativeEntry(0, 1)),
  )
  assert_true(
    @nmf.sparsity([-1.0e-300]) is Err(@nmf.NMFError::NegativeEntry(0, 0)),
  )
  // Non-finite input is a numerical error, never silently replaced by 0.
  assert_true(@nmf.sparsity([nan()]) is Err(@nmf.NMFError::NumericalError(_)))
  assert_true(@nmf.sparsity([inf()]) is Err(@nmf.NMFError::NumericalError(_)))
  assert_true(
    @nmf.sparsity([1.0, 2.0, nan()]) is Err(@nmf.NMFError::NumericalError(_)),
  )
  assert_true(
    @nmf.sparsity([1.0, inf()]) is Err(@nmf.NMFError::NumericalError(_)),
  )
  // Non-finite is checked before negativity, so -Infinity is NumericalError
  // rather than NegativeEntry.
  assert_true(@nmf.sparsity([-inf()]) is Err(@nmf.NMFError::NumericalError(_)))
  assert_true(
    @nmf.sparsity([1.0, -inf()]) is Err(@nmf.NMFError::NumericalError(_)),
  )
}

// ---------------------------------------------------------------------
// G. Extreme magnitudes.
// ---------------------------------------------------------------------

///|
test "hoyer G: huge and tiny magnitudes stay finite and correct" {
  // A naive Σx² would overflow here; scaling by the maximum keeps it finite.
  mtx_assert_close(
    mtx_sparsity([1.0e308, 1.0e308, 0.0]),
    0.43417375120630186,
    "1e308 pair",
  )
  // A naive Σx² would underflow to zero here.
  mtx_assert_close(
    mtx_sparsity([1.0e-308, 2.0e-308]),
    0.17520617977219377,
    "1e-308 pair",
  )
  // Subnormal entries: equal values are still "not sparse".
  assert_eq(@nmf.sparsity([5.0e-324, 5.0e-324]), Ok(0.0))
  assert_eq(@nmf.sparsity([5.0e-324, 0.0]), Ok(1.0))
  // Huge dynamic range: the tiny entry cannot change the result materially,
  // but the vector must not overflow or produce NaN.
  mtx_assert_close(
    mtx_sparsity([1.0e308, 1.0e308, 1.0e-308]),
    0.43417375120630186,
    "wide dynamic range",
  )
  mtx_assert_close(
    mtx_sparsity([1.0e308, 1.0e-308]),
    1.0,
    "two-entry wide range",
  )
  // Tiny vectors are scale equivalents of small integer vectors.
  mtx_assert_close(
    mtx_sparsity([1.0e-300, 2.0e-300, 3.0e-300]),
    mtx_sparsity([1.0, 2.0, 3.0]),
    "tiny triple",
  )
}

// ---------------------------------------------------------------------
// H. Matrix direction and W/H semantics.
// ---------------------------------------------------------------------

///|
test "hoyer H: row and column directions are distinct on a non-square matrix" {
  // 2×3: rows [1,1,1], [1,0,1]; columns [1,1], [1,0], [1,1].
  let m = mat(2, 3, [1.0, 1.0, 1.0, 1.0, 0.0, 1.0])
  assert_eq(@nmf.sparsity(mtx_row(m, 0)), Ok(0.0))
  mtx_assert_close(
    mtx_sparsity(mtx_row(m, 1)),
    0.43417375120630186,
    "row 1 of 2x3",
  )
  assert_eq(@nmf.sparsity(mtx_col(m, 0)), Ok(0.0))
  assert_eq(@nmf.sparsity(mtx_col(m, 1)), Ok(1.0))
  assert_eq(@nmf.sparsity(mtx_col(m, 2)), Ok(0.0))
  // The two directions must not be confused: the same matrix gives different
  // multisets of values.
  let row_values = [mtx_sparsity(mtx_row(m, 0)), mtx_sparsity(mtx_row(m, 1))]
  let col_values = [
    mtx_sparsity(mtx_col(m, 0)),
    mtx_sparsity(mtx_col(m, 1)),
    mtx_sparsity(mtx_col(m, 2)),
  ]
  assert_eq(row_values.length(), 2)
  assert_eq(col_values.length(), 3)
  assert_true(col_values[1] > row_values[1])
}

///|
test "hoyer H: W columns and H rows are the meaningful directions" {
  // W is m×k: each column is a basis vector of length m = 4.
  let w = mat(4, 2, [0.5, 0.0, 0.5, 0.0, 0.5, 1.0, 0.5, 1.0])
  assert_eq(@nmf.sparsity(mtx_col(w, 0)), Ok(0.0))
  mtx_assert_close(
    mtx_sparsity(mtx_col(w, 1)),
    0.5857864376269051,
    "W column 1",
  )
  // H is k×n: each row is a component activation vector of length n = 5.
  let h = mat(2, 5, [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
  assert_eq(@nmf.sparsity(mtx_row(h, 0)), Ok(0.0))
  assert_eq(@nmf.sparsity(mtx_row(h, 1)), Ok(1.0))
  // A whole-matrix flattening is not a substitute for either direction: the
  // flattened vectors have different lengths and different values.
  let flat = Array::make(8, 0.0)
  for i in 0..<4 {
    for j in 0..<2 {
      flat[i * 2 + j] = unwrap_double(w.get(i, j))
    }
  }
  mtx_assert_close(
    mtx_sparsity(flat),
    0.28386477150941114,
    "flattened W is a different quantity",
  )
}

// ---------------------------------------------------------------------
// I. Range property on generated non-negative vectors.
// ---------------------------------------------------------------------

///|
priv struct MtxRng {
  mut state : UInt64
}

///|
fn MtxRng::next_double(self : MtxRng) -> Double {
  self.state = (self.state * 1664525UL + 1013904223UL) & 0xFFFFFFFFUL
  (self.state >> 8).to_double() / 16777216.0
}

///|
test "hoyer I: values stay in [0,1] for generated non-negative vectors" {
  let rng : MtxRng = { state: 20261116UL, }
  // Decades used to exercise wide dynamic ranges without a power operator.
  let scales : Array[Double] = [
    1.0e-6, 1.0e-4, 1.0e-2, 1.0, 1.0e2, 1.0e4, 1.0e6,
  ]
  for trial in 0..<200 {
    let n = 2 + (rng.next_double() * 7.0).to_int()
    let v = Array::make(n, 0.0)
    let mut nonzeros = 0
    for i in 0..<n {
      // Mix exact zeros with random magnitudes spanning many decades.
      let draw = rng.next_double()
      let value = if draw < 0.3 {
        0.0
      } else {
        draw * scales[(rng.next_double() * 7.0).to_int()]
      }
      v[i] = value
      if value != 0.0 {
        nonzeros = nonzeros + 1
      }
    }
    let s = mtx_sparsity(v)
    assert_true(s >= 0.0)
    assert_true(s <= 1.0)
    // A vector with no non-zero entry must hit the all-zero convention.
    if nonzeros == 0 {
      assert_eq(s, 1.0)
    }
    // A single non-zero entry must hit maximal sparsity exactly.
    if nonzeros == 1 {
      assert_eq(s, 1.0)
    }
    if trial == 0 {
      assert_true(!s.is_nan() && !s.is_inf())
    }
  }
}

// ---------------------------------------------------------------------
// J. The metric is independent of the NMF training path.
// ---------------------------------------------------------------------

///|
test "hoyer J: measuring a fitted model does not change its state" {
  let v = mat(4, 3, [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 2.0, 9.0, 1.0,
  ])
  let model = @nmf.NMF::new(
    @nmf.NMFConfig::new(n_components=2, l1_reg=0.1, l2_reg=0.2, seed=42UL),
  )
  assert_eq(model.fit(v), Ok(()))
  let loss_before = unwrap_double(model.loss_value())
  let err_before = unwrap_double(model.reconstruction_error(v))
  let w = unwrap_matrix(model.w())
  let h = unwrap_matrix(model.h())
  let w_sparsity = Array::make(w.cols(), 0.0)
  for j in 0..<w.cols() {
    w_sparsity[j] = mtx_sparsity(mtx_col(w, j))
  }
  let h_sparsity = Array::make(h.rows(), 0.0)
  for i in 0..<h.rows() {
    h_sparsity[i] = mtx_sparsity(mtx_row(h, i))
  }
  for s in w_sparsity {
    assert_true(s >= 0.0 && s <= 1.0)
  }
  for s in h_sparsity {
    assert_true(s >= 0.0 && s <= 1.0)
  }
  // Pure measurement: the trained model is untouched.
  assert_eq(model.loss_value(), Ok(loss_before))
  assert_eq(model.reconstruction_error(v), Ok(err_before))
  let w_after = unwrap_matrix(model.w())
  let h_after = unwrap_matrix(model.h())
  for i in 0..<w.rows() {
    for j in 0..<w.cols() {
      assert_eq(unwrap_double(w.get(i, j)), unwrap_double(w_after.get(i, j)))
    }
  }
  for i in 0..<h.rows() {
    for j in 0..<h.cols() {
      assert_eq(unwrap_double(h.get(i, j)), unwrap_double(h_after.get(i, j)))
    }
  }
}
