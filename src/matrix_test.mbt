// Blackbox tests for the public Matrix API.

///|
fn mat(rows : Int32, cols : Int32, data : Array[Double]) -> @nmf.Matrix {
  unwrap_matrix(@nmf.Matrix::new(rows, cols, data))
}

///|
test "matrix: creation and dimension queries" {
  let m = mat(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  assert_eq(m.rows(), 2)
  assert_eq(m.cols(), 3)
}

///|
test "matrix: get reads row-major entries" {
  let m = mat(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  assert_eq(m.get(0, 0), Ok(1.0))
  assert_eq(m.get(0, 2), Ok(3.0))
  assert_eq(m.get(1, 0), Ok(4.0))
  assert_eq(m.get(1, 2), Ok(6.0))
}

///|
test "matrix: set writes and rejects non-finite values" {
  let m = mat(2, 2, [0.0, 0.0, 0.0, 0.0])
  assert_eq(m.set(1, 1, 7.5), Ok(()))
  assert_eq(m.get(1, 1), Ok(7.5))
  assert_true(m.set(0, 0, nan()) is Err(@nmf.NumericalError(_)))
  assert_true(m.set(0, 0, inf()) is Err(@nmf.NumericalError(_)))
  assert_true(m.set(0, 0, -inf()) is Err(@nmf.NumericalError(_)))
  // rejected writes must not modify the entry
  assert_eq(m.get(0, 0), Ok(0.0))
}

///|
test "matrix: get/set out of bounds" {
  let m = mat(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  assert_true(m.get(2, 0) is Err(@nmf.DimensionMismatch(_)))
  assert_true(m.get(0, 3) is Err(@nmf.DimensionMismatch(_)))
  assert_true(m.get(-1, 0) is Err(@nmf.DimensionMismatch(_)))
  assert_true(m.get(0, -1) is Err(@nmf.DimensionMismatch(_)))
  assert_true(m.set(2, 0, 1.0) is Err(@nmf.DimensionMismatch(_)))
  assert_true(m.set(0, 3, 1.0) is Err(@nmf.DimensionMismatch(_)))
}

///|
test "matrix: negative entries are allowed (generic container)" {
  let m = mat(1, 2, [-3.0, -0.5])
  assert_eq(m.get(0, 0), Ok(-3.0))
}

///|
test "matrix: negative dimensions rejected" {
  assert_true(@nmf.Matrix::new(-1, 2, []) is Err(@nmf.InvalidMatrixShape(_)))
  assert_true(@nmf.Matrix::new(2, -1, []) is Err(@nmf.InvalidMatrixShape(_)))
  assert_true(@nmf.Matrix::zeros(-2, 3) is Err(@nmf.InvalidMatrixShape(_)))
}

///|
test "matrix: rows*cols overflow rejected" {
  assert_true(
    @nmf.Matrix::new(2147483647, 2, []) is Err(@nmf.InvalidMatrixShape(_)),
  )
  assert_true(
    @nmf.Matrix::zeros(65536, 65536) is Err(@nmf.InvalidMatrixShape(_)),
  )
}

///|
test "matrix: data length mismatch rejected" {
  assert_true(
    @nmf.Matrix::new(2, 3, [1.0, 2.0]) is Err(@nmf.DimensionMismatch(_)),
  )
  assert_true(
    @nmf.Matrix::new(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    is Err(@nmf.DimensionMismatch(_)),
  )
}

///|
test "matrix: NaN and Infinity entries rejected at creation" {
  assert_true(
    @nmf.Matrix::new(1, 2, [1.0, nan()]) is Err(@nmf.NumericalError(_)),
  )
  assert_true(
    @nmf.Matrix::new(1, 2, [inf(), 1.0]) is Err(@nmf.NumericalError(_)),
  )
  assert_true(
    @nmf.Matrix::new(1, 2, [1.0, -inf()]) is Err(@nmf.NumericalError(_)),
  )
}

///|
test "matrix: zero-sized dimensions allowed with empty data" {
  let m = match @nmf.Matrix::zeros(0, 3) {
    Ok(m) => m
    Err(e) => abort("unexpected: \{e}")
  }
  assert_eq(m.rows(), 0)
  assert_eq(m.cols(), 3)
  assert_true(m.get(0, 0) is Err(@nmf.DimensionMismatch(_)))
  assert_true(@nmf.Matrix::new(0, 3, [1.0]) is Err(@nmf.DimensionMismatch(_)))
}

///|
test "matrix: zeros creates all-zero entries" {
  let m = match @nmf.Matrix::zeros(2, 2) {
    Ok(m) => m
    Err(e) => abort("unexpected: \{e}")
  }
  assert_eq(m.get(0, 0), Ok(0.0))
  assert_eq(m.get(1, 1), Ok(0.0))
}

///|
test "matrix: matmul hand-computed 2x3 by 3x2" {
  let a = mat(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  let b = mat(3, 2, [7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
  let c = match @nmf.matmul(a, b) {
    Ok(c) => c
    Err(e) => abort("unexpected: \{e}")
  }
  assert_eq(c.rows(), 2)
  assert_eq(c.cols(), 2)
  assert_eq(c.get(0, 0), Ok(58.0))
  assert_eq(c.get(0, 1), Ok(64.0))
  assert_eq(c.get(1, 0), Ok(139.0))
  assert_eq(c.get(1, 1), Ok(154.0))
}

///|
test "matrix: matmul dimension mismatch" {
  let a = mat(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  let b = mat(2, 2, [1.0, 2.0, 3.0, 4.0])
  assert_true(@nmf.matmul(a, b) is Err(@nmf.DimensionMismatch(_)))
}

///|
test "matrix: matmul_at_b hand-computed A^T A" {
  let a = mat(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  let c = match @nmf.matmul_at_b(a, a) {
    Ok(c) => c
    Err(e) => abort("unexpected: \{e}")
  }
  assert_eq(c.rows(), 3)
  assert_eq(c.cols(), 3)
  assert_eq(c.get(0, 0), Ok(17.0))
  assert_eq(c.get(0, 1), Ok(22.0))
  assert_eq(c.get(0, 2), Ok(27.0))
  assert_eq(c.get(1, 1), Ok(29.0))
  assert_eq(c.get(1, 2), Ok(36.0))
  assert_eq(c.get(2, 2), Ok(45.0))
}

///|
test "matrix: matmul_a_bt hand-computed A A^T" {
  let a = mat(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  let c = match @nmf.matmul_a_bt(a, a) {
    Ok(c) => c
    Err(e) => abort("unexpected: \{e}")
  }
  assert_eq(c.rows(), 2)
  assert_eq(c.cols(), 2)
  assert_eq(c.get(0, 0), Ok(14.0))
  assert_eq(c.get(0, 1), Ok(32.0))
  assert_eq(c.get(1, 0), Ok(32.0))
  assert_eq(c.get(1, 1), Ok(77.0))
}

///|
test "matrix: transpose-multiply dimension mismatches" {
  let a = mat(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  let b = mat(3, 2, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  assert_true(@nmf.matmul_at_b(a, b) is Err(@nmf.DimensionMismatch(_)))
  assert_true(@nmf.matmul_a_bt(a, b) is Err(@nmf.DimensionMismatch(_)))
}

///|
test "matrix: transpose swaps dimensions and entries" {
  let a = mat(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  let t = @nmf.transpose(a)
  assert_eq(t.rows(), 3)
  assert_eq(t.cols(), 2)
  assert_eq(t.get(0, 0), Ok(1.0))
  assert_eq(t.get(2, 0), Ok(3.0))
  assert_eq(t.get(0, 1), Ok(4.0))
  assert_eq(t.get(2, 1), Ok(6.0))
  // double transpose is identity
  let tt = @nmf.transpose(t)
  for i in 0..<2 {
    for j in 0..<3 {
      assert_eq(tt.get(i, j), a.get(i, j))
    }
  }
}

///|
test "matrix: (A^T B)^T == B^T A property" {
  let a = mat(3, 2, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  let b = mat(3, 2, [7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
  let lhs = match @nmf.matmul_at_b(a, b) {
    Ok(m) => @nmf.transpose(m)
    Err(e) => abort("unexpected: \{e}")
  }
  let rhs = match @nmf.matmul_at_b(b, a) {
    Ok(m) => m
    Err(e) => abort("unexpected: \{e}")
  }
  for i in 0..<lhs.rows() {
    for j in 0..<lhs.cols() {
      assert_eq(lhs.get(i, j), rhs.get(i, j))
    }
  }
}

///|
test "matrix: matmul with non-square identity" {
  let a = mat(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  let eye = mat(3, 3, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
  let c = match @nmf.matmul(a, eye) {
    Ok(c) => c
    Err(e) => abort("unexpected: \{e}")
  }
  for i in 0..<2 {
    for j in 0..<3 {
      assert_eq(c.get(i, j), a.get(i, j))
    }
  }
}

///|
test "matrix: frobenius norm hand-computed" {
  let a = mat(2, 2, [3.0, 4.0, 0.0, 0.0])
  let n = unwrap_double(@nmf.frobenius_norm(a))
  assert_true((n - 5.0).abs() < 1.0e-12)
  let z = unwrap_matrix(@nmf.Matrix::zeros(2, 2))
  assert_eq(@nmf.frobenius_norm(z), Ok(0.0))
}

///|
test "matrix: frobenius norm overflow is an error, not Infinity" {
  let big = mat(1, 2, [1.0e200, 1.0e200])
  assert_true(@nmf.frobenius_norm(big) is Err(@nmf.NumericalError(_)))
}
