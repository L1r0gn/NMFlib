// Blackbox tests for the Frobenius objective F = ½‖V − WH‖²_F.

///|
test "loss: hand-computed objective equals 7" {
  // V=[[1,2],[3,4]], W=[[1],[1]], H=[[1,1]] -> WH=[[1,1],[1,1]]
  // diffs [0,1,2,3], sum of squares 14, F = 7
  let v = mat(2, 2, [1.0, 2.0, 3.0, 4.0])
  let w = mat(2, 1, [1.0, 1.0])
  let h = mat(1, 2, [1.0, 1.0])
  assert_eq(@nmf.frobenius_objective(v, w, h), Ok(7.0))
}

///|
test "loss: exact factorization gives zero objective" {
  // W=[[1],[3]], H=[[2,4]] -> WH=[[2,4],[6,12]] exactly
  let v = mat(2, 2, [2.0, 4.0, 6.0, 12.0])
  let w = mat(2, 1, [1.0, 3.0])
  let h = mat(1, 2, [2.0, 4.0])
  assert_eq(@nmf.frobenius_objective(v, w, h), Ok(0.0))
}

///|
test "loss: all-zero matrices give zero objective" {
  let v = unwrap_matrix(@nmf.Matrix::zeros(3, 3))
  let w = unwrap_matrix(@nmf.Matrix::zeros(3, 2))
  let h = unwrap_matrix(@nmf.Matrix::zeros(2, 3))
  assert_eq(@nmf.frobenius_objective(v, w, h), Ok(0.0))
}

///|
test "loss: objective is symmetric in the residual sign" {
  // F(V, W, H) with V=0 equals ½‖WH‖²
  let v = unwrap_matrix(@nmf.Matrix::zeros(2, 2))
  let w = mat(2, 1, [1.0, 2.0])
  let h = mat(1, 2, [3.0, 4.0])
  // WH = [[3,4],[6,8]], sum of squares = 9+16+36+64 = 125
  assert_eq(@nmf.frobenius_objective(v, w, h), Ok(62.5))
}

///|
test "loss: dimension mismatches are rejected" {
  let v = mat(2, 2, [1.0, 2.0, 3.0, 4.0])
  let w_ok = mat(2, 1, [1.0, 1.0])
  let h_ok = mat(1, 2, [1.0, 1.0])
  let w_bad_rows = mat(1, 1, [1.0])
  let h_bad_cols = mat(1, 1, [1.0])
  let w_bad_rank = mat(2, 2, [1.0, 0.0, 0.0, 1.0])
  assert_true(
    @nmf.frobenius_objective(v, w_bad_rows, h_ok)
    is Err(@nmf.DimensionMismatch(_)),
  )
  assert_true(
    @nmf.frobenius_objective(v, w_ok, h_bad_cols)
    is Err(@nmf.DimensionMismatch(_)),
  )
  assert_true(
    @nmf.frobenius_objective(v, w_bad_rank, h_ok)
    is Err(@nmf.DimensionMismatch(_)),
  )
}

///|
test "loss: overflow returns NumericalError, never Infinity" {
  let v = mat(1, 1, [1.0])
  let w = mat(1, 1, [1.0e200])
  let h = mat(1, 1, [1.0e200])
  assert_true(@nmf.frobenius_objective(v, w, h) is Err(@nmf.NumericalError(_)))
}

///|
test "loss: matches direct residual computation on a rank-2 case" {
  // V=[[1,2],[3,4]], W=I2, H=[[0.5,1],[2,3]]
  // WH=[[0.5,1],[2,3]], diffs [0.5,1,1,1], sum=0.25+1+1+1=3.25, F=1.625
  let v = mat(2, 2, [1.0, 2.0, 3.0, 4.0])
  let w = mat(2, 2, [1.0, 0.0, 0.0, 1.0])
  let h = mat(2, 2, [0.5, 1.0, 2.0, 3.0])
  assert_eq(@nmf.frobenius_objective(v, w, h), Ok(1.625))
}
