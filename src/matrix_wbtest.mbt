// Whitebox tests: unchecked kernels, internal helpers, clone semantics.

///|
fn wmat(rows : Int32, cols : Int32, data : Array[Double]) -> Matrix {
  match Matrix::new(rows, cols, data) {
    Ok(m) => m
    Err(e) => abort("unexpected matrix error: \{e}")
  }
}

///|
test "wb matrix: clone is independent of the original" {
  let a = wmat(2, 2, [1.0, 2.0, 3.0, 4.0])
  let b = a.clone()
  match b.set(0, 0, 99.0) {
    Ok(_) => ()
    Err(e) => abort("set failed: \{e}")
  }
  assert_eq(a.get(0, 0), Ok(1.0))
  assert_eq(b.get(0, 0), Ok(99.0))
}

///|
test "wb matrix: zeros_buf and all_finite" {
  let z = zeros_buf(2, 3)
  assert_eq(z.rows, 2)
  assert_eq(z.cols, 3)
  assert_eq(z.data.length(), 6)
  assert_true(all_finite(z.data))
  assert_true(all_finite([1.0, -2.0, 0.0]))
  assert_true(!all_finite([1.0, 0.0 / 0.0]))
  assert_true(!all_finite([1.0 / 0.0]))
  assert_true(!all_finite([-1.0 / 0.0]))
}

///|
test "wb matrix: matmul_into clears dirty output buffer" {
  let a = wmat(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  let b = wmat(3, 2, [7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
  let out = wmat(2, 2, [1000.0, 1000.0, 1000.0, 1000.0])
  matmul_into(a, b, out)
  assert_eq(out.get(0, 0), Ok(58.0))
  assert_eq(out.get(0, 1), Ok(64.0))
  assert_eq(out.get(1, 0), Ok(139.0))
  assert_eq(out.get(1, 1), Ok(154.0))
}

///|
test "wb matrix: kernels agree with checked ops on random inputs" {
  let rng = SplitMix64::new(2026UL)
  for _ in 0..<50 {
    let n = 1 + (rng.next_u64() % 5UL).to_int()
    let p = 1 + (rng.next_u64() % 5UL).to_int()
    let m = 1 + (rng.next_u64() % 5UL).to_int()
    let a_data = Array::make(n * p, 0.0)
    let b_data = Array::make(p * m, 0.0)
    for idx in 0..<a_data.length() {
      a_data[idx] = rng.next_double() * 10.0
    }
    for idx in 0..<b_data.length() {
      b_data[idx] = rng.next_double() * 10.0
    }
    let a = wmat(n, p, a_data)
    let b = wmat(p, m, b_data)
    // matmul_into vs checked matmul
    let checked = match matmul(a, b) {
      Ok(c) => c
      Err(e) => abort("matmul failed: \{e}")
    }
    let out = zeros_buf(n, m)
    matmul_into(a, b, out)
    for idx in 0..<out.data.length() {
      assert_eq(out.data[idx], checked.data[idx])
    }
    // matmul_at_b_into == checked AᵀA
    let at_b = match matmul_at_b(a, a) {
      Ok(x) => x
      Err(e) => abort("matmul_at_b failed: \{e}")
    }
    let at_b_out = zeros_buf(p, p)
    matmul_at_b_into(a, a, at_b_out)
    for idx in 0..<at_b_out.data.length() {
      assert_eq(at_b_out.data[idx], at_b.data[idx])
    }
    // matmul_a_bt_into == a · transpose(b2) where b2 has a's col count
    let b2_data = Array::make(m * p, 0.0)
    for idx in 0..<b2_data.length() {
      b2_data[idx] = rng.next_double() * 10.0
    }
    let b2 = wmat(m, p, b2_data)
    let a_bt = match matmul_a_bt(a, b2) {
      Ok(x) => x
      Err(e) => abort("matmul_a_bt failed: \{e}")
    }
    let a_bt_out = zeros_buf(n, m)
    matmul_a_bt_into(a, b2, a_bt_out)
    for idx in 0..<a_bt_out.data.length() {
      assert_eq(a_bt_out.data[idx], a_bt.data[idx])
    }
  }
}

///|
test "wb matrix: validate_shape boundary values" {
  assert_eq(validate_shape(0, 0), Ok(()))
  assert_eq(validate_shape(46340, 46340), Ok(()))
  assert_true(validate_shape(46341, 46341) is Err(InvalidMatrixShape(_)))
  assert_true(validate_shape(-1, 0) is Err(InvalidMatrixShape(_)))
}
