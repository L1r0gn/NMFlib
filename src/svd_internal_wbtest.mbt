// Whitebox tests for the internal truncated SVD used by NNDSVD.

///|
fn svd_test_mat(rows : Int32, cols : Int32, data : Array[Double]) -> Matrix {
  match Matrix::new(rows, cols, data) {
    Ok(m) => m
    Err(e) => abort("unexpected matrix error: \{e}")
  }
}

///|
fn svd_test_components(
  r : Result[Array[SvdComponent], NMFError],
) -> Array[SvdComponent] {
  match r {
    Ok(c) => c
    Err(e) => abort("unexpected SVD error: \{e}")
  }
}

///|
fn svd_test_double(r : Result[Double, NMFError]) -> Double {
  match r {
    Ok(x) => x
    Err(e) => abort("unexpected: \{e}")
  }
}

///|
fn svd_test_factor_ok(c : SvdComponent, m : Int32, n : Int32) -> Unit raise {
  assert_eq(c.u.length(), m)
  assert_eq(c.v.length(), n)
  assert_true(c.sigma >= 0.0)
  assert_true(!c.sigma.is_nan() && !c.sigma.is_inf())
  for x in c.u {
    assert_true(!x.is_nan() && !x.is_inf())
  }
  for x in c.v {
    assert_true(!x.is_nan() && !x.is_inf())
  }
  if c.sigma == 0.0 {
    for x in c.u {
      assert_eq(x, 0.0)
    }
    for x in c.v {
      assert_eq(x, 0.0)
    }
  } else {
    let un = svd_test_double(svd_vector_norm(c.u))
    let vn = svd_test_double(svd_vector_norm(c.v))
    assert_true((un - 1.0).abs() < 1.0e-8)
    assert_true((vn - 1.0).abs() < 1.0e-8)
  }
}

///|
fn svd_test_reconstruct(
  m : Int32,
  n : Int32,
  comps : Array[SvdComponent],
) -> Matrix {
  let out = zeros_buf(m, n)
  for r in 0..<comps.length() {
    let c = comps[r]
    for i in 0..<m {
      for j in 0..<n {
        out.data[i * n + j] = out.data[i * n + j] + c.sigma * c.u[i] * c.v[j]
      }
    }
  }
  out
}

///|
fn svd_test_relerr(a : Matrix, b : Matrix) -> Double {
  let m = a.rows
  let n = a.cols
  let mut s = 0.0
  for i in 0..<m {
    for j in 0..<n {
      let d = a.data[i * n + j] - b.data[i * n + j]
      s = s + d * d
    }
  }
  let den = svd_test_double(frobenius_norm(a))
  if den == 0.0 {
    if s == 0.0 {
      0.0
    } else {
      1.0e300
    }
  } else {
    s.sqrt() / den
  }
}

///|
fn svd_test_dot(a : Array[Double], b : Array[Double]) -> Double {
  let mut s = 0.0
  for i in 0..<a.length() {
    s = s + a[i] * b[i]
  }
  s
}

///|
test "wb svd: rectangular diagonal matrix has known singular values" {
  // diag(3, 2) embedded in 3x2: singular values 3 and 2.
  let a = svd_test_mat(3, 2, [3.0, 0.0, 0.0, 2.0, 0.0, 0.0])
  let comps = svd_test_components(svd_top_k(a, 2))
  assert_eq(comps.length(), 2)
  svd_test_factor_ok(comps[0], 3, 2)
  svd_test_factor_ok(comps[1], 3, 2)
  assert_true((comps[0].sigma - 3.0).abs() < 1.0e-7)
  assert_true((comps[1].sigma - 2.0).abs() < 1.0e-7)
  assert_true(comps[0].sigma >= comps[1].sigma)
  let rec = svd_test_reconstruct(3, 2, comps)
  assert_true(svd_test_relerr(a, rec) < 1.0e-8)
  // R v = sigma u for the dominant triplet.
  for i in 0..<3 {
    let mut rv = 0.0
    for j in 0..<2 {
      rv = rv + a.data[i * 2 + j] * comps[0].v[j]
    }
    assert_true((rv - comps[0].sigma * comps[0].u[i]).abs() < 1.0e-8)
  }
}

///|
test "wb svd: rank-1 matrix is recovered exactly" {
  let u0 = [1.0, 2.0, 3.0]
  let v0 = [2.0, 1.0]
  let a = svd_test_mat(3, 2, [
    u0[0] * v0[0],
    u0[0] * v0[1],
    u0[1] * v0[0],
    u0[1] * v0[1],
    u0[2] * v0[0],
    u0[2] * v0[1],
  ])
  let comps = svd_test_components(svd_top_k(a, 1))
  svd_test_factor_ok(comps[0], 3, 2)
  let expected = (14.0 * 5.0).sqrt()
  assert_true((comps[0].sigma - expected).abs() < 1.0e-6)
  let u_un = 14.0.sqrt()
  let v_un = 5.0.sqrt()
  let mut u_dot = 0.0
  let mut v_dot = 0.0
  for i in 0..<3 {
    u_dot = u_dot + comps[0].u[i] * (u0[i] / u_un)
  }
  for j in 0..<2 {
    v_dot = v_dot + comps[0].v[j] * (v0[j] / v_un)
  }
  // Sign is arbitrary; the absolute normalized inner product must be 1.
  assert_true((u_dot.abs() - 1.0).abs() < 1.0e-6)
  assert_true((v_dot.abs() - 1.0).abs() < 1.0e-6)
  let rec = svd_test_reconstruct(3, 2, comps)
  assert_true(svd_test_relerr(a, rec) < 1.0e-8)
}

///|
test "wb svd: known rank-2 product is reconstructed by two components" {
  let a_true = svd_test_mat(5, 2, [
    1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 0.0, 0.0, 2.0,
  ])
  let b_true = svd_test_mat(2, 4, [1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 3.0, 2.0])
  let a = match matmul(a_true, b_true) {
    Ok(m) => m
    Err(e) => abort("unexpected matmul error: \{e}")
  }
  let comps = svd_test_components(svd_top_k(a, 2))
  svd_test_factor_ok(comps[0], 5, 4)
  svd_test_factor_ok(comps[1], 5, 4)
  assert_true(comps[0].sigma >= comps[1].sigma)
  let rec = svd_test_reconstruct(5, 4, comps)
  assert_true(svd_test_relerr(a, rec) < 1.0e-6)
}

///|
test "wb svd: rank deficiency yields zero triplets without division" {
  // Rank-1 data, but request three components.
  let a = svd_test_mat(3, 3, [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0])
  let comps = svd_test_components(svd_top_k(a, 3))
  svd_test_factor_ok(comps[0], 3, 3)
  svd_test_factor_ok(comps[1], 3, 3)
  svd_test_factor_ok(comps[2], 3, 3)
  assert_true(comps[0].sigma > 0.0)
  assert_eq(comps[1].sigma, 0.0)
  assert_eq(comps[2].sigma, 0.0)
  let rec = svd_test_reconstruct(3, 3, comps)
  assert_true(svd_test_relerr(a, rec) < 1.0e-8)
  // All-zero input: every component is zero and finite.
  let z = zeros_buf(3, 4)
  let zcomps = svd_test_components(svd_top_k(z, 2))
  svd_test_factor_ok(zcomps[0], 3, 4)
  svd_test_factor_ok(zcomps[1], 3, 4)
  assert_eq(zcomps[0].sigma, 0.0)
  assert_eq(zcomps[1].sigma, 0.0)
}

///|
test "wb svd: duplicate singular values stay finite and orthogonal" {
  let a = svd_test_mat(3, 3, [2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0])
  let comps = svd_test_components(svd_top_k(a, 3))
  svd_test_factor_ok(comps[0], 3, 3)
  svd_test_factor_ok(comps[1], 3, 3)
  svd_test_factor_ok(comps[2], 3, 3)
  assert_true((comps[0].sigma - 2.0).abs() < 1.0e-7)
  assert_true((comps[1].sigma - 2.0).abs() < 1.0e-7)
  assert_true((comps[2].sigma - 1.0).abs() < 1.0e-7)
  assert_true(comps[0].sigma >= comps[1].sigma)
  assert_true(comps[1].sigma >= comps[2].sigma)
  assert_true(svd_test_dot(comps[0].u, comps[1].u).abs() < 1.0e-6)
  assert_true(svd_test_dot(comps[0].v, comps[1].v).abs() < 1.0e-6)
  let rec = svd_test_reconstruct(3, 3, comps)
  assert_true(svd_test_relerr(a, rec) < 1.0e-6)
}

///|
test "wb svd: tiny positive matrix stays finite" {
  let a = svd_test_mat(2, 2, [1.0e-100, 2.0e-100, 3.0e-100, 4.0e-100])
  let comps = svd_test_components(svd_top_k(a, 2))
  svd_test_factor_ok(comps[0], 2, 2)
  svd_test_factor_ok(comps[1], 2, 2)
  assert_true(comps[0].sigma > 0.0)
  assert_true(!comps[0].sigma.is_nan() && !comps[0].sigma.is_inf())
  let rec = svd_test_reconstruct(2, 2, comps)
  assert_true(svd_test_relerr(a, rec) < 1.0e-6)
}

///|
test "wb svd: fixed seeds make repeated extraction bit-identical" {
  let a = svd_test_mat(4, 3, [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 2.0, 9.0, 1.0,
  ])
  let c1 = svd_test_components(svd_top_k(a, 2))
  let c2 = svd_test_components(svd_top_k(a, 2))
  for r in 0..<2 {
    assert_eq(c1[r].sigma, c2[r].sigma)
    for i in 0..<4 {
      assert_eq(c1[r].u[i], c2[r].u[i])
    }
    for j in 0..<3 {
      assert_eq(c1[r].v[j], c2[r].v[j])
    }
  }
}
