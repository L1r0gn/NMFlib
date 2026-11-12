// Whitebox tests: SplitMix64 determinism, range, and random initialization.

///|
test "wb random: SplitMix64 known-answer vectors (seed 0)" {
  let rng = SplitMix64::new(0UL)
  assert_eq(rng.next_u64(), 0xE220A8397B1DCDAFUL)
  assert_eq(rng.next_u64(), 0x6E789E6AA1B965F4UL)
  assert_eq(rng.next_u64(), 0x006C45D188009454FUL)
}

///|
test "wb random: same seed reproduces the sequence exactly" {
  let a = SplitMix64::new(42UL)
  let b = SplitMix64::new(42UL)
  for _ in 0..<100 {
    assert_eq(a.next_u64(), b.next_u64())
  }
}

///|
test "wb random: different seeds diverge" {
  let a = SplitMix64::new(1UL)
  let b = SplitMix64::new(2UL)
  let mut differs = false
  for _ in 0..<10 {
    if a.next_u64() != b.next_u64() {
      differs = true
    }
  }
  assert_true(differs)
}

///|
test "wb random: next_double stays in (0, 1]" {
  let rng = SplitMix64::new(7UL)
  for _ in 0..<1000 {
    let x = rng.next_double()
    assert_true(x > 0.0)
    assert_true(x <= 1.0)
  }
}

///|
fn rng_v() -> Matrix {
  match Matrix::new(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) {
    Ok(m) => m
    Err(e) => abort("unexpected: \{e}")
  }
}

///|
test "wb random: init_scale follows sqrt(mean/k), zero V falls back to 1" {
  let v = rng_v()
  // mean([1..6]) = 3.5, k = 2 -> sqrt(1.75)
  match init_scale(v, 2) {
    Ok(s) => assert_true((s - 1.75.sqrt()).abs() < 1.0e-12)
    Err(e) => abort("unexpected: \{e}")
  }
  let z = zeros_buf(2, 3)
  assert_eq(init_scale(z, 2), Ok(1.0))
}

///|
test "wb random: init_scale overflow is an error" {
  let big = match Matrix::new(1, 2, [1.0e308, 1.0e308]) {
    Ok(m) => m
    Err(e) => abort("unexpected: \{e}")
  }
  assert_true(init_scale(big, 1) is Err(NumericalError(_)))
}

///|
test "wb random: random_init dims, strict positivity, reproducibility" {
  let v = rng_v()
  let (w1, h1) = match random_init(v, 2, 42UL) {
    Ok(pair) => pair
    Err(e) => abort("unexpected: \{e}")
  }
  assert_eq(w1.rows, 2)
  assert_eq(w1.cols, 2)
  assert_eq(h1.rows, 2)
  assert_eq(h1.cols, 3)
  for x in w1.data {
    assert_true(x > 0.0)
    assert_true(x.is_nan() == false)
  }
  for x in h1.data {
    assert_true(x > 0.0)
  }
  let (w2, h2) = match random_init(v, 2, 42UL) {
    Ok(pair) => pair
    Err(e) => abort("unexpected: \{e}")
  }
  for idx in 0..<w1.data.length() {
    assert_eq(w1.data[idx], w2.data[idx])
  }
  for idx in 0..<h1.data.length() {
    assert_eq(h1.data[idx], h2.data[idx])
  }
  let (w3, _) = match random_init(v, 2, 43UL) {
    Ok(pair) => pair
    Err(e) => abort("unexpected: \{e}")
  }
  let mut differs = false
  for idx in 0..<w1.data.length() {
    if w1.data[idx] != w3.data[idx] {
      differs = true
    }
  }
  assert_true(differs)
}

///|
test "wb random: random_init on all-zero V stays strictly positive" {
  let z = zeros_buf(3, 3)
  let (w, h) = match random_init(z, 2, 9UL) {
    Ok(pair) => pair
    Err(e) => abort("unexpected: \{e}")
  }
  for x in w.data {
    assert_true(x > 0.0)
    assert_true(x <= 1.0)
  }
  for x in h.data {
    assert_true(x > 0.0)
  }
}

///|
test "wb random: random_init_w dims and reproducibility" {
  let x = rng_v()
  let w1 = match random_init_w(x, 2, 5UL) {
    Ok(w) => w
    Err(e) => abort("unexpected: \{e}")
  }
  assert_eq(w1.rows, 2)
  assert_eq(w1.cols, 2)
  let w2 = match random_init_w(x, 2, 5UL) {
    Ok(w) => w
    Err(e) => abort("unexpected: \{e}")
  }
  for idx in 0..<w1.data.length() {
    assert_eq(w1.data[idx], w2.data[idx])
    assert_true(w1.data[idx] > 0.0)
  }
}
