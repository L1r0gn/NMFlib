// Blackbox tests for generalized KL divergence D(V || WH).

///|
fn kl_value(v : @nmf.Matrix, wh : @nmf.Matrix) -> Double {
  match @nmf.kl_divergence(v, wh) {
    Ok(x) => x
    Err(e) => abort("unexpected KL error: \{e}")
  }
}

///|
priv struct KlRng {
  mut state : UInt64
}

///|
fn KlRng::new(seed : UInt64) -> KlRng {
  { state: seed, }
}

///|
fn KlRng::next_double(self : KlRng) -> Double {
  self.state = (self.state * 1664525UL + 1013904223UL) & 0xFFFFFFFFUL
  (self.state >> 8).to_double() / 16777216.0
}

///|
test "kl: equal matrices have zero divergence" {
  let v = mat(2, 2, [1.0, 2.0, 3.0, 4.0])
  let wh = mat(2, 2, [1.0, 2.0, 3.0, 4.0])
  assert_eq(@nmf.kl_divergence(v, wh), Ok(0.0))
  let z = mat(2, 2, [0.0, 0.0, 0.0, 0.0])
  assert_eq(@nmf.kl_divergence(z, z), Ok(0.0))
}

///|
test "kl: V=0 and WH>0 equals sum(WH)" {
  let z = mat(2, 2, [0.0, 0.0, 0.0, 0.0])
  let wh = mat(2, 2, [1.0, 2.0, 3.0, 4.0])
  assert_eq(@nmf.kl_divergence(z, wh), Ok(10.0))
  // Mixed zeros: V=0 contributes WH on the two V=0 entries.
  let v = mat(1, 2, [0.0, 0.0])
  let wh2 = mat(1, 2, [2.0, 3.0])
  assert_eq(@nmf.kl_divergence(v, wh2), Ok(5.0))
}

///|
test "kl: hand-computed mixed zero terms" {
  // (0,0) 1log1-1+1=0; (0,1) V=0,WH=1 -> 1;
  // (1,0) V=0,WH=1 -> 1; (1,1) 2log1-2+2=0. Total 2.
  let v = mat(2, 2, [1.0, 0.0, 0.0, 2.0])
  let wh = mat(2, 2, [1.0, 1.0, 1.0, 2.0])
  assert_eq(@nmf.kl_divergence(v, wh), Ok(2.0))
}

///|
test "kl: single positive term follows the formula" {
  let v = mat(1, 1, [2.0])
  let wh = mat(1, 1, [1.0])
  let val = kl_value(v, wh)
  assert_true((val - 0.3862943611198906).abs() < 1.0e-12)
  let v2 = mat(1, 1, [1.0])
  let wh2 = mat(1, 1, [2.0])
  let val2 = kl_value(v2, wh2)
  assert_true((val2 - 0.3068528194400547).abs() < 1.0e-12)
}

///|
test "kl: V>0 and WH=0 is an explicit NumericalError" {
  let v = mat(1, 1, [1.0])
  let wh = mat(1, 1, [0.0])
  assert_true(@nmf.kl_divergence(v, wh) is Err(@nmf.NumericalError(_)))
  let v2 = mat(2, 2, [0.0, 1.0, 0.0, 0.0])
  let wh2 = mat(2, 2, [0.0, 0.0, 0.0, 0.0])
  assert_true(@nmf.kl_divergence(v2, wh2) is Err(@nmf.NumericalError(_)))
  // V=0, WH=0 remains a legal zero term.
  let z = mat(1, 1, [0.0])
  assert_eq(@nmf.kl_divergence(z, z), Ok(0.0))
}

///|
test "kl: dimensions and negative entries are rejected" {
  let v = mat(2, 2, [1.0, 2.0, 3.0, 4.0])
  let wrong = mat(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  assert_true(@nmf.kl_divergence(v, wrong) is Err(@nmf.DimensionMismatch(_)))
  let neg_v = mat(1, 1, [-1.0])
  let one = mat(1, 1, [1.0])
  assert_true(@nmf.kl_divergence(neg_v, one) is Err(@nmf.NegativeEntry(0, 0)))
  let neg_wh = mat(1, 1, [-2.0])
  assert_true(@nmf.kl_divergence(one, neg_wh) is Err(@nmf.NegativeEntry(0, 0)))
}

///|
test "kl: tiny and large finite values remain finite" {
  let tiny = mat(1, 1, [1.0e-100])
  assert_eq(@nmf.kl_divergence(tiny, tiny), Ok(0.0))
  let tiny2 = mat(1, 1, [1.0e-200])
  let val = kl_value(tiny, tiny2)
  assert_true(val > 0.0 && !val.is_nan() && !val.is_inf())
  let big = mat(1, 1, [1.0e200])
  let big2 = mat(1, 1, [1.0e100])
  let val2 = kl_value(big, big2)
  assert_true(val2 > 0.0 && !val2.is_nan() && !val2.is_inf())
  // V*log(V/WH) overflows for this extreme pair; the public API reports it.
  let huge = mat(1, 1, [1.0e308])
  let tiny_wh = mat(1, 1, [1.0e-308])
  assert_true(@nmf.kl_divergence(huge, tiny_wh) is Err(@nmf.NumericalError(_)))
}

///|
test "kl: NaN and Infinity cannot enter a Matrix" {
  assert_true(@nmf.Matrix::new(1, 1, [nan()]) is Err(@nmf.NumericalError(_)))
  assert_true(@nmf.Matrix::new(1, 1, [inf()]) is Err(@nmf.NumericalError(_)))
}

///|
test "kl: divergence is non-negative on random non-negative inputs" {
  let rng = KlRng::new(20261116UL)
  for _ in 0..<50 {
    let rows = 1 + (rng.next_double() * 4.0).to_int()
    let cols = 1 + (rng.next_double() * 4.0).to_int()
    let v_data = Array::make(rows * cols, 0.0)
    let wh_data = Array::make(rows * cols, 0.0)
    for idx in 0..<v_data.length() {
      v_data[idx] = rng.next_double() * 5.0
      // Strictly positive reconstruction avoids the infinite V>0,WH=0 case.
      wh_data[idx] = 0.01 + rng.next_double() * 5.0
    }
    let v = match @nmf.Matrix::new(rows, cols, v_data) {
      Ok(m) => m
      Err(e) => abort("unexpected matrix error: \{e}")
    }
    let wh = match @nmf.Matrix::new(rows, cols, wh_data) {
      Ok(m) => m
      Err(e) => abort("unexpected matrix error: \{e}")
    }
    let val = kl_value(v, wh)
    assert_true(val >= 0.0)
    assert_true(!val.is_nan() && !val.is_inf())
  }
}

///|
test "kl: zero rows and columns are finite" {
  let v = mat(3, 3, [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
  let wh = mat(3, 3, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
  let val = kl_value(v, wh)
  // Zero V entries contribute 1 each; the single 1 entry contributes 0.
  assert_true((val - 8.0).abs() < 1.0e-12)
  assert_true(!val.is_nan() && !val.is_inf())
}
