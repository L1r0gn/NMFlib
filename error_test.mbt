// Blackbox tests for NMFError equality and display behavior.

///|
test "error: Eq compares variant and payload" {
  assert_eq(@nmf.NMFError::NotFitted, @nmf.NMFError::NotFitted)
  assert_eq(@nmf.NMFError::InvalidRank(3), @nmf.NMFError::InvalidRank(3))
  assert_true(@nmf.NMFError::InvalidRank(3) != @nmf.NMFError::InvalidRank(4))
  assert_true(@nmf.NMFError::NotFitted != @nmf.NMFError::InvalidRank(0))
  assert_eq(
    @nmf.NMFError::NegativeEntry(1, 2),
    @nmf.NMFError::NegativeEntry(1, 2),
  )
  assert_true(
    @nmf.NMFError::NegativeEntry(1, 2) != @nmf.NMFError::NegativeEntry(2, 1),
  )
  assert_eq(
    @nmf.NMFError::DimensionMismatch("x"),
    @nmf.NMFError::DimensionMismatch("x"),
  )
  assert_true(
    @nmf.NMFError::DimensionMismatch("x") != @nmf.NMFError::NumericalError("x"),
  )
}

///|
test "error: Show messages locate the problem" {
  assert_eq(@nmf.NMFError::NotFitted.to_string(), "NotFitted")
  assert_eq(
    @nmf.NMFError::NegativeEntry(2, 5).to_string(),
    "NegativeEntry at (2, 5)",
  )
  assert_eq(@nmf.NMFError::InvalidRank(0).to_string(), "InvalidRank: 0")
  assert_true(
    @nmf.NMFError::DimensionMismatch("inner dims")
    .to_string()
    .contains("inner dims"),
  )
  assert_true(
    @nmf.NMFError::NumericalError("overflow").to_string().contains("overflow"),
  )
}
