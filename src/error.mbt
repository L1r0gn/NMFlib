///|
/// Unified error type for all public NMFlib APIs.
///
/// Every fallible entry point returns `Result[T, NMFError]`. Illegal input is
/// never silently clamped, reshaped, or repaired.
pub(all) enum NMFError {
  /// Matrix dimensions are incompatible (data length, matmul inner dims,
  /// or API shape constraints). The message describes the mismatch.
  DimensionMismatch(String)
  /// A matrix entry that must be non-negative is negative, at (row, col).
  NegativeEntry(Int32, Int32)
  /// The requested rank is outside `1 <= k <= min(rows, cols)`.
  InvalidRank(Int32)
  /// A configuration value is illegal or requests an unsupported combination.
  InvalidConfig(String)
  /// The matrix shape itself is invalid (negative dimensions, overflow,
  /// or an empty matrix where a non-empty one is required).
  InvalidMatrixShape(String)
  /// A non-finite value (NaN/Infinity) was supplied or produced.
  NumericalError(String)
  /// A model-dependent API was called before a successful `fit`.
  NotFitted
}

///|
pub impl Debug for NMFError with fn to_repr(self) {
  match self {
    DimensionMismatch(msg) =>
      @debug.Repr::ctor("DimensionMismatch", [(None, @debug.Repr::string(msg))])
    NegativeEntry(i, j) =>
      @debug.Repr::ctor("NegativeEntry", [
        (None, @debug.Repr::integer(i.to_string())),
        (None, @debug.Repr::integer(j.to_string())),
      ])
    InvalidRank(k) =>
      @debug.Repr::ctor("InvalidRank", [
        (None, @debug.Repr::integer(k.to_string())),
      ])
    InvalidConfig(msg) =>
      @debug.Repr::ctor("InvalidConfig", [(None, @debug.Repr::string(msg))])
    InvalidMatrixShape(msg) =>
      @debug.Repr::ctor("InvalidMatrixShape", [(None, @debug.Repr::string(msg))])
    NumericalError(msg) =>
      @debug.Repr::ctor("NumericalError", [(None, @debug.Repr::string(msg))])
    NotFitted => @debug.Repr::ctor("NotFitted", [])
  }
}

///|
pub extend NMFError with Debug::{to_repr}

///|
pub impl Eq for NMFError with fn equal(self, other) {
  match (self, other) {
    (DimensionMismatch(a), DimensionMismatch(b)) => a == b
    (NegativeEntry(i1, j1), NegativeEntry(i2, j2)) => i1 == i2 && j1 == j2
    (InvalidRank(a), InvalidRank(b)) => a == b
    (InvalidConfig(a), InvalidConfig(b)) => a == b
    (InvalidMatrixShape(a), InvalidMatrixShape(b)) => a == b
    (NumericalError(a), NumericalError(b)) => a == b
    (NotFitted, NotFitted) => true
    _ => false
  }
}

///|
pub extend NMFError with Eq::{equal, not_equal}

///|
pub impl Show for NMFError with fn to_string(self) {
  match self {
    DimensionMismatch(msg) => "DimensionMismatch: \{msg}"
    NegativeEntry(i, j) => "NegativeEntry at (\{i}, \{j})"
    InvalidRank(k) => "InvalidRank: \{k}"
    InvalidConfig(msg) => "InvalidConfig: \{msg}"
    InvalidMatrixShape(msg) => "InvalidMatrixShape: \{msg}"
    NumericalError(msg) => "NumericalError: \{msg}"
    NotFitted => "NotFitted"
  }
}

///|
pub extend NMFError with Show::{to_string, output}
