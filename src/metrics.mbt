///|
/// Hoyer sparsity of a single non-negative vector (Hoyer, JMLR 2004).
///
/// ```text
/// S(x) = (√n − ‖x‖₁/‖x‖₂) / (√n − 1)
/// ```
///
/// For a non-negative, non-zero vector with `n >= 2` the value lies in
/// `[0, 1]`: `0` when every entry is equal (no sparsity), `1` when exactly one
/// entry is non-zero (maximal sparsity). The measure is invariant under
/// positive scaling (`S(cx) = S(x)`, `c > 0`) and under permutation of the
/// entries.
///
/// Boundary conventions (PLAN §8.7, applied in this order):
///
/// - **empty input** (`n = 0`) → `Err(InvalidConfig)`: the formula has no
///   meaning without entries;
/// - **single entry** (`n = 1`) → `0.0`: the denominator `√n − 1` is zero, so
///   the formula is undefined; a one-dimensional vector carries no sparsity
///   information. This rule takes precedence over the all-zero rule, hence
///   `sparsity([0.0]) == Ok(0.0)`;
/// - **all-zero vector** (`n >= 2`, every entry `0`) → `1.0`: the convention
///   that a vector with no non-zero component is maximally sparse (the formula
///   would be `0/0`);
/// - **negative entry** → `Err(NegativeEntry(0, i))`: the vector is treated as
///   a single row, so the first index is always `0` and the second is the
///   position of the offending entry;
/// - **NaN / ±Infinity** → `Err(NumericalError)`. This check runs before the
///   negativity check, so `-Infinity` is reported as a numerical error rather
///   than as a negative entry. Nothing is silently replaced by `0`.
///
/// Numerical policy: the ratio `‖x‖₁/‖x‖₂` is scale invariant, so the
/// computation first divides every entry by `a = max x_i`, giving `y_i ∈
/// [0, 1]`. Both sums are then bounded by `n` and cannot overflow (a naive
/// `Σ x_i²` would overflow for entries near `Double` max and underflow for
/// subnormals), and both are accumulated with Kahan compensation. The result
/// is checked against `[0, 1]`; only a rounding-scale residue (`|S| <= 1e-12`,
/// which is what equal entries and one-hot vectors can produce) is snapped to
/// `0`, and a value exceeding `1` by at most `1e-12` to `1`; a larger
/// excursion is reported as `NumericalError` instead of being hidden.
///
/// The value is a pure function of the vector: it never reads or mutates model
/// state. To measure an NMF factor, extract the direction that carries the
/// meaning — `W`'s columns are basis vectors (length `m`) and `H`'s rows are
/// component activations (length `n`) — with `Matrix::get`, and pass each
/// extracted array to this function. No matrix-wide average is provided here,
/// because an aggregate would silently choose a direction and a rule for
/// zero vectors.
pub fn sparsity(x : Array[Double]) -> Result[Double, NMFError] {
  let n = x.length()
  if n == 0 {
    return Err(InvalidConfig("sparsity: empty vector has no defined value"))
  }
  for i in 0..<n {
    let v = x[i]
    if v.is_nan() || v.is_inf() {
      return Err(NumericalError("sparsity: non-finite entry at index \{i}"))
    }
    if v < 0.0 {
      return Err(NegativeEntry(0, i))
    }
  }
  // n = 1 is checked before the all-zero case: the formula's denominator is
  // zero there, and PLAN §8.7 fixes the value at 0.0.
  if n == 1 {
    return Ok(0.0)
  }
  let mut max_entry = 0.0
  for v in x {
    if v > max_entry {
      max_entry = v
    }
  }
  if max_entry == 0.0 {
    // Non-negative and finite, so a zero maximum means every entry is zero.
    return Ok(1.0)
  }
  // Scale by the maximum so that y_i ∈ [0, 1]; the ratio is unchanged and the
  // sums below are bounded by n. Kahan compensation keeps the two sums
  // accurate for long vectors.
  let mut sum_abs = 0.0
  let mut comp_abs = 0.0
  let mut sum_sq = 0.0
  let mut comp_sq = 0.0
  for v in x {
    let y = v / max_entry
    let t_abs = y - comp_abs
    let next_abs = sum_abs + t_abs
    comp_abs = next_abs - sum_abs - t_abs
    sum_abs = next_abs
    let square = y * y
    let t_sq = square - comp_sq
    let next_sq = sum_sq + t_sq
    comp_sq = next_sq - sum_sq - t_sq
    sum_sq = next_sq
  }
  // sum_sq >= 1 because the largest y is exactly 1.0, so this division cannot
  // divide by zero or underflow.
  let root_n = n.to_double().sqrt()
  let ratio = sum_abs / sum_sq.sqrt()
  let raw = (root_n - ratio) / (root_n - 1.0)
  // Rounding guard: for equal entries the exact value is 0 but the division
  // can land a few ulps on either side of it, and `ratio` can exceed `√n` by a
  // few ulps for a one-hot vector. Only such residues are snapped; anything
  // further outside [0, 1] is a defect and is reported below, not truncated.
  let clamped = if raw.abs() <= 1.0e-12 {
    0.0
  } else if raw > 1.0 && raw <= 1.0 + 1.0e-12 {
    1.0
  } else {
    raw
  }
  if clamped < 0.0 || clamped > 1.0 || clamped.is_nan() || clamped.is_inf() {
    return Err(
      NumericalError("sparsity: result fell outside [0, 1]: \{clamped}"),
    )
  }
  Ok(clamped)
}
