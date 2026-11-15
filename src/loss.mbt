///|
/// Unchecked `Σ (V − WH)²` from a precomputed product `WH`.
/// Caller guarantees `wh` has the same shape as `v`.
fn sq_diff_from_wh(v : Matrix, wh : Matrix) -> Double {
  let mut s = 0.0
  for idx in 0..<v.data.length() {
    let diff = v.data[idx] - wh.data[idx]
    s = s + diff * diff
  }
  s
}

///|
/// Frobenius objective `F = ½‖V − WH‖²_F`.
///
/// The ½ coefficient matches the Lee–Seung MU derivation: the multiplicative
/// updates in `solver_mu.mbt` minimize exactly this objective, and the
/// convergence test compares successive values of it. The reported
/// `reconstruction_error` is the norm `‖V − WH‖_F = sqrt(2F)` (sklearn
/// semantics), kept distinct from this objective.
///
/// Dimensions must satisfy `V: n×d`, `W: n×k`, `H: k×d`. Overflow of the
/// accumulated sum of squares returns `NumericalError` (never a silent
/// Infinity).
pub fn frobenius_objective(
  v : Matrix,
  w : Matrix,
  h : Matrix,
) -> Result[Double, NMFError] {
  if w.rows != v.rows || h.cols != v.cols || w.cols != h.rows {
    return Err(
      DimensionMismatch(
        "frobenius_objective shapes: V \{v.rows}x\{v.cols}, W \{w.rows}x\{w.cols}, H \{h.rows}x\{h.cols}",
      ),
    )
  }
  let wh = zeros_buf(v.rows, v.cols)
  matmul_into(w, h, wh)
  let s = sq_diff_from_wh(v, wh)
  if s.is_nan() || s.is_inf() {
    return Err(
      NumericalError("Frobenius objective overflowed to a non-finite value"),
    )
  }
  Ok(0.5 * s)
}

///|
/// Generalized KL divergence
/// `D(V‖WH) = Σ [ V log(V/WH) − V + WH ]`.
///
/// Uses the log-difference form `V(log V − log WH) − V + WH` rather than
/// `V/WH` to avoid ratio overflow/underflow. Zero conventions:
///
/// - `V=0, WH=0`: term 0 (the `0·log 0` limit, no division);
/// - `V=0, WH>0`: term `WH` (the log term is 0 by limit);
/// - `V>0, WH=0`: the divergence is `+∞`; returns `NumericalError` because
///   the public Result type cannot represent an infinite objective.
///
/// Negative entries return `NegativeEntry`; non-finite terms or accumulated
/// sums return `NumericalError`. A tiny negative residue caused by rounding
/// in the theoretically non-negative sum is clamped to zero only within
/// `1e-12·(1 + Σ|term|)`; a clearly negative value is reported as an error.
pub fn kl_divergence(v : Matrix, wh : Matrix) -> Result[Double, NMFError] {
  if v.rows != wh.rows || v.cols != wh.cols {
    return Err(
      DimensionMismatch(
        "kl_divergence shapes: V \{v.rows}x\{v.cols}, WH \{wh.rows}x\{wh.cols}",
      ),
    )
  }
  let mut sum = 0.0
  let mut compensation = 0.0
  let mut magnitude = 0.0
  for idx in 0..<v.data.length() {
    let x = v.data[idx]
    let y = wh.data[idx]
    let i = idx / v.cols
    let j = idx % v.cols
    if x.is_nan() || x.is_inf() || y.is_nan() || y.is_inf() {
      return Err(
        NumericalError("kl_divergence: non-finite value at (\{i},\{j})"),
      )
    }
    if x < 0.0 || y < 0.0 {
      return Err(NegativeEntry(i, j))
    }
    let term = if x == 0.0 {
      // V=0: 0·log(0/WH) is 0; when WH=0 the whole term is 0, not 0/0.
      y
    } else if y == 0.0 {
      return Err(
        NumericalError("kl_divergence is infinite: V>0 and WH=0 at (\{i},\{j})"),
      )
    } else {
      let value = x * (@math.ln(x) - @math.ln(y)) - x + y
      if value.is_nan() || value.is_inf() {
        return Err(
          NumericalError("kl_divergence: non-finite term at (\{i},\{j})"),
        )
      }
      value
    }
    if term.is_nan() || term.is_inf() {
      return Err(
        NumericalError("kl_divergence: non-finite term at (\{i},\{j})"),
      )
    }
    // Kahan summation keeps cancellation from corrupting the non-negative sum.
    let adjusted = term - compensation
    let next = sum + adjusted
    compensation = next - sum - adjusted
    sum = next
    if sum.is_nan() || sum.is_inf() {
      return Err(NumericalError("kl_divergence sum overflowed"))
    }
    magnitude = magnitude + term.abs()
    if magnitude.is_nan() || magnitude.is_inf() {
      return Err(NumericalError("kl_divergence magnitude overflowed"))
    }
  }
  if sum < 0.0 {
    let tolerance = 1.0e-12 * (1.0 + magnitude)
    if -sum <= tolerance {
      return Ok(0.0)
    }
    return Err(
      NumericalError("kl_divergence became negative beyond rounding tolerance"),
    )
  }
  if sum == 0.0 {
    return Ok(0.0)
  }
  Ok(sum)
}
