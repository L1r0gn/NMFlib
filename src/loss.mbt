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
