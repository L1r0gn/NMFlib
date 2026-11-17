///|
/// L1/L2 regularization strengths applied to **both** factors.
///
/// The penalty is
/// ```text
/// Ω(W,H) = l1·(ΣW + ΣH) + (l2/2)·(‖W‖²_F + ‖H‖²_F)
/// ```
/// Since W and H are non-negative, `ΣW = ‖W‖₁` and `ΣH = ‖H‖₁`, so the first
/// term is exactly the L1 penalty on both factors. Both strengths are
/// validated at the API boundary (`l1/l2 >= 0`, finite) and are never
/// repaired silently here.
priv struct Reg {
  l1 : Double
  l2 : Double
}

///|
/// Zero regularization. Every regularized expression in this file returns the
/// exact pre-T12 value when `reg == no_reg`: adding `0.0` and
/// `0.0 * non_negative` is exact in IEEE-754, so the original update path and
/// its bit-for-bit results are preserved without a separate branch.
#warnings("-unused_value")
let no_reg : Reg = { l1: 0.0, l2: 0.0, }

///|
/// Regularization penalty `Ω(W,H)` for non-negative factors.
///
/// Returns `Ok(0.0)` immediately for `no_reg`, so a zero-regularization run
/// performs no extra arithmetic and cannot fail on new error paths. Negative
/// or non-finite factor entries and non-finite intermediate sums are reported
/// as `NumericalError`; the penalty is never silently clamped.
fn reg_penalty(w : Matrix, h : Matrix, reg : Reg) -> Result[Double, NMFError] {
  if reg.l1 == 0.0 && reg.l2 == 0.0 {
    return Ok(0.0)
  }
  let mut l1_sum = 0.0
  let mut l2_sum = 0.0
  for idx in 0..<w.data.length() {
    let x = w.data[idx]
    if x.is_nan() || x.is_inf() || x < 0.0 {
      return Err(
        NumericalError("regularization penalty: invalid W entry at \{idx}"),
      )
    }
    l1_sum = l1_sum + x
    l2_sum = l2_sum + x * x
    if l1_sum.is_nan() || l1_sum.is_inf() || l2_sum.is_nan() || l2_sum.is_inf() {
      return Err(NumericalError("regularization penalty: W sum overflowed"))
    }
  }
  for idx in 0..<h.data.length() {
    let x = h.data[idx]
    if x.is_nan() || x.is_inf() || x < 0.0 {
      return Err(
        NumericalError("regularization penalty: invalid H entry at \{idx}"),
      )
    }
    l1_sum = l1_sum + x
    l2_sum = l2_sum + x * x
    if l1_sum.is_nan() || l1_sum.is_inf() || l2_sum.is_nan() || l2_sum.is_inf() {
      return Err(NumericalError("regularization penalty: H sum overflowed"))
    }
  }
  let penalty = reg.l1 * l1_sum + 0.5 * reg.l2 * l2_sum
  if penalty.is_nan() || penalty.is_inf() {
    return Err(
      NumericalError("regularization penalty overflowed to a non-finite value"),
    )
  }
  Ok(penalty)
}

///|
/// Regularized Frobenius objective from a precomputed product `WH`:
/// `½‖V − WH‖²_F + Ω(W,H)`.
///
/// This is the quantity that `loss_value` reports and that the convergence
/// test compares for Frobenius training; it is **not** the reconstruction
/// norm returned by `NMF::reconstruction_error`.
fn frobenius_reg_from_wh(
  v : Matrix,
  wh : Matrix,
  w : Matrix,
  h : Matrix,
  reg : Reg,
) -> Result[Double, NMFError] {
  let mut f = 0.5 * sq_diff_from_wh(v, wh)
  match reg_penalty(w, h, reg) {
    Ok(p) => f = f + p
    Err(e) => return Err(e)
  }
  if f.is_nan() || f.is_inf() {
    return Err(
      NumericalError("regularized Frobenius objective became non-finite"),
    )
  }
  Ok(f)
}

///|
/// Regularized generalized-KL objective from a precomputed product `WH`:
/// `D_KL(V‖WH) + Ω(W,H)`.
///
/// The KL term itself is unchanged from T11 (including its zero conventions);
/// the regularization penalty is added on top and never merged into the
/// divergence.
fn kl_reg_from_wh(
  v : Matrix,
  wh : Matrix,
  w : Matrix,
  h : Matrix,
  reg : Reg,
) -> Result[Double, NMFError] {
  let base = match kl_divergence(v, wh) {
    Ok(x) => x
    Err(e) => return Err(e)
  }
  let penalty = match reg_penalty(w, h, reg) {
    Ok(p) => p
    Err(e) => return Err(e)
  }
  let f = base + penalty
  if f.is_nan() || f.is_inf() {
    return Err(NumericalError("regularized KL objective became non-finite"))
  }
  Ok(f)
}

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
