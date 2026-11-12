///|
/// Initialization scale `sqrt(mean(V) / k)` (sklearn-style).
/// For an all-zero input the mean is 0 and the scale falls back to 1 so the
/// initial factors stay strictly positive.
fn init_scale(v : Matrix, k : Int32) -> Result[Double, NMFError] {
  let mut sum = 0.0
  for x in v.data {
    sum = sum + x
  }
  if sum.is_nan() || sum.is_inf() {
    return Err(NumericalError("sum of V overflowed while computing init scale"))
  }
  let mean = sum / (v.rows * v.cols).to_double()
  let scale = if mean > 0.0 { (mean / k.to_double()).sqrt() } else { 1.0 }
  Ok(scale)
}

///|
/// Fills a buffer with strictly positive uniform entries times `scale`.
fn random_fill(buf : Matrix, scale : Double, rng : SplitMix64) -> Unit {
  for idx in 0..<buf.data.length() {
    buf.data[idx] = rng.next_double() * scale
  }
}

///|
/// Random initialization of both factors: W (n×k), H (k×d), strictly
/// positive, reproducible for a fixed seed.
fn random_init(
  v : Matrix,
  k : Int32,
  seed : UInt64,
) -> Result[(Matrix, Matrix), NMFError] {
  let scale = match init_scale(v, k) {
    Ok(s) => s
    Err(e) => return Err(e)
  }
  let rng = SplitMix64::new(seed)
  let w = zeros_buf(v.rows, k)
  let h = zeros_buf(k, v.cols)
  random_fill(w, scale, rng)
  random_fill(h, scale, rng)
  Ok((w, h))
}

///|
/// Random initialization of W only (m×k), used by `transform` where H is
/// fixed from training.
fn random_init_w(
  x : Matrix,
  k : Int32,
  seed : UInt64,
) -> Result[Matrix, NMFError] {
  let scale = match init_scale(x, k) {
    Ok(s) => s
    Err(e) => return Err(e)
  }
  let rng = SplitMix64::new(seed)
  let w = zeros_buf(x.rows, k)
  random_fill(w, scale, rng)
  Ok(w)
}
