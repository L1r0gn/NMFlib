// Runnable quickstart: `moon run examples/quickstart` from the repo root.
// Mirrors the snippet in README.md.

///|
fn demo() -> Result[Unit, @nmf.NMFError] {
  // 1. Build a non-negative matrix V (3x3, row-major).
  let v = match
    @nmf.Matrix::new(3, 3, [1.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 6.0]) {
    Ok(m) => m
    Err(e) => return Err(e)
  }
  // 2. Configure and fit (MU + Frobenius + Random, fixed seed).
  let config = @nmf.NMFConfig::new(n_components=2, seed=42UL)
  let model = @nmf.NMF::new(config)
  match model.fit(v) {
    Ok(_) => ()
    Err(e) => return Err(e)
  }
  // 3. Read factors and the reconstruction error ||V - WH||_F.
  let w = match model.w() {
    Ok(m) => m
    Err(e) => return Err(e)
  }
  let h = match model.h() {
    Ok(m) => m
    Err(e) => return Err(e)
  }
  let err = match model.reconstruction_error(v) {
    Ok(x) => x
    Err(e) => return Err(e)
  }
  let n_iter = match model.n_iter() {
    Ok(n) => n
    Err(e) => return Err(e)
  }
  let converged = match model.converged() {
    Ok(c) => c
    Err(e) => return Err(e)
  }
  println("W: \{w.rows()}x\{w.cols()}, H: \{h.rows()}x\{h.cols()}")
  println("iterations=\{n_iter}, converged=\{converged}")
  println("reconstruction error = \{err}")
  Ok(())
}

///|
fn main {
  match demo() {
    Ok(_) => ()
    Err(e) => println("NMF failed: \{e}")
  }
}
