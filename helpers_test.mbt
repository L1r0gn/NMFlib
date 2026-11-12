// Shared helpers for blackbox tests.
// IEEE special values are constructed by arithmetic: no core API needed.

///|
fn nan() -> Double {
  0.0 / 0.0
}

///|
fn inf() -> Double {
  1.0 / 0.0
}

///|
fn unwrap_matrix(m : Result[@nmf.Matrix, @nmf.NMFError]) -> @nmf.Matrix {
  match m {
    Ok(m) => m
    Err(e) => abort("unexpected matrix error: \{e}")
  }
}

///|
fn unwrap_double(x : Result[Double, @nmf.NMFError]) -> Double {
  match x {
    Ok(x) => x
    Err(e) => abort("unexpected error: \{e}")
  }
}
