// T16 validation runner. Reads case literals from cases_gen.mbt (generated
// by run_experiments.py), fits each case through the PUBLIC NMFlib API, and
// prints one CSV line per case:
//   id,ok,n_iter,converged,loss_value,reconstruction_error,
//   w_rows,w_cols,h_rows,h_cols,<W row-major>,<H row-major>
// or on failure:
//   id,err,<error message>
// No production code is modified or extended by this module.

///|
struct T16Case {
  id : String
  v : @nmf.Matrix
  k : Int32
  solver : @nmf.Solver
  loss : @nmf.Loss
  init : @nmf.Init
  max_iter : Int32
  tol : Double
  seed : UInt64
  l1 : Double
  l2 : Double
}

///|
fn mat(rows : Int32, cols : Int32, data : Array[Double]) -> @nmf.Matrix {
  match @nmf.Matrix::new(rows, cols, data) {
    Ok(m) => m
    Err(e) => {
      println("invalid literal matrix: \{e}")
      panic()
    }
  }
}

///|
fn[T] unwrap(r : Result[T, @nmf.NMFError]) -> T {
  match r {
    Ok(x) => x
    Err(e) => {
      println("unexpected error: \{e}")
      panic()
    }
  }
}

///|
fn matrix_csv(m : @nmf.Matrix) -> Array[String] {
  let out : Array[String] = []
  for i in 0..<m.rows() {
    for j in 0..<m.cols() {
      out.push(unwrap(m.get(i, j)).to_string())
    }
  }
  out
}

///|
fn main {
  for c in t16_cases {
    let config = @nmf.NMFConfig::new(
      n_components=c.k,
      solver=c.solver,
      loss=c.loss,
      init=c.init,
      max_iter=c.max_iter,
      tol=c.tol,
      seed=c.seed,
      l1_reg=c.l1,
      l2_reg=c.l2,
    )
    let model = @nmf.NMF::new(config)
    match model.fit(c.v) {
      Ok(_) => {
        let w = unwrap(model.w())
        let h = unwrap(model.h())
        let fields : Array[String] = [
          c.id,
          "ok",
          unwrap(model.n_iter()).to_string(),
          unwrap(model.converged()).to_string(),
          unwrap(model.loss_value()).to_string(),
          unwrap(model.reconstruction_error(c.v)).to_string(),
          w.rows().to_string(),
          w.cols().to_string(),
          h.rows().to_string(),
          h.cols().to_string(),
        ]
        for s in matrix_csv(w) {
          fields.push(s)
        }
        for s in matrix_csv(h) {
          fields.push(s)
        }
        println(fields.join(","))
      }
      Err(e) => println("\{c.id},err,\{e}")
    }
  }
}
