///|
/// NMF solver selection. Both solvers minimize the Frobenius objective;
/// `CD` currently supports only `Loss::Frobenius` (CD + KL is rejected at
/// validation time).
#warnings("-unused_constructor")
pub(all) enum Solver {
  MU
  CD
}

///|
/// Loss function selection. `KLDivergence` lands in T11 (MU only);
/// selecting it before then returns `InvalidConfig`.
#warnings("-unused_constructor")
pub(all) enum Loss {
  Frobenius
  KLDivergence
}

///|
/// Initialization selection. `NNDSVD` and `NNDSVDa` are deterministic
/// (seed-independent); `NNDSVDa` additionally replaces exact zeros in W/H
/// with the global mean of V.
#warnings("-unused_constructor")
pub(all) enum Init {
  Random
  NNDSVD
  NNDSVDa
}

///|
/// Training configuration. Construct with `NMFConfig::new`; validation
/// happens at `fit` time so illegal values surface as `Err`, never as
/// silently repaired parameters.
pub struct NMFConfig {
  n_components : Int32
  solver : Solver
  loss : Loss
  init : Init
  max_iter : Int32
  tol : Double
  seed : UInt64
  l1_reg : Double
  l2_reg : Double
}

///|
/// Configuration with sklearn-style defaults: MU / Frobenius / Random,
/// max_iter=200, tol=1e-4, seed=42, no regularization.
pub fn NMFConfig::new(
  n_components~ : Int32,
  solver~ : Solver = MU,
  loss~ : Loss = Frobenius,
  init~ : Init = Random,
  max_iter~ : Int32 = 200,
  tol~ : Double = 1.0e-4,
  seed~ : UInt64 = 42UL,
  l1_reg~ : Double = 0.0,
  l2_reg~ : Double = 0.0,
) -> NMFConfig {
  { n_components, solver, loss, init, max_iter, tol, seed, l1_reg, l2_reg, }
}

///|
/// Validates configuration. Unsupported solver/loss combinations and any
/// nonzero regularization are rejected explicitly (remaining implementations
/// arrive in T11–T12); nothing is silently substituted. All `Init` variants
/// are supported.
fn validate_config(c : NMFConfig) -> Result[Unit, NMFError] {
  if c.solver is CD && c.loss is KLDivergence {
    return Err(InvalidConfig("unsupported combination: CD solver with KL loss"))
  }
  match c.loss {
    KLDivergence =>
      return Err(InvalidConfig("KL loss is not implemented yet (planned T11)"))
    Frobenius => ()
  }
  if c.max_iter < 1 {
    return Err(InvalidConfig("max_iter must be >= 1, got \{c.max_iter}"))
  }
  if c.tol < 0.0 || c.tol.is_nan() || c.tol.is_inf() {
    return Err(InvalidConfig("tol must be finite and >= 0, got \{c.tol}"))
  }
  if c.l1_reg != 0.0 || c.l2_reg != 0.0 {
    return Err(
      InvalidConfig(
        "l1_reg/l2_reg are not implemented yet (planned T12); both must be 0.0",
      ),
    )
  }
  if c.n_components < 1 {
    return Err(InvalidRank(c.n_components))
  }
  Ok(())
}

///|
/// Successful training state, kept separate from configuration and from the
/// mutable model handle.
struct Fitted {
  w : Matrix
  h : Matrix
  n_iter : Int32
  loss_value : Double
  converged : Bool
}

///|
/// NMF model handle: configuration plus an optional fitted state.
///
/// `NMF::new(config)` creates an untrained model; `fit` trains it. Any
/// model-dependent call before a successful fit returns `Err(NotFitted)`.
/// A failed fit resets the model to the untrained state (no half-fitted
/// results are ever exposed).
pub struct NMF {
  config : NMFConfig
  mut fitted : Fitted?
}

///|
/// Creates an untrained model with the given configuration.
pub fn NMF::new(config : NMFConfig) -> NMF {
  { config, fitted: None, }
}

///|
/// Fits `V ≈ W·H` with the configured solver/loss/init.
///
/// Validation: configuration legality, non-empty V, non-negative entries
/// (`NegativeEntry(i, j)`), and `1 <= k <= min(rows, cols)` (`InvalidRank`).
/// On success the model exposes W, H, and training metadata; on any error
/// the model is left untrained.
pub fn NMF::fit(self : NMF, v : Matrix) -> Result[Unit, NMFError] {
  self.fitted = None
  match validate_config(self.config) {
    Err(e) => return Err(e)
    Ok(_) => ()
  }
  if v.rows == 0 || v.cols == 0 {
    return Err(InvalidMatrixShape("V must be non-empty for fitting"))
  }
  for i in 0..<v.rows {
    for j in 0..<v.cols {
      if v.data[i * v.cols + j] < 0.0 {
        return Err(NegativeEntry(i, j))
      }
    }
  }
  let k = self.config.n_components
  let max_k = if v.rows < v.cols { v.rows } else { v.cols }
  if k > max_k {
    return Err(InvalidRank(k))
  }
  let result = match self.config.init {
    Random =>
      match self.config.solver {
        MU =>
          mu_fit_summary(
            mu_frobenius_fit(
              v,
              k,
              self.config.max_iter,
              self.config.tol,
              self.config.seed,
            ),
          )
        CD =>
          cd_fit_summary(
            cd_frobenius_fit(
              v,
              k,
              self.config.max_iter,
              self.config.tol,
              self.config.seed,
            ),
          )
      }
    NNDSVD =>
      // Standard NNDSVD is deterministic and does not use NMFConfig.seed.
      match nndsvd_init(v, k) {
        Ok(pair) => {
          let (w0, h0) = pair
          fit_from_initial(
            v,
            w0,
            h0,
            self.config.solver,
            self.config.max_iter,
            self.config.tol,
          )
        }
        Err(e) => Err(e)
      }
    NNDSVDa =>
      // NNDSVDa reuses standard NNDSVD and fills its exact zeros with mean(V).
      match nndsvda_init(v, k) {
        Ok(pair) => {
          let (w0, h0) = pair
          fit_from_initial(
            v,
            w0,
            h0,
            self.config.solver,
            self.config.max_iter,
            self.config.tol,
          )
        }
        Err(e) => Err(e)
      }
  }
  let (w, h, n_iter, loss_value, converged) = match result {
    Ok(r) => r
    Err(e) => return Err(e)
  }
  self.fitted = Some({ w, h, n_iter, loss_value, converged, })
  Ok(())
}

///|
/// Runs the selected solver from already-initialized non-negative factors.
fn fit_from_initial(
  v : Matrix,
  w : Matrix,
  h : Matrix,
  solver : Solver,
  max_iter : Int32,
  tol : Double,
) -> Result[(Matrix, Matrix, Int32, Double, Bool), NMFError] {
  match solver {
    MU => mu_fit_summary(mu_frobenius_fit_from(v, w, h, max_iter, tol))
    CD => cd_fit_summary(cd_frobenius_fit_from(v, w, h, max_iter, tol))
  }
}

///|
/// Converts a MU fit result to the uniform tuple used by `NMF::fit`.
fn mu_fit_summary(
  r : Result[MuFit, NMFError],
) -> Result[(Matrix, Matrix, Int32, Double, Bool), NMFError] {
  match r {
    Ok(x) => Ok((x.w, x.h, x.n_iter, x.loss_value, x.converged))
    Err(e) => Err(e)
  }
}

///|
/// Converts a CD fit result to the uniform tuple used by `NMF::fit`.
fn cd_fit_summary(
  r : Result[CdFit, NMFError],
) -> Result[(Matrix, Matrix, Int32, Double, Bool), NMFError] {
  match r {
    Ok(x) => Ok((x.w, x.h, x.n_iter, x.loss_value, x.converged))
    Err(e) => Err(e)
  }
}

///|
/// Fits and returns the training coefficients W (n×k) in one call.
pub fn NMF::fit_transform(self : NMF, v : Matrix) -> Result[Matrix, NMFError] {
  match self.fit(v) {
    Err(e) => return Err(e)
    Ok(_) => ()
  }
  self.w()
}

///|
/// Solves `X ≈ W·H` for new data X with H fixed from training, using the
/// same solver/loss/iteration settings. X must be non-negative and have the
/// same column count as the training data.
pub fn NMF::transform(self : NMF, x : Matrix) -> Result[Matrix, NMFError] {
  let f = match self.fitted {
    Some(f) => f
    None => return Err(NotFitted)
  }
  if x.cols != f.h.cols {
    return Err(
      DimensionMismatch(
        "transform: x has \{x.cols} columns, model expects \{f.h.cols}",
      ),
    )
  }
  if x.rows == 0 {
    return Err(InvalidMatrixShape("transform: x must have at least one row"))
  }
  for i in 0..<x.rows {
    for j in 0..<x.cols {
      if x.data[i * x.cols + j] < 0.0 {
        return Err(NegativeEntry(i, j))
      }
    }
  }
  match self.config.solver {
    MU =>
      mu_frobenius_transform(
        x,
        f.h,
        self.config.max_iter,
        self.config.tol,
        self.config.seed,
      )
    CD =>
      cd_frobenius_transform(
        x,
        f.h,
        self.config.max_iter,
        self.config.tol,
        self.config.seed,
      )
  }
}

///|
/// Reconstructs `W·H` from coefficients W (m×k). W's column count must equal
/// the model rank.
pub fn NMF::inverse_transform(
  self : NMF,
  w : Matrix,
) -> Result[Matrix, NMFError] {
  let f = match self.fitted {
    Some(f) => f
    None => return Err(NotFitted)
  }
  if w.cols != f.h.rows {
    return Err(
      DimensionMismatch(
        "inverse_transform: w has \{w.cols} columns, model rank is \{f.h.rows}",
      ),
    )
  }
  matmul(w, f.h)
}

///|
/// Reconstruction error `‖V − WH‖_F` (Frobenius norm, sklearn semantics).
/// Always the norm, regardless of the training loss; equals
/// `sqrt(2 * loss_value)` for Frobenius training on the same V.
/// V must have the training dimensions.
pub fn NMF::reconstruction_error(
  self : NMF,
  v : Matrix,
) -> Result[Double, NMFError] {
  let f = match self.fitted {
    Some(f) => f
    None => return Err(NotFitted)
  }
  if v.rows != f.w.rows || v.cols != f.h.cols {
    return Err(
      DimensionMismatch(
        "reconstruction_error: v is \{v.rows}x\{v.cols}, training data was \{f.w.rows}x\{f.h.cols}",
      ),
    )
  }
  let wh = zeros_buf(v.rows, v.cols)
  matmul_into(f.w, f.h, wh)
  let s = sq_diff_from_wh(v, wh)
  if s.is_nan() || s.is_inf() {
    return Err(
      NumericalError("reconstruction error overflowed to a non-finite value"),
    )
  }
  Ok(s.sqrt())
}

///|
/// Basis matrix W (n×k), as a defensive copy. `NotFitted` before training.
pub fn NMF::w(self : NMF) -> Result[Matrix, NMFError] {
  match self.fitted {
    Some(f) => Ok(f.w.clone())
    None => Err(NotFitted)
  }
}

///|
/// Component matrix H (k×d), as a defensive copy. `NotFitted` before training.
pub fn NMF::h(self : NMF) -> Result[Matrix, NMFError] {
  match self.fitted {
    Some(f) => Ok(f.h.clone())
    None => Err(NotFitted)
  }
}

///|
/// Number of solver iterations actually executed. `NotFitted` before training.
pub fn NMF::n_iter(self : NMF) -> Result[Int32, NMFError] {
  match self.fitted {
    Some(f) => Ok(f.n_iter)
    None => Err(NotFitted)
  }
}

///|
/// Final true objective value `½‖V − WH‖²_F`. `NotFitted` before training.
pub fn NMF::loss_value(self : NMF) -> Result[Double, NMFError] {
  match self.fitted {
    Some(f) => Ok(f.loss_value)
    None => Err(NotFitted)
  }
}

///|
/// Whether the relative-objective-change criterion was met. False when
/// max_iter was exhausted without convergence. `NotFitted` before training.
pub fn NMF::converged(self : NMF) -> Result[Bool, NMFError] {
  match self.fitted {
    Some(f) => Ok(f.converged)
    None => Err(NotFitted)
  }
}
