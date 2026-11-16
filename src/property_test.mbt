// Property-based tests (T14): random legal inputs must satisfy the five
// properties from PLAN §9 — W/H non-negative, correct dimensions, same-seed
// reproducibility, no panic on legal input, and typed errors on illegal input
// (the last one is exercised by `fuzz_test.mbt`; here every generated input is
// legal, so the property is that the call returns `Ok`).
//
// Determinism: one fixed seed per test function, the shared `TestRng` from
// `e2e_test.mbt` (independent of the library's SplitMix64), no clock, no
// external randomness. Every failure message carries the seed and case index,
// plus the generation parameters needed to reproduce the case.
//
// Why MU-KL never gets NNDSVD/NNDSVDa in the main scan: those initializers
// produce structural zeros, and MU-KL reports `NumericalError` when the
// initial `WH` is zero at a position where `V > 0` (documented T11 behaviour,
// covered by unit tests). The main scan therefore uses Random for MU-KL, and a
// separate test exercises the deterministic initializers on strictly positive
// data, where the leading NNDSVD component keeps `WH > 0` everywhere.

///|
let prop_seed : UInt64 = 20261116UL

///|
fn prop_mat(rows : Int32, cols : Int32, data : Array[Double]) -> @nmf.Matrix {
  match @nmf.Matrix::new(rows, cols, data) {
    Ok(m) => m
    Err(e) => abort("unexpected matrix error: \{e}")
  }
}

///|
fn prop_require(cond : Bool, case : Int32, detail : String) -> Unit {
  if !cond {
    abort("property failed: seed=\{prop_seed} case=\{case} \{detail}")
  }
}

///|
fn prop_factor_ok(
  m : @nmf.Matrix,
  rows : Int32,
  cols : Int32,
  case : Int32,
  what : String,
) -> Unit {
  prop_require(m.rows() == rows, case, "\{what}.rows=\{m.rows()} want \{rows}")
  prop_require(m.cols() == cols, case, "\{what}.cols=\{m.cols()} want \{cols}")
  for i in 0..<rows {
    for j in 0..<cols {
      let x = match m.get(i, j) {
        Ok(x) => x
        Err(e) => abort("seed=\{prop_seed} case=\{case} \{what}.get: \{e}")
      }
      prop_require(x >= 0.0, case, "\{what}[\{i},\{j}] negative: \{x}")
      prop_require(!x.is_nan() && !x.is_inf(), case, "\{what} non-finite")
    }
  }
}

///|
fn prop_sq_error(v : @nmf.Matrix, w : @nmf.Matrix, h : @nmf.Matrix) -> Double {
  let mut s = 0.0
  for i in 0..<v.rows() {
    for j in 0..<v.cols() {
      let mut wh = 0.0
      for p in 0..<w.cols() {
        wh = wh + unwrap_double(w.get(i, p)) * unwrap_double(h.get(p, j))
      }
      let diff = unwrap_double(v.get(i, j)) - wh
      s = s + diff * diff
    }
  }
  s
}

///|
fn prop_penalty(
  w : @nmf.Matrix,
  h : @nmf.Matrix,
  l1 : Double,
  l2 : Double,
) -> Double {
  let mut sum = 0.0
  let mut sq = 0.0
  for i in 0..<w.rows() {
    for j in 0..<w.cols() {
      let x = unwrap_double(w.get(i, j))
      sum = sum + x
      sq = sq + x * x
    }
  }
  for i in 0..<h.rows() {
    for j in 0..<h.cols() {
      let x = unwrap_double(h.get(i, j))
      sum = sum + x
      sq = sq + x * x
    }
  }
  l1 * sum + 0.5 * l2 * sq
}

///|
fn prop_data(
  rng : TestRng,
  rows : Int32,
  cols : Int32,
  with_zeros : Bool,
  scale : Double,
) -> Array[Double] {
  let data = Array::make(rows * cols, 0.0)
  for i in 0..<rows {
    for j in 0..<cols {
      // Exact zeros are a legal and important NMF case. The pattern uses
      // (i + j) % 3 so that every row and every column still has a positive
      // entry: an all-zero column would drive the matching H column to zero
      // in MU-KL and make a later transform report the documented infinite-KL
      // error whenever the new data is positive there. Sprinkling zeros keeps
      // every MU-KL fit well-posed without weakening the input variety.
      let value = if with_zeros && (i + j) % 3 == 0 {
        0.0
      } else {
        0.05 + rng.next_double() * 3.0
      }
      data[i * cols + j] = value * scale
    }
  }
  data
}

///|
fn prop_init_name(init : @nmf.Init) -> String {
  match init {
    @nmf.Init::Random => "Random"
    @nmf.Init::NNDSVD => "NNDSVD"
    @nmf.Init::NNDSVDa => "NNDSVDa"
  }
}

///|
fn prop_solver_name(kind : Int32) -> String {
  if kind == 0 {
    "MU+Frobenius"
  } else if kind == 1 {
    "CD+Frobenius"
  } else {
    "MU+KLDivergence"
  }
}

// ---------------------------------------------------------------------
// 200 legal cases: five properties + transform + reported-value checks.
// ---------------------------------------------------------------------

///|
test "prop: 200 random legal inputs satisfy the five properties" {
  let rng = TestRng::new(prop_seed)
  let mut transform_calls = 0
  // Coverage counters: they make the "varied configurations" claim checkable
  // and fail loudly if a future edit collapses the generator to one shape.
  let mut count_mu_frob = 0
  let mut count_cd = 0
  let mut count_kl = 0
  let mut count_reg = 0
  let mut count_det_init = 0
  let mut count_zeros = 0
  for case in 0..<200 {
    // ---- generation -------------------------------------------------
    let rows = 2 + rng.next_int(8) // 2..9
    let cols = 2 + rng.next_int(8) // 2..9
    let max_k = if rows < cols { rows } else { cols }
    let k = 1 + rng.next_int(max_k)
    let kind = case % 3 // 0 MU+Frob, 1 CD+Frob, 2 MU+KL
    let with_zeros = case % 4 != 0 // 75% of cases contain exact zeros
    let scale = if case % 7 == 0 {
      1.0e-3
    } else if case % 11 == 0 {
      1.0e3
    } else {
      1.0
    }
    // MU-KL is paired with Random init and strictly positive data in this
    // scan; deterministic initializers get their own test below.
    let use_deterministic = case % 6 == 4 || case % 6 == 5
    let random_init = kind == 2 || !use_deterministic
    let init = if random_init {
      @nmf.Init::Random
    } else if case % 6 == 4 {
      @nmf.Init::NNDSVD
    } else {
      @nmf.Init::NNDSVDa
    }
    let data = prop_data(rng, rows, cols, with_zeros && random_init, scale)
    let v = prop_mat(rows, cols, data)
    let has_reg = case % 5 == 0
    let l1 = if has_reg { 0.05 } else { 0.0 }
    let l2 = if has_reg { 0.1 } else { 0.0 }
    let max_iter = 5 + rng.next_int(30)
    let solver = if kind == 1 { @nmf.Solver::CD } else { @nmf.Solver::MU }
    let loss = if kind == 2 {
      @nmf.Loss::KLDivergence
    } else {
      @nmf.Loss::Frobenius
    }
    let config = @nmf.NMFConfig::new(
      n_components=k,
      solver~,
      loss~,
      init~,
      max_iter~,
      tol=1.0e-6,
      seed=(case + 1).to_uint64(),
      l1_reg=l1,
      l2_reg=l2,
    )
    if kind == 0 {
      count_mu_frob = count_mu_frob + 1
    } else if kind == 1 {
      count_cd = count_cd + 1
    } else {
      count_kl = count_kl + 1
    }
    if has_reg {
      count_reg = count_reg + 1
    }
    if !random_init {
      count_det_init = count_det_init + 1
    }
    if with_zeros && random_init {
      count_zeros = count_zeros + 1
    }
    // ---- property 4: legal input does not panic (call completes) -----
    let model = @nmf.NMF::new(config)
    match model.fit(v) {
      Ok(_) => ()
      Err(e) =>
        abort(
          "seed=\{prop_seed} case=\{case} \{prop_solver_name(kind)} init=\{prop_init_name(init)} size=\{rows}x\{cols} k=\{k} reg=\{l1},\{l2}: unexpected \{e}",
        )
    }
    // ---- property 1 + 2: non-negative factors, exact dimensions ------
    let w = match model.w() {
      Ok(m) => m
      Err(e) => abort("seed=\{prop_seed} case=\{case} w(): \{e}")
    }
    let h = match model.h() {
      Ok(m) => m
      Err(e) => abort("seed=\{prop_seed} case=\{case} h(): \{e}")
    }
    prop_factor_ok(w, rows, k, case, "W")
    prop_factor_ok(h, k, cols, case, "H")
    // Convergence metadata must stay honest for any budget.
    let n_iter = match model.n_iter() {
      Ok(n) => n
      Err(e) => abort("seed=\{prop_seed} case=\{case} n_iter(): \{e}")
    }
    let converged = match model.converged() {
      Ok(c) => c
      Err(e) => abort("seed=\{prop_seed} case=\{case} converged(): \{e}")
    }
    prop_require(
      n_iter >= 1 && n_iter <= max_iter,
      case,
      "n_iter=\{n_iter} budget=\{max_iter}",
    )
    prop_require(
      converged || n_iter == max_iter,
      case,
      "unconverged early stop",
    )
    // ---- reported values are consistent with the factors -------------
    let reported = match model.loss_value() {
      Ok(x) => x
      Err(e) => abort("seed=\{prop_seed} case=\{case} loss_value(): \{e}")
    }
    prop_require(
      !reported.is_nan() && !reported.is_inf() && reported >= 0.0,
      case,
      "loss_value=\{reported}",
    )
    let sq = prop_sq_error(v, w, h)
    let err = match model.reconstruction_error(v) {
      Ok(x) => x
      Err(e) =>
        abort("seed=\{prop_seed} case=\{case} reconstruction_error: \{e}")
    }
    let err_tol = 1.0e-9 * (1.0 + err)
    prop_require(
      (err - sq.sqrt()).abs() <= err_tol,
      case,
      "reconstruction_error",
    )
    let penalty = prop_penalty(w, h, l1, l2)
    if kind != 2 {
      // Frobenius: loss_value is exactly ½‖V−WH‖² + Ω.
      let want = 0.5 * sq + penalty
      let tol = 1.0e-9 * (1.0 + want)
      prop_require((reported - want).abs() <= tol, case, "Frobenius loss_value")
    } else {
      // KL: the divergence is non-negative, so loss_value >= Ω must hold.
      prop_require(
        reported >= penalty - 1.0e-9 * (1.0 + penalty),
        case,
        "KL >= Ω",
      )
    }
    // ---- property 3: same seed reproduces the whole run --------------
    let twin = @nmf.NMF::new(config)
    match twin.fit(v) {
      Ok(_) => ()
      Err(e) => abort("seed=\{prop_seed} case=\{case} twin fit: \{e}")
    }
    let w2 = match twin.w() {
      Ok(m) => m
      Err(e) => abort("seed=\{prop_seed} case=\{case} twin w: \{e}")
    }
    let h2 = match twin.h() {
      Ok(m) => m
      Err(e) => abort("seed=\{prop_seed} case=\{case} twin h: \{e}")
    }
    prop_require(
      twin.n_iter() == model.n_iter(),
      case,
      "n_iter not reproducible",
    )
    prop_require(
      twin.loss_value() == model.loss_value(),
      case,
      "loss not reproducible",
    )
    for i in 0..<rows {
      for j in 0..<k {
        prop_require(
          unwrap_double(w.get(i, j)) == unwrap_double(w2.get(i, j)),
          case,
          "W not reproducible",
        )
      }
    }
    for i in 0..<k {
      for j in 0..<cols {
        prop_require(
          unwrap_double(h.get(i, j)) == unwrap_double(h2.get(i, j)),
          case,
          "H not reproducible",
        )
      }
    }
    // ---- transform: fixed H, new W, model untouched ------------------
    let x_rows = 1 + rng.next_int(4)
    let x = prop_mat(x_rows, cols, prop_data(rng, x_rows, cols, true, scale))
    let w_new = match model.transform(x) {
      Ok(m) => m
      Err(e) =>
        abort("seed=\{prop_seed} case=\{case} transform: unexpected \{e}")
    }
    transform_calls = transform_calls + 1
    prop_factor_ok(w_new, x_rows, k, case, "transform W")
    let h_after = match model.h() {
      Ok(m) => m
      Err(e) =>
        abort("seed=\{prop_seed} case=\{case} h() after transform: \{e}")
    }
    for i in 0..<k {
      for j in 0..<cols {
        prop_require(
          unwrap_double(h.get(i, j)) == unwrap_double(h_after.get(i, j)),
          case,
          "transform mutated H",
        )
      }
    }
    // Transform is deterministic for the same input and config.
    let w_new2 = match model.transform(x) {
      Ok(m) => m
      Err(e) => abort("seed=\{prop_seed} case=\{case} transform twice: \{e}")
    }
    for i in 0..<x_rows {
      for j in 0..<k {
        prop_require(
          unwrap_double(w_new.get(i, j)) == unwrap_double(w_new2.get(i, j)),
          case,
          "transform not deterministic",
        )
      }
    }
    // inverse_transform(W) = W·H, with the training dimensions.
    let recon = match model.inverse_transform(w) {
      Ok(m) => m
      Err(e) => abort("seed=\{prop_seed} case=\{case} inverse_transform: \{e}")
    }
    prop_factor_ok(recon, rows, cols, case, "inverse_transform")
    // ---- sparsity of the factors stays inside [0,1] ------------------
    for j in 0..<k {
      let column = Array::make(rows, 0.0)
      for i in 0..<rows {
        column[i] = unwrap_double(w.get(i, j))
      }
      let s = match @nmf.sparsity(column) {
        Ok(x) => x
        Err(e) => abort("seed=\{prop_seed} case=\{case} W sparsity: \{e}")
      }
      prop_require(s >= 0.0 && s <= 1.0, case, "W sparsity out of range: \{s}")
    }
    for i in 0..<k {
      let row = Array::make(cols, 0.0)
      for j in 0..<cols {
        row[j] = unwrap_double(h.get(i, j))
      }
      let s = match @nmf.sparsity(row) {
        Ok(x) => x
        Err(e) => abort("seed=\{prop_seed} case=\{case} H sparsity: \{e}")
      }
      prop_require(s >= 0.0 && s <= 1.0, case, "H sparsity out of range: \{s}")
    }
  }
  assert_eq(transform_calls, 200)
  assert_eq(count_mu_frob + count_cd + count_kl, 200)
  assert_true(count_mu_frob >= 50 && count_cd >= 50 && count_kl >= 50)
  assert_true(count_reg >= 30)
  assert_true(count_det_init >= 30)
  assert_true(count_zeros >= 100)
}

// ---------------------------------------------------------------------
// Deterministic initializers with MU-KL (strictly positive data).
// ---------------------------------------------------------------------

///|
test "prop: deterministic initializers work with MU-KL on positive data" {
  let rng = TestRng::new(prop_seed + 1UL)
  for case in 0..<12 {
    let rows = 3 + rng.next_int(5)
    let cols = 3 + rng.next_int(5)
    let max_k = if rows < cols { rows } else { cols }
    let k = 1 + rng.next_int(if max_k > 2 { 2 } else { max_k })
    // Strictly positive: the leading NNDSVD component is strictly positive, so
    // the initial WH has no zero where V > 0 (the documented MU-KL
    // precondition) and the fit must succeed.
    let data = prop_data(rng, rows, cols, false, 1.0)
    let v = prop_mat(rows, cols, data)
    let init = if case % 2 == 0 {
      @nmf.Init::NNDSVD
    } else {
      @nmf.Init::NNDSVDa
    }
    let model = @nmf.NMF::new(
      @nmf.NMFConfig::new(
        n_components=k,
        loss=@nmf.Loss::KLDivergence,
        init~,
        max_iter=20,
        tol=1.0e-6,
        seed=(case + 1).to_uint64(),
        l1_reg=0.0,
        l2_reg=0.05,
      ),
    )
    match model.fit(v) {
      Ok(_) => ()
      Err(e) =>
        abort(
          "seed=\{prop_seed + 1UL} case=\{case} MU-KL init=\{prop_init_name(init)} size=\{rows}x\{cols} k=\{k}: \{e}",
        )
    }
    let w = match model.w() {
      Ok(m) => m
      Err(e) => abort("case=\{case} w: \{e}")
    }
    let h = match model.h() {
      Ok(m) => m
      Err(e) => abort("case=\{case} h: \{e}")
    }
    prop_factor_ok(w, rows, k, case, "W")
    prop_factor_ok(h, k, cols, case, "H")
  }
}

// ---------------------------------------------------------------------
// Loss monotonicity (PLAN §9: per-algorithm, not a blanket claim).
// ---------------------------------------------------------------------

///|
test "prop: MU and CD objectives are non-increasing along a run" {
  let rng = TestRng::new(prop_seed + 2UL)
  for case in 0..<8 {
    let rows = 4 + rng.next_int(4)
    let cols = 3 + rng.next_int(4)
    let max_k = if rows < cols { rows } else { cols }
    let k = 1 + rng.next_int(if max_k > 3 { 3 } else { max_k })
    let v = prop_mat(rows, cols, prop_data(rng, rows, cols, true, 1.0))
    for kind in 0..<3 {
      let solver = if kind == 1 { @nmf.Solver::CD } else { @nmf.Solver::MU }
      let loss = if kind == 2 {
        @nmf.Loss::KLDivergence
      } else {
        @nmf.Loss::Frobenius
      }
      // tol = 0 forces the full budget, so each run is a prefix of the next
      // and the recorded objectives must form a non-increasing sequence.
      let mut prev = 1.0e300
      for budget in [1, 2, 3, 5, 8] {
        let model = @nmf.NMF::new(
          @nmf.NMFConfig::new(
            n_components=k,
            solver~,
            loss~,
            init=@nmf.Init::Random,
            max_iter=budget,
            tol=0.0,
            seed=(case + 1).to_uint64(),
            l1_reg=0.02,
            l2_reg=0.05,
          ),
        )
        match model.fit(v) {
          Ok(_) => ()
          Err(e) =>
            abort(
              "seed=\{prop_seed + 2UL} case=\{case} \{prop_solver_name(kind)} budget=\{budget}: \{e}",
            )
        }
        let f = match model.loss_value() {
          Ok(x) => x
          Err(e) => abort("case=\{case} loss: \{e}")
        }
        prop_require(
          f <= prev + 1.0e-9 * (1.0 + prev.abs()),
          case,
          "\{prop_solver_name(kind)} objective increased: \{f} > \{prev}",
        )
        prev = f
      }
    }
  }
}

// ---------------------------------------------------------------------
// Hoyer sparsity random properties (T13 metric, T14 properties).
// ---------------------------------------------------------------------

///|
test "prop: sparsity satisfies range, scale, and permutation invariance" {
  let rng = TestRng::new(prop_seed + 3UL)
  for case in 0..<100 {
    let n = 2 + rng.next_int(9) // 2..10
    let base = Array::make(n, 0.0)
    let mut nonzeros = 0
    for i in 0..<n {
      let value = if rng.next_double() < 0.25 {
        0.0
      } else {
        rng.next_double() * 3.0 + 1.0e-3
      }
      base[i] = value
      if value != 0.0 {
        nonzeros = nonzeros + 1
      }
    }
    let s = match @nmf.sparsity(base) {
      Ok(x) => x
      Err(e) => abort("seed=\{prop_seed + 3UL} case=\{case}: \{e}")
    }
    prop_require(s >= 0.0 && s <= 1.0, case, "S outside [0,1]: \{s}")
    // An all-zero vector takes the PLAN convention.
    if nonzeros == 0 {
      prop_require(s == 1.0, case, "all-zero must be 1.0")
      continue
    }
    // Scale invariance for positive factors that stay finite.
    for factor in [2.0, 1000.0, 1.0e-120] {
      let scaled = Array::make(n, 0.0)
      for i in 0..<n {
        scaled[i] = base[i] * factor
      }
      let s2 = match @nmf.sparsity(scaled) {
        Ok(x) => x
        Err(e) => abort("case=\{case} scale \{factor}: \{e}")
      }
      prop_require(
        (s2 - s).abs() <= 1.0e-9 * (1.0 + s.abs()),
        case,
        "scale \{factor} changed S: \{s2} vs \{s}",
      )
    }
    // Permutation invariance (reverse order, then a rotated order).
    let reversed = Array::make(n, 0.0)
    for i in 0..<n {
      reversed[i] = base[n - 1 - i]
    }
    let rotated = Array::make(n, 0.0)
    for i in 0..<n {
      rotated[i] = base[(i + 1) % n]
    }
    for perm in [reversed, rotated] {
      let s3 = match @nmf.sparsity(perm) {
        Ok(x) => x
        Err(e) => abort("case=\{case} permutation: \{e}")
      }
      prop_require(
        (s3 - s).abs() <= 1.0e-9 * (1.0 + s.abs()),
        case,
        "permutation changed S: \{s3} vs \{s}",
      )
    }
  }
  // Fixed conventions on the random-scan boundary values.
  assert_eq(@nmf.sparsity([0.0, 0.0, 0.0]), Ok(1.0))
  assert_eq(@nmf.sparsity([7.5]), Ok(0.0))
}
