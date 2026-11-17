// NMFlib Topic Modeling MVP: `moon run examples/topic_modeling` from the repo root.
//
// Pipeline: fixed corpus -> whitespace tokenizer -> deterministic vocabulary
// (first-occurrence order) -> document-term matrix V (docs x words) ->
// NMF (MU + generalized KL + Random init) -> topic keywords from H rows,
// Hoyer sparsity per topic, document topic activations from W rows, and
// transform() inference for unseen documents. Every stage is self-checked;
// any failure aborts instead of printing fabricated results.

///|
let rank : Int32 = 3

///|
let seed : UInt64 = 42UL

///|
let max_iter : Int32 = 500

///|
let tol : Double = 1.0e-4

///|
// Selected by the parameter study documented in README.md (smallest non-zero
// L1 that yields meaningful sparsity without topic collapse).
let l1_reg : Double = 0.001

///|
let l2_reg : Double = 0.0

///|
let top_k : Int32 = 6

///|
fn require(cond : Bool, msg : String) -> Unit {
  if !cond {
    abort("Topic modeling self-check failed: \{msg}")
  }
}

///|
fn unwrap_matrix(
  r : Result[@nmf.Matrix, @nmf.NMFError],
  context : String,
) -> @nmf.Matrix {
  match r {
    Ok(m) => m
    Err(e) => abort("\{context} failed: \{e}")
  }
}

///|
fn unwrap_double(r : Result[Double, @nmf.NMFError], context : String) -> Double {
  match r {
    Ok(x) => x
    Err(e) => abort("\{context} failed: \{e}")
  }
}

///|
// Whitespace tokenizer: splits on space/tab/CR/LF, drops empty tokens.
// Input is already lowercase; no stemming, stopwords, or punctuation rules.
fn tokenize(text : String) -> Array[String] {
  let words : Array[String] = []
  let buf : Array[Char] = []
  for c in text {
    if c == ' ' || c == '\t' || c == '\n' || c == '\r' {
      if buf.length() > 0 {
        words.push(String::from_array(buf))
        buf.clear()
      }
    } else {
      buf.push(c)
    }
  }
  if buf.length() > 0 {
    words.push(String::from_array(buf))
  }
  words
}

///|
fn vocab_index(vocab : Array[String], word : String) -> Int32 {
  for j, w in vocab {
    if w == word {
      return j
    }
  }
  -1
}

///|
// Deterministic vocabulary: words in order of first appearance in the corpus.
fn build_vocabulary(docs : Array[String]) -> Array[String] {
  let vocab : Array[String] = []
  for doc in docs {
    for word in tokenize(doc) {
      if vocab_index(vocab, word) < 0 {
        vocab.push(word)
      }
    }
  }
  vocab
}

///|
// Builds a document-term matrix (rows = documents, cols = vocabulary) plus
// per-document OOV and known-token counts. OOV words are counted and ignored;
// the vocabulary is never extended.
fn doc_term_matrix(
  docs : Array[String],
  vocab : Array[String],
) -> (@nmf.Matrix, Array[Int32], Array[Int32]) {
  let m = docs.length()
  let n = vocab.length()
  let data = Array::make(m * n, 0.0)
  let oov = Array::make(m, 0)
  let known = Array::make(m, 0)
  for i, doc in docs {
    for word in tokenize(doc) {
      let j = vocab_index(vocab, word)
      if j < 0 {
        oov[i] = oov[i] + 1
      } else {
        data[i * n + j] = data[i * n + j] + 1.0
        known[i] = known[i] + 1
      }
    }
  }
  let v = unwrap_matrix(@nmf.Matrix::new(m, n, data), "Matrix::new")
  (v, oov, known)
}

///|
fn matrix_row(m : @nmf.Matrix, i : Int32) -> Array[Double] {
  let row : Array[Double] = []
  for j in 0..<m.cols() {
    row.push(m.data[i * m.cols() + j])
  }
  row
}

///|
fn check_shape(
  name : String,
  m : @nmf.Matrix,
  rows : Int32,
  cols : Int32,
) -> Unit {
  require(
    m.rows() == rows && m.cols() == cols,
    "\{name} shape \{m.rows()}x\{m.cols()}, expected \{rows}x\{cols}",
  )
}

///|
fn check_nonneg_finite(name : String, m : @nmf.Matrix) -> Unit {
  for i in 0..<m.rows() {
    for j in 0..<m.cols() {
      let x = m.data[i * m.cols() + j]
      require(
        !x.is_nan() && !x.is_inf(),
        "\{name} has non-finite entry at (\{i}, \{j})",
      )
      require(x >= 0.0, "\{name} has negative entry at (\{i}, \{j}): \{x}")
    }
  }
}

///|
fn check_no_zero_rows(name : String, m : @nmf.Matrix) -> Unit {
  for i in 0..<m.rows() {
    require(row_sum(matrix_row(m, i)) > 0.0, "\{name} row \{i} is all zero")
  }
}

///|
fn check_no_zero_cols(name : String, m : @nmf.Matrix) -> Unit {
  for j in 0..<m.cols() {
    let mut s = 0.0
    for i in 0..<m.rows() {
      s = s + m.data[i * m.cols() + j]
    }
    require(s > 0.0, "\{name} column \{j} is all zero")
  }
}

///|
fn check_finite_scalar(name : String, x : Double) -> Unit {
  require(!x.is_nan() && !x.is_inf(), "\{name} is not finite: \{x}")
}

///|
// Top-K (word, weight) pairs by descending weight; ties keep vocabulary order.
// Simple selection sort: vocabulary and rank are tiny, no dependencies needed.
fn top_keywords(
  row : Array[Double],
  vocab : Array[String],
  k : Int32,
) -> Array[(String, Double)] {
  let pairs : Array[(String, Double)] = []
  for j, w in row {
    pairs.push((vocab[j], w))
  }
  let n = pairs.length()
  for i in 0..<n {
    let mut best = i
    for j in (i + 1)..<n {
      if pairs[j].1 > pairs[best].1 {
        best = j
      }
    }
    let tmp = pairs[i]
    pairs[i] = pairs[best]
    pairs[best] = tmp
  }
  let out : Array[(String, Double)] = []
  for i in 0..<k {
    if i < n {
      out.push(pairs[i])
    }
  }
  out
}

///|
fn argmax(row : Array[Double]) -> Int32 {
  let mut best = 0
  for j in 1..<row.length() {
    if row[j] > row[best] {
      best = j
    }
  }
  best
}

///|
fn row_sum(row : Array[Double]) -> Double {
  let mut s = 0.0
  for x in row {
    s = s + x
  }
  s
}

///|
fn fmt4(x : Double) -> Double {
  (x * 10000.0).round() / 10000.0
}

///|
fn pad(s : String, width : Int32) -> String {
  let mut out = s
  for _ in s.length()..<width {
    out = "\{out} "
  }
  out
}

///|
fn print_activations(row : Array[Double]) -> Unit {
  let total = row_sum(row)
  require(total > 0.0, "document has zero total topic activation")
  for t, x in row {
    println("    Topic \{t}: \{fmt4(x / total)} (normalized activation)")
  }
}

///|
fn main {
  println("========================================")
  println("NMFlib Topic Modeling MVP")
  println("========================================")
  // [1] Dataset: vocabulary and document-term matrix V (docs x words).
  println("")
  println("[1] Dataset")
  let vocab = build_vocabulary(training_docs)
  require(training_docs.length() > 0, "empty corpus")
  require(vocab.length() > 0, "empty vocabulary")
  let (v, _, _) = doc_term_matrix(training_docs, vocab)
  let m = training_docs.length()
  let n = vocab.length()
  check_shape("V", v, m, n)
  check_nonneg_finite("V", v)
  check_no_zero_rows("V", v)
  check_no_zero_cols("V", v)
  println("Documents: \{m}")
  println("Vocabulary size: \{n}")
  println("Matrix shape: \{v.rows()} x \{v.cols()} (documents x words)")
  let mut vocab_line = ""
  for j, word in vocab {
    vocab_line = if j == 0 { word } else { "\{vocab_line}, \{word}" }
  }
  println("Vocabulary (first-occurrence order): \{vocab_line}")
  // [2] Model configuration.
  println("")
  println("[2] Model Configuration")
  println("Solver: MU")
  println("Loss: Generalized KL")
  println("Init: Random")
  println("Rank: \{rank}")
  println("L1: \{l1_reg}")
  println("L2: \{l2_reg}")
  println("Seed: \{seed}")
  println("max_iter: \{max_iter}, tol: \{tol}")
  let config = @nmf.NMFConfig::new(
    n_components=rank,
    solver=MU,
    loss=KLDivergence,
    init=Random,
    max_iter~,
    tol~,
    seed~,
    l1_reg~,
    l2_reg~,
  )
  let model = @nmf.NMF::new(config)
  // [3] Training and factor checks.
  println("")
  println("[3] Training")
  match model.fit(v) {
    Ok(_) => ()
    Err(e) => abort("fit failed: \{e}")
  }
  let w = unwrap_matrix(model.w(), "w()")
  let h = unwrap_matrix(model.h(), "h()")
  check_shape("W", w, m, rank)
  check_shape("H", h, rank, n)
  check_nonneg_finite("W", w)
  check_nonneg_finite("H", h)
  let n_iter = match model.n_iter() {
    Ok(x) => x
    Err(e) => abort("n_iter() failed: \{e}")
  }
  let converged = match model.converged() {
    Ok(x) => x
    Err(e) => abort("converged() failed: \{e}")
  }
  let loss = unwrap_double(model.loss_value(), "loss_value()")
  let recon = unwrap_double(
    model.reconstruction_error(v),
    "reconstruction_error",
  )
  check_finite_scalar("loss_value", loss)
  check_finite_scalar("reconstruction_error", recon)
  println("Iterations: \{n_iter}")
  println("Converged: \{converged}")
  println("Full objective (loss_value, includes regularization): \{loss}")
  println("Reconstruction error (Frobenius ||V - WH||_F): \{recon}")
  // [4] Topic keywords from H rows (topic order is not fixed by NMF).
  println("")
  println("[4] Discovered Topics")
  require(h.rows() == rank, "topic count \{h.rows()} != rank \{rank}")
  for t in 0..<rank {
    let row = matrix_row(h, t)
    let keywords = top_keywords(row, vocab, top_k)
    require(
      keywords.length() == top_k,
      "topic \{t}: fewer than \{top_k} keywords available",
    )
    println("")
    println("==============================")
    println("Topic \{t}")
    println("==============================")
    for (word, weight) in keywords {
      println("\{pad(word, 14)}\{fmt4(weight)}")
    }
  }
  // [5] Hoyer sparsity per topic: extract each H row, call the public API.
  println("")
  println("[5] Topic Sparsity (Hoyer, per H row)")
  let mut sparsity_sum = 0.0
  for t in 0..<rank {
    let row = matrix_row(h, t)
    let s = unwrap_double(@nmf.sparsity(row), "sparsity(topic \{t})")
    require(s >= 0.0 && s <= 1.0, "topic \{t} sparsity out of [0, 1]: \{s}")
    sparsity_sum = sparsity_sum + s
    println("Topic \{t} sparsity: \{fmt4(s)}")
  }
  println("Average topic sparsity: \{fmt4(sparsity_sum / rank.to_double())}")
  // [6] Document topic activations from W rows (relative, not probabilities).
  println("")
  println("[6] Document Topic Distribution")
  for i in 0..<m {
    let row = matrix_row(w, i)
    let dominant = argmax(row)
    println(
      "Document \{i} [expected: \{expected_labels[i]}] -> Topic \{dominant}",
    )
    print_activations(row)
  }
  // [7] New-document inference with the trained model (no retraining).
  println("")
  println("[7] New Document Inference (transform)")
  let (v_new, oov, known) = doc_term_matrix(new_docs, vocab)
  check_shape("V_new", v_new, new_docs.length(), n)
  check_nonneg_finite("V_new", v_new)
  let valid_rows : Array[Int32] = []
  for i in 0..<new_docs.length() {
    println(
      "New document \{i}: \{known[i]} known tokens, \{oov[i]} OOV tokens (ignored)",
    )
    if known[i] == 0 {
      println(
        "New document \{i}: all tokens are OOV; skipped (no topic inference)",
      )
    } else {
      let s = row_sum(matrix_row(v_new, i))
      require(
        s > 0.0,
        "new document \{i} has known tokens but zero term counts (bug)",
      )
      valid_rows.push(i)
    }
  }
  if valid_rows.length() > 0 {
    let h_before = h.data.copy()
    let data : Array[Double] = []
    for i in valid_rows {
      for x in matrix_row(v_new, i) {
        data.push(x)
      }
    }
    let v_valid = unwrap_matrix(
      @nmf.Matrix::new(valid_rows.length(), n, data),
      "Matrix::new(V_valid)",
    )
    let w_new = unwrap_matrix(model.transform(v_valid), "transform")
    check_shape("W_new", w_new, valid_rows.length(), rank)
    check_nonneg_finite("W_new", w_new)
    let h_after = unwrap_matrix(model.h(), "h() after transform")
    require(h_after.data == h_before, "transform modified the trained H")
    for r, i in valid_rows {
      let row = matrix_row(w_new, r)
      let dominant = argmax(row)
      println("New document \{i} -> Topic \{dominant}")
      print_activations(row)
    }
  }
  println("")
  println("========================================")
  println("Topic Modeling Completed (all self-checks passed)")
  println("========================================")
}
