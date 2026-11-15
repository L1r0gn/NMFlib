///|
/// Dense row-major matrix backed by a flat `Array[Double]`.
///
/// `Matrix` is a general numerical container: entries may be any finite
/// double (negatives included). Non-negativity is enforced only at the NMF
/// input boundary (`NMF::fit` / `NMF::transform`), not here.
///
/// Creation rejects negative dimensions, `rows * cols` overflow, data-length
/// mismatches, and NaN/Infinity entries. Zero-sized dimensions are allowed
/// (with empty data); NMF fitting rejects empty matrices separately.
pub struct Matrix {
  data : Array[Double]
  rows : Int32
  cols : Int32
}

///|
/// Checks dimensions and the `rows * cols <= Int::MAX` bound.
fn validate_shape(rows : Int32, cols : Int32) -> Result[Unit, NMFError] {
  if rows < 0 || cols < 0 {
    return Err(InvalidMatrixShape("negative dimensions: \{rows}x\{cols}"))
  }
  if rows.to_int64() * cols.to_int64() > 2147483647L {
    return Err(InvalidMatrixShape("rows*cols overflows Int: \{rows}x\{cols}"))
  }
  Ok(())
}

///|
/// Creates a matrix from row-major data, validating shape, length, and
/// finiteness of every entry.
pub fn Matrix::new(
  rows : Int32,
  cols : Int32,
  data : Array[Double],
) -> Result[Matrix, NMFError] {
  match validate_shape(rows, cols) {
    Err(e) => return Err(e)
    Ok(_) => ()
  }
  let size = rows * cols
  if data.length() != size {
    return Err(
      DimensionMismatch(
        "data length \{data.length()} does not match \{rows}x\{cols}=\{size}",
      ),
    )
  }
  for idx in 0..<size {
    let x = data[idx]
    if x.is_nan() {
      return Err(NumericalError("NaN entry at flat index \{idx}"))
    }
    if x.is_inf() {
      return Err(NumericalError("infinite entry at flat index \{idx}"))
    }
  }
  Ok({ data, rows, cols, })
}

///|
/// Creates a zero matrix with the given dimensions.
pub fn Matrix::zeros(rows : Int32, cols : Int32) -> Result[Matrix, NMFError] {
  match validate_shape(rows, cols) {
    Err(e) => return Err(e)
    Ok(_) => ()
  }
  Ok({ data: Array::make(rows * cols, 0.0), rows, cols, })
}

///|
/// Number of rows.
pub fn Matrix::rows(self : Matrix) -> Int32 {
  self.rows
}

///|
/// Number of columns.
pub fn Matrix::cols(self : Matrix) -> Int32 {
  self.cols
}

///|
/// Reads entry `(i, j)`; out-of-bounds indices return `DimensionMismatch`.
pub fn Matrix::get(
  self : Matrix,
  i : Int32,
  j : Int32,
) -> Result[Double, NMFError] {
  if i < 0 || i >= self.rows || j < 0 || j >= self.cols {
    return Err(
      DimensionMismatch(
        "index (\{i},\{j}) out of bounds for \{self.rows}x\{self.cols}",
      ),
    )
  }
  Ok(self.data[i * self.cols + j])
}

///|
/// Writes entry `(i, j)`. Rejects out-of-bounds indices and NaN/Infinity.
pub fn Matrix::set(
  self : Matrix,
  i : Int32,
  j : Int32,
  v : Double,
) -> Result[Unit, NMFError] {
  if i < 0 || i >= self.rows || j < 0 || j >= self.cols {
    return Err(
      DimensionMismatch(
        "index (\{i},\{j}) out of bounds for \{self.rows}x\{self.cols}",
      ),
    )
  }
  if v.is_nan() {
    return Err(NumericalError("cannot set NaN at (\{i},\{j})"))
  }
  if v.is_inf() {
    return Err(NumericalError("cannot set Infinity at (\{i},\{j})"))
  }
  self.data[i * self.cols + j] = v
  Ok(())
}

///|
/// Defensive copy (public accessors return copies so callers cannot mutate
/// model internals through `set`).
fn Matrix::clone(self : Matrix) -> Matrix {
  { data: self.data.copy(), rows: self.rows, cols: self.cols, }
}

///|
/// Unchecked internal zero buffer. Callers guarantee valid positive dims.
fn zeros_buf(rows : Int32, cols : Int32) -> Matrix {
  { data: Array::make(rows * cols, 0.0), rows, cols, }
}

///|
/// True when every entry is finite (no NaN, no Infinity).
fn all_finite(data : Array[Double]) -> Bool {
  for x in data {
    if x.is_nan() || x.is_inf() {
      return false
    }
  }
  true
}

///|
/// Returns `Aᵀ` (no failure mode).
pub fn transpose(a : Matrix) -> Matrix {
  let out = Array::make(a.rows * a.cols, 0.0)
  for i in 0..<a.rows {
    for j in 0..<a.cols {
      out[j * a.rows + i] = a.data[i * a.cols + j]
    }
  }
  { data: out, rows: a.cols, cols: a.rows, }
}

///|
/// Unchecked `out = A · B`. Caller guarantees `a.cols == b.rows` and
/// `out` is `a.rows × b.cols`. Row-major i-p-j loop for cache locality.
fn matmul_into(a : Matrix, b : Matrix, out : Matrix) -> Unit {
  for idx in 0..<out.data.length() {
    out.data[idx] = 0.0
  }
  for i in 0..<a.rows {
    let a_row = i * a.cols
    let out_row = i * out.cols
    for p in 0..<a.cols {
      let a_ip = a.data[a_row + p]
      let b_row = p * b.cols
      for j in 0..<b.cols {
        out.data[out_row + j] = out.data[out_row + j] + a_ip * b.data[b_row + j]
      }
    }
  }
}

///|
/// Unchecked `out = Aᵀ · B`. Caller guarantees `a.rows == b.rows` and
/// `out` is `a.cols × b.cols`.
fn matmul_at_b_into(a : Matrix, b : Matrix, out : Matrix) -> Unit {
  for idx in 0..<out.data.length() {
    out.data[idx] = 0.0
  }
  for i in 0..<a.rows {
    let a_row = i * a.cols
    let b_row = i * b.cols
    for r in 0..<a.cols {
      let a_ir = a.data[a_row + r]
      let out_row = r * out.cols
      for j in 0..<b.cols {
        out.data[out_row + j] = out.data[out_row + j] + a_ir * b.data[b_row + j]
      }
    }
  }
}

///|
/// Unchecked `out = A · Bᵀ`. Caller guarantees `a.cols == b.cols` and
/// `out` is `a.rows × b.rows`. Both operands are read row-wise.
fn matmul_a_bt_into(a : Matrix, b : Matrix, out : Matrix) -> Unit {
  for i in 0..<a.rows {
    let a_row = i * a.cols
    let out_row = i * out.cols
    for j in 0..<b.rows {
      let b_row = j * b.cols
      let mut s = 0.0
      for p in 0..<a.cols {
        s = s + a.data[a_row + p] * b.data[b_row + p]
      }
      out.data[out_row + j] = s
    }
  }
}

///|
/// Checked `A · B`; inner dimensions must match.
pub fn matmul(a : Matrix, b : Matrix) -> Result[Matrix, NMFError] {
  if a.cols != b.rows {
    return Err(
      DimensionMismatch(
        "matmul inner dims: \{a.rows}x\{a.cols} · \{b.rows}x\{b.cols}",
      ),
    )
  }
  let out = {
    data: Array::make(a.rows * b.cols, 0.0),
    rows: a.rows,
    cols: b.cols,
  }
  matmul_into(a, b, out)
  Ok(out)
}

///|
/// Checked `Aᵀ · B`; the row counts of `a` and `b` must match.
pub fn matmul_at_b(a : Matrix, b : Matrix) -> Result[Matrix, NMFError] {
  if a.rows != b.rows {
    return Err(
      DimensionMismatch(
        "matmul_at_b row dims: \{a.rows}x\{a.cols}ᵀ · \{b.rows}x\{b.cols}",
      ),
    )
  }
  let out = {
    data: Array::make(a.cols * b.cols, 0.0),
    rows: a.cols,
    cols: b.cols,
  }
  matmul_at_b_into(a, b, out)
  Ok(out)
}

///|
/// Checked `A · Bᵀ`; the column counts of `a` and `b` must match.
pub fn matmul_a_bt(a : Matrix, b : Matrix) -> Result[Matrix, NMFError] {
  if a.cols != b.cols {
    return Err(
      DimensionMismatch(
        "matmul_a_bt col dims: \{a.rows}x\{a.cols} · \{b.rows}x\{b.cols}ᵀ",
      ),
    )
  }
  let out = {
    data: Array::make(a.rows * b.rows, 0.0),
    rows: a.rows,
    cols: b.rows,
  }
  matmul_a_bt_into(a, b, out)
  Ok(out)
}

///|
/// Frobenius norm `sqrt(Σ x²)`. Returns `NumericalError` if the accumulated
/// sum of squares overflows to a non-finite value.
pub fn frobenius_norm(a : Matrix) -> Result[Double, NMFError] {
  let mut s = 0.0
  for x in a.data {
    s = s + x * x
  }
  if s.is_nan() || s.is_inf() {
    return Err(
      NumericalError("Frobenius norm overflowed to a non-finite value"),
    )
  }
  Ok(s.sqrt())
}
