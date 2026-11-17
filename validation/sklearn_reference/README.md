# T16: scikit-learn reference validation

Independent numerical cross-validation of NMFlib against
`sklearn.decomposition.NMF` (scikit-learn **1.7.2**). This directory is a
**validation tool only**: Python/sklearn are NOT production dependencies of
the MoonBit package, and no NMFlib public API was added or changed for it.

## What is compared

The shared mathematical target (both implementations are re-evaluated with
the independent `objective.py`, never with their own reported losses):

```
J(W,H) = D(V||WH) + l1*(sum W + sum H) + (l2/2)*(||W||_F^2 + ||H||_F^2)
D = 1/2*||V-WH||_F^2 (Frobenius)  or  sum[V log(V/WH) - V + WH] (KL)
```

Matrix direction: `V (m x n) = X (n_samples x n_features)`, `W: m x k`,
`H: k x n`. Regularization mapping (hand-verified in `selftest.py`):

```
r = l1/(l1+l2),  alpha_W = (l1+l2)/n_features,  alpha_H = (l1+l2)/n_samples
```

`alpha_H` is always set explicitly (never sklearn's default `"same"`, which
is wrong for non-square inputs). KL zero conventions follow NMFlib exactly:
`V=0,WH=0 -> 0`; `V=0,WH>0 -> WH`; `V>0,WH=0 -> error` (no epsilon anywhere).

Validated combinations: MU+Frobenius, CD+Frobenius, MU+KL.
CD+KL remains `InvalidConfig` in NMFlib and is not a reference combination.

## Known implementation differences (recorded, not "fixed")

See `nmf_reference.SEMANTIC_DIFFERENCES` and `results/metadata.json`:
different stopping criteria (NMFlib: relative change of the full regularized
objective every iteration; sklearn CD: KKT violation ratio; sklearn MU:
relative error change every 10 iterations on the UNregularized error),
different MU update order, sklearn's `EPSILON=1.19e-7` floor in MU-KL ratios
vs NMFlib's explicit `NumericalError` for `V>0, WH=0`, different random-init
distributions (same scale `sqrt(mean/k)`, different RNG), sklearn's
`reconstruction_err_` being sqrt(beta-divergence) (sqrt(2*KL) under KL loss)
while NMFlib's `reconstruction_error` is always the Frobenius norm, and
sklearn's lack of a `converged` attribute. Identical seeds do NOT imply
identical initial W/H; conclusions rely on multi-seed statistics and
independent objectives, never on element-wise W/H equality.

## Layout

| file | role |
|---|---|
| `objective.py` | independent Frobenius/KL/L1/L2/full-objective evaluators |
| `generate_data.py` | 4 fixed-seed datasets (A1/A2 low-rank, B noisy, C sparse counts with true rank 2 < configured k=4 and exact zeros) |
| `nmf_reference.py` | sklearn wrapper, parameter mapping, version recording |
| `moonbit_runner/` | standalone MoonBit module (local path dependency on NMFlib) fitting generated case literals via the public API; prints W/H + metadata CSV |
| `run_experiments.py` | orchestrates both sides, independent re-evaluation, results.csv + metadata.json |
| `analyze_results.py` | summary.csv, analysis.md, plots/ |
| `selftest.py` | hand-computed objective/mapping checks, KL boundaries, reproducibility, schema |

## Reproduce

```bash
conda create -n nmflib-t16 python=3.10 -y
conda activate nmflib-t16
pip install -r requirements.txt

python generate_data.py        # fixed datasets -> data/
python selftest.py             # hand-verification BEFORE batch experiments
python run_experiments.py      # full grid: 336 cases x 2 implementations
python analyze_results.py      # summary.csv, analysis.md, plots/

python run_experiments.py --smoke   # tiny subset used by CI
```

The MoonBit side requires the `moon` toolchain on PATH; the runner builds
against the repository via a local path dependency (`moonbit_runner/moon.mod`).

Full grid: 4 datasets x 3 combos x 4 regularization configs x 5 seeds
(11/23/42/77/101) with Random init = 240 cases, plus a deterministic
NNDSVD/NNDSVDa layer (4 x 3 x 4 x 2 = 96 cases), per implementation.
`max_iter=500`, `tol=1e-6` on both sides; actual stopping semantics differ
and are recorded per implementation.

Per-fit timing is not exposed by the NMFlib public API, so NMFlib rows carry
no `fit_seconds` (the runner's total wall time is in `metadata.json`);
sklearn rows carry real per-fit timings. No production code was modified to
obtain timing.

Reproducibility was verified empirically with two consecutive full-grid
runs: every `results.csv` column is bit-identical across runs **except**
`fit_seconds` (336 sklearn wall timings) and `moonbit_runner_wall_seconds`
in `metadata.json` — wall-clock measurements are inherently
nondeterministic. Datasets, parameters, final W/H, independently
re-evaluated objectives and crosscheck values reproduce exactly.