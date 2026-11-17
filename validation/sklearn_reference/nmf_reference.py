"""scikit-learn NMF reference wrapper for T16.

Responsibilities:
  * version recording (no assumptions about the installed sklearn);
  * solver / loss / init mapping to NMFlib's three valid combinations;
  * L1/L2 regularization mapping with alpha_H ALWAYS set explicitly
    (never sklearn's default "same", which is wrong for non-square inputs);
  * fit + honest metadata extraction (sklearn's NMF exposes n_iter_ but NO
    converged attribute; we report that fact instead of inventing a value).

Regularization mapping (verified by hand in selftest.py before any batch
experiment). NMFlib optimizes
    J = D(V||WH) + l1*(sum W + sum H) + (l2/2)*(||W||_F^2 + ||H||_F^2)
sklearn 1.7.2 optimizes (source: _compute_regularization + class docstring)
    J = D(X||WH) + alpha_W*r*n*||W||_1 + alpha_H*r*m*||H||_1
        + (alpha_W*(1-r)*n/2)*||W||_F^2 + (alpha_H*(1-r)*m/2)*||H||_F^2
with m=n_samples, n=n_features. Matching coefficients gives
    r = l1/(l1+l2),  alpha_W = (l1+l2)/n,  alpha_H = (l1+l2)/m
and for l1=l2=0: alpha_W=alpha_H=l1_ratio=0.
"""

from __future__ import annotations

import platform
import sys
import time

import numpy as np
import scipy
import sklearn
from sklearn.decomposition import NMF

SOLVER_MAP = {"MU": "mu", "CD": "cd"}
LOSS_MAP = {"frobenius": "frobenius", "kl": "kullback-leibler"}
INIT_MAP = {"Random": "random", "NNDSVD": "nndsvd", "NNDSVDa": "nndsvda"}

# Documented behavioral differences between the two implementations,
# recorded from the installed sklearn 1.7.2 source (sklearn/decomposition/
# _nmf.py) and NMFlib src/solver_mu.mbt, src/solver_cd.mbt, src/loss.mbt.
SEMANTIC_DIFFERENCES = {
    "stopping_nmflib": (
        "every iteration: |F_t - F_{t-1}| / max(1, |F_{t-1}|) < tol, where F "
        "is the full regularized objective (1/2||V-WH||_F^2 or KL, plus Omega)"
    ),
    "stopping_sklearn_cd": (
        "every iteration: KKT violation / violation_init <= tol "
        "(violation_init measured after iteration 1); break if violation_init==0"
    ),
    "stopping_sklearn_mu": (
        "every 10 iterations: (previous_error - error)/error_at_init < tol, "
        "error = ||X-WH||_F for frobenius, sqrt(2*KL) for kullback-leibler; "
        "regularization is NOT part of this monitored error"
    ),
    "update_order": (
        "NMFlib MU/CD update H first then W; sklearn MU updates W first then "
        "H (CD: W then H as well). Same alternating scheme, different path."
    ),
    "kl_zero_protection": (
        "sklearn MU-KL floors WH at EPSILON=1.1920929e-07 (float32 eps) inside "
        "the V/WH ratio, maps zero W column sums to 1.0 and zero denominators "
        "to EPSILON, and zeroes H entries below float64 eps after updates; "
        "V>0 with WH=0 therefore yields a large finite ratio instead of an "
        "error. NMFlib returns NumericalError for V>0 and WH=0, normalizes "
        "zero-denominator components to 0, and never adds epsilon to the KL "
        "objective itself."
    ),
    "random_init": (
        "both use scale sqrt(mean(V)/k); sklearn draws |standard_normal| "
        "(half-normal, numpy RNG), NMFlib draws uniform(0,1) (SplitMix64). "
        "Same integer seed does NOT give the same initial W/H."
    ),
    "nndsvd": (
        "sklearn nndsvd/nndsvda use randomized SVD (depends on random_state); "
        "NMFlib uses deterministic internal power iteration + deflation "
        "(seed-independent). Element-wise equality is not expected."
    ),
    "converged_field": (
        "NMFlib exposes converged(); sklearn NMF has NO converged attribute, "
        "only n_iter_. sklearn rows record converged as empty and "
        "hit_max_iter = (n_iter_ >= max_iter)."
    ),
    "reconstruction_err_": (
        "sklearn reconstruction_err_ = _beta_divergence(X,W,H,beta_loss,"
        "square_root=True): ||X-WH||_F for frobenius loss but sqrt(2*KL) for "
        "kullback-leibler loss (both solvers). NMFlib reconstruction_error is "
        "ALWAYS ||V-WH||_F regardless of loss. Independent evaluation uses "
        "the Frobenius norm for both sides; sklearn's reported value is "
        "crosschecked against its own documented semantics."
    ),
}


def environment_info() -> dict:
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "platform": platform.platform(),
        "dtype": "float64",
    }


def map_regularization(l1_reg: float, l2_reg: float, m: int, n: int) -> dict:
    """Map NMFlib (l1_reg, l2_reg) to sklearn (alpha_W, alpha_H, l1_ratio).

    m = n_samples = V rows, n = n_features = V cols. alpha_H is always an
    explicit float, never "same".
    """
    if l1_reg < 0 or l2_reg < 0:
        raise ValueError("regularization strengths must be >= 0")
    total = l1_reg + l2_reg
    if total == 0.0:
        return {"alpha_W": 0.0, "alpha_H": 0.0, "l1_ratio": 0.0}
    return {
        "alpha_W": total / n,
        "alpha_H": total / m,
        "l1_ratio": l1_reg / total,
    }


def sklearn_penalty_from_params(
    w: np.ndarray,
    h: np.ndarray,
    alpha_W: float,
    alpha_H: float,
    l1_ratio: float,
    m: int,
    n: int,
) -> float:
    """sklearn's documented regularization term, computed independently.

    Used by selftest.py to cross-check the mapping against NMFlib's Omega
    on hand-built factors (no sklearn internals involved).
    """
    r = l1_ratio
    return float(
        alpha_W * r * n * np.sum(w)
        + alpha_H * r * m * np.sum(h)
        + 0.5 * alpha_W * (1.0 - r) * n * np.sum(w * w)
        + 0.5 * alpha_H * (1.0 - r) * m * np.sum(h * h)
    )


def fit_sklearn(
    v: np.ndarray,
    k: int,
    solver: str,
    loss: str,
    init: str,
    seed: int,
    l1_reg: float,
    l2_reg: float,
    max_iter: int,
    tol: float,
) -> dict:
    """Fit sklearn NMF with mapped parameters; return factors + metadata."""
    if solver == "CD" and loss == "kl":
        raise ValueError("CD+KL is unsupported in NMFlib (InvalidConfig)")
    m, n = v.shape
    reg = map_regularization(l1_reg, l2_reg, m, n)
    model = NMF(
        n_components=k,
        init=INIT_MAP[init],
        solver=SOLVER_MAP[solver],
        beta_loss=LOSS_MAP[loss],
        tol=tol,
        max_iter=max_iter,
        random_state=seed,
        alpha_W=reg["alpha_W"],
        alpha_H=reg["alpha_H"],
        l1_ratio=reg["l1_ratio"],
        shuffle=False,
    )
    t0 = time.perf_counter()
    # fit_transform returns the TRAINING W (m x k); components_ is H (k x n).
    # NMF.transform would re-solve W with H fixed, which is not what we want.
    w = np.asarray(model.fit_transform(v), dtype=np.float64)
    fit_seconds = time.perf_counter() - t0
    h = np.asarray(model.components_, dtype=np.float64)  # k x n
    return {
        "w": w,
        "h": h,
        "n_iter": int(model.n_iter_),
        # sklearn NMF provides no converged attribute; do not fabricate one.
        "converged": None,
        "hit_max_iter": bool(model.n_iter_ >= max_iter),
        "reported_reconstruction_err": float(model.reconstruction_err_),
        # NOTE: sqrt(beta-divergence), i.e. sqrt(2*KL) under KL loss.
        "fit_seconds": fit_seconds,
        "sklearn_params": reg,
    }