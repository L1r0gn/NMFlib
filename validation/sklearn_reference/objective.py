"""Independent NMF objective evaluators for T16 reference validation.

Pure NumPy implementations derived from the mathematical definitions in
NMFlib's `src/loss.mbt` (Frobenius with the 1/2 coefficient, generalized KL
with NMFlib's exact zero conventions, L1/L2 penalty). No sklearn internals
are imported or copied here: this module is the independent referee used to
re-evaluate the final W/H produced by BOTH implementations.

KL zero conventions (identical to NMFlib `kl_divergence`):
  V=0, WH=0 -> term 0
  V=0, WH>0 -> term WH
  V>0, WH=0 -> KLInfiniteError (the divergence is +infinity; never silently
               replaced by an epsilon-regularized finite value)
"""

from __future__ import annotations

import numpy as np


class KLInfiniteError(ValueError):
    """Raised when V>0 and WH=0: generalized KL is +infinity."""


def _check_nonneg_finite(name: str, a: np.ndarray) -> None:
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains NaN or Infinity")
    if np.any(a < 0.0):
        raise ValueError(f"{name} contains negative entries")


def frobenius_data_loss(v: np.ndarray, wh: np.ndarray) -> float:
    """L_F = 1/2 * ||V - WH||_F^2 (NMFlib `frobenius_objective`)."""
    _check_nonneg_finite("V", v)
    _check_nonneg_finite("WH", wh)
    d = v - wh
    return float(0.5 * np.sum(d * d))


def reconstruction_error(v: np.ndarray, wh: np.ndarray) -> float:
    """E_F = ||V - WH||_F (sklearn `reconstruction_err_` semantics)."""
    _check_nonneg_finite("V", v)
    _check_nonneg_finite("WH", wh)
    d = v - wh
    return float(np.sqrt(np.sum(d * d)))


def kl_divergence(v: np.ndarray, wh: np.ndarray) -> float:
    """D_KL(V||WH) = sum[V log(V/WH) - V + WH] with NMFlib zero conventions.

    Uses the log-difference form V*(log V - log WH) - V + WH on the strictly
    positive part, matching NMFlib's overflow-safe formulation. No epsilon
    is added anywhere.
    """
    _check_nonneg_finite("V", v)
    _check_nonneg_finite("WH", wh)
    if v.shape != wh.shape:
        raise ValueError(f"shape mismatch: V {v.shape} vs WH {wh.shape}")
    bad = (v > 0.0) & (wh == 0.0)
    if np.any(bad):
        i, j = np.argwhere(bad)[0]
        raise KLInfiniteError(f"KL is infinite: V>0 and WH=0 at ({i},{j})")
    total = float(np.sum(wh[v == 0.0]))  # V=0 rows contribute WH (0 if WH=0)
    pos = v > 0.0
    x = v[pos]
    y = wh[pos]
    terms = x * (np.log(x) - np.log(y)) - x + y
    return total + float(np.sum(terms))


def l1_penalty(w: np.ndarray, h: np.ndarray) -> float:
    """P1 = ||W||_1 + ||H||_1 (element sums, factors are non-negative)."""
    _check_nonneg_finite("W", w)
    _check_nonneg_finite("H", h)
    return float(np.sum(w) + np.sum(h))


def l2_penalty(w: np.ndarray, h: np.ndarray) -> float:
    """P2 = 1/2 * (||W||_F^2 + ||H||_F^2)."""
    _check_nonneg_finite("W", w)
    _check_nonneg_finite("H", h)
    return float(0.5 * (np.sum(w * w) + np.sum(h * h)))


def regularization_penalty(
    w: np.ndarray, h: np.ndarray, l1_reg: float, l2_reg: float
) -> float:
    """Omega(W,H) = l1_reg*P1 + l2_reg*P2 (NMFlib definition)."""
    return l1_reg * l1_penalty(w, h) + l2_reg * l2_penalty(w, h)


def data_loss(v: np.ndarray, wh: np.ndarray, loss: str) -> float:
    if loss == "frobenius":
        return frobenius_data_loss(v, wh)
    if loss == "kl":
        return kl_divergence(v, wh)
    raise ValueError(f"unknown loss: {loss}")


def full_objective(
    v: np.ndarray,
    w: np.ndarray,
    h: np.ndarray,
    l1_reg: float,
    l2_reg: float,
    loss: str,
) -> float:
    """J = D(V||WH) + l1_reg*P1 + l2_reg*P2, the shared optimization target."""
    wh = w @ h
    return data_loss(v, wh, loss) + regularization_penalty(w, h, l1_reg, l2_reg)


def factor_health(w: np.ndarray, h: np.ndarray) -> dict:
    """Non-negativity / finiteness audit of factors and their product."""
    wh = w @ h
    return {
        "w_finite": bool(np.all(np.isfinite(w))),
        "h_finite": bool(np.all(np.isfinite(h))),
        "wh_finite": bool(np.all(np.isfinite(wh))),
        "w_nonneg": bool(np.all(w >= 0.0)),
        "h_nonneg": bool(np.all(h >= 0.0)),
        "wh_nonneg": bool(np.all(wh >= 0.0)),
        "w_nnzs": int(np.count_nonzero(w)),
        "h_nnzs": int(np.count_nonzero(h)),
    }