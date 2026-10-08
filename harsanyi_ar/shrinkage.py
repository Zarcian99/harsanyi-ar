"""
Empirical-Bayes (zero-mean Fay-Herriot) shrinkage of the ALE cell increments.
Section "Stability in cells with few data" of the paper, eq. (gamma).

Each cell C contributes an increment Delta_C, the mean over its points of the
difference of the residual across the cell, modelled as a noisy measurement of the
true increment with sampling variance

    Var(Delta_C) = v / |N(C)|,

where v is the pooled within-cell variance and |N(C)| the number of points in the
cell. What is accumulated is gamma_C * Delta_C, with

    gamma_C = tau^2 / (tau^2 + Var(Delta_C)),

tau^2 being the between-cell variance. Only the residual correction is shrunk; the
Harsanyi dividends are never modified.

tau^2 estimators: 'mom' (moments), 'ml' (maximum likelihood) and 'adm' (adjusted
maximum likelihood; Li & Lahiri 2010, Yoshimori & Lahiri 2014). 'adm' is strictly
positive and is the one used in the paper.
"""
from __future__ import annotations
import numpy as np


def _neg_loglik(A: float, D: np.ndarray, V: np.ndarray) -> float:
    """Negative log-likelihood (up to constants) of D_c ~ N(0, A + V_c)."""
    s = A + V
    return 0.5 * float(np.sum(np.log(s) + D ** 2 / s))


def _solve_tau2(D: np.ndarray, V: np.ndarray, adjusted: bool) -> float:
    """Minimise over A = tau^2 >= 0 by log-spaced grid search plus ternary refinement.
    With adjusted=True the likelihood is multiplied by A, which rules out A = 0."""
    grid = np.concatenate([[0.0], np.geomspace(1e-8, 1e3, 300)])
    pen = (lambda A: (np.log(A) if (adjusted and A > 0) else 0.0))
    val = np.array([_neg_loglik(A, D, V) - pen(A) for A in grid])
    if adjusted:
        val[0] = np.inf
    A0 = grid[int(np.argmin(val))]
    lo, hi = max(A0 * 0.5, 0.0), A0 * 2 + 1e-9
    for _ in range(40):
        a = lo + (hi - lo) / 3
        b = hi - (hi - lo) / 3
        fa = _neg_loglik(a, D, V) - pen(a)
        fb = _neg_loglik(b, D, V) - pen(b)
        if fa < fb:
            hi = b
        else:
            lo = a
    return 0.5 * (lo + hi)


def estimate_tau2(Delta, VarD, method: str = "adm") -> float:
    """Between-cell variance tau^2 by 'mom', 'ml' or 'adm'."""
    D = np.asarray(Delta, float).ravel()
    V = np.asarray(VarD, float).ravel()
    if D.size == 0:
        return 0.0
    if method == "mom":
        w = 1.0 / np.maximum(V, 1e-30)
        sw = w.sum()
        return max(0.0, (w * D ** 2).sum() / sw - (w * V).sum() / sw)
    if method in ("ml", "adm"):
        return _solve_tau2(D, V, adjusted=(method == "adm"))
    raise ValueError(f"unknown tau^2 method: {method!r}")


def pooled_variance(values: np.ndarray, cell_mean_at_point: np.ndarray,
                    count_at_point: np.ndarray) -> float:
    """Pooled within-cell variance v over the cells with at least two points
    (falls back to the global variance when fewer than five points qualify)."""
    res = values - cell_mean_at_point
    big = count_at_point >= 2
    v = float((res[big] ** 2).mean()) if big.sum() >= 5 else float(np.var(values))
    return max(v, 1e-12)


def shrink_increments(cell_mean: np.ndarray, count: np.ndarray, pooled_var: float,
                      tau2_method: str = "adm"):
    """
    Shrink the cell increments Delta_C = cell_mean (elementwise, so it serves 1-D and
    2-D grids). Returns (gamma * Delta, gamma, tau2). Empty cells get gamma = 0.
    """
    ne = count > 0
    Delta = cell_mean
    VarD = pooled_var / np.maximum(count, 1.0)
    tau2 = estimate_tau2(Delta[ne], VarD[ne], tau2_method)
    gamma = np.where(ne, tau2 / (tau2 + VarD + 1e-30), 0.0)
    return gamma * Delta, gamma, tau2
