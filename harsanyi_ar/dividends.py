"""
Harsanyi dividends of the masking game, up to order 2. Section "Local attribution
and Harsanyi dividends" of the paper.

For an instance x and a coalition S, the masked point x_[S] keeps the coordinates in S
and sets the rest to the baseline, eq. (enmascarado). With v_x(S) = f(x_[S]), eq. (vS),
the dividends are, eq. (dividendo),

    d_i(x_i)       = f(x_[i])  - f(baseline),
    d_ij(x_i, x_j) = f(x_[ij]) - f(x_[i]) - f(x_[j]) + f(baseline).

They are computed exactly, from 1 + d + d(d-1)/2 evaluations of the model per
instance. The model only has to expose predict: (N, d) array -> (N,) array.
"""
from __future__ import annotations
from typing import Callable, List, Tuple
import numpy as np

from .numerics import sum_terms

Pair = Tuple[int, int]
Predict = Callable[[np.ndarray], np.ndarray]


def all_pairs(d: int) -> List[Pair]:
    """Ordered list of pairs (i, j), i < j."""
    return [(i, j) for i in range(d) for j in range(i + 1, d)]


def _call(predict: Predict, X: np.ndarray) -> np.ndarray:
    return np.asarray(predict(X), dtype=float).ravel()


MAX_ROWS = 2_000_000      # rows per model call when evaluating the masked points


def compute_dividends(predict: Predict, X: np.ndarray, baseline: np.ndarray):
    """
    Returns
    -------
    f0    : float, f(baseline)
    d1    : (N, d) order-one dividends d_i(x_i)
    d2    : (N, P) order-two dividends d_ij(x_i, x_j), P = d(d-1)/2
    pairs : list of pairs, in the column order of d2
    """
    X = np.asarray(X, float)
    N, d = X.shape
    pairs = all_pairs(d)
    P = len(pairs)
    base = np.asarray(baseline, float)[None, :]
    f0 = float(_call(predict, base)[0])

    keep1 = np.eye(d)
    keep2 = np.zeros((P, d))
    for p, (i, j) in enumerate(pairs):
        keep2[p, i] = keep2[p, j] = 1.0
    keep = np.vstack([keep1, keep2])                                   # (d+P, d)

    # masked points x_[T] for every set T, evaluated in batches of at most MAX_ROWS rows so
    # that memory stays bounded for large d (the batches do not change the values)
    sets_per_batch = max(1, MAX_ROWS // N)
    vals = np.empty((d + P) * N)
    for s0 in range(0, d + P, sets_per_batch):
        k = keep[s0:s0 + sets_per_batch][:, None, :]                   # (b, 1, d)
        masked = X[None, :, :] * k + base[None, :, :] * (1.0 - k)      # (b, N, d)
        vals[s0 * N:(s0 + len(k)) * N] = _call(predict, masked.reshape(-1, d))
    f1 = vals[:d * N].reshape(d, N)
    f2 = vals[d * N:].reshape(P, N)

    d1 = (f1 - f0).T
    ii = np.array([i for i, _ in pairs], dtype=int)
    jj = np.array([j for _, j in pairs], dtype=int)
    d2 = (f2 - f1[ii] - f1[jj] + f0).T if P else np.zeros((N, 0))
    return f0, d1, d2, pairs


def truncated_prediction(predict: Predict, X: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """Order-2 truncation f^k(x) = f(baseline) + sum_i d_i + sum_ij d_ij, eq. (truncamiento)."""
    X = np.asarray(X, float)
    f0, d1, d2, _ = compute_dividends(predict, X, baseline)
    terms = [np.full(X.shape[0], f0)]
    terms += [d1[:, i] for i in range(d1.shape[1])]
    terms += [d2[:, p] for p in range(d2.shape[1])]
    return sum_terms(terms)
