"""
Numerical utilities: compensated summation and an exactly-summed R^2.

The surrogate adds 1 + d + d(d-1)/2 pieces per instance and the ALE curves add one
increment per grid cell. Compensated (Neumaier) summation keeps the rounding error
independent of the number of summands (appendix "Compensated summation" of the paper).
"""
from __future__ import annotations
import math
import numpy as np


def fsum_1d(a) -> float:
    """Correctly rounded sum of a 1-D array (math.fsum)."""
    return math.fsum(np.asarray(a, dtype=float).ravel().tolist())


def mean_1d(a) -> float:
    """Correctly rounded mean of a 1-D array."""
    a = np.asarray(a, dtype=float).ravel()
    return fsum_1d(a) / a.size


def sum_terms(terms) -> np.ndarray:
    """Neumaier-compensated elementwise sum of a list of equally shaped arrays."""
    if len(terms) == 0:
        raise ValueError("sum_terms received an empty list")
    s = np.array(terms[0], dtype=np.float64, copy=True)
    c = np.zeros_like(s)
    for x in terms[1:]:
        x = np.asarray(x, dtype=np.float64)
        t = s + x
        big_s = np.abs(s) >= np.abs(x)
        c += np.where(big_s, (s - t) + x, (x - t) + s)
        s = t
    return s + c


def cumsum0(incr) -> np.ndarray:
    """Compensated cumulative sum with a leading zero: (B,) increments -> (B+1,) nodes."""
    incr = np.asarray(incr, dtype=np.float64).ravel()
    out = np.empty(incr.size + 1, dtype=np.float64)
    out[0] = 0.0
    s = 0.0
    c = 0.0
    for k in range(incr.size):
        x = float(incr[k])
        t = s + x
        if abs(s) >= abs(x):
            c += (s - t) + x
        else:
            c += (x - t) + s
        s = t
        out[k + 1] = s + c
    return out


def cumsum2d_nodes(incr) -> np.ndarray:
    """Compensated double cumulative sum: (Bi, Bj) cell increments -> (Bi+1, Bj+1) nodes."""
    incr = np.asarray(incr, dtype=np.float64)
    bi, bj = incr.shape
    col_nodes = np.zeros((bi + 1, bj))
    for b in range(bj):
        col_nodes[:, b] = cumsum0(incr[:, b])
    nodes = np.zeros((bi + 1, bj + 1))
    for a in range(bi + 1):
        nodes[a, :] = cumsum0(col_nodes[a, :])
    return nodes


def r2_score(y_true, y_pred) -> float:
    """Coefficient of determination with exactly rounded sums of squares."""
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    mean = fsum_1d(yt) / yt.size
    ss_res = fsum_1d((yt - yp) ** 2)
    ss_tot = fsum_1d((yt - mean) ** 2)
    return 1.0 - ss_res / ss_tot
