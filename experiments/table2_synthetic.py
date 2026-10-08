"""
Table "Reconstruction in distribution" (tab:ar): what the residual correction recovers.

R^2 of the order-2 surrogate against the true function on 5000 test samples, with the
dividends alone and with kappa + dividends + AR; mean over five seeds. Inputs are
standard Gaussian in R^4, independent or pairwise equicorrelated (rho = 0.7). The
"black box" is the synthetic function itself (ALE route, 4000 fitting points out of a
background sample of 20000, twenty quantile bins, no shrinkage, baseline 0).

    python experiments/table2_synthetic.py
"""
from __future__ import annotations
import numpy as np
from scipy.stats import norm

from common import Tee, N_BINS, TAU2
from harsanyi_ar import HarsanyiARSurrogate

D, N_BG, N_FIT, N_TEST, SEEDS = 4, 20_000, 4_000, 5_000, range(5)


def sample(rng: np.random.RandomState, n: int, rho: float) -> np.ndarray:
    """N(0, 1) marginals; Corr(X_i, X_j) = rho for i != j."""
    Z = rng.normal(size=(n, D))
    if rho == 0.0:
        return Z
    C = np.full((D, D), rho)
    np.fill_diagonal(C, 1.0)
    return Z @ np.linalg.cholesky(C).T


# softplus network with fixed (untrained) weights, eq. (f_red)
_rw = np.random.RandomState(7)
_W = _rw.normal(scale=6.0, size=(6, D))
_B = _rw.normal(scale=0.5, size=6)
_V = _rw.normal(scale=1.0, size=6)


def f_net(X):
    return np.logaddexp(0.0, X @ _W.T + _B) @ _V


def q(x):
    """Midpoint quantile of the octile that contains x (x standard normal)."""
    return norm.ppf((2.0 * np.ceil(8.0 * norm.cdf(x)) - 1.0) / 16.0)


def f_step(X):
    """Eq. (f_esc): constant on each of the 8^3 boxes, order 3, independent of x_4."""
    t1, t2, t3 = q(X[:, 0]), q(X[:, 1]), q(X[:, 2])
    return t1 + 0.8 * t2 + 0.7 * t3 + 0.9 * t1 * t2 + t1 * t2 * t3 ** 2


ROWS = [
    ("pure order 2",            lambda X: X[:, 0] * X[:, 1] + X[:, 2] * X[:, 3],                       [0.0]),
    ("order-3 cosine",          lambda X: np.cos(X[:, 0]) * np.cos(X[:, 1]) * np.cos(X[:, 2]),         [0.0]),
    ("order-3 monomial",        lambda X: X[:, 0] * X[:, 1] * X[:, 2],                                 [0.0, 0.7]),
    ("additive + order 3",      lambda X: 2 * (X[:, 0] + X[:, 1] + X[:, 2]) + X[:, 0] * X[:, 1] * X[:, 2], [0.0, 0.7]),
    ("softplus neural network", f_net,                                                                 [0.0]),
    ("step function",           f_step,                                                                [0.0]),
    ("additive + order 2",      lambda X: X[:, 0] + X[:, 1] + X[:, 0] * X[:, 1],                       [0.7]),
]


def evaluate(f, rho: float):
    div, full = [], []
    for s in SEEDS:
        rng = np.random.RandomState(s)
        X_bg, X_test = sample(rng, N_BG, rho), sample(rng, N_TEST, rho)
        Xfit = X_bg[rng.choice(N_BG, size=N_FIT, replace=False)]
        sur = HarsanyiARSurrogate(lambda Z: f(np.asarray(Z, float)), np.zeros(D), N_BINS, TAU2).fit(Xfit)
        div.append(sur.fidelity_dividends_only(X_test))
        full.append(sur.fidelity(X_test))
    return np.mean(div), np.std(div), np.mean(full), np.std(full)


if __name__ == "__main__":
    out = Tee("table2_synthetic.txt")
    for rho in (0.0, 0.7):
        out(f"\n== {'independent' if rho == 0.0 else f'correlated (rho = {rho} pairwise)'} ==")
        out(f"{'function':26s} {'dividends only':>18s} {'kappa + dividends + AR':>26s}")
        for name, f, regimes in ROWS:
            if rho in regimes:
                dm, ds, fm, fs = evaluate(f, rho)
                dm = 0.0 if abs(dm) < 5e-4 else dm          # print -0.000 as 0.000
                out(f"{name:26s} {dm:10.3f} +- {ds:.3f} {fm:16.3f} +- {fs:.3f}")
