"""
Does shrinkage flatten the estimated corrections in the tails? Shrinkage multiplies each
increment by gamma_C before it is accumulated, and the outer cells hold few observations,
so the estimated curve could lose range where the data are scarce.

Function f = cos x1 cos x2 cos x3 on independent standard Gaussian inputs in R^4, whose
corrections are known in closed form. With mu = E[cos X] - 1 and c_j = cos x_j, the residual
of the order-2 truncation is r = (c1 - 1)(c2 - 1)(c3 - 1), its ANOVA components are
w_1 = mu^2 (c1 - 1 - mu) and w_12 = mu (c1 - 1 - mu)(c2 - 1 - mu), and the corrections,
eq. (ar), are AR_1 = w_1 - w_1(0) = mu^2 (c1 - 1) and AR_12 = mu [(c1 - 1 - mu)(c2 - 1 - mu) - mu^2].

For fitting samples of 200 and 4000 points, with and without shrinkage, five seeds, the
error of the estimated AR_1 and AR_12 against the true ones is averaged over 5000 test
points, in the outer deciles of x1 against deciles 3 to 8 (order one), and in the corners where
both x1 and x2 lie in their outer deciles against the region where both lie in deciles 3 to 8
(order two).

    python experiments/tails_check.py
"""
from __future__ import annotations
import warnings
import numpy as np
from scipy.stats import norm

warnings.simplefilter("ignore")
from common import Tee, TAU2, SEEDS
from harsanyi_ar import HarsanyiARSurrogate

D, N_TEST = 4, 5000
MU = np.exp(-0.5) - 1.0
f = lambda X: np.cos(X[:, 0]) * np.cos(X[:, 1]) * np.cos(X[:, 2])
ar1_true = lambda x: MU ** 2 * (np.cos(x) - 1.0)
ar12_true = lambda x, y: MU * ((np.cos(x) - 1.0 - MU) * (np.cos(y) - 1.0 - MU) - MU ** 2)
DECILE_EDGES = norm.ppf(np.linspace(0, 1, 11))


def decile(x):
    return np.clip(np.searchsorted(DECILE_EDGES, x) - 1, 0, 9)


if __name__ == "__main__":
    out = Tee("tails_check.txt")
    out(f"seeds = {list(SEEDS)}; f = cos x1 cos x2 cos x3, independent N(0,1) inputs; error = estimated - true, "
        f"averaged over {N_TEST} test points; true AR_1 ranges from {ar1_true(0):.3f} at 0 to {ar1_true(np.pi):.3f} at pi")
    out(f"\n{'n':>5s} {'shrink':>7s} | {'order 1: decile 1':>18s} {'deciles 3-8':>12s} {'decile 10':>10s} {'rmse':>7s} | "
        f"{'order 2: corners':>17s} {'centre':>8s} {'rmse':>7s}")
    for n_fit in (200, 4000):
        for shrink in (True, False):
            e1 = {k: [] for k in ("d1", "mid", "d10", "rmse")}; e2 = {k: [] for k in ("corner", "centre", "rmse")}
            for seed in SEEDS:
                rng = np.random.RandomState(seed)
                Xfit, Xte = rng.normal(size=(n_fit, D)), rng.normal(size=(N_TEST, D))
                sur = HarsanyiARSurrogate(f, np.zeros(D), None, TAU2, shrink).fit(Xfit)
                err1 = sur.ar1[0](Xte[:, 0]) - ar1_true(Xte[:, 0])
                dec = decile(Xte[:, 0])
                e1["d1"].append(err1[dec == 0].mean()); e1["mid"].append(err1[(dec >= 2) & (dec <= 7)].mean())
                e1["d10"].append(err1[dec == 9].mean()); e1["rmse"].append(np.sqrt((err1 ** 2).mean()))
                err2 = sur.ar2[(0, 1)](Xte) - ar12_true(Xte[:, 0], Xte[:, 1])
                d0, d1 = decile(Xte[:, 0]), decile(Xte[:, 1])
                corner = ((d0 == 0) | (d0 == 9)) & ((d1 == 0) | (d1 == 9))
                centre = (d0 >= 2) & (d0 <= 7) & (d1 >= 2) & (d1 <= 7)
                e2["corner"].append(err2[corner].mean()); e2["centre"].append(err2[centre].mean())
                e2["rmse"].append(np.sqrt((err2 ** 2).mean()))
            m = lambda v: np.mean(v)
            out(f"{n_fit:>5d} {'on' if shrink else 'off':>7s} | {m(e1['d1']):>+18.4f} {m(e1['mid']):>+12.4f} "
                f"{m(e1['d10']):>+10.4f} {m(e1['rmse']):>7.4f} | {m(e2['corner']):>+17.4f} {m(e2['centre']):>+8.4f} "
                f"{m(e2['rmse']):>7.4f}")
    out(f"\nreference: true AR_1 mean in decile 1 and 10 = {ar1_true(norm.ppf(0.05)):.4f}, "
        f"true AR_12 at the corners (|x| = 1.64) = {ar12_true(1.64, 1.64):.4f}, at the centre (0.5, 0.5) = {ar12_true(0.5, 0.5):.4f}")
