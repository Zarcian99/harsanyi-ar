"""
Role of the intercept kappa (Section "Estimating the intercept"): fidelity with and without
kappa, on synthetic functions whose inputs are Gaussian or standardised log-normal (skewed),
independent or pairwise correlated (rho = 0.7, Gaussian copula), and on California.

kappa is the mean of the residual after the correction, eq. (ap_mejora_kappa): the mean of
the discarded dividends over the data plus the baseline values of their lower-order
components. It is zero when the discarded part averages to its value at the baseline, as
for x1 x2 x3 with inputs of mean zero, and not when that part is even or convex, as for
cos x1 cos x2 cos x3 or (x1 x2 x3)^2; skewed or dependent inputs alone do not make it
large (log-normal rows). The loss of fidelity without kappa is about kappa^2 / Var f, so
the table also reports kappa / sd(f).

    python experiments/ablation_kappa.py
"""
from __future__ import annotations
import warnings
import numpy as np

warnings.simplefilter("ignore")
from common import Tee, N_BINS, TAU2, SEEDS, load, fit_subsample, train_mlp, NFIT, XGB
from harsanyi_ar import HarsanyiARSurrogate
from table2_synthetic import f_net, D, N_BG, N_FIT, N_TEST

LN_MEAN, LN_SD = np.exp(0.5), np.sqrt((np.e - 1.0) * np.e)     # moments of exp(Z), Z ~ N(0, 1)


def sample(rng, n, rho, lognormal):
    Z = rng.normal(size=(n, D))
    if rho:
        C = np.full((D, D), rho); np.fill_diagonal(C, 1.0)
        Z = Z @ np.linalg.cholesky(C).T
    return (np.exp(Z) - LN_MEAN) / LN_SD if lognormal else Z    # mean 0, variance 1 marginals


FUNCTIONS = [("order-3 monomial", lambda X: X[:, 0] * X[:, 1] * X[:, 2]),
             ("softplus neural network", f_net),
             ("order-3 cosine", lambda X: np.cos(X[:, 0]) * np.cos(X[:, 1]) * np.cos(X[:, 2])),
             ("even order 3", lambda X: X[:, 0] ** 2 * X[:, 1] ** 2 * X[:, 2] ** 2)]
INPUTS = [("Gaussian, independent", 0.0, False), ("Gaussian, rho = 0.7", 0.7, False),
          ("log-normal, independent", 0.0, True), ("log-normal, rho = 0.7", 0.7, True)]


def synthetic_row(f, rho, lognormal):
    rel, without, with_ = [], [], []
    for s in SEEDS:
        rng = np.random.RandomState(s)
        X_bg, X_test = sample(rng, N_BG, rho, lognormal), sample(rng, N_TEST, rho, lognormal)
        Xfit = X_bg[rng.choice(N_BG, size=N_FIT, replace=False)]
        g = lambda Z: f(np.asarray(Z, float))
        sur = HarsanyiARSurrogate(g, np.zeros(D), N_BINS, TAU2).fit(Xfit)
        rel.append(sur.kappa / np.std(g(X_test)))
        with_.append(sur.fidelity(X_test))
        sur.kappa = 0.0
        without.append(sur.fidelity(X_test))
    return np.mean(rel), np.mean(without), np.std(without), np.mean(with_), np.std(with_)


def california_row(which):
    sp = load("california")
    rel, without, with_ = [], [], []
    for seed in SEEDS:
        if which == "xgb":
            import xgboost as xgb
            m = xgb.XGBRegressor(random_state=seed, **XGB["california"]).fit(sp.Ztr, sp.ytr)
            pred = lambda A, m=m: m.predict(np.asarray(A, np.float32))
        else:
            from harsanyi_ar import TorchModel
            pred = TorchModel(train_mlp(sp, "california", "softplus", seed), sp.d).predict
        Xfit = sp.Ztr[fit_subsample(len(sp.Ztr), NFIT, seed)]
        sur = HarsanyiARSurrogate(pred, np.zeros(sp.d), N_BINS, TAU2).fit(Xfit)
        rel.append(sur.kappa / np.std(pred(sp.Zte)))
        with_.append(sur.fidelity(sp.Zte))
        sur.kappa = 0.0
        without.append(sur.fidelity(sp.Zte))
    return np.mean(rel), np.mean(without), np.std(without), np.mean(with_), np.std(with_)


if __name__ == "__main__":
    out = Tee("ablation_kappa.txt")
    out(f"seeds = {list(SEEDS)}; synthetic: {N_FIT} fitting points, {N_TEST} test points; "
        f"California: {NFIT} fitting rows, test set")
    out(f"{'function':26s} {'inputs':26s} {'kappa/sd(f)':>12s} {'without kappa':>18s} {'with kappa':>18s}")
    for fname, f in FUNCTIONS:
        for iname, rho, ln in INPUTS:
            r = synthetic_row(f, rho, ln)
            out(f"{fname:26s} {iname:26s} {r[0]:>+12.3f} {r[1]:>9.3f} +- {r[2]:.3f} {r[3]:>9.3f} +- {r[4]:.3f}")
    for which, label in (("xgb", "California, XGBoost"), ("net", "California, network")):
        r = california_row(which)
        out(f"{label:26s} {'data':26s} {r[0]:>+12.3f} {r[1]:>9.3f} +- {r[2]:.3f} {r[3]:>9.3f} +- {r[4]:.3f}")
