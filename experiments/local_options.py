"""
The two ways of explaining a new point (Section "Cost"), compared on fidelity:
  1. interpolate the components linearly from the fitting points, at no evaluation of the
     model: order one by linear interpolation along the sorted fitting values of x_j, order
     two by linear interpolation on the Delaunay triangulation of the fitting points of
     (x_i, x_j), nearest fitting point outside its convex hull;
  2. recompute the dividend at x, eq. (dividendo), 1 + d + d(d-1)/2 evaluations per point
     (the way used in the paper).
California and cancer, both base models, five seeds, fidelity on the test set.

    python experiments/local_options.py
"""
from __future__ import annotations
import warnings
import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

warnings.simplefilter("ignore")
from common import load, fit_subsample, train_mlp, mean_std, Tee, TAU2, NFIT, SEEDS, XGB
from harsanyi_ar import HarsanyiARSurrogate
from harsanyi_ar.numerics import r2_score


def interpolate_from_fitting_points(sur, Xfit, X):
    """Option 1: surrogate prediction at X from the components known at the fitting points."""
    f0, c1f, c2f, pairs = sur.components(Xfit)
    n, d = X.shape
    c1 = np.zeros((n, d)); c2 = np.zeros((n, len(pairs)))
    for j in range(d):
        xs, inv = np.unique(Xfit[:, j], return_inverse=True)
        ys = np.bincount(inv, weights=c1f[:, j]) / np.bincount(inv)      # mean over repeated values
        c1[:, j] = np.interp(X[:, j], xs, ys)
    for p, (i, j) in enumerate(pairs):
        P = Xfit[:, [i, j]]
        v = LinearNDInterpolator(P, c2f[:, p])(X[:, [i, j]])
        m = np.isnan(v)
        if m.any():
            v[m] = NearestNDInterpolator(P, c2f[:, p])(X[m][:, [i, j]])
        c2[:, p] = v
    return f0 + sur.kappa + c1.sum(1) + c2.sum(1)


def base_model(sp, dataset, which, seed):
    if which == "xgb":
        import xgboost as xgb
        if sp.task == "regression":
            m = xgb.XGBRegressor(random_state=seed, **XGB[dataset]).fit(sp.Ztr, sp.ytr)
            return lambda A: m.predict(np.asarray(A, np.float32))
        m = xgb.XGBClassifier(random_state=seed, eval_metric="logloss", **XGB[dataset]).fit(sp.Ztr, sp.ytr)
        return lambda A: m.predict(np.asarray(A, np.float32), output_margin=True)
    from harsanyi_ar import TorchModel
    return TorchModel(train_mlp(sp, dataset, "softplus", seed), sp.d).predict


if __name__ == "__main__":
    out = Tee("local_options.txt")
    out(f"seeds = {list(SEEDS)}; fidelity on the test set")
    out(f"{'':22s} {'1. interpolated':>16s} {'2. exact dividends':>19s}")
    for dataset in ("california", "cancer"):
        sp = load(dataset)
        for which in ("xgb", "net"):
            f1, f2 = [], []
            for seed in SEEDS:
                Xfit = sp.Ztr[fit_subsample(len(sp.Ztr), NFIT, seed)] if dataset == "california" else sp.Ztr
                pred = base_model(sp, dataset, which, seed)
                sur = HarsanyiARSurrogate(pred, np.zeros(sp.d), None, TAU2).fit(Xfit)
                fm = pred(sp.Zte)
                f1.append(r2_score(fm, interpolate_from_fitting_points(sur, Xfit, sp.Zte)))
                f2.append(r2_score(fm, sur.predict(sp.Zte)))
            out(f"{dataset + ', ' + ('XGBoost' if which == 'xgb' else 'network'):22s} {mean_std(f1):>16s} {mean_std(f2):>19s}")
