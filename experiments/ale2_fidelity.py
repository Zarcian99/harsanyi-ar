"""
Fidelity of pure ALE of order <= 2 against the surrogate model of the paper, on the same
models, fitting samples and test sets (California, cancer and Wine; XGBoost and the softplus
network; five seeds). ALE-2 is the sum of the ALE curves and surfaces of f (not of the
residual), built with the same grids, differences, accumulation and discount as the package,
plus a constant fitted as the mean of f minus the components on the fitting sample; it is
read at a point by interpolation, so explaining a new instance costs no model evaluation.

    python experiments/ale2_fidelity.py
"""
from __future__ import annotations
import itertools
import time
import warnings
import numpy as np

warnings.simplefilter("ignore")
from common import load, fit_subsample, train_mlp, mean_std, Tee, TAU2, NFIT, SEEDS, XGB
from harsanyi_ar import HarsanyiARSurrogate
from harsanyi_ar.ale import quantile_edges, cell_index, build_ar1, build_ar2, terrell_scott_bins
from harsanyi_ar.numerics import r2_score, sum_terms


class ALE2:
    """Pure ALE surrogate of order <= 2, with the building blocks of the package."""
    def __init__(self, predict, n_bins=None):
        self.predict, self.n_bins = predict, n_bins

    def fit(self, Xfit):
        n, d = Xfit.shape
        K = terrell_scott_bins(n) if self.n_bins is None else self.n_bins
        self.d = d
        self.edges = [quantile_edges(Xfit[:, j], K) for j in range(d)]
        idx = [cell_index(Xfit[:, j], self.edges[j]) for j in range(d)]
        self.curves, self.surfaces = [], {}
        for j in range(d):
            A, B = Xfit.copy(), Xfit.copy()
            A[:, j], B[:, j] = self.edges[j][idx[j] + 1], self.edges[j][idx[j]]
            self.curves.append(build_ar1(self.edges[j], idx[j], self.predict(A) - self.predict(B), 0.0, TAU2, False))
        for i, j in itertools.combinations(range(d), 2):
            ei, ej, iid, jid = self.edges[i], self.edges[j], idx[i], idx[j]
            second = np.zeros(n)
            for ai, aj, sgn in ((ei[iid + 1], ej[jid + 1], 1.0), (ei[iid + 1], ej[jid], -1.0),
                                (ei[iid], ej[jid + 1], -1.0), (ei[iid], ej[jid], 1.0)):
                Y = Xfit.copy(); Y[:, i] = ai; Y[:, j] = aj
                second += sgn * self.predict(Y)
            self.surfaces[(i, j)] = build_ar2(i, j, Xfit[:, i], Xfit[:, j], ei, ej, iid, jid, second, 0.0, 0.0, TAU2, False)
        self.c = float(np.mean(self.predict(Xfit) - self._sum(Xfit)))
        return self

    def _sum(self, X):
        terms = [self.curves[j](X[:, j]) for j in range(self.d)] + [s(X) for s in self.surfaces.values()]
        return sum_terms(terms)

    def predict_surrogate(self, X):
        return self.c + self._sum(np.asarray(X, float))

    def fidelity(self, X):
        return r2_score(self.predict(np.asarray(X, float)), self.predict_surrogate(X))


def outputs(sp, dataset, which, seed):
    if which == "xgb":
        import xgboost as xgb
        if sp.task == "regression":
            m = xgb.XGBRegressor(random_state=seed, **XGB[dataset]).fit(sp.Ztr, sp.ytr)
            return [lambda A, m=m: m.predict(np.asarray(A, np.float32))]
        m = xgb.XGBClassifier(random_state=seed, eval_metric="logloss" if sp.task == "binary" else "mlogloss",
                              **XGB[dataset]).fit(sp.Ztr, sp.ytr)
        if sp.task == "binary":
            return [lambda A, m=m: m.predict(np.asarray(A, np.float32), output_margin=True)]
        return [lambda A, m=m, c=c: m.predict(np.asarray(A, np.float32), output_margin=True)[:, c] for c in range(3)]
    from harsanyi_ar import TorchModel
    net = train_mlp(sp, dataset, "softplus", seed)
    return [TorchModel(net, sp.d, c).predict for c in range(3 if sp.task == "multiclass" else 1)]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--models", default="xgb,net"); ap.add_argument("--datasets", default="california,cancer,wine")
    ap.add_argument("--tag", default=""); a = ap.parse_args()
    out = Tee(f"ale2_fidelity{('_' + a.tag) if a.tag else ''}.txt")
    out(f"seeds = {list(SEEDS)}; fidelity on the test set; ALE-2 = ALE curves and surfaces of the model, no dividends")
    out(f"{'':24s} {'ALE order <= 2':>16s} {'this paper':>16s} {'ALE-2 fit (s)':>14s} {'ours fit (s)':>13s}")
    for which in a.models.split(","):
        for dataset in a.datasets.split(","):
            sp = load(dataset); fa, fo, ta, to = [], [], [], []
            for seed in SEEDS:
                Xfit = sp.Ztr[fit_subsample(len(sp.Ztr), NFIT, seed)] if dataset == "california" else sp.Ztr
                va, vo = [], []
                for pred in outputs(sp, dataset, which, seed):
                    t = time.perf_counter(); ale = ALE2(pred).fit(Xfit); ta.append(time.perf_counter() - t)
                    t = time.perf_counter(); sur = HarsanyiARSurrogate(pred, np.zeros(sp.d), None, TAU2).fit(Xfit); to.append(time.perf_counter() - t)
                    va.append(ale.fidelity(sp.Zte)); vo.append(sur.fidelity(sp.Zte))
                fa.append(np.mean(va)); fo.append(np.mean(vo))
            out(f"{dataset + ', ' + ('XGBoost' if which == 'xgb' else 'network'):24s} {mean_std(fa):>16s} {mean_std(fo):>16s} {np.mean(ta):>14.2f} {np.mean(to):>13.2f}")
