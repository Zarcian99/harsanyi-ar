"""
Per-instance fidelity: the residual h(x) = f(x) - surrogate(x) that the surrogate model leaves
unexplained at each test instance. For the models of the fidelity table (XGBoost and the softplus
network on California, cancer and Wine; five seeds; fitting rows as in the paper), the quantiles
50/90/99 and the maximum, over the test instances, of
  |h(x)| / sd(f)                          comparable across datasets;
  |h(x)| / |f(x) - f(0) - kappa|          share of what the surrogate explains at x (median only:
                                          its upper quantiles blow up where f(x) is close to f(0)+kappa).
On Wine the three class logits are pooled, each scaled by its own sd. Mean over seeds.

    python experiments/local_residual.py
"""
from __future__ import annotations
import warnings
import numpy as np

warnings.simplefilter("ignore")
from common import load, fit_subsample, Tee, N_BINS, TAU2, NFIT, SEEDS
from harsanyi_ar import HarsanyiARSurrogate
from projected_indices import outputs


def residuals(sp, preds, Xfit):
    rel_sd, rel_ex = [], []
    for p in preds:
        s = HarsanyiARSurrogate(p, np.zeros(sp.d), N_BINS, TAU2).fit(Xfit)
        fx = p(sp.Zte); h = fx - s.predict(sp.Zte)
        f0 = float(p(np.zeros((1, sp.d)))[0])
        rel_sd.append(np.abs(h) / fx.std())
        rel_ex.append(np.abs(h) / np.abs(fx - f0 - s.kappa))
    return np.concatenate(rel_sd), np.concatenate(rel_ex)


if __name__ == "__main__":
    out = Tee("local_residual.txt")
    out(f"seeds = {list(SEEDS)}; test instances; mean over seeds")
    out(f"{'':22s} {'|h|/sd(f): p50':>15s} {'p90':>7s} {'p99':>7s} {'max':>7s}   {'|h|/explained: p50':>19s}")
    for dataset in ("california", "cancer", "wine"):
        sp = load(dataset)
        for which in ("xgb", "net"):
            q = []
            for seed in SEEDS:
                preds = outputs(sp, dataset, which, seed)
                Xfit = sp.Ztr[fit_subsample(len(sp.Ztr), NFIT, seed)] if dataset == "california" else sp.Ztr
                a, b = residuals(sp, preds, Xfit)
                q.append(list(np.percentile(a, [50, 90, 99])) + [a.max(), np.median(b)])
                out(f"  {dataset} {which} seed {seed}: " + "  ".join(f"{v:.3f}" for v in q[-1]))
            m = np.mean(q, axis=0)
            name = f"{dataset}, {'XGBoost' if which == 'xgb' else 'network'}"
            out(f"{name:22s} {m[0]:15.3f} {m[1]:7.3f} {m[2]:7.3f} {m[3]:7.3f}   {m[4]:19.3f}")
