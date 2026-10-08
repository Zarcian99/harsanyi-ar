"""
Ablation of the estimation choices on California (Section "Stability in cells with few
data" and the quantile grid of Section "Sample approximation"), on XGBoost and on the
softplus network, five seeds, fidelity on the test set:

  A. shrinkage off (the default) / on, for fitting samples n_fit in {200, 500, 1000, 4000, 16512}
     (the whole training set), with the bins given by the Terrell-Scott rule;
  B. intercept kappa on / off, n_fit = 4000;
  C. number of bins per variable fixed at K in {5, 10, 20, 40} for every n_fit, against the
     rule (the rows of block A without shrinkage).

Each row also reports the fraction of empty cells of the order-2 grids. The networks are
trained once per seed and shared by every configuration.

    python experiments/ablation.py --models xgb,net
"""
from __future__ import annotations
import argparse
import itertools
import warnings
import numpy as np

warnings.simplefilter("ignore")
from common import load, fit_subsample, train_mlp, mean_std, Tee, TAU2, SEEDS, XGB
from harsanyi_ar import HarsanyiARSurrogate
from harsanyi_ar.ale import quantile_edges, cell_index

N_FITS = (200, 500, 1000, 4000, 16512)
K_FIXED = (5, 10, 20, 40)


def empty_fraction(Xfit, n_bins):
    """Fraction of empty cells over the order-2 grids of the fitting sample."""
    d = Xfit.shape[1]
    edges = [quantile_edges(Xfit[:, j], n_bins) for j in range(d)]
    idx = [cell_index(Xfit[:, j], edges[j]) for j in range(d)]
    fr = []
    for i, j in itertools.combinations(range(d), 2):
        bi, bj = edges[i].size - 1, edges[j].size - 1
        cnt = np.bincount(idx[i] * bj + idx[j], minlength=bi * bj)
        fr.append((cnt == 0).mean())
    return float(np.mean(fr))


def models(sp, which):
    """One scalar predictor per seed."""
    out = {}
    for seed in SEEDS:
        if which == "xgb":
            import xgboost as xgb
            m = xgb.XGBRegressor(random_state=seed, **XGB["california"]).fit(sp.Ztr, sp.ytr)
            out[seed] = lambda A, m=m: m.predict(np.asarray(A, np.float32))
        else:
            from harsanyi_ar import TorchModel
            out[seed] = TorchModel(train_mlp(sp, "california", "softplus", seed), sp.d).predict
    return out


def run(sp, preds, out, label, n_fit, n_bins=None, shrink=False, intercept=True):
    fid, emp, K = [], [], None
    for seed in SEEDS:
        Xfit = sp.Ztr[fit_subsample(len(sp.Ztr), n_fit, seed)] if n_fit < len(sp.Ztr) else sp.Ztr
        sur = HarsanyiARSurrogate(preds[seed], np.zeros(sp.d), n_bins, TAU2, shrink, intercept).fit(Xfit)
        fid.append(sur.fidelity(sp.Zte)); K = sur.n_bins
        emp.append(empty_fraction(Xfit, sur.n_bins))
    out(f"  {label:44s} n_fit={n_fit:5d}  K={K:2d}  fidelity = {mean_std(fid)}   empty cells = {np.mean(emp):.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="xgb,net")
    ap.add_argument("--blocks", default="A,B,C")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    out = Tee(f"ablation{('_' + a.tag) if a.tag else ''}.txt")
    out(f"seeds = {list(SEEDS)}; California; fidelity on the test set; empty cells = fraction over the order-2 grids")
    sp = load("california")
    for which in a.models.split(","):
        preds = models(sp, which)
        out(f"\n== {'XGBoost' if which == 'xgb' else 'network (softplus)'} ==")
        if "A" in a.blocks:
            for n_fit in N_FITS:
                for shrink in (True, False):
                    run(sp, preds, out, f"A  shrinkage {'on ' if shrink else 'off'}", n_fit, shrink=shrink)
        if "B" in a.blocks:
            run(sp, preds, out, "B  intercept kappa off", 4000, intercept=False)
        if "C" in a.blocks:
            for n_fit in N_FITS:
                for K in K_FIXED:
                    run(sp, preds, out, f"C  fixed K = {K}", n_fit, n_bins=K)
