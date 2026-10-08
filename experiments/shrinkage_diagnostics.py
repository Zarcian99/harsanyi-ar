"""
How much the shrinkage of Section "Stability in cells with few data" intervenes, on
California, for XGBoost and the softplus network, by fitting size (five seeds).

For every grid (order one and order two) the fit computes the pooled within-cell variance v,
the between-cell variance tau^2 and the factors gamma_C = tau^2 / (tau^2 + v / |N(C)|),
eq. (gamma). Reported, separately for the order-one and order-two grids:
  shrink   mean over the non-empty cells of 1 - gamma_C: the fraction of the increment that
           shrinkage removes (0 = no intervention);
  v/tau2   median over the grids of v / tau^2: within-cell spread of the differences relative
           to the spread of the true increments between cells;
  empty    fraction of empty cells.

    python experiments/shrinkage_diagnostics.py
"""
from __future__ import annotations
import warnings
import numpy as np

warnings.simplefilter("ignore")
from common import load, fit_subsample, Tee, TAU2, SEEDS      # puts the package on the path
import harsanyi_ar.ale as ale_mod
from ablation import models, N_FITS
from harsanyi_ar import HarsanyiARSurrogate

RECORD: list = []
_original = ale_mod.shrink_increments


def recording_shrink(cell_mean, count, pooled_var, tau2_method="adm"):
    out, gamma, tau2 = _original(cell_mean, count, pooled_var, tau2_method)
    ne = count > 0
    RECORD.append(dict(order=cell_mean.ndim, shrink=float(np.mean(1.0 - gamma[ne])) if ne.any() else 0.0,
                       ratio=float(pooled_var / tau2) if tau2 > 0 else np.inf,
                       empty=float(np.mean(~ne))))
    return out, gamma, tau2


ale_mod.shrink_increments = recording_shrink

if __name__ == "__main__":
    out = Tee("shrinkage_diagnostics.txt")
    out(f"seeds = {list(SEEDS)}; California; shrink = mean of 1 - gamma_C over non-empty cells; "
        f"v/tau2 = median over grids; empty = fraction of empty cells")
    sp = load("california")
    for which in ("xgb", "net"):
        preds = models(sp, which)
        out(f"\n== {'XGBoost' if which == 'xgb' else 'network (softplus)'} ==")
        out(f"{'n_fit':>6s} {'K':>3s} | {'order 1: shrink':>15s} {'v/tau2':>8s} | "
            f"{'order 2: shrink':>15s} {'v/tau2':>8s} {'empty':>6s}")
        for n_fit in N_FITS:
            rows = {1: [], 2: []}
            for seed in SEEDS:
                Xfit = sp.Ztr[fit_subsample(len(sp.Ztr), n_fit, seed)] if n_fit < len(sp.Ztr) else sp.Ztr
                RECORD.clear()
                sur = HarsanyiARSurrogate(preds[seed], np.zeros(sp.d), None, TAU2, shrink=True).fit(Xfit)
                for r in RECORD:
                    rows[r["order"]].append(r)
            s1 = np.mean([r["shrink"] for r in rows[1]]); q1 = np.median([r["ratio"] for r in rows[1]])
            s2 = np.mean([r["shrink"] for r in rows[2]]); q2 = np.median([r["ratio"] for r in rows[2]])
            e2 = np.mean([r["empty"] for r in rows[2]])
            out(f"{n_fit:>6d} {sur.n_bins:>3d} | {s1:>15.3f} {q1:>8.2f} | {s2:>15.3f} {q2:>8.2f} {e2:>6.2f}")
