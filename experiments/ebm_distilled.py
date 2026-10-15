"""
Order-2 EBM fitted to the predictions of each network of the activation table (tab:activacion),
against the surrogate model, on California: the six activations, the five seeds and the 4000
fitting rows of the paper. The EBM (interpret, default hyperparameters, pairwise interactions)
learns from the network's predictions on the fitting rows, not from the labels, as the surrogate
model does. For each, the fidelity (R^2 to the network on the test set) and the time, on one
thread for every library: for the surrogate model, its fit (which evaluates the network at the
masked points) and its predictions on the test set; for the EBM, the network's predictions on
the fitting rows, its fit and its predictions on the test set. The network itself is shared and
not timed.

    python experiments/ebm_distilled.py
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"                     # one thread for every library, so that the times compare
import time
import warnings
import numpy as np

warnings.simplefilter("ignore")
from common import load, fit_subsample, train_mlp, mean_std, Tee, N_BINS, TAU2, NFIT, SEEDS
from harsanyi_ar import HarsanyiARSurrogate, TorchModel
from harsanyi_ar.numerics import r2_score

ACTIVATIONS = ["softplus", "sigmoid", "tanh", "silu", "mish", "gelu"]


def ebm_fidelity(sp, predicts, Xfit, seed, rows=None):
    """Fidelity of an EBM fitted to each output (mean over outputs), fit time and prediction time."""
    from interpret.glassbox import ExplainableBoostingRegressor
    X = Xfit if rows is None else rows
    fids, t_fit, t_pred = [], 0.0, 0.0
    for pred in predicts:
        t = time.perf_counter()
        e = ExplainableBoostingRegressor(random_state=seed, n_jobs=1).fit(X, pred(X))
        t_fit += time.perf_counter() - t
        t = time.perf_counter(); yhat = e.predict(sp.Zte); t_pred += time.perf_counter() - t
        fids.append(r2_score(pred(sp.Zte), yhat))
    return float(np.mean(fids)), t_fit, t_pred


def surrogate_fidelity(sp, predicts, Xfit):
    """Fidelity of the surrogate model (mean over outputs), fit time and prediction time."""
    fids, t_fit, t_pred = [], 0.0, 0.0
    for p in predicts:
        t = time.perf_counter(); s = HarsanyiARSurrogate(p, np.zeros(sp.d), N_BINS, TAU2).fit(Xfit)
        t_fit += time.perf_counter() - t
        t = time.perf_counter(); yhat = s.predict(sp.Zte); t_pred += time.perf_counter() - t
        fids.append(r2_score(p(sp.Zte), yhat))
    return float(np.mean(fids)), t_fit, t_pred


def net_outputs(sp, dataset, act, seed):
    net = train_mlp(sp, dataset, act, seed)
    n_out = 3 if sp.task == "multiclass" else 1
    return [TorchModel(net, sp.d, c).predict for c in range(n_out)]


def fitting_rows(sp, dataset, seed):
    return sp.Ztr[fit_subsample(len(sp.Ztr), NFIT, seed)] if dataset == "california" else sp.Ztr


def run(seeds, tag=""):
    import torch
    torch.set_num_threads(1)
    out = Tee(f"ebm_distilled{('_' + tag) if tag else ''}.txt")
    out(f"seeds = {seeds}; California; the network of each activation, explained by the surrogate model and by an")
    out("order-2 EBM fitted to the network's predictions. Network R^2: against the data, on the test set. Fidelity:")
    out("R^2 to the network on the test set. Time: fit + predictions on the test set, one thread for every library.")
    sp = load("california")
    out(f"\n{'activation':10s} {'network R^2':>15s} {'surrogate fid.':>15s} {'surrogate time':>15s} {'EBM fid.':>15s} {'EBM time':>13s}")
    for act in ACTIVATIONS:
        perf, fs, ts, fe, te = [], [], [], [], []
        for seed in seeds:
            preds = net_outputs(sp, "california", act, seed)
            perf.append(r2_score(sp.yte, preds[0](sp.Zte)))
            Xfit = fitting_rows(sp, "california", seed)
            f1, t1, t2 = surrogate_fidelity(sp, preds, Xfit)
            f2, t3, t4 = ebm_fidelity(sp, preds, Xfit, seed)
            fs.append(f1); ts.append(t1 + t2); fe.append(f2); te.append(t3 + t4)
            out(f"  {act} seed {seed}: network R^2 {perf[-1]:.4f}  surrogate {f1:.4f} ({t1:.1f} + {t2:.1f} s)"
                f"  EBM {f2:.4f} ({t3:.1f} + {t4:.1f} s)")
        out(f"{act:10s} {mean_std(perf):>15s} {mean_std(fs):>15s} {np.mean(ts):>13.1f} s {mean_std(fe):>15s} {np.mean(te):>11.1f} s")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    run([int(s) for s in a.seeds.split(",")], a.tag)
