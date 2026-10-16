"""
Sensitivity of the fidelity to the baseline, on California, for the six activation functions
of Table "tab:activacion" (section on activations).

The surrogate model masks the absent variables at the baseline. The default baseline is the
vector 0, the training mean of the standardized features. Here each network is also explained
with two other baselines that lie in a denser region of the data:
  - the coordinate-wise median of the training features;
  - the training instance closest (in Euclidean distance) to that median: a real point.
Fidelity = R^2 between the surrogate model and the network on the test split; mean +- std over
the five seeds; the networks are the same as in table1_activations.py.

    python experiments/baseline_sensitivity.py
"""
from __future__ import annotations
import argparse, time, warnings
import numpy as np

warnings.simplefilter("ignore")
from common import load, fit_subsample, train_mlp, mean_std, Tee, N_BINS, TAU2, NFIT, SEEDS
from harsanyi_ar import HarsanyiARSurrogate, TorchModel, r2_score

ACTIVATIONS = ["softplus", "sigmoid", "tanh", "silu", "mish", "gelu"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--activations", default=",".join(ACTIVATIONS))
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    out = Tee("baseline_sensitivity.txt")
    sp = load("california")
    median = np.median(sp.Ztr, axis=0)
    nearest = sp.Ztr[np.argmin(((sp.Ztr - median) ** 2).sum(1))]
    out(f"seeds = {seeds}; California, test split, order 2, {N_BINS} bins, no shrinkage")
    out("baselines (standardized units):")
    out("  mean    = " + np.array2string(np.zeros(sp.d), precision=2, separator=", "))
    out("  median  = " + np.array2string(median, precision=2, separator=", "))
    out("  nearest = " + np.array2string(nearest, precision=2, separator=", "))
    out(f"\n{'activation':10s} {'network R2':>18s} | {'mean (0)':>18s} {'median':>18s} {'nearest instance':>18s}")
    for act in a.activations.split(","):
        fid = {"mean": [], "median": [], "nearest": []}; perf = []
        t0 = time.time()
        for seed in seeds:
            net = train_mlp(sp, "california", act, seed)
            m = TorchModel(net, sp.d, 0)
            Xfit = sp.Ztr[fit_subsample(len(sp.Ztr), NFIT, seed)]
            perf.append(r2_score(sp.yte, m.predict(sp.Zte)))
            for name, b in (("mean", np.zeros(sp.d)), ("median", median), ("nearest", nearest)):
                fid[name].append(HarsanyiARSurrogate(m.predict, b, N_BINS, TAU2).fit(Xfit).fidelity(sp.Zte))
        out(f"{act:10s} {mean_std(perf):>18s} | {mean_std(fid['mean']):>18s} {mean_std(fid['median']):>18s} "
            f"{mean_std(fid['nearest']):>18s}    ({time.time()-t0:.0f} s)")
