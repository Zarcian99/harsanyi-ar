"""
Finite differences (the method of the paper, harsanyi_ar) against the derivative variant
(experiments/dale_variant.py) on the same networks and fitting points: fidelity and
fitting time, California, seeds 0-4. All the results of the paper use finite differences.

    python experiments/route_comparison.py
"""
from __future__ import annotations
import time
import warnings
import numpy as np

warnings.simplefilter("ignore")
from common import load, fit_subsample, train_mlp, mean_std, Tee, N_BINS, TAU2, NFIT
from harsanyi_ar import HarsanyiARSurrogate, TorchModel
from dale_variant import HarsanyiARSurrogateDALE

if __name__ == "__main__":
    out = Tee("route_comparison.txt")
    ca = load("california")
    out(f"{'activation':10s} {'finite differences':>32s} {'derivatives':>32s}")
    for act in ("softplus", "sigmoid", "tanh", "silu", "mish", "gelu"):
        R = {k: [] for k in ("f_fd", "t_fd", "f_de", "t_de")}
        for seed in range(5):
            model = TorchModel(train_mlp(ca, "california", act, seed), ca.d)
            Xfit = ca.Ztr[fit_subsample(len(ca.Ztr), NFIT, seed)]
            t = time.perf_counter()
            s = HarsanyiARSurrogate(model.predict, np.zeros(ca.d), N_BINS, TAU2).fit(Xfit)
            R["t_fd"].append(time.perf_counter() - t); R["f_fd"].append(s.fidelity(ca.Zte))
            t = time.perf_counter()
            s = HarsanyiARSurrogateDALE(model, N_BINS, TAU2).fit(Xfit)
            R["t_de"].append(time.perf_counter() - t); R["f_de"].append(s.fidelity(ca.Zte))
        out(f"{act:10s} {mean_std(R['f_fd']):>20s} {np.mean(R['t_fd']):8.1f} s {mean_std(R['f_de']):>22s} {np.mean(R['t_de']):8.1f} s")
