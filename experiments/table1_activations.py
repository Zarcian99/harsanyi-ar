"""
Table "Fidelity by activation function and dataset" (tab:activacion).

Fidelity = R^2 between the order-2 surrogate and the network it explains, on the test
split; mean +- std over five seeds (network initialisation, training order and, on
California, fitting subsample). In classification the logit is decomposed; on Wine the
fidelity is averaged over the three class logits. The predictive performance of each
network (R^2, AUC, accuracy) is reported next to it.

    python experiments/table1_activations.py [--datasets california,cancer,wine]
"""
from __future__ import annotations
import argparse
import warnings
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

warnings.simplefilter("ignore")
from common import load, fit_subsample, train_mlp, mean_std, Tee, N_BINS, TAU2, NFIT, SEEDS
from harsanyi_ar import HarsanyiARSurrogate, TorchModel, r2_score

ACTIVATIONS = ["softplus", "sigmoid", "tanh", "silu", "mish", "gelu"]


def performance(sp, outputs):
    if sp.task == "regression":
        return r2_score(sp.yte, outputs[:, 0])
    if sp.task == "binary":
        return roc_auc_score(sp.yte, outputs[:, 0])
    return accuracy_score(sp.yte, outputs.argmax(1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="california,cancer,wine")
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    out = Tee(f"table1_activations{('_' + a.tag) if a.tag else ''}.txt")
    out(f"seeds = {seeds}")
    for dataset in a.datasets.split(","):
        sp = load(dataset)
        n_out = 3 if sp.task == "multiclass" else 1
        metric = {"regression": "R2", "binary": "AUC", "multiclass": "accuracy"}[sp.task]
        out(f"\n== {dataset} ==")
        for act in ACTIVATIONS:
            fid, perf = [], []
            for seed in seeds:
                net = train_mlp(sp, dataset, act, seed)
                Xfit = sp.Ztr[fit_subsample(len(sp.Ztr), NFIT, seed)] if dataset == "california" else sp.Ztr
                models = [TorchModel(net, sp.d, c) for c in range(n_out)]
                fid.append(float(np.mean([HarsanyiARSurrogate(m.predict, np.zeros(sp.d), N_BINS, TAU2)
                                          .fit(Xfit).fidelity(sp.Zte) for m in models])))
                perf.append(performance(sp, np.column_stack([m.predict(sp.Zte) for m in models])))
            out(f"  {act:9s} fidelity = {mean_std(fid)}   network {metric} = {mean_std(perf)}")
