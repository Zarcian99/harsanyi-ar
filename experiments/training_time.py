"""
Time to obtain each model of the table "Predictive performance against the data"
(tab:predictores), on one CPU core, in one run: the transparent baselines, the two black boxes,
and the surrogate model fitted to each black box (fitting sample of the paper: 4000 training rows on California, the whole training
set on cancer and Wine; on Wine, the three surrogates of the class logits). For the surrogate
model the fit is reported on its own: the black box it explains already exists, and its own
training time is in the black-box row. Every thread pool (numpy, XGBoost, torch, EBM) is set
to one thread (the EBM is timed in ebm_distilled.py, against the surrogate model). Mean and standard deviation over the seeds; the deterministic baselines are
timed once per seed as well.

    python experiments/training_time.py [--datasets california,cancer,wine] [--seeds 0,1,2]
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import time
import warnings
import numpy as np

warnings.simplefilter("ignore")
from common import load, fit_subsample, train_mlp, Tee, TAU2, NFIT, XGB
from harsanyi_ar import HarsanyiARSurrogate


def timed(fn):
    t = time.perf_counter(); r = fn(); return time.perf_counter() - t, r


def ms(v):
    v = np.asarray(v, float)
    return f"{v.mean():9.2f} +- {v.std():6.2f}"


def baselines(sp):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures, SplineTransformer
    from sklearn.linear_model import RidgeCV, LogisticRegression
    if sp.task == "regression":
        alphas = np.logspace(-3, 3, 13)
        return {"linear / logistic": make_pipeline(StandardScaler(), RidgeCV(alphas=alphas)),
                "degree-2 polynomial": make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False), RidgeCV(alphas=alphas)),
                "GAM with splines": make_pipeline(StandardScaler(), SplineTransformer(n_knots=12, degree=3), RidgeCV(alphas=alphas))}
    lr = lambda: LogisticRegression(max_iter=5000, C=1.0)
    return {"linear / logistic": make_pipeline(StandardScaler(), lr()),
            "degree-2 polynomial": make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False), lr()),
            "GAM with splines": make_pipeline(StandardScaler(), SplineTransformer(n_knots=8, degree=3), lr())}


def outputs_of(sp, model_predict_margin, n_out):
    return [(lambda A, c=c: model_predict_margin(A)[:, c]) for c in range(n_out)] if n_out > 1 else [model_predict_margin]


if __name__ == "__main__":
    import argparse
    import platform
    import torch
    import xgboost as xgb
    torch.set_num_threads(1)
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="california,cancer,wine"); ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    out = Tee(f"training_time{('_' + a.tag) if a.tag else ''}.txt")
    cpu = platform.processor() or "unknown"
    if platform.system() == "Windows":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    else:
        try:
            for line in open("/proc/cpuinfo"):
                if line.lower().startswith("model name"):
                    cpu = line.split(":", 1)[1].strip(); break
        except OSError:
            pass
    out(f"machine: {cpu}; {platform.platform()}; python {platform.python_version()}")
    out(f"training time in seconds, one thread for every library; mean +- std over seeds {seeds}")
    from harsanyi_ar import TorchModel
    for dataset in a.datasets.split(","):
        sp = load(dataset)
        n_out = 3 if sp.task == "multiclass" else 1
        T = {k: [] for k in ("linear / logistic", "degree-2 polynomial", "GAM with splines",
                             "XGBoost (black box)", "network (black box)",
                             "surrogate fit on XGBoost", "surrogate fit on network")}
        for seed in seeds:
            for name, m in baselines(sp).items():
                T[name].append(timed(lambda m=m: m.fit(sp.Xtr, sp.ytr))[0])
            if sp.task == "regression":
                t, m = timed(lambda: xgb.XGBRegressor(random_state=seed, n_jobs=1, **XGB[dataset]).fit(sp.Ztr, sp.ytr))
                margin = lambda A, m=m: m.predict(np.asarray(A, np.float32))
            else:
                ev = "logloss" if sp.task == "binary" else "mlogloss"
                t, m = timed(lambda: xgb.XGBClassifier(random_state=seed, n_jobs=1, eval_metric=ev, **XGB[dataset]).fit(sp.Ztr, sp.ytr))
                margin = lambda A, m=m: m.predict(np.asarray(A, np.float32), output_margin=True)
            T["XGBoost (black box)"].append(t)
            t, net = timed(lambda: train_mlp(sp, dataset, "softplus", seed))
            T["network (black box)"].append(t)
            Xfit = sp.Ztr[fit_subsample(len(sp.Ztr), NFIT, seed)] if dataset == "california" else sp.Ztr
            fx = outputs_of(sp, margin, n_out)
            T["surrogate fit on XGBoost"].append(sum(timed(lambda f=f: HarsanyiARSurrogate(f, np.zeros(sp.d), None, TAU2).fit(Xfit))[0] for f in fx))
            fn = [TorchModel(net, sp.d, c).predict for c in range(n_out)]
            T["surrogate fit on network"].append(sum(timed(lambda f=f: HarsanyiARSurrogate(f, np.zeros(sp.d), None, TAU2).fit(Xfit))[0] for f in fn))
            out(f"  {dataset} seed {seed}: " + "  ".join(f"{k} {v[-1]:.2f}" for k, v in T.items()))
        out(f"\n== {dataset} ==")
        for k, v in T.items():
            out(f"  {k:28s} {ms(v)} s")
