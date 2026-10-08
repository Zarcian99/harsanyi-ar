"""
Section "Cost": model evaluations and wall-clock time (XGBoost on California, seed 0).

  * building the surrogate: number of model evaluations and time, relative to training;
  * explaining one instance with recomputed dividends: exactly 1 + d + d(d-1)/2
    evaluations of the model, whatever the instance.

    python experiments/cost.py
"""
from __future__ import annotations
import time
import warnings
import numpy as np

warnings.simplefilter("ignore")
from common import load, fit_subsample, Tee, N_BINS, TAU2, NFIT, XGB_CALIFORNIA
from harsanyi_ar import HarsanyiARSurrogate
import xgboost as xgb

if __name__ == "__main__":
    out = Tee("cost.txt")
    ca = load("california")
    t0 = time.perf_counter()
    booster = xgb.XGBRegressor(random_state=0, **XGB_CALIFORNIA).fit(ca.Ztr, ca.ytr)
    t_train = time.perf_counter() - t0

    calls = {"n": 0, "rows": 0}

    def predict(A):
        A = np.asarray(A, np.float32)
        calls["n"] += 1
        calls["rows"] += len(A)
        return booster.predict(A)

    Xfit = ca.Ztr[fit_subsample(len(ca.Ztr), NFIT, 0)]
    t0 = time.perf_counter()
    sur = HarsanyiARSurrogate(predict, np.zeros(ca.d), N_BINS, TAU2).fit(Xfit)
    t_build = time.perf_counter() - t0
    out(f"training XGBoost        : {t_train:.2f} s")
    out(f"building the surrogate  : {t_build:.2f} s  ({len(Xfit)} fitting points, "
        f"{calls['rows']} model evaluations in {calls['n']} batched calls)")

    calls.update(n=0, rows=0)
    sur.explain_local(ca.Zte[:1])
    d = ca.d
    out(f"explaining one instance : {calls['rows']} model evaluations "
        f"(expected 1 + d + d(d-1)/2 = {1 + d + d * (d - 1) // 2})")

    t0 = time.perf_counter()
    for _ in range(3):
        sur.predict(ca.Zte)
    out(f"surrogate prediction    : {1000 * (time.perf_counter() - t0) / 3 / len(ca.Zte):.3f} ms per instance")
