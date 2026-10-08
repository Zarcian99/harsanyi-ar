"""
Tree depth against order-2 fidelity (supports the remark on tree depth in
"Limitations"): a tree of depth p contains interactions of order <= p only, so deeper
ensembles leave more structure above the truncation order.

    python experiments/xgb_depth_sweep.py
"""
from __future__ import annotations
import warnings
import numpy as np

warnings.simplefilter("ignore")
from common import load, fit_subsample, Tee, N_BINS, TAU2, NFIT
from harsanyi_ar import HarsanyiARSurrogate, r2_score
import xgboost as xgb

if __name__ == "__main__":
    out = Tee("xgb_depth_sweep.txt")
    ca = load("california")
    Xfit = ca.Ztr[fit_subsample(len(ca.Ztr), NFIT, 0)]
    out(f"{'depth':>5s} {'trees':>6s} {'XGBoost R2':>11s} {'fid. dividends':>15s} {'fidelity':>9s} {'system R2':>10s}")
    for depth, trees in ((2, 400), (3, 300), (4, 200), (6, 200)):
        m = xgb.XGBRegressor(random_state=0, n_estimators=trees, max_depth=depth, learning_rate=0.1,
                             subsample=0.8, colsample_bytree=0.8).fit(ca.Ztr, ca.ytr)
        predict = lambda A, m=m: m.predict(np.asarray(A, np.float32))
        sur = HarsanyiARSurrogate(predict, np.zeros(ca.d), N_BINS, TAU2).fit(Xfit)
        out(f"{depth:5d} {trees:6d} {r2_score(ca.yte, predict(ca.Zte)):11.3f} "
            f"{sur.fidelity_dividends_only(ca.Zte):15.3f} {sur.fidelity(ca.Zte):9.3f} "
            f"{r2_score(ca.yte, sur.predict(ca.Zte)):10.3f}")
