"""
Does it matter that the surrogate is fitted on rows the base model was trained on?
California: the base model is trained on the training set minus 4000 held-out rows; the
surrogate is fitted (a) on those 4000 held-out rows or (b) on 4000 rows used to train the
model; fidelity on the test set, five seeds.

    python experiments/fit_sample_check.py
"""
from __future__ import annotations
import warnings
import numpy as np

warnings.simplefilter("ignore")
from common import load, Tee, TAU2, SEEDS, XGB, NET_HIDDEN, NET_LR, NET_EPOCHS, NET_BATCH
from harsanyi_ar import HarsanyiARSurrogate
from harsanyi_ar.numerics import r2_score

N_HOLD = 4000


def train_net(X, y, seed):
    import torch
    from harsanyi_ar import MLP
    torch.set_default_dtype(torch.float64); torch.manual_seed(seed)
    net = MLP(X.shape[1], "softplus", hidden=NET_HIDDEN, d_out=1)
    opt = torch.optim.Adam(net.parameters(), lr=NET_LR)
    Xt, yt = torch.as_tensor(X), torch.as_tensor(y).reshape(-1, 1)
    g = torch.Generator().manual_seed(seed)
    for _ in range(NET_EPOCHS["california"]):
        perm = torch.randperm(len(Xt), generator=g)
        for s in range(0, len(Xt), NET_BATCH["california"]):
            b = perm[s:s + NET_BATCH["california"]]
            opt.zero_grad(); torch.nn.functional.mse_loss(net(Xt[b]), yt[b]).backward(); opt.step()
    return net


if __name__ == "__main__":
    out = Tee("fit_sample_check.txt")
    sp = load("california")
    out(f"seeds = {list(SEEDS)}; base model trained on {len(sp.Ztr) - N_HOLD} rows; surrogate on {N_HOLD} rows")
    for which in ("xgb", "net"):
        a, b, perf = [], [], []
        for seed in SEEDS:
            rs = np.random.RandomState(seed)
            perm = rs.permutation(len(sp.Ztr)); hold, used = perm[:N_HOLD], perm[N_HOLD:]
            Xm, ym = sp.Ztr[used], sp.ytr[used]
            if which == "xgb":
                import xgboost as xgb
                m = xgb.XGBRegressor(random_state=seed, **XGB["california"]).fit(Xm, ym)
                pred = lambda A, m=m: m.predict(np.asarray(A, np.float32))
            else:
                from harsanyi_ar import TorchModel
                pred = TorchModel(train_net(Xm, ym, seed), sp.d).predict
            perf.append(r2_score(sp.yte, pred(sp.Zte)))
            for X_fit, acc in ((sp.Ztr[hold], a), (Xm[rs.choice(len(Xm), N_HOLD, replace=False)], b)):
                acc.append(HarsanyiARSurrogate(pred, np.zeros(sp.d), None, TAU2).fit(X_fit).fidelity(sp.Zte))
        name = "XGBoost" if which == "xgb" else "network"
        out(f"  {name:8s} model R2 = {np.mean(perf):.3f}   fidelity, surrogate fitted on held-out rows = "
            f"{np.mean(a):.3f} +- {np.std(a):.3f}   on rows used to train the model = {np.mean(b):.3f} +- {np.std(b):.3f}")
