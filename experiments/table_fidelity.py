"""
Fidelity of the order-2 surrogate to the model it explains, with and without the
correction: XGBoost and the softplus network on California, cancer and Wine.

Fidelity = R^2 between the surrogate and the model. In classification it is measured on
the logit; on Wine, averaged over the three class logits. Three columns per row:
  dividends only (test)   truncation f^k, eq. (truncamiento), on the test split;
  surrogate (test)        kappa + dividends + AR, eq. (subrogado), on the test split;
  surrogate (fit)         the same surrogate on its own fitting sample (in-sample).
Mean +- std over SEEDS (model seed and, on California, the fitting subsample). The
fitting sample, the number of bins and every hyperparameter come from common.py.

    python experiments/table_fidelity.py --models xgb,net [--datasets california,cancer,wine]

The network rows need torch; the XGBoost rows need xgboost only.
"""
from __future__ import annotations
import argparse
import warnings
import numpy as np

warnings.simplefilter("ignore")
from common import load, fit_subsample, train_mlp, mean_std, Tee, N_BINS, TAU2, NFIT, SEEDS, XGB
from harsanyi_ar import HarsanyiARSurrogate

DATASETS = ("california", "cancer", "wine")


def xgb_scalar_outputs(sp, dataset, seed):
    """Scalar outputs of the XGBoost model: the prediction (regression) or the logit of
    each class (classification), as functions (N, d) -> (N,)."""
    import xgboost as xgb
    if sp.task == "regression":
        m = xgb.XGBRegressor(random_state=seed, **XGB[dataset]).fit(sp.Ztr, sp.ytr)
        return [lambda A, m=m: m.predict(np.asarray(A, np.float32))]
    metric = "logloss" if sp.task == "binary" else "mlogloss"
    m = xgb.XGBClassifier(random_state=seed, eval_metric=metric, **XGB[dataset]).fit(sp.Ztr, sp.ytr)
    if sp.task == "binary":
        return [lambda A, m=m: m.predict(np.asarray(A, np.float32), output_margin=True)]
    return [lambda A, m=m, c=c: m.predict(np.asarray(A, np.float32), output_margin=True)[:, c]
            for c in range(3)]


def net_scalar_outputs(sp, dataset, seed):
    from harsanyi_ar import TorchModel
    net = train_mlp(sp, dataset, "softplus", seed)
    n_out = 3 if sp.task == "multiclass" else 1
    return [TorchModel(net, sp.d, c).predict for c in range(n_out)]


def fidelity_row(sp, dataset, outputs_fn, out):
    """Per seed: (dividends only on test, surrogate on test, surrogate on the fitting sample)."""
    rows = []
    for seed in SEEDS:
        Xfit = sp.Ztr[fit_subsample(len(sp.Ztr), NFIT, seed)] if dataset == "california" else sp.Ztr
        vals = []
        for predict in outputs_fn(sp, dataset, seed):
            sur = HarsanyiARSurrogate(predict, np.zeros(sp.d), N_BINS, TAU2).fit(Xfit)
            vals.append((sur.fidelity_dividends_only(sp.Zte), sur.fidelity(sp.Zte), sur.fidelity(Xfit)))
            n_bins = sur.n_bins
        rows.append(np.mean(vals, axis=0))
        out(f"    seed {seed}: dividends only {rows[-1][0]:.4f}   surrogate test {rows[-1][1]:.4f}"
            f"   surrogate fit {rows[-1][2]:.4f}   (n_fit = {len(Xfit)}, K = {n_bins})")
    return np.array(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="xgb,net")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    out = Tee(f"table_fidelity{('_' + a.tag) if a.tag else ''}.txt")
    out(f"seeds = {list(SEEDS)}; fidelity = R^2 to the model (logit in classification, "
        f"mean over the class logits on Wine)")
    summary = []
    for model in a.models.split(","):
        fn = {"xgb": xgb_scalar_outputs, "net": net_scalar_outputs}[model]
        name = {"xgb": "XGBoost", "net": "network"}[model]
        for dataset in a.datasets.split(","):
            out(f"\n== {dataset}, {name} ==")
            R = fidelity_row(load(dataset), dataset, fn, out)
            summary.append((f"{dataset}, {name}", R))
    out(f"\n{'':22s} {'dividends only (test)':>22s} {'surrogate (test)':>18s} {'surrogate (fit)':>17s}")
    for label, R in summary:
        out(f"{label:22s} {mean_std(R[:, 0]):>22s} {mean_std(R[:, 1]):>18s} {mean_std(R[:, 2]):>17s}")
