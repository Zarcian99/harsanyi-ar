"""
Table "Predictive performance against the data" (tab:predictores) and the fidelity
figures quoted in the text.

Rows (select with --rows, comma separated; default: all but ebm):
  baselines  linear/logistic, degree-2 polynomial, GAM with splines     (single fit)
  xgb        XGBoost black box and our system on XGBoost
  net        our system on the softplus network
  ebm        Explainable Boosting Machine (slow: several minutes per California fit)

Models with randomness: mean +- std over --seeds (model seed and, on California, fitting
subsample); fixed split. Linear, polynomial and spline baselines are deterministic (single
fit). Classification surrogates decompose the logit (one surrogate per class logit on Wine;
predicted class = argmax of the three surrogates, probabilities = softmax). On Wine the
log-loss is reported next to the accuracy.

    python experiments/table3_predictors.py
    python experiments/table3_predictors.py --rows ebm
"""
from __future__ import annotations
import argparse
import warnings
import numpy as np

warnings.simplefilter("ignore")
from common import (load, fit_subsample, train_mlp, mean_std, Tee, N_BINS, TAU2,
                    NFIT, XGB_CALIFORNIA, XGB_CANCER, XGB_WINE)
from harsanyi_ar import HarsanyiARSurrogate, r2_score
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss


def softmax(L):
    L = L - L.max(axis=1, keepdims=True)
    E = np.exp(L)
    return E / E.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------- baselines
def baselines(out):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures, SplineTransformer
    from sklearn.linear_model import RidgeCV, LogisticRegression

    out("\n== Transparent baselines (single fit) ==")
    ca = load("california")
    alphas = np.logspace(-3, 3, 13)
    reg = {
        "linear": make_pipeline(StandardScaler(), RidgeCV(alphas=alphas)),
        "degree-2 polynomial": make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False),
                                             RidgeCV(alphas=alphas)),
        "GAM with splines": make_pipeline(StandardScaler(), SplineTransformer(n_knots=12, degree=3),
                                          RidgeCV(alphas=alphas)),
    }
    for name, m in reg.items():
        m.fit(ca.Xtr, ca.ytr)
        out(f"  California  {name:22s} R2       = {r2_score(ca.yte, m.predict(ca.Xte)):.3f}")

    def clf(kind):
        lr = LogisticRegression(max_iter=5000, C=1.0)
        if kind == "logistic":
            return make_pipeline(StandardScaler(), lr)
        if kind == "degree-2 polynomial":
            return make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False), lr)
        return make_pipeline(StandardScaler(), SplineTransformer(n_knots=8, degree=3), lr)

    bc, wi = load("cancer"), load("wine")
    for kind in ("logistic", "degree-2 polynomial", "GAM with splines"):
        m = clf(kind).fit(bc.Xtr, bc.ytr)
        out(f"  Cancer      {kind:22s} AUC      = {roc_auc_score(bc.yte, m.predict_proba(bc.Xte)[:, 1]):.3f}")
        m = clf(kind).fit(wi.Xtr, wi.ytr)
        out(f"  Wine        {kind:22s} accuracy = {accuracy_score(wi.yte, m.predict(wi.Xte)):.3f}"
            f"   log-loss = {log_loss(wi.yte, m.predict_proba(wi.Xte), labels=[0, 1, 2]):.3f}")


# ---------------------------------------------------------------- XGBoost
def xgb_rows(out, seeds):
    import xgboost as xgb

    out(f"\n== XGBoost black box and our system on XGBoost  ==")
    ca = load("california")
    R = {k: [] for k in ("bb", "sys", "fid", "fid_div", "kappa")}
    for seed in seeds:
        m = xgb.XGBRegressor(random_state=seed, **XGB_CALIFORNIA).fit(ca.Ztr, ca.ytr)
        predict = lambda A, m=m: m.predict(np.asarray(A, np.float32))
        Xfit = ca.Ztr[fit_subsample(len(ca.Ztr), NFIT, seed)]
        sur = HarsanyiARSurrogate(predict, np.zeros(ca.d), N_BINS, TAU2).fit(Xfit)
        R["bb"].append(r2_score(ca.yte, predict(ca.Zte)))
        R["sys"].append(r2_score(ca.yte, sur.predict(ca.Zte)))
        R["fid"].append(sur.fidelity(ca.Zte))
        R["fid_div"].append(sur.fidelity_dividends_only(ca.Zte))
        R["kappa"].append(sur.kappa)
        out(f"  seed {seed}: XGBoost R2={R['bb'][-1]:.4f}  system R2={R['sys'][-1]:.4f}  "
            f"fidelity={R['fid'][-1]:.4f}  (dividends only {R['fid_div'][-1]:.4f})  kappa={sur.kappa:+.4f}")
    out(f"  California  XGBoost (black box)          R2 = {mean_std(R['bb'])}")
    out(f"  California  our system, XGBoost     R2 = {mean_std(R['sys'])}")
    out(f"  California  fidelity to XGBoost: dividends only {mean_std(R['fid_div'])}"
        f"  ->  surrogate {mean_std(R['fid'])}")

    bc = load("cancer")
    R = {k: [] for k in ("bb", "sys", "fid")}
    for seed in seeds:
        m = xgb.XGBClassifier(random_state=seed, eval_metric="logloss", **XGB_CANCER).fit(bc.Ztr, bc.ytr)
        margin = lambda A, m=m: m.predict(np.asarray(A, np.float32), output_margin=True)
        sur = HarsanyiARSurrogate(margin, np.zeros(bc.d), N_BINS, TAU2).fit(bc.Ztr)
        R["bb"].append(roc_auc_score(bc.yte, margin(bc.Zte)))
        R["sys"].append(roc_auc_score(bc.yte, sur.predict(bc.Zte)))
        R["fid"].append(sur.fidelity(bc.Zte))
    out(f"  Cancer      XGBoost (black box)          AUC = {mean_std(R['bb'])}")
    out(f"  Cancer      our system, XGBoost     AUC = {mean_std(R['sys'])}"
        f"   (fidelity on the logit {mean_std(R['fid'])})")

    wi = load("wine")
    R = {k: [] for k in ("bb", "sys", "bb_ll", "sys_ll", "fid")}
    for seed in seeds:
        m = xgb.XGBClassifier(random_state=seed, eval_metric="mlogloss", **XGB_WINE).fit(wi.Ztr, wi.ytr)
        S = np.zeros((len(wi.Zte), 3)); fid = []
        for c in range(3):
            margin_c = lambda A, c=c: m.predict(np.asarray(A, np.float32), output_margin=True)[:, c]
            sur = HarsanyiARSurrogate(margin_c, np.zeros(wi.d), N_BINS, TAU2).fit(wi.Ztr)
            S[:, c] = sur.predict(wi.Zte); fid.append(sur.fidelity(wi.Zte))
        Zte32 = np.asarray(wi.Zte, np.float32)
        R["bb"].append(accuracy_score(wi.yte, m.predict(Zte32)))
        R["bb_ll"].append(log_loss(wi.yte, m.predict_proba(Zte32), labels=[0, 1, 2]))
        R["sys"].append(accuracy_score(wi.yte, S.argmax(1)))
        R["sys_ll"].append(log_loss(wi.yte, softmax(S), labels=[0, 1, 2]))
        R["fid"].append(np.mean(fid))
    out(f"  Wine        XGBoost (black box)          accuracy = {mean_std(R['bb'])}   log-loss = {mean_std(R['bb_ll'])}")
    out(f"  Wine        our system, XGBoost     accuracy = {mean_std(R['sys'])}   log-loss = {mean_std(R['sys_ll'])}"
        f"   (mean fidelity on the logits {mean_std(R['fid'])})")


# ---------------------------------------------------------------- network
def net_rows(out, seeds):
    from harsanyi_ar import TorchModel

    out(f"\n== Our system on the softplus network  ==")
    ca = load("california")
    R = {k: [] for k in ("net", "sys", "fid", "fid_div")}
    for seed in seeds:
        model = TorchModel(train_mlp(ca, "california", "softplus", seed), ca.d)
        Xfit = ca.Ztr[fit_subsample(len(ca.Ztr), NFIT, seed)]
        sur = HarsanyiARSurrogate(model.predict, np.zeros(ca.d), N_BINS, TAU2).fit(Xfit)
        R["net"].append(r2_score(ca.yte, model.predict(ca.Zte)))
        R["sys"].append(r2_score(ca.yte, sur.predict(ca.Zte)))
        R["fid"].append(sur.fidelity(ca.Zte))
        R["fid_div"].append(sur.fidelity_dividends_only(ca.Zte))
        out(f"  seed {seed}: network R2={R['net'][-1]:.4f}  system R2={R['sys'][-1]:.4f}  "
            f"fidelity={R['fid'][-1]:.4f}  (dividends only {R['fid_div'][-1]:.4f})  kappa={sur.kappa:+.4f}")
    out(f"  California  network                       R2 = {mean_std(R['net'])}")
    out(f"  California  our system, network     R2 = {mean_std(R['sys'])}")
    out(f"  California  fidelity to the network: dividends only {mean_std(R['fid_div'])}"
        f"  ->  surrogate {mean_std(R['fid'])}")

    bc = load("cancer")
    R = {k: [] for k in ("bb", "sys", "fid")}
    for seed in seeds:
        model = TorchModel(train_mlp(bc, "cancer", "softplus", seed), bc.d)
        sur = HarsanyiARSurrogate(model.predict, np.zeros(bc.d), N_BINS, TAU2).fit(bc.Ztr)
        R["bb"].append(roc_auc_score(bc.yte, model.predict(bc.Zte)))
        R["sys"].append(roc_auc_score(bc.yte, sur.predict(bc.Zte)))
        R["fid"].append(sur.fidelity(bc.Zte))
    out(f"  Cancer      network                       AUC = {mean_std(R['bb'])}")
    out(f"  Cancer      our system, network     AUC = {mean_std(R['sys'])}"
        f"   (fidelity on the logit {mean_std(R['fid'])})")

    wi = load("wine")
    R = {k: [] for k in ("bb", "sys", "bb_ll", "sys_ll", "fid")}
    for seed in seeds:
        net = train_mlp(wi, "wine", "softplus", seed)
        S = np.zeros((len(wi.Zte), 3)); L = np.zeros_like(S); fid = []
        for c in range(3):
            model = TorchModel(net, wi.d, output_index=c)
            sur = HarsanyiARSurrogate(model.predict, np.zeros(wi.d), N_BINS, TAU2).fit(wi.Ztr)
            S[:, c], L[:, c] = sur.predict(wi.Zte), model.predict(wi.Zte); fid.append(sur.fidelity(wi.Zte))
        R["bb"].append(accuracy_score(wi.yte, L.argmax(1)))
        R["bb_ll"].append(log_loss(wi.yte, softmax(L), labels=[0, 1, 2]))
        R["sys"].append(accuracy_score(wi.yte, S.argmax(1)))
        R["sys_ll"].append(log_loss(wi.yte, softmax(S), labels=[0, 1, 2]))
        R["fid"].append(np.mean(fid))
    out(f"  Wine        network                       accuracy = {mean_std(R['bb'])}   log-loss = {mean_std(R['bb_ll'])}")
    out(f"  Wine        our system, network     accuracy = {mean_std(R['sys'])}   log-loss = {mean_std(R['sys_ll'])}"
        f"   (mean fidelity on the logits {mean_std(R['fid'])})")


# ---------------------------------------------------------------- EBM
def ebm_rows(out, seeds):
    from interpret.glassbox import ExplainableBoostingRegressor, ExplainableBoostingClassifier

    out(f"\n== EBM (default hyperparameters) ==")
    bc, wi = load("cancer"), load("wine")
    auc = [roc_auc_score(bc.yte, ExplainableBoostingClassifier(random_state=s).fit(bc.Xtr, bc.ytr)
                         .predict_proba(bc.Xte)[:, 1]) for s in seeds]
    out(f"  Cancer      EBM   AUC      = {mean_std(auc)}")
    acc, ll = [], []
    for s in seeds:
        m = ExplainableBoostingClassifier(random_state=s).fit(wi.Xtr, wi.ytr)
        acc.append(accuracy_score(wi.yte, m.predict(wi.Xte)))
        ll.append(log_loss(wi.yte, m.predict_proba(wi.Xte), labels=[0, 1, 2]))
    out(f"  Wine        EBM   accuracy = {mean_std(acc)}   log-loss = {mean_std(ll)}")
    ca = load("california"); r2 = []
    for seed in seeds:
        m = ExplainableBoostingRegressor(random_state=seed).fit(ca.Xtr, ca.ytr)
        r2.append(r2_score(ca.yte, m.predict(ca.Xte)))
        out(f"  seed {seed}: EBM California R2={r2[-1]:.4f}")
    out(f"  California  EBM   R2       = {mean_std(r2)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="baselines,xgb,net")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    rows = a.rows.split(",")
    out = Tee(f"table3_predictors{('_' + a.tag) if a.tag else ''}.txt")
    out(f"seeds = {seeds}")
    if "baselines" in rows: baselines(out)
    if "xgb" in rows: xgb_rows(out, seeds)
    if "net" in rows: net_rows(out, seeds)
    if "ebm" in rows: ebm_rows(out, seeds)
