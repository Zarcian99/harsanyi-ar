"""
Cost of the surrogate model against the methods it is related to, measured as the number
of evaluations of the model (hardware independent) and as wall time, on the same model and
the same instances. Nothing else is compared: the methods return different objects (a global
decomposition plus local attributions here, interaction indices or Shapley values of one
instance there, global curves in ALE), so only their cost admits a common scale, which is
how Faith-Shap (Tsai et al. 2023), SHAP-IQ (Fumagalli et al. 2023) and shapiq (Muschalik
et al. 2024) compare estimators: budgets of model evaluations, with wall time as a secondary
measure.

Setting A. California, XGBoost (seed 0, as in the paper), d = 8. The local methods explain
the same N_LOCAL test instances with the single-baseline game v_x(T) = f(x_[T]) of the paper
(baseline 0); their cost is reported per instance. The global methods are fitted once on the
4000 fitting rows. Methods:
  ours            fit (dividends + corrections) and explanation of a new instance
                  (dividends at x + interpolation of the corrections);
  ALE, order <= 2 the global curves and surfaces of all variables and pairs, estimated with
                  the same grids, differences and accumulation (no dividends);
  Shapley-Taylor  interaction index of order 2, exact, shapiq ExactComputer;
  Faith-Shap      interaction index of order 2, exact, shapiq ExactComputer;
  KernelSHAP      Shapley values, shap.KernelExplainer, default budget 2d + 2048;
  TreeSHAP        Shapley values, shap.TreeExplainer (path-dependent value function);
  TreeSHAP-IQ     shapiq.TreeExplainer, interventional with the baseline as the only reference
                  row (the game of the paper): Shapley values and both interaction indices,
                  exact, by tree traversal without model evaluations.

Setting B. Scaling in d: XGBoost on synthetic data with d in {8, 20, 50}, 4000 fitting rows,
the same quantities; the exact indices are timed on a few instances at d = 20 and are
infeasible at d = 50 (2^50 evaluations per instance).

    python experiments/timing_comparison.py
"""
from __future__ import annotations
import itertools
import time
import warnings
import numpy as np

warnings.simplefilter("ignore")
from common import load, fit_subsample, Tee, TAU2, NFIT, XGB
from harsanyi_ar import HarsanyiARSurrogate
from harsanyi_ar.ale import quantile_edges, cell_index, build_ar1, build_ar2, terrell_scott_bins

N_LOCAL, N_KERNEL, REPS = 200, 50, 3


class Counting:
    """Wraps predict and counts the rows it is asked to evaluate."""
    def __init__(self, predict):
        self.predict, self.n = predict, 0

    def __call__(self, A):
        A = np.asarray(A, float); self.n += len(A)
        return self.predict(A)

    def reset(self):
        self.n = 0


def masked_game(f, x, d):
    """Value function of the single-baseline game of the paper, for shapiq: coalitions (bool
    matrix) -> f at the masked points x_[T]."""
    def value(coalitions):
        C = np.asarray(coalitions, bool).reshape(-1, d)
        return f(C * x[None, :])
    return value


def ale_order2(f, Xfit, n_bins):
    """Pure ALE of order <= 2 of f on the fitting sample, with the grids, differences and
    accumulation of the package but no dividends; returns nothing, it is timed."""
    n, d = Xfit.shape
    edges = [quantile_edges(Xfit[:, j], n_bins) for j in range(d)]
    idx = [cell_index(Xfit[:, j], edges[j]) for j in range(d)]
    for j in range(d):
        A, B = Xfit.copy(), Xfit.copy()
        A[:, j], B[:, j] = edges[j][idx[j] + 1], edges[j][idx[j]]
        build_ar1(edges[j], idx[j], f(A) - f(B), 0.0, TAU2, False)
    for i, j in itertools.combinations(range(d), 2):
        ei, ej, iid, jid = edges[i], edges[j], idx[i], idx[j]
        second = np.zeros(n)
        for ai, aj, sgn in ((ei[iid + 1], ej[jid + 1], 1.0), (ei[iid + 1], ej[jid], -1.0),
                            (ei[iid], ej[jid + 1], -1.0), (ei[iid], ej[jid], 1.0)):
            Y = Xfit.copy(); Y[:, i] = ai; Y[:, j] = aj
            second += sgn * f(Y)
        build_ar2(i, j, Xfit[:, i], Xfit[:, j], ei, ej, iid, jid, second, 0.0, 0.0, TAU2, False)


def time_it(fn, reps=REPS):
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def run_setting(out, label, model_predict, Xfit, Xlocal, d, exact_instances=None):
    import shap, shapiq
    f = Counting(model_predict)
    out(f"\n== {label} (d = {d}, {len(Xfit)} fitting rows, {len(Xlocal)} explained instances) ==")
    out(f"{'method':38s} {'evaluations':>16s} {'seconds':>12s}   note")

    # ours: fit
    f.reset(); t = time_it(lambda: HarsanyiARSurrogate(f, np.zeros(d), None, TAU2).fit(Xfit), 1)
    out(f"{'ours, fit (global model)':38s} {f.n:>16,d} {t:>12.2f}   once")
    sur = HarsanyiARSurrogate(f, np.zeros(d), None, TAU2).fit(Xfit)
    # ours: explain new instances
    f.reset(); t = time_it(lambda: sur.explain_local(Xlocal), 1)
    out(f"{'ours, explain (per instance)':38s} {f.n / len(Xlocal):>16,.0f} {t / len(Xlocal):>12.4f}   dividends at x + interpolation")

    # pure ALE of order <= 2
    K = terrell_scott_bins(len(Xfit))
    f.reset(); t = time_it(lambda: ale_order2(f, Xfit, K), 1)
    out(f"{'ALE order <= 2 (global curves only)':38s} {f.n:>16,d} {t:>12.2f}   once; no local explanation")

    # exact interaction indices via shapiq
    inst = Xlocal if exact_instances is None else Xlocal[:exact_instances]
    for index, name in (("STII", "Shapley-Taylor, order 2, exact"), ("FSII", "Faith-Shap, order 2, exact")):
        if d > 20 or (d > 8 and index == "FSII"):
            # FSII in shapiq solves a weighted least squares with a 2^d x 2^d weight matrix:
            # 8 TiB at d = 20; at d = 50, 2^50 game evaluations for either index
            out(f"{name:38s} {2 ** d:>16,d} {'--':>12s}   per instance; infeasible")
            continue
        f.reset(); t0 = time.perf_counter()
        for x in inst:
            shapiq.ExactComputer(masked_game(f, x, d), n_players=d)(index=index, order=2)
        t = time.perf_counter() - t0
        out(f"{name:38s} {f.n / len(inst):>16,.0f} {t / len(inst):>12.4f}   per instance; 2^d game evaluations")

    # KernelSHAP (single-baseline background) and TreeSHAP
    kinst = Xlocal[:N_KERNEL]
    f.reset(); t0 = time.perf_counter()
    ke = shap.KernelExplainer(f, np.zeros((1, d)))
    ke.shap_values(kinst, silent=True)
    t = time.perf_counter() - t0
    out(f"{'KernelSHAP (Shapley values)':38s} {f.n / len(kinst):>16,.0f} {t / len(kinst):>12.4f}   per instance; budget 2d + 2048")
    return sur


def tree_shap_row(out, model, Xlocal, d):
    """Tree-specific exact algorithms: TreeSHAP (shap, path-dependent value function) and
    TreeSHAP-IQ (shapiq, interventional with the baseline as the only reference row, i.e. the
    game of the paper): Shapley values and the two interaction indices, no model evaluations."""
    import shap, shapiq
    te = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    t = time_it(lambda: te.shap_values(Xlocal), 3)
    out(f"{'TreeSHAP (Shapley values, path-dep.)':38s} {'0':>16s} {t / len(Xlocal):>12.4f}   per instance; tree traversal")
    for index, order, name in (("SV", 1, "TreeSHAP-IQ, Shapley values"), ("STII", 2, "TreeSHAP-IQ, Shapley-Taylor, order 2"),
                               ("FSII", 2, "TreeSHAP-IQ, Faith-Shap, order 2")):
        tx = shapiq.TreeExplainer(model=model, mode="interventional", reference_dataset=np.zeros((1, d)),
                                  index=index, max_order=order)
        t0 = time.perf_counter()
        for x in Xlocal:
            tx.explain(x)
        out(f"{name:38s} {'0':>16s} {(time.perf_counter() - t0) / len(Xlocal):>12.4f}   per instance; tree traversal, same game")


if __name__ == "__main__":
    import argparse
    import xgboost as xgb
    ap = argparse.ArgumentParser(); ap.add_argument("--settings", default="A,8,20,50"); ap.add_argument("--tag", default="")
    args = ap.parse_args(); settings = args.settings.split(",")
    out = Tee(f"timing_comparison{('_' + args.tag) if args.tag else ''}.txt")
    out(f"wall times are medians of {REPS} runs (fits: single run) on one CPU core; evaluations = rows passed to the model")

    # ---- Setting A: California, XGBoost
    ca = load("california") if "A" in settings else None
    if ca is not None:
        m = xgb.XGBRegressor(random_state=0, **XGB["california"]).fit(ca.Ztr, ca.ytr)
        pred = lambda A: m.predict(np.asarray(A, np.float32))
        Xfit = ca.Ztr[fit_subsample(len(ca.Ztr), NFIT, 0)]
        Xlocal = ca.Zte[:N_LOCAL]
        run_setting(out, "A. California, XGBoost", pred, Xfit, Xlocal, ca.d)
        tree_shap_row(out, m, Xlocal, ca.d)

    # ---- Setting B: scaling in d on synthetic data
    for d in [int(x) for x in settings if x != "A"]:
        rs = np.random.RandomState(d)
        X = rs.normal(size=(20_000, d))
        y = (X[:, 0] * X[:, 1] + np.sin(X[:, 2]) * X[:, 3] + 0.5 * X[:, :5].sum(1) ** 2 / 5
             + 0.3 * X[:, 1] * X[:, 2] * X[:, 3] + rs.normal(scale=0.5, size=len(X)))
        md = xgb.XGBRegressor(random_state=0, **XGB["california"]).fit(X[:16_000], y[:16_000])
        pd_ = lambda A, md=md: md.predict(np.asarray(A, np.float32))
        run_setting(out, f"B. synthetic, XGBoost, d = {d}", pd_, X[:NFIT], X[16_000:16_000 + N_LOCAL], d,
                    exact_instances=1 if d == 20 else None)
        tree_shap_row(out, md, X[16_000:16_000 + N_LOCAL], d)
