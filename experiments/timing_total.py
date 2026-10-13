"""
Total time to explain N instances, N in 1, 10, 100, 1000, 10000, measured directly for each
method computed the way its authors compute it, with the reference library:

  surrogate model    this paper: fit on the fitting sample, then the N explanations (dividends at
                     x, corrections by interpolation);
  ALE, order <= 2    Apley & Zhu (2020): the curves of every variable and the surfaces of every
                     pair, then the N readings by interpolation; the algorithm of the R package
                     ALEPlot, run here with the grids and accumulation of harsanyi_ar (the number
                     of model evaluations, 2 per point and variable and 4 per point and pair, is
                     the algorithm's);
  Shapley-Taylor     Sundararajan et al. (2020): exactly, over all 2^d coalitions (shapiq
                     ExactComputer), and by the permutation sampling the paper proposes (shapiq
                     PermutationSamplingSTII);
  Faith-Shap         Tsai et al. (2023): exactly, the weighted least squares over all 2^d
                     coalitions (shapiq ExactComputer; a 2^d x 2^d weight matrix, so only d <= 8
                     here), and by the sampled weighted regression of the paper (shapiq
                     RegressionFSII);
  KernelSHAP         Lundberg & Lee (2017): shap.KernelExplainer with its default budget;
  TreeSHAP-IQ        Muschalik et al. (2024): shapiq.TreeExplainer, interventional, baseline as
                     reference, Shapley-Taylor of order 2 (exact, by tree traversal).

The sampled regression of Faith-Shap and KernelSHAP use the same budget, the default of
KernelSHAP, 2d + 2048 model evaluations per instance. The permutation sampling of
Shapley-Taylor is given two permutations per instance, 2 (4 C(d,2) + 2d) evaluations, the
number of permutations that budget affords at d = 20 (at d = 50 that budget does not complete
one permutation and the estimator returns zeros). All local methods explain the same instances of the same model with
the variables masked at the same baseline (the game v_x of the paper). The N instances cycle
through the 200 test instances; every method is called once before being timed. A size N is
not run when the previous run, times ten, exceeds MAX_SECONDS; it is reported as "not run"
and nothing is extrapolated. Where the exact index is available (d = 8; Shapley-Taylor at
d = 20, on two instances) the relative squared error of each estimator at its budget is
reported.

    python experiments/timing_total.py [--settings A,20,50] [--tag name]
"""
from __future__ import annotations
import time
import warnings
import numpy as np

warnings.simplefilter("ignore")
from common import load, fit_subsample, Tee, TAU2, NFIT, XGB
from harsanyi_ar import HarsanyiARSurrogate
from ale2_fidelity import ALE2


def masked_game(f, x, d):
    """Value function of the single-baseline game of the paper, for shapiq: coalitions (bool
    matrix) -> f at the masked points x_[T], the variables outside T at the baseline 0."""
    def value(coalitions):
        C = np.asarray(coalitions, bool).reshape(-1, d)
        return f(C * x[None, :])
    return value

NS = (1, 10, 100, 1000, 10_000, 100_000)
MAX_SECONDS = 3600
N_ERR = {8: 50, 20: 2}


def cyc(X, N):
    return X[np.arange(N) % len(X)]


def timed(fn):
    t = time.perf_counter(); fn(); return time.perf_counter() - t


def rel_error(approx, exact):
    keys = [k for k in exact.dict_values if len(k) >= 1]
    a = np.array([approx.dict_values.get(k, 0.0) for k in keys]); e = np.array([exact.dict_values[k] for k in keys])
    return float(np.sum((a - e) ** 2) / np.sum(e ** 2))


def run_method(out, name, explain_N, fit_seconds=0.0, note="", cap=None):
    cap = MAX_SECONDS if cap is None else cap
    explain_N(1)                                                   # warm-up call, not timed
    last = None
    for N in NS:
        if last is not None and last * 10 > cap:
            out(f"{name:48s} {N:>7d} {'--':>12s}   not run (would exceed {cap} s)")
            continue
        last = timed(lambda: explain_N(N))
        out(f"{name:48s} {N:>7d} {fit_seconds + last:>12.3f}   measured{(', ' + note) if note else ''}")


def run_setting(out, label, model, predict, Xfit, Xlocal, d, methods=None):
    """methods: subset of {ours, ale, exact, sampled, kernel, tree, errors}; None runs all."""
    import shap, shapiq
    want = lambda m: methods is None or m in methods
    from math import comb
    budget = 2 * d + 2048
    budget_perm = 2 * (4 * comb(d, 2) + 2 * d)
    out(f"\n== {label} (d = {d}; budget {budget} evaluations; permutation sampling: 2 permutations, {budget_perm} evaluations) ==")
    out(f"{'method':48s} {'N':>7s} {'seconds':>12s}   status")

    if want("ours"):
        sur = None
        def fit_ours():
            nonlocal sur
            sur = HarsanyiARSurrogate(predict, np.zeros(d), None, TAU2).fit(Xfit)
        t_fit = timed(fit_ours)
        run_method(out, "surrogate model (fit, then explanations)", lambda N: sur.explain_local(cyc(Xlocal, N)),
                   fit_seconds=t_fit, note=f"fit {t_fit:.2f} s")

    if want("ale"):
        ale = None
        def fit_ale():
            nonlocal ale
            ale = ALE2(predict).fit(Xfit)
        t_ale = timed(fit_ale)
        run_method(out, "ALE, order <= 2 (fit, then readings)", lambda N: ale.predict_surrogate(cyc(Xlocal, N)),
                   fit_seconds=t_ale, note=f"fit {t_ale:.2f} s")

    def exact_N(index):
        def go(N):
            for x in cyc(Xlocal, N):
                shapiq.ExactComputer(masked_game(predict, x, d), n_players=d)(index=index, order=2)
        return go
    if want("exact"):
        if d <= 20:
            run_method(out, "Shapley-Taylor, computed exactly", exact_N("STII"))
        else:
            out(f"{'Shapley-Taylor, computed exactly':48s} {'all':>7s} {'--':>12s}   not feasible (2^d evaluations per instance)")
        if d <= 8:
            run_method(out, "Faith-Shap, computed exactly", exact_N("FSII"))
        else:
            out(f"{'Faith-Shap, computed exactly':48s} {'all':>7s} {'--':>12s}   not feasible (2^d x 2^d weight matrix)")

    def sampled(make, b):
        def go(N):
            for x in cyc(Xlocal, N):
                make().approximate(b, masked_game(predict, x, d))
        return go
    if (want("sampled") or want("perm")) and budget < 2 ** d:
        run_method(out, "Shapley-Taylor, permutation sampling", sampled(lambda: shapiq.PermutationSamplingSTII(n=d, max_order=2, random_state=0), budget_perm))
    if (want("sampled") or want("regression")) and budget < 2 ** d:
        run_method(out, "Faith-Shap, sampled weighted regression", sampled(lambda: shapiq.RegressionFSII(n=d, max_order=2, random_state=0), budget))
    elif want("sampled"):
        out(f"{'Shapley-Taylor, permutation sampling':48s} {'all':>7s} {'--':>12s}   budget >= 2^d: the estimator is the exact index")
        out(f"{'Faith-Shap, sampled weighted regression':48s} {'all':>7s} {'--':>12s}   budget >= 2^d: the estimator is the exact index")

    if want("kernel"):
        def kernel(N):
            shap.KernelExplainer(predict, np.zeros((1, d))).shap_values(cyc(Xlocal, N), silent=True)
        run_method(out, "KernelSHAP", kernel)

    if want("tree"):
        tx = shapiq.TreeExplainer(model=model, mode="interventional", reference_dataset=np.zeros((1, d)), index="STII", max_order=2)
        def tree(N):
            for x in cyc(Xlocal, N):
                tx.explain(x)
        run_method(out, "TreeSHAP-IQ, Shapley-Taylor", tree)

    n_err = N_ERR.get(d, 0)
    if want("errors") and n_err and budget < 2 ** d:
        out(f"{'relative squared error at the budget, over':48s} {n_err:>7d}   instances:")
        for index, name, make in (("STII", "Shapley-Taylor, permutation sampling", lambda: shapiq.PermutationSamplingSTII(n=d, max_order=2, random_state=0)),
                                  ("FSII", "Faith-Shap, sampled weighted regression", lambda: shapiq.RegressionFSII(n=d, max_order=2, random_state=0))):
            if index == "FSII" and d > 8:
                out(f"{'  ' + name:48s} {'--':>7s} {'--':>12s}   exact value not available")
                continue
            errs = []
            for x in Xlocal[:n_err]:
                ex = shapiq.ExactComputer(masked_game(predict, x, d), n_players=d)(index=index, order=2)
                errs.append(rel_error(make().approximate(budget_perm if index == "STII" else budget, masked_game(predict, x, d)), ex))
            out(f"{'  ' + name:48s} {'':>7s} {np.mean(errs):>12.3f}   mean relative squared error")


if __name__ == "__main__":
    import argparse
    import xgboost as xgb
    ap = argparse.ArgumentParser(); ap.add_argument("--settings", default="A,20,50"); ap.add_argument("--tag", default="")
    ap.add_argument("--methods", default=None, help="comma-separated subset of ours,ale,exact,sampled (= perm,regression),kernel,tree,errors")
    args = ap.parse_args(); settings = args.settings.split(",")
    methods = None if args.methods is None else set(args.methods.split(","))
    out = Tee(f"timing_total{('_' + args.tag) if args.tag else ''}.txt")
    import platform
    cpu = platform.processor() or "unknown"
    try:
        for line in open("/proc/cpuinfo"):
            if line.lower().startswith("model name"):
                cpu = line.split(":", 1)[1].strip(); break
    except OSError:
        pass
    out(f"machine: {cpu}; {platform.platform()}; python {platform.python_version()}")
    out("total time on one CPU core, measured; N instances cycle through 200 test instances; one warm-up call per method")
    if "A" in settings:
        ca = load("california")
        m = xgb.XGBRegressor(random_state=0, **XGB["california"]).fit(ca.Ztr, ca.ytr)
        pred = lambda A: m.predict(np.asarray(A, np.float32))
        run_setting(out, "A. California, XGBoost", m, pred, ca.Ztr[fit_subsample(len(ca.Ztr), NFIT, 0)], ca.Zte[:200], ca.d, methods)
    for d in [int(x) for x in settings if x != "A"]:
        rs = np.random.RandomState(d)
        X = rs.normal(size=(20_000, d))
        y = (X[:, 0] * X[:, 1] + np.sin(X[:, 2]) * X[:, 3] + 0.5 * X[:, :5].sum(1) ** 2 / 5
             + 0.3 * X[:, 1] * X[:, 2] * X[:, 3] + rs.normal(scale=0.5, size=len(X)))
        md = xgb.XGBRegressor(random_state=0, **XGB["california"]).fit(X[:16_000], y[:16_000])
        pd_ = lambda A, md=md: md.predict(np.asarray(A, np.float32))
        run_setting(out, f"B. synthetic, XGBoost, d = {d}", md, pd_, X[:NFIT], X[16_000:16_200], d, methods)
