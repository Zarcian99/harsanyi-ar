"""
Order-2 models built from Shapley-Taylor and Faith-Shap, against the dividends, ALE and the
surrogate model: XGBoost and the softplus network on California, cancer and Wine, the five seeds
and the fitting rows of the paper (4000 on California, the whole training set otherwise);
fidelity = R^2 to the model on the test set, averaged over the class logits on Wine. The exact
indices need the model at all 2^d coalitions of every fitting row, which is feasible here:
d = 8 (California), 10 (cancer) and 13 (Wine).

The indices at a point x are not functions of their own variables: each order-two value also
carries a fixed share of the higher-order dividends, which depends on all of x. To obtain an
additive model of order 2 from them, each index is projected onto its own variables,
g_S(x_S) = E[I_S(X) | X_S = x_S]. The dividends of order <= 2 are kept exact, as in the surrogate
model, and only the part the index adds to them, I_S - d_S, is projected: on the same grids as
the surrogate model (the same quantile edges, number of bins and baseline node), by cell means
placed at the cell centres and interpolated linearly (bilinearly for pairs), empty cells at the
overall mean.
Each model gets the same kind of intercept as the surrogate: the mean residual on the fitting rows.

Exact indices from the dividends of the masked game (Möbius transform over the 2^8 coalitions),
checked against shapiq.ExactComputer on a few instances:
  Shapley-Taylor, k = 2:  I_j = d_j,   I_S = sum_{T >= S} d_T / C(|T|, 2) for |S| = 2;
  Faith-Shap, k = 2:      E_S = d_S + (-1)^(2-|S|) |S|/(2+|S|) C(2,|S|) sum_{T > S, |T| > 2} C(|T|-1, 2) / C(|T|+1, 2+|S|) d_T.

    python experiments/projected_indices.py
"""
from __future__ import annotations
import itertools
import warnings
from math import comb
import numpy as np

warnings.simplefilter("ignore")
from common import load, fit_subsample, train_mlp, mean_std, Tee, N_BINS, TAU2, NFIT, SEEDS, XGB
from harsanyi_ar import HarsanyiARSurrogate, TorchModel
from harsanyi_ar.ale import quantile_edges, cell_index, terrell_scott_bins, bilinear
from harsanyi_ar.numerics import r2_score
from ale2_fidelity import ALE2

K_ORDER = 2


def game_values(predict, X, d):
    """f at the 2^d masked points of every row: array (n, 2^d), column m = coalition with bit mask m."""
    masks = np.array([[(m >> j) & 1 for j in range(d)] for m in range(2 ** d)], dtype=float)
    P = (X[:, None, :] * masks[None, :, :]).reshape(-1, d)
    return predict(P).reshape(len(X), 2 ** d)


def moebius(V, d):
    """Harsanyi dividends of every coalition, by the fast Möbius transform over bit masks."""
    D = V.copy()
    for j in range(d):
        bit = 1 << j
        for m in range(2 ** d):
            if m & bit:
                D[:, m] -= D[:, m ^ bit]
    return D


def index_matrices(d):
    """Matrices mapping the 2^d dividends to the order-1 and order-2 values of Shapley-Taylor and
    Faith-Shap (k = 2). Rows: the sets S of size 1 and 2, in the order of `sets`."""
    sets = [S for s in (1, 2) for S in itertools.combinations(range(d), s)]
    mask = lambda S: sum(1 << j for j in S)
    sz = [bin(m).count("1") for m in range(2 ** d)]
    ST = np.zeros((len(sets), 2 ** d)); FS = np.zeros((len(sets), 2 ** d))
    for r, S in enumerate(sets):
        ms, s = mask(S), len(S)
        ST[r, ms] = 1.0; FS[r, ms] = 1.0
        for m in range(2 ** d):
            if (m & ms) == ms and m != ms:
                t = sz[m]
                if s == 2:
                    ST[r, m] += 1.0 / comb(t, 2)
                if t > K_ORDER:
                    FS[r, m] += ((-1) ** (K_ORDER - s) * s / (K_ORDER + s) * comb(K_ORDER, s)
                                 * comb(t - 1, K_ORDER) / comb(t + K_ORDER - 1, K_ORDER + s))
    return sets, ST, FS


def check_against_shapiq(predict, X, d, sets, ST, FS):
    import shapiq
    D = moebius(game_values(predict, X, d), d)
    for i, x in enumerate(X):
        def game(C, x=x):
            C = np.asarray(C, bool).reshape(-1, d)
            return predict(C * x[None, :])
        ec = shapiq.ExactComputer(game, n_players=d)
        for name, M in (("STII", ST), ("FSII", FS)):
            ref = ec(index=name, order=2).dict_values
            mine = M @ D[i]
            for r, S in enumerate(sets):
                assert np.isclose(mine[r], ref.get(S, 0.0), atol=1e-6 * max(1.0, abs(ref.get(S, 0.0)))), (name, S)


class Projection:
    """Additive model of order 2 built from an index: exact dividends plus the projection of what
    the index adds to them onto the variables of each set."""

    def __init__(self, sets, M, K):
        self.sets, self.M, self.K = sets, M, K

    def fit(self, Xfit, D, f0, ffit, node=None):
        d = Xfit.shape[1]
        self.edges = [quantile_edges(Xfit[:, j], self.K, node) for j in range(d)]
        self.centres = [0.5 * (e[:-1] + e[1:]) for e in self.edges]
        idx = [cell_index(Xfit[:, j], self.edges[j]) for j in range(d)]
        I = D @ self.M.T                                    # (n, sets): index values at the fitting rows
        div = D[:, [sum(1 << j for j in S) for S in self.sets]]
        R = I - div                                         # what the index adds to the dividends
        self.tables = []
        for r, S in enumerate(self.sets):
            shape = tuple(len(self.centres[j]) for j in S)
            flat = np.ravel_multi_index(tuple(idx[j] for j in S), shape)
            cnt = np.bincount(flat, minlength=int(np.prod(shape)))
            sm = np.bincount(flat, weights=R[:, r], minlength=int(np.prod(shape)))
            g = np.where(cnt > 0, sm / np.maximum(cnt, 1), R[:, r].mean()).reshape(shape)
            self.tables.append(g)
        self.f0 = f0
        self.c = 0.0
        self.c = float(np.mean(ffit - self._predict(Xfit, div)))
        return self

    def _projected(self, X):
        out = np.zeros(len(X))
        for (S, g) in zip(self.sets, self.tables):
            if len(S) == 1:
                j = S[0]
                out += np.interp(X[:, j], self.centres[j], g)
            else:
                i, j = S
                out += bilinear(self.centres[i], self.centres[j], g, X[:, i], X[:, j])
        return out

    def _predict(self, X, div):
        return self.f0 + div.sum(axis=1) + self._projected(X) + self.c

    def predict(self, X, div):
        return self._predict(X, div)


def dividends_up_to_2(predict, X, d, sets):
    """Exact dividends of order 1 and 2 at the rows of X (masked game restricted to |T| <= 2)."""
    f0 = float(predict(np.zeros((1, d)))[0])
    single = {j: predict(np.where(np.arange(d) == j, X, 0.0)) for j in range(d)}
    out = np.zeros((len(X), len(sets)))
    for r, S in enumerate(sets):
        if len(S) == 1:
            out[:, r] = single[S[0]] - f0
        else:
            i, j = S
            keep = np.isin(np.arange(d), S)
            out[:, r] = predict(np.where(keep, X, 0.0)) - single[i] - single[j] + f0
    return out, f0


def outputs(sp, dataset, which, seed):
    """The functions to explain: the regression output, the binary logit, or the three logits."""
    if which == "net":
        n_out = 3 if sp.task == "multiclass" else 1
        net = train_mlp(sp, dataset, "softplus", seed)
        return [TorchModel(net, sp.d, c).predict for c in range(n_out)]
    import xgboost as xgb
    if sp.task == "regression":
        m = xgb.XGBRegressor(random_state=seed, **XGB[dataset]).fit(sp.Ztr, sp.ytr)
        return [lambda A, m=m: m.predict(np.asarray(A, np.float32)).astype(float)]
    ev = "logloss" if sp.task == "binary" else "mlogloss"
    m = xgb.XGBClassifier(random_state=seed, eval_metric=ev, **XGB[dataset]).fit(sp.Ztr, sp.ytr)
    margin = lambda A: m.predict(np.asarray(A, np.float32), output_margin=True).astype(float)
    if sp.task == "multiclass":
        return [(lambda A, c=c: margin(A)[:, c]) for c in range(3)]
    return [margin]


def fidelities(sp, predict, Xfit, with_indices, sets=None, ST=None, FS=None):
    d = sp.d
    fte = predict(sp.Zte)
    sur = HarsanyiARSurrogate(predict, np.zeros(d), N_BINS, TAU2).fit(Xfit)
    r = {"div": r2_score(fte, sur.predict_dividends_only(sp.Zte)),
         "ale": ALE2(predict).fit(Xfit).fidelity(sp.Zte),
         "sur": sur.fidelity(sp.Zte), "st": np.nan, "fs": np.nan}
    if with_indices:
        D = moebius(game_values(predict, Xfit, d), d)
        _, f0 = dividends_up_to_2(predict, Xfit, d, sets)
        div_te, _ = dividends_up_to_2(predict, sp.Zte, d, sets)
        K = terrell_scott_bins(len(Xfit)) if N_BINS is None else N_BINS
        for key, M in (("st", ST), ("fs", FS)):
            pr = Projection(sets, M, K).fit(Xfit, D, f0, predict(Xfit), node=sur._node(0.0))
            r[key] = r2_score(fte, pr.predict(sp.Zte, div_te))
    return r


def run(seeds, tag="", datasets=("california", "cancer", "wine")):
    out = Tee(f"projected_indices{('_' + tag) if tag else ''}.txt")
    out(f"seeds = {seeds}; fidelity = R^2 to the model on the test set (mean over the logits on Wine)")
    out(f"\n{'':22s} {'dividends':>16s} {'ALE':>16s} {'Shapley-Taylor':>16s} {'Faith-Shap':>16s} {'surrogate':>16s}")
    out(f"{'':22s} {'':>16s} {'':>16s} {'projected':>16s} {'projected':>16s} {'':>16s}")
    for dataset in datasets:
        sp = load(dataset); d = sp.d
        with_indices = d <= 13
        sets, ST, FS = index_matrices(d) if with_indices else (None, None, None)
        checked = False
        for which in ("xgb", "net"):
            res = {k: [] for k in ("div", "ale", "st", "fs", "sur")}
            for seed in seeds:
                preds = outputs(sp, dataset, which, seed)
                Xfit = sp.Ztr[fit_subsample(len(sp.Ztr), NFIT, seed)] if dataset == "california" else sp.Ztr
                if with_indices and not checked:
                    check_against_shapiq(preds[0], Xfit[:3], d, sets, ST, FS); checked = True
                rs_ = [fidelities(sp, p, Xfit, with_indices, sets, ST, FS) for p in preds]
                for k in res:
                    res[k].append(float(np.mean([r[k] for r in rs_])))
                out(f"  {dataset} {which} seed {seed}: " + "  ".join(f"{k} {v[-1]:.4f}" for k, v in res.items()))
            name = f"{dataset}, {'XGBoost' if which == 'xgb' else 'network'}"
            cell = lambda k: mean_std(res[k]) if not np.isnan(res[k][0]) else "--"
            out(f"{name:22s} " + " ".join(f"{cell(k):>16s}" for k in ("div", "ale", "st", "fs", "sur")))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--tag", default="")
    ap.add_argument("--datasets", default="california,cancer,wine")
    a = ap.parse_args()
    run([int(s) for s in a.seeds.split(",")], a.tag, a.datasets.split(","))
