"""
Order-k surrogate model, for any truncation order k (Sections "The order-k surrogate model"
and "Estimation" of the paper). `HarsanyiARSurrogate` in surrogate.py is the order-2
implementation used for the experiments; this module generalises every step to k and, at
k = 2, reproduces it.

Pieces, for every set S with 1 <= |S| <= k:
  d_S    Harsanyi dividend, eq. (dividendo), from the model at the masked points x_[T],
         T subset of S;
  AR_S   accumulated residual: the ALE component of r = f - sum_{|S|<=k} d_S on S, estimated
         on the quantile grid of S from the differences of r across the cells
         (ap_ale_tfc), obtained from the model alone through the identity (ap_deriv_cerrada);
         cell means (ap_ale_incr), accumulation (ap_ale_acum), multilinear interpolation
         (ap_ale_interp), re-anchoring at the baseline (ap_ale_anclaje), and the discount of
         the lower orders by the estimated ALE operators (ap_ar_centrado), applied
         recursively down to order one;
  kappa  the mean over the fitting sample of r - sum_S AR_S.
The surrogate is f(0) + kappa + sum_S [d_S + AR_S]; its Shapley values share each component
equally among the members of S, eq. (shapley).

Cost: the fit evaluates the model O(n d^k) times (dividends and differences at the n fitting
points); explaining a new point costs sum_{s<=k} C(d, s) evaluations (its dividends).
"""
from __future__ import annotations
import itertools
from math import comb
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from .ale import quantile_edges, cell_index, terrell_scott_bins
from .numerics import cumsum0, sum_terms, r2_score
from .shrinkage import pooled_variance, shrink_increments

Predict = Callable[[np.ndarray], np.ndarray]
SetT = Tuple[int, ...]


def _call(predict: Predict, X: np.ndarray) -> np.ndarray:
    return np.asarray(predict(np.asarray(X, float)), float).reshape(-1)


def _subsets(S: SetT, proper: bool = False, nonempty: bool = False):
    """Subsets of S in increasing size (tuples in the order of S)."""
    top = len(S) - 1 if proper else len(S)
    for s in range(1 if nonempty else 0, top + 1):
        for T in itertools.combinations(S, s):
            yield T


# --------------------------------------------------------------------------- #
#  Piecewise-multilinear functions on a product grid
# --------------------------------------------------------------------------- #
class GridFunction:
    """A function of x_S given by its values at the nodes of the product grid of S and
    extended between nodes by the multilinear interpolant (ap_ale_interp); constant beyond
    the outer nodes, as np.interp."""

    def __init__(self, vars_: SetT, nodes: List[np.ndarray], values: np.ndarray):
        self.vars, self.nodes, self.values = tuple(vars_), [np.asarray(n, float) for n in nodes], np.asarray(values, float)
        assert self.values.shape == tuple(n.size for n in self.nodes)

    def at(self, Z: np.ndarray) -> np.ndarray:
        """Values at the coordinates Z (N, |S|), in the order of self.vars."""
        Z = np.asarray(Z, float).reshape(-1, len(self.vars))
        base, frac = [], []
        for a, n in enumerate(self.nodes):
            g = np.interp(Z[:, a], n, np.arange(n.size))
            i0 = np.clip(np.floor(g).astype(int), 0, n.size - 2)
            base.append(i0); frac.append(g - i0)
        out = np.zeros(len(Z))
        for corner in itertools.product((0, 1), repeat=len(self.vars)):
            w = np.ones(len(Z)); idx = []
            for a, c in enumerate(corner):
                w = w * (frac[a] if c else 1.0 - frac[a]); idx.append(base[a] + c)
            out += w * self.values[tuple(idx)]
        return out

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Values at full points X (N, d): reads the coordinates of self.vars."""
        return self.at(np.asarray(X, float)[:, list(self.vars)])

    def mesh(self) -> np.ndarray:
        """All nodes of the grid as a (prod(len), |S|) array, in C order of self.values."""
        M = np.meshgrid(*self.nodes, indexing="ij")
        return np.column_stack([m.ravel() for m in M])


# --------------------------------------------------------------------------- #
#  The surrogate model
# --------------------------------------------------------------------------- #
class HarsanyiARSurrogateK:
    """
    predict   : model, (N, d) -> (N,).
    baseline  : (d,) point where the variables are masked (the mean of the standardised data).
    order     : truncation order k.
    n_bins    : bins per variable of the quantile grids; None applies the Terrell-Scott rule.
    shrink    : Fay-Herriot shrinkage of the cell increments (off by default).
    max_rows  : rows per model call, to bound memory.
    """

    def __init__(self, predict: Predict, baseline: np.ndarray, order: int = 2, n_bins: int | None = None,
                 tau2_method: str = "adm", shrink: bool = False, fit_intercept: bool = True,
                 max_rows: int = 2_000_000):
        self._predict_model = predict
        self.baseline = np.asarray(baseline, float).reshape(-1)
        self.d = self.baseline.size
        self.k = int(order)
        if not 1 <= self.k <= self.d:
            raise ValueError("order must be between 1 and d")
        self.n_bins_arg = n_bins
        self.tau2_method, self.shrink, self.fit_intercept = tau2_method, bool(shrink), bool(fit_intercept)
        self.max_rows = int(max_rows)
        self.sets: List[SetT] = [S for s in range(1, self.k + 1) for S in itertools.combinations(range(self.d), s)]
        self._fitted = False

    # ------------------------------------------------------------- dividends
    def _masked_values(self, X: np.ndarray) -> Dict[SetT, np.ndarray]:
        """f at the masked points x_[T] for every T with |T| <= k, including T = () -> f(0)."""
        X = np.asarray(X, float)
        sets = [()] + self.sets
        out = {T: np.empty(len(X)) for T in sets}
        rows = max(1, self.max_rows // max(1, len(sets)))
        for s0 in range(0, len(X), rows):
            Xc = X[s0:s0 + rows]
            P = np.tile(self.baseline, (len(sets) * len(Xc), 1))
            for t, T in enumerate(sets):
                block = P[t * len(Xc):(t + 1) * len(Xc)]
                if T:
                    block[:, list(T)] = Xc[:, list(T)]
            vals = _call(self._predict_model, P)
            for t, T in enumerate(sets):
                out[T][s0:s0 + len(Xc)] = vals[t * len(Xc):(t + 1) * len(Xc)]
        return out

    def dividends(self, X: np.ndarray) -> Tuple[float, Dict[SetT, np.ndarray]]:
        """f(0) and the dividends d_S(x_S), |S| <= k, eq. (dividendo), at the points X."""
        v = self._masked_values(X)
        div = {S: sum_terms([((-1) ** (len(S) - len(T))) * v[T] for T in _subsets(S)]) for S in self.sets}
        return float(v[()][0]) if len(X) else float(_call(self._predict_model, self.baseline[None, :])[0]), div

    # ------------------------------------------------------------- differences
    def _coefficients(self, S: SetT) -> List[Tuple[SetT, float]]:
        """Masked contexts R (S subset of R, |R| <= k) and the coefficients -c_R of
        (ap_deriv_cerrada); the unmasked point enters with coefficient +1."""
        rest = [j for j in range(self.d) if j not in S]
        out: List[Tuple[SetT, float]] = []
        for extra in range(0, self.k - len(S) + 1):
            for E in itertools.combinations(rest, extra):
                R = tuple(sorted(S + E))
                m = self.k - len(R)
                c = 1.0 if m == 0 else ((-1) ** m) * comb(self.d - len(R) - 1, m)
                out.append((R, -c))
        return out

    def _corner_differences(self, g_eval: Callable[[np.ndarray], np.ndarray], P: np.ndarray, S: SetT,
                            lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        """Alternating sum of g over the 2^|S| corners of the cells (lo, hi) of the points P,
        eq. (ap_ale_tfc): the corner for T is upper along T and lower along S \\ T."""
        out = np.zeros(len(P))
        for corner in itertools.product((0, 1), repeat=len(S)):
            Q = P.copy()
            for a, j in enumerate(S):
                Q[:, j] = hi[:, a] if corner[a] else lo[:, a]
            out += ((-1) ** (len(S) - sum(corner))) * g_eval(Q)
        return out

    def _residual_differences(self, S: SetT, X: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        """Delta_S r at the fitting points, from the model alone, eq. (ap_deriv_cerrada)."""
        total = np.zeros(len(X))
        contexts = [(None, 1.0)] + self._coefficients(S)
        rows = max(1, self.max_rows // (2 ** len(S)))
        for R, coef in contexts:
            P = X.copy()
            if R is not None:
                mask = np.ones(self.d, bool); mask[list(R)] = False
                P[:, mask] = self.baseline[mask]
            for s0 in range(0, len(X), rows):
                sl = slice(s0, s0 + rows)
                total[sl] += coef * self._corner_differences(lambda Q: _call(self._predict_model, Q), P[sl], S, lo[sl], hi[sl])
        return total

    # ------------------------------------------------------------- grid machinery
    def _cells(self, S: SetT, X: np.ndarray):
        """Cell index per axis and the lower and upper corners of the cell of each point."""
        idx = np.column_stack([self.cidx[j] if X is self._Xfit else cell_index(X[:, j], self.edges[j]) for j in S])
        lo = np.column_stack([self.edges[j][idx[:, a]] for a, j in enumerate(S)])
        hi = np.column_stack([self.edges[j][idx[:, a] + 1] for a, j in enumerate(S)])
        return idx, lo, hi

    def _cell_means(self, S: SetT, idx: np.ndarray, values: np.ndarray, shrink: bool) -> np.ndarray:
        """Mean of the differences in each cell, eq. (ap_ale_incr), 0 in empty cells,
        optionally shrunk (Section "Stability in cells with few data")."""
        shape = tuple(self.edges[j].size - 1 for j in S)
        flat = np.ravel_multi_index(tuple(idx.T), shape)
        n_cells = int(np.prod(shape))
        cnt = np.bincount(flat, minlength=n_cells)
        mean = np.where(cnt > 0, np.bincount(flat, weights=values, minlength=n_cells) / np.maximum(cnt, 1), 0.0)
        if shrink:
            v = pooled_variance(values, mean[flat], cnt[flat])
            mean, _, _ = shrink_increments(mean, cnt, v, self.tau2_method)
            mean = np.asarray(mean, float).reshape(-1)
        return mean.reshape(shape)

    @staticmethod
    def _accumulate(means: np.ndarray) -> np.ndarray:
        """nu at the nodes, eq. (ap_ale_acum): compensated cumulative sums axis by axis."""
        nodes = means
        for axis in range(means.ndim):
            nodes = np.apply_along_axis(cumsum0, axis, nodes)
        return nodes

    def _re_anchor(self, S: SetT, nu: GridFunction) -> GridFunction:
        """L_S at the nodes, eq. (ap_ale_anclaje): alternating sum of nu with the coordinates
        outside T at the baseline, so that L_S vanishes when any variable of S is at it."""
        mesh = nu.mesh()
        vals = np.zeros(len(mesh))
        for T in _subsets(S):
            Z = mesh.copy()
            for a, j in enumerate(S):
                if j not in T:
                    Z[:, a] = self.baseline[j]
            vals += ((-1) ** (len(S) - len(T))) * nu.at(Z)
        return GridFunction(S, nu.nodes, vals.reshape(nu.values.shape))

    def _L(self, U: SetT, g: GridFunction) -> GridFunction:
        """L_U of a function g of x_V (U subset of V): differences of g across the cells of
        the grid of U at the fitting points (ap_ar_incremento), with the other coordinates
        of V at their observed values; means without shrinkage; accumulation; re-anchoring."""
        idx, lo, hi = self._cells(U, self._Xfit)
        diffs = self._corner_differences(lambda Q: g(Q), self._Xfit, U, lo, hi)
        nodes = self._accumulate(self._cell_means(U, idx, diffs, shrink=False))
        return self._re_anchor(U, GridFunction(U, [self.edges[j] for j in U], nodes))

    def _H(self, S: SetT, L: GridFunction) -> GridFunction:
        """The estimated ALE component of a re-anchored curve L of x_S, eq. (ap_ar_centrado):
        L minus the components of its lower-order content, found recursively. The constant
        term is left out: it cancels once the value at the baseline is removed."""
        vals = L.values.copy()
        mesh = L.mesh()
        for U in _subsets(S, proper=True, nonempty=True):
            HU = self._H(U, self._L(U, L))
            vals -= HU.at(mesh[:, [S.index(j) for j in U]]).reshape(vals.shape)
        return GridFunction(S, L.nodes, vals)

    # ------------------------------------------------------------- fit
    def fit(self, Xfit: np.ndarray) -> "HarsanyiARSurrogateK":
        X = np.asarray(Xfit, float)
        if X.ndim != 2 or X.shape[1] != self.d:
            raise ValueError("Xfit must be (n, d) with d = len(baseline)")
        self._Xfit = X
        n = len(X)
        self.n_bins = terrell_scott_bins(n) if self.n_bins_arg is None else int(self.n_bins_arg)
        self.edges = [quantile_edges(X[:, j], self.n_bins) for j in range(self.d)]
        self.cidx = [cell_index(X[:, j], self.edges[j]) for j in range(self.d)]

        self.f0, div = self.dividends(X)
        r = _call(self._predict_model, X) - self.f0 - sum_terms([div[S] for S in self.sets])

        self.ar: Dict[SetT, GridFunction] = {}
        for S in self.sets:
            idx, lo, hi = self._cells(S, X)
            diffs = self._residual_differences(S, X, lo, hi)
            nodes = self._accumulate(self._cell_means(S, idx, diffs, self.shrink))
            L = self._re_anchor(S, GridFunction(S, [self.edges[j] for j in S], nodes))
            H = self._H(S, L)
            offset = float(H.at(self.baseline[list(S)][None, :])[0])
            self.ar[S] = GridFunction(S, H.nodes, H.values - offset)

        corr = sum_terms([self.ar[S](X) for S in self.sets])
        self.kappa = float(np.mean(r - corr)) if self.fit_intercept else 0.0
        self._fitted = True
        return self

    # ------------------------------------------------------------- use
    def components(self, X: np.ndarray) -> Tuple[float, Dict[SetT, np.ndarray]]:
        """f(0) + kappa and the components d_S + AR_S at X."""
        if not self._fitted:
            raise RuntimeError("fit first")
        X = np.asarray(X, float)
        _, div = self.dividends(X)
        return self.f0 + self.kappa, {S: div[S] + self.ar[S](X) for S in self.sets}

    def predict(self, X: np.ndarray) -> np.ndarray:
        c0, comp = self.components(X)
        return c0 + sum_terms([comp[S] for S in self.sets])

    def fidelity(self, X: np.ndarray) -> float:
        return r2_score(_call(self._predict_model, X), self.predict(X))

    def explain_local(self, X: np.ndarray) -> dict:
        """Intercept, the components by set, and the Shapley values of the surrogate,
        eq. (shapley): each component shared equally among the members of its set."""
        c0, comp = self.components(X)
        X = np.asarray(X, float)
        phi = np.zeros((len(X), self.d))
        for S, v in comp.items():
            for j in S:
                phi[:, j] += v / len(S)
        return {"intercept": c0, "components": comp, "shapley": phi}

    def effect(self, S: Sequence[int], Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Global effect of the set S at the coordinates Z (N, |S|): the dividend d_S alone
        and the component d_S + AR_S."""
        S = tuple(S)
        X = np.tile(self.baseline, (len(Z), 1)); X[:, list(S)] = np.asarray(Z, float)
        _, div = self.dividends(X)
        return div[S], div[S] + self.ar[S].at(np.asarray(Z, float))
