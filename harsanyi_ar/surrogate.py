"""
Order-2 surrogate model, eq. (subrogado) of the paper with k = 2:

    f~(x) = f(baseline) + kappa + sum_i [d_i + AR_i](x_i) + sum_{i<j} [d_ij + AR_ij](x_i, x_j).

  d_S    Harsanyi dividends, computed exactly from the model (dividends.py).
  AR_S   accumulated-residual correction, estimated once on the fitting sample, inside
         the support of the data, with empirical-Bayes shrinkage (ale.py, shrinkage.py).
  kappa  intercept correction, eq. (kappa_est): mean over the fitting sample of
         r^k - sum_S AR_S, i.e. of f minus the anchored surrogate.

Every component d_S + AR_S vanishes at the baseline. The same components give the
global effects (as functions of x_S), the local attribution (evaluated at x) and the
Shapley values of the surrogate, eq. (shapley).

The correction accumulates finite differences of the residual, so only predict() is
required, whatever the model.
"""
from __future__ import annotations
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np

from .numerics import sum_terms, r2_score, mean_1d
from .dividends import compute_dividends, truncated_prediction, all_pairs
from .ale import (quantile_edges, cell_index, build_ar1, build_ar2, AR1Curve, AR2Surface,
                  terrell_scott_bins, bilinear)

Pair = Tuple[int, int]
Predict = Callable[[np.ndarray], np.ndarray]


class _SurrogateBase:
    """
    n_bins : bins per variable of the quantile grid, common to every order. None (default)
             applies the rule of Terrell & Scott (1985), K = ceil((2n)^(1/3)) with n the
             size of the fitting sample (ale.terrell_scott_bins); an integer fixes it.
    shrink : Fay-Herriot shrinkage of the cell increments (shrinkage.py). Off by default: it
             trades variance for bias, which pays only when the differences of the model vary
             strongly within the cells (tree ensembles) and flattens the corrections in the
             tails (Section "Practical effect of the optional design choices").
    local  : how the dividend of a component is evaluated at a new point (Section "Cost").
             "exact" (default): recomputed at x, eq. (dividendo), 1 + d + d(d-1)/2 evaluations
             of the model per point; the component d_S + AR_S is then the same function the
             global plots draw, evaluated exactly at x. "interpolate": evaluated at the nodes
             of the grid of S during the fit (the baseline is added as a node) and interpolated
             multilinearly between them, like the correction; explaining then costs no
             evaluation of the model, at a price in fidelity where the dividends are curved
             between nodes (on California, 0.03 to 0.07 of R^2).
    """

    def __init__(self, predict: Predict, baseline, n_bins: Optional[int] = None,
                 tau2_method: str = "adm", shrink: bool = False, fit_intercept: bool = True,
                 local: str = "exact"):
        self._predict_model: Predict = lambda A: np.asarray(predict(A), float).ravel()
        self.baseline = np.asarray(baseline, float).ravel()
        self.d = self.baseline.size
        self.n_bins_spec: Optional[int] = None if n_bins is None else int(n_bins)
        self.n_bins: Optional[int] = self.n_bins_spec      # resolved in fit()
        self.tau2_method = tau2_method
        self.shrink = bool(shrink)
        self.fit_intercept = bool(fit_intercept)
        if local not in ("interpolate", "exact"):
            raise ValueError("local must be 'interpolate' or 'exact'")
        self.local = local
        self.div1_nodes: Dict[int, np.ndarray] = {}          # d_j at the nodes of the grid of x_j
        self.div2_nodes: Dict[Pair, np.ndarray] = {}         # d_ij at the nodes of the grid of (x_i, x_j)
        self.pairs: List[Pair] = all_pairs(self.d)
        self.ar1: Dict[int, AR1Curve] = {}
        self.ar2: Dict[Pair, AR2Surface] = {}
        self.kappa = 0.0
        self.f0 = float(self._predict_model(self.baseline[None, :])[0])
        self._fitted = False

    # ------------------------------------------------------------------ fit
    def _fit_corrections(self, Xfit: np.ndarray) -> None:
        raise NotImplementedError

    def fit(self, Xfit):
        Xfit = np.asarray(Xfit, float)
        self.n_bins = terrell_scott_bins(Xfit.shape[0]) if self.n_bins_spec is None else self.n_bins_spec
        self._fit_corrections(Xfit)
        if self.local == "interpolate":
            self._fit_node_dividends()
        self._fitted = True
        self.kappa = 0.0
        if self.fit_intercept:
            self.kappa = mean_1d(self._predict_model(Xfit) - self.predict_anchored(Xfit))
        return self

    def _node(self, value: float):
        """The baseline coordinate is a grid node only in the interpolated mode, where the
        components must be exact there; the default grids are the plain quantile grids."""
        return float(value) if self.local == "interpolate" else None

    def _fit_node_dividends(self) -> None:
        """Dividends at the nodes of the grids, for the interpolated evaluation: d_j at the
        K+1 nodes of x_j and d_ij at the (K+1)^2 nodes of (x_i, x_j), eq. (dividendo)."""
        b = self.baseline
        f1 = {}
        for j in range(self.d):
            nodes = self.ar1[j].nodes_x
            P = np.tile(b, (nodes.size, 1)); P[:, j] = nodes
            f1[j] = self._predict_model(P)
            self.div1_nodes[j] = f1[j] - self.f0
        for (i, j) in self.pairs:
            ni, nj = self.ar2[(i, j)].ni, self.ar2[(i, j)].nj
            GI, GJ = np.meshgrid(ni, nj, indexing="ij")
            P = np.tile(b, (GI.size, 1)); P[:, i] = GI.ravel(); P[:, j] = GJ.ravel()
            fij = self._predict_model(P).reshape(GI.shape)
            self.div2_nodes[(i, j)] = fij - f1[i][:, None] - f1[j][None, :] + self.f0

    def _check(self):
        if not self._fitted:
            raise RuntimeError("surrogate not fitted: call fit(Xfit) first")

    # ----------------------------------------------------------- components
    def components(self, X):
        """f(baseline), order-one components (N, d), order-two components (N, P), pairs."""
        self._check()
        X = np.asarray(X, float)
        if self.local == "exact":
            f0, d1, d2, pairs = compute_dividends(self._predict_model, X, self.baseline)
        else:
            f0, pairs = self.f0, self.pairs
            d1 = np.column_stack([np.interp(X[:, j], self.ar1[j].nodes_x, self.div1_nodes[j]) for j in range(self.d)])
            d2 = (np.column_stack([bilinear(self.ar2[p].ni, self.ar2[p].nj, self.div2_nodes[p], X[:, p[0]], X[:, p[1]])
                                   for p in pairs]) if pairs else np.zeros((len(X), 0)))
        c1 = d1 + np.column_stack([self.ar1[j](X[:, j]) for j in range(self.d)])
        c2 = d2 + np.column_stack([self.ar2[p](X) for p in pairs]) if pairs else d2
        return f0, c1, c2, pairs

    # ----------------------------------------------------------- prediction
    def predict_anchored(self, X) -> np.ndarray:
        """Surrogate without the intercept correction (value f(baseline) at the baseline)."""
        f0, c1, c2, _ = self.components(X)
        terms = [np.full(c1.shape[0], f0)]
        terms += [c1[:, i] for i in range(c1.shape[1])]
        terms += [c2[:, p] for p in range(c2.shape[1])]
        return sum_terms(terms)

    def predict(self, X) -> np.ndarray:
        """Surrogate prediction f~(x), eq. (subrogado) without h^k."""
        return self.predict_anchored(X) + self.kappa

    def predict_dividends_only(self, X) -> np.ndarray:
        """Truncation f^k(x): dividends of order <= 2, no correction, eq. (truncamiento)."""
        return truncated_prediction(self._predict_model, X, self.baseline)

    # -------------------------------------------------------------- fidelity
    def fidelity(self, X) -> float:
        """R^2 between the surrogate and the model it explains."""
        return r2_score(self._predict_model(np.asarray(X, float)), self.predict(X))

    def fidelity_anchored(self, X) -> float:
        return r2_score(self._predict_model(np.asarray(X, float)), self.predict_anchored(X))

    def fidelity_dividends_only(self, X) -> float:
        return r2_score(self._predict_model(np.asarray(X, float)), self.predict_dividends_only(X))

    # ------------------------------------------------------ local explanation
    def explain_local(self, X) -> dict:
        """
        intercept  : f(baseline) + kappa
        order1     : (N, d)  d_j + AR_j
        order2     : {(i, j): (N,)}  d_ij + AR_ij
        shapley    : (N, d)  phi_j = order1_j + 1/2 sum_{l != j} order2_jl, eq. (shapley)
        prediction : (N,)    intercept + sum_j phi_j
        """
        f0, c1, c2, pairs = self.components(X)
        shap = c1.copy()
        for p, (i, j) in enumerate(pairs):
            shap[:, i] += 0.5 * c2[:, p]
            shap[:, j] += 0.5 * c2[:, p]
        intercept = f0 + self.kappa
        pred = sum_terms([np.full(c1.shape[0], intercept)]
                         + [c1[:, i] for i in range(c1.shape[1])]
                         + [c2[:, p] for p in range(c2.shape[1])])
        return {"intercept": intercept, "f0": f0, "kappa": self.kappa,
                "order1": c1, "order2": {pairs[p]: c2[:, p] for p in range(len(pairs))},
                "shapley": shap, "prediction": pred}

    # ---------------------------------------------------------- global effects
    def effect_1d(self, j: int, grid: np.ndarray):
        """Order-one global effect on a grid of x_j: (dividend d_j, component d_j + AR_j)."""
        self._check()
        grid = np.asarray(grid, float)
        P = np.tile(self.baseline, (grid.size, 1))
        P[:, j] = grid
        dj = self._predict_model(P) - self.f0
        return dj, dj + self.ar1[j](grid)

    def effect_2d(self, i: int, j: int, grid_i: np.ndarray, grid_j: np.ndarray):
        """Order-two global effect d_ij + AR_ij on the mesh grid_i x grid_j.
        Returns an array of shape (len(grid_j), len(grid_i))."""
        self._check()
        GI, GJ = np.meshgrid(np.asarray(grid_i, float), np.asarray(grid_j, float))
        n = GI.size
        pij = np.tile(self.baseline, (n, 1)); pij[:, i] = GI.ravel(); pij[:, j] = GJ.ravel()
        pi = np.tile(self.baseline, (n, 1)); pi[:, i] = GI.ravel()
        pj = np.tile(self.baseline, (n, 1)); pj[:, j] = GJ.ravel()
        dij = (self._predict_model(pij) - self._predict_model(pi)
               - self._predict_model(pj) + self.f0)
        key = (i, j) if i < j else (j, i)
        return (dij + self.ar2[key](pij)).reshape(GI.shape)


class HarsanyiARSurrogate(_SurrogateBase):
    """
    The correction accumulates finite differences of the residual across the corners of
    each grid cell, eq. (ap_ale_incr). Works with non-differentiable models (tree
    ensembles); only predict: (N, d) -> (N,) is required.
    """

    def _fit_corrections(self, Xfit: np.ndarray) -> None:
        b = self.baseline
        N, d = Xfit.shape
        # order 1: difference of r^k = f - f^k across the cell, D_j r = r(z+, x_-j) - r(z-, x_-j).
        # Only the dividends that contain j change with x_j, so (finite-difference form of
        # eq. (ap_deriv_cerrada) with k = 2)
        #   D_j r^k = D_j f(x) + (d - 2) D_j f(x_[j]) - sum_{l != j} D_j f(x_[jl]),
        # i.e. 2 (d + 1) model evaluations per point and variable.
        for j in range(d):
            edges = quantile_edges(Xfit[:, j], self.n_bins, self._node(b[j]))
            idx = cell_index(Xfit[:, j], edges)
            up, lo = edges[idx + 1], edges[idx]

            def diff_j(rest: np.ndarray) -> np.ndarray:
                A = rest.copy(); A[:, j] = up
                B = rest.copy(); B[:, j] = lo
                return self._predict_model(A) - self._predict_model(B)

            base_rows = np.tile(b, (N, 1))
            diff = diff_j(Xfit) + (d - 2) * diff_j(base_rows)
            for l in range(d):
                if l != j:
                    masked = base_rows.copy(); masked[:, l] = Xfit[:, l]
                    diff = diff - diff_j(masked)
            self.ar1[j] = build_ar1(edges, idx, diff, b[j], self.tau2_method, self.shrink)
        # order 2: second difference of r^k across the four corners. Every dividend of
        # f^k other than d_ij has zero second difference in (x_i, x_j), so
        #   D2 r^k = D2 f(., x_-ij) - D2 f(., baseline_-ij).
        for (i, j) in self.pairs:
            ei = quantile_edges(Xfit[:, i], self.n_bins, self._node(b[i]))
            ej = quantile_edges(Xfit[:, j], self.n_bins, self._node(b[j]))
            iid, jid = cell_index(Xfit[:, i], ei), cell_index(Xfit[:, j], ej)
            ui, li, uj, lj = ei[iid + 1], ei[iid], ej[jid + 1], ej[jid]

            def second_difference(rest: np.ndarray) -> np.ndarray:
                out = np.zeros(N)
                for ai, aj, sign in ((ui, uj, 1.0), (ui, lj, -1.0), (li, uj, -1.0), (li, lj, 1.0)):
                    Y = rest.copy(); Y[:, i] = ai; Y[:, j] = aj
                    out += sign * self._predict_model(Y)
                return out

            second = second_difference(Xfit) - second_difference(np.tile(b, (N, 1)))
            self.ar2[(i, j)] = build_ar2(i, j, Xfit[:, i], Xfit[:, j], ei, ej, iid, jid, second,
                                         b[i], b[j], self.tau2_method, self.shrink)
