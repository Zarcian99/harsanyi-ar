"""
Derivative variant of the correction (DALE-type accumulation; Gkolemis et al. 2023),
kept outside the package and used only by experiments/route_comparison.py.

The method of the paper accumulates finite differences of the residual across the cells
of the grid (harsanyi_ar.surrogate). This variant, for differentiable torch models only,
accumulates instead the derivative of the residual, obtained from derivatives of the
model at masked points, times the cell length or area. Everything else (dividends,
quantile grid, shrinkage, compensated accumulation, re-anchoring, discount of the lower
orders, intercept) is the same computation as in the package; the pieces that need the
cell volume are copied here so that the package does not carry them.

Residual derivatives. f(x_[R]) does not depend on the coordinates outside R,
eq. (ap_deriv_mascara), so differentiating every dividend of f^k leaves only the masked
points that contain S, eq. (ap_deriv_doble):

    d^|S| r^k / dx_S (x) = d^|S| f / dx_S (x)
                           - sum_{|T|<=k} sum_{S subset R subset T} (-1)^{|T \\ R|} d^|S| f / dx_S (x_[R]).

Requires torch (torch.func). `model` is a harsanyi_ar.TorchModel.
"""
from __future__ import annotations
import itertools
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harsanyi_ar.numerics import sum_terms, r2_score, mean_1d, cumsum0, cumsum2d_nodes
from harsanyi_ar.dividends import compute_dividends, all_pairs
from harsanyi_ar.ale import (quantile_edges, cell_index, AR1Curve, AR2Surface, terrell_scott_bins,
                             ale_curve_of_surface)
from harsanyi_ar.shrinkage import estimate_tau2, pooled_variance

Coalition = Tuple[int, ...]
Pair = Tuple[int, int]


# --------------------------------------------------------------------------- #
#  Residual derivatives from model derivatives at masked points
# --------------------------------------------------------------------------- #
def coalitions_up_to(d: int, k: int) -> List[Coalition]:
    """Non-empty coalitions of size 1..k."""
    out: List[Coalition] = []
    for m in range(1, k + 1):
        out.extend(itertools.combinations(range(d), m))
    return out


def subset_signs(T: Coalition):
    """Yields (R, (-1)^{|T \\ R|}) over the subsets R of T."""
    T = tuple(T)
    for r in range(len(T) + 1):
        for R in itertools.combinations(T, r):
            yield R, (-1.0) ** (len(T) - len(R))


def _mask(X: torch.Tensor, R: Coalition, baseline: torch.Tensor) -> torch.Tensor:
    m = torch.zeros(X.shape[-1], dtype=X.dtype, device=X.device)
    if len(R):
        m[list(R)] = 1.0
    return X * m + baseline * (1.0 - m)


def _batched(fn, Xin: torch.Tensor, chunk: int) -> torch.Tensor:
    outs = []
    for s in range(0, Xin.shape[0], chunk):
        outs.append(torch.func.vmap(fn)(Xin[s:s + chunk]).detach())
    return torch.cat(outs, 0)


def residual_derivatives(model, X: torch.Tensor, k: int = 2, chunk: int = 512):
    """
    Returns grad_r (N, d) and, if k >= 2, hess_r (N, d, d) of the residual r^k.
    `model` must expose .single (R^d -> scalar) and .baseline (tensor).
    """
    N, d = X.shape
    baseline = model.baseline
    coalitions = coalitions_up_to(d, k)
    jac = torch.func.jacrev(model.single)
    hes = torch.func.hessian(model.single) if k >= 2 else None

    needed = sorted({R for T in coalitions for R, _ in subset_signs(T) if len(R) >= 1},
                    key=lambda t: (len(t), t))
    needed_hess = [R for R in needed if len(R) >= 2]

    jac_inputs = torch.cat([X] + [_mask(X, R, baseline) for R in needed], dim=0)
    J = _batched(jac, jac_inputs, chunk)
    grad_masked = {R: J[(n + 1) * N:(n + 2) * N] for n, R in enumerate(needed)}
    grad_r = J[:N].clone()
    for T in coalitions:
        for R, sign in subset_signs(T):
            for i in R:
                grad_r[:, i] -= sign * grad_masked[R][:, i]

    hess_r = None
    if k >= 2:
        hess_inputs = torch.cat([X] + [_mask(X, R, baseline) for R in needed_hess], dim=0)
        H = _batched(hes, hess_inputs, chunk)
        hess_masked = {R: H[(n + 1) * N:(n + 2) * N] for n, R in enumerate(needed_hess)}
        hess_r = H[:N].clone()
        for T in coalitions:
            for R, sign in subset_signs(T):
                Rl = list(R)
                for a in range(len(Rl)):
                    for b in range(a + 1, len(Rl)):
                        i, j = Rl[a], Rl[b]
                        hess_r[:, i, j] -= sign * hess_masked[R][:, i, j]
                        hess_r[:, j, i] = hess_r[:, i, j]

    grad_np = grad_r.detach().cpu().numpy()
    hess_np = hess_r.detach().cpu().numpy() if hess_r is not None else None
    return grad_np, hess_np


# --------------------------------------------------------------------------- #
#  Accumulation of cell-mean derivatives times cell volume (shrunk)
# --------------------------------------------------------------------------- #
def shrink_increments_volume(cell_mean: np.ndarray, volume: np.ndarray, count: np.ndarray,
                             pooled_var: float, tau2_method: str = "adm"):
    """
    Shrink the increments Delta_C = cell_mean * volume, where cell_mean is a mean
    derivative and volume the cell length or area, with Var(Delta_C) = volume^2 v / |N(C)|.
    Returns (gamma * Delta, gamma, tau2). Empty cells get gamma = 0.
    """
    ne = count > 0
    Delta = cell_mean * volume
    VarD = volume ** 2 * pooled_var / np.maximum(count, 1.0)
    tau2 = estimate_tau2(Delta[ne], VarD[ne], tau2_method)
    gamma = np.where(ne, tau2 / (tau2 + VarD + 1e-30), 0.0)
    return gamma * Delta, gamma, tau2


def build_ar1_volume(edges: np.ndarray, idx: np.ndarray, values: np.ndarray, volume: np.ndarray,
                     base: float = 0.0, tau2_method: str = "adm", shrink: bool = True) -> AR1Curve:
    """Order-one curve from per-point derivatives `values` and cell lengths `volume`."""
    nb = edges.size - 1
    sums = np.bincount(idx, weights=values, minlength=nb)
    cnts = np.bincount(idx, minlength=nb)
    cell_mean = np.where(cnts > 0, sums / np.maximum(cnts, 1), 0.0)

    if shrink:
        v = pooled_variance(values, cell_mean[idx], cnts[idx])
        incr, _, _ = shrink_increments_volume(cell_mean, volume, cnts, v, tau2_method)
    else:
        incr = cell_mean * volume

    nodes = cumsum0(incr)
    return AR1Curve(edges, nodes, float(np.interp(base, edges, nodes)))


def build_ar2_volume(i: int, j: int, zi: np.ndarray, zj: np.ndarray,
                     ei: np.ndarray, ej: np.ndarray, iid: np.ndarray, jid: np.ndarray,
                     values: np.ndarray, volume: np.ndarray,
                     base_i: float = 0.0, base_j: float = 0.0,
                     tau2_method: str = "adm", shrink: bool = True) -> AR2Surface:
    """Order-two surface from per-point cross derivatives `values` and cell areas `volume`."""
    bi, bj = ei.size - 1, ej.size - 1

    flat = iid * bj + jid
    sums = np.bincount(flat, weights=values, minlength=bi * bj)
    cnts = np.bincount(flat, minlength=bi * bj)
    cell_mean = np.where(cnts > 0, sums / np.maximum(cnts, 1), 0.0).reshape(bi, bj)
    cell_n = cnts.reshape(bi, bj)

    if shrink:
        v = pooled_variance(values, cell_mean[iid, jid], cell_n[iid, jid])
        incr, _, _ = shrink_increments_volume(cell_mean, volume, cell_n, v, tau2_method)
    else:
        incr = cell_mean * volume

    nodes = cumsum2d_nodes(incr)
    nu_i0 = np.array([np.interp(base_j, ej, nodes[a, :]) for a in range(bi + 1)])
    nu_0j = np.array([np.interp(base_i, ei, nodes[:, b]) for b in range(bj + 1)])
    nu_00 = float(np.interp(base_i, ei, nu_i0))
    surface = nodes - nu_i0[:, None] - nu_0j[None, :] + nu_00

    zero = AR1Curve(np.array([0.0, 1.0]), np.zeros(2), 0.0)
    tmp = AR2Surface(i, j, surface, ei, ej, zero, zero, 0.0)
    disc_i = ale_curve_of_surface(tmp.raw, zi, zj, ei, iid, base_i, along_first=True)
    disc_j = ale_curve_of_surface(tmp.raw, zj, zi, ej, jid, base_j, along_first=False)
    out = AR2Surface(i, j, surface, ei, ej, disc_i, disc_j, 0.0)
    out.offset = float(out.evaluate(np.array([base_i]), np.array([base_j]))[0])
    return out


# --------------------------------------------------------------------------- #
#  The surrogate with derivative-based corrections
# --------------------------------------------------------------------------- #
class HarsanyiARSurrogateDALE:
    """
    Order-2 surrogate whose corrections AR_j, AR_ij accumulate derivatives of the
    residual. Same dividends, intercept and prediction as harsanyi_ar.HarsanyiARSurrogate;
    only the per-cell quantity differs. `model` is a harsanyi_ar.TorchModel.
    """

    def __init__(self, model, n_bins=None, tau2_method: str = "adm",
                 shrink: bool = False, fit_intercept: bool = True, chunk: int = 512):
        self.model = model
        self.chunk = int(chunk)
        self._predict_model = lambda A: np.asarray(model.predict(A), float).ravel()
        self.baseline = np.asarray(model.baseline.cpu().numpy(), float).ravel()
        self.d = self.baseline.size
        self.n_bins_spec = None if n_bins is None else int(n_bins)
        self.n_bins = self.n_bins_spec                      # resolved in fit(), as in the package
        self.tau2_method = tau2_method
        self.shrink = bool(shrink)
        self.fit_intercept = bool(fit_intercept)
        self.pairs: List[Pair] = all_pairs(self.d)
        self.ar1: Dict[int, AR1Curve] = {}
        self.ar2: Dict[Pair, AR2Surface] = {}
        self.kappa = 0.0
        self.f0 = float(self._predict_model(self.baseline[None, :])[0])
        self._fitted = False

    def _fit_corrections(self, Xfit: np.ndarray) -> None:
        Xt = torch.as_tensor(Xfit, dtype=self.model.dtype)
        grad_r, hess_r = residual_derivatives(self.model, Xt, k=2, chunk=self.chunk)
        b = self.baseline
        for j in range(self.d):
            edges = quantile_edges(Xfit[:, j], self.n_bins)
            idx = cell_index(Xfit[:, j], edges)
            self.ar1[j] = build_ar1_volume(edges, idx, grad_r[:, j], np.diff(edges), b[j],
                                           self.tau2_method, self.shrink)
        for (i, j) in self.pairs:
            ei = quantile_edges(Xfit[:, i], self.n_bins)
            ej = quantile_edges(Xfit[:, j], self.n_bins)
            iid, jid = cell_index(Xfit[:, i], ei), cell_index(Xfit[:, j], ej)
            area = np.diff(ei)[:, None] * np.diff(ej)[None, :]
            self.ar2[(i, j)] = build_ar2_volume(i, j, Xfit[:, i], Xfit[:, j], ei, ej, iid, jid,
                                                hess_r[:, i, j], area, b[i], b[j],
                                                self.tau2_method, self.shrink)

    def fit(self, Xfit):
        Xfit = np.asarray(Xfit, float)
        self.n_bins = terrell_scott_bins(Xfit.shape[0]) if self.n_bins_spec is None else self.n_bins_spec
        self._fit_corrections(Xfit)
        self._fitted = True
        self.kappa = 0.0
        if self.fit_intercept:
            self.kappa = mean_1d(self._predict_model(Xfit) - self.predict_anchored(Xfit))
        return self

    def components(self, X):
        if not self._fitted:
            raise RuntimeError("surrogate not fitted: call fit(Xfit) first")
        X = np.asarray(X, float)
        f0, d1, d2, pairs = compute_dividends(self._predict_model, X, self.baseline)
        c1 = d1 + np.column_stack([self.ar1[j](X[:, j]) for j in range(self.d)])
        c2 = d2 + np.column_stack([self.ar2[p](X) for p in pairs]) if pairs else d2
        return f0, c1, c2, pairs

    def predict_anchored(self, X) -> np.ndarray:
        f0, c1, c2, _ = self.components(X)
        terms = [np.full(c1.shape[0], f0)]
        terms += [c1[:, i] for i in range(c1.shape[1])]
        terms += [c2[:, p] for p in range(c2.shape[1])]
        return sum_terms(terms)

    def predict(self, X) -> np.ndarray:
        return self.predict_anchored(X) + self.kappa

    def fidelity(self, X) -> float:
        return r2_score(self._predict_model(np.asarray(X, float)), self.predict(X))
