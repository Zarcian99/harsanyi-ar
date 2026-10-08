"""
Accumulated-residual (AR) curves: quantile grid, shrunk accumulation, discounting of
the lower orders and re-anchoring at the baseline. Section "Estimation" of the paper.

The quantity averaged per cell is the finite difference of the residual r^k across the
cell, eq. (ap_ale_tfc), so the cell mean is already the increment Delta_C of
eq. (ap_ale_incr).

Pipeline for a set S (|S| = 1 or 2):
  1. quantile grid on x_S;
  2. cell means Delta_C, shrunk by gamma_C (shrinkage.py);
  3. compensated accumulation to the grid nodes (numerics.py), nu(z) = sum of the
     increments of the cells in the box [z_0, z]; between nodes nu is interpolated
     multilinearly;
  4. re-anchoring of nu at the baseline, eq. (ap_ale_anclaje), with nu evaluated by
     interpolation;
  5. |S| = 2: discount of the lower orders, eq. (ap_ar_centrado): the ALE curve of the
     accumulated surface along each of its variables (differences of the surface across
     the cells of that variable at the fitting points, cell means without shrinkage,
     accumulation, re-anchoring) is subtracted from the surface;
  6. subtraction of the value at the baseline, so that AR_S(baseline) = 0, eq. (ar).
With this discount the estimated operators are pseudo-orthogonal exactly, whatever the
dependence between the variables: the ALE curve along x_i of AR_ij is identically zero
(Section "Discounting the lower orders").
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .numerics import cumsum0, cumsum2d_nodes
from .shrinkage import pooled_variance, shrink_increments


# --------------------------------------------------------------------------- #
#  Grid
# --------------------------------------------------------------------------- #
def terrell_scott_bins(n: int) -> int:
    """
    Number of bins per variable by the oversmoothed rule of Terrell & Scott (1985),
    K = ceil((2n)^(1/3)), with n the size of the fitting sample: the smallest number of
    bins that does not oversmooth any density, hence the one that leaves most points per
    bin among the n^(1/3) rules. n = 4000 gives K = 20. The cube root is taken in integer
    arithmetic so that exact cubes are not rounded off.
    """
    n = int(n)
    if n < 1:
        raise ValueError("the fitting sample is empty")
    m = 2 * n
    k = max(1, int(round(m ** (1.0 / 3.0))))
    while k ** 3 < m:
        k += 1
    while k > 1 and (k - 1) ** 3 >= m:
        k -= 1
    return k


def quantile_edges(z: np.ndarray, n_bins: int, node: float | None = None) -> np.ndarray:
    """Empirical-quantile bin edges (duplicates collapsed). With `node`, that value (the
    baseline coordinate) is added as a node of the grid, so that every component is exact
    at the baseline: the interpolated dividends vanish there and the re-anchoring
    (ap_ale_anclaje) reads nu at a node."""
    q = np.quantile(z, np.linspace(0, 1, n_bins + 1))
    if node is not None and q[0] <= node <= q[-1]:
        q = np.append(q, node)
    e = np.unique(q)
    if e.size < 2:
        e = np.array([z.min(), z.max() + 1e-9])
    return e


def cell_index(z: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Index of the bin that contains each value of z."""
    return np.clip(np.digitize(z, edges[1:-1]), 0, edges.size - 2)


# --------------------------------------------------------------------------- #
#  Order 1
# --------------------------------------------------------------------------- #
@dataclass
class AR1Curve:
    """AR_j(x_j): linear interpolation of the accumulated values between grid nodes,
    zero at the baseline."""
    nodes_x: np.ndarray        # (B+1,) grid nodes (bin edges)
    nodes_v: np.ndarray        # (B+1,) accumulated value nu at each node
    offset: float              # nu(baseline)

    def __call__(self, z: np.ndarray) -> np.ndarray:
        return np.interp(z, self.nodes_x, self.nodes_v) - self.offset


def build_ar1(edges: np.ndarray, idx: np.ndarray, values: np.ndarray,
              base: float = 0.0, tau2_method: str = "adm", shrink: bool = True) -> AR1Curve:
    """
    edges  : (B+1,) bin edges of x_j.
    idx    : (N,) bin of each fitting point.
    values : (N,) per-point difference of the residual across its cell, eq. (ap_ale_tfc).
    base   : baseline coordinate of x_j.
    """
    nb = edges.size - 1
    sums = np.bincount(idx, weights=values, minlength=nb)
    cnts = np.bincount(idx, minlength=nb)
    cell_mean = np.where(cnts > 0, sums / np.maximum(cnts, 1), 0.0)

    if shrink:
        v = pooled_variance(values, cell_mean[idx], cnts[idx])
        incr, _, _ = shrink_increments(cell_mean, cnts, v, tau2_method)
    else:
        incr = cell_mean

    nodes = cumsum0(incr)
    return AR1Curve(edges, nodes, float(np.interp(base, edges, nodes)))


# --------------------------------------------------------------------------- #
#  Order 2
# --------------------------------------------------------------------------- #
def bilinear(ni: np.ndarray, nj: np.ndarray, g: np.ndarray, zi: np.ndarray, zj: np.ndarray) -> np.ndarray:
    """Bilinear interpolation of the node values g (len(ni), len(nj)) at the points (zi, zj),
    eq. (ap_ale_interp) for |S| = 2; constant beyond the outer nodes, as np.interp."""
    gi = np.interp(zi, ni, np.arange(ni.size))
    gj = np.interp(zj, nj, np.arange(nj.size))
    i0 = np.clip(np.floor(gi).astype(int), 0, ni.size - 2)
    j0 = np.clip(np.floor(gj).astype(int), 0, nj.size - 2)
    ti, tj = gi - i0, gj - j0
    return (g[i0, j0] * (1 - ti) * (1 - tj) + g[i0 + 1, j0] * ti * (1 - tj)
            + g[i0, j0 + 1] * (1 - ti) * tj + g[i0 + 1, j0 + 1] * ti * tj)


@dataclass
class AR2Surface:
    """AR_ij(x_i, x_j): bilinear interpolation between grid nodes of the re-anchored
    accumulated surface, minus its ALE curves along x_i and along x_j (the lower-order
    discount, eq. ap_ar_centrado), zero at the baseline."""
    i: int
    j: int
    surface: np.ndarray        # (Bi+1, Bj+1) re-anchored accumulated values at the nodes
    ni: np.ndarray             # (Bi+1,) grid nodes of x_i
    nj: np.ndarray             # (Bj+1,) grid nodes of x_j
    disc_i: AR1Curve           # ALE curve of the surface along x_i, re-anchored at the baseline
    disc_j: AR1Curve           # ALE curve of the surface along x_j, re-anchored at the baseline
    offset: float

    def raw(self, zi: np.ndarray, zj: np.ndarray) -> np.ndarray:
        return bilinear(self.ni, self.nj, self.surface, zi, zj)

    def evaluate(self, zi: np.ndarray, zj: np.ndarray) -> np.ndarray:
        return self.raw(zi, zj) - self.disc_i(zi) - self.disc_j(zj) - self.offset

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.evaluate(X[:, self.i], X[:, self.j])


def ale_curve_of_surface(raw_fn, axis_vals: np.ndarray, other_vals: np.ndarray, edges: np.ndarray,
                         idx: np.ndarray, base: float, along_first: bool) -> AR1Curve:
    """
    ALE curve of a bivariate function g(x_i, x_j) along one of its variables, estimated on
    the fitting points, eq. (ap_ar_centrado) steps 1-3: differences of g across the cell of
    that variable that contains each fitting point, with the other coordinate kept at its
    observed value, eq. (ap_ar_incremento); cell means without shrinkage; accumulation;
    re-anchoring at the baseline.
    raw_fn(a, b) evaluates g at (a, b); along_first says whether the curve runs along the
    first argument.
    """
    lo, hi = edges[idx], edges[idx + 1]
    if along_first:
        diff = raw_fn(hi, other_vals) - raw_fn(lo, other_vals)
    else:
        diff = raw_fn(other_vals, hi) - raw_fn(other_vals, lo)
    nb = edges.size - 1
    cnts = np.bincount(idx, minlength=nb)
    means = np.where(cnts > 0, np.bincount(idx, weights=diff, minlength=nb) / np.maximum(cnts, 1), 0.0)
    nodes = cumsum0(means)
    return AR1Curve(edges, nodes, float(np.interp(base, edges, nodes)))


def build_ar2(i: int, j: int, zi: np.ndarray, zj: np.ndarray,
              ei: np.ndarray, ej: np.ndarray, iid: np.ndarray, jid: np.ndarray,
              values: np.ndarray, base_i: float = 0.0, base_j: float = 0.0,
              tau2_method: str = "adm", shrink: bool = True) -> AR2Surface:
    """
    zi, zj   : (N,) coordinates of the fitting points.
    ei, ej   : bin edges; iid, jid: bin of each fitting point on each axis.
    values   : (N,) per-point second difference of the residual across its cell,
               eq. (ap_ale_tfc).
    """
    bi, bj = ei.size - 1, ej.size - 1

    flat = iid * bj + jid
    sums = np.bincount(flat, weights=values, minlength=bi * bj)
    cnts = np.bincount(flat, minlength=bi * bj)
    cell_mean = np.where(cnts > 0, sums / np.maximum(cnts, 1), 0.0).reshape(bi, bj)
    cell_n = cnts.reshape(bi, bj)

    if shrink:
        v = pooled_variance(values, cell_mean[iid, jid], cell_n[iid, jid])
        incr, _, _ = shrink_increments(cell_mean, cell_n, v, tau2_method)
    else:
        incr = cell_mean

    nodes = cumsum2d_nodes(incr)                                   # nu at the grid nodes
    # re-anchoring, eq. (ap_ale_anclaje): nu(x_i, x_j) - nu(x_i, 0) - nu(0, x_j) + nu(0, 0),
    # with nu off the nodes given by its multilinear interpolant
    nu_i0 = np.array([np.interp(base_j, ej, nodes[a, :]) for a in range(bi + 1)])
    nu_0j = np.array([np.interp(base_i, ei, nodes[:, b]) for b in range(bj + 1)])
    nu_00 = float(np.interp(base_i, ei, nu_i0))
    surface = nodes - nu_i0[:, None] - nu_0j[None, :] + nu_00

    # discount of the lower orders, eq. (ap_ar_centrado): the ALE curve of the surface
    # along each variable, estimated on the fitting points, is subtracted from it; the
    # constant term (its mean) cancels in AR_S once the value at the baseline is removed
    zero = AR1Curve(np.array([0.0, 1.0]), np.zeros(2), 0.0)
    tmp = AR2Surface(i, j, surface, ei, ej, zero, zero, 0.0)
    disc_i = ale_curve_of_surface(tmp.raw, zi, zj, ei, iid, base_i, along_first=True)
    disc_j = ale_curve_of_surface(tmp.raw, zj, zi, ej, jid, base_j, along_first=False)

    out = AR2Surface(i, j, surface, ei, ej, disc_i, disc_j, 0.0)
    out.offset = float(out.evaluate(np.array([base_i]), np.array([base_j]))[0])
    return out
