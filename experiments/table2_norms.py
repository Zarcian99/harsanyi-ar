"""
Norms behind the table "Reconstruction in distribution" (tab:ar), computable because the functions
are known. Same fitting samples and surrogate models as table2_synthetic.py (so the same correction
A = kappa + sum AR per seed); the expectations are taken as exactly as each function allows:
  ||r||^2         r = f - f^k, the residual of the dividends of order <= 2;
  ||Pi r||^2      Pi r = Pi f - f^k, its projection onto the sums of functions of at most two
                  variables (f^k already belongs there): the most the correction can recover;
  ||Pi r - A||^2  the distance from the estimated correction to that projection.
How Pi f is obtained:
  independent inputs: the ANOVA truncation, from g_T(x_T) = E f(x_T, X_-T), |T| <= 2,
    - polynomials and the cosine product: closed form;
    - the softplus network: E[softplus(a + s Z)] tabulated once per unit and set T by dense
      quadrature (trapezoid, 4001 nodes on [-9, 9]) on a fine grid of a, read by cubic spline;
    - the step function: exact, by enumerating the 8^3 octile cells (probability 1/8 each);
  correlated inputs (polynomials only): least squares on all monomials of degree <= 3 in one or two
    variables, with the normal equations built from exact Gaussian moments (Stein's identity), so
    the projection is exact.
Norms that involve A are means over 2 000 000 fresh draws (common to all seeds); ||r||^2 and
||Pi r||^2 are exact where the function allows (step function, correlated polynomials) and means
over the same 2 000 000 draws otherwise. The gain is checked two ways,
||r||^2 - ||r - A||^2 against ||Pi r||^2 - ||Pi r - A||^2.

    python experiments/table2_norms.py
"""
from __future__ import annotations
import itertools
from functools import lru_cache
import numpy as np
from scipy.stats import norm
from scipy.interpolate import CubicSpline

from common import Tee, N_BINS, TAU2
from harsanyi_ar import HarsanyiARSurrogate
from table2_synthetic import ROWS, sample, D, N_BG, N_FIT, SEEDS, q, _W, _B, _V

N_EVAL = 2_000_000
SETS = [()] + [(j,) for j in range(D)] + list(itertools.combinations(range(D), 2))
M_COS = np.exp(-0.5)                                     # E cos X, X ~ N(0, 1)
OCT = norm.ppf((2 * np.arange(1, 9) - 1) / 16.0)          # octile midpoints


def inclusion_exclusion(g):
    """ANOVA truncation at order 2 from the conditional means g[T], T in SETS."""
    out = g[()] * (1 - D + D * (D - 1) / 2)              # g_0 with its signed count
    for j in range(D):
        out = out + g[(j,)] * (1 - (D - 1))              # each single: +1, -1 per pair containing it
    for T in SETS[1 + D:]:
        out = out + g[T]
    return out


# ------------------------------------------------------------------ independent: closed forms
def pi_cos(X):
    g = {T: np.prod([np.cos(X[:, j]) for j in T if j < 3], axis=0) * M_COS ** (3 - sum(j < 3 for j in T))
         if T else np.full(len(X), M_COS ** 3) for T in SETS}
    g = {T: (v if np.ndim(v) else np.full(len(X), v)) for T, v in g.items()}
    return inclusion_exclusion(g)


def pi_net(X):
    za = np.linspace(-9, 9, 4001); wz = norm.pdf(za); wz /= np.trapezoid(wz, za)
    g = {}
    for T in SETS:
        rest = [j for j in range(D) if j not in T]
        tot = np.zeros(len(X))
        for k in range(len(_V)):
            a = (X[:, list(T)] @ _W[k, list(T)] if T else np.zeros(len(X))) + _B[k]
            s = np.sqrt(np.sum(_W[k, rest] ** 2))
            grid = np.linspace(a.min() - 1, a.max() + 1, 6001)
            vals = np.trapezoid(np.logaddexp(0.0, grid[:, None] + s * za[None, :]) * wz[None, :], za, axis=1)
            tot += _V[k] * CubicSpline(grid, vals)(a)
        g[T] = tot
    return inclusion_exclusion(g)


def pi_step_cells():
    """f_step on the 8^3 octile cells and its ANOVA truncation there (x_4 is irrelevant)."""
    I = np.array(list(itertools.product(range(8), repeat=3)))
    Xc = np.column_stack([OCT[I[:, 0]], OCT[I[:, 1]], OCT[I[:, 2]], np.zeros(len(I))])
    return I, Xc


def pi_step(f, I, Xc, idx):
    """Exact ANOVA truncation of f_step, read at the cells idx (n, 3) of the evaluation points."""
    F = f(Xc).reshape(8, 8, 8)
    g = {(): np.full(len(idx), F.mean())}
    for j in range(D):
        g[(j,)] = (F.mean(axis=tuple(a for a in range(3) if a != j))[idx[:, j]] if j < 3
                   else np.full(len(idx), F.mean()))
    for j, l in itertools.combinations(range(D), 2):
        if l < 3:
            other = tuple(a for a in range(3) if a not in (j, l))
            g[(j, l)] = F.mean(axis=other)[idx[:, j], idx[:, l]]
        else:                                             # pairs with x_4: as the single
            g[(j, l)] = g[(j,)]
    return inclusion_exclusion(g)


# ------------------------------------------------------------------ correlated: exact moments
def moment_fn(C):
    @lru_cache(maxsize=None)
    def m(a):
        a = list(a)
        if sum(a) == 0:
            return 1.0
        if sum(a) % 2:
            return 0.0
        i = next(t for t, v in enumerate(a) if v > 0)
        a[i] -= 1
        tot = 0.0
        for j in range(D):
            if a[j] > 0:
                b = a.copy(); b[j] -= 1
                tot += C[i, j] * a[j] * m(tuple(b))
        return tot
    return m


def basis_exps():
    E = [tuple([0] * D)]
    for j in range(D):
        for p in (1, 2, 3):
            e = [0] * D; e[j] = p; E.append(tuple(e))
    for j, l in itertools.combinations(range(D), 2):
        for p, r in ((1, 1), (1, 2), (2, 1)):
            e = [0] * D; e[j] = p; e[l] = r; E.append(tuple(e))
    return E


def pi_correlated_cubic(rho):
    """Exact projection of x1 x2 x3 onto F_2 under equicorrelated N(0, 1) inputs."""
    C = np.full((D, D), rho); np.fill_diagonal(C, 1.0)
    m = moment_fn(C)
    E = basis_exps()
    add = lambda u, v: tuple(x + y for x, y in zip(u, v))
    G = np.array([[m(add(u, v)) for v in E] for u in E])
    t = (1, 1, 1, 0)
    b = np.array([m(add(u, t)) for u in E])
    coef = np.linalg.solve(G, b)
    norm_r = m(add(t, t))
    norm_pr = float(coef @ G @ coef)
    proj = lambda X: np.column_stack([np.prod(X ** np.array(e), axis=1) for e in E]) @ coef
    return proj, norm_r, norm_pr


# ------------------------------------------------------------------ table
def run():
    out = Tee("table2_norms.txt")
    for rho in (0.0, 0.7):
        out(f"\n== {'independent' if rho == 0.0 else f'correlated (rho = {rho} pairwise)'} ==")
        out(f"{'function':26s} {'||r||^2':>9s} {'||Pi r||^2':>11s} {'||Pi r - A||^2':>15s}"
            f"   gain {'direct':>8s} {'identity':>9s}")
        Xe = sample(np.random.RandomState(999), N_EVAL, rho)
        for name, f, regimes in ROWS:
            if rho not in regimes:
                continue
            fe = f(Xe)
            rng0 = np.random.RandomState(0)                 # the dividends do not depend on the fit
            sur0 = HarsanyiARSurrogate(lambda Z: f(np.asarray(Z, float)), np.zeros(D), N_BINS, TAU2).fit(
                sample(rng0, N_FIT, rho))
            fk = sur0.predict_dividends_only(Xe)
            exact_r = exact_pr = None
            if rho == 0.0:
                if name == "order-3 cosine":
                    Pf = pi_cos(Xe)
                    # r = (c1-1)(c2-1)(c3-1), c = cos x; its ANOVA part of order 3 is v1 v2 v3, v = c - E c
                    Ec, Ec2 = np.exp(-0.5), (1 + np.exp(-2.0)) / 2
                    exact_r = (Ec2 - 2 * Ec + 1) ** 3
                    exact_pr = exact_r - (Ec2 - Ec ** 2) ** 3
                elif name == "softplus neural network":
                    Pf = pi_net(Xe)
                elif name == "step function":
                    I, Xc = pi_step_cells()
                    idx = np.column_stack([np.searchsorted(OCT, q(Xe[:, j])) for j in range(3)])
                    Pf = pi_step(f, I, Xc, idx)
                    # exact norms over the 512 equiprobable cells
                    fkc = sur0.predict_dividends_only(Xc)
                    Pfc = pi_step(f, I, Xc, I)
                    exact_r = float(np.mean((f(Xc) - fkc) ** 2))
                    exact_pr = float(np.mean((Pfc - fkc) ** 2))
                elif name == "order-3 monomial":
                    Pf = np.zeros(len(Xe)); exact_r, exact_pr = 1.0, 0.0      # r = x1 x2 x3
                elif name == "additive + order 3":
                    Pf = 2 * (Xe[:, 0] + Xe[:, 1] + Xe[:, 2]); exact_r, exact_pr = 1.0, 0.0
                else:                                       # pure order 2: f itself
                    Pf = fe
            else:
                if name == "additive + order 2":
                    Pf = fe
                else:                                       # r = x1 x2 x3 for both cubic rows
                    proj, exact_r, exact_pr = pi_correlated_cubic(rho)
                    Pf = proj(Xe) + (fe - Xe[:, 0] * Xe[:, 1] * Xe[:, 2])
            r, Pr = fe - fk, Pf - fk
            n_r = exact_r if exact_r is not None else float(np.mean(r ** 2))
            n_p = exact_pr if exact_pr is not None else float(np.mean(Pr ** 2))
            n_e, g_dir, g_id = [], [], []
            for s in SEEDS:
                rng = np.random.RandomState(s)
                X_bg, _ = sample(rng, N_BG, rho), sample(rng, 5000, rho)
                Xfit = X_bg[rng.choice(N_BG, size=N_FIT, replace=False)]
                sur = HarsanyiARSurrogate(lambda Z: f(np.asarray(Z, float)), np.zeros(D), N_BINS, TAU2).fit(Xfit)
                A = sur.predict(Xe) - fk
                e = float(np.mean((Pr - A) ** 2))
                n_e.append(e)
                g_dir.append(float(np.mean(r ** 2) - np.mean((r - A) ** 2)))
                g_id.append(float(np.mean(Pr ** 2)) - e)
            out(f"{name:26s} {n_r:9.4f} {n_p:11.4f} {np.mean(n_e):15.4f}   {np.mean(g_dir):13.4f} {np.mean(g_id):9.4f}")


if __name__ == "__main__":
    run()
