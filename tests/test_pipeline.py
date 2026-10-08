"""
Checks of the properties stated in the paper. Run with `pytest` or `python tests/test_pipeline.py`.
Needs numpy only.
"""
from __future__ import annotations
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harsanyi_ar import HarsanyiARSurrogate, truncated_prediction
from harsanyi_ar.numerics import sum_terms, cumsum0

D = 4
cosine3 = lambda X: np.cos(X[:, 0]) * np.cos(X[:, 1]) * np.cos(X[:, 2])


def _fit(f, n=3000, seed=0, **kw):
    rs = np.random.RandomState(seed)
    Xfit, Xte = rs.normal(size=(n, D)), rs.normal(size=(2000, D))
    return HarsanyiARSurrogate(f, np.zeros(D), **kw).fit(Xfit), Xfit, Xte


def test_compensated_summation():
    assert sum_terms([np.array([1e16]), np.array([1.0]), np.array([-1e16]), np.array([1.0])])[0] == 2.0
    incr = np.full(10_000, 0.1)
    assert abs(cumsum0(incr)[-1] - 1000.0) < 1e-12


def test_terrell_scott_rule():
    """K = ceil((2n)^(1/3)), exact on cubes: 2n = 1000 gives 10, 2n = 1002 gives 11."""
    from harsanyi_ar.ale import terrell_scott_bins
    for n, k in ((4, 2), (142, 7), (455, 10), (500, 10), (501, 11), (4000, 20),
                 (16512, 33), (1_000_000, 126)):
        assert terrell_scott_bins(n) == k, (n, terrell_scott_bins(n))
    sur, Xfit, Xte = _fit(cosine3, n=4000)
    assert sur.n_bins == 20
    fixed = HarsanyiARSurrogate(cosine3, np.zeros(D), n_bins=20).fit(Xfit)
    assert np.array_equal(sur.predict(Xte), fixed.predict(Xte))


def test_dividends_exact_for_order_two():
    f = lambda X: 1.0 + X[:, 0] - 2 * X[:, 3] ** 2 + X[:, 0] * X[:, 1] + np.sin(X[:, 2]) * X[:, 3]
    X = np.random.RandomState(1).normal(size=(500, D))
    assert np.max(np.abs(truncated_prediction(f, X, np.zeros(D)) - f(X))) < 1e-12


def test_no_residual_no_correction():
    f = lambda X: X[:, 0] * X[:, 1] + X[:, 2] * X[:, 3]
    sur, _, Xte = _fit(f)
    assert abs(sur.kappa) < 1e-12 and sur.fidelity(Xte) > 1 - 1e-12


def test_components_vanish_at_baseline():
    sur, _, _ = _fit(cosine3)
    _, c1, c2, _ = sur.components(np.zeros((1, D)))
    assert np.max(np.abs(c1)) < 1e-12 and np.max(np.abs(c2)) < 1e-12


def test_reanchored_surface_vanishes_on_the_baseline_axes():
    """Eq. (ap_ale_anclaje): L_S vanishes as soon as one variable of S returns to the baseline."""
    sur, _, _ = _fit(cosine3)
    t, zero = np.linspace(-2, 2, 21), np.zeros(21)
    for surface in sur.ar2.values():
        assert np.max(np.abs(surface.raw(t, zero))) < 1e-12
        assert np.max(np.abs(surface.raw(zero, t))) < 1e-12


def test_pseudo_orthogonality_of_the_estimators():
    """Section "Discounting the lower orders": the estimated operators are pseudo-orthogonal
    exactly, whatever the dependence. The ALE curve along x_i of AR_ij, estimated on the
    fitting points as in eq. (ap_ar_centrado), is identically zero; so is the one along x_j."""
    from harsanyi_ar.ale import quantile_edges, cell_index, ale_curve_of_surface
    rs = np.random.RandomState(4)
    Z = rs.normal(size=(3000, D)) @ np.linalg.cholesky(0.3 * np.eye(D) + 0.7 * np.ones((D, D))).T
    sur = HarsanyiARSurrogate(cosine3, np.zeros(D)).fit(Z)                # dependent inputs
    for (i, j), ar in sur.ar2.items():
        g = lambda a, b: ar.evaluate(a, b)
        ei, ej = ar.ni, ar.nj
        iid, jid = cell_index(Z[:, i], ei), cell_index(Z[:, j], ej)
        ci = ale_curve_of_surface(g, Z[:, i], Z[:, j], ei, iid, 0.0, along_first=True)
        cj = ale_curve_of_surface(g, Z[:, j], Z[:, i], ej, jid, 0.0, along_first=False)
        assert np.max(np.abs(ci.nodes_v)) < 1e-10 and np.max(np.abs(cj.nodes_v)) < 1e-10
        assert abs(ar(np.zeros((1, D)))[0]) < 1e-12


def test_interpolated_mode():
    """local='interpolate': the components are read from the grids, the baseline is a node so
    they vanish there, and a bilinear function is reproduced exactly inside the grid."""
    rs = np.random.RandomState(0)
    Xfit, Xte = rs.normal(size=(4000, D)), rs.normal(size=(2000, D))
    f = lambda X: X[:, 0] * X[:, 1] + X[:, 2] * X[:, 3]
    sur = HarsanyiARSurrogate(f, np.zeros(D), local="interpolate").fit(Xfit)
    f0, c1, c2, _ = sur.components(np.zeros((1, D)))
    assert np.max(np.abs(c1)) < 1e-12 and np.max(np.abs(c2)) < 1e-12
    inside = np.all((Xte >= Xfit.min(0)) & (Xte <= Xfit.max(0)), axis=1)
    assert sur.fidelity(Xte[inside]) > 1 - 1e-10


def test_efficiency_of_shapley_values():
    sur, _, Xte = _fit(cosine3)
    e = sur.explain_local(Xte[:50])
    assert np.max(np.abs(e["intercept"] + e["shapley"].sum(1) - sur.predict(Xte[:50]))) < 1e-12


def test_kappa_is_mean_corrected_residual():
    sur, Xfit, _ = _fit(cosine3)
    assert abs(np.mean(cosine3(Xfit) - sur.predict(Xfit))) < 1e-12
    assert abs(sur.kappa - np.mean(cosine3(Xfit) - sur.predict_anchored(Xfit))) < 1e-12


def test_correction_improves_under_independence():
    sur, _, Xte = _fit(cosine3)
    assert sur.fidelity(Xte) > sur.fidelity_dividends_only(Xte) + 0.2


def test_residual_differences_identity():
    """Eq. (ap_deriv_cerrada) with k = 2, in finite differences: the corrections built from
    evaluations of f at masked points coincide with those built from the residual
    r = f - f^k evaluated directly at the corners of the cells. Without shrinkage, whose
    numerical tau^2 search would amplify rounding differences of 1e-16 to 1e-10."""
    from harsanyi_ar.ale import quantile_edges, cell_index, build_ar1, build_ar2
    sur, Xfit, _ = _fit(cosine3, shrink=False)
    b = np.zeros(D)
    r = lambda A: cosine3(A) - truncated_prediction(cosine3, A, b)
    for j in range(D):
        edges = quantile_edges(Xfit[:, j], sur.n_bins)
        idx = cell_index(Xfit[:, j], edges)
        A, B = Xfit.copy(), Xfit.copy()
        A[:, j], B[:, j] = edges[idx + 1], edges[idx]
        direct = build_ar1(edges, idx, r(A) - r(B), b[j], sur.tau2_method, sur.shrink)
        assert np.max(np.abs(direct.nodes_v - sur.ar1[j].nodes_v)) < 1e-10
    for (i, j) in sur.pairs:
        ei, ej = quantile_edges(Xfit[:, i], sur.n_bins), quantile_edges(Xfit[:, j], sur.n_bins)
        iid, jid = cell_index(Xfit[:, i], ei), cell_index(Xfit[:, j], ej)
        second = np.zeros(len(Xfit))
        for ai, aj, sign in ((ei[iid + 1], ej[jid + 1], 1.0), (ei[iid + 1], ej[jid], -1.0),
                             (ei[iid], ej[jid + 1], -1.0), (ei[iid], ej[jid], 1.0)):
            Y = Xfit.copy(); Y[:, i] = ai; Y[:, j] = aj
            second += sign * r(Y)
        direct = build_ar2(i, j, Xfit[:, i], Xfit[:, j], ei, ej, iid, jid, second,
                           b[i], b[j], sur.tau2_method, sur.shrink)
        got = sur.ar2[(i, j)]
        assert np.max(np.abs(direct.surface - got.surface)) < 1e-10
        assert np.max(np.abs(direct.disc_i.nodes_v - got.disc_i.nodes_v)) < 1e-10
        assert abs(direct.offset - got.offset) < 1e-10


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")
