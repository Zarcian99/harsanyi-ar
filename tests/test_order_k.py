"""Checks of the order-k implementation (harsanyi_ar/order_k.py). Run: python tests/test_order_k.py"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from harsanyi_ar import HarsanyiARSurrogate, HarsanyiARSurrogateK

D = 4


def smooth(X):
    return np.sin(X[:, 0]) * X[:, 1] + X[:, 2] ** 2 * X[:, 3] + np.exp(X[:, 0] * X[:, 2] / 3) + np.cos(X[:, 1] * X[:, 3] * X[:, 2])


def sample(seed, n, rho=0.0):
    Z = np.random.RandomState(seed).normal(size=(n, D))
    if rho:
        C = np.full((D, D), rho); np.fill_diagonal(C, 1.0)
        Z = Z @ np.linalg.cholesky(C).T
    return Z


def test_order2_reproduces_the_paper_implementation():
    """At k = 2 the general code gives the same surrogate as surrogate.py, with and without
    shrinkage, with the same grids (same number of bins)."""
    Xfit, Xte = sample(0, 1500), sample(1, 500)
    for shrink in (False, True):
        a = HarsanyiARSurrogate(smooth, np.zeros(D), None, "adm", shrink).fit(Xfit)
        b = HarsanyiARSurrogateK(smooth, np.zeros(D), order=2, shrink=shrink).fit(Xfit)
        assert abs(a.kappa - b.kappa) < 1e-7
        assert np.max(np.abs(a.predict(Xte) - b.predict(Xte))) < 1e-7
        ea, eb = a.explain_local(Xte), b.explain_local(Xte)
        assert np.max(np.abs(ea["shapley"] - eb["shapley"])) < 1e-7


def test_order3_exact_on_an_order3_function():
    """With k = 3 the dividends reproduce an order-3 function exactly: no residual, no correction."""
    f = lambda X: X[:, 0] * X[:, 1] * X[:, 2] + 0.5 * X[:, 1] * X[:, 3] + X[:, 0] ** 2 - X[:, 3]
    Xfit, Xte = sample(0, 2000), sample(1, 500)
    sur = HarsanyiARSurrogateK(f, np.zeros(D), order=3).fit(Xfit)
    assert abs(sur.kappa) < 1e-10
    assert max(np.max(np.abs(sur.ar[S](Xte))) for S in sur.sets) < 1e-9
    assert sur.fidelity(Xte) > 1 - 1e-12


def test_components_vanish_at_baseline_and_efficiency():
    Xfit, Xte = sample(0, 1500, rho=0.6), sample(1, 300, rho=0.6)
    sur = HarsanyiARSurrogateK(smooth, np.zeros(D), order=3).fit(Xfit)
    c0, comp = sur.components(np.zeros((1, D)))
    assert max(abs(float(v[0])) for v in comp.values()) < 1e-10
    e = sur.explain_local(Xte)
    assert np.max(np.abs(e["intercept"] + e["shapley"].sum(1) - sur.predict(Xte))) < 1e-10


def test_pseudo_orthogonality_of_the_estimators_at_order3():
    """H_U(AR_S) = 0 for every nonempty U strictly inside S, on dependent inputs: the
    estimated operators are pseudo-orthogonal whatever the dependence (eq. ap_ar_pseudoort)."""
    Xfit = sample(0, 1500, rho=0.7)
    sur = HarsanyiARSurrogateK(smooth, np.zeros(D), order=3).fit(Xfit)
    worst = 0.0
    for S in sur.sets:
        if len(S) < 2:
            continue
        for s in range(1, len(S)):
            import itertools
            for U in itertools.combinations(S, s):
                HU = sur._H(U, sur._L(U, sur.ar[S]))
                worst = max(worst, float(np.max(np.abs(HU.values - HU.values.flat[0]))))   # constant = zero component
    assert worst < 1e-9, worst


def test_higher_order_recovers_more_under_independence():
    """cos x1 cos x2 cos x3 cos x4 has content at every order up to four: the fidelity of the
    corrected surrogate does not decrease with k, and the correction never subtracts."""
    f = lambda X: np.cos(X[:, 0]) * np.cos(X[:, 1]) * np.cos(X[:, 2]) * np.cos(X[:, 3])
    Xfit, Xte = sample(0, 3000), sample(1, 2000)
    prev = -np.inf
    for k in (1, 2, 3):
        sur = HarsanyiARSurrogateK(f, np.zeros(D), order=k).fit(Xfit)
        fid = sur.fidelity(Xte)
        f0, div = sur.dividends(Xte)
        from harsanyi_ar.numerics import r2_score, sum_terms
        fid_div = r2_score(f(Xte), f0 + sum_terms([div[S] for S in sur.sets]))
        assert fid >= fid_div - 1e-6, (k, fid, fid_div)
        assert fid >= prev - 1e-6, (k, fid, prev)
        prev = fid
    sur4 = HarsanyiARSurrogateK(f, np.zeros(D), order=4).fit(Xfit)
    assert sur4.fidelity(Xte) > 1 - 1e-12


if __name__ == "__main__":
    import inspect
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and inspect.isfunction(fn):
            try:
                fn(); print("ok ", name)
            except AssertionError as e:
                fails += 1; print("FAIL", name, e)
    print(f"{5 - fails} of 5 tests passed")
