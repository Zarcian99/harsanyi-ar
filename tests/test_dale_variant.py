"""
Checks of the derivative variant of experiments/dale_variant.py (used only by
experiments/route_comparison.py). Skipped when torch is not installed.
Run with `pytest tests/test_dale_variant.py`.
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
torch = pytest.importorskip("torch")

from harsanyi_ar import HarsanyiARSurrogate, TorchModel, MLP
from harsanyi_ar.numerics import r2_score
from dale_variant import HarsanyiARSurrogateDALE, residual_derivatives

D = 4


def _torch_model(seed=0):
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(seed)
    return TorchModel(MLP(D, "softplus", hidden=16), D)


def test_residual_derivative_closed_form():
    """Eq. (ap_deriv_cerrada) with k = 2, in derivatives: c_R = (-1)^{k-|R|} binom(d-|R|-1, k-|R|)."""
    model = _torch_model()
    X = torch.randn(40, D)
    grad_r, hess_r = residual_derivatives(model, X, k=2)
    jac = torch.func.vmap(torch.func.jacrev(model.single))
    hes = torch.func.vmap(torch.func.hessian(model.single))

    def masked(R):
        m = torch.zeros(D); m[list(R)] = 1.0
        return X * m
    for i in range(D):
        expected = jac(X)[:, i] + (D - 2) * jac(masked([i]))[:, i]
        for j in range(D):
            if j != i:
                expected = expected - jac(masked([i, j]))[:, i]
        assert np.max(np.abs(grad_r[:, i] - expected.detach().numpy())) < 1e-10
    for i in range(D):
        for j in range(i + 1, D):
            expected = hes(X)[:, i, j] - hes(masked([i, j]))[:, i, j]
            assert np.max(np.abs(hess_r[:, i, j] - expected.detach().numpy())) < 1e-10


def test_finite_differences_and_derivatives_agree():
    """On a smooth network the two accumulations give the same surrogate up to grid error."""
    model = _torch_model(1)
    rs = np.random.RandomState(0)
    Xfit, Xte = rs.normal(size=(3000, D)), rs.normal(size=(1000, D))
    a = HarsanyiARSurrogate(model.predict, np.zeros(D)).fit(Xfit).predict(Xte)
    b = HarsanyiARSurrogateDALE(model).fit(Xfit).predict(Xte)
    assert r2_score(a, b) > 0.999
