"""
harsanyi_ar: order-2 interpretable surrogate of a trained model, built from Harsanyi
dividends plus an accumulated-residual (ALE-based) correction with empirical-Bayes
shrinkage.

    from harsanyi_ar import HarsanyiARSurrogate            # any model, numpy only
    from harsanyi_ar import TorchModel, MLP                # torch networks of the experiments
"""
from .numerics import r2_score, fsum_1d, mean_1d
from .dividends import compute_dividends, truncated_prediction, all_pairs
from .shrinkage import estimate_tau2
from .surrogate import HarsanyiARSurrogate

__all__ = ["HarsanyiARSurrogate", "TorchModel", "MLP",
           "compute_dividends", "truncated_prediction", "all_pairs", "estimate_tau2",
           "r2_score", "fsum_1d", "mean_1d"]

_TORCH_NAMES = {"TorchModel": "model", "MLP": "model", "ACTIVATIONS": "model"}


def __getattr__(name):            # torch is imported only if the torch classes are used
    if name in _TORCH_NAMES:
        import importlib
        return getattr(importlib.import_module(f".{_TORCH_NAMES[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
