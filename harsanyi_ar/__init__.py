"""
harsanyi_ar: interpretable order-k surrogate of a trained model, built from Harsanyi
dividends plus an accumulated-residual (ALE-based) correction.

    from harsanyi_ar import HarsanyiARSurrogate            # order 2, the implementation of the paper
    from harsanyi_ar import HarsanyiARSurrogateK           # any order k (reproduces the above at k = 2)
    from harsanyi_ar import TorchModel, MLP                # torch networks of the experiments
"""
from .numerics import r2_score, fsum_1d, mean_1d
from .dividends import compute_dividends, truncated_prediction, all_pairs
from .shrinkage import estimate_tau2
from .surrogate import HarsanyiARSurrogate
from .order_k import HarsanyiARSurrogateK, GridFunction

__all__ = ["HarsanyiARSurrogate", "HarsanyiARSurrogateK", "GridFunction", "TorchModel", "MLP",
           "compute_dividends", "truncated_prediction", "all_pairs", "estimate_tau2",
           "r2_score", "fsum_1d", "mean_1d"]

_TORCH_NAMES = {"TorchModel": "model", "MLP": "model", "ACTIVATIONS": "model"}


def __getattr__(name):            # torch is imported only if the torch classes are used
    if name in _TORCH_NAMES:
        import importlib
        return getattr(importlib.import_module(f".{_TORCH_NAMES[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
