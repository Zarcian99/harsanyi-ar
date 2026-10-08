"""
Thin wrapper around a differentiable torch network, plus the MLP used in the paper.
Requires torch. Inputs are assumed standardised, so the baseline (feature means) is 0.
"""
from __future__ import annotations
from typing import Optional
import numpy as np
import torch
import torch.nn as nn

ACTIVATIONS = {
    "softplus": nn.Softplus, "sigmoid": nn.Sigmoid, "tanh": nn.Tanh,
    "silu": nn.SiLU, "mish": nn.Mish, "gelu": nn.GELU,
}


class MLP(nn.Module):
    """Multilayer perceptron with two hidden layers."""

    def __init__(self, d_in: int, activation: str = "softplus", hidden: int = 64, d_out: int = 1):
        super().__init__()
        act = ACTIVATIONS[activation.lower()]
        self.net = nn.Sequential(nn.Linear(d_in, hidden), act(),
                                 nn.Linear(hidden, hidden), act(),
                                 nn.Linear(hidden, d_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TorchModel:
    """
    Scalar view of a torch network: output `output_index` of `net` (a logit in
    classification, the prediction in regression).

      predict(X)  numpy (N, d) -> numpy (N,)         model evaluations / dividends
      single(z)   tensor (d,)  -> scalar tensor      for torch.func.jacrev / hessian
    """

    def __init__(self, net: nn.Module, d: int, output_index: int = 0,
                 baseline: Optional[np.ndarray] = None, dtype: torch.dtype = torch.float64):
        self.net = net.to(dtype).eval()
        self.d = int(d)
        self.output_index = int(output_index)
        self.dtype = dtype
        b = np.zeros(self.d) if baseline is None else np.asarray(baseline, float)
        self.baseline = torch.tensor(b, dtype=dtype)

    def single(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z.unsqueeze(0))[0, self.output_index]

    def predict(self, X) -> np.ndarray:
        with torch.no_grad():
            Z = torch.as_tensor(np.asarray(X, float), dtype=self.dtype)
            return self.net(Z)[:, self.output_index].cpu().numpy()
