"""Encoder + Decoder MLPs (content embedding <-> latent)."""

from __future__ import annotations

import torch
import torch.nn as nn


def _mlp(dims: list[int]) -> nn.Sequential:
    """MLP over the given layer widths: ReLU between layers, none after the last."""
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden: list[int], latent_dim: int):
        super().__init__()
        self.net = _mlp([input_dim, *hidden, latent_dim])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, hidden: list[int], output_dim: int):
        super().__init__()
        self.net = _mlp([latent_dim, *reversed(hidden), output_dim])

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
