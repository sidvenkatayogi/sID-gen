"""Codebook health metrics."""

from __future__ import annotations

import torch


def codebook_utilization(indices: torch.Tensor, codebook_size: int) -> list[float]:
    """Fraction of codes used at least once, per level. `indices` is (N, L)."""
    L = indices.shape[1]
    out: list[float] = []
    for l in range(L):
        used = torch.unique(indices[:, l]).numel()
        out.append(used / codebook_size)
    return out


def codebook_perplexity(indices: torch.Tensor, codebook_size: int) -> list[float]:
    """exp(entropy of code-usage), per level. Range [1, K]: 1 = collapsed to one
    code, K = perfectly uniform."""
    L = indices.shape[1]
    out: list[float] = []
    for l in range(L):
        counts = torch.bincount(indices[:, l], minlength=codebook_size).float()
        p = counts / counts.sum().clamp(min=1)
        nonzero = p > 0
        entropy = -(p[nonzero] * p[nonzero].log()).sum()
        out.append(float(entropy.exp().item()))
    return out


def sid_uniqueness(indices: torch.Tensor) -> tuple[float, int]:
    """(unique_fraction, num_duplicates) over the SID tuples. `indices` is (N, L)."""
    N = indices.shape[0]
    seen: set[tuple[int, ...]] = set()
    dupes = 0
    for row in indices.tolist():
        t = tuple(row)
        if t in seen:
            dupes += 1
        else:
            seen.add(t)
    return (len(seen) / N, dupes)
