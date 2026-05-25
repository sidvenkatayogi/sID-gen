"""Codebook health metrics."""

from __future__ import annotations

import torch


def codebook_utilization(indices: torch.Tensor, codebook_size: int) -> list[float]:
    """Fraction of codes used at least once, per level.

    indices: (N, L) integer tensor of chosen code IDs across the dataset.
    Returns a list of L floats in [0, 1].
    """
    L = indices.shape[1]
    out: list[float] = []
    for l in range(L):
        used = torch.unique(indices[:, l]).numel()
        out.append(used / codebook_size)
    return out


def codebook_perplexity(indices: torch.Tensor, codebook_size: int) -> list[float]:
    """Perplexity = exp(entropy of code-usage distribution), per level.

    Range is [1, K]. 1 means all assignments collapsed to a single code;
    K means perfectly uniform usage. A healthy run has perplexity well
    above 1 (but doesn't need to hit K — natural data has structure).
    """
    L = indices.shape[1]
    out: list[float] = []
    for l in range(L):
        counts = torch.bincount(indices[:, l], minlength=codebook_size).float()
        p = counts / counts.sum().clamp(min=1)
        # Shannon entropy, with 0 log 0 := 0.
        nonzero = p > 0
        entropy = -(p[nonzero] * p[nonzero].log()).sum()
        out.append(float(entropy.exp().item()))
    return out


def sid_uniqueness(indices: torch.Tensor) -> tuple[float, int]:
    """Fraction of rows with a unique SID tuple, plus the collision count.

    indices: (N, L). Returns (unique_fraction, num_duplicates).
    """
    N = indices.shape[0]
    # Convert each row to a tuple of ints to use Python set semantics.
    seen: set[tuple[int, ...]] = set()
    dupes = 0
    for row in indices.tolist():
        t = tuple(row)
        if t in seen:
            dupes += 1
        else:
            seen.add(t)
    return (len(seen) / N, dupes)
