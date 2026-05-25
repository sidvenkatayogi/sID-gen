"""Residual vector quantizer.

L codebooks, each holding K vectors of dim D. A latent z is quantized by
walking the levels: at each level pick the nearest code, subtract it, pass
the residual to the next level. L=1 is plain VQ (no residual at all, since
the loop runs once).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualQuantizer(nn.Module):
    def __init__(self, num_levels: int, codebook_size: int, latent_dim: int):
        super().__init__()
        self.L = num_levels
        self.K = codebook_size
        self.D = latent_dim

        # One codebook per level, shape (K, D). Stored as a single (L, K, D)
        # tensor for cleanliness. Random init for now — we'll replace with
        # k-means init in a later step.
        codebooks = torch.randn(num_levels, codebook_size, latent_dim) * 0.01
        self.codebooks = nn.Parameter(codebooks)

    def _nearest_code(self, r: torch.Tensor, level: int) -> tuple[torch.Tensor, torch.Tensor]:
        """For each row of r (B, D), find its nearest entry in codebook[level].

        Returns:
            e:   (B, D) the selected code vectors
            idx: (B,)   the chosen index in [0, K)
        """
        cb = self.codebooks[level]                          # (K, D)

        # Squared Euclidean distance: ||r - c||^2 = ||r||^2 - 2 r·c + ||c||^2
        # We can drop ||r||^2 (constant per row) for argmin, but we keep the
        # full expression — clearer, and the cost is negligible at K=256.
        r2 = (r * r).sum(dim=1, keepdim=True)               # (B, 1)
        c2 = (cb * cb).sum(dim=1)                           # (K,)
        rc = r @ cb.t()                                     # (B, K)
        dist = r2 - 2 * rc + c2                             # (B, K)

        idx = dist.argmin(dim=1)                            # (B,)
        e = cb[idx]                                         # (B, D)
        return e, idx
