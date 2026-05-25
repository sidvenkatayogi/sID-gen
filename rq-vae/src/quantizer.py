"""Residual vector quantizer.

L codebooks, each holding K vectors of dim D. A latent z is quantized by
walking the levels: at each level pick the nearest code, subtract it, pass
the residual to the next level. L=1 is plain VQ (no residual at all, since
the loop runs once).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualQuantizer(nn.Module):
    def __init__(
        self,
        num_levels: int,
        codebook_size: int,
        latent_dim: int,
        use_ema: bool = True,
        ema_decay: float = 0.99,
        ema_eps: float = 1e-5,
    ):
        super().__init__()
        self.L = num_levels
        self.K = codebook_size
        self.D = latent_dim
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.ema_eps = ema_eps

        # One codebook per level, shape (K, D). Stored as a single (L, K, D)
        # tensor for cleanliness. Random init for now — we'll replace with
        # k-means init in a later step.
        codebooks = torch.randn(num_levels, codebook_size, latent_dim) * 0.01
        self.codebooks = nn.Parameter(codebooks, requires_grad=not use_ema)

        # EMA statistics. Buffers (not Parameters) — they're state, not
        # learned. They move with the module (cpu/gpu, state_dict) but the
        # optimizer ignores them.
        if use_ema:
            self.register_buffer("cluster_size", torch.zeros(num_levels, codebook_size))
            self.register_buffer("cluster_sum", torch.zeros(num_levels, codebook_size, latent_dim))

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

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """Residual quantization over L levels.

        For each input z:
            r_0 = z
            for l in 0..L-1:
                e_l, idx_l = nearest_code(r_l, codebook l)
                z_hat += e_l
                r_{l+1} = r_l - e_l

        Returns:
            z_hat:     (B, D)   sum of chosen codes across all levels
            indices:   (B, L)   per-level chosen code indices (the SID)
            residuals: list of L tensors (B, D) — r_l entering each level
                       (kept so the loss can compute || r_l - sg[e_l] ||^2)
        """
        B, D = z.shape
        assert D == self.D, f"latent dim {D} != expected {self.D}"

        r = z
        z_hat = torch.zeros_like(z)
        idx_list: list[torch.Tensor] = []
        residuals: list[torch.Tensor] = []

        for l in range(self.L):
            residuals.append(r)
            e, idx = self._nearest_code(r, l)
            if self.training and self.use_ema:
                self._ema_update(r, idx, l)
            z_hat = z_hat + e
            r = r - e
            idx_list.append(idx)

        indices = torch.stack(idx_list, dim=1)              # (B, L)
        return z_hat, indices, residuals

    @torch.no_grad()
    def _ema_update(self, r: torch.Tensor, idx: torch.Tensor, level: int) -> None:
        """One EMA step for codebook[level], given this batch's assignments.

        We track two running statistics per code:
            cluster_size[k]  — soft count of how often code k gets picked
            cluster_sum[k]   — running vector sum of residuals assigned to k

        New code value is cluster_sum[k] / cluster_size[k] — i.e. the mean
        residual that landed on it. Laplace smoothing keeps unused codes
        from blowing up to NaN (their counts approach 0).
        """
        # Per-code assignment matrix this batch: one_hot[b, k] = 1 iff
        # input b was assigned to code k.
        one_hot = F.one_hot(idx, num_classes=self.K).type(r.dtype)   # (B, K)

        # n_k = how many inputs landed on code k this batch.
        n = one_hot.sum(dim=0)                                       # (K,)
        # m_k = sum of all residuals assigned to code k this batch.
        #   one_hot.T : (K, B),  r : (B, D)  ->  (K, D)
        m = one_hot.t() @ r                                          # (K, D)

        # Exponential moving average: new = decay * old + (1 - decay) * batch.
        # In-place ops (mul_, add_) avoid allocating a new buffer each step.
        self.cluster_size[level].mul_(self.ema_decay).add_(n, alpha=1 - self.ema_decay)
        self.cluster_sum[level].mul_(self.ema_decay).add_(m, alpha=1 - self.ema_decay)

        # Laplace-smoothed counts: prevents div-by-zero for codes that
        # haven't been picked yet (cluster_size near 0).
        N = self.cluster_size[level].sum()
        smoothed = (self.cluster_size[level] + self.ema_eps) / (N + self.K * self.ema_eps) * N

        # Overwrite the codebook entries in place. .data sidesteps autograd
        # bookkeeping — fine because we're in no_grad and the param has
        # requires_grad=False in EMA mode anyway.
        self.codebooks.data[level] = self.cluster_sum[level] / smoothed.unsqueeze(1)
