"""Residual vector quantizer.

L codebooks, each K vectors of dim D. A latent z is quantized by walking the
levels: at each level pick the nearest code, subtract it, pass the residual to
the next level. L=1 is plain VQ.
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

        # One codebook per level, stored as a single (L, K, D) tensor. Random
        # init; kmeans_init() overwrites it before training when enabled.
        codebooks = torch.randn(num_levels, codebook_size, latent_dim) * 0.01
        self.codebooks = nn.Parameter(codebooks, requires_grad=not use_ema)

        # EMA statistics are buffers (state, not learned parameters).
        if use_ema:
            self.register_buffer("cluster_size", torch.zeros(num_levels, codebook_size))
            self.register_buffer("cluster_sum", torch.zeros(num_levels, codebook_size, latent_dim))

    def _nearest_code(self, r: torch.Tensor, level: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Nearest code in codebook[level] for each row of r (B, D).
        Returns (selected codes (B, D), chosen indices (B,))."""
        cb = self.codebooks[level]                          # (K, D)
        # ||r - c||^2 = ||r||^2 - 2 r·c + ||c||^2
        r2 = (r * r).sum(dim=1, keepdim=True)               # (B, 1)
        c2 = (cb * cb).sum(dim=1)                           # (K,)
        rc = r @ cb.t()                                     # (B, K)
        dist = r2 - 2 * rc + c2                             # (B, K)
        idx = dist.argmin(dim=1)                            # (B,)
        return cb[idx], idx

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """Residual quantization over L levels. Returns:
            z_hat:     (B, D)   sum of chosen codes across all levels
            indices:   (B, L)   per-level chosen code indices (the SID)
            residuals: list of L tensors (B, D), r_l entering each level
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
        """One EMA step for codebook[level]. Tracks per-code cluster_size (soft
        count) and cluster_sum (residual sum); the new code value is their
        Laplace-smoothed ratio (the mean residual that landed on it)."""
        one_hot = F.one_hot(idx, num_classes=self.K).type(r.dtype)   # (B, K)
        n = one_hot.sum(dim=0)                                       # (K,)
        m = one_hot.t() @ r                                          # (K, D)

        self.cluster_size[level].mul_(self.ema_decay).add_(n, alpha=1 - self.ema_decay)
        self.cluster_sum[level].mul_(self.ema_decay).add_(m, alpha=1 - self.ema_decay)

        # Laplace smoothing prevents div-by-zero for never-picked codes.
        N = self.cluster_size[level].sum()
        smoothed = (self.cluster_size[level] + self.ema_eps) / (N + self.K * self.ema_eps) * N
        self.codebooks.data[level] = self.cluster_sum[level] / smoothed.unsqueeze(1)

    @torch.no_grad()
    def kmeans_init(self, z: torch.Tensor, random_state: int = 0) -> None:
        """Seed each codebook from k-means over the residuals entering that level.

        Sequential by necessity: codebook[l]'s input distribution depends on
        codebooks 0..l-1, so we fit level by level, quantize, and pass the
        residual on. Also seeds the EMA buffers so the first EMA step doesn't
        wash out the centroids.
        """
        from sklearn.cluster import KMeans

        r = z
        B = r.shape[0]
        for l in range(self.L):
            r_np = r.detach().cpu().numpy()
            n_clusters = min(self.K, B)
            km = KMeans(
                n_clusters=n_clusters,
                n_init=1,
                random_state=random_state + l,
            ).fit(r_np)

            centroids = torch.tensor(
                km.cluster_centers_, dtype=r.dtype, device=r.device
            )                                                        # (n_clusters, D)
            labels = torch.tensor(km.labels_, dtype=torch.long, device=r.device)  # (B,)

            # Fewer rows than codes: pad remaining slots with random rows of r.
            if n_clusters < self.K:
                extra = self.K - n_clusters
                pad_idx = torch.randint(0, B, (extra,), device=r.device)
                pad = r[pad_idx]
                centroids = torch.cat([centroids, pad], dim=0)       # (K, D)

            self.codebooks.data[l] = centroids

            if self.use_ema:
                counts = torch.bincount(labels, minlength=self.K).type(r.dtype)   # (K,)
                sums = torch.zeros(self.K, self.D, dtype=r.dtype, device=r.device)
                sums.index_add_(0, labels, r)                        # (K, D)
                if n_clusters < self.K:
                    counts[n_clusters:] = 1.0
                    sums[n_clusters:] = centroids[n_clusters:]
                self.cluster_size[l] = counts
                self.cluster_sum[l] = sums

            e, _ = self._nearest_code(r, l)
            r = r - e

    @torch.no_grad()
    def reinit_dead_codes(
        self,
        residuals: list[torch.Tensor],
        usage_threshold: float = 1.0,
    ) -> list[int]:
        """Resurrect codes whose EMA usage fell below `usage_threshold`, drawing
        replacements from that level's residuals in the current batch. Returns
        the per-level count reinitialized."""
        assert self.use_ema, "reinit relies on EMA cluster_size statistics"
        assert len(residuals) == self.L

        n_reinit: list[int] = []
        for l in range(self.L):
            dead = self.cluster_size[l] < usage_threshold       # (K,) bool
            n_dead = int(dead.sum().item())
            n_reinit.append(n_dead)
            if n_dead == 0:
                continue

            r_l = residuals[l]                                  # (B, D)
            B = r_l.shape[0]
            pick = torch.randint(0, B, (n_dead,), device=r_l.device)
            new_codes = r_l[pick]                               # (n_dead, D)

            self.codebooks.data[l][dead] = new_codes
            # Reset EMA buffers to (sum=new, size=1) so the next update blends
            # from the new vector rather than overwriting it with ~0/~0.
            self.cluster_sum[l][dead] = new_codes
            self.cluster_size[l][dead] = 1.0

        return n_reinit
