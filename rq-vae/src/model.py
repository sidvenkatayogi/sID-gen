"""RQ-VAE: encoder + residual quantizer + decoder, with the training loss.

    x  -encoder->  z  -quantizer->  z_hat  -STE->  z_hat_ste  -decoder->  x_hat

Loss = L_recon + L_rq, where L_recon = ||x - x_hat||^2 and L_rq sums the
per-level commitment (and, in "loss" mode, codebook) terms. In EMA mode the
codebook term is handled by the EMA update, so only the commitment term is
kept in the backprop loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Decoder, Encoder
from .quantizer import ResidualQuantizer


@dataclass
class RQVAEOutput:
    loss: torch.Tensor
    recon_loss: torch.Tensor
    rq_loss: torch.Tensor
    x_hat: torch.Tensor
    indices: torch.Tensor              # (B, L) — the SIDs
    residuals: list[torch.Tensor]      # length L, each (B, D) — for dead-code reinit


class RQVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        encoder_hidden: list[int],
        latent_dim: int,
        num_levels: int,
        codebook_size: int,
        commitment_beta: float = 0.25,
        codebook_update: str = "ema",
        ema_decay: float = 0.99,
        ema_eps: float = 1e-5,
        normalize_latent: bool = False,
    ):
        super().__init__()
        self.encoder = Encoder(input_dim, encoder_hidden, latent_dim)
        self.decoder = Decoder(latent_dim, encoder_hidden, input_dim)
        self.quantizer = ResidualQuantizer(
            num_levels,
            codebook_size,
            latent_dim,
            use_ema=(codebook_update == "ema"),
            ema_decay=ema_decay,
            ema_eps=ema_eps,
        )
        self.beta = commitment_beta
        self.codebook_update = codebook_update
        # Project latents onto the unit sphere before quantizing. This prevents
        # magnitude collapse (||z|| -> 0, which makes residuals vanish and the
        # deeper codebooks redundant): with ||z||=1 the encoder must separate
        # items by angle, so all levels carry information. See ViT-VQGAN.
        self.normalize_latent = normalize_latent

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encoder output, optionally L2-normalized onto the unit sphere. Shared
        by forward() and the k-means codebook init so both see the same space."""
        z = self.encoder(x)
        if self.normalize_latent:
            z = F.normalize(z, dim=-1)
        return z

    def forward(self, x: torch.Tensor) -> RQVAEOutput:
        z = self.encode(x)                                   # (B, D)
        z_hat, indices, residuals = self.quantizer(z)        # z_hat: (B, D)

        # Straight-through estimator: forward value is z_hat, but gradient flows
        # to the encoder as if this were the identity z -> z.
        z_hat_ste = z + (z_hat - z).detach()
        x_hat = self.decoder(z_hat_ste)

        recon_loss = F.mse_loss(x_hat, x)

        rq_loss = x.new_zeros(())
        L = len(residuals)
        for l in range(L):
            r_l = residuals[l]
            e_l = self.quantizer.codebooks[l][indices[:, l]]   # (B, D)

            commit = F.mse_loss(r_l, e_l.detach())
            if self.codebook_update == "loss":
                codebook = F.mse_loss(r_l.detach(), e_l)
                rq_loss = rq_loss + codebook + self.beta * commit
            else:
                rq_loss = rq_loss + self.beta * commit

        loss = recon_loss + rq_loss
        return RQVAEOutput(
            loss=loss,
            recon_loss=recon_loss,
            rq_loss=rq_loss,
            x_hat=x_hat,
            indices=indices,
            residuals=residuals,
        )
