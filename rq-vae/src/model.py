"""RQ-VAE: encoder + residual quantizer + decoder, with the training loss.

Forward pass:
    x  -encoder->  z  -quantizer->  z_hat  -STE->  z_hat_ste  -decoder->  x_hat

Loss (SPEC):
    L = L_recon + L_rq
    L_recon = || x - x_hat ||^2
    L_rq    = sum_l ( || sg[r_l] - e_l ||^2  +  beta * || r_l - sg[e_l] ||^2 )

When codebook updates are EMA (default), the codebook term || sg[r_l] - e_l ||^2
is handled by the EMA update (added later) — we drop it from the backprop loss
and keep only the commitment term.
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
    indices: torch.Tensor   # (B, L) — the SIDs


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

    def forward(self, x: torch.Tensor) -> RQVAEOutput:
        z = self.encoder(x)                                  # (B, D)
        z_hat, indices, residuals = self.quantizer(z)        # z_hat: (B, D)

        # ---- Straight-through estimator ----------------------------------
        # The quantizer's z_hat is built from cb[idx] lookups, and argmin
        # blocks gradient. Passing z_hat to the decoder as-is would give
        # the encoder zero gradient from the recon loss.
        #
        # Trick: in the forward pass, the value is z_hat (quantized). In the
        # backward pass, gradient flows as if the operation were the identity
        # z -> z. Mathematically:
        #     z_hat_ste = z + (z_hat - z).detach()
        # Forward: z + (z_hat - z) = z_hat  ✓
        # Backward: (z_hat - z) has no grad (detached), so d/dz = 1.
        z_hat_ste = z + (z_hat - z).detach()
        x_hat = self.decoder(z_hat_ste)

        # ---- Losses ------------------------------------------------------
        recon_loss = F.mse_loss(x_hat, x)

        # L_rq: per-level codebook + commitment terms.
        # We already picked codes during the quantizer forward — just look
        # them up by the stored indices instead of redoing the search.
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
                # EMA mode: codebook term is handled by EMA update (added later).
                rq_loss = rq_loss + self.beta * commit

        loss = recon_loss + rq_loss
        return RQVAEOutput(
            loss=loss,
            recon_loss=recon_loss,
            rq_loss=rq_loss,
            x_hat=x_hat,
            indices=indices,
        )
