"""T5-style encoder-decoder transformer for TIGER.

Architecture (SPEC §6, paper's "small" config):
    d_model=128, d_ff=1024, num_heads=6, head_dim=64, layers=4+4,
    relative position biases (T5-style), tied embeddings, ReLU FFN.

Compared to the paper's ~13M param claim, this exact config produces ~5M
trainable params — the paper figure appears to be optimistic for these
hyperparameters. Functionally still matches; printing `model.num_params()`
gives the actual count at instantiation.

Conventions:
    - All masks passed *into* attention modules are BOOLEAN where
      True == "keep this position", False == "mask it out". Internally,
      attention turns them into additive -inf where masked.
    - RMSNorm + pre-norm (T5 default).
    - Relative position bias lives on the first self-attention of each stack
      (encoder + decoder). Subsequent layers receive the bias tensor as input
      so we compute it once per forward pass and reuse it. This matches T5X.
    - Cross-attention has NO relative position bias (no meaningful relative
      distance between an encoder token and a decoder token).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Building blocks
# ============================================================================
class RMSNorm(nn.Module):
    """Root-mean-square layer norm (T5 default). Only a per-feature scale —
    no bias, no centering. Lighter than LayerNorm and is what T5 actually uses."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # rms = sqrt(mean(x^2)) along the last dim. Stay in fp32 for the
        # reciprocal-sqrt; precision matters here in bf16 training.
        x_dtype = x.dtype
        x32 = x.float()
        rms = x32.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        out = (x32 * rms).to(x_dtype)
        return out * self.weight


class FeedForward(nn.Module):
    """Plain two-layer FFN with ReLU activation, matching SPEC §6.

    Some T5 variants use gated activations (GeGLU/SwiGLU) but the original
    T5 and this spec use vanilla ReLU(W1 x) W2.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.down(self.drop(F.relu(self.up(x)))))


# ============================================================================
# T5 relative position bias
# ============================================================================
class RelativePositionBias(nn.Module):
    """T5-style learned relative position bias.

    For each (query_pos, key_pos), we look up a per-head bias from a small
    embedding table indexed by a *bucketed* relative position. Bucketing is
    logarithmic for larger distances so we get fine resolution near 0 and
    coarse resolution out at ±max_distance.

    Returns a `(num_heads, q_len, k_len)` tensor added to attention scores
    BEFORE the softmax. Shared across layers within a stack (we compute it
    once per forward and pass it down).
    """

    def __init__(
        self,
        num_buckets: int,
        max_distance: int,
        num_heads: int,
        bidirectional: bool,
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.num_heads = num_heads
        self.bidirectional = bidirectional
        self.bias = nn.Embedding(num_buckets, num_heads)

    @staticmethod
    def _bucket(
        relative_position: torch.Tensor,
        bidirectional: bool,
        num_buckets: int,
        max_distance: int,
    ) -> torch.Tensor:
        """Port of T5's `_relative_position_bucket`.

        bidirectional=True splits buckets into negative/positive halves.
        bidirectional=False (decoder self-attn) treats positive offsets
        (future positions) as bucket 0 — they're masked anyway by the
        causal mask, so the bucket assignment doesn't matter for them.
        """
        ret = torch.zeros_like(relative_position)
        # We want the bucket of `key_pos - query_pos`; bigger means the key
        # is to the right of the query. T5's convention is to negate first
        # because they bucket `query - key` (it doesn't matter — the
        # embedding table is learned either way).
        n = -relative_position

        if bidirectional:
            num_buckets //= 2
            ret = (n < 0).to(torch.long) * num_buckets
            n = torch.abs(n)
        else:
            n = torch.clamp(n, min=0)

        # Bucket 0..max_exact-1 are reserved for "exact" small distances.
        # Beyond that we use a log-spaced bucket: bucket = max_exact +
        # round(log(n/max_exact) / log(max_distance/max_exact) *
        # (num_buckets - max_exact)).
        max_exact = num_buckets // 2
        is_small = n < max_exact

        # `clamp(min=1)` keeps log(0) from blowing up — `is_small` filters
        # those rows away anyway.
        val_if_large = max_exact + (
            torch.log(n.float().clamp(min=1) / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.minimum(
            val_if_large, torch.full_like(val_if_large, num_buckets - 1)
        )

        ret = ret + torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, q_len: int, k_len: int, device: torch.device) -> torch.Tensor:
        """Compute the bias for q_len queries and k_len keys.

        Returns `(num_heads, q_len, k_len)`.
        """
        q_pos = torch.arange(q_len, dtype=torch.long, device=device)[:, None]   # (q,1)
        k_pos = torch.arange(k_len, dtype=torch.long, device=device)[None, :]   # (1,k)
        relative = k_pos - q_pos                                                # (q,k)
        buckets = self._bucket(
            relative,
            bidirectional=self.bidirectional,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )                                                                       # (q,k) long
        # self.bias: (num_buckets, num_heads). Index gives (q, k, num_heads),
        # then permute to (num_heads, q, k).
        b = self.bias(buckets)
        return b.permute(2, 0, 1)


# ============================================================================
# Multi-head attention (self or cross, with optional relative bias)
# ============================================================================
class MultiHeadAttention(nn.Module):
    """Generic multi-head attention.

    Used three ways:
        - encoder self-attn  (q,k,v from encoder; bidirectional)
        - decoder self-attn  (q,k,v from decoder; causal)
        - cross-attn         (q from decoder, k,v from encoder; no causal,
                              no relative bias)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim         # e.g. 6*64 = 384

        # All four projections are unbiased — matches T5.
        self.q_proj = nn.Linear(d_model, self.inner_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.inner_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.inner_dim, bias=False)
        self.o_proj = nn.Linear(self.inner_dim, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.out_drop = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, L, inner_dim) -> (B, num_heads, L, head_dim)
        B, L, _ = x.shape
        return x.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, num_heads, L, head_dim) -> (B, L, inner_dim)
        B, H, L, D = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, H * D)

    def forward(
        self,
        q_input: torch.Tensor,                    # (B, L_q, d_model)
        kv_input: torch.Tensor,                   # (B, L_k, d_model)
        *,
        attn_mask: torch.Tensor | None = None,    # (B, L_q, L_k) bool, True=keep
        position_bias: torch.Tensor | None = None,  # (num_heads, L_q, L_k) additive
    ) -> torch.Tensor:
        q = self._split_heads(self.q_proj(q_input))    # (B,H,L_q,Dh)
        k = self._split_heads(self.k_proj(kv_input))   # (B,H,L_k,Dh)
        v = self._split_heads(self.v_proj(kv_input))   # (B,H,L_k,Dh)

        # T5 actually does NOT divide by sqrt(head_dim) — the relative
        # position bias absorbs the scaling. To stay closer to a generic
        # transformer (and because some readers expect it), we keep the
        # sqrt-scaling. The bias is learned, so the model can adapt.
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # scores: (B, H, L_q, L_k)

        if position_bias is not None:
            # Broadcast over batch dim: bias is (H, L_q, L_k) -> (1, H, L_q, L_k)
            scores = scores + position_bias.unsqueeze(0)

        if attn_mask is not None:
            # attn_mask: (B, L_q, L_k) bool. False -> -inf.
            mask = attn_mask.unsqueeze(1)                        # (B,1,L_q,L_k)
            scores = scores.masked_fill(~mask, float("-inf"))

        # Softmax in fp32 for numerical stability under bf16/fp16.
        attn = scores.float().softmax(dim=-1).to(q.dtype)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)                              # (B,H,L_q,Dh)
        out = self._merge_heads(out)
        return self.out_drop(self.o_proj(out))


# ============================================================================
# Encoder / Decoder blocks (pre-norm)
# ============================================================================
class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, head_dim: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, num_heads, head_dim, dropout)
        self.norm2 = RMSNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,                         # (B, L, d_model)
        attn_mask: torch.Tensor,                 # (B, L, L)
        position_bias: torch.Tensor,             # (H, L, L)
    ) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.drop(self.self_attn(h, h, attn_mask=attn_mask, position_bias=position_bias))
        h = self.norm2(x)
        x = x + self.drop(self.ffn(h))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, head_dim: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, num_heads, head_dim, dropout)
        self.norm2 = RMSNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, head_dim, dropout)
        self.norm3 = RMSNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,                         # (B, L_dec, d_model)
        encoder_out: torch.Tensor,               # (B, L_enc, d_model)
        self_attn_mask: torch.Tensor,            # (B, L_dec, L_dec) causal
        cross_attn_mask: torch.Tensor,           # (B, L_dec, L_enc) keep-encoder-tokens
        self_position_bias: torch.Tensor,        # (H, L_dec, L_dec)
    ) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.drop(
            self.self_attn(h, h, attn_mask=self_attn_mask, position_bias=self_position_bias)
        )
        h = self.norm2(x)
        # No position_bias for cross-attn (T5 convention).
        x = x + self.drop(self.cross_attn(h, encoder_out, attn_mask=cross_attn_mask))
        h = self.norm3(x)
        x = x + self.drop(self.ffn(h))
        return x


# ============================================================================
# Full model
# ============================================================================
@dataclass
class TigerConfig:
    vocab_size: int = 3027
    d_model: int = 128
    d_ff: int = 1024
    num_heads: int = 6
    head_dim: int = 64
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    dropout: float = 0.1
    max_enc_len: int = 82
    max_dec_len: int = 5
    pad_id: int = 3024
    rel_num_buckets: int = 32
    rel_max_distance: int = 128
    tie_embeddings: bool = True
    initializer_range: float = 0.02


class TigerTransformer(nn.Module):
    """The whole TIGER transformer in one module.

    Forward signature (training):
        out = model(encoder_input_ids, encoder_attn_mask, decoder_input_ids)
        out.logits   # (B, L_dec, vocab)

    For beam search use:
        enc_out = model.encode(encoder_input_ids, encoder_attn_mask)
        logits  = model.decode(decoder_input_ids, enc_out, encoder_attn_mask)
        # logits: (B, L_dec_so_far, vocab) — slice [:, -1, :] for current step
    """

    def __init__(self, config: TigerConfig):
        super().__init__()
        self.config = config

        self.embed = nn.Embedding(config.vocab_size, config.d_model)

        self.enc_rel_bias = RelativePositionBias(
            num_buckets=config.rel_num_buckets,
            max_distance=config.rel_max_distance,
            num_heads=config.num_heads,
            bidirectional=True,
        )
        self.dec_rel_bias = RelativePositionBias(
            num_buckets=config.rel_num_buckets,
            max_distance=config.rel_max_distance,
            num_heads=config.num_heads,
            bidirectional=False,
        )

        self.encoder_layers = nn.ModuleList(
            [
                EncoderBlock(
                    config.d_model, config.num_heads, config.head_dim, config.d_ff, config.dropout
                )
                for _ in range(config.num_encoder_layers)
            ]
        )
        self.encoder_final_norm = RMSNorm(config.d_model)

        self.decoder_layers = nn.ModuleList(
            [
                DecoderBlock(
                    config.d_model, config.num_heads, config.head_dim, config.d_ff, config.dropout
                )
                for _ in range(config.num_decoder_layers)
            ]
        )
        self.decoder_final_norm = RMSNorm(config.d_model)

        self.embed_drop = nn.Dropout(config.dropout)

        if config.tie_embeddings:
            self.lm_head: nn.Linear | None = None    # use self.embed.weight at fwd time
        else:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self._init_weights()

    # ---- Init -------------------------------------------------------------
    def _init_weights(self) -> None:
        """Normal init with a small std. The exact init doesn't matter much
        for a model this small but small std keeps early training stable."""
        std = self.config.initializer_range
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, std)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                m.weight.data.normal_(0.0, std)

    # ---- Convenience ------------------------------------------------------
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ---- Mask helpers -----------------------------------------------------
    @staticmethod
    def _padding_keep_mask(attn_mask_1d: torch.Tensor, q_len: int) -> torch.Tensor:
        """Turn a 1-D attention mask (B, L_k) into a 2-D keep-mask (B, q_len, L_k).

        Every query attends to the same set of non-pad keys — the broadcast
        over the query dimension is just `unsqueeze(1)`.
        """
        return attn_mask_1d.bool().unsqueeze(1).expand(-1, q_len, -1)

    @staticmethod
    def _causal_keep_mask(L: int, device: torch.device) -> torch.Tensor:
        """(L, L) lower-triangular True mask. True = keep, False = mask out."""
        return torch.ones(L, L, dtype=torch.bool, device=device).tril()

    # ---- Forward pieces ---------------------------------------------------
    def encode(
        self,
        encoder_input_ids: torch.Tensor,       # (B, L_enc) long
        encoder_attn_mask: torch.Tensor,       # (B, L_enc) long {0,1}
    ) -> torch.Tensor:
        """Run the encoder stack. Returns hidden states (B, L_enc, d_model)."""
        B, L_enc = encoder_input_ids.shape
        x = self.embed_drop(self.embed(encoder_input_ids))

        keep_mask = self._padding_keep_mask(encoder_attn_mask, q_len=L_enc)   # (B, L, L)
        pos_bias = self.enc_rel_bias(L_enc, L_enc, encoder_input_ids.device)  # (H, L, L)

        for blk in self.encoder_layers:
            x = blk(x, attn_mask=keep_mask, position_bias=pos_bias)
        return self.encoder_final_norm(x)

    def decode(
        self,
        decoder_input_ids: torch.Tensor,       # (B, L_dec) long
        encoder_hidden: torch.Tensor,          # (B, L_enc, d_model)
        encoder_attn_mask: torch.Tensor,       # (B, L_enc) long {0,1}
    ) -> torch.Tensor:
        """Run the decoder stack on a prefix of length L_dec. Returns
        hidden states (B, L_dec, d_model)."""
        B, L_dec = decoder_input_ids.shape
        L_enc = encoder_hidden.shape[1]
        device = decoder_input_ids.device

        x = self.embed_drop(self.embed(decoder_input_ids))

        # decoder self-attn: causal mask, no padding inside the decoder
        # (every example has exactly L_dec real positions during training,
        # and during inference we always feed real tokens).
        causal = self._causal_keep_mask(L_dec, device)                      # (L_dec, L_dec)
        self_keep = causal.unsqueeze(0).expand(B, -1, -1)                   # (B, L_dec, L_dec)
        self_pos_bias = self.dec_rel_bias(L_dec, L_dec, device)             # (H, L_dec, L_dec)

        # cross-attn: each decoder position can attend to all non-padded
        # encoder positions.
        cross_keep = self._padding_keep_mask(encoder_attn_mask, q_len=L_dec)  # (B, L_dec, L_enc)

        for blk in self.decoder_layers:
            x = blk(
                x,
                encoder_out=encoder_hidden,
                self_attn_mask=self_keep,
                cross_attn_mask=cross_keep,
                self_position_bias=self_pos_bias,
            )
        return self.decoder_final_norm(x)

    def logits_from_hidden(self, decoder_hidden: torch.Tensor) -> torch.Tensor:
        """Project decoder hidden states to vocabulary logits.

        Tied: `logits = h @ embed.weight.T`. Untied: a separate linear.
        """
        if self.lm_head is None:
            return decoder_hidden @ self.embed.weight.T
        return self.lm_head(decoder_hidden)

    def forward(
        self,
        encoder_input_ids: torch.Tensor,
        encoder_attn_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """End-to-end forward pass. Returns logits (B, L_dec, vocab)."""
        enc = self.encode(encoder_input_ids, encoder_attn_mask)
        dec = self.decode(decoder_input_ids, enc, encoder_attn_mask)
        return self.logits_from_hidden(dec)


__all__ = ["TigerTransformer", "TigerConfig"]
