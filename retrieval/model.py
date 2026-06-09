"""T5 encoder-decoder for TIGER, backed by HuggingFace `transformers`.

This module used to hand-roll the entire T5 stack (RMSNorm, relative position
bias, multi-head attention, beam-search-friendly encode/decode). We now delegate
the architecture to `transformers.T5ForConditionalGeneration` — the *same* T5
design, but a trusted, heavily-tested implementation. The point is to rule out
a subtle bug in the from-scratch backbone as the cause of poor retrieval (our
results were sitting at the paper's random-ID floor). As a concrete example, the
hand-rolled version was missing T5's tied-embedding `d_model**-0.5` logit
scaling; HF applies it for free.

The PUBLIC INTERFACE is intentionally identical to the old module, so
`dataset.py`, `decode.py`, `eval.py`, and `train.py` keep working untouched:

    model  = TigerTransformer(TigerConfig(...))
    logits = model(enc_ids, enc_mask, dec_in)              # (B, L_dec, vocab)
    enc    = model.encode(enc_ids, enc_mask)               # (B, L_enc, d_model)
    dec    = model.decode(dec_in, enc, enc_mask)           # (B, L_dec, d_model)
    step   = model.logits_from_hidden(dec[:, -1, :])       # (B, vocab)

`encode`/`decode`/`logits_from_hidden` exist for the custom beam search in
`decode.py`, which re-runs the decoder on a growing prefix each step (no KV
cache) and slices the per-position vocabulary. We keep that beam search as-is.

Mapping our `TigerConfig` onto `transformers.T5Config`:
    d_model            -> d_model
    head_dim           -> d_kv          (inner_dim = num_heads * d_kv)
    num_heads          -> num_heads
    d_ff               -> d_ff          (feed_forward_proj="relu": vanilla T5 FFN)
    num_encoder_layers -> num_layers
    num_decoder_layers -> num_decoder_layers
    rel_num_buckets    -> relative_attention_num_buckets
    rel_max_distance   -> relative_attention_max_distance
    dropout            -> dropout_rate
    tie_embeddings     -> tie_word_embeddings
    pad_id             -> pad_token_id
`max_enc_len`/`max_dec_len`/`initializer_range` have no T5 equivalent (T5 uses
relative position bias, so there is no max-length embedding); we keep them on
the dataclass only so the existing construction call in `train.py` is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import T5Config, T5ForConditionalGeneration


# ============================================================================
# Config (unchanged public shape — train.py constructs this exactly as before)
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
    max_enc_len: int = 82          # unused by T5 (relative position bias); kept for compat
    max_dec_len: int = 5           # unused by T5; kept for compat
    pad_id: int = 3024
    rel_num_buckets: int = 32
    rel_max_distance: int = 128
    tie_embeddings: bool = True
    initializer_range: float = 0.02  # unused by T5 (uses initializer_factor); kept for compat


# ============================================================================
# Model (thin wrapper over HuggingFace T5)
# ============================================================================
class TigerTransformer(nn.Module):
    """TIGER's T5 encoder-decoder, implemented via `T5ForConditionalGeneration`.

    We feed `decoder_input_ids` explicitly (our dataset already prepends BOS),
    so HF does NOT do its usual right-shift — there is no `labels` argument in
    the forward path; the loss is still computed in `train.py`.
    """

    def __init__(self, config: TigerConfig):
        super().__init__()
        self.config = config

        t5_config = T5Config(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            d_kv=config.head_dim,
            d_ff=config.d_ff,
            num_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            num_heads=config.num_heads,
            relative_attention_num_buckets=config.rel_num_buckets,
            relative_attention_max_distance=config.rel_max_distance,
            dropout_rate=config.dropout,
            feed_forward_proj="relu",          # vanilla T5 ReLU FFN (matches SPEC §6)
            tie_word_embeddings=config.tie_embeddings,
            pad_token_id=config.pad_id,
            decoder_start_token_id=config.pad_id,  # unused (we pass decoder_input_ids); set for completeness
            use_cache=False,                   # beam search re-runs the prefix; no KV cache
        )
        self.t5 = T5ForConditionalGeneration(t5_config)

        # T5 multiplies the decoder hidden state by d_model**-0.5 before the
        # (tied) LM head. We must replicate that in `logits_from_hidden` so the
        # beam-search path matches the training-forward path exactly.
        self._logit_scale = config.d_model ** -0.5 if config.tie_embeddings else 1.0

    # ---- Convenience ------------------------------------------------------
    def num_params(self) -> int:
        # Tied weights are a single Parameter object, so .parameters() yields
        # them once — no double counting.
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ---- Forward pieces (used by beam search in decode.py) ----------------
    def encode(
        self,
        encoder_input_ids: torch.Tensor,       # (B, L_enc) long
        encoder_attn_mask: torch.Tensor,       # (B, L_enc) {0,1}
    ) -> torch.Tensor:
        """Run the encoder stack. Returns hidden states (B, L_enc, d_model)."""
        out = self.t5.encoder(
            input_ids=encoder_input_ids,
            attention_mask=encoder_attn_mask,
            return_dict=True,
        )
        return out.last_hidden_state

    def decode(
        self,
        decoder_input_ids: torch.Tensor,       # (B, L_dec) long
        encoder_hidden: torch.Tensor,          # (B, L_enc, d_model)
        encoder_attn_mask: torch.Tensor,       # (B, L_enc) {0,1}
    ) -> torch.Tensor:
        """Run the decoder stack on a prefix. Returns (B, L_dec, d_model).

        The decoder T5Stack applies its own causal mask (it is built with
        is_decoder=True), so we pass no decoder attention mask — our decoder
        prefixes never contain padding.
        """
        out = self.t5.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden,
            encoder_attention_mask=encoder_attn_mask,
            use_cache=False,
            return_dict=True,
        )
        return out.last_hidden_state

    def logits_from_hidden(self, decoder_hidden: torch.Tensor) -> torch.Tensor:
        """Project decoder hidden states to vocabulary logits, matching T5's
        tied-embedding scaling (`hidden * d_model**-0.5` then the LM head)."""
        return self.t5.lm_head(decoder_hidden * self._logit_scale)

    def forward(
        self,
        encoder_input_ids: torch.Tensor,
        encoder_attn_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """End-to-end forward pass (training). Returns logits (B, L_dec, vocab).

        Uses the full `T5ForConditionalGeneration` forward, which applies the
        same `d_model**-0.5` scaling + LM head as `logits_from_hidden`, so the
        training and beam-search logits are consistent.
        """
        out = self.t5(
            input_ids=encoder_input_ids,
            attention_mask=encoder_attn_mask,
            decoder_input_ids=decoder_input_ids,
            use_cache=False,
            return_dict=True,
        )
        return out.logits


__all__ = ["TigerTransformer", "TigerConfig"]
