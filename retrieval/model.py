"""TIGER's T5 encoder-decoder, backed by HuggingFace `transformers`.

The architecture is delegated to `transformers.T5ForConditionalGeneration`.
The public interface is what the rest of the package depends on:

    model  = TigerTransformer(TigerConfig(...))
    logits = model(enc_ids, enc_mask, dec_in)        # (B, L_dec, vocab)
    enc    = model.encode(enc_ids, enc_mask)         # (B, L_enc, d_model)
    dec    = model.decode(dec_in, enc, enc_mask)     # (B, L_dec, d_model)
    step   = model.logits_from_hidden(dec[:, -1, :]) # (B, vocab)

`encode`/`decode`/`logits_from_hidden` exist for the custom beam search in
`decode.py`, which re-runs the decoder on a growing prefix each step.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import T5Config, T5ForConditionalGeneration


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
    max_enc_len: int = 82            # unused by T5 (relative position bias); kept for compat
    max_dec_len: int = 5            # unused by T5; kept for compat
    pad_id: int = 3024
    rel_num_buckets: int = 32
    rel_max_distance: int = 128
    tie_embeddings: bool = True
    initializer_range: float = 0.02  # unused by T5; kept for compat


class TigerTransformer(nn.Module):
    """T5 encoder-decoder. We feed `decoder_input_ids` explicitly (the dataset
    already prepends BOS), so HF does no right-shift; loss is computed in
    `train.py`."""

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
            feed_forward_proj="relu",
            tie_word_embeddings=config.tie_embeddings,
            pad_token_id=config.pad_id,
            decoder_start_token_id=config.pad_id,
            use_cache=False,
        )
        self.t5 = T5ForConditionalGeneration(t5_config)

        # T5 scales the decoder hidden state by d_model**-0.5 before the tied
        # LM head; replicate it in `logits_from_hidden` so the beam-search path
        # matches the training-forward path.
        self._logit_scale = config.d_model ** -0.5 if config.tie_embeddings else 1.0

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def encode(
        self,
        encoder_input_ids: torch.Tensor,
        encoder_attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        out = self.t5.encoder(
            input_ids=encoder_input_ids,
            attention_mask=encoder_attn_mask,
            return_dict=True,
        )
        return out.last_hidden_state

    def decode(
        self,
        decoder_input_ids: torch.Tensor,
        encoder_hidden: torch.Tensor,
        encoder_attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        # The decoder stack applies its own causal mask; our prefixes never pad.
        out = self.t5.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden,
            encoder_attention_mask=encoder_attn_mask,
            use_cache=False,
            return_dict=True,
        )
        return out.last_hidden_state

    def logits_from_hidden(self, decoder_hidden: torch.Tensor) -> torch.Tensor:
        return self.t5.lm_head(decoder_hidden * self._logit_scale)

    def forward(
        self,
        encoder_input_ids: torch.Tensor,
        encoder_attn_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        out = self.t5(
            input_ids=encoder_input_ids,
            attention_mask=encoder_attn_mask,
            decoder_input_ids=decoder_input_ids,
            use_cache=False,
            return_dict=True,
        )
        return out.logits


__all__ = ["TigerTransformer", "TigerConfig"]
