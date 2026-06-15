"""Beam search over the 4 codeword tokens.

At decoder step t the vocabulary is masked to the per-position slice
`[256*t, 256*(t+1))`, so every decoded tuple is structurally valid (each token
lives in the right position). Whether the tuple maps to a real item is checked
later via `sid_to_item`. Beam scores are additive log-probabilities. Inference
only — do not call inside an autograd graph.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tiger.retrieval.model import TigerTransformer
from tiger.retrieval.vocab import BOS_ID, CODEBOOK_SIZE, NUM_SID_POSITIONS


@torch.no_grad()
def beam_decode(
    model: TigerTransformer,
    encoder_input_ids: torch.Tensor,    # (B, L_enc)
    encoder_attn_mask: torch.Tensor,    # (B, L_enc)
    beam_width: int = 50,
) -> list[list[tuple[tuple[int, int, int, int], float]]]:
    """Return per-row beams ranked by score:
    `out[b] = [((c0,c1,c2,c3), log_prob), ...]`, length `beam_width`, sorted desc."""
    model.eval()
    device = encoder_input_ids.device
    B, _ = encoder_input_ids.shape
    K = beam_width

    enc_hidden = model.encode(encoder_input_ids, encoder_attn_mask)        # (B, L_enc, D)
    L_enc, D = enc_hidden.shape[1], enc_hidden.shape[2]

    # Step 0: score c0 in B-space, then fan out to K beams per row. Doing this
    # in B-space (not pre-replicated B*K with -inf padding) avoids topk
    # tie-breaking among -inf entries producing duplicate c0 picks.
    bos_input = torch.full((B, 1), BOS_ID, dtype=torch.long, device=device)
    dec_hidden = model.decode(bos_input, enc_hidden, encoder_attn_mask)    # (B, 1, D)
    step0_logits = model.logits_from_hidden(dec_hidden[:, -1, :])          # (B, V)
    step0_slice = step0_logits[:, 0:CODEBOOK_SIZE]                         # (B, 256)
    step0_logp = F.log_softmax(step0_slice, dim=-1)
    step0_scores, step0_codes = step0_logp.topk(K, dim=-1)                 # (B, K)

    bos_tile = torch.full((B, K, 1), BOS_ID, dtype=torch.long, device=device)
    c0_tok = step0_codes.unsqueeze(-1)                                     # slice_start=0
    cur_input = torch.cat([bos_tile, c0_tok], dim=-1).view(B * K, -1)      # (B*K, 2)
    scores = step0_scores                                                  # (B, K)

    enc_hidden_rep = (
        enc_hidden.unsqueeze(1).expand(-1, K, -1, -1).reshape(B * K, L_enc, D)
    )
    enc_mask_rep = (
        encoder_attn_mask.unsqueeze(1).expand(-1, K, -1).reshape(B * K, L_enc)
    )

    for t in range(1, NUM_SID_POSITIONS):              # t in {1,2,3} predicts c1, c2, c3
        dec_hidden = model.decode(cur_input, enc_hidden_rep, enc_mask_rep)  # (B*K, t+1, D)
        logits = model.logits_from_hidden(dec_hidden[:, -1, :])             # (B*K, V)

        slice_start = CODEBOOK_SIZE * t
        slice_end = CODEBOOK_SIZE * (t + 1)
        valid_logits = logits[:, slice_start:slice_end]                     # (B*K, 256)
        logp = F.log_softmax(valid_logits, dim=-1).view(B, K, CODEBOOK_SIZE)

        combined = scores.unsqueeze(-1) + logp                              # (B, K, 256)
        flat = combined.view(B, K * CODEBOOK_SIZE)
        topk_scores, topk_idx = flat.topk(K, dim=-1)                        # (B, K)

        beam_idx = topk_idx // CODEBOOK_SIZE
        code_idx = topk_idx % CODEBOOK_SIZE
        token_idx = code_idx + slice_start

        cur_input_3d = cur_input.view(B, K, -1)
        gather_idx = beam_idx.unsqueeze(-1).expand(-1, -1, cur_input_3d.size(-1))
        new_prefix = torch.gather(cur_input_3d, dim=1, index=gather_idx)
        new_prefix = torch.cat([new_prefix, token_idx.unsqueeze(-1)], dim=-1)
        cur_input = new_prefix.view(B * K, -1)
        scores = topk_scores

    # cur_input is (B*K, 5) = [BOS, c0_tok, c1_tok, c2_tok, c3_tok].
    final = cur_input.view(B, K, NUM_SID_POSITIONS + 1)
    codeword_tokens = final[:, :, 1:]                                        # drop BOS -> (B, K, 4)
    pos_offset = (
        torch.arange(NUM_SID_POSITIONS, device=device) * CODEBOOK_SIZE
    ).view(1, 1, -1)
    codes = codeword_tokens - pos_offset                                     # (B, K, 4)

    codes_cpu = codes.cpu().tolist()
    scores_cpu = scores.cpu().tolist()
    out: list[list[tuple[tuple[int, int, int, int], float]]] = []
    for b in range(B):
        beams: list[tuple[tuple[int, int, int, int], float]] = []
        for k in range(K):
            sid = tuple(codes_cpu[b][k])
            beams.append((sid, float(scores_cpu[b][k])))  # type: ignore[arg-type]
        out.append(beams)
    return out
