"""Beam search over the 4 codeword tokens.

Decoding strategy (SPEC §8):
    - Always decode exactly 4 codewords (positions c0, c1, c2, c3).
    - At decoder step t, mask the vocabulary so only tokens in the per-position
      slice `[256*t, 256*(t+1))` are sampled. Anything else gets -inf, which
      guarantees the decoded tuple is *structurally* valid (i.e. each token
      lives in the right position-slice). Whether it maps to a real item is
      checked later via `sid_to_item`.

Layout used inside the beam search:
    - Step 0 is special-cased: we start with one beam per row (B total),
      score the first codeword, and *fan out* to K beams per row.
    - Steps 1..3 work with `(B*K, ...)` tensors where K is beam width.
      We reshape to `(B, K, ...)` for top-K selection and flat back for
      the next decoder pass.
    - Beam scores are log-probabilities (additive across steps).

Beam search is a pure inference operation; do not call this from inside a
training step's autograd graph.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from retrieval.model import TigerTransformer
from retrieval.vocab import BOS_ID, CODEBOOK_SIZE, NUM_SID_POSITIONS


@torch.no_grad()
def beam_decode(
    model: TigerTransformer,
    encoder_input_ids: torch.Tensor,    # (B, L_enc) long
    encoder_attn_mask: torch.Tensor,    # (B, L_enc) {0,1}
    beam_width: int = 50,
) -> list[list[tuple[tuple[int, int, int, int], float]]]:
    """Return per-row beams ranked by score.

    Output layout:
        out[b] = [((c0,c1,c2,c3), log_prob), ...]   length beam_width, sorted desc
    """
    model.eval()
    device = encoder_input_ids.device
    B, _ = encoder_input_ids.shape
    K = beam_width

    # ---- Encode once -------------------------------------------------------
    enc_hidden = model.encode(encoder_input_ids, encoder_attn_mask)        # (B, L_enc, D)
    L_enc, D = enc_hidden.shape[1], enc_hidden.shape[2]

    # ---- Step 0: B beams in, B*K beams out ---------------------------------
    # Every example starts with prefix [BOS]. We score the first codeword,
    # then fan out to K beams per row. Doing this in B-space (not B*K) avoids
    # the tie-breaking issue that arises if we pre-replicate to K beams with
    # `(K-1)` of them held at -inf — topk's tie-breaking among -inf entries
    # is implementation-defined and can produce duplicate c0 picks.
    bos_input = torch.full((B, 1), BOS_ID, dtype=torch.long, device=device)
    dec_hidden = model.decode(bos_input, enc_hidden, encoder_attn_mask)    # (B, 1, D)
    step0_logits = model.logits_from_hidden(dec_hidden[:, -1, :])          # (B, V)
    step0_slice = step0_logits[:, 0:CODEBOOK_SIZE]                         # (B, 256)
    step0_logp = F.log_softmax(step0_slice, dim=-1)                        # (B, 256)
    step0_scores, step0_codes = step0_logp.topk(K, dim=-1)                 # (B, K)

    # Build (B*K, 2) prefixes: [BOS, c0_tok]
    bos_tile = torch.full((B, K, 1), BOS_ID, dtype=torch.long, device=device)
    c0_tok = (step0_codes + 0).unsqueeze(-1)                               # (B, K, 1); slice_start=0
    cur_input = torch.cat([bos_tile, c0_tok], dim=-1).view(B * K, -1)      # (B*K, 2)
    scores = step0_scores                                                  # (B, K)

    # ---- Replicate encoder outputs across the K beams ---------------------
    enc_hidden_rep = (
        enc_hidden.unsqueeze(1).expand(-1, K, -1, -1).reshape(B * K, L_enc, D)
    )
    enc_mask_rep = (
        encoder_attn_mask.unsqueeze(1).expand(-1, K, -1).reshape(B * K, L_enc)
    )

    # ---- Steps 1..3: full (B, K) beam search per step ----------------------
    for t in range(1, NUM_SID_POSITIONS):              # t in {1,2,3} predicts c1, c2, c3
        dec_hidden = model.decode(cur_input, enc_hidden_rep, enc_mask_rep)  # (B*K, t+1, D)
        logits = model.logits_from_hidden(dec_hidden[:, -1, :])             # (B*K, V)

        slice_start = CODEBOOK_SIZE * t
        slice_end = CODEBOOK_SIZE * (t + 1)
        valid_logits = logits[:, slice_start:slice_end]                     # (B*K, 256)
        logp = F.log_softmax(valid_logits, dim=-1).view(B, K, CODEBOOK_SIZE)

        # combined[b, k, v] = running_score[b, k] + new_logp[b, k, v]
        combined = scores.unsqueeze(-1) + logp                              # (B, K, 256)
        flat = combined.view(B, K * CODEBOOK_SIZE)                          # (B, K*256)
        topk_scores, topk_idx = flat.topk(K, dim=-1)                        # (B, K)

        beam_idx = topk_idx // CODEBOOK_SIZE                                # (B, K)
        code_idx = topk_idx % CODEBOOK_SIZE                                 # (B, K)
        token_idx = code_idx + slice_start

        # Gather previous prefixes along the beam axis, then append new token.
        cur_input_3d = cur_input.view(B, K, -1)
        gather_idx = beam_idx.unsqueeze(-1).expand(-1, -1, cur_input_3d.size(-1))
        new_prefix = torch.gather(cur_input_3d, dim=1, index=gather_idx)    # (B, K, t+1)
        new_prefix = torch.cat([new_prefix, token_idx.unsqueeze(-1)], dim=-1)
        cur_input = new_prefix.view(B * K, -1)
        scores = topk_scores

    # ---- Convert codeword tokens back to (c0, c1, c2, c3) tuples ----------
    # cur_input final shape (B*K, 5) = [BOS, c0_tok, c1_tok, c2_tok, c3_tok].
    final = cur_input.view(B, K, NUM_SID_POSITIONS + 1)
    # Drop the BOS column; subtract position offset to recover raw codes.
    codeword_tokens = final[:, :, 1:]                                        # (B, K, 4)
    pos_offset = (
        torch.arange(NUM_SID_POSITIONS, device=device) * CODEBOOK_SIZE
    ).view(1, 1, -1)
    codes = codeword_tokens - pos_offset                                     # (B, K, 4)

    # ---- Materialize to Python list ---------------------------------------
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
