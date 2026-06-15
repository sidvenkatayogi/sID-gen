"""Token vocabulary, SID encoding, and user-ID hashing.

The decoder emits tokens from a single flat 3027-token vocabulary:

    [   0,  256)  256  codeword tokens for SID position 0  (c0)
    [ 256,  512)  256  codeword tokens for SID position 1  (c1)
    [ 512,  768)  256  codeword tokens for SID position 2  (c2)
    [ 768, 1024)  256  codeword tokens for SID position 3  (c3, collision suffix)
    [1024, 3024) 2000  user-ID tokens (hashed buckets)
    [3024, 3027)    3  PAD, BOS, EOS

Per-position offsetting lets the same code integer appear at any of the four
SID positions with a distinct embedding row.
"""

from __future__ import annotations

import hashlib
from typing import Iterable

CODEBOOK_SIZE: int = 256
NUM_SID_POSITIONS: int = 4        # c0, c1, c2 (RQ-VAE) + c3 (collision suffix)
NUM_USER_BUCKETS: int = 2000

CODEWORD_VOCAB: int = CODEBOOK_SIZE * NUM_SID_POSITIONS  # 1024
USER_TOKEN_START: int = CODEWORD_VOCAB                   # 1024
SPECIAL_TOKEN_START: int = USER_TOKEN_START + NUM_USER_BUCKETS  # 3024

PAD_ID: int = SPECIAL_TOKEN_START + 0       # 3024
BOS_ID: int = SPECIAL_TOKEN_START + 1       # 3025
EOS_ID: int = SPECIAL_TOKEN_START + 2       # 3026

VOCAB_SIZE: int = SPECIAL_TOKEN_START + 3   # 3027


def sid_to_tokens(sid: Iterable[int]) -> list[int]:
    """Convert a length-4 SID tuple to its 4-token form (token = 256*p + code)."""
    sid_list = list(sid)
    assert len(sid_list) == NUM_SID_POSITIONS, (
        f"SID must have {NUM_SID_POSITIONS} codes, got {len(sid_list)}"
    )
    out: list[int] = []
    for p, code in enumerate(sid_list):
        assert 0 <= code < CODEBOOK_SIZE, (
            f"code at position {p} out of range [0, {CODEBOOK_SIZE}): {code}"
        )
        out.append(CODEBOOK_SIZE * p + code)
    return out


def tokens_to_sid(tokens: Iterable[int]) -> tuple[int, int, int, int]:
    """Inverse of `sid_to_tokens`: subtract the per-position offset."""
    tok_list = list(tokens)
    assert len(tok_list) == NUM_SID_POSITIONS, (
        f"need {NUM_SID_POSITIONS} tokens, got {len(tok_list)}"
    )
    codes = tuple(tok - CODEBOOK_SIZE * p for p, tok in enumerate(tok_list))
    for p, c in enumerate(codes):
        assert 0 <= c < CODEBOOK_SIZE, (
            f"decoded code at position {p} out of range: token={tok_list[p]} code={c}"
        )
    return codes  # type: ignore[return-value]


def position_token_range(position: int) -> tuple[int, int]:
    """`[start, end)` of valid token IDs at SID position `position` (0..3)."""
    assert 0 <= position < NUM_SID_POSITIONS
    return (CODEBOOK_SIZE * position, CODEBOOK_SIZE * (position + 1))


def user_token(user_id: str) -> int:
    """Map a user_id to a token in `[USER_TOKEN_START, SPECIAL_TOKEN_START)`.

    MD5 (not Python's built-in `hash`, which is salted per process) keeps the
    bucket stable across training and inference runs.
    """
    digest = hashlib.md5(user_id.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % NUM_USER_BUCKETS
    return USER_TOKEN_START + bucket


def build_encoder_input(
    user_id: str,
    history_sids: list[tuple[int, int, int, int]],
    pad_to: int,
) -> tuple[list[int], list[int]]:
    """Assemble `[user_tok, sid(i_1)[0..3], ..., sid(i_h)[0..3]]`, right-padded.

    Returns `(input_ids, attention_mask)`, both length `pad_to`; the mask is 1
    on real positions and 0 on padding.
    """
    toks: list[int] = [user_token(user_id)]
    for sid in history_sids:
        toks.extend(sid_to_tokens(sid))
    assert len(toks) <= pad_to, (
        f"encoder input length {len(toks)} exceeds pad_to={pad_to} "
        f"(history len={len(history_sids)})"
    )

    attn = [1] * len(toks) + [0] * (pad_to - len(toks))
    toks = toks + [PAD_ID] * (pad_to - len(toks))
    return toks, attn


def build_decoder_io(
    target_sid: tuple[int, int, int, int],
) -> tuple[list[int], list[int]]:
    """Teacher-forcing decoder input/target for one SID:

        decoder_input  = [BOS, c0, c1, c2, c3]
        decoder_target = [c0, c1, c2, c3, EOS]
    """
    tgt_toks = sid_to_tokens(target_sid)
    decoder_input = [BOS_ID, *tgt_toks]
    decoder_target = [*tgt_toks, EOS_ID]
    return decoder_input, decoder_target
