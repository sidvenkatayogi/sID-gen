"""Token vocabulary, SID encoding, and user-ID hashing.

The decoder emits tokens from a single flat 3027-vocab. Layout (SPEC §4):

    [   0,  256)  256  codeword tokens for SID position 0  (c0)
    [ 256,  512)  256  codeword tokens for SID position 1  (c1)
    [ 512,  768)  256  codeword tokens for SID position 2  (c2)
    [ 768, 1024)  256  codeword tokens for SID position 3  (c3, collision suffix)
    [1024, 3024) 2000  user-ID tokens (hashed buckets)
    [3024, 3027)    3  PAD, BOS, EOS

Per-position offsetting is the trick that lets the same integer (e.g. `5`)
appear at any of the four SID positions without confusing the model — the
embedding rows for `(pos=0, code=5)`, `(pos=1, code=5)`, etc. are distinct
slots in the table. The paper uses this exact scheme.

User IDs are hashed into 2000 buckets via MD5 (NOT Python's built-in
`hash()` — that is salted per-process and would change between training and
inference runs).
"""

from __future__ import annotations

import hashlib
from typing import Iterable

# ---- Constants (must match SPEC §4) -----------------------------------------
CODEBOOK_SIZE: int = 256         # K per RQ-VAE level
NUM_SID_POSITIONS: int = 4       # c0, c1, c2 (RQ-VAE) + c3 (collision suffix)
NUM_USER_BUCKETS: int = 2000

CODEWORD_VOCAB: int = CODEBOOK_SIZE * NUM_SID_POSITIONS  # 1024
USER_TOKEN_START: int = CODEWORD_VOCAB                   # 1024
SPECIAL_TOKEN_START: int = USER_TOKEN_START + NUM_USER_BUCKETS  # 3024

PAD_ID: int = SPECIAL_TOKEN_START + 0       # 3024
BOS_ID: int = SPECIAL_TOKEN_START + 1       # 3025
EOS_ID: int = SPECIAL_TOKEN_START + 2       # 3026

VOCAB_SIZE: int = SPECIAL_TOKEN_START + 3   # 3027


# ---- SID <-> token round-trip ----------------------------------------------
def sid_to_tokens(sid: Iterable[int]) -> list[int]:
    """Convert a length-4 SID tuple `(c0, c1, c2, c3)` to its 4-token form.

    Token at position p = `CODEBOOK_SIZE * p + code`. The offset is what
    keeps positions distinguishable inside a single embedding table.
    """
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
    """Inverse of `sid_to_tokens`. Subtracts the position offset.

    Tolerant of any tokens in `[0, 1024)`; does not validate that each token
    is in its expected per-position range — beam search guarantees that by
    masking, and downstream code may want to inspect raw decoded tokens for
    debugging.
    """
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


# ---- Per-position decode mask ----------------------------------------------
def position_token_range(position: int) -> tuple[int, int]:
    """`[start, end)` of valid token IDs at SID position `position` (0..3)."""
    assert 0 <= position < NUM_SID_POSITIONS
    return (CODEBOOK_SIZE * position, CODEBOOK_SIZE * (position + 1))


# ---- User-ID hashing (stable across processes) -----------------------------
def user_token(user_id: str) -> int:
    """Map a user_id string to a token in `[USER_TOKEN_START, SPECIAL_TOKEN_START)`.

    MD5 is used instead of Python's built-in `hash` because PYTHONHASHSEED
    randomizes the latter per process — same string would map to different
    buckets across runs and break checkpoint reuse.
    """
    digest = hashlib.md5(user_id.encode("utf-8")).digest()
    # 8 bytes is plenty of entropy for a mod-2000.
    bucket = int.from_bytes(digest[:8], "big") % NUM_USER_BUCKETS
    return USER_TOKEN_START + bucket


# ---- Sequence assembly helpers ---------------------------------------------
def build_encoder_input(
    user_id: str,
    history_sids: list[tuple[int, int, int, int]],
    pad_to: int,
) -> tuple[list[int], list[int]]:
    """Assemble the encoder input from §5.1.

        [user_tok(u), sid(i_1)[0..3], sid(i_2)[0..3], ..., sid(i_h)[0..3]]

    Returns `(input_ids, attention_mask)` both of length `pad_to`. Right-pad
    with PAD; the mask is 1 on real positions, 0 on padding.
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
    """Decoder input / target for predicting one SID (§5.2).

        decoder_input  = [BOS, c0, c1, c2, c3]      # length 5
        decoder_target = [c0, c1, c2, c3, EOS]      # length 5

    Standard teacher-forcing shift: position `t` in input predicts position
    `t` in target. The spec text drops `c3` from the input but says length
    is 5 — we follow the length-5 reading, which is the only consistent
    teacher-forcing interpretation.
    """
    tgt_toks = sid_to_tokens(target_sid)              # 4 codeword tokens
    decoder_input = [BOS_ID, *tgt_toks]               # BOS + 4 = 5
    decoder_target = [*tgt_toks, EOS_ID]              # 4 + EOS = 5
    return decoder_input, decoder_target
