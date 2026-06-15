"""Torch Dataset producing encoder/decoder tensors for the TIGER transformer.

Each row of `{train,val,test}.jsonl` becomes one example:

    encoder input:   [user_token(u), sid(i_1)[0..3], ..., sid(i_h)[0..3]]  (right-padded)
    encoder mask:    1 on real positions, 0 on PAD
    decoder input:   [BOS, c0, c1, c2, c3]
    decoder labels:  [c0, c1, c2, c3, EOS]

History is truncated to the most recent 20 items.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import Dataset

from tiger.retrieval.vocab import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    build_decoder_io,
    build_encoder_input,
    sid_to_tokens,
    user_token,
)

HISTORY_CAP: int = 20
MAX_ENC_LEN: int = 82        # 1 user token + 4*20 codeword tokens, rounded up
MAX_DEC_LEN: int = 5         # BOS + 4 codewords / 4 codewords + EOS


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_item_to_sid(path: Path) -> dict[str, tuple[int, int, int, int]]:
    with open(path, "r") as f:
        raw = json.load(f)
    out: dict[str, tuple[int, int, int, int]] = {}
    for k, v in raw.items():
        assert len(v) == 4, f"item {k} has SID of length {len(v)} (expected 4)"
        out[k] = tuple(v)  # type: ignore[assignment]
    return out


class TigerSequenceDataset(Dataset):
    """One example per row in a jsonl split.

    Each item is a dict of tensors: `encoder_input_ids`, `encoder_attn_mask`,
    `decoder_input_ids`, `decoder_labels`, and `target_sid` (the gold item's
    raw 4-code SID, used by eval).
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        item_to_sid: dict[str, tuple[int, int, int, int]] | str | Path,
        history_cap: int = HISTORY_CAP,
        max_enc_len: int = MAX_ENC_LEN,
    ):
        self.history_cap = history_cap
        self.max_enc_len = max_enc_len

        if isinstance(item_to_sid, (str, Path)):
            self.item_to_sid = _load_item_to_sid(Path(item_to_sid))
        else:
            self.item_to_sid = dict(item_to_sid)

        all_rows = _load_jsonl(Path(jsonl_path))

        kept: list[dict] = []
        dropped = 0
        for r in all_rows:
            if r["target"] not in self.item_to_sid:
                dropped += 1
                continue
            kept.append(r)
        if dropped:
            print(
                f"[dataset] {jsonl_path}: dropped {dropped}/{len(all_rows)} rows "
                f"(target has no SID)"
            )
        self.rows: list[dict] = kept

    def __len__(self) -> int:
        return len(self.rows)

    def _history_sids(self, history: Sequence[str]) -> list[tuple[int, int, int, int]]:
        """Keep the most recent `history_cap` items and look up their SIDs.
        Items absent from the SID map are dropped (empty in a well-formed run)."""
        recent = history[-self.history_cap :]
        sids: list[tuple[int, int, int, int]] = []
        for iid in recent:
            sid = self.item_to_sid.get(iid)
            if sid is not None:
                sids.append(sid)
        return sids

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        history_sids = self._history_sids(row["history"])
        target_sid = self.item_to_sid[row["target"]]

        enc_ids, enc_mask = build_encoder_input(
            user_id=row["user_id"],
            history_sids=history_sids,
            pad_to=self.max_enc_len,
        )
        dec_in, dec_tgt = build_decoder_io(target_sid)

        return {
            "encoder_input_ids": torch.tensor(enc_ids, dtype=torch.long),
            "encoder_attn_mask": torch.tensor(enc_mask, dtype=torch.long),
            "decoder_input_ids": torch.tensor(dec_in, dtype=torch.long),
            "decoder_labels":    torch.tensor(dec_tgt, dtype=torch.long),
            "target_sid":        torch.tensor(list(target_sid), dtype=torch.long),
        }


def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in batch[0]}


__all__ = [
    "TigerSequenceDataset",
    "collate",
    "HISTORY_CAP",
    "MAX_ENC_LEN",
    "MAX_DEC_LEN",
    "BOS_ID",
    "EOS_ID",
    "PAD_ID",
]
