"""Retrieval metrics: Recall@K and NDCG@K.

For each test user, we have:
    - a single ground-truth next item (`gold`)
    - a ranked list of K predicted items (best first; `None` for slots we
      couldn't fill, e.g. when beam search produced too many invalid SIDs)

Recall@K:
    1 if `gold` appears anywhere in the top-K predictions, else 0.
    Averaged across users.

NDCG@K (single-relevant-item form, SPEC §9):
    `1 / log2(rank + 1)` if `gold` hits at rank ∈ [1, K], else 0.
    Averaged across users. With one relevant item the IDCG is `1/log2(2) = 1`,
    so this matches the standard formulation without an extra normalization.
"""

from __future__ import annotations

import math
from typing import Sequence


def _rank_of_gold(preds: Sequence[str | None], gold: str) -> int | None:
    """1-based rank of `gold` in `preds`, or None if absent."""
    for i, p in enumerate(preds):
        if p == gold:
            return i + 1
    return None


def compute_retrieval_metrics(
    all_predictions: list[list[str | None]],
    all_gold: list[str],
    ks: tuple[int, ...] = (5, 10),
) -> dict[str, float]:
    """Compute Recall@k and NDCG@k for every k in `ks`.

    Each row of `all_predictions` should be at least max(ks) long. Slots
    that beam search couldn't fill can be `None` — they simply never match.
    """
    assert len(all_predictions) == len(all_gold)
    n = len(all_gold)

    out: dict[str, float] = {}
    for k in ks:
        recall_hits = 0
        ndcg_sum = 0.0
        for preds, gold in zip(all_predictions, all_gold):
            rank = _rank_of_gold(preds[:k], gold)
            if rank is not None:
                recall_hits += 1
                ndcg_sum += 1.0 / math.log2(rank + 1)
        out[f"recall@{k}"] = recall_hits / max(n, 1)
        out[f"ndcg@{k}"] = ndcg_sum / max(n, 1)

    return out
