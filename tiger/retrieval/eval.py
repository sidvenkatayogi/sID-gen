"""Retrieval metrics: Recall@K and NDCG@K.

Each user has one ground-truth next item and a ranked list of K predictions
(`None` for slots beam search couldn't fill). With a single relevant item the
IDCG is 1, so NDCG@K = `1 / log2(rank + 1)` if the gold item hits at rank <= K,
else 0 — averaged over users. Recall@K is the hit fraction.
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


# Paper's reported Beauty test numbers (Rajput et al., NeurIPS 2023).
PAPER_METRICS = {"recall@5": 0.0454, "ndcg@5": 0.0321, "recall@10": 0.0648, "ndcg@10": 0.0384}


def compare_to_paper(metrics: dict[str, float], paper: dict[str, float] = PAPER_METRICS):
    """Tidy ours-vs-paper comparison table (returns a pandas DataFrame)."""
    import pandas as pd

    rows = [
        {
            "metric": k,
            "ours": round(metrics[k], 4),
            "paper": ref,
            "rel_%": round((metrics[k] - ref) / ref * 100, 1),
        }
        for k, ref in paper.items()
    ]
    return pd.DataFrame(rows)
