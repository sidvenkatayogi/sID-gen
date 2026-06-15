"""Evaluate a trained checkpoint on a split and print metrics (vs. the paper)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tiger.retrieval.eval import compare_to_paper
from tiger.retrieval.train import evaluate_checkpoint


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/tiger_beauty/checkpoints/best.pt"),
    )
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--beam-width", type=int, default=50)
    ap.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args()

    metrics = evaluate_checkpoint(
        args.checkpoint,
        args.data_dir,
        split=args.split,
        beam_width=args.beam_width,
        top_k=args.top_k,
    )
    print(json.dumps(metrics, indent=2))
    if args.split == "test":
        print(compare_to_paper(metrics).to_string(index=False))


if __name__ == "__main__":
    main()
