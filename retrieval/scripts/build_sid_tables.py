"""Build `item_to_sid.json` + `sid_to_item.json` from the trained RQ-VAE.

Pipeline:
    rq-vae best.pt           -- loaded
    amazon_beauty_items.csv  -- item IDs in row order
    amazon_beauty_embeddings.npy -- (N, D) standardized content embeddings
        |
        v   restrict to items appearing in the 5-core sequences
            (if --sequences-dir is given)
        |
        v   encode through the RQ-VAE -> (c0, c1, c2) per item
        |
        v   compute c3 = collision-breaking suffix
            (items sharing (c0,c1,c2) get c3 = 0, 1, 2, ... in sorted item_id order;
            isolated items get c3 = 0)
        |
        v   write the two JSONs

The collision-breaking c3 is computed **over the filtered subset only** —
running it over all 259K Beauty items would waste codes on items the TIGER
model will never see (it only encodes items that appear in the 5-core
sequences).

Usage:
    python retrieval/scripts/build_sid_tables.py \\
        --checkpoint rq-vae/outputs/amazon_beauty_checkpoints/best.pt \\
        --items-csv  rq-vae/outputs/amazon_beauty_items.csv \\
        --embeddings rq-vae/outputs/amazon_beauty_embeddings.npy \\
        --sequences-dir data/processed/ \\
        --output-dir   data/
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from retrieval.vocab import CODEBOOK_SIZE


# ----------------------------------------------------------------------------
# RQ-VAE checkpoint loading: we call into the existing rq-vae package without
# modifying it. The package lives at <project_root>/rq-vae/src/, with `src` as
# the top-level Python module name — so we add `rq-vae/` to sys.path.
# ----------------------------------------------------------------------------
def _add_rqvae_to_path(project_root: Path) -> None:
    rqvae_root = project_root / "rq-vae"
    assert rqvae_root.exists(), f"rq-vae not found at {rqvae_root}"
    if str(rqvae_root) not in sys.path:
        sys.path.insert(0, str(rqvae_root))


def load_rqvae(checkpoint: Path, device: torch.device):
    """Build the RQ-VAE from a checkpoint. Imports happen here (not at
    module top) so the sys.path injection above takes effect first."""
    from src.model import RQVAE  # type: ignore

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    m = cfg["model"]
    model = RQVAE(
        input_dim=cfg["data"]["input_dim"],
        encoder_hidden=m["encoder_hidden"],
        latent_dim=m["latent_dim"],
        num_levels=m["num_levels"],
        codebook_size=m["codebook_size"],
        commitment_beta=m["commitment_beta"],
        codebook_update=m["codebook_update"],
        ema_decay=m["ema_decay"],
        ema_eps=m["ema_eps"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


@torch.no_grad()
def encode_to_codes(model, x: torch.Tensor, batch_size: int = 1024) -> np.ndarray:
    """Run the encoder + quantizer over `x` (N, D). Returns `(N, L)` numpy ints."""
    chunks: list[torch.Tensor] = []
    for i in range(0, x.shape[0], batch_size):
        out = model(x[i : i + batch_size])
        chunks.append(out.indices.cpu())
    return torch.cat(chunks, dim=0).numpy().astype(np.int64)


# ----------------------------------------------------------------------------
# Collision-breaking c3 suffix.
# ----------------------------------------------------------------------------
def assign_collision_suffix(
    item_ids: list[str],
    codes: np.ndarray,                  # (N, L=3) of c0,c1,c2
) -> np.ndarray:                        # (N,) of c3
    """For each (c0,c1,c2) bucket, assign c3 = 0,1,2,... in sorted item_id order.

    Isolated items get c3 = 0 (which is also what the rank-0 element of any
    bucket gets — they share the same assignment rule).
    """
    assert len(item_ids) == codes.shape[0]
    assert codes.shape[1] == 3

    # Group row indices by (c0, c1, c2).
    buckets: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for row, (c0, c1, c2) in enumerate(codes.tolist()):
        buckets[(c0, c1, c2)].append(row)

    c3 = np.zeros(len(item_ids), dtype=np.int64)
    max_bucket = 0
    for key, rows in buckets.items():
        # Sort by item_id string so the suffix is deterministic across runs.
        rows_sorted = sorted(rows, key=lambda r: item_ids[r])
        for rank, r in enumerate(rows_sorted):
            c3[r] = rank
        max_bucket = max(max_bucket, len(rows_sorted))

    assert max_bucket <= CODEBOOK_SIZE, (
        f"collision bucket size {max_bucket} exceeds c3 capacity {CODEBOOK_SIZE} — "
        f"increase NUM_SID_POSITIONS or widen the c3 vocab"
    )
    print(f"[sid] max collision bucket size = {max_bucket} (c3 capacity = {CODEBOOK_SIZE})")
    return c3


# ----------------------------------------------------------------------------
# Item-set filtering: only emit SIDs for items that appear in the preprocessed
# sequences. The TIGER model will never see anything outside this set.
# ----------------------------------------------------------------------------
def collect_items_from_sequences(sequences_dir: Path) -> set[str]:
    """Read train/val/test.jsonl from `sequences_dir`, return the union of
    item IDs that appear anywhere (in history or as a target)."""
    items: set[str] = set()
    for split in ("train.jsonl", "val.jsonl", "test.jsonl"):
        path = sequences_dir / split
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing — run retrieval/scripts/preprocess_beauty.py first"
            )
        with open(path, "r") as f:
            for line in f:
                rec = json.loads(line)
                items.update(rec["history"])
                items.add(rec["target"])
    print(f"[sid] {len(items)} unique items across train/val/test sequences")
    return items


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def build_tables(
    checkpoint: Path,
    items_csv: Path,
    embeddings: Path,
    sequences_dir: Path | None,
    output_dir: Path,
) -> None:
    # This file lives at <project_root>/retrieval/scripts/build_sid_tables.py,
    # so the project root (which holds rq-vae/) is three parents up.
    project_root = Path(__file__).resolve().parent.parent.parent
    _add_rqvae_to_path(project_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[sid] device={device}")

    # ---- 1. Load metadata + embeddings -------------------------------------
    df = pd.read_csv(items_csv)
    assert "item_id" in df.columns, f"{items_csv} must have an item_id column"
    item_ids_all = df["item_id"].astype(str).tolist()

    emb = np.load(embeddings)
    assert emb.shape[0] == len(item_ids_all), (
        f"embeddings rows {emb.shape[0]} != items_csv rows {len(item_ids_all)}"
    )
    print(f"[sid] loaded {len(item_ids_all)} items, embeddings {emb.shape}")

    # ---- 2. Restrict to items appearing in 5-core sequences ----------------
    if sequences_dir is not None:
        wanted = collect_items_from_sequences(sequences_dir)
        keep_mask = np.array([iid in wanted for iid in item_ids_all], dtype=bool)
        missing = wanted - set(np.array(item_ids_all)[keep_mask].tolist())
        if missing:
            print(
                f"[sid] WARN: {len(missing)} sequence items have no embedding "
                f"(showing 5): {sorted(missing)[:5]}"
            )
        emb = emb[keep_mask]
        item_ids = [iid for iid, k in zip(item_ids_all, keep_mask) if k]
        print(f"[sid] filtered to {len(item_ids)} items present in both")
    else:
        item_ids = item_ids_all
        print(f"[sid] no --sequences-dir; emitting SIDs for all {len(item_ids)} items")

    # ---- 3. Encode through RQ-VAE -----------------------------------------
    model, cfg = load_rqvae(checkpoint, device)
    L = cfg["model"]["num_levels"]
    K = cfg["model"]["codebook_size"]
    assert L == 3, f"this spec assumes 3 RQ-VAE levels, got {L}"
    assert K == CODEBOOK_SIZE, (
        f"RQ-VAE codebook size {K} != vocab CODEBOOK_SIZE {CODEBOOK_SIZE} — "
        f"adjust retrieval/vocab.py if you re-trained with a different K"
    )

    x = torch.from_numpy(emb).to(device)
    codes = encode_to_codes(model, x)                       # (N, 3)
    print(f"[sid] encoded to codes, shape={codes.shape}")

    # ---- 4. Collision-breaking c3 -----------------------------------------
    c3 = assign_collision_suffix(item_ids, codes)           # (N,)
    full = np.concatenate([codes, c3[:, None]], axis=1)     # (N, 4)

    # ---- 5. Write JSONs ---------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    item_to_sid: dict[str, list[int]] = {}
    sid_to_item: dict[str, str] = {}
    for iid, row in zip(item_ids, full.tolist()):
        item_to_sid[iid] = row
        key = "_".join(str(c) for c in row)
        if key in sid_to_item:
            # Should be impossible after the c3 assignment — sanity check.
            raise RuntimeError(
                f"SID collision after c3 assignment: {key} maps to both "
                f"{sid_to_item[key]} and {iid}"
            )
        sid_to_item[key] = iid

    with open(output_dir / "item_to_sid.json", "w") as f:
        json.dump(item_to_sid, f)
    with open(output_dir / "sid_to_item.json", "w") as f:
        json.dump(sid_to_item, f)

    n_isolated = int((c3 == 0).sum() - (codes.shape[0] - len(set(map(tuple, codes.tolist())))))
    print(f"[sid] wrote {output_dir/'item_to_sid.json'}  ({len(item_to_sid)} items)")
    print(f"[sid] wrote {output_dir/'sid_to_item.json'}")
    unique_prefixes = len({tuple(r[:3]) for r in full.tolist()})
    print(
        f"[sid] {unique_prefixes} unique (c0,c1,c2) prefixes across "
        f"{len(item_ids)} items  (avg collisions/prefix = "
        f"{len(item_ids)/max(unique_prefixes,1):.2f})"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("rq-vae/outputs/amazon_beauty_checkpoints/best.pt"),
        help="trained RQ-VAE checkpoint (best.pt)",
    )
    ap.add_argument(
        "--items-csv",
        type=Path,
        default=Path("rq-vae/outputs/amazon_beauty_items.csv"),
        help="per-item metadata produced by the rq-vae dataloader",
    )
    ap.add_argument(
        "--embeddings",
        type=Path,
        default=Path("rq-vae/outputs/amazon_beauty_embeddings.npy"),
        help="content embeddings (N, D) produced by the rq-vae dataloader",
    )
    ap.add_argument(
        "--sequences-dir",
        type=Path,
        default=Path("data/processed"),
        help=(
            "directory containing train/val/test.jsonl. SIDs are computed "
            "only for items appearing in these splits (pass /dev/null or an "
            "empty path to use all items)"
        ),
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="where to write item_to_sid.json and sid_to_item.json",
    )
    args = ap.parse_args()

    seq_dir: Path | None = args.sequences_dir
    if seq_dir is not None and (not seq_dir.exists() or str(seq_dir) in ("/dev/null", "")):
        seq_dir = None

    build_tables(
        checkpoint=args.checkpoint,
        items_csv=args.items_csv,
        embeddings=args.embeddings,
        sequences_dir=seq_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
