"""Build `item_to_sid.json` + `sid_to_item.json` from the trained RQ-VAE.

Encodes the items appearing in the 5-core sequences through the frozen RQ-VAE
to get `(c0, c1, c2)`, assigns a collision-breaking `c3` (items sharing a
prefix get c3 = 0, 1, 2, ... in sorted item_id order; isolated items get 0),
and writes the two lookup JSONs. The c3 suffix is computed over the filtered
subset only — items outside the sequences are never seen by the TIGER model.

Usage (run from the project root):
    python -m tiger.scripts.build_sid_tables \\
        --checkpoint outputs/amazon_beauty_checkpoints/best.pt \\
        --items-csv  outputs/amazon_beauty_items.csv \\
        --embeddings outputs/amazon_beauty_embeddings.npy \\
        --sequences-dir data/processed/ \\
        --output-dir   data/
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tiger.retrieval.vocab import CODEBOOK_SIZE
from tiger.rqvae.model import RQVAE


def load_rqvae(checkpoint: Path, device: torch.device):
    """Rebuild the RQ-VAE from a checkpoint."""
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


def assign_collision_suffix(
    item_ids: list[str],
    codes: np.ndarray,                  # (N, 3) of c0,c1,c2
) -> np.ndarray:                        # (N,) of c3
    """For each (c0,c1,c2) bucket, assign c3 = 0,1,2,... in sorted item_id order."""
    assert len(item_ids) == codes.shape[0]
    assert codes.shape[1] == 3

    buckets: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for row, (c0, c1, c2) in enumerate(codes.tolist()):
        buckets[(c0, c1, c2)].append(row)

    c3 = np.zeros(len(item_ids), dtype=np.int64)
    max_bucket = 0
    for key, rows in buckets.items():
        rows_sorted = sorted(rows, key=lambda r: item_ids[r])
        for rank, r in enumerate(rows_sorted):
            c3[r] = rank
        max_bucket = max(max_bucket, len(rows_sorted))

    assert max_bucket <= CODEBOOK_SIZE, (
        f"collision bucket size {max_bucket} exceeds c3 capacity {CODEBOOK_SIZE}"
    )
    print(f"[sid] max collision bucket size = {max_bucket} (c3 capacity = {CODEBOOK_SIZE})")
    return c3


def collect_items_from_sequences(sequences_dir: Path) -> set[str]:
    """Union of item IDs appearing anywhere in train/val/test.jsonl."""
    items: set[str] = set()
    for split in ("train.jsonl", "val.jsonl", "test.jsonl"):
        path = sequences_dir / split
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing — run `python -m tiger.scripts.preprocess_beauty` first"
            )
        with open(path, "r") as f:
            for line in f:
                rec = json.loads(line)
                items.update(rec["history"])
                items.add(rec["target"])
    print(f"[sid] {len(items)} unique items across train/val/test sequences")
    return items


def build_tables(
    checkpoint: Path,
    items_csv: Path,
    embeddings: Path,
    sequences_dir: Path | None,
    output_dir: Path,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[sid] device={device}")

    df = pd.read_csv(items_csv)
    assert "item_id" in df.columns, f"{items_csv} must have an item_id column"
    item_ids_all = df["item_id"].astype(str).tolist()

    emb = np.load(embeddings)
    assert emb.shape[0] == len(item_ids_all), (
        f"embeddings rows {emb.shape[0]} != items_csv rows {len(item_ids_all)}"
    )
    print(f"[sid] loaded {len(item_ids_all)} items, embeddings {emb.shape}")

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

    model, cfg = load_rqvae(checkpoint, device)
    L = cfg["model"]["num_levels"]
    K = cfg["model"]["codebook_size"]
    assert L == 3, f"this pipeline assumes 3 RQ-VAE levels, got {L}"
    assert K == CODEBOOK_SIZE, (
        f"RQ-VAE codebook size {K} != vocab CODEBOOK_SIZE {CODEBOOK_SIZE} — "
        f"adjust tiger/retrieval/vocab.py if you re-trained with a different K"
    )

    x = torch.from_numpy(emb).to(device)
    codes = encode_to_codes(model, x)                       # (N, 3)
    print(f"[sid] encoded to codes, shape={codes.shape}")

    c3 = assign_collision_suffix(item_ids, codes)           # (N,)
    full = np.concatenate([codes, c3[:, None]], axis=1)     # (N, 4)

    output_dir.mkdir(parents=True, exist_ok=True)
    item_to_sid: dict[str, list[int]] = {}
    sid_to_item: dict[str, str] = {}
    for iid, row in zip(item_ids, full.tolist()):
        item_to_sid[iid] = row
        key = "_".join(str(c) for c in row)
        if key in sid_to_item:
            raise RuntimeError(
                f"SID collision after c3 assignment: {key} maps to both "
                f"{sid_to_item[key]} and {iid}"
            )
        sid_to_item[key] = iid

    with open(output_dir / "item_to_sid.json", "w") as f:
        json.dump(item_to_sid, f)
    with open(output_dir / "sid_to_item.json", "w") as f:
        json.dump(sid_to_item, f)

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
        default=Path("outputs/amazon_beauty_checkpoints/best.pt"),
        help="trained RQ-VAE checkpoint (best.pt)",
    )
    ap.add_argument(
        "--items-csv",
        type=Path,
        default=Path("outputs/amazon_beauty_items.csv"),
        help="per-item metadata produced by the rqvae dataloader",
    )
    ap.add_argument(
        "--embeddings",
        type=Path,
        default=Path("outputs/amazon_beauty_embeddings.npy"),
        help="content embeddings (N, D) produced by the rqvae dataloader",
    )
    ap.add_argument(
        "--sequences-dir",
        type=Path,
        default=Path("data/processed"),
        help=(
            "directory with train/val/test.jsonl. SIDs are computed only for "
            "items in these splits (pass an empty/missing path to use all items)"
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
