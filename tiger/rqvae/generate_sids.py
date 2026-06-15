"""Load a trained RQ-VAE and emit per-item Semantic IDs.

Deterministic: encoder -> residual quantizer -> integer tuple. No decoder, no
sampling — the same checkpoint + embeddings always produce the same SIDs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import load_config
from .metrics import codebook_perplexity, codebook_utilization, sid_uniqueness
from .model import RQVAE


def build_model_from_checkpoint(ckpt_path: Path, device: torch.device) -> tuple[RQVAE, dict]:
    """Rebuild the RQ-VAE from a checkpoint (which stores its own config)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
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
        normalize_latent=m.get("normalize_latent", False),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


@torch.no_grad()
def encode_all(model: RQVAE, x: torch.Tensor, batch_size: int = 512) -> torch.Tensor:
    """Run the encoder + quantizer on every embedding. Returns indices (N, L)."""
    chunks: list[torch.Tensor] = []
    for i in range(0, x.shape[0], batch_size):
        xb = x[i : i + batch_size]
        out = model(xb)
        chunks.append(out.indices.cpu())
    return torch.cat(chunks, dim=0)


def format_sid(row: list[int]) -> str:
    return "-".join(str(c) for c in row)


def generate(cfg: dict, checkpoint: str | None = None) -> None:
    if checkpoint is None:
        checkpoint = str(Path(cfg["output"]["checkpoints_dir"]) / "best.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[gen] device={device}")

    emb_path = Path(cfg["data"]["embeddings_path"])
    items_csv = Path(cfg["data"]["items_csv"])
    assert emb_path.exists() and items_csv.exists(), "run the dataloader first"

    x = torch.from_numpy(np.load(emb_path)).to(device)         # (N, D)
    df = pd.read_csv(items_csv)
    assert len(df) == x.shape[0], "items/embeddings length mismatch"

    model, _ = build_model_from_checkpoint(Path(checkpoint), device)
    print(f"[gen] loaded {checkpoint}  L={model.quantizer.L} K={model.quantizer.K}")

    indices = encode_all(model, x)                              # (N, L) on cpu
    K = model.quantizer.K

    util = codebook_utilization(indices, K)
    ppl = codebook_perplexity(indices, K)
    uniq_frac, dupes = sid_uniqueness(indices)
    print(f"[gen] utilization per level: {[f'{u:.3f}' for u in util]}")
    print(f"[gen] perplexity per level:  {[f'{p:.2f}' for p in ppl]}")
    print(f"[gen] sid unique fraction:   {uniq_frac:.4f}  collisions={dupes}")

    sid_strings = [format_sid(row) for row in indices.tolist()]
    out_df = df.copy()
    out_df["sid"] = sid_strings
    out_path = Path(cfg["output"]["sids_csv"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"[gen] wrote {out_path}  ({len(out_df)} rows)")

    print("\n[sample] 10 items and their SIDs:")
    for i in range(min(10, len(out_df))):
        row = out_df.iloc[i]
        print(f"  {row['sid']:>12}  {row.to_dict()}")

    metrics_path = Path(cfg["output"]["metrics_json"])
    metrics: dict = {}
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
    metrics.update(
        {
            "final_utilization": util,
            "final_perplexity": ppl,
            "final_sid_unique_fraction": uniq_frac,
            "final_sid_collisions": dupes,
        }
    )
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[gen] updated {metrics_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default=None, help="defaults to {name}_checkpoints/best.pt")
    args = ap.parse_args()
    generate(load_config(args.config), args.checkpoint)


if __name__ == "__main__":
    main()
