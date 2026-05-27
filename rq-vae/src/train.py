"""Train the RQ-VAE on cached item content embeddings."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .config import load_config
from .metrics import codebook_perplexity, codebook_utilization, sid_uniqueness
from .model import RQVAE


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply dotted overrides like 'model.num_levels=1'."""
    for o in overrides:
        key, value = o.split("=", 1)
        # Cheap value coercion: try int -> float -> bool -> str.
        try:
            value_cast: object = int(value)
        except ValueError:
            try:
                value_cast = float(value)
            except ValueError:
                if value.lower() in ("true", "false"):
                    value_cast = value.lower() == "true"
                else:
                    value_cast = value
        cur = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            cur = cur[p]
        cur[parts[-1]] = value_cast
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def compute_epoch_metrics(model: RQVAE, loader: DataLoader, device: torch.device) -> dict:
    """Run the dataset through the model in eval mode and report metrics."""
    model.eval()
    all_idx: list[torch.Tensor] = []
    recon_losses: list[float] = []
    for (x,) in loader:
        x = x.to(device)
        out = model(x)
        all_idx.append(out.indices.cpu())
        recon_losses.append(float(out.recon_loss.item()))
    indices = torch.cat(all_idx, dim=0)
    util = codebook_utilization(indices, model.quantizer.K)
    ppl = codebook_perplexity(indices, model.quantizer.K)
    uniq_frac, dupes = sid_uniqueness(indices)
    model.train()
    return {
        "recon_loss": float(np.mean(recon_losses)),
        "utilization": util,
        "perplexity": ppl,
        "sid_unique_fraction": uniq_frac,
        "sid_collisions": dupes,
    }


def train(cfg: dict) -> list[dict]:
    set_seed(cfg["train"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}")

    # ---- Data --------------------------------------------------------------
    emb_path = Path(cfg["data"]["embeddings_path"])
    assert emb_path.exists(), f"missing {emb_path} — run the dataloader first"
    x = torch.from_numpy(np.load(emb_path))                          # (N, input_dim)
    print(f"[train] loaded embeddings {tuple(x.shape)}")

    dataset = TensorDataset(x)
    loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        drop_last=False,
    )

    # ---- Model -------------------------------------------------------------
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
    print(f"[train] model L={m['num_levels']}  K={m['codebook_size']}  D={m['latent_dim']}")

    # In EMA mode the codebook isn't optimizer-tracked (requires_grad=False),
    # so filtering by requires_grad keeps the optimizer pointed at the right
    # tensors automatically.
    opt_name = cfg["train"].get("optimizer", "adam").lower()
    opt_cls = {"adam": torch.optim.Adam, "adagrad": torch.optim.Adagrad}[opt_name]
    optim = opt_cls(
        (p for p in model.parameters() if p.requires_grad),
        lr=cfg["train"]["lr"],
    )

    # ---- k-means codebook init --------------------------------------------
    # Run the entire corpus through the (random-init) encoder and use k-means
    # centroids of the resulting latents to seed each codebook. Using all the
    # data (not one batch) matters: if N <= K at any level, k-means assigns
    # one point per cluster, residuals at the next level collapse to ~0, and
    # k-means there finds 1 distinct cluster.
    if cfg["train"].get("kmeans_init", False):
        with torch.no_grad():
            z0 = model.encoder(x.to(device))
        model.quantizer.kmeans_init(z0, random_state=cfg["train"]["seed"])
        print(f"[train] k-means init done on full corpus ({z0.shape[0]} latents)")

    # ---- Checkpoint setup --------------------------------------------------
    ckpt_dir = Path(cfg["output"]["checkpoints_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_recon = float("inf")

    # ---- Training loop -----------------------------------------------------
    step = 0
    reinit_enabled = cfg["train"].get("reinit_enabled", True)
    reinit_every = cfg["train"]["reinit_every"]
    reinit_thr = cfg["train"]["reinit_usage_threshold"]
    epochs = cfg["train"]["epochs"]
    history: list[dict] = []

    for epoch in range(epochs):
        model.train()
        running_total = 0.0
        running_recon = 0.0
        running_rq = 0.0
        n_seen = 0

        for (xb,) in loader:
            xb = xb.to(device)
            out = model(xb)

            optim.zero_grad()
            out.loss.backward()
            optim.step()

            B = xb.shape[0]
            running_total += float(out.loss.item()) * B
            running_recon += float(out.recon_loss.item()) * B
            running_rq    += float(out.rq_loss.item()) * B
            n_seen += B

            step += 1
            if reinit_enabled and step % reinit_every == 0 and model.quantizer.use_ema:
                n_reinit = model.quantizer.reinit_dead_codes(
                    out.residuals, usage_threshold=reinit_thr
                )
                if any(n > 0 for n in n_reinit):
                    print(f"[train] step {step}: reinit {n_reinit} dead codes per level")

        # Epoch-end logging + eval-style metrics.
        metrics = compute_epoch_metrics(model, loader, device)
        metrics["epoch"] = epoch + 1
        metrics["train_loss"] = running_total / n_seen
        metrics["train_recon"] = running_recon / n_seen
        metrics["train_rq"] = running_rq / n_seen
        history.append(metrics)
        print(
            f"[train] epoch {epoch+1:3d}/{epochs}  "
            f"loss={running_total/n_seen:.4f}  "
            f"recon={running_recon/n_seen:.4f}  "
            f"rq={running_rq/n_seen:.4f}  "
            f"util={[f'{u:.2f}' for u in metrics['utilization']]}  "
            f"ppl={[f'{p:.1f}' for p in metrics['perplexity']]}  "
            f"uniq={metrics['sid_unique_fraction']:.3f}"
        )

        if metrics["recon_loss"] < best_recon:
            best_recon = metrics["recon_loss"]
            torch.save({"model": model.state_dict(), "config": cfg}, ckpt_dir / "best.pt")

    # Final checkpoint + final metrics dump.
    torch.save({"model": model.state_dict(), "config": cfg}, ckpt_dir / "final.pt")
    final_metrics = compute_epoch_metrics(model, loader, device)
    metrics_path = Path(cfg["output"]["metrics_json"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=2)
    history_path = Path(cfg["output"]["history_json"])
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[train] wrote {metrics_path}")
    print(f"[train] wrote {history_path}")
    print(f"[train] final: {final_metrics}")
    return history


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--override",
        action="append",
        default=[],
        help="dotted key=value override, e.g. --override model.num_levels=1",
    )
    args = ap.parse_args()
    cfg = apply_overrides(load_config(args.config), args.override)
    train(cfg)


if __name__ == "__main__":
    main()
