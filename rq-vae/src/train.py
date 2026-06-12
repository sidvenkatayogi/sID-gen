"""Train the RQ-VAE on cached item content embeddings."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from .config import load_config
from .metrics import codebook_perplexity, codebook_utilization, sid_uniqueness
from .model import RQVAE


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply dotted overrides like 'model.num_levels=1' (int -> float -> bool -> str)."""
    for o in overrides:
        key, value = o.split("=", 1)
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
def compute_epoch_metrics(model: RQVAE, x: torch.Tensor, batch_size: int) -> dict:
    """Full-corpus eval pass. `x` is expected to already live on the model's
    device. eval() gates off the EMA codebook update (quantizer.forward), so this
    pass does not mutate training state."""
    model.eval()
    all_idx: list[torch.Tensor] = []
    recon_weighted = 0.0
    n_seen = 0
    N = x.shape[0]
    for start in range(0, N, batch_size):
        xb = x[start : start + batch_size]
        out = model(xb)
        all_idx.append(out.indices)                       # keep on device; one .cpu() below
        B = xb.shape[0]
        recon_weighted += float(out.recon_loss.item()) * B
        n_seen += B
    indices = torch.cat(all_idx, dim=0).cpu()
    util = codebook_utilization(indices, model.quantizer.K)
    ppl = codebook_perplexity(indices, model.quantizer.K)
    uniq_frac, dupes = sid_uniqueness(indices)
    model.train()
    return {
        "recon_loss": recon_weighted / n_seen,
        "utilization": util,
        "perplexity": ppl,
        "sid_unique_fraction": uniq_frac,
        "sid_collisions": dupes,
    }


def train(cfg: dict) -> list[dict]:
    set_seed(cfg["train"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}")

    emb_path = Path(cfg["data"]["embeddings_path"])
    assert emb_path.exists(), f"missing {emb_path} — run the dataloader first"
    # The corpus is small (~tens of MB), so keep it resident on-device for the
    # whole run instead of re-copying each batch through a DataLoader. Batches
    # are formed by slicing a per-epoch random permutation (see the loop below).
    x = torch.from_numpy(np.load(emb_path)).to(device)               # (N, input_dim)
    N = x.shape[0]
    batch_size = cfg["train"]["batch_size"]
    print(f"[train] loaded embeddings {tuple(x.shape)}")

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
    print(f"[train] model L={m['num_levels']}  K={m['codebook_size']}  D={m['latent_dim']}")

    # In EMA mode the codebook has requires_grad=False, so filtering keeps the
    # optimizer pointed only at the encoder/decoder weights.
    opt_name = cfg["train"].get("optimizer", "adam").lower()
    opt_cls = {"adam": torch.optim.Adam, "adagrad": torch.optim.Adagrad}[opt_name]
    optim = opt_cls(
        (p for p in model.parameters() if p.requires_grad),
        lr=cfg["train"]["lr"],
    )

    # k-means init over the full corpus (not one batch): if N <= K at any level,
    # next-level residuals collapse to ~0 and k-means there finds one cluster.
    if cfg["train"].get("kmeans_init", False):
        with torch.no_grad():
            z0 = model.encode(x)
        model.quantizer.kmeans_init(z0, random_state=cfg["train"]["seed"])
        print(f"[train] k-means init done on full corpus ({z0.shape[0]} latents)")

    ckpt_dir = Path(cfg["output"]["checkpoints_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_recon = float("inf")

    step = 0
    reinit_enabled = cfg["train"].get("reinit_enabled", True)
    reinit_every = cfg["train"]["reinit_every"]
    reinit_thr = cfg["train"]["reinit_usage_threshold"]
    epochs = cfg["train"]["epochs"]
    # The full-corpus eval pass roughly doubles per-epoch cost, so run it only
    # every `eval_every` epochs (and always on the final epoch). best.pt is
    # chosen among evaluated epochs — coarser eval_every => coarser selection.
    eval_every = cfg["train"].get("eval_every", 50)
    history: list[dict] = []

    for epoch in range(epochs):
        model.train()
        # Loss accumulators stay on-device; we only sync (.item()) once per epoch
        # below, rather than per batch, to avoid stalling the GPU pipeline.
        running_total = torch.zeros((), device=device)
        running_recon = torch.zeros((), device=device)
        running_rq = torch.zeros((), device=device)
        n_seen = 0

        perm = torch.randperm(N, device=device)
        for start in range(0, N, batch_size):
            xb = x[perm[start : start + batch_size]]
            out = model(xb)

            optim.zero_grad()
            out.loss.backward()
            optim.step()

            B = xb.shape[0]
            running_total += out.loss.detach() * B
            running_recon += out.recon_loss.detach() * B
            running_rq    += out.rq_loss.detach() * B
            n_seen += B

            step += 1
            if reinit_enabled and step % reinit_every == 0 and model.quantizer.use_ema:
                n_reinit = model.quantizer.reinit_dead_codes(
                    out.residuals, usage_threshold=reinit_thr
                )
                if any(n > 0 for n in n_reinit):
                    print(f"[train] step {step}: reinit {n_reinit} dead codes per level")

        train_loss = float(running_total.item()) / n_seen
        train_recon = float(running_recon.item()) / n_seen
        train_rq = float(running_rq.item()) / n_seen

        if (epoch + 1) % eval_every == 0 or (epoch + 1) == epochs:
            metrics = compute_epoch_metrics(model, x, batch_size)
            metrics["epoch"] = epoch + 1
            metrics["train_loss"] = train_loss
            metrics["train_recon"] = train_recon
            metrics["train_rq"] = train_rq
            history.append(metrics)
            print(
                f"[train] epoch {epoch+1:3d}/{epochs}  "
                f"loss={train_loss:.4f}  "
                f"recon={train_recon:.4f}  "
                f"rq={train_rq:.4f}  "
                f"util={[f'{u:.2f}' for u in metrics['utilization']]}  "
                f"ppl={[f'{p:.1f}' for p in metrics['perplexity']]}  "
                f"uniq={metrics['sid_unique_fraction']:.3f}"
            )

            if metrics["recon_loss"] < best_recon:
                best_recon = metrics["recon_loss"]
                torch.save({"model": model.state_dict(), "config": cfg}, ckpt_dir / "best.pt")
        else:
            print(
                f"[train] epoch {epoch+1:3d}/{epochs}  "
                f"loss={train_loss:.4f}  recon={train_recon:.4f}  rq={train_rq:.4f}"
            )

    torch.save({"model": model.state_dict(), "config": cfg}, ckpt_dir / "final.pt")
    final_metrics = compute_epoch_metrics(model, x, batch_size)
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
