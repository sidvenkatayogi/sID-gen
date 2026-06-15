"""Training loop for the TIGER transformer.

Defaults (see `configs/retrieval/beauty.yaml`): 200K steps, batch 256, Adafactor with a
linear-warmup -> inverse-sqrt LR schedule peaking at 0.01, grad clip 1.0.

The peak LR of 0.01 is an *Adafactor* learning rate (TIGER is trained in T5X,
whose default optimizer is Adafactor). Adafactor clips its per-step update RMS,
so 0.01 is stable; AdamW at 0.01 is far too hot.

Validation runs beam decode on the val split every `eval_every` steps; the best
checkpoint by val NDCG@10 is saved as `best.pt`. Outputs land in `output_dir`:
`checkpoints/{best,last}.pt`, `metrics.json`, `val_curve.png`.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from tiger.retrieval.dataset import TigerSequenceDataset, collate
from tiger.retrieval.decode import beam_decode
from tiger.retrieval.eval import compute_retrieval_metrics
from tiger.retrieval.model import TigerConfig, TigerTransformer
from tiger.retrieval.vocab import PAD_ID


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_lr_schedule(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    peak_lr: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup to peak_lr over `warmup_steps`, then inverse-sqrt decay.
    Base LR is set to `peak_lr`, so the warmup factor is just step/warmup."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return math.sqrt(warmup_steps / step)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def pick_amp_dtype() -> torch.dtype | None:
    """bf16 on CUDA when supported, else None (fp32)."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return None


@torch.no_grad()
def validate(
    model: TigerTransformer,
    val_loader: DataLoader,
    sid_to_item: dict[str, str],
    device: torch.device,
    beam_width: int,
    top_k: int,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Beam-decode the val set and return Recall@k / NDCG@k / invalid@k.
    `max_batches=None` runs the whole set; pass an int to subsample."""
    model.eval()
    all_pred_items: list[list[str | None]] = []
    all_gold_items: list[str] = []
    invalid_count = 0
    seen = 0

    for bi, batch in enumerate(val_loader):
        if max_batches is not None and bi >= max_batches:
            break
        enc_ids = batch["encoder_input_ids"].to(device)
        enc_mask = batch["encoder_attn_mask"].to(device)
        target_sid = batch["target_sid"]                       # (B, 4) on cpu

        pred_sids = beam_decode(model, enc_ids, enc_mask, beam_width=beam_width)

        for b in range(enc_ids.size(0)):
            beams = pred_sids[b]
            top_items: list[str | None] = []
            row_invalid = 0
            for sid_tuple, _score in beams:
                key = "_".join(str(c) for c in sid_tuple)
                item = sid_to_item.get(key)
                if item is None:
                    row_invalid += 1
                    continue
                top_items.append(item)
                if len(top_items) >= top_k:
                    break

            while len(top_items) < top_k:
                top_items.append(None)

            invalid_count += row_invalid
            all_pred_items.append(top_items)

            gold_key = "_".join(str(c) for c in target_sid[b].tolist())
            all_gold_items.append(sid_to_item[gold_key])
            seen += 1

    metrics = compute_retrieval_metrics(all_pred_items, all_gold_items, ks=(5, 10))
    metrics["invalid_rate"] = invalid_count / max(seen, 1) / top_k
    model.train()
    return metrics


def train(
    cfg: dict,
    data_dir: Path,
    output_dir: Path,
) -> None:
    train_cfg = cfg["train"]
    set_seed(train_cfg["seed"])
    device = pick_device()
    amp_dtype = pick_amp_dtype() if device.type == "cuda" else None
    print(f"[train] device={device}  amp_dtype={amp_dtype}")

    item_to_sid_path = data_dir / "item_to_sid.json"
    sid_to_item_path = data_dir / "sid_to_item.json"
    with open(sid_to_item_path) as f:
        sid_to_item: dict[str, str] = json.load(f)

    train_ds = TigerSequenceDataset(data_dir / "processed/train.jsonl", item_to_sid_path)
    val_ds = TigerSequenceDataset(data_dir / "processed/val.jsonl", item_to_sid_path)
    print(f"[train] train={len(train_ds)}  val={len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=train_cfg.get("num_workers", 0),
        collate_fn=collate,
        persistent_workers=train_cfg.get("num_workers", 0) > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.get("eval_batch_size", 64),
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )

    mcfg = cfg["model"]
    model_config = TigerConfig(
        vocab_size=mcfg["vocab_size"],
        d_model=mcfg["d_model"],
        d_ff=mcfg["d_ff"],
        num_heads=mcfg["num_heads"],
        head_dim=mcfg["head_dim"],
        num_encoder_layers=mcfg["num_encoder_layers"],
        num_decoder_layers=mcfg["num_decoder_layers"],
        dropout=mcfg["dropout"],
        max_enc_len=mcfg["max_enc_len"],
        max_dec_len=mcfg["max_dec_len"],
        pad_id=PAD_ID,
        rel_num_buckets=mcfg.get("rel_num_buckets", 32),
        rel_max_distance=mcfg.get("rel_max_distance", 128),
        tie_embeddings=mcfg.get("tie_embeddings", True),
        initializer_range=mcfg.get("initializer_range", 0.02),
    )
    model = TigerTransformer(model_config).to(device)
    print(f"[train] model params: {model.num_params():,}")

    # Adafactor (T5X default): no momentum, beta2_decay=-0.8 matches T5X's
    # decay_rate=0.8, and the update-RMS clip keeps a 0.01 LR stable. The AdamW
    # betas in the YAML are unused here.
    optim = torch.optim.Adafactor(
        model.parameters(),
        lr=train_cfg["peak_lr"],
        beta2_decay=-0.8,
        weight_decay=train_cfg.get("weight_decay", 0.0),
    )
    scheduler = make_lr_schedule(
        optim, warmup_steps=train_cfg["warmup_steps"], peak_lr=train_cfg["peak_lr"]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    total_steps = train_cfg["total_steps"]
    eval_every = train_cfg["eval_every"]
    log_every = train_cfg.get("log_every", 50)
    grad_clip = train_cfg.get("grad_clip", 1.0)
    beam_width = train_cfg.get("eval_beam_width", 50)
    top_k = train_cfg.get("eval_top_k", 10)
    eval_max_batches = train_cfg.get("eval_max_batches", None)
    # Early stop after `patience` consecutive evals without val NDCG@10
    # improvement (None disables). best.pt is always the peak regardless.
    patience = train_cfg.get("early_stop_patience", None)

    best_ndcg = -1.0
    evals_since_improve = 0
    eval_history: list[dict] = []
    step = 0
    start = time.time()
    train_iter = iter(train_loader)

    model.train()
    while step < total_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        enc_ids = batch["encoder_input_ids"].to(device, non_blocking=True)
        enc_mask = batch["encoder_attn_mask"].to(device, non_blocking=True)
        dec_in = batch["decoder_input_ids"].to(device, non_blocking=True)
        dec_tgt = batch["decoder_labels"].to(device, non_blocking=True)

        if amp_dtype is not None:
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                logits = model(enc_ids, enc_mask, dec_in)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    dec_tgt.reshape(-1),
                    ignore_index=PAD_ID,
                    label_smoothing=train_cfg.get("label_smoothing", 0.0),
                )
        else:
            logits = model(enc_ids, enc_mask, dec_in)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                dec_tgt.reshape(-1),
                ignore_index=PAD_ID,
                label_smoothing=train_cfg.get("label_smoothing", 0.0),
            )

        optim.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optim.step()
        scheduler.step()

        step += 1
        if step % log_every == 0:
            elapsed = time.time() - start
            cur_lr = scheduler.get_last_lr()[0]
            print(
                f"[train] step={step:>7d}/{total_steps}  "
                f"loss={loss.item():.4f}  lr={cur_lr:.2e}  "
                f"step_per_s={step/max(elapsed,1e-6):.1f}"
            )

        if step % eval_every == 0 or step == total_steps:
            metrics = validate(
                model,
                val_loader,
                sid_to_item,
                device=device,
                beam_width=beam_width,
                top_k=top_k,
                max_batches=eval_max_batches,
            )
            metrics["step"] = step
            eval_history.append(metrics)
            print(
                f"[eval] step={step}  "
                f"R@5={metrics['recall@5']:.4f}  N@5={metrics['ndcg@5']:.4f}  "
                f"R@10={metrics['recall@10']:.4f}  N@10={metrics['ndcg@10']:.4f}  "
                f"invalid@10={metrics['invalid_rate']:.4f}"
            )

            if metrics["ndcg@10"] > best_ndcg:
                best_ndcg = metrics["ndcg@10"]
                evals_since_improve = 0
                _save_checkpoint(
                    model, optim, scheduler, cfg, step, metrics,
                    ckpt_dir / "best.pt"
                )
                print(f"[eval] new best NDCG@10={best_ndcg:.4f} -> saved best.pt")
            else:
                evals_since_improve += 1
                print(
                    f"[eval] no improvement ({evals_since_improve}/{patience}) "
                    f"— best NDCG@10={best_ndcg:.4f}"
                    if patience is not None
                    else f"[eval] no improvement — best NDCG@10={best_ndcg:.4f}"
                )

            _save_checkpoint(model, optim, scheduler, cfg, step, metrics, ckpt_dir / "last.pt")

            if patience is not None and evals_since_improve >= patience:
                print(
                    f"[train] early stop at step {step}: no val NDCG@10 improvement "
                    f"in {patience} evals (best={best_ndcg:.4f})"
                )
                break

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(
            {
                "eval_history": eval_history,
                "best_ndcg_at_10": best_ndcg,
                "model_params": model.num_params(),
            },
            f,
            indent=2,
        )
    print(f"[train] wrote {output_dir/'metrics.json'}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = [h["step"] for h in eval_history]
        plt.figure(figsize=(7, 4))
        plt.plot(steps, [h["ndcg@10"] for h in eval_history], marker="o", label="NDCG@10")
        plt.plot(steps, [h["recall@10"] for h in eval_history], marker="x", label="Recall@10")
        plt.xlabel("step")
        plt.legend()
        plt.title("TIGER (Beauty) — validation curve")
        plt.tight_layout()
        plt.savefig(output_dir / "val_curve.png", dpi=120)
        print(f"[train] wrote {output_dir/'val_curve.png'}")
    except Exception as exc:
        print(f"[train] skipped val_curve.png: {exc}")


def _save_checkpoint(
    model: TigerTransformer,
    optim: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    cfg: dict,
    step: int,
    metrics: dict,
    path: Path,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": cfg,
            "step": step,
            "metrics": metrics,
        },
        path,
    )


def pick_device() -> torch.device:
    return torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )


def train_from_config(
    config_path: str | Path,
    data_dir: str | Path = "data",
    output_dir: str | Path = "outputs/tiger_beauty",
    quick: bool = False,
) -> dict:
    """Load a YAML config and run training. `quick=True` trims steps for a smoke
    test that will not match the paper. Returns the (possibly trimmed) config."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    if quick:
        cfg["train"].update(
            total_steps=4000, warmup_steps=500, eval_every=1000, eval_max_batches=20
        )
    train(cfg, Path(data_dir), Path(output_dir))
    return cfg


def build_model_from_checkpoint(
    ckpt_path: str | Path, device: torch.device
) -> tuple[TigerTransformer, dict]:
    """Rebuild a TigerTransformer from a training checkpoint (which stores its
    own config). Returns (model in eval mode, raw checkpoint dict)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    m = ckpt["config"]["model"]
    model = TigerTransformer(TigerConfig(
        vocab_size=m["vocab_size"], d_model=m["d_model"], d_ff=m["d_ff"],
        num_heads=m["num_heads"], head_dim=m["head_dim"],
        num_encoder_layers=m["num_encoder_layers"], num_decoder_layers=m["num_decoder_layers"],
        dropout=m["dropout"], max_enc_len=m["max_enc_len"], max_dec_len=m["max_dec_len"],
        pad_id=PAD_ID, rel_num_buckets=m.get("rel_num_buckets", 32),
        rel_max_distance=m.get("rel_max_distance", 128),
        tie_embeddings=m.get("tie_embeddings", True),
        initializer_range=m.get("initializer_range", 0.02),
    )).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def evaluate_checkpoint(
    ckpt_path: str | Path,
    data_dir: str | Path = "data",
    split: str = "test",
    beam_width: int = 50,
    top_k: int = 10,
    eval_batch_size: int = 64,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Beam-decode `{split}.jsonl` with a saved checkpoint and return metrics."""
    device = pick_device()
    data_dir = Path(data_dir)
    model, _ = build_model_from_checkpoint(ckpt_path, device)
    with open(data_dir / "sid_to_item.json") as f:
        sid_to_item = json.load(f)
    ds = TigerSequenceDataset(
        data_dir / "processed" / f"{split}.jsonl", data_dir / "item_to_sid.json"
    )
    loader = DataLoader(ds, batch_size=eval_batch_size, shuffle=False, collate_fn=collate)
    return validate(
        model, loader, sid_to_item, device=device,
        beam_width=beam_width, top_k=top_k, max_batches=max_batches,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/retrieval/beauty.yaml"))
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/tiger_beauty"))
    args = ap.parse_args()
    train_from_config(args.config, args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
