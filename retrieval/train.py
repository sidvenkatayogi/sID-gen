"""Training loop for the TIGER transformer.

Hyperparameters (SPEC §7):
    200K steps, batch 256, Adafactor (T5X-style, wd=0), peak LR 0.01,
    10K linear warmup -> inverse-sqrt decay, grad clip 1.0, dropout 0.1,
    bf16 if available, seed 42.

    Label smoothing is 0.1 (not the spec's 0.0): the decoder otherwise grows
    overconfident on memorized codeword combinations, blowing up the invalid-SID
    rate and collapsing val metrics ~25K steps in. Smoothing keeps a little mass
    spread across the vocab and is the standard T5-style cure for this.

NOTE on the optimizer: the paper/SPEC's peak LR of 0.01 is an *Adafactor*
learning rate (TIGER is trained in T5X, whose default optimizer is Adafactor).
Adafactor clips its per-step update RMS so 0.01 is stable; AdamW at 0.01 is
~10-30x too hot (train loss floors high and val NDCG peaks during warmup then
degrades). We therefore use Adafactor here, not AdamW.

Validation cadence (SPEC §7.2):
    every 5K steps -- run beam decode on val set -> NDCG@10 / Recall@10.
    Best checkpoint by val NDCG@10 is saved as `best.pt`. Training early-stops
    after `early_stop_patience` evals with no val NDCG@10 improvement (config;
    null disables it).

Outputs:
    checkpoints/{best.pt, last.pt}
    metrics.json (final + per-eval-step)
    val_curve.png (NDCG@10 vs step)
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

from retrieval.dataset import TigerSequenceDataset, collate
from retrieval.decode import beam_decode
from retrieval.eval import compute_retrieval_metrics
from retrieval.model import TigerConfig, TigerTransformer
from retrieval.vocab import PAD_ID


# ----------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------
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

    The lambda multiplies the base LR set on the optimizer. We set base LR
    to `peak_lr` so warmup is just `step / warmup_steps`.
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        # After warmup: factor decays as sqrt(warmup / step), so at step
        # =warmup the factor is 1.0 and shrinks from there.
        return math.sqrt(warmup_steps / step)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def pick_amp_dtype() -> torch.dtype | None:
    """bf16 if the device supports it; otherwise None (fp32)."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    # MPS (Apple Silicon) supports bf16 in recent torch but autocast is touchy
    # on it. fp32 is fine for a model this small.
    return None


# ----------------------------------------------------------------------------
# Validation: beam-decode the val split and compute retrieval metrics.
# ----------------------------------------------------------------------------
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
    """Run beam decode on the val set, return Recall@k / NDCG@k / invalid@k.

    `max_batches=None` runs the whole set; pass an integer to subsample
    during long training runs.
    """
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

        # beam_decode -> for each row, top-`beam_width` SID tuples + scores.
        pred_sids = beam_decode(model, enc_ids, enc_mask, beam_width=beam_width)
        # pred_sids: list of length B, each a list of (sid_tuple, score) sorted desc

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

            # If we couldn't fill top_k from this many beams, pad with None
            # so per-rank metric calcs still work — they just won't hit.
            while len(top_items) < top_k:
                top_items.append(None)

            invalid_count += row_invalid
            all_pred_items.append(top_items)

            # Reverse-lookup gold item from its SID. We saved target_sid on
            # cpu so this is just a dict lookup per row.
            gold_key = "_".join(str(c) for c in target_sid[b].tolist())
            all_gold_items.append(sid_to_item[gold_key])
            seen += 1

    metrics = compute_retrieval_metrics(all_pred_items, all_gold_items, ks=(5, 10))
    # Invalid rate: invalid beams / total beams visited, averaged across rows.
    # (Each row visited beam_width beams; not all were tried since we early-exit
    # once top_k valid are found. Using `invalid_count / (seen * top_k)` is a
    # decent proxy that doesn't require tracking per-row beam visits.)
    metrics["invalid_rate"] = invalid_count / max(seen, 1) / top_k
    model.train()
    return metrics


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def train(
    cfg: dict,
    data_dir: Path,
    output_dir: Path,
) -> None:
    train_cfg = cfg["train"]
    set_seed(train_cfg["seed"])
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    amp_dtype = pick_amp_dtype() if device.type == "cuda" else None
    print(f"[train] device={device}  amp_dtype={amp_dtype}")

    # ---- Data -------------------------------------------------------------
    item_to_sid_path = data_dir / "item_to_sid.json"
    sid_to_item_path = data_dir / "sid_to_item.json"
    with open(sid_to_item_path) as f:
        sid_to_item: dict[str, str] = json.load(f)

    train_ds = TigerSequenceDataset(data_dir / "processed/train.jsonl", item_to_sid_path)
    val_ds = TigerSequenceDataset(data_dir / "processed/val.jsonl", item_to_sid_path)
    print(f"[train] train={len(train_ds)}  val={len(val_ds)}")

    # Workers help on Linux/CUDA; on macOS/MPS keep it single-process to
    # avoid the fork-vs-spawn footgun. PyTorch defaults are fine for now.
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

    # ---- Model ------------------------------------------------------------
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

    # ---- Optimizer + schedule --------------------------------------------
    # Adafactor (T5X default), driven by an external linear-warmup ->
    # inverse-sqrt LR schedule peaking at `peak_lr`. beta2_decay=-0.8 matches
    # T5X's decay_rate=0.8; d=1.0 (default) is the update-RMS clip threshold
    # that keeps a 0.01 LR stable. No first-moment/momentum term, matching the
    # T5X config. The AdamW betas in the YAML are intentionally unused here.
    optim = torch.optim.Adafactor(
        model.parameters(),
        lr=train_cfg["peak_lr"],
        beta2_decay=-0.8,
        weight_decay=train_cfg.get("weight_decay", 0.0),
    )
    scheduler = make_lr_schedule(
        optim, warmup_steps=train_cfg["warmup_steps"], peak_lr=train_cfg["peak_lr"]
    )

    # ---- Train state ------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    total_steps = train_cfg["total_steps"]
    eval_every = train_cfg["eval_every"]
    log_every = train_cfg.get("log_every", 50)
    grad_clip = train_cfg.get("grad_clip", 1.0)
    beam_width = train_cfg.get("eval_beam_width", 50)
    top_k = train_cfg.get("eval_top_k", 10)
    eval_max_batches = train_cfg.get("eval_max_batches", None)   # None = full val
    # Early stopping: halt after `patience` consecutive evals with no improvement
    # in val NDCG@10. None disables it (train the full `total_steps`). best.pt is
    # always the peak checkpoint regardless, so this only saves wasted compute on
    # the post-peak overfitting tail.
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

            # Early stop: peak is already preserved in best.pt, so bail out of the
            # overfitting tail once val NDCG@10 has stalled for `patience` evals.
            if patience is not None and evals_since_improve >= patience:
                print(
                    f"[train] early stop at step {step}: no val NDCG@10 improvement "
                    f"in {patience} evals (best={best_ndcg:.4f})"
                )
                break

    # ---- Persist eval history --------------------------------------------
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

    # Best-effort plot: skip silently if matplotlib isn't around.
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("retrieval/configs/beauty.yaml"))
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/tiger_beauty"))
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    train(cfg, args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
