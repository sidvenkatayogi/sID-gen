## experiments and implementations for TIGER (Transformer Index for GEnerative Recommenders) in pytorch

This repo has two sibling packages that compose into the full TIGER pipeline:

```
tiger/                      # project root
├── rq-vae/                 # upstream: content embedding -> Semantic IDs
└── tiger/                  # this package: TIGER encoder-decoder transformer
```

The RQ-VAE produces 3-code Semantic IDs from item content embeddings. The
TIGER transformer treats those IDs (plus a collision-breaking 4th code and
a user-ID hash) as a flat token vocabulary, and learns to autoregressively
decode the SID of the next item in a user's history.

## Components

| Path | Purpose |
|---|---|
| `tiger/vocab.py` | Token layout, SID round-trip, MD5 user-ID hashing |
| `tiger/dataset.py` | Torch Dataset producing encoder/decoder tensors |
| `tiger/model.py` | T5-style encoder-decoder, from-scratch PyTorch |
| `tiger/decode.py` | Beam search with per-position vocabulary masking |
| `tiger/eval.py` | Recall@K and NDCG@K metrics |
| `tiger/train.py` | Training loop (AdamW + linear-warmup-then-inv-sqrt) |
| `retrieval/scripts/preprocess_beauty.py` | 5-core reviews → train/val/test.jsonl (leave-one-out) |
| `retrieval/scripts/build_sid_tables.py` | RQ-VAE checkpoint → `item_to_sid.json` + `sid_to_item.json` |
| `retrieval/configs/beauty.yaml` | All hyperparameters (model + train) |
| `tests/` | Vocab round-trip, dataset shapes, beam decode validity |

## End-to-end run (Amazon Beauty)

Assumes `rq-vae` has already been trained end-to-end (its
`outputs/amazon_beauty_checkpoints/best.pt` exists, plus `*_items.csv` and
`*_embeddings.npy`).

```bash
# 1. Preprocess Beauty reviews into per-user sequences with leave-one-out split.
python retrieval/scripts/preprocess_beauty.py --download

# 2. Run the trained RQ-VAE over the items in those sequences and emit the
#    SID lookup tables. The collision-breaking c3 suffix is computed only
#    over items that show up in the sequences.
python retrieval/scripts/build_sid_tables.py \
    --checkpoint rq-vae/outputs/amazon_beauty_checkpoints/best.pt \
    --items-csv  rq-vae/outputs/amazon_beauty_items.csv \
    --embeddings rq-vae/outputs/amazon_beauty_embeddings.npy \
    --sequences-dir data/processed \
    --output-dir   data

# 3. Train the TIGER transformer.
python -m retrieval.train --config retrieval/configs/beauty.yaml

# 4. (Optional) Run the unit tests.
python -m pytest tests/
```

Outputs land in `outputs/tiger_beauty/`:
- `checkpoints/best.pt` — best by val NDCG@10
- `checkpoints/last.pt` — most recent
- `metrics.json` — full eval history + best score
- `val_curve.png` — NDCG@10 / Recall@10 vs step

## Target metrics (SPEC §9, paper, Beauty test split)

| Metric | Paper | Hit if within ±5% |
|---|---|---|
| Recall@5 | 0.0454 | 0.0431 – 0.0477 |
| NDCG@5 | 0.0321 | 0.0305 – 0.0337 |
| Recall@10 | 0.0648 | 0.0616 – 0.0680 |
| NDCG@10 | 0.0384 | 0.0365 – 0.0403 |

## Implementation notes

- **From-scratch T5**: encoder-decoder with RMSNorm, T5-style relative
  position bias (bucketed, log-spaced beyond a threshold). See `tiger/model.py`.
- **Param count**: ~4.85M with the spec config — the spec's "~13M" claim is
  optimistic for these particular hyperparameters (`d_model=128`, `d_ff=1024`,
  4+4 layers); the architecture itself matches.
- **Vocab size 3027**: 256×4 codeword tokens + 2000 user-ID buckets + 3
  specials (PAD/BOS/EOS).
- **User-ID hashing**: MD5 → mod 2000. NOT Python's `hash()` (salted per
  process — would break checkpoint reuse).
- **Beam search**: width 50, per-position vocab mask guarantees structural
  validity of each decoded token. Invalid (c0,c1,c2,c3) tuples that don't
  map to a real item are discarded; we return top-K of what's left.

## References
* [Recommender Systems with Generative Retrieval](https://arxiv.org/pdf/2305.05065) by Shashank Rajput, Nikhil Mehta, Anima Singh, Raghunandan H. Keshavan, Trung Vu, Lukasz Heldt, Lichan Hong, Yi Tay, Vinh Q. Tran, Jonah Samost, Maciej Kula, Ed H. Chi, Maheswaran Sathiamoorthy
* [Autoregressive Image Generation using Residual Quantization](https://arxiv.org/abs/2203.01941) by Doyup Lee, Chiheon Kim, Saehoon Kim, Minsu Cho, Wook-Shin Han
