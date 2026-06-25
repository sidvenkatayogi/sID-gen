# TIGER — Recommender Systems with Generative Retrieval

An external PyTorch reproduction of **TIGER**, from the NeurIPS '23 paper
*"Recommender Systems with Generative Retrieval"* (Rajput et al.), trained and
evaluated on Amazon Beauty.

**Paper:** http://arxiv.org/abs/2305.05065

The pipeline has two stages, both living under the `tiger/` package: an RQ-VAE
that turns item content embeddings into Semantic IDs, and a T5 encoder-decoder
that decodes the next item's Semantic ID from a user's history.

## Project layout

```
tiger/
├── tiger/                       # the importable package (run via `python -m tiger.*`)
│   ├── rqvae/                   # stage 1: residual-quantized VAE -> Semantic IDs
│   │   ├── config.py            #   YAML loading + output-path resolution
│   │   ├── encoder.py           #   MLP encoder/decoder
│   │   ├── quantizer.py         #   residual vector quantizer (codebooks)
│   │   ├── model.py             #   RQVAE module
│   │   ├── metrics.py           #   codebook perplexity/utilization, SID uniqueness
│   │   ├── train.py             #   training loop
│   │   ├── generate_sids.py     #   encode items -> SID csv
│   │   ├── dataloader_beauty.py #   Amazon Beauty content embeddings
│   │   └── dataloader_ml1m.py   #   MovieLens-1M content embeddings
│   ├── retrieval/               # stage 2: T5 encoder-decoder over Semantic IDs
│   │   ├── vocab.py             #   SID <-> token vocab, special ids
│   │   ├── dataset.py           #   user-history -> (input, target) sequences
│   │   ├── model.py             #   TigerTransformer (HF T5 wrapper)
│   │   ├── decode.py            #   constrained beam search
│   │   ├── eval.py              #   Recall@k / NDCG@k, paper comparison
│   │   ├── train.py             #   training loop
│   │   └── evaluate.py          #   evaluate a checkpoint on a split
│   └── scripts/                 # pipeline glue (run via `python -m tiger.scripts.*`)
│       ├── preprocess_beauty.py #   reviews -> per-user leave-one-out sequences
│       └── build_sid_tables.py  #   build item<->SID lookup tables (+ c3 suffix)
├── configs/
│   ├── rqvae/                   # beauty.yaml, ml1m.yaml
│   └── retrieval/               # beauty.yaml
├── notebooks/                   # exploratory notebooks (gitignored)
├── data/                        # datasets + processed sequences (gitignored)
├── outputs/                     # checkpoints, metrics, SID artifacts (gitignored)
├── requirements.txt
└── README.md
```

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

All commands run from the project root (the cwd every path is resolved against).

```bash
# 1. Train the RQ-VAE and emit item content embeddings + Semantic IDs.
python -m tiger.rqvae.dataloader_beauty --config configs/rqvae/beauty.yaml --download
python -m tiger.rqvae.train             --config configs/rqvae/beauty.yaml
python -m tiger.rqvae.generate_sids     --config configs/rqvae/beauty.yaml \
    --checkpoint outputs/amazon_beauty_checkpoints/best.pt

# 2. Preprocess Beauty reviews into per-user sequences (leave-one-out split).
python -m tiger.scripts.preprocess_beauty --download

# 3. Build the SID lookup tables (the collision-breaking c3 is computed here).
python -m tiger.scripts.build_sid_tables \
    --checkpoint outputs/amazon_beauty_checkpoints/best.pt \
    --items-csv  outputs/amazon_beauty_items.csv \
    --embeddings outputs/amazon_beauty_embeddings.npy \
    --sequences-dir data/processed \
    --output-dir   data

# 4. Train the retrieval transformer.
python -m tiger.retrieval.train --config configs/retrieval/beauty.yaml

# 5. Evaluate the best checkpoint on the test split.
python -m tiger.retrieval.evaluate --checkpoint outputs/tiger_beauty/checkpoints/best.pt
```

Retrieval outputs land in `outputs/tiger_beauty/` (`checkpoints/best.pt`,
`metrics.json`, `val_curve.png`).

## Results

We compare on Beauty against the numbers reported in the original paper. There is no official released implementation. Metrics are Recall
and NDCG at @5 and @10 on the test split, from the best checkpoint by validation NDCG@10.

| Metric | This repo | Paper |
|---|---|---|
| Recall@5 | 0.0312 | 0.0454 |
| NDCG@5 | 0.0210 | 0.0321 |
| Recall@10 | 0.0486 | 0.0648 |
| NDCG@10 | 0.0265 | 0.0384 |

Invalid-ID rate @10 ≈ 0.0006. Best checkpoint at step 20K (val NDCG@10 = 0.0377).

## Implementation differences

- **Parameter count**: Using the paper's configuration gives us a model with ~4.85M params vs. the paper's stated ~13M (T5 encoder-decoder,
  4 + 4 layers, `d_model=128`). It remains unsure which components are under parameterized.
- **Early Stopping**: Training stops if the validation metric (NDCG@10) does not improve for 25000 steps

## Future Work
- Improve hyperparameters or RQVAE training behavior to match or improve on original paper's reported metrics. Current test metrics ended up at ~70% of paper reported.
- Add implementations for [PLUM](https://arxiv.org/abs/2510.07784) and [STATIC](https://arxiv.org/abs/2602.22647)

## References

- [Recommender Systems with Generative Retrieval](https://arxiv.org/abs/2305.05065) — Rajput et al., NeurIPS 2023.
- [Autoregressive Image Generation using Residual Quantization](https://arxiv.org/abs/2203.01941) — Lee et al. (RQ-VAE).
