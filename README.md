# TIGER — Recommender Systems with Generative Retrieval

An external PyTorch reproduction of **TIGER**, from the NeurIPS '23 paper
*"Recommender Systems with Generative Retrieval"* (Rajput et al.), trained and
evaluated on Amazon Beauty.

**Paper:** http://arxiv.org/abs/2305.05065

The pipeline has two stages, each its own package:

- **`rq-vae/`** — a residual-quantized VAE that turns item content embeddings
  into Semantic IDs.
- **`retrieval/`** — a T5 encoder-decoder that decodes the next item's Semantic
  ID from a user's history.

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

All commands run from the project root.

```bash
# 1. Train the RQ-VAE and emit item content embeddings + Semantic IDs.
cd rq-vae
python -m src.amazon_beauty_dataloader --config configs/amazon_beauty_config.yaml --download
python -m src.train         --config configs/amazon_beauty_config.yaml
python -m src.generate_sids --config configs/amazon_beauty_config.yaml \
    --checkpoint outputs/amazon_beauty_checkpoints/best.pt
cd ..

# 2. Preprocess Beauty reviews into per-user sequences (leave-one-out split).
python scripts/preprocess_beauty.py --download

# 3. Build the SID lookup tables (the collision-breaking c3 is computed here).
python scripts/build_sid_tables.py \
    --checkpoint rq-vae/outputs/amazon_beauty_checkpoints/best.pt \
    --items-csv  rq-vae/outputs/amazon_beauty_items.csv \
    --embeddings rq-vae/outputs/amazon_beauty_embeddings.npy \
    --sequences-dir data/processed \
    --output-dir   data

# 4. Train the retrieval transformer.
python -m retrieval.train --config retrieval/configs/beauty.yaml

# 5. Evaluate the best checkpoint on the test split.
python -m retrieval.evaluate --checkpoint outputs/tiger_beauty/checkpoints/best.pt
```

Retrieval outputs land in `outputs/tiger_beauty/` (`checkpoints/best.pt`,
`metrics.json`, `val_curve.png`).

## Results

Because there is no official TIGER reference implementation, we compare on
Beauty against the numbers reported in the original paper. Metrics are Recall
and NDCG at @5 and @10 on the test split, from the best checkpoint by
validation NDCG@10.

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
## References

- [Recommender Systems with Generative Retrieval](https://arxiv.org/abs/2305.05065) — Rajput et al., NeurIPS 2023.
- [Autoregressive Image Generation using Residual Quantization](https://arxiv.org/abs/2203.01941) — Lee et al. (RQ-VAE).
