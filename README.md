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

| Metric | This repo | Paper | Relative |
|---|---|---|---|
| Recall@5 | 0.0258 | 0.0454 | −43% |
| NDCG@5 | 0.0176 | 0.0321 | −45% |
| Recall@10 | 0.0390 | 0.0648 | −40% |
| NDCG@10 | 0.0218 | 0.0384 | −43% |

Invalid-ID rate @10 ≈ 0.0002.
Changes in model size, hyperparameters, or optimizer didn't seem to help :/

## Implementation differences

- **Parameter count**: ~4.85M vs. the paper's stated ~13M (T5 encoder-decoder,
  4 + 4 layers, `d_model=128`).
- **Backbone**: T5 via HuggingFace `transformers`, with a from-scratch beam
  search (width 50, per-position vocabulary masking; structurally-invalid IDs
  are filtered out).
- **Optimizer**: Adafactor (T5X-style), peak LR 0.01, linear warmup then
  inverse-sqrt decay, with early stopping on validation NDCG@10.
- **Content embeddings**: `sentence-t5-base` rather than the paper's larger encoder.
- **Vocabulary**: 3027 tokens (256×4 codewords + 2000 hashed user-ID buckets +
  PAD/BOS/EOS); user IDs hashed with MD5.

## References

- [Recommender Systems with Generative Retrieval](https://arxiv.org/abs/2305.05065) — Rajput et al., NeurIPS 2023.
- [Autoregressive Image Generation using Residual Quantization](https://arxiv.org/abs/2203.01941) — Lee et al. (RQ-VAE).
