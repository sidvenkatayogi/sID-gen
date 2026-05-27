# RQ-VAE

Residual-quantized VAE that turns per-item content embeddings into Semantic IDs
(SIDs): tuples of integer codes, one per quantizer level.

The pipeline has three stages:

1. **Dataloader** — dataset-specific. Reads your raw data, builds text per
   item, embeds it, and writes three artifacts to disk.
2. **Train** (`src.train`) — dataset-agnostic. Reads the embeddings and trains
   the RQ-VAE.
3. **Generate SIDs** (`src.generate_sids`) — dataset-agnostic. Loads a
   checkpoint and emits a CSV mapping items to SIDs.

Every stage is driven by a single YAML config. The config's top-level `name`
field is the only switch you flip per dataset — all artifact paths are derived
from it under `outputs/`, so two configs with different names produce disjoint
outputs and can run side-by-side.

## Adding a new dataset

You only need to write a dataloader. Train and inference are reused as-is.

### 1. Write a dataloader

Create `src/<name>_dataloader.py`. The ml1m loader (`src/ml1m_dataloader.py`)
is the reference. Your loader must:

- Read its own config block (whatever keys you need — `data.*`).
- Produce a `(N, input_dim)` `float32` numpy array of content embeddings.
- Write three files to the paths the resolved config specifies (these are
  injected automatically by `src.config.load_config` based on `name`):
  - `data.embeddings_path` (`outputs/{name}_embeddings.npy`) —
    `np.save` the `(N, input_dim)` array.
  - `data.items_csv` (`outputs/{name}_items.csv`) — one row per item, in
    the same order as the embeddings. Schema is up to you; whatever columns
    you write here will be carried through to the SID output CSV.
  - `data.stats_path` (`outputs/{name}_stats.npz`) —
    `np.savez(..., mean=..., std=...)` with the per-dimension stats used to
    standardize the embeddings (or zeros/ones if you skip standardization).

Load configs via `from src.config import load_config` — never `yaml.safe_load`
directly. The helper validates `name` and writes the resolved paths back into
the dict.

The standardization, embedding model, etc. are conveniences — the contract
with the rest of the pipeline is just those three output files.

### 2. Write a config

Copy `ml1m_config.yaml` and adjust. The required keys are:

```yaml
name: mydataset                           # drives every output path

data:
  # Anything your dataloader needs (raw paths, encoding, embedding model, ...).
  input_dim: 384                          # must match the embedding dim you produce
  standardize: true

model:
  encoder_hidden: [256, 128]
  latent_dim: 32
  num_levels: 3                           # L: depth of the residual quantizer
  codebook_size: 64                       # K: codes per level
  commitment_beta: 0.25
  codebook_update: ema                    # "ema" or "loss"
  ema_decay: 0.99
  ema_eps: 1.0e-5

train:
  optimizer: adam
  lr: 1.0e-3
  batch_size: 256
  epochs: 300
  kmeans_init: true
  reinit_enabled: true
  reinit_every: 250
  reinit_usage_threshold: 0.5
  seed: 0
```

There is no `output:` block and no explicit path keys — everything under
`outputs/` is named from `name`. Pick a unique `name` per config and the
artifacts won't collide.

## Running the pipeline

All three stages take `--config <path>` (required).

### Prepare data

```bash
python -m src.<name>_dataloader --config <name>_config.yaml [--download]
```

Writes `embeddings_path`, `items_csv`, `stats_path`.

### Train

```bash
python -m src.train --config <name>_config.yaml [--override key.path=value ...]
```

`--override` patches dotted keys in the loaded config, e.g.
`--override model.num_levels=4 --override train.epochs=100`. Useful for sweeps
without copying configs.

Outputs (with `name: foo`):
- `outputs/foo_checkpoints/best.pt` — lowest epoch recon loss
- `outputs/foo_checkpoints/final.pt` — last epoch
- `outputs/foo_metrics.json` — final-epoch metrics (recon, utilization,
  perplexity, unique-SID fraction)
- `outputs/foo_history.json` — per-epoch metrics

Checkpoints embed the full config they were trained with, so inference can
rebuild the model without re-reading the yaml.

### Generate SIDs

```bash
python -m src.generate_sids --config <name>_config.yaml [--checkpoint path/to/best.pt]
```

Default checkpoint is `outputs/checkpoints/best.pt`; pass `--checkpoint` to
point at the config-specific one (`outputs/{name}_checkpoints/best.pt`).

Outputs (with `name: foo`):
- `outputs/foo_sids.csv` — every column from `outputs/foo_items.csv` plus a
  `sid` column formatted as `c0-c1-...-c{L-1}`.
- `outputs/foo_metrics.json` — updated in place with final-inference
  utilization, perplexity, SID uniqueness, and collision count.
