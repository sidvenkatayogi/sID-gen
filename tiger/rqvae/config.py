"""Config loading + path resolution.

The top-level `name` field drives every output path under `outputs/`, so configs
with different names produce disjoint artifacts. The output root defaults to
`outputs/` but can be overridden per-config with a top-level `output_dir` field
(e.g. `output_dir: outputs_base` to keep a separate run's artifacts apart).
`load_config` injects the resolved paths:

  data.embeddings_path   = outputs/{name}_embeddings.npy
  data.items_csv         = outputs/{name}_items.csv
  data.stats_path        = outputs/{name}_stats.npz
  output.sids_csv        = outputs/{name}_sids.csv
  output.checkpoints_dir = outputs/{name}_checkpoints
  output.metrics_json    = outputs/{name}_metrics.json
  output.history_json    = outputs/{name}_history.json
"""

from __future__ import annotations

from pathlib import Path

import yaml

OUTPUTS_DIR = Path("outputs")


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    _resolve_paths(cfg)
    return cfg


def _resolve_paths(cfg: dict) -> None:
    name = cfg.get("name")
    assert name, "config must define a top-level `name`"

    out_dir = Path(cfg.get("output_dir", OUTPUTS_DIR))

    cfg.setdefault("data", {})
    cfg.setdefault("output", {})
    cfg["data"]["embeddings_path"] = str(out_dir / f"{name}_embeddings.npy")
    cfg["data"]["items_csv"]       = str(out_dir / f"{name}_items.csv")
    cfg["data"]["stats_path"]      = str(out_dir / f"{name}_stats.npz")
    cfg["output"]["sids_csv"]        = str(out_dir / f"{name}_sids.csv")
    cfg["output"]["checkpoints_dir"] = str(out_dir / f"{name}_checkpoints")
    cfg["output"]["metrics_json"]    = str(out_dir / f"{name}_metrics.json")
    cfg["output"]["history_json"]    = str(out_dir / f"{name}_history.json")
