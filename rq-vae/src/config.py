"""Config loading + path resolution.

The config has a top-level `name` (e.g. `ml1m`). All output paths in the
pipeline are derived from it under `outputs/`, so two configs with different
names produce disjoint artifacts and can run side-by-side.

Resolved paths injected into the config:
  data.embeddings_path  = outputs/{name}_embeddings.npy
  data.items_csv        = outputs/{name}_items.csv
  data.stats_path       = outputs/{name}_stats.npz
  output.sids_csv       = outputs/{name}_sids.csv
  output.checkpoints_dir= outputs/{name}_checkpoints
  output.metrics_json   = outputs/{name}_metrics.json
  output.history_json   = outputs/{name}_history.json
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

    cfg.setdefault("data", {})
    cfg.setdefault("output", {})
    cfg["data"]["embeddings_path"] = str(OUTPUTS_DIR / f"{name}_embeddings.npy")
    cfg["data"]["items_csv"]       = str(OUTPUTS_DIR / f"{name}_items.csv")
    cfg["data"]["stats_path"]      = str(OUTPUTS_DIR / f"{name}_stats.npz")
    cfg["output"]["sids_csv"]        = str(OUTPUTS_DIR / f"{name}_sids.csv")
    cfg["output"]["checkpoints_dir"] = str(OUTPUTS_DIR / f"{name}_checkpoints")
    cfg["output"]["metrics_json"]    = str(OUTPUTS_DIR / f"{name}_metrics.json")
    cfg["output"]["history_json"]    = str(OUTPUTS_DIR / f"{name}_history.json")
