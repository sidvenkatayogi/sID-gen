"""Amazon Beauty loader + content embedding builder.

    meta_Beauty.json.gz -> per-item text -> sentence-t5 -> standardize ->
    cache (embeddings.npy + items.csv + stats.npz)

The metadata is the McAuley UCSD Amazon Reviews dump (one gzipped JSON object
per line) with fields like asin, title, brand, categories, price, description.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

from .config import load_config

META_URL = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Beauty.json.gz"


def download_meta(dest_path: Path) -> Path:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists():
        print(f"[data] already downloaded at {dest_path}")
        return dest_path
    print(f"[data] downloading {META_URL}")
    with urlopen(META_URL) as resp:
        buf = resp.read()
    dest_path.write_bytes(buf)
    print(f"[data] wrote {dest_path}")
    return dest_path


def _flatten_categories(cats) -> str:
    """Flatten `categories` (list of category paths) to a deduped, ordered CSV."""
    if not isinstance(cats, list):
        return ""
    seen: dict[str, None] = {}
    for path in cats:
        if isinstance(path, list):
            for c in path:
                if c not in seen:
                    seen[c] = None
    return ", ".join(seen.keys())


def load_meta(meta_path: Path) -> pd.DataFrame:
    """Parse meta_Beauty.json.gz. The lines are Python-repr-ish, so fall back to
    eval when JSON parsing fails."""
    rows: list[dict] = []
    with gzip.open(meta_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                obj = eval(line, {"__builtins__": {}}, {})
            rows.append(obj)
    df = pd.DataFrame(rows)
    keep = [c for c in ("asin", "title", "brand", "categories", "price", "description") if c in df.columns]
    df = df[keep].copy()
    df = df.rename(columns={"asin": "item_id"})
    df = df.dropna(subset=["item_id"]).drop_duplicates(subset=["item_id"]).reset_index(drop=True)
    if "categories" in df.columns:
        df["categories"] = df["categories"].apply(_flatten_categories)
    for col in ("title", "brand", "description"):
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
    return df


def collect_sequence_items(sequences_dir: Path) -> set[str]:
    """Union of item_ids across train/val/test.jsonl — the 5-core filtered set the
    recommender actually uses. Mirrors tiger/scripts/build_sid_tables.py so the RQ-VAE
    trains on exactly the items that will later be assigned SIDs."""
    items: set[str] = set()
    for split in ("train.jsonl", "val.jsonl", "test.jsonl"):
        path = sequences_dir / split
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing — run `python -m tiger.scripts.preprocess_beauty` first, or set "
                f"data.sequences_dir to null to embed the full meta catalog"
            )
        with open(path, "r") as f:
            for line in f:
                rec = json.loads(line)
                items.update(rec["history"])
                items.add(rec["target"])
    return items


def build_texts(df: pd.DataFrame) -> list[str]:
    """One text per item: title, brand, categories, price, description."""
    texts: list[str] = []
    for _, r in df.iterrows():
        parts: list[str] = []
        if r.get("title"):
            parts.append(f"Title: {r['title']}")
        if r.get("brand"):
            parts.append(f"Brand: {r['brand']}")
        if r.get("categories"):
            parts.append(f"Categories: {r['categories']}")
        price = r.get("price")
        if pd.notna(price):
            parts.append(f"Price: {price}")
        if r.get("description"):
            parts.append(f"Description: {r['description']}")
        texts.append(". ".join(parts))
    return texts


def embed_texts(texts: list[str], model_name: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    print(f"[data] loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"[data] encoding {len(texts)} item texts")
    emb = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype(np.float32)
    return emb


def standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-dimension z-score, guarding zero-variance dims. Returns (x_std, mean, std)."""
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std_safe = np.where(std < 1e-8, 1.0, std)
    x_std = (x - mean) / std_safe
    return x_std.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def prepare(config: dict, download: bool) -> None:
    data_cfg = config["data"]

    meta_path = Path(data_cfg["meta_path"])
    if download or not meta_path.exists():
        download_meta(meta_path)

    df = load_meta(meta_path)
    print(f"[data] loaded {len(df)} beauty items")

    # Restrict to the 5-core item set the recommender uses, instead of embedding
    # the whole ~259k meta catalog. Keeps the RQ-VAE's codebook focused on items
    # that actually get SIDs and cuts encoding time ~20x.
    seq_dir = data_cfg.get("sequences_dir")
    if seq_dir:
        wanted = collect_sequence_items(Path(seq_dir))
        before = len(df)
        df = df[df["item_id"].astype(str).isin(wanted)].reset_index(drop=True)
        missing = len(wanted) - len(df)
        print(
            f"[data] filtered to 5-core items: {len(df)}/{before} kept "
            f"({len(wanted)} sequence items, {missing} have no metadata)"
        )

    texts = build_texts(df)
    emb = embed_texts(texts, data_cfg["embedding_model"])
    assert emb.shape[1] == data_cfg["input_dim"], (
        f"embedding dim {emb.shape[1]} != configured input_dim {data_cfg['input_dim']}"
    )

    if data_cfg.get("standardize", True):
        emb_std, mean, std = standardize(emb)
    else:
        emb_std = emb
        mean = np.zeros(emb.shape[1], dtype=np.float32)
        std = np.ones(emb.shape[1], dtype=np.float32)

    emb_path = Path(data_cfg["embeddings_path"])
    stats_path = Path(data_cfg["stats_path"])
    items_csv = Path(data_cfg["items_csv"])
    for p in (emb_path, stats_path, items_csv):
        p.parent.mkdir(parents=True, exist_ok=True)

    np.save(emb_path, emb_std)
    np.savez(stats_path, mean=mean, std=std)
    df.to_csv(items_csv, index=False)

    print(f"[data] wrote {emb_path}  shape={emb_std.shape} dtype={emb_std.dtype}")
    print(f"[data] wrote {stats_path}")
    print(f"[data] wrote {items_csv}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--download", action="store_true")
    args = ap.parse_args()
    prepare(load_config(args.config), download=args.download)


if __name__ == "__main__":
    main()
