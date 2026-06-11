"""MovieLens 1M loader + content embedding builder.

    ml-1m.zip -> movies.dat -> text per movie -> sentence-transformer ->
    standardize -> cache (embeddings.npy + movies.csv + stats.npz)
"""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

from .config import load_config

ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


def download_ml1m(dest_dir: Path) -> Path:
    """Download and extract ml-1m.zip into dest_dir. Returns path to ml-1m/."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    extracted = dest_dir / "ml-1m"
    if (extracted / "movies.dat").exists():
        print(f"[data] already extracted at {extracted}")
        return extracted

    print(f"[data] downloading {ML1M_URL}")
    with urlopen(ML1M_URL) as resp:
        buf = io.BytesIO(resp.read())
    with zipfile.ZipFile(buf) as zf:
        zf.extractall(dest_dir)
    print(f"[data] extracted to {extracted}")
    return extracted


def load_movies(movies_path: Path, encoding: str = "latin-1") -> pd.DataFrame:
    """Parse movies.dat (columns movieId, title, genres). The `::` separator
    forces the python engine; latin-1 handles the accented characters."""
    df = pd.read_csv(
        movies_path,
        sep="::",
        engine="python",
        header=None,
        names=["movieId", "title", "genres"],
        encoding=encoding,
    )
    return df


def build_texts(df: pd.DataFrame) -> list[str]:
    """One text per movie: 'Title. Genres: g1, g2, g3.'"""
    texts = []
    for title, genres in zip(df["title"], df["genres"]):
        genres_csv = ", ".join(genres.split("|"))
        texts.append(f"{title}. Genres: {genres_csv}.")
    return texts


def embed_texts(texts: list[str], model_name: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    print(f"[data] loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"[data] encoding {len(texts)} movie texts")
    emb = model.encode(
        texts,
        batch_size=64,
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

    movies_path = Path(data_cfg["movies_path"])
    if download or not movies_path.exists():
        download_ml1m(movies_path.parent.parent)  # data/ml-1m/movies.dat -> data/

    df = load_movies(movies_path, encoding=data_cfg["encoding"])
    print(f"[data] loaded {len(df)} movies")

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
