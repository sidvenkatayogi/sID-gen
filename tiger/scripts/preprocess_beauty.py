"""Build train/val/test sequences from the Amazon Beauty 5-core reviews.

Parses the reviews dump, applies an iterative 5-core filter, sorts each user's
items by timestamp, and writes a leave-one-out split as jsonl rows of the form
`{"user_id": ..., "history": [...], "target": ...}`:

    train:  (history=[i_1..i_{k-1}], target=i_k)  for k in [2, n-2]
    val:    (history=[i_1..i_{n-2}], target=i_{n-1})
    test:   (history=[i_1..i_{n-1}], target=i_n)

History truncation to 20 happens in the Dataset, not here. Expected stats after
preprocessing: users 22,363 / items 12,101 / mean seq len 8.87 / median 6.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
from urllib.request import urlopen

REVIEWS_URL = (
    "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_5.json.gz"
)


def download(dest_path: Path) -> None:
    if dest_path.exists():
        print(f"[prep] already downloaded at {dest_path}")
        return
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[prep] downloading {REVIEWS_URL}")
    with urlopen(REVIEWS_URL) as resp:
        buf = resp.read()
    dest_path.write_bytes(buf)
    print(f"[prep] wrote {dest_path} ({len(buf)/1e6:.1f} MB)")


def parse_reviews(path: Path) -> list[tuple[str, str, int]]:
    """Return list of (user_id, item_id, timestamp). Missing fields drop the row.
    McAuley dumps are Python-repr-ish, so fall back to eval when JSON fails."""
    rows: list[tuple[str, str, int]] = []
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                obj = eval(line, {"__builtins__": {}}, {})
            u = obj.get("reviewerID")
            i = obj.get("asin")
            t = obj.get("unixReviewTime")
            if u is None or i is None or t is None:
                continue
            rows.append((str(u), str(i), int(t)))
    print(f"[prep] parsed {len(rows)} reviews")
    return rows


def five_core_filter(
    rows: list[tuple[str, str, int]],
    min_count: int = 5,
) -> list[tuple[str, str, int]]:
    """Iteratively drop users/items with fewer than `min_count` interactions
    until both sides satisfy the threshold."""
    cur = rows
    for it in range(20):
        user_counts = Counter(u for u, _, _ in cur)
        item_counts = Counter(i for _, i, _ in cur)
        keep = [
            (u, i, t)
            for (u, i, t) in cur
            if user_counts[u] >= min_count and item_counts[i] >= min_count
        ]
        if len(keep) == len(cur):
            print(f"[prep] 5-core filter converged after {it} pass(es)")
            return keep
        cur = keep
    raise RuntimeError("5-core filter did not converge in 20 passes")


def build_user_sequences(
    rows: list[tuple[str, str, int]],
) -> dict[str, list[str]]:
    """Per user, the chronologically sorted item list (stable on timestamp ties)."""
    grouped: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for u, i, t in rows:
        grouped[u].append((t, i))
    seqs: dict[str, list[str]] = {}
    for u, lst in grouped.items():
        lst.sort(key=lambda x: x[0])
        seqs[u] = [i for _, i in lst]
    return seqs


def split_leave_one_out(
    seqs: dict[str, list[str]],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Generate train / val / test rows from per-user sequences. Train uses the
    sliding-window scheme: every prefix of length 1..n-3 is one example."""
    train: list[dict] = []
    val: list[dict] = []
    test: list[dict] = []

    for u, seq in seqs.items():
        n = len(seq)
        if n < 3:
            continue

        for k in range(2, n - 1):
            train.append(
                {"user_id": u, "history": seq[: k - 1], "target": seq[k - 1]}
            )

        val.append({"user_id": u, "history": seq[: n - 2], "target": seq[n - 2]})
        test.append({"user_id": u, "history": seq[: n - 1], "target": seq[n - 1]})

    return train, val, test


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[prep] wrote {path} ({len(rows)} rows)")


def report_stats(seqs: dict[str, list[str]]) -> None:
    lengths = sorted(len(s) for s in seqs.values())
    items = set()
    for s in seqs.values():
        items.update(s)
    n = len(lengths)
    mean = sum(lengths) / n
    median = lengths[n // 2]
    print(
        f"[prep] users={n}  items={len(items)}  "
        f"mean_seq_len={mean:.2f}  median_seq_len={median}  "
        f"(expected: 22,363 / 12,101 / 8.87 / 6)"
    )


def preprocess(reviews_path: Path, output_dir: Path, download_flag: bool) -> None:
    if download_flag or not reviews_path.exists():
        download(reviews_path)

    rows = parse_reviews(reviews_path)
    rows = five_core_filter(rows, min_count=5)
    print(f"[prep] after 5-core: {len(rows)} reviews")

    seqs = build_user_sequences(rows)
    report_stats(seqs)

    train, val, test = split_leave_one_out(seqs)
    write_jsonl(train, output_dir / "train.jsonl")
    write_jsonl(val, output_dir / "val.jsonl")
    write_jsonl(test, output_dir / "test.jsonl")

    print(f"[prep] done: train={len(train)}  val={len(val)}  test={len(test)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--reviews",
        type=Path,
        default=Path("data/raw/reviews_Beauty_5.json.gz"),
        help="path to the 5-core reviews dump",
    )
    ap.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    ap.add_argument(
        "--download",
        action="store_true",
        help="force re-download even if the file exists",
    )
    args = ap.parse_args()
    preprocess(args.reviews, args.output_dir, args.download)


if __name__ == "__main__":
    main()
