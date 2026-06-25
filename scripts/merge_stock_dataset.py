"""
Final step — Merge all shard parquets into a single stock dataset.

Reads every backtest_dataset/STOCKS/shards/shard_*.parquet produced by
build_stock_dataset.py and concatenates them into one file:

    backtest_dataset/STOCKS/stock_dataset.parquet

Rows are deduplicated on (ticker, date_str) — if a ticker was (re)processed
in more than one shard, the last occurrence wins. Output is sorted by
(ticker, date_str).

Usage (from backtester_api/):
    python -m scripts.merge_stock_dataset
    python -m scripts.merge_stock_dataset --shards-dir backtest_dataset/STOCKS/shards
    python -m scripts.merge_stock_dataset --out backtest_dataset/STOCKS/stock_dataset.parquet
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SHARDS_DIR = Path("backtest_dataset/STOCKS/shards")
OUTPUT_PATH = Path("backtest_dataset/STOCKS/stock_dataset.parquet")


def merge(shards_dir: Path, out_path: Path) -> None:
    shard_files = sorted(shards_dir.glob("shard_*.parquet"))
    if not shard_files:
        raise FileNotFoundError(f"No shard_*.parquet files found in {shards_dir}.")

    logger.info("Merging %d shard file(s) from %s", len(shard_files), shards_dir)

    parts = []
    for path in shard_files:
        df = pd.read_parquet(path)
        logger.info("  %-32s %d rows, %d tickers", path.name, len(df), df["ticker"].nunique())
        parts.append(df)

    merged = (pd.concat(parts, ignore_index=True)
                .drop_duplicates(subset=["ticker", "date_str"], keep="last")
                .sort_values(["ticker", "date_str"])
                .reset_index(drop=True))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path, index=False, compression="zstd")
    logger.info(
        "Wrote %s: %d rows, %d tickers, %s -> %s  (%.1f MB)",
        out_path, len(merged), merged["ticker"].nunique(),
        merged["date_str"].min(), merged["date_str"].max(),
        out_path.stat().st_size / 1e6,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge per-shard parquets into a single stock_dataset.parquet."
    )
    parser.add_argument("--shards-dir", default=str(SHARDS_DIR),
                        help=f"Directory holding shard_*.parquet (default: {SHARDS_DIR}).")
    parser.add_argument("--out", default=str(OUTPUT_PATH),
                        help=f"Output parquet path (default: {OUTPUT_PATH}).")
    args = parser.parse_args()

    merge(Path(args.shards_dir), Path(args.out))


if __name__ == "__main__":
    main()
