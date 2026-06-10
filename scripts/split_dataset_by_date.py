import os
import sys
sys.path.insert(0, os.path.abspath("."))

import pandas as pd
import time as tm
from pathlib import Path


def split_dataset_by_date(timeframe: str):
    """
    Reads full_dataset.parquet for the given timeframe and writes one .parquet
    file per trading date into backtest_dataset/full/{timeframe}/dates/.

    Filename: date_str with '-' replaced by '_', e.g. 2026_01_01.parquet
    Rows within each file are sorted by ticker then date.
    """
    input_path = Path(f"backtest_dataset/full/{timeframe}/full_dataset.parquet")

    if not input_path.exists():
        print(f"File not found: {input_path}")
        return

    print(f"Reading {input_path} ...")
    df = pd.read_parquet(input_path)
    n_dates = df["date_str"].nunique()
    print(f"  {len(df):,} rows | {df['ticker'].nunique()} tickers | {n_dates} dates")

    df = df.sort_values(["date_str", "ticker", "date"]).reset_index(drop=True)

    output_dir = Path(f"backtest_dataset/full/{timeframe}/dates")
    output_dir.mkdir(parents=True, exist_ok=True)

    start = tm.perf_counter()

    for i, (date_str, group) in enumerate(df.groupby("date_str", sort=True), 1):
        filename = date_str.replace("-", "_") + ".parquet"
        group.reset_index(drop=True).to_parquet(output_dir / filename, index=False)

        if i % 100 == 0 or i == n_dates:
            elapsed = tm.perf_counter() - start
            print(f"  {i}/{n_dates} dates written ({elapsed:.1f}s)")

    elapsed = tm.perf_counter() - start
    print(f"Done. {n_dates} files saved to {output_dir} ({elapsed:.1f}s)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Split full_dataset.parquet into one file per trading date."
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default=None,
        help="Timeframe to process: '5m', '15m', or 'all' (default).",
    )
    args = parser.parse_args()

    timeframe = args.timeframe or "all"

    if timeframe == "all":
        for tf in ["5m", "15m"]:
            print(f"\n=== {tf} ===")
            split_dataset_by_date(tf)
    else:
        split_dataset_by_date(timeframe)
