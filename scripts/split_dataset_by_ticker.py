import os
import sys
sys.path.insert(0, os.path.abspath("."))

import pandas as pd
import time as tm
from pathlib import Path


def split_dataset_by_ticker(timeframe: str):
    """
    Reads full_dataset.parquet for the given timeframe and writes one .parquet
    file per ticker into backtest_dataset/full/{timeframe}/tickers/.
    Row order from the original dataset is preserved (sorted by date).
    """
    input_path = Path(f'backtest_dataset/full/{timeframe}/full_dataset.parquet')

    if not input_path.exists():
        print(f"File not found: {input_path}")
        return

    print(f"Reading {input_path}...")
    df = pd.read_parquet(input_path)
    print(f"  {len(df):,} rows | {df['ticker'].nunique()} tickers")

    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)

    output_dir = Path(f'backtest_dataset/full/{timeframe}/tickers')
    output_dir.mkdir(parents=True, exist_ok=True)

    tickers = df['ticker'].unique()
    start = tm.perf_counter()

    for i, ticker in enumerate(tickers, 1):
        ticker_df = df[df['ticker'] == ticker].reset_index(drop=True)
        ticker_df.to_parquet(output_dir / f'{ticker}.parquet', index=False)

        if i % 500 == 0:
            elapsed = tm.perf_counter() - start
            print(f"  {i}/{len(tickers)} tickers written ({elapsed:.1f}s)")

    elapsed = tm.perf_counter() - start
    print(f"Done. {len(tickers)} files saved to {output_dir} ({elapsed:.1f}s)")

    # Row counts summary
    counts_path = Path(f'backtest_dataset/full/{timeframe}/ticker_row_counts.parquet')
    counts = (
        df.groupby('ticker')
        .size()
        .reset_index(name='row_count')
        .sort_values('row_count', ascending=False)
        .reset_index(drop=True)
    )
    counts.to_parquet(counts_path, index=False)
    print(f"Row counts saved to {counts_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Split full_dataset.parquet into one file per ticker.")
    parser.add_argument(
        "--timeframe",
        type=str,
        default=None,
        help="Timeframe to process: '5m', '15m', or 'all'. Defaults to 'all'."
    )
    args = parser.parse_args()

    timeframe = args.timeframe or "all"

    if timeframe == "all":
        for tf in ["5m", "15m"]:
            print(f"\n=== {tf} ===")
            split_dataset_by_ticker(tf)
    else:
        split_dataset_by_ticker(timeframe)
