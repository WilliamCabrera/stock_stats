"""
Filter the stock dataset to tickers with a low float.

Reads backtest_dataset/STOCKS/stock_dataset.parquet, takes each ticker's most
recent non-null float (and the market_cap from that same day), keeps the
tickers whose float is below the threshold, and writes:

    ticker, float, market_cap

Output: backtest_dataset/STOCKS/low_float_tickers.parquet

Usage (from backtester_api/):
    python -m scripts.filter_low_float
    python -m scripts.filter_low_float --max-float 20000000
    python -m scripts.filter_low_float --max-float 10e6 --out somewhere.parquet
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATASET_PATH = Path("backtest_dataset/STOCKS/stock_dataset.parquet")
OUTPUT_PATH  = Path("backtest_dataset/STOCKS/low_float_tickers.parquet")


def filter_low_float(dataset: Path, out_path: Path, max_float: float) -> pd.DataFrame:
    df = pd.read_parquet(dataset, columns=["ticker", "date_str", "float", "market_cap"])

    # Keep each ticker's most recent day that actually has a float value.
    df = df.dropna(subset=["float"])
    latest = (df.sort_values(["ticker", "date_str"])
                .groupby("ticker", as_index=False)
                .tail(1))

    low = (latest[latest["float"] < max_float]
           [["ticker", "float", "market_cap"]]
           .sort_values("float")
           .reset_index(drop=True))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    low.to_parquet(out_path, index=False, compression="zstd")
    logger.info(
        "Tickers with float < %s: %d / %d -> %s",
        f"{max_float:,.0f}", len(low), latest["ticker"].nunique(), out_path,
    )
    return low


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter stock dataset to low-float tickers.")
    parser.add_argument("--dataset", default=str(DATASET_PATH),
                        help=f"Input dataset parquet (default: {DATASET_PATH}).")
    parser.add_argument("--out", default=str(OUTPUT_PATH),
                        help=f"Output parquet (default: {OUTPUT_PATH}).")
    parser.add_argument("--max-float", type=float, default=20_000_000,
                        help="Keep tickers with float strictly below this (default: 20,000,000).")
    args = parser.parse_args()

    low = filter_low_float(Path(args.dataset), Path(args.out), args.max_float)
    print(low.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
