"""
Build the TQQQ index dataset for backtest_dataset/INDICES/TQQQ/{5m,1h}.

Fetches the last 5 years of OHLCV candles from Massive via fetch_candles
in 1-month batches (so no single request is too large and no data is lost),
then writes the merged result to:

    backtest_dataset/INDICES/TQQQ/5m/tqqq_full_dataset.parquet
    backtest_dataset/INDICES/TQQQ/1h/tqqq_full_dataset.parquet

Usage (from backtester_api/):
    python -m scripts.build_tqqq_dataset --timeframe 5m
    python -m scripts.build_tqqq_dataset --timeframe 1h
    python -m scripts.build_tqqq_dataset                 # both timeframes

    # Custom date range
    python -m scripts.build_tqqq_dataset --timeframe 5m --from 2023-01-01 --to 2024-12-31
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.abspath("."))

from app.utils.massive import fetch_candles

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TICKER       = "TQQQ"
OUTPUT_BASE  = Path("backtest_dataset/INDICES") / TICKER
OUTPUT_NAME  = "tqqq_full_dataset.parquet"
TIMEFRAMES   = ["5m", "1h"]

YEARS_BACK = 5

SESSION_START = "04:00"
SESSION_END   = "20:00"

COLUMNS = ["ticker", "date", "date_str", "open", "high", "low", "close", "volume"]


def default_from_date(today: date | None = None) -> date:
    """Start date: 5 years before today."""
    today = today or date.today()
    try:
        return today.replace(year=today.year - YEARS_BACK)
    except ValueError:  # Feb 29 on a non-leap target year
        return today.replace(year=today.year - YEARS_BACK, day=28)


# ---------------------------------------------------------------------------
# Monthly batching
# ---------------------------------------------------------------------------

def month_ranges(from_date: date, to_date: date) -> list[tuple[date, date]]:
    """
    Split [from_date, to_date] into consecutive ~1-month windows.

    Each window is [start, end] inclusive and windows do not overlap, so
    concatenating the batches yields the full range without duplicates.
    """
    ranges: list[tuple[date, date]] = []
    start = from_date
    while start <= to_date:
        # First day of the next month
        if start.month == 12:
            next_month = date(start.year + 1, 1, 1)
        else:
            next_month = date(start.year, start.month + 1, 1)
        end = min(next_month - timedelta(days=1), to_date)
        ranges.append((start, end))
        start = next_month
    return ranges


# ---------------------------------------------------------------------------
# Fetch + build
# ---------------------------------------------------------------------------

def build_timeframe(timeframe: str, from_date: date, to_date: date) -> None:
    """Fetch all monthly batches for one timeframe and write the parquet."""
    batches = month_ranges(from_date, to_date)
    logger.info(
        "Building %s %s dataset: %s → %s  (%d monthly batches)",
        TICKER, timeframe, from_date, to_date, len(batches),
    )

    parts: list[pd.DataFrame] = []
    for n, (start, end) in enumerate(batches, 1):
        candles = fetch_candles(
            TICKER,
            start.isoformat(),
            end.isoformat(),
            timeframe=timeframe,
            session_start=SESSION_START,
            session_end=SESSION_END,
        )
        if not candles:
            logger.warning("Batch %d/%d  %s → %s: no candles.", n, len(batches), start, end)
            continue

        df = pd.DataFrame(candles)
        dt_et          = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert("America/New_York")
        df["date"]     = pd.to_datetime(dt_et.dt.strftime("%Y-%m-%dT%H:%M:%S"))
        df["date_str"] = dt_et.dt.strftime("%Y-%m-%d")
        df["ticker"]   = TICKER
        parts.append(df[COLUMNS])

        logger.info("Batch %d/%d  %s → %s: %d bars", n, len(batches), start, end, len(df))

    if not parts:
        logger.error("No data fetched for %s %s — nothing written.", TICKER, timeframe)
        return

    full = (
        pd.concat(parts, ignore_index=True)
        .drop_duplicates(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    out_dir = OUTPUT_BASE / timeframe
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / OUTPUT_NAME
    full.to_parquet(path, index=False, compression="zstd")
    logger.info(
        "Wrote %s: %d rows, %s → %s  (%.1f MB)",
        path, len(full), full["date_str"].min(), full["date_str"].max(),
        path.stat().st_size / 1e6,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build TQQQ dataset (5m / 1h, last 5 years) in backtest_dataset/INDICES/TQQQ/."
    )
    parser.add_argument(
        "--timeframe",
        choices=TIMEFRAMES,
        help="Timeframe to build. Omit to build both (5m and 1h).",
    )
    parser.add_argument(
        "--from", dest="from_date", default=default_from_date().isoformat(),
        help="Start date YYYY-MM-DD (default: 5 years ago).",
    )
    parser.add_argument(
        "--to", dest="to_date", default=date.today().isoformat(),
        help="End date YYYY-MM-DD inclusive (default: today).",
    )
    args = parser.parse_args()

    from_date = date.fromisoformat(args.from_date)
    to_date   = date.fromisoformat(args.to_date)
    if from_date > to_date:
        parser.error(f"--from {from_date} is after --to {to_date}")

    timeframes = [args.timeframe] if args.timeframe else TIMEFRAMES
    for tf in timeframes:
        build_timeframe(tf, from_date, to_date)


if __name__ == "__main__":
    main()
