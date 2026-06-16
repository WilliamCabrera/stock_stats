"""
Build the TQQQ index dataset for backtest_dataset/INDICES/TQQQ/{5m,10m,1h,1d}.

Fetches the last 5 years of OHLCV candles from Massive in 1-month batches,
computes indicators, and joins cross-timeframe columns:

    1d  → sma_9/20/50/200, atr_14, daily_range, daily_range_ma10
    1h  → sma_9/20/50/200, atr_14, daily_range_ma10
    5m  → sma_9/20/50/200, atr_14, daily_range_ma10, h1_9am_high, h1_9am_low
    10m → sma_9/20/50/200, atr_14, daily_range_ma10, h1_9am_high, h1_9am_low

Output:
    backtest_dataset/INDICES/TQQQ/{5m,10m,1h,1d}/tqqq_full_dataset.parquet

Usage (from backtester_api/):
    python -m scripts.build_tqqq_dataset              # all timeframes
    python -m scripts.build_tqqq_dataset --timeframe 5m
    python -m scripts.build_tqqq_dataset --timeframe 10m
    python -m scripts.build_tqqq_dataset --timeframe 1h
    python -m scripts.build_tqqq_dataset --timeframe 1d

    # Custom date range
    python -m scripts.build_tqqq_dataset --from 2023-01-01 --to 2024-12-31
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

from app.utils.indicators import compute_atr, compute_sma
from app.utils.massive import fetch_candles

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TICKER     = "TQQQ"
BASE       = Path("backtest_dataset/INDICES/TQQQ")
NAME       = "tqqq_full_dataset.parquet"
TIMEFRAMES = ["5m", "10m", "1h", "1d"]
YEARS_BACK = 5
SESSION    = ("04:00", "20:00")
COLS       = ["ticker", "date", "date_str", "open", "high", "low", "close", "volume"]


def default_from() -> date:
    today = date.today()
    try:
        return today.replace(year=today.year - YEARS_BACK)
    except ValueError:
        return today.replace(year=today.year - YEARS_BACK, day=28)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def month_ranges(from_date: date, to_date: date) -> list[tuple[date, date]]:
    ranges, start = [], from_date
    while start <= to_date:
        next_m = date(start.year + (start.month // 12), start.month % 12 + 1, 1)
        ranges.append((start, min(next_m - timedelta(days=1), to_date)))
        start = next_m
    return ranges


def fetch_tf(tf: str, from_date: date, to_date: date) -> pd.DataFrame:
    use_session = tf != "1d"
    batches = month_ranges(from_date, to_date)
    logger.info("Fetching %s %s: %s → %s  (%d batches)", TICKER, tf, from_date, to_date, len(batches))
    parts = []
    for n, (s, e) in enumerate(batches, 1):
        kw = dict(timeframe=tf)
        if use_session:
            kw["session_start"], kw["session_end"] = SESSION
        candles = fetch_candles(TICKER, s.isoformat(), e.isoformat(), **kw)
        if not candles:
            logger.warning("Batch %d/%d  %s→%s: no data", n, len(batches), s, e)
            continue
        df = pd.DataFrame(candles)
        dt             = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert("America/New_York")
        df["date"]     = pd.to_datetime(dt.dt.strftime("%Y-%m-%dT%H:%M:%S"))
        df["date_str"] = dt.dt.strftime("%Y-%m-%d")
        df["ticker"]   = TICKER
        parts.append(df[COLS])
        logger.info("Batch %d/%d  %s→%s: %d bars", n, len(batches), s, e, len(df))
    return (pd.concat(parts, ignore_index=True)
              .drop_duplicates(subset=["date"])
              .sort_values("date").reset_index(drop=True))


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["sma_9"]   = compute_sma(df, window=9)
    df["sma_20"]  = compute_sma(df, window=20)
    df["sma_50"]  = compute_sma(df, window=50)
    df["sma_200"] = compute_sma(df, window=200)
    df["atr_14"]  = compute_atr(df, window=14)
    return df


def save(df: pd.DataFrame, tf: str) -> Path:
    path = BASE / tf / NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression="zstd")
    logger.info("Wrote %s: %d rows, %s → %s  (%.1f MB)",
                path, len(df), df["date_str"].min(), df["date_str"].max(),
                path.stat().st_size / 1e6)
    return path


# ---------------------------------------------------------------------------
# Per-timeframe builders
# ---------------------------------------------------------------------------

def build_1d(from_date: date, to_date: date) -> pd.DataFrame:
    df = fetch_tf("1d", from_date, to_date)
    df = add_indicators(df)
    df["sma_100"]          = compute_sma(df, window=100)
    df["daily_range"]      = df["high"] - df["low"]
    df["daily_range_ma10"] = df["daily_range"].rolling(10, min_periods=1).mean()
    save(df, "1d")
    return df


def build_1h(from_date: date, to_date: date, df1d: pd.DataFrame | None = None) -> pd.DataFrame:
    if df1d is None:
        df1d = pd.read_parquet(BASE / "1d" / NAME, columns=["date_str", "daily_range_ma10"])
    df = fetch_tf("1h", from_date, to_date)
    df = add_indicators(df)
    df = df.merge(df1d[["date_str", "daily_range_ma10"]], on="date_str", how="left")
    save(df, "1h")
    return df


def _h1_9am(df1h: pd.DataFrame) -> pd.DataFrame:
    return (df1h[df1h["date"].dt.hour == 9][["date_str", "high", "low"]]
               .rename(columns={"high": "h1_9am_high", "low": "h1_9am_low"}))


def build_5m(from_date: date, to_date: date,
             df1d: pd.DataFrame | None = None,
             df1h: pd.DataFrame | None = None) -> pd.DataFrame:
    if df1d is None:
        df1d = pd.read_parquet(BASE / "1d" / NAME, columns=["date_str", "daily_range_ma10"])
    if df1h is None:
        df1h = pd.read_parquet(BASE / "1h" / NAME, columns=["date", "date_str", "high", "low"])
    df = fetch_tf("5m", from_date, to_date)
    df = add_indicators(df)
    df = df.merge(df1d[["date_str", "daily_range_ma10"]], on="date_str", how="left")
    df = df.merge(_h1_9am(df1h), on="date_str", how="left")
    save(df, "5m")
    return df


def build_10m(from_date: date, to_date: date,
              df1d: pd.DataFrame | None = None,
              df1h: pd.DataFrame | None = None) -> pd.DataFrame:
    if df1d is None:
        df1d = pd.read_parquet(BASE / "1d" / NAME, columns=["date_str", "daily_range_ma10"])
    if df1h is None:
        df1h = pd.read_parquet(BASE / "1h" / NAME, columns=["date", "date_str", "high", "low"])
    df = fetch_tf("10m", from_date, to_date)
    df = add_indicators(df)
    df = df.merge(df1d[["date_str", "daily_range_ma10"]], on="date_str", how="left")
    df = df.merge(_h1_9am(df1h), on="date_str", how="left")
    save(df, "10m")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build TQQQ dataset (5m/10m/1h/1d, last 5 years) in backtest_dataset/INDICES/TQQQ/."
    )
    parser.add_argument("--timeframe", choices=TIMEFRAMES,
                        help="Timeframe to build. Omit to build all.")
    parser.add_argument("--from", dest="from_date", default=default_from().isoformat(),
                        help="Start date YYYY-MM-DD (default: 5 years ago).")
    parser.add_argument("--to", dest="to_date", default=date.today().isoformat(),
                        help="End date YYYY-MM-DD inclusive (default: today).")
    args = parser.parse_args()

    from_date = date.fromisoformat(args.from_date)
    to_date   = date.fromisoformat(args.to_date)
    if from_date > to_date:
        parser.error(f"--from {from_date} is after --to {to_date}")

    tf = args.timeframe

    if tf is None:
        df1d = build_1d(from_date, to_date)
        df1h = build_1h(from_date, to_date, df1d)
        build_5m(from_date, to_date, df1d, df1h)
        build_10m(from_date, to_date, df1d, df1h)
    elif tf == "1d":
        build_1d(from_date, to_date)
    elif tf == "1h":
        build_1h(from_date, to_date)
    elif tf == "5m":
        build_5m(from_date, to_date)
    elif tf == "10m":
        build_10m(from_date, to_date)


if __name__ == "__main__":
    main()
