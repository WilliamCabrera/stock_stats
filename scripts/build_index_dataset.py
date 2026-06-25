"""
Build a full OHLCV dataset for any index or large-cap ticker.

Fetches the last N years of candles from Massive in 1-month batches,
computes indicators, and joins cross-timeframe columns:

    1d  → sma_9/20/50/100/200, atr_14, daily_range, daily_range_ma10
    1h  → sma_9/20/50/200, atr_14, daily_range_ma10
    5m  → sma_9/20/50/200, atr_14, daily_range_ma10, h1_9am_high, h1_9am_low
    10m → sma_9/20/50/200, atr_14, daily_range_ma10, h1_9am_high, h1_9am_low

Output:
    backtest_dataset/INDICES/{TICKER}/{tf}/{ticker_lower}_full_dataset.parquet

Build order: 1d → 1h → 5m / 10m (each step joins columns from the previous one).

Usage (from backtester_api/):
    python -m scripts.build_index_dataset --ticker SPY
    python -m scripts.build_index_dataset --ticker AAPL --timeframe 5m
    python -m scripts.build_index_dataset --ticker QQQ --from 2023-01-01 --to 2024-12-31
    python -m scripts.build_index_dataset --ticker NVDA --years 3
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

TIMEFRAMES   = ["1d", "1h", "5m", "10m"]
SESSION      = ("04:00", "20:00")
DEFAULT_YEARS = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base(ticker: str) -> Path:
    return Path("backtest_dataset/INDICES") / ticker.upper()


def _parquet_name(ticker: str) -> str:
    return f"{ticker.lower()}_full_dataset.parquet"


def _default_from(years: int) -> date:
    today = date.today()
    try:
        return today.replace(year=today.year - years)
    except ValueError:
        return today.replace(year=today.year - years, day=28)


def _month_ranges(from_date: date, to_date: date) -> list[tuple[date, date]]:
    ranges, start = [], from_date
    while start <= to_date:
        next_m = date(start.year + (start.month // 12), start.month % 12 + 1, 1)
        ranges.append((start, min(next_m - timedelta(days=1), to_date)))
        start = next_m
    return ranges


def _fetch(ticker: str, tf: str, from_date: date, to_date: date) -> pd.DataFrame:
    kw: dict = dict(timeframe=tf)
    if tf != "1d":
        kw["session_start"], kw["session_end"] = SESSION
    batches = _month_ranges(from_date, to_date)
    logger.info("Fetching %s %s: %s → %s  (%d batches)", ticker, tf, from_date, to_date, len(batches))
    parts = []
    for n, (s, e) in enumerate(batches, 1):
        candles = fetch_candles(ticker, s.isoformat(), e.isoformat(), **kw)
        if not candles:
            logger.warning("Batch %d/%d  %s→%s: no data", n, len(batches), s, e)
            continue
        df = pd.DataFrame(candles)
        dt             = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert("America/New_York")
        df["date"]     = pd.to_datetime(dt.dt.strftime("%Y-%m-%dT%H:%M:%S"))
        df["date_str"] = dt.dt.strftime("%Y-%m-%d")
        df["ticker"]   = ticker.upper()
        cols = ["ticker", "date", "date_str", "open", "high", "low", "close", "volume"]
        parts.append(df[cols])
        logger.info("Batch %d/%d  %s→%s: %d bars", n, len(batches), s, e, len(df))
    if not parts:
        return pd.DataFrame()
    return (pd.concat(parts, ignore_index=True)
              .drop_duplicates(subset=["date"])
              .sort_values("date").reset_index(drop=True))


def _indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["sma_9"]   = compute_sma(df, window=9)
    df["sma_20"]  = compute_sma(df, window=20)
    df["sma_50"]  = compute_sma(df, window=50)
    df["sma_200"] = compute_sma(df, window=200)
    df["atr_14"]  = compute_atr(df, window=14)
    return df


def _save(df: pd.DataFrame, ticker: str, tf: str) -> Path:
    path = _base(ticker) / tf / _parquet_name(ticker)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression="zstd")
    logger.info("Wrote %s: %d rows, %s → %s  (%.1f MB)",
                path, len(df), df["date_str"].min(), df["date_str"].max(),
                path.stat().st_size / 1e6)
    return path


def _h1_9am(df1h: pd.DataFrame) -> pd.DataFrame:
    return (df1h[df1h["date"].dt.hour == 9][["date_str", "high", "low"]]
               .rename(columns={"high": "h1_9am_high", "low": "h1_9am_low"}))


# ---------------------------------------------------------------------------
# Per-timeframe builders
# ---------------------------------------------------------------------------

def build_1d(ticker: str, from_date: date, to_date: date) -> pd.DataFrame:
    df = _fetch(ticker, "1d", from_date, to_date)
    if df.empty:
        logger.error("%s 1d: no data fetched.", ticker)
        return df
    df = _indicators(df)
    df["sma_100"]          = compute_sma(df, window=100)
    df["daily_range"]      = df["high"] - df["low"]
    df["daily_range_ma10"] = df["daily_range"].rolling(10, min_periods=1).mean()
    _save(df, ticker, "1d")
    return df


def build_1h(ticker: str, from_date: date, to_date: date,
             df1d: pd.DataFrame | None = None) -> pd.DataFrame:
    if df1d is None:
        path = _base(ticker) / "1d" / _parquet_name(ticker)
        df1d = pd.read_parquet(path, columns=["date_str", "daily_range_ma10"])
    df = _fetch(ticker, "1h", from_date, to_date)
    if df.empty:
        logger.error("%s 1h: no data fetched.", ticker)
        return df
    df = _indicators(df)
    df = df.merge(df1d[["date_str", "daily_range_ma10"]], on="date_str", how="left")
    _save(df, ticker, "1h")
    return df


def build_intraday(ticker: str, tf: str, from_date: date, to_date: date,
                   df1d: pd.DataFrame | None = None,
                   df1h: pd.DataFrame | None = None) -> pd.DataFrame:
    if df1d is None:
        df1d = pd.read_parquet(_base(ticker) / "1d" / _parquet_name(ticker),
                               columns=["date_str", "daily_range_ma10"])
    if df1h is None:
        df1h = pd.read_parquet(_base(ticker) / "1h" / _parquet_name(ticker),
                               columns=["date", "date_str", "high", "low"])
    df = _fetch(ticker, tf, from_date, to_date)
    if df.empty:
        logger.error("%s %s: no data fetched.", ticker, tf)
        return df
    df = _indicators(df)
    df = df.merge(df1d[["date_str", "daily_range_ma10"]], on="date_str", how="left")
    df = df.merge(_h1_9am(df1h), on="date_str", how="left")
    _save(df, ticker, tf)
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a full OHLCV + indicators dataset for any index or large-cap ticker."
    )
    parser.add_argument("--ticker", required=True,
                        help="Ticker symbol, e.g. SPY, AAPL, TQQQ.")
    parser.add_argument("--timeframe", choices=TIMEFRAMES,
                        help="Timeframe to build. Omit to build all (1d → 1h → 5m → 10m).")
    parser.add_argument("--from", dest="from_date",
                        help="Start date YYYY-MM-DD (default: --years ago).")
    parser.add_argument("--to", dest="to_date", default=date.today().isoformat(),
                        help="End date YYYY-MM-DD inclusive (default: today).")
    parser.add_argument("--years", type=int, default=DEFAULT_YEARS,
                        help=f"Years of history to fetch when --from is omitted (default: {DEFAULT_YEARS}).")
    args = parser.parse_args()

    ticker    = args.ticker.upper()
    from_date = date.fromisoformat(args.from_date) if args.from_date else _default_from(args.years)
    to_date   = date.fromisoformat(args.to_date)

    if from_date > to_date:
        parser.error(f"--from {from_date} is after --to {to_date}")

    tf = args.timeframe

    if tf is None:
        df1d = build_1d(ticker, from_date, to_date)
        df1h = build_1h(ticker, from_date, to_date, df1d)
        build_intraday(ticker, "5m",  from_date, to_date, df1d, df1h)
        build_intraday(ticker, "10m", from_date, to_date, df1d, df1h)
    elif tf == "1d":
        build_1d(ticker, from_date, to_date)
    elif tf == "1h":
        build_1h(ticker, from_date, to_date)
    else:
        build_intraday(ticker, tf, from_date, to_date)


if __name__ == "__main__":
    main()
