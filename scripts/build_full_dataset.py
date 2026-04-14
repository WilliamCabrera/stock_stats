"""
Build a full dataset from stock_data_filtered starting from 2024-01-01
up to the most recent available date.

Unlike build_walkforward_datasets.py (which splits data into IS/OOS folds),
this script produces a single parquet file with all filtered trading days.

Output layout:
    backtest_dataset/full/{timeframe}/full_dataset.parquet

Each parquet has one row per bar with columns:
    ticker, date (naive ET datetime), date_str,
    open, high, low, close, volume,
    atr, RVOL_daily, SMA_VOLUME_20_5m, vwap, previous_day_close,
    sma_9, sma_200,
    donchian_upper, donchian_lower, donchian_basis

Usage (from backtester_api/):
    python -m scripts.build_full_dataset              # default: 5m
    python -m scripts.build_full_dataset --tf 15m
    python -m scripts.build_full_dataset --tf 1h
    python -m scripts.build_full_dataset --tf 5m --from-date 2023-01-01
"""
from __future__ import annotations

import logging
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.abspath("."))

from app.utils.indicators import compute_atr, compute_donchian, compute_rvol, compute_sma, compute_vwap
from app.utils.massive import fetch_candles
from app.utils.pipeline_data_collection import fetch_stock_data_filtered

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_FROM_DATE = date(2024, 1, 1)
OUTPUT_BASE       = Path("backtest_dataset/full")

VALID_TIMEFRAMES = {"1m", "5m", "15m", "30m", "1h", "1d"}

# Calendar-day lookback per timeframe — enough to warm up SMA 200.
_LOOKBACK_DAYS: dict[str, int] = {
    "1m":  10,
    "5m":  15,
    "15m": 30,
    "30m": 50,
    "1h":  80,
    "1d":  310,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_ticker_df(
    ticker: str,
    from_date: date,
    to_date: date,
    daily_lookup: dict,
    timeframe: str,
) -> pd.DataFrame | None:
    """
    Fetch candles for `ticker` at the given `timeframe` between from_date and
    to_date, compute all needed indicators, and return a tidy DataFrame.

    `daily_lookup` maps (ticker, date_str) -> previous_close from
    stock_data_filtered.
    """
    candles = fetch_candles(
        ticker,
        from_date.isoformat(),
        to_date.isoformat(),
        timeframe=timeframe,
        session_start="04:00",
        session_end="20:00",
    )
    if not candles:
        return None

    df = pd.DataFrame(candles)

    # Naive ET datetime (consistent with strategy code)
    dt_et          = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert("America/New_York")
    df["date"]     = pd.to_datetime(dt_et.dt.strftime("%Y-%m-%dT%H:%M:%S"))
    df["date_str"] = dt_et.dt.strftime("%Y-%m-%d")
    df["ticker"]   = ticker

    # Indicators (computed on full fetched window for proper warmup)
    df["vwap"]             = compute_vwap(df).values
    df["atr"]              = compute_atr(df).values
    df["RVOL_daily"]       = compute_rvol(df).values
    df["SMA_VOLUME_20_5m"] = compute_sma(df, window=20, column="volume").values
    df["sma_9"]            = compute_sma(df, window=9).values
    df["sma_200"]          = compute_sma(df, window=200).values

    donchian = compute_donchian(df, period=5, offset=1)
    df["donchian_upper"] = donchian["donchian_upper"].values
    df["donchian_lower"] = donchian["donchian_lower"].values
    df["donchian_basis"] = donchian["donchian_basis"].values

    # previous_day_close from stock_data_filtered daily summary
    df["previous_day_close"] = df["date_str"].map(
        lambda ds: daily_lookup.get((ticker, ds), float("nan"))
    )

    keep = [
        "ticker", "date", "date_str",
        "open", "high", "low", "close", "volume",
        "atr", "RVOL_daily", "SMA_VOLUME_20_5m", "vwap", "previous_day_close",
        "sma_9", "sma_200",
        "donchian_upper", "donchian_lower", "donchian_basis",
    ]
    return df[keep]


# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------

def build_full(timeframe: str = "5m", from_date: date = DEFAULT_FROM_DATE) -> None:
    """
    Build and save a single parquet dataset for all stock_data_filtered rows
    from `from_date` up to the latest available date.

    Args:
        timeframe:  Candle timeframe string (e.g. "5m", "15m", "1h").
        from_date:  Earliest date to include (default: 2024-01-01).
    """
    if timeframe not in VALID_TIMEFRAMES:
        raise ValueError(f"Invalid timeframe '{timeframe}'. Valid: {sorted(VALID_TIMEFRAMES)}")

    logger.info("Timeframe: %s  |  From: %s", timeframe, from_date)

    # 1. Load daily summary from DB
    logger.info("Loading stock_data_filtered from PostgREST...")
    filtered_df = fetch_stock_data_filtered()
    if filtered_df.empty:
        logger.error("stock_data_filtered is empty — aborting.")
        return

    # Normalise date column
    if "date_str" not in filtered_df.columns:
        raise ValueError("Expected column 'date_str' in stock_data_filtered.")
    filtered_df["date_col"] = pd.to_datetime(filtered_df["date_str"]).dt.date

    # Rename previous_close if the DB column uses a different name
    if "previous_close" not in filtered_df.columns and "previous_day_close" in filtered_df.columns:
        filtered_df = filtered_df.rename(columns={"previous_day_close": "previous_close"})

    # 2. Filter to the requested date range
    to_date = filtered_df["date_col"].max()
    mask    = (filtered_df["date_col"] >= from_date) & (filtered_df["date_col"] <= to_date)
    window_df = filtered_df[mask]

    if window_df.empty:
        logger.error("No rows in stock_data_filtered for %s → %s. Aborting.", from_date, to_date)
        return

    tickers = sorted(window_df["ticker"].unique())
    logger.info(
        "%d tickers, %d ticker-days  (%s → %s)",
        len(tickers), len(window_df), from_date, to_date,
    )

    # 3. Build (ticker, date_str) → previous_close lookup
    daily_lookup = {
        (row["ticker"], row["date_str"]): row["previous_close"]
        for _, row in window_df.iterrows()
    }

    # Fetch start with extra lookback for indicator warmup
    lookback_days = _LOOKBACK_DAYS.get(timeframe, 30)
    fetch_from    = from_date - timedelta(days=lookback_days)

    # 4. Fetch candles per ticker, compute indicators, trim to window
    parts: list[pd.DataFrame] = []
    for n, ticker in enumerate(tickers, 1):
        if n % 100 == 0:
            logger.info("Fetching ticker %d / %d ...", n, len(tickers))
        try:
            df_tf = _build_ticker_df(ticker, fetch_from, to_date, daily_lookup, timeframe)
            if df_tf is None or df_tf.empty:
                continue
            # Trim lookback rows
            df_tf = df_tf[df_tf["date_str"] >= from_date.isoformat()]
            # Keep only gapper days present in stock_data_filtered
            ticker_dates = set(window_df.loc[window_df["ticker"] == ticker, "date_str"])
            df_tf = df_tf[df_tf["date_str"].isin(ticker_dates)]
            if not df_tf.empty:
                parts.append(df_tf)
        except Exception as exc:
            logger.warning("Failed for %s — %s", ticker, exc)

    if not parts:
        logger.error("No %s data fetched for any ticker.", timeframe)
        return

    # 5. Concatenate, sort, and save
    result = pd.concat(parts, ignore_index=True).sort_values(["ticker", "date"])

    output_path = OUTPUT_BASE / timeframe / "full_dataset.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False, compression="zstd")

    logger.info(
        "Saved %d rows → %s  (%.1f MB)",
        len(result), output_path, output_path.stat().st_size / 1e6,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build a full candle dataset from stock_data_filtered (2024 → latest)."
    )
    parser.add_argument(
        "--tf",
        default="5m",
        help="Candle timeframe (e.g. 5m, 15m, 1h). Default: 5m",
    )
    parser.add_argument(
        "--from-date",
        default="2024-01-01",
        help="Earliest date to include (YYYY-MM-DD). Default: 2024-01-01",
    )
    args = parser.parse_args()

    build_full(
        timeframe=args.tf,
        from_date=date.fromisoformat(args.from_date),
    )
