"""
Incremental updater for backtest_dataset/full/{5m,15m}/full_dataset.parquet.

Steps:
  1. Read full/5m/full_dataset.parquet  → find the most recent date_str
  2. Query stock_data_filtered via PostgREST for rows with date_str > that date
  3. For each ticker+new_dates, fetch 5m and 15m candles from Massive,
     compute indicators (with proper SMA warmup lookback)
  4. Upsert the new rows into full/5m/full_dataset.parquet and
                                full/15m/full_dataset.parquet
  5. Upsert into full/5m/tickers/<ticker>.parquet and
                  full/15m/tickers/<ticker>.parquet
  6. Upsert the stock_data_filtered rows into
     backtest_dataset/pending_backtest.parquet so a future pipeline can run
     strategies against the new ticker-days and append the resulting trades.

Usage (from backtester_api/):
    python -m scripts.update_full_dataset
    python -m scripts.update_full_dataset --dry-run   # shows what would be fetched
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import httpx

sys.path.insert(0, os.path.abspath("."))

from app.config import get_settings
from app.utils.indicators import (
    compute_atr,
    compute_donchian,
    compute_rvol,
    compute_sma,
    compute_vwap,
)
from app.utils.massive import fetch_candles

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OUTPUT_BASE          = Path("backtest_dataset/full")
PENDING_BACKTEST_PATH = Path("backtest_dataset/pending_backtest.parquet")

# Extra calendar days fetched before the target window for indicator warmup.
_LOOKBACK_DAYS: dict[str, int] = {
    "5m":  15,
    "15m": 30,
}

TIMEFRAMES = list(_LOOKBACK_DAYS.keys())

COLUMNS = [
    "ticker", "date", "date_str",
    "open", "high", "low", "close", "volume",
    "atr", "RVOL_daily", "SMA_VOLUME_20_5m", "vwap", "previous_day_close",
    "sma_9", "sma_200",
    "donchian_upper", "donchian_lower", "donchian_basis",
]


# ---------------------------------------------------------------------------
# Step 1 — find the latest date already stored
# ---------------------------------------------------------------------------

def get_latest_date_str() -> str:
    """Return the most recent date_str in the 5m full_dataset.parquet."""
    path = OUTPUT_BASE / "5m" / "full_dataset.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Full dataset not found: {path}")
    df = pd.read_parquet(path, columns=["date_str"])
    latest = df["date_str"].max()
    logger.info("Latest date in 5m full_dataset: %s", latest)
    return latest


# ---------------------------------------------------------------------------
# Step 2 — query stock_data_filtered for new entries
# ---------------------------------------------------------------------------

def fetch_new_filtered_entries(since_date_str: str) -> pd.DataFrame:
    """
    Fetch stock_data_filtered rows where date_str > since_date_str.

    Uses PostgREST horizontal filter syntax:
        /stock_data_filtered?date_str=gt.<date>&order=date_str.asc
    """
    settings   = get_settings()
    base_url   = settings.postgrest_url.rstrip("/") + "/stock_data_filtered"
    headers    = {
        "Authorization": f"Bearer {settings.postgrest_token}",
        "Accept":        "application/json",
    }

    page_size = 10_000
    all_rows: list[dict] = []
    offset = 0

    with httpx.Client(timeout=60) as client:
        while True:
            url = (
                f"{base_url}"
                f"?date_str=gt.{since_date_str}"
                f"&order=date_str.asc"
                f"&limit={page_size}&offset={offset}"
            )
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
            rows = resp.json()
            if not rows:
                break
            all_rows.extend(rows)
            if len(rows) < page_size:
                break
            offset += page_size

    if not all_rows:
        logger.info("No new entries in stock_data_filtered after %s.", since_date_str)
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    logger.info(
        "Fetched %d new stock_data_filtered rows (%d tickers) after %s.",
        len(df), df["ticker"].nunique(), since_date_str,
    )
    return df


# ---------------------------------------------------------------------------
# Step 3 — fetch candles + compute indicators for one ticker
# ---------------------------------------------------------------------------

def _build_ticker_df(
    ticker: str,
    from_date: date,
    to_date: date,
    daily_lookup: dict[tuple[str, str], float],
    timeframe: str,
) -> pd.DataFrame | None:
    """
    Fetch candles for `ticker` in [from_date, to_date], compute all indicators,
    and return a DataFrame aligned to COLUMNS.

    `daily_lookup` maps (ticker, date_str) → previous_close.
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

    dt_et          = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert("America/New_York")
    df["date"]     = pd.to_datetime(dt_et.dt.strftime("%Y-%m-%dT%H:%M:%S"))
    df["date_str"] = dt_et.dt.strftime("%Y-%m-%d")
    df["ticker"]   = ticker

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

    df["previous_day_close"] = df["date_str"].map(
        lambda ds: daily_lookup.get((ticker, ds), float("nan"))
    )

    return df[COLUMNS]


# ---------------------------------------------------------------------------
# Step 4+5 — upsert helpers
# ---------------------------------------------------------------------------

def _upsert_full_dataset(new_df: pd.DataFrame, timeframe: str) -> None:
    """
    Upsert new_df into full/{{timeframe}}/full_dataset.parquet.

    Rows whose (ticker, date_str) already exist are replaced; new ones are
    appended. Result is sorted by (ticker, date).
    """
    path = OUTPUT_BASE / timeframe / "full_dataset.parquet"
    if not path.exists():
        logger.warning("full_dataset.parquet not found for %s — creating it.", timeframe)
        new_df.sort_values(["ticker", "date"]).reset_index(drop=True).to_parquet(
            path, index=False, compression="zstd"
        )
        logger.info("Created %s (%d rows)", path, len(new_df))
        return

    existing = pd.read_parquet(path)
    key      = set(zip(new_df["ticker"], new_df["date_str"]))
    existing = existing[
        ~existing.apply(lambda r: (r["ticker"], r["date_str"]) in key, axis=1)
    ]
    merged = (
        pd.concat([existing, new_df], ignore_index=True)
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )
    merged.to_parquet(path, index=False, compression="zstd")
    logger.info(
        "Updated %s: +%d new rows → %d total  (%.1f MB)",
        path, len(new_df), len(merged), path.stat().st_size / 1e6,
    )


def _upsert_ticker_parquet(df: pd.DataFrame, timeframe: str) -> None:
    """
    Upsert df into full/{{timeframe}}/tickers/<ticker>.parquet.

    Creates the directory and file if they don't exist. Replaces existing
    rows for the same date_str; new rows are appended.
    """
    ticker   = df["ticker"].iloc[0]
    out_dir  = OUTPUT_BASE / timeframe / "tickers"
    out_dir.mkdir(parents=True, exist_ok=True)
    path     = out_dir / f"{ticker}.parquet"

    if path.exists():
        existing = pq.read_table(path).to_pandas()
        existing = existing[~existing["date_str"].isin(df["date_str"])]
        df = pd.concat([existing, df], ignore_index=True)

    df = df.sort_values(["date"]).reset_index(drop=True)
    pq.write_table(
        pa.Table.from_pandas(df, preserve_index=False),
        path,
        compression="zstd",
    )
    logger.debug("ticker parquet  %-6s  %s  rows=%d", ticker, timeframe, len(df))


# ---------------------------------------------------------------------------
# Step 6 — pending backtest queue
# ---------------------------------------------------------------------------

def _upsert_pending_backtest(new_rows: pd.DataFrame) -> None:
    """
    Upsert stock_data_filtered rows into pending_backtest.parquet.

    The file acts as a queue for the backtest pipeline: each row is one
    ticker-day that needs to be run through every strategy.  Rows are keyed
    by (ticker, date_str); re-running the update pipeline for the same dates
    is safe — existing rows are replaced rather than duplicated.

    Columns: all columns returned by stock_data_filtered (ticker, date_str,
    gap, gap_perc, previous_close, open, high, low, close, volume, …).
    """
    if new_rows.empty:
        return

    path = PENDING_BACKTEST_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        existing = pd.read_parquet(path)
        key      = set(zip(new_rows["ticker"], new_rows["date_str"]))
        existing = existing[
            ~existing.apply(lambda r: (r["ticker"], r["date_str"]) in key, axis=1)
        ]
        new_rows = pd.concat([existing, new_rows], ignore_index=True)

    new_rows = new_rows.sort_values(["date_str", "ticker"]).reset_index(drop=True)
    new_rows.to_parquet(path, index=False, compression="zstd")
    logger.info(
        "pending_backtest: %d rows queued → %d total  (%s)",
        len(new_rows), len(new_rows), path,
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(dry_run: bool = False) -> None:
    # Step 1
    latest_date_str = get_latest_date_str()

    # Step 2
    filtered_df = fetch_new_filtered_entries(latest_date_str)
    if filtered_df.empty:
        logger.info("Dataset is already up to date. Nothing to do.")
        return

    # Normalise: ensure previous_close column name
    if "previous_close" not in filtered_df.columns and "previous_day_close" in filtered_df.columns:
        filtered_df = filtered_df.rename(columns={"previous_day_close": "previous_close"})

    tickers = sorted(filtered_df["ticker"].unique())
    logger.info("New data spans %d tickers, %d ticker-days.", len(tickers), len(filtered_df))

    if dry_run:
        print(filtered_df[["ticker", "date_str"]].to_string(index=False))
        return

    # Step 3+4+5 — process each ticker across both timeframes
    for_full: dict[str, list[pd.DataFrame]] = {tf: [] for tf in TIMEFRAMES}

    for n, ticker in enumerate(tickers, 1):
        ticker_rows = filtered_df[filtered_df["ticker"] == ticker]
        new_dates   = set(ticker_rows["date_str"])
        min_date    = date.fromisoformat(ticker_rows["date_str"].min())
        max_date    = date.fromisoformat(ticker_rows["date_str"].max())

        daily_lookup = {
            (ticker, row["date_str"]): row["previous_close"]
            for _, row in ticker_rows.iterrows()
        }

        if n % 50 == 0 or n == 1:
            logger.info("Processing ticker %d / %d  (%s)", n, len(tickers), ticker)

        for tf in TIMEFRAMES:
            lookback  = _LOOKBACK_DAYS[tf]
            fetch_from = min_date - timedelta(days=lookback)

            try:
                df_tf = _build_ticker_df(ticker, fetch_from, max_date, daily_lookup, tf)
            except Exception as exc:
                logger.warning("Failed fetch  ticker=%-6s  tf=%s  %s", ticker, tf, exc)
                continue

            if df_tf is None or df_tf.empty:
                logger.debug("No candles  ticker=%-6s  tf=%s", ticker, tf)
                continue

            # Keep only the new date_str values (trim warmup + existing dates)
            df_tf = df_tf[df_tf["date_str"].isin(new_dates)]
            if df_tf.empty:
                continue

            for_full[tf].append(df_tf)
            _upsert_ticker_parquet(df_tf, tf)

    # Step 4 — write full_dataset.parquet for each timeframe
    for tf in TIMEFRAMES:
        parts = for_full[tf]
        if not parts:
            logger.info("No new %s data to write.", tf)
            continue
        new_df = pd.concat(parts, ignore_index=True).sort_values(["ticker", "date"])
        _upsert_full_dataset(new_df, tf)

    # Step 6 — queue new ticker-days for the backtest pipeline
    _upsert_pending_backtest(filtered_df)

    logger.info("Update complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Incrementally update full_dataset.parquet (5m + 15m) from stock_data_filtered."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print which ticker-days would be fetched without writing anything.",
    )
    args = parser.parse_args()
    run(dry_run=args.dry_run)
