
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.abspath("."))
import numpy as np
import pandas as pd
import asyncio
import aiohttp
import multiprocessing
from typing import List, Tuple, Dict, Any
import logging
from datetime import date, datetime

import httpx

from app.config import get_settings
from app.utils.market_utils import (
    ticker_chunks,
    chunk_date_range,
    process_data_minutes,
    sync_data_with_prev_day_close,
    _save_state,
    fetch_live_tickers,
)
from app.utils.pipeline_v1 import (
    fetch_data_1_min,
    fetch_and_process,
    _fetch_split,
    _fetch_daily_ohlc,
    _apply_gap_logic,
    process_batch_worker,
    async_batch_runner,
    save_ticker_to_db,
    save_ticker_parquet,
    group_parameters_by_ticker,
    partition_tickers_into_batches,
    main_multiprocess_pipeline,
    MAX_CONCURRENT_REQUESTS,
)
import json
import time

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_DELISTED_CSV = os.path.join(_PROJECT_ROOT, "data", "delisted", "delisted_stocklight.csv")

# How many months of history to fetch before the delisting date
_HISTORY_START = date(2020, 1, 1)


def _parse_delisting_date(date_str: str) -> date | None:
    """
    Parse the delisting date string from the CSV.

    The CSV uses a format like "Apr 30th 2025". Strips ordinal suffixes
    (st, nd, rd, th) before parsing.

    Returns a date object or None if parsing fails.
    """
    import re
    cleaned = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", date_str.strip())
    for fmt in ("%b %d %Y", "%B %d %Y"):
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue
    logger.warning("Could not parse delisting date: %r", date_str)
    return None


def load_delisted_tickers(csv_path: str = _DELISTED_CSV) -> pd.DataFrame:
    """
    Read the delisted CSV and return a cleaned DataFrame with columns:
        ticker        — equity symbol (from ``code``)
        delisting_date — date object (from ``date``)

    Rows with unparseable dates or invalid tickers (containing '.' or not
    purely alphabetic) are dropped with a warning.
    """
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"code": "ticker"})

    # Drop tickers that contain dots or non-alpha chars (warrants, units, etc.)
    original_len = len(df)
    df = df[df["ticker"].str.match(r"^[A-Z]+$", na=False)].copy()
    dropped = original_len - len(df)
    if dropped:
        logger.info("Dropped %d non-standard tickers (warrants/units/etc.)", dropped)

    df["delisting_date"] = df["date"].apply(_parse_delisting_date)
    df = df.dropna(subset=["delisting_date"])
    df = df[["ticker", "delisting_date"]].drop_duplicates(subset=["ticker"])

    logger.info("Loaded %d delisted tickers from %s", len(df), csv_path)
    return df


def build_delisted_chunks(
    delisted_df: pd.DataFrame,
    history_start: date = _HISTORY_START,
) -> list[tuple[str, str, str]]:
    """
    Build monthly (ticker, from_date, to_date) chunks for each delisted ticker.

    The to_date for each ticker is its delisting date.
    The from_date is always ``history_start`` (default 2020-01-01).

    Args:
        delisted_df:    DataFrame with columns ``ticker`` and ``delisting_date``.
        history_start:  Fixed start date for all tickers (default 2020-01-01).

    Returns:
        Flat list of (ticker, from_str, to_str) tuples.
    """
    all_chunks: list[tuple[str, str, str]] = []
    for _, row in delisted_df.iterrows():
        ticker         = row["ticker"]
        raw_date       = row["delisting_date"]

        try:
            delisting_date = raw_date if isinstance(raw_date, date) else date.fromisoformat(str(raw_date))
        except (ValueError, TypeError):
            delisting_date = None

        if delisting_date is None or pd.isna(raw_date):
            delisting_date = date.today()
            logger.debug("%s — delisting_date inválida, usando fecha actual %s.", ticker, delisting_date)

        if delisting_date <= history_start:
            logger.debug("Skipping %s — delisting date %s is on or before history_start %s.", ticker, delisting_date, history_start)
            continue
        all_chunks.extend(chunk_date_range(ticker, history_start, delisting_date))

    logger.info("Built %d chunks for %d delisted tickers.", len(all_chunks), len(delisted_df))
    return all_chunks


_TICKERS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "all_tickers_merged.csv")


def compare_tickers(
    tickers_csv: str = _TICKERS_CSV,
    delisted_csv: str = _DELISTED_CSV,
) -> pd.DataFrame:
    """
    Fetch the currently active ticker list from Massive and cross-reference it
    against the local CSV plus the delisted CSV to produce a single deduplicated
    DataFrame.

    Each ticker appears exactly once. ``status`` is set to:
        "listed"   — ticker is present in the live Massive API response
        "delisted" — ticker is NOT in the live Massive API response

    Steps:
        1. Load all known tickers from ``tickers_csv`` (local cache).
        2. Load delisted tickers from ``delisted_csv``.
        3. Fetch the live active-stock list from Massive (same logic as sync_tickers).
        4. Build a unified deduplicated DataFrame; mark each ticker by status.

    Returns:
        DataFrame with columns: ticker, company_name, stock_market,
        delisting_date (NaT if not delisted), status.
    """
    import httpx

    settings = get_settings()
    api_key  = settings.massive_api_key
    base_url = settings.massive_base_url

    # 1. Local CSV cache
    local_df = pd.read_csv(tickers_csv)
    local_df["ticker"] = local_df["ticker"].str.upper()

    # 2. Delisted CSV
    delisted_df = load_delisted_tickers(delisted_csv)
    delisted_df["ticker"] = delisted_df["ticker"].str.upper()

    # 3. Fetch live active list from Massive — read-only, no CSV merge/save
    try:
        live_tickers = fetch_live_tickers(api_key, base_url)
    except Exception as exc:
        logger.warning("compare_tickers: API unavailable (%s) — status based on local CSV only.", exc)
        live_tickers = set(local_df["ticker"].dropna())

    # 4. Union of all known tickers (local cache + delisted + live API), deduplicated
    known_tickers = set(local_df["ticker"].dropna()) | set(delisted_df["ticker"].dropna())
    new_from_api  = pd.DataFrame(
        [{"ticker": t} for t in sorted(live_tickers - known_tickers)]
    )
    all_tickers = (
        pd.concat([
            local_df[["ticker", "company_name", "stock_market"]],
            delisted_df[["ticker"]],
            new_from_api,
        ], ignore_index=True)
        .drop_duplicates(subset=["ticker"])
    )

    # Attach delisting_date where available
    all_tickers = all_tickers.merge(
        delisted_df[["ticker", "delisting_date"]], on="ticker", how="left"
    )

    # 5. Mark status: listed if in live Massive response, else delisted
    all_tickers["status"] = all_tickers["ticker"].apply(
        lambda t: "listed" if t in live_tickers else "delisted"
    )

    all_tickers = all_tickers.sort_values(["status", "ticker"]).reset_index(drop=True)

    counts = all_tickers["status"].value_counts()
    print(f"\n{'='*55}")
    print(f"  Live tickers from Massive:  {len(live_tickers):>6}")
    print(f"  Total unique tickers:       {len(all_tickers):>6}")
    print(f"{'='*55}")
    print(f"  status='listed'   :         {counts.get('listed', 0):>6}")
    print(f"  status='delisted' :         {counts.get('delisted', 0):>6}")
    print(f"{'='*55}\n")

    return all_tickers


def run_delisted_pipeline(
    history_start: date = _HISTORY_START,
    num_processes: int = 8,
) -> None:
    """
    Full pipeline for delisted tickers:

    1. Call compare_tickers() to get the live-vs-local status for all tickers.
    2. Filter to status == "delisted" — these are tickers no longer active on Massive.
    3. Build monthly fetch chunks from ``history_start`` to each ticker's delisting_date.
    4. Run the same multiprocess pipeline as pipeline_v1.

    Args:
        history_start:  Fixed start date for all tickers (default 2020-01-01).
        num_processes:  Number of parallel worker processes (default 8).
    """
    all_df = compare_tickers()
    delisted_df = all_df[all_df["status"] == "delisted"][["ticker", "delisting_date"]].copy()

    if delisted_df.empty:
        logger.warning("No delisted tickers found — aborting.")
        return

    all_chunks = build_delisted_chunks(delisted_df, history_start=history_start)
    if not all_chunks:
        logger.warning("No chunks generated — aborting.")
        return

    main_multiprocess_pipeline(
        parameters=all_chunks,
        num_processes=num_processes,
        connectionParams={},
    )


if __name__ == "__main__":
    from app.utils.logging_config import setup_logging
    setup_logging()
    run_delisted_pipeline()
    #result = compare_tickers()
    #print(result)
