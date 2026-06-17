"""
Incremental daily updater for backtest_dataset/STOCKS/stock_dataset.parquet.

Finds the most recent date already in stock_dataset.parquet, then for every
ticker in the universe:
  1. Fetches new 1D candles (with a warmup window for SMA/ATR accuracy).
  2. Computes SMA 9/20/50/100/200 and ATR 14.
  3. Fetches market_cap + float via /v3/reference/tickers/<ticker>?date=...
  4. Collects results in a staging parquet, then merges into stock_dataset.parquet.
  5. Records tickers that failed into update_failures.json; they are retried
     automatically on the next run.

Designed to run nightly via Ofelia after market close (e.g. 21:30 ET).

Usage (from backtester_api/):
    python -m scripts.update_stock_dataset
    python -m scripts.update_stock_dataset --dry-run
    python -m scripts.update_stock_dataset --concurrency 80 --ticker-concurrency 30
    python -m scripts.update_stock_dataset --no-retry-failures
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import sys
import time as tm
from datetime import date, timedelta
from pathlib import Path

import httpx
import pandas as pd

sys.path.insert(0, os.path.abspath("."))

from app.config import get_settings
from app.utils.indicators import compute_atr, compute_sma
from app.utils.massive import fetch_candles

# ---------------------------------------------------------------------------
# Logging — console (INFO) + rotating file (WARNING+), same pattern as
# backtest_helpers.py so Ofelia's save-folder captures warnings.
# ---------------------------------------------------------------------------

_LOG_DIR = Path(os.path.abspath(".")) / "logs" / "stocks"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setLevel(logging.INFO)
_console_handler.setFormatter(logging.Formatter("%(message)s"))

_file_handler = logging.FileHandler(
    _LOG_DIR / "update_stock_dataset.log", encoding="utf-8"
)
_file_handler.setLevel(logging.WARNING)
_file_handler.setFormatter(
    logging.Formatter("%(asctime)s %(levelname)-8s %(message)s",
                      datefmt="%Y-%m-%d %H:%M:%S")
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(_console_handler)
logger.addHandler(_file_handler)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

STOCK_DATASET_PATH = Path("backtest_dataset/STOCKS/stock_dataset.parquet")
STAGING_PATH       = Path("backtest_dataset/STOCKS/update_staging.parquet")
FAILURES_PATH      = Path("backtest_dataset/STOCKS/update_failures.json")
UNIVERSE_PATH      = Path("backtest_dataset/UNIVERSE/stock_universe.parquet")

_RETRY_STATUS = {429, 500, 502, 503, 504}

# Calendar days fetched before the target window so SMA-200 and ATR are warm.
_WARMUP_CALENDAR_DAYS = 300

OUTPUT_COLUMNS = [
    "ticker", "date", "date_str",
    "open", "high", "low", "close", "volume",
    "sma_9", "sma_20", "sma_50", "sma_100", "sma_200", "atr_14",
    "market_cap", "float", "shares_outstanding",
]


# ---------------------------------------------------------------------------
# Failure record
# ---------------------------------------------------------------------------

def _load_failures() -> list[dict]:
    if not FAILURES_PATH.exists():
        return []
    try:
        return json.loads(FAILURES_PATH.read_text())
    except Exception as exc:
        logger.warning("Could not load failures file: %s", exc)
        return []


def _save_failures(failures: list[dict]) -> None:
    FAILURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    FAILURES_PATH.write_text(json.dumps(failures, indent=2, default=str))
    logger.warning("Failures saved → %s  (%d tickers)", FAILURES_PATH, len(failures))


# ---------------------------------------------------------------------------
# Staging helpers (crash recovery: new rows are collected here first)
# ---------------------------------------------------------------------------

def _load_staging() -> pd.DataFrame:
    if not STAGING_PATH.exists():
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    df = pd.read_parquet(STAGING_PATH)
    logger.info("Resuming from staging: %d rows, %d tickers already done.",
                len(df), df["ticker"].nunique())
    return df


def _write_staging(df: pd.DataFrame) -> None:
    tmp = STAGING_PATH.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False, compression="zstd")
    os.replace(tmp, STAGING_PATH)


# ---------------------------------------------------------------------------
# Step 1 — find latest date already stored
# ---------------------------------------------------------------------------

def _latest_date() -> str:
    if not STOCK_DATASET_PATH.exists():
        raise FileNotFoundError(f"stock_dataset.parquet not found: {STOCK_DATASET_PATH}")
    df = pd.read_parquet(STOCK_DATASET_PATH, columns=["date_str"])
    return df["date_str"].max()


# ---------------------------------------------------------------------------
# Step 2 — candles + indicators (sync, runs in asyncio thread pool)
# ---------------------------------------------------------------------------

def _build_candles_df(ticker: str, from_date: date, to_date: date) -> pd.DataFrame:
    candles = fetch_candles(
        ticker,
        from_date.isoformat(),
        to_date.isoformat(),
        timeframe="1d",
        adjusted=True,
    )
    if not candles:
        return pd.DataFrame()

    df = pd.DataFrame(candles)
    dt = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert("America/New_York")
    df["date"]     = pd.to_datetime(dt.dt.strftime("%Y-%m-%dT%H:%M:%S"))
    df["date_str"] = dt.dt.strftime("%Y-%m-%d")
    df["ticker"]   = ticker

    df = (df.drop_duplicates(subset=["date_str"])
            .sort_values("date_str")
            .reset_index(drop=True))

    df["sma_9"]   = compute_sma(df, window=9)
    df["sma_20"]  = compute_sma(df, window=20)
    df["sma_50"]  = compute_sma(df, window=50)
    df["sma_100"] = compute_sma(df, window=100)
    df["sma_200"] = compute_sma(df, window=200)
    df["atr_14"]  = compute_atr(df, window=14)

    return df


# ---------------------------------------------------------------------------
# Step 3 — market cap + float (async reference endpoint)
# ---------------------------------------------------------------------------

async def _fetch_reference(
    client: httpx.AsyncClient,
    base: str,
    api_key: str,
    ticker: str,
    date_str: str,
    sem: asyncio.Semaphore,
    retries: int,
) -> tuple[str, dict | None]:
    url = f"{base}/v3/reference/tickers/{ticker}?date={date_str}&apiKey={api_key}"
    async with sem:
        for attempt in range(retries + 1):
            try:
                resp = await client.get(url)
            except (httpx.TimeoutException, httpx.TransportError):
                if attempt == retries:
                    return date_str, None
                await asyncio.sleep(min(0.5 * 2 ** attempt, 10) + random.random() * 0.25)
                continue

            if resp.status_code == 200:
                return date_str, resp.json().get("results")
            if resp.status_code in _RETRY_STATUS and attempt < retries:
                await asyncio.sleep(min(0.5 * 2 ** attempt, 10) + random.random() * 0.25)
                continue
            return date_str, None
    return date_str, None


async def _fetch_marketcaps(
    client: httpx.AsyncClient,
    ticker: str,
    date_strs: list[str],
    sem: asyncio.Semaphore,
    retries: int,
    base: str,
    api_key: str,
) -> pd.DataFrame:
    tasks = [
        _fetch_reference(client, base, api_key, ticker, ds, sem, retries)
        for ds in date_strs
    ]
    results = await asyncio.gather(*tasks)

    rows = []
    for ds, res in results:
        if not res:
            continue
        rows.append({
            "date_str":                       ds,
            "market_cap":                     res.get("market_cap"),
            "weighted_shares_outstanding":    res.get("weighted_shares_outstanding"),
            "share_class_shares_outstanding": res.get("share_class_shares_outstanding"),
        })

    if not rows:
        return pd.DataFrame(columns=[
            "date_str", "market_cap",
            "weighted_shares_outstanding", "share_class_shares_outstanding",
        ])
    return pd.DataFrame(rows)


def _attach_marketcap(df: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(ref, on="date_str", how="left")
    df["float"]              = df["weighted_shares_outstanding"].ffill().bfill()
    df["shares_outstanding"] = df["share_class_shares_outstanding"].ffill().bfill()
    df["market_cap"]         = df["market_cap"].ffill().bfill()
    return df.drop(columns=["weighted_shares_outstanding", "share_class_shares_outstanding"])


# ---------------------------------------------------------------------------
# Per-ticker async worker
# ---------------------------------------------------------------------------

async def _process_ticker(
    ticker: str,
    from_date: date,
    to_date: date,
    new_dates: set[str],
    client: httpx.AsyncClient,
    req_sem: asyncio.Semaphore,
    tick_sem: asyncio.Semaphore,
    retries: int,
    base: str,
    api_key: str,
) -> tuple[str, str, pd.DataFrame | None]:
    """Returns (ticker, status, df_or_None).  status: 'ok' | 'empty' | 'error'."""
    async with tick_sem:
        try:
            df = await asyncio.to_thread(_build_candles_df, ticker, from_date, to_date)
        except Exception as exc:
            logger.warning("candles failed  %-6s  %s", ticker, exc)
            return ticker, "error", None

        if df.empty:
            return ticker, "empty", None

        # Trim to only the new dates (warmup rows are discarded after indicators)
        new_df = df[df["date_str"].isin(new_dates)].copy()
        if new_df.empty:
            return ticker, "empty", None

        try:
            ref = await _fetch_marketcaps(
                client, ticker, new_df["date_str"].tolist(),
                req_sem, retries, base, api_key,
            )
        except Exception as exc:
            logger.warning("marketcap failed  %-6s  %s", ticker, exc)
            ref = pd.DataFrame(columns=[
                "date_str", "market_cap",
                "weighted_shares_outstanding", "share_class_shares_outstanding",
            ])

        new_df = _attach_marketcap(new_df, ref)
        return ticker, "ok", new_df.reindex(columns=OUTPUT_COLUMNS)


# ---------------------------------------------------------------------------
# Async driver
# ---------------------------------------------------------------------------

async def _run_async(
    tickers: list[str],
    already_done: set[str],
    from_date: date,
    to_date: date,
    new_dates: set[str],
    concurrency: int,
    ticker_concurrency: int,
    retries: int,
    flush_every: int,
    staging: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict]]:
    settings = get_settings()
    base     = settings.massive_base_url.rstrip("/")
    api_key  = settings.massive_api_key

    req_sem  = asyncio.Semaphore(concurrency)
    tick_sem = asyncio.Semaphore(ticker_concurrency)
    limits   = httpx.Limits(
        max_connections=concurrency + 10,
        max_keepalive_connections=concurrency,
    )

    parts: list[pd.DataFrame] = [staging] if not staging.empty else []
    new_failures: list[dict]  = []
    tally = {"ok": 0, "empty": 0, "error": 0}
    done  = 0
    since_flush = 0
    total = len(tickers)

    async with httpx.AsyncClient(limits=limits, timeout=30) as client:
        tasks = [
            asyncio.create_task(
                _process_ticker(
                    tk, from_date, to_date, new_dates,
                    client, req_sem, tick_sem, retries, base, api_key,
                )
            )
            for tk in tickers
            if tk not in already_done
        ]

        for fut in asyncio.as_completed(tasks):
            ticker, status, df = await fut
            tally[status] = tally.get(status, 0) + 1
            done += 1

            if status == "error":
                new_failures.append({
                    "ticker":    ticker,
                    "date_strs": sorted(new_dates),
                    "error":     "fetch_failed",
                })
            elif df is not None:
                parts.append(df)
                since_flush += 1

            # Periodic checkpoint to staging so a crash loses at most
            # flush_every tickers.
            if since_flush >= flush_every and parts:
                combined = pd.concat(parts, ignore_index=True)
                _write_staging(combined)
                parts = [combined]
                since_flush = 0
                logger.info("Checkpoint → staging  done=%d/%d  tally=%s",
                            done, total, tally)

            if done % 1000 == 0 or done == total:
                logger.info("Progress %d/%d  %s", done, total, tally)

    final = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=OUTPUT_COLUMNS)
    return final, new_failures


# ---------------------------------------------------------------------------
# Upsert stock_dataset.parquet (vectorized key-based replace)
# ---------------------------------------------------------------------------

def _upsert_stock_dataset(new_df: pd.DataFrame) -> None:
    if new_df.empty:
        logger.info("No new rows to upsert.")
        return

    if not STOCK_DATASET_PATH.exists():
        STOCK_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        new_df.sort_values(["ticker", "date_str"]).reset_index(drop=True).to_parquet(
            STOCK_DATASET_PATH, index=False, compression="zstd"
        )
        logger.info("Created %s (%d rows)", STOCK_DATASET_PATH, len(new_df))
        return

    existing = pd.read_parquet(STOCK_DATASET_PATH)

    # Vectorized key removal — no row-wise apply on 12M rows
    keys = pd.MultiIndex.from_frame(new_df[["ticker", "date_str"]])
    existing_keys = pd.MultiIndex.from_frame(existing[["ticker", "date_str"]])
    existing = existing[~existing_keys.isin(keys)]

    merged = (
        pd.concat([existing, new_df], ignore_index=True)
        .sort_values(["ticker", "date_str"])
        .reset_index(drop=True)
    )
    tmp = STOCK_DATASET_PATH.with_suffix(".parquet.tmp")
    merged.to_parquet(tmp, index=False, compression="zstd")
    os.replace(tmp, STOCK_DATASET_PATH)
    logger.info(
        "Upserted stock_dataset.parquet: +%d rows → %d total  (%.1f MB)",
        len(new_df), len(merged), STOCK_DATASET_PATH.stat().st_size / 1e6,
    )


# ---------------------------------------------------------------------------
# Detect new trading dates by probing SPY
# ---------------------------------------------------------------------------

def _detect_new_dates(since_date_str: str, to_date: date) -> set[str]:
    """
    Returns the set of trading date_str values strictly after since_date_str.
    Uses SPY as a proxy — SPY trades on every US equity market day.
    Falls back to all calendar days if SPY fetch fails.
    """
    try:
        start = (date.fromisoformat(since_date_str) + timedelta(days=1)).isoformat()
        candles = fetch_candles("SPY", start, to_date.isoformat(),
                                timeframe="1d", adjusted=True)
        if candles:
            probe_df = pd.DataFrame(candles)
            dt = (pd.to_datetime(probe_df["time"], unit="s", utc=True)
                    .dt.tz_convert("America/New_York"))
            dates = set(dt.dt.strftime("%Y-%m-%d").tolist())
            logger.info("New trading dates via SPY: %s", sorted(dates))
            return dates
    except Exception as exc:
        logger.warning("SPY probe failed (%s) — falling back to calendar days.", exc)

    d = date.fromisoformat(since_date_str) + timedelta(days=1)
    fallback: set[str] = set()
    while d <= to_date:
        fallback.add(d.isoformat())
        d += timedelta(days=1)
    return fallback


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_incremental_stock_dataset_update(
    concurrency: int = 50,
    ticker_concurrency: int = 20,
    retries: int = 3,
    flush_every: int = 500,
    retry_failures: bool = True,
) -> None:
    """
    Incremental nightly update of stock_dataset.parquet.

    Safe to run multiple times — already-present (ticker, date_str) pairs are
    replaced, not duplicated.  Crashed runs resume from the staging file.
    """
    _t0 = tm.time()
    logger.info("=" * 60)
    logger.info("run_incremental_stock_dataset_update started  %s",
                tm.strftime("%Y-%m-%d %H:%M:%S"))

    settings = get_settings()
    if not settings.massive_api_key:
        logger.error("MASSIVE_API_KEY not set — aborting.")
        return

    # ── 1. Find the date window ──────────────────────────────────────────────
    try:
        latest_date_str = _latest_date()
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return

    to_date = date.today()
    if latest_date_str >= to_date.isoformat():
        logger.info("stock_dataset already up to date (%s). Nothing to do.",
                    latest_date_str)
        return

    new_dates = _detect_new_dates(latest_date_str, to_date)
    if not new_dates:
        logger.info("No new trading dates found after %s.", latest_date_str)
        return

    from_date = date.fromisoformat(latest_date_str) - timedelta(days=_WARMUP_CALENDAR_DAYS)
    logger.info(
        "Window: fetch from %s, keep only %s  (warmup=%d calendar days)",
        from_date.isoformat(), sorted(new_dates), _WARMUP_CALENDAR_DAYS,
    )

    # ── 2. Load universe ─────────────────────────────────────────────────────
    if not UNIVERSE_PATH.exists():
        logger.error("Universe file not found: %s", UNIVERSE_PATH)
        return
    universe_tickers = pd.read_parquet(UNIVERSE_PATH, columns=["ticker"])["ticker"].tolist()
    logger.info("Universe: %d tickers", len(universe_tickers))

    # ── 3. Merge in retry tickers from previous failures ─────────────────────
    prev_failures = _load_failures() if retry_failures else []
    retry_tickers = [f["ticker"] for f in prev_failures]
    if retry_tickers:
        logger.info("Retrying %d tickers from previous run failures.", len(retry_tickers))

    all_tickers = sorted(set(universe_tickers) | set(retry_tickers))
    logger.info("Total tickers to process: %d", len(all_tickers))

    # ── 4. Resume from staging if a previous run crashed ─────────────────────
    staging = _load_staging()
    already_done: set[str] = set()
    if not staging.empty:
        # Any ticker already in staging for these new_dates is considered done
        staging_for_new = staging[staging["date_str"].isin(new_dates)]
        already_done = set(staging_for_new["ticker"].unique())
        if already_done:
            logger.info("Resuming — %d tickers already in staging, skipping them.",
                        len(already_done))

    # ── 5. Async fetch ────────────────────────────────────────────────────────
    combined, new_failures = asyncio.run(
        _run_async(
            all_tickers, already_done,
            from_date, to_date, new_dates,
            concurrency, ticker_concurrency,
            retries, flush_every, staging,
        )
    )

    # ── 6. Merge into stock_dataset.parquet ───────────────────────────────────
    new_rows = combined[combined["date_str"].isin(new_dates)] if not combined.empty else pd.DataFrame()
    _upsert_stock_dataset(new_rows)

    # ── 7. Clean up staging ───────────────────────────────────────────────────
    if STAGING_PATH.exists():
        STAGING_PATH.unlink()
        logger.info("Staging file removed.")

    # ── 8. Save / clear failure record ───────────────────────────────────────
    if new_failures:
        _save_failures(new_failures)
    else:
        if FAILURES_PATH.exists():
            FAILURES_PATH.unlink()
            logger.info("No failures — cleared %s.", FAILURES_PATH)

    elapsed = tm.time() - _t0
    logger.info(
        "run_incremental_stock_dataset_update completed in %.1fs  "
        "new_rows=%d  failures=%d",
        elapsed,
        len(new_rows),
        len(new_failures),
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Incrementally update stock_dataset.parquet with new daily candles."
    )
    parser.add_argument("--concurrency",        type=int, default=50,
                        help="Max concurrent reference (market_cap) requests (default: 50).")
    parser.add_argument("--ticker-concurrency", type=int, default=20,
                        help="Max tickers processed in parallel (default: 20).")
    parser.add_argument("--retries",            type=int, default=3,
                        help="Retries per request on 429/5xx/timeout (default: 3).")
    parser.add_argument("--flush-every",        type=int, default=500,
                        help="Write staging checkpoint every N tickers (default: 500).")
    parser.add_argument("--no-retry-failures",  action="store_true",
                        help="Ignore previous failure record (don't retry).")
    parser.add_argument("--dry-run",            action="store_true",
                        help="Show what would run without fetching anything.")
    args = parser.parse_args()

    if args.dry_run:
        try:
            latest = _latest_date()
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}")
            return
        to_date   = date.today()
        new_dates = _detect_new_dates(latest, to_date)
        print(f"Latest date in stock_dataset : {latest}")
        print(f"New trading dates detected   : {sorted(new_dates) or 'none'}")
        if UNIVERSE_PATH.exists():
            n = len(pd.read_parquet(UNIVERSE_PATH, columns=["ticker"]))
            print(f"Universe tickers             : {n}")
        failures = _load_failures()
        print(f"Pending failures (retry)     : {len(failures)}")
        staging = _load_staging()
        if not staging.empty:
            print(f"Staging rows (from crash)    : {len(staging)}")
        return

    run_incremental_stock_dataset_update(
        concurrency=args.concurrency,
        ticker_concurrency=args.ticker_concurrency,
        retries=args.retries,
        flush_every=args.flush_every,
        retry_failures=not args.no_retry_failures,
    )


if __name__ == "__main__":
    main()
