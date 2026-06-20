"""
Build full_dataset_temp.parquet for 5m (and optionally 15m) candles covering
ticker-days that exist in stock_data_missing_from_full.parquet but are absent
from backtest_dataset/full/{tf}/full_dataset.parquet.

Logic mirrors update_full_dataset.py exactly but reads from the local parquet
instead of PostgREST, and writes to full_dataset_temp.parquet to avoid
touching the existing dataset.

Supports resuming: tickers already written to the temp file are skipped.

Usage (from backtester_api/):
    python -m scripts.build_missing_dataset              # 5m only, 8 workers
    python -m scripts.build_missing_dataset --tf 15m
    python -m scripts.build_missing_dataset --tf both
    python -m scripts.build_missing_dataset --workers 16
    python -m scripts.build_missing_dataset --dry-run
    python -m scripts.build_missing_dataset --flush      # overwrite temp file
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.abspath("."))

from app.utils.indicators import (
    compute_atr,
    compute_donchian,
    compute_rvol,
    compute_sma,
    compute_vwap,
)
from app.utils.massive import fetch_candles

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MISSING_PATH  = Path("backtest_dataset/STOCKS/stock_data_missing_from_full.parquet")
OUTPUT_BASE   = Path("backtest_dataset/full")
FAILURES_PATH = Path("logs/build_missing_failures.json")

_LOOKBACK_DAYS: dict[str, int] = {
    "5m":  15,
    "15m": 30,
}

COLUMNS = [
    "ticker", "date", "date_str",
    "open", "high", "low", "close", "volume",
    "atr", "RVOL_daily", "SMA_VOLUME_20_5m", "vwap", "previous_day_close",
    "sma_9", "sma_200",
    "donchian_upper", "donchian_lower", "donchian_basis",
]

CHECKPOINT_EVERY = 50
DEFAULT_WORKERS  = 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _temp_path(timeframe: str) -> Path:
    return OUTPUT_BASE / timeframe / "full_dataset_temp.parquet"


def _already_done(timeframe: str) -> set[tuple[str, str]]:
    path = _temp_path(timeframe)
    if not path.exists():
        return set()
    df = pd.read_parquet(path, columns=["ticker", "date_str"])
    return set(zip(df["ticker"], df["date_str"]))


def _build_ticker_df(
    ticker: str,
    from_date: date,
    to_date: date,
    daily_lookup: dict[tuple[str, str], float],
    timeframe: str,
) -> pd.DataFrame | None:
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


def _flush(parts: list[pd.DataFrame], timeframe: str, lock: threading.Lock) -> None:
    if not parts:
        return
    new_df = pd.concat(parts, ignore_index=True).sort_values(["ticker", "date"])
    path   = _temp_path(timeframe)
    path.parent.mkdir(parents=True, exist_ok=True)

    with lock:
        if path.exists():
            existing = pd.read_parquet(path)
            key      = set(zip(new_df["ticker"], new_df["date_str"]))
            existing = existing[
                ~existing.apply(lambda r: (r["ticker"], r["date_str"]) in key, axis=1)
            ]
            new_df = pd.concat([existing, new_df], ignore_index=True).sort_values(["ticker", "date"])

        pq.write_table(
            pa.Table.from_pandas(new_df, preserve_index=False),
            path,
            compression="zstd",
        )
    logger.info(
        "Checkpoint → %s  total_rows=%d  (%.1f MB)",
        path, len(new_df), path.stat().st_size / 1e6,
    )


def _fmt_duration(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s   = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    timeframes: list[str],
    dry_run: bool = False,
    flush: bool = False,
    workers: int = DEFAULT_WORKERS,
) -> None:
    if not MISSING_PATH.exists():
        raise FileNotFoundError(f"Missing-from-full file not found: {MISSING_PATH}")

    missing_df = pd.read_parquet(MISSING_PATH)
    logger.info(
        "Loaded %s: %d rows, %d tickers, dates %s → %s",
        MISSING_PATH, len(missing_df),
        missing_df["ticker"].nunique(),
        missing_df["date_str"].min(),
        missing_df["date_str"].max(),
    )

    if "previous_close" not in missing_df.columns and "stock_float" in missing_df.columns:
        raise ValueError("Cannot find 'previous_close' column in missing file.")

    tickers = sorted(missing_df["ticker"].unique())
    logger.info("Total tickers to process: %d  |  workers: %d", len(tickers), workers)

    if dry_run:
        print(f"[DRY RUN] Would process {len(tickers)} tickers across timeframes: {timeframes}")
        print(missing_df[["ticker", "date_str"]].drop_duplicates().head(20).to_string(index=False))
        return

    if flush:
        for tf in timeframes:
            p = _temp_path(tf)
            if p.exists():
                p.unlink()
                logger.info("Flushed existing temp file: %s", p)

    total_start = time.perf_counter()

    for tf in timeframes:
        logger.info("=== Timeframe: %s  (workers=%d) ===", tf, workers)
        tf_start = time.perf_counter()

        done_pairs = _already_done(tf)
        logger.info("Already done pairs in temp file: %d", len(done_pairs))

        # Build work list: only tickers with remaining dates
        work = []
        for ticker in tickers:
            ticker_rows    = missing_df[missing_df["ticker"] == ticker]
            target_dates   = set(ticker_rows["date_str"])
            remaining      = {d for d in target_dates if (ticker, d) not in done_pairs}
            if remaining:
                work.append((ticker, ticker_rows, remaining))

        total_work = len(work)
        logger.info("Tickers to fetch: %d  (skipping %d already done)", total_work, len(tickers) - total_work)

        if not total_work:
            logger.info("[%s] Nothing to do.", tf)
            continue

        # Shared state (thread-safe)
        parts_buf: list[pd.DataFrame] = []
        failures:  list[dict]          = []
        buf_lock   = threading.Lock()
        file_lock  = threading.Lock()
        counter    = {"processed": 0, "skipped_empty": 0}
        start_time = time.perf_counter()

        def process_ticker(item: tuple) -> None:
            ticker, ticker_rows, remaining_dates = item

            min_date   = date.fromisoformat(min(remaining_dates))
            max_date   = date.fromisoformat(max(remaining_dates))
            lookback   = _LOOKBACK_DAYS[tf]
            fetch_from = min_date - timedelta(days=lookback)

            daily_lookup: dict[tuple[str, str], float] = {
                (ticker, row["date_str"]): row["previous_close"]
                for _, row in ticker_rows[ticker_rows["date_str"].isin(remaining_dates)].iterrows()
            }

            try:
                df_tf = _build_ticker_df(ticker, fetch_from, max_date, daily_lookup, tf)
            except Exception as exc:
                logger.warning("FAIL fetch  ticker=%-8s  tf=%s  %s", ticker, tf, exc)
                with buf_lock:
                    failures.append({"ticker": ticker, "tf": tf, "error": str(exc)})
                return

            if df_tf is None or df_tf.empty:
                with buf_lock:
                    counter["skipped_empty"] += 1
                return

            df_tf = df_tf[df_tf["date_str"].isin(remaining_dates)]
            if df_tf.empty:
                with buf_lock:
                    counter["skipped_empty"] += 1
                return

            flush_now = False
            with buf_lock:
                parts_buf.append(df_tf)
                counter["processed"] += 1
                n = counter["processed"]
                if n % CHECKPOINT_EVERY == 0:
                    flush_now = True
                    snapshot  = parts_buf.copy()
                    parts_buf.clear()

            if flush_now:
                _flush(snapshot, tf, file_lock)
                elapsed = time.perf_counter() - start_time
                rate    = n / elapsed if elapsed > 0 else 0
                eta     = (total_work - n) / rate if rate > 0 else 0
                logger.info(
                    "[%s] %d / %d  |  %.1f t/s  |  elapsed %s  |  ETA %s",
                    tf, n, total_work,
                    rate,
                    _fmt_duration(elapsed),
                    _fmt_duration(eta),
                )

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(process_ticker, item): item[0] for item in work}
            done_count = 0
            for fut in as_completed(futures):
                done_count += 1
                exc = fut.exception()
                if exc:
                    logger.error("Unhandled error for %s: %s", futures[fut], exc)

        # Final flush
        with buf_lock:
            remaining_parts = parts_buf.copy()
            parts_buf.clear()
        _flush(remaining_parts, tf, file_lock)

        tf_elapsed = time.perf_counter() - tf_start
        final_path = _temp_path(tf)
        total_rows = len(pd.read_parquet(final_path, columns=["ticker"])) if final_path.exists() else 0
        logger.info(
            "[%s] Done. processed=%d  skipped_empty=%d  failures=%d  rows=%d  time=%s",
            tf,
            counter["processed"],
            counter["skipped_empty"],
            len(failures),
            total_rows,
            _fmt_duration(tf_elapsed),
        )

        if failures:
            import json
            FAILURES_PATH.parent.mkdir(parents=True, exist_ok=True)
            fail_path = FAILURES_PATH.with_stem(f"build_missing_failures_{tf}")
            with open(fail_path, "w") as f:
                json.dump(failures, f, indent=2)
            logger.warning("Failures saved to %s (%d entries)", fail_path, len(failures))

    total_elapsed = time.perf_counter() - total_start
    logger.info("All timeframes complete. Total time: %s", _fmt_duration(total_elapsed))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build full_dataset_temp.parquet for missing ticker-days."
    )
    parser.add_argument(
        "--tf",
        default="5m",
        choices=["5m", "15m", "both"],
        help="Timeframe to process (default: 5m). Use 'both' for 5m+15m.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Parallel HTTP workers (default: {DEFAULT_WORKERS}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be processed without fetching or writing anything.",
    )
    parser.add_argument(
        "--flush",
        action="store_true",
        help="Delete existing temp file before starting (disables resume).",
    )
    args = parser.parse_args()

    tfs = ["5m", "15m"] if args.tf == "both" else [args.tf]
    run(timeframes=tfs, dry_run=args.dry_run, flush=args.flush, workers=args.workers)
