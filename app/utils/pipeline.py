"""
Data injection pipeline.

Orchestrates:
    1. Ticker sync   — fetch the latest active-stock list from Massive/Polygon.
    2. Chunk gen     — split the 5-year history into monthly date ranges.
    3. Parallel fetch — download 1-min candles (04:00–20:00 ET) for every
                        chunk concurrently, bounded by a shared semaphore.
    4. Aggregation   — once all chunks for a ticker arrive (in batches to cap
                        memory), apply process_data_minutes and emit daily rows.

Usage:
    import asyncio
    from app.utils.pipeline import run_pipeline

    # Blocking call from a script / Celery task:
    results = asyncio.run(run_pipeline())

    # With an incremental storage callback:
    def save(ticker, df):
        df.to_parquet(f"data/{ticker}.parquet")

    asyncio.run(run_pipeline(on_ticker_done=save))
"""
from __future__ import annotations

import asyncio
import json
import logging
import multiprocessing as mp
from collections import defaultdict
from datetime import date, datetime
from typing import Callable
from zoneinfo import ZoneInfo

import httpx
import pandas as pd

import numpy as np

from app.config import get_settings
from app.utils.market_utils import (
    sync_tickers,
    ticker_chunks,
    process_data_minutes,
    sync_data_with_prev_day_close,
)

logger = logging.getLogger(__name__)

# ── Low-level helpers ──────────────────────────────────────────────────────────

_ET = ZoneInfo("America/New_York")


def _et_to_ms(date_str: str, hour: int, minute: int = 0) -> int:
    """Convert a date + local ET hour:minute to UTC milliseconds."""
    d  = date.fromisoformat(date_str)
    dt = datetime(d.year, d.month, d.day, hour, minute, tzinfo=_ET)
    return int(dt.timestamp() * 1000)


# ── Async chunk fetch ──────────────────────────────────────────────────────────

async def _fetch_chunk(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    ticker: str,
    from_date: str,
    to_date: str,
    api_key: str,
    base_url: str,
) -> tuple[list[dict], bool]:
    """
    Fetch one monthly chunk of 1-min candles within the 04:00–20:00 ET window.

    Handles pagination automatically. On any network/API error the failure is
    logged and whatever bars were collected so far are returned (possibly []).

    Returns:
        (bars, had_error) where had_error is True when a connection/server error
        exhausted all retries or an unexpected exception occurred.  A 4xx client
        error (non-retryable) is *not* considered a failure.

    Returns bars in process_data_minutes format: {t (UTC ms), o, h, l, c, v}.
    """
    from_ms = _et_to_ms(from_date, 4)
    to_ms   = _et_to_ms(to_date,  20)

    url: str | None = (
        f"{base_url}/v2/aggs/ticker/{ticker}/range/1/minute"
        f"/{from_ms}/{to_ms}"
        f"?adjusted=false&sort=asc&limit=50000&apiKey={api_key}"
    )

    bars: list[dict] = []
    had_error = False
    _MAX_RETRIES = 4
    _RETRY_STATUSES = {429, 500, 502, 503, 504}

    async with semaphore:
        while url:
            for attempt in range(_MAX_RETRIES):
                safe_url = url.split("&apiKey=")[0]
                try:
                    resp = await client.get(url)
                    resp.raise_for_status()
                    data = resp.json()

                    bars.extend(
                        {"t": b["t"], "o": b["o"], "h": b["h"], "l": b["l"], "c": b["c"], "v": b["v"]}
                        for b in (data.get("results") or [])
                    )

                    next_url = data.get("next_url")
                    url = f"{next_url}&apiKey={api_key}" if next_url else None
                    break  # success — exit retry loop

                except httpx.TimeoutException as exc:
                    wait = 2 ** attempt
                    logger.warning(
                        "TIMEOUT  ticker=%-6s  %s→%s  attempt=%d  retry_in=%ds  url=%s  %s",
                        ticker, from_date, to_date, attempt + 1, wait, safe_url, exc,
                    )
                    if attempt < _MAX_RETRIES - 1:
                        await asyncio.sleep(wait)
                    else:
                        logger.error(
                            "GIVE_UP  ticker=%-6s  %s→%s  all %d attempts failed (timeout)  url=%s",
                            ticker, from_date, to_date, _MAX_RETRIES, safe_url,
                        )
                        had_error = True
                        url = None

                except httpx.HTTPStatusError as exc:
                    status = exc.response.status_code
                    if status in _RETRY_STATUSES:
                        wait = 2 ** attempt
                        logger.warning(
                            "HTTP_%d  ticker=%-6s  %s→%s  attempt=%d  retry_in=%ds  url=%s",
                            status, ticker, from_date, to_date, attempt + 1, wait, safe_url,
                        )
                        if attempt < _MAX_RETRIES - 1:
                            await asyncio.sleep(wait)
                        else:
                            logger.error(
                                "GIVE_UP  ticker=%-6s  %s→%s  all %d attempts failed (HTTP %d)  url=%s",
                                ticker, from_date, to_date, _MAX_RETRIES, status, safe_url,
                            )
                            had_error = True
                            url = None
                    else:
                        # 4xx client errors (except 429) are not retried
                        logger.error(
                            "HTTP_%d  ticker=%-6s  %s→%s  url=%s  %s",
                            status, ticker, from_date, to_date, safe_url, exc,
                        )
                        url = None
                    break

                except Exception as exc:
                    logger.error(
                        "ERROR    ticker=%-6s  %s→%s  %s: %s  url=%s",
                        ticker, from_date, to_date, type(exc).__name__, exc, safe_url,
                    )
                    had_error = True
                    url = None
                    break

    return bars, had_error


# ── Split fetch ────────────────────────────────────────────────────────────────

async def _fetch_split(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    ticker: str,
    api_key: str,
    base_url: str,
) -> list[dict]:
    """
    Fetch all stock-split events for a ticker from the Massive/Polygon API.

    Returns a list of dicts with keys:
        execution_date               (ISO date string, e.g. "2020-08-31")
        historical_adjustment_factor (float = split_from / split_to)

    Returns [] on any error so enrichment can still proceed without adjustments.
    """
    url: str | None = (
        f"{base_url}/v3/reference/splits"
        f"?ticker={ticker}&limit=1000&apiKey={api_key}"
    )
    splits: list[dict] = []
    _MAX_RETRIES    = 4
    _RETRY_STATUSES = {429, 500, 502, 503, 504}

    async with semaphore:
        while url:
            for attempt in range(_MAX_RETRIES):
                try:
                    resp = await client.get(url)
                    resp.raise_for_status()
                    data = resp.json()

                    for item in data.get("results") or []:
                        split_from = item.get("split_from", 1) or 1
                        split_to   = item.get("split_to",   1) or 1
                        splits.append({
                            "execution_date":               item["execution_date"],
                            "historical_adjustment_factor": split_from / split_to,
                        })

                    next_url = data.get("next_url")
                    url = f"{next_url}&apiKey={api_key}" if next_url else None
                    break  # success

                except httpx.TimeoutException as exc:
                    wait = 2 ** attempt
                    logger.warning(
                        "SPLIT_TIMEOUT  ticker=%-6s  attempt=%d  retry_in=%ds  %s",
                        ticker, attempt + 1, wait, exc,
                    )
                    if attempt < _MAX_RETRIES - 1:
                        await asyncio.sleep(wait)
                    else:
                        logger.error("SPLIT_GIVE_UP  ticker=%-6s  all %d attempts failed", ticker, _MAX_RETRIES)
                        url = None

                except httpx.HTTPStatusError as exc:
                    status = exc.response.status_code
                    if status in _RETRY_STATUSES:
                        wait = 2 ** attempt
                        logger.warning(
                            "SPLIT_HTTP_%d  ticker=%-6s  attempt=%d  retry_in=%ds",
                            status, ticker, attempt + 1, wait,
                        )
                        if attempt < _MAX_RETRIES - 1:
                            await asyncio.sleep(wait)
                        else:
                            logger.error("SPLIT_GIVE_UP  ticker=%-6s  HTTP %d", ticker, status)
                            url = None
                    else:
                        logger.error("SPLIT_HTTP_%d  ticker=%-6s  %s", status, ticker, exc)
                        url = None
                    break

                except Exception as exc:
                    logger.error("SPLIT_ERROR  ticker=%-6s  %s: %s", ticker, type(exc).__name__, exc)
                    url = None
                    break

    return splits


# ── State persistence ──────────────────────────────────────────────────────────

_PIPELINE_STATE_FILE = "pipeline_state.json"


def _load_failed_tickers(state_file: str) -> set[str]:
    """Load the set of tickers that failed in the previous pipeline run.

    Returns an empty set if the file does not exist or cannot be parsed.
    Failures are connection errors / unexpected exceptions — not missing data.
    """
    try:
        with open(state_file) as f:
            data = json.load(f)
        failed: set[str] = set(data.get("failed_tickers", []))
        if failed:
            logger.info("state: %d failed tickers from previous run", len(failed))
        return failed
    except FileNotFoundError:
        return set()
    except Exception as exc:
        logger.warning("state: could not load %s — %s", state_file, exc)
        return set()


def _save_state(state_file: str, failed_tickers: set[str]) -> None:
    """Persist the set of failed tickers to a JSON file for the next run."""
    try:
        data = {
            "failed_tickers": sorted(failed_tickers),
            "count":          len(failed_tickers),
            "timestamp":      datetime.now(_ET).isoformat(),
        }
        with open(state_file, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(
            "state: saved %d failed tickers to %s",
            len(failed_tickers), state_file,
        )
    except Exception as exc:
        logger.warning("state: could not save %s — %s", state_file, exc)


# ── Storage schema — final column order written to the DB / parquet ───────────

_OUTPUT_COLS: list[str] = [
    "ticker", "date_str",
    "gap", "gap_perc", "daily_range", "day_range_perc",
    "previous_close",
    "open", "high", "low", "close", "volume",
    "premarket_volume", "market_hours_volume",
    "high_pm", "low_pm", "pm_open", "highest_in_pm", "high_pm_time",
    "high_mh",
    "ah_open", "ah_close", "ah_high", "ah_low", "ah_range", "ah_range_perc", "ah_volume",
    "market_cap", "stock_float", "daily_200_sma",
    "split_date_str", "split_adjust_factor",
    "time",
]


# ── Post-process enrichment ────────────────────────────────────────────────────

def _enrich(df: pd.DataFrame, splits: list[dict]) -> pd.DataFrame:
    """
    Enrich a daily-summary DataFrame after process_data_minutes.

    Steps:
        1. Compute ``previous_close``:
             - For consecutive trading days (gap ≤ 3 calendar days) use the
               previous row's close.
             - For the first row or after a gap > 3 days, fall back to open.
        2. Call sync_data_with_prev_day_close with the pre-fetched splits to
           set split_adjust_factor and override previous_close on split dates.
        3. Derive gap, gap_perc, daily_range, day_range_perc.
        4. Add placeholder columns: market_cap, stock_float, daily_200_sma.

    The DataFrame is modified in-place and also returned.
    """
    if df.empty:
        return df

    # ── 1. previous_close ─────────────────────────────────────────────────────
    dates         = pd.to_datetime(df["date_str"])
    prev_dates    = dates.shift(1)
    day_gap       = (dates - prev_dates).dt.days   # NaN for first row

    prev_close = df["close"].shift(1)
    # Fallback: first bar of the session for that day.
    # pm_open is the 4am-ish candle; fall back to the 9:30 open when absent.
    session_open = df["pm_open"].where(
        df["pm_open"].notna() & (df["pm_open"] > 0),
        df["open"],
    )
    # Use previous close only when rows are consecutive (≤ 3 calendar days)
    use_prev = day_gap.notna() & (day_gap <= 3)
    df["previous_close"] = np.where(use_prev, prev_close, session_open)

    # ── 2. split adjustment + override previous_close on split day ────────────
    df = sync_data_with_prev_day_close(df, fetch_split=lambda _: splits)

    # ── 3. derived columns ────────────────────────────────────────────────────
    prev = df["previous_close"]

    df["gap"]           = (df["open"] - prev).round(3)
    df["gap_perc"]      = np.where(
        prev > 0,
        ((df["open"] - prev) / prev * 100).round(3),
        0.0,
    )
    df["daily_range"]   = (df["high"] - df["low"]).round(3)
    df["day_range_perc"] = np.where(
        prev > 0,
        ((df["high"] - df["low"]) / prev * 100).round(3),
        0.0,
    )

    # ── 4. placeholder columns ────────────────────────────────────────────────
    df["market_cap"]    = -1
    df["stock_float"]   = -1
    df["daily_200_sma"] = -1.0

    # ── 5. final cleanup: drop internals, fill remaining NaN, select & order ──
    df = df.drop(columns=["day"], errors="ignore")
    df = df.fillna(-1)

    # Select & reorder to the storage schema; tolerate missing columns so that
    # unit tests with minimal DataFrames still work.
    present = [c for c in _OUTPUT_COLS if c in df.columns]
    df = df[present]

    # Remove rows with no price data (both open and close are 0)
    df = df[~((df["open"] == 0) & (df["close"] == 0))].reset_index(drop=True)

    return df


# ── Per-ticker aggregation ─────────────────────────────────────────────────────

async def _process_ticker(
    ticker: str,
    date_ranges: list[tuple[str, str]],
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    api_key: str,
    base_url: str,
    batch_months: int,
    on_ticker_done: Callable[[str, pd.DataFrame], None] | None,
) -> tuple[str, pd.DataFrame | None, bool]:
    """
    Fetch all monthly chunks for a single ticker in memory-bounded batches,
    apply process_data_minutes to each batch, and concatenate the daily rows.

    Batches always align to full calendar months so no trading day is ever
    split across two process_data_minutes calls.

    Returns:
        (ticker, df, had_error) where had_error is True when any chunk suffered
        a connection/server error or an unexpected exception occurred.  Missing
        data (ticker has no history) is *not* considered an error.
    """
    daily_frames: list[pd.DataFrame] = []
    had_error = False

    try:
        for i in range(0, len(date_ranges), batch_months):
            batch = date_ranges[i : i + batch_months]

            # Fetch all chunks in this batch concurrently
            chunk_results: list[tuple[list[dict], bool]] = await asyncio.gather(*[
                _fetch_chunk(client, semaphore, ticker, fd, td, api_key, base_url)
                for fd, td in batch
            ])

            # Track errors and flatten bars
            candles: list[dict] = []
            for bars, chunk_had_error in chunk_results:
                if chunk_had_error:
                    had_error = True
                candles.extend(bars)

            # Sort chronologically
            candles.sort(key=lambda b: b["t"])

            if candles:
                # Run CPU-heavy pandas work in a thread so the event loop stays
                # free to handle incoming HTTP responses from other tickers.
                daily = await asyncio.to_thread(process_data_minutes, candles)
                if daily is not None and not daily.empty:
                    daily_frames.append(daily)

            # Release batch memory before the next iteration
            del candles, chunk_results

        if not daily_frames:
            logger.warning("no data  ticker=%s", ticker)
            return ticker, None, had_error

        result = (
            pd.concat(daily_frames, ignore_index=True)
            .sort_values("date_str")
            .reset_index(drop=True)
        )
        result.insert(0, "ticker", ticker)

        # ── Enrichment: splits + derived columns ──────────────────────────────────
        splits = await _fetch_split(client, semaphore, ticker, api_key, base_url)
        result = await asyncio.to_thread(_enrich, result, splits)

        if on_ticker_done is not None:
            try:
                on_ticker_done(ticker, result)
            except Exception as exc:
                logger.error("on_ticker_done failed for %s: %s", ticker, exc)

        logger.info("done  ticker=%-6s  days=%d", ticker, len(result))
        return ticker, result, had_error

    except Exception as exc:
        logger.error("FATAL_ERROR  ticker=%-6s  %s: %s", ticker, type(exc).__name__, exc)
        return ticker, None, True


# ── Pipeline entry point ───────────────────────────────────────────────────────

async def run_pipeline(
    tickers_df: pd.DataFrame | None = None,
    months_back: int = 59,
    max_concurrent: int = 50,
    batch_months: int = 6,
    on_ticker_done: Callable[[str, pd.DataFrame], None] | None = None,
) -> tuple[dict[str, pd.DataFrame], set[str]]:
    """
    Full data injection pipeline.

    Steps:
        1. **Ticker sync** — call sync_tickers() to update the local CSV, or
           use the provided DataFrame directly.
        2. **Chunk generation** — call ticker_chunks() to split the
           ``months_back``-month history into monthly (ticker, from, to) tuples.
        3. **Parallel fetch** — dispatch coroutines for all tickers at once.
           A shared asyncio.Semaphore caps simultaneous HTTP requests at
           ``max_concurrent``, maximising throughput without overwhelming the
           API.  True async I/O (httpx.AsyncClient) avoids thread-pool limits.
        4. **Batch aggregation** — each ticker's chunks are processed in
           groups of ``batch_months`` full calendar months.  process_data_minutes
           is called per batch so memory stays bounded.  Batches never split a
           trading day.

    Args:
        tickers_df:     Ticker list DataFrame (columns: ticker, company_name,
                        stock_market).  If None, sync_tickers() is called first
                        to fetch and persist the latest active-stock list.
        months_back:    History window in months (default 59 ≈ 4y 11m).
        max_concurrent: Maximum simultaneous HTTP requests.  Tune this to stay
                        within Massive/Polygon rate limits (default 50).
        batch_months:   Months of 1-min data processed at a time per ticker.
                        Lower values reduce peak memory usage; higher values
                        allow more intra-ticker concurrency (default 6).
        on_ticker_done: Optional callback(ticker: str, df: pd.DataFrame) called
                        as each ticker completes.  Use this for incremental
                        storage (e.g. write to DB or parquet) so the full
                        result dict does not need to live in RAM simultaneously.

    Returns:
        (results, failed) where results maps ticker → daily summary DataFrame
        and failed is the set of tickers that suffered connection/server errors.
    """
    # Step 1 — ticker list
    if tickers_df is None:
        logger.info("syncing ticker list...")
        tickers_df = sync_tickers()
    logger.info("tickers=%d", len(tickers_df))

    # Step 2 — chunk generation
    all_chunks = ticker_chunks(tickers_df, months_back=months_back)
    logger.info("total chunks=%d", len(all_chunks))

    by_ticker: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for ticker, fd, td in all_chunks:
        by_ticker[ticker].append((fd, td))

    # Steps 3+4 — concurrent fetch + aggregation
    settings  = get_settings()
    semaphore = asyncio.Semaphore(max_concurrent)

    async with httpx.AsyncClient(timeout=httpx.Timeout(30, connect=10)) as client:
        tasks = [
            _process_ticker(
                ticker, date_ranges,
                client, semaphore,
                settings.massive_api_key,
                settings.massive_base_url,
                batch_months,
                on_ticker_done,
            )
            for ticker, date_ranges in by_ticker.items()
        ]
        logger.info("dispatching %d ticker tasks...", len(tasks))
        all_results = await asyncio.gather(*tasks)

    results = {t: df for t, df, _  in all_results if df is not None}
    failed  = {t for t, df, err in all_results if err}
    logger.info(
        "pipeline complete: %d/%d tickers produced data, %d failed",
        len(results), len(by_ticker), len(failed),
    )
    return results, failed


# ── Database injection ─────────────────────────────────────────────────────────

async def save_to_db(
    results: dict[str, pd.DataFrame],
    postgrest_url: str | None = None,
    token: str | None = None,
    batch_size: int = 5_000,
) -> None:
    """
    Push pipeline output to PostgreSQL via the PostgREST ``upsert_stock_data`` RPC.

    Converts each DataFrame in *results* to a list of JSON records and POSTs
    them to ``POST /rpc/upsert_stock_data`` in batches of *batch_size* rows to
    keep request sizes manageable.  Rows that already exist in the DB are
    updated (INSERT … ON CONFLICT DO UPDATE).

    Args:
        results:        Dict returned by run_pipeline / run_pipeline_multiprocess.
        postgrest_url:  Base URL of the PostgREST service (e.g. "http://localhost:3031").
                        If None, read from settings.postgrest_url.
        token:          JWT token for the ``web_admin`` role.
                        If None, read from settings.postgrest_token.
        batch_size:     Maximum number of rows per HTTP request (default 5 000).
    """
    settings = get_settings()
    url   = (postgrest_url or settings.postgrest_url).rstrip("/") + "/rpc/upsert_stock_data"
    token = token or settings.postgrest_token

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type":  "application/json",
        "Prefer":        "return=minimal",
    }

    # Columns that must be integer in the DB (BIGINT).
    # Pandas stores them as float64 after fillna(-1) because NaN requires float dtype.
    # PostgreSQL's jsonb_to_recordset rejects "175326.0" for a bigint field,
    # so we cast them back to int64 before serialisation.
    _BIGINT_COLS = frozenset({
        "volume", "premarket_volume", "market_hours_volume",
        "ah_volume", "high_pm_time", "time",
    })

    # Flatten all DataFrames into a single list of records
    records: list[dict] = []
    for df in results.values():
        if df is not None and not df.empty:
            df = df.copy()
            for col in _BIGINT_COLS:
                if col in df.columns:
                    df[col] = df[col].astype("int64")
            records.extend(df.to_dict(orient="records"))

    if not records:
        logger.warning("save_to_db: no records to insert")
        return

    total   = len(records)
    batches = (total + batch_size - 1) // batch_size
    logger.info("save_to_db: %d records across %d batch(es) → %s", total, batches, url)

    async with httpx.AsyncClient(timeout=httpx.Timeout(60, connect=10)) as client:
        for i in range(0, total, batch_size):
            batch  = records[i : i + batch_size]
            batch_n = i // batch_size + 1
            try:
                resp = await client.post(url, headers=headers, json={"p_data": batch})
                resp.raise_for_status()
                logger.info(
                    "save_to_db: batch %d/%d  rows=%d  status=%d",
                    batch_n, batches, len(batch), resp.status_code,
                )
            except httpx.HTTPStatusError as exc:
                logger.error(
                    "save_to_db: batch %d/%d HTTP %d — %s",
                    batch_n, batches, exc.response.status_code, exc.response.text,
                )
                raise
            except Exception as exc:
                logger.error(
                    "save_to_db: batch %d/%d failed — %s: %s",
                    batch_n, batches, type(exc).__name__, exc,
                )
                raise


# ── Sample run (50 random tickers) ────────────────────────────────────────────

def _print_ticker(ticker: str, df: pd.DataFrame) -> None:
    """Print one ticker's daily summary to stdout."""
    print(f"\n{'─'*60}")
    print(f"  {ticker}  ({len(df)} trading days)")
    print(f"{'─'*60}")
    with pd.option_context(
        "display.max_columns", None,
        "display.width", 120,
        "display.float_format", "{:.3f}".format,
    ):
        print(df.to_string(index=False))


def run_sample(
    n: int = 50,
    num_processes: int = 8,
    max_concurrent: int = 100,
    save_db: bool = False,
) -> None:
    """
    Run the multiprocess pipeline on ``n`` randomly selected tickers and
    print each ticker's daily summary to stdout.

    Args:
        n:               Number of random tickers to sample (default 50).
        num_processes:   Subprocess count (default 8).
        max_concurrent:  HTTP requests per subprocess (default 100).
        save_db:         If True, inject results into the database via
                         save_to_db() after the pipeline finishes (default False).

    Usage:
        from app.utils.pipeline import run_sample
        run_sample()
        run_sample(n=10, num_processes=4, save_db=True)
    """
    import time
    from app.utils.market_utils import _TICKERS_CSV

    tickers_df = pd.read_csv(_TICKERS_CSV).sample(n=n, random_state=None)

    print(f"\n{'='*60}")
    print(f"Sample run — {n} tickers  |  {num_processes} processes × {max_concurrent} concurrent")
    print(f"{'='*60}\n")

    t0 = time.perf_counter()
    results = run_pipeline_multiprocess(
        tickers_df=tickers_df,
        num_processes=min(num_processes, n),   # no more processes than tickers
        max_concurrent=max_concurrent,
        state_file=None,   # no state tracking for random samples
    )
    elapsed = time.perf_counter() - t0

    for ticker, df in sorted(results.items()):
        _print_ticker(ticker, df)

    print(f"\n{'='*60}")
    print(f"Done — {len(results)}/{n} tickers produced data in {elapsed:.1f}s")
    print(f"{'='*60}\n")

    if save_db:
        print("Injecting into database…")
        asyncio.run(save_to_db(results))
        print("Database injection complete.")


# ── Multiprocess pipeline ──────────────────────────────────────────────────────
# _mp_worker must be defined at module level so multiprocessing can pickle it.

def _mp_worker(
    worker_id: int,
    ticker_subset: pd.DataFrame,
    months_back: int,
    max_concurrent: int,
    batch_months: int,
) -> tuple[dict[str, pd.DataFrame], set[str]]:
    """
    Entry point for each subprocess.  Runs its own asyncio event loop with its
    own httpx.AsyncClient and semaphore, fully independent of all other workers.
    """
    logging.basicConfig(
        level=logging.INFO,
        format=f"[W{worker_id}] %(levelname)s %(message)s",
    )
    return asyncio.run(
        run_pipeline(
            tickers_df=ticker_subset,
            months_back=months_back,
            max_concurrent=max_concurrent,
            batch_months=batch_months,
        )
    )


def run_pipeline_multiprocess(
    tickers_df: pd.DataFrame | None = None,
    num_processes: int = 8,
    months_back: int = 59,
    max_concurrent: int = 100,
    batch_months: int = 6,
    state_file: str | None = _PIPELINE_STATE_FILE,
) -> dict[str, pd.DataFrame]:
    """
    Run the pipeline across ``num_processes`` independent subprocesses.

    Each subprocess owns its own asyncio event loop and HTTP semaphore, so
    total concurrent HTTP requests = ``num_processes × max_concurrent``
    (e.g. 8 × 100 = 800).  Tickers are distributed round-robin so every
    subprocess handles a roughly equal share.

    This mirrors the architecture of the original multiprocessing pipeline:
    multiprocessing for CPU/GIL parallelism, asyncio inside each process for
    I/O concurrency.

    Args:
        tickers_df:    Ticker list DataFrame.  If None, sync_tickers() is
                       called in the main process before spawning workers.
        num_processes: Number of subprocesses (default 8, one per CPU core).
        months_back:   History window in months (default 59 = 4y 11m).
        max_concurrent: Concurrent HTTP requests per subprocess (default 100).
        batch_months:  Months of 1-min data processed per batch per ticker
                       (default 6).

    Returns:
        Merged dict mapping ticker → daily summary DataFrame.

    Usage:
        from app.utils.pipeline import run_pipeline_multiprocess

        results = run_pipeline_multiprocess(num_processes=8)

        # With storage: use on_ticker_done inside each worker by subclassing,
        # or post-process the returned dict:
        for ticker, df in results.items():
            df.to_parquet(f"data/{ticker}.parquet")
    """
    import time

    if tickers_df is None:
        logging.info("syncing ticker list before spawning workers...")
        tickers_df = sync_tickers()

    tickers = tickers_df["ticker"].dropna().unique().tolist()

    # Load previously-failed tickers and move them to the front of the queue
    if state_file is not None:
        prev_failed = _load_failed_tickers(state_file)
        if prev_failed:
            failed_first = [t for t in tickers if t in prev_failed]
            rest         = [t for t in tickers if t not in prev_failed]
            tickers      = failed_first + rest
            logging.info(
                "state: prioritizing %d previously-failed tickers",
                len(failed_first),
            )

    logging.info(
        "run_pipeline_multiprocess: %d tickers → %d processes × %d concurrent",
        len(tickers), num_processes, max_concurrent,
    )

    # Partition tickers round-robin across workers (same strategy as old pipeline)
    partitions: list[list[str]] = [[] for _ in range(num_processes)]
    for i, t in enumerate(tickers):
        partitions[i % num_processes].append(t)

    subsets = [
        tickers_df[tickers_df["ticker"].isin(part)].copy()
        for part in partitions
        if part
    ]

    worker_args = [
        (wid, subset, months_back, max_concurrent, batch_months)
        for wid, subset in enumerate(subsets, start=1)
    ]

    t0 = time.perf_counter()

    # "spawn" is safer than "fork" when mixing asyncio + multiprocessing
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(subsets)) as pool:
        worker_results: list[tuple[dict, set[str]]] = pool.starmap(_mp_worker, worker_args)

    elapsed = time.perf_counter() - t0

    merged: dict[str, pd.DataFrame] = {}
    all_failed: set[str] = set()
    for r, failed_set in worker_results:
        merged.update(r)
        all_failed.update(failed_set)

    if state_file is not None:
        _save_state(state_file, all_failed)

    logging.info(
        "multiprocess pipeline done: %d tickers in %.1fs  (%d failed)",
        len(merged), elapsed, len(all_failed),
    )
    return merged


if __name__ == "__main__":
    run_sample()
