"""
Steps 2-4 — Build the daily dataset for the whole stock universe (sharded).

For every ticker in the universe file (see build_stock_universe.py):

    Step 2  Fetch ~N years of 1D candles (adjusted=true).             1 request/ticker
    Step 3  Compute SMA 9/20/50/100/200 and ATR 14.                   local
    Step 4  Fetch market_cap + float per trading day from
            /v3/reference/tickers/<ticker>?date=YYYY-MM-DD            ~252 req/ticker/year

Output is ONE parquet per shard:

    backtest_dataset/STOCKS/shards/shard_<I>_of_<N>.parquet

with columns:
    ticker, date, date_str, open, high, low, close, volume,
    sma_9, sma_20, sma_50, sma_100, sma_200, atr_14,
    market_cap, float, shares_outstanding

After all shards finish, merge them into a single file with:
    python -m scripts.merge_stock_dataset

Step 4 is the bottleneck (>1M requests across the universe). It is mitigated by:
  * Only requesting actual trading days (taken from the candles, not the calendar).
  * Async concurrency with a global request semaphore (--concurrency).
  * Optional sampling: --marketcap-step N hits the reference endpoint every N
    trading days, forward-fills the share counts, and recomputes
    market_cap = close * float for the in-between days (shares change rarely,
    so this is accurate while cutting requests by ~N×).
  * Resume: the shard parquet is rewritten every --flush-every tickers; on
    restart the tickers it already contains are skipped, so the job can be
    killed/restarted freely.

Run it in parallel by sharding the universe across processes/terminals:

    # terminal 1                  # terminal 2              # ... terminal 8
    --num-shards 8 --shard 0      --num-shards 8 --shard 1  ...  --shard 7

Usage (from backtester_api/):
    python -m scripts.build_stock_dataset                       # 1 shard, 5y
    python -m scripts.build_stock_dataset --num-shards 8 --shard 0
    python -m scripts.build_stock_dataset --skip-marketcap      # fast first pass
    python -m scripts.build_stock_dataset --marketcap-step 5    # sample weekly
    python -m scripts.build_stock_dataset --tickers AAPL,MSFT --no-resume
    python -m scripts.build_stock_dataset --limit 20            # smoke test
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import random
import sys
from datetime import date
from pathlib import Path

import httpx
import pandas as pd

sys.path.insert(0, os.path.abspath("."))

from app.config import get_settings
from app.utils.indicators import compute_atr, compute_sma
from app.utils.massive import fetch_candles

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

UNIVERSE_PATH = Path("backtest_dataset/UNIVERSE/stock_universe.parquet")
SHARDS_DIR    = Path("backtest_dataset/STOCKS/shards")
DEFAULT_YEARS = 5

OUTPUT_COLUMNS = [
    "ticker", "date", "date_str",
    "open", "high", "low", "close", "volume",
    "sma_9", "sma_20", "sma_50", "sma_100", "sma_200", "atr_14",
    "market_cap", "float", "shares_outstanding",
]

# Transient HTTP statuses worth retrying.
_RETRY_STATUS = {429, 500, 502, 503, 504}


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def _default_from(years: int) -> date:
    today = date.today()
    try:
        return today.replace(year=today.year - years)
    except ValueError:  # Feb 29
        return today.replace(year=today.year - years, day=28)


# ---------------------------------------------------------------------------
# Step 2 + 3 — candles and indicators (sync; 1 request per ticker)
# ---------------------------------------------------------------------------

def _build_candles_df(ticker: str, from_date: date, to_date: date) -> pd.DataFrame:
    """Fetch adjusted 1D candles and attach SMA/ATR indicators."""
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
# Step 4 — market cap + float per trading day (async)
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
    """
    Fetch /v3/reference/tickers/<ticker>?date=<date_str> with retry/backoff.

    Returns (date_str, results_dict_or_None). A None result means "no data"
    (404 / before list date / exhausted retries) and is handled by fill-forward.
    """
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
            return date_str, None  # 404 and other non-retryable -> missing
    return date_str, None


async def _fetch_marketcaps(
    client: httpx.AsyncClient,
    ticker: str,
    date_strs: list[str],
    sem: asyncio.Semaphore,
    retries: int,
) -> pd.DataFrame:
    """
    Fetch reference data for the given trading days and return a DataFrame:
        date_str, market_cap, weighted_shares_outstanding, share_class_shares_outstanding
    """
    settings = get_settings()
    base = settings.massive_base_url.rstrip("/")
    api_key = settings.massive_api_key

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
            "date_str": ds,
            "market_cap": res.get("market_cap"),
            "weighted_shares_outstanding": res.get("weighted_shares_outstanding"),
            "share_class_shares_outstanding": res.get("share_class_shares_outstanding"),
        })

    if not rows:
        return pd.DataFrame(
            columns=["date_str", "market_cap",
                     "weighted_shares_outstanding", "share_class_shares_outstanding"]
        )
    return pd.DataFrame(rows)


def _attach_marketcap(df: pd.DataFrame, ref: pd.DataFrame, step: int) -> pd.DataFrame:
    """
    Merge reference data onto the candle DataFrame and derive the final
    market_cap / float / shares_outstanding columns.

    step == 1  -> market_cap taken straight from the endpoint (ffill gaps).
    step  > 1  -> share counts forward-filled, market_cap = close * float.
    """
    df = df.merge(ref, on="date_str", how="left")

    df["float"] = df["weighted_shares_outstanding"].ffill().bfill()
    df["shares_outstanding"] = df["share_class_shares_outstanding"].ffill().bfill()

    if step == 1:
        df["market_cap"] = df["market_cap"].ffill().bfill()
    else:
        df["market_cap"] = df["close"] * df["float"]

    return df.drop(columns=["weighted_shares_outstanding", "share_class_shares_outstanding"])


# ---------------------------------------------------------------------------
# Per-ticker orchestration
# ---------------------------------------------------------------------------

def _sample_dates(date_strs: list[str], step: int) -> list[str]:
    """Every `step`-th trading day, always including the first and last."""
    if step <= 1:
        return date_strs
    sampled = date_strs[::step]
    if date_strs[-1] not in sampled:
        sampled.append(date_strs[-1])
    return sampled


async def process_ticker(
    ticker: str,
    from_date: date,
    to_date: date,
    client: httpx.AsyncClient,
    req_sem: asyncio.Semaphore,
    args: argparse.Namespace,
) -> tuple[str, str, pd.DataFrame | None]:
    """
    Process one ticker end-to-end.

    Returns (ticker, status, df_or_None). status in {ok, empty, error}.
    The driver collects the DataFrames and flushes them to the shard parquet.
    """
    # Steps 2 + 3 — candles + indicators (sync, off the event loop).
    try:
        df = await asyncio.to_thread(_build_candles_df, ticker, from_date, to_date)
    except Exception as exc:
        logger.warning("candles failed  %-6s  %s", ticker, exc)
        return ticker, "error", None

    if df.empty:
        return ticker, "empty", None

    # Step 4 — market cap + float per trading day.
    if args.skip_marketcap:
        df["market_cap"] = pd.NA
        df["float"] = pd.NA
        df["shares_outstanding"] = pd.NA
    else:
        date_strs = df["date_str"].tolist()
        sample = _sample_dates(date_strs, args.marketcap_step)
        try:
            ref = await _fetch_marketcaps(client, ticker, sample, req_sem, args.retries)
        except Exception as exc:
            logger.warning("marketcap failed  %-6s  %s", ticker, exc)
            ref = pd.DataFrame(
                columns=["date_str", "market_cap",
                         "weighted_shares_outstanding", "share_class_shares_outstanding"]
            )
        df = _attach_marketcap(df, ref, args.marketcap_step)

    return ticker, "ok", df.reindex(columns=OUTPUT_COLUMNS)


# ---------------------------------------------------------------------------
# Shard I/O (single parquet per shard, rewritten for resume/checkpoint)
# ---------------------------------------------------------------------------

def _shard_path(out_dir: Path, shard: int, num_shards: int) -> Path:
    return out_dir / f"shard_{shard:03d}_of_{num_shards:03d}.parquet"


def _write_shard(parts: list[pd.DataFrame], path: Path) -> None:
    """Concatenate all collected parts and write the shard parquet atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    combined = (pd.concat(parts, ignore_index=True)
                  .drop_duplicates(subset=["ticker", "date_str"], keep="last")
                  .sort_values(["ticker", "date_str"])
                  .reset_index(drop=True))
    tmp = path.with_suffix(".parquet.tmp")
    combined.to_parquet(tmp, index=False, compression="zstd")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Universe selection
# ---------------------------------------------------------------------------

def load_tickers(args: argparse.Namespace) -> list[str]:
    """Resolve the list of tickers to process (universe file, shard, overrides)."""
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        path = Path(args.universe)
        if not path.exists():
            raise FileNotFoundError(
                f"Universe file not found: {path}. Run build_stock_universe first."
            )
        tickers = pd.read_parquet(path, columns=["ticker"])["ticker"].tolist()

    if args.num_shards > 1:
        tickers = [t for i, t in enumerate(tickers) if i % args.num_shards == args.shard]
        logger.info("Shard %d/%d -> %d tickers.", args.shard, args.num_shards, len(tickers))

    if args.limit:
        tickers = tickers[: args.limit]

    return tickers


# ---------------------------------------------------------------------------
# Async driver
# ---------------------------------------------------------------------------

async def run(args: argparse.Namespace) -> None:
    from_date = date.fromisoformat(args.from_date) if args.from_date else _default_from(args.years)
    to_date   = date.fromisoformat(args.to_date)
    if from_date > to_date:
        raise SystemExit(f"--from {from_date} is after --to {to_date}")

    if not get_settings().massive_api_key:
        raise SystemExit("MASSIVE_API_KEY is not set in the environment.")

    tickers   = load_tickers(args)
    out_dir   = Path(args.out_dir)
    shard_path = _shard_path(out_dir, args.shard, args.num_shards)

    # Resume: skip tickers already present in this shard's parquet.
    parts: list[pd.DataFrame] = []
    if args.resume and shard_path.exists():
        existing = pd.read_parquet(shard_path)
        done = set(existing["ticker"].unique())
        before = len(tickers)
        tickers = [t for t in tickers if t not in done]
        parts.append(existing)
        logger.info("Resume: %d/%d tickers already in %s — %d remaining.",
                    before - len(tickers), before, shard_path.name, len(tickers))

    if not tickers:
        logger.info("Nothing to do for shard %d.", args.shard)
        return

    logger.info(
        "Shard %d/%d: %d tickers  %s -> %s  concurrency=%d  ticker_concurrency=%d  "
        "marketcap_step=%d  skip_marketcap=%s",
        args.shard, args.num_shards, len(tickers), from_date, to_date,
        args.concurrency, args.ticker_concurrency, args.marketcap_step, args.skip_marketcap,
    )

    req_sem  = asyncio.Semaphore(args.concurrency)
    tick_sem = asyncio.Semaphore(args.ticker_concurrency)
    limits = httpx.Limits(
        max_connections=args.concurrency + 10,
        max_keepalive_connections=args.concurrency,
    )

    tally: dict[str, int] = {"ok": 0, "empty": 0, "error": 0}
    done_count = 0
    since_flush = 0
    total = len(tickers)

    async with httpx.AsyncClient(limits=limits, timeout=30) as client:
        async def worker(tk: str):
            async with tick_sem:
                return await process_ticker(tk, from_date, to_date, client, req_sem, args)

        tasks = [asyncio.create_task(worker(tk)) for tk in tickers]
        for fut in asyncio.as_completed(tasks):
            ticker, status, df = await fut
            tally[status] = tally.get(status, 0) + 1
            done_count += 1

            if df is not None:
                parts.append(df)
                since_flush += 1

            # Periodic checkpoint so a crash loses at most --flush-every tickers.
            if since_flush >= args.flush_every:
                _write_shard(parts, shard_path)
                since_flush = 0
                logger.info("Checkpoint -> %s  (%d tickers, %d rows)",
                            shard_path.name, len(parts) - (1 if args.resume else 0),
                            sum(len(p) for p in parts))

            if done_count % 100 == 0 or done_count == total:
                logger.info("Progress %d/%d  %s", done_count, total, tally)

    if since_flush > 0:  # unflushed new tickers remain
        _write_shard(parts, shard_path)
    logger.info("Shard %d done. %s -> %s", args.shard, tally, shard_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build per-shard daily OHLCV + indicators + market cap/float for the stock universe."
    )
    parser.add_argument("--universe", default=str(UNIVERSE_PATH),
                        help=f"Universe parquet (default: {UNIVERSE_PATH}).")
    parser.add_argument("--out-dir", default=str(SHARDS_DIR),
                        help=f"Shard output directory (default: {SHARDS_DIR}).")
    parser.add_argument("--tickers",
                        help="Comma-separated tickers to process instead of the universe file.")

    parser.add_argument("--years", type=int, default=DEFAULT_YEARS,
                        help=f"Years of history when --from is omitted (default: {DEFAULT_YEARS}).")
    parser.add_argument("--from", dest="from_date",
                        help="Start date YYYY-MM-DD (default: --years ago).")
    parser.add_argument("--to", dest="to_date", default=date.today().isoformat(),
                        help="End date YYYY-MM-DD inclusive (default: today).")

    parser.add_argument("--concurrency", type=int, default=50,
                        help="Max concurrent reference (market cap) requests (default: 50).")
    parser.add_argument("--ticker-concurrency", type=int, default=8,
                        help="Max tickers processed concurrently (default: 8).")
    parser.add_argument("--retries", type=int, default=5,
                        help="Retries per reference request on 429/5xx/timeout (default: 5).")
    parser.add_argument("--flush-every", type=int, default=200,
                        help="Rewrite the shard parquet every N completed tickers (default: 200).")

    parser.add_argument("--marketcap-step", type=int, default=1,
                        help="Sample the reference endpoint every N trading days; "
                             "share counts are forward-filled and market_cap recomputed "
                             "as close*float for in-between days (default: 1 = every day).")
    parser.add_argument("--skip-marketcap", action="store_true",
                        help="Skip step 4 entirely (candles + indicators only).")

    parser.add_argument("--num-shards", type=int, default=1,
                        help="Total number of parallel shards (default: 1).")
    parser.add_argument("--shard", type=int, default=0,
                        help="This process's shard index in [0, num-shards) (default: 0).")

    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="Reprocess tickers even if already present in the shard parquet.")
    parser.add_argument("--limit", type=int,
                        help="Process only the first N tickers (after sharding) — for testing.")

    args = parser.parse_args()

    if not 0 <= args.shard < args.num_shards:
        parser.error(f"--shard must be in [0, {args.num_shards}); got {args.shard}")

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
