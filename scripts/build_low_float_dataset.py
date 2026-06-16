"""
Build a per-day session dataset for the low-float tickers (last N years).

Derived from app.utils.market_utils.process_data_minutes — same session logic,
with corrections for this use case (see process_minute_bars below):

  * UNADJUSTED 1-minute candles (real traded prices).
  * Prices kept at 6 decimals (low-float names trade sub-penny; the original
    round(3) destroyed price information).
  * Missing-session fields are NaN, not the -1 sentinel (cleaner for parquet
    analytics; np.where(prev > 0, ...) and pm_open > 0 still behave correctly).

Per ticker:
  1. Fetch 1m unadjusted candles (04:00-20:00 ET) for the window.
  2. process_minute_bars  -> per-day session summary.
  3. _apply_gap_logic (reused) -> previous_close (+ split override),
     gap, gap_perc, daily_range, day_range_perc.
  4. Merge market_cap + stock_float from stock_dataset.parquet.

Output is ONE parquet per shard (resumable, run shards in parallel):
    backtest_dataset/LOW_FLOAT/shards/shard_<I>_of_<N>.parquet

Columns:
    ticker, date_str,
    gap, gap_perc, daily_range, day_range_perc, previous_close,
    open, high, low, close, volume,
    premarket_volume, market_hours_volume,
    high_pm, low_pm, pm_open, highest_in_pm, high_pm_time, high_mh,
    ah_open, ah_close, ah_high, ah_low, ah_range, ah_range_perc, ah_volume,
    market_cap, stock_float,
    split_date_str, split_adjust_factor,
    time
(daily_200_sma intentionally omitted for now.)

Usage (from backtester_api/):
    python -m scripts.build_low_float_dataset --tickers BHAT,SMFL --limit 2   # smoke test
    python -m scripts.build_low_float_dataset --num-shards 8 --shard 0
    python -m scripts.build_low_float_dataset --years 5 --ticker-concurrency 6
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from datetime import date, time as dtime
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("."))

from app.config import get_settings
from app.utils.massive import fetch_candles
from app.utils.pipeline_data_collection import _apply_gap_logic

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LOW_FLOAT_TICKERS = Path("backtest_dataset/STOCKS/low_float_tickers.parquet")
STOCK_DATASET     = Path("backtest_dataset/STOCKS/stock_dataset.parquet")
SHARDS_DIR        = Path("backtest_dataset/LOW_FLOAT/shards")
DEFAULT_YEARS     = 5

# Session boundaries (America/New_York local time) — unchanged from market_utils.
_PM_START, _PM_END = dtime(4, 0),  dtime(9, 30)
_MH_START, _MH_END = dtime(9, 30), dtime(16, 0)
_AH_START, _AH_END = dtime(16, 0), dtime(20, 0)

OUTPUT_COLUMNS = [
    "ticker", "date_str",
    "gap", "gap_perc", "daily_range", "day_range_perc", "previous_close",
    "open", "high", "low", "close", "volume",
    "premarket_volume", "market_hours_volume",
    "high_pm", "low_pm", "pm_open", "highest_in_pm", "high_pm_time", "high_mh",
    "ah_open", "ah_close", "ah_high", "ah_low", "ah_range", "ah_range_perc", "ah_volume",
    "market_cap", "stock_float",
    "split_date_str", "split_adjust_factor",
    "time",
]

# Price columns kept at 6 decimals (sub-penny low-float names).
_PRICE_COLS = [
    "open", "high", "low", "close", "previous_close",
    "high_pm", "low_pm", "pm_open", "high_mh",
    "ah_open", "ah_close", "ah_high", "ah_low",
]


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def _default_from(years: int) -> date:
    today = date.today()
    try:
        return today.replace(year=today.year - years)
    except ValueError:
        return today.replace(year=today.year - years, day=28)


# ---------------------------------------------------------------------------
# Core — per-day session summary (corrected fork of process_data_minutes)
# ---------------------------------------------------------------------------

def process_minute_bars(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate 1-minute candles into a per-day, per-session summary.

    Expects the columns produced by app.utils.massive.fetch_candles:
        open, high, low, close, volume, date (tz-aware America/New_York),
        day (datetime.date).

    Returns one row per trading day. Differences from process_data_minutes:
      * prices rounded to 6 decimals (not 3) — keeps sub-penny precision;
      * missing-session prices left as NaN (not -1);
      * errors raised, not swallowed by a bare print.
    """
    if df.empty:
        return df

    df = df.copy()
    t = df["date"].dt.time

    df["_pm"] = (t >= _PM_START) & (t < _PM_END)
    df["_mh"] = (t >= _MH_START) & (t < _MH_END)
    df["_ah"] = (t >= _AH_START) & (t <= _AH_END)

    pm = df[df["_pm"]].sort_values(["day", "date"])
    mh = df[df["_mh"]].sort_values(["day", "date"])
    ah = df[df["_ah"]].sort_values(["day", "date"])

    # ── Market hours: the day's backbone (open/close/high/low/volume) ──────────
    mh_agg = mh.groupby("day", as_index=False).agg(
        open                = ("open",   "first"),
        close               = ("close",  "last"),
        high_mh             = ("high",   "max"),
        low_mh              = ("low",    "min"),
        market_hours_volume = ("volume", "sum"),
    )

    # ── Pre-market ─────────────────────────────────────────────────────────────
    pm_agg = pm.groupby("day", as_index=False).agg(
        high_pm          = ("high",   "max"),
        low_pm           = ("low",    "min"),
        pm_open          = ("open",   "first"),
        premarket_volume = ("volume", "sum"),
    )

    # Time (epoch ms) at which the pre-market high was first reached.
    if not pm.empty:
        pm_h = pm.merge(pm_agg[["day", "high_pm"]], on="day", how="left")
        pm_h = pm_h[pm_h["high"] == pm_h["high_pm"]]
        pm_h = (pm_h.sort_values(["day", "date"])
                    .groupby("day", as_index=False).first()[["day", "date"]])
        pm_h["high_pm_time"] = pm_h["date"].astype("int64") // 10**6
        pm_h = pm_h[["day", "high_pm_time"]]
    else:
        pm_h = pd.DataFrame(columns=["day", "high_pm_time"])

    # ── After hours ────────────────────────────────────────────────────────────
    ah_agg = ah.groupby("day", as_index=False).agg(
        ah_open   = ("open",   "first"),
        ah_close  = ("close",  "last"),
        ah_high   = ("high",   "max"),
        ah_low    = ("low",    "min"),
        ah_volume = ("volume", "sum"),
    )
    ah_agg["ah_range"]      = ah_agg["ah_high"] - ah_agg["ah_open"]
    ah_agg["ah_range_perc"] = 100 * ah_agg["ah_range"] / ah_agg["ah_open"]

    # ── Assemble: market hours is the spine; every trading day has an mh row ────
    out = mh_agg.merge(pm_agg, on="day", how="left") \
                .merge(pm_h,   on="day", how="left") \
                .merge(ah_agg, on="day", how="left")

    # Full-day high/low include pre-market (after-hours excluded, as before).
    out["high"] = np.maximum(out["high_mh"], out["high_pm"].fillna(-np.inf))
    out["low"]  = np.minimum(out["low_mh"],  out["low_pm"].fillna(np.inf))
    out["highest_in_pm"] = out["high_pm"] >= out["high_mh"]   # NaN high_pm -> False

    # `volume` mirrors market-hours volume (kept for schema parity).
    out["volume"] = out["market_hours_volume"]

    # Epoch ms of ET midnight for the day, and the date string.
    out["time"] = (pd.to_datetime(out["day"])
                   .dt.tz_localize("America/New_York")
                   .astype("int64") // 10**6)
    out["date_str"] = pd.to_datetime(out["day"]).dt.strftime("%Y-%m-%d")

    # Volumes are counts → 0 when the session had no candles.
    for col in ("volume", "market_hours_volume", "premarket_volume", "ah_volume"):
        out[col] = out[col].fillna(0)

    return out


# ---------------------------------------------------------------------------
# Splits (sync httpx) + daily previous_close
# ---------------------------------------------------------------------------

def _fetch_splits(ticker: str, client: httpx.Client) -> list[dict]:
    """Return split events as [{execution_date, historical_adjustment_factor}]."""
    settings = get_settings()
    url: str | None = (
        f"{settings.massive_base_url.rstrip('/')}/v3/reference/splits"
        f"?ticker={ticker}&limit=1000&apiKey={settings.massive_api_key}"
    )
    out: list[dict] = []
    while url:
        resp = client.get(url)
        if resp.status_code != 200:
            break
        data = resp.json()
        for item in data.get("results") or []:
            sf = item.get("split_from", 1) or 1
            st = item.get("split_to", 1) or 1
            out.append({
                "execution_date": item["execution_date"],
                "historical_adjustment_factor": sf / st,
            })
        nxt = data.get("next_url")
        url = f"{nxt}&apiKey={settings.massive_api_key}" if nxt else None
    return out


def _daily_prev_close(ticker: str, from_date: date, to_date: date) -> pd.DataFrame:
    """date_str -> previous_close from UNADJUSTED 1d bars (shift by one session)."""
    candles = fetch_candles(ticker, from_date.isoformat(), to_date.isoformat(),
                            timeframe="1d", adjusted=False)
    if not candles:
        return pd.DataFrame(columns=["date_str", "previous_close"])
    d = pd.DataFrame(candles)
    d["date_str"] = (pd.to_datetime(d["time"], unit="s", utc=True)
                     .dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d"))
    d = d.sort_values("date_str").reset_index(drop=True)
    d["previous_close"] = d["close"].shift(1)
    return d[["date_str", "previous_close"]]


# ---------------------------------------------------------------------------
# Per-ticker pipeline (sync; runs in a thread)
# ---------------------------------------------------------------------------

def build_ticker(
    ticker: str,
    from_date: date,
    to_date: date,
    mc_lookup: pd.DataFrame,
    client: httpx.Client,
) -> pd.DataFrame | None:
    """Full per-ticker build → one DataFrame aligned to OUTPUT_COLUMNS, or None."""
    candles = fetch_candles(ticker, from_date.isoformat(), to_date.isoformat(),
                            timeframe="1m", adjusted=False,
                            session_start="04:00", session_end="20:00")
    if not candles:
        return None

    raw = pd.DataFrame(candles)
    # NOTE: fetch_candles' own 'date'/'day' columns are broken (massive.py builds
    # them with unit='ms' on a seconds value → 1970). Derive from 'time' (seconds),
    # exactly like build_index_dataset / build_stock_dataset do.
    raw["date"] = pd.to_datetime(raw["time"], unit="s", utc=True).dt.tz_convert("America/New_York")
    raw["day"]  = raw["date"].dt.date

    daily = process_minute_bars(raw)
    if daily.empty:
        return None
    daily["ticker"] = ticker

    # Gap logic + previous_close (+ split override) — reuse the proven function.
    prev = _daily_prev_close(ticker, from_date, to_date)
    splits = _fetch_splits(ticker, client)
    daily = _apply_gap_logic(daily, prev, splits)

    # market_cap + stock_float from the daily stock dataset.
    mc = mc_lookup[mc_lookup["ticker"] == ticker][["date_str", "market_cap", "stock_float"]]
    daily = daily.merge(mc, on="date_str", how="left")
    daily["market_cap"]  = daily.groupby("ticker")["market_cap"].ffill().bfill()
    daily["stock_float"] = daily.groupby("ticker")["stock_float"].ffill().bfill()

    # Precision: prices at 6 decimals, derived percents at 3.
    for col in _PRICE_COLS:
        if col in daily:
            daily[col] = daily[col].round(6)
    for col in ("gap", "gap_perc", "daily_range", "day_range_perc", "ah_range", "ah_range_perc"):
        if col in daily:
            daily[col] = daily[col].round(3)

    daily["time"] = daily["time"].astype("int64")
    daily["high_pm_time"] = daily["high_pm_time"].astype("Int64")  # nullable

    return daily.reindex(columns=OUTPUT_COLUMNS).sort_values("date_str").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Shard I/O (single parquet per shard, checkpointed for resume)
# ---------------------------------------------------------------------------

def _shard_path(out_dir: Path, shard: int, num_shards: int) -> Path:
    return out_dir / f"shard_{shard:03d}_of_{num_shards:03d}.parquet"


def _write_shard(parts: list[pd.DataFrame], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    combined = (pd.concat(parts, ignore_index=True)
                  .drop_duplicates(subset=["ticker", "date_str"], keep="last")
                  .sort_values(["ticker", "date_str"]).reset_index(drop=True))
    tmp = path.with_suffix(".parquet.tmp")
    combined.to_parquet(tmp, index=False, compression="zstd")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Universe / market-cap lookup
# ---------------------------------------------------------------------------

def load_tickers(args: argparse.Namespace) -> list[str]:
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        path = Path(args.tickers_file)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run filter_low_float first.")
        tickers = pd.read_parquet(path, columns=["ticker"])["ticker"].tolist()

    if args.num_shards > 1:
        tickers = [t for i, t in enumerate(tickers) if i % args.num_shards == args.shard]
        logger.info("Shard %d/%d -> %d tickers.", args.shard, args.num_shards, len(tickers))
    if args.limit:
        tickers = tickers[: args.limit]
    return tickers


def load_marketcap_lookup(tickers: list[str]) -> pd.DataFrame:
    """Slim (ticker, date_str, market_cap, stock_float) table for the given tickers."""
    if not STOCK_DATASET.exists():
        logger.warning("%s not found — market_cap/stock_float will be NaN.", STOCK_DATASET)
        return pd.DataFrame(columns=["ticker", "date_str", "market_cap", "stock_float"])
    df = pd.read_parquet(STOCK_DATASET, columns=["ticker", "date_str", "market_cap", "float"])
    df = df[df["ticker"].isin(set(tickers))].rename(columns={"float": "stock_float"})
    return df.reset_index(drop=True)


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

    tickers    = load_tickers(args)
    out_dir    = Path(args.out_dir)
    shard_path = _shard_path(out_dir, args.shard, args.num_shards)

    parts: list[pd.DataFrame] = []
    if args.resume and shard_path.exists():
        existing = pd.read_parquet(shard_path)
        done = set(existing["ticker"].unique())
        before = len(tickers)
        tickers = [t for t in tickers if t not in done]
        parts.append(existing)
        logger.info("Resume: %d already done, %d remaining.", before - len(tickers), len(tickers))

    if not tickers:
        logger.info("Nothing to do for shard %d.", args.shard)
        return

    logger.info("Loading market_cap/stock_float lookup for %d tickers…", len(tickers))
    mc_lookup = load_marketcap_lookup(tickers)

    logger.info("Shard %d/%d: %d tickers  %s -> %s  ticker_concurrency=%d",
                args.shard, args.num_shards, len(tickers), from_date, to_date,
                args.ticker_concurrency)

    tick_sem = asyncio.Semaphore(args.ticker_concurrency)
    tally = {"ok": 0, "empty": 0, "error": 0}
    done_count = since_flush = 0
    total = len(tickers)

    with httpx.Client(timeout=60) as client:
        async def worker(tk: str):
            async with tick_sem:
                try:
                    return tk, "ok", await asyncio.to_thread(
                        build_ticker, tk, from_date, to_date, mc_lookup, client)
                except Exception as exc:
                    logger.warning("build failed  %-6s  %s", tk, exc)
                    return tk, "error", None

        tasks = [asyncio.create_task(worker(tk)) for tk in tickers]
        for fut in asyncio.as_completed(tasks):
            tk, status, df = await fut
            if status == "ok" and (df is None or df.empty):
                status = "empty"
            tally[status] = tally.get(status, 0) + 1
            done_count += 1
            if df is not None and not df.empty:
                parts.append(df)
                since_flush += 1
            if since_flush >= args.flush_every:
                _write_shard(parts, shard_path)
                since_flush = 0
                logger.info("Checkpoint -> %s  (%d rows)", shard_path.name,
                            sum(len(p) for p in parts))
            if done_count % 50 == 0 or done_count == total:
                logger.info("Progress %d/%d  %s", done_count, total, tally)

    if since_flush > 0:
        _write_shard(parts, shard_path)
    logger.info("Shard %d done. %s -> %s", args.shard, tally, shard_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build per-day session dataset for low-float tickers (1m unadjusted)."
    )
    parser.add_argument("--tickers-file", default=str(LOW_FLOAT_TICKERS),
                        help=f"Parquet with the ticker list (default: {LOW_FLOAT_TICKERS}).")
    parser.add_argument("--out-dir", default=str(SHARDS_DIR),
                        help=f"Shard output directory (default: {SHARDS_DIR}).")
    parser.add_argument("--tickers", help="Comma-separated tickers (overrides the file).")

    parser.add_argument("--years", type=int, default=DEFAULT_YEARS,
                        help=f"Years of history when --from is omitted (default: {DEFAULT_YEARS}).")
    parser.add_argument("--from", dest="from_date", help="Start date YYYY-MM-DD.")
    parser.add_argument("--to", dest="to_date", default=date.today().isoformat(),
                        help="End date YYYY-MM-DD inclusive (default: today).")

    parser.add_argument("--ticker-concurrency", type=int, default=6,
                        help="Tickers processed concurrently (default: 6). 1m fetches are heavy.")
    parser.add_argument("--flush-every", type=int, default=50,
                        help="Rewrite the shard parquet every N tickers (default: 50).")

    parser.add_argument("--num-shards", type=int, default=1, help="Total parallel shards.")
    parser.add_argument("--shard", type=int, default=0, help="This shard index [0, num-shards).")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="Reprocess tickers already present in the shard parquet.")
    parser.add_argument("--limit", type=int, help="Process only first N tickers (after shard).")

    args = parser.parse_args()
    if not 0 <= args.shard < args.num_shards:
        parser.error(f"--shard must be in [0, {args.num_shards}); got {args.shard}")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
