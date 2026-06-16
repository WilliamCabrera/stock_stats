"""
Incremental updater for backtest_dataset/INDICES/{ticker}/{tf}/*_full_dataset.parquet.

Mirrors the workflow of update_full_dataset.py but for index / large-cap tickers
that live in backtest_dataset/INDICES/ instead of backtest_dataset/full/.

Steps per index ticker:
  1. Read the existing 1d parquet → find the most recent date_str
  2. Fetch new candles for each timeframe from (latest + 1 day) to today,
     with a warmup prefix so rolling indicators are computed correctly
  3. Compute indicators:
       1d  → sma_9/20/50/200, atr_14, daily_range, daily_range_ma10
       1h  → sma_9/20/50/200, atr_14, daily_range_ma10  (joined from 1d)
       5m  → sma_9/20/50/200, atr_14, daily_range_ma10, h1_9am_high/low
       10m → sma_9/20/50/200, atr_14, daily_range_ma10, h1_9am_high/low
  4. Upsert new rows (warmup rows discarded) into each timeframe parquet

Usage (from backtester_api/):
    python -m scripts.update_indices_dataset                  # all registered indices
    python -m scripts.update_indices_dataset --ticker TQQQ    # one ticker
    python -m scripts.update_indices_dataset --ticker QQQ --timeframe 5m
    python -m scripts.update_indices_dataset --dry-run        # print plan, write nothing
    python -m scripts.update_indices_dataset --from 2025-01-01  # override start date
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
# Registry — add new indices here
# ---------------------------------------------------------------------------
INDICES: dict[str, dict] = {
    "TQQQ": {"name": "tqqq", "timeframes": ["1d", "1h", "5m", "10m"]},
    "QQQ":  {"name": "qqq",  "timeframes": ["1d", "1h", "5m", "10m"]},
}

BASE        = Path("backtest_dataset/INDICES")
SESSION     = ("04:00", "20:00")
WARMUP_DAYS = 20   # extra calendar days fetched before the new window for indicator warmup


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dataset_path(ticker: str, tf: str) -> Path:
    name = INDICES[ticker]["name"]
    return BASE / ticker / tf / f"{name}_full_dataset.parquet"


def _latest_date(ticker: str, tf: str = "1d") -> date | None:
    path = _dataset_path(ticker, tf)
    if not path.exists():
        return None
    df = pd.read_parquet(path, columns=["date_str"])
    if df.empty:
        return None
    return date.fromisoformat(df["date_str"].max())


def _fetch_tf(ticker: str, tf: str, from_date: date, to_date: date) -> pd.DataFrame:
    kw: dict = dict(timeframe=tf)
    if tf != "1d":
        kw["session_start"], kw["session_end"] = SESSION
    candles = fetch_candles(ticker, from_date.isoformat(), to_date.isoformat(), **kw)
    if not candles:
        return pd.DataFrame()
    df        = pd.DataFrame(candles)
    dt        = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert("America/New_York")
    df["date"]     = pd.to_datetime(dt.dt.strftime("%Y-%m-%dT%H:%M:%S"))
    df["date_str"] = dt.dt.strftime("%Y-%m-%d")
    df["ticker"]   = ticker
    cols = ["ticker", "date", "date_str", "open", "high", "low", "close", "volume"]
    return df[cols].drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)


def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sma_9"]   = compute_sma(df, window=9)
    df["sma_20"]  = compute_sma(df, window=20)
    df["sma_50"]  = compute_sma(df, window=50)
    df["sma_200"] = compute_sma(df, window=200)
    df["atr_14"]  = compute_atr(df, window=14)
    return df


def _upsert(path: Path, new_rows: pd.DataFrame, new_date_strs: set[str]) -> None:
    """Replace rows for new_date_strs in the existing parquet and append brand-new ones."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_parquet(path)
        existing = existing[~existing["date_str"].isin(new_date_strs)]
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        combined = new_rows
    combined = combined.sort_values("date").reset_index(drop=True)
    combined.to_parquet(path, index=False, compression="zstd")
    logger.info(
        "Upserted %s: +%d rows for %d date(s) → %d total  (%.1f MB)",
        path, len(new_rows), len(new_date_strs), len(combined), path.stat().st_size / 1e6,
    )


def _h1_9am(df1h: pd.DataFrame) -> pd.DataFrame:
    return (
        df1h[df1h["date"].dt.hour == 9][["date_str", "high", "low"]]
        .rename(columns={"high": "h1_9am_high", "low": "h1_9am_low"})
    )


# ---------------------------------------------------------------------------
# Per-timeframe updaters
# ---------------------------------------------------------------------------

def _update_1d(
    ticker: str, warmup_from: date, to_date: date, fetch_from: date
) -> tuple[pd.DataFrame, set[str]]:
    """
    Fetch 1d candles (warmup_from → to_date), compute indicators, upsert new rows.
    Returns (full_df_with_warmup, new_date_strs) so downstream timeframes can join.
    """
    df = _fetch_tf(ticker, "1d", warmup_from, to_date)
    if df.empty:
        logger.warning("%s 1d: no candles fetched.", ticker)
        return pd.DataFrame(), set()

    df = _add_indicators(df)
    df["sma_100"]          = compute_sma(df, window=100)
    df["daily_range"]      = df["high"] - df["low"]
    df["daily_range_ma10"] = df["daily_range"].rolling(10, min_periods=1).mean()

    new_date_strs = set(df[df["date_str"] >= fetch_from.isoformat()]["date_str"])
    new_rows      = df[df["date_str"].isin(new_date_strs)].copy()

    if new_rows.empty:
        logger.info("%s 1d: nothing new after warmup trim.", ticker)
    else:
        _upsert(_dataset_path(ticker, "1d"), new_rows, new_date_strs)

    return df, new_date_strs   # df includes warmup rows for cross-timeframe joins


def _update_1h(
    ticker: str, warmup_from: date, to_date: date,
    new_date_strs: set[str], df1d: pd.DataFrame,
) -> pd.DataFrame:
    """Fetch 1h, join daily_range_ma10 from df1d, upsert new rows."""
    df = _fetch_tf(ticker, "1h", warmup_from, to_date)
    if df.empty:
        logger.warning("%s 1h: no candles fetched.", ticker)
        return pd.DataFrame()

    df = _add_indicators(df)
    df = df.merge(df1d[["date_str", "daily_range_ma10"]], on="date_str", how="left")

    new_rows = df[df["date_str"].isin(new_date_strs)].copy()
    if new_rows.empty:
        logger.info("%s 1h: nothing new after warmup trim.", ticker)
    else:
        _upsert(_dataset_path(ticker, "1h"), new_rows, new_date_strs)

    return df   # includes warmup rows for h1_9am join downstream


def _update_intraday(
    ticker: str, tf: str, warmup_from: date, to_date: date,
    new_date_strs: set[str], df1d: pd.DataFrame, df1h: pd.DataFrame,
) -> None:
    """Fetch 5m or 10m, join daily_range_ma10 and h1_9am, upsert new rows."""
    df = _fetch_tf(ticker, tf, warmup_from, to_date)
    if df.empty:
        logger.warning("%s %s: no candles fetched.", ticker, tf)
        return

    df = _add_indicators(df)
    df = df.merge(df1d[["date_str", "daily_range_ma10"]], on="date_str", how="left")
    if not df1h.empty:
        df = df.merge(_h1_9am(df1h), on="date_str", how="left")

    new_rows = df[df["date_str"].isin(new_date_strs)].copy()
    if new_rows.empty:
        logger.info("%s %s: nothing new after warmup trim.", ticker, tf)
    else:
        _upsert(_dataset_path(ticker, tf), new_rows, new_date_strs)


# ---------------------------------------------------------------------------
# Main update logic per ticker
# ---------------------------------------------------------------------------

def update_index(
    ticker: str,
    timeframes: list[str] | None = None,
    from_date: date | None = None,
    to_date: date | None = None,
    dry_run: bool = False,
) -> None:
    cfg     = INDICES[ticker]
    tfs     = timeframes or cfg["timeframes"]
    to_date = to_date or date.today()

    # Determine fetch_from: explicit --from override, else latest_date + 1 day
    latest = _latest_date(ticker, "1d")
    if from_date is not None:
        fetch_from = from_date
    elif latest is None:
        logger.error(
            "%s: no existing 1d dataset found at %s. Run build script first.",
            ticker, _dataset_path(ticker, "1d"),
        )
        return
    else:
        fetch_from = latest + timedelta(days=1)

    if fetch_from > to_date:
        logger.info("%s is already up to date (latest: %s).", ticker, latest)
        return

    warmup_from = fetch_from - timedelta(days=WARMUP_DAYS)

    logger.info(
        "%s  fetch_from=%s  to=%s  warmup_from=%s  timeframes=%s",
        ticker, fetch_from, to_date, warmup_from, tfs,
    )

    if dry_run:
        print(
            f"[dry-run] {ticker}: would fetch {tfs} "
            f"from {fetch_from} to {to_date} (warmup from {warmup_from})"
        )
        return

    # Update in dependency order: 1d → 1h → 5m / 10m
    df1d: pd.DataFrame = pd.DataFrame()
    df1h: pd.DataFrame = pd.DataFrame()
    new_date_strs: set[str] = set()

    if "1d" in tfs:
        df1d, new_date_strs = _update_1d(ticker, warmup_from, to_date, fetch_from)
    else:
        # Load existing 1d for cross-joins even when not updating it
        path_1d = _dataset_path(ticker, "1d")
        if path_1d.exists():
            df1d = pd.read_parquet(path_1d, columns=["date_str", "daily_range_ma10"])
        # Best-effort date range (actual trading days will be a subset)
        new_date_strs = {
            d.strftime("%Y-%m-%d")
            for d in pd.date_range(fetch_from, to_date)
        }

    if not new_date_strs:
        logger.info("%s: no new trading days found.", ticker)
        return

    if "1h" in tfs:
        df1h = _update_1h(ticker, warmup_from, to_date, new_date_strs, df1d)

    for tf in [t for t in tfs if t not in ("1d", "1h")]:
        _update_intraday(ticker, tf, warmup_from, to_date, new_date_strs, df1d, df1h)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Incrementally update INDICES datasets (TQQQ, QQQ, …)."
    )
    parser.add_argument(
        "--ticker",
        choices=list(INDICES.keys()),
        help="Index to update. Omit to update all registered indices.",
    )
    parser.add_argument(
        "--timeframe",
        choices=["1d", "1h", "5m", "10m"],
        help="Timeframe to update. Omit to update all timeframes for the ticker.",
    )
    parser.add_argument(
        "--from", dest="from_date",
        help="Override start date YYYY-MM-DD (default: latest date in parquet + 1 day).",
    )
    parser.add_argument(
        "--to", dest="to_date", default=date.today().isoformat(),
        help="End date YYYY-MM-DD inclusive (default: today).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be fetched without writing anything.",
    )
    args = parser.parse_args()

    from_date = date.fromisoformat(args.from_date) if args.from_date else None
    to_date   = date.fromisoformat(args.to_date)
    tickers   = [args.ticker] if args.ticker else list(INDICES.keys())
    tfs       = [args.timeframe] if args.timeframe else None

    for ticker in tickers:
        update_index(
            ticker,
            timeframes=tfs,
            from_date=from_date,
            to_date=to_date,
            dry_run=args.dry_run,
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
