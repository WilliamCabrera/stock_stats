"""
Build walk-forward backtest datasets from stock_data_filtered.

Walk-forward windows  (IS = 1 year, OOS = 6 months, slide = 6 months):

    Fold 1: IS  [d0,       d0+12M)   OOS [d0+12M, d0+18M)
    Fold 2: IS  [d0+6M,   d0+18M)   OOS [d0+18M, d0+24M)
    Fold 3: IS  [d0+12M,  d0+24M)   OOS [d0+24M, d0+30M)

Output layout:
    backtest_dataset/walkforward/{timeframe}/
        fold_1/in_sample.parquet
        fold_1/out_of_sample.parquet
        fold_2/...
        fold_3/...

Each parquet has one row per bar with columns:
    ticker, date (naive ET datetime), date_str,
    open, high, low, close, volume,
    atr, RVOL_daily, SMA_VOLUME_20_5m, vwap, previous_day_close,
    sma_9, sma_200,
    donchian_upper, donchian_lower, donchian_basis

Usage (from backtester_api/):
    python -m scripts.build_walkforward_datasets              # default: 5m
    python -m scripts.build_walkforward_datasets --tf 15m
    python -m scripts.build_walkforward_datasets --tf 1h
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
from app.utils.pipeline_v1 import fetch_stock_data_filtered

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
IS_MONTHS        = 12   # in-sample window length (months)
OOS_MONTHS       = 6    # out-of-sample window length (months)
SLIDE_MONTHS     = 6    # how many months to slide each fold
N_FOLDS          = 3
OUTPUT_BASE      = Path("backtest_dataset/walkforward")

VALID_TIMEFRAMES = {"1m", "5m", "15m", "30m", "1h", "1d"}

# Calendar-day lookback per timeframe — enough to warm up SMA 200.
# Formula: ceil(200 / bars_per_trading_day) * (365/252) + buffer
_LOOKBACK_DAYS: dict[str, int] = {
    "1m":  10,   # 390 bars/day → 1 trading day needed
    "5m":  15,   # 78 bars/day  → 3 trading days needed
    "15m": 30,   # 26 bars/day  → 8 trading days needed
    "30m": 50,   # 13 bars/day  → 16 trading days needed
    "1h":  80,   # 6.5 bars/day → 31 trading days needed
    "1d":  310,  # 1 bar/day    → 200 trading days needed
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_months(d: date, months: int) -> date:
    """Add `months` calendar months to a date, clamping to month-end if needed."""
    month = d.month - 1 + months
    year  = d.year + month // 12
    month = month % 12 + 1
    import calendar
    last_day = calendar.monthrange(year, month)[1]
    return date(year, month, min(d.day, last_day))


def _walk_forward_folds(oldest: date) -> list[dict]:
    """
    Return a list of dicts describing each fold:
        {fold, is_start, is_end, oos_start, oos_end}
    """
    folds = []
    for i in range(N_FOLDS):
        slide     = SLIDE_MONTHS * i
        is_start  = _add_months(oldest, slide)
        is_end    = _add_months(is_start, IS_MONTHS)
        oos_start = is_end
        oos_end   = _add_months(oos_start, OOS_MONTHS)
        folds.append({
            "fold":      i + 1,
            "is_start":  is_start,
            "is_end":    is_end,
            "oos_start": oos_start,
            "oos_end":   oos_end,
        })
    return folds


def _build_ticker_df(ticker: str, from_date: date, to_date: date,
                     daily_lookup: dict, timeframe: str) -> pd.DataFrame | None:
    """
    Fetch candles for `ticker` at the given `timeframe` between from_date and to_date,
    compute all needed indicators, and return a tidy DataFrame.

    `daily_lookup` maps (ticker, date_str) -> previous_close from stock_data_filtered.
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

    # --- Naive ET datetime index (consistent with strategy code) ---
    dt_et   = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert("America/New_York")
    df["date"]     = pd.to_datetime(dt_et.dt.strftime("%Y-%m-%dT%H:%M:%S"))
    df["date_str"] = dt_et.dt.strftime("%Y-%m-%d")
    df["ticker"]   = ticker

    # --- Indicators (computed on full fetched window for proper warmup) ---
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

    # --- previous_day_close from stock_data_filtered daily summary ---
    df["previous_day_close"] = df["date_str"].map(
        lambda ds: daily_lookup.get((ticker, ds), float("nan"))
    )

    # Keep only the lookback-free columns (drop raw Massive extras)
    keep = ["ticker", "date", "date_str",
            "open", "high", "low", "close", "volume",
            "atr", "RVOL_daily", "SMA_VOLUME_20_5m", "vwap", "previous_day_close",
            "sma_9", "sma_200",
            "donchian_upper", "donchian_lower", "donchian_basis"]
    return df[keep]


def _build_split(
    filtered_df: pd.DataFrame,
    split_name: str,
    start_date: date,
    end_date: date,
    output_path: Path,
    timeframe: str,
) -> None:
    """
    Build and save one parquet (IS or OOS) for the given date window.

    Only tickers/days present in stock_data_filtered are included.
    A lookback window before start_date is fetched for indicator warmup
    but then trimmed from the output.
    """
    # Filter daily summary to this window
    mask = (filtered_df["date_col"] >= start_date) & (filtered_df["date_col"] < end_date)
    window_df = filtered_df[mask]

    if window_df.empty:
        logger.warning("%s: no rows in stock_data_filtered for %s → %s", split_name, start_date, end_date)
        return

    tickers = sorted(window_df["ticker"].unique())
    logger.info("%s: %d tickers, %d ticker-days  (%s → %s)",
                split_name, len(tickers), len(window_df), start_date, end_date)

    # Build (ticker, date_str) → previous_close lookup from the daily summary
    daily_lookup = {
        (row["ticker"], row["date_str"]): row["previous_close"]
        for _, row in window_df.iterrows()
    }

    # Fetch start with extra lookback for indicator warmup (SMA 200 needs more bars for slower TFs)
    lookback_days = _LOOKBACK_DAYS.get(timeframe, 30)
    fetch_from = start_date - timedelta(days=lookback_days)

    parts: list[pd.DataFrame] = []
    for n, ticker in enumerate(tickers, 1):
        if n % 100 == 0:
            logger.info("%s: fetching ticker %d / %d ...", split_name, n, len(tickers))
        try:
            df_5m = _build_ticker_df(ticker, fetch_from, end_date, daily_lookup, timeframe)
            if df_5m is None or df_5m.empty:
                continue
            # Trim to the actual window (discard lookback rows)
            df_5m = df_5m[df_5m["date_str"] >= start_date.isoformat()]
            # Keep only the gapper days (days present in stock_data_filtered)
            ticker_dates = set(window_df.loc[window_df["ticker"] == ticker, "date_str"])
            df_5m = df_5m[df_5m["date_str"].isin(ticker_dates)]
            if not df_5m.empty:
                parts.append(df_5m)
        except Exception as exc:
            logger.warning("%s: failed for %s — %s", split_name, ticker, exc)

    if not parts:
        logger.error("%s: no %s data fetched.", split_name, timeframe)
        return

    result = pd.concat(parts, ignore_index=True).sort_values(["ticker", "date"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False, compression="zstd")
    logger.info("%s: saved %d rows → %s  (%.1f MB)",
                split_name, len(result), output_path,
                output_path.stat().st_size / 1e6)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_all(timeframe: str = "5m") -> None:
    if timeframe not in VALID_TIMEFRAMES:
        raise ValueError(f"Invalid timeframe '{timeframe}'. Valid: {sorted(VALID_TIMEFRAMES)}")

    logger.info("Timeframe: %s", timeframe)

    # 1. Load daily summary from DB
    logger.info("Loading stock_data_filtered from PostgREST...")
    filtered_df = fetch_stock_data_filtered()
    if filtered_df.empty:
        logger.error("stock_data_filtered is empty — aborting.")
        return

    # Normalise date column (PostgREST returns date_str as string)
    if "date_str" not in filtered_df.columns:
        raise ValueError("Expected column 'date_str' in stock_data_filtered.")
    filtered_df["date_col"] = pd.to_datetime(filtered_df["date_str"]).dt.date

    # Rename previous_close if the DB column uses a different name
    if "previous_close" not in filtered_df.columns and "previous_day_close" in filtered_df.columns:
        filtered_df = filtered_df.rename(columns={"previous_day_close": "previous_close"})

    oldest = filtered_df["date_col"].min()
    logger.info("Oldest date in stock_data_filtered: %s", oldest)

    # 2. Compute walk-forward folds
    folds = _walk_forward_folds(oldest)
    for f in folds:
        logger.info(
            "Fold %d — IS: %s → %s   OOS: %s → %s",
            f["fold"], f["is_start"], f["is_end"], f["oos_start"], f["oos_end"],
        )

    # 3. Build each split
    for f in folds:
        fold_dir = OUTPUT_BASE / timeframe / f"fold_{f['fold']}"

        _build_split(
            filtered_df  = filtered_df,
            split_name   = f"fold_{f['fold']}/in_sample",
            start_date   = f["is_start"],
            end_date     = f["is_end"],
            output_path  = fold_dir / "in_sample.parquet",
            timeframe    = timeframe,
        )
        _build_split(
            filtered_df  = filtered_df,
            split_name   = f"fold_{f['fold']}/out_of_sample",
            start_date   = f["oos_start"],
            end_date     = f["oos_end"],
            output_path  = fold_dir / "out_of_sample.parquet",
            timeframe    = timeframe,
        )

    logger.info("Done. Output: %s", OUTPUT_BASE.resolve())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build walk-forward backtest datasets.")
    parser.add_argument("--tf", default="5m", help="Candle timeframe (e.g. 5m, 15m, 1h). Default: 5m")
    args = parser.parse_args()
    build_all(timeframe=args.tf)
