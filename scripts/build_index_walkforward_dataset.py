"""
Build walk-forward folds for any index or large-cap ticker dataset.

Slices backtest_dataset/INDICES/{TICKER}/{tf}/{ticker_lower}_full_dataset.parquet
(no API re-fetch) into walk-forward folds:

    Fold 1: IS [d0,        d0+IS)    OOS [d0+IS,         d0+IS+OOS)
    Fold 2: IS [d0+slide,  ...]      OOS [d0+slide+IS,   ...]
    ...

Default windows: IS=24M / OOS=12M / slide=12M → 3 folds covering a 5-year range.

Output:
    backtest_dataset/INDICES/{TICKER}/walkforward/{tf}/
        fold_1/in_sample.parquet
        fold_1/out_of_sample.parquet
        fold_2/...
        fold_3/...
        final_oos.parquet   ← everything after the last fold's OOS end

Usage (from backtester_api/):
    python -m scripts.build_index_walkforward_dataset --ticker SPY
    python -m scripts.build_index_walkforward_dataset --ticker AAPL --timeframe 5m
    python -m scripts.build_index_walkforward_dataset --ticker QQQ --is-months 12 --oos-months 6 --slide-months 6

Requires {ticker_lower}_full_dataset.parquet — run build_index_dataset.py first.
"""
from __future__ import annotations

import argparse
import calendar
import logging
import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.abspath("."))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TIMEFRAMES   = ["1d", "1h", "5m", "10m"]
IS_MONTHS    = 24
OOS_MONTHS   = 12
SLIDE_MONTHS = 12
N_FOLDS      = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parquet_name(ticker: str) -> str:
    return f"{ticker.lower()}_full_dataset.parquet"


def _base(ticker: str) -> Path:
    return Path("backtest_dataset/INDICES") / ticker.upper()


def _add_months(d: date, months: int) -> date:
    month    = d.month - 1 + months
    year     = d.year + month // 12
    month    = month % 12 + 1
    last_day = calendar.monthrange(year, month)[1]
    return date(year, month, min(d.day, last_day))


def _folds(oldest: date, is_m: int, oos_m: int, slide_m: int, n: int) -> list[dict]:
    return [
        {
            "fold":      i + 1,
            "is_start":  _add_months(oldest, slide_m * i),
            "is_end":    _add_months(oldest, slide_m * i + is_m),
            "oos_start": _add_months(oldest, slide_m * i + is_m),
            "oos_end":   _add_months(oldest, slide_m * i + is_m + oos_m),
        }
        for i in range(n)
    ]


def _write_split(df: pd.DataFrame, label: str, start: date, end: date, path: Path) -> None:
    sl = (df[(df["date_str"] >= start.isoformat()) & (df["date_str"] < end.isoformat())]
            .sort_values("date").reset_index(drop=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    sl.to_parquet(path, index=False, compression="zstd")
    if sl.empty:
        logger.warning("%s: no rows for %s → %s — written empty.", label, start, end)
    else:
        logger.info("%s: %d rows  (%s → %s)  → %s  (%.1f MB)",
                    label, len(sl), sl["date_str"].min(), sl["date_str"].max(),
                    path, path.stat().st_size / 1e6)


# ---------------------------------------------------------------------------
# Per-timeframe builder
# ---------------------------------------------------------------------------

def build_timeframe(
    ticker: str, tf: str,
    is_m: int, oos_m: int, slide_m: int, n_folds: int,
) -> None:
    input_path = _base(ticker) / tf / _parquet_name(ticker)
    if not input_path.exists():
        logger.error(
            "%s not found — run `python -m scripts.build_index_dataset --ticker %s` first.",
            input_path, ticker,
        )
        return

    df     = pd.read_parquet(input_path)
    oldest = date.fromisoformat(df["date_str"].min())
    logger.info("%s %s: %d rows, %s → %s", ticker, tf, len(df), oldest, df["date_str"].max())

    wf_base = _base(ticker) / "walkforward" / tf
    fold_defs = _folds(oldest, is_m, oos_m, slide_m, n_folds)

    for f in fold_defs:
        logger.info("Fold %d — IS: %s → %s   OOS: %s → %s",
                    f["fold"], f["is_start"], f["is_end"], f["oos_start"], f["oos_end"])

    for f in fold_defs:
        fd = wf_base / f"fold_{f['fold']}"
        _write_split(df, f"{tf}/fold_{f['fold']}/in_sample",
                     f["is_start"], f["is_end"], fd / "in_sample.parquet")
        _write_split(df, f"{tf}/fold_{f['fold']}/out_of_sample",
                     f["oos_start"], f["oos_end"], fd / "out_of_sample.parquet")

    last_end = fold_defs[-1]["oos_end"]
    final    = df[df["date_str"] >= last_end.isoformat()].sort_values("date").reset_index(drop=True)
    fp       = wf_base / "final_oos.parquet"
    fp.parent.mkdir(parents=True, exist_ok=True)
    final.to_parquet(fp, index=False, compression="zstd")
    if final.empty:
        logger.info("%s %s final_oos: no data yet after %s", ticker, tf, last_end)
    else:
        logger.info("%s %s final_oos: %d rows  (%s → %s)  → %s",
                    ticker, tf, len(final), final["date_str"].min(), final["date_str"].max(), fp)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build walk-forward folds for any index or large-cap ticker."
    )
    parser.add_argument("--ticker", required=True,
                        help="Ticker symbol, e.g. SPY, AAPL, TQQQ.")
    parser.add_argument("--timeframe", choices=TIMEFRAMES,
                        help="Timeframe to build. Omit to build all.")
    parser.add_argument("--is-months",    type=int, default=IS_MONTHS,
                        help=f"In-sample window in months (default: {IS_MONTHS}).")
    parser.add_argument("--oos-months",   type=int, default=OOS_MONTHS,
                        help=f"Out-of-sample window in months (default: {OOS_MONTHS}).")
    parser.add_argument("--slide-months", type=int, default=SLIDE_MONTHS,
                        help=f"Slide between folds in months (default: {SLIDE_MONTHS}).")
    parser.add_argument("--folds",        type=int, default=N_FOLDS,
                        help=f"Number of folds (default: {N_FOLDS}).")
    args = parser.parse_args()

    ticker = args.ticker.upper()
    tfs    = [args.timeframe] if args.timeframe else TIMEFRAMES

    for tf in tfs:
        build_timeframe(ticker, tf, args.is_months, args.oos_months, args.slide_months, args.folds)


if __name__ == "__main__":
    main()
