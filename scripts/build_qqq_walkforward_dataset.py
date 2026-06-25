"""
Build walk-forward folds for the QQQ index dataset.

Slices backtest_dataset/INDICES/QQQ/{5m,10m,1h,1d}/qqq_full_dataset.parquet
(no API re-fetch) into walk-forward folds:

    Fold 1: IS [d0,      d0+24M)   OOS [d0+24M, d0+36M)
    Fold 2: IS [d0+12M,  d0+36M)   OOS [d0+36M, d0+48M)
    Fold 3: IS [d0+24M,  d0+48M)   OOS [d0+48M, d0+60M)

Output layout:
    backtest_dataset/INDICES/QQQ/walkforward/{timeframe}/
        fold_1/in_sample.parquet
        fold_1/out_of_sample.parquet
        fold_2/...
        fold_3/...
        final_oos.parquet   ← data after last fold's OOS end

Usage (from backtester_api/):
    python -m scripts.build_qqq_walkforward_dataset          # all timeframes
    python -m scripts.build_qqq_walkforward_dataset --timeframe 5m
    python -m scripts.build_qqq_walkforward_dataset --timeframe 10m
    python -m scripts.build_qqq_walkforward_dataset --timeframe 1h
    python -m scripts.build_qqq_walkforward_dataset --timeframe 1d

    # Custom windows (months)
    python -m scripts.build_qqq_walkforward_dataset --is-months 12 --oos-months 6 --slide-months 6

Requires qqq_full_dataset.parquet (run make qqq-dataset first).
"""
from __future__ import annotations

import argparse
import calendar
import logging
from datetime import date
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TICKER       = "QQQ"
BASE         = Path("backtest_dataset/INDICES/QQQ")
NAME         = "qqq_full_dataset.parquet"
WF_BASE      = BASE / "walkforward"
TIMEFRAMES   = ["5m", "10m", "1h", "1d"]

IS_MONTHS    = 24
OOS_MONTHS   = 12
SLIDE_MONTHS = 12
N_FOLDS      = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_months(d: date, months: int) -> date:
    month    = d.month - 1 + months
    year     = d.year + month // 12
    month    = month % 12 + 1
    last_day = calendar.monthrange(year, month)[1]
    return date(year, month, min(d.day, last_day))


def _walk_forward_folds(oldest: date, is_m: int, oos_m: int, slide_m: int, n: int) -> list[dict]:
    return [{"fold":      i + 1,
             "is_start":  _add_months(oldest, slide_m * i),
             "is_end":    _add_months(oldest, slide_m * i + is_m),
             "oos_start": _add_months(oldest, slide_m * i + is_m),
             "oos_end":   _add_months(oldest, slide_m * i + is_m + oos_m)}
            for i in range(n)]


def _write_split(df: pd.DataFrame, label: str, start: date, end: date, path: Path) -> None:
    sl = (df[(df["date_str"] >= start.isoformat()) & (df["date_str"] < end.isoformat())]
            .sort_values("date").reset_index(drop=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    sl.to_parquet(path, index=False, compression="zstd")
    if sl.empty:
        logger.warning("%s: no rows for %s → %s — skipped.", label, start, end)
    else:
        logger.info("%s: %d rows  (%s → %s)  → %s  (%.1f MB)",
                    label, len(sl), sl["date_str"].min(), sl["date_str"].max(),
                    path, path.stat().st_size / 1e6)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_timeframe(timeframe: str, is_m: int, oos_m: int, slide_m: int, n_folds: int) -> None:
    input_path = BASE / timeframe / NAME
    if not input_path.exists():
        logger.error("%s not found — run `make qqq-dataset` first.", input_path)
        return

    df     = pd.read_parquet(input_path)
    oldest = date.fromisoformat(df["date_str"].min())
    logger.info("%s %s: %d rows, %s → %s",
                TICKER, timeframe, len(df), oldest, df["date_str"].max())

    folds = _walk_forward_folds(oldest, is_m, oos_m, slide_m, n_folds)
    for f in folds:
        logger.info("Fold %d — IS: %s → %s   OOS: %s → %s",
                    f["fold"], f["is_start"], f["is_end"], f["oos_start"], f["oos_end"])

    for f in folds:
        fd = WF_BASE / timeframe / f"fold_{f['fold']}"
        _write_split(df, f"{timeframe}/fold_{f['fold']}/in_sample",
                     f["is_start"], f["is_end"], fd / "in_sample.parquet")
        _write_split(df, f"{timeframe}/fold_{f['fold']}/out_of_sample",
                     f["oos_start"], f["oos_end"], fd / "out_of_sample.parquet")

    last_end = folds[-1]["oos_end"]
    final    = df[df["date_str"] >= last_end.isoformat()].sort_values("date").reset_index(drop=True)
    fp       = WF_BASE / timeframe / "final_oos.parquet"
    fp.parent.mkdir(parents=True, exist_ok=True)
    final.to_parquet(fp, index=False, compression="zstd")
    if final.empty:
        logger.info("%s final_oos: no data yet after %s (populate by running make qqq-dataset)", timeframe, last_end)
    else:
        logger.info("%s final_oos: %d rows  (%s → %s)  → %s",
                    timeframe, len(final), final["date_str"].min(), final["date_str"].max(), fp)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build QQQ walk-forward folds from the full QQQ dataset."
    )
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

    timeframes = [args.timeframe] if args.timeframe else TIMEFRAMES
    for tf in timeframes:
        build_timeframe(tf, args.is_months, args.oos_months, args.slide_months, args.folds)


if __name__ == "__main__":
    main()
