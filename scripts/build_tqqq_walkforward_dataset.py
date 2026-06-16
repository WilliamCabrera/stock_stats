"""
Build walk-forward folds for the TQQQ index dataset.

Slices backtest_dataset/INDICES/TQQQ/{5m,1h,1d}/tqqq_full_dataset.parquet
(no API re-fetch) into the same fold structure as backtest_dataset/walkforward:

    Fold 1: IS [d0,        d0+24M)   OOS [d0+24M, d0+36M)
    Fold 2: IS [d0+12M,    d0+36M)   OOS [d0+36M, d0+48M)
    Fold 3: IS [d0+24M,    d0+48M)   OOS [d0+48M, d0+60M)

Window sizes keep the same proportions as the stock walkforward
(IS = 2 × OOS, slide = OOS) but scaled so 3 folds cover the full
5-year TQQQ range.

Output layout:
    backtest_dataset/INDICES/TQQQ/walkforward/{timeframe}/
        fold_1/in_sample.parquet
        fold_1/out_of_sample.parquet
        fold_2/...
        fold_3/...

Usage (from backtester_api/):
    python -m scripts.build_tqqq_walkforward_dataset                  # both 5m and 1h
    python -m scripts.build_tqqq_walkforward_dataset --timeframe 5m
    python -m scripts.build_tqqq_walkforward_dataset --timeframe 1h

    # Custom windows (months)
    python -m scripts.build_tqqq_walkforward_dataset --is-months 12 --oos-months 6 --slide-months 6
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
TICKER       = "TQQQ"
INPUT_BASE   = Path("backtest_dataset/INDICES") / TICKER
INPUT_NAME   = "tqqq_full_dataset.parquet"
OUTPUT_BASE  = INPUT_BASE / "walkforward"
TIMEFRAMES   = ["5m", "10m", "1h", "1d"]

IS_MONTHS    = 24   # in-sample window length (months)
OOS_MONTHS   = 12   # out-of-sample window length (months)
SLIDE_MONTHS = 12   # how many months to slide each fold
N_FOLDS      = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_months(d: date, months: int) -> date:
    """Add `months` calendar months to a date, clamping to month-end if needed."""
    month = d.month - 1 + months
    year  = d.year + month // 12
    month = month % 12 + 1
    last_day = calendar.monthrange(year, month)[1]
    return date(year, month, min(d.day, last_day))


def _walk_forward_folds(
    oldest: date,
    is_months: int,
    oos_months: int,
    slide_months: int,
    n_folds: int,
) -> list[dict]:
    """
    Return a list of dicts describing each fold:
        {fold, is_start, is_end, oos_start, oos_end}
    """
    folds = []
    for i in range(n_folds):
        slide     = slide_months * i
        is_start  = _add_months(oldest, slide)
        is_end    = _add_months(is_start, is_months)
        oos_start = is_end
        oos_end   = _add_months(oos_start, oos_months)
        folds.append({
            "fold":      i + 1,
            "is_start":  is_start,
            "is_end":    is_end,
            "oos_start": oos_start,
            "oos_end":   oos_end,
        })
    return folds


def _write_split(
    df: pd.DataFrame,
    split_name: str,
    start_date: date,
    end_date: date,
    output_path: Path,
) -> None:
    """Slice df to [start_date, end_date) by date_str and write the parquet."""
    mask  = (df["date_str"] >= start_date.isoformat()) & (df["date_str"] < end_date.isoformat())
    split = df[mask].sort_values("date").reset_index(drop=True)

    if split.empty:
        logger.warning("%s: no rows for %s → %s — skipped.", split_name, start_date, end_date)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    split.to_parquet(output_path, index=False, compression="zstd")
    logger.info(
        "%s: %d rows  (%s → %s)  → %s  (%.1f MB)",
        split_name, len(split), split["date_str"].min(), split["date_str"].max(),
        output_path, output_path.stat().st_size / 1e6,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_timeframe(
    timeframe: str,
    is_months: int,
    oos_months: int,
    slide_months: int,
    n_folds: int,
) -> None:
    """Slice the full TQQQ dataset for one timeframe into walk-forward folds."""
    input_path = INPUT_BASE / timeframe / INPUT_NAME
    if not input_path.exists():
        logger.error(
            "%s not found — run `make tqqq-dataset` first to build the full dataset.",
            input_path,
        )
        return

    df = pd.read_parquet(input_path)
    oldest = date.fromisoformat(df["date_str"].min())
    newest = date.fromisoformat(df["date_str"].max())
    logger.info(
        "%s %s: %d rows, %s → %s", TICKER, timeframe, len(df), oldest, newest,
    )

    folds = _walk_forward_folds(oldest, is_months, oos_months, slide_months, n_folds)
    for f in folds:
        logger.info(
            "Fold %d — IS: %s → %s   OOS: %s → %s",
            f["fold"], f["is_start"], f["is_end"], f["oos_start"], f["oos_end"],
        )

    for f in folds:
        fold_dir = OUTPUT_BASE / timeframe / f"fold_{f['fold']}"
        _write_split(
            df, f"{timeframe}/fold_{f['fold']}/in_sample",
            f["is_start"], f["is_end"], fold_dir / "in_sample.parquet",
        )
        _write_split(
            df, f"{timeframe}/fold_{f['fold']}/out_of_sample",
            f["oos_start"], f["oos_end"], fold_dir / "out_of_sample.parquet",
        )

    # Final OOS: everything in the full dataset after the last fold's OOS end.
    # Grows automatically as new data is fetched via make tqqq-dataset.
    last_oos_end = folds[-1]["oos_end"]
    final_oos = df[df["date_str"] >= last_oos_end.isoformat()].sort_values("date").reset_index(drop=True)
    out_path = OUTPUT_BASE / timeframe / "final_oos.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final_oos.to_parquet(out_path, index=False, compression="zstd")
    if final_oos.empty:
        logger.info("%s final_oos: no data yet after %s (populate by running make tqqq-dataset)", timeframe, last_oos_end)
    else:
        logger.info(
            "%s final_oos: %d rows  (%s → %s)  → %s",
            timeframe, len(final_oos), final_oos["date_str"].min(), final_oos["date_str"].max(), out_path,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build TQQQ walk-forward folds from the full TQQQ dataset."
    )
    parser.add_argument(
        "--timeframe",
        choices=TIMEFRAMES,
        help="Timeframe to build. Omit to build all (5m, 1h, 1d).",
    )
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
