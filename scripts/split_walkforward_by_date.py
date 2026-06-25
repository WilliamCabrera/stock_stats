"""
Split walkforward in_sample / out_of_sample parquets into one file per date.

Output structure:
  backtest_dataset/walkforward/<timeframe>/fold_<n>/dates_IS/<date>.parquet
  backtest_dataset/walkforward/<timeframe>/fold_<n>/dates_OOS/<date>.parquet

Usage (from backtester_api/):
    python -m scripts.split_walkforward_by_date                  # all timeframes, all folds
    python -m scripts.split_walkforward_by_date --timeframe 5m
    python -m scripts.split_walkforward_by_date --timeframe 15m --fold 2
"""
import argparse
import sys
import time as tm
from pathlib import Path

import pandas as pd

WF_ROOT = Path("backtest_dataset/walkforward")


def split_parquet_by_date(input_path: Path, output_dir: Path) -> int:
    if not input_path.exists():
        print(f"  [SKIP] Not found: {input_path}")
        return 0

    df = pd.read_parquet(input_path)
    df = df.sort_values(["date_str", "ticker", "date"]).reset_index(drop=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_dates = df["date_str"].nunique()
    start = tm.perf_counter()

    for i, (date_str, group) in enumerate(df.groupby("date_str", sort=True), 1):
        filename = date_str.replace("-", "_") + ".parquet"
        group.reset_index(drop=True).to_parquet(output_dir / filename, index=False)

        if i % 100 == 0 or i == n_dates:
            print(f"    {i}/{n_dates} dates  ({tm.perf_counter() - start:.1f}s)")

    return n_dates


def split_fold(timeframe: str, fold: str) -> None:
    fold_dir = WF_ROOT / timeframe / fold
    print(f"\n── {timeframe}/{fold} ──")

    for split, out_name in [("in_sample", "dates_IS"), ("out_of_sample", "dates_OOS")]:
        src = fold_dir / f"{split}.parquet"
        dst = fold_dir / out_name
        print(f"  {split}.parquet → {out_name}/")
        n = split_parquet_by_date(src, dst)
        if n:
            print(f"  ✓ {n} archivos escritos en {dst}")


def main(timeframe: str | None, fold: int | None) -> None:
    timeframes = [timeframe] if timeframe else ["5m", "15m"]

    for tf in timeframes:
        tf_dir = WF_ROOT / tf
        if not tf_dir.exists():
            print(f"[SKIP] {tf_dir} no existe")
            continue

        folds = ([f"fold_{fold}"] if fold else
                 sorted(p.name for p in tf_dir.iterdir()
                        if p.is_dir() and p.name.startswith("fold_")))

        for f in folds:
            split_fold(tf, f)

    print("\nListo.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split walkforward IS/OOS parquets into one file per date."
    )
    parser.add_argument("--timeframe", type=str, default=None,
                        help="'5m' o '15m' (default: ambos)")
    parser.add_argument("--fold", type=int, default=None,
                        help="Número de fold: 1, 2 o 3 (default: todos)")
    args = parser.parse_args()
    main(args.timeframe, args.fold)
