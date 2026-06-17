"""
Enrich every trade parquet under strategies/iterative with market_cap and float.

Reads the lookup from backtest_dataset/STOCKS/stock_dataset.parquet on (ticker, date_str)
and joins it to every *.parquet found under strategies/iterative (recursively).
Files are rewritten atomically in-place. Already-enriched files are skipped unless
--force is passed.

Usage:
    python -m scripts.enrich_trades_with_fundamentals
    python -m scripts.enrich_trades_with_fundamentals --force
    python -m scripts.enrich_trades_with_fundamentals --trades-dir strategies/iterative/UP-TO-DATE
    python -m scripts.enrich_trades_with_fundamentals --stock-dataset path/to/stock_dataset.parquet
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_STOCK_DATASET = Path("backtest_dataset/STOCKS/stock_dataset.parquet")
DEFAULT_TRADES_DIR    = Path("strategies/iterative")


def load_lookup(stock_dataset: Path) -> pd.DataFrame:
    logger.info("Loading market_cap/float lookup from %s …", stock_dataset)
    df = pd.read_parquet(stock_dataset, columns=["ticker", "date_str", "market_cap", "float"])
    # Keep most-recent row per (ticker, date_str) in case of duplicates
    df = df.drop_duplicates(subset=["ticker", "date_str"], keep="last")
    logger.info("Lookup: %d rows, %d unique tickers", len(df), df["ticker"].nunique())
    return df


def enrich_file(path: Path, lookup: pd.DataFrame, force: bool) -> bool:
    trades = pd.read_parquet(path)

    if not force and "market_cap" in trades.columns and "float" in trades.columns:
        logger.info("  skip (already enriched): %s", path)
        return False

    n_before = len(trades)

    # Drop any stale columns so the merge is clean
    trades = trades.drop(columns=["market_cap", "float"], errors="ignore")

    enriched = trades.merge(lookup, on=["ticker", "date_str"], how="left")
    assert len(enriched) == n_before, f"Row count changed after merge: {n_before} → {len(enriched)}"

    matched = enriched["market_cap"].notna().sum()
    logger.info(
        "  %s: %d/%d rows matched (%.1f%%)",
        path.name, matched, n_before, 100 * matched / n_before if n_before else 0,
    )

    # Atomic write: write to .tmp then rename
    tmp = path.with_suffix(".parquet.tmp")
    enriched.to_parquet(tmp, index=False, compression="zstd")
    os.replace(tmp, path)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Add market_cap + float to all trade parquets.")
    parser.add_argument("--stock-dataset", default=str(DEFAULT_STOCK_DATASET),
                        help=f"Stock dataset parquet (default: {DEFAULT_STOCK_DATASET}).")
    parser.add_argument("--trades-dir", default=str(DEFAULT_TRADES_DIR),
                        help=f"Root dir to search for trade parquets (default: {DEFAULT_TRADES_DIR}).")
    parser.add_argument("--force", action="store_true",
                        help="Re-enrich even files that already have market_cap/float columns.")
    args = parser.parse_args()

    lookup = load_lookup(Path(args.stock_dataset))

    trade_files = sorted(Path(args.trades_dir).rglob("*.parquet"))
    logger.info("Found %d parquet files under %s", len(trade_files), args.trades_dir)

    updated = skipped = 0
    for f in trade_files:
        try:
            changed = enrich_file(f, lookup, force=args.force)
            if changed:
                updated += 1
            else:
                skipped += 1
        except Exception as exc:
            logger.error("  ERROR enriching %s: %s", f, exc)

    logger.info("Done. updated=%d skipped=%d", updated, skipped)


if __name__ == "__main__":
    main()
