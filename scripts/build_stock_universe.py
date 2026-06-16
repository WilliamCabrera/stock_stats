"""
Step 1 — Build the clean universe of US stock tickers.

Fetches every ticker from Massive's reference endpoint (active + inactive),
removes duplicates, and drops symbols with special characters (warrants,
units, preferred shares, etc.). Only plain alphabetic tickers survive:

    "ABC"      -> kept
    "APH.WD"   -> dropped (dot)
    "AHW$WC"   -> dropped (dollar sign)
    "BRK.B"    -> dropped (dot)

The endpoint paginates via `next_url`; both `active=true` and `active=false`
are fetched so the universe includes listed *and* delisted names. When a
ticker appears in both, the active record wins.

Output:
    backtest_dataset/UNIVERSE/stock_universe.parquet
    backtest_dataset/UNIVERSE/stock_universe.csv   (--csv, on by default)

Usage (from backtester_api/):
    python -m scripts.build_stock_universe
    python -m scripts.build_stock_universe --active all      # default
    python -m scripts.build_stock_universe --active true     # listed only
    python -m scripts.build_stock_universe --no-csv
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path

import httpx
import pandas as pd

sys.path.insert(0, os.path.abspath("."))

from app.config import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("backtest_dataset/UNIVERSE")

# A valid ticker is uppercase letters only — anything with '.', '$', '/', digits
# or other punctuation is a warrant / unit / preferred / non-common-share symbol.
VALID_TICKER_RE = re.compile(r"^[A-Z]+$")

PAGE_LIMIT = 1000

# Reference columns worth keeping if the API returns them.
KEEP_COLUMNS = [
    "ticker", "name", "market", "locale", "primary_exchange",
    "type", "active", "currency_name", "cik", "composite_figi", "list_date",
]


def _fetch_reference_page(client: httpx.Client, url: str, api_key: str) -> tuple[list[dict], str | None]:
    """Fetch one page of /v3/reference/tickers; return (rows, next_url)."""
    resp = client.get(url)
    resp.raise_for_status()
    data = resp.json()
    rows = data.get("results") or []
    next_url = data.get("next_url")
    if next_url:
        next_url = f"{next_url}&apiKey={api_key}"
    return rows, next_url


def fetch_all_tickers(active: str) -> list[dict]:
    """
    Fetch every ticker page for the given `active` filter ("true"/"false").

    Returns the raw reference rows (unfiltered, may contain duplicates across
    pages — the API itself does not duplicate within a single active filter).
    """
    settings = get_settings()
    if not settings.massive_api_key:
        raise ValueError("MASSIVE_API_KEY is not set in the environment.")

    base = settings.massive_base_url.rstrip("/")
    url: str | None = (
        f"{base}/v3/reference/tickers"
        f"?market=stocks&active={active}&order=asc&limit={PAGE_LIMIT}"
        f"&sort=ticker&apiKey={settings.massive_api_key}"
    )

    rows: list[dict] = []
    page = 0
    with httpx.Client(timeout=60) as client:
        while url:
            page += 1
            batch, url = _fetch_reference_page(client, url, settings.massive_api_key)
            rows.extend(batch)
            logger.info("active=%s  page %d  +%d rows  (total %d)", active, page, len(batch), len(rows))
    return rows


def build_universe(active: str) -> pd.DataFrame:
    """
    Build the deduplicated, cleaned ticker universe.

    `active`:
        "all"   -> fetch active=true then active=false (active record wins)
        "true"  -> listed only
        "false" -> delisted only
    """
    if active == "all":
        filters = ["true", "false"]
    else:
        filters = [active]

    # Keyed by ticker; iterate active=true first so it takes precedence on dupes.
    by_ticker: dict[str, dict] = {}
    raw_count = 0
    for flt in filters:
        for row in fetch_all_tickers(flt):
            raw_count += 1
            sym = row.get("ticker")
            if not sym or sym in by_ticker:
                continue
            by_ticker[sym] = row

    df = pd.DataFrame(by_ticker.values())
    logger.info("Collected %d raw rows -> %d unique tickers.", raw_count, len(df))

    # Drop symbols with special characters / non-letter glyphs.
    mask = df["ticker"].str.fullmatch(VALID_TICKER_RE)
    dropped = int((~mask).sum())
    df = df[mask].copy()
    logger.info("Dropped %d tickers with special characters -> %d clean tickers.", dropped, len(df))

    # Keep a stable, useful column subset.
    cols = [c for c in KEEP_COLUMNS if c in df.columns]
    df = df[cols].sort_values("ticker").reset_index(drop=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the clean universe of US stock tickers (active + delisted)."
    )
    parser.add_argument(
        "--active", choices=["all", "true", "false"], default="all",
        help="Which listings to include (default: all = active + delisted).",
    )
    parser.add_argument(
        "--out-dir", default=str(OUTPUT_DIR),
        help=f"Output directory (default: {OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--no-csv", dest="csv", action="store_false",
        help="Skip writing the .csv copy (parquet is always written).",
    )
    args = parser.parse_args()

    df = build_universe(args.active)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = out_dir / "stock_universe.parquet"
    df.to_parquet(parquet_path, index=False, compression="zstd")
    logger.info("Wrote %s (%d tickers, %.2f MB)", parquet_path, len(df), parquet_path.stat().st_size / 1e6)

    if args.csv:
        csv_path = out_dir / "stock_universe.csv"
        df.to_csv(csv_path, index=False)
        logger.info("Wrote %s", csv_path)


if __name__ == "__main__":
    main()
