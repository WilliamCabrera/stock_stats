from __future__ import annotations

import asyncio
import logging
import os
from io import StringIO

import aiohttp
import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = "/app/data/delisted"
BASE_URL = "https://stocklight.com/stocks/us/delisted?page={page}"
TOTAL_PAGES = 1135
CONCURRENCY = 20

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


async def _fetch_page(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    page: int,
) -> pd.DataFrame | None:
    url = BASE_URL.format(page=page)
    async with sem:
        try:
            async with session.get(url, headers=HEADERS, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                resp.raise_for_status()
                html = await resp.text()
        except Exception as e:
            logger.warning("Page %d failed: %s", page, e)
            return None

    try:
        tables = pd.read_html(StringIO(html))
        if not tables:
            return None
        df = tables[0]
        df.columns = [c.strip() for c in df.columns]
        # Keep only the columns we need
        df = df[["Date", "Exchange", "Code", "Name"]].rename(
            columns={"Date": "date", "Exchange": "exchange", "Code": "code", "Name": "name"}
        )
        return df
    except Exception as e:
        logger.warning("Parse error on page %d: %s", page, e)
        return None


async def _fetch_all() -> pd.DataFrame:
    sem = asyncio.Semaphore(CONCURRENCY)
    connector = aiohttp.TCPConnector(limit=CONCURRENCY)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [_fetch_page(session, sem, p) for p in range(1, TOTAL_PAGES + 1)]
        results = []
        for i, coro in enumerate(asyncio.as_completed(tasks), 1):
            df = await coro
            if df is not None:
                results.append(df)
            if i % 100 == 0:
                logger.info("Progress: %d / %d pages fetched", i, TOTAL_PAGES)
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def fetch_delisted_all() -> pd.DataFrame:
    """
    Scrape all delisted US stocks from stocklight.com (1135 pages).
    Returns a DataFrame with columns: date, exchange, code, name.
    Saves result to /app/data/delisted/delisted_stocklight.csv.
    """
    logger.info("Starting full scrape of %d pages from stocklight.com", TOTAL_PAGES)
    df = asyncio.run(_fetch_all())

    if df.empty:
        logger.error("No data retrieved")
        return df

    # Drop duplicates that might appear across pages
    df = df.drop_duplicates(subset=["code", "date"]).reset_index(drop=True)

    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = os.path.join(DATA_DIR, "delisted_stocklight.csv")
    df.to_csv(out_path, index=False)
    logger.info("Saved %d rows to %s", len(df), out_path)

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = fetch_delisted_all()
    print(f"\nTotal rows: {len(df)}")
    print(df.head(10).to_string(index=False))
