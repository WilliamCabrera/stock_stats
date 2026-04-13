"""
Generic market data utilities.

Functions that are not tied to any specific data provider and can be
reused across different parts of the application.
"""
from __future__ import annotations

import json
import logging
import re
from calendar import monthrange
from datetime import date, datetime, time as dtime, time
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd


import asyncio
import aiohttp

logger = logging.getLogger(__name__)

_MONTHS_BACK = 59   # 4 years and 11 months
_TICKERS_CSV = Path(__file__).parent / "all_tickers_merged.csv"
_TICKER_RE   = re.compile(r"^[A-Za-z]+$")

_ET = ZoneInfo("America/New_York")


def _months_ago(d: date, months: int) -> date:
    """Return the date that is exactly ``months`` calendar months before ``d``.

    If the resulting month has fewer days than ``d.day``, the last day of that
    month is used (e.g. March 31 − 1 month → February 28/29).
    """
    total   = d.year * 12 + d.month - months
    year    = (total - 1) // 12
    month   = (total - 1) % 12 + 1
    _, last = monthrange(year, month)
    return date(year, month, min(d.day, last))


def chunk_date_range(
    ticker: str,
    from_date: str | date,
    to_date: str | date,
) -> list[tuple[str, str, str]]:
    """
    Split a date range into monthly chunks aligned to calendar months.

    The first chunk starts on from_date and ends on the last day of that
    calendar month.  Subsequent chunks cover full calendar months.  The
    last chunk ends on to_date (which may be before the month's end).

    Args:
        ticker:    Equity symbol, e.g. "NVDA".
        from_date: Start date. Accepts "YYYY-MM-DD" or a date object.
        to_date:   End date (inclusive). Accepts "YYYY-MM-DD" or a date object.

    Returns:
        List of (ticker, from_str, to_str) tuples with "YYYY-MM-DD" strings.

    Example:
        chunk_date_range("NVDA", "2020-03-12", "2020-05-13")
        → [
            ("NVDA", "2020-03-12", "2020-03-31"),
            ("NVDA", "2020-04-01", "2020-04-30"),
            ("NVDA", "2020-05-01", "2020-05-13"),
          ]
    """
    if isinstance(from_date, str):
        from_date = date.fromisoformat(from_date[:10])
    if isinstance(to_date, str):
        to_date = date.fromisoformat(to_date[:10])

    chunks: list[tuple[str, str, str]] = []
    chunk_start = from_date

    while chunk_start <= to_date:
        _, last_day = monthrange(chunk_start.year, chunk_start.month)
        month_end = date(chunk_start.year, chunk_start.month, last_day)
        chunk_end = min(month_end, to_date)

        chunks.append((
            ticker,
            chunk_start.strftime("%Y-%m-%d"),
            chunk_end.strftime("%Y-%m-%d"),
        ))

        # Advance to the first day of the next month
        if chunk_start.month == 12:
            chunk_start = date(chunk_start.year + 1, 1, 1)
        else:
            chunk_start = date(chunk_start.year, chunk_start.month + 1, 1)

    return chunks


def ticker_chunks(
    tickers: list[str] | pd.DataFrame,
    months_back: int = _MONTHS_BACK,
) -> list[tuple[str, str, str]]:
    """
    Return monthly date-range chunks for every ticker in a DataFrame.

    The date range spans from (today - months_back months) to today.

    Args:
        tickers_df:  DataFrame with at least a ``ticker`` column
                     (same schema as all_tickers_merged.csv).
        months_back: How many calendar months of history to request
                     (default: 59 = 4 years and 11 months).

    Returns:
        Flat list of (ticker, from_str, to_str) tuples, one per monthly chunk
        per ticker.

    Example:
        >>> import pandas as pd
        >>> df = pd.read_csv("all_tickers_merged.csv")
        >>> chunks = ticker_chunks(df)
        >>> chunks[0]
        ('A', '2021-03-28', '2021-03-31')
    """
    to_date   = date.today()
    from_date = _months_ago(to_date, months_back)

    tickers = tickers_df["ticker"].dropna().unique().tolist() if isinstance(tickers, pd.Series) else tickers

    result: list[tuple[str, str, str]] = []
    for ticker in tickers:
        result.extend(chunk_date_range(ticker, from_date, to_date))

    return result


def fetch_live_tickers(api_key: str, base_url: str) -> set[str]:
    """
    Fetch the full active-stock ticker list from Massive (paginated).

    Returns a set of uppercase ticker symbols (A-Z only).
    Raises on connection errors so the caller can decide how to handle them.
    """
    import httpx

    list_url: str | None = (
        f"{base_url}/v3/reference/tickers"
        f"?market=stocks&active=true&order=asc&limit=1000&sort=ticker"
        f"&apiKey={api_key}"
    )
    live: set[str] = set()
    with httpx.Client(timeout=30) as client:
        while list_url:
            resp = client.get(list_url)
            resp.raise_for_status()
            data = resp.json()
            for item in data.get("results") or []:
                t = item.get("ticker", "")
                if _TICKER_RE.match(t):
                    live.add(t.upper())
            next_url = data.get("next_url")
            list_url = f"{next_url}&apiKey={api_key}" if next_url else None

    logger.info("fetch_live_tickers: %d active tickers from Massive.", len(live))
    return live


def sync_tickers(
    csv_path: str | Path = _TICKERS_CSV,
) -> pd.DataFrame:
    """
    Sync the local ticker CSV against the Massive/Polygon active-stocks list.

    Steps:
        1. Load the existing ticker list from ``csv_path``.
        2. Fetch all active stock tickers from the reference/tickers endpoint
           (auto-paginated).
        3. Identify tickers not yet in the CSV that contain only A-Z letters.
        4. For each new ticker fetch its detail endpoint to obtain
           ``primary_exchange``.
        5. Append new rows, save the updated CSV, and return the DataFrame.

    Args:
        csv_path: Path to ``all_tickers_merged.csv`` (or equivalent).

    Returns:
        Updated DataFrame with columns: ticker, company_name, stock_market.
    """
    import httpx
    from app.config import get_settings

    settings  = get_settings()
    api_key   = settings.massive_api_key
    base_url  = settings.massive_base_url

    # 1. Load existing tickers ------------------------------------------------
    csv_path = Path(csv_path)
    existing = pd.read_csv(csv_path)
    known    = set(existing["ticker"].dropna().str.upper())

    # 2. Fetch full active-stock list (paginated) ------------------------------
    import httpx
    list_url: str | None = (
        f"{base_url}/v3/reference/tickers"
        f"?market=stocks&active=true&order=asc&limit=1000&sort=ticker"
        f"&apiKey={api_key}"
    )
    fetched: list[dict] = []
    try:
        with httpx.Client(timeout=30) as client:
            while list_url:
                resp = client.get(list_url)
                resp.raise_for_status()
                data = resp.json()
                fetched.extend(data.get("results") or [])
                next_url  = data.get("next_url")
                list_url  = f"{next_url}&apiKey={api_key}" if next_url else None
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as exc:
        logger.warning("sync_tickers: no se pudo conectar a la API (%s) — usando CSV existente.", exc)
        return existing

    # 3. Find genuinely new tickers (A-Z only) ---------------------------------
    candidates: list[dict] = []
    for item in fetched:
        t = item.get("ticker", "")
        if _TICKER_RE.match(t) and t.upper() not in known:
            candidates.append({"ticker": t, "company_name": item.get("name", "")})

    if not candidates:
        return existing

    # 4. Fetch primary_exchange for each new ticker ----------------------------
    with httpx.Client(timeout=30) as client:
        for row in candidates:
            try:
                resp = client.get(
                    f"{base_url}/v3/reference/tickers/{row['ticker']}",
                    params={"apiKey": api_key},
                )
                resp.raise_for_status()
                row["stock_market"] = (
                    resp.json().get("results", {}).get("primary_exchange", "")
                )
            except Exception:
                row["stock_market"] = ""

    # 5. Merge and save --------------------------------------------------------
    new_df  = pd.DataFrame(candidates, columns=["ticker", "company_name", "stock_market"])
    updated = pd.concat([existing, new_df], ignore_index=True)
    updated.to_csv(csv_path, index=False)

    return updated



def _save_state(state_file: str, failed_tickers: list[dict]) -> None:
    """Persist failed ticker details to a JSON file for the next run."""
    try:
        data = {
            "count":     len(failed_tickers),
            "timestamp": datetime.now(_ET).isoformat(),
            "failures":  failed_tickers,
        }
        with open(state_file, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(
            "state: saved %d failed tickers to %s",
            len(failed_tickers), state_file,
        )
    except Exception as exc:
        logger.warning("state: could not save %s — %s", state_file, exc)

# ── Session boundaries (America/New_York local time) ──────────────────────────
_ET = ZoneInfo("America/New_York")

_PM_START  = dtime(4,  0)   # pre-market  04:00 ET
_PM_END    = dtime(9, 30)   # pre-market  ends before 09:30 ET
_MH_START  = dtime(9, 30)   # market hours 09:30 ET
_MH_END    = dtime(16,  0)  # market hours ends before 16:00 ET
_AH_START  = dtime(16,  0)  # after hours  16:00 ET
_AH_END    = dtime(20,  0)  # after hours  20:00 ET (inclusive)

_KEY = ["ticker", "date"]


def daily_sessions(df: "pd.DataFrame", time_unit: str = "ms") -> "pd.DataFrame":
    """
    Aggregate intraday candles into per-day, per-ticker session summaries.

    Candles are classified into three sessions using America/New_York time:
        Pre-market  (pm):  04:00 – 09:29 ET
        Market hours (mh): 09:30 – 15:59 ET
        After hours  (ah): 16:00 – 20:00 ET

    Args:
        df:        DataFrame with columns:
                   ticker, open, high, low, close, volume, time
        time_unit: Unit of the `time` column — "ms" (milliseconds, default)
                   or "s" (seconds).

    Returns:
        One row per (ticker, date) with columns:
            high_pm, low_pm, premarket_volume
            day_open, day_close, high_mh, day_low
            ah_open, ah_close, ah_high, ah_low, ah_volume
            ah_range      (= ah_high  - ah_open)
            ah_range_pct  (= ah_range / ah_open * 100)
    """
    import pandas as pd

    df = df.copy()

    # ── Convert timestamps to ET local time ───────────────────────────────────
    df["dt_et"] = (
        pd.to_datetime(df["time"], unit=time_unit, utc=True)
        .dt.tz_convert(_ET)
    )
    df["date"]    = df["dt_et"].dt.date
    df["time_et"] = df["dt_et"].dt.time

    # ── Classify each candle into a session ───────────────────────────────────
    t = df["time_et"]
    df["session"] = None
    df.loc[(t >= _PM_START) & (t < _PM_END),  "session"] = "pm"
    df.loc[(t >= _MH_START) & (t < _MH_END),  "session"] = "mh"
    df.loc[(t >= _AH_START) & (t <= _AH_END), "session"] = "ah"

    # Sort so that first/last aggregations follow chronological order
    df = df.sort_values(["ticker", "date", "dt_et"])

    # ── Pre-market ────────────────────────────────────────────────────────────
    pm_agg = (
        df[df["session"] == "pm"]
        .groupby(_KEY)
        .agg(
            high_pm           = ("high",   "max"),
            low_pm            = ("low",    "min"),
            premarket_volume  = ("volume", "sum"),
        )
    )

    # ── Market hours ──────────────────────────────────────────────────────────
    mh_agg = (
        df[df["session"] == "mh"]
        .groupby(_KEY)
        .agg(
            day_open  = ("open",  "first"),
            day_close = ("close", "last"),
            high_mh   = ("high",  "max"),
            day_low   = ("low",   "min"),
        )
    )

    # ── After hours ───────────────────────────────────────────────────────────
    ah_agg = (
        df[df["session"] == "ah"]
        .groupby(_KEY)
        .agg(
            ah_open   = ("open",   "first"),
            ah_close  = ("close",  "last"),
            ah_high   = ("high",   "max"),
            ah_low    = ("low",    "min"),
            ah_volume = ("volume", "sum"),
        )
    )
    ah_agg["ah_range"]     = ah_agg["ah_high"] - ah_agg["ah_open"]
    ah_agg["ah_range_pct"] = (ah_agg["ah_range"] / ah_agg["ah_open"]) * 100

    # ── Join all sessions on (ticker, date) ───────────────────────────────────
    all_keys = (
        df[df["session"].notna()]
        .groupby(_KEY)
        .size()
        .rename("_n")
        .reset_index()
        .set_index(_KEY)
    )

    result = (
        all_keys
        .join(pm_agg)
        .join(mh_agg)
        .join(ah_agg)
        .drop(columns="_n")
        .reset_index()
    )

    return result


def process_data_minutes(data: pd.DataFrame) -> "pd.DataFrame | None":
    """
    Convert a list of raw Massive/Polygon API bars into a per-day OHLCV summary.

    Input keys (Polygon shorthand): o, c, h, l, v, t  (t = UTC milliseconds)

    Sessions (America/New_York):
        Pre-market  (pm):  04:00 – 09:29 ET
        Market hours (mh): 09:30 – 15:59 ET
        After hours  (ah): 16:00 – 20:00 ET

    Returns one row per trading day with columns:
        open, close, high, low, volume        — market-hours OHLCV
        high_pm, low_pm, premarket_volume
        high_mh, low_mh, market_hours_volume
        ah_open, ah_close, ah_high, ah_low, ah_volume, ah_range, ah_range_perc
        high_pm_time                          — epoch ms when pm high was reached
        highest_in_pm                         — bool: pm high >= mh high
        time                                  — epoch ms of ET midnight for the day
        date_str                              — "YYYY-MM-DD"
    """
    import numpy as np
    import pandas as pd

    df = pd.DataFrame(data)
    if df.empty:
        return df

    try:
        df.rename(
            columns={"o": "open", "c": "close", "h": "high", "l": "low", "v": "volume", "t": "time"},
            inplace=True,
        )

        # ── Convert ms timestamp to ET local datetime ─────────────────────────
        df["dt_et"] = pd.to_datetime(df["time"], unit="ms", utc=True).dt.tz_convert("America/New_York")
        df["day"]   = df["dt_et"].dt.date
        df["time"]  = df["dt_et"].dt.time   # overwrite ms column with time-of-day objects

        # ── Session masks ─────────────────────────────────────────────────────
        # FIX 1: pre-market starts at 04:00, not midnight
        df["is_premarket"]    = (df["time"] >= time(4,  0)) & (df["time"] <  time(9, 30))
        df["is_market_hours"] = (df["time"] >= time(9, 30)) & (df["time"] <  time(16, 0))
        df["is_after_hours"]  = (df["time"] >= time(16, 0)) & (df["time"] <= time(20, 0))

        # ── Pre-market ────────────────────────────────────────────────────────
        # FIX 4: sort by ['day', 'time'], not just 'time'
        before_930 = df[df["is_premarket"]].sort_values(["day", "time"]).reset_index(drop=True)

        highest_pre_market = (
            before_930.groupby("day", as_index=False)["high"].max()
            .rename(columns={"high": "high_pm"})
        )
        lowest_pre_market = (
            before_930.groupby("day", as_index=False)["low"].min()
            .rename(columns={"low": "low_pm"})
        )
        premarket_volume = (
            before_930.groupby("day", as_index=False)["volume"].sum()
            .rename(columns={"volume": "premarket_volume"})
        )
        premarket_open = (
            before_930.groupby("day", as_index=False)["open"].first()
            .rename(columns={"open": "pm_open"})
        )

        # Time at which the pre-market high is reached
        pm_high_with_time = before_930.merge(highest_pre_market, on="day", how="left")
        pm_high_with_time = pm_high_with_time[pm_high_with_time["high"] == pm_high_with_time["high_pm"]]
        pm_high_time = (
            pm_high_with_time
            .sort_values(["day", "dt_et"])
            .groupby("day", as_index=False)
            .first()[["day", "dt_et"]]
        )
        # FIX 9: tz-aware epoch conversion (nanoseconds UTC ÷ 10^6 = ms)
        pm_high_time["high_pm_time"] = pm_high_time["dt_et"].astype("int64") // 10**6
        pm_high_time = pm_high_time[["day", "high_pm_time"]]

        # ── Market hours ──────────────────────────────────────────────────────
        after_930 = df[df["is_market_hours"]].sort_values(["day", "time"]).reset_index(drop=True)

        # FIX 3: day_open = first candle open of market hours (not exact-time match)
        daily_open = after_930.groupby("day", as_index=False)["open"].first()
        # FIX 2: day_close = last candle close of market hours (not open at 16:00)
        daily_close = after_930.groupby("day", as_index=False)["close"].last()

        highest_market_hours = (
            after_930.groupby("day", as_index=False)["high"].max()
            .rename(columns={"high": "high_mh"})
        )
        lowest_market_hours = (
            after_930.groupby("day", as_index=False)["low"].min()
            .rename(columns={"low": "low_mh"})
        )
        market_hours_volume = (
            after_930.groupby("day", as_index=False)["volume"].sum()
            .rename(columns={"volume": "market_hours_volume"})
        )

        # ── After hours ───────────────────────────────────────────────────────
        after_hours = df[df["is_after_hours"]].sort_values(["day", "time"]).reset_index(drop=True)

        # FIX 5 + FIX 6: compute all AH fields in one groupby, no column-select drop,
        # no redundant separate groupbys, no extra .reset_index() after as_index=False
        after_hour_daily_ohlc = after_hours.groupby("day", as_index=False).agg(
            ah_open   = ("open",   "first"),
            ah_close  = ("close",  "last"),
            ah_high   = ("high",   "max"),
            ah_low    = ("low",    "min"),
            ah_volume = ("volume", "sum"),
        )
        after_hour_daily_ohlc["ah_range"]      = after_hour_daily_ohlc["ah_high"] - after_hour_daily_ohlc["ah_open"]
        after_hour_daily_ohlc["ah_range_perc"] = 100 * after_hour_daily_ohlc["ah_range"] / after_hour_daily_ohlc["ah_open"]

        # ── Assemble daily summary ────────────────────────────────────────────
        daily_ohlc = (
            after_930.groupby("day").agg(
                high   = ("high",   "max"),
                low    = ("low",    "min"),
                volume = ("volume", "sum"),
            ).reset_index()
        )

        # FIX 8: merge everything first, apply fillna once at the end
        daily_ohlc = daily_ohlc.merge(daily_open,             on="day", how="left")
        daily_ohlc = daily_ohlc.merge(daily_close,            on="day", how="left")
        daily_ohlc = daily_ohlc.merge(premarket_volume,       on="day", how="left")
        daily_ohlc = daily_ohlc.merge(premarket_open,         on="day", how="left")
        daily_ohlc = daily_ohlc.merge(market_hours_volume,    on="day", how="left")
        daily_ohlc = daily_ohlc.merge(highest_market_hours,   on="day", how="left")
        daily_ohlc = daily_ohlc.merge(lowest_market_hours,    on="day", how="left")
        daily_ohlc = daily_ohlc.merge(highest_pre_market,     on="day", how="left")
        daily_ohlc = daily_ohlc.merge(lowest_pre_market,      on="day", how="left")
        daily_ohlc = daily_ohlc.merge(after_hour_daily_ohlc,  on="day", how="left")
        daily_ohlc = daily_ohlc.merge(pm_high_time,           on="day", how="left")

        # FIX 7: high = max(high_pm, high_mh) — not max(high_mh, open)
        daily_ohlc["high"] = np.maximum(daily_ohlc["high_mh"], daily_ohlc["high_pm"].fillna(0))
        daily_ohlc["low"]  = np.minimum(daily_ohlc["low_mh"],  daily_ohlc["low_pm"].fillna(float("inf")))

        daily_ohlc["highest_in_pm"] = daily_ohlc["high_pm"] >= daily_ohlc["high_mh"]

        # FIX 10: localize date to ET before converting to epoch ms (midnight ET, not midnight UTC)
        daily_ohlc["time"] = (
            pd.to_datetime(daily_ohlc["day"])
            .dt.tz_localize("America/New_York")
            .astype("int64") // 10**6
        )
        daily_ohlc["date_str"] = pd.to_datetime(daily_ohlc["day"]).dt.strftime("%Y-%m-%d")

        # FIX 8 cont.: single fillna with sentinel values, after all merges are done
        daily_ohlc.fillna(
            {
                "volume":              0,
                "premarket_volume":    0,
                "market_hours_volume": 0,
                "high_pm":            -1,
                "low_pm":             -1,
                "high_mh":            -1,
                "low_mh":             -1,
                "ah_open":            -1,
                "ah_close":           -1,
                "ah_high":            -1,
                "ah_low":             -1,
                "ah_volume":           0,
                "ah_range":            0,
                "ah_range_perc":       0,
                "high_pm_time":       -1,
            },
            inplace=True,
        )

        # Keep epoch-ms columns as int64 (prevents scientific notation / float drift)
        daily_ohlc["time"]         = daily_ohlc["time"].astype("int64")
        daily_ohlc["high_pm_time"] = daily_ohlc["high_pm_time"].astype("int64")

        # Round all float columns to 3 decimal places
        float_cols = daily_ohlc.select_dtypes(include="float").columns
        daily_ohlc[float_cols] = daily_ohlc[float_cols].round(3)

        return daily_ohlc

    except Exception as e:
        print(f"error: {e}")
        return None


def sync_data_with_prev_day_close(
    df: pd.DataFrame,
    fetch_split,
) -> pd.DataFrame:
    """
    Enrich a daily-summary DataFrame with stock-split adjustment factors and
    set ``previous_close`` to the market-open price on each split day.

    For every ticker the function:
        1. Calls ``fetch_split(ticker)`` to retrieve split events.
        2. Tags each pre-split row with its ``historical_adjustment_factor``
           and the split's execution date (``split_date_str``).
        3. Rows with no applicable split receive ``split_adjust_factor = 1``.
        4. On the split execution day sets ``previous_close = open``.

    Args:
        df:          Daily-summary DataFrame that contains at minimum the
                     columns: ticker, time (ET midnight epoch ms), date_str,
                     open.
        fetch_split: Callable(ticker: str) → list[dict] | None.
                     Each dict must have the keys ``execution_date`` (ISO
                     date string) and ``historical_adjustment_factor`` (float).

    Returns:
        The same DataFrame with three new columns added in-place:
        ``split_adjust_factor``, ``split_date_str``, ``previous_close``.
    """
    try:
        df["split_adjust_factor"] = np.nan
        df["split_date_str"]      = ""

        seen_tickers: set[str] = set()

        for i in range(len(df)):
            row    = df.iloc[i]          # positional — safe on any index
            ticker = row["ticker"]

            if ticker in seen_tickers:   # each ticker is processed only once
                continue
            seen_tickers.add(ticker)

            res = fetch_split(ticker)

            if not res:                  # None or empty list
                df.loc[df["ticker"] == ticker, "split_adjust_factor"] = 1
                continue

            for split in res:
                time_param = int(
                    pd.to_datetime(split["execution_date"], utc=True).timestamp() * 1000
                )
                mask = (df["ticker"] == ticker) & (df["time"] < time_param)

                df.loc[mask, "split_adjust_factor"] = split["historical_adjustment_factor"]
                df.loc[mask, "split_date_str"]      = (
                    pd.to_datetime(time_param, unit="ms", utc=True).strftime("%Y-%m-%d")
                )

            # Rows with no split event assigned → neutral factor
            no_split = (df["ticker"] == ticker) & df["split_adjust_factor"].isna()
            df.loc[no_split, "split_adjust_factor"] = 1

            # On each split execution day set previous_close = open
            ticker_mask  = df["ticker"] == ticker
            split_dates  = df.loc[ticker_mask, "split_date_str"].str.strip().unique()
            date_col     = df.loc[ticker_mask, "date_str"].str.strip()

            for sd in split_dates:
                if sd:
                    idx = ticker_mask & (df["date_str"].str.strip() == sd)
                    df.loc[idx, "previous_close"] = df.loc[idx, "open"]

    except Exception as e:
        print(f"sync_data_with_prev_day_close: {e}")

    return df


def append_single_parquet(df, path):
    """
    Save data to parquet file.It appends if file exists.
    
    Args:
        df (dataframe): data
        path (string): path to parquet file 
    """
    if os.path.exists(path):
        old = pd.read_parquet(path)
        df = pd.concat([old, df], ignore_index=True)

    df.to_parquet(path)
    

