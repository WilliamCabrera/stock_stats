"""
Massive.com (Polygon.io-compatible) market data client.

Provides sync and async versions of fetch_candles so the same logic
can be used from both Celery tasks (sync) and FastAPI routes (async).

Supported timeframes: "1m", "5m", "30m", "1h", "1d"

Usage (sync — Celery tasks):
    from app.utils.massive import fetch_candles
    candles = fetch_candles("AAPL", "2024-01-02", "2024-01-10", timeframe="5m")

    # Session filter: local time, e.g. 09:30–16:00 ET
    candles = fetch_candles(
        "AAPL", "2024-01-02", "2024-01-10",
        timeframe="5m",
        session_start="09:30",
        session_end="16:00",
    )

Usage (async — FastAPI routes):
    from app.utils.massive import fetch_candles_async
    candles = await fetch_candles_async("AAPL", "2024-01-02", "2024-01-10",
                                        timeframe="5m",
                                        session_start="09:30",
                                        session_end="16:00")
"""
from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, time, timezone
from typing import Literal

import httpx
import pandas as pd

from app.config import get_settings
from app.utils.time_utils import local_to_ms, local_time_to_utc_str, tz_label, tz_offset_hours

logger = logging.getLogger(__name__)

# ── Types ─────────────────────────────────────────────────────────────────────

TimeFrame = Literal["1m", "5m", "15m", "30m", "1h", "1d"]

Candle = dict  # {time: int (UTC seconds), open, high, low, close, volume}

_TF_MAP: dict[str, tuple[int, str]] = {
    "1m":  (1,  "minute"),
    "5m":  (5,  "minute"),
    "15m": (15, "minute"),
    "30m": (30, "minute"),
    "1h":  (1,  "hour"),
    "1d":  (1,  "day"),
}

# Daily timeframes carry no meaningful intraday time — skip session filter.
_DAILY_TIMEFRAMES = {"1d"}

# ── Internal helpers ──────────────────────────────────────────────────────────

def _parse_time(t: str) -> time:
    """Parse a 'HH:MM' string into a datetime.time object."""
    try:
        h, m = t.strip().split(":")
        return time(int(h), int(m))
    except (ValueError, AttributeError) as exc:
        raise ValueError(
            f"Invalid time format '{t}'. Expected 'HH:MM', e.g. '09:30'."
        ) from exc


def _bars_to_dataframe(results: list[dict]) -> pd.DataFrame:
    """Convert raw Massive API bars to a candles DataFrame (vectorized)."""
    df = pd.DataFrame(results)
    df.rename(columns={"o": "open", "c": "close", "h": "high",
                        "l": "low",  "v": "volume", "t": "time"}, inplace=True)
    df["time"] = df["time"] // 1000  # milliseconds → UTC seconds
    df["date"] = pd.to_datetime(df["time"], unit='ms', utc=True)  # -5 means New York timezone
    df["date"] = df["date"].dt.tz_convert("America/New_York")
    df["day"] =  df["date"].dt.date 
    
    return df


def _raw_data_to_dataframe(data):

   
    df = pd.DataFrame(data)

    try:
        df.rename(columns={'o': 'open', 'c': 'close', 'h': 'high', 'l': 'low', 'v': 'volume', 't': 'time'}, inplace=True)
        # Convert timestamp to datetime
        df["date"] = pd.to_datetime(df["time"], unit='ms', utc=True)  # -5 means New York timezone
        df["date"] = df["date"].dt.tz_convert("America/New_York")
        df["day"] =  df["date"].dt.date #pd.to_datetime(pd.to_datetime(df["time"], unit='ms').dt.date).dt.strftime('%Y-%m-%d') 
        
        #df['dayX'] = pd.to_datetime(df['day'])
        #df["day1"] =  pd.to_datetime(pd.to_datetime(df["time"], unit='ms').dt.date).dt.strftime('%Y-%m-%d') 

    except  Exception as e:
       print(f' error: {e}')
    
    return df


def _build_url(
    ticker: str,
    from_date: str,
    to_date: str,
    timeframe: str,
    adjusted: bool,
    limit: int,
    api_key: str,
    base_url: str,
) -> str:
    multiplier, timespan = _TF_MAP[timeframe]
    adj = "true" if adjusted else "false"
    return (
        f"{base_url}/v2/aggs/ticker/{ticker}/range"
        f"/{multiplier}/{timespan}/{from_date}/{to_date}"
        f"?adjusted={adj}&sort=asc&limit={limit}&apiKey={api_key}"
    )


# ── Sync client (used by Celery tasks) ───────────────────────────────────────

def fetch_candles(
    ticker: str,
    from_date: str | date | datetime,
    to_date: str | date | datetime,
    timeframe: TimeFrame = "5m",
    adjusted: bool = False,
    limit: int = 50_000,
    session_start: str | None = None,
    session_end: str | None = None,
) -> list[Candle]:
    """
    Fetch OHLCV candles from Massive.com synchronously.

    Handles pagination automatically — all pages are fetched and merged
    before returning.

    Dates are interpreted in the system's local timezone (TZ env var or
    /etc/localtime) and sent to Massive as UTC milliseconds.

    Args:
        ticker:        Equity symbol, e.g. "AAPL".
        from_date:     Start date. Accepts "YYYY-MM-DD", "YYYY-MM-DDTHH:MM:SS",
                       a date, or a datetime.
        to_date:       End date (inclusive).
        timeframe:     One of "1m", "5m", "15m", "30m", "1h", "1d".
        adjusted:      Whether to return split/dividend-adjusted prices.
        limit:         Max bars per API request (Massive cap: 50 000).
        session_start: Local time 'HH:MM' — keep bars at or after this time each day.
                       Ignored for "1d".
        session_end:   Local time 'HH:MM' — keep bars at or before this time each day.
                       Ignored for "1d".

    Returns:
        List of candle dicts sorted by time ascending.

    Raises:
        ValueError: Unknown timeframe, invalid time format, or missing API key.
        httpx.HTTPStatusError: Non-2xx response from Massive.
    """
    
    
    if timeframe not in _TF_MAP:
        raise ValueError(f"Unknown timeframe '{timeframe}'. Valid: {list(_TF_MAP)}")

    settings = get_settings()
    if not settings.massive_api_key:
        raise ValueError("MASSIVE_API_KEY is not set in the environment.")

    
    # When session times are given, combine them with the dates so Massive
    # receives exact UTC ms boundaries for the first and last day.
    if session_start:
        fmt = "%H:%M:%S" if len(session_start) > 5 else "%H:%M"
        _from_date = f"{from_date}T{datetime.strptime(session_start, fmt).strftime('%H:%M:%S')}"
        _to_date   = f"{to_date}T{datetime.strptime(session_end, fmt).strftime('%H:%M:%S')}"
    else:
        _from_date = from_date
        _to_date   = to_date

    # Convert local dates → UTC ms so Massive uses exact time boundaries.
    from_ts = local_to_ms(_from_date, end_of_day=False)
    to_ts   = local_to_ms(_to_date,   end_of_day=not bool(session_end))

    url: str | None = _build_url(
        ticker=ticker,
        from_date=from_ts,
        to_date=to_ts,
        timeframe=timeframe,
        adjusted=adjusted,
        limit=limit,
        api_key=settings.massive_api_key,
        base_url=settings.massive_base_url,
    )
    
    
    
    
    # Accumulate raw results across all pages, then convert once
    all_results: list[dict] = []

    with httpx.Client(timeout=30) as client:
        while url:
            logger.debug("Fetching %s", url)
            resp = client.get(url)
            resp.raise_for_status()
            data = resp.json()
            all_results.extend(data.get("results") or [])
            next_url = data.get("next_url")
            url = f"{next_url}&apiKey={settings.massive_api_key}" if next_url else None

    if not all_results:
        return []

    # Single vectorized conversion
    df = _bars_to_dataframe(all_results)

    # Session filter (vectorized)
    if timeframe not in _DAILY_TIMEFRAMES and (session_start or session_end):
        try:
            utc_start = local_time_to_utc_str(session_start) if session_start else None
            utc_end   = local_time_to_utc_str(session_end)   if session_end   else None
            start = _parse_time(utc_start) if utc_start else time(0, 0)
            end   = _parse_time(utc_end)   if utc_end   else time(23, 59, 59)
            t_col = pd.to_datetime(df["time"], unit="s", utc=True).dt.time
            if start <= end:
                df = df[t_col.between(start, end)]
            else:
                df = df[(t_col >= start) | (t_col <= end)]
        except Exception as e:
            logger.warning("Session filter failed (%s) — returning unfiltered candles.", e)

    all_candles = df.to_dict(orient="records")

    logger.info(
        "fetch_candles: ticker=%s tf=%s bars=%d from=%s to=%s tz=%s session=%s-%s",
        ticker, timeframe, len(all_candles),
        from_ts, to_ts, tz_label(),
        session_start or "*", session_end or "*",
        extra={"utc_offset": tz_offset_hours()},
    )

    return all_candles


# ── Async client (used by FastAPI routes) ────────────────────────────────────

async def fetch_candles_async(
    ticker: str,
    from_date: str | date | datetime,
    to_date: str | date | datetime,
    timeframe: TimeFrame = "5m",
    adjusted: bool = False,
    limit: int = 50_000,
    session_start: str | None = None,
    session_end: str | None = None,
) -> list[Candle]:
    """
    Async version of fetch_candles — non-blocking, suitable for FastAPI routes.

    Runs the sync implementation in a thread pool so the event loop is never
    blocked by network I/O.
    """
    return await asyncio.to_thread(
        fetch_candles,
        ticker,
        from_date,
        to_date,
        timeframe,
        adjusted,
        limit,
        session_start,
        session_end,
    )


