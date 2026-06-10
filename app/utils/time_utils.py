"""
Timezone and datetime utilities.

Helpers for resolving the system's local timezone and converting
local dates/times to UTC representations used by external APIs.
"""
from __future__ import annotations

import os
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo


def _get_local_tz() -> ZoneInfo:
    """
    Return the system's local timezone as a ZoneInfo object.

    Resolution order:
      1. TZ environment variable (e.g.  TZ=America/New_York  in docker-compose)
      2. /etc/localtime symlink   (standard on Linux/macOS)
      3. Fallback: UTC
    """
    tz_env = os.environ.get("TZ")
    if tz_env:
        return ZoneInfo(tz_env)

    try:
        link = os.readlink("/etc/localtime")
        if "zoneinfo/" in link:
            return ZoneInfo(link.split("zoneinfo/")[-1])
    except OSError:
        pass

    return ZoneInfo("UTC")


def tz_offset_hours() -> int:
    """
    Return the current UTC offset of the local timezone as a signed integer.

    Examples:
        America/New_York in winter (EST) → -5
        America/New_York in summer (EDT) → -4
        UTC                              →  0
        Europe/Paris in summer (CEST)    → +2
    """
    tz = _get_local_tz()
    offset = datetime.now(tz).utcoffset()
    return int(offset.total_seconds()) // 3600


def tz_label() -> str:
    """
    Return a human-readable timezone label including the current UTC offset.

    Example outputs:
        "America/New_York (UTC-05:00)"   — EST (winter)
        "America/New_York (UTC-04:00)"   — EDT (summer)
        "UTC (UTC+00:00)"
    """
    tz = _get_local_tz()
    hours = tz_offset_hours()
    sign = "+" if hours >= 0 else "-"
    return f"{tz} (UTC{sign}{abs(hours):02d}:00)"


def local_time_to_utc_str(local_hhmm: str) -> str:
    """
    Convert a local 'HH:MM' time string to a UTC 'HH:MM' string.

    Uses the same offset as tz_offset_hours() for consistency with local_to_ms().

    Examples (EST, UTC-5):  "09:30"  →  "14:30"
    Examples (EDT, UTC-4):  "09:30"  →  "13:30"
    """
    h, m = map(int, local_hhmm.strip().split(":"))
    offset = tz_offset_hours()                # e.g. -5 for EST
    total_minutes = h * 60 + m - offset * 60  # UTC = local - offset
    total_minutes %= 24 * 60                  # wrap around midnight
    return f"{total_minutes // 60:02d}:{total_minutes % 60:02d}"


def local_to_ms(d: str | date | datetime, end_of_day: bool = False) -> int:
    """
    Convert a local date/datetime to UTC milliseconds.

    Algorithm (explicit offset subtraction):
        1. Parse the input as a naive local datetime (no tzinfo attached).
        2. Read the UTC offset as a signed integer (e.g. -5 for EST, -4 for EDT).
        3. Subtract the offset:  UTC = local_time - offset_hours
           e.g. 09:30 local (UTC-5)  →  09:30 - (-5h)  =  14:30 UTC
        4. Stamp the result as UTC and return Unix milliseconds.

    Args:
        d:          "YYYY-MM-DD", "YYYY-MM-DDTHH:MM:SS", a date, or a datetime.
        end_of_day: When True and no explicit time is present, sets time to
                    23:59:59 so the full end-day is included in the request.

    Returns:
        UTC milliseconds (int).
    """
    offset_hours = tz_offset_hours()          # e.g. -5 (EST) or -4 (EDT)
    offset_delta = timedelta(hours=offset_hours)

    # ── normalise input to a naive local datetime ─────────────────────────────
    if isinstance(d, datetime):
        local_dt = d.replace(tzinfo=None)      # strip any existing tzinfo
    elif isinstance(d, date):
        h, m, s = (23, 59, 59) if end_of_day else (0, 0, 0)
        local_dt = datetime(d.year, d.month, d.day, h, m, s)
    else:
        s = str(d).replace(" ", "T")           # normalise "2024-01-02 09:30" → "T"
        if "T" in s:
            local_dt = datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S")
        else:
            local_dt = datetime.strptime(s[:10], "%Y-%m-%d")
            if end_of_day:
                local_dt = local_dt.replace(hour=23, minute=59, second=59)

    # ── subtract offset to get UTC, then stamp as UTC ─────────────────────────
    utc_dt = (local_dt - offset_delta).replace(tzinfo=timezone.utc)
    return int(utc_dt.timestamp() * 1000)
