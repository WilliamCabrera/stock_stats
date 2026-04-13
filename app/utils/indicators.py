"""
Technical indicators computed from fetch_candles DataFrames.

Each function accepts a DataFrame with at minimum the columns produced by
fetch_candles (time, open, high, low, close, volume) and returns a pd.Series
aligned to the input index.

Usage:
    from app.utils.indicators import compute_vwap, compute_atr, compute_rvol, compute_sma

    df = pd.DataFrame(fetch_candles("NVDA", "2026-03-25", "2026-03-25"))
    df["vwap"] = compute_vwap(df)
    df["atr"]  = compute_atr(df, window=14)
    df["rvol"] = compute_rvol(df, window=20)
    df["sma"]  = compute_sma(df, window=20)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Compute intraday VWAP, resetting at the start of each trading day (ET).

    Formula: cumsum(typical_price * volume) / cumsum(volume)
             typical_price = (high + low + close) / 3

    Args:
        df: DataFrame with columns time (Unix seconds UTC), high, low, close, volume.

    Returns:
        pd.Series named "vwap", aligned to df's index.
    """
    d = df.copy()
    d["_tp"]      = (d["high"] + d["low"] + d["close"]) / 3
    d["_tp_vol"]  = d["_tp"] * d["volume"]
    d["_date_et"] = (
        pd.to_datetime(d["time"], unit="s", utc=True)
        .dt.tz_convert("America/New_York")
        .dt.date
    )
    d["_cum_tp_vol"] = d.groupby("_date_et")["_tp_vol"].cumsum()
    d["_cum_vol"]    = d.groupby("_date_et")["volume"].cumsum()
    return (d["_cum_tp_vol"] / d["_cum_vol"]).rename("vwap")


def compute_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Average True Range (ATR).

    True Range = max(high - low, |high - prev_close|, |low - prev_close|)
    ATR = rolling mean of TR over `window` periods.

    The first bar has no prev_close so TR falls back to high - low.

    Args:
        df:     DataFrame with columns high, low, close.
        window: Lookback period (default 14).

    Returns:
        pd.Series named "atr".
    """
    high  = df["high"]
    low   = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    return tr.rolling(window, min_periods=1).mean().rename("atr")


def compute_rvol(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Relative Volume (RVOL).

    RVOL = current bar volume / rolling mean volume over `window` bars.
    Values > 1 indicate above-average activity; < 1 below-average.

    Args:
        df:     DataFrame with column volume.
        window: Lookback period for the average (default 20).

    Returns:
        pd.Series named "rvol".
    """
    avg_vol = df["volume"].rolling(window, min_periods=1).mean()
    rvol = df["volume"] / avg_vol.replace(0, np.nan)
    return rvol.rename("rvol")


def compute_sma(df: pd.DataFrame, window: int = 20, column: str = "close") -> pd.Series:
    """
    Simple Moving Average (SMA).

    Args:
        df:     DataFrame with the target column.
        window: Lookback period (default 20).
        column: Column to average (default "close").

    Returns:
        pd.Series named "sma_{window}".
    """
    return df[column].rolling(window, min_periods=1).mean().rename(f"sma_{window}")


def compute_donchian(
    df: pd.DataFrame,
    period: int = 5,
    offset: int = 1,
) -> pd.DataFrame:
    """
    Donchian Channel.

    upper = highest high over `period` bars, shifted by `offset` (avoids look-ahead).
    lower = lowest  low  over `period` bars, shifted by `offset`.
    basis = (upper + lower) / 2

    Args:
        df:     DataFrame with columns high, low.
        period: Lookback window in bars (default 5).
        offset: Shift to apply after rolling — 1 means the channel reflects
                the previous `period` closed bars (no look-ahead, default 1).

    Returns:
        pd.DataFrame with columns donchian_upper, donchian_lower, donchian_basis.
    """
    upper = df["high"].rolling(period, min_periods=1).max().shift(offset)
    lower = df["low"].rolling(period, min_periods=1).min().shift(offset)
    basis = (upper + lower) / 2
    return pd.DataFrame(
        {"donchian_upper": upper, "donchian_lower": lower, "donchian_basis": basis},
        index=df.index,
    )


def compute_close_atr_band(
    df: pd.DataFrame,
    factor: float = 1.0,
    atr_window: int = 14,
    direction: str = "above",
) -> pd.Series:
    """
    Custom indicator: close ± factor * ATR.

    Useful as a dynamic stop, target, or band relative to price action.

    Args:
        df:         DataFrame with columns high, low, close.
        factor:     ATR multiplier (default 1.0).
        atr_window: Lookback for ATR (default 14).
        direction:  "above" → close + factor * ATR
                    "below" → close - factor * ATR

    Returns:
        pd.Series named "close_atr_above" or "close_atr_below".

    Example:
        df["stop"] = compute_close_atr_band(df, factor=2.0, direction="above")
    """
    atr = compute_atr(df, window=atr_window)
    if direction == "above":
        return (df["close"] + factor * atr).rename("close_atr_above")
    elif direction == "below":
        return (df["close"] - factor * atr).rename("close_atr_below")
    else:
        raise ValueError(f"direction must be 'above' or 'below', got '{direction}'")
