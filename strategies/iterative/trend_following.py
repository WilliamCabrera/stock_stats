import os
import sys
sys.path.insert(0, os.path.abspath("."))

import logging
import numpy as np
import pandas as pd
import time as tm
from pathlib import Path

from strategies.iterative.strategy_base import enforce_schema

DATASET_ROOT = Path(os.path.abspath(".")) / "backtest_dataset" / "full"

_LOG_DIR = Path(os.path.abspath(".")) / "logs" / "iterative"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setLevel(logging.INFO)
_console_handler.setFormatter(logging.Formatter("%(message)s"))

_file_handler = logging.FileHandler(_LOG_DIR / "trend_following.log", encoding="utf-8")
_file_handler.setLevel(logging.WARNING)
_file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(_console_handler)
logger.addHandler(_file_handler)


def ema100_trend_follower_iterative(
    candles: pd.DataFrame,
    gap_pct: float = 0.40,       # unused — kept for run_backtest interface
    stop_pct: float = 0.50,      # unused
    tp_pct: float = 0.20,        # unused
    slippage: float = 0.001,
    timeframe_minutes: int = 5,
    atr_period: int = 14,
    ema_period: int = 100,
    lookback_days: int = 20,
    pyramid_max: int = 3,
    pyramid_atr_mult: float = 0.5,
    trail_atr_mult: float = 2.0,
    ticker_parquet_path: "Path | None" = None,
) -> pd.DataFrame:
    """
    Trend follower for indices / large caps with 5 rules:
      1. EMA100 as trend filter (long bias above, short bias below)
      2. Enter long on new 20-day high, short on new 20-day low
      3. Pyramid up to 3 units every 0.5x ATR advance
      4. ATR-trailing stop at 2.0x ATR(14) below recent peak / above trough
      5. Execute on next bar's open

    Historical daily indicators (EMA100, ATR14, 20d-high/low) are computed from
    the ticker's full intraday parquet, resampled to daily, using only prior days
    (no lookahead). Each pyramid unit is recorded as a separate trade.
    """
    STRATEGY = f"ema100_trend_follower_iterative_{ema_period}_{lookback_days}_{atr_period}"
    CLOSE_HOUR = 16
    tf_str = f"{timeframe_minutes}m"
    expected_delta = pd.Timedelta(minutes=timeframe_minutes)

    df = candles.reset_index(drop=True)
    if len(df) < 3:
        return enforce_schema(pd.DataFrame())

    ticker = df["ticker"].iloc[0]
    current_date_str = df["date_str"].iloc[0]

    # ── Load historical data and resample to daily ───────────────────────────
    ticker_path = ticker_parquet_path if ticker_parquet_path is not None else DATASET_ROOT / tf_str / "tickers" / f"{ticker}.parquet"
    if not ticker_path.exists():
        return enforce_schema(pd.DataFrame())
    try:
        full_df = pd.read_parquet(ticker_path, columns=["date_str", "open", "high", "low", "close"])
    except Exception:
        return enforce_schema(pd.DataFrame())

    prior = full_df[full_df["date_str"] < current_date_str]
    daily = (
        prior.groupby("date_str", sort=True)
        .agg(open=("open", "first"), high=("high", "max"),
             low=("low", "min"), close=("close", "last"))
    )
    if len(daily) < max(ema_period, lookback_days + atr_period):
        return enforce_schema(pd.DataFrame())

    ema100 = daily["close"].ewm(span=ema_period, adjust=False).mean().iloc[-1]
    tr = pd.concat([
        daily["high"] - daily["low"],
        (daily["high"] - daily["close"].shift(1)).abs(),
        (daily["low"]  - daily["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr14  = tr.ewm(span=atr_period, adjust=False).mean().iloc[-1]
    high20 = daily["high"].rolling(lookback_days).max().iloc[-1]
    low20  = daily["low"].rolling(lookback_days).min().iloc[-1]

    if any(pd.isna(v) for v in [ema100, atr14, high20, low20]) or atr14 == 0:
        return enforce_schema(pd.DataFrame())

    # ── Intraday setup ────────────────────────────────────────────────────────
    before_close = df["date"].dt.hour < CLOSE_HOUR
    if not before_close.any():
        return enforce_schema(pd.DataFrame())
    last_valid_idx = before_close[before_close].index[-1]
    no_gap = df["date"].diff() == expected_delta

    def _flush(units_list, exit_px, exit_time, t_highs, t_lows):
        recs = []
        for u in units_list:
            ep = u["entry_price"]
            d  = u["direction"]
            if d == "long":
                pnl = exit_px - ep
                mae = max(ep - min(t_lows), 0.0)  if t_lows else 0.0
                mfe = max(max(t_highs) - ep, 0.0) if t_highs else 0.0
            else:
                pnl = ep - exit_px
                mae = max(max(t_highs) - ep, 0.0) if t_highs else 0.0
                mfe = max(ep - min(t_lows), 0.0)  if t_lows else 0.0
            recs.append({
                "ticker":             ticker,
                "date_str":           u["entry_date_str"],
                "type":               d,
                "entry_price":        round(ep, 4),
                "exit_price":         round(exit_px, 4),
                "stop_loss_price":    round(u["sl_price"], 4),
                "take_profit_price":  float("nan"),
                "risk_reward_ratio":  float("nan"),
                "pnl":                round(pnl, 4),
                "Return":             round(pnl / ep, 4) if ep else 0.0,
                "MAE":                round(mae, 4),
                "mae_pct":            round(mae / ep * 100, 4) if ep else 0.0,
                "MFE":                round(mfe, 4),
                "mfe_pct":            round(mfe / ep * 100, 4) if ep else 0.0,
                "rvol_daily":         u["signal_rvol"],
                "previous_day_close": u["signal_prev_close"],
                "volume":             u["signal_volume"],
                "entry_volume":       u["entry_volume"],
                "entry_time":         u["entry_time"],
                "exit_time":          exit_time,
                "strategy":           STRATEGY,
            })
        return recs

    trades: list[dict]    = []
    position              = None        # None | "long" | "short"
    units: list[dict]     = []
    trail_stop            = 0.0
    peak_price            = 0.0
    trough_price          = float("inf")
    pending_direction     = None        # pending entry direction
    pending_exit          = False
    next_pyramid_level    = float("inf")
    trade_highs: list[float] = []
    trade_lows:  list[float] = []
    signal_volume = float("nan")
    signal_rvol   = float("nan")
    signal_prev_close = float("nan")

    for i in range(1, last_valid_idx + 1):
        row = df.iloc[i]

        # ── 1. Execute pending exit ──────────────────────────────────────────
        if pending_exit:
            exit_px = row["open"] * (1 - slippage) if position == "long" else row["open"] * (1 + slippage)
            trades.extend(_flush(units, exit_px, row["date"], trade_highs, trade_lows))
            position = None; units = []; pending_exit = False
            pending_direction = None; trade_highs = []; trade_lows = []
            next_pyramid_level = float("inf")
            continue

        # ── 2. Execute pending entry (open new or pyramid add) ───────────────
        if pending_direction is not None:
            if position is None:
                ep = row["open"] * (1 + slippage) if pending_direction == "long" else row["open"] * (1 - slippage)
                position     = pending_direction
                trail_stop   = ep - trail_atr_mult * atr14 if position == "long" else ep + trail_atr_mult * atr14
                peak_price   = row["high"]
                trough_price = row["low"]
                next_pyramid_level = ep + pyramid_atr_mult * atr14 if position == "long" else ep - pyramid_atr_mult * atr14
                trade_highs  = [row["high"]]
                trade_lows   = [row["low"]]
            else:
                ep = row["open"] * (1 + slippage) if position == "long" else row["open"] * (1 - slippage)
                next_pyramid_level = ep + pyramid_atr_mult * atr14 if position == "long" else ep - pyramid_atr_mult * atr14
                trade_highs.append(row["high"])
                trade_lows.append(row["low"])

            units.append({
                "direction":         position,
                "entry_price":       ep,
                "sl_price":          trail_stop,
                "entry_time":        row["date"],
                "entry_volume":      row["volume"],
                "entry_date_str":    row["date_str"],
                "signal_volume":     signal_volume,
                "signal_rvol":       signal_rvol,
                "signal_prev_close": signal_prev_close,
            })
            pending_direction = None

        elif position is not None:
            trade_highs.append(row["high"])
            trade_lows.append(row["low"])

        # ── 3. Update trailing stop and check exit ───────────────────────────
        if position is not None:
            if position == "long":
                peak_price = max(peak_price, row["high"])
                trail_stop = peak_price - trail_atr_mult * atr14
                hit_stop   = row["low"] <= trail_stop
            else:
                trough_price = min(trough_price, row["low"])
                trail_stop   = trough_price + trail_atr_mult * atr14
                hit_stop     = row["high"] >= trail_stop

            is_last = i == last_valid_idx
            if hit_stop and not is_last:
                pending_exit = True
            elif is_last:
                exit_px = row["close"] * (1 - slippage) if position == "long" else row["close"] * (1 + slippage)
                trades.extend(_flush(units, exit_px, row["date"], trade_highs, trade_lows))
                position = None; units = []; trade_highs = []; trade_lows = []
                next_pyramid_level = float("inf")

        # ── 4. Signal detection (fires on next bar) ──────────────────────────
        next_i = i + 1
        if next_i > last_valid_idx or not no_gap.iloc[next_i] or pending_exit or pending_direction is not None:
            continue

        # Pyramid: add unit when price advances 0.5x ATR beyond last entry
        if position is not None and len(units) < pyramid_max:
            if position == "long" and row["high"] >= next_pyramid_level:
                pending_direction = "long"
            elif position == "short" and row["low"] <= next_pyramid_level:
                pending_direction = "short"

        # New breakout entry
        if position is None:
            if row["close"] > ema100 and row["high"] > high20:
                pending_direction  = "long"
                signal_volume      = row["volume"]
                signal_rvol        = row.get("RVOL_daily", float("nan"))
                signal_prev_close  = row.get("previous_day_close", float("nan"))
            elif row["close"] < ema100 and row["low"] < low20:
                pending_direction  = "short"
                signal_volume      = row["volume"]
                signal_rvol        = row.get("RVOL_daily", float("nan"))
                signal_prev_close  = row.get("previous_day_close", float("nan"))

    logger.debug("%-6s  %s  → %d trade(s)", ticker, current_date_str, len(trades))
    return enforce_schema(pd.DataFrame(trades))


if __name__ == "__main__":
    from strategies.iterative.backtest_helpers import run_backtest
    _t0 = tm.time()
    run_backtest(
        timeframe="5m",
        strategy_func=ema100_trend_follower_iterative,
        slippage=0.001,
        out_put_name="ema100_trend_follower_iterative_100_20_14",
    )
    print(f"completado en {tm.time() - _t0:.2f}s")
