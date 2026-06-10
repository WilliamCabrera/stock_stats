"""
Base module for iterative strategies.

Every strategy function must:
  1. Accept (candles: pd.DataFrame, **params) and return a pd.DataFrame.
  2. Return a DataFrame with exactly TRADE_COLUMNS (use enforce_schema).
  3. Execute ALL entries and exits at the open of the NEXT bar (never the
     signal bar itself). Set a pending flag on the signal bar and act on i+1.

Usage
-----
    from strategies.iterative.strategy_base import TRADE_COLUMNS, enforce_schema

    def my_strategy(candles, gap_pct=0.4, stop_pct=0.5, tp_pct=0.2,
                    slippage=0.001, timeframe_minutes=5):
        ...
        trades = [...]            # list of dicts
        df = pd.DataFrame(trades)
        return enforce_schema(df) # always call before returning
"""
from __future__ import annotations

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Canonical output schema — must match UP-TO-DATE/{tf}/**/*.parquet
# ---------------------------------------------------------------------------
TRADE_COLUMNS: list[str] = [
    "ticker",
    "date_str",
    "type",
    "entry_price",
    "exit_price",
    "stop_loss_price",
    "take_profit_price",
    "risk_reward_ratio",
    "pnl",
    "Return",
    "MAE",
    "mae_pct",
    "MFE",
    "mfe_pct",
    "rvol_daily",
    "previous_day_close",
    "volume",
    "entry_volume",
    "entry_time",
    "exit_time",
    "strategy",
    "timeframe",
]


def enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df has exactly TRADE_COLUMNS in the right order.
    Missing columns are filled with NaN so every strategy output is
    compatible with the trades parquet files.
    """
    if df.empty:
        return pd.DataFrame(columns=TRADE_COLUMNS)
    for col in TRADE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[TRADE_COLUMNS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Strategy template — copy, rename, and implement your signal logic
# ---------------------------------------------------------------------------
"""
def my_strategy_template(
    candles: pd.DataFrame,
    gap_pct: float = 0.40,
    stop_pct: float = 0.50,
    tp_pct: float = 0.20,
    slippage: float = 0.001,
    timeframe_minutes: int = 5,
) -> pd.DataFrame:
    STRATEGY = f"my_strategy_template_{gap_pct}_{stop_pct}_{tp_pct}"
    CLOSE_HOUR = 16
    expected_delta = pd.Timedelta(minutes=timeframe_minutes)

    df = candles.reset_index(drop=True)
    if len(df) < 3:
        return enforce_schema(pd.DataFrame())

    ticker = df["ticker"].iloc[0]

    before_close = df["date"].dt.hour < CLOSE_HOUR
    if not before_close.any():
        return enforce_schema(pd.DataFrame())
    last_valid_idx = before_close[before_close].index[-1]

    no_gap = df["date"].diff() == expected_delta

    # ── Compute your signal vector here ─────────────────────────────────
    signal = pd.Series(False, index=df.index)   # replace with real logic

    trades: list[dict] = []
    position       = None
    pending_entry  = False
    pending_exit   = False
    entry_price    = sl_price = tp_price = 0.0
    entry_time     = entry_volume = entry_date_str = None
    signal_volume  = signal_rvol = signal_prev_close = None
    trade_highs: list[float] = []
    trade_lows:  list[float] = []

    for i in range(1, last_valid_idx + 1):
        row = df.iloc[i]

        # ── Exit at OPEN of this bar (signal was on the previous bar) ────
        if pending_exit:
            exit_price = row["open"] * (1 + slippage)   # short: buy to cover
            pnl  = entry_price - exit_price
            rr   = (entry_price - tp_price) / (sl_price - entry_price)
            mae  = max(trade_highs) - entry_price
            mfe  = entry_price - min(trade_lows)
            trades.append({
                "ticker":             ticker,
                "date_str":           entry_date_str,
                "type":               "short",
                "entry_price":        entry_price,
                "exit_price":         exit_price,
                "stop_loss_price":    round(sl_price, 4),
                "take_profit_price":  round(tp_price, 4),
                "risk_reward_ratio":  round(rr, 4),
                "pnl":                round(pnl, 4),
                "Return":             round(pnl / entry_price, 4),
                "MAE":                round(mae, 4),
                "mae_pct":            round(mae / entry_price * 100, 4),
                "MFE":                round(mfe, 4),
                "mfe_pct":            round(mfe / entry_price * 100, 4),
                "rvol_daily":         signal_rvol,
                "previous_day_close": signal_prev_close,
                "volume":             signal_volume,
                "entry_volume":       entry_volume,
                "entry_time":         entry_time,
                "exit_time":          row["date"],
                "strategy":           STRATEGY,
                # "timeframe" is added by run_backtest after the call
            })
            position    = None
            pending_exit = False
            trade_highs = []
            trade_lows  = []
            continue   # don't look for a new signal on the exit bar

        # ── Enter at OPEN of this bar (signal was on the previous bar) ───
        if pending_entry:
            entry_price    = row["open"] * (1 - slippage)   # short: sell
            sl_price       = entry_price * (1 + stop_pct)
            tp_price       = entry_price * (1 - tp_pct)
            entry_time     = row["date"]
            entry_volume   = row["volume"]
            entry_date_str = row["date_str"]
            position       = "short"
            pending_entry  = False
            trade_highs    = [row["high"]]
            trade_lows     = [row["low"]]

        elif position == "short":
            trade_highs.append(row["high"])
            trade_lows.append(row["low"])

        # ── Check SL / TP / forced EOD close ────────────────────────────
        if position == "short":
            hit_tp  = row["low"]  <= tp_price
            hit_sl  = row["high"] >= sl_price
            is_last = i == last_valid_idx

            if (hit_tp or hit_sl) and not is_last:
                pending_exit = True
            elif is_last:
                exit_price = row["close"] * (1 + slippage)
                pnl  = entry_price - exit_price
                rr   = (entry_price - tp_price) / (sl_price - entry_price)
                mae  = max(trade_highs) - entry_price
                mfe  = entry_price - min(trade_lows)
                trades.append({
                    "ticker":             ticker,
                    "date_str":           entry_date_str,
                    "type":               "short",
                    "entry_price":        entry_price,
                    "exit_price":         exit_price,
                    "stop_loss_price":    round(sl_price, 4),
                    "take_profit_price":  round(tp_price, 4),
                    "risk_reward_ratio":  round(rr, 4),
                    "pnl":                round(pnl, 4),
                    "Return":             round(pnl / entry_price, 4),
                    "MAE":                round(mae, 4),
                    "mae_pct":            round(mae / entry_price * 100, 4),
                    "MFE":                round(mfe, 4),
                    "mfe_pct":            round(mfe / entry_price * 100, 4),
                    "rvol_daily":         signal_rvol,
                    "previous_day_close": signal_prev_close,
                    "volume":             signal_volume,
                    "entry_volume":       entry_volume,
                    "entry_time":         entry_time,
                    "exit_time":          row["date"],
                    "strategy":           STRATEGY,
                })
                position    = None
                trade_highs = []
                trade_lows  = []

        # ── Detect signal — entry will happen on the NEXT bar ────────────
        next_i = i + 1
        if (
            position is None
            and signal.iloc[i]
            and next_i <= last_valid_idx
            and no_gap.iloc[next_i]
        ):
            pending_entry     = True
            signal_volume     = row["volume"]
            signal_rvol       = row["RVOL_daily"]
            signal_prev_close = row["previous_day_close"]

    return enforce_schema(pd.DataFrame(trades))
"""
