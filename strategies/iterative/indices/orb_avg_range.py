import os
import sys
sys.path.insert(0, os.path.abspath("."))

import logging
import pandas as pd
import time as tm
from pathlib import Path

from strategies.iterative.strategy_base import enforce_schema

_LOG_DIR = Path(os.path.abspath(".")) / "logs" / "iterative"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setLevel(logging.INFO)
_console_handler.setFormatter(logging.Formatter("%(message)s"))

_file_handler = logging.FileHandler(_LOG_DIR / "orb_avg_range.log", encoding="utf-8")
_file_handler.setLevel(logging.WARNING)
_file_handler.setFormatter(logging.Formatter(
    "%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
))

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(_console_handler)
logger.addHandler(_file_handler)


def orb_avg_range_iterative(
    candles: pd.DataFrame,
    gap_pct: float = 0.40,       # unused — kept for runner interface
    stop_pct: float = 0.50,      # unused
    tp_pct: float = 0.20,        # unused
    slippage: float = 0.001,
    timeframe_minutes: int = 5,
    ticker_parquet_path: "Path | None" = None,  # unused — kept for indices runner interface
) -> pd.DataFrame:
    """
    Opening Range Breakout — Average Range (ORB_AvgRange).

    Opening range: the 9:00–10:00 am ET 1-hour bar, already precomputed in
    the INDICES dataset as h1_9am_high / h1_9am_low.
    Signals are evaluated from 10:00 am onward. One trade per day.

    Long
      Signal : close > h1_9am_high  (breakout above the range)
      Entry  : next-bar open  (+ slippage)
      TP     : h1_9am_low + daily_range_ma10
      SL     : h1_9am_low

    Short
      Signal : close < h1_9am_low   (breakdown below the range)
      Entry  : next-bar open  (- slippage)
      TP     : h1_9am_high - daily_range_ma10
      SL     : h1_9am_high

    All entries and exits execute on the OPEN of the bar after the signal.
    Forced EOD exit at the last bar's close if still in position.
    """
    STRATEGY   = "orb_avg_range_iterative"
    CLOSE_HOUR = 16
    ORB_HOUR   = 10   # signals only valid at or after this hour
    df = candles.reset_index(drop=True)
    if len(df) < 3:
        return enforce_schema(pd.DataFrame())

    ticker           = df["ticker"].iloc[0]
    current_date_str = df["date_str"].iloc[0]

    for col in ("h1_9am_high", "h1_9am_low", "daily_range_ma10"):
        if col not in df.columns or df[col].isna().all():
            logger.debug("%-6s  %s — %s missing/null, skip", ticker, current_date_str, col)
            return enforce_schema(pd.DataFrame())

    orb_high  = float(df["h1_9am_high"].iloc[0])
    orb_low   = float(df["h1_9am_low"].iloc[0])
    avg_range = float(df["daily_range_ma10"].iloc[0])

    if any(pd.isna(v) for v in [orb_high, orb_low, avg_range]) or avg_range <= 0:
        return enforce_schema(pd.DataFrame())

    long_tp  = orb_low  + avg_range   # long  TP: project avg range up from the range low
    long_sl  = orb_low                # long  SL: range low
    short_tp = orb_high - avg_range   # short TP: project avg range down from the range high
    short_sl = orb_high               # short SL: range high

    before_close = df["date"].dt.hour < CLOSE_HOUR
    if not before_close.any():
        return enforce_schema(pd.DataFrame())
    last_valid_idx = before_close[before_close].index[-1]

    trades:        list[dict] = []
    position       = None        # None | "long" | "short"
    pending_entry  = None        # None | "long" | "short"
    pending_exit   = False
    entry_price    = sl_price = tp_price = 0.0
    entry_time     = entry_volume = entry_date_str = None
    signal_volume  = float("nan")
    signal_rvol    = float("nan")
    signal_prev_close = float("nan")
    trade_highs:   list[float] = []
    trade_lows:    list[float] = []
    traded_today   = False       # one trade per day — no re-entry after first exit

    def _record(pos, ep, xp, sl, tp, et, xtime, t_highs, t_lows):
        if pos == "long":
            pnl = xp - ep
            mae = max(ep - min(t_lows),  0.0) if t_lows  else 0.0
            mfe = max(max(t_highs) - ep, 0.0) if t_highs else 0.0
        else:
            pnl = ep - xp
            mae = max(max(t_highs) - ep, 0.0) if t_highs else 0.0
            mfe = max(ep - min(t_lows),  0.0) if t_lows  else 0.0
        rr = abs(tp - ep) / abs(sl - ep) if abs(sl - ep) > 0 else float("nan")
        return {
            "ticker":             ticker,
            "date_str":           entry_date_str,
            "type":               pos,
            "entry_price":        round(ep, 4),
            "exit_price":         round(xp, 4),
            "stop_loss_price":    round(sl, 4),
            "take_profit_price":  round(tp, 4),
            "risk_reward_ratio":  round(rr, 4),
            "pnl":                round(pnl, 4),
            "Return":             round(pnl / ep, 4) if ep else 0.0,
            "MAE":                round(mae, 4),
            "mae_pct":            round(mae / ep * 100, 4) if ep else 0.0,
            "MFE":                round(mfe, 4),
            "mfe_pct":            round(mfe / ep * 100, 4) if ep else 0.0,
            "rvol_daily":         signal_rvol,
            "previous_day_close": signal_prev_close,
            "volume":             signal_volume,
            "entry_volume":       entry_volume,
            "entry_time":         et,
            "exit_time":          xtime,
            "strategy":           STRATEGY,
        }

    for i in range(1, last_valid_idx + 1):
        row = df.iloc[i]

        # ── Execute pending exit at this bar's open ──────────────────────────
        if pending_exit:
            xp = row["open"] * (1 - slippage) if position == "long" else row["open"] * (1 + slippage)
            trades.append(_record(position, entry_price, xp, sl_price, tp_price,
                                  entry_time, row["date"], trade_highs, trade_lows))
            position     = None
            pending_exit = False
            trade_highs  = []
            trade_lows   = []
            continue

        # ── Execute pending entry at this bar's open ─────────────────────────
        if pending_entry is not None:
            if pending_entry == "long":
                entry_price = row["open"] * (1 + slippage)
                sl_price    = long_sl
                tp_price    = long_tp
            else:
                entry_price = row["open"] * (1 - slippage)
                sl_price    = short_sl
                tp_price    = short_tp
            position       = pending_entry
            entry_time     = row["date"]
            entry_volume   = row["volume"]
            entry_date_str = row["date_str"]
            pending_entry  = None
            trade_highs    = [row["high"]]
            trade_lows     = [row["low"]]

        elif position is not None:
            trade_highs.append(row["high"])
            trade_lows.append(row["low"])

        # ── Check SL / TP / forced EOD close ────────────────────────────────
        if position is not None:
            is_last = i == last_valid_idx
            if position == "long":
                hit_tp = row["high"] >= tp_price   # high reaches TP (above entry)
                hit_sl = row["low"]  <= sl_price   # low drops to SL (below entry)
            else:
                hit_tp = row["low"]  <= tp_price   # low drops to TP (below entry)
                hit_sl = row["high"] >= sl_price   # high rises to SL (above entry)

            if (hit_tp or hit_sl) and not is_last:
                pending_exit = True
            elif is_last:
                xp = row["close"] * (1 - slippage) if position == "long" else row["close"] * (1 + slippage)
                trades.append(_record(position, entry_price, xp, sl_price, tp_price,
                                      entry_time, row["date"], trade_highs, trade_lows))
                position    = None
                trade_highs = []
                trade_lows  = []

        # ── Signal detection — entry executes on the NEXT bar ────────────────
        next_i = i + 1
        if (
            position is None
            and not traded_today
            and pending_entry is None
            and row["date"].hour >= ORB_HOUR
            and next_i <= last_valid_idx
        ):
            if row["close"] > orb_high:
                pending_entry     = "long"
                traded_today      = True
                signal_volume     = row["volume"]
                signal_rvol       = row.get("RVOL_daily", float("nan"))
                signal_prev_close = row.get("previous_day_close", float("nan"))
            elif row["close"] < orb_low:
                pending_entry     = "short"
                traded_today      = True
                signal_volume     = row["volume"]
                signal_rvol       = row.get("RVOL_daily", float("nan"))
                signal_prev_close = row.get("previous_day_close", float("nan"))

    logger.debug("%-6s  %s  → %d trade(s)", ticker, current_date_str, len(trades))
    return enforce_schema(pd.DataFrame(trades))


if __name__ == "__main__":
    import argparse
    from strategies.iterative.backtest_helpers import (
        _run_indices_uptodate,
        _run_indices_walkforward,
    )

    parser = argparse.ArgumentParser(description="Run ORB_AvgRange backtest")
    parser.add_argument(
        "--mode",
        choices=["uptodate", "walkforward"],
        default="uptodate",
        help="uptodate: full dataset  |  walkforward: IS/OOS folds (default: uptodate)",
    )
    parser.add_argument("--from-date", dest="from_date", default=None, help="YYYY-MM-DD")
    parser.add_argument("--to-date",   dest="to_date",   default=None, help="YYYY-MM-DD")
    args = parser.parse_args()

    _ENTRY = {
        "strategy_name": "orb_avg_range_iterative",
        "strategy_func": orb_avg_range_iterative,
        "data_root":     "backtest_dataset/INDICES",
        "tickers":       ["QQQ", "TQQQ"],
        "timeframes":    ["10m"],
    }
    _PARAMS = {
        "slippage":     0,
        "gap_pct":      0,
        "stop_pct":     0,
        "tp_pct":       0,
        "out_put_name": "orb_avg_range_iterative",
    }

    _t0 = tm.time()
    if args.mode == "walkforward":
        _run_indices_walkforward(_ENTRY, _PARAMS, from_date=args.from_date, to_date=args.to_date)
    else:
        _run_indices_uptodate(_ENTRY, _PARAMS, from_date=args.from_date, to_date=args.to_date)
    print(f"completado en {tm.time() - _t0:.2f}s")
