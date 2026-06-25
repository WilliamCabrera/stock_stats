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

_file_handler = logging.FileHandler(_LOG_DIR / "orb_first_candle.log", encoding="utf-8")
_file_handler.setLevel(logging.WARNING)
_file_handler.setFormatter(logging.Formatter(
    "%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
))

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(_console_handler)
logger.addHandler(_file_handler)


def orb_first_candle_iterative(
    candles: pd.DataFrame,
    gap_pct: float = 0.40,
    stop_pct: float = 0.50,
    tp_pct: float = 0.20,
    slippage: float = 0.001,
    timeframe_minutes: int = 5,
    ticker_parquet_path: "Path | None" = None,
) -> pd.DataFrame:
    """
    Opening Range Breakout — First Candle Direction (ORB_FirstCandle).

    Signal: colour of the first 5-min candle (9:30–9:35 AM ET).
      Green (close > open) → LONG   at the open of the 9:35 bar
      Red   (close < open) → SHORT  at the open of the 9:35 bar

    Long
      SL   : low  of the 9:30 candle
      Risk : entry − SL
      TP   : entry + 10 × risk

    Short
      SL   : high of the 9:30 candle
      Risk : SL − entry
      TP   : entry − 10 × risk

    Forced EOD exit at the close of the last bar before 4 PM.
    """
    STRATEGY   = "orb_first_candle_iterative"
    CLOSE_HOUR = 16
    ORB_HOUR   = 9
    ORB_MINUTE = 30
    TP_MULTIPLE = 10
    df = candles.reset_index(drop=True)
    if len(df) < 3:
        return enforce_schema(pd.DataFrame())

    ticker           = df["ticker"].iloc[0]
    current_date_str = df["date_str"].iloc[0]

    # Locate the 9:30 AM candle (the opening range signal bar)
    orb_mask = (df["date"].dt.hour == ORB_HOUR) & (df["date"].dt.minute == ORB_MINUTE)
    if not orb_mask.any():
        logger.debug("%-6s  %s — no 9:30 candle, skip", ticker, current_date_str)
        return enforce_schema(pd.DataFrame())

    orb_idx = int(orb_mask.idxmax())
    orb_row = df.iloc[orb_idx]

    orb_open  = float(orb_row["open"])
    orb_close = float(orb_row["close"])
    orb_high  = float(orb_row["high"])
    orb_low   = float(orb_row["low"])

    if orb_close == orb_open:
        logger.debug("%-6s  %s — doji, skip", ticker, current_date_str)
        return enforce_schema(pd.DataFrame())

    signal_direction = "long" if orb_close > orb_open else "short"

    before_close = df["date"].dt.hour < CLOSE_HOUR
    if not before_close.any():
        return enforce_schema(pd.DataFrame())
    last_valid_idx = int(before_close[before_close].index[-1])

    # Entry fires on the bar right after the ORB candle
    entry_bar_idx = orb_idx + 1
    if entry_bar_idx > last_valid_idx:
        logger.debug("%-6s  %s — no valid bar after ORB candle, skip", ticker, current_date_str)
        return enforce_schema(pd.DataFrame())

    pending_entry     = signal_direction
    signal_volume     = float(orb_row["volume"])
    signal_rvol       = float(orb_row.get("RVOL_daily", float("nan")))
    signal_prev_close = float(orb_row.get("previous_day_close", float("nan")))

    trades:       list[dict] = []
    position      = None        # None | "long" | "short"
    pending_exit  = False
    entry_price   = sl_price = tp_price = 0.0
    entry_time    = entry_volume = entry_date_str = None
    trade_highs:  list[float] = []
    trade_lows:   list[float] = []

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

    for i in range(entry_bar_idx, last_valid_idx + 1):
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
                sl_price    = orb_low
                risk        = entry_price - sl_price
            else:
                entry_price = row["open"] * (1 - slippage)
                sl_price    = orb_high
                risk        = sl_price - entry_price

            if risk <= 0:
                logger.debug("%-6s  %s — zero/negative risk after slippage, skip", ticker, current_date_str)
                return enforce_schema(pd.DataFrame())

            tp_price       = (entry_price + TP_MULTIPLE * risk) if pending_entry == "long" else (entry_price - TP_MULTIPLE * risk)
            position       = pending_entry
            entry_time     = row["date"]
            entry_volume   = float(row["volume"])
            entry_date_str = row["date_str"]
            pending_entry  = None
            trade_highs    = [float(row["high"])]
            trade_lows     = [float(row["low"])]

        elif position is not None:
            trade_highs.append(float(row["high"]))
            trade_lows.append(float(row["low"]))

        # ── Check SL / TP / forced EOD close ────────────────────────────────
        if position is not None:
            is_last = i == last_valid_idx
            if position == "long":
                hit_tp = row["high"] >= tp_price
                hit_sl = row["low"]  <= sl_price
            else:
                hit_tp = row["low"]  <= tp_price
                hit_sl = row["high"] >= sl_price

            if (hit_tp or hit_sl) and not is_last:
                pending_exit = True
            elif is_last:
                xp = row["close"] * (1 - slippage) if position == "long" else row["close"] * (1 + slippage)
                trades.append(_record(position, entry_price, xp, sl_price, tp_price,
                                      entry_time, row["date"], trade_highs, trade_lows))
                position    = None
                trade_highs = []
                trade_lows  = []

    logger.debug("%-6s  %s  → %d trade(s)", ticker, current_date_str, len(trades))
    return enforce_schema(pd.DataFrame(trades))


if __name__ == "__main__":
    import argparse
    from strategies.iterative.backtest_helpers import (
        _run_indices_uptodate,
        _run_indices_walkforward,
    )

    parser = argparse.ArgumentParser(description="Run ORB_FirstCandle backtest")
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
        "strategy_name": "orb_first_candle_iterative",
        "strategy_func": orb_first_candle_iterative,
        "data_root":     "backtest_dataset/INDICES",
        "tickers":       ["QQQ", "TQQQ"],
        "timeframes":    ["5m"],
    }
    _PARAMS = {
        "slippage":     0,
        "gap_pct":      0,
        "stop_pct":     0,
        "tp_pct":       0,
        "out_put_name": "orb_first_candle_iterative",
    }

    _t0 = tm.time()
    if args.mode == "walkforward":
        _run_indices_walkforward(_ENTRY, _PARAMS, from_date=args.from_date, to_date=args.to_date)
    else:
        _run_indices_uptodate(_ENTRY, _PARAMS, from_date=args.from_date, to_date=args.to_date)
    print(f"completado en {tm.time() - _t0:.2f}s")
