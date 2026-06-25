import os
import sys
sys.path.insert(0, os.path.abspath("."))

import logging
import pandas as pd
import time as tm
from pathlib import Path

from strategies.iterative.strategy_base import enforce_schema


def _precompute(
    df: pd.DataFrame,
    sma_window: int = 100,
    atr_period: int = 14,
    **kwargs,
) -> pd.DataFrame:
    """
    Add sma_{sma_window} and atr_{atr_period} columns to df if not already present.
    Must be called on the FULL ticker dataset (not per-day slices) for accurate history.
    ATR uses Wilder's smoothing (alpha = 1/period).
    """
    df = df.copy().sort_values("date").reset_index(drop=True)

    sma_col = f"sma_{sma_window}"
    atr_col = f"atr_{atr_period}"

    if sma_col not in df.columns:
        df[sma_col] = df["close"].rolling(window=sma_window, min_periods=sma_window).mean()
        logger.info("[precompute] %s computed (%d valid / %d rows)", sma_col, df[sma_col].notna().sum(), len(df))

    if atr_col not in df.columns:
        prev_close = df["close"].shift(1)
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"]  - prev_close).abs(),
        ], axis=1).max(axis=1)
        df[atr_col] = tr.ewm(alpha=1.0 / atr_period, min_periods=atr_period, adjust=False).mean()
        logger.info("[precompute] %s computed (%d valid / %d rows)", atr_col, df[atr_col].notna().sum(), len(df))

    return df

_LOG_DIR = Path(os.path.abspath(".")) / "logs" / "iterative"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _ch = logging.StreamHandler(sys.stdout)
    _ch.setLevel(logging.INFO)
    _ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_ch)
    _fh = logging.FileHandler(_LOG_DIR / "sma_crossover_trail.log", encoding="utf-8")
    _fh.setLevel(logging.WARNING)
    _fh.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(_fh)


def sma_crossover_trail_iterative(
    candles: pd.DataFrame,
    sma_window: int   = 100,
    atr_period: int   = 14,
    factor:     float = 2.0,
    max_hold_days: int = 1000,
    slippage:   float = 0.001,
    timeframe_minutes: int = 60,
    gap_pct: float = 0,
    stop_pct: float = 0,
    tp_pct:  float = 0,
    **kwargs,
) -> pd.DataFrame:
    """
    Generic SMA Crossover with ATR Trailing Stop — indices.

    Parameters
    ----------
    sma_window    : SMA period to use. Column sma_{sma_window} must exist in the data.
    atr_period    : ATR period for the trailing stop. Column atr_{atr_period} must exist.
    factor        : ATR multiplier.  trailing_sl = sma ± factor * atr.
    max_hold_days : Maximum bars to hold a trade expressed in trading days.
                    Converted to bars internally using timeframe_minutes.
                    Default 1000 = effectively unlimited.

    Long signal : SMA rising AND green candle with open < sma < close, no doji.

    Entry      : next bar open + slippage.
    Trailing SL: sma - factor*atr, recalculated every bar.
                 Exit when bar low breaches the trailing SL.
    No fixed TP (take_profit_price = NaN).
    EOD exit at the last bar of the day (sin restricción de hora).
    """
    SMA_COL  = f"sma_{sma_window}"
    ATR_COL  = f"atr_{atr_period}"
    STRATEGY = "sma_crossover_trail_iterative"

    bars_per_day  = max(1, int(6.5 * 60 / timeframe_minutes))
    max_hold_bars = max_hold_days * bars_per_day

    df = candles.reset_index(drop=True)
    if len(df) < 2:
        return enforce_schema(pd.DataFrame())

    if SMA_COL not in df.columns:
        logger.warning("[%s] Column '%s' not found — add it to the parquet first", STRATEGY, SMA_COL)
        return enforce_schema(pd.DataFrame())
    if ATR_COL not in df.columns:
        logger.warning("[%s] Column '%s' not found — add it to the parquet first", STRATEGY, ATR_COL)
        return enforce_schema(pd.DataFrame())

    ticker           = str(df["ticker"].iloc[0])
    current_date_str = str(df["date_str"].iloc[0])

    last_valid_idx = len(df) - 1

    trades:        list[dict] = []
    position:      str | None = None
    pending_entry: str | None = None
    pending_exit               = False

    entry_price    = 0.0
    initial_sl     = float("nan")
    sl_price       = float("nan")
    entry_time     = None
    entry_date_str = current_date_str
    entry_volume   = float("nan")
    bars_held      = 0
    signal_volume  = float("nan")
    signal_rvol    = float("nan")
    signal_prev_close = float("nan")
    trade_highs:   list[float] = []
    trade_lows:    list[float] = []

    def _record(pos, ep, xp, sl, et, xtime, t_highs, t_lows, open_trade=False):
        import math
        closed = not open_trade and not math.isnan(xp)
        if closed:
            if pos == "long":
                pnl = xp - ep
                mae = max(ep - min(t_lows),  0.0) if t_lows  else 0.0
                mfe = max(max(t_highs) - ep, 0.0) if t_highs else 0.0
            else:
                pnl = ep - xp
                mae = max(max(t_highs) - ep, 0.0) if t_highs else 0.0
                mfe = max(ep - min(t_lows),  0.0) if t_lows  else 0.0
        else:
            pnl = mae = mfe = float("nan")
        return {
            "ticker":             ticker,
            "date_str":           entry_date_str,
            "type":               pos,
            "entry_price":        round(ep, 4),
            "exit_price":         round(xp, 4) if closed else float("nan"),
            "stop_loss_price":    round(sl, 4),
            "take_profit_price":  float("nan"),
            "risk_reward_ratio":  float("nan"),
            "pnl":                round(pnl, 4) if closed else float("nan"),
            "Return":             round(pnl / ep, 4) if (closed and ep) else float("nan"),
            "MAE":                round(mae, 4) if closed else float("nan"),
            "mae_pct":            round(mae / ep * 100, 4) if (closed and ep) else float("nan"),
            "MFE":                round(mfe, 4) if closed else float("nan"),
            "mfe_pct":            round(mfe / ep * 100, 4) if (closed and ep) else float("nan"),
            "rvol_daily":         signal_rvol,
            "previous_day_close": signal_prev_close,
            "volume":             signal_volume,
            "entry_volume":       entry_volume,
            "entry_time":         et,
            "exit_time":          xtime,
            "strategy":           STRATEGY,
            "is_open":            open_trade,
        }

    for i in range(1, last_valid_idx + 1):
        row  = df.iloc[i]
        prev = df.iloc[i - 1]

        # ── 1. Execute pending exit at this bar's open ────────────────────────
        if pending_exit:
            xp = row["open"] * (1 - slippage) if position == "long" else row["open"] * (1 + slippage)
            trades.append(_record(position, entry_price, xp, initial_sl,
                                  entry_time, row["date"], trade_highs, trade_lows))
            logger.debug("%-6s  %s  EXIT  %s @ %.4f", ticker, current_date_str, position, xp)
            position     = None
            pending_exit = False
            bars_held    = 0
            trade_highs  = []
            trade_lows   = []
            # no continue — allow signal check on same bar

        # ── 2. Execute pending entry at this bar's open ───────────────────────
        if pending_entry is not None and position is None:
            sma_e = row.get(SMA_COL, float("nan"))
            atr_e = row.get(ATR_COL, float("nan"))

            if pd.isna(sma_e) or pd.isna(atr_e):
                pending_entry = None
            else:
                if pending_entry == "long":
                    entry_price = float(row["open"]) * (1 + slippage)
                    initial_sl  = float(sma_e) - factor * float(atr_e)
                else:
                    entry_price = float(row["open"]) * (1 - slippage)
                    initial_sl  = float(sma_e) + factor * float(atr_e)

                risk = (entry_price - initial_sl) if pending_entry == "long" else (initial_sl - entry_price)
                if risk <= 0:
                    logger.debug("%-6s  %s — zero/negative risk at entry, skip", ticker, current_date_str)
                    pending_entry = None
                else:
                    sl_price       = initial_sl
                    position       = pending_entry
                    entry_time     = row["date"]
                    entry_date_str = str(row["date_str"])
                    entry_volume   = float(row["volume"])
                    bars_held      = 1
                    trade_highs    = [float(row["high"])]
                    trade_lows     = [float(row["low"])]
                    pending_entry  = None
                    logger.debug("%-6s  %s  ENTER %s @ %.4f  sl=%.4f",
                                 ticker, current_date_str, position, entry_price, sl_price)

        elif position is not None:
            bars_held += 1
            trade_highs.append(float(row["high"]))
            trade_lows.append(float(row["low"]))

        # ── 3. Check trailing SL / max hold / forced EOD close ────────────────
        if position is not None:
            is_last    = i == last_valid_idx
            hit_time   = bars_held >= max_hold_bars

            sma_now = row.get(SMA_COL, float("nan"))
            atr_now = row.get(ATR_COL, float("nan"))
            if pd.notna(sma_now) and pd.notna(atr_now):
                new_sl = float(sma_now) - factor * float(atr_now)
                sl_price = max(sl_price, new_sl)  # trailing: solo sube, nunca baja

            if position == "long":
                hit_sl = float(row["low"]) <= sl_price
            else:
                hit_sl = float(row["high"]) >= sl_price

            if hit_sl or hit_time:
                if is_last:
                    # SL/time hit on the very last bar: close immediately at SL level
                    xp = sl_price * (1 - slippage) if position == "long" else sl_price * (1 + slippage)
                    trades.append(_record(position, entry_price, xp, initial_sl,
                                          entry_time, row["date"], trade_highs, trade_lows))
                    logger.debug("%-6s  %s  SL+LAST  %s @ %.4f", ticker, current_date_str, position, xp)
                    position    = None
                    bars_held   = 0
                    trade_highs = []
                    trade_lows  = []
                else:
                    # Exit executes at next bar's open; keep position set so the
                    # pending-exit handler (step 1) knows the direction
                    pending_exit = True
            elif is_last:
                # Reached the end of available data with SL not hit — trade still open.
                # No exit_price / exit_time / pnl: those are NaN until the trade closes.
                # stop_loss_price = current trailing SL (not initial) so the user can
                # see exactly where the stop is right now.
                trades.append(_record(position, entry_price, float("nan"), sl_price,
                                      entry_time, None, trade_highs, trade_lows,
                                      open_trade=True))
                logger.debug("%-6s  %s  STILL OPEN  %s  trailing_sl=%.4f",
                             ticker, current_date_str, position, sl_price)
                position    = None
                bars_held   = 0
                trade_highs = []
                trade_lows  = []

        # ── 4. Check for new signal (only when flat and not the last bar) ─────
        if position is None and pending_entry is None and not pending_exit and i < last_valid_idx:
            sma_now  = row.get(SMA_COL,  float("nan"))
            sma_prev = prev.get(SMA_COL, float("nan"))

            if pd.isna(sma_now) or pd.isna(sma_prev):
                continue

            open_  = float(row["open"])
            close_ = float(row["close"])
            rng    = float(row["high"]) - float(row["low"])
            body   = abs(close_ - open_)

            is_doji    = (body / rng) < 0.1 if rng > 0 else True
            sma_rising = float(sma_now) > float(sma_prev)
            long_cross = open_ < float(sma_now) < close_

            if not is_doji and sma_rising and long_cross:
                pending_entry     = "long"
                signal_volume     = float(row["volume"])
                signal_rvol       = float(row.get("RVOL_daily",        float("nan")))
                signal_prev_close = float(row.get("previous_day_close", float("nan")))
                logger.debug("%-6s  %s  SIGNAL long  %s=%.4f", ticker, current_date_str, SMA_COL, float(sma_now))

    logger.debug("%-6s  %s  → %d trade(s)", ticker, current_date_str, len(trades))
    return enforce_schema(pd.DataFrame(trades))


if __name__ == "__main__":
    import argparse
    from strategies.iterative.backtest_helpers import _run_indices_uptodate

    parser = argparse.ArgumentParser(description="Generic SMA Crossover Trail backtest on indices")
    parser.add_argument("--from-date",     dest="from_date",     default=None,  help="YYYY-MM-DD")
    parser.add_argument("--to-date",       dest="to_date",       default=None,  help="YYYY-MM-DD")
    parser.add_argument("--sma-window",    dest="sma_window",    type=int,   default=100,  help="SMA period (default: 100)")
    parser.add_argument("--atr-period",    dest="atr_period",    type=int,   default=14,   help="ATR period (default: 14)")
    parser.add_argument("--factor",        type=float, default=2.0,   help="ATR multiplier (default: 2.0)")
    parser.add_argument("--max-hold-days", dest="max_hold_days", type=int,   default=1000, help="Max hold days (default: 1000)")
    parser.add_argument("--tickers",       nargs="+",  default=["QQQ", "TQQQ"])
    args = parser.parse_args()

    _name = f"sma_crossover_trail_{args.sma_window}_atr{args.atr_period}_f{args.factor}"

    _ENTRY = {
        "strategy_name": "sma_crossover_trail_iterative",
        "strategy_func": sma_crossover_trail_iterative,
        "precompute_fn": _precompute,
        "data_root":     "backtest_dataset/INDICES",
        "tickers":       args.tickers,
        "timeframes":    ["1h"],
    }
    _PARAMS = {
        "slippage":      0.001,
        "gap_pct":       0,
        "stop_pct":      0,
        "tp_pct":        0,
        "sma_window":    args.sma_window,
        "atr_period":    args.atr_period,
        "factor":        args.factor,
        "max_hold_days": args.max_hold_days,
        "out_put_name":  _name,
    }

    _t0 = tm.time()
    _run_indices_uptodate(_ENTRY, _PARAMS, from_date=args.from_date, to_date=args.to_date)
    print(f"Completado en {tm.time() - _t0:.2f}s")
