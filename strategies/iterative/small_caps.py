import os
import sys
sys.path.insert(0, os.path.abspath("."))
import vectorbt as vbt
import pandas as pd
pd.options.display.float_format = '{:.3f}'.format

import json
import logging
import functools
import numpy as np
#from utils import helpers
#from utils import helpers as utils_helpers, trade_metrics as tme
import itertools
import time as tm
from functools import wraps
#from small_caps_strategies import commons
from pprint import pprint
from pathlib import Path
import webbrowser
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from app.utils.market_utils import  append_single_parquet
from app.utils.charts import plot_candles_df, trades_to_markers, CHARTS_DIR, _write_fullscreen_html
from app.utils.indicators import compute_close_atr_band, compute_vwap
from app.utils.massive import fetch_candles
from app.utils.trade_metrics import (analysis_and_plot, summary_report, get_mae_mfe)
from strategies.iterative.strategy_base import enforce_schema

DATASET_ROOT   = Path(os.path.abspath(".")) / "backtest_dataset" / "full"
DATASET_ROOT_1 = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_LOG_DIR = Path(os.path.abspath(".")) / "logs" / "iterative"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setLevel(logging.INFO)
_console_handler.setFormatter(logging.Formatter("%(message)s"))

_file_handler = logging.FileHandler(_LOG_DIR / "small_caps.log", encoding="utf-8")
_file_handler.setLevel(logging.WARNING)
_file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(_console_handler)
logger.addHandler(_file_handler)


def backside_short_lower_low_fix_stop_iterative(
    candles: pd.DataFrame,
    gap_pct: float = 0.40,
    stop_pct: float = 0.5,
    tp_pct: float = 0.20,
    slippage: float = 0.001,
    timeframe_minutes: int = 5,
) -> pd.DataFrame:
    """
    Short iterativo: vela roja que rompe el low de la vela verde anterior.

    Condiciones de entrada (barra i):
      1. Vela actual roja  (close < open)
      2. Vela anterior verde (prev_close > prev_open)
      3. close[i] < low[i-1]  (lower low)
      4. high[i] >= previous_day_close * (1 + gap_pct)
      5. open[i] > vwap[i-1]  (abre sobre VWAP de la vela anterior — sin lookahead)
      6. bar[i-1] es el verdadero anterior (sin huecos de tiempo)

    Entrada:  open de la vela siguiente (OVS), siempre que OVS sea < 16:00
    SL:       OVS * (1 + stop_pct)
    TP:       OVS * (1 - tp_pct)
    Cierre forzado en la última vela antes de las 16:00.
    """
    STRATEGY = f"backside_short_lower_low_fix_stop_iterative_{gap_pct}_{stop_pct}_{tp_pct}"
    CLOSE_HOUR = 16
    expected_delta = pd.Timedelta(minutes=timeframe_minutes)

    df = candles.reset_index(drop=True)
    if len(df) < 3:
        logger.debug("%-6s  skipped — less than 3 candles", candles["ticker"].iloc[0] if len(candles) else "?")
        return pd.DataFrame()

    ticker = df["ticker"].iloc[0]

    # ── Índice de la última vela válida (antes de las 16:00) ─────────────
    before_close = df["date"].dt.hour < CLOSE_HOUR
    if not before_close.any():
        return pd.DataFrame()
    last_valid_idx = before_close[before_close].index[-1]

    # ── Condición: sin hueco — bar[i] es exactamente 1 timeframe después ─
    no_gap = df["date"].diff() == expected_delta  # True en i si bar[i-1] es el real anterior

    prev_close = df["close"].shift(1)
    prev_open  = df["open"].shift(1)
    prev_low   = df["low"].shift(1)
    prev_vwap  = df["vwap"].shift(1)

    red        = df["close"] < df["open"]
    green_prev = prev_close > prev_open
    lower_low  = df["close"] < prev_low
    gap_cond   = df["high"] >= df["previous_day_close"] * (1 + gap_pct)
    above_vwap = df["open"] > prev_vwap

    signal = red & green_prev & lower_low & gap_cond & above_vwap & no_gap

    # debug_df = df[["ticker", "date", "open", "high", "low", "close", "vwap", "previous_day_close"]].copy()
    # debug_df["prev_close"]  = prev_close
    # debug_df["prev_open"]   = prev_open
    # debug_df["prev_low"]    = prev_low
    # debug_df["prev_vwap"]   = prev_vwap
    # debug_df["no_gap"]      = no_gap
    # debug_df["red"]         = red
    # debug_df["green_prev"]  = green_prev
    # debug_df["lower_low"]   = lower_low
    # debug_df["gap_cond"]    = gap_cond
    # debug_df["above_vwap"]  = above_vwap
    # debug_df["signal"]      = signal
    
    

    trades: list[dict] = []
    position       = None
    pending_entry  = False
    pending_exit   = False
    entry_price    = sl_price = tp_price = 0.0
    entry_time     = entry_volume = entry_date_str = None
    signal_volume  = signal_rvol = signal_prev_close = None
    trade_highs: list[float] = []
    trade_lows:  list[float] = []

    #print('===== DEBUG: backside_short_lower_low_fix_stop_iterative =====')
    #print(debug_df[["date", "open", "high", "low", "close", "vwap", "previous_day_close", "prev_close", "prev_open", "prev_low", "prev_vwap", "no_gap", "red", "green_prev", "lower_low", "gap_cond", "above_vwap", "signal"]][0:35])

    for i in range(1, last_valid_idx + 1):
        row = df.iloc[i]

        # ── Cerrar posición al open de esta vela (señal en barra anterior) ─
        if pending_exit:
            exit_price = row["open"] * (1 + slippage)   # short: compra para cerrar — paga más
            pnl = entry_price - exit_price
            rr  = (entry_price - tp_price) / (sl_price - entry_price)
            mae     = max(trade_highs) - entry_price
            mfe     = entry_price - min(trade_lows)
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
            pending_exit = False
            trade_highs = []
            trade_lows  = []
            continue   # no buscar nueva señal en la barra de salida

        # ── Abrir posición al open de esta vela (señal en barra anterior) ──
        if pending_entry:
            entry_price    = row["open"] * (1 - slippage)   # short: vende — recibe menos
            sl_price       = entry_price * (1 + stop_pct)
            tp_price       = entry_price * (1 - tp_pct)
            entry_time     = row["date"]
            entry_volume   = row["volume"]
            entry_date_str = row["date_str"]
            position       = "short"
            pending_entry  = False
            trade_highs    = [row["high"]]
            trade_lows     = [row["low"]]

        # ── Acumular highs/lows en barras intermedias ─────────────────────
        elif position == "short":
            trade_highs.append(row["high"])
            trade_lows.append(row["low"])

        # ── Detectar SL/TP o cierre forzado EOD ──────────────────────────
        if position == "short":
            hit_tp  = row["low"]  <= tp_price
            hit_sl  = row["high"] >= sl_price
            is_last = i == last_valid_idx

            if (hit_tp or hit_sl) and not is_last:
                # Salida al open de la siguiente vela
                pending_exit = True

            elif is_last:
                # Cierre forzado pre-16:00 al close de la última vela válida
                exit_price = row["close"] * (1 + slippage)   # short: compra para cerrar — paga más
                pnl = entry_price - exit_price
                rr  = (entry_price - tp_price) / (sl_price - entry_price)
                mae     = max(trade_highs) - entry_price
                mfe     = entry_price - min(trade_lows)
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

        # ── Detectar señal — entrada en la próxima barra ─────────────────
        # La próxima barra debe existir, ser consecutiva y estar antes de 16:00
        next_i = i + 1
        if (
            position is None
            and signal.iloc[i]
            and next_i <= last_valid_idx
            and no_gap.iloc[next_i]        # OVS también es consecutiva
        ):
            pending_entry     = True
            signal_volume     = row["volume"]
            signal_rvol       = row["RVOL_daily"]
            signal_prev_close = row["previous_day_close"]
            logger.debug("SIGNAL  %-6s  %s  bar=%s  close=%.4f", ticker, row["date_str"], row["date"], row["close"])

    n_trades = len(trades)
    logger.debug("%-6s  %s  → %d trade(s)", ticker, df["date_str"].iloc[0], n_trades)
    return enforce_schema(pd.DataFrame(trades))

def gap_crap_iterative(
    candles: pd.DataFrame,
    gap_pct: float = 0.40,
    stop_pct: float = 0.5,
    tp_pct: float = 0.20,
    slippage: float = 0.001,
    timeframe_minutes: int = 5,
) -> pd.DataFrame:
    """
    Short iterativo: si el close de la vela de las 9:25 AM supera gap_pct sobre el
    cierre del día anterior, abre short al open de la siguiente barra.

    Condición de entrada (barra 9:25):
      1. close[9:25] >= previous_day_close * (1 + gap_pct)

    Entrada:  open de la vela siguiente (sin filtro no_gap — la barra de 9:25 es
              pre-mercado y la siguiente siempre es 9:30, el open regular)
    SL:       entry * (1 + stop_pct)
    TP:       entry * (1 - tp_pct)
    Cierre forzado en la última vela antes de las 16:00.
    """
    STRATEGY = f"gap_crap_iterative_{gap_pct}_{stop_pct}_{tp_pct}"
    CLOSE_HOUR = 16

    df = candles.reset_index(drop=True)
    if len(df) < 3:
        logger.debug("%-6s  skipped — less than 3 candles", candles["ticker"].iloc[0] if len(candles) else "?")
        return pd.DataFrame()

    ticker = df["ticker"].iloc[0]

    before_close = df["date"].dt.hour < CLOSE_HOUR
    if not before_close.any():
        return pd.DataFrame()
    last_valid_idx = before_close[before_close].index[-1]

    signal_time = (df["date"].dt.hour == 9) & (df["date"].dt.minute == 25)
    gap_cond    = df["close"] >= df["previous_day_close"] * (1 + gap_pct)
    signal      = signal_time & gap_cond

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

        if pending_exit:
            exit_price = row["open"] * (1 + slippage)
            pnl = entry_price - exit_price
            rr  = (entry_price - tp_price) / (sl_price - entry_price)
            mae = max(trade_highs) - entry_price
            mfe = entry_price - min(trade_lows)
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
            position     = None
            pending_exit = False
            trade_highs  = []
            trade_lows   = []
            continue

        if pending_entry:
            entry_price    = row["open"] * (1 - slippage)
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

        if position == "short":
            hit_tp  = row["low"]  <= tp_price
            hit_sl  = row["high"] >= sl_price
            is_last = i == last_valid_idx

            if (hit_tp or hit_sl) and not is_last:
                pending_exit = True
            elif is_last:
                exit_price = row["close"] * (1 + slippage)
                pnl = entry_price - exit_price
                rr  = (entry_price - tp_price) / (sl_price - entry_price)
                mae = max(trade_highs) - entry_price
                mfe = entry_price - min(trade_lows)
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

        # Señal: solo en la barra de las 9:25 AM — no se verifica no_gap porque
        # la siguiente barra siempre es el open regular (9:30)
        next_i = i + 1
        if (
            position is None
            and signal.iloc[i]
            and next_i <= last_valid_idx
        ):
            pending_entry     = True
            signal_volume     = row["volume"]
            signal_rvol       = row["RVOL_daily"]
            signal_prev_close = row["previous_day_close"]
            logger.debug("SIGNAL  %-6s  %s  bar=%s  close=%.4f", ticker, row["date_str"], row["date"], row["close"])

    n_trades = len(trades)
    logger.debug("%-6s  %s  → %d trade(s)", ticker, df["date_str"].iloc[0], n_trades)
    return enforce_schema(pd.DataFrame(trades))


def short_push_exhaustion_iterative(
    candles: pd.DataFrame,
    gap_pct: float = 0.40,
    stop_pct: float = 0.5,
    tp_pct: float = 0.20,
    slippage: float = 0.001,
    timeframe_minutes: int = 5,
) -> pd.DataFrame:
    """
    Short iterativo: desaceleración del push alcista medida por una vela roja con
    topping tail >= 1.5x la distancia (open - low), por encima de VWAP, con volumen
    superior al de la barra anterior y > 40 000.

    Condiciones de entrada (barra i):
      1. Vela roja (close < open)
      2. Topping tail (high - open) >= 1.5 * (open - low)   — upper shadow dominante
      3. open[i] > vwap[i-1]                                 — por encima de VWAP
      4. volume[i] > volume[i-1]  AND  volume[i] > 40 000
      5. (high[i] - low[i]) >= 0.20 * (high[i-1] - low[i-1])  — range mínimo 20% del anterior
      6. high[i] >= previous_day_close * (1 + gap_pct)       — contexto de gap-up
      7. bar[i-1] es el verdadero anterior (sin huecos)

    Entrada:  open de la vela siguiente
    SL:       entry * (1 + stop_pct)
    TP:       entry * (1 - tp_pct)
    Cierre forzado en la última vela antes de las 16:00.
    """
    STRATEGY = f"short_push_exhaustion_iterative_{gap_pct}_{stop_pct}_{tp_pct}"
    CLOSE_HOUR = 16
    expected_delta = pd.Timedelta(minutes=timeframe_minutes)
    MIN_VOLUME = 40_000

    df = candles.reset_index(drop=True)
    if len(df) < 3:
        logger.debug("%-6s  skipped — less than 3 candles", candles["ticker"].iloc[0] if len(candles) else "?")
        return pd.DataFrame()

    ticker = df["ticker"].iloc[0]

    before_close = df["date"].dt.hour < CLOSE_HOUR
    if not before_close.any():
        return pd.DataFrame()
    last_valid_idx = before_close[before_close].index[-1]

    no_gap      = df["date"].diff() == expected_delta
    prev_vwap   = df["vwap"].shift(1)
    prev_volume = df["volume"].shift(1)

    red           = df["close"] < df["open"]
    topping_tail  = df["high"] - df["open"]
    low_to_open   = (df["open"] - df["low"]).clip(lower=1e-8)   # evitar /0
    exhaustion    = topping_tail >= 1.5 * low_to_open
    above_vwap    = df["open"] > prev_vwap
    vol_surge     = (df["volume"] > prev_volume) & (df["volume"] > MIN_VOLUME)
    gap_cond      = df["high"] >= df["previous_day_close"] * (1 + gap_pct)
    prev_range    = (df["high"].shift(1) - df["low"].shift(1)).clip(lower=1e-8)
    range_filter  = (df["high"] - df["low"]) >= 0.20 * prev_range

    signal = red & exhaustion & above_vwap & vol_surge & gap_cond & range_filter & no_gap

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

        if pending_exit:
            exit_price = row["open"] * (1 + slippage)
            pnl = entry_price - exit_price
            rr  = (entry_price - tp_price) / (sl_price - entry_price)
            mae = max(trade_highs) - entry_price
            mfe = entry_price - min(trade_lows)
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
            position     = None
            pending_exit = False
            trade_highs  = []
            trade_lows   = []
            continue

        if pending_entry:
            entry_price    = row["open"] * (1 - slippage)
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

        if position == "short":
            hit_tp  = row["low"]  <= tp_price
            hit_sl  = row["high"] >= sl_price
            is_last = i == last_valid_idx

            if (hit_tp or hit_sl) and not is_last:
                pending_exit = True
            elif is_last:
                exit_price = row["close"] * (1 + slippage)
                pnl = entry_price - exit_price
                rr  = (entry_price - tp_price) / (sl_price - entry_price)
                mae = max(trade_highs) - entry_price
                mfe = entry_price - min(trade_lows)
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
            logger.debug("SIGNAL  %-6s  %s  bar=%s  close=%.4f", ticker, row["date_str"], row["date"], row["close"])

    n_trades = len(trades)
    logger.debug("%-6s  %s  → %d trade(s)", ticker, df["date_str"].iloc[0], n_trades)
    return enforce_schema(pd.DataFrame(trades))


def push_rejection_iterative(
    candles: pd.DataFrame,
    gap_pct: float = 0.40,
    stop_pct: float = 0.5,
    tp_pct: float = 0.20,
    slippage: float = 0.001,
    timeframe_minutes: int = 5,
) -> pd.DataFrame:
    """
    Short iterativo: rechazo del push alcista cuando la vela roja cruza VWAP de
    arriba hacia abajo con fuerza suficiente.

    Condiciones de entrada (barra i):
      1. Vela roja (close < open)
      2. open[i] > vwap[i]  AND  close[i] < vwap[i]  — cruza VWAP de arriba a abajo
      3. (high[i] - low[i]) >= 0.30 * (high[i-1] - low[i-1])  — range >= 30% de la barra anterior
      4. Vela anterior verde (close[i-1] > open[i-1])
      5. (high[i-1] - low[i-1]) >= 0.40 * open[i-1]  — push anterior >= 40% del open
      6. (open[i] - close[i]) > (close[i] - low[i])   — cuerpo mayor que bottom tail
      7. bar[i-1] es el verdadero anterior (sin huecos)

    Entrada:  open de la vela siguiente
    SL:       entry * (1 + stop_pct)
    TP:       entry * (1 - tp_pct)
    Cierre forzado en la última vela antes de las 16:00.
    """
    STRATEGY = f"push_rejection_iterative_{gap_pct}_{stop_pct}_{tp_pct}"
    CLOSE_HOUR = 16
    expected_delta = pd.Timedelta(minutes=timeframe_minutes)

    df = candles.reset_index(drop=True)
    if len(df) < 3:
        logger.debug("%-6s  skipped — less than 3 candles", candles["ticker"].iloc[0] if len(candles) else "?")
        return pd.DataFrame()

    ticker = df["ticker"].iloc[0]

    before_close = df["date"].dt.hour < CLOSE_HOUR
    if not before_close.any():
        return pd.DataFrame()
    last_valid_idx = before_close[before_close].index[-1]

    no_gap = df["date"].diff() == expected_delta

    red          = df["close"] < df["open"]
    vwap_cross   = (df["open"] > df["vwap"]) & (df["close"] < df["vwap"])
    cur_range    = df["high"] - df["low"]
    prev_range   = (df["high"].shift(1) - df["low"].shift(1)).clip(lower=1e-8)
    range_30pct  = cur_range >= 0.30 * prev_range
    green_prev   = df["close"].shift(1) > df["open"].shift(1)
    push_size    = prev_range >= 0.40 * df["open"].shift(1).clip(lower=1e-8)
    body         = df["open"] - df["close"]
    bottom_tail  = (df["close"] - df["low"]).clip(lower=0)
    body_gt_tail = body > bottom_tail

    signal = red & vwap_cross & range_30pct & green_prev & push_size & body_gt_tail & no_gap

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

        if pending_exit:
            exit_price = row["open"] * (1 + slippage)
            pnl = entry_price - exit_price
            rr  = (entry_price - tp_price) / (sl_price - entry_price)
            mae = max(trade_highs) - entry_price
            mfe = entry_price - min(trade_lows)
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
            position     = None
            pending_exit = False
            trade_highs  = []
            trade_lows   = []
            continue

        if pending_entry:
            entry_price    = row["open"] * (1 - slippage)
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

        if position == "short":
            hit_tp  = row["low"]  <= tp_price
            hit_sl  = row["high"] >= sl_price
            is_last = i == last_valid_idx

            if (hit_tp or hit_sl) and not is_last:
                pending_exit = True
            elif is_last:
                exit_price = row["close"] * (1 + slippage)
                pnl = entry_price - exit_price
                rr  = (entry_price - tp_price) / (sl_price - entry_price)
                mae = max(trade_highs) - entry_price
                mfe = entry_price - min(trade_lows)
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
            logger.debug("SIGNAL  %-6s  %s  bar=%s  close=%.4f", ticker, row["date_str"], row["date"], row["close"])

    n_trades = len(trades)
    logger.debug("%-6s  %s  → %d trade(s)", ticker, df["date_str"].iloc[0], n_trades)
    return enforce_schema(pd.DataFrame(trades))

def sma9_momentum_long_iterative(
    candles: pd.DataFrame,
    min_dist_pct: float = 0.08,
    max_dist_pct: float = 1.0,
    min_volume: int = 40_000,
    target_rr: float = 10.0,
    max_hold_hours: float = 1,
    slippage: float = 0.001,
    timeframe_minutes: int = 5,
    **kwargs,  # absorbs gap_pct / stop_pct / tp_pct passed by run_backtest
) -> pd.DataFrame:
    """
    Long iterativo: vela verde con SMA9 en pendiente positiva.

    Condiciones de entrada (barra i):
      1. Vela verde (close > open)
      2. Pendiente positiva SMA9: sma_9[i] > sma_9[i-1]
      3. volume[i] >= min_volume  (default 40 000)
      4. (close[i] - day_low[i]) / day_low[i] entre min_dist_pct y max_dist_pct
         donde day_low[i] = min(low[0..i]) — mínimo intradiario acumulado
      5. No doji: cuerpo / rango total >= 10%
      6. bar[i-1] es el verdadero anterior (sin huecos de tiempo)

    Entrada:  open de la vela siguiente (long: compra)
    SL:       day_low en la barra de señal
    TP:       entry + target_rr × (entry − SL)
    Tiempo:   si a las max_hold_hours desde la entrada no se alcanzó TP/SL,
              cierra al open de la siguiente vela
    Cierre forzado en la última vela antes de las 16:00.
    """
    STRATEGY = f"sma9_momentum_long_iterative_{min_dist_pct}_{max_dist_pct}_{target_rr}"
    CLOSE_HOUR = 16
    expected_delta = pd.Timedelta(minutes=timeframe_minutes)
    MIN_BODY_RATIO = 0.10

    df = candles.reset_index(drop=True)
    if len(df) < 3:
        logger.debug("%-6s  skipped — less than 3 candles", candles["ticker"].iloc[0] if len(candles) else "?")
        return pd.DataFrame()

    ticker = df["ticker"].iloc[0]

    before_close = df["date"].dt.hour < CLOSE_HOUR
    if not before_close.any():
        return pd.DataFrame()
    last_valid_idx = before_close[before_close].index[-1]

    no_gap    = df["date"].diff() == expected_delta
    day_low   = df["low"].cummin()
    prev_sma9 = df["sma_9"].shift(1)

    green      = df["close"] > df["open"]
    sma9_up    = df["sma_9"] > prev_sma9
    vol_cond   = df["volume"] >= min_volume
    dist_pct   = (df["close"] - day_low) / day_low.clip(lower=1e-8)
    dist_cond  = (dist_pct > min_dist_pct) & (dist_pct < max_dist_pct)
    body_ratio = (df["close"] - df["open"]) / (df["high"] - df["low"]).clip(lower=1e-8)
    no_doji    = body_ratio >= MIN_BODY_RATIO

    signal = green & sma9_up & vol_cond & dist_cond & no_doji & no_gap

    hold_delta = pd.Timedelta(hours=max_hold_hours)

    trades: list[dict] = []
    position           = None
    pending_entry      = False
    pending_exit       = False
    entry_price        = sl_price = tp_price = 0.0
    entry_time         = entry_volume = entry_date_str = None
    signal_volume      = signal_rvol = signal_prev_close = None
    signal_day_low     = None
    time_exit_deadline = None
    trade_highs: list[float] = []
    trade_lows:  list[float] = []

    for i in range(1, last_valid_idx + 1):
        row = df.iloc[i]

        # ── Cerrar posición al open de esta vela ─────────────────────────
        if pending_exit:
            exit_price = row["open"] * (1 - slippage)
            pnl        = exit_price - entry_price
            rr_actual  = (tp_price - entry_price) / (entry_price - sl_price) if (entry_price - sl_price) > 1e-8 else 0
            mae        = entry_price - min(trade_lows)
            mfe        = max(trade_highs) - entry_price
            trades.append({
                "ticker":             ticker,
                "date_str":           entry_date_str,
                "type":               "long",
                "entry_price":        entry_price,
                "exit_price":         exit_price,
                "stop_loss_price":    round(sl_price, 4),
                "take_profit_price":  round(tp_price, 4),
                "risk_reward_ratio":  round(rr_actual, 4),
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
            position           = None
            pending_exit       = False
            time_exit_deadline = None
            trade_highs        = []
            trade_lows         = []
            continue

        # ── Abrir posición al open de esta vela ──────────────────────────
        if pending_entry:
            entry_price = row["open"] * (1 + slippage)
            sl_price    = signal_day_low
            risk        = entry_price - sl_price
            if risk <= 1e-8:
                # next bar opened below day low — skip trade
                pending_entry = False
            else:
                tp_price           = entry_price + target_rr * risk
                entry_time         = row["date"]
                entry_volume       = row["volume"]
                entry_date_str     = row["date_str"]
                time_exit_deadline = entry_time + hold_delta
                position           = "long"
                pending_entry      = False
                trade_highs        = [row["high"]]
                trade_lows         = [row["low"]]

        elif position == "long":
            trade_highs.append(row["high"])
            trade_lows.append(row["low"])

        # ── Detectar SL/TP/tiempo o cierre forzado EOD ───────────────────
        if position == "long":
            hit_tp   = row["high"] >= tp_price
            hit_sl   = row["low"]  <= sl_price
            hit_time = row["date"] >= time_exit_deadline
            is_last  = i == last_valid_idx

            if (hit_tp or hit_sl or hit_time) and not is_last:
                pending_exit = True

            elif is_last:
                exit_price = row["close"] * (1 - slippage)
                pnl        = exit_price - entry_price
                rr_actual  = (tp_price - entry_price) / (entry_price - sl_price) if (entry_price - sl_price) > 1e-8 else 0
                mae        = entry_price - min(trade_lows)
                mfe        = max(trade_highs) - entry_price
                trades.append({
                    "ticker":             ticker,
                    "date_str":           entry_date_str,
                    "type":               "long",
                    "entry_price":        entry_price,
                    "exit_price":         exit_price,
                    "stop_loss_price":    round(sl_price, 4),
                    "take_profit_price":  round(tp_price, 4),
                    "risk_reward_ratio":  round(rr_actual, 4),
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
                position           = None
                time_exit_deadline = None
                trade_highs        = []
                trade_lows         = []

        # ── Detectar señal — entrada en la próxima barra ─────────────────
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
            signal_day_low    = day_low.iloc[i]
            logger.debug("SIGNAL  %-6s  %s  bar=%s  close=%.4f", ticker, row["date_str"], row["date"], row["close"])

    n_trades = len(trades)
    logger.debug("%-6s  %s  → %d trade(s)", ticker, df["date_str"].iloc[0], n_trades)
    return enforce_schema(pd.DataFrame(trades))


def _dynamic_target_rr(dist_pct: float) -> float:
    """
    Target R:R tiered by how far the entry close is from the intraday low.

    dist_pct = (close − day_low) / day_low

      0 – 10 %  →  10
     10 – 20 %  →   8
     20 – 30 %  →   7
     30 – 40 %  →   6
     40 – 50 %  →   5
     50 – 60 %  →   4
     60 – 70 %  →   3
     70 %+      →   2
    """
    if dist_pct < 0.10: return 10.0
    if dist_pct < 0.20: return  4.0
    if dist_pct < 0.30: return  4.0
    if dist_pct < 0.40: return  3.0
    if dist_pct < 0.50: return  2.0
    if dist_pct < 0.60: return  2.0
    if dist_pct < 0.70: return  1.0
    return 1.0


def sma9_momentum_long_dynrr_iterative(
    candles: pd.DataFrame,
    min_dist_pct: float = 0.08,
    max_dist_pct: float = 0.80,
    min_volume: int = 40_000,
    max_hold_hours: float = 1.0,
    slippage: float = 0.001,
    timeframe_minutes: int = 5,
    **kwargs,
) -> pd.DataFrame:
    """
    Igual que sma9_momentum_long_iterative pero con target_rr dinámico:
    cuanto más lejos está el cierre del mínimo intradiario, menor el objetivo.

      dist  0–10 %  → rr 10
      dist 10–20 %  → rr  8
      dist 20–30 %  → rr  7
      … (ver _dynamic_target_rr)

    El rr se fija en la barra de señal y no cambia durante el trade.
    """
    STRATEGY = "sma9_momentum_long_dynrr_iterative"
    CLOSE_HOUR = 16
    expected_delta = pd.Timedelta(minutes=timeframe_minutes)
    MIN_BODY_RATIO = 0.10

    df = candles.reset_index(drop=True)
    if len(df) < 3:
        return pd.DataFrame()

    ticker = df["ticker"].iloc[0]

    before_close = df["date"].dt.hour < CLOSE_HOUR
    if not before_close.any():
        return pd.DataFrame()
    last_valid_idx = before_close[before_close].index[-1]

    no_gap    = df["date"].diff() == expected_delta
    day_low   = df["low"].cummin()
    prev_sma9 = df["sma_9"].shift(1)

    green      = df["close"] > df["open"]
    sma9_up    = df["sma_9"] > prev_sma9
    vol_cond   = df["volume"] >= min_volume
    dist_pct   = (df["close"] - day_low) / day_low.clip(lower=1e-8)
    dist_cond  = (dist_pct > min_dist_pct) & (dist_pct < max_dist_pct)
    body_ratio = (df["close"] - df["open"]) / (df["high"] - df["low"]).clip(lower=1e-8)
    no_doji    = body_ratio >= MIN_BODY_RATIO

    signal = green & sma9_up & vol_cond & dist_cond & no_doji & no_gap

    hold_delta = pd.Timedelta(hours=max_hold_hours)

    trades: list[dict] = []
    position           = None
    pending_entry      = False
    pending_exit       = False
    entry_price        = sl_price = tp_price = 0.0
    entry_time         = entry_volume = entry_date_str = None
    signal_volume      = signal_rvol = signal_prev_close = None
    signal_day_low     = None
    signal_target_rr   = None
    time_exit_deadline = None
    trade_highs: list[float] = []
    trade_lows:  list[float] = []

    def _record(exit_price, exit_time):
        pnl       = exit_price - entry_price
        rr_actual = (tp_price - entry_price) / (entry_price - sl_price) if (entry_price - sl_price) > 1e-8 else 0
        mae       = entry_price - min(trade_lows)
        mfe       = max(trade_highs) - entry_price
        return {
            "ticker":             ticker,
            "date_str":           entry_date_str,
            "type":               "long",
            "entry_price":        entry_price,
            "exit_price":         exit_price,
            "stop_loss_price":    round(sl_price, 4),
            "take_profit_price":  round(tp_price, 4),
            "risk_reward_ratio":  round(rr_actual, 4),
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
            "exit_time":          exit_time,
            "strategy":           STRATEGY,
        }

    for i in range(1, last_valid_idx + 1):
        row = df.iloc[i]

        # ── Cerrar posición al open de esta vela ─────────────────────────
        if pending_exit:
            trades.append(_record(row["open"] * (1 - slippage), row["date"]))
            position           = None
            pending_exit       = False
            time_exit_deadline = None
            trade_highs        = []
            trade_lows         = []
            continue

        # ── Abrir posición al open de esta vela ──────────────────────────
        if pending_entry:
            entry_price = row["open"] * (1 + slippage)
            sl_price    = signal_day_low
            risk        = entry_price - sl_price
            if risk <= 1e-8:
                pending_entry = False
            else:
                tp_price           = entry_price + signal_target_rr * risk
                entry_time         = row["date"]
                entry_volume       = row["volume"]
                entry_date_str     = row["date_str"]
                time_exit_deadline = entry_time + hold_delta
                position           = "long"
                pending_entry      = False
                trade_highs        = [row["high"]]
                trade_lows         = [row["low"]]

        elif position == "long":
            trade_highs.append(row["high"])
            trade_lows.append(row["low"])

        # ── Detectar SL / TP / tiempo / EOD ─────────────────────────────
        if position == "long":
            hit_tp   = row["high"] >= tp_price
            hit_sl   = row["low"]  <= sl_price
            hit_time = row["date"] >= time_exit_deadline
            is_last  = i == last_valid_idx

            if (hit_tp or hit_sl or hit_time) and not is_last:
                pending_exit = True
            elif is_last:
                trades.append(_record(row["close"] * (1 - slippage), row["date"]))
                position           = None
                time_exit_deadline = None
                trade_highs        = []
                trade_lows         = []

        # ── Detectar señal ───────────────────────────────────────────────
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
            signal_day_low    = day_low.iloc[i]
            signal_target_rr  = _dynamic_target_rr(float(dist_pct.iloc[i]))

    return enforce_schema(pd.DataFrame(trades))


# ======== small caps backtesting helpers ========
# Moved to strategies/iterative/backtest_helpers.py
from strategies.iterative.backtest_helpers import (
    plot_ticker,
    run_backtest,
    run_walkforward_backtest,
    run_up_to_date_backtest,
    run_iterative_incremental_backtest,
    _default_strategies,
)

if __name__ == "__main__":
    # python -m scripts.split_dataset_by_date --timeframe 5m --input-file backtest_dataset/pending_candles_5m.parquet
    _t0 = tm.time()
    run_walkforward_backtest()
    run_up_to_date_backtest()
    print(f"run_backtest full  completado en {tm.time() - _t0:.2f}s")
    pass


