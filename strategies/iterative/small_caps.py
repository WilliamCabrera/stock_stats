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

def plot_ticker(
    ticker: str,
    timeframe: str = "5m",
    date: str | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
    show_sma9: bool = True,
    show_sma200: bool = True,
    show_donchian: bool = True,
    show_vwap: bool = True,
    output: str | None = None,
    height: int = 800,
) -> go.Figure:
    """
    Grafica velas + indicadores desde el dataset local.

    Ruta: backtest_dataset/full/<timeframe>/tickers/<ticker>.parquet

    Args:
        ticker:        Símbolo, e.g. "AAOI"
        timeframe:     "5m" o "15m"
        date:          Día exacto "YYYY-MM-DD" (atajo para from_date == to_date)
        from_date:     Inicio de rango "YYYY-MM-DD" (inclusive)
        to_date:       Fin de rango   "YYYY-MM-DD" (inclusive)
        show_sma9:     Mostrar SMA 9
        show_sma200:   Mostrar SMA 200
        show_donchian: Mostrar canal Donchian (upper / basis / lower)
        show_vwap:     Mostrar VWAP
        output:        Ruta HTML de salida. None = auto-guardar y abrir browser.
        height:        Altura del gráfico en píxeles.

    Ejemplos:
        plot_ticker("AAOI", date="2024-11-08")
        plot_ticker("AAOI", from_date="2024-11-01", to_date="2024-11-30")
        plot_ticker("AAOI", show_sma200=False, show_donchian=False)
    """
    parquet_path = DATASET_ROOT / timeframe / "tickers" / f"{ticker}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {parquet_path}")

    df = pd.read_parquet(parquet_path)

    # date es un atajo para from_date == to_date
    if date is not None:
        from_date = from_date or date
        to_date   = to_date   or date

    if from_date is not None or to_date is not None:
        dates = df["date"].dt.strftime("%Y-%m-%d")
        mask = pd.Series(True, index=df.index)
        if from_date:
            mask &= dates >= from_date
        if to_date:
            mask &= dates <= to_date
        df = df[mask].copy()
        if df.empty:
            label = f"{from_date} → {to_date}" if from_date != to_date else from_date
            raise ValueError(f"No hay datos para {ticker} en {label}")

    # ── Layout ─────────────────────────────────────────────────────────────
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
    )

    # ── Velas ───────────────────────────────────────────────────────────────
    fig.add_trace(
        go.Candlestick(
            x=df["date"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name=ticker,
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
            showlegend=True,
        ),
        row=1, col=1,
    )

    # ── Volumen ─────────────────────────────────────────────────────────────
    bar_colors = [
        "#26a69a" if c >= o else "#ef5350"
        for o, c in zip(df["open"], df["close"])
    ]
    fig.add_trace(
        go.Bar(
            x=df["date"],
            y=df["volume"],
            name="Volumen",
            marker_color=bar_colors,
            showlegend=False,
        ),
        row=2, col=1,
    )

    # ── SMA 9 ───────────────────────────────────────────────────────────────
    if show_sma9 and "sma_9" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["sma_9"],
                mode="lines",
                name="SMA 9",
                line=dict(color="#ffeb3b", width=1),
                hovertemplate="SMA9: %{y:.2f}<extra></extra>",
            ),
            row=1, col=1,
        )

    # ── SMA 200 ─────────────────────────────────────────────────────────────
    if show_sma200 and "sma_200" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["sma_200"],
                mode="lines",
                name="SMA 200",
                line=dict(color="#ff9800", width=1.5),
                hovertemplate="SMA200: %{y:.2f}<extra></extra>",
            ),
            row=1, col=1,
        )

    # ── Donchian ────────────────────────────────────────────────────────────
    if show_donchian and all(c in df.columns for c in ["donchian_upper", "donchian_lower", "donchian_basis"]):
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["donchian_upper"],
                mode="lines",
                name="Donchian Upper",
                line=dict(color="rgba(100,181,246,0.6)", width=1),
                legendgroup="donchian",
                hovertemplate="Upper: %{y:.2f}<extra></extra>",
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["donchian_lower"],
                mode="lines",
                name="Donchian Lower",
                line=dict(color="rgba(100,181,246,0.6)", width=1),
                fill="tonexty",
                fillcolor="rgba(100,181,246,0.08)",
                legendgroup="donchian",
                hovertemplate="Lower: %{y:.2f}<extra></extra>",
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["donchian_basis"],
                mode="lines",
                name="Donchian Basis",
                line=dict(color="rgba(100,181,246,0.9)", width=1, dash="dot"),
                legendgroup="donchian",
                hovertemplate="Basis: %{y:.2f}<extra></extra>",
            ),
            row=1, col=1,
        )

    # ── VWAP ────────────────────────────────────────────────────────────────
    if show_vwap and "vwap" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["vwap"],
                mode="lines",
                name="VWAP",
                line=dict(color="#ce93d8", width=1.5),
                hovertemplate="VWAP: %{y:.2f}<extra></extra>",
            ),
            row=1, col=1,
        )

    # ── Styling ─────────────────────────────────────────────────────────────
    if from_date and to_date and from_date != to_date:
        date_label = f"{from_date} → {to_date}"
    elif from_date or to_date:
        date_label = from_date or to_date
    else:
        date_label = ""
    title = f"{ticker}  ({timeframe})" + (f"  {date_label}" if date_label else "")
    fig.update_layout(
        title=title,
        height=height,
        autosize=True,
        template="plotly_dark",
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            itemclick="toggle",
            itemdoubleclick="toggleothers",
        ),
        margin=dict(l=50, r=50, t=60, b=30),
    )
    fig.update_xaxes(
        rangebreaks=[
            dict(bounds=["sat", "mon"]),
            dict(bounds=[20, 4], pattern="hour"),
        ],
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikecolor="#555",
        spikethickness=1,
        spikedash="dot",
    )
    fig.update_yaxes(
        title_text="Precio",
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikecolor="#555",
        spikethickness=1,
        spikedash="dot",
        row=1, col=1,
    )
    fig.update_yaxes(title_text="Volumen", row=2, col=1)

    # ── Output ───────────────────────────────────────────────────────────────
    range_slug = date_label.replace(" → ", "_to_") if date_label else "all"
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in f"{ticker}_{timeframe}_{range_slug}")
    if output:
        _write_fullscreen_html(fig, output)
        print(f"Chart guardado → {output}")
    else:
        path = CHARTS_DIR / f"{safe}.html"
        _write_fullscreen_html(fig, str(path))
        print(f"Chart guardado → {path}")
        webbrowser.open(path.as_uri())

    return fig

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


def run_backtest(
    timeframe: str,
    strategy_func=None,
    from_date: str | None = None,
    to_date: str | None = None,
    slippage: float = 0.001,
    gap_pct: float = 0.40,
    stop_pct: float = 0.50,
    tp_pct: float = 0.20,
    dates_dir: str | Path | None = None,
    out_dir: str | Path | None = None,
    out_put_name=None
) -> pd.DataFrame:
    """
    Corre un backtest iterativo leyendo cada archivo de fechas en dates_dir.

    Cada archivo .parquet cubre un día y contiene todos los tickers activos ese día.
    Por cada ticker del día se llama a strategy_func(candles) → pd.DataFrame de trades.

    Args:
        timeframe:     "5m" o "15m"
        strategy_func: Callable(candles: pd.DataFrame) → pd.DataFrame con columnas
                       [entry_time, exit_time, entry_price, exit_price, pnl, ...]
                       Usa functools.partial para pasar parámetros extra.
        from_date:     Filtrar desde "YYYY-MM-DD" (inclusive). None = desde el inicio.
        to_date:       Filtrar hasta "YYYY-MM-DD" (inclusive). None = hasta el final.
        dates_dir:     Directorio con los archivos .parquet por fecha.
                       Default: backtest_dataset/full/<timeframe>/dates/
        out_dir:       Directorio de salida para el parquet de trades y el checkpoint.
                       Default: strategies/iterative/UP-TO-DATE/<timeframe>/<strategy>/

    Returns:
        DataFrame con todos los trades.

    Ejemplo:
        from functools import partial
        func = partial(backside_short_lower_low_fix_stop_iterative, gap_pct=0.4)

        # Full dataset (default)
        trades = run_backtest("5m", func)

        # Walk-forward OOS fold 1
        trades = run_backtest(
            "5m", func,
            dates_dir="backtest_dataset/walkforward/5m/fold_1/dates_OOS",
            out_dir="strategies/iterative/WF/5m/fold_1/backside_short_lower_low_fix_stop_iterative",
        )
    """
    if strategy_func is None:
        return pd.DataFrame()
    
    # ── Nombre de la estrategia (soporta functools.partial) ──────────────
    _func = strategy_func.func if isinstance(strategy_func, functools.partial) else strategy_func
    strategy_name = _func.__name__

    if out_put_name is None:
        out_put_name = f"{strategy_name}_{gap_pct}_{stop_pct}_{tp_pct}"

    # ── Rutas de salida y checkpoint ─────────────────────────────────────
    _out_dir = Path(out_dir) if out_dir is not None else DATASET_ROOT_1 / "UP-TO-DATE" / timeframe / strategy_name
    _out_dir.mkdir(parents=True, exist_ok=True)
    out_path        = _out_dir / f"{out_put_name}.parquet"
    checkpoint_path = _out_dir / f"{out_put_name}_checkpoint.json"

    logger.info("run_backtest  strategy=%s  tf=%s  params=%s  out=%s", strategy_name, timeframe, out_put_name, _out_dir)

    # ── Cargar checkpoint (resume) ────────────────────────────────────────
    resume_from: str | None = None
    if checkpoint_path.exists():
        try:
            ck = json.loads(checkpoint_path.read_text())
            resume_from = ck.get("last_completed_date")
            logger.info("[RESUME] Continuando desde %s", resume_from)
        except Exception:
            pass

    # ── Listar y filtrar archivos de fechas ───────────────────────────────
    _dates_dir = Path(dates_dir) if dates_dir is not None else DATASET_ROOT / timeframe / "dates"
    if not _dates_dir.exists():
        raise FileNotFoundError(f"Directorio no encontrado: {_dates_dir}")

    date_files = sorted(_dates_dir.glob("*.parquet"))
    if not date_files:
        raise FileNotFoundError(f"No hay archivos parquet en {_dates_dir}")

    def _file_date(p: Path) -> str:
        return p.stem.replace("_", "-")

    if from_date or to_date:
        date_files = [
            p for p in date_files
            if (not from_date or _file_date(p) >= from_date)
            and (not to_date   or _file_date(p) <= to_date)
        ]

    # Saltar fechas ya completadas
    if resume_from:
        date_files = [p for p in date_files if _file_date(p) > resume_from]

    # ── Loop principal ────────────────────────────────────────────────────
    def _append_trades(new_df: pd.DataFrame) -> None:
        if out_path.exists():
            existing = pd.read_parquet(out_path)
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined = combined.drop_duplicates(
                subset=["ticker", "date_str", "entry_time"], keep="last"
            )
        else:
            combined = new_df
        cols_first = [c for c in ["ticker", "date_str"] if c in combined.columns]
        rest = [c for c in combined.columns if c not in cols_first]
        combined[cols_first + rest].to_parquet(out_path, index=False)

    tf_minutes = int(timeframe[:-1])
    total_dates = len(date_files)
    logger.info("Procesando %d fechas  [%s → %s]", total_dates, _file_date(date_files[0]) if date_files else "?", _file_date(date_files[-1]) if date_files else "?")

    total_trades_session = 0

    for idx, parquet_file in enumerate(date_files, 1):
        day_str = _file_date(parquet_file)

        try:
            day_df = pd.read_parquet(parquet_file)
        except Exception as e:
            logger.warning("[%d/%d] %s — error leyendo parquet: %s", idx, total_dates, day_str, e)
            continue

        n_tickers = day_df["ticker"].nunique()
        day_trades: list[pd.DataFrame] = []

        for ticker in day_df["ticker"].unique():
            candles = day_df[day_df["ticker"] == ticker].copy().reset_index(drop=True)
            if len(candles) < 2:
                continue
            try:
                trades = strategy_func(
                    candles,
                    gap_pct=gap_pct,
                    stop_pct=stop_pct,
                    tp_pct=tp_pct,
                    slippage=slippage,
                    timeframe_minutes=tf_minutes,
                )
            except Exception as e:
                logger.warning("%-6s  %s — strategy error: %s", ticker, day_str, e)
                continue
            if trades is not None and not trades.empty:
                trades["timeframe"] = timeframe
                day_trades.append(trades)

        if day_trades:
            day_result = pd.concat(day_trades, ignore_index=True)
            _append_trades(day_result)
            total_trades_session += len(day_result)
            logger.info("[%d/%d] %s  tickers=%d  trades=%d", idx, total_dates, day_str, n_tickers, len(day_result))
        else:
            logger.info("[%d/%d] %s  tickers=%d  trades=0", idx, total_dates, day_str, n_tickers)

        # ── Guardar checkpoint ────────────────────────────────────────────
        checkpoint_path.write_text(json.dumps({"last_completed_date": day_str}))

    # ── Backtest completado: borrar checkpoint ────────────────────────────
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info("Checkpoint eliminado — backtest completo.")

    if not out_path.exists():
        return pd.DataFrame()

    result = pd.read_parquet(out_path)
    logger.info("run_backtest completo  strategy=%s  tf=%s  session_trades=%d  total_acumulado=%d  → %s",
                strategy_name, timeframe, total_trades_session, len(result), out_path)
    return result

def run_walkforward_backtest(from_date: str | None = None, to_date: str | None = None, strategies: list | None = None):
    """
    Walk-forward backtest driven by STRATEGIES registry.

    For every entry in STRATEGIES, and for every params set in entry["params"],
    runs run_backtest across all folds (IS + OOS) for both 5m and 15m timeframes.
    """
    _t0 = tm.time()
    _strategies = strategies if strategies is not None else _default_strategies()

    FOLDS = [
        ("5m",  "fold_1", "IN-SAMPLE",     "dates_IS",  "tier_1"),
        ("5m",  "fold_2", "IN-SAMPLE",     "dates_IS",  "tier_2"),
        ("5m",  "fold_3", "IN-SAMPLE",     "dates_IS",  "tier_3"),
        ("5m",  "fold_1", "OUT-OF-SAMPLE", "dates_OOS", "tier_1"),
        ("5m",  "fold_2", "OUT-OF-SAMPLE", "dates_OOS", "tier_2"),
        ("5m",  "fold_3", "OUT-OF-SAMPLE", "dates_OOS", "tier_3"),
        ("15m", "fold_1", "IN-SAMPLE",     "dates_IS",  "tier_1"),
        ("15m", "fold_2", "IN-SAMPLE",     "dates_IS",  "tier_2"),
        ("15m", "fold_3", "IN-SAMPLE",     "dates_IS",  "tier_3"),
        ("15m", "fold_1", "OUT-OF-SAMPLE", "dates_OOS", "tier_1"),
        ("15m", "fold_2", "OUT-OF-SAMPLE", "dates_OOS", "tier_2"),
        ("15m", "fold_3", "OUT-OF-SAMPLE", "dates_OOS", "tier_3"),
    ]

    for entry in _strategies:
        strategy_func = entry["strategy_func"]
        strategy_name = entry["strategy_name"]

        for p in entry["params"]:
            slippage     = p["slippage"]
            gap_pct      = p["gap_pct"]
            stop_pct     = p["stop_pct"]
            tp_pct       = p["tp_pct"]
            out_put_name = p["out_put_name"]

            logger.info("WF  strategy=%s  params=%s", strategy_name, out_put_name)

            for tf, fold, split, dates_subdir, tier in FOLDS:
                dates_dir = f"backtest_dataset/walkforward/{tf}/{fold}/{dates_subdir}"
                out_dir   = f"strategies/iterative/WF/{split}/{tf}/{tier}/{out_put_name}"
                run_backtest(
                    timeframe=tf,
                    strategy_func=strategy_func,
                    from_date=from_date,
                    to_date=to_date,
                    slippage=slippage,
                    gap_pct=gap_pct,
                    stop_pct=stop_pct,
                    tp_pct=tp_pct,
                    dates_dir=dates_dir,
                    out_dir=out_dir,
                    out_put_name=out_put_name,
                )

    logger.info("run_walkforward_backtest completado en %.2fs", tm.time() - _t0)

def run_up_to_date_backtest(from_date: str | None = None, to_date: str | None = None, strategies: list | None = None):
    """
    Up-to-date backtest driven by STRATEGIES registry.

    For every entry in STRATEGIES, and for every params set in entry["params"],
    runs run_backtest on the full dataset for both 5m and 15m timeframes.
    """
    _t0 = tm.time()
    _strategies = strategies if strategies is not None else _default_strategies()

    for entry in _strategies:
        strategy_func = entry["strategy_func"]
        strategy_name = entry["strategy_name"]

        for p in entry["params"]:
            slippage     = p["slippage"]
            gap_pct      = p["gap_pct"]
            stop_pct     = p["stop_pct"]
            tp_pct       = p["tp_pct"]
            out_put_name = p["out_put_name"]

            logger.info("UP-TO-DATE  strategy=%s  params=%s", strategy_name, out_put_name)

            for tf in ["5m", "15m"]:
                run_backtest(
                    timeframe=tf,
                    strategy_func=strategy_func,
                    from_date=from_date,
                    to_date=to_date,
                    slippage=slippage,
                    gap_pct=gap_pct,
                    stop_pct=stop_pct,
                    tp_pct=tp_pct,
                    dates_dir=f"backtest_dataset/full/{tf}/dates",
                    out_dir=f"strategies/iterative/UP-TO-DATE/{tf}/{out_put_name}",
                    out_put_name=out_put_name,
                )

    logger.info("run_up_to_date_backtest completado en %.2fs", tm.time() - _t0)

def run_iterative_incremental_backtest(strategies: list | None = None):
    """
    Incremental iterative backtest driven by STRATEGIES registry.

    For every entry in STRATEGIES, and for every params set in entry["params"],
    runs run_backtest on the pending dates for both 5m and 15m timeframes.

    Reads pending dates from:
        backtest_dataset/pending_candles_{5m,15m}.parquet
    Reads candle data from:
        backtest_dataset/full/{timeframe}/dates/{YYYY_MM_DD}.parquet
    Appends trades to (same location as run_up_to_date_backtest):
        strategies/iterative/UP-TO-DATE/{timeframe}/{out_put_name}/{out_put_name}.parquet
    """
    _t0 = tm.time()
    _strategies = strategies if strategies is not None else _default_strategies()

    pending_dates_cache: dict[str, list[str]] = {}

    for entry in _strategies:
        strategy_func = entry["strategy_func"]
        strategy_name = entry["strategy_name"]

        for p in entry["params"]:
            slippage    = p["slippage"]
            gap_pct     = p["gap_pct"]
            stop_pct    = p["stop_pct"]
            tp_pct      = p["tp_pct"]
            out_put_name = p["out_put_name"]

            logger.info("INCREMENTAL  strategy=%s  params=%s", strategy_name, out_put_name)

            for timeframe in ["5m", "15m"]:
                if timeframe not in pending_dates_cache:
                    base_path = Path(os.path.abspath(".")) / "backtest_dataset"
                    pending_path = base_path / f"pending_candles_{timeframe}.parquet"
                    if not pending_path.exists():
                        logger.info("[%s] No pending candles — skipping.", timeframe)
                        pending_dates_cache[timeframe] = []
                        continue
                    pending_df = pd.read_parquet(pending_path, columns=["date_str"])
                    if pending_df.empty:
                        logger.info("[%s] Pending candles vacíos — skipping.", timeframe)
                        pending_dates_cache[timeframe] = []
                        continue
                    pending_dates_cache[timeframe] = sorted(pending_df["date_str"].unique())

                pending_dates = pending_dates_cache.get(timeframe, [])
                if not pending_dates:
                    continue

                min_date = pending_dates[0]
                max_date = pending_dates[-1]
                logger.info("[%s] Pending dates: %s → %s (%d date(s))", timeframe, min_date, max_date, len(pending_dates))

                run_backtest(
                    timeframe=timeframe,
                    strategy_func=strategy_func,
                    from_date=min_date,
                    to_date=max_date,
                    slippage=slippage,
                    gap_pct=gap_pct,
                    stop_pct=stop_pct,
                    tp_pct=tp_pct,
                    dates_dir=f"backtest_dataset/full/{timeframe}/dates",
                    out_dir=f"strategies/iterative/UP-TO-DATE/{timeframe}/{out_put_name}",
                    out_put_name=out_put_name,
                )

    logger.info("run_iterative_incremental_backtest completado en %.2fs", tm.time() - _t0)
    return


def _default_strategies() -> list:
    from strategies.iterative.strategies_registry import STRATEGIES
    return STRATEGIES

if __name__ == "__main__":
    # python -m scripts.split_dataset_by_date --timeframe 5m --input-file backtest_dataset/pending_candles_5m.parquet
    #_t0 = tm.time()
    run_walkforward_backtest()
    run_up_to_date_backtest()
    #print(f"run_backtest full  completado en {tm.time() - _t0:.2f}s")
    pass


