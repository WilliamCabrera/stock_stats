"""
Corre una estrategia iterativa sobre datos locales (o Massive como fallback) y
grafica los trades con entry/exit markers y líneas de SL/TP.

Fuentes de datos (en orden):
  1. backtest_dataset/full/<tf>/dates/<date>.parquet  (local, sin API key)
  2. Massive.com  (fallback si el ticker no está en el parquet local)

Uso (desde backtester_api/):
    python -m scripts.test_strategy_local --ticker CRVO --date 2026-06-16
    python -m scripts.test_strategy_local --ticker CRVO --date 2026-06-16 --tf 15m
    python -m scripts.test_strategy_local --ticker CRVO --date 2026-06-16 --strategy backside_short_lower_low_fix_stop_iterative
"""
import argparse
import os
import sys
import zoneinfo
from datetime import date as _date, timedelta

sys.path.insert(0, os.path.abspath("."))

import pandas as pd
import plotly.graph_objects as go

from strategies.iterative.backtest_helpers import plot_ticker
from app.utils.charts import (
    plot_candles_df, trades_to_markers, _write_fullscreen_html, CHARTS_DIR,
)
from app.utils.indicators import compute_vwap, compute_sma

STRATEGY_MAP = {
    "sma9_momentum_long_iterative": None,
    "backside_short_lower_low_fix_stop_iterative": None,
    "gap_crap_iterative": None,
    "short_push_exhaustion_iterative": None,
    "push_rejection_iterative": None,
}

ET = zoneinfo.ZoneInfo("America/New_York")

# Days of warmup to fetch before the target date (for SMA9 computation)
_SMA_WARMUP_DAYS = 10


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_strategy(name: str):
    from strategies.iterative import small_caps
    fn = getattr(small_caps, name, None)
    if fn is None:
        raise ValueError(f"Estrategia '{name}' no encontrada en small_caps.py")
    return fn


def _dates_parquet(tf: str, date: str) -> str:
    return f"backtest_dataset/full/{tf}/dates/{date.replace('-', '_')}.parquet"


def _to_et_dt(ts: str) -> pd.Timestamp:
    return pd.Timestamp(ts).tz_localize(ET)


def _add_markers(fig, markers, name, symbol, color, textposition):
    if not markers:
        return
    fig.add_trace(
        go.Scatter(
            x=[_to_et_dt(m["time"]) for m in markers],
            y=[m["price"] for m in markers],
            mode="markers+text",
            marker=dict(symbol=symbol, size=14, color=color),
            text=[m.get("label", name) for m in markers],
            textposition=textposition,
            name=name,
            hovertemplate="%{text}<br>%{y:.2f}<extra></extra>",
        ),
        row=1, col=1,
    )


def _add_sl_tp_lines(fig, trades: pd.DataFrame):
    sl_colors = ["#ff8a65", "#ffb74d", "#fff176", "#80cbc4"]
    tp_colors = ["#69f0ae", "#40c4ff", "#ea80fc", "#ff6d00"]
    for i, row in trades.iterrows():
        fig.add_shape(type="line",
            x0=row["entry_time"], x1=row["exit_time"],
            y0=row["stop_loss_price"], y1=row["stop_loss_price"],
            line=dict(color=sl_colors[i % len(sl_colors)], width=1, dash="dot"),
            row=1, col=1,
        )
        fig.add_shape(type="line",
            x0=row["entry_time"], x1=row["exit_time"],
            y0=row["take_profit_price"], y1=row["take_profit_price"],
            line=dict(color=tp_colors[i % len(tp_colors)], width=1, dash="dot"),
            row=1, col=1,
        )


# ── data loading ─────────────────────────────────────────────────────────────

def _load_local(ticker: str, date: str, tf: str) -> pd.DataFrame | None:
    """Returns candles DataFrame if ticker found locally, else None."""
    parquet = _dates_parquet(tf, date)
    if not os.path.exists(parquet):
        return None
    df = pd.read_parquet(parquet)
    candles = df[df["ticker"] == ticker].copy().reset_index(drop=True)
    return candles if not candles.empty else None


def _load_from_massive(ticker: str, date: str, tf: str) -> pd.DataFrame:
    """
    Fetch candles from Massive with a warmup window, compute SMA9/VWAP/
    previous_day_close, then return only the bars for `date` in the same
    column layout expected by the strategy.
    """
    from app.utils.massive import fetch_candles

    warmup_from = (_date.fromisoformat(date) - timedelta(days=_SMA_WARMUP_DAYS)).isoformat()
    print(f"  Massive: fetching {ticker} {warmup_from} → {date} ({tf})")

    raw = fetch_candles(ticker, warmup_from, date, timeframe=tf)
    if not raw:
        raise ValueError(f"Massive no devolvió datos para {ticker} en {date}")

    df = pd.DataFrame(raw)

    # Convert unix-s UTC → naive ET datetime (same as local parquet)
    df["date"] = (
        pd.to_datetime(df["time"], unit="s", utc=True)
        .dt.tz_convert(ET)
        .dt.tz_localize(None)
    )
    df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")
    df["ticker"]   = ticker

    # Indicators across the full warmup window (no lookahead)
    df["sma_9"] = compute_sma(df, window=9)
    df["vwap"]  = compute_vwap(df)

    # previous_day_close: close of the last bar of each prior trading day
    # Group by date, take the last close, shift by one calendar position
    last_close_by_day = (
        df.groupby("date_str")["close"].last().rename("previous_day_close")
    )
    prev_close_map = last_close_by_day.shift(1).to_dict()
    df["previous_day_close"] = df["date_str"].map(prev_close_map)

    # RVOL not computable without a longer history → NaN (only stored in trade record)
    df["RVOL_daily"] = float("nan")

    # Filter to requested date only
    candles = df[df["date_str"] == date].copy().reset_index(drop=True)
    if candles.empty:
        raise ValueError(f"Massive no devolvió barras para {ticker} en {date}")
    return candles


# ── chart via Massive data ────────────────────────────────────────────────────

def _plot_from_massive(ticker: str, date: str, tf: str, candles: pd.DataFrame, out_path: str) -> go.Figure:
    """
    Build a plotly Figure from Massive candles (uses plot_candles_df since
    the ticker may not exist in the local tickers/ parquet).
    """
    # plot_candles_df expects `time` (unix-s UTC) — derive it from the ET datetime
    chart_df = candles.copy()
    chart_df["time"] = (
        pd.to_datetime(chart_df["date"])
        .dt.tz_localize(ET)
        .astype("int64") // 10 ** 9
    )
    indicators = {}
    if "sma_9" in chart_df.columns:
        indicators["SMA 9"] = chart_df["sma_9"].reset_index(drop=True)
    if "vwap" in chart_df.columns:
        indicators["VWAP"]  = chart_df["vwap"].reset_index(drop=True)

    prev_close = candles["previous_day_close"].iloc[0] if "previous_day_close" in candles.columns else None

    fig = plot_candles_df(
        chart_df,
        title=f"{ticker}  ({tf})  {date}  [Massive]",
        prev_close=prev_close,
        indicators=indicators if indicators else None,
        output=out_path,   # save now; markers/lines added after, re-saved by caller
    )
    return fig


# ── main ─────────────────────────────────────────────────────────────────────

def run(ticker: str, date: str, tf: str, strategy_name: str):
    # 1. Try local first
    candles = _load_local(ticker, date, tf)
    source = "local"
    if candles is None:
        print(f"  {ticker} no encontrado en datos locales — usando Massive")
        candles = _load_from_massive(ticker, date, tf)
        source = "massive"

    # 2. Run strategy
    strategy_fn = _load_strategy(strategy_name)
    trades = strategy_fn(candles, timeframe_minutes=int(tf[:-1]))

    print(f"\n{'─'*60}")
    print(f"Estrategia : {strategy_name}")
    print(f"Ticker     : {ticker}   Fecha: {date}   TF: {tf}   Fuente: {source}")
    print(f"Trades     : {len(trades)}")
    if not trades.empty:
        cols = ["type", "entry_price", "exit_price", "stop_loss_price",
                "take_profit_price", "pnl", "Return", "entry_time", "exit_time"]
        print(trades[cols].to_string(index=True))
    print(f"{'─'*60}\n")

    # 3. Chart
    out_path = str(CHARTS_DIR / f"{ticker}_{tf}_{date}_{strategy_name}.html")

    if source == "local":
        fig = plot_ticker(
            ticker, timeframe=tf, date=date,
            show_sma9=True, show_sma200=False,
            show_donchian=False, show_vwap=True,
            output=out_path,
        )
    else:
        fig = _plot_from_massive(ticker, date, tf, candles, out_path)

    if not trades.empty:
        entries, exits, short_entries, short_exits = trades_to_markers(trades)
        _add_markers(fig, entries,       "Entry",       "triangle-up",   "#00e676", "bottom center")
        _add_markers(fig, exits,         "Exit",        "triangle-down", "#ff1744", "top center")
        _add_markers(fig, short_entries, "Short Entry", "triangle-down", "#ff1744", "top center")
        _add_markers(fig, short_exits,   "Short Exit",  "triangle-up",   "#00e676", "bottom center")
        _add_sl_tp_lines(fig, trades)

    import webbrowser
    from pathlib import Path
    _write_fullscreen_html(fig, out_path)
    print(f"Chart guardado → {out_path}")
    webbrowser.open(Path(out_path).resolve().as_uri())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Testea una estrategia iterativa con datos locales (o Massive como fallback)."
    )
    parser.add_argument("--ticker",   type=str, required=True, help="Símbolo, e.g. CRVO")
    parser.add_argument("--date",     type=str, required=True, help="Fecha YYYY-MM-DD")
    parser.add_argument("--tf",       type=str, default="5m",  help="Timeframe: 5m o 15m (default: 5m)")
    parser.add_argument("--strategy", type=str, default="sma9_momentum_long_iterative",
                        help=f"Función estrategia. Opciones: {list(STRATEGY_MAP)}")
    args = parser.parse_args()
    run(args.ticker, args.date, args.tf, args.strategy)
