"""
Corre una estrategia de índices sobre datos locales y grafica los trades.
El gráfico muestra todos los datos entre la fecha de entry y exit del trade.

Fuente de datos: backtest_dataset/INDICES/{ticker}/{tf}/{ticker_lower}_full_dataset.parquet
Si el indicador requerido no existe en el parquet, se calcula automáticamente desde OHLC.

Uso (desde backtester_api/):
    python -m scripts.test_strategy_indices --ticker QQQ --date 2026-06-09 --tf 1h --strategy sma_crossover_trail_iterative
    python -m scripts.test_strategy_indices --ticker QQQ --date 2026-06-09 --tf 1h --strategy sma_crossover_trail_iterative --sma-window 50 --factor 1.5
    python -m scripts.test_strategy_indices --ticker TQQQ --date 2026-05-28 --tf 1h --strategy sma_crossover_trail_iterative --factor 2.0
"""
import argparse
import importlib
import os
import sys
import zoneinfo
from datetime import date as _date, timedelta
from pathlib import Path

sys.path.insert(0, os.path.abspath("."))

import pandas as pd
import plotly.graph_objects as go

from app.utils.charts import (
    plot_candles_df, trades_to_markers, _write_fullscreen_html, CHARTS_DIR,
)

ET = zoneinfo.ZoneInfo("America/New_York")

# ── strategy registry ─────────────────────────────────────────────────────────
# name → module_path  (all are indices strategies)
STRATEGY_MAP = {
    "sma_crossover_trail_iterative": "strategies.iterative.indices.sma_crossover_trail_iterative",
    "orb_first_candle_iterative":    "strategies.iterative.indices.orb_first_candle",
    "orb_avg_range_iterative":       "strategies.iterative.indices.orb_avg_range",
}


# ── strategy loading ──────────────────────────────────────────────────────────

def _load_strategy(name: str):
    """Returns (strategy_fn, precompute_fn | None)."""
    module_path = STRATEGY_MAP.get(name)
    if module_path is None:
        raise ValueError(f"Estrategia '{name}' no encontrada. Opciones: {list(STRATEGY_MAP)}")
    module = importlib.import_module(module_path)
    fn = getattr(module, name, None)
    if fn is None:
        raise ValueError(f"Función '{name}' no encontrada en {module_path}")
    return fn, getattr(module, "_precompute", None)


# ── data loading ──────────────────────────────────────────────────────────────

def _indices_parquet(ticker: str, tf: str) -> Path:
    return Path(f"backtest_dataset/INDICES/{ticker}/{tf}/{ticker.lower()}_full_dataset.parquet")


def _load_full(ticker: str, tf: str) -> pd.DataFrame:
    path = _indices_parquet(ticker, tf)
    if not path.exists():
        raise FileNotFoundError(f"Parquet no encontrado: {path}")
    return pd.read_parquet(path).sort_values("date").reset_index(drop=True)


def _slice(full_df: pd.DataFrame, from_date: str, to_date: str) -> pd.DataFrame:
    mask = (full_df["date_str"] >= from_date) & (full_df["date_str"] <= to_date)
    return full_df[mask].copy().reset_index(drop=True)


# ── chart helpers ─────────────────────────────────────────────────────────────

def _to_et_dt(ts) -> pd.Timestamp:
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


def _add_sl_lines(fig, trades: pd.DataFrame, chart_end: str | None = None):
    """Dibuja línea horizontal de SL (entry → exit). Trades abiertos se extienden hasta chart_end."""
    sl_colors = ["#ff8a65", "#ffb74d", "#fff176", "#80cbc4"]
    tp_colors = ["#69f0ae", "#40c4ff", "#ea80fc", "#ff6d00"]
    for i, row in trades.iterrows():
        if pd.notna(row.get("stop_loss_price")):
            x1 = row["exit_time"] if pd.notna(row.get("exit_time")) else chart_end
            fig.add_shape(type="line",
                x0=row["entry_time"], x1=x1,
                y0=row["stop_loss_price"], y1=row["stop_loss_price"],
                line=dict(color=sl_colors[i % len(sl_colors)], width=1, dash="dot"),
                row=1, col=1,
            )
        if pd.notna(row.get("take_profit_price")):
            x1 = row["exit_time"] if pd.notna(row.get("exit_time")) else chart_end
            fig.add_shape(type="line",
                x0=row["entry_time"], x1=x1,
                y0=row["take_profit_price"], y1=row["take_profit_price"],
                line=dict(color=tp_colors[i % len(tp_colors)], width=1, dash="dot"),
                row=1, col=1,
            )


def _build_chart_df(candles: pd.DataFrame) -> pd.DataFrame:
    """Añade columna `time` (unix-s UTC) requerida por plot_candles_df."""
    out = candles.copy()
    if "time" not in out.columns:
        out["time"] = (
            pd.to_datetime(out["date"])
            .dt.tz_localize(ET)
            .astype("int64") // 10**9
        )
    return out


def _make_figure(candles: pd.DataFrame, ticker: str, tf: str,
                 date_label: str, out_path: str) -> go.Figure:
    chart_df = _build_chart_df(candles)

    # Añade todos los indicadores SMA/VWAP disponibles
    indicators = {}
    for col, label in [
        ("sma_9",   "SMA 9"),  ("sma_20",  "SMA 20"),  ("sma_50",  "SMA 50"),
        ("sma_100", "SMA 100"), ("sma_200", "SMA 200"), ("vwap",    "VWAP"),
    ]:
        if col in chart_df.columns and chart_df[col].notna().any():
            indicators[label] = chart_df[col].reset_index(drop=True)

    prev_close = None
    if "previous_day_close" in candles.columns:
        first_valid = candles["previous_day_close"].dropna()
        if not first_valid.empty:
            prev_close = float(first_valid.iloc[0])

    return plot_candles_df(
        chart_df,
        title=f"{ticker}  ({tf})  {date_label}",
        prev_close=prev_close,
        indicators=indicators or None,
        output=out_path,
    )


# ── main ──────────────────────────────────────────────────────────────────────

def run(ticker: str, date: str, tf: str, strategy_name: str, strategy_kwargs: dict):
    tf_minutes = int(tf[:-1]) * (60 if tf.endswith("h") else 1)

    # 1. Cargar dataset completo + precompute
    strategy_fn, precompute_fn = _load_strategy(strategy_name)

    full_df = _load_full(ticker, tf)
    if precompute_fn is not None:
        full_df = precompute_fn(full_df, **strategy_kwargs)

    # 2. Correr estrategia desde la fecha solicitada hasta el fin del dataset
    # Se pasan todos los datos desde `date` para que el trade pueda durar varios días
    day_candles = _slice(full_df, date, full_df["date_str"].max())
    if day_candles.empty:
        print(f"Sin datos para {ticker} en {date} (tf={tf}).")
        return

    trades = strategy_fn(day_candles, timeframe_minutes=tf_minutes, **strategy_kwargs)

    print(f"\n{'─'*60}")
    print(f"Estrategia : {strategy_name}  {strategy_kwargs if strategy_kwargs else ''}")
    print(f"Ticker     : {ticker}   Fecha: {date}   TF: {tf}")
    print(f"Trades     : {len(trades)}")
    if not trades.empty:
        cols = ["type", "entry_price", "exit_price", "stop_loss_price",
                "take_profit_price", "pnl", "Return", "entry_time", "exit_time"]
        df_print = trades[cols].copy()
        if "is_open" in trades.columns:
            df_print.insert(0, "estado", trades["is_open"].map({True: "(OPEN)", False: "closed"}))
        print(df_print.to_string(index=True))
    print(f"{'─'*60}\n")

    # 3. Determinar rango del gráfico: min(entry_date) → max(exit_date)
    # Los trades abiertos tienen exit_time=NaT; se usa el fin del dataset como fallback.
    if not trades.empty:
        chart_from = pd.to_datetime(trades["entry_time"]).dt.date.min().isoformat()
        closed_exits = pd.to_datetime(trades["exit_time"]).dropna()
        if not closed_exits.empty:
            chart_to = closed_exits.dt.date.max().isoformat()
        else:
            chart_to = full_df["date_str"].max()
    else:
        chart_from = chart_to = date

    date_label   = f"{chart_from} → {chart_to}" if chart_from != chart_to else chart_from
    chart_candles = _slice(full_df, chart_from, chart_to)

    # 4. Graficar
    out_path = str(CHARTS_DIR / f"{ticker}_{tf}_{date}_{strategy_name}.html")
    fig = _make_figure(chart_candles, ticker, tf, date_label, out_path)

    if not trades.empty:
        entries, exits, short_entries, short_exits = trades_to_markers(trades)
        _add_markers(fig, entries,       "Entry",       "triangle-up",   "#00e676", "bottom center")
        _add_markers(fig, exits,         "Exit",        "triangle-down", "#ff1744", "top center")
        _add_markers(fig, short_entries, "Short Entry", "triangle-down", "#ff1744", "top center")
        _add_markers(fig, short_exits,   "Short Exit",  "triangle-up",   "#00e676", "bottom center")
        _add_sl_lines(fig, trades, chart_end=chart_to)

    import webbrowser
    _write_fullscreen_html(fig, out_path)
    print(f"Chart guardado → {out_path}")
    webbrowser.open(Path(out_path).resolve().as_uri())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Testea una estrategia de índices con datos locales."
    )
    parser.add_argument("--ticker",        type=str,   required=True,
                        help="Ticker del índice, e.g. QQQ, TQQQ")
    parser.add_argument("--date",          type=str,   required=True,
                        help="Fecha de señal YYYY-MM-DD")
    parser.add_argument("--tf",            type=str,   default="1h",
                        help="Timeframe: 1h, 5m, 10m (default: 1h)")
    parser.add_argument("--strategy",      type=str,   default="sma_crossover_trail_iterative",
                        help=f"Estrategia. Opciones: {list(STRATEGY_MAP)}")
    # Parámetros estrategia genérica
    parser.add_argument("--sma-window",    dest="sma_window",    type=int,   default=None,
                        help="Período SMA (default: usa el de la función)")
    parser.add_argument("--atr-period",    dest="atr_period",    type=int,   default=None,
                        help="Período ATR (default: usa el de la función)")
    parser.add_argument("--factor",        type=float, default=None,
                        help="Multiplicador ATR para trailing SL")
    parser.add_argument("--max-hold-days", dest="max_hold_days", type=int,   default=None,
                        help="Días máximos en trade")
    args = parser.parse_args()

    strategy_kwargs = {}
    if args.sma_window    is not None: strategy_kwargs["sma_window"]    = args.sma_window
    if args.atr_period    is not None: strategy_kwargs["atr_period"]    = args.atr_period
    if args.factor        is not None: strategy_kwargs["factor"]        = args.factor
    if args.max_hold_days is not None: strategy_kwargs["max_hold_days"] = args.max_hold_days

    run(args.ticker, args.date, args.tf, args.strategy, strategy_kwargs)
