import os
import sys
sys.path.insert(0, os.path.abspath("."))

import json
import logging
import functools
import time as tm
from pathlib import Path

import pandas as pd
import webbrowser
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.utils.charts import CHARTS_DIR, _write_fullscreen_html

DATASET_ROOT   = Path(os.path.abspath(".")) / "backtest_dataset" / "full"
DATASET_ROOT_1 = Path(__file__).resolve().parent

_LOG_DIR = Path(os.path.abspath(".")) / "logs" / "iterative"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setLevel(logging.INFO)
_console_handler.setFormatter(logging.Formatter("%(message)s"))

_file_handler = logging.FileHandler(_LOG_DIR / "backtest_helpers.log", encoding="utf-8")
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

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
    )

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

    if show_sma9 and "sma_9" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["date"], y=df["sma_9"],
                mode="lines", name="SMA 9",
                line=dict(color="#ffeb3b", width=1),
                hovertemplate="SMA9: %{y:.2f}<extra></extra>",
            ),
            row=1, col=1,
        )

    if show_sma200 and "sma_200" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["date"], y=df["sma_200"],
                mode="lines", name="SMA 200",
                line=dict(color="#ff9800", width=1.5),
                hovertemplate="SMA200: %{y:.2f}<extra></extra>",
            ),
            row=1, col=1,
        )

    if show_donchian and all(c in df.columns for c in ["donchian_upper", "donchian_lower", "donchian_basis"]):
        fig.add_trace(
            go.Scatter(
                x=df["date"], y=df["donchian_upper"],
                mode="lines", name="Donchian Upper",
                line=dict(color="rgba(100,181,246,0.6)", width=1),
                legendgroup="donchian",
                hovertemplate="Upper: %{y:.2f}<extra></extra>",
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["date"], y=df["donchian_lower"],
                mode="lines", name="Donchian Lower",
                line=dict(color="rgba(100,181,246,0.6)", width=1),
                fill="tonexty", fillcolor="rgba(100,181,246,0.08)",
                legendgroup="donchian",
                hovertemplate="Lower: %{y:.2f}<extra></extra>",
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["date"], y=df["donchian_basis"],
                mode="lines", name="Donchian Basis",
                line=dict(color="rgba(100,181,246,0.9)", width=1, dash="dot"),
                legendgroup="donchian",
                hovertemplate="Basis: %{y:.2f}<extra></extra>",
            ),
            row=1, col=1,
        )

    if show_vwap and "vwap" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["date"], y=df["vwap"],
                mode="lines", name="VWAP",
                line=dict(color="#ce93d8", width=1.5),
                hovertemplate="VWAP: %{y:.2f}<extra></extra>",
            ),
            row=1, col=1,
        )

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
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            itemclick="toggle", itemdoubleclick="toggleothers",
        ),
        margin=dict(l=50, r=50, t=60, b=30),
    )
    fig.update_xaxes(
        rangebreaks=[
            dict(bounds=["sat", "mon"]),
            dict(bounds=[20, 4], pattern="hour"),
        ],
        showspikes=True, spikemode="across", spikesnap="cursor",
        spikecolor="#555", spikethickness=1, spikedash="dot",
    )
    fig.update_yaxes(
        title_text="Precio",
        showspikes=True, spikemode="across", spikesnap="cursor",
        spikecolor="#555", spikethickness=1, spikedash="dot",
        row=1, col=1,
    )
    fig.update_yaxes(title_text="Volumen", row=2, col=1)

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
    """
    if strategy_func is None:
        return pd.DataFrame()

    _func = strategy_func.func if isinstance(strategy_func, functools.partial) else strategy_func
    strategy_name = _func.__name__

    if out_put_name is None:
        out_put_name = f"{strategy_name}_{gap_pct}_{stop_pct}_{tp_pct}"

    _out_dir = Path(out_dir) if out_dir is not None else DATASET_ROOT_1 / "UP-TO-DATE" / timeframe / strategy_name
    _out_dir.mkdir(parents=True, exist_ok=True)
    out_path        = _out_dir / f"{out_put_name}.parquet"
    checkpoint_path = _out_dir / f"{out_put_name}_checkpoint.json"

    logger.info("run_backtest  strategy=%s  tf=%s  params=%s  out=%s", strategy_name, timeframe, out_put_name, _out_dir)

    resume_from: str | None = None
    if checkpoint_path.exists():
        try:
            ck = json.loads(checkpoint_path.read_text())
            resume_from = ck.get("last_completed_date")
            logger.info("[RESUME] Continuando desde %s", resume_from)
        except Exception:
            pass

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

    if resume_from:
        date_files = [p for p in date_files if _file_date(p) > resume_from]

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
    logger.info("Procesando %d fechas  [%s → %s]",
                total_dates,
                _file_date(date_files[0]) if date_files else "?",
                _file_date(date_files[-1]) if date_files else "?")

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

        checkpoint_path.write_text(json.dumps({"last_completed_date": day_str}))

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
            slippage     = p["slippage"]
            gap_pct      = p["gap_pct"]
            stop_pct     = p["stop_pct"]
            tp_pct       = p["tp_pct"]
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

    # Clear pending candles that were just processed
    base_path = Path(os.path.abspath(".")) / "backtest_dataset"
    for timeframe in ["5m", "15m"]:
        processed_dates = pending_dates_cache.get(timeframe, [])
        if not processed_dates:
            continue
        pending_path = base_path / f"pending_candles_{timeframe}.parquet"
        if not pending_path.exists():
            continue
        remaining = pd.read_parquet(pending_path)
        remaining = remaining[~remaining["date_str"].isin(processed_dates)]
        if remaining.empty:
            pending_path.unlink()
            logger.info("[%s] pending_candles eliminado (todas las fechas procesadas).", timeframe)
        else:
            remaining.to_parquet(pending_path, index=False, compression="zstd")
            logger.info(
                "[%s] pending_candles: %d fechas procesadas eliminadas, %d restantes.",
                timeframe, len(processed_dates), len(remaining["date_str"].unique()),
            )

    logger.info("run_iterative_incremental_backtest completado en %.2fs", tm.time() - _t0)


def _default_strategies() -> list:
    from strategies.iterative.strategies_registry import STRATEGIES
    return STRATEGIES


if __name__ == "__main__":
    #run_walkforward_backtest()
    #run_up_to_date_backtest()
    pass