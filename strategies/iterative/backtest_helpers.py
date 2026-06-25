import os
import sys
sys.path.insert(0, os.path.abspath("."))

import json
import logging
import functools
import time as tm
from pathlib import Path

import httpx
import pandas as pd
import webbrowser
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.config import get_settings
from app.utils.charts import CHARTS_DIR, _write_fullscreen_html

DATASET_ROOT   = Path(os.path.abspath(".")) / "backtest_dataset" / "full"
DATASET_ROOT_WF = Path(os.path.abspath(".")) / "backtest_dataset" / "walkforward"
DATASET_ROOT_1 = Path(__file__).resolve().parent


def _wf_cutoff_date(timeframe: str) -> str | None:
    """
    Returns the last OOS date across all walkforward folds for a given timeframe,
    as "YYYY-MM-DD". Returns None if no walkforward data exists.

    Used by run_up_to_date_backtest to avoid re-running dates already covered by
    the walkforward validation period.
    """
    wf_root = DATASET_ROOT_WF / timeframe
    if not wf_root.exists():
        return None
    last_date = None
    for fold_dir in wf_root.iterdir():
        oos_dir = fold_dir / "dates_OOS"
        if not oos_dir.exists():
            continue
        for f in oos_dir.glob("*.parquet"):
            d = f.stem.replace("_", "-")
            if last_date is None or d > last_date:
                last_date = d
    return last_date

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


_RETRY_STATUS = {429, 500, 502, 503, 504}


def _fetch_reference_sync(
    client: httpx.Client,
    base: str,
    api_key: str,
    ticker: str,
    date_str: str,
    retries: int = 3,
) -> dict | None:
    url = f"{base}/v3/reference/tickers/{ticker}?date={date_str}&apiKey={api_key}"
    for attempt in range(retries + 1):
        try:
            resp = client.get(url)
        except (httpx.TimeoutException, httpx.TransportError):
            if attempt == retries:
                return None
            tm.sleep(min(0.5 * 2 ** attempt, 10))
            continue
        if resp.status_code == 200:
            return resp.json().get("results")
        if resp.status_code in _RETRY_STATUS and attempt < retries:
            tm.sleep(min(0.5 * 2 ** attempt, 10))
            continue
        return None
    return None


def _enrich_trades_with_fundamentals(trades: pd.DataFrame) -> pd.DataFrame:
    """Attach market_cap and float to trades via the Massive reference endpoint."""
    if trades.empty or "ticker" not in trades.columns or "date_str" not in trades.columns:
        return trades

    settings = get_settings()
    if not settings.massive_api_key:
        logger.warning("MASSIVE_API_KEY no configurado — omitiendo enriquecimiento de fundamentals.")
        return trades

    base = settings.massive_base_url.rstrip("/")
    api_key = settings.massive_api_key

    pairs = trades[["ticker", "date_str"]].drop_duplicates()
    rows = []
    with httpx.Client(timeout=30) as client:
        for _, row in pairs.iterrows():
            res = _fetch_reference_sync(client, base, api_key, row["ticker"], row["date_str"])
            if res:
                rows.append({
                    "ticker":      row["ticker"],
                    "date_str":    row["date_str"],
                    "market_cap":  res.get("market_cap"),
                    "float":       res.get("weighted_shares_outstanding"),
                })

    if not rows:
        logger.warning("_enrich_trades_with_fundamentals: no se obtuvo ningún dato de fundamentals.")
        return trades

    lookup = pd.DataFrame(rows)
    trades = trades.drop(columns=["market_cap", "float"], errors="ignore")
    enriched = trades.merge(lookup, on=["ticker", "date_str"], how="left")
    matched = enriched["market_cap"].notna().sum()
    logger.info("Fundamentals enriquecidos: %d/%d trades con market_cap.", matched, len(enriched))
    return enriched


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
    out_put_name=None,
    **extra_kwargs,
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
    out_path        = _out_dir / f"{out_put_name}_{timeframe}.parquet"
    checkpoint_path = _out_dir / f"{out_put_name}_{timeframe}_checkpoint.json"

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

        expected_delta = pd.Timedelta(minutes=tf_minutes)
        for ticker in day_df["ticker"].unique():
            candles = day_df[day_df["ticker"] == ticker].copy().reset_index(drop=True)
            if len(candles) < 2:
                continue
            rth = candles[(candles["date"].dt.hour >= 9) & (candles["date"].dt.hour < 16)]
            if len(rth) > 1 and (rth["date"].diff().iloc[1:] != expected_delta).any():
                logger.debug("%-6s  %s — gaps in RTH candles, skip", ticker, day_str)
                continue
            try:
                trades = strategy_func(
                    candles,
                    gap_pct=gap_pct,
                    stop_pct=stop_pct,
                    tp_pct=tp_pct,
                    slippage=slippage,
                    timeframe_minutes=tf_minutes,
                    **extra_kwargs,
                )
            except Exception as e:
                logger.warning("%-6s  %s — strategy error: %s", ticker, day_str, e)
                continue
            if trades is not None and not trades.empty:
                trades["timeframe"] = timeframe
                day_trades.append(trades)

        if day_trades:
            day_result = pd.concat(day_trades, ignore_index=True)
            #day_result = _enrich_trades_with_fundamentals(day_result)
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
        dataset       = entry.get("dataset", "small_caps")

        if dataset == "indices":
            for p in entry["params"]:
                logger.info("WF  strategy=%s  dataset=indices  params=%s", strategy_name, p["out_put_name"])
                _run_indices_walkforward(entry, p, from_date=from_date, to_date=to_date)
            continue

        if dataset != "small_caps":
            logger.info("WF  strategy=%s — omitido (dataset=%s)", strategy_name, dataset)
            continue

        for p in entry["params"]:
            slippage     = p["slippage"]
            gap_pct      = p["gap_pct"]
            stop_pct     = p["stop_pct"]
            tp_pct       = p["tp_pct"]
            out_put_name = p["out_put_name"]
            extra_params = {k: v for k, v in p.items()
                            if k not in ("slippage", "gap_pct", "stop_pct", "tp_pct", "out_put_name")}

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
                    **extra_params,
                )

    logger.info("run_walkforward_backtest completado en %.2fs", tm.time() - _t0)


def run_up_to_date_backtest(
    from_date: str | None = None,
    to_date: str | None = None,
    strategies: list | None = None,
    skip_walkforward_dates: bool = True,
):
    """
    Up-to-date backtest driven by STRATEGIES registry.

    For every entry in STRATEGIES, and for every params set in entry["params"],
    runs run_backtest on the full dataset for both 5m and 15m timeframes.

    Args:
        from_date:              Start date "YYYY-MM-DD". If None and
                                skip_walkforward_dates=True, auto-detected as the
                                day after the last OOS date in the walkforward folds.
        to_date:                End date "YYYY-MM-DD". None = up to latest available.
        strategies:             Override the registry. None = use STRATEGIES.
        skip_walkforward_dates: If True (default), small_caps strategies will only
                                run on dates after the walkforward OOS period ends,
                                avoiding double-counting with run_walkforward_backtest.
                                Pass False to run on the full historical range.
    """
    _t0 = tm.time()
    _strategies = strategies if strategies is not None else _default_strategies()

    for entry in _strategies:
        strategy_name = entry["strategy_name"]
        dataset       = entry.get("dataset", "small_caps")

        for p in entry["params"]:
            out_put_name = p["out_put_name"]
            logger.info("UP-TO-DATE  strategy=%s  dataset=%s  params=%s", strategy_name, dataset, out_put_name)

            if dataset == "indices":
                _run_indices_uptodate(entry, p, from_date=from_date, to_date=to_date)
            else:
                strategy_func = entry["strategy_func"]
                for tf in ["5m", "15m"]:
                    # Auto-detect WF cutoff so UP-TO-DATE never overlaps with WF folds
                    effective_from = from_date
                    if effective_from is None and skip_walkforward_dates:
                        cutoff = _wf_cutoff_date(tf)
                        if cutoff is not None:
                            # cutoff is the last OOS date; start the day after
                            from datetime import date, timedelta
                            next_day = (date.fromisoformat(cutoff) + timedelta(days=1)).isoformat()
                            effective_from = next_day
                            logger.info(
                                "UP-TO-DATE  tf=%s  WF cutoff=%s → from_date=%s",
                                tf, cutoff, effective_from,
                            )

                    _extra = {k: v for k, v in p.items()
                              if k not in ("slippage", "gap_pct", "stop_pct", "tp_pct", "out_put_name")}
                    run_backtest(
                        timeframe=tf,
                        strategy_func=strategy_func,
                        from_date=effective_from,
                        to_date=to_date,
                        slippage=p["slippage"],
                        gap_pct=p["gap_pct"],
                        stop_pct=p["stop_pct"],
                        tp_pct=p["tp_pct"],
                        dates_dir=f"backtest_dataset/full/{tf}/dates",
                        out_dir=f"strategies/iterative/UP-TO-DATE/{tf}/{out_put_name}",
                        out_put_name=out_put_name,
                        **_extra,
                    )

    logger.info("run_up_to_date_backtest completado en %.2fs", tm.time() - _t0)


def run_iterative_incremental_backtest(strategies: list | None = None):
    """
    Incremental iterative backtest driven by STRATEGIES registry.

    Covers all datasets declared in the registry:

    small_caps
        Reads pending dates from backtest_dataset/pending_candles_{5m,15m}.parquet
        (written by update_full_dataset.py). Clears processed dates on completion.
        Output: strategies/iterative/UP-TO-DATE/{tf}/{out_put_name}/

    indices
        Derives new dates by comparing max(date_str) in the existing output parquet
        against the dates available in backtest_dataset/INDICES/{ticker}/{tf}/.
        If no output exists yet, bootstraps the full history on first run.
        Output: strategies/iterative/UP-TO-DATE/INDICES/{tf}/{out_put_name}/
    """
    _t0 = tm.time()
    _strategies = strategies if strategies is not None else _default_strategies()

    pending_dates_cache: dict[str, list[str]] = {}

    for entry in _strategies:
        strategy_func = entry["strategy_func"]
        strategy_name = entry["strategy_name"]
        dataset       = entry.get("dataset", "small_caps")

        if dataset == "indices":
            for p in entry["params"]:
                logger.info("INCREMENTAL  strategy=%s  dataset=indices  params=%s", strategy_name, p["out_put_name"])
                _run_indices_incremental(entry, p)
            continue

        if dataset != "small_caps":
            logger.info("INCREMENTAL  strategy=%s — omitido (dataset=%s)", strategy_name, dataset)
            continue

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

                _extra = {k: v for k, v in p.items()
                          if k not in ("slippage", "gap_pct", "stop_pct", "tp_pct", "out_put_name")}
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
                    **_extra,
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


def _run_indices_walkforward(
    entry: dict,
    p: dict,
    from_date: str | None = None,
    to_date: str | None = None,
) -> None:
    """
    Walk-forward backtest for index strategies.

    Reads per-ticker IS/OOS fold parquets from:
        {data_root}/{ticker}/walkforward/{tf}/fold_{N}/{in_sample,out_of_sample}.parquet

    Passes the full-history parquet as ticker_parquet_path so strategies that
    need historical context (EMA, ATR) have access to prior days outside the fold.

    Output:
        strategies/iterative/WF/INDICES/{IN-SAMPLE,OUT-OF-SAMPLE}/{tf}/{tier}/{out_put_name}/
    """
    strategy_func = entry["strategy_func"]
    strategy_name = entry["strategy_name"]
    data_root     = Path(os.path.abspath(".")) / entry["data_root"]
    tickers       = entry.get("tickers", [])
    timeframes    = entry.get("timeframes", ["5m"])
    out_put_name  = p["out_put_name"]

    FOLDS = [
        (1, "IN-SAMPLE",     "in_sample",     "tier_1"),
        (2, "IN-SAMPLE",     "in_sample",     "tier_2"),
        (3, "IN-SAMPLE",     "in_sample",     "tier_3"),
        (1, "OUT-OF-SAMPLE", "out_of_sample", "tier_1"),
        (2, "OUT-OF-SAMPLE", "out_of_sample", "tier_2"),
        (3, "OUT-OF-SAMPLE", "out_of_sample", "tier_3"),
    ]

    for tf in timeframes:
        tf_minutes = int(tf[:-1])

        for fold_n, split, file_stem, tier in FOLDS:
            out_dir = (
                Path(os.path.abspath("."))
                / "strategies" / "iterative" / "WF" / "INDICES"
                / split / tf / tier / out_put_name
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{out_put_name}_{tf}.parquet"

            for ticker in tickers:
                fold_parquet = (
                    data_root / ticker / "walkforward" / tf
                    / f"fold_{fold_n}" / f"{file_stem}.parquet"
                )
                if not fold_parquet.exists():
                    logger.warning("[INDICES WF] %s  fold_%d/%s  %s — parquet no encontrado: %s",
                                   ticker, fold_n, file_stem, tf, fold_parquet)
                    continue

                # Full-history parquet for historical context (EMA, ATR lookbacks, etc.)
                full_parquet = data_root / ticker / tf / f"{ticker.lower()}_full_dataset.parquet"
                ticker_parquet_path = full_parquet if full_parquet.exists() else fold_parquet

                fold_df = pd.read_parquet(fold_parquet)
                if "date_str" not in fold_df.columns:
                    logger.warning("[INDICES WF] %s fold_%d/%s — columna date_str no encontrada.", ticker, fold_n, file_stem)
                    continue

                if entry.get("multi_day"):
                    # Multi-day: precompute on full history for correct SMA/ATR lookback,
                    # then slice to this fold's date range and run at once.
                    # Trades reaching the fold boundary without SL hit → is_open=True.
                    if full_parquet.exists():
                        full_df = pd.read_parquet(full_parquet).sort_values("date").reset_index(drop=True)
                        precompute_fn = entry.get("precompute_fn")
                        if precompute_fn is not None:
                            full_df = precompute_fn(full_df, **{k: v for k, v in p.items() if k != "out_put_name"})
                        fold_min = fold_df["date_str"].min()
                        fold_max = fold_df["date_str"].max()
                        if from_date:
                            fold_min = max(fold_min, from_date)
                        if to_date:
                            fold_max = min(fold_max, to_date)
                        run_df = full_df[(full_df["date_str"] >= fold_min) & (full_df["date_str"] <= fold_max)]
                    else:
                        run_df = fold_df

                    if run_df.empty:
                        continue

                    extra_params = {k: v for k, v in p.items()
                                    if k not in ("out_put_name", "slippage", "gap_pct", "stop_pct", "tp_pct")}
                    logger.info("[INDICES WF][MULTI_DAY] %s  fold_%d/%s  %s  %s  %d barras",
                                strategy_name, fold_n, split, tf, ticker, len(run_df))
                    try:
                        trades = strategy_func(
                            run_df,
                            gap_pct=p["gap_pct"],
                            stop_pct=p["stop_pct"],
                            tp_pct=p["tp_pct"],
                            slippage=p["slippage"],
                            timeframe_minutes=tf_minutes,
                            **extra_params,
                        )
                    except Exception as e:
                        logger.warning("[INDICES WF][MULTI_DAY] %-6s  fold_%d/%s — error: %s", ticker, fold_n, split, e)
                        continue

                    if trades is not None and not trades.empty:
                        trades["timeframe"] = tf
                        if out_path.exists():
                            existing = pd.read_parquet(out_path)
                            existing = existing[existing["ticker"] != ticker]
                            combined = pd.concat([existing, trades], ignore_index=True)
                        else:
                            combined = trades
                        combined.to_parquet(out_path, index=False)
                        open_n = int(trades["is_open"].sum()) if "is_open" in trades.columns else 0
                        logger.info("[INDICES WF][MULTI_DAY] %s  fold_%d/%s  %s  %s  → %d trades (%d open)",
                                    strategy_name, fold_n, split, tf, ticker, len(trades), open_n)

                else:
                    if from_date:
                        fold_df = fold_df[fold_df["date_str"] >= from_date]
                    if to_date:
                        fold_df = fold_df[fold_df["date_str"] <= to_date]

                    dates = sorted(fold_df["date_str"].unique())
                    if not dates:
                        continue

                    logger.info("[INDICES WF] %s  fold_%d/%s  %s  %s  %d fechas",
                                strategy_name, fold_n, split, tf, ticker, len(dates))

                    day_trades: list[pd.DataFrame] = []
                    expected_delta = pd.Timedelta(minutes=tf_minutes)
                    for day_str in dates:
                        candles = fold_df[fold_df["date_str"] == day_str].copy().reset_index(drop=True)
                        if len(candles) < 2:
                            continue
                        rth = candles[(candles["date"].dt.hour >= 9) & (candles["date"].dt.hour < 16)]
                        if len(rth) > 1 and (rth["date"].diff().iloc[1:] != expected_delta).any():
                            logger.debug("%-6s  %s — gaps in RTH candles, skip", ticker, day_str)
                            continue
                        try:
                            trades = strategy_func(
                                candles,
                                gap_pct=p["gap_pct"],
                                stop_pct=p["stop_pct"],
                                tp_pct=p["tp_pct"],
                                slippage=p["slippage"],
                                timeframe_minutes=tf_minutes,
                                ticker_parquet_path=ticker_parquet_path,
                            )
                        except Exception as e:
                            logger.warning("[INDICES WF] %-6s  %s — strategy error: %s", ticker, day_str, e)
                            continue
                        if trades is not None and not trades.empty:
                            trades["timeframe"] = tf
                            day_trades.append(trades)

                    if day_trades:
                        new_df = pd.concat(day_trades, ignore_index=True)
                        if out_path.exists():
                            existing = pd.read_parquet(out_path)
                            combined = pd.concat([existing, new_df], ignore_index=True)
                            combined = combined.drop_duplicates(
                                subset=["ticker", "date_str", "entry_time"], keep="last"
                            )
                        else:
                            combined = new_df
                        combined.to_parquet(out_path, index=False)
                        logger.info("[INDICES WF] %s  fold_%d/%s  %s  %s  → %d trades escritos",
                                    strategy_name, fold_n, split, tf, ticker, len(new_df))


def _run_indices_incremental(entry: dict, p: dict) -> None:
    """
    Incremental backtest for index strategies.

    Determines which dates are new by comparing max(date_str) in the existing
    output parquet against the dates available in the ticker's full parquet.
    Delegates to _run_indices_uptodate with from_date set to the day after the
    last already-processed date.

    If no output parquet exists yet (first run), processes the full history so
    the output is bootstrapped and subsequent incremental runs work correctly.
    """
    strategy_name = entry["strategy_name"]
    data_root     = Path(os.path.abspath(".")) / entry["data_root"]
    timeframes    = entry.get("timeframes", ["5m"])
    out_put_name  = p["out_put_name"]

    for tf in timeframes:
        out_path = (
            Path(os.path.abspath("."))
            / "strategies" / "iterative" / "UP-TO-DATE" / "INDICES"
            / tf / out_put_name / f"{out_put_name}_{tf}.parquet"
        )

        from_date: str | None = None

        if out_path.exists():
            try:
                existing  = pd.read_parquet(out_path, columns=["date_str"])
                last_date = existing["date_str"].max()
                from_date = (pd.Timestamp(last_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                logger.info("[INDICES INCREMENTAL] %s %s — último date=%s  →  from=%s",
                            strategy_name, tf, last_date, from_date)
            except Exception as e:
                logger.warning("[INDICES INCREMENTAL] %s %s — error leyendo output: %s; corriendo full.", strategy_name, tf, e)
        else:
            logger.info("[INDICES INCREMENTAL] %s %s — sin output previo, bootstrapping full history.", strategy_name, tf)

        # Verify there is actually new data before launching
        tickers = entry.get("tickers", [])
        has_new = False
        for ticker in tickers:
            parquet_path = data_root / ticker / tf / f"{ticker.lower()}_full_dataset.parquet"
            if not parquet_path.exists():
                continue
            dates_available = pd.read_parquet(parquet_path, columns=["date_str"])["date_str"]
            if from_date is None or (dates_available > from_date).any():
                has_new = True
                break

        if not has_new:
            logger.info("[INDICES INCREMENTAL] %s %s — sin fechas nuevas, skipping.", strategy_name, tf)
            continue

        _run_indices_uptodate(entry, p, from_date=from_date)


def _run_indices_uptodate(
    entry: dict,
    p: dict,
    from_date: str | None = None,
    to_date: str | None = None,
) -> None:
    """
    Up-to-date backtest for index strategies.

    Reads per-ticker parquet files from:
        {data_root}/{ticker}/{tf}/{ticker_lower}_full_dataset.parquet

    Path template comes from entry["data_root"] — nothing is hardcoded here.
    """
    strategy_func = entry["strategy_func"]
    strategy_name = entry["strategy_name"]
    data_root     = Path(os.path.abspath(".")) / entry["data_root"]
    tickers       = entry.get("tickers", [])
    timeframes    = entry.get("timeframes", ["5m"])
    slippage      = p["slippage"]
    gap_pct       = p["gap_pct"]
    stop_pct      = p["stop_pct"]
    tp_pct        = p["tp_pct"]
    out_put_name  = p["out_put_name"]

    for tf in timeframes:
        tf_minutes = int(tf[:-1])
        out_dir    = Path(os.path.abspath(".")) / "strategies" / "iterative" / "UP-TO-DATE" / "INDICES" / tf / out_put_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path        = out_dir / f"{out_put_name}_{tf}.parquet"
        checkpoint_path = out_dir / f"{out_put_name}_{tf}_checkpoint.json"

        resume_from: str | None = None
        if checkpoint_path.exists():
            try:
                ck = json.loads(checkpoint_path.read_text())
                resume_from = ck.get("last_completed_date")
                logger.info("[INDICES][RESUME] %s %s — continuando desde %s", strategy_name, tf, resume_from)
            except Exception:
                pass

        for ticker in tickers:
            # Path resolved from registry data_root — no dataset logic inside strategy
            parquet_path = data_root / ticker / tf / f"{ticker.lower()}_full_dataset.parquet"
            if not parquet_path.exists():
                logger.warning("[INDICES] %s %s %s — parquet no encontrado: %s", strategy_name, tf, ticker, parquet_path)
                continue

            full_df = pd.read_parquet(parquet_path)
            if "date_str" not in full_df.columns:
                logger.warning("[INDICES] %s %s %s — columna date_str no encontrada.", strategy_name, tf, ticker)
                continue

            precompute_fn = entry.get("precompute_fn")
            if precompute_fn is not None:
                full_df = precompute_fn(full_df, **{k: v for k, v in p.items() if k != "out_put_name"})

            total_session = 0

            if entry.get("multi_day"):
                # Multi-day strategy: NEVER filter by from_date or resume_from.
                # Cutting history would orphan any trade that entered before the cutoff,
                # making it impossible to detect its open/close status.
                # Only to_date is respected (useful for bounded testing).
                # The output parquet is fully overwritten each run so there are no
                # stale "open" records from previous executions.
                run_df = full_df[full_df["date_str"] <= to_date] if to_date else full_df
                if run_df.empty:
                    continue
                extra_params = {k: v for k, v in p.items()
                                if k not in ("out_put_name", "slippage", "gap_pct", "stop_pct", "tp_pct")}
                logger.info("[INDICES][MULTI_DAY] %s  %s  %s  %d barras", strategy_name, tf, ticker, len(run_df))
                try:
                    trades = strategy_func(
                        run_df,
                        gap_pct=gap_pct,
                        stop_pct=stop_pct,
                        tp_pct=tp_pct,
                        slippage=slippage,
                        timeframe_minutes=tf_minutes,
                        **extra_params,
                    )
                except Exception as e:
                    logger.warning("[INDICES][MULTI_DAY] %-6s — strategy error: %s", ticker, e)
                    continue

                if trades is not None and not trades.empty:
                    trades["timeframe"] = tf
                    # Replace only this ticker's rows — preserve other tickers already saved
                    if out_path.exists():
                        existing = pd.read_parquet(out_path)
                        existing = existing[existing["ticker"] != ticker]
                        combined = pd.concat([existing, trades], ignore_index=True)
                    else:
                        combined = trades
                    combined.to_parquet(out_path, index=False)
                    total_session += len(trades)
                    open_n = int(trades["is_open"].sum()) if "is_open" in trades.columns else 0
                    logger.info("[INDICES][MULTI_DAY] %s  %s  %s  %d trades (%d open)",
                                strategy_name, tf, ticker, len(trades), open_n)

            else:
                # Day-by-day strategy: apply all date filters and iterate per date
                if from_date:
                    full_df = full_df[full_df["date_str"] >= from_date]
                if to_date:
                    full_df = full_df[full_df["date_str"] <= to_date]
                if resume_from:
                    full_df = full_df[full_df["date_str"] > resume_from]

                dates = sorted(full_df["date_str"].unique())
                if not dates:
                    continue

                logger.info("[INDICES] %s  %s  %s  %d fechas", strategy_name, tf, ticker, len(dates))
                expected_delta = pd.Timedelta(minutes=tf_minutes)
                for day_str in dates:
                    candles = full_df[full_df["date_str"] == day_str].copy().reset_index(drop=True)
                    if len(candles) < 2:
                        checkpoint_path.write_text(json.dumps({"last_completed_date": day_str}))
                        continue
                    rth = candles[(candles["date"].dt.hour >= 9) & (candles["date"].dt.hour < 16)]
                    if len(rth) > 1 and (rth["date"].diff().iloc[1:] != expected_delta).any():
                        logger.debug("%-6s  %s — gaps in RTH candles, skip", ticker, day_str)
                        checkpoint_path.write_text(json.dumps({"last_completed_date": day_str}))
                        continue
                    try:
                        trades = strategy_func(
                            candles,
                            gap_pct=gap_pct,
                            stop_pct=stop_pct,
                            tp_pct=tp_pct,
                            slippage=slippage,
                            timeframe_minutes=tf_minutes,
                            ticker_parquet_path=parquet_path,
                        )
                    except Exception as e:
                        logger.warning("[INDICES] %-6s  %s — strategy error: %s", ticker, day_str, e)
                        checkpoint_path.write_text(json.dumps({"last_completed_date": day_str}))
                        continue

                    if trades is not None and not trades.empty:
                        trades["timeframe"] = tf
                        if out_path.exists():
                            existing = pd.read_parquet(out_path)
                            combined = pd.concat([existing, trades], ignore_index=True)
                            combined = combined.drop_duplicates(
                                subset=["ticker", "date_str", "entry_time"], keep="last"
                            )
                        else:
                            combined = trades
                        combined.to_parquet(out_path, index=False)
                        total_session += len(trades)

                    checkpoint_path.write_text(json.dumps({"last_completed_date": day_str}))

            logger.info("[INDICES] %s  %s  %s  session_trades=%d", strategy_name, tf, ticker, total_session)

        if checkpoint_path.exists():
            checkpoint_path.unlink()
            logger.info("[INDICES] Checkpoint eliminado — %s %s completo.", strategy_name, tf)


def _default_strategies() -> list:
    from strategies.iterative.strategies_registry import STRATEGIES
    return STRATEGIES


if __name__ == "__main__":
    #run_walkforward_backtest()
    #run_up_to_date_backtest()
    pass