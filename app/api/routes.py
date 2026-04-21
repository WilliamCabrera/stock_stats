from fastapi import APIRouter, HTTPException, Query
from typing import Literal
from pathlib import Path
from celery.result import AsyncResult

from app.schemas import (
    BacktestRequest,
    JobSubmitted,
    JobStatus,
    IndicatorRequest,
)
from app.tasks.backtest import run_backtest, compute_indicators
from app.utils.massive import fetch_candles_async, TimeFrame
from app.config import get_settings

router = APIRouter()


# ── Market data ──────────────────────────────────────────────────────────────

@router.get("/candles/{ticker}", tags=["market data"])
async def candles(
    ticker: str,
    from_date: str = Query(..., alias="from", example="2024-01-02"),
    to_date: str = Query(..., alias="to", example="2024-01-10"),
    timeframe: TimeFrame = Query(default="5m"),
    adjusted: bool = Query(default=False),
    session_start: str | None = Query(
        default=None,
        example="14:30",
        description="time 'HH:MM' — keep bars at or after this time each day. "
                    " Ignored for 1d.",
    ),
    session_end: str | None = Query(
        default=None,
        example="21:00",
        description="'HH:MM' — keep bars at or before this time each day. "
                    " Ignored for 1d.",
    ),
):
    """
    Fetch OHLCV candles from Massive.com for a given ticker and date range.

    Returns a list of {time, open, high, low, close, volume} objects
    where `time` is a UTC Unix timestamp in seconds.

    Dates are interpreted in the server's local timezone (set via the `TZ`
    environment variable or `/etc/localtime`) and sent to Massive as UTC
    milliseconds. Accepts "YYYY-MM-DD" or "YYYY-MM-DDTHH:MM:SS".

    Use `session_start` / `session_end` (UTC 'HH:MM') for additional per-day
    filtering on multi-day ranges.
    """
    try:
        data = await fetch_candles_async(
            ticker.upper(), from_date, to_date, timeframe, adjusted,
            session_start=session_start, session_end=session_end,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Massive API error: {exc}") from exc
    return {"ticker": ticker.upper(), "timeframe": timeframe, "candles": data, "count": len(data)}


# ── Strategies ───────────────────────────────────────────────────────────────

@router.get("/strategies", tags=["strategies"])
async def list_strategies():
    """
    Discover available strategies by scanning the backtest_dataset directory.

    Scans:
    - backtest_dataset/full/<timeframe>/trades/<strategy>/
    - backtest_dataset/walkforward/<timeframe>/fold_<n>/trades/<strategy>/

    Returns the union of all strategy names plus availability metadata.
    """
    settings = get_settings()
    base = Path(settings.dataset_path)

    # { strategy_name: { "full": [timeframes], "walkforward": { timeframe: [folds] } } }
    result: dict[str, dict] = {}

    def _add_full(strategy: str, timeframe: str):
        entry = result.setdefault(strategy, {"full": [], "walkforward": {}})
        if timeframe not in entry["full"]:
            entry["full"].append(timeframe)

    def _add_walkforward(strategy: str, timeframe: str, fold: int):
        entry = result.setdefault(strategy, {"full": [], "walkforward": {}})
        folds = entry["walkforward"].setdefault(timeframe, [])
        if fold not in folds:
            folds.append(fold)

    # { strategy_name: set of variant strings }
    variants: dict[str, set] = {}

    def _read_variants(strategy: str, parquet_path: Path):
        if not parquet_path.exists():
            return
        import pandas as pd
        col = pd.read_parquet(parquet_path, columns=["strategy"])["strategy"]
        variants.setdefault(strategy, set()).update(col.dropna().unique().tolist())

    # Scan full/<timeframe>/trades/
    full_root = base / "full"
    if full_root.is_dir():
        for tf_dir in sorted(full_root.iterdir()):
            trades_dir = tf_dir / "trades"
            if trades_dir.is_dir():
                for strategy_dir in sorted(trades_dir.iterdir()):
                    if strategy_dir.is_dir():
                        _add_full(strategy_dir.name, tf_dir.name)
                        _read_variants(
                            strategy_dir.name,
                            strategy_dir / f"{strategy_dir.name}_full_{tf_dir.name}_trades.parquet",
                        )

    # Scan walkforward/<timeframe>/fold_<n>/trades/
    wf_root = base / "walkforward"
    if wf_root.is_dir():
        for tf_dir in sorted(wf_root.iterdir()):
            if not tf_dir.is_dir():
                continue
            for fold_dir in sorted(tf_dir.iterdir()):
                if not fold_dir.is_dir() or not fold_dir.name.startswith("fold_"):
                    continue
                try:
                    fold_n = int(fold_dir.name.split("_", 1)[1])
                except ValueError:
                    continue
                trades_dir = fold_dir / "trades"
                if trades_dir.is_dir():
                    for strategy_dir in sorted(trades_dir.iterdir()):
                        if strategy_dir.is_dir():
                            _add_walkforward(strategy_dir.name, tf_dir.name, fold_n)

    strategies = [
        {"name": name, **availability, "variants": sorted(variants.get(name, []))}
        for name, availability in sorted(result.items())
    ]
    return {"strategies": strategies, "count": len(strategies)}


# ── Trades ───────────────────────────────────────────────────────────────────

import pandas as pd

# Two-level cache:
#   _df_cache   : raw DataFrame (datetimes intact) — used for analysis
#   _trades_cache: serialized list[dict]            — used for paginated /trades
_df_cache: dict[tuple[str, str], pd.DataFrame] = {}
_trades_cache: dict[tuple[str, str], list[dict]] = {}

_TRADE_COLUMNS = [
    "ticker", "type", "entry_price", "exit_price", "stop_loss_price",
    "pnl", "Return", "rvol_daily", "previous_day_close", "volume",
    "entry_time", "exit_time", "strategy", "is_profit", "gap_prct",
]


def _parquet_path(strategy: str, timeframe: str) -> Path:
    settings = get_settings()
    return (
        Path(settings.dataset_path)
        / "full" / timeframe / "trades" / strategy
        / f"{strategy}_full_{timeframe}_trades.parquet"
    )


def _load_df(strategy: str, timeframe: str) -> pd.DataFrame:
    """Return the raw (datetime-typed) DataFrame, cached after first load."""
    key = (strategy, timeframe)
    if key not in _df_cache:
        path = _parquet_path(strategy, timeframe)
        if not path.exists():
            raise FileNotFoundError(str(path))
        df = pd.read_parquet(path)
        df["is_profit"] = df["pnl"] > 0
        if "gap_prct" not in df.columns:
            df["gap_prct"] = None
        _df_cache[key] = df
    return _df_cache[key]


def _load_trades(strategy: str, timeframe: str) -> list[dict]:
    """Return serialized records (ISO datetime strings), cached after first call."""
    key = (strategy, timeframe)
    if key not in _trades_cache:
        df = _load_df(strategy, timeframe).copy()
        for col in ("entry_time", "exit_time"):
            if col in df.columns:
                df[col] = df[col].dt.strftime("%Y-%m-%dT%H:%M:%S")
        _trades_cache[key] = df[_TRADE_COLUMNS].to_dict(orient="records")
    return _trades_cache[key]


@router.get("/trades", tags=["trades"])
async def list_trades(
    strategy: str = Query(..., description="Strategy folder name, e.g. backside_short_lower_low"),
    timeframe: str = Query(..., description="Timeframe folder name, e.g. 15m or 5m"),
    variant: str | None = Query(default=None, description="Filter by strategy column value; 'all' returns every variant"),
    page: int = Query(default=1, ge=1, description="Page number (1-based)"),
    page_size: int = Query(default=500, ge=1, le=2000, description="Trades per page"),
):
    """
    Return paginated trades for a given strategy and timeframe from backtest_dataset/full.

    Pass `variant=all` (or omit it) to return every parameter combination.
    Pass a specific variant string to filter by the `strategy` column.

    The DataFrame is cached in memory after the first request — subsequent pages
    are served from RAM with no disk I/O.

    Response includes pagination metadata: total, page, page_size, pages.
    """
    try:
        all_trades = _load_trades(strategy, timeframe)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"No trades file found: {exc}. Check strategy name and timeframe.",
        ) from exc

    # Filter by variant when a specific value is requested (not "all" / empty)
    if variant and variant.lower() not in ("all", ""):
        all_trades = [t for t in all_trades if t.get("strategy") == variant]

    total = len(all_trades)
    pages = (total + page_size - 1) // page_size
    offset = (page - 1) * page_size
    slice_ = all_trades[offset : offset + page_size]

    return {
        "trades": slice_,
        "total": total,
        "page": page,
        "page_size": page_size,
        "pages": pages,
    }


# ── Summary / Analysis ───────────────────────────────────────────────────────

# ── Shared filter dependency & helpers ───────────────────────────────────────

from dataclasses import dataclass
from fastapi import Depends
import math
import numpy as np


@dataclass
class AnalysisFilters:
    strategy: str
    timeframe: str
    variant: str | None
    ticker: str | None
    price_min: float | None
    price_max: float | None
    volume_min: float | None
    volume_max: float | None
    time_from: str | None
    time_to: str | None
    initial_capital: float
    risk_pct: float


def _analysis_filters(
    strategy: str = Query(default="backside_short_lower_low", description="Strategy folder name"),
    timeframe: str = Query(default="5m", description="Timeframe, e.g. 5m or 15m"),
    variant: str | None = Query(default=None, description="Filter by strategy column value"),
    ticker: str | None = Query(default=None, description="Filter by ticker (case-insensitive)"),
    price_min: float | None = Query(default=None, description="Min entry_price"),
    price_max: float | None = Query(default=None, description="Max entry_price"),
    volume_min: float | None = Query(default=None, description="Min volume"),
    volume_max: float | None = Query(default=None, description="Max volume"),
    time_from: str | None = Query(default=None, description="entry_time >= HH:MM"),
    time_to: str | None = Query(default=None, description="entry_time <= HH:MM"),
    initial_capital: float = Query(default=1_000.0, description="Starting capital"),
    risk_pct: float = Query(default=0.01, ge=0.0, le=1.0, description="Risk per trade as fraction of capital (0.01 = 1%)"),
) -> AnalysisFilters:
    return AnalysisFilters(
        strategy=strategy, timeframe=timeframe, variant=variant, ticker=ticker,
        price_min=price_min, price_max=price_max,
        volume_min=volume_min, volume_max=volume_max,
        time_from=time_from, time_to=time_to,
        initial_capital=initial_capital, risk_pct=risk_pct,
    )


def _apply_filters(f: AnalysisFilters) -> pd.DataFrame:
    """Load and filter the trades DataFrame. Raises HTTPException on bad input."""
    try:
        df = _load_df(f.strategy, f.timeframe).copy()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"No trades file found: {exc}") from exc

    if f.variant is not None and f.variant.lower() not in ("all", ""):
        df = df[df["strategy"] == f.variant]
    if f.ticker is not None:
        df = df[df["ticker"].str.upper() == f.ticker.upper()]
    if f.price_min is not None:
        df = df[df["entry_price"] >= f.price_min]
    if f.price_max is not None:
        df = df[df["entry_price"] <= f.price_max]
    if f.volume_min is not None:
        df = df[df["volume"] >= f.volume_min]
    if f.volume_max is not None:
        df = df[df["volume"] <= f.volume_max]
    if f.time_from is not None or f.time_to is not None:
        t_min = pd.to_datetime(df["entry_time"]).dt.hour * 60 + pd.to_datetime(df["entry_time"]).dt.minute
        mask = pd.Series(True, index=df.index)
        if f.time_from is not None:
            h, m = map(int, f.time_from.split(":"))
            mask &= t_min >= h * 60 + m
        if f.time_to is not None:
            h, m = map(int, f.time_to.split(":"))
            mask &= t_min <= h * 60 + m
        df = df[mask]

    if df.empty:
        raise HTTPException(status_code=422, detail="No trades match the given filters.")
    if len(df) < 2:
        raise HTTPException(status_code=422, detail="At least 2 trades are required.")
    return df


def _build_equity(df: pd.DataFrame, f: AnalysisFilters):
    from app.utils.trade_metrics import equity_from_r, equity_returns
    equity_df = equity_from_r(df, initial_capital=f.initial_capital, risk_pct=f.risk_pct)
    equity = equity_df["equity"]
    returns = equity_returns(equity)
    return equity, returns


def _clean(v):
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


# ── Analysis endpoints ────────────────────────────────────────────────────────

@router.get("/summary", tags=["analysis"])
async def get_summary(f: AnalysisFilters = Depends(_analysis_filters)):
    """
    Filter trades and return a full performance summary (R-based metrics + equity stats).
    """
    from app.utils.trade_metrics import summary_report

    df = _apply_filters(f)
    try:
        report = summary_report(df, initial_capital=f.initial_capital, risk_pct=f.risk_pct)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"summary_report error: {exc}") from exc

    winners = int(df["is_profit"].sum())
    losers  = len(df) - winners

    return {
        "trades_count": len(df),
        "winners": winners,
        "losers": losers,
        "summary": {k: _clean(v) for k, v in report.items()},
    }


@router.get("/equity", tags=["analysis"])
async def get_equity(f: AnalysisFilters = Depends(_analysis_filters)):
    """
    Return the equity curve and CAGR reference line, aggregated to one point per day.
    Each daily value is the last equity recorded that day (end-of-day).
    """
    df = _apply_filters(f)
    equity, _ = _build_equity(df, f)

    # Aggregate: last equity value per calendar day
    daily = equity.resample("D").last().dropna()

    years = (daily.index[-1] - daily.index[0]).days / 365.25
    cagr = (daily.iloc[-1] / daily.iloc[0]) ** (1 / years) - 1
    cagr_line = daily.iloc[0] * (1 + cagr) ** np.linspace(0, years, len(daily))

    points = [
        {
            "time": t.date().isoformat(),
            "equity": round(float(eq), 4),
            "cagr_equity": round(float(cg), 4),
        }
        for t, eq, cg in zip(daily.index, daily, cagr_line)
    ]
    return {
        "trades_count": len(df),
        "days_count": len(points),
        "initial_capital": f.initial_capital,
        "final_equity": round(float(daily.iloc[-1]), 4),
        "cagr": round(cagr, 6),
        "curve": points,
    }


@router.get("/drawdown", tags=["analysis"])
async def get_drawdown(f: AnalysisFilters = Depends(_analysis_filters)):
    """
    Return the drawdown series aggregated to one point per day.
    Drawdown is computed on the daily equity series (last trade of each day).
    """
    from app.utils.trade_metrics import drawdown_series

    df = _apply_filters(f)
    equity, _ = _build_equity(df, f)

    # Aggregate equity to daily, then compute drawdown on that
    daily = equity.resample("D").last().dropna()
    dd = drawdown_series(daily)

    points = [
        {"time": t.date().isoformat(), "drawdown": round(float(v), 6)}
        for t, v in zip(dd.index, dd)
    ]
    return {
        "trades_count": len(df),
        "days_count": len(points),
        "max_drawdown": round(float(dd.min()), 6),
        "series": points,
    }


@router.get("/returns-histogram", tags=["analysis"])
async def get_returns_histogram(
    bins: int = Query(default=50, ge=5, le=200, description="Number of histogram bins"),
    f: AnalysisFilters = Depends(_analysis_filters),
):
    """
    Return the equity-returns distribution pre-binned for charting, plus VaR 5%.
    """
    df = _apply_filters(f)
    _, returns = _build_equity(df, f)

    var_5 = float(returns.quantile(0.05))
    counts, edges = np.histogram(returns.dropna(), bins=bins)

    histogram = [
        {"x": round(float(edges[i]), 6), "count": int(counts[i])}
        for i in range(len(counts))
    ]
    return {
        "trades_count": len(df),
        "var_5pct": round(var_5, 6),
        "bins": len(histogram),
        "histogram": histogram,
    }


# ── Backtest ─────────────────────────────────────────────────────────────────

@router.post("/backtest", response_model=JobSubmitted, status_code=202, tags=["backtest"])
async def submit_backtest(req: BacktestRequest, ohlcv: list[dict]):
    """
    Submit a backtest job. Returns immediately with a job_id.
    Poll GET /jobs/{job_id} for status and results.

    The caller must supply the raw OHLCV data (list of
    {time, open, high, low, close, volume} objects with `time` as UTC seconds).
    """
    task = run_backtest.apply_async(
        args=[ohlcv, req.ticker, req.strategy, req.params],
        queue="backtest",
    )
    return JobSubmitted(job_id=task.id)


# ── Job status ────────────────────────────────────────────────────────────────

@router.get("/jobs/{job_id}", response_model=JobStatus, tags=["jobs"])
async def get_job(job_id: str):
    """Poll for job status and result."""
    result: AsyncResult = AsyncResult(job_id)

    state = result.state
    if state == "PENDING":
        return JobStatus(job_id=job_id, status="pending")
    if state in ("STARTED", "PROGRESS"):
        return JobStatus(job_id=job_id, status="running")
    if state == "SUCCESS":
        return JobStatus(job_id=job_id, status="success", result=result.result)
    if state == "FAILURE":
        return JobStatus(
            job_id=job_id,
            status="failure",
            error=str(result.result),
        )
    # REVOKED or unknown
    return JobStatus(job_id=job_id, status="failure", error=f"Unexpected state: {state}")


@router.delete("/jobs/{job_id}", status_code=204, tags=["jobs"])
async def cancel_job(job_id: str):
    """Revoke a pending or running job."""
    AsyncResult(job_id).revoke(terminate=True)


# ── Indicators ───────────────────────────────────────────────────────────────

@router.post("/indicators", tags=["analysis"])
async def indicators(req: IndicatorRequest):
    """
    Compute a TA indicator synchronously (lightweight — no job queue needed).
    For heavy batch computations use the /backtest endpoint instead.
    """
    task = compute_indicators.apply_async(
        args=[req.prices, req.indicator, req.params],
        queue="backtest",
    )
    # Wait up to 30 s for a lightweight indicator computation
    try:
        return task.get(timeout=30)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
