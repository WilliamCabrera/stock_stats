"""
Walk-forward analysis endpoints.

Reads from:
  backtest_dataset/walkforward/<timeframe>/fold_<fold>/trades/<strategy>/
    <strategy>_in_sample_trades.parquet
    <strategy>_out_of_sample_trades.parquet

All endpoints require `strategy`, `timeframe`, and `fold`.
`sample` controls which split to load: "in_sample" | "out_of_sample" | "both"
(default: "out_of_sample").
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from pathlib import Path
from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from app.config import get_settings
from app.api.routes import _build_equity, _clean, _TRADE_COLUMNS

router = APIRouter(prefix="/wf", tags=["walkforward"])

# ── Cache ────────────────────────────────────────────────────────────────────
# key: (strategy, timeframe, fold, sample)   sample ∈ {"in_sample", "out_of_sample"}
_wf_df_cache: dict[tuple, pd.DataFrame] = {}
_wf_trades_cache: dict[tuple, list[dict]] = {}


# ── Data loading ─────────────────────────────────────────────────────────────

def _wf_parquet_path(strategy: str, timeframe: str, fold: int, sample: str) -> Path:
    settings = get_settings()
    return (
        Path(settings.dataset_path)
        / "walkforward" / timeframe / f"fold_{fold}" / "trades" / strategy
        / f"{strategy}_{sample}_trades.parquet"
    )


def _load_wf_df_single(strategy: str, timeframe: str, fold: int, sample: str) -> pd.DataFrame:
    key = (strategy, timeframe, fold, sample)
    if key not in _wf_df_cache:
        path = _wf_parquet_path(strategy, timeframe, fold, sample)
        if not path.exists():
            raise FileNotFoundError(str(path))
        df = pd.read_parquet(path)
        df["is_profit"] = df["pnl"] > 0
        if "gap_prct" not in df.columns:
            df["gap_prct"] = None
        _wf_df_cache[key] = df
    return _wf_df_cache[key]


def _load_wf_df(strategy: str, timeframe: str, fold: int, sample: str) -> pd.DataFrame:
    if sample != "both":
        return _load_wf_df_single(strategy, timeframe, fold, sample)

    parts = []
    for s in ("in_sample", "out_of_sample"):
        try:
            part = _load_wf_df_single(strategy, timeframe, fold, s).copy()
            part["sample_type"] = s
            parts.append(part)
        except FileNotFoundError:
            pass
    if not parts:
        raise FileNotFoundError(
            f"No parquet files found for {strategy}/{timeframe}/fold_{fold}"
        )
    return pd.concat(parts, ignore_index=True).sort_values("entry_time")


def _load_wf_trades(strategy: str, timeframe: str, fold: int, sample: str) -> list[dict]:
    if sample == "both":
        # No single-file cache for "both" — assemble on the fly
        df = _load_wf_df(strategy, timeframe, fold, "both").copy()
        for col in ("entry_time", "exit_time"):
            if col in df.columns:
                df[col] = df[col].dt.strftime("%Y-%m-%dT%H:%M:%S")
        cols = [c for c in _TRADE_COLUMNS if c in df.columns]
        return df[cols].to_dict(orient="records")

    key = (strategy, timeframe, fold, sample)
    if key not in _wf_trades_cache:
        df = _load_wf_df_single(strategy, timeframe, fold, sample).copy()
        for col in ("entry_time", "exit_time"):
            if col in df.columns:
                df[col] = df[col].dt.strftime("%Y-%m-%dT%H:%M:%S")
        cols = [c for c in _TRADE_COLUMNS if c in df.columns]
        _wf_trades_cache[key] = df[cols].to_dict(orient="records")
    return _wf_trades_cache[key]


# ── Shared filter dependency ──────────────────────────────────────────────────

@dataclass
class WfFilters:
    strategy: str
    timeframe: str
    fold: int
    sample: str
    variant: str | None
    ticker: str | None
    date_from: str | None
    date_to: str | None
    initial_capital: float
    risk_pct: float


def _wf_filters(
    strategy: str = Query(..., description="Strategy folder name, e.g. backside_short_lower_low"),
    timeframe: str = Query(..., description="Timeframe folder name, e.g. 5m or 15m"),
    fold: int = Query(..., ge=1, description="Walk-forward fold number"),
    sample: str = Query(
        default="out_of_sample",
        description="Which split to load: in_sample | out_of_sample | both",
    ),
    variant: str | None = Query(default=None, description="Filter by strategy column value"),
    ticker: str | None = Query(default=None, description="Filter by ticker (case-insensitive)"),
    date_from: str | None = Query(default=None, description="entry_time >= YYYY-MM-DD"),
    date_to: str | None = Query(default=None, description="entry_time <= YYYY-MM-DD"),
    initial_capital: float = Query(default=1_000.0),
    risk_pct: float = Query(default=0.01, ge=0.0, le=1.0),
) -> WfFilters:
    if sample not in ("in_sample", "out_of_sample", "both"):
        raise HTTPException(
            status_code=422,
            detail="sample must be one of: in_sample, out_of_sample, both",
        )
    return WfFilters(
        strategy=strategy, timeframe=timeframe, fold=fold, sample=sample,
        variant=variant, ticker=ticker,
        date_from=date_from, date_to=date_to,
        initial_capital=initial_capital, risk_pct=risk_pct,
    )


def _apply_wf_filters(f: WfFilters) -> pd.DataFrame:
    try:
        df = _load_wf_df(f.strategy, f.timeframe, f.fold, f.sample).copy()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"No trades file found: {exc}") from exc

    if f.variant is not None and f.variant.lower() not in ("all", ""):
        variants = {v.strip() for v in f.variant.split(",") if v.strip()}
        if variants:
            df = df[df["strategy"].isin(variants)]
    if f.ticker is not None:
        df = df[df["ticker"].str.upper() == f.ticker.upper()]
    if f.date_from is not None:
        df = df[df["entry_time"].dt.date >= pd.to_datetime(f.date_from).date()]
    if f.date_to is not None:
        df = df[df["entry_time"].dt.date <= pd.to_datetime(f.date_to).date()]

    if df.empty:
        raise HTTPException(status_code=422, detail="No trades match the given filters.")
    if len(df) < 2:
        raise HTTPException(status_code=422, detail="At least 2 trades are required.")
    return df


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/trades")
async def wf_list_trades(
    strategy: str = Query(...),
    timeframe: str = Query(...),
    fold: int = Query(..., ge=1),
    sample: str = Query(default="out_of_sample"),
    variant: str | None = Query(default=None),
    ticker: str | None = Query(default=None),
    date_from: str | None = Query(default=None),
    date_to: str | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=500, ge=1, le=2000),
):
    if sample not in ("in_sample", "out_of_sample", "both"):
        raise HTTPException(status_code=422, detail="sample must be in_sample | out_of_sample | both")
    try:
        all_trades = _load_wf_trades(strategy, timeframe, fold, sample)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"No trades file found: {exc}") from exc

    if variant and variant.lower() not in ("all", ""):
        variant_set = {v.strip() for v in variant.split(",") if v.strip()}
        if variant_set:
            all_trades = [t for t in all_trades if t.get("strategy") in variant_set]
    if ticker:
        ticker_upper = ticker.upper()
        all_trades = [t for t in all_trades if (t.get("ticker") or "").upper() == ticker_upper]
    if date_from:
        all_trades = [t for t in all_trades if (t.get("entry_time") or "")[:10] >= date_from]
    if date_to:
        all_trades = [t for t in all_trades if (t.get("entry_time") or "")[:10] <= date_to]

    total = len(all_trades)
    pages = (total + page_size - 1) // page_size
    offset = (page - 1) * page_size
    return {
        "trades": all_trades[offset: offset + page_size],
        "total": total,
        "page": page,
        "page_size": page_size,
        "pages": pages,
    }


@router.get("/summary")
async def wf_get_summary(f: WfFilters = Depends(_wf_filters)):
    from app.utils.trade_metrics import summary_report

    df = _apply_wf_filters(f)
    try:
        report = summary_report(df, initial_capital=f.initial_capital, risk_pct=f.risk_pct)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"summary_report error: {exc}") from exc

    winners = int(df["is_profit"].sum())
    losers = len(df) - winners
    return {
        "trades_count": len(df),
        "winners": winners,
        "losers": losers,
        "summary": {k: _clean(v) for k, v in report.items()},
    }


@router.get("/equity")
async def wf_get_equity(f: WfFilters = Depends(_wf_filters)):
    df = _apply_wf_filters(f)
    equity, _ = _build_equity(df, f)

    daily = equity.resample("D").last().dropna()
    years = (daily.index[-1] - daily.index[0]).days / 365.25
    cagr = (daily.iloc[-1] / daily.iloc[0]) ** (1 / years) - 1 if years > 0 else 0.0
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


@router.get("/drawdown")
async def wf_get_drawdown(f: WfFilters = Depends(_wf_filters)):
    from app.utils.trade_metrics import drawdown_series

    df = _apply_wf_filters(f)
    equity, _ = _build_equity(df, f)
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


@router.get("/returns-histogram")
async def wf_get_returns_histogram(
    bins: int = Query(default=50, ge=5, le=200),
    f: WfFilters = Depends(_wf_filters),
):
    df = _apply_wf_filters(f)
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


@router.get("/stress-test")
async def wf_get_stress_test(
    top_pct: float = Query(default=5.0, ge=0.0, le=50.0),
    bins: int = Query(default=50, ge=5, le=200),
    f: WfFilters = Depends(_wf_filters),
):
    from app.utils.trade_metrics import summary_report

    df = _apply_wf_filters(f)
    original_count = len(df)

    n_trim = max(1, int(np.ceil(original_count * top_pct / 100))) if top_pct > 0 else 0
    df_trimmed = df.sort_values("pnl", ascending=False).iloc[n_trim:].copy()

    _, returns = _build_equity(df_trimmed, f)
    var_5 = float(returns.quantile(0.05)) if len(returns) else 0.0
    counts, edges = np.histogram(returns.dropna(), bins=bins)
    histogram = [
        {"x": round(float(edges[i]), 6), "count": int(counts[i])}
        for i in range(len(counts))
    ]

    try:
        report = summary_report(df_trimmed, initial_capital=f.initial_capital, risk_pct=f.risk_pct)
    except Exception:
        report = {}

    winners = int(df_trimmed["is_profit"].sum())
    losers = len(df_trimmed) - winners
    return {
        "original_count": original_count,
        "trimmed_count": n_trim,
        "remaining_count": len(df_trimmed),
        "top_pct": top_pct,
        "var_5pct": round(var_5, 6),
        "bins": len(histogram),
        "histogram": histogram,
        "trades_count": len(df_trimmed),
        "winners": winners,
        "losers": losers,
        "summary": {k: _clean(v) for k, v in report.items()},
    }


@router.get("/stress-test/equity")
async def wf_get_stress_test_equity(
    top_pct: float = Query(default=5.0, ge=0.0, le=50.0),
    f: WfFilters = Depends(_wf_filters),
):
    df = _apply_wf_filters(f)
    original_count = len(df)

    n_trim = max(1, int(np.ceil(original_count * top_pct / 100))) if top_pct > 0 else 0
    df_trimmed = df.sort_values("pnl", ascending=False).iloc[n_trim:].copy()

    equity, _ = _build_equity(df_trimmed, f)
    daily = equity.resample("D").last().dropna()

    if len(daily) < 2:
        return {"trades_count": len(df_trimmed), "days_count": 0, "curve": []}

    years = (daily.index[-1] - daily.index[0]).days / 365.25
    cagr = (daily.iloc[-1] / daily.iloc[0]) ** (1 / years) - 1 if years > 0 else 0.0
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
        "original_count": original_count,
        "trimmed_count": n_trim,
        "trades_count": len(df_trimmed),
        "days_count": len(points),
        "initial_capital": f.initial_capital,
        "final_equity": round(float(daily.iloc[-1]), 4),
        "cagr": round(cagr, 6),
        "curve": points,
    }
