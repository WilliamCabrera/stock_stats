"""
Small-caps backtest results API.

Serves pre-computed trades and performance reports from the backtest_dataset
directory. Covers three evaluation periods:

  in-sample     — training window (2021-2022)
  out-of-sample — first unseen validation window (2022-2023)
  walk-forward  — 3-fold temporal cross-validation

Endpoint map
────────────────────────────────────────────────────────────────
GET /smallcaps/strategies/in-sample
GET /smallcaps/strategies/out-of-sample
GET /smallcaps/strategies/walk-forward

GET /smallcaps/in-sample/{strategy}/trades
GET /smallcaps/in-sample/{strategy}/report

GET /smallcaps/out-of-sample/{strategy}/trades
GET /smallcaps/out-of-sample/{strategy}/report

GET /smallcaps/walk-forward/{strategy}/trades?fold=1&split=in_sample
GET /smallcaps/walk-forward/{strategy}/report?fold=1&split=in_sample
────────────────────────────────────────────────────────────────

Note: the source data folder uses the typo "walk_fordward" — that is intentional
and matches the actual filenames on disk.
"""
import logging
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from app.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/smallcaps", tags=["smallcaps — backtest results"])

# Maps API period keys → actual folder names on disk
# (walk_fordward is a typo preserved from the original dataset)
_PERIOD_FOLDER = {
    "in_sample":      "in_sample",
    "out_of_sample":  "out_of_sample",
    "walk_forward":   "walk_fordward",
}

Split = Literal["in_sample", "out_sample"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _trades_dir(period_key: str) -> Path:
    root = Path(get_settings().dataset_path)
    return root / _PERIOD_FOLDER[period_key] / "trades"


def _load(path: Path) -> list[dict]:
    """Read a parquet file and return JSON-serialisable records."""
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path.name}")

    df = pd.read_parquet(path)

    # Convert any datetime column to ISO-8601 strings
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].astype(str)

    # Replace NaN / ±Inf with None so FastAPI can serialise to JSON
    df = df.where(pd.notnull(df), None)

    return df.to_dict(orient="records")


def _strategy_list(period_key: str) -> list[str]:
    d = _trades_dir(period_key)
    if not d.exists():
        return []
    return sorted(p.name for p in d.iterdir() if p.is_dir())


# ── Strategy discovery ────────────────────────────────────────────────────────

@router.get(
    "/strategies/in-sample",
    summary="List strategies available in the in-sample period",
)
async def strategies_in_sample():
    strategies = _strategy_list("in_sample")
    return {"period": "in_sample", "count": len(strategies), "strategies": strategies}


@router.get(
    "/strategies/out-of-sample",
    summary="List strategies available in the out-of-sample period",
)
async def strategies_out_of_sample():
    strategies = _strategy_list("out_of_sample")
    return {"period": "out_of_sample", "count": len(strategies), "strategies": strategies}


@router.get(
    "/strategies/walk-forward",
    summary="List strategies available in the walk-forward analysis (3 folds)",
)
async def strategies_walk_forward():
    strategies = _strategy_list("walk_forward")
    return {
        "period": "walk_forward",
        "folds": 3,
        "splits": ["in_sample", "out_sample"],
        "count": len(strategies),
        "strategies": strategies,
    }


# ── In-sample ─────────────────────────────────────────────────────────────────

@router.get(
    "/in-sample/{strategy}/trades",
    summary="Individual trades — in-sample",
    description="All executed trades for a strategy in the in-sample training period.",
)
async def in_sample_trades(strategy: str):
    path = _trades_dir("in_sample") / strategy / f"{strategy}_in_sample_trades.parquet"
    records = _load(path)
    return {"period": "in_sample", "strategy": strategy, "count": len(records), "trades": records}


@router.get(
    "/in-sample/{strategy}/report",
    summary="Performance report — in-sample",
    description="Aggregated metrics per parameter combination for a strategy in the in-sample period.",
)
async def in_sample_report(strategy: str):
    path = _trades_dir("in_sample") / strategy / f"{strategy}_in_sample_trade_stats.parquet"
    records = _load(path)
    return {"period": "in_sample", "strategy": strategy, "combinations": len(records), "report": records}


# ── Out-of-sample ─────────────────────────────────────────────────────────────

@router.get(
    "/out-of-sample/{strategy}/trades",
    summary="Individual trades — out-of-sample",
    description="All executed trades for a strategy in the out-of-sample validation period.",
)
async def out_of_sample_trades(strategy: str):
    path = _trades_dir("out_of_sample") / strategy / f"{strategy}_out_of_sample_trades.parquet"
    records = _load(path)
    return {"period": "out_of_sample", "strategy": strategy, "count": len(records), "trades": records}


@router.get(
    "/out-of-sample/{strategy}/report",
    summary="Performance report — out-of-sample",
    description="Aggregated metrics per parameter combination for a strategy in the out-of-sample period.",
)
async def out_of_sample_report(strategy: str):
    path = _trades_dir("out_of_sample") / strategy / f"{strategy}_out_of_sample_trade_stats.parquet"
    records = _load(path)
    return {"period": "out_of_sample", "strategy": strategy, "combinations": len(records), "report": records}


# ── Walk-forward ──────────────────────────────────────────────────────────────

@router.get(
    "/walk-forward/{strategy}/trades",
    summary="Individual trades — walk-forward fold",
    description=(
        "Executed trades for a specific walk-forward fold.\n\n"
        "- **fold** `1 | 2 | 3` — temporal cross-validation fold\n"
        "- **split** `in_sample` (training window) | `out_sample` (validation window)"
    ),
)
async def walk_forward_trades(
    strategy: str,
    fold: int = Query(..., ge=1, le=3, description="Fold number (1–3)"),
    split: Split = Query("in_sample", description="'in_sample' or 'out_sample'"),
):
    filename = f"walk_fordward_{split}_{fold}_trades.parquet"
    path = _trades_dir("walk_forward") / strategy / filename
    records = _load(path)
    return {
        "period": "walk_forward",
        "strategy": strategy,
        "fold": fold,
        "split": split,
        "count": len(records),
        "trades": records,
    }


@router.get(
    "/walk-forward/{strategy}/report",
    summary="Performance report — walk-forward fold",
    description=(
        "Aggregated metrics for a specific walk-forward fold.\n\n"
        "- **fold** `1 | 2 | 3` — temporal cross-validation fold\n"
        "- **split** `in_sample` (training window) | `out_sample` (validation window)"
    ),
)
async def walk_forward_report(
    strategy: str,
    fold: int = Query(..., ge=1, le=3, description="Fold number (1–3)"),
    split: Split = Query("in_sample", description="'in_sample' or 'out_sample'"),
):
    filename = f"walk_fordward_{strategy}_{split}_{fold}_trade_stats.parquet"
    path = _trades_dir("walk_forward") / strategy / filename
    records = _load(path)
    return {
        "period": "walk_forward",
        "strategy": strategy,
        "fold": fold,
        "split": split,
        "combinations": len(records),
        "report": records,
    }
