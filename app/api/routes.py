from fastapi import APIRouter, HTTPException, Query
from typing import Literal
from celery.result import AsyncResult

from app.schemas import (
    BacktestRequest,
    JobSubmitted,
    JobStatus,
    IndicatorRequest,
)
from app.tasks.backtest import run_backtest, compute_indicators
from app.utils.massive import fetch_candles_async, TimeFrame

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
