from pydantic import BaseModel, Field
from typing import Any, Literal
from datetime import date


# ── Backtest request/response ────────────────────────────────────────────────

class BacktestRequest(BaseModel):
    ticker: str = Field(..., examples=["AAPL"])
    from_date: date = Field(..., examples=["2023-01-01"])
    to_date: date = Field(..., examples=["2024-01-01"])
    strategy: str = Field(default="sma_crossover", examples=["sma_crossover"])
    params: dict[str, Any] = Field(default_factory=dict, examples=[{"fast": 10, "slow": 50}])


class JobSubmitted(BaseModel):
    job_id: str
    status: Literal["pending"] = "pending"


# ── Job status ───────────────────────────────────────────────────────────────

class JobStatus(BaseModel):
    job_id: str
    status: Literal["pending", "running", "success", "failure"]
    result: Any | None = None
    error: str | None = None


# ── Analysis results ─────────────────────────────────────────────────────────

class EquityCurvePoint(BaseModel):
    time: str   # ISO date string
    value: float


class PerformanceMetrics(BaseModel):
    total_return: float
    sharpe_ratio: float | None
    max_drawdown: float
    win_rate: float
    total_trades: int
    profit_factor: float | None


class BacktestResult(BaseModel):
    ticker: str
    strategy: str
    params: dict[str, Any]
    metrics: PerformanceMetrics
    equity_curve: list[EquityCurvePoint]
    trades: list[dict[str, Any]]


# ── Indicator request ─────────────────────────────────────────────────────────

class IndicatorRequest(BaseModel):
    prices: list[float] = Field(..., description="Close prices in chronological order")
    indicator: str = Field(..., examples=["sma", "ema", "rsi", "macd", "bbands"])
    params: dict[str, Any] = Field(default_factory=dict)
