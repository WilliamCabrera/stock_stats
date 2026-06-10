"""
Celery tasks — heavy CPU work runs in isolated worker processes.
Each task receives plain-JSON-serialisable arguments and returns
a plain dict that is stored in Redis for the API to retrieve.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import ta
import vectorbt as vbt
from celery import Task
from celery.utils.log import get_task_logger

from app.worker.celery_app import celery

logger = get_task_logger(__name__)


# ── Base task with retry behaviour ───────────────────────────────────────────

class BaseBacktestTask(Task):
    abstract = True
    max_retries = 2
    default_retry_delay = 5


# ── Helper: build price DataFrame from raw OHLCV list ───────────────────────

def _build_df(ohlcv: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(ohlcv)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("time").sort_index()
    return df


# ── Strategies ───────────────────────────────────────────────────────────────

def _strategy_sma_crossover(df: pd.DataFrame, fast: int = 10, slow: int = 50) -> tuple:
    """Simple SMA crossover: buy when fast > slow, sell when fast < slow."""
    close = df["close"]
    fast_ma = ta.trend.sma_indicator(close, window=fast)
    slow_ma = ta.trend.sma_indicator(close, window=slow)
    entries = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
    exits = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
    return entries, exits


def _strategy_rsi_mean_reversion(
    df: pd.DataFrame, period: int = 14, oversold: float = 30, overbought: float = 70
) -> tuple:
    """RSI mean-reversion: buy oversold, sell overbought."""
    close = df["close"]
    rsi = ta.momentum.rsi(close, window=period)
    entries = rsi < oversold
    exits = rsi > overbought
    return entries, exits


STRATEGIES = {
    "sma_crossover": _strategy_sma_crossover,
    "rsi_mean_reversion": _strategy_rsi_mean_reversion,
}


# ── Main backtest task ────────────────────────────────────────────────────────

@celery.task(base=BaseBacktestTask, bind=True, name="app.tasks.backtest.run_backtest")
def run_backtest(self, ohlcv: list[dict], ticker: str, strategy: str, params: dict) -> dict:
    """
    Run a vectorbt backtest on the provided OHLCV data.

    Args:
        ohlcv:    list of {time, open, high, low, close, volume} dicts
        ticker:   symbol name (for labelling)
        strategy: strategy name key from STRATEGIES dict
        params:   strategy-specific parameters

    Returns:
        Serialisable dict with metrics, equity_curve and trades.
    """
    logger.info("Starting backtest: ticker=%s strategy=%s params=%s", ticker, strategy, params)
    self.update_state(state="PROGRESS", meta={"step": "building dataframe"})

    df = _build_df(ohlcv)
    close = df["close"]

    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Available: {list(STRATEGIES)}")

    self.update_state(state="PROGRESS", meta={"step": "computing signals"})
    strategy_fn = STRATEGIES[strategy]
    entries, exits = strategy_fn(df, **params)

    self.update_state(state="PROGRESS", meta={"step": "running vectorbt portfolio"})
    pf = vbt.Portfolio.from_signals(
        close,
        entries=entries,
        exits=exits,
        freq="5T",          # 5-minute bars — adjust if needed
        init_cash=10_000,
    )

    self.update_state(state="PROGRESS", meta={"step": "extracting results"})

    # Metrics
    stats = pf.stats()
    trades_df = pf.trades.records_readable

    equity = pf.value()
    equity_curve = [
        {"time": str(idx.date()), "value": round(float(v), 4)}
        for idx, v in equity.items()
    ]

    trades = trades_df.to_dict(orient="records") if not trades_df.empty else []
    # Make sure all values are JSON-serialisable
    for t in trades:
        for k, v in t.items():
            if isinstance(v, (np.integer,)):
                t[k] = int(v)
            elif isinstance(v, (np.floating, float)) and not np.isfinite(v):
                t[k] = None
            elif isinstance(v, np.floating):
                t[k] = float(v)
            elif hasattr(v, "isoformat"):
                t[k] = v.isoformat()

    def _safe(v):
        if v is None:
            return None
        f = float(v)
        return round(f, 4) if np.isfinite(f) else None

    result = {
        "ticker": ticker,
        "strategy": strategy,
        "params": params,
        "metrics": {
            "total_return": _safe(stats.get("Total Return [%]")),
            "sharpe_ratio": _safe(stats.get("Sharpe Ratio")),
            "max_drawdown": _safe(stats.get("Max Drawdown [%]")),
            "win_rate": _safe(stats.get("Win Rate [%]")),
            "total_trades": int(stats.get("Total Trades", 0)),
            "profit_factor": _safe(stats.get("Profit Factor")),
        },
        "equity_curve": equity_curve,
        "trades": trades,
    }
    logger.info("Backtest complete: %d trades", result["metrics"]["total_trades"])
    return result


# ── Indicator-only task (lighter, no portfolio simulation) ────────────────────

@celery.task(name="app.tasks.backtest.compute_indicators")
def compute_indicators(prices: list[float], indicator: str, params: dict) -> dict:
    """Compute a single TA indicator and return the values."""
    close = pd.Series(prices, dtype=float)

    ind = indicator.lower()
    if ind == "sma":
        values = ta.trend.sma_indicator(close, window=params.get("window", 20))
    elif ind == "ema":
        values = ta.trend.ema_indicator(close, window=params.get("window", 20))
    elif ind == "rsi":
        values = ta.momentum.rsi(close, window=params.get("window", 14))
    elif ind == "macd":
        macd_obj = ta.trend.MACD(
            close,
            window_slow=params.get("slow", 26),
            window_fast=params.get("fast", 12),
            window_sign=params.get("signal", 9),
        )
        return {
            "macd": [None if np.isnan(v) else round(float(v), 6) for v in macd_obj.macd()],
            "signal": [None if np.isnan(v) else round(float(v), 6) for v in macd_obj.macd_signal()],
            "hist": [None if np.isnan(v) else round(float(v), 6) for v in macd_obj.macd_diff()],
        }
    elif ind == "bbands":
        bb = ta.volatility.BollingerBands(close, window=params.get("window", 20))
        return {
            "upper": [None if np.isnan(v) else round(float(v), 6) for v in bb.bollinger_hband()],
            "mid": [None if np.isnan(v) else round(float(v), 6) for v in bb.bollinger_mavg()],
            "lower": [None if np.isnan(v) else round(float(v), 6) for v in bb.bollinger_lband()],
        }
    else:
        raise ValueError(f"Unknown indicator '{indicator}'")

    return {
        ind: [None if np.isnan(v) else round(float(v), 6) for v in values]
    }
