import os
import sys
sys.path.insert(0, os.path.abspath("."))

from strategies.iterative.small_caps import (
    backside_short_lower_low_fix_stop_iterative,
    gap_crap_iterative,
    short_push_exhaustion_iterative,
    push_rejection_iterative,
)
from strategies.iterative.trend_following import ema100_trend_follower_iterative
from strategies.iterative.orb_avg_range import orb_avg_range_iterative
from strategies.iterative.orb_first_candle import orb_first_candle_iterative

# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY REGISTRY
#
# Single source of truth binding each strategy to its dataset, tickers, and
# parameter sets. All runner functions in backtest_helpers.py read from this
# list — no routing or data-path logic lives in the runners themselves.
#
# Full spec: STRATEGY_REGISTRY_SPEC.md
#
# Required fields (every entry)
# ──────────────────────────────
#   strategy_name  str        Unique key. Must match the function name.
#   strategy_func  callable   Imported strategy function.
#   dataset        str        "small_caps" | "indices"
#   params         list[dict] One dict per param set. Fields:
#                               slippage, gap_pct, stop_pct, tp_pct, out_put_name
#
# Additional fields for dataset="indices"
# ────────────────────────────────────────
#   data_root  str        Relative path to the index dataset root.
#                         Full path: {data_root}/{ticker}/{tf}/{ticker_lower}_full_dataset.parquet
#   tickers    list[str]  Tickers to run on, e.g. ["QQQ", "TQQQ"].
#   timeframes list[str]  Optional. Defaults to ["5m"].
#
# Dataset routing summary
# ────────────────────────
#   small_caps → backtest_dataset/full/{tf}/dates/  (per-date parquet, all tickers)
#                output: strategies/iterative/UP-TO-DATE/{tf}/{out_put_name}/
#                supports: up-to-date, walkforward, incremental
#
#   indices    → {data_root}/{ticker}/{tf}/  (per-ticker monolithic parquet)
#                output: strategies/iterative/UP-TO-DATE/INDICES/{tf}/{out_put_name}/
#                supports: up-to-date only
# ─────────────────────────────────────────────────────────────────────────────

STRATEGIES = [
    {
        "strategy_name": "backside_short_lower_low_fix_stop_iterative",
        "strategy_func": backside_short_lower_low_fix_stop_iterative,
        "dataset": "small_caps",
        "params": [
            {"slippage": 0.001, "gap_pct": 0.40, "stop_pct": 0.50, "tp_pct": 0.20, "out_put_name": "backside_short_lower_low_fix_stop_iterative_0.4_0.5_0.2"},
        ],
    },
    {
        "strategy_name": "gap_crap_iterative",
        "strategy_func": gap_crap_iterative,
        "dataset": "small_caps",
        "params": [
            {"slippage": 0.001, "gap_pct": 0.40, "stop_pct": 0.50, "tp_pct": 0.20, "out_put_name": "gap_crap_iterative_0.4_0.5_0.2"},
        ],
    },
    {
        "strategy_name": "short_push_exhaustion_iterative",
        "strategy_func": short_push_exhaustion_iterative,
        "dataset": "small_caps",
        "params": [
            {"slippage": 0.001, "gap_pct": 0.40, "stop_pct": 0.50, "tp_pct": 0.20, "out_put_name": "short_push_exhaustion_iterative_0.4_0.5_0.2"},
        ],
    },
    {
        "strategy_name": "push_rejection_iterative",
        "strategy_func": push_rejection_iterative,
        "dataset": "small_caps",
        "params": [
            {"slippage": 0.001, "gap_pct": 0.40, "stop_pct": 0.50, "tp_pct": 0.20, "out_put_name": "push_rejection_iterative_0.4_0.5_0.2"},
        ],
    },
    {
        "strategy_name": "orb_first_candle_iterative",
        "strategy_func": orb_first_candle_iterative,
        "dataset": "indices",
        "data_root": "backtest_dataset/INDICES",
        "tickers": ["QQQ", "TQQQ"],
        "timeframes": ["5m"],
        "params": [
            {"slippage": 0, "gap_pct": 0, "stop_pct": 0, "tp_pct": 0, "out_put_name": "orb_first_candle_iterative"},
        ],
    },
    {
        "strategy_name": "orb_avg_range_iterative",
        "strategy_func": orb_avg_range_iterative,
        "dataset": "indices",
        "data_root": "backtest_dataset/INDICES",
        "tickers": ["QQQ", "TQQQ"],
        "timeframes": ["10m"],
        "params": [
            {"slippage": 0.001, "gap_pct": 0.40, "stop_pct": 0.50, "tp_pct": 0.20, "out_put_name": "orb_avg_range_iterative"},
        ],
    },
    # {
    #     "strategy_name": "ema100_trend_follower_iterative",
    #     "strategy_func": ema100_trend_follower_iterative,
    #     "dataset": "indices",
    #     # data_root: root of the per-ticker index dataset.
    #     # Path pattern: {data_root}/{ticker}/{tf}/{ticker_lower}_full_dataset.parquet
    #     "data_root": "backtest_dataset/INDICES",
    #     "tickers": ["QQQ", "TQQQ"],
    #     "timeframes": ["1d"],
    #     "params": [
    #         {"slippage": 0.001, "gap_pct": 0.40, "stop_pct": 0.50, "tp_pct": 0.20, "out_put_name": "ema100_trend_follower_iterative_100_20_14"},
    #     ],
    # },
]
