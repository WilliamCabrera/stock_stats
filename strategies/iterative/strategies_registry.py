import os
import sys
sys.path.insert(0, os.path.abspath("."))

from strategies.iterative.small_caps import (
    backside_short_lower_low_fix_stop_iterative,
    gap_crap_iterative,
    short_push_exhaustion_iterative,
    push_rejection_iterative,
)

STRATEGIES = [
    # {
    #     "strategy_name": "backside_short_lower_low_fix_stop_iterative",
    #     "strategy_func": backside_short_lower_low_fix_stop_iterative,
    #     "params": [
    #         {"slippage": 0.001, "gap_pct": 0.40, "stop_pct": 0.50, "tp_pct": 0.20, "out_put_name": "backside_short_lower_low_fix_stop_iterative_0.4_0.5_0.2"},    
    #     ],
    # },
    # {
    #     "strategy_name": "gap_crap_iterative",
    #     "strategy_func": gap_crap_iterative,
    #     "params": [
    #         {"slippage": 0.001, "gap_pct": 0.40, "stop_pct": 0.50, "tp_pct": 0.20, "out_put_name": "gap_crap_iterative_0.4_0.5_0.2"},
    #     ],
    # },
    # {
    #     "strategy_name": "short_push_exhaustion_iterative",
    #     "strategy_func": short_push_exhaustion_iterative,
    #     "params": [
    #         {"slippage": 0.001, "gap_pct": 0.40, "stop_pct": 0.50, "tp_pct": 0.20, "out_put_name": "short_push_exhaustion_iterative_0.4_0.5_0.2"},
    #     ],
    # },
    {
        "strategy_name": "push_rejection_iterative",
        "strategy_func": push_rejection_iterative,
        "params": [
            {"slippage": 0.001, "gap_pct": 0.40, "stop_pct": 0.50, "tp_pct": 0.20, "out_put_name": "push_rejection_iterative_0.4_0.5_0.2"},
        ],
    },
]
