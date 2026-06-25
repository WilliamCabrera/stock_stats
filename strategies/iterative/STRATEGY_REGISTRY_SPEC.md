# Strategy Registry — Spec & Documentation

## Overview

`strategies_registry.py` is the single source of truth that binds a strategy function to a dataset,
its execution parameters, and the tickers/timeframes it should run on. All runner functions in
`backtest_helpers.py` read from this registry — nothing about routing or data paths is hardcoded in
the runners themselves.

---

## Entry Schema

Each entry in `STRATEGIES` is a `dict` with the following fields:

### Required fields (all datasets)

| Field | Type | Description |
|---|---|---|
| `strategy_name` | `str` | Unique identifier. Used in log messages and output directory names. Must match the function name. |
| `strategy_func` | `callable` | The strategy function imported at the top of the file. |
| `dataset` | `str` | Which dataset family to run against. One of `"small_caps"` or `"indices"`. Controls which runner path is taken. |
| `params` | `list[dict]` | One or more parameter sets. Each element is a `dict` (see [Params schema](#params-schema) below). |

### Required fields — `dataset: "indices"` only

| Field | Type | Description |
|---|---|---|
| `data_root` | `str` | Relative path to the root of the per-ticker index dataset (from the project root). The runner constructs the full path as `{data_root}/{ticker}/{tf}/{ticker_lower}_full_dataset.parquet`. |
| `tickers` | `list[str]` | Tickers to run the strategy on, e.g. `["QQQ", "TQQQ"]`. |

### Optional fields

| Field | Type | Default | Description |
|---|---|---|---|
| `timeframes` | `list[str]` | `["5m"]` (indices) / `["5m", "15m"]` (small caps) | Timeframes to iterate over. Must exist in the target dataset. |

---

## Params Schema

Each element of `params` is a `dict` with the following fields:

| Field | Type | Description |
|---|---|---|
| `slippage` | `float` | Fraction applied to entry/exit price to simulate market impact (e.g. `0.001` = 0.1%). |
| `gap_pct` | `float` | Minimum gap percent required for entry signal. Strategy-specific semantics. |
| `stop_pct` | `float` | Stop loss distance as a fraction of entry price. |
| `tp_pct` | `float` | Take profit distance as a fraction of entry price. |
| `out_put_name` | `str` | Unique name for this param set. Used as the output filename stem and directory name. Convention: `{strategy_name}_{gap_pct}_{stop_pct}_{tp_pct}`. |

Multiple elements in `params` let you backtest the same strategy with different parameter combinations
in a single run.

---

## Dataset Types

### `small_caps`

- **Data source:** `backtest_dataset/full/{tf}/dates/YYYY_MM_DD.parquet`
  Each date file contains all active tickers for that day.
- **Runner:** `run_backtest()` — iterates date files, calls `strategy_func(candles)` per ticker per day.
- **WF support:** yes (`run_walkforward_backtest`)
- **Incremental support:** yes (`run_iterative_incremental_backtest`)
- **Output:** `strategies/iterative/UP-TO-DATE/{tf}/{out_put_name}/{out_put_name}.parquet`

### `indices`

- **Data source:** `{data_root}/{ticker}/{tf}/{ticker_lower}_full_dataset.parquet`
  One monolithic parquet per ticker; the runner splits it by `date_str` internally.
- **Runner:** `_run_indices_uptodate()` — iterates tickers from `entry["tickers"]`, splits by date,
  calls `strategy_func(candles, ticker_parquet_path=...)` per day. The resolved parquet path is
  passed directly to the strategy so the strategy itself contains no path logic.
- **WF support:** no (skipped with a log warning)
- **Incremental support:** no (skipped with a log warning)
- **Output:** `strategies/iterative/UP-TO-DATE/INDICES/{tf}/{out_put_name}/{out_put_name}.parquet`

---

## How the runners use the registry

```
run_up_to_date_backtest()
    └─ for entry in STRATEGIES:
          if entry["dataset"] == "indices":
              _run_indices_uptodate(entry, p)    ← uses data_root, tickers, timeframes
          else:
              run_backtest(dates_dir="backtest_dataset/full/{tf}/dates", ...)

run_walkforward_backtest()
    └─ skips entries where dataset != "small_caps"

run_iterative_incremental_backtest()
    └─ skips entries where dataset != "small_caps"
```

---

## Strategy function interface

Every strategy function must accept these keyword arguments (passed by the runners):

```python
def my_strategy(
    candles: pd.DataFrame,      # one day × one ticker
    gap_pct: float,
    stop_pct: float,
    tp_pct: float,
    slippage: float,
    timeframe_minutes: int,
) -> pd.DataFrame:              # trades, or empty DataFrame
```

Strategies targeting the `"indices"` dataset additionally receive:

```python
    ticker_parquet_path: Path,  # full path to the ticker's historical parquet
                                # (resolved by the runner from data_root in registry)
```

The strategy uses `ticker_parquet_path` to load historical data for indicator computation
(EMA, ATR, etc.). When `None` (default), the strategy falls back to the small-caps path.

Return value must conform to the schema enforced by `enforce_schema()` in `strategy_base.py`.

---

## Adding a new strategy

### Small-caps strategy

1. Implement the function in the appropriate module (e.g. `small_caps.py`).
2. Import it at the top of `strategies_registry.py`.
3. Add an entry to `STRATEGIES`:

```python
{
    "strategy_name": "my_new_strategy",
    "strategy_func": my_new_strategy,
    "dataset": "small_caps",
    "params": [
        {"slippage": 0.001, "gap_pct": 0.40, "stop_pct": 0.50, "tp_pct": 0.20,
         "out_put_name": "my_new_strategy_0.4_0.5_0.2"},
    ],
},
```

### Index strategy

1. Implement the function. The function signature must include `ticker_parquet_path: Path | None = None`.
2. Import it at the top of `strategies_registry.py`.
3. Add an entry to `STRATEGIES`:

```python
{
    "strategy_name": "my_index_strategy",
    "strategy_func": my_index_strategy,
    "dataset": "indices",
    "data_root": "backtest_dataset/INDICES",
    "tickers": ["QQQ", "TQQQ", "SPY"],
    "timeframes": ["5m"],
    "params": [
        {"slippage": 0.001, "gap_pct": 0.40, "stop_pct": 0.50, "tp_pct": 0.20,
         "out_put_name": "my_index_strategy_0.4_0.5_0.2"},
    ],
},
```

---

## Output layout

```
strategies/iterative/
├── UP-TO-DATE/
│   ├── 5m/
│   │   └── {out_put_name}/
│   │       └── {out_put_name}.parquet          ← small_caps
│   ├── 15m/
│   │   └── {out_put_name}/
│   │       └── {out_put_name}.parquet          ← small_caps
│   └── INDICES/
│       └── 5m/
│           └── {out_put_name}/
│               └── {out_put_name}.parquet      ← indices
└── WF/
    ├── IN-SAMPLE/
    │   ├── 5m/  └── {tier}/  └── {out_put_name}/  └── {out_put_name}.parquet
    │   └── 15m/ └── ...
    └── OUT-OF-SAMPLE/
        └── ...                                 ← small_caps only
```

---

## Invariants

- `strategy_name` must be unique across all entries.
- `out_put_name` must be unique across all entries and param sets (it is used as a directory name).
- `dataset: "indices"` entries must include `data_root` and `tickers`.
- Strategy functions must be pure with respect to the `candles` input — do not mutate the DataFrame.
- Adding an entry here is sufficient to include a strategy in all applicable runner passes. No changes
  to `backtest_helpers.py` are needed unless a new dataset type is introduced.
