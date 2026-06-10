# New Iterative Strategy

Create a new iterative backtest strategy function for this project.

## Usage

```
/new-strategy <function_name> [description of the signal logic]
```

**Example:**
```
/new-strategy gap_up_reversal_short  Short stocks that gap up >30% and show a lower low on the 5m chart
```

---

## Rules — enforce these on every strategy you create

### 1. Output schema
The return value MUST always be `enforce_schema(pd.DataFrame(trades))`.  
Import it from `strategies.iterative.strategy_base`:

```python
from strategies.iterative.strategy_base import enforce_schema
```

The 22 required columns are defined in `TRADE_COLUMNS` in that module. Any column not populated by the strategy is filled with `NaN` automatically by `enforce_schema`. Never build the DataFrame with a hand-rolled column list.

### 2. Next-bar-open execution — no lookahead
All entries AND exits must happen at the **open of the next bar** after the signal bar. Use the `pending_entry` / `pending_exit` boolean flag pattern:

- On the **signal bar** (`i`): set `pending_entry = True` (or `pending_exit = True`).
- On the **next bar** (`i+1`): execute at `row["open"]` and then `continue` (do not look for a new signal on the execution bar).

Never execute on the same bar that produced the signal.

### 3. Function signature
```python
def <function_name>(
    candles: pd.DataFrame,
    gap_pct: float = 0.40,
    stop_pct: float = 0.50,
    tp_pct: float = 0.20,
    slippage: float = 0.001,
    timeframe_minutes: int = 5,
) -> pd.DataFrame:
```

`candles` contains one trading day for one ticker. The function is called once per ticker per date by `run_backtest`.

### 4. Register the strategy
After writing the function in `strategies/iterative/small_caps.py`, add an entry to `strategies/iterative/strategies_registry.py`:

```python
{
    "strategy_name": "<function_name>",
    "strategy_func": <function_name>,
    "params": [
        {
            "slippage": 0.001,
            "gap_pct": 0.40,
            "stop_pct": 0.50,
            "tp_pct": 0.20,
            "out_put_name": "<function_name>_0.4_0.5_0.2",
        },
    ],
},
```

Add the import at the top of `strategies_registry.py`:
```python
from strategies.iterative.small_caps import <function_name>
```

---

## Full template — copy and fill in signal logic

```python
from strategies.iterative.strategy_base import enforce_schema

def $ARGUMENTS(
    candles: pd.DataFrame,
    gap_pct: float = 0.40,
    stop_pct: float = 0.50,
    tp_pct: float = 0.20,
    slippage: float = 0.001,
    timeframe_minutes: int = 5,
) -> pd.DataFrame:
    STRATEGY = f"$ARGUMENTS_{gap_pct}_{stop_pct}_{tp_pct}"
    CLOSE_HOUR = 16
    expected_delta = pd.Timedelta(minutes=timeframe_minutes)

    df = candles.reset_index(drop=True)
    if len(df) < 3:
        return enforce_schema(pd.DataFrame())

    ticker = df["ticker"].iloc[0]

    before_close = df["date"].dt.hour < CLOSE_HOUR
    if not before_close.any():
        return enforce_schema(pd.DataFrame())
    last_valid_idx = before_close[before_close].index[-1]

    no_gap = df["date"].diff() == expected_delta

    # ── YOUR SIGNAL VECTOR ───────────────────────────────────────────────────
    # Replace this with real signal logic. signal.iloc[i] == True means
    # "enter on the open of bar i+1".
    signal = pd.Series(False, index=df.index)

    trades: list[dict] = []
    position       = None
    pending_entry  = False
    pending_exit   = False
    entry_price    = sl_price = tp_price = 0.0
    entry_time     = entry_volume = entry_date_str = None
    signal_volume  = signal_rvol = signal_prev_close = None
    trade_highs: list[float] = []
    trade_lows:  list[float] = []

    for i in range(1, last_valid_idx + 1):
        row = df.iloc[i]

        # ── EXIT at open of this bar (signal was set on the previous bar) ────
        if pending_exit:
            exit_price = row["open"] * (1 + slippage)   # short: buy-to-cover
            pnl  = entry_price - exit_price
            rr   = (entry_price - tp_price) / (sl_price - entry_price)
            mae  = max(trade_highs) - entry_price
            mfe  = entry_price - min(trade_lows)
            trades.append({
                "ticker":             ticker,
                "date_str":           entry_date_str,
                "type":               "short",
                "entry_price":        entry_price,
                "exit_price":         exit_price,
                "stop_loss_price":    round(sl_price, 4),
                "take_profit_price":  round(tp_price, 4),
                "risk_reward_ratio":  round(rr, 4),
                "pnl":                round(pnl, 4),
                "Return":             round(pnl / entry_price, 4),
                "MAE":                round(mae, 4),
                "mae_pct":            round(mae / entry_price * 100, 4),
                "MFE":                round(mfe, 4),
                "mfe_pct":            round(mfe / entry_price * 100, 4),
                "rvol_daily":         signal_rvol,
                "previous_day_close": signal_prev_close,
                "volume":             signal_volume,
                "entry_volume":       entry_volume,
                "entry_time":         entry_time,
                "exit_time":          row["date"],
                "strategy":           STRATEGY,
                # "timeframe" is injected by run_backtest after this call
            })
            position     = None
            pending_exit = False
            trade_highs  = []
            trade_lows   = []
            continue   # don't look for a new signal on the exit bar

        # ── ENTER at open of this bar (signal was set on the previous bar) ───
        if pending_entry:
            entry_price    = row["open"] * (1 - slippage)   # short: sell
            sl_price       = entry_price * (1 + stop_pct)
            tp_price       = entry_price * (1 - tp_pct)
            entry_time     = row["date"]
            entry_volume   = row["volume"]
            entry_date_str = row["date_str"]
            position       = "short"
            pending_entry  = False
            trade_highs    = [row["high"]]
            trade_lows     = [row["low"]]

        elif position == "short":
            trade_highs.append(row["high"])
            trade_lows.append(row["low"])

        # ── SL / TP / forced EOD close ───────────────────────────────────────
        if position == "short":
            hit_tp  = row["low"]  <= tp_price
            hit_sl  = row["high"] >= sl_price
            is_last = i == last_valid_idx

            if (hit_tp or hit_sl) and not is_last:
                pending_exit = True
            elif is_last:
                exit_price = row["close"] * (1 + slippage)
                pnl  = entry_price - exit_price
                rr   = (entry_price - tp_price) / (sl_price - entry_price)
                mae  = max(trade_highs) - entry_price
                mfe  = entry_price - min(trade_lows)
                trades.append({
                    "ticker":             ticker,
                    "date_str":           entry_date_str,
                    "type":               "short",
                    "entry_price":        entry_price,
                    "exit_price":         exit_price,
                    "stop_loss_price":    round(sl_price, 4),
                    "take_profit_price":  round(tp_price, 4),
                    "risk_reward_ratio":  round(rr, 4),
                    "pnl":                round(pnl, 4),
                    "Return":             round(pnl / entry_price, 4),
                    "MAE":                round(mae, 4),
                    "mae_pct":            round(mae / entry_price * 100, 4),
                    "MFE":                round(mfe, 4),
                    "mfe_pct":            round(mfe / entry_price * 100, 4),
                    "rvol_daily":         signal_rvol,
                    "previous_day_close": signal_prev_close,
                    "volume":             signal_volume,
                    "entry_volume":       entry_volume,
                    "entry_time":         entry_time,
                    "exit_time":          row["date"],
                    "strategy":           STRATEGY,
                })
                position    = None
                trade_highs = []
                trade_lows  = []

        # ── SIGNAL DETECTION — entry fires on the NEXT bar ───────────────────
        next_i = i + 1
        if (
            position is None
            and signal.iloc[i]            # <-- replace with your condition
            and next_i <= last_valid_idx
            and no_gap.iloc[next_i]
        ):
            pending_entry     = True
            signal_volume     = row["volume"]
            signal_rvol       = row.get("RVOL_daily")
            signal_prev_close = row.get("previous_day_close")

    return enforce_schema(pd.DataFrame(trades))
```

---

## Steps to complete after inserting the template

1. Replace the `signal` vector with your real entry condition (computed before the loop or inside it).
2. Adjust SL/TP direction if the strategy is long (reverse the slippage signs and price comparisons).
3. Add any extra fields you compute to the `trades.append({...})` dicts — `enforce_schema` will keep only the 22 canonical columns, so extras are silently dropped.
4. Add the function to `strategies_registry.py` (see Rule 4 above).
5. Run a quick smoke test:
   ```
   python strategies/iterative/small_caps.py
   ```
