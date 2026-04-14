# Backtester API

Backtesting engine for intraday small-cap strategies built on vectorbt. All commands are run from the project root (`backtester_api/`).

---

## Scripts

### `scripts/pipeline.py`
Fetches market data from the data source and injects it into the database.

```bash
# Full pipeline — all tickers
python scripts/pipeline.py run

# With tuning params
python scripts/pipeline.py run --procs 8 --concur 100

# Random sample of N tickers
python scripts/pipeline.py sample --n 50 --procs 4 --concur 100
```

---

### `scripts/build_walkforward_datasets.py`
Builds walk-forward datasets split into in-sample (IS) and out-of-sample (OOS) folds.
Each fold uses a 12-month IS window and a 6-month OOS window, sliding by 6 months.

Output: `backtest_dataset/walkforward/{timeframe}/fold_{1,2,3}/{in_sample,out_of_sample}.parquet`

```bash
python -m scripts.build_walkforward_datasets              # default: 5m
python -m scripts.build_walkforward_datasets --tf 15m
python -m scripts.build_walkforward_datasets --tf 1h
```

---

### `scripts/build_full_dataset.py`
Builds a single parquet with all filtered trading days from 2024-01-01 to the most recent available date. Used as the out-of-sample full dataset for final validation.

Output: `backtest_dataset/full/{timeframe}/full_dataset.parquet`

```bash
python -m scripts.build_full_dataset              # default: 5m
python -m scripts.build_full_dataset --tf 15m
python -m scripts.build_full_dataset --tf 1h
python -m scripts.build_full_dataset --tf 5m --from-date 2023-01-01
```

---

### `scripts/split_dataset_by_ticker.py`
Splits `full_dataset.parquet` into one `.parquet` file per ticker, preserving chronological order. Also generates a `ticker_row_counts.parquet` summary.

Output:
- `backtest_dataset/full/{timeframe}/tickers/{TICKER}.parquet`
- `backtest_dataset/full/{timeframe}/ticker_row_counts.parquet`

```bash
# Both timeframes (5m and 15m)
python scripts/split_dataset_by_ticker.py

# Single timeframe
python scripts/split_dataset_by_ticker.py --timeframe 5m
python scripts/split_dataset_by_ticker.py --timeframe 15m
```

---

### `scripts/plot_trade.py`
Plots a single trade from a trades parquet file with candles and entry/exit markers.

```bash
# First trade in the file
python -m scripts.plot_trade

# By position index
python -m scripts.plot_trade --index 7

# By ticker and date
python -m scripts.plot_trade --ticker ABOS --date 2022-09-28
```

---

## Dataset layout

```
backtest_dataset/
├── walkforward/
│   ├── 5m/
│   │   ├── fold_1/
│   │   │   ├── in_sample.parquet
│   │   │   └── out_of_sample.parquet
│   │   ├── fold_2/
│   │   └── fold_3/
│   └── 15m/
│       └── ...
└── full/
    ├── 5m/
    │   ├── full_dataset.parquet
    │   ├── ticker_row_counts.parquet
    │   ├── tickers/
    │   │   ├── AAPL.parquet
    │   │   └── ...
    │   └── trades/
    └── 15m/
        └── ...
```
