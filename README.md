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

## Automated daily pipeline

Three jobs run automatically every day via **Ofelia** (the Docker-based cron scheduler). They execute inside the `pipeline` container and are defined in `docker-compose.yml`.

| Hora (ET) | Job                    | Descripción                                    |
|-----------|------------------------|------------------------------------------------|
| **20:30** | `incremental-pipeline` | Ingesta de datos de mercado en PostgreSQL      |
| **02:00** | `update-full-dataset`  | Fetchea velas 5m/15m y actualiza los parquets  |
| **04:00** | `incremental-backtest` | Corre todas las estrategias sobre datos nuevos |

**20:30 — `incremental-pipeline`**
Detecta los tickers con actividad reciente (gappers, movers), fetchea sus datos OHLCV del día desde Massive.com, los inserta/actualiza en la tabla `stock_data` de PostgreSQL y refresca la vista materializada `stock_data_filtered` para que solo queden los tickers que cumplen los filtros de calidad (gap > 40 %, prev-close > $0.10, etc.).

**02:00 — `update-full-dataset`**
Lee `stock_data_filtered` para encontrar los ticker-días nuevos (fecha > última fecha en el parquet). Para cada uno fetchea las velas de 5m y 15m desde Massive, calcula los indicadores (ATR, VWAP, RVOL, SMAs, Donchian) y actualiza cuatro destinos: `full_dataset.parquet`, el fichero por ticker en `tickers/`, y los ficheros temporales `pending_candles_5m.parquet` / `pending_candles_15m.parquet`. También encola los ticker-días en `pending_backtest.parquet` para uso posterior.

**04:00 — `incremental-backtest`**
Lee `pending_candles_5m.parquet` y `pending_candles_15m.parquet` generados en el paso anterior y corre todas las estrategias (`backside_short_lower_low`, `short_push_exhaustion`, `gap_crap_strategy`) sobre esos datos. Los trades resultantes se agregan (append) a los ficheros de trades de cada estrategia en `full/{timeframe}/trades/`, manteniéndolos siempre al día sin necesidad de re-ejecutar el backtest histórico completo.

```
20:30 → stock_data (PostgreSQL) → stock_data_filtered (MV refresh)
02:00 → full_dataset.parquet + tickers/ + pending_candles_*.parquet
04:00 → full/{5m,15m}/trades/{strategy}/*_trades.parquet  (append)
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
