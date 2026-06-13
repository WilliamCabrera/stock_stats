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

### `scripts/split_dataset_by_date.py`
Splits `full_dataset.parquet` into one `.parquet` file per trading date. Used by the iterative backtest engine (`run_backtest`) to read data day by day.

Output: `backtest_dataset/full/{timeframe}/dates/{YYYY_MM_DD}.parquet`

```bash
# Both timeframes (5m and 15m)
python -m scripts.split_dataset_by_date

# Single timeframe
python -m scripts.split_dataset_by_date --timeframe 5m
python -m scripts.split_dataset_by_date --timeframe 15m
```

---

### `scripts/split_walkforward_by_date.py`
Splits each fold's `in_sample.parquet` and `out_of_sample.parquet` into one `.parquet` file per trading date.

Output:
- `backtest_dataset/walkforward/{timeframe}/fold_{n}/dates_IS/{YYYY_MM_DD}.parquet`
- `backtest_dataset/walkforward/{timeframe}/fold_{n}/dates_OOS/{YYYY_MM_DD}.parquet`

```bash
# All timeframes and all folds
python -m scripts.split_walkforward_by_date

# Single timeframe
python -m scripts.split_walkforward_by_date --timeframe 5m
python -m scripts.split_walkforward_by_date --timeframe 15m

# Single fold
python -m scripts.split_walkforward_by_date --timeframe 5m --fold 2
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

### `scripts/build_tqqq_dataset.py`
Builds the TQQQ index dataset for the last 5 years in 5m and 1h timeframes. Candles are fetched from Massive via `fetch_candles` in **1-month batches** to avoid overloading responses and losing data, then merged, deduplicated and written to:

```
backtest_dataset/INDICES/TQQQ/5m/tqqq_full_dataset.parquet
backtest_dataset/INDICES/TQQQ/1h/tqqq_full_dataset.parquet
```

```bash
# Both timeframes (5m and 1h)
make tqqq-dataset
python -m scripts.build_tqqq_dataset

# Single timeframe
make tqqq-dataset-5m
make tqqq-dataset-1h
python -m scripts.build_tqqq_dataset --timeframe 5m

# Custom date range
python -m scripts.build_tqqq_dataset --timeframe 1h --from 2023-01-01 --to 2024-12-31
```

Columns: `ticker, date, date_str, open, high, low, close, volume` (session 04:00–20:00 ET, zstd compression).

---

### `scripts/build_tqqq_walkforward_dataset.py`
Slices the full TQQQ dataset (no API re-fetch) into walk-forward folds with the same structure as `backtest_dataset/walkforward`. Windows keep the stock walkforward proportions (IS = 2 × OOS, slide = OOS) but scaled to **IS 24M / OOS 12M / slide 12M** so 3 folds cover the full 5-year range:

```
Fold 1: IS [d0,      d0+24M)   OOS [d0+24M, d0+36M)
Fold 2: IS [d0+12M,  d0+36M)   OOS [d0+36M, d0+48M)
Fold 3: IS [d0+24M,  d0+48M)   OOS [d0+48M, d0+60M)
```

Output:

```
backtest_dataset/INDICES/TQQQ/walkforward/{5m,1h}/fold_{1,2,3}/in_sample.parquet
backtest_dataset/INDICES/TQQQ/walkforward/{5m,1h}/fold_{1,2,3}/out_of_sample.parquet
```

```bash
# Both timeframes (5m and 1h)
make tqqq-walkforward
python -m scripts.build_tqqq_walkforward_dataset

# Single timeframe
make tqqq-walkforward-5m
make tqqq-walkforward-1h
python -m scripts.build_tqqq_walkforward_dataset --timeframe 5m

# Custom windows (months)
python -m scripts.build_tqqq_walkforward_dataset --is-months 12 --oos-months 6 --slide-months 6
```

Requires `tqqq_full_dataset.parquet` (run `make tqqq-dataset` first).

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
Lee `stock_data_filtered` para encontrar los ticker-días nuevos (fecha > última fecha en el parquet). Para cada uno fetchea las velas de 5m y 15m desde Massive, calcula los indicadores (ATR, VWAP, RVOL, SMAs, Donchian) y actualiza cinco destinos: `full_dataset.parquet`, el fichero por ticker en `tickers/`, los ficheros por fecha en `dates/` (`YYYY_MM_DD.parquet`), y los ficheros temporales `pending_candles_5m.parquet` / `pending_candles_15m.parquet`.

**04:00 — `incremental-backtest`**
Lee `pending_candles_5m.parquet` y `pending_candles_15m.parquet` generados en el paso anterior y corre todas las estrategias (`backside_short_lower_low`, `short_push_exhaustion`, `gap_crap_strategy`) sobre esos datos. Los trades resultantes se agregan (append) a los ficheros de trades de cada estrategia en `full/{timeframe}/trades/`, manteniéndolos siempre al día sin necesidad de re-ejecutar el backtest histórico completo.

```
20:30 → stock_data (PostgreSQL) → stock_data_filtered (MV refresh)
02:00 → full_dataset.parquet + tickers/ + dates/ + pending_candles_*.parquet
04:00 → full/{5m,15m}/trades/{strategy}/*_trades.parquet  (append)
```

---

## Despliegue y actualización de contenedores

Los scripts se montan como volumen (`.:/app`), por lo que los cambios en archivos `.py` se reflejan inmediatamente sin necesidad de rebuild. Sin embargo, **Ofelia** lee los labels del contenedor `pipeline` al arrancar, por lo que cualquier cambio en `docker-compose.yml` requiere recrear el contenedor.

### Actualizar solo scripts (sin cambios en dependencias ni `docker-compose.yml`)

No se requiere ninguna acción — el siguiente cronjob ya usará el código nuevo.

### Actualizar tras cambios en `docker-compose.yml` (labels de Ofelia, variables de entorno, recursos)

```bash
docker compose up -d --force-recreate pipeline
```

Esto recrea el contenedor con la nueva configuración sin reconstruir la imagen.

### Actualizar tras cambios en `Dockerfile` o `requirements.txt`

```bash
docker compose up -d --build pipeline
```

Esto reconstruye la imagen e inicia el contenedor con ella.

### Reiniciar todo el stack

```bash
docker compose down && docker compose up -d
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
