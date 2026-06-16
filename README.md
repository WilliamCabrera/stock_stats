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
Builds the TQQQ index dataset for the last 5 years in **5m, 10m, 1h and 1d** timeframes. Candles are fetched from Massive via `fetch_candles` in 1-month batches, indicators are computed, and cross-timeframe columns are joined before writing:

```
backtest_dataset/INDICES/TQQQ/1d/tqqq_full_dataset.parquet
backtest_dataset/INDICES/TQQQ/1h/tqqq_full_dataset.parquet
backtest_dataset/INDICES/TQQQ/5m/tqqq_full_dataset.parquet
backtest_dataset/INDICES/TQQQ/10m/tqqq_full_dataset.parquet
```

Build order: `1d → 1h → 5m / 10m` (each step joins columns from the previous one).

| Timeframe | Extra columns |
|-----------|--------------|
| 1d | `sma_9/20/50/200`, `atr_14`, `daily_range`, `daily_range_ma10` |
| 1h | `sma_9/20/50/200`, `atr_14`, `daily_range_ma10` |
| 5m / 10m | `sma_9/20/50/200`, `atr_14`, `daily_range_ma10`, `h1_9am_high`, `h1_9am_low` |

```bash
# All timeframes (1d → 1h → 5m → 10m)
make tqqq-dataset
python -m scripts.build_tqqq_dataset

# Single timeframe
make tqqq-dataset-5m
make tqqq-dataset-10m
make tqqq-dataset-1h
make tqqq-dataset-1d
python -m scripts.build_tqqq_dataset --timeframe 5m

# Custom date range
python -m scripts.build_tqqq_dataset --from 2023-01-01 --to 2024-12-31
```

---

### `scripts/build_qqq_dataset.py`
Builds the QQQ index dataset for the last 5 years in **5m, 10m, 1h and 1d** timeframes. Same structure and indicators as the TQQQ dataset.

```
backtest_dataset/INDICES/QQQ/1d/qqq_full_dataset.parquet
backtest_dataset/INDICES/QQQ/1h/qqq_full_dataset.parquet
backtest_dataset/INDICES/QQQ/5m/qqq_full_dataset.parquet
backtest_dataset/INDICES/QQQ/10m/qqq_full_dataset.parquet
```

```bash
# All timeframes
make qqq-dataset
python -m scripts.build_qqq_dataset

# Single timeframe
make qqq-dataset-5m
make qqq-dataset-10m
make qqq-dataset-1h
make qqq-dataset-1d

# Custom date range
python -m scripts.build_qqq_dataset --from 2023-01-01 --to 2024-12-31
```

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

### `scripts/build_qqq_walkforward_dataset.py`

Slices the QQQ full dataset (no API re-fetch) into walk-forward folds with the same window structure as the TQQQ script: **IS 24M / OOS 12M / slide 12M**, producing 3 folds. Also writes a `final_oos.parquet` with data after the last fold's OOS end.

Output:

```text
backtest_dataset/INDICES/QQQ/walkforward/{5m,10m,1h,1d}/
    fold_1/in_sample.parquet
    fold_1/out_of_sample.parquet
    fold_2/...
    fold_3/...
    final_oos.parquet
```

```bash
# All timeframes
make qqq-walkforward
python -m scripts.build_qqq_walkforward_dataset

# Single timeframe
make qqq-walkforward-5m
make qqq-walkforward-10m
make qqq-walkforward-1h
make qqq-walkforward-1d

# Custom windows (months)
python -m scripts.build_qqq_walkforward_dataset --is-months 12 --oos-months 6 --slide-months 6
```

Requires `qqq_full_dataset.parquet` (run `make qqq-dataset` first).

---

### `scripts/build_index_dataset.py`

Versión genérica de los scripts `build_tqqq_dataset.py` / `build_qqq_dataset.py`. Acepta cualquier ticker como parámetro y construye el dataset completo con todos los indicadores en los cuatro timeframes.

Output: `backtest_dataset/INDICES/{TICKER}/{tf}/{ticker_lower}_full_dataset.parquet`

Indicadores por timeframe (mismos que TQQQ / QQQ):

| Timeframe | Columnas extra |
|---|---|
| 1d | `sma_9/20/50/100/200`, `atr_14`, `daily_range`, `daily_range_ma10` |
| 1h | `sma_9/20/50/200`, `atr_14`, `daily_range_ma10` |
| 5m / 10m | `sma_9/20/50/200`, `atr_14`, `daily_range_ma10`, `h1_9am_high`, `h1_9am_low` |

```bash
# Todos los timeframes (1d → 1h → 5m → 10m)
python -m scripts.build_index_dataset --ticker SPY
make build-index TICKER=SPY

# Timeframe específico
python -m scripts.build_index_dataset --ticker AAPL --timeframe 5m
make build-index TICKER=AAPL TIMEFRAME=5m

# Rango de fechas personalizado
python -m scripts.build_index_dataset --ticker NVDA --from 2022-01-01 --to 2024-12-31

# Número de años de historia (default: 5)
make build-index TICKER=MSFT YEARS=3
python -m scripts.build_index_dataset --ticker MSFT --years 3
```

---

### `scripts/build_index_walkforward_dataset.py`

Versión genérica de `build_tqqq_walkforward_dataset.py` / `build_qqq_walkforward_dataset.py`. Parte el full dataset de cualquier ticker en folds IS/OOS sin re-fetchear datos.

Output: `backtest_dataset/INDICES/{TICKER}/walkforward/{tf}/fold_{1..N}/{in_sample,out_of_sample}.parquet` + `final_oos.parquet`

```bash
# Todos los timeframes, ventanas por defecto (IS=24M / OOS=12M / slide=12M)
python -m scripts.build_index_walkforward_dataset --ticker SPY
make build-index-wf TICKER=SPY

# Timeframe específico
python -m scripts.build_index_walkforward_dataset --ticker AAPL --timeframe 1d
make build-index-wf TICKER=AAPL TIMEFRAME=1d

# Ventanas personalizadas
python -m scripts.build_index_walkforward_dataset --ticker QQQ --is-months 12 --oos-months 6 --slide-months 6
make build-index-wf TICKER=QQQ IS=12 OOS=6 SLIDE=6
```

Requiere el full dataset del ticker (ejecutar `build_index_dataset.py` primero).

---

### `scripts/update_indices_dataset.py`

Incremental updater for INDICES datasets — mirrors `update_full_dataset.py` but for `backtest_dataset/INDICES/`. Reads the latest `date_str` from the existing 1d parquet, fetches from that date + 1 day to today, recomputes all indicators, and upserts the new rows. Updates timeframes in dependency order (`1d → 1h → 5m / 10m`) so cross-timeframe joins are always consistent.

Registered indices: **TQQQ**, **QQQ** (add new ones to the `INDICES` dict in the script).

```bash
# All registered indices, all timeframes
make update-indices
python -m scripts.update_indices_dataset

# Single index
python -m scripts.update_indices_dataset --ticker TQQQ
python -m scripts.update_indices_dataset --ticker QQQ

# Single timeframe
python -m scripts.update_indices_dataset --ticker TQQQ --timeframe 5m

# Override start date
python -m scripts.update_indices_dataset --from 2025-01-01

# Dry run — print plan without writing
python -m scripts.update_indices_dataset --dry-run
```

---

### Full stock-market dataset (`build_stock_universe` → `build_stock_dataset` → `merge_stock_dataset`)

Builds a daily dataset for **every US stock ticker** (NASDAQ/NYSE/etc, active **and** delisted) from Massive. Three stages:

**1. `scripts/build_stock_universe.py`** — fetches all tickers (`v3/reference/tickers`, paginated, `active=true` + `active=false`), removes duplicates and any symbol with special characters (`ABC` ✓, `APH.WD` ✗, `AHW$WC` ✗).

Output: `backtest_dataset/UNIVERSE/stock_universe.parquet` (+ `.csv`)

```bash
make stock-universe
python -m scripts.build_stock_universe
python -m scripts.build_stock_universe --active true   # listed only
```

**2. `scripts/build_stock_dataset.py`** — for each ticker fetches ~5y of adjusted **1D** candles (1 request/ticker), computes **SMA 9/20/50/100/200** and **ATR 14**, and fetches **market_cap + float** per trading day (`v3/reference/tickers/<ticker>?date=...`). Step 4 is the bottleneck (>1M requests across the universe), so the job is **sharded** (one parquet per shard), **resumable** (checkpoints every `--flush-every` tickers; restart skips tickers already present), and uses **async concurrency** with retry/backoff. `--marketcap-step N` samples the reference endpoint every N trading days and recomputes `market_cap = close × float` for in-between days (shares change rarely), cutting requests ~N×.

Output: `backtest_dataset/STOCKS/shards/shard_<I>_of_<N>.parquet`

```bash
# Single process (slow but simple)
make stock-dataset
python -m scripts.build_stock_dataset

# Parallel — one terminal per shard (recommended)
make stock-dataset NUM_SHARDS=8 SHARD=0    # ... up to SHARD=7
python -m scripts.build_stock_dataset --num-shards 8 --shard 0

# Launch all shards in parallel in the background (logs → /tmp/stock_shard_*.log)
make stock-dataset-all NUM_SHARDS=8

# Tuning / faster passes
python -m scripts.build_stock_dataset --concurrency 50 --ticker-concurrency 8
python -m scripts.build_stock_dataset --marketcap-step 5     # sample weekly
python -m scripts.build_stock_dataset --skip-marketcap       # candles + indicators only
python -m scripts.build_stock_dataset --tickers AAPL,MSFT --limit 5   # smoke test
```

Make vars: `YEARS=5 SCONCUR=50 STEP=1 NUM_SHARDS=1 SHARD=0`.

**3. `scripts/merge_stock_dataset.py`** — concatenates all shards into one file, deduplicated on `(ticker, date_str)`.

Output: `backtest_dataset/STOCKS/stock_dataset.parquet`

```bash
make stock-merge
python -m scripts.merge_stock_dataset
```

Columns: `ticker, date, date_str, open, high, low, close, volume, sma_9, sma_20, sma_50, sma_100, sma_200, atr_14, market_cap, float, shares_outstanding`.

---

### Low-float session dataset (`filter_low_float` → `build_low_float_dataset` → merge)

Builds a per-day, per-session dataset (pre-market / market-hours / after-hours) from **1-minute unadjusted** candles for the low-float tickers. Three stages:

**1. `scripts/filter_low_float.py`** — reads `stock_dataset.parquet` and keeps each ticker's most recent non-null float; selects those below the threshold.

Output: `backtest_dataset/STOCKS/low_float_tickers.parquet` (columns `ticker, float, market_cap`)

```bash
python -m scripts.filter_low_float                  # default < 20,000,000
python -m scripts.filter_low_float --max-float 10e6
```

**2. `scripts/build_low_float_dataset.py`** — for each ticker fetches 1m unadjusted candles (04:00–20:00 ET), aggregates per-day sessions via `process_minute_bars` (a corrected fork of `market_utils.process_data_minutes`: prices kept at 6 decimals for sub-penny names, NaN instead of `-1` for empty sessions), reuses `_apply_gap_logic` for `previous_close`/gap/range + split adjustment, and merges `market_cap`/`stock_float` from the stock dataset. **Sharded, resumable, async** (same harness as `build_stock_dataset`). A full 1m fetch per ticker is paginated and fully merged before any per-day aggregation, so a day's candles are never split across responses.

Output: `backtest_dataset/LOW_FLOAT/shards/shard_<I>_of_<N>.parquet`

```bash
# Single process / smoke test
python -m scripts.build_low_float_dataset --limit 10 --num-shards 1
python -m scripts.build_low_float_dataset --tickers JRSH,ZNB --from 2026-06-15 --to 2026-06-15

# Parallel — one terminal per shard
make low-float NUM_SHARDS=8 SHARD=0    # ... up to SHARD=7
python -m scripts.build_low_float_dataset --num-shards 8 --shard 0

# Launch all shards in parallel in the background (logs → /tmp/lowfloat_shard_*.log)
make low-float-all NUM_SHARDS=8        # bump TCONCUR=10 if the API plan allows
```

Make vars: `YEARS=5 TCONCUR=6 NUM_SHARDS=1 SHARD=0`.

**3. Merge** — concatenates the shards (dedup on `ticker, date_str`).

Output: `backtest_dataset/LOW_FLOAT/low_float_dataset.parquet`

```bash
make low-float-merge
python -m scripts.merge_stock_dataset \
    --shards-dir backtest_dataset/LOW_FLOAT/shards \
    --out backtest_dataset/LOW_FLOAT/low_float_dataset.parquet
```

Columns: `ticker, date_str, gap, gap_perc, daily_range, day_range_perc, previous_close, open, high, low, close, volume, premarket_volume, market_hours_volume, high_pm, low_pm, pm_open, highest_in_pm, high_pm_time, high_mh, ah_open, ah_close, ah_high, ah_low, ah_range, ah_range_perc, ah_volume, market_cap, stock_float, split_date_str, split_adjust_factor, time`.

Notes: prices are **unadjusted** (6-decimal precision); `volume` is kept fractional as returned by the source; the day's `high`/`low` cover pre-market + market-hours (after-hours lives only in the `ah_*` columns); `market_cap`/`stock_float` are forward/back-filled per ticker.

---

## Iterative strategies

### `strategies/iterative/small_caps.py`

Intraday strategies for small-cap gappers. Each function receives one day of 5m or 15m candles for one ticker and returns a `pd.DataFrame` of trades. Registered strategies:

| Function | Signal |
|---|---|
| `backside_short_lower_low_fix_stop_iterative` | Red bar breaking the low of the prior green bar, above VWAP |
| `gap_crap_iterative` | Short at 9:25 close if gapped ≥ 40 % over prior close |
| `short_push_exhaustion_iterative` | Red bar with dominant topping tail, vol surge, above VWAP |
| `push_rejection_iterative` | Red bar crossing VWAP downward with body > bottom tail |

### `strategies/iterative/trend_following.py`

Trend-following strategies for indices and large caps.

#### `ema100_trend_follower_iterative`

Classic turtle-style breakout with ATR-trailing stop and pyramiding:

1. **EMA100 filter** — long bias when price is above EMA100, short bias below.
2. **Breakout entry** — long on new 20-day high, short on new 20-day low.
3. **Pyramid** — adds up to 3 units as price advances 0.5 × ATR(14) beyond each entry.
4. **ATR trailing stop** — 2.0 × ATR(14) below the running peak (long) or above the trough (short); updated on every bar.
5. **Next-bar execution** — all entries and exits fire at the open of the bar after the signal.

Historical daily indicators (EMA100, ATR14, 20-day high/low) are computed from the ticker's full intraday parquet using only prior days (no lookahead). Each pyramid unit is recorded as a separate trade.

```python
from strategies.iterative.trend_following import ema100_trend_follower_iterative
```

---

## Automated daily pipeline

Four jobs run automatically every day via **Ofelia** (the Docker-based cron scheduler). They execute inside the `pipeline` container and are defined in `docker-compose.yml`.

| Hora (ET) | Job                       | Descripción                                             |
|-----------|---------------------------|---------------------------------------------------------|
| **20:30** | `incremental-pipeline`    | Ingesta de datos de mercado en PostgreSQL               |
| **22:00** | `update-indices-dataset`  | Actualiza los parquets de TQQQ y QQQ (todos los TFs)   |
| **02:00** | `update-full-dataset`     | Fetchea velas 5m/15m de small caps y actualiza parquets |
| **04:00** | `incremental-backtest`    | Corre todas las estrategias sobre datos nuevos          |

**20:30 — `incremental-pipeline`**
Detecta los tickers con actividad reciente (gappers, movers), fetchea sus datos OHLCV del día desde Massive.com, los inserta/actualiza en la tabla `stock_data` de PostgreSQL y refresca la vista materializada `stock_data_filtered` para que solo queden los tickers que cumplen los filtros de calidad (gap > 40 %, prev-close > $0.10, etc.).

**22:00 — `update-indices-dataset`**
Ejecuta `scripts/update_indices_dataset.py` para todos los índices registrados (TQQQ, QQQ). Para cada uno detecta la última fecha en el parquet 1d, fetchea los datos nuevos con warmup de 20 días calendario, recalcula los indicadores (SMA 9/20/50/200, ATR14, daily_range_ma10, h1_9am_high/low) y actualiza los cuatro timeframes en orden de dependencia (1d → 1h → 5m → 10m).

**02:00 — `update-full-dataset`**
Lee `stock_data_filtered` para encontrar los ticker-días nuevos (fecha > última fecha en el parquet). Para cada uno fetchea las velas de 5m y 15m desde Massive, calcula los indicadores (ATR, VWAP, RVOL, SMAs, Donchian) y actualiza cinco destinos: `full_dataset.parquet`, el fichero por ticker en `tickers/`, los ficheros por fecha en `dates/` (`YYYY_MM_DD.parquet`), y los ficheros temporales `pending_candles_5m.parquet` / `pending_candles_15m.parquet`.

**04:00 — `incremental-backtest`**
Lee `pending_candles_5m.parquet` y `pending_candles_15m.parquet` generados en el paso anterior y corre todas las estrategias registradas en `strategies_registry.py` sobre esos datos. Los trades resultantes se agregan a los ficheros de trades de cada estrategia en `UP-TO-DATE/{timeframe}/{strategy}/`, manteniéndolos siempre al día sin necesidad de re-ejecutar el backtest histórico completo.

```text
20:30 → stock_data (PostgreSQL) → stock_data_filtered (MV refresh)
22:00 → INDICES/{TQQQ,QQQ}/{1d,1h,5m,10m}/*_full_dataset.parquet  (upsert)
02:00 → full_dataset.parquet + tickers/ + dates/ + pending_candles_*.parquet
04:00 → UP-TO-DATE/{5m,15m}/{strategy}/*.parquet  (append)
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

```text
backtest_dataset/
├── full/                              ← small caps (5m, 15m)
│   ├── 5m/
│   │   ├── full_dataset.parquet       ← all tickers merged
│   │   ├── ticker_row_counts.parquet
│   │   ├── tickers/                   ← one file per ticker
│   │   │   ├── AAPL.parquet
│   │   │   └── ...
│   │   └── dates/                     ← one file per trading day
│   │       ├── 2024_01_02.parquet
│   │       └── ...
│   └── 15m/
│       └── ...
├── walkforward/                       ← small caps walk-forward folds
│   ├── 5m/
│   │   ├── fold_1/
│   │   │   ├── dates_IS/
│   │   │   └── dates_OOS/
│   │   ├── fold_2/
│   │   └── fold_3/
│   └── 15m/
│       └── ...
├── INDICES/                           ← index / large-cap datasets
│   ├── TQQQ/
│   │   ├── 1d/tqqq_full_dataset.parquet
│   │   ├── 1h/tqqq_full_dataset.parquet
│   │   ├── 5m/tqqq_full_dataset.parquet
│   │   ├── 10m/tqqq_full_dataset.parquet
│   │   └── walkforward/
│   │       └── {5m,10m,1h,1d}/fold_{1,2,3}/{in_sample,out_of_sample}.parquet
│   └── QQQ/
│       ├── 1d/qqq_full_dataset.parquet
│       ├── 1h/qqq_full_dataset.parquet
│       ├── 5m/qqq_full_dataset.parquet
│       ├── 10m/qqq_full_dataset.parquet
│       └── walkforward/
│           └── {5m,10m,1h,1d}/fold_{1,2,3}/{in_sample,out_of_sample}.parquet
├── pending_candles_5m.parquet         ← new small-cap days queued for backtest
└── pending_candles_15m.parquet
```

### INDICES columns

All INDICES parquets share a base set of columns. Cross-timeframe columns are joined at build/update time:

| Column | Timeframes | Source |
|---|---|---|
| `ticker, date, date_str, open, high, low, close, volume` | all | Massive API |
| `sma_9, sma_20, sma_50, sma_200` | all | computed per-timeframe |
| `sma_100` | 1d only | computed on daily close |
| `atr_14` | all | computed per-timeframe |
| `daily_range, daily_range_ma10` | all | joined from 1d |
| `h1_9am_high, h1_9am_low` | 5m, 10m | joined from 1h (9:00 bar) |
