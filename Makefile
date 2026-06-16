COMPOSE  = docker compose
WORKERS ?= 1
PROCS   ?= 4
CONCUR  ?= 10

.PHONY: build start stop restart logs ps shell test clean pipeline pipeline-sample clean-charts reload-scheduler dashboard signal-candle-dashboard \
        tqqq-dataset tqqq-dataset-5m tqqq-dataset-10m tqqq-dataset-1h tqqq-dataset-1d tqqq-walkforward tqqq-walkforward-5m tqqq-walkforward-10m tqqq-walkforward-1h tqqq-walkforward-1d \
        qqq-dataset qqq-dataset-5m qqq-dataset-10m qqq-dataset-1h qqq-dataset-1d qqq-walkforward qqq-walkforward-5m qqq-walkforward-10m qqq-walkforward-1h qqq-walkforward-1d \
        update-indices build-index build-index-wf \
        stock-universe stock-dataset stock-dataset-all stock-merge

## Build all images (no cache)
build:
	$(COMPOSE) build --no-cache

## Build incrementally (uses cache)
build-fast:
	$(COMPOSE) build

## Start all services in background
start:
	$(COMPOSE) up -d

## Start and scale workers  →  make start-workers WORKERS=3
start-workers:
	$(COMPOSE) up -d --scale worker=$(WORKERS)

## Stop all services (keep volumes)
stop:
	$(COMPOSE) down

## Stop and delete volumes
clean:
	$(COMPOSE) down -v --remove-orphans

## Restart all services
restart:
	$(COMPOSE) restart

## Restart a single service  →  make restart-api
restart-api:
	$(COMPOSE) restart api

restart-worker:
	$(COMPOSE) restart worker

restart-flower:
	$(COMPOSE) restart flower

## Rebuild and restart everything
rebuild:
	$(COMPOSE) down
	$(COMPOSE) build
	$(COMPOSE) up -d

## Show running containers
ps:
	$(COMPOSE) ps

## Follow logs for all services (Ctrl-C to exit)
logs:
	$(COMPOSE) logs -f

logs-api:
	$(COMPOSE) logs -f api

logs-worker:
	$(COMPOSE) logs -f worker

## Open a shell in the api container
shell:
	$(COMPOSE) exec api bash

## Quick health check
health:
	@curl -sf http://localhost:8000/health && echo " ✓ API is up" || echo " ✗ API is down"

## Run tests inside the api container
test:
	$(COMPOSE) exec api python -m pytest tests/ -v

## Run a specific test file  →  make test-file FILE=tests/utils/test_market_utils.py
test-file:
	$(COMPOSE) exec api python -m pytest $(FILE) -v


## Recreate pipeline + ofelia so Ofelia picks up updated labels from docker-compose
reload-scheduler:
	$(COMPOSE) up -d --force-recreate pipeline ofelia

pipeline-data-collection:
	docker exec -it backtester_api-api-1 python3 app/utils/pipeline_data_collection.py

pipeline-delisted:
	docker exec -it backtester_api-api-1 python3 app/utils/pipeline_delisted.py


## Launch backtest dashboard  →  make dashboard
## Optional: make dashboard STRAT=strategies/iterative/OOS/5m PORT=8502
STRAT ?= strategies
PORT  ?= 8501
dashboard:
	.venv/bin/streamlit run $(STRAT)/dashboard.py --server.port $(PORT)

## Launch signal candle size dashboard  →  make signal-candle-dashboard
signal-candle-dashboard:
	.venv/bin/streamlit run strategies/signal_candle_dashboard.py --server.port 8503
	.venv/bin/streamlit run $(STRAT)/dashboard.py --server.port $(PORT)

## Delete temporary chart HTML files from /tmp
## Use  make clean-charts FORCE=1  to skip confirmation
clean-charts:
	@bash scripts/clean_tmp_charts.sh $(FORCE)

## Build TQQQ dataset (last 5 years) → backtest_dataset/INDICES/TQQQ/{5m,1h}/tqqq_full_dataset.parquet
## Build TQQQ dataset (last 5 years) → backtest_dataset/INDICES/TQQQ/{5m,10m,1h,1d}/tqqq_full_dataset.parquet
tqqq-dataset:
	.venv/bin/python -m scripts.build_tqqq_dataset

tqqq-dataset-5m:
	.venv/bin/python -m scripts.build_tqqq_dataset --timeframe 5m

tqqq-dataset-10m:
	.venv/bin/python -m scripts.build_tqqq_dataset --timeframe 10m

tqqq-dataset-1h:
	.venv/bin/python -m scripts.build_tqqq_dataset --timeframe 1h

tqqq-dataset-1d:
	.venv/bin/python -m scripts.build_tqqq_dataset --timeframe 1d

## Build TQQQ walk-forward folds → backtest_dataset/INDICES/TQQQ/walkforward/{5m,10m,1h,1d}/fold_{1,2,3}/
tqqq-walkforward:
	.venv/bin/python -m scripts.build_tqqq_walkforward_dataset

tqqq-walkforward-5m:
	.venv/bin/python -m scripts.build_tqqq_walkforward_dataset --timeframe 5m

tqqq-walkforward-10m:
	.venv/bin/python -m scripts.build_tqqq_walkforward_dataset --timeframe 10m

tqqq-walkforward-1h:
	.venv/bin/python -m scripts.build_tqqq_walkforward_dataset --timeframe 1h

tqqq-walkforward-1d:
	.venv/bin/python -m scripts.build_tqqq_walkforward_dataset --timeframe 1d

## Incrementally update all INDICES datasets (TQQQ + QQQ) → same parquets as build scripts
update-indices:
	.venv/bin/python -m scripts.update_indices_dataset

## Generic build for any ticker  →  backtest_dataset/INDICES/{TICKER}/
## Usage: make build-index TICKER=SPY   (add TIMEFRAME=5m and/or YEARS=3 optionally)
TICKER   ?= SPY
TIMEFRAME ?=
YEARS    ?= 5
_TF_ARG  := $(if $(TIMEFRAME),--timeframe $(TIMEFRAME),)
_YR_ARG  := --years $(YEARS)

build-index:
	.venv/bin/python -m scripts.build_index_dataset --ticker $(TICKER) $(_TF_ARG) $(_YR_ARG)

## Generic walk-forward build for any ticker
## Usage: make build-index-wf TICKER=SPY   (add TIMEFRAME=5m, IS=24, OOS=12, SLIDE=12 optionally)
IS    ?= 24
OOS   ?= 12
SLIDE ?= 12

build-index-wf:
	.venv/bin/python -m scripts.build_index_walkforward_dataset --ticker $(TICKER) $(_TF_ARG) \
	    --is-months $(IS) --oos-months $(OOS) --slide-months $(SLIDE)

## Build QQQ dataset (last 5 years) → backtest_dataset/INDICES/QQQ/{5m,10m,1h,1d}/qqq_full_dataset.parquet
qqq-dataset:
	.venv/bin/python -m scripts.build_qqq_dataset

qqq-dataset-5m:
	.venv/bin/python -m scripts.build_qqq_dataset --timeframe 5m

qqq-dataset-10m:
	.venv/bin/python -m scripts.build_qqq_dataset --timeframe 10m

qqq-dataset-1h:
	.venv/bin/python -m scripts.build_qqq_dataset --timeframe 1h

qqq-dataset-1d:
	.venv/bin/python -m scripts.build_qqq_dataset --timeframe 1d

## Build QQQ walk-forward folds → backtest_dataset/INDICES/QQQ/walkforward/{5m,10m,1h,1d}/fold_{1,2,3}/
qqq-walkforward:
	.venv/bin/python -m scripts.build_qqq_walkforward_dataset

qqq-walkforward-5m:
	.venv/bin/python -m scripts.build_qqq_walkforward_dataset --timeframe 5m

qqq-walkforward-10m:
	.venv/bin/python -m scripts.build_qqq_walkforward_dataset --timeframe 10m

qqq-walkforward-1h:
	.venv/bin/python -m scripts.build_qqq_walkforward_dataset --timeframe 1h

qqq-walkforward-1d:
	.venv/bin/python -m scripts.build_qqq_walkforward_dataset --timeframe 1d

## ──────────────────────────────────────────────────────────────────────────
## Full stock-market dataset (every NASDAQ/NYSE/etc ticker, active + delisted)
## Pipeline:  stock-universe  →  stock-dataset (×NUM_SHARDS)  →  stock-merge
## Vars: YEARS=5  CONCUR=50  STEP=1  NUM_SHARDS=1  SHARD=0
## ──────────────────────────────────────────────────────────────────────────
NUM_SHARDS ?= 1
SHARD      ?= 0
STEP       ?= 1
SCONCUR    ?= 50

## Step 1 — build the clean ticker universe → backtest_dataset/UNIVERSE/stock_universe.parquet
stock-universe:
	.venv/bin/python -m scripts.build_stock_universe

## Steps 2-4 — build ONE shard → backtest_dataset/STOCKS/shards/shard_<SHARD>_of_<NUM_SHARDS>.parquet
## Usage: make stock-dataset NUM_SHARDS=8 SHARD=0   (run once per shard, one terminal each)
stock-dataset:
	.venv/bin/python -m scripts.build_stock_dataset \
	    --num-shards $(NUM_SHARDS) --shard $(SHARD) \
	    --years $(YEARS) --concurrency $(SCONCUR) --marketcap-step $(STEP)

## Launch ALL shards in parallel (background) → logs in /tmp/stock_shard_*.log; blocks until done
## Usage: make stock-dataset-all NUM_SHARDS=8
stock-dataset-all:
	@for i in $$(seq 0 $$(($(NUM_SHARDS)-1))); do \
	    echo "→ shard $$i / $(NUM_SHARDS)"; \
	    .venv/bin/python -m scripts.build_stock_dataset --num-shards $(NUM_SHARDS) --shard $$i \
	        --years $(YEARS) --concurrency $(SCONCUR) --marketcap-step $(STEP) \
	        > /tmp/stock_shard_$$i.log 2>&1 & \
	done; \
	echo "Launched $(NUM_SHARDS) shards. Tail logs:  tail -f /tmp/stock_shard_*.log"; \
	wait; \
	echo "All shards finished."

## Final — merge all shards → backtest_dataset/STOCKS/stock_dataset.parquet
stock-merge:
	.venv/bin/python -m scripts.merge_stock_dataset
