COMPOSE  = docker compose
WORKERS ?= 1
PROCS   ?= 4
CONCUR  ?= 10

.PHONY: build start stop restart logs ps shell test clean pipeline pipeline-sample clean-charts reload-scheduler dashboard tqqq-dataset tqqq-dataset-5m tqqq-dataset-1h tqqq-walkforward tqqq-walkforward-5m tqqq-walkforward-1h

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

## Delete temporary chart HTML files from /tmp
## Use  make clean-charts FORCE=1  to skip confirmation
clean-charts:
	@bash scripts/clean_tmp_charts.sh $(FORCE)

## Build TQQQ dataset (last 5 years) → backtest_dataset/INDICES/TQQQ/{5m,1h}/tqqq_full_dataset.parquet
tqqq-dataset:
	.venv/bin/python -m scripts.build_tqqq_dataset

tqqq-dataset-5m:
	.venv/bin/python -m scripts.build_tqqq_dataset --timeframe 5m

tqqq-dataset-1h:
	.venv/bin/python -m scripts.build_tqqq_dataset --timeframe 1h

## Build TQQQ walk-forward folds → backtest_dataset/INDICES/TQQQ/walkforward/{5m,1h}/fold_{1,2,3}/
tqqq-walkforward:
	.venv/bin/python -m scripts.build_tqqq_walkforward_dataset

tqqq-walkforward-5m:
	.venv/bin/python -m scripts.build_tqqq_walkforward_dataset --timeframe 5m

tqqq-walkforward-1h:
	.venv/bin/python -m scripts.build_tqqq_walkforward_dataset --timeframe 1h
