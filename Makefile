COMPOSE  = docker compose
WORKERS ?= 1
PROCS   ?= 4
CONCUR  ?= 10

.PHONY: build start stop restart logs ps shell test clean pipeline pipeline-sample clean-charts reload-scheduler

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


## Delete temporary chart HTML files from /tmp
## Use  make clean-charts FORCE=1  to skip confirmation
clean-charts:
	@bash scripts/clean_tmp_charts.sh $(FORCE)
