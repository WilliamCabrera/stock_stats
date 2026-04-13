import logging

from celery import Celery
from celery.signals import task_failure, worker_init

from app.config import get_settings
from app.utils.logging_config import setup_logging

settings = get_settings()

# ── Initialise logging when the worker process starts ────────────────────────

@worker_init.connect
def _on_worker_init(**kwargs):
    setup_logging()
    logging.getLogger(__name__).info("Celery worker initialised")


# ── Log every task failure with full traceback ────────────────────────────────

@task_failure.connect
def _on_task_failure(task_id, exception, traceback, einfo, *args, **kwargs):
    logging.getLogger(__name__).error(
        "Task %s failed: %s",
        task_id,
        exception,
        exc_info=(type(exception), exception, traceback),
        extra={"task_id": task_id},
    )


celery = Celery(
    "backtester",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.tasks.backtest"],
)

celery.conf.update(
    # Serialisation
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    # Timeouts
    task_soft_time_limit=300,   # 5 min: raises SoftTimeLimitExceeded
    task_time_limit=360,        # 6 min: hard kill
    result_expires=3600,        # keep results 1 h in Redis
    # Multiprocessing: each worker process handles one CPU-bound task at a time
    worker_prefetch_multiplier=1,
    task_acks_late=True,        # ack only after completion (safe retry on crash)
    # Routing
    task_default_queue="default",
    task_routes={
        "app.tasks.backtest.*": {"queue": "backtest"},
    },
)
