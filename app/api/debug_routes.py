"""
Debug endpoints for verifying the error-logging pipeline.

┌─────────────────────────────────────────────┬─────────────────────────┐
│ Endpoint                                    │ Destination             │
├─────────────────────────────────────────────┼─────────────────────────┤
│ GET /debug/error/unhandled                  │ errors.log  (middleware)│
│ GET /debug/error/logged                     │ errors.log  (direct)    │
│ GET /debug/error/caught                     │ errors.log  (exc_info)  │
│ GET /debug/warn/http-404                    │ app.log only (WARNING)  │
└─────────────────────────────────────────────┴─────────────────────────┘

Usage:
    curl http://localhost:8000/api/v1/debug/error/unhandled
    tail -f logs/errors.log | python -m json.tool
"""
import logging

from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/debug", tags=["debug"])


@router.get("/error/unhandled")
async def trigger_unhandled_exception():
    """
    Raises a bare RuntimeError — NOT caught in the route handler.

    Flow: route → middleware except Exception → logger.exception() → **errors.log**
    Response: 500 Internal Server Error
    """
    raise RuntimeError("Intentional unhandled RuntimeError — check errors.log")


@router.get("/error/logged")
async def trigger_logged_error():
    """
    Calls logger.error() directly with custom extra fields.

    Flow: logger.error() → JsonFormatter → **errors.log**
    Response: 200 OK (the error is only in the log, not in the HTTP response)
    """
    logger.error(
        "Intentional direct logger.error() call — debug endpoint",
        extra={"test": True, "source": "debug_routes", "endpoint": "/debug/error/logged"},
    )
    return {"logged": True, "level": "ERROR", "destination": "errors.log"}


@router.get("/error/caught")
async def trigger_caught_exception():
    """
    Catches a ZeroDivisionError and logs it with exc_info=True.

    Flow: except → logger.error(exc_info=True) → JsonFormatter (traceback) → **errors.log**
    Response: 200 OK with the caught error details
    """
    try:
        _ = 1 / 0
    except ZeroDivisionError:
        logger.error(
            "Caught ZeroDivisionError — logged with full traceback",
            exc_info=True,
            extra={"test": True, "operation": "1 / 0"},
        )
    return {"logged": True, "level": "ERROR", "exc_info": True, "destination": "errors.log"}


@router.get("/warn/http-404")
async def trigger_http_warning():
    """
    Raises HTTPException 404 — FastAPI handles it before the middleware sees an exception.

    Flow: HTTPException → FastAPI handler → 404 response →
          middleware logs as WARNING (status >= 400) → **app.log only** (NOT errors.log)

    Use this to confirm that 4xx errors do NOT appear in errors.log.
    """
    raise HTTPException(status_code=404, detail="Intentional 404 — check app.log (not errors.log)")
