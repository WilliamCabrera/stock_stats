import logging
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from app.config import get_settings
from app.api.routes import router
from app.api.debug_routes import router as debug_router
from app.api.smallcaps_routes import router as smallcaps_router
from app.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger.info("Backtester API starting up")
    yield
    logger.info("Backtester API shutting down")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Backtester API",
        description="Async backtesting engine — vectorbt + TA + pandas",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request / response logging middleware ─────────────────────────────────
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            elapsed = (time.perf_counter() - start) * 1000
            logger.exception(
                "Unhandled exception during request",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "elapsed_ms": round(elapsed, 1),
                },
            )
            return JSONResponse(status_code=500, content={"detail": "Internal server error"})

        elapsed = (time.perf_counter() - start) * 1000
        level = logging.WARNING if response.status_code >= 400 else logging.INFO
        logger.log(
            level,
            "%s %s → %s  (%.1f ms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "elapsed_ms": round(elapsed, 1),
            },
        )
        return response

    app.include_router(router, prefix="/api/v1")
    app.include_router(debug_router, prefix="/api/v1")
    app.include_router(smallcaps_router, prefix="/api/v1")

    @app.get("/health", tags=["health"])
    async def health():
        return {"status": "ok"}

    return app


app = create_app()
