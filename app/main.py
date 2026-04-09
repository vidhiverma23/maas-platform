"""
FastAPI Application Entry Point
================================
Application factory with lifespan management for startup/shutdown hooks.

Startup:
- Configure structured logging
- Initialize database tables (dev mode)
- Verify Redis connectivity
- Create model storage directory

Shutdown:
- Close database connection pool
- Close Redis connection pool
- Clear model cache

Design:
- Lifespan context manager (replaces deprecated on_event)
- CORS enabled for development
- Versioned API prefix (/api/v1)
- Auto-generated OpenAPI docs at /docs
"""

import os
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.utils.logger import setup_logging, get_logger


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.
    Code before `yield` runs at startup; after `yield` runs at shutdown.
    """
    # ── Startup ──────────────────────────────────────────────
    setup_logging()
    logger = get_logger("startup")

    settings = get_settings()
    logger.info(
        "application_starting",
        app_name=settings.app_name,
        env=settings.app_env,
        version=settings.app_version,
    )

    # Create model storage directory
    storage_path = Path(settings.model_storage_path)
    storage_path.mkdir(parents=True, exist_ok=True)
    logger.info("model_storage_initialized", path=str(storage_path))

    # Initialize database tables (dev only — production uses Alembic)
    if not settings.is_production:
        try:
            from app.database.connection import init_db
            await init_db()
            logger.info("database_tables_created")
        except Exception as e:
            logger.error("database_init_failed", error=str(e))

    # Verify Redis connectivity
    try:
        from app.api.dependencies import get_redis
        redis_client = await get_redis()
        await redis_client.ping()
        logger.info("redis_connected", url=settings.redis_url)
    except Exception as e:
        logger.warning("redis_connection_failed", error=str(e))

    logger.info("application_ready")

    yield  # ── Application runs here ──

    # ── Shutdown ─────────────────────────────────────────────
    logger = get_logger("shutdown")
    logger.info("application_shutting_down")

    # Clear model cache
    from app.services.model_loader import model_loader
    await model_loader.clear_cache()

    # Close Redis
    try:
        from app.api.dependencies import close_redis
        await close_redis()
        logger.info("redis_disconnected")
    except Exception as e:
        logger.warning("redis_close_failed", error=str(e))

    # Close database
    try:
        from app.database.connection import close_db
        await close_db()
        logger.info("database_disconnected")
    except Exception as e:
        logger.warning("database_close_failed", error=str(e))

    logger.info("application_stopped")


def create_app() -> FastAPI:
    """
    Application factory — creates and configures the FastAPI instance.
    """
    settings = get_settings()

    app = FastAPI(
        title="AI Model-as-a-Service (MaaS) Platform",
        description=(
            "A scalable backend system for uploading, registering, deploying, "
            "and serving machine learning models via REST APIs. "
            "Supports sklearn, PyTorch, and ONNX models."
        ),
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ── CORS Middleware ──────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if not settings.is_production else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Response-Time"],
    )

    # ── Request Logging Middleware ───────────────────────────
    from app.api.middleware import RequestLoggingMiddleware, register_exception_handlers
    app.add_middleware(RequestLoggingMiddleware)

    # ── Exception Handlers ───────────────────────────────────
    register_exception_handlers(app)

    # ── Route Registration ───────────────────────────────────
    from app.api.routes.health import router as health_router
    from app.api.routes.models import router as models_router
    from app.api.routes.inference import router as inference_router

    app.include_router(health_router)
    app.include_router(models_router)
    app.include_router(inference_router)

    # ── Root Endpoint ────────────────────────────────────────
    @app.get("/", tags=["Root"])
    async def root():
        return {
            "service": "AI Model-as-a-Service (MaaS) Platform",
            "version": settings.app_version,
            "docs": "/docs",
            "health": "/health",
            "metrics": "/metrics",
        }

    return app


# Create the application instance
app = create_app()
