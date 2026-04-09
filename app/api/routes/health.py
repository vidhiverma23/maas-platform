"""
Health & Metrics Endpoints
==========================
Operational endpoints for monitoring, alerting, and orchestration.

Endpoints:
- GET /health  — Liveness probe (is the process alive?)
- GET /ready   — Readiness probe (are dependencies connected?)
- GET /metrics — Prometheus-format metrics for scraping

Design:
- /health is cheap and fast — just returns 200
- /ready checks DB and Redis connectivity
- /metrics returns Prometheus exposition format
- Separate from business routes for clean monitoring config
"""

import time

from fastapi import APIRouter, Response
from fastapi.responses import JSONResponse

from app.api.dependencies import RedisClient, DBSession
from app.models.schemas import HealthResponse
from app.services.model_loader import model_loader
from app.utils.metrics import get_metrics
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["Operations"])

# Track process start time for uptime calculation
_start_time = time.monotonic()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    description="Returns 200 if the process is alive. Use for Kubernetes liveness probes.",
)
async def health_check() -> HealthResponse:
    """Basic liveness check — always returns healthy if the process is running."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        uptime_seconds=round(time.monotonic() - _start_time, 2),
        checks={"process": "ok"},
    )


@router.get(
    "/ready",
    response_model=HealthResponse,
    summary="Readiness probe",
    description="Checks database and Redis connectivity. Use for Kubernetes readiness probes.",
)
async def readiness_check(
    db: DBSession,
    redis_client: RedisClient,
) -> JSONResponse:
    """
    Deep health check — verifies all dependencies are reachable.
    Returns 503 if any dependency is unavailable.
    """
    checks: dict[str, str] = {}
    overall_status = "healthy"

    # Check PostgreSQL
    try:
        from sqlalchemy import text
        await db.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as e:
        checks["database"] = f"error: {str(e)[:100]}"
        overall_status = "unhealthy"
        logger.error("readiness_check_db_failed", error=str(e))

    # Check Redis
    try:
        await redis_client.ping()
        checks["redis"] = "ok"
    except Exception as e:
        checks["redis"] = f"error: {str(e)[:100]}"
        overall_status = "unhealthy"
        logger.error("readiness_check_redis_failed", error=str(e))

    # Model loader status
    cache_stats = model_loader.cache_stats()
    checks["model_cache"] = f"{cache_stats['cached_models']}/{cache_stats['max_capacity']} models loaded"

    status_code = 200 if overall_status == "healthy" else 503

    return JSONResponse(
        status_code=status_code,
        content={
            "status": overall_status,
            "version": "1.0.0",
            "uptime_seconds": round(time.monotonic() - _start_time, 2),
            "checks": checks,
        },
    )


@router.get(
    "/metrics",
    summary="Prometheus metrics",
    description="Returns metrics in Prometheus exposition format for scraping.",
)
async def metrics() -> Response:
    """Export Prometheus metrics for monitoring stack integration."""
    return Response(
        content=get_metrics(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
