"""
Request Middleware
==================
Cross-cutting concerns that apply to every request:
- Request ID generation and propagation
- Request/response logging with latency tracking
- Global exception handling (MaaSException → JSON responses)
- Rate limiting enforcement

Design:
- Middleware ordering matters: outermost runs first
- structlog contextvars for request-scoped log context
- Exception handlers registered on the FastAPI app, not as middleware
"""

import time
import uuid

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from app.utils.exceptions import MaaSException
from app.utils.logger import get_logger
from app.utils.metrics import ACTIVE_REQUESTS, REQUEST_COUNT, REQUEST_LATENCY

logger = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware that:
    1. Assigns a unique request ID (X-Request-ID header)
    2. Binds request context to structlog for all downstream logs
    3. Records request latency in Prometheus histogram
    4. Logs request/response summary
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Generate or propagate request ID
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        client_ip = request.client.host if request.client else "unknown"

        # Bind request context to structlog for this request scope
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            client_ip=client_ip,
            method=request.method,
            path=request.url.path,
        )

        # Track active requests
        ACTIVE_REQUESTS.inc()
        start_time = time.monotonic()

        try:
            response = await call_next(request)

            # Calculate latency
            latency = time.monotonic() - start_time

            # Record metrics
            endpoint = request.url.path
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=endpoint,
                status_code=response.status_code,
            ).inc()
            REQUEST_LATENCY.labels(
                method=request.method,
                endpoint=endpoint,
            ).observe(latency)

            # Add response headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{latency * 1000:.2f}ms"

            # Log request summary
            logger.info(
                "request_completed",
                status_code=response.status_code,
                latency_ms=round(latency * 1000, 2),
            )

            return response

        except Exception as exc:
            latency = time.monotonic() - start_time
            logger.error(
                "request_failed",
                error=str(exc),
                latency_ms=round(latency * 1000, 2),
            )
            raise
        finally:
            ACTIVE_REQUESTS.dec()
            structlog.contextvars.clear_contextvars()


def register_exception_handlers(app: FastAPI) -> None:
    """
    Register global exception handlers on the FastAPI app.
    Converts typed exceptions to consistent JSON error responses.
    """

    @app.exception_handler(MaaSException)
    async def maas_exception_handler(
        request: Request, exc: MaaSException
    ) -> JSONResponse:
        """Handle all MaaS platform exceptions."""
        request_id = request.headers.get("X-Request-ID", "unknown")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error_code": exc.error_code,
                "message": exc.message,
                "detail": exc.detail,
                "request_id": request_id,
            },
            headers={"X-Request-ID": request_id},
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """
        Catch-all for unhandled exceptions.
        Logs full traceback but returns a sanitized error to the client.
        """
        request_id = request.headers.get("X-Request-ID", "unknown")
        logger.error(
            "unhandled_exception",
            error_type=type(exc).__name__,
            error=str(exc),
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error_code": "INTERNAL_ERROR",
                "message": "An unexpected error occurred",
                "detail": {},
                "request_id": request_id,
            },
            headers={"X-Request-ID": request_id},
        )
