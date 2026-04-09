"""
Structured Logging Configuration
=================================
Uses structlog for machine-parseable JSON logs in production and
human-readable colored logs in development.

Key decisions:
- JSON output in production for log aggregation (ELK, Azure Monitor)
- Console output in development for readability
- Request-scoped context binding (request_id, client_ip, etc.)
- Performance: processors are configured once at startup
"""

import logging
import sys
from typing import Any

import structlog
from app.config import get_settings


def setup_logging() -> None:
    """
    Configure structlog processors and stdlib logging bridge.
    Call once at application startup.
    """
    settings = get_settings()

    # Shared processors for both renderers
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,       # Pull in request-scoped vars
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if settings.log_format == "json":
        # Production: JSON lines for machine parsing
        renderer = structlog.processors.JSONRenderer()
    else:
        # Development: colored, human-readable output
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Bridge stdlib logging → structlog formatting
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

    # Suppress noisy third-party loggers
    for noisy_logger in ("uvicorn.access", "sqlalchemy.engine", "httpx"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """
    Get a named, structured logger.

    Usage:
        logger = get_logger(__name__)
        logger.info("model_loaded", model_id="abc-123", latency_ms=42)
    """
    return structlog.get_logger(name or __name__)
