"""
FastAPI Dependency Injection
============================
Shared dependencies injected into route handlers via FastAPI's
Depends() system. This is the wiring layer that connects
infrastructure (DB, Redis) to business logic (services).

Design:
- Database session: scoped per request via async generator
- Redis client: long-lived connection pool, shared across requests
- Services: instantiated per request with injected dependencies
- Rate limiter: shared singleton backed by Redis
"""

from collections.abc import AsyncGenerator
from typing import Annotated

import redis.asyncio as redis
from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database.connection import get_db_session
from app.services.cache_service import CacheService
from app.services.model_registry import ModelRegistry
from app.utils.rate_limiter import RateLimiter

settings = get_settings()

# ── Redis Connection Pool ────────────────────────────────────
# Created once at module load; reused for all requests.
# Connection pool handles reconnects and health checks internally.
_redis_pool: redis.Redis | None = None


async def get_redis() -> redis.Redis:
    """Get the shared Redis client instance."""
    global _redis_pool
    if _redis_pool is None:
        _redis_pool = redis.from_url(
            settings.redis_url,
            decode_responses=True,
            max_connections=50,
            socket_connect_timeout=5,
            socket_keepalive=True,
            retry_on_timeout=True,
        )
    return _redis_pool


async def close_redis() -> None:
    """Close Redis connection pool on shutdown."""
    global _redis_pool
    if _redis_pool is not None:
        await _redis_pool.close()
        _redis_pool = None


# ── Dependency Types ─────────────────────────────────────────
# Using Annotated types for clean dependency injection syntax

async def get_model_registry(
    db: AsyncSession = Depends(get_db_session),
) -> ModelRegistry:
    """Get a ModelRegistry service scoped to the current request's DB session."""
    return ModelRegistry(db)


async def get_cache_service(
    redis_client: redis.Redis = Depends(get_redis),
) -> CacheService:
    """Get a CacheService backed by the shared Redis pool."""
    return CacheService(redis_client)


async def get_rate_limiter(
    redis_client: redis.Redis = Depends(get_redis),
) -> RateLimiter:
    """Get the rate limiter backed by the shared Redis pool."""
    return RateLimiter(redis_client)


# ── Type Aliases for Route Handlers ──────────────────────────
# These make route handler signatures cleaner:
#   async def create_model(registry: RegistryDep, ...):

DBSession = Annotated[AsyncSession, Depends(get_db_session)]
RedisClient = Annotated[redis.Redis, Depends(get_redis)]
RegistryDep = Annotated[ModelRegistry, Depends(get_model_registry)]
CacheDep = Annotated[CacheService, Depends(get_cache_service)]
RateLimiterDep = Annotated[RateLimiter, Depends(get_rate_limiter)]
