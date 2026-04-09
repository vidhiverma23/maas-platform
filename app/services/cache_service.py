"""
Redis Cache Service
===================
Prediction caching layer to avoid redundant inference for identical inputs.

Design:
- Cache key = SHA-256(model_id + version + sorted(input_data))
- JSON serialization for cache values
- Configurable TTL (default 300s)
- Graceful degradation: cache errors never break inference
- Hit/miss metrics for monitoring cache effectiveness

Why SHA-256 cache keys?
- Deterministic: same input always produces the same key
- Compact: 64-char hex string regardless of input size
- Collision-resistant: SHA-256 is cryptographically strong
"""

import hashlib
import json
from typing import Any

import redis.asyncio as redis

from app.config import get_settings
from app.utils.logger import get_logger
from app.utils.metrics import CACHE_HITS, CACHE_MISSES

logger = get_logger(__name__)
settings = get_settings()


class CacheService:
    """Redis-backed prediction cache."""

    def __init__(self, redis_client: redis.Redis) -> None:
        self._redis = redis_client
        self._enabled = settings.cache_enabled
        self._ttl = settings.cache_ttl_seconds

    def _make_key(
        self,
        model_id: str,
        version: int,
        input_data: Any,
        parameters: dict | None = None,
    ) -> str:
        """
        Generate a deterministic cache key from prediction inputs.
        Sorting ensures dict key order doesn't affect the hash.
        """
        payload = json.dumps(
            {
                "model_id": model_id,
                "version": version,
                "input": input_data,
                "params": parameters or {},
            },
            sort_keys=True,
            default=str,
        )
        digest = hashlib.sha256(payload.encode()).hexdigest()
        return f"pred_cache:{digest}"

    async def get(
        self,
        model_id: str,
        version: int,
        input_data: Any,
        parameters: dict | None = None,
    ) -> list[Any] | None:
        """
        Look up a cached prediction.
        Returns None on miss or if caching is disabled.
        """
        if not self._enabled:
            return None

        key = self._make_key(model_id, version, input_data, parameters)

        try:
            cached = await self._redis.get(key)
            if cached is not None:
                CACHE_HITS.inc()
                logger.debug("cache_hit", key=key[:20])
                return json.loads(cached)

            CACHE_MISSES.inc()
            return None

        except (redis.RedisError, json.JSONDecodeError) as e:
            # Cache failures should never block inference
            logger.warning("cache_get_error", error=str(e))
            CACHE_MISSES.inc()
            return None

    async def set(
        self,
        model_id: str,
        version: int,
        input_data: Any,
        predictions: list[Any],
        parameters: dict | None = None,
        ttl: int | None = None,
    ) -> None:
        """
        Store a prediction result in cache.
        No-op if caching is disabled.
        """
        if not self._enabled:
            return

        key = self._make_key(model_id, version, input_data, parameters)
        ttl = ttl or self._ttl

        try:
            value = json.dumps(predictions, default=str)
            await self._redis.set(key, value, ex=ttl)
            logger.debug("cache_set", key=key[:20], ttl=ttl)
        except redis.RedisError as e:
            logger.warning("cache_set_error", error=str(e))

    async def invalidate_model(self, model_id: str) -> int:
        """
        Invalidate all cached predictions for a model.
        Uses SCAN to avoid blocking Redis with KEYS command.
        """
        if not self._enabled:
            return 0

        # Note: This is a best-effort pattern scan.
        # In production, consider maintaining a set of keys per model.
        deleted = 0
        try:
            async for key in self._redis.scan_iter(
                match="pred_cache:*", count=100
            ):
                await self._redis.delete(key)
                deleted += 1
        except redis.RedisError as e:
            logger.warning("cache_invalidate_error", error=str(e))

        return deleted

    async def flush_all(self) -> None:
        """Clear the entire prediction cache."""
        try:
            # Only delete prediction cache keys, not rate limit keys
            async for key in self._redis.scan_iter(
                match="pred_cache:*", count=100
            ):
                await self._redis.delete(key)
            logger.info("prediction_cache_flushed")
        except redis.RedisError as e:
            logger.warning("cache_flush_error", error=str(e))

    async def stats(self) -> dict:
        """Return cache statistics."""
        try:
            info = await self._redis.info("memory")
            db_size = await self._redis.dbsize()
            return {
                "enabled": self._enabled,
                "ttl_seconds": self._ttl,
                "total_keys": db_size,
                "used_memory": info.get("used_memory_human", "unknown"),
            }
        except redis.RedisError:
            return {"enabled": self._enabled, "status": "unavailable"}
