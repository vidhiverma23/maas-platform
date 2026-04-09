"""
Distributed Token-Bucket Rate Limiter
======================================
Redis-backed rate limiter that works across multiple FastAPI instances.

Algorithm: Sliding window counter using Redis INCR + EXPIRE.
- Each client IP gets a Redis key: `rate_limit:{ip}`
- Key incremented on each request; TTL set to the window duration
- When count exceeds the limit, requests are rejected with 429

Why Redis-based (not in-memory)?
- Consistent limits across horizontal replicas behind Nginx
- Survives process restarts
- Atomic operations prevent race conditions
"""

import redis.asyncio as redis

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class RateLimiter:
    """Distributed rate limiter backed by Redis."""

    def __init__(self, redis_client: redis.Redis) -> None:
        self._redis = redis_client
        self._settings = get_settings()

    @property
    def limit(self) -> int:
        return self._settings.rate_limit_requests

    @property
    def window(self) -> int:
        return self._settings.rate_limit_window_seconds

    async def is_allowed(self, client_ip: str) -> tuple[bool, int, int]:
        """
        Check if a request from `client_ip` is within rate limits.

        Returns:
            (allowed, remaining, reset_ttl)
            - allowed: True if the request should proceed
            - remaining: Number of requests left in the window
            - reset_ttl: Seconds until the window resets
        """
        key = f"rate_limit:{client_ip}"

        try:
            # Use pipeline for atomic increment + TTL check
            pipe = self._redis.pipeline(transaction=True)
            pipe.incr(key)
            pipe.ttl(key)
            results = await pipe.execute()

            current_count: int = results[0]
            ttl: int = results[1]

            # First request in window — set expiry
            if ttl == -1:
                await self._redis.expire(key, self.window)
                ttl = self.window

            remaining = max(0, self.limit - current_count)
            allowed = current_count <= self.limit

            if not allowed:
                logger.warning(
                    "rate_limit_exceeded",
                    client_ip=client_ip,
                    count=current_count,
                    limit=self.limit,
                )

            return allowed, remaining, ttl

        except redis.RedisError as e:
            # Fail open: if Redis is down, allow the request.
            # This is a deliberate choice — availability over strictness.
            logger.error("rate_limiter_redis_error", error=str(e))
            return True, self.limit, self.window

    async def get_usage(self, client_ip: str) -> dict:
        """Get current rate limit usage for a client (for headers)."""
        key = f"rate_limit:{client_ip}"
        try:
            count = await self._redis.get(key)
            ttl = await self._redis.ttl(key)
            current = int(count) if count else 0
            return {
                "limit": self.limit,
                "remaining": max(0, self.limit - current),
                "reset": max(0, ttl),
            }
        except redis.RedisError:
            return {"limit": self.limit, "remaining": self.limit, "reset": self.window}
