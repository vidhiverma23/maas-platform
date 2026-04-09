"""
Dynamic Model Loader with LRU Cache
=====================================
Loads ML models from disk into memory on demand, with an eviction
policy to prevent unbounded memory growth.

Key decisions:
- TTL-based LRU cache: models evicted after inactivity period
- Thread-safe: uses asyncio.Lock for concurrent access protection
- Multi-format: dispatches to sklearn/ONNX/PyTorch loaders
- Warm-up: pre-loads models at startup if configured
- Memory tracking: gauge metric for loaded model count

Why not functools.lru_cache?
- Need async support
- Need TTL-based eviction (not just size-based)
- Need to track access time for LRU ordering
"""

import asyncio
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np

from app.models.database import MLModelVersion
from app.utils.exceptions import ModelLoadError
from app.utils.logger import get_logger
from app.utils.metrics import MODEL_LOAD_LATENCY, MODELS_LOADED

logger = get_logger(__name__)

# Maximum number of models to keep in memory simultaneously
MAX_CACHED_MODELS = 50
# Evict models not accessed within this window (seconds)
MODEL_TTL_SECONDS = 3600  # 1 hour


class CachedModel:
    """Container for a loaded model with access tracking."""

    __slots__ = ("model", "version_id", "framework", "loaded_at", "last_accessed")

    def __init__(self, model: Any, version_id: str, framework: str) -> None:
        self.model = model
        self.version_id = version_id
        self.framework = framework
        self.loaded_at = time.monotonic()
        self.last_accessed = time.monotonic()

    def touch(self) -> None:
        """Update last-access time (for LRU eviction)."""
        self.last_accessed = time.monotonic()

    @property
    def age_seconds(self) -> float:
        return time.monotonic() - self.last_accessed


class ModelLoader:
    """
    Thread-safe model loader with LRU cache and TTL eviction.
    Supports sklearn (pickle), ONNX, and PyTorch models.
    """

    def __init__(self) -> None:
        self._cache: dict[str, CachedModel] = {}
        self._lock = asyncio.Lock()

    async def get_model(self, version: MLModelVersion) -> Any:
        """
        Get a loaded model instance, loading from disk if not cached.
        Thread-safe via asyncio.Lock.
        """
        cache_key = str(version.id)

        # Fast path: cache hit
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            cached.touch()
            return cached.model

        # Slow path: load from disk (with lock to prevent thundering herd)
        async with self._lock:
            # Double-check after acquiring lock
            if cache_key in self._cache:
                cached = self._cache[cache_key]
                cached.touch()
                return cached.model

            # Evict stale entries before loading
            await self._evict_stale()

            # Evict LRU if at capacity
            if len(self._cache) >= MAX_CACHED_MODELS:
                await self._evict_lru()

            # Load the model
            model = await self._load_from_disk(version)

            self._cache[cache_key] = CachedModel(
                model=model,
                version_id=cache_key,
                framework=version.framework,
            )
            MODELS_LOADED.set(len(self._cache))

            logger.info(
                "model_loaded_to_cache",
                version_id=cache_key,
                framework=version.framework,
                cache_size=len(self._cache),
            )
            return model

    async def _load_from_disk(self, version: MLModelVersion) -> Any:
        """
        Load a model file based on its format.
        Runs CPU-bound loading in a thread pool to avoid blocking the event loop.
        """
        file_path = Path(version.file_path)
        if not file_path.exists():
            raise ModelLoadError(
                str(version.model_id),
                f"Model file not found: {file_path}",
            )

        start = time.monotonic()

        try:
            loop = asyncio.get_event_loop()
            model = await loop.run_in_executor(
                None,
                self._load_sync,
                file_path,
                version.format,
                version.framework,
            )
        except ModelLoadError:
            raise
        except Exception as e:
            raise ModelLoadError(str(version.model_id), str(e))

        elapsed = time.monotonic() - start
        MODEL_LOAD_LATENCY.labels(model_type=version.framework).observe(elapsed)

        logger.info(
            "model_loaded_from_disk",
            version_id=str(version.id),
            format=version.format,
            load_time_ms=round(elapsed * 1000, 2),
        )
        return model

    @staticmethod
    def _load_sync(file_path: Path, fmt: str, framework: str) -> Any:
        """
        Synchronous model loading — runs in thread pool.
        Dispatches to the appropriate loader based on format.
        """
        if fmt in ("pkl", "joblib"):
            return ModelLoader._load_pickle(file_path)
        elif fmt == "onnx":
            return ModelLoader._load_onnx(file_path)
        elif fmt == "pt":
            return ModelLoader._load_pytorch(file_path)
        else:
            raise ModelLoadError("unknown", f"Unsupported format: {fmt}")

    @staticmethod
    def _load_pickle(path: Path) -> Any:
        """Load a pickle/joblib serialized sklearn model."""
        with open(path, "rb") as f:
            return pickle.load(f)  # noqa: S301 — trusted internal models

    @staticmethod
    def _load_onnx(path: Path) -> Any:
        """
        Load an ONNX model using onnxruntime.
        Returns an InferenceSession which is the runtime handle.
        """
        try:
            import onnxruntime as ort

            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            # Use all available CPU cores for intra-op parallelism
            sess_options.intra_op_num_threads = 0
            return ort.InferenceSession(str(path), sess_options)
        except ImportError:
            raise ModelLoadError("onnx", "onnxruntime not installed")

    @staticmethod
    def _load_pytorch(path: Path) -> Any:
        """Load a PyTorch model (TorchScript format)."""
        try:
            import torch

            model = torch.jit.load(str(path), map_location="cpu")
            model.eval()
            return model
        except ImportError:
            raise ModelLoadError("pytorch", "torch not installed")

    # ── Cache Management ─────────────────────────────────────

    async def _evict_stale(self) -> None:
        """Remove models that haven't been accessed within the TTL."""
        stale_keys = [
            key for key, cached in self._cache.items()
            if cached.age_seconds > MODEL_TTL_SECONDS
        ]
        for key in stale_keys:
            del self._cache[key]
            logger.info("model_evicted_ttl", version_id=key)

        if stale_keys:
            MODELS_LOADED.set(len(self._cache))

    async def _evict_lru(self) -> None:
        """Evict the least-recently-used model."""
        if not self._cache:
            return

        lru_key = min(self._cache, key=lambda k: self._cache[k].last_accessed)
        del self._cache[lru_key]
        MODELS_LOADED.set(len(self._cache))
        logger.info("model_evicted_lru", version_id=lru_key)

    async def clear_cache(self) -> None:
        """Clear all cached models (for shutdown or testing)."""
        async with self._lock:
            self._cache.clear()
            MODELS_LOADED.set(0)
            logger.info("model_cache_cleared")

    def cache_stats(self) -> dict:
        """Return cache statistics for monitoring."""
        return {
            "cached_models": len(self._cache),
            "max_capacity": MAX_CACHED_MODELS,
            "ttl_seconds": MODEL_TTL_SECONDS,
            "models": [
                {
                    "version_id": key,
                    "framework": cached.framework,
                    "age_seconds": round(cached.age_seconds, 1),
                }
                for key, cached in self._cache.items()
            ],
        }


# ── Singleton ────────────────────────────────────────────────
# Single instance shared across all requests in a worker process
model_loader = ModelLoader()
