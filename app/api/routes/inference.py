"""
Inference API Route
===================
The core prediction endpoint that accepts input data and returns
model predictions via the inference pipeline.

Flow:
1. Rate limit check
2. Cache lookup (Redis)
3. Model resolution (registry → version → loader)
4. Inference (engine)
5. Cache store
6. Response

Design:
- Single endpoint handles all models via model_id routing
- Supports version pinning or "latest" default
- Redis caching for repeated predictions
- Rate limiting per client IP
- Latency tracking in response headers
"""

import time

from fastapi import APIRouter, Request

from app.api.dependencies import CacheDep, RateLimiterDep, RegistryDep
from app.models.schemas import PredictionRequest, PredictionResponse
from app.services.inference_engine import inference_engine
from app.services.model_loader import model_loader
from app.utils.exceptions import RateLimitExceededError
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1", tags=["Inference"])


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Run model prediction",
    description=(
        "Submit input data for inference against a registered model. "
        "Supports model_id-based routing with optional version pinning."
    ),
    responses={
        429: {"description": "Rate limit exceeded"},
        404: {"description": "Model or version not found"},
        422: {"description": "Invalid input data"},
    },
)
async def predict(
    request: Request,
    payload: PredictionRequest,
    registry: RegistryDep,
    cache: CacheDep,
    rate_limiter: RateLimiterDep,
) -> PredictionResponse:
    """
    Run inference on a registered model.

    Pipeline:
    1. Check rate limit for client IP
    2. Look up cached prediction (Redis)
    3. Resolve model version from registry
    4. Load model into memory (LRU cache)
    5. Run inference via engine
    6. Cache the result
    7. Return response with latency metadata
    """
    start_time = time.monotonic()

    # ── Step 1: Rate Limiting ────────────────────────────────
    client_ip = request.client.host if request.client else "unknown"
    allowed, remaining, reset_ttl = await rate_limiter.is_allowed(client_ip)

    if not allowed:
        raise RateLimitExceededError(
            client_ip=client_ip,
            limit=rate_limiter.limit,
            window=rate_limiter.window,
        )

    # ── Step 2: Cache Lookup ─────────────────────────────────
    # Normalize input for cache key generation
    input_for_cache = payload.input_data
    if isinstance(input_for_cache, list):
        # Convert to hashable form
        input_for_cache = payload.input_data

    cached_predictions = await cache.get(
        model_id=payload.model_id,
        version=payload.version or 0,  # 0 means "latest"
        input_data=input_for_cache,
        parameters=payload.parameters,
    )

    if cached_predictions is not None:
        # Resolve model name for the response (even on cache hit)
        model = await registry.get_model(payload.model_id)
        version_obj = await registry.get_version(payload.model_id, payload.version)
        elapsed_ms = round((time.monotonic() - start_time) * 1000, 2)

        logger.info(
            "prediction_cache_hit",
            model_id=payload.model_id,
            latency_ms=elapsed_ms,
        )

        return PredictionResponse(
            model_id=str(model.id),
            model_name=model.name,
            version=version_obj.version_number,
            predictions=cached_predictions,
            latency_ms=elapsed_ms,
            cached=True,
            metadata={
                "rate_limit_remaining": remaining,
                "rate_limit_reset": reset_ttl,
            },
        )

    # ── Step 3: Resolve Model Version ────────────────────────
    model = await registry.get_model(payload.model_id)
    version_obj = await registry.get_version(payload.model_id, payload.version)

    # ── Step 4: Load Model ───────────────────────────────────
    loaded_model = await model_loader.get_model(version_obj)

    # ── Step 5: Run Inference ────────────────────────────────
    predictions = await inference_engine.predict(
        model=loaded_model,
        version=version_obj,
        input_data=payload.input_data,
        parameters=payload.parameters,
    )

    # ── Step 6: Cache Result ─────────────────────────────────
    await cache.set(
        model_id=payload.model_id,
        version=version_obj.version_number,
        input_data=input_for_cache,
        predictions=predictions,
        parameters=payload.parameters,
    )

    # ── Step 7: Response ─────────────────────────────────────
    elapsed_ms = round((time.monotonic() - start_time) * 1000, 2)

    return PredictionResponse(
        model_id=str(model.id),
        model_name=model.name,
        version=version_obj.version_number,
        predictions=predictions,
        latency_ms=elapsed_ms,
        cached=False,
        metadata={
            "framework": version_obj.framework,
            "rate_limit_remaining": remaining,
            "rate_limit_reset": reset_ttl,
        },
    )
