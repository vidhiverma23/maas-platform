"""
Model Management API Routes
============================
CRUD endpoints for ML model registration and version management.

Endpoints:
- POST   /api/v1/models                     — Register a new model
- GET    /api/v1/models                     — List all models (paginated)
- GET    /api/v1/models/{model_id}          — Get model details
- PATCH  /api/v1/models/{model_id}          — Update model metadata
- DELETE /api/v1/models/{model_id}          — Delete model + all versions
- POST   /api/v1/models/{model_id}/versions — Upload a new model version
- GET    /api/v1/models/{model_id}/versions/{version} — Get version details
- PATCH  /api/v1/models/{model_id}/versions/{version}/status — Update status

Design:
- All routes delegate to ModelRegistry service (no business logic here)
- Pydantic schemas for request/response validation
- Rate limiting on mutation endpoints
"""

import math

from fastapi import APIRouter, File, Form, Query, UploadFile

from app.api.dependencies import CacheDep, RateLimiterDep, RegistryDep
from app.models.database import ModelStatus, ModelType
from app.models.schemas import (
    ModelCreate,
    ModelListResponse,
    ModelResponse,
    ModelUpdate,
    ModelVersionCreate,
    ModelVersionResponse,
    ModelVersionStatusUpdate,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/models", tags=["Models"])


# ── Model CRUD ───────────────────────────────────────────────

@router.post(
    "",
    response_model=ModelResponse,
    status_code=201,
    summary="Register a new model",
    description="Create a new model entity. Upload versions separately.",
)
async def create_model(
    data: ModelCreate,
    registry: RegistryDep,
) -> ModelResponse:
    """Register a new ML model with metadata."""
    model = await registry.create_model(data)
    return ModelResponse.model_validate(model)


@router.get(
    "",
    response_model=ModelListResponse,
    summary="List all models",
    description="Get paginated list of registered models with optional filters.",
)
async def list_models(
    registry: RegistryDep,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    model_type: str | None = Query(None, description="Filter by model type"),
    owner: str | None = Query(None, description="Filter by owner"),
) -> ModelListResponse:
    """List all registered models with pagination."""
    models, total = await registry.list_models(
        page=page,
        page_size=page_size,
        model_type=model_type,
        owner=owner,
    )
    return ModelListResponse(
        items=[ModelResponse.model_validate(m) for m in models],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=math.ceil(total / page_size) if total > 0 else 0,
    )


@router.get(
    "/{model_id}",
    response_model=ModelResponse,
    summary="Get model details",
    description="Retrieve a model by UUID or name, including all versions.",
)
async def get_model(
    model_id: str,
    registry: RegistryDep,
) -> ModelResponse:
    """Get model details with all versions."""
    model = await registry.get_model(model_id)
    return ModelResponse.model_validate(model)


@router.patch(
    "/{model_id}",
    response_model=ModelResponse,
    summary="Update model metadata",
    description="Update description, owner, or tags of an existing model.",
)
async def update_model(
    model_id: str,
    data: ModelUpdate,
    registry: RegistryDep,
) -> ModelResponse:
    """Update model metadata."""
    model = await registry.update_model(model_id, data)
    return ModelResponse.model_validate(model)


@router.delete(
    "/{model_id}",
    status_code=204,
    summary="Delete model",
    description="Delete a model and all its versions (irreversible).",
)
async def delete_model(
    model_id: str,
    registry: RegistryDep,
    cache: CacheDep,
) -> None:
    """Delete a model and all its versions."""
    await registry.delete_model(model_id)
    await cache.invalidate_model(model_id)


# ── Version Management ───────────────────────────────────────

@router.post(
    "/{model_id}/versions",
    response_model=ModelVersionResponse,
    status_code=201,
    summary="Upload a new model version",
    description=(
        "Upload a model file (pkl, onnx, pt, joblib) and create a new version. "
        "Version number is auto-incremented."
    ),
)
async def upload_version(
    model_id: str,
    registry: RegistryDep,
    file: UploadFile = File(..., description="Model file (pkl, onnx, pt, joblib)"),
    version_tag: str | None = Form(None, description="Optional version tag"),
    framework: str = Form("sklearn", description="ML framework"),
    max_batch_size: int = Form(32, description="Max inference batch size"),
    timeout_seconds: float = Form(30.0, description="Inference timeout"),
) -> ModelVersionResponse:
    """Upload a model file and create a new version."""
    version_data = ModelVersionCreate(
        version_tag=version_tag,
        framework=framework,
        max_batch_size=max_batch_size,
        timeout_seconds=timeout_seconds,
    )
    version = await registry.create_version(model_id, file, version_data)
    return ModelVersionResponse.model_validate(version)


@router.get(
    "/{model_id}/versions/{version}",
    response_model=ModelVersionResponse,
    summary="Get version details",
)
async def get_version(
    model_id: str,
    version: int,
    registry: RegistryDep,
) -> ModelVersionResponse:
    """Get details of a specific model version."""
    model_version = await registry.get_version(model_id, version)
    return ModelVersionResponse.model_validate(model_version)


@router.patch(
    "/{model_id}/versions/{version}/status",
    response_model=ModelVersionResponse,
    summary="Update version status",
    description="Change the lifecycle status of a model version (ready, deprecated, etc.).",
)
async def update_version_status(
    model_id: str,
    version: int,
    data: ModelVersionStatusUpdate,
    registry: RegistryDep,
) -> ModelVersionResponse:
    """Update model version status."""
    model_version = await registry.update_version_status(
        model_id, version, data.status
    )
    return ModelVersionResponse.model_validate(model_version)
