"""
Pydantic Request/Response Schemas
==================================
Strict validation at the API boundary. All input is validated here
before reaching business logic — defense in depth.

Design:
- Separate Create/Update/Response schemas to control field exposure
- UUID serialization as strings in responses
- Enum reuse from database models for consistency
- ConfigDict for ORM mode (from_attributes) for seamless SQLAlchemy conversion
"""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.models.database import ModelStatus, ModelType


# ── Model Schemas ────────────────────────────────────────────

class ModelCreate(BaseModel):
    """Schema for registering a new model (POST /models)."""
    name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Unique model name (e.g. 'fraud-detector')",
        examples=["fraud-detector"],
    )
    description: str | None = Field(
        None,
        max_length=2000,
        description="Human-readable model description",
    )
    model_type: ModelType = Field(
        default=ModelType.SKLEARN,
        description="ML framework type",
    )
    owner: str = Field(
        default="system",
        max_length=255,
        description="Model owner or team name",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Searchable tags",
        examples=[["classification", "tabular", "production"]],
    )


class ModelUpdate(BaseModel):
    """Schema for updating model metadata (PATCH /models/{id})."""
    description: str | None = None
    owner: str | None = None
    tags: list[str] | None = None


class ModelResponse(BaseModel):
    """Schema for model in API responses."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    description: str | None
    model_type: ModelType
    owner: str | None
    tags: list[str] | None
    created_at: datetime
    updated_at: datetime
    versions: list["ModelVersionResponse"] = []


class ModelListResponse(BaseModel):
    """Paginated list of models."""
    items: list[ModelResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


# ── Model Version Schemas ────────────────────────────────────

class ModelVersionCreate(BaseModel):
    """Schema for creating a model version (via file upload)."""
    version_tag: str | None = Field(
        None,
        max_length=50,
        description="Optional semantic version tag (e.g. 'v1.2.0')",
    )
    framework: str = Field(
        default="sklearn",
        description="ML framework used to train the model",
    )
    input_schema: dict[str, Any] | None = Field(
        None,
        description="Expected input shape/types (JSON schema)",
    )
    output_schema: dict[str, Any] | None = Field(
        None,
        description="Expected output shape/types (JSON schema)",
    )
    metrics: dict[str, Any] | None = Field(
        None,
        description="Training/validation metrics",
        examples=[{"accuracy": 0.95, "f1_score": 0.92}],
    )
    max_batch_size: int = Field(
        default=32,
        ge=1,
        le=1024,
        description="Maximum inference batch size",
    )
    timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description="Inference timeout in seconds",
    )


class ModelVersionResponse(BaseModel):
    """Schema for model version in API responses."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    model_id: UUID
    version_number: int
    version_tag: str | None
    framework: str
    format: str
    status: ModelStatus
    file_size_bytes: int | None
    file_hash: str | None
    metrics: dict[str, Any] | None
    input_schema: dict[str, Any] | None
    output_schema: dict[str, Any] | None
    max_batch_size: int | None
    timeout_seconds: float | None
    created_at: datetime
    updated_at: datetime


class ModelVersionStatusUpdate(BaseModel):
    """Schema for updating a version's status."""
    status: ModelStatus


# ── Inference Schemas ────────────────────────────────────────

class PredictionRequest(BaseModel):
    """Schema for prediction input (POST /predict)."""
    model_id: str = Field(
        ...,
        description="UUID or name of the model to use",
    )
    version: int | None = Field(
        None,
        description="Specific version number (latest if omitted)",
    )
    input_data: list[list[float]] | dict[str, Any] = Field(
        ...,
        description="Input features — 2D array for tabular, or dict for structured",
        examples=[[[1.0, 2.0, 3.0, 4.0]]],
    )
    parameters: dict[str, Any] | None = Field(
        None,
        description="Optional inference parameters (threshold, top_k, etc.)",
    )


class PredictionResponse(BaseModel):
    """Schema for prediction output."""
    model_id: str
    model_name: str
    version: int
    predictions: list[Any]
    latency_ms: float
    cached: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Health Schemas ───────────────────────────────────────────

class HealthResponse(BaseModel):
    """Health check response."""
    status: str  # "healthy" | "degraded" | "unhealthy"
    version: str
    uptime_seconds: float
    checks: dict[str, str]  # {"database": "ok", "redis": "ok"}


class ErrorResponse(BaseModel):
    """Standard error response format."""
    error_code: str
    message: str
    detail: dict[str, Any] = Field(default_factory=dict)
    request_id: str | None = None
