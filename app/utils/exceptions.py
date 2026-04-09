"""
Custom Exception Hierarchy
==========================
Typed exceptions for clean error propagation across layers.
FastAPI exception handlers in middleware.py translate these
into appropriate HTTP responses.

Design:
- Base MaaSException carries an error_code for machine-readable errors
- Subclasses map to specific HTTP status codes
- All exceptions carry contextual detail for debugging
"""

from typing import Any


class MaaSException(Exception):
    """Base exception for all MaaS platform errors."""

    def __init__(
        self,
        message: str,
        error_code: str = "INTERNAL_ERROR",
        status_code: int = 500,
        detail: dict[str, Any] | None = None,
    ) -> None:
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.detail = detail or {}
        super().__init__(self.message)


class ModelNotFoundException(MaaSException):
    """Raised when a requested model does not exist."""

    def __init__(self, model_id: str) -> None:
        super().__init__(
            message=f"Model '{model_id}' not found",
            error_code="MODEL_NOT_FOUND",
            status_code=404,
            detail={"model_id": model_id},
        )


class ModelVersionNotFoundException(MaaSException):
    """Raised when a specific model version does not exist."""

    def __init__(self, model_id: str, version: str) -> None:
        super().__init__(
            message=f"Model '{model_id}' version '{version}' not found",
            error_code="MODEL_VERSION_NOT_FOUND",
            status_code=404,
            detail={"model_id": model_id, "version": version},
        )


class ModelLoadError(MaaSException):
    """Raised when a model file cannot be loaded into memory."""

    def __init__(self, model_id: str, reason: str) -> None:
        super().__init__(
            message=f"Failed to load model '{model_id}': {reason}",
            error_code="MODEL_LOAD_ERROR",
            status_code=500,
            detail={"model_id": model_id, "reason": reason},
        )


class ModelUploadError(MaaSException):
    """Raised when model file upload fails validation."""

    def __init__(self, reason: str) -> None:
        super().__init__(
            message=f"Model upload failed: {reason}",
            error_code="MODEL_UPLOAD_ERROR",
            status_code=400,
            detail={"reason": reason},
        )


class InferenceError(MaaSException):
    """Raised when prediction fails during inference."""

    def __init__(self, model_id: str, reason: str) -> None:
        super().__init__(
            message=f"Inference failed for model '{model_id}': {reason}",
            error_code="INFERENCE_ERROR",
            status_code=500,
            detail={"model_id": model_id, "reason": reason},
        )


class InvalidInputError(MaaSException):
    """Raised when prediction input data is malformed."""

    def __init__(self, reason: str) -> None:
        super().__init__(
            message=f"Invalid input data: {reason}",
            error_code="INVALID_INPUT",
            status_code=422,
            detail={"reason": reason},
        )


class RateLimitExceededError(MaaSException):
    """Raised when client exceeds the request rate limit."""

    def __init__(self, client_ip: str, limit: int, window: int) -> None:
        super().__init__(
            message=f"Rate limit exceeded: {limit} requests per {window}s",
            error_code="RATE_LIMIT_EXCEEDED",
            status_code=429,
            detail={
                "client_ip": client_ip,
                "limit": limit,
                "window_seconds": window,
            },
        )


class StorageError(MaaSException):
    """Raised when model file storage operations fail."""

    def __init__(self, operation: str, reason: str) -> None:
        super().__init__(
            message=f"Storage {operation} failed: {reason}",
            error_code="STORAGE_ERROR",
            status_code=500,
            detail={"operation": operation, "reason": reason},
        )
