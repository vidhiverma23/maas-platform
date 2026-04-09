"""
Model Registry Service
======================
CRUD operations for ML model metadata in PostgreSQL.
This is the single source of truth for model existence and status.

Design:
- All DB operations go through this service (not direct ORM in routes)
- Pagination built-in for list operations
- File storage is handled here alongside metadata
- SHA-256 hash for model file integrity verification
"""

import hashlib
import os
import shutil
from pathlib import Path
from uuid import UUID

import aiofiles
from fastapi import UploadFile
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.database import MLModel, MLModelVersion, ModelStatus
from app.models.schemas import ModelCreate, ModelUpdate, ModelVersionCreate
from app.utils.exceptions import (
    ModelNotFoundException,
    ModelUploadError,
    ModelVersionNotFoundException,
    StorageError,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class ModelRegistry:
    """Service for managing ML model metadata and file storage."""

    def __init__(self, db: AsyncSession) -> None:
        self._db = db
        self._storage_path = Path(settings.model_storage_path)
        self._storage_path.mkdir(parents=True, exist_ok=True)

    # ── Model CRUD ───────────────────────────────────────────

    async def create_model(self, data: ModelCreate) -> MLModel:
        """Register a new model entity."""
        model = MLModel(
            name=data.name,
            description=data.description,
            model_type=data.model_type,
            owner=data.owner,
            tags=data.tags,
        )
        self._db.add(model)
        await self._db.flush()  # Populate ID without committing
        await self._db.refresh(model)

        logger.info(
            "model_created",
            model_id=str(model.id),
            name=model.name,
            model_type=model.model_type.value,
        )
        return model

    async def get_model(self, model_id: str) -> MLModel:
        """
        Fetch a model by UUID or name.
        Supports lookup by either for flexible API usage.
        """
        # Try UUID lookup first
        try:
            uuid_val = UUID(model_id)
            stmt = select(MLModel).where(MLModel.id == uuid_val)
        except ValueError:
            # Fall back to name lookup
            stmt = select(MLModel).where(MLModel.name == model_id)

        result = await self._db.execute(stmt)
        model = result.scalar_one_or_none()

        if not model:
            raise ModelNotFoundException(model_id)
        return model

    async def list_models(
        self,
        page: int = 1,
        page_size: int = 20,
        model_type: str | None = None,
        owner: str | None = None,
    ) -> tuple[list[MLModel], int]:
        """
        List models with pagination and optional filters.
        Returns (models, total_count) for pagination metadata.
        """
        stmt = select(MLModel)

        # Apply filters
        if model_type:
            stmt = stmt.where(MLModel.model_type == model_type)
        if owner:
            stmt = stmt.where(MLModel.owner == owner)

        # Get total count
        count_stmt = select(func.count()).select_from(stmt.subquery())
        total = (await self._db.execute(count_stmt)).scalar() or 0

        # Apply pagination
        offset = (page - 1) * page_size
        stmt = stmt.order_by(MLModel.created_at.desc()).offset(offset).limit(page_size)

        result = await self._db.execute(stmt)
        models = list(result.scalars().all())

        return models, total

    async def update_model(self, model_id: str, data: ModelUpdate) -> MLModel:
        """Update model metadata (description, owner, tags)."""
        model = await self.get_model(model_id)

        update_data = data.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(model, field, value)

        await self._db.flush()
        await self._db.refresh(model)

        logger.info("model_updated", model_id=str(model.id), fields=list(update_data.keys()))
        return model

    async def delete_model(self, model_id: str) -> None:
        """Delete a model and all its versions (cascade)."""
        model = await self.get_model(model_id)
        model_uuid = str(model.id)

        # Remove files from storage
        model_dir = self._storage_path / model_uuid
        if model_dir.exists():
            shutil.rmtree(model_dir)

        await self._db.delete(model)
        await self._db.flush()

        logger.info("model_deleted", model_id=model_uuid, name=model.name)

    # ── Version Management ───────────────────────────────────

    async def create_version(
        self,
        model_id: str,
        file: UploadFile,
        version_data: ModelVersionCreate,
    ) -> MLModelVersion:
        """
        Upload a model file and create a new version.
        Steps:
        1. Validate file format and size
        2. Determine next version number
        3. Store file on disk with content hash
        4. Create version record in DB
        """
        model = await self.get_model(model_id)
        model_uuid = str(model.id)

        # Validate file extension
        if not file.filename:
            raise ModelUploadError("Filename is required")

        file_ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
        if file_ext not in settings.supported_formats_list:
            raise ModelUploadError(
                f"Unsupported format '.{file_ext}'. "
                f"Supported: {settings.supported_formats_list}"
            )

        # Determine next version number
        latest_version = await self._get_latest_version_number(model.id)
        next_version = latest_version + 1

        # Create storage directory
        version_dir = self._storage_path / model_uuid / str(next_version)
        version_dir.mkdir(parents=True, exist_ok=True)
        file_path = version_dir / f"model.{file_ext}"

        # Stream file to disk and compute hash
        file_size, file_hash = await self._save_file(file, file_path)

        if file_size > settings.max_model_size_bytes:
            # Clean up oversized file
            os.remove(file_path)
            raise ModelUploadError(
                f"File size ({file_size / 1024 / 1024:.1f}MB) exceeds "
                f"limit ({settings.max_model_size_mb}MB)"
            )

        # Create version record
        version = MLModelVersion(
            model_id=model.id,
            version_number=next_version,
            version_tag=version_data.version_tag,
            file_path=str(file_path),
            file_size_bytes=file_size,
            file_hash=file_hash,
            framework=version_data.framework,
            format=file_ext,
            status=ModelStatus.READY,  # Mark as ready post-upload
            metrics=version_data.metrics,
            input_schema=version_data.input_schema,
            output_schema=version_data.output_schema,
            max_batch_size=version_data.max_batch_size,
            timeout_seconds=version_data.timeout_seconds,
        )
        self._db.add(version)
        await self._db.flush()
        await self._db.refresh(version)

        logger.info(
            "model_version_created",
            model_id=model_uuid,
            version=next_version,
            format=file_ext,
            size_mb=round(file_size / 1024 / 1024, 2),
        )
        return version

    async def get_version(
        self,
        model_id: str,
        version: int | None = None,
    ) -> MLModelVersion:
        """
        Get a specific version, or the latest READY version if version is None.
        """
        model = await self.get_model(model_id)

        if version is not None:
            stmt = select(MLModelVersion).where(
                MLModelVersion.model_id == model.id,
                MLModelVersion.version_number == version,
            )
        else:
            # Get latest READY version
            stmt = (
                select(MLModelVersion)
                .where(
                    MLModelVersion.model_id == model.id,
                    MLModelVersion.status == ModelStatus.READY,
                )
                .order_by(MLModelVersion.version_number.desc())
                .limit(1)
            )

        result = await self._db.execute(stmt)
        model_version = result.scalar_one_or_none()

        if not model_version:
            version_str = str(version) if version else "latest"
            raise ModelVersionNotFoundException(model_id, version_str)

        return model_version

    async def update_version_status(
        self,
        model_id: str,
        version: int,
        status: ModelStatus,
    ) -> MLModelVersion:
        """Update the status of a model version."""
        model_version = await self.get_version(model_id, version)
        model_version.status = status
        await self._db.flush()
        await self._db.refresh(model_version)
        logger.info(
            "version_status_updated",
            model_id=model_id,
            version=version,
            status=status.value,
        )
        return model_version

    # ── Internal Helpers ─────────────────────────────────────

    async def _get_latest_version_number(self, model_id: UUID) -> int:
        """Get the highest version number for a model, or 0 if none."""
        stmt = (
            select(func.max(MLModelVersion.version_number))
            .where(MLModelVersion.model_id == model_id)
        )
        result = await self._db.execute(stmt)
        return result.scalar() or 0

    @staticmethod
    async def _save_file(file: UploadFile, path: Path) -> tuple[int, str]:
        """
        Stream uploaded file to disk and compute SHA-256 hash.
        Returns (file_size_bytes, hex_digest).
        """
        hasher = hashlib.sha256()
        total_size = 0
        chunk_size = 64 * 1024  # 64KB chunks for memory efficiency

        try:
            async with aiofiles.open(path, "wb") as f:
                while chunk := await file.read(chunk_size):
                    await f.write(chunk)
                    hasher.update(chunk)
                    total_size += len(chunk)
        except OSError as e:
            raise StorageError("write", str(e))

        return total_size, hasher.hexdigest()
