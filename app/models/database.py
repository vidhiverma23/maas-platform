"""
SQLAlchemy ORM Models
=====================
Database schema for model metadata and version tracking.

Design:
- MLModel: top-level model entity (name, description, owner)
- MLModelVersion: versioned artifact with file path, metrics, status
- One-to-many: each MLModel has multiple MLModelVersions
- UUID primary keys for distributed safety (no sequential ID guessing)
- Timestamps for audit trail
"""

import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import relationship

from app.database.connection import Base


class ModelStatus(str, enum.Enum):
    """Lifecycle states for a model version."""
    UPLOADING = "uploading"
    REGISTERED = "registered"
    VALIDATING = "validating"
    READY = "ready"         # Available for inference
    FAILED = "failed"       # Validation or loading failed
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class ModelType(str, enum.Enum):
    """Supported ML framework types."""
    SKLEARN = "sklearn"
    PYTORCH = "pytorch"
    ONNX = "onnx"
    CUSTOM = "custom"


class MLModel(Base):
    """
    Top-level model entity.
    Represents a logical model (e.g. "fraud-detector") that can
    have multiple deployed versions.
    """
    __tablename__ = "ml_models"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
    )
    name = Column(String(255), nullable=False, unique=True, index=True)
    description = Column(Text, nullable=True)
    owner = Column(String(255), nullable=True, default="system")
    model_type = Column(
        Enum(ModelType, name="model_type_enum"),
        nullable=False,
        default=ModelType.SKLEARN,
    )
    tags = Column(JSON, nullable=True, default=list)  # ["classification", "tabular"]

    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    # Relationships
    versions = relationship(
        "MLModelVersion",
        back_populates="model",
        cascade="all, delete-orphan",
        order_by="MLModelVersion.version_number.desc()",
        lazy="selectin",  # Eager-load versions to avoid N+1
    )

    # Indexes for common query patterns
    __table_args__ = (
        Index("idx_models_owner", "owner"),
        Index("idx_models_type", "model_type"),
        Index("idx_models_created", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<MLModel(name={self.name}, type={self.model_type})>"


class MLModelVersion(Base):
    """
    Versioned model artifact.
    Each version has its own file, metrics, and deployment status.
    """
    __tablename__ = "ml_model_versions"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
    )
    model_id = Column(
        UUID(as_uuid=True),
        ForeignKey("ml_models.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    version_number = Column(Integer, nullable=False, default=1)
    version_tag = Column(String(50), nullable=True)  # "v1.2.0" or "latest"

    # File info
    file_path = Column(String(512), nullable=False)
    file_size_bytes = Column(Integer, nullable=True)
    file_hash = Column(String(64), nullable=True)  # SHA-256 for integrity

    # Framework / format
    framework = Column(String(50), nullable=False, default="sklearn")
    format = Column(String(20), nullable=False, default="pkl")  # pkl, onnx, pt

    # Status
    status = Column(
        Enum(ModelStatus, name="model_status_enum"),
        nullable=False,
        default=ModelStatus.REGISTERED,
    )

    # Performance metrics (stored after validation)
    metrics = Column(JSON, nullable=True, default=dict)  # {"accuracy": 0.95, ...}
    input_schema = Column(JSON, nullable=True)   # Expected input shape/types
    output_schema = Column(JSON, nullable=True)  # Expected output shape/types

    # Inference config
    max_batch_size = Column(Integer, nullable=True, default=32)
    timeout_seconds = Column(Float, nullable=True, default=30.0)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    # Relationships
    model = relationship("MLModel", back_populates="versions")

    # Composite indexes
    __table_args__ = (
        Index("idx_version_model_number", "model_id", "version_number", unique=True),
        Index("idx_version_status", "status"),
    )

    def __repr__(self) -> str:
        return f"<MLModelVersion(model_id={self.model_id}, v={self.version_number})>"
