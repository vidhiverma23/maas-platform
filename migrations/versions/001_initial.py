"""Initial schema — ML models and versions

Revision ID: 001_initial
Revises: None
Create Date: 2024-01-01
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSON, UUID

# revision identifiers
revision: str = "001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create model_type enum
    model_type_enum = sa.Enum(
        "sklearn", "pytorch", "onnx", "custom",
        name="model_type_enum",
    )
    model_type_enum.create(op.get_bind(), checkfirst=True)

    # Create model_status enum
    model_status_enum = sa.Enum(
        "uploading", "registered", "validating", "ready",
        "failed", "deprecated", "archived",
        name="model_status_enum",
    )
    model_status_enum.create(op.get_bind(), checkfirst=True)

    # Create ml_models table
    op.create_table(
        "ml_models",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False, unique=True),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("owner", sa.String(255), nullable=True, server_default="system"),
        sa.Column(
            "model_type",
            model_type_enum,
            nullable=False,
            server_default="sklearn",
        ),
        sa.Column("tags", JSON, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
    )

    op.create_index("idx_models_owner", "ml_models", ["owner"])
    op.create_index("idx_models_type", "ml_models", ["model_type"])
    op.create_index("idx_models_created", "ml_models", ["created_at"])

    # Create ml_model_versions table
    op.create_table(
        "ml_model_versions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "model_id",
            UUID(as_uuid=True),
            sa.ForeignKey("ml_models.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("version_number", sa.Integer, nullable=False, server_default="1"),
        sa.Column("version_tag", sa.String(50), nullable=True),
        sa.Column("file_path", sa.String(512), nullable=False),
        sa.Column("file_size_bytes", sa.Integer, nullable=True),
        sa.Column("file_hash", sa.String(64), nullable=True),
        sa.Column("framework", sa.String(50), nullable=False, server_default="sklearn"),
        sa.Column("format", sa.String(20), nullable=False, server_default="pkl"),
        sa.Column(
            "status",
            model_status_enum,
            nullable=False,
            server_default="registered",
        ),
        sa.Column("metrics", JSON, nullable=True),
        sa.Column("input_schema", JSON, nullable=True),
        sa.Column("output_schema", JSON, nullable=True),
        sa.Column("max_batch_size", sa.Integer, nullable=True, server_default="32"),
        sa.Column("timeout_seconds", sa.Float, nullable=True, server_default="30.0"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
    )

    op.create_index("idx_version_model_number", "ml_model_versions", ["model_id", "version_number"], unique=True)
    op.create_index("idx_version_status", "ml_model_versions", ["status"])


def downgrade() -> None:
    op.drop_table("ml_model_versions")
    op.drop_table("ml_models")

    # Drop enums
    sa.Enum(name="model_status_enum").drop(op.get_bind(), checkfirst=True)
    sa.Enum(name="model_type_enum").drop(op.get_bind(), checkfirst=True)
