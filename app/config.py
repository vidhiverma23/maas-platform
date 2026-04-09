"""
Centralized Configuration Management
=====================================
Uses pydantic-settings for type-safe environment variable parsing.
All config flows through this single source of truth — no scattered
os.getenv() calls throughout the codebase.

Key decisions:
- Pydantic v2 Settings with `model_config` for .env file loading
- Cached singleton via lru_cache to avoid re-parsing on every import
- Sensible defaults for local development, overridden in production
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Silently ignore unknown env vars
    )

    # --- Application ---
    app_name: str = "maas-platform"
    app_env: Literal["development", "staging", "production"] = "development"
    app_debug: bool = True
    app_version: str = "1.0.0"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_workers: int = 1

    # --- PostgreSQL ---
    postgres_user: str = "maas_user"
    postgres_password: str = "maas_secret_password"
    postgres_db: str = "maas_db"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    database_url: str = (
        "postgresql+asyncpg://maas_user:maas_secret_password@localhost:5432/maas_db"
    )

    # --- Redis ---
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_url: str = "redis://localhost:6379/0"

    # --- Model Storage ---
    model_storage_path: str = "./model_storage"
    max_model_size_mb: int = 500
    supported_model_formats: str = "pkl,onnx,pt,joblib"

    # --- Rate Limiting ---
    rate_limit_requests: int = 100
    rate_limit_window_seconds: int = 60

    # --- Caching ---
    cache_ttl_seconds: int = 300
    cache_enabled: bool = True

    # --- Logging ---
    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "json"

    @property
    def supported_formats_list(self) -> list[str]:
        """Parse comma-separated format string into a list."""
        return [fmt.strip().lower() for fmt in self.supported_model_formats.split(",")]

    @property
    def max_model_size_bytes(self) -> int:
        """Convert MB limit to bytes for file upload validation."""
        return self.max_model_size_mb * 1024 * 1024

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Cached settings singleton.
    Call this function anywhere you need config — it's parsed once
    and reused for the lifetime of the process.
    """
    return Settings()
