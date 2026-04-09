"""
Async Database Connection Manager
==================================
SQLAlchemy 2.0 async engine with connection pooling tuned
for production workloads behind Nginx with multiple replicas.

Key decisions:
- AsyncSession with expire_on_commit=False to avoid lazy-load pitfalls
- Connection pool sized for concurrent inference workload
- Pool pre-ping to detect stale connections after PostgreSQL restarts
- Separate engine for migrations (Alembic uses sync engine)
"""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.config import get_settings

settings = get_settings()

# ── Async Engine ─────────────────────────────────────────────
# pool_size=20: supports 20 concurrent DB connections per worker
# max_overflow=10: allows 10 additional connections under burst
# pool_pre_ping=True: validates connections before checkout
engine = create_async_engine(
    settings.database_url,
    echo=settings.app_debug and not settings.is_production,
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True,
    pool_recycle=3600,  # Recycle connections every hour
)

# ── Session Factory ──────────────────────────────────────────
async_session_factory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,  # Prevent lazy-load issues in async context
    autoflush=False,
)


# ── Base Model ───────────────────────────────────────────────
class Base(DeclarativeBase):
    """Base class for all ORM models. Provides __tablename__ convention."""
    pass


# ── Session Dependency ───────────────────────────────────────
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that yields a scoped database session.
    Automatically commits on success and rolls back on exception.
    """
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ── Lifecycle Helpers ────────────────────────────────────────
async def init_db() -> None:
    """
    Create all tables (for development/testing).
    Production should use Alembic migrations exclusively.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    """Dispose of the connection pool on shutdown."""
    await engine.dispose()
