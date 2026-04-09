# ============================================================
# Multi-Stage Dockerfile — AI MaaS Platform
# ============================================================
# Stage 1: Build dependencies in a separate layer for caching
# Stage 2: Slim runtime image with only what's needed
#
# Build:  docker build -t maas-platform .
# Run:    docker run -p 8000:8000 maas-platform
# ============================================================

# ── Stage 1: Builder ─────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build dependencies for native extensions (asyncpg, hiredis)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: Runtime ─────────────────────────────────────────
FROM python:3.12-slim AS runtime

# Security: run as non-root user
RUN groupadd -r maas && useradd -r -g maas -s /bin/false maas

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY . .

# Create model storage directory
RUN mkdir -p /app/model_storage && chown -R maas:maas /app

# Switch to non-root user
USER maas

# Environment defaults (overridden by docker-compose / Kubernetes)
ENV APP_HOST=0.0.0.0 \
    APP_PORT=8000 \
    APP_WORKERS=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Health check for orchestration
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Run with uvicorn — production config
# Workers controlled by APP_WORKERS env var
CMD uvicorn app.main:app \
    --host ${APP_HOST} \
    --port ${APP_PORT} \
    --workers ${APP_WORKERS} \
    --loop uvloop \
    --http httptools \
    --access-log \
    --log-level info
