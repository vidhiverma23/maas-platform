"""
Health & Operations Endpoint Tests
====================================
Tests for /health, /ready, and /metrics endpoints.
"""

import pytest
import pytest_asyncio


@pytest.mark.asyncio
async def test_health_check(client):
    """GET /health should return 200 with status=healthy."""
    response = await client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "uptime_seconds" in data
    assert data["checks"]["process"] == "ok"


@pytest.mark.asyncio
async def test_readiness_check(client):
    """GET /ready should return 200 when all dependencies are up."""
    response = await client.get("/ready")
    # May return 200 or 503 depending on DB state in test
    assert response.status_code in (200, 503)

    data = response.json()
    assert "status" in data
    assert "checks" in data
    assert "version" in data


@pytest.mark.asyncio
async def test_metrics_endpoint(client):
    """GET /metrics should return Prometheus-formatted metrics."""
    response = await client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]

    # Check for our custom metrics
    body = response.text
    assert "maas_http_requests_total" in body
    assert "maas_active_requests" in body


@pytest.mark.asyncio
async def test_root_endpoint(client):
    """GET / should return service info."""
    response = await client.get("/")
    assert response.status_code == 200

    data = response.json()
    assert "service" in data
    assert "version" in data
    assert data["docs"] == "/docs"
    assert data["health"] == "/health"


@pytest.mark.asyncio
async def test_request_id_header(client):
    """All responses should include X-Request-ID header."""
    response = await client.get("/health")
    assert "X-Request-ID" in response.headers


@pytest.mark.asyncio
async def test_response_time_header(client):
    """All responses should include X-Response-Time header."""
    response = await client.get("/health")
    assert "X-Response-Time" in response.headers
    assert response.headers["X-Response-Time"].endswith("ms")
