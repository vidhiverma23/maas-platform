"""
Model Management API Tests
============================
Tests for model CRUD and version upload endpoints.
"""

import io
import pickle

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression


@pytest.mark.asyncio
async def test_create_model(client):
    """POST /api/v1/models should register a new model."""
    payload = {
        "name": "test-model",
        "description": "A test sklearn model",
        "model_type": "sklearn",
        "owner": "test-team",
        "tags": ["classification", "test"],
    }
    response = await client.post("/api/v1/models", json=payload)
    assert response.status_code == 201

    data = response.json()
    assert data["name"] == "test-model"
    assert data["description"] == "A test sklearn model"
    assert data["model_type"] == "sklearn"
    assert data["owner"] == "test-team"
    assert data["tags"] == ["classification", "test"]
    assert "id" in data
    assert "created_at" in data


@pytest.mark.asyncio
async def test_create_duplicate_model(client):
    """Creating a model with the same name should fail."""
    payload = {"name": "duplicate-model", "model_type": "sklearn"}

    response1 = await client.post("/api/v1/models", json=payload)
    assert response1.status_code == 201

    response2 = await client.post("/api/v1/models", json=payload)
    assert response2.status_code == 500  # IntegrityError


@pytest.mark.asyncio
async def test_list_models(client):
    """GET /api/v1/models should return paginated list."""
    # Create a model first
    await client.post(
        "/api/v1/models",
        json={"name": "list-test-model", "model_type": "sklearn"},
    )

    response = await client.get("/api/v1/models")
    assert response.status_code == 200

    data = response.json()
    assert "items" in data
    assert "total" in data
    assert "page" in data
    assert "page_size" in data
    assert "total_pages" in data
    assert data["total"] >= 1


@pytest.mark.asyncio
async def test_list_models_pagination(client):
    """Pagination should work correctly."""
    # Create multiple models
    for i in range(5):
        await client.post(
            "/api/v1/models",
            json={"name": f"page-model-{i}", "model_type": "sklearn"},
        )

    # Get first page with size 2
    response = await client.get("/api/v1/models?page=1&page_size=2")
    assert response.status_code == 200

    data = response.json()
    assert len(data["items"]) == 2
    assert data["page"] == 1
    assert data["page_size"] == 2


@pytest.mark.asyncio
async def test_get_model(client):
    """GET /api/v1/models/{id} should return model details."""
    # Create a model
    create_response = await client.post(
        "/api/v1/models",
        json={
            "name": "get-test-model",
            "description": "Test description",
            "model_type": "sklearn",
        },
    )
    model_id = create_response.json()["id"]

    # Get by UUID
    response = await client.get(f"/api/v1/models/{model_id}")
    assert response.status_code == 200
    assert response.json()["name"] == "get-test-model"

    # Get by name
    response = await client.get("/api/v1/models/get-test-model")
    assert response.status_code == 200
    assert response.json()["name"] == "get-test-model"


@pytest.mark.asyncio
async def test_get_nonexistent_model(client):
    """Getting a non-existent model should return 404."""
    response = await client.get("/api/v1/models/nonexistent-model")
    assert response.status_code == 404

    data = response.json()
    assert data["error_code"] == "MODEL_NOT_FOUND"


@pytest.mark.asyncio
async def test_update_model(client):
    """PATCH /api/v1/models/{id} should update metadata."""
    # Create a model
    create_response = await client.post(
        "/api/v1/models",
        json={"name": "update-test", "model_type": "sklearn"},
    )
    model_id = create_response.json()["id"]

    # Update
    response = await client.patch(
        f"/api/v1/models/{model_id}",
        json={
            "description": "Updated description",
            "owner": "new-team",
            "tags": ["updated"],
        },
    )
    assert response.status_code == 200

    data = response.json()
    assert data["description"] == "Updated description"
    assert data["owner"] == "new-team"
    assert data["tags"] == ["updated"]


@pytest.mark.asyncio
async def test_delete_model(client):
    """DELETE /api/v1/models/{id} should remove the model."""
    # Create a model
    create_response = await client.post(
        "/api/v1/models",
        json={"name": "delete-test", "model_type": "sklearn"},
    )
    model_id = create_response.json()["id"]

    # Delete
    response = await client.delete(f"/api/v1/models/{model_id}")
    assert response.status_code == 204

    # Verify it's gone
    get_response = await client.get(f"/api/v1/models/{model_id}")
    assert get_response.status_code == 404


@pytest.mark.asyncio
async def test_upload_model_version(client):
    """POST /api/v1/models/{id}/versions should upload a model file."""
    # Create a model
    create_response = await client.post(
        "/api/v1/models",
        json={"name": "upload-test", "model_type": "sklearn"},
    )
    model_id = create_response.json()["id"]

    # Create a sample sklearn model in memory
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 1, 0, 1])
    model = LogisticRegression()
    model.fit(X, y)

    model_bytes = io.BytesIO()
    pickle.dump(model, model_bytes)
    model_bytes.seek(0)

    # Upload
    response = await client.post(
        f"/api/v1/models/{model_id}/versions",
        files={"file": ("model.pkl", model_bytes, "application/octet-stream")},
        data={"framework": "sklearn", "version_tag": "v1.0.0"},
    )
    assert response.status_code == 201

    data = response.json()
    assert data["version_number"] == 1
    assert data["framework"] == "sklearn"
    assert data["format"] == "pkl"
    assert data["status"] == "ready"
    assert data["file_size_bytes"] > 0
    assert data["file_hash"] is not None


@pytest.mark.asyncio
async def test_upload_unsupported_format(client):
    """Uploading an unsupported file format should fail."""
    create_response = await client.post(
        "/api/v1/models",
        json={"name": "format-test", "model_type": "sklearn"},
    )
    model_id = create_response.json()["id"]

    response = await client.post(
        f"/api/v1/models/{model_id}/versions",
        files={"file": ("model.xyz", b"fake data", "application/octet-stream")},
        data={"framework": "sklearn"},
    )
    assert response.status_code == 400
    assert "Unsupported format" in response.json()["message"]
