"""
Inference API Tests
====================
Tests for the /predict endpoint and inference pipeline.
"""

import io
import pickle

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression


async def _create_model_with_version(client) -> tuple[str, str]:
    """
    Helper: Create a model and upload a trained version.
    Returns (model_id, model_name).
    """
    # Register model
    create_response = await client.post(
        "/api/v1/models",
        json={"name": "inference-test-model", "model_type": "sklearn"},
    )
    data = create_response.json()
    model_id = data["id"]
    model_name = data["name"]

    # Train and upload a version
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 1, 0, 1])
    model = LogisticRegression()
    model.fit(X, y)

    model_bytes = io.BytesIO()
    pickle.dump(model, model_bytes)
    model_bytes.seek(0)

    await client.post(
        f"/api/v1/models/{model_id}/versions",
        files={"file": ("model.pkl", model_bytes, "application/octet-stream")},
        data={"framework": "sklearn"},
    )

    return model_id, model_name


@pytest.mark.asyncio
async def test_predict_success(client):
    """POST /api/v1/predict should return predictions."""
    model_id, model_name = await _create_model_with_version(client)

    response = await client.post(
        "/api/v1/predict",
        json={
            "model_id": model_id,
            "input_data": [[1.0, 2.0]],
        },
    )
    assert response.status_code == 200

    data = response.json()
    assert data["model_id"] == model_id
    assert data["model_name"] == model_name
    assert data["version"] == 1
    assert "predictions" in data
    assert len(data["predictions"]) == 1
    assert "latency_ms" in data
    assert data["cached"] is False


@pytest.mark.asyncio
async def test_predict_by_model_name(client):
    """Should be able to predict using model name instead of UUID."""
    model_id, model_name = await _create_model_with_version(client)

    response = await client.post(
        "/api/v1/predict",
        json={
            "model_id": model_name,
            "input_data": [[1.0, 2.0]],
        },
    )
    assert response.status_code == 200
    assert response.json()["model_name"] == model_name


@pytest.mark.asyncio
async def test_predict_batch(client):
    """Should handle batch predictions (multiple samples)."""
    model_id, _ = await _create_model_with_version(client)

    response = await client.post(
        "/api/v1/predict",
        json={
            "model_id": model_id,
            "input_data": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        },
    )
    assert response.status_code == 200

    data = response.json()
    assert len(data["predictions"]) == 3


@pytest.mark.asyncio
async def test_predict_nonexistent_model(client):
    """Predicting on a non-existent model should return 404."""
    response = await client.post(
        "/api/v1/predict",
        json={
            "model_id": "nonexistent-model",
            "input_data": [[1.0, 2.0]],
        },
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_predict_invalid_input(client):
    """Predicting with invalid input should return an error."""
    model_id, _ = await _create_model_with_version(client)

    # Missing input_data
    response = await client.post(
        "/api/v1/predict",
        json={"model_id": model_id},
    )
    assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_predict_includes_latency(client):
    """Response should include latency_ms metadata."""
    model_id, _ = await _create_model_with_version(client)

    response = await client.post(
        "/api/v1/predict",
        json={
            "model_id": model_id,
            "input_data": [[1.0, 2.0]],
        },
    )
    data = response.json()
    assert "latency_ms" in data
    assert isinstance(data["latency_ms"], float)
    assert data["latency_ms"] >= 0
