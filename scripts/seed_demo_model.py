"""
Demo Model Seed Script
=======================
Creates a sample sklearn model and registers it with the platform
via the API. Used for local development and demo purposes.

Usage:
    python scripts/seed_demo_model.py

Prerequisites:
    - The MaaS API must be running at http://localhost:80
    - Or pass --url flag to specify the API base URL
"""

import argparse
import io
import pickle
import sys
from pathlib import Path

import numpy as np
import requests
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def create_iris_classifier() -> tuple[bytes, dict]:
    """
    Train a RandomForest on the Iris dataset and return
    the pickled model bytes and performance metrics.
    """
    # Load and split data
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Evaluate
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # Serialize
    buffer = io.BytesIO()
    pickle.dump(model, buffer)
    model_bytes = buffer.getvalue()

    metrics = {
        "accuracy": round(accuracy, 4),
        "n_samples_train": len(X_train),
        "n_samples_test": len(X_test),
        "n_features": X_train.shape[1],
        "n_classes": len(set(y_train)),
        "feature_names": list(iris.feature_names),
        "target_names": list(iris.target_names),
    }

    return model_bytes, metrics


def create_simple_regressor() -> tuple[bytes, dict]:
    """
    Train a simple LogisticRegression for a second demo model.
    """
    np.random.seed(42)
    X = np.random.randn(200, 3)
    y = (X[:, 0] + X[:, 1] * 2 > 0).astype(int)

    model = LogisticRegression()
    model.fit(X, y)

    accuracy = model.score(X, y)

    buffer = io.BytesIO()
    pickle.dump(model, buffer)
    model_bytes = buffer.getvalue()

    metrics = {
        "accuracy": round(accuracy, 4),
        "n_features": 3,
    }

    return model_bytes, metrics


def seed(base_url: str) -> None:
    """Register demo models via the API."""
    print(f"\n{'='*60}")
    print("  AI MaaS Platform — Demo Model Seeder")
    print(f"{'='*60}\n")
    print(f"API URL: {base_url}\n")

    # ── Check health ─────────────────────────────────────────
    try:
        health = requests.get(f"{base_url}/health", timeout=5)
        print(f"✅ API is healthy: {health.json()['status']}\n")
    except requests.ConnectionError:
        print("❌ Cannot connect to the API. Is it running?")
        sys.exit(1)

    # ── Seed Model 1: Iris Classifier ────────────────────────
    print("📦 Creating Iris Classifier model...")
    model_bytes, metrics = create_iris_classifier()

    # Register model
    response = requests.post(
        f"{base_url}/api/v1/models",
        json={
            "name": "iris-classifier",
            "description": "Random Forest classifier trained on the Iris dataset. "
                          "Predicts flower species from sepal/petal measurements.",
            "model_type": "sklearn",
            "owner": "data-science-team",
            "tags": ["classification", "sklearn", "iris", "demo"],
        },
    )
    if response.status_code == 201:
        model_id = response.json()["id"]
        print(f"   ✅ Model registered: {model_id}")
    else:
        print(f"   ⚠️  Registration response: {response.status_code} - {response.text}")
        # Try to get existing model
        response = requests.get(f"{base_url}/api/v1/models/iris-classifier")
        model_id = response.json()["id"]
        print(f"   ℹ️  Using existing model: {model_id}")

    # Upload version
    print("   📤 Uploading model version...")
    response = requests.post(
        f"{base_url}/api/v1/models/{model_id}/versions",
        files={"file": ("model.pkl", model_bytes, "application/octet-stream")},
        data={
            "framework": "sklearn",
            "version_tag": "v1.0.0",
            "max_batch_size": "64",
        },
    )
    if response.status_code == 201:
        version = response.json()
        print(f"   ✅ Version {version['version_number']} uploaded "
              f"({version['file_size_bytes']} bytes)")
    else:
        print(f"   ⚠️  Upload response: {response.status_code}")

    # Test prediction
    print("   🔮 Testing prediction...")
    response = requests.post(
        f"{base_url}/api/v1/predict",
        json={
            "model_id": "iris-classifier",
            "input_data": [[5.1, 3.5, 1.4, 0.2]],  # Should be Setosa
        },
    )
    if response.status_code == 200:
        pred = response.json()
        print(f"   ✅ Prediction: {pred['predictions']} "
              f"(latency: {pred['latency_ms']}ms)")
    else:
        print(f"   ⚠️  Prediction response: {response.status_code} - {response.text}")

    # ── Seed Model 2: Simple Classifier ──────────────────────
    print("\n📦 Creating Simple Classifier model...")
    model_bytes2, metrics2 = create_simple_regressor()

    response = requests.post(
        f"{base_url}/api/v1/models",
        json={
            "name": "simple-classifier",
            "description": "Logistic Regression binary classifier for demo purposes.",
            "model_type": "sklearn",
            "owner": "ml-platform-team",
            "tags": ["classification", "sklearn", "binary", "demo"],
        },
    )
    if response.status_code == 201:
        model_id2 = response.json()["id"]
        print(f"   ✅ Model registered: {model_id2}")

        response = requests.post(
            f"{base_url}/api/v1/models/{model_id2}/versions",
            files={"file": ("model.pkl", model_bytes2, "application/octet-stream")},
            data={"framework": "sklearn", "version_tag": "v1.0.0"},
        )
        if response.status_code == 201:
            version2 = response.json()
            print(f"   ✅ Version {version2['version_number']} uploaded")
    else:
        print(f"   ⚠️  Registration response: {response.status_code}")

    # ── Summary ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  ✅ Seeding complete!")
    print(f"{'='*60}")
    print(f"\n📖 API Docs:     {base_url}/docs")
    print(f"❤️  Health:       {base_url}/health")
    print(f"📊 Metrics:      {base_url}/metrics")
    print(f"🔮 Predict:      POST {base_url}/api/v1/predict")
    print()
    print("Example cURL command:")
    print(f'  curl -X POST {base_url}/api/v1/predict \\')
    print('    -H "Content-Type: application/json" \\')
    print('    -d \'{"model_id": "iris-classifier", "input_data": [[5.1, 3.5, 1.4, 0.2]]}\'')
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed demo models into MaaS Platform")
    parser.add_argument(
        "--url",
        default="http://localhost:80",
        help="Base URL of the MaaS API (default: http://localhost:80)",
    )
    args = parser.parse_args()
    seed(args.url)
