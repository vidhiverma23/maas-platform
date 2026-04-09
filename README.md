<p align="center">
  <h1 align="center">🚀 AI Model-as-a-Service (MaaS) Platform</h1>
  <p align="center">
    A production-grade backend system for uploading, registering, deploying,
    and serving machine learning models via REST APIs.
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python" alt="Python" />
  <img src="https://img.shields.io/badge/FastAPI-0.115-green?logo=fastapi" alt="FastAPI" />
  <img src="https://img.shields.io/badge/PostgreSQL-16-blue?logo=postgresql" alt="PostgreSQL" />
  <img src="https://img.shields.io/badge/Redis-7-red?logo=redis" alt="Redis" />
  <img src="https://img.shields.io/badge/Docker-Compose-blue?logo=docker" alt="Docker" />
  <img src="https://img.shields.io/badge/Nginx-LB-green?logo=nginx" alt="Nginx" />
  <img src="https://img.shields.io/badge/Azure-Ready-blue?logo=microsoftazure" alt="Azure" />
  <img src="https://img.shields.io/badge/K8s-Ready-blue?logo=kubernetes" alt="K8s" />
</p>

---

## 📋 Table of Contents

- [System Design](#-system-design)
- [Architecture](#-architecture)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [API Reference](#-api-reference)
- [API Examples (cURL)](#-api-examples-curl)
- [Configuration](#-configuration)
- [Testing](#-testing)
- [Deployment](#-deployment)
- [Monitoring](#-monitoring)

---

## 🧠 System Design

### High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER                              │
│   (Web Apps, Mobile, SDKs, CI/CD Pipelines, Data Scientists)     │
└─────────────────────────────┬────────────────────────────────────┘
                              │ HTTPS
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                    NGINX LOAD BALANCER                            │
│   ┌─────────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│   │ Rate Limit   │  │ Round-Robin  │  │ Health Check Bypass   │  │
│   │ (50r/s API)  │  │ Upstream     │  │ (/health, /metrics)   │  │
│   └─────────────┘  └──────────────┘  └───────────────────────┘  │
└─────────────────────────────┬────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
┌────────────────┐ ┌────────────────┐ ┌────────────────┐
│  FastAPI :8000 │ │  FastAPI :8000 │ │  FastAPI :8000 │
│  (Instance 1)  │ │  (Instance 2)  │ │  (Instance 3)  │
│  ┌──────────┐  │ │  ┌──────────┐  │ │  ┌──────────┐  │
│  │ Middleware│  │ │  │ Middleware│  │ │  │ Middleware│  │
│  │ - ReqID  │  │ │  │ - ReqID  │  │ │  │ - ReqID  │  │
│  │ - Logging│  │ │  │ - Logging│  │ │  │ - Logging│  │
│  │ - Metrics│  │ │  │ - Metrics│  │ │  │ - Metrics│  │
│  └──────────┘  │ │  └──────────┘  │ │  └──────────┘  │
│  ┌──────────┐  │ │  ┌──────────┐  │ │  ┌──────────┐  │
│  │ Services │  │ │  │ Services │  │ │  │ Services │  │
│  │ -Registry│  │ │  │ -Registry│  │ │  │ -Registry│  │
│  │ -Loader  │  │ │  │ -Loader  │  │ │  │ -Loader  │  │
│  │ -Infer.  │  │ │  │ -Infer.  │  │ │  │ -Infer.  │  │
│  │ -Cache   │  │ │  │ -Cache   │  │ │  │ -Cache   │  │
│  └──────────┘  │ │  └──────────┘  │ │  └──────────┘  │
└───────┬────────┘ └───────┬────────┘ └───────┬────────┘
        │                  │                  │
        └──────────────────┼──────────────────┘
                    ┌──────┴──────┐
              ┌─────┴─────┐ ┌────┴─────┐  ┌──────────┐
              │ PostgreSQL│ │  Redis   │  │  Model   │
              │    :5432  │ │  :6379   │  │  Storage │
              │           │ │          │  │  (Volume)│
              │ - Models  │ │ - Cache  │  │          │
              │ - Versions│ │ - Rate   │  │ - .pkl   │
              │ - Metadata│ │   Limit  │  │ - .onnx  │
              │           │ │          │  │ - .pt    │
              └───────────┘ └──────────┘  └──────────┘
```

### Request Flow (Inference)

```
Client Request → Nginx (rate limit + LB)
    → FastAPI Middleware (request ID, logging, metrics)
        → Rate Limiter Check (Redis)
        → Cache Lookup (Redis)
            → HIT:  Return cached prediction
            → MISS: Continue ↓
        → Model Resolution (PostgreSQL)
        → Model Loading (LRU memory cache + disk)
        → Inference Engine (thread pool executor)
        → Cache Store (Redis, TTL=300s)
    → Response (with latency, request ID headers)
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Async FastAPI** | Non-blocking I/O for high concurrency; ML inference offloaded to thread pool |
| **LRU Model Cache** | Avoid reloading models from disk on every request; TTL eviction prevents OOM |
| **Redis Dual-Use** | Single Redis instance for both prediction caching and distributed rate limiting |
| **Multi-Format Support** | Unified interface across sklearn/ONNX/PyTorch for team flexibility |
| **Token-Bucket Rate Limiter** | Redis-based for consistency across replicas; fail-open for availability |
| **UUID Primary Keys** | No sequential ID guessing; safe for distributed systems |
| **SHA-256 File Hashing** | Model integrity verification without relying on filename |
| **Prometheus Metrics** | Industry-standard; integrates with Grafana, Azure Monitor, Datadog |

---

## ✨ Features

### Core
- ✅ Upload ML models (pickle, ONNX, PyTorch, joblib)
- ✅ Register models with metadata (name, version, type, tags)
- ✅ Model versioning with status lifecycle (registered → ready → deprecated)
- ✅ Dynamic model loading at runtime with LRU memory cache
- ✅ REST API endpoint `/predict` with model_id-based routing
- ✅ Batch prediction support

### Performance
- ✅ Fully async FastAPI endpoints
- ✅ Redis prediction caching (SHA-256 cache keys)
- ✅ Thread pool executor for CPU-bound inference
- ✅ Connection pooling (database + Redis)

### Infrastructure
- ✅ Multi-stage Dockerfile (slim production image)
- ✅ Docker Compose with 3 API replicas + PostgreSQL + Redis + Nginx
- ✅ Nginx load balancing (least-conn algorithm)
- ✅ Kubernetes manifests (Deployment, Service, HPA, ConfigMap, Secret)
- ✅ Azure-ready deployment structure

### Observability
- ✅ Structured JSON logging (structlog)
- ✅ Prometheus metrics endpoint (`/metrics`)
- ✅ Request latency tracking (histogram)
- ✅ Request ID propagation (X-Request-ID header)
- ✅ Health check (`/health`) and readiness probe (`/ready`)

### Security
- ✅ Distributed rate limiting (Redis-backed, per-IP)
- ✅ Nginx-level rate limiting (defense in depth)
- ✅ Non-root Docker user
- ✅ Input validation (Pydantic v2 schemas)
- ✅ File size limits for uploads

---

## 🛠 Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **API Framework** | FastAPI 0.115 | Async REST API with auto-docs |
| **Language** | Python 3.12 | Type hints, async/await |
| **Database** | PostgreSQL 16 | Model metadata storage |
| **Async Driver** | asyncpg | High-performance PostgreSQL driver |
| **ORM** | SQLAlchemy 2.0 | Async ORM with connection pooling |
| **Migrations** | Alembic | Schema version management |
| **Cache/Queue** | Redis 7 | Prediction caching + rate limiting |
| **Validation** | Pydantic v2 | Request/response schema validation |
| **ML Runtimes** | sklearn, ONNX, PyTorch | Multi-framework inference |
| **Logging** | structlog | Structured JSON logging |
| **Metrics** | prometheus_client | Prometheus-compatible metrics |
| **Load Balancer** | Nginx | Traffic distribution + rate limiting |
| **Containerization** | Docker + Compose | Multi-service orchestration |
| **Orchestration** | Kubernetes | Production auto-scaling |
| **CI/CD** | GitHub Actions | Automated testing + deployment |
| **Cloud** | Azure | Container deployment target |

---

## 📁 Project Structure

```
AI Model-as-a-Service (MaaS) Platform/
├── app/                        # Application source code
│   ├── __init__.py
│   ├── main.py                 # FastAPI app factory + lifespan
│   ├── config.py               # Pydantic-settings configuration
│   ├── api/                    # API layer (routes + middleware)
│   │   ├── dependencies.py     # Dependency injection wiring
│   │   ├── middleware.py       # Request logging, error handling
│   │   └── routes/
│   │       ├── health.py       # /health, /ready, /metrics
│   │       ├── models.py       # Model CRUD + version upload
│   │       └── inference.py    # /predict endpoint
│   ├── models/                 # Data models
│   │   ├── database.py         # SQLAlchemy ORM models
│   │   └── schemas.py          # Pydantic request/response schemas
│   ├── services/               # Business logic layer
│   │   ├── model_registry.py   # Model CRUD + file storage
│   │   ├── model_loader.py     # LRU model loading cache
│   │   ├── inference_engine.py # Multi-framework inference
│   │   └── cache_service.py    # Redis prediction caching
│   ├── database/
│   │   └── connection.py       # Async SQLAlchemy engine
│   └── utils/
│       ├── logger.py           # Structured logging setup
│       ├── metrics.py          # Prometheus metrics registry
│       ├── rate_limiter.py     # Redis rate limiter
│       └── exceptions.py       # Custom exception hierarchy
├── infrastructure/
│   ├── nginx/nginx.conf        # Nginx load balancer config
│   └── kubernetes/             # K8s deployment manifests
│       ├── deployment.yaml
│       ├── service.yaml
│       └── hpa.yaml
├── migrations/                 # Alembic database migrations
│   ├── env.py
│   └── versions/001_initial.py
├── tests/                      # Test suite
│   ├── conftest.py             # Shared fixtures
│   ├── test_health.py
│   ├── test_models.py
│   └── test_inference.py
├── scripts/
│   └── seed_demo_model.py      # Demo data seeder
├── .github/workflows/ci.yml   # CI/CD pipeline
├── Dockerfile                  # Multi-stage build
├── docker-compose.yml          # Full stack orchestration
├── requirements.txt            # Pinned dependencies
├── alembic.ini                 # Migration config
├── pyproject.toml              # Pytest config
├── .env.example                # Environment template
└── README.md                   # This file
```

---

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose installed
- Python 3.12+ (for local development)
- 4GB+ RAM available

### Option 1: Docker Compose (Recommended)

```bash
# 1. Clone the repository
cd "AI Model-as-a-Service (MaaS) Platform"

# 2. Copy environment config
cp .env.example .env

# 3. Start the full stack (3 API replicas + PostgreSQL + Redis + Nginx)
docker-compose up --build

# 4. Wait for services to be healthy, then seed demo models
python scripts/seed_demo_model.py --url http://localhost:80

# 5. Open API docs
open http://localhost:80/docs
```

### Option 2: Local Development

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start PostgreSQL and Redis (via Docker)
docker run -d --name maas-postgres \
  -e POSTGRES_USER=maas_user \
  -e POSTGRES_PASSWORD=maas_secret_password \
  -e POSTGRES_DB=maas_db \
  -p 5432:5432 postgres:16-alpine

docker run -d --name maas-redis -p 6379:6379 redis:7-alpine

# 4. Configure environment
cp .env.example .env
# Edit .env: set POSTGRES_HOST=localhost, REDIS_HOST=localhost

# 5. Run the application
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 6. Open API docs
open http://localhost:8000/docs
```

---

## 📡 API Reference

### Operations

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness probe |
| `GET` | `/ready` | Readiness probe (checks DB + Redis) |
| `GET` | `/metrics` | Prometheus metrics |
| `GET` | `/docs` | Swagger UI |

### Model Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/models` | Register a new model |
| `GET` | `/api/v1/models` | List models (paginated) |
| `GET` | `/api/v1/models/{id}` | Get model details |
| `PATCH` | `/api/v1/models/{id}` | Update model metadata |
| `DELETE` | `/api/v1/models/{id}` | Delete model + versions |
| `POST` | `/api/v1/models/{id}/versions` | Upload model version |
| `GET` | `/api/v1/models/{id}/versions/{v}` | Get version details |
| `PATCH` | `/api/v1/models/{id}/versions/{v}/status` | Update version status |

### Inference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/predict` | Run model prediction |

---

## 📝 API Examples (cURL)

### Register a Model

```bash
curl -X POST http://localhost:80/api/v1/models \
  -H "Content-Type: application/json" \
  -d '{
    "name": "fraud-detector",
    "description": "XGBoost fraud detection model for transactions",
    "model_type": "sklearn",
    "owner": "risk-team",
    "tags": ["classification", "fraud", "production"]
  }'
```

### Upload a Model Version

```bash
curl -X POST http://localhost:80/api/v1/models/fraud-detector/versions \
  -F "file=@model.pkl" \
  -F "framework=sklearn" \
  -F "version_tag=v1.0.0" \
  -F "max_batch_size=64"
```

### Run Prediction

```bash
# By model name (uses latest version)
curl -X POST http://localhost:80/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "fraud-detector",
    "input_data": [[100.0, 1.5, 0.0, 42.0]],
    "parameters": {"return_probabilities": true}
  }'

# By model ID with specific version
curl -X POST http://localhost:80/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "version": 2,
    "input_data": [[100.0, 1.5, 0.0, 42.0]]
  }'
```

### Batch Prediction

```bash
curl -X POST http://localhost:80/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "iris-classifier",
    "input_data": [
      [5.1, 3.5, 1.4, 0.2],
      [6.7, 3.0, 5.2, 2.3],
      [5.8, 2.7, 4.1, 1.0]
    ]
  }'
```

### List Models (Paginated)

```bash
curl "http://localhost:80/api/v1/models?page=1&page_size=10&model_type=sklearn"
```

### Health & Metrics

```bash
# Liveness
curl http://localhost:80/health

# Readiness (checks DB + Redis)
curl http://localhost:80/ready

# Prometheus metrics
curl http://localhost:80/metrics
```

---

## ⚙️ Configuration

All configuration is done via environment variables. See `.env.example` for the complete list.

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_ENV` | `development` | Environment (development/staging/production) |
| `APP_WORKERS` | `1` | Uvicorn worker count |
| `DATABASE_URL` | `postgresql+asyncpg://...` | PostgreSQL connection string |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection string |
| `MODEL_STORAGE_PATH` | `./model_storage` | Model file storage directory |
| `MAX_MODEL_SIZE_MB` | `500` | Maximum upload file size |
| `RATE_LIMIT_REQUESTS` | `100` | Requests per window |
| `RATE_LIMIT_WINDOW_SECONDS` | `60` | Rate limit window duration |
| `CACHE_TTL_SECONDS` | `300` | Prediction cache TTL |
| `LOG_FORMAT` | `json` | Log output format (json/console) |

---

## 🧪 Testing

```bash
# Install test dependencies
pip install -r requirements.txt
pip install aiosqlite

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=app --cov-report=term-missing

# Run specific test file
pytest tests/test_health.py -v

# Run specific test
pytest tests/test_models.py::test_create_model -v
```

---

## 🚢 Deployment

### Azure Container Instances

```bash
# Build and push to ACR
az acr build --registry <acr-name> --image maas-platform:latest .

# Deploy
az container create \
  --resource-group maas-rg \
  --name maas-api \
  --image <acr-name>.azurecr.io/maas-platform:latest \
  --cpu 2 --memory 4 \
  --ports 8000 \
  --environment-variables APP_ENV=production APP_WORKERS=4
```

### Kubernetes

```bash
# Create namespace
kubectl create namespace maas

# Apply all manifests
kubectl apply -f infrastructure/kubernetes/

# Check deployment status
kubectl get pods -n maas
kubectl get hpa -n maas
```

---

## 📊 Monitoring

### Prometheus Metrics Available

| Metric | Type | Description |
|--------|------|-------------|
| `maas_http_requests_total` | Counter | Total HTTP requests |
| `maas_http_request_duration_seconds` | Histogram | Request latency |
| `maas_inference_total` | Counter | Inference requests |
| `maas_inference_duration_seconds` | Histogram | Inference latency |
| `maas_models_loaded` | Gauge | Models in memory cache |
| `maas_cache_hits_total` | Counter | Cache hits |
| `maas_cache_misses_total` | Counter | Cache misses |
| `maas_active_requests` | Gauge | In-flight requests |

### Response Headers

Every response includes:
- `X-Request-ID` — Unique request identifier for tracing
- `X-Response-Time` — Request processing time in milliseconds

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  Built with ❤️ for production ML infrastructure
</p>
