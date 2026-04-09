"""
Prometheus-Compatible Metrics Collector
=======================================
Lightweight in-process metrics using prometheus_client.
Exposes counters, histograms, and gauges for the /metrics endpoint.

Key decisions:
- Use prometheus_client directly (no middleware overhead)
- Custom histogram buckets tuned for ML inference latencies
- Thread-safe counters for concurrent request tracking
"""

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

# Custom registry to avoid polluting the default global registry
# with metrics from third-party libraries
REGISTRY = CollectorRegistry()

# ── Request Metrics ──────────────────────────────────────────

REQUEST_COUNT = Counter(
    name="maas_http_requests_total",
    documentation="Total number of HTTP requests received",
    labelnames=["method", "endpoint", "status_code"],
    registry=REGISTRY,
)

REQUEST_LATENCY = Histogram(
    name="maas_http_request_duration_seconds",
    documentation="HTTP request latency in seconds",
    labelnames=["method", "endpoint"],
    # Buckets tuned for API + inference latencies:
    # fast responses (5ms) through heavy inference (30s)
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
    registry=REGISTRY,
)

# ── Inference Metrics ────────────────────────────────────────

INFERENCE_COUNT = Counter(
    name="maas_inference_total",
    documentation="Total number of inference requests",
    labelnames=["model_id", "model_type", "status"],
    registry=REGISTRY,
)

INFERENCE_LATENCY = Histogram(
    name="maas_inference_duration_seconds",
    documentation="Model inference latency in seconds (excludes I/O)",
    labelnames=["model_id", "model_type"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 5.0, 10.0),
    registry=REGISTRY,
)

# ── Model Metrics ────────────────────────────────────────────

MODELS_LOADED = Gauge(
    name="maas_models_loaded",
    documentation="Number of models currently loaded in memory",
    registry=REGISTRY,
)

MODEL_LOAD_LATENCY = Histogram(
    name="maas_model_load_duration_seconds",
    documentation="Time to load a model from storage into memory",
    labelnames=["model_type"],
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0),
    registry=REGISTRY,
)

# ── Cache Metrics ────────────────────────────────────────────

CACHE_HITS = Counter(
    name="maas_cache_hits_total",
    documentation="Total number of prediction cache hits",
    registry=REGISTRY,
)

CACHE_MISSES = Counter(
    name="maas_cache_misses_total",
    documentation="Total number of prediction cache misses",
    registry=REGISTRY,
)

# ── System Metrics ───────────────────────────────────────────

ACTIVE_REQUESTS = Gauge(
    name="maas_active_requests",
    documentation="Number of requests currently being processed",
    registry=REGISTRY,
)


def get_metrics() -> bytes:
    """
    Serialize all registered metrics in Prometheus exposition format.
    Called by the /metrics endpoint handler.
    """
    return generate_latest(REGISTRY)
