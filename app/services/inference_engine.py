"""
Inference Engine
================
Unified prediction interface that dispatches inference to the
appropriate runtime based on model framework type.

Design:
- Single `predict()` entry point accepts model + input
- Framework-specific handlers normalize input/output
- Runs inference in thread pool (CPU-bound work off event loop)
- Metrics instrumentation on every prediction
- Support for both single and batch predictions
"""

import asyncio
import time
from typing import Any

import numpy as np

from app.models.database import MLModelVersion
from app.utils.exceptions import InferenceError, InvalidInputError
from app.utils.logger import get_logger
from app.utils.metrics import INFERENCE_COUNT, INFERENCE_LATENCY

logger = get_logger(__name__)


class InferenceEngine:
    """
    Stateless inference engine.
    Given a loaded model object and input data, produces predictions.
    """

    async def predict(
        self,
        model: Any,
        version: MLModelVersion,
        input_data: list[list[float]] | dict[str, Any],
        parameters: dict[str, Any] | None = None,
    ) -> list[Any]:
        """
        Run inference on the given model.

        Args:
            model: Loaded model object (sklearn estimator, ONNX session, etc.)
            version: Model version metadata (for framework dispatch)
            input_data: Raw input from the request
            parameters: Optional inference parameters

        Returns:
            List of predictions
        """
        framework = version.framework.lower()
        model_id = str(version.model_id)
        start = time.monotonic()

        try:
            # Validate and normalize input
            processed_input = self._preprocess_input(input_data, framework)

            # Dispatch to framework-specific handler in thread pool
            loop = asyncio.get_event_loop()
            predictions = await loop.run_in_executor(
                None,
                self._dispatch_predict,
                model,
                framework,
                processed_input,
                parameters or {},
            )

            elapsed = time.monotonic() - start

            # Record success metrics
            INFERENCE_COUNT.labels(
                model_id=model_id,
                model_type=framework,
                status="success",
            ).inc()
            INFERENCE_LATENCY.labels(
                model_id=model_id,
                model_type=framework,
            ).observe(elapsed)

            logger.info(
                "inference_completed",
                model_id=model_id,
                framework=framework,
                input_shape=self._describe_shape(processed_input),
                output_count=len(predictions),
                latency_ms=round(elapsed * 1000, 2),
            )
            return predictions

        except (InferenceError, InvalidInputError):
            INFERENCE_COUNT.labels(
                model_id=model_id, model_type=framework, status="error"
            ).inc()
            raise
        except Exception as e:
            INFERENCE_COUNT.labels(
                model_id=model_id, model_type=framework, status="error"
            ).inc()
            raise InferenceError(model_id, str(e))

    # ── Input Preprocessing ──────────────────────────────────

    def _preprocess_input(
        self,
        input_data: list[list[float]] | dict[str, Any],
        framework: str,
    ) -> np.ndarray | dict[str, Any]:
        """
        Convert raw input to the format expected by each framework.
        - sklearn: numpy 2D array
        - ONNX: numpy 2D array (float32)
        - PyTorch: numpy array (will be converted to tensor in handler)
        """
        if isinstance(input_data, dict):
            # Structured input — pass through for frameworks that support it
            return input_data

        try:
            arr = np.array(input_data, dtype=np.float32)

            # Ensure 2D: single sample → batch of 1
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            elif arr.ndim != 2:
                raise InvalidInputError(
                    f"Expected 1D or 2D input, got {arr.ndim}D array"
                )

            return arr

        except (ValueError, TypeError) as e:
            raise InvalidInputError(
                f"Cannot convert input to numeric array: {e}"
            )

    # ── Framework Dispatch ───────────────────────────────────

    def _dispatch_predict(
        self,
        model: Any,
        framework: str,
        input_data: np.ndarray | dict,
        parameters: dict[str, Any],
    ) -> list[Any]:
        """
        Synchronous prediction dispatch (runs in thread pool).
        Routes to the correct handler based on framework type.
        """
        if framework == "sklearn":
            return self._predict_sklearn(model, input_data, parameters)
        elif framework == "onnx":
            return self._predict_onnx(model, input_data, parameters)
        elif framework == "pytorch":
            return self._predict_pytorch(model, input_data, parameters)
        else:
            raise InferenceError(
                "unknown", f"Unsupported framework: {framework}"
            )

    @staticmethod
    def _predict_sklearn(
        model: Any,
        input_data: np.ndarray,
        parameters: dict[str, Any],
    ) -> list[Any]:
        """
        Sklearn prediction.
        Supports both predict() and predict_proba() via parameters.
        """
        try:
            if parameters.get("return_probabilities", False):
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(input_data)
                    return probs.tolist()
                else:
                    # Fallback to predict if model doesn't support probabilities
                    pass

            predictions = model.predict(input_data)
            return predictions.tolist()

        except Exception as e:
            raise InferenceError("sklearn", f"Prediction failed: {e}")

    @staticmethod
    def _predict_onnx(
        model: Any,  # ort.InferenceSession
        input_data: np.ndarray,
        parameters: dict[str, Any],
    ) -> list[Any]:
        """
        ONNX Runtime prediction.
        Automatically detects input name from the model's metadata.
        """
        try:
            input_name = model.get_inputs()[0].name
            result = model.run(None, {input_name: input_data})

            # ONNX returns a list of outputs — take the first
            predictions = result[0]
            if isinstance(predictions, np.ndarray):
                return predictions.tolist()
            return list(predictions)

        except Exception as e:
            raise InferenceError("onnx", f"ONNX inference failed: {e}")

    @staticmethod
    def _predict_pytorch(
        model: Any,  # torch.jit.ScriptModule
        input_data: np.ndarray,
        parameters: dict[str, Any],
    ) -> list[Any]:
        """
        PyTorch (TorchScript) prediction.
        Runs in eval mode with no_grad for inference efficiency.
        """
        try:
            import torch

            tensor = torch.from_numpy(input_data).float()

            with torch.no_grad():
                output = model(tensor)

            if isinstance(output, torch.Tensor):
                return output.numpy().tolist()
            return list(output)

        except ImportError:
            raise InferenceError("pytorch", "torch not installed")
        except Exception as e:
            raise InferenceError("pytorch", f"PyTorch inference failed: {e}")

    # ── Utilities ────────────────────────────────────────────

    @staticmethod
    def _describe_shape(data: np.ndarray | dict) -> str:
        """Describe input shape for logging."""
        if isinstance(data, np.ndarray):
            return str(data.shape)
        elif isinstance(data, dict):
            return f"dict({len(data)} keys)"
        return "unknown"


# ── Singleton ────────────────────────────────────────────────
inference_engine = InferenceEngine()
