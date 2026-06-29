# Description: Prediction service for loading models and running inference.
# Description: Handles preprocessing, model loading, and cluster assignment.

"""Prediction service for the API."""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from scry.api.schemas import CLUSTER_DEFINITIONS
from scry.model import TemporalXDEC
from scry.utils.tracing import get_tracer

tracer = get_tracer(__name__)
logger = logging.getLogger(__name__)


class ModelSchemaError(ValueError):
    """Raised when a checkpoint has no usable feature schema for by-name alignment."""


class Predictor:
    """Prediction service for cluster assignment.

    Loads a trained X-DEC model and provides prediction methods.

    Attributes:
        model: The loaded TemporalXDEC model.
        device: Device the model runs on (cpu, cuda, mps).
        config: Model configuration from checkpoint.
        normalization: Normalization parameters (mean, std).
        is_loaded: Whether a model is loaded.
    """

    def __init__(self, model_path: str):
        """Initialize predictor with a trained model.

        Args:
            model_path: Path to the saved model checkpoint.

        Raises:
            FileNotFoundError: If model file doesn't exist.
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.device = self._detect_device()
        self._load_model()
        self.is_loaded = True

    def _detect_device(self) -> str:
        """Detect best available device.

        Returns:
            Device string (cuda, mps, or cpu).
        """
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_model(self) -> None:
        """Load model from checkpoint.

        Raises:
            ModelSchemaError: If the checkpoint has no feature schema, or the
                schema is inconsistent with the model dimensions. Without a
                schema, incoming metrics cannot be aligned by name and
                predictions would be unreliable.
        """
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

        self.config = checkpoint["config"]
        self.normalization = checkpoint.get("normalization", {"mean": None, "std": None})
        self.cat_normalization = checkpoint.get("categorical_normalization")

        schema = checkpoint.get("feature_schema")
        if not schema or "numerical" not in schema or "categorical" not in schema:
            raise ModelSchemaError(
                "Model checkpoint has no feature_schema. It was trained before Scry "
                "persisted feature names, so incoming metrics cannot be aligned by "
                "name and predictions would be unreliable. Retrain with the current "
                "scripts/extract_features.py and scripts/train_model.py."
            )
        self.feature_schema = schema
        self.numerical_features = list(schema["numerical"])
        self.categorical_features = list(schema["categorical"])

        # The schema order must match the model dimensions, or by-name placement
        # would write into the wrong columns.
        if len(self.numerical_features) != self.config["num_numerical"]:
            raise ModelSchemaError(
                f"feature_schema numerical count ({len(self.numerical_features)}) "
                f"does not match model num_numerical ({self.config['num_numerical']}). "
                "Retrain the model."
            )
        if len(self.categorical_features) != self.config["num_categorical"]:
            raise ModelSchemaError(
                f"feature_schema categorical count ({len(self.categorical_features)}) "
                f"does not match model num_categorical ({self.config['num_categorical']}). "
                "Retrain the model."
            )
        if self.categorical_features and self.cat_normalization is None:
            raise ModelSchemaError(
                "Model checkpoint has a categorical feature_schema but no "
                "categorical_normalization params. Retrain the model."
            )

        self._num_index = {name: i for i, name in enumerate(self.numerical_features)}
        self._cat_index = {name: i for i, name in enumerate(self.categorical_features)}

        # Create model with saved config
        self.model = TemporalXDEC(
            num_numerical=self.config["num_numerical"],
            num_categorical=self.config["num_categorical"],
            seq_len=self.config["seq_len"],
            num_hidden=self.config["num_hidden"],
            cat_hidden=self.config["cat_hidden"],
            latent_dim=self.config["latent_dim"],
            n_clusters=self.config["n_clusters"],
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _fit_length(values: np.ndarray, seq_len: int) -> np.ndarray:
        """Fit a 1-D series to seq_len.

        Takes the most recent ``seq_len`` values, or pads with zeros at the
        beginning when the series is shorter.
        """
        n = len(values)
        if n == seq_len:
            return values
        if n > seq_len:
            return values[-seq_len:]
        padded = np.zeros(seq_len, dtype=np.float32)
        if n:
            padded[-n:] = values
        return padded

    def _preprocess_numerical(self, metrics: dict[str, list[float]]) -> torch.Tensor:
        """Preprocess numerical metrics into model input, aligned by name.

        Each incoming series is placed into the column its name occupies in the
        model's feature schema. Unknown names are ignored; schema features absent
        from the input are left at the feature mean (normalized 0).

        Args:
            metrics: Dict mapping metric names to time series values.

        Returns:
            Tensor of shape (1, seq_len, num_numerical).
        """
        seq_len = self.config["seq_len"]
        num_features = self.config["num_numerical"]

        data = np.zeros((seq_len, num_features), dtype=np.float32)
        filled = np.zeros(num_features, dtype=bool)

        for name, values in metrics.items():
            col = self._num_index.get(name)
            if col is None or len(values) == 0:
                continue
            # Mirror training NaN handling (forward/backward fill, then zero).
            series = pd.Series(np.asarray(values, dtype=np.float32))
            series = series.ffill().bfill().fillna(0.0)
            data[:, col] = self._fit_length(series.to_numpy(dtype=np.float32), seq_len)
            filled[col] = True

        # Normalize using the persisted per-feature params.
        if self.normalization["mean"] is not None:
            mean = np.asarray(self.normalization["mean"], dtype=np.float32)
            std = np.asarray(self.normalization["std"], dtype=np.float32)
            std = np.where(std == 0, 1.0, std)  # Avoid division by zero
            data = (data - mean) / std

        # Genuinely missing features map to the feature mean (normalized 0), the
        # neutral input, rather than to -mean/std from a raw-zero column.
        missing = np.where(~filled)[0]
        if missing.size:
            data[:, missing] = 0.0

        tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        return tensor.to(self.device)

    def _preprocess_categorical(self, metrics: dict[str, list[int]]) -> torch.Tensor:
        """Preprocess categorical metrics into model input, aligned by name.

        Applies the persisted per-feature min/max encoding so serving matches
        training. Unknown names are ignored; schema features absent from the
        input stay at 0, the default state.

        Args:
            metrics: Dict mapping metric names to time series values (0/1).

        Returns:
            Tensor of shape (1, seq_len, num_categorical).
        """
        seq_len = self.config["seq_len"]
        num_features = self.config["num_categorical"]

        data = np.zeros((seq_len, num_features), dtype=np.float32)

        cat_min = cat_max = None
        if self.cat_normalization is not None:
            cat_min = np.asarray(self.cat_normalization["min"], dtype=np.float32)
            cat_max = np.asarray(self.cat_normalization["max"], dtype=np.float32)

        for name, values in metrics.items():
            col = self._cat_index.get(name)
            if col is None or len(values) == 0:
                continue
            arr = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0)
            arr = self._fit_length(arr, seq_len)
            if cat_min is not None:
                lo, hi = cat_min[col], cat_max[col]
                if hi > lo:
                    arr = (arr - lo) / (hi - lo)
                else:
                    # All-same training value: replicate the encode fallback.
                    arr = np.full_like(arr, 1.0 if hi > 0 else 0.0)
                arr = np.clip(arr, 0.0, 1.0)
            data[:, col] = arr

        tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        return tensor.to(self.device)

    def predict(
        self,
        numerical_metrics: dict[str, list[float]],
        categorical_metrics: dict[str, list[int]],
    ) -> dict[str, Any]:
        """Run prediction on input metrics.

        Args:
            numerical_metrics: Dict of numerical metric time series.
            categorical_metrics: Dict of categorical metric time series.

        Returns:
            Dict with cluster_id, cluster_name, confidence, action, priority.
        """
        with tracer.start_as_current_span("predict") as span:
            span.set_attribute("numerical_metrics.count", len(numerical_metrics))
            span.set_attribute("categorical_metrics.count", len(categorical_metrics))

            # Preprocess inputs
            with tracer.start_as_current_span("preprocess_numerical") as num_span:
                num_span.set_attribute("features.count", len(numerical_metrics))
                x_num = self._preprocess_numerical(numerical_metrics)

            with tracer.start_as_current_span("preprocess_categorical") as cat_span:
                cat_span.set_attribute("features.count", len(categorical_metrics))
                x_cat = self._preprocess_categorical(categorical_metrics)

            # Feature coverage: which schema features the request actually supplied.
            # A present-but-empty series is treated as missing, matching preprocessing.
            num_missing = [n for n in self.numerical_features if not numerical_metrics.get(n)]
            cat_missing = [n for n in self.categorical_features if not categorical_metrics.get(n)]
            span.set_attribute(
                "features.numerical.present", len(self.numerical_features) - len(num_missing)
            )
            span.set_attribute("features.numerical.missing", len(num_missing))
            span.set_attribute(
                "features.categorical.present",
                len(self.categorical_features) - len(cat_missing),
            )
            span.set_attribute("features.categorical.missing", len(cat_missing))
            if num_missing or cat_missing:
                logger.debug(
                    "prediction with partial coverage: %d/%d numerical missing, "
                    "%d/%d categorical missing",
                    len(num_missing), len(self.numerical_features),
                    len(cat_missing), len(self.categorical_features),
                )

            # Run inference
            with tracer.start_as_current_span("model_inference") as inf_span:
                inf_span.set_attribute("model.device", str(self.device))
                inf_span.set_attribute("model.n_clusters", self.config["n_clusters"])

                with torch.no_grad():
                    outputs = self.model(x_num, x_cat)
                    q = outputs["q"]  # Soft cluster assignments

                    # Get hard assignment and confidence
                    confidence, cluster_id = q.max(dim=1)
                    cluster_id = cluster_id.item()
                    confidence = confidence.item()

                inf_span.set_attribute("result.cluster_id", cluster_id)
                inf_span.set_attribute("result.confidence", confidence)

            # Get cluster info
            cluster_info = CLUSTER_DEFINITIONS[cluster_id]
            span.set_attribute("result.cluster_name", cluster_info.name)
            span.set_attribute("result.priority", cluster_info.priority)

            return {
                "cluster_id": cluster_id,
                "cluster_name": cluster_info.name,
                "confidence": confidence,
                "action": cluster_info.action,
                "priority": cluster_info.priority,
            }

    def get_embedding(
        self,
        numerical_metrics: dict[str, list[float]],
        categorical_metrics: dict[str, list[int]],
    ) -> np.ndarray:
        """Get latent embedding for input metrics.

        Uses the mean (mu) of the latent distribution for deterministic results.

        Args:
            numerical_metrics: Dict of numerical metric time series.
            categorical_metrics: Dict of categorical metric time series.

        Returns:
            Numpy array of shape (latent_dim,).
        """
        with tracer.start_as_current_span("get_embedding") as span:
            span.set_attribute("latent_dim", self.config["latent_dim"])

            # Preprocess inputs
            x_num = self._preprocess_numerical(numerical_metrics)
            x_cat = self._preprocess_categorical(categorical_metrics)

            # Get embedding using mu (deterministic) instead of sampled z
            with tracer.start_as_current_span("encode_latent"):
                with torch.no_grad():
                    _, mu, _ = self.model.xvae.encode(x_num, x_cat)

            return mu.cpu().numpy().squeeze()
