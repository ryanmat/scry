# Description: Prediction service for loading models and running inference.
# Description: Handles preprocessing, model loading, and cluster assignment.

"""Prediction service for the API."""

from pathlib import Path
from typing import Any

import numpy as np
import torch

from scry.api.schemas import CLUSTER_DEFINITIONS
from scry.model import TemporalXDEC
from scry.utils.tracing import get_tracer

tracer = get_tracer(__name__)


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
        """Load model from checkpoint."""
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

        self.config = checkpoint["config"]
        self.normalization = checkpoint.get("normalization", {"mean": None, "std": None})

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

    def _preprocess_numerical(self, metrics: dict[str, list[float]]) -> torch.Tensor:
        """Preprocess numerical metrics into model input.

        Args:
            metrics: Dict mapping metric names to time series values.

        Returns:
            Tensor of shape (1, seq_len, num_numerical).
        """
        seq_len = self.config["seq_len"]
        num_features = self.config["num_numerical"]

        # Create zero tensor
        data = np.zeros((seq_len, num_features), dtype=np.float32)

        # Fill in available metrics
        for i, (name, values) in enumerate(metrics.items()):
            if i >= num_features:
                break

            values = np.array(values, dtype=np.float32)

            # Truncate or pad
            if len(values) > seq_len:
                # Take most recent values
                values = values[-seq_len:]
            elif len(values) < seq_len:
                # Pad with zeros at the beginning
                padded = np.zeros(seq_len, dtype=np.float32)
                padded[-len(values):] = values
                values = padded

            data[:, i] = values

        # Normalize if we have normalization params
        if self.normalization["mean"] is not None:
            mean = np.array(self.normalization["mean"], dtype=np.float32)
            std = np.array(self.normalization["std"], dtype=np.float32)
            std = np.where(std == 0, 1.0, std)  # Avoid division by zero
            data = (data - mean) / std

        # Add batch dimension and convert to tensor
        tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        return tensor.to(self.device)

    def _preprocess_categorical(self, metrics: dict[str, list[int]]) -> torch.Tensor:
        """Preprocess categorical metrics into model input.

        Args:
            metrics: Dict mapping metric names to time series values (0/1).

        Returns:
            Tensor of shape (1, seq_len, num_categorical).
        """
        seq_len = self.config["seq_len"]
        num_features = self.config["num_categorical"]

        # Create zero tensor
        data = np.zeros((seq_len, num_features), dtype=np.float32)

        # Fill in available metrics
        for i, (name, values) in enumerate(metrics.items()):
            if i >= num_features:
                break

            values = np.array(values, dtype=np.float32)

            # Truncate or pad
            if len(values) > seq_len:
                # Take most recent values
                values = values[-seq_len:]
            elif len(values) < seq_len:
                # Pad with zeros at the beginning
                padded = np.zeros(seq_len, dtype=np.float32)
                padded[-len(values):] = values
                values = padded

            data[:, i] = values

        # Add batch dimension and convert to tensor
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
