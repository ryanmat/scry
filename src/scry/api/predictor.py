# Description: Prediction service for loading models and running inference.
# Description: Handles preprocessing, model loading, and cluster assignment.

"""Prediction service for the API."""

import logging
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from scry.api.schemas import CLUSTER_DEFINITIONS
from scry.data.windowing import build_windows
from scry.model import TemporalXDEC
from scry.model.reconstruction import reconstruction_errors
from scry.utils.tracing import get_tracer

tracer = get_tracer(__name__)
logger = logging.getLogger(__name__)


class ModelSchemaError(ValueError):
    """Raised when a checkpoint has no usable feature schema for by-name alignment."""


def _severity_from_ratio(ratio: float) -> int:
    """Map a reconstruction ratio (error / threshold) to a 1-4 severity band.

    1: at/below threshold; 2: 1-1.5x; 3: 1.5-2x; 4: above 2x. The ``Scry_Anomaly``
    datapoint alerts warn at ratio > 1 and error at ratio > 2, so severity >= 2 is
    the warn band and severity 4 is the error band.
    """
    if ratio > 2.0:
        return 4
    if ratio > 1.5:
        return 3
    if ratio > 1.0:
        return 2
    return 1


def _frame_from_series(
    numerical_metrics: dict[str, list[float]],
    categorical_metrics: dict[str, list[int]],
) -> pd.DataFrame:
    """Build a canonical long-format frame from parallel request series.

    The series are treated as parallel samples aligned at the end (most recent
    last) on a shared synthetic 1-minute grid, so unequal-length series land on one
    timeline the way a real capture would, rather than each being sliced
    independently. This is what lets the request path window and normalize
    identically to the bake/validation path.
    """
    columns = ["resource_id", "metric_name", "timestamp", "value"]
    series = {**numerical_metrics, **categorical_metrics}
    lengths = [len(v) for v in series.values() if v]
    if not lengths:
        return pd.DataFrame(columns=columns)

    grid_len = max(lengths)
    base = pd.Timestamp("2000-01-01T00:00:00Z")
    grid = base + pd.to_timedelta(np.arange(grid_len), unit="m")

    rows = []
    for name, values in series.items():
        if not values:
            continue
        offset = grid_len - len(values)  # end-align: last sample is most recent
        for i, value in enumerate(values):
            rows.append(
                {
                    "resource_id": "request",
                    "metric_name": name,
                    "timestamp": grid[offset + i],
                    "value": float(value),
                }
            )
    return pd.DataFrame(rows, columns=columns)


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

        # Serving block (persisted by scripts/bake_serving_threshold.py) carries the
        # healthy reconstruction threshold. An env override lets an operator retune
        # it without re-baking. Absent both, the reconstruction endpoint still runs
        # and returns the raw error with a null ratio.
        self.serving = checkpoint.get("serving")
        # The env override is parsed exactly once here: it is an operator retune fixed
        # for the process lifetime, and re-parsing per request would log a warning on
        # every scored request when the value is invalid.
        self._recon_env_threshold = self._env_recon_threshold()
        self.recon_threshold = (
            self._recon_env_threshold
            if self._recon_env_threshold is not None
            else self._serving_recon_threshold(self.serving)
        )
        # Per-resource thresholds (baked via --per-resource-margin) refine the global
        # for resources whose identity the caller resolved; the env override, being an
        # explicit operator retune, beats both.
        self.recon_thresholds_per_resource = self._resolve_per_resource_thresholds(self.serving)
        if self.recon_threshold is None:
            logger.warning(
                "no reconstruction threshold configured (no serving block and no "
                "SCRY_RECON_THRESHOLD); /anomaly/reconstruction will report a null ratio "
                "and never flag anomalies until a threshold is baked via "
                "scripts/bake_serving_threshold.py"
            )

    @staticmethod
    def _env_recon_threshold() -> float | None:
        """The validated SCRY_RECON_THRESHOLD env override, or None when unset/invalid."""
        env_value = os.environ.get("SCRY_RECON_THRESHOLD")
        if env_value is None:
            return None
        try:
            parsed = float(env_value)
        except ValueError:
            logger.warning("ignoring invalid SCRY_RECON_THRESHOLD=%r (not a float)", env_value)
            return None
        if parsed > 0:
            return parsed
        logger.warning("ignoring non-positive SCRY_RECON_THRESHOLD=%r", env_value)
        return None

    @staticmethod
    def _serving_recon_threshold(serving: dict[str, Any] | None) -> float | None:
        """The serving block's global threshold, or None when absent or unusable.

        A non-positive value is rejected with a warning, since the reconstruction
        error is a non-negative MSE and a threshold <= 0 cannot discriminate. This
        keeps ``/health/detailed`` and the scoring path in agreement rather than
        advertising a threshold the scorer would ignore.
        """
        if serving and serving.get("threshold") is not None:
            threshold = float(serving["threshold"])
            if threshold > 0:
                return threshold
            logger.warning("ignoring non-positive serving threshold %r in checkpoint", threshold)
        return None

    @staticmethod
    def _resolve_per_resource_thresholds(serving: dict[str, Any] | None) -> dict[str, float]:
        """The serving block's per-resource threshold map, validated entry by entry.

        Non-numeric and non-positive entries are dropped with a warning; the
        affected resource then serves the global threshold like any unknown one.
        """
        if not serving:
            return {}
        thresholds: dict[str, float] = {}
        for rid, value in (serving.get("per_resource") or {}).items():
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                logger.warning(
                    "ignoring non-numeric per-resource threshold for %r: %r", rid, value
                )
                continue
            if math.isfinite(parsed) and parsed > 0:
                thresholds[str(rid)] = parsed
            else:
                logger.warning(
                    "ignoring non-finite or non-positive per-resource threshold for %r: %r",
                    rid,
                    value,
                )
        return thresholds

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

    def reconstruction_error(
        self,
        numerical_metrics: dict[str, list[float]],
        categorical_metrics: dict[str, list[int]],
    ) -> dict[str, Any]:
        """Score the latest reconstruction window from parallel request series.

        The series are treated as parallel samples on a shared, end-aligned grid
        (most recent last) and scored through the same windowing the threshold was
        calibrated with. See :meth:`score_reconstruction`.

        Args:
            numerical_metrics: Dict of numerical metric time series.
            categorical_metrics: Dict of categorical metric time series.

        Returns:
            The score dict (see :meth:`score_reconstruction`).
        """
        return self.score_reconstruction(_frame_from_series(numerical_metrics, categorical_metrics))

    def score_reconstruction(
        self, df_long: pd.DataFrame, *, resource_id: str | None = None
    ) -> dict[str, Any]:
        """Score the most recent reconstruction window of a long-format frame.

        Windows the frame in the model's feature order with its stored
        normalization -- the exact path ``scripts/bake_serving_threshold.py`` and
        the incident-validation harness use -- and scores the most recent full
        window against the persisted healthy threshold. A frame with fewer than
        seq_len distinct timestamps cannot form a full window, so it is reported as
        not-scored (``reconstruction_error`` and ``ratio`` None) rather than scored
        on a front-padded window; this matches the bake path, which drops
        sub-seq_len captures, and avoids a spurious anomaly at cold start. A window
        that carries no numerical data at all (zero coverage) is likewise reported
        not-scored, so a numerical collection outage is not masked as a healthy
        all-neutral score. When no threshold is configured the error is still
        returned with a null ratio.

        Args:
            df_long: Canonical long-format metrics for a single resource.
            resource_id: The resource's canonical identity, when the caller has
                resolved it (the lookup path passes the id matched in the data
                source). Selects that resource's baked per-resource threshold if
                one exists; unknown or absent ids serve the global threshold, and
                the ``SCRY_RECON_THRESHOLD`` env override beats both.

        Returns:
            Dict with reconstruction_error, threshold, ratio, is_anomaly, severity,
            and coverage. reconstruction_error and ratio are None when no full
            window is available. coverage is measured over the scored window.
        """
        with tracer.start_as_current_span("score_reconstruction") as span:
            seq_len = int(self.config["seq_len"])
            threshold = self.recon_threshold
            if resource_id is not None and self._recon_env_threshold is None:
                threshold = self.recon_thresholds_per_resource.get(str(resource_id), threshold)

            error, coverage = self._latest_window(df_long, seq_len)
            span.set_attribute("reconstruction.coverage", coverage)
            if error is None:
                span.set_attribute("reconstruction.scored", False)
                return {
                    "reconstruction_error": None,
                    "threshold": threshold,
                    "ratio": None,
                    "is_anomaly": False,
                    "severity": 1,
                    "coverage": coverage,
                }

            if threshold is not None and threshold > 0:
                ratio: float | None = error / threshold
                is_anomaly = ratio > 1.0
                severity = _severity_from_ratio(ratio)
            else:
                ratio = None
                is_anomaly = False
                severity = 1

            span.set_attribute("reconstruction.error", error)
            span.set_attribute("reconstruction.is_anomaly", is_anomaly)
            return {
                "reconstruction_error": error,
                "threshold": threshold,
                "ratio": ratio,
                "is_anomaly": is_anomaly,
                "severity": severity,
                "coverage": coverage,
            }

    def _latest_window(self, df_long: pd.DataFrame, seq_len: int) -> tuple[float | None, float]:
        """Score the most recent full window; return (error, coverage).

        Trims to the most recent seq_len distinct timestamps so the cost is one
        window regardless of the lookback, and measures coverage over that scored
        window (not the whole frame, so a feature that stopped reporting before the
        window is counted as absent). Returns error None -- not scored -- when there
        are fewer than seq_len distinct timestamps or the window carries no
        numerical data at all (zero coverage), rather than scoring an all-neutral
        window that would read as healthy and mask a collection outage.
        """
        if df_long.empty:
            return None, 0.0
        ts = pd.to_datetime(df_long["timestamp"], utc=True)
        distinct = np.sort(ts.unique())
        if distinct.size < seq_len:
            return None, self._numerical_coverage(df_long)
        recent = set(distinct[-seq_len:])
        df_recent = df_long[ts.isin(recent)]

        coverage = self._numerical_coverage(df_recent)
        if coverage == 0.0:
            # No numerical feature was observed in the scored window; an all-neutral
            # window would score as healthy, so report not-scored instead.
            return None, 0.0

        windows = build_windows(
            df_recent,
            numerical_features=self.numerical_features,
            categorical_features=self.categorical_features,
            normalization=self.normalization,
            cat_normalization=self.cat_normalization,
            seq_len=seq_len,
            step=1,
        )
        if windows.x_num.shape[0] == 0:
            return None, coverage
        errors = reconstruction_errors(self.model, windows.x_num, windows.x_cat, self.device)
        # The most recent window is the one whose last sample is latest.
        latest = int(np.argmax(windows.end_times.values))
        return float(errors[latest]), coverage

    def _numerical_coverage(self, df_long: pd.DataFrame) -> float:
        """Fraction of the model's numerical features present (any non-null) in the frame."""
        if not self.numerical_features:
            return 1.0
        present_names = set(df_long.loc[df_long["value"].notna(), "metric_name"].unique())
        present = sum(1 for name in self.numerical_features if name in present_names)
        return present / len(self.numerical_features)
