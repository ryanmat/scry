# Description: Pydantic schemas for API request/response models.
# Description: Defines validation rules for prediction requests and responses.

"""API request/response schemas."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ClusterName(str, Enum):
    """Valid cluster names."""

    NORMAL = "NORMAL"
    PRE_SCALE = "PRE_SCALE"
    PRE_FAILURE = "PRE_FAILURE"
    ACTIVE_DEGRADATION = "ACTIVE_DEGRADATION"
    ANOMALY = "ANOMALY"


class Action(str, Enum):
    """Valid remediation actions."""

    NONE = "NONE"
    SCALE = "SCALE"
    DIAGNOSTIC = "DIAGNOSTIC"
    REMEDIATE = "REMEDIATE"
    ALERT = "ALERT"


class Priority(str, Enum):
    """Valid priority levels."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class PredictionRequest(BaseModel):
    """Request schema for cluster prediction.

    Attributes:
        resource_id: Identifier for the infrastructure resource.
        numerical_metrics: Dict mapping metric names to time series values.
        categorical_metrics: Dict mapping metric names to time series values (0/1).
        window_minutes: Length of the prediction window in minutes.
    """

    resource_id: str = Field(..., min_length=1, description="Resource identifier")
    numerical_metrics: dict[str, list[float]] = Field(
        ..., min_length=1, description="Numerical metric time series"
    )
    categorical_metrics: dict[str, list[int]] = Field(
        ..., min_length=1, description="Categorical metric time series (0/1 values)"
    )
    window_minutes: int = Field(default=30, ge=1, le=120, description="Prediction window size")

    @field_validator("numerical_metrics")
    @classmethod
    def validate_numerical_not_empty(cls, v: dict) -> dict:
        """Ensure numerical_metrics is not empty."""
        if not v:
            raise ValueError("numerical_metrics cannot be empty")
        return v

    @field_validator("categorical_metrics")
    @classmethod
    def validate_categorical_not_empty(cls, v: dict) -> dict:
        """Ensure categorical_metrics is not empty."""
        if not v:
            raise ValueError("categorical_metrics cannot be empty")
        return v


class PredictionResponse(BaseModel):
    """Response schema for cluster prediction.

    Attributes:
        resource_id: Identifier for the infrastructure resource.
        cluster_id: Assigned cluster ID (0-4).
        cluster_name: Human-readable cluster name.
        confidence: Prediction confidence (0-1).
        action: Recommended remediation action.
        priority: Action priority level.
    """

    resource_id: str = Field(..., description="Resource identifier")
    cluster_id: int = Field(..., ge=0, le=4, description="Cluster ID (0-4)")
    cluster_name: Literal["NORMAL", "PRE_SCALE", "PRE_FAILURE", "ACTIVE_DEGRADATION", "ANOMALY"] = (
        Field(..., description="Cluster name")
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")
    action: Literal["NONE", "SCALE", "DIAGNOSTIC", "REMEDIATE", "ALERT"] = Field(
        ..., description="Recommended action"
    )
    priority: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = Field(
        ..., description="Action priority"
    )


class HealthResponse(BaseModel):
    """Response schema for health check.

    Attributes:
        status: Health status ("healthy" or "unhealthy").
        model_loaded: Whether the model is loaded.
        version: API version.
    """

    status: str = Field(..., description="Health status")
    model_loaded: bool = Field(..., description="Model loaded status")
    version: str = Field(..., description="API version")


class DetailedHealthResponse(BaseModel):
    """Response schema for the detailed health check.

    Attributes:
        status: Health status ("healthy" or "unhealthy").
        model_loaded: Whether the X-DEC model is loaded.
        version: API version.
        model_load_time_ms: Time taken to load the X-DEC model in milliseconds.
        model_version: Model identifier derived from checkpoint config.
        model_path: Filesystem path to the model file.
        datasource: Configured data source descriptor, or None if unconfigured.
        chronos_loaded: Whether the Chronos forecasting model is loaded.
        drift_configured: Whether an operator has attached a drift detector.
        forecast_anomaly_configured: Whether an operator has attached a
            forecast-anomaly detector.
        accuracy_configured: Whether an operator has attached an accuracy tracker.
        uptime_seconds: Seconds since the API started.
    """

    status: str = Field(..., description="Health status")
    model_loaded: bool = Field(..., description="X-DEC model loaded status")
    version: str = Field(..., description="API version")
    model_load_time_ms: float | None = Field(None, description="Model load time in ms")
    model_version: str | None = Field(None, description="Model config identifier")
    model_path: str = Field(..., description="Path to model file")
    datasource: str | None = Field(None, description="Configured data source descriptor")
    chronos_loaded: bool = Field(..., description="Chronos model loaded")
    recon_threshold: float | None = Field(
        None, description="Configured healthy reconstruction threshold, if any"
    )
    drift_configured: bool = Field(
        False, description="Drift detector attached; /drift serves 503 until then"
    )
    forecast_anomaly_configured: bool = Field(
        False, description="Forecast-anomaly detector attached; /anomaly serves 503 until then"
    )
    accuracy_configured: bool = Field(
        False, description="Accuracy tracker attached; /accuracy serves 503 until then"
    )
    uptime_seconds: float = Field(..., description="Seconds since API start")


class ClusterInfo(BaseModel):
    """Information about a cluster.

    Attributes:
        id: Cluster ID (0-4).
        name: Cluster name.
        action: Recommended action for this cluster.
        priority: Priority level for this cluster.
        description: Human-readable description.
    """

    id: int = Field(..., ge=0, le=4, description="Cluster ID")
    name: str = Field(..., description="Cluster name")
    action: str = Field(..., description="Recommended action")
    priority: str = Field(..., description="Priority level")
    description: str = Field(..., description="Cluster description")


# Cluster definitions
CLUSTER_DEFINITIONS = [
    ClusterInfo(
        id=0,
        name="NORMAL",
        action="NONE",
        priority="LOW",
        description="System operating normally",
    ),
    ClusterInfo(
        id=1,
        name="PRE_SCALE",
        action="SCALE",
        priority="MEDIUM",
        description="Consider proactive scaling",
    ),
    ClusterInfo(
        id=2,
        name="PRE_FAILURE",
        action="DIAGNOSTIC",
        priority="HIGH",
        description="Collect diagnostics, prepare remediation",
    ),
    ClusterInfo(
        id=3,
        name="ACTIVE_DEGRADATION",
        action="REMEDIATE",
        priority="CRITICAL",
        description="Immediate action required",
    ),
    ClusterInfo(
        id=4,
        name="ANOMALY",
        action="ALERT",
        priority="HIGH",
        description="Unknown pattern, investigate",
    ),
]


def get_cluster_info() -> list[ClusterInfo]:
    """Get all cluster definitions.

    Returns:
        List of ClusterInfo objects for all 5 clusters.
    """
    return CLUSTER_DEFINITIONS


# -- Forecast schemas --


class ForecastRequest(BaseModel):
    """Request schema for metric forecasting.

    Attributes:
        resource_id: Identifier for the infrastructure resource.
        metrics: Dict mapping metric names to historical time series values.
        horizons: Forecast horizons in timesteps (e.g. [15, 60, 240, 1440]
            for 15min, 1h, 4h, 24h at 1-min resolution).
    """

    resource_id: str = Field(..., min_length=1, description="Resource identifier")
    metrics: dict[str, list[float]] = Field(
        ..., min_length=1, description="Metric time series (name -> values)"
    )
    horizons: list[int] = Field(
        default=[15, 60, 240, 1440],
        min_length=1,
        description="Forecast horizons in timesteps",
    )

    @field_validator("metrics")
    @classmethod
    def validate_metrics_not_empty(cls, v: dict) -> dict:
        """Ensure metrics is not empty and each series has data."""
        if not v:
            raise ValueError("metrics cannot be empty")
        for name, values in v.items():
            if not values:
                raise ValueError(f"metric '{name}' has empty time series")
        return v

    @field_validator("horizons")
    @classmethod
    def validate_horizons_positive(cls, v: list[int]) -> list[int]:
        """Ensure all horizons are positive integers."""
        for h in v:
            if h < 1:
                raise ValueError(f"horizon must be positive, got {h}")
        return v


class HorizonForecast(BaseModel):
    """Forecast at a single horizon.

    Attributes:
        horizon: Forecast horizon in timesteps.
        median: Median (p50) forecast value.
        lower: Lower quantile (p10) forecast value.
        upper: Upper quantile (p90) forecast value.
    """

    horizon: int = Field(..., ge=1, description="Horizon in timesteps")
    median: float = Field(..., description="Median forecast value")
    lower: float = Field(..., description="Lower quantile forecast")
    upper: float = Field(..., description="Upper quantile forecast")


class MetricForecast(BaseModel):
    """Forecast for a single metric across all horizons.

    Attributes:
        metric_name: Name of the forecasted metric.
        horizons: Per-horizon forecast values.
    """

    metric_name: str = Field(..., description="Metric name")
    horizons: list[HorizonForecast] = Field(..., description="Per-horizon forecasts")


class ForecastResponse(BaseModel):
    """Response schema for metric forecasting.

    Attributes:
        resource_id: Identifier for the infrastructure resource.
        forecasts: Per-metric forecast results.
        model_id: HuggingFace model ID used for forecasting.
    """

    resource_id: str = Field(..., description="Resource identifier")
    forecasts: list[MetricForecast] = Field(..., description="Per-metric forecasts")
    model_id: str = Field(..., description="Model used for forecasting")


# -- Reconstruction anomaly schemas --


class ReconstructionRequest(BaseModel):
    """Request schema for the reconstruction-anomaly score.

    Attributes:
        resource_id: Identifier for the infrastructure resource.
        numerical_metrics: Dict mapping metric names to time series values.
        categorical_metrics: Dict mapping metric names to time series values (0/1).
            Optional so purely-numerical models can be scored.
    """

    resource_id: str = Field(..., min_length=1, description="Resource identifier")
    numerical_metrics: dict[str, list[float]] = Field(
        ..., min_length=1, description="Numerical metric time series"
    )
    categorical_metrics: dict[str, list[int]] = Field(
        default_factory=dict, description="Categorical metric time series (0/1 values)"
    )

    @field_validator("numerical_metrics")
    @classmethod
    def validate_numerical_not_empty(cls, v: dict) -> dict:
        """Ensure numerical_metrics is not empty."""
        if not v:
            raise ValueError("numerical_metrics cannot be empty")
        return v


class ReconstructionResponse(BaseModel):
    """Response schema for the reconstruction-anomaly score.

    Attributes:
        resource_id: Identifier for the infrastructure resource.
        reconstruction_error: Per-window numerical reconstruction MSE (from the latent mean);
            None when the window was too short to score.
        threshold: Healthy reconstruction threshold, or None when unconfigured.
        ratio: reconstruction_error / threshold, or None when unconfigured.
        is_anomaly: True when the ratio exceeds 1 (error above the healthy threshold).
        severity: Severity band (1 low .. 4 critical) derived from the ratio.
        coverage: Fraction of the model's numerical features present in the input.
        timestamp: UTC ISO8601 time of scoring.
    """

    resource_id: str = Field(..., description="Resource identifier")
    reconstruction_error: float | None = Field(
        None, ge=0.0, description="Per-window numerical reconstruction MSE; None when not scored"
    )
    threshold: float | None = Field(None, description="Healthy reconstruction threshold")
    ratio: float | None = Field(
        None, description="reconstruction_error / threshold; None when no threshold is configured"
    )
    is_anomaly: bool = Field(..., description="True when the ratio exceeds 1")
    severity: int = Field(..., ge=1, le=4, description="Severity band (1 low .. 4 critical)")
    coverage: float = Field(
        ..., ge=0.0, le=1.0, description="Fraction of model numerical features present"
    )
    timestamp: str = Field(..., description="UTC ISO8601 time of scoring")
