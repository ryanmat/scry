# Description: FastAPI prediction API for exposing cluster predictions
# Description: and recommended remediation actions.

"""FastAPI prediction service for Scry."""

from scry.api.main import app, create_app
from scry.api.predictor import Predictor
from scry.api.schemas import (
    ClusterInfo,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    get_cluster_info,
)

__all__ = [
    "app",
    "create_app",
    "Predictor",
    "PredictionRequest",
    "PredictionResponse",
    "HealthResponse",
    "ClusterInfo",
    "get_cluster_info",
]
