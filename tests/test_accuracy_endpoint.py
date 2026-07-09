# Description: Tests for /accuracy API endpoint contract.
# Description: Validates the unconfigured 503 refusal, metric response, and flat key structure.

"""Tests for /accuracy API endpoint."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from scry.model import TemporalXDEC
from scry.model.forecasting.accuracy import ForecastAccuracyTracker


@pytest.fixture
def temp_model_path():
    """Create a temporary model file for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_model.pt"

        model = TemporalXDEC(
            num_numerical=9,
            num_categorical=8,
            seq_len=30,
            num_hidden=64,
            cat_hidden=32,
            latent_dim=8,
            n_clusters=5,
        )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": {
                    "num_numerical": 9,
                    "num_categorical": 8,
                    "seq_len": 30,
                    "num_hidden": 64,
                    "cat_hidden": 32,
                    "latent_dim": 8,
                    "n_clusters": 5,
                },
                "normalization": {
                    "mean": np.zeros(9),
                    "std": np.ones(9),
                },
                "categorical_normalization": {"min": np.zeros(8), "max": np.ones(8)},
                "feature_schema": {
                    "numerical": [f"num{i}" for i in range(9)],
                    "categorical": [f"cat{i}" for i in range(8)],
                    "profile": "test",
                },
            },
            model_path,
        )

        yield model_path


@pytest.fixture
def test_client(temp_model_path):
    """Create a test client without accuracy tracker."""
    with patch.dict("os.environ", {"MODEL_PATH": str(temp_model_path)}):
        from scry.api.main import create_app

        app = create_app(model_path=str(temp_model_path))
        client = TestClient(app)
        yield client


@pytest.fixture
def test_client_with_tracker(temp_model_path):
    """Create a test client with accuracy tracker configured."""
    with patch.dict("os.environ", {"MODEL_PATH": str(temp_model_path)}):
        from scry.api.main import create_app

        app = create_app(model_path=str(temp_model_path))

        tracker = ForecastAccuracyTracker(horizons=["15m", "1h", "4h", "24h"])
        # Record some observations
        for h in ["15m", "1h", "4h", "24h"]:
            for i in range(5):
                tracker.record_forecast(h, actual=float(i), median=float(i) + 0.1, lower=float(i) - 1, upper=float(i) + 1)
        for i in range(5):
            tracker.record_cluster(cluster_id=0, confidence=0.9)

        app.state.accuracy_tracker = tracker
        client = TestClient(app)
        yield client


class TestAccuracyEndpointNoTracker:
    """Tests for /accuracy when no tracker is configured."""

    def test_accuracy_endpoint_no_tracker(self, test_client):
        """The shipped default app (no tracker attached) serves 503, not zero metrics.

        Runs against plain create_app() with no app.state monkeypatching, so this
        fails if an unconfigured /accuracy fabricates ApiStatus=1 with all-zero
        metrics.
        """
        response = test_client.get("/accuracy")
        assert response.status_code == 503
        assert response.json()["detail"] == "Accuracy tracking not configured"

    def test_health_detailed_reports_accuracy_unconfigured(self, test_client):
        """/health/detailed exposes the wiring state without probing /accuracy."""
        response = test_client.get("/health/detailed")
        assert response.status_code == 200
        assert response.json()["accuracy_configured"] is False

    def test_health_detailed_reports_accuracy_configured(self, test_client_with_tracker):
        """The flag flips once an operator attaches a tracker."""
        response = test_client_with_tracker.get("/health/detailed")
        assert response.status_code == 200
        assert response.json()["accuracy_configured"] is True


class TestAccuracyEndpointWithTracker:
    """Tests for /accuracy when tracker is active."""

    def test_accuracy_endpoint_with_tracker(self, test_client_with_tracker):
        """Returns full metrics dict with per-horizon and stability values."""
        response = test_client_with_tracker.get("/accuracy")
        assert response.status_code == 200

        data = response.json()
        assert data["ObservationCount"] == 5
        assert data["ApiStatus"] == 1
        # Check a per-horizon value exists
        assert "Picp15m" in data
        assert "Mae1h" in data
        assert "TransitionRate" in data

    def test_accuracy_response_has_required_keys(self, test_client_with_tracker):
        """All flat keys the Groovy script expects are present."""
        response = test_client_with_tracker.get("/accuracy")
        data = response.json()

        # Per-horizon keys (4 metrics x 4 horizons = 16)
        for metric in ["Picp", "Mae", "Mase", "Mpiw"]:
            for horizon in ["15m", "1h", "4h", "24h"]:
                key = f"{metric}{horizon}"
                assert key in data, f"Missing required key: {key}"

        # Stability keys
        for key in ["TransitionRate", "ConfidenceStd", "DominantClusterPct"]:
            assert key in data, f"Missing required key: {key}"

        # Operational keys
        for key in ["ObservationCount", "ApiStatus", "ApiLatencyMs", "timestamp"]:
            assert key in data, f"Missing required key: {key}"
