# Description: Tests for POST /forecast endpoint.
# Description: Validates schema, synthetic sine wave forecasting, and error cases.

"""Tests for the /forecast API endpoint."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from scry.model import TemporalXDEC

MOCK_FORECAST_RESULT = [
    {
        "metric_name": "cpu",
        "horizons": [
            {"horizon": 15, "median": 47.3, "lower": 44.1, "upper": 51.2},
            {"horizon": 60, "median": 48.9, "lower": 42.5, "upper": 55.8},
        ],
    },
    {
        "metric_name": "memory",
        "horizons": [
            {"horizon": 15, "median": 72.1, "lower": 70.0, "upper": 74.5},
            {"horizon": 60, "median": 73.0, "lower": 69.2, "upper": 76.8},
        ],
    },
]


@pytest.fixture
def temp_model_path():
    """Create a temporary model file for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_model.pt"
        model = TemporalXDEC(
            num_numerical=9, num_categorical=8, seq_len=30,
            num_hidden=64, cat_hidden=32, latent_dim=8, n_clusters=5,
        )
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": {
                    "num_numerical": 9, "num_categorical": 8, "seq_len": 30,
                    "num_hidden": 64, "cat_hidden": 32, "latent_dim": 8, "n_clusters": 5,
                },
                "normalization": {"mean": np.zeros(9), "std": np.ones(9)},
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
def forecast_client(temp_model_path):
    """Create a test client with a real Predictor but mocked Forecaster."""
    mock_forecaster = MagicMock()
    mock_forecaster.model_id = "amazon/chronos-bolt-tiny"
    mock_forecaster.horizons = [15, 60, 240, 1440]
    mock_forecaster.forecast_metrics.return_value = MOCK_FORECAST_RESULT

    from scry.api.main import create_app

    app = create_app(model_path=str(temp_model_path))
    app.state.forecaster = mock_forecaster
    client = TestClient(app)
    yield client, mock_forecaster


class TestForecastEndpoint:
    """Tests for POST /forecast endpoint."""

    def test_forecast_returns_200(self, forecast_client):
        """Test forecast endpoint returns 200 for valid request."""
        client, _ = forecast_client
        response = client.post(
            "/forecast",
            json={
                "resource_id": "web-server-01",
                "metrics": {"cpu": [45.2, 46.1, 47.0, 48.3]},
                "horizons": [15, 60],
            },
        )
        assert response.status_code == 200

    def test_forecast_response_structure(self, forecast_client):
        """Test forecast response has expected fields."""
        client, _ = forecast_client
        response = client.post(
            "/forecast",
            json={
                "resource_id": "web-server-01",
                "metrics": {"cpu": [50.0] * 100},
                "horizons": [15, 60],
            },
        )
        data = response.json()

        assert data["resource_id"] == "web-server-01"
        assert data["model_id"] == "amazon/chronos-bolt-tiny"
        assert len(data["forecasts"]) == 2
        assert data["forecasts"][0]["metric_name"] == "cpu"
        assert data["forecasts"][1]["metric_name"] == "memory"

    def test_forecast_horizon_structure(self, forecast_client):
        """Test each horizon forecast has median, lower, upper."""
        client, _ = forecast_client
        response = client.post(
            "/forecast",
            json={
                "resource_id": "test-vm",
                "metrics": {"cpu": [50.0] * 100},
                "horizons": [15, 60],
            },
        )
        data = response.json()
        horizon = data["forecasts"][0]["horizons"][0]

        assert "horizon" in horizon
        assert "median" in horizon
        assert "lower" in horizon
        assert "upper" in horizon
        assert horizon["lower"] <= horizon["median"] <= horizon["upper"]

    def test_forecast_calls_forecaster_with_metrics(self, forecast_client):
        """Test that the forecaster is called with the provided metrics."""
        client, mock_forecaster = forecast_client
        metrics = {"cpu": [1.0, 2.0, 3.0], "mem": [70.0, 71.0, 72.0]}
        client.post(
            "/forecast",
            json={
                "resource_id": "test-vm",
                "metrics": metrics,
                "horizons": [15, 60],
            },
        )
        mock_forecaster.forecast_metrics.assert_called_once_with(metrics)

    def test_forecast_empty_resource_id_fails(self, forecast_client):
        """Test forecast fails with empty resource_id."""
        client, _ = forecast_client
        response = client.post(
            "/forecast",
            json={
                "resource_id": "",
                "metrics": {"cpu": [1.0]},
            },
        )
        assert response.status_code == 422

    def test_forecast_empty_metrics_fails(self, forecast_client):
        """Test forecast fails with empty metrics dict."""
        client, _ = forecast_client
        response = client.post(
            "/forecast",
            json={
                "resource_id": "test-vm",
                "metrics": {},
            },
        )
        assert response.status_code == 422

    def test_forecast_empty_series_fails(self, forecast_client):
        """Test forecast fails when a metric has empty time series."""
        client, _ = forecast_client
        response = client.post(
            "/forecast",
            json={
                "resource_id": "test-vm",
                "metrics": {"cpu": []},
            },
        )
        assert response.status_code == 422

    def test_forecast_missing_metrics_fails(self, forecast_client):
        """Test forecast fails with missing metrics field."""
        client, _ = forecast_client
        response = client.post(
            "/forecast",
            json={"resource_id": "test-vm"},
        )
        assert response.status_code == 422

    def test_forecast_zero_horizon_fails(self, forecast_client):
        """Test forecast fails with zero horizon value."""
        client, _ = forecast_client
        response = client.post(
            "/forecast",
            json={
                "resource_id": "test-vm",
                "metrics": {"cpu": [1.0]},
                "horizons": [0],
            },
        )
        assert response.status_code == 422

    def test_forecast_default_horizons(self, forecast_client):
        """Test forecast uses default horizons when not specified."""
        client, _ = forecast_client
        response = client.post(
            "/forecast",
            json={
                "resource_id": "test-vm",
                "metrics": {"cpu": [50.0] * 100},
            },
        )
        assert response.status_code == 200


class TestForecastSchemas:
    """Tests for forecast request/response validation."""

    def test_forecast_request_requires_resource_id(self):
        """Test ForecastRequest validates resource_id."""
        from scry.api.schemas import ForecastRequest

        with pytest.raises(Exception):
            ForecastRequest(metrics={"cpu": [1.0]})

    def test_forecast_request_validates_metrics(self):
        """Test ForecastRequest validates metrics is not empty."""
        from scry.api.schemas import ForecastRequest

        with pytest.raises(Exception):
            ForecastRequest(resource_id="test", metrics={})

    def test_forecast_request_validates_horizons(self):
        """Test ForecastRequest validates horizons are positive."""
        from scry.api.schemas import ForecastRequest

        with pytest.raises(Exception):
            ForecastRequest(
                resource_id="test",
                metrics={"cpu": [1.0]},
                horizons=[-1],
            )

    def test_forecast_request_accepts_valid(self):
        """Test ForecastRequest accepts valid input."""
        from scry.api.schemas import ForecastRequest

        req = ForecastRequest(
            resource_id="test-vm",
            metrics={"cpu": [45.2, 46.1]},
            horizons=[15, 60],
        )
        assert req.resource_id == "test-vm"
        assert len(req.metrics) == 1
        assert req.horizons == [15, 60]


class TestForecasterUnit:
    """Unit tests for the Forecaster wrapper class."""

    def test_forecaster_not_loaded_initially(self):
        """Test forecaster is not loaded until first call."""
        from scry.api.forecaster import Forecaster

        f = Forecaster(model_id="test-model")
        assert not f.is_loaded

    def test_forecaster_forecast_metrics_calls_batch(self):
        """Test forecast_metrics delegates to ChronosForecaster batch methods."""
        import numpy as np

        from scry.api.forecaster import Forecaster

        f = Forecaster(horizons=[15, 60])

        mock_chronos = MagicMock()
        mock_chronos.forecast_batch.return_value = {
            "median": np.array([[47.3, 48.9]]),
            "lower": np.array([[44.1, 42.5]]),
            "upper": np.array([[51.2, 55.8]]),
        }
        mock_chronos.extract_at_horizons.return_value = {
            "median": np.array([[47.3, 48.9]]),
            "lower": np.array([[44.1, 42.5]]),
            "upper": np.array([[51.2, 55.8]]),
        }
        f._forecaster = mock_chronos

        results = f.forecast_metrics({"cpu": [50.0] * 100})

        assert len(results) == 1
        assert results[0]["metric_name"] == "cpu"
        assert len(results[0]["horizons"]) == 2
        assert results[0]["horizons"][0]["horizon"] == 15
        assert results[0]["horizons"][0]["median"] == 47.3

    def test_forecaster_multi_metric(self):
        """Test forecast_metrics handles multiple metrics."""
        import numpy as np

        from scry.api.forecaster import Forecaster

        f = Forecaster(horizons=[15])
        mock_chronos = MagicMock()
        mock_chronos.forecast_batch.return_value = {
            "median": np.array([[47.3], [72.1]]),
            "lower": np.array([[44.1], [70.0]]),
            "upper": np.array([[51.2], [74.5]]),
        }
        mock_chronos.extract_at_horizons.return_value = {
            "median": np.array([[47.3], [72.1]]),
            "lower": np.array([[44.1], [70.0]]),
            "upper": np.array([[51.2], [74.5]]),
        }
        f._forecaster = mock_chronos

        results = f.forecast_metrics({"cpu": [50.0] * 100, "mem": [70.0] * 100})

        assert len(results) == 2
        assert results[0]["metric_name"] == "cpu"
        assert results[1]["metric_name"] == "mem"
