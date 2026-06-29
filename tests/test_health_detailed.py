# Description: Tests for /health/detailed endpoint.
# Description: Validates model diagnostics, the data source descriptor, and uptime reporting.

"""Tests for the detailed health check endpoint."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from scry.model import TemporalXDEC


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
    """Create a test client with a loaded model.

    Pins SCRY_DATA_URI to a fixed object-store URI AND resets the Config
    singleton so the new env actually takes effect. Without the reset, whichever
    test triggered the first get_config() call locks in that value for the rest
    of the suite (the developer's local .env or whatever pytest collected
    first), and the datasource descriptor assertions become non-hermetic.
    """
    from scry.utils.config import reset_config

    with patch.dict(
        "os.environ",
        {"MODEL_PATH": str(temp_model_path), "SCRY_DATA_URI": "data/metrics/**/*.parquet"},
    ):
        reset_config()
        from scry.api.main import create_app

        app = create_app(model_path=str(temp_model_path))
        client = TestClient(app)
        yield client
    reset_config()


class TestHealthDetailed:
    """Tests for /health/detailed endpoint."""

    def test_returns_200(self, test_client):
        """Endpoint returns 200 when model is loaded."""
        response = test_client.get("/health/detailed")
        assert response.status_code == 200

    def test_response_has_all_fields(self, test_client):
        """Response contains every DetailedHealthResponse field."""
        data = test_client.get("/health/detailed").json()

        expected_fields = [
            "status",
            "model_loaded",
            "version",
            "model_load_time_ms",
            "model_version",
            "model_path",
            "datasource",
            "chronos_loaded",
            "uptime_seconds",
        ]
        for field in expected_fields:
            assert field in data, f"Missing field: {field}"

    def test_status_healthy(self, test_client):
        """Status is healthy when model is loaded."""
        data = test_client.get("/health/detailed").json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True

    def test_model_load_time_populated(self, test_client):
        """Model load time is recorded and positive."""
        data = test_client.get("/health/detailed").json()
        assert data["model_load_time_ms"] is not None
        assert data["model_load_time_ms"] > 0

    def test_model_version_from_config(self, test_client):
        """Model version string reflects checkpoint config."""
        data = test_client.get("/health/detailed").json()
        assert data["model_version"] == "xdec-k5-d8"

    def test_uptime_positive(self, test_client):
        """Uptime is positive (app started before request)."""
        data = test_client.get("/health/detailed").json()
        assert data["uptime_seconds"] > 0

    def test_datasource_descriptor_reported(self, test_client):
        """The datasource descriptor is reported and non-empty."""
        data = test_client.get("/health/detailed").json()
        assert data["datasource"] is not None
        assert data["datasource"].startswith("object-store:")

    def test_datasource_reflects_config(self, test_client):
        """The datasource descriptor reflects the configured object-store URI."""
        data = test_client.get("/health/detailed").json()
        assert data["datasource"] == "object-store: data/metrics/**/*.parquet"

    def test_chronos_not_loaded_initially(self, test_client):
        """Chronos model is lazy-loaded, not loaded at startup."""
        data = test_client.get("/health/detailed").json()
        assert data["chronos_loaded"] is False

    def test_model_path_reported(self, test_client, temp_model_path):
        """Model path matches the configured path."""
        data = test_client.get("/health/detailed").json()
        assert data["model_path"] == str(temp_model_path)
