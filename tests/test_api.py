# Description: Tests for FastAPI endpoints.
# Description: Tests health, predict, and cluster endpoints.

"""Tests for the FastAPI application."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient

from scry.api.predictor import Predictor
from scry.model import TemporalXDEC


@pytest.fixture
def temp_model_path():
    """Create a temporary model file for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_model.pt"

        # Create and save a model
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
            },
            model_path,
        )

        yield model_path


@pytest.fixture
def test_client(temp_model_path):
    """Create a test client with a loaded model."""
    # Patch the model path environment variable
    with patch.dict("os.environ", {"MODEL_PATH": str(temp_model_path)}):
        from scry.api.main import create_app

        app = create_app(model_path=str(temp_model_path))
        client = TestClient(app)
        yield client


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_returns_200(self, test_client):
        """Test health endpoint returns 200."""
        response = test_client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self, test_client):
        """Test health response has expected fields."""
        response = test_client.get("/health")
        data = response.json()

        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data

    def test_health_shows_model_loaded(self, test_client):
        """Test health shows model is loaded."""
        response = test_client.get("/health")
        data = response.json()

        assert data["status"] == "healthy"
        assert data["model_loaded"] is True


class TestPredictEndpoint:
    """Tests for /predict endpoint."""

    def test_predict_returns_200(self, test_client):
        """Test predict endpoint returns 200 for valid request."""
        response = test_client.post(
            "/predict",
            json={
                "resource_id": "test-pod-123",
                "numerical_metrics": {
                    "cpuUsageNanoCores": [1000000.0] * 30,
                },
                "categorical_metrics": {
                    "kubePodStatusReady": [1] * 30,
                },
            },
        )
        assert response.status_code == 200

    def test_predict_response_structure(self, test_client):
        """Test predict response has expected fields."""
        response = test_client.post(
            "/predict",
            json={
                "resource_id": "test-pod-123",
                "numerical_metrics": {"cpu": [1.0] * 30},
                "categorical_metrics": {"ready": [1] * 30},
            },
        )
        data = response.json()

        assert "resource_id" in data
        assert "cluster_id" in data
        assert "cluster_name" in data
        assert "confidence" in data
        assert "action" in data
        assert "priority" in data

    def test_predict_returns_resource_id(self, test_client):
        """Test predict echoes back resource_id."""
        response = test_client.post(
            "/predict",
            json={
                "resource_id": "my-special-pod",
                "numerical_metrics": {"cpu": [1.0] * 30},
                "categorical_metrics": {"ready": [1] * 30},
            },
        )
        data = response.json()

        assert data["resource_id"] == "my-special-pod"

    def test_predict_cluster_id_valid(self, test_client):
        """Test predict returns valid cluster_id."""
        response = test_client.post(
            "/predict",
            json={
                "resource_id": "test-pod",
                "numerical_metrics": {"cpu": [1.0] * 30},
                "categorical_metrics": {"ready": [1] * 30},
            },
        )
        data = response.json()

        assert 0 <= data["cluster_id"] <= 4

    def test_predict_confidence_valid(self, test_client):
        """Test predict returns valid confidence."""
        response = test_client.post(
            "/predict",
            json={
                "resource_id": "test-pod",
                "numerical_metrics": {"cpu": [1.0] * 30},
                "categorical_metrics": {"ready": [1] * 30},
            },
        )
        data = response.json()

        assert 0.0 <= data["confidence"] <= 1.0

    def test_predict_empty_resource_id_fails(self, test_client):
        """Test predict fails with empty resource_id."""
        response = test_client.post(
            "/predict",
            json={
                "resource_id": "",
                "numerical_metrics": {"cpu": [1.0]},
                "categorical_metrics": {"ready": [1]},
            },
        )
        assert response.status_code == 422  # Validation error

    def test_predict_empty_metrics_fails(self, test_client):
        """Test predict fails with empty metrics."""
        response = test_client.post(
            "/predict",
            json={
                "resource_id": "test-pod",
                "numerical_metrics": {},
                "categorical_metrics": {"ready": [1]},
            },
        )
        assert response.status_code == 422  # Validation error

    def test_predict_missing_field_fails(self, test_client):
        """Test predict fails with missing required field."""
        response = test_client.post(
            "/predict",
            json={
                "resource_id": "test-pod",
                # Missing numerical_metrics and categorical_metrics
            },
        )
        assert response.status_code == 422


class TestPredictLookupEndpoint:
    """Tests for /predict/lookup GET endpoint.

    The endpoint pulls a resource's recent metrics through the configured
    DataSource seam and predicts. With no source configured it returns an
    honest 503; with a source but no matching metrics, 404. The happy path is
    exercised by mocking the seam so the test needs no live data source.
    """

    def test_predict_lookup_happy_path(self, test_client):
        """A configured source with usable metrics yields a 200 prediction."""
        df = pd.DataFrame({"resource_id": ["my-test-pod"]})
        canned = {
            "cluster_id": 0,
            "cluster_name": "NORMAL",
            "confidence": 0.9,
            "action": "NONE",
            "priority": "LOW",
        }
        with (
            patch("scry.api.main._resource_metrics", new=AsyncMock(return_value=df)),
            patch("scry.api.main._split_by_profile", return_value=({"cpu": [1.0]}, {})),
            patch.object(Predictor, "predict", return_value=canned),
        ):
            response = test_client.get("/predict/lookup?resource_id=my-test-pod")

        assert response.status_code == 200
        data = response.json()
        assert data["resource_id"] == "my-test-pod"
        assert data["cluster_name"] == "NORMAL"
        for field in ("cluster_id", "confidence", "action", "priority"):
            assert field in data

    def test_predict_lookup_no_source_returns_503(self, test_client):
        """With no data source configured, the endpoint returns an honest 503."""
        with patch("scry.api.main._resource_metrics", new=AsyncMock(return_value=None)):
            response = test_client.get("/predict/lookup?resource_id=test-pod")
        assert response.status_code == 503
        assert "No data source configured" in response.json()["detail"]

    def test_predict_lookup_resource_not_found_returns_404(self, test_client):
        """A configured source with no matching metrics returns 404."""
        with patch(
            "scry.api.main._resource_metrics",
            new=AsyncMock(return_value=pd.DataFrame()),
        ):
            response = test_client.get("/predict/lookup?resource_id=ghost")
        assert response.status_code == 404

    def test_predict_lookup_missing_resource_id_fails(self, test_client):
        """Test predict lookup fails without resource_id."""
        response = test_client.get("/predict/lookup")
        assert response.status_code == 422  # Validation error


class TestClustersEndpoint:
    """Tests for /clusters endpoint."""

    def test_clusters_returns_200(self, test_client):
        """Test clusters endpoint returns 200."""
        response = test_client.get("/clusters")
        assert response.status_code == 200

    def test_clusters_returns_list(self, test_client):
        """Test clusters endpoint returns a list."""
        response = test_client.get("/clusters")
        data = response.json()

        assert isinstance(data, list)
        assert len(data) == 5

    def test_clusters_structure(self, test_client):
        """Test each cluster has expected fields."""
        response = test_client.get("/clusters")
        data = response.json()

        for cluster in data:
            assert "id" in cluster
            assert "name" in cluster
            assert "action" in cluster
            assert "priority" in cluster
            assert "description" in cluster

    def test_clusters_order(self, test_client):
        """Test clusters are in order by ID."""
        response = test_client.get("/clusters")
        data = response.json()

        names = [c["name"] for c in data]
        assert names == ["NORMAL", "PRE_SCALE", "PRE_FAILURE", "ACTIVE_DEGRADATION", "ANOMALY"]


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root_returns_200(self, test_client):
        """Test root endpoint returns 200."""
        response = test_client.get("/")
        assert response.status_code == 200

    def test_root_returns_info(self, test_client):
        """Test root endpoint returns API info."""
        response = test_client.get("/")
        data = response.json()

        assert "name" in data
        assert "version" in data
        assert "docs" in data


class TestOpenAPI:
    """Tests for OpenAPI documentation."""

    def test_openapi_available(self, test_client):
        """Test OpenAPI schema is available."""
        response = test_client.get("/openapi.json")
        assert response.status_code == 200

    def test_docs_available(self, test_client):
        """Test Swagger UI is available."""
        response = test_client.get("/docs")
        assert response.status_code == 200
