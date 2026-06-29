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
                "categorical_normalization": {
                    "min": np.zeros(8),
                    "max": np.ones(8),
                },
                "feature_schema": {
                    "numerical": [
                        "cpuUsageNanoCores", "memoryUsageBytes", "networkRxBytes",
                        "networkTxBytes", "fsUsedBytes", "cpuLimits", "memoryLimits",
                        "memoryRequests", "memoryWorkingSetBytes",
                    ],
                    "categorical": [
                        "kubePodStatusReady", "podConditionPhase", "kubePodStatusScheduled",
                        "kubePodContainerStatusRunning", "kubePodContainerStatusWaiting",
                        "kubePodContainerStatusTerminated", "kubePodContainerStatusReady",
                        "status",
                    ],
                    "profile": "test",
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


# ARO node ids from the captured data: 3 masters + 5 workers.
_ARO_IDS = [
    "rm-oc-cluster-27t8q-master-0-node-rm-aro-cluster",
    "rm-oc-cluster-27t8q-master-1-node-rm-aro-cluster",
    "rm-oc-cluster-27t8q-master-2-node-rm-aro-cluster",
    "rm-oc-cluster-27t8q-worker-eastus1-jm9mz-node-rm-aro-cluster",
    "rm-oc-cluster-27t8q-worker-eastus1-vb6vs-node-rm-aro-cluster",
    "rm-oc-cluster-27t8q-worker-eastus2-lp7tj-node-rm-aro-cluster",
    "rm-oc-cluster-27t8q-worker-eastus2-xwg6l-node-rm-aro-cluster",
    "rm-oc-cluster-27t8q-worker-eastus3-75cpl-node-rm-aro-cluster",
]


def _resolution_frame() -> pd.DataFrame:
    """A canonical frame of petclinic-vm plus the 8 ARO nodes (IP host_names)."""
    ids = ["petclinic-vm", *_ARO_IDS]
    hosts = ["127.0.0.1", *[f"10.1.12.{i}" for i in range(len(_ARO_IDS))]]
    return pd.DataFrame(
        {
            "resource_id": ids,
            "host_name": hosts,
            "metric_name": ["m"] * len(ids),
            "value": [1.0] * len(ids),
        }
    )


class TestResourceResolution:
    """Tests for the exact-preferred -> substring resolution helper."""

    def test_exact_resolves_petclinic_vm(self):
        from scry.api.main import _resolve_rows

        out = _resolve_rows(_resolution_frame(), "petclinic-vm")
        assert set(out["resource_id"]) == {"petclinic-vm"}

    def test_exact_is_case_insensitive(self):
        from scry.api.main import _resolve_rows

        out = _resolve_rows(_resolution_frame(), "PETCLINIC-VM")
        assert set(out["resource_id"]) == {"petclinic-vm"}

    def test_unique_substring_resolves_master_0(self):
        from scry.api.main import _resolve_rows

        out = _resolve_rows(_resolution_frame(), "master-0")
        assert set(out["resource_id"]) == {
            "rm-oc-cluster-27t8q-master-0-node-rm-aro-cluster"
        }

    def test_ambiguous_substring_worker_matches_five(self):
        from scry.api.main import _resolve_rows

        out = _resolve_rows(_resolution_frame(), "worker")
        assert out["resource_id"].nunique() == 5

    def test_unknown_returns_empty(self):
        from scry.api.main import _resolve_rows

        out = _resolve_rows(_resolution_frame(), "ghost")
        assert out.empty

    def test_exact_preferred_over_substring(self):
        from scry.api.main import _resolve_rows

        df = pd.DataFrame(
            {
                "resource_id": ["node", "node-2", "node-3"],
                "host_name": ["h", "h", "h"],
                "metric_name": ["m"] * 3,
                "value": [1.0] * 3,
            }
        )
        out = _resolve_rows(df, "node")
        assert set(out["resource_id"]) == {"node"}

    def test_exact_host_name_resolves(self):
        from scry.api.main import _resolve_rows

        df = pd.DataFrame(
            {
                "resource_id": ["res-a", "res-b"],
                "host_name": ["10.0.0.5", "10.0.0.6"],
                "metric_name": ["m", "m"],
                "value": [1.0, 1.0],
            }
        )
        out = _resolve_rows(df, "10.0.0.5")
        assert set(out["resource_id"]) == {"res-a"}

    def test_exact_host_beats_substring(self):
        from scry.api.main import _resolve_rows

        # The needle exactly matches res-a's host and is a substring of res-b's
        # host (10.0.0.50); the exact host_name match must win.
        df = pd.DataFrame(
            {
                "resource_id": ["res-a", "res-b"],
                "host_name": ["10.0.0.5", "10.0.0.50"],
                "metric_name": ["m", "m"],
                "value": [1.0, 1.0],
            }
        )
        out = _resolve_rows(df, "10.0.0.5")
        assert set(out["resource_id"]) == {"res-a"}

    def test_substring_fallback_is_literal_not_regex(self):
        from scry.api.main import _resolve_rows

        # Needle "a.c" matches the literal host "xa.cy" but must NOT regex-match
        # "xabcy" (where '.' would be a wildcard).
        df = pd.DataFrame(
            {
                "resource_id": ["dotted", "plain"],
                "host_name": ["xa.cy", "xabcy"],
                "metric_name": ["m", "m"],
                "value": [1.0, 1.0],
            }
        )
        out = _resolve_rows(df, "a.c")
        assert set(out["resource_id"]) == {"dotted"}


class TestPredictLookupAmbiguity:
    """The lookup endpoint refuses to pool multiple resources into one prediction."""

    def test_ambiguous_lookup_returns_409(self, test_client):
        workers = [i for i in _ARO_IDS if "worker" in i]
        df = pd.DataFrame(
            {
                "resource_id": workers,
                "host_name": [f"10.1.12.{i}" for i in range(len(workers))],
                "metric_name": ["m"] * len(workers),
                "value": [1.0] * len(workers),
            }
        )
        with (
            patch("scry.api.main._resource_metrics", new=AsyncMock(return_value=df)),
            patch.object(Predictor, "predict") as mock_predict,
        ):
            response = test_client.get("/predict/lookup?resource_id=worker")

        assert response.status_code == 409
        detail = response.json()["detail"]
        assert len(detail["candidates"]) == 5
        assert detail["candidates"] == sorted(workers)
        mock_predict.assert_not_called()

    def test_ambiguous_lookup_caps_candidates_at_20(self, test_client):
        ids = [f"node-{i:03d}" for i in range(25)]
        df = pd.DataFrame(
            {
                "resource_id": ids,
                "host_name": [f"10.0.0.{i}" for i in range(25)],
                "metric_name": ["m"] * 25,
                "value": [1.0] * 25,
            }
        )
        with patch("scry.api.main._resource_metrics", new=AsyncMock(return_value=df)):
            response = test_client.get("/predict/lookup", params={"resource_id": "node"})

        assert response.status_code == 409
        candidates = response.json()["detail"]["candidates"]
        assert len(candidates) == 20  # capped from 25 matches
        assert candidates == sorted(ids)[:20]


class TestSchemaLessStartup:
    """A schema-less checkpoint degrades the API to unhealthy rather than crashing."""

    def test_schema_less_checkpoint_degrades_health(self, tmp_path):
        from scry.api.main import create_app

        model = TemporalXDEC(
            num_numerical=9,
            num_categorical=8,
            seq_len=30,
            num_hidden=64,
            cat_hidden=32,
            latent_dim=8,
            n_clusters=5,
        )
        path = tmp_path / "old_model.pt"
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
                "normalization": {"mean": np.zeros(9), "std": np.ones(9)},
            },
            path,
        )

        app = create_app(model_path=str(path))
        client = TestClient(app)
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json()["model_loaded"] is False
