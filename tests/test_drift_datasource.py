# Description: Tests for /drift API endpoint and DataSource contract.
# Description: Validates drift response keys match what the Groovy collector script expects.

"""Tests for drift endpoint and DataSource contract."""

import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from scry.model import TemporalXDEC
from scry.model.drift import DriftDetector


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
    """Create a test client with a loaded model."""
    with patch.dict("os.environ", {"MODEL_PATH": str(temp_model_path)}):
        from scry.api.main import create_app

        app = create_app(model_path=str(temp_model_path))
        client = TestClient(app)
        yield client


@pytest.fixture
def test_client_with_drift(temp_model_path):
    """Create a test client with drift detector configured."""
    with patch.dict("os.environ", {"MODEL_PATH": str(temp_model_path)}):
        from scry.api.main import create_app

        app = create_app(model_path=str(temp_model_path))

        # Configure drift detector on app state
        detector = DriftDetector(
            n_features=9,
            feature_names=[f"feature_{i}" for i in range(9)],
            psi_threshold=0.2,
        )
        app.state.drift_detector = detector
        app.state.reference_data = np.random.randn(100, 9)
        app.state.current_data = np.random.randn(100, 9)
        app.state.error_stream = np.random.randn(200)

        client = TestClient(app)
        yield client


class TestDriftEndpointNoReferenceData:
    """Tests for /drift when no reference data is configured."""

    def test_drift_endpoint_no_reference_data(self, test_client):
        """Endpoint returns graceful fallback when no detector configured."""
        response = test_client.get("/drift")
        assert response.status_code == 200

        data = response.json()
        assert data["feature_drift"]["has_drift"] is False
        assert data["prediction_drift"]["has_drift"] is False
        assert "timestamp" in data


class TestDriftEndpointWithDetector:
    """Tests for /drift when detector is configured with data."""

    def test_drift_endpoint_with_detector(self, test_client_with_drift):
        """Endpoint returns full PSI/ADWIN metrics when detector is active."""
        response = test_client_with_drift.get("/drift")
        assert response.status_code == 200

        data = response.json()
        assert "feature_drift" in data
        assert "prediction_drift" in data
        assert "timestamp" in data

        # Feature drift should have PSI details
        fd = data["feature_drift"]
        assert "has_drift" in fd
        assert "psi_per_feature" in fd
        assert "max_psi" in fd
        assert "threshold" in fd

        # Prediction drift should have ADWIN details
        pd = data["prediction_drift"]
        assert "has_drift" in pd
        assert "score" in pd
        assert "threshold" in pd
        assert "mean_before" in pd
        assert "mean_after" in pd

    def test_drift_response_has_required_keys(self, test_client_with_drift):
        """All keys the Groovy DataSource script expects are present."""
        response = test_client_with_drift.get("/drift")
        data = response.json()

        # Keys the Scry_Drift.xml Groovy script will parse
        required_top_keys = ["feature_drift", "prediction_drift", "timestamp"]
        for key in required_top_keys:
            assert key in data, f"Missing required key: {key}"

        required_feature_keys = ["has_drift", "max_psi", "threshold"]
        for key in required_feature_keys:
            assert key in data["feature_drift"], f"Missing feature_drift key: {key}"

        required_prediction_keys = ["has_drift", "mean_before", "mean_after"]
        for key in required_prediction_keys:
            assert key in data["prediction_drift"], f"Missing prediction_drift key: {key}"


class TestDriftXmlStructure:
    """Tests for Scry_Drift.xml DataSource structure."""

    def test_drift_xml_structure_valid(self):
        """Parse XML and verify datapoint structure matches script DS pattern.

        Script DataSources use namevalue post-processing: rawDataFieldName must be
        "output" and postProcessorParam holds the key name parsed from Key=Value output.
        """
        xml_path = Path(__file__).parent.parent / "logicmodules" / "datasources" / "Scry_Drift.xml"
        if not xml_path.exists():
            pytest.skip("Scry_Drift.xml not yet created")

        tree = ET.parse(xml_path)
        root = tree.getroot()

        expected_datapoints = [
            "HasFeatureDrift",
            "MaxPsiValue",
            "PsiThreshold",
            "HasPredictionDrift",
            "DriftScore",
            "DriftThreshold",
            "MeanErrorBefore",
            "MeanErrorAfter",
            "ApiStatus",
            "ApiLatencyMs",
        ]

        # Verify each datapoint uses correct script DS pattern
        param_fields = []
        for dp in root.iter("dataPoint"):
            raw_field = dp.find("rawDataFieldName")
            post_method = dp.find("postProcessorMethod")
            post_param = dp.find("postProcessorParam")
            assert raw_field is not None and raw_field.text == "output", (
                f"rawDataFieldName must be 'output' for script DS, got '{raw_field.text if raw_field is not None else None}'"
            )
            assert post_method is not None and post_method.text == "namevalue", (
                "postProcessorMethod must be 'namevalue' for script DS"
            )
            if post_param is not None and post_param.text:
                param_fields.append(post_param.text)

        for field in expected_datapoints:
            assert field in param_fields, f"Missing postProcessorParam: {field}"
