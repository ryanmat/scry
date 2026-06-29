# Description: Tests for /anomaly API endpoint and DataSource contract.
# Description: Validates anomaly response keys and severity mapping for LM integration.

"""Tests for anomaly endpoint and DataSource contract."""

import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from scry.model import TemporalXDEC
from scry.model.forecasting.anomaly_detector import ForecastAnomalyDetector

METRIC_NAMES = [
    "cpuUsagePercent",
    "memoryUsagePercent",
    "networkBytesIn",
    "networkBytesOut",
    "podRestartCount",
    "containerCpuRequest",
    "containerCpuLimit",
    "containerMemoryRequest",
    "containerMemoryLimit",
]


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
def test_client_with_anomaly(temp_model_path):
    """Create a test client with anomaly detector configured."""
    with patch.dict("os.environ", {"MODEL_PATH": str(temp_model_path)}):
        from scry.api.main import create_app

        app = create_app(model_path=str(temp_model_path))

        # Configure anomaly detector on app state
        detector = ForecastAnomalyDetector(metric_names=METRIC_NAMES)
        app.state.anomaly_detector = detector

        # Provide last actuals and forecast data for detection
        app.state.last_actuals = np.array([25.0, 40.0, 1e6, 5e5, 0.0, 100.0, 500.0, 128.0, 512.0])
        app.state.last_forecast = {
            "median": np.array([24.0, 39.0, 1e6, 5e5, 0.0, 100.0, 500.0, 128.0, 512.0]),
            "lower": np.array([20.0, 35.0, 8e5, 4e5, 0.0, 90.0, 450.0, 120.0, 480.0]),
            "upper": np.array([30.0, 45.0, 1.2e6, 6e5, 1.0, 110.0, 550.0, 140.0, 540.0]),
        }

        client = TestClient(app)
        yield client


class TestAnomalyEndpointNoForecaster:
    """Tests for /anomaly when no detector is configured."""

    def test_anomaly_endpoint_no_forecaster(self, test_client):
        """Returns graceful fallback when no anomaly detector configured."""
        response = test_client.get("/anomaly")
        assert response.status_code == 200

        data = response.json()
        assert data["is_anomaly"] is False
        assert data["anomaly_score"] == 0.0
        assert data["violated_metrics"] == []
        assert data["severity"] == "low"
        assert "timestamp" in data


class TestAnomalyEndpointWithDetector:
    """Tests for /anomaly when detector is configured."""

    def test_anomaly_endpoint_with_detector(self, test_client_with_anomaly):
        """Returns anomaly detection results when detector is active."""
        response = test_client_with_anomaly.get("/anomaly")
        assert response.status_code == 200

        data = response.json()
        assert "is_anomaly" in data
        assert "anomaly_score" in data
        assert "violated_metrics" in data
        assert "severity" in data
        assert "metric_count" in data
        assert "timestamp" in data
        assert isinstance(data["violated_metrics"], list)
        assert isinstance(data["anomaly_score"], (int, float))
        assert data["metric_count"] == len(METRIC_NAMES)

    def test_anomaly_response_keys_match_datasource(self, test_client_with_anomaly):
        """All keys the Groovy DataSource script expects are present."""
        response = test_client_with_anomaly.get("/anomaly")
        data = response.json()

        required_keys = [
            "is_anomaly",
            "anomaly_score",
            "violated_metrics",
            "severity",
            "metric_count",
            "timestamp",
        ]
        for key in required_keys:
            assert key in data, f"Missing required key: {key}"


class TestAnomalySeverityMapping:
    """Tests for anomaly severity mapping."""

    def test_anomaly_severity_mapping(self):
        """Severity levels map correctly: low=1, medium=2, high=3, critical=4."""
        severity_map = {"low": 1, "medium": 2, "high": 3, "critical": 4}

        detector = ForecastAnomalyDetector(metric_names=["cpu"])

        # No violation => low severity
        actuals = np.array([25.0])
        forecast = {
            "median": np.array([25.0]),
            "lower": np.array([20.0]),
            "upper": np.array([30.0]),
        }
        result = detector.detect(actuals, forecast)
        assert result["severity"] == "low"
        assert severity_map[result["severity"]] == 1

        # Medium violation: score between 0.5 and 1.0
        actuals_med = np.array([33.0])  # 3 above upper(30), width=10, score=0.3... adjust
        forecast_med = {
            "median": np.array([25.0]),
            "lower": np.array([20.0]),
            "upper": np.array([30.0]),
        }
        # score = (35 - 30) / (30 - 20) = 0.5 => medium
        actuals_med = np.array([35.0])
        result_med = detector.detect(actuals_med, forecast_med)
        assert result_med["severity"] == "medium"
        assert severity_map[result_med["severity"]] == 2

        # High violation: score between 1.0 and 2.0
        # score = (40 - 30) / 10 = 1.0 => high
        actuals_high = np.array([40.0])
        result_high = detector.detect(actuals_high, forecast_med)
        assert result_high["severity"] == "high"
        assert severity_map[result_high["severity"]] == 3

        # Critical violation: score >= 2.0
        # score = (50 - 30) / 10 = 2.0 => critical
        actuals_crit = np.array([50.0])
        result_crit = detector.detect(actuals_crit, forecast_med)
        assert result_crit["severity"] == "critical"
        assert severity_map[result_crit["severity"]] == 4


class TestAnomalyXmlStructure:
    """Tests for Scry_Anomaly.xml DataSource structure."""

    def test_anomaly_xml_structure_valid(self):
        """Parse XML and verify datapoint structure matches script DS pattern.

        Script DataSources use namevalue post-processing: rawDataFieldName must be
        "output" and postProcessorParam holds the key name parsed from Key=Value output.
        """
        xml_path = Path(__file__).parent.parent / "logicmodules" / "datasources" / "Scry_Anomaly.xml"
        if not xml_path.exists():
            pytest.skip("Scry_Anomaly.xml not yet created")

        tree = ET.parse(xml_path)
        root = tree.getroot()

        expected_datapoints = [
            "IsAnomaly",
            "AnomalyScore",
            "ViolatedMetricCount",
            "Severity",
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
