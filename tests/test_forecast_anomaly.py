# Description: Unit tests for forecast-based anomaly detection.
# Description: Tests detection logic, scoring, and multi-metric aggregation.

"""Tests for scry.model.forecasting.anomaly_detector module."""

import numpy as np
import pytest


class TestForecastAnomalyDetector:
    """Tests for ForecastAnomalyDetector class."""

    @pytest.fixture
    def detector(self):
        """Create a ForecastAnomalyDetector."""
        from scry.model.forecasting.anomaly_detector import ForecastAnomalyDetector

        return ForecastAnomalyDetector(metric_names=["cpu", "memory", "network"])

    def test_detect_anomaly_when_outside_interval(self, detector) -> None:
        """Should detect anomaly when actual exceeds upper prediction interval."""
        actuals = np.array([100.0, 50.0, 30.0])
        forecast = {
            "median": np.array([60.0, 48.0, 28.0]),
            "lower": np.array([40.0, 35.0, 20.0]),
            "upper": np.array([80.0, 60.0, 35.0]),
        }

        result = detector.detect(actuals, forecast)

        assert result["is_anomaly"] is True
        # cpu=100 > upper=80, so cpu should be violated
        assert "cpu" in result["violated_metrics"]

    def test_no_anomaly_within_interval(self, detector) -> None:
        """Should not detect anomaly when all actuals are within intervals."""
        actuals = np.array([60.0, 48.0, 28.0])
        forecast = {
            "median": np.array([60.0, 48.0, 28.0]),
            "lower": np.array([40.0, 35.0, 20.0]),
            "upper": np.array([80.0, 60.0, 35.0]),
        }

        result = detector.detect(actuals, forecast)

        assert result["is_anomaly"] is False
        assert len(result["violated_metrics"]) == 0

    def test_anomaly_score_proportional_to_violation_magnitude(self, detector) -> None:
        """Anomaly score should be larger for bigger violations."""
        forecast = {
            "median": np.array([50.0, 50.0, 50.0]),
            "lower": np.array([40.0, 40.0, 40.0]),
            "upper": np.array([60.0, 60.0, 60.0]),
        }

        # Small violation
        actuals_small = np.array([65.0, 50.0, 50.0])
        result_small = detector.detect(actuals_small, forecast)

        # Large violation
        actuals_large = np.array([100.0, 50.0, 50.0])
        result_large = detector.detect(actuals_large, forecast)

        assert result_large["anomaly_score"] > result_small["anomaly_score"]

    def test_multi_metric_anomaly_aggregation(self, detector) -> None:
        """Should aggregate anomalies across multiple metrics."""
        actuals = np.array([100.0, 100.0, 28.0])  # cpu and memory violated
        forecast = {
            "median": np.array([60.0, 48.0, 28.0]),
            "lower": np.array([40.0, 35.0, 20.0]),
            "upper": np.array([80.0, 60.0, 35.0]),
        }

        result = detector.detect(actuals, forecast)

        assert result["is_anomaly"] is True
        assert "cpu" in result["violated_metrics"]
        assert "memory" in result["violated_metrics"]
        assert result["severity"] in ["low", "medium", "high", "critical"]

    def test_detect_returns_per_metric_scores(self, detector) -> None:
        """Detection result should include per-metric anomaly scores."""
        actuals = np.array([100.0, 50.0, 30.0])
        forecast = {
            "median": np.array([60.0, 48.0, 28.0]),
            "lower": np.array([40.0, 35.0, 20.0]),
            "upper": np.array([80.0, 60.0, 35.0]),
        }

        result = detector.detect(actuals, forecast)

        assert "per_metric_scores" in result
        assert len(result["per_metric_scores"]) == 3

    def test_below_lower_also_detected(self, detector) -> None:
        """Anomaly below the lower bound should also be detected."""
        actuals = np.array([10.0, 48.0, 28.0])  # cpu=10 < lower=40
        forecast = {
            "median": np.array([60.0, 48.0, 28.0]),
            "lower": np.array([40.0, 35.0, 20.0]),
            "upper": np.array([80.0, 60.0, 35.0]),
        }

        result = detector.detect(actuals, forecast)

        assert result["is_anomaly"] is True
        assert "cpu" in result["violated_metrics"]
