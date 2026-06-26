# Description: Tests for ForecastAccuracyTracker covering per-horizon and stability metrics.
# Description: Validates PICP, MPIW, MAE, MASE, transition rate, confidence stability.

"""Tests for ForecastAccuracyTracker."""

import math

import pytest

from scry.model.forecasting.accuracy import ForecastAccuracyTracker


class TestPICP:
    """Tests for Prediction Interval Coverage Probability."""

    def test_picp_all_covered(self):
        """All actuals inside interval yields PICP = 1.0."""
        tracker = ForecastAccuracyTracker(horizons=["15m"])
        for actual in [1.0, 2.0, 3.0, 4.0, 5.0]:
            tracker.record_forecast("15m", actual=actual, median=actual, lower=actual - 1, upper=actual + 1)

        metrics = tracker.compute_metrics()
        assert metrics["horizons"]["15m"]["picp"] == pytest.approx(1.0)

    def test_picp_none_covered(self):
        """All actuals outside interval yields PICP = 0.0."""
        tracker = ForecastAccuracyTracker(horizons=["15m"])
        for actual in [10.0, 20.0, 30.0]:
            tracker.record_forecast("15m", actual=actual, median=0.0, lower=0.0, upper=1.0)

        metrics = tracker.compute_metrics()
        assert metrics["horizons"]["15m"]["picp"] == pytest.approx(0.0)

    def test_picp_partial_coverage(self):
        """3 of 5 actuals inside interval yields PICP = 0.6."""
        tracker = ForecastAccuracyTracker(horizons=["15m"])
        # 3 covered
        for actual in [1.0, 2.0, 3.0]:
            tracker.record_forecast("15m", actual=actual, median=actual, lower=0.0, upper=5.0)
        # 2 not covered
        for actual in [10.0, 20.0]:
            tracker.record_forecast("15m", actual=actual, median=0.0, lower=0.0, upper=5.0)

        metrics = tracker.compute_metrics()
        assert metrics["horizons"]["15m"]["picp"] == pytest.approx(0.6)


class TestMPIW:
    """Tests for Mean Prediction Interval Width."""

    def test_mpiw_calculation(self):
        """MPIW is the mean of (upper - lower) across observations."""
        tracker = ForecastAccuracyTracker(horizons=["1h"])
        # widths: 2.0, 4.0, 6.0 -> mean = 4.0
        tracker.record_forecast("1h", actual=1.0, median=1.0, lower=0.0, upper=2.0)
        tracker.record_forecast("1h", actual=3.0, median=3.0, lower=1.0, upper=5.0)
        tracker.record_forecast("1h", actual=5.0, median=5.0, lower=2.0, upper=8.0)

        metrics = tracker.compute_metrics()
        assert metrics["horizons"]["1h"]["mpiw"] == pytest.approx(4.0)


class TestMAE:
    """Tests for Mean Absolute Error."""

    def test_mae_calculation(self):
        """MAE is the mean of |actual - median| across observations."""
        tracker = ForecastAccuracyTracker(horizons=["4h"])
        # errors: |1-2|=1, |3-1|=2, |5-6|=1 -> mean = 4/3
        tracker.record_forecast("4h", actual=1.0, median=2.0, lower=0.0, upper=4.0)
        tracker.record_forecast("4h", actual=3.0, median=1.0, lower=0.0, upper=4.0)
        tracker.record_forecast("4h", actual=5.0, median=6.0, lower=0.0, upper=8.0)

        metrics = tracker.compute_metrics()
        assert metrics["horizons"]["4h"]["mae"] == pytest.approx(4.0 / 3.0)


class TestMASE:
    """Tests for Mean Absolute Scaled Error."""

    def test_mase_above_one_worse_than_naive(self):
        """MASE > 1.0 means model is worse than naive persistence forecast."""
        tracker = ForecastAccuracyTracker(horizons=["24h"])
        # actuals: 1, 2, 3 -> naive errors: |2-1|=1, |3-2|=1 -> mean naive = 1.0
        # medians: 5, 5, 5 -> model errors: |1-5|=4, |2-5|=3, |3-5|=2 -> mean model = 3.0
        # MASE = 3.0 / 1.0 = 3.0
        tracker.record_forecast("24h", actual=1.0, median=5.0, lower=0.0, upper=10.0)
        tracker.record_forecast("24h", actual=2.0, median=5.0, lower=0.0, upper=10.0)
        tracker.record_forecast("24h", actual=3.0, median=5.0, lower=0.0, upper=10.0)

        metrics = tracker.compute_metrics()
        assert metrics["horizons"]["24h"]["mase"] > 1.0

    def test_mase_below_one_better_than_naive(self):
        """MASE < 1.0 means model is better than naive persistence forecast."""
        tracker = ForecastAccuracyTracker(horizons=["15m"])
        # actuals: 1, 3, 5 -> naive errors: |3-1|=2, |5-3|=2 -> mean naive = 2.0
        # medians: 1.1, 2.9, 5.1 -> model errors: 0.1, 0.1, 0.1 -> mean model = 0.1
        # MASE = 0.1 / 2.0 = 0.05
        tracker.record_forecast("15m", actual=1.0, median=1.1, lower=0.0, upper=2.0)
        tracker.record_forecast("15m", actual=3.0, median=2.9, lower=2.0, upper=4.0)
        tracker.record_forecast("15m", actual=5.0, median=5.1, lower=4.0, upper=6.0)

        metrics = tracker.compute_metrics()
        assert metrics["horizons"]["15m"]["mase"] < 1.0


class TestPerHorizon:
    """Tests for multi-horizon metric computation."""

    def test_per_horizon_metrics(self):
        """Returns a dict keyed by each horizon name with all four metrics."""
        horizons = ["15m", "1h", "4h", "24h"]
        tracker = ForecastAccuracyTracker(horizons=horizons)
        for h in horizons:
            tracker.record_forecast(h, actual=1.0, median=1.0, lower=0.0, upper=2.0)
            tracker.record_forecast(h, actual=2.0, median=2.0, lower=1.0, upper=3.0)

        metrics = tracker.compute_metrics()
        for h in horizons:
            assert h in metrics["horizons"]
            assert "picp" in metrics["horizons"][h]
            assert "mpiw" in metrics["horizons"][h]
            assert "mae" in metrics["horizons"][h]
            assert "mase" in metrics["horizons"][h]


class TestTransitionRate:
    """Tests for cluster transition rate metric."""

    def test_transition_rate_stable(self):
        """Same cluster repeated yields transition rate = 0.0."""
        tracker = ForecastAccuracyTracker(horizons=["15m"])
        for _ in range(10):
            tracker.record_cluster(cluster_id=0, confidence=0.9)

        metrics = tracker.compute_metrics()
        assert metrics["stability"]["transition_rate"] == pytest.approx(0.0)

    def test_transition_rate_unstable(self):
        """Alternating clusters yields transition rate = 1.0."""
        tracker = ForecastAccuracyTracker(horizons=["15m"])
        for i in range(10):
            tracker.record_cluster(cluster_id=i % 2, confidence=0.9)

        metrics = tracker.compute_metrics()
        assert metrics["stability"]["transition_rate"] == pytest.approx(1.0)


class TestConfidenceStability:
    """Tests for confidence standard deviation metric."""

    def test_confidence_stability(self):
        """Constant confidence yields std dev near 0.0."""
        tracker = ForecastAccuracyTracker(horizons=["15m"])
        for _ in range(10):
            tracker.record_cluster(cluster_id=0, confidence=0.85)

        metrics = tracker.compute_metrics()
        assert metrics["stability"]["confidence_std"] == pytest.approx(0.0, abs=1e-10)


class TestDominantClusterPct:
    """Tests for dominant cluster percentage metric."""

    def test_dominant_cluster_pct(self):
        """80% in cluster 0 yields dominant_cluster_pct = 80.0."""
        tracker = ForecastAccuracyTracker(horizons=["15m"])
        for _ in range(8):
            tracker.record_cluster(cluster_id=0, confidence=0.9)
        for _ in range(2):
            tracker.record_cluster(cluster_id=1, confidence=0.9)

        metrics = tracker.compute_metrics()
        assert metrics["stability"]["dominant_cluster_pct"] == pytest.approx(80.0)


class TestRecordAndComputeCycle:
    """Tests for the full record -> compute workflow."""

    def test_record_and_compute_cycle(self):
        """Full workflow: record observations, compute metrics, verify structure."""
        tracker = ForecastAccuracyTracker(horizons=["15m", "1h"])

        # Record forecasts
        for _ in range(5):
            tracker.record_forecast("15m", actual=1.0, median=1.0, lower=0.0, upper=2.0)
            tracker.record_forecast("1h", actual=2.0, median=2.5, lower=1.0, upper=4.0)

        # Record clusters
        for _ in range(5):
            tracker.record_cluster(cluster_id=0, confidence=0.9)

        metrics = tracker.compute_metrics()

        # Top-level keys
        assert "horizons" in metrics
        assert "stability" in metrics
        assert "observation_count" in metrics
        assert "timestamp" in metrics

        # Observation count matches
        assert metrics["observation_count"] == 5

        # All horizon metrics present
        assert "15m" in metrics["horizons"]
        assert "1h" in metrics["horizons"]


class TestEmptyHistory:
    """Tests for behavior with no recorded observations."""

    def test_empty_history_returns_defaults(self):
        """No observations yield safe default values."""
        tracker = ForecastAccuracyTracker(horizons=["15m", "1h"])
        metrics = tracker.compute_metrics()

        # Horizons should return 0.0 for all metrics when empty
        for h in ["15m", "1h"]:
            assert metrics["horizons"][h]["picp"] == pytest.approx(0.0)
            assert metrics["horizons"][h]["mpiw"] == pytest.approx(0.0)
            assert metrics["horizons"][h]["mae"] == pytest.approx(0.0)
            # MASE should be NaN when no data (no naive baseline)
            assert math.isnan(metrics["horizons"][h]["mase"])

        # Stability defaults
        assert metrics["stability"]["transition_rate"] == pytest.approx(0.0)
        assert metrics["stability"]["confidence_std"] == pytest.approx(0.0)
        assert metrics["stability"]["dominant_cluster_pct"] == pytest.approx(0.0)
        assert metrics["observation_count"] == 0
