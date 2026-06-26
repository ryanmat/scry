# Description: Unit tests for feature and concept drift detection.
# Description: Tests PSI for feature drift and ADWIN for concept drift.

"""Tests for scry.model.drift module."""

import numpy as np


class TestPopulationStabilityIndex:
    """Tests for PSI-based feature drift detection."""

    def test_psi_detects_feature_distribution_shift(self) -> None:
        """PSI should detect a shifted distribution."""
        from scry.model.drift import DriftDetector

        detector = DriftDetector(n_features=3, feature_names=["cpu", "memory", "network"])

        # Reference: normal distribution centered at 0
        rng = np.random.default_rng(42)
        reference = rng.normal(0, 1, (1000, 3))

        # Current: shifted distribution centered at 3
        current = rng.normal(3, 1, (1000, 3))

        result = detector.check_feature_drift(reference, current)

        assert result["has_drift"] is True
        # PSI > 0.2 indicates significant drift
        for feature_name in ["cpu", "memory", "network"]:
            assert result["psi_per_feature"][feature_name] > 0.2

    def test_psi_stable_for_identical_distributions(self) -> None:
        """PSI should be near zero for same distribution."""
        from scry.model.drift import DriftDetector

        detector = DriftDetector(n_features=2, feature_names=["cpu", "memory"])

        rng = np.random.default_rng(42)
        reference = rng.normal(0, 1, (1000, 2))
        current = rng.normal(0, 1, (1000, 2))

        result = detector.check_feature_drift(reference, current)

        assert result["has_drift"] is False
        for feature_name in ["cpu", "memory"]:
            assert result["psi_per_feature"][feature_name] < 0.2


class TestAdaptiveWindowing:
    """Tests for ADWIN-based concept drift detection."""

    def test_adwin_detects_error_rate_change(self) -> None:
        """ADWIN should detect a change point in error rate."""
        from scry.model.drift import DriftDetector

        detector = DriftDetector(n_features=1, feature_names=["metric"])

        # Error stream: low errors then sudden increase
        rng = np.random.default_rng(42)
        errors_stable = rng.uniform(0, 0.1, 500)
        errors_drifted = rng.uniform(0.4, 0.6, 500)
        error_stream = np.concatenate([errors_stable, errors_drifted])

        result = detector.check_prediction_drift(error_stream)

        assert result["has_drift"] is True
        assert result["change_point_index"] is not None

    def test_adwin_stable_for_constant_errors(self) -> None:
        """ADWIN should not detect drift for constant error rate."""
        from scry.model.drift import DriftDetector

        detector = DriftDetector(n_features=1, feature_names=["metric"])

        # Stable error rate with very low variance
        rng = np.random.default_rng(42)
        error_stream = rng.uniform(0.14, 0.16, 1000)

        result = detector.check_prediction_drift(error_stream)

        assert result["has_drift"] is False


class TestDriftDetectorCombined:
    """Tests for combined drift detection."""

    def test_drift_status_dict_structure(self) -> None:
        """Drift status should return structured dict."""
        from scry.model.drift import DriftDetector

        detector = DriftDetector(n_features=2, feature_names=["cpu", "memory"])

        rng = np.random.default_rng(42)
        reference = rng.normal(0, 1, (1000, 2))
        current = rng.normal(0, 1, (1000, 2))
        errors = np.random.uniform(0.1, 0.2, 500)

        status = detector.get_drift_status(reference, current, errors)

        assert "feature_drift" in status
        assert "prediction_drift" in status
        assert "timestamp" in status
