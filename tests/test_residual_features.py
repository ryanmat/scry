# Description: Unit tests for residual feature computation from forecasts.
# Description: Tests shape, zero residuals, uncertainty width, and finiteness.

"""Tests for scry.model.forecasting.residual_features module."""

import numpy as np


class TestResidualFeatures:
    """Tests for compute_residual_features function."""

    def test_residual_shape(self) -> None:
        """Output shape should be (n_samples, n_metrics * n_horizons * 2)."""
        from scry.model.forecasting.residual_features import compute_residual_features

        n_metrics = 9
        n_horizons = 4
        n_samples = 50

        actuals = np.random.randn(n_samples, n_metrics, n_horizons)
        forecast = {
            "median": np.random.randn(n_samples, n_metrics, n_horizons),
            "lower": np.random.randn(n_samples, n_metrics, n_horizons) - 1,
            "upper": np.random.randn(n_samples, n_metrics, n_horizons) + 1,
        }

        result = compute_residual_features(actuals, forecast)

        # 9 metrics * 4 horizons = 36 residuals + 36 uncertainties = 72
        expected_features = n_metrics * n_horizons * 2
        assert result.shape == (n_samples, expected_features)

    def test_zero_residual_for_perfect_forecast(self) -> None:
        """When forecast median exactly matches actuals, residuals should be zero."""
        from scry.model.forecasting.residual_features import compute_residual_features

        n_metrics = 3
        n_horizons = 2
        n_samples = 10

        actuals = np.random.randn(n_samples, n_metrics, n_horizons)
        forecast = {
            "median": actuals.copy(),
            "lower": actuals - 0.5,
            "upper": actuals + 0.5,
        }

        result = compute_residual_features(actuals, forecast)

        # First half are residuals (should be zero), second half are uncertainties
        n_residuals = n_metrics * n_horizons
        residuals = result[:, :n_residuals]
        np.testing.assert_array_almost_equal(residuals, 0.0)

    def test_uncertainty_width_non_negative(self) -> None:
        """Uncertainty width (upper - lower) should be non-negative."""
        from scry.model.forecasting.residual_features import compute_residual_features

        n_metrics = 5
        n_horizons = 3
        n_samples = 20

        actuals = np.random.randn(n_samples, n_metrics, n_horizons)
        lower = np.random.randn(n_samples, n_metrics, n_horizons)
        upper = lower + np.abs(np.random.randn(n_samples, n_metrics, n_horizons))

        forecast = {
            "median": (lower + upper) / 2,
            "lower": lower,
            "upper": upper,
        }

        result = compute_residual_features(actuals, forecast)

        # Second half are uncertainty widths
        n_residuals = n_metrics * n_horizons
        uncertainties = result[:, n_residuals:]
        assert (uncertainties >= 0).all()

    def test_features_are_finite(self) -> None:
        """All output features should be finite (no NaN or inf)."""
        from scry.model.forecasting.residual_features import compute_residual_features

        n_metrics = 9
        n_horizons = 4
        n_samples = 30

        actuals = np.random.randn(n_samples, n_metrics, n_horizons)
        forecast = {
            "median": np.random.randn(n_samples, n_metrics, n_horizons),
            "lower": np.random.randn(n_samples, n_metrics, n_horizons) - 1,
            "upper": np.random.randn(n_samples, n_metrics, n_horizons) + 1,
        }

        result = compute_residual_features(actuals, forecast)

        assert np.isfinite(result).all()
