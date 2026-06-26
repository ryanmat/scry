# Description: Unit tests for the Chronos-2 forecasting wrapper.
# Description: Tests forecast output structure, shapes, and prediction interval ordering.

"""Tests for scry.model.forecasting.chronos_wrapper module.

Uses amazon/chronos-bolt-tiny (9MB) for fast real inference.
"""

import pytest
import torch

# Skip all tests if chronos-forecasting is not installed
chronos = pytest.importorskip("chronos")


@pytest.fixture(scope="module")
def forecaster():
    """Create a ChronosForecaster with tiny model (shared across tests)."""
    from scry.model.forecasting.chronos_wrapper import ChronosForecaster

    return ChronosForecaster(
        model_id="amazon/chronos-bolt-tiny",
        device="cpu",
        horizons=[15, 60],
        quantile_levels=[0.1, 0.5, 0.9],
    )


@pytest.fixture
def sample_context() -> torch.Tensor:
    """Sample context tensor for a single metric (512 timesteps)."""
    return torch.randn(512)


@pytest.fixture
def sample_context_multi() -> list[torch.Tensor]:
    """Sample context tensors for multiple metrics (9 metrics, 512 timesteps each)."""
    return [torch.randn(512) for _ in range(9)]


class TestChronosForecaster:
    """Tests for ChronosForecaster class."""

    def test_forecast_returns_expected_keys(self, forecaster, sample_context) -> None:
        """forecast should return dict with median, lower, upper keys."""
        result = forecaster.forecast(sample_context)

        assert "median" in result
        assert "lower" in result
        assert "upper" in result

    def test_forecast_shapes_match_horizons(self, forecaster, sample_context) -> None:
        """Forecast output shapes should match the configured horizons."""
        result = forecaster.forecast(sample_context)

        # With horizons=[15, 60], max horizon is 60
        # Output should have prediction_length = max(horizons) = 60
        assert result["median"].shape[0] == max(forecaster.horizons)
        assert result["lower"].shape[0] == max(forecaster.horizons)
        assert result["upper"].shape[0] == max(forecaster.horizons)

    def test_prediction_intervals_ordered(self, forecaster, sample_context) -> None:
        """lower <= median <= upper should hold (approximately, for quantiles)."""
        result = forecaster.forecast(sample_context)

        # For quantile forecasts, lower (10th) <= median (50th) <= upper (90th)
        assert (result["lower"] <= result["median"] + 1e-6).all()
        assert (result["median"] <= result["upper"] + 1e-6).all()

    def test_multivariate_forecast_handles_multiple_metrics(
        self, forecaster, sample_context_multi
    ) -> None:
        """forecast_batch should handle multiple metric time series."""
        result = forecaster.forecast_batch(sample_context_multi)

        assert "median" in result
        # Should have results for each metric: (n_metrics, max_horizon)
        assert result["median"].shape[0] == 9
        assert result["median"].shape[1] == max(forecaster.horizons)

    def test_forecast_at_specific_horizons(self, forecaster, sample_context) -> None:
        """forecast_at_horizons should return values at specified horizon offsets."""
        result = forecaster.forecast(sample_context)
        horizon_values = forecaster.extract_at_horizons(result)

        # Should have one value per horizon
        assert len(horizon_values["median"]) == len(forecaster.horizons)
