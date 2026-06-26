# Description: Wrapper for Chronos-2 foundation model for time-series forecasting.
# Description: Provides per-metric and batch forecasting with quantile predictions.

"""Chronos-2 forecasting wrapper for infrastructure metrics."""


import numpy as np
import torch
from chronos import BaseChronosPipeline


class ChronosForecaster:
    """Wrapper around Chronos-2 for infrastructure metric forecasting.

    Provides quantile forecasts at configurable horizons. Supports both
    single-metric and batch (multi-metric) forecasting.

    Args:
        model_id: HuggingFace model ID (e.g. "amazon/chronos-2" or
            "amazon/chronos-bolt-tiny" for testing).
        device: Device for inference ("cpu", "cuda").
        horizons: List of forecast horizons in timesteps (e.g. [15, 60, 240, 1440]
            for 15min, 1h, 4h, 24h at 1-min resolution).
        quantile_levels: Quantile levels for prediction intervals.
    """

    def __init__(
        self,
        model_id: str = "amazon/chronos-bolt-tiny",
        device: str = "cpu",
        horizons: list[int] | None = None,
        quantile_levels: list[float] | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.horizons = horizons or [15, 60, 240, 1440]
        self.quantile_levels = quantile_levels or [0.1, 0.5, 0.9]

        self._pipeline = BaseChronosPipeline.from_pretrained(
            model_id,
            device_map=device,
        )

    @property
    def prediction_length(self) -> int:
        """Maximum horizon determines prediction length."""
        return max(self.horizons)

    def forecast(self, context: torch.Tensor) -> dict[str, np.ndarray]:
        """Forecast a single metric time series.

        Args:
            context: 1D tensor of historical values (context_length,).

        Returns:
            Dict with keys:
                - median: (prediction_length,) median forecast
                - lower: (prediction_length,) lower quantile
                - upper: (prediction_length,) upper quantile
        """
        # predict_quantiles expects list of tensors or 2D tensor
        quantiles, mean = self._pipeline.predict_quantiles(
            [context],
            prediction_length=self.prediction_length,
            quantile_levels=self.quantile_levels,
        )

        # quantiles shape: (1, prediction_length, n_quantiles)
        q = quantiles[0].numpy()

        return {
            "lower": q[:, 0],
            "median": q[:, 1],
            "upper": q[:, 2],
        }

    def forecast_batch(self, contexts: list[torch.Tensor]) -> dict[str, np.ndarray]:
        """Forecast multiple metric time series in a single batch.

        Args:
            contexts: List of 1D tensors, one per metric.

        Returns:
            Dict with keys:
                - median: (n_metrics, prediction_length) median forecasts
                - lower: (n_metrics, prediction_length) lower quantiles
                - upper: (n_metrics, prediction_length) upper quantiles
        """
        quantiles, mean = self._pipeline.predict_quantiles(
            contexts,
            prediction_length=self.prediction_length,
            quantile_levels=self.quantile_levels,
        )

        # quantiles shape: (n_metrics, prediction_length, n_quantiles)
        q = quantiles.numpy()

        return {
            "lower": q[:, :, 0],
            "median": q[:, :, 1],
            "upper": q[:, :, 2],
        }

    def extract_at_horizons(
        self, forecast: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Extract forecast values at specific horizon offsets.

        Args:
            forecast: Dict from forecast() with (prediction_length,) arrays.

        Returns:
            Dict with same keys but (n_horizons,) arrays.
        """
        result = {}
        for key, values in forecast.items():
            if values.ndim == 1:
                # Single metric: extract at horizon indices (0-indexed, so h-1)
                result[key] = np.array([values[h - 1] for h in self.horizons])
            else:
                # Batch: (n_metrics, prediction_length) -> (n_metrics, n_horizons)
                result[key] = np.column_stack(
                    [values[:, h - 1] for h in self.horizons]
                )
        return result
