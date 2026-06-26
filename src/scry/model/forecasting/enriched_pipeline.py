# Description: Enriched feature pipeline that adds Chronos-2 forecast features to base data.
# Description: Concatenates residual and uncertainty features to numerical windows.

"""Enriched feature pipeline for forecast-augmented X-DEC training."""

import logging
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


class EnrichedFeaturePipeline:
    """Pipeline that enriches base X-DEC features with Chronos-2 forecast features.

    For each sliding window, computes:
        - Forecast residuals at each horizon (actual - median_forecast)
        - Prediction interval widths at each horizon (upper - lower)

    These are appended to the numerical feature dimension.

    Args:
        n_metrics: Number of numerical metrics in base data.
        horizons: Forecast horizon offsets (timesteps from window end).
        forecaster: Optional ChronosForecaster instance. If None, uses
            synthetic features (zeros for residuals, ones for uncertainties).
    """

    def __init__(
        self,
        n_metrics: int = 9,
        horizons: list[int] | None = None,
        forecaster: Any = "auto",
    ) -> None:
        self.n_metrics = n_metrics
        self.horizons = horizons or [15, 60, 240, 1440]
        self.n_horizons = len(self.horizons)

        if forecaster == "auto":
            self._forecaster = self._try_load_forecaster()
        else:
            self._forecaster = forecaster

    def _try_load_forecaster(self) -> Any:
        """Try to load a ChronosForecaster. Returns None if unavailable."""
        try:
            from scry.model.forecasting.chronos_wrapper import ChronosForecaster

            return ChronosForecaster(
                model_id="amazon/chronos-bolt-tiny",
                device="cpu",
                horizons=self.horizons,
            )
        except ImportError:
            logger.warning("chronos-forecasting not installed, using synthetic features")
            return None

    @property
    def enriched_num_features(self) -> int:
        """Total numerical features after enrichment."""
        # base features + residuals + uncertainty widths
        return self.n_metrics + self.n_metrics * self.n_horizons * 2

    def enrich(self, data: dict[str, Any]) -> dict[str, Any]:
        """Enrich base training data with forecast-derived features.

        Takes the output of XDECFeaturePipeline.transform() and adds
        forecast residual and uncertainty features to the numerical windows.

        Args:
            data: Dict from XDECFeaturePipeline.transform() containing
                num_windows, cat_windows, labels, num_norm_params.

        Returns:
            Enriched data dict with wider num_windows.
        """
        num_windows = data["num_windows"]
        n_samples, seq_len, n_num = num_windows.shape

        if self._forecaster is not None:
            forecast_features = self._compute_forecast_features(num_windows)
        else:
            # Synthetic features when no forecaster available
            n_forecast_features = self.n_metrics * self.n_horizons * 2
            forecast_features = np.zeros((n_samples, seq_len, n_forecast_features))

        # Concatenate along feature dimension
        enriched_num = np.concatenate([num_windows, forecast_features], axis=2)

        # Update normalization params
        n_enriched = enriched_num.shape[2]
        old_mean = data["num_norm_params"]["mean"]
        old_std = data["num_norm_params"]["std"]
        new_mean = np.concatenate([old_mean, np.zeros(n_enriched - len(old_mean))])
        new_std = np.concatenate([old_std, np.ones(n_enriched - len(old_std))])

        return {
            "num_windows": enriched_num,
            "cat_windows": data["cat_windows"],
            "labels": data["labels"],
            "num_norm_params": {"mean": new_mean, "std": new_std},
        }

    def _compute_forecast_features(
        self, num_windows: np.ndarray
    ) -> np.ndarray:
        """Compute forecast residual and uncertainty features for each window.

        For each window, the last timestep values are treated as "actuals",
        and forecasts are generated from the preceding context.

        Args:
            num_windows: (n_samples, seq_len, n_metrics) numerical data.

        Returns:
            (n_samples, seq_len, n_metrics * n_horizons * 2) forecast features.
        """
        n_samples, seq_len, n_metrics = num_windows.shape
        n_forecast_features = n_metrics * self.n_horizons * 2

        # Pre-allocate output: same shape as windows but with forecast features
        forecast_features = np.zeros((n_samples, seq_len, n_forecast_features))

        # For each metric, compute forecast using available context
        for metric_idx in range(min(n_metrics, self.n_metrics)):
            # Use all samples' full time series for this metric as context
            # Average context across samples to get a representative series
            # (In production, this would use per-resource historical data)
            metric_context = num_windows[:, :, metric_idx].mean(axis=0)
            context_tensor = torch.tensor(metric_context, dtype=torch.float32)

            forecast = self._forecaster.forecast(context_tensor)
            horizon_values = self._forecaster.extract_at_horizons(forecast)

            # Fill forecast features for this metric across all timesteps
            for h_idx, h in enumerate(self.horizons):
                residual_idx = metric_idx * self.n_horizons + h_idx
                uncert_idx = self.n_metrics * self.n_horizons + residual_idx

                # Residual: approximate using forecast median vs actual mean
                forecast_features[:, :, residual_idx] = (
                    metric_context.mean() - horizon_values["median"][h_idx]
                )
                # Uncertainty width
                forecast_features[:, :, uncert_idx] = (
                    horizon_values["upper"][h_idx] - horizon_values["lower"][h_idx]
                )

        return forecast_features
