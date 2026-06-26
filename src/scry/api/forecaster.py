# Description: Singleton wrapper around ChronosForecaster for API use.
# Description: Lazy-loads the Chronos model on first request, reuses across calls.

"""Forecaster service for the API."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

log = logging.getLogger(__name__)


class Forecaster:
    """Singleton forecaster service wrapping ChronosForecaster.

    Lazy-loads the Chronos foundation model on first call.
    Provides a forecast_metrics method that takes multiple metrics
    and returns per-metric, per-horizon forecasts with quantiles.

    Args:
        model_id: HuggingFace model ID for Chronos.
        device: Inference device (cpu, cuda).
        horizons: Forecast horizons in timesteps.
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
        self._forecaster: Any = None

    def _load(self) -> None:
        """Lazy-load the ChronosForecaster on first use."""
        if self._forecaster is not None:
            return

        from scry.model.forecasting.chronos_wrapper import ChronosForecaster

        log.info(
            "Loading ChronosForecaster model=%s device=%s horizons=%s",
            self.model_id,
            self.device,
            self.horizons,
        )
        self._forecaster = ChronosForecaster(
            model_id=self.model_id,
            device=self.device,
            horizons=self.horizons,
            quantile_levels=self.quantile_levels,
        )
        log.info("ChronosForecaster loaded")

    @property
    def is_loaded(self) -> bool:
        """Whether the Chronos model is loaded."""
        return self._forecaster is not None

    def forecast_metrics(
        self,
        metrics: dict[str, list[float]],
    ) -> list[dict[str, Any]]:
        """Forecast multiple metrics and extract values at configured horizons.

        Args:
            metrics: Dict mapping metric names to historical time series values.

        Returns:
            List of per-metric forecast dicts, each containing:
                - metric_name: str
                - horizons: list of {horizon, median, lower, upper} dicts
        """
        self._load()

        metric_names = list(metrics.keys())
        contexts = [
            torch.tensor(values, dtype=torch.float32)
            for values in metrics.values()
        ]

        batch_forecast = self._forecaster.forecast_batch(contexts)
        at_horizons = self._forecaster.extract_at_horizons(batch_forecast)

        results = []
        for i, name in enumerate(metric_names):
            horizon_results = []
            for j, h in enumerate(self.horizons):
                horizon_results.append({
                    "horizon": h,
                    "median": float(np.round(at_horizons["median"][i, j], 4)),
                    "lower": float(np.round(at_horizons["lower"][i, j], 4)),
                    "upper": float(np.round(at_horizons["upper"][i, j], 4)),
                })
            results.append({
                "metric_name": name,
                "horizons": horizon_results,
            })

        return results
