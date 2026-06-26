# Description: Compute residual features from forecast predictions vs actuals.
# Description: Produces residual and uncertainty width features for X-DEC enrichment.

"""Residual feature computation for forecast-enriched X-DEC training."""


import numpy as np


def compute_residual_features(
    actuals: np.ndarray,
    forecast: dict[str, np.ndarray],
) -> np.ndarray:
    """Compute residual and uncertainty features from forecasts vs actuals.

    For each (metric, horizon) pair, computes:
        - residual = actual - median_forecast
        - uncertainty_width = upper - lower

    Args:
        actuals: Array of shape (n_samples, n_metrics, n_horizons).
        forecast: Dict with keys "median", "lower", "upper", each
            of shape (n_samples, n_metrics, n_horizons).

    Returns:
        Feature array of shape (n_samples, n_metrics * n_horizons * 2).
        First half is residuals, second half is uncertainty widths.
    """
    n_samples, n_metrics, n_horizons = actuals.shape

    # Residuals: actual - median forecast
    residuals = actuals - forecast["median"]

    # Uncertainty widths: upper - lower quantile
    uncertainties = forecast["upper"] - forecast["lower"]

    # Flatten metric and horizon dimensions
    residuals_flat = residuals.reshape(n_samples, n_metrics * n_horizons)
    uncertainties_flat = uncertainties.reshape(n_samples, n_metrics * n_horizons)

    # Concatenate: [residuals | uncertainties]
    return np.concatenate([residuals_flat, uncertainties_flat], axis=1)
