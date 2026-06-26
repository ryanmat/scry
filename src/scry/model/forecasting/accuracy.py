# Description: Forecast accuracy tracker for per-horizon and cluster stability metrics.
# Description: Computes PICP, MPIW, MAE, MASE and cluster transition/confidence metrics.

"""Forecast accuracy tracking for prediction quality measurement."""

from collections import deque
from datetime import datetime, timezone

import numpy as np


class ForecastAccuracyTracker:
    """Tracks forecast accuracy per horizon and cluster stability.

    Maintains rolling buffers of actual vs forecast observations per horizon,
    plus cluster assignment history. Computes coverage, width, error, and
    stability metrics without requiring ground truth labels.

    Args:
        horizons: List of horizon names (e.g. ["15m", "1h", "4h", "24h"]).
        max_history: Maximum observations to retain per horizon.
    """

    def __init__(self, horizons: list[str], max_history: int = 500) -> None:
        self.horizons = horizons
        self.max_history = max_history

        # Per-horizon rolling buffers: {horizon: deque of (actual, median, lower, upper)}
        self._forecast_history: dict[str, deque] = {
            h: deque(maxlen=max_history) for h in horizons
        }

        # Cluster assignment rolling buffer: deque of (cluster_id, confidence)
        self._cluster_history: deque = deque(maxlen=max_history)

    def record_forecast(
        self,
        horizon_name: str,
        actual: float,
        median: float,
        lower: float,
        upper: float,
    ) -> None:
        """Record one actual vs forecast observation for a horizon.

        Args:
            horizon_name: Which horizon this observation belongs to.
            actual: The observed actual value.
            median: The forecast median (point estimate).
            lower: Lower bound of prediction interval.
            upper: Upper bound of prediction interval.
        """
        if horizon_name not in self._forecast_history:
            return
        self._forecast_history[horizon_name].append((actual, median, lower, upper))

    def record_cluster(self, cluster_id: int, confidence: float) -> None:
        """Record one cluster prediction.

        Args:
            cluster_id: Assigned cluster ID.
            confidence: Prediction confidence (0-1).
        """
        self._cluster_history.append((cluster_id, confidence))

    def _compute_horizon_metrics(self, horizon_name: str) -> dict[str, float]:
        """Compute PICP, MPIW, MAE, and MASE for a single horizon.

        Args:
            horizon_name: Horizon to compute metrics for.

        Returns:
            Dict with picp, mpiw, mae, mase values.
        """
        history = self._forecast_history[horizon_name]

        if len(history) == 0:
            return {"picp": 0.0, "mpiw": 0.0, "mae": 0.0, "mase": float("nan")}

        actuals = np.array([obs[0] for obs in history])
        medians = np.array([obs[1] for obs in history])
        lowers = np.array([obs[2] for obs in history])
        uppers = np.array([obs[3] for obs in history])

        # PICP: fraction of actuals inside [lower, upper]
        covered = np.sum((actuals >= lowers) & (actuals <= uppers))
        picp = float(covered / len(actuals))

        # MPIW: mean prediction interval width
        mpiw = float(np.mean(uppers - lowers))

        # MAE: mean absolute error between median and actual
        mae = float(np.mean(np.abs(actuals - medians)))

        # MASE: MAE normalized by naive forecast (persistence model) error
        # Naive forecast = previous actual value
        if len(actuals) < 2:
            mase = float("nan")
        else:
            naive_errors = np.abs(np.diff(actuals))
            mean_naive = float(np.mean(naive_errors))
            if mean_naive < 1e-10:
                mase = float("nan")
            else:
                mase = mae / mean_naive

        return {"picp": picp, "mpiw": mpiw, "mae": mae, "mase": mase}

    def _compute_stability_metrics(self) -> dict[str, float]:
        """Compute cluster stability metrics from assignment history.

        Returns:
            Dict with transition_rate, confidence_std, dominant_cluster_pct.
        """
        if len(self._cluster_history) == 0:
            return {
                "transition_rate": 0.0,
                "confidence_std": 0.0,
                "dominant_cluster_pct": 0.0,
            }

        cluster_ids = [obs[0] for obs in self._cluster_history]
        confidences = np.array([obs[1] for obs in self._cluster_history])

        # Transition rate: fraction of consecutive predictions that change cluster
        if len(cluster_ids) < 2:
            transition_rate = 0.0
        else:
            transitions = sum(
                1 for i in range(1, len(cluster_ids)) if cluster_ids[i] != cluster_ids[i - 1]
            )
            transition_rate = transitions / (len(cluster_ids) - 1)

        # Confidence stability: std dev of confidence values
        confidence_std = float(np.std(confidences))

        # Dominant cluster: percentage of time in most common cluster
        from collections import Counter

        counts = Counter(cluster_ids)
        dominant_count = counts.most_common(1)[0][1]
        dominant_cluster_pct = (dominant_count / len(cluster_ids)) * 100.0

        return {
            "transition_rate": transition_rate,
            "confidence_std": confidence_std,
            "dominant_cluster_pct": dominant_cluster_pct,
        }

    def compute_metrics(self) -> dict:
        """Compute all accuracy and stability metrics.

        Returns:
            Dict with:
                - horizons: per-horizon metrics (picp, mpiw, mae, mase)
                - stability: cluster stability metrics
                - observation_count: total cluster observations recorded
                - timestamp: ISO 8601 timestamp
        """
        horizons_metrics = {
            h: self._compute_horizon_metrics(h) for h in self.horizons
        }
        stability_metrics = self._compute_stability_metrics()

        return {
            "horizons": horizons_metrics,
            "stability": stability_metrics,
            "observation_count": len(self._cluster_history),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
