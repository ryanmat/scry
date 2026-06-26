# Description: Forecast-based anomaly detection using prediction intervals.
# Description: Detects anomalies when actuals fall outside Chronos-2 confidence intervals.

"""Forecast-based anomaly detector for infrastructure metrics."""


import numpy as np


class ForecastAnomalyDetector:
    """Detects anomalies by comparing actuals against forecast prediction intervals.

    An anomaly is flagged when the actual value falls outside the
    [lower, upper] quantile interval for any metric. The anomaly score
    is the max normalized distance outside the interval across all metrics.

    Args:
        metric_names: Names of the metrics being monitored.
        severity_thresholds: Dict mapping severity levels to score thresholds.
    """

    def __init__(
        self,
        metric_names: list[str],
        severity_thresholds: dict[str, float] | None = None,
    ) -> None:
        self.metric_names = metric_names
        self.severity_thresholds = severity_thresholds or {
            "low": 0.0,
            "medium": 0.5,
            "high": 1.0,
            "critical": 2.0,
        }

    def detect(
        self,
        actuals: np.ndarray,
        forecast: dict[str, np.ndarray],
    ) -> dict:
        """Detect anomalies by comparing actuals to prediction intervals.

        Args:
            actuals: Array of actual values (n_metrics,).
            forecast: Dict with "median", "lower", "upper" arrays (n_metrics,).

        Returns:
            Dict with:
                - is_anomaly: Whether any metric is anomalous
                - anomaly_score: Max normalized violation magnitude
                - violated_metrics: List of metric names that violated bounds
                - per_metric_scores: Per-metric anomaly scores
                - severity: "low", "medium", "high", or "critical"
        """
        lower = forecast["lower"]
        upper = forecast["upper"]
        interval_width = upper - lower

        # Compute per-metric violations
        per_metric_scores = np.zeros(len(self.metric_names))
        violated_metrics = []

        for i, name in enumerate(self.metric_names):
            width = max(interval_width[i], 1e-10)  # avoid division by zero

            if actuals[i] > upper[i]:
                # Above upper bound
                per_metric_scores[i] = (actuals[i] - upper[i]) / width
                violated_metrics.append(name)
            elif actuals[i] < lower[i]:
                # Below lower bound
                per_metric_scores[i] = (lower[i] - actuals[i]) / width
                violated_metrics.append(name)

        anomaly_score = float(per_metric_scores.max())
        is_anomaly = len(violated_metrics) > 0

        # Determine severity
        severity = "low"
        for level, threshold in sorted(
            self.severity_thresholds.items(), key=lambda x: x[1], reverse=True
        ):
            if anomaly_score >= threshold:
                severity = level
                break

        return {
            "is_anomaly": is_anomaly,
            "anomaly_score": anomaly_score,
            "violated_metrics": violated_metrics,
            "per_metric_scores": per_metric_scores.tolist(),
            "severity": severity,
        }

    def detect_batch(
        self,
        actuals_batch: np.ndarray,
        forecast_batch: dict[str, np.ndarray],
    ) -> list[dict]:
        """Detect anomalies for a batch of samples.

        Args:
            actuals_batch: (n_samples, n_metrics) actual values.
            forecast_batch: Dict with "median", "lower", "upper" of
                shape (n_samples, n_metrics).

        Returns:
            List of detection results, one per sample.
        """
        results = []
        for i in range(actuals_batch.shape[0]):
            sample_forecast = {
                k: v[i] for k, v in forecast_batch.items()
            }
            results.append(self.detect(actuals_batch[i], sample_forecast))
        return results
