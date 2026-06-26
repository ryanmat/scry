# Description: Feature and concept drift detection for deployed models.
# Description: Uses PSI for feature drift and ADWIN for prediction drift.

"""Drift detection for Scry models.

Provides two detection mechanisms:
    - PSI (Population Stability Index): Detects changes in input feature
      distributions between reference and current data.
    - ADWIN (Adaptive Windowing): Detects change points in prediction
      error streams indicating concept drift.
"""

from datetime import datetime, timezone

import numpy as np


class DriftDetector:
    """Combined feature and concept drift detector.

    Args:
        n_features: Number of input features.
        feature_names: Names for each feature.
        psi_threshold: PSI value above which drift is significant (default: 0.2).
        adwin_delta: ADWIN confidence parameter (default: 0.01).
        n_bins: Number of bins for PSI histogram (default: 10).
    """

    def __init__(
        self,
        n_features: int,
        feature_names: list[str] | None = None,
        psi_threshold: float = 0.2,
        adwin_delta: float = 0.01,
        n_bins: int = 10,
    ) -> None:
        self.n_features = n_features
        self.feature_names = feature_names or [f"feature_{i}" for i in range(n_features)]
        self.psi_threshold = psi_threshold
        self.adwin_delta = adwin_delta
        self.n_bins = n_bins

    def check_feature_drift(
        self,
        reference_data: np.ndarray,
        current_data: np.ndarray,
    ) -> dict:
        """Check for feature distribution drift using PSI.

        PSI = sum((current_pct - reference_pct) * ln(current_pct / reference_pct))
        PSI < 0.1: no significant change
        0.1 <= PSI < 0.2: moderate change
        PSI >= 0.2: significant drift

        Args:
            reference_data: Reference data (n_samples, n_features).
            current_data: Current data (n_samples, n_features).

        Returns:
            Dict with has_drift, psi_per_feature, max_psi.
        """
        psi_per_feature = {}

        for i, name in enumerate(self.feature_names):
            psi = self._compute_psi(reference_data[:, i], current_data[:, i])
            psi_per_feature[name] = float(psi)

        max_psi = max(psi_per_feature.values())
        has_drift = bool(max_psi >= self.psi_threshold)

        return {
            "has_drift": has_drift,
            "psi_per_feature": psi_per_feature,
            "max_psi": float(max_psi),
            "threshold": float(self.psi_threshold),
        }

    def _compute_psi(
        self, reference: np.ndarray, current: np.ndarray
    ) -> float:
        """Compute PSI between two 1D distributions.

        Args:
            reference: Reference distribution values.
            current: Current distribution values.

        Returns:
            PSI value (0 = identical, higher = more drift).
        """
        eps = 1e-10

        # Create bins from combined data range
        combined = np.concatenate([reference, current])
        bins = np.linspace(combined.min() - eps, combined.max() + eps, self.n_bins + 1)

        # Compute bin percentages
        ref_hist, _ = np.histogram(reference, bins=bins)
        cur_hist, _ = np.histogram(current, bins=bins)

        ref_pct = ref_hist / len(reference) + eps
        cur_pct = cur_hist / len(current) + eps

        # PSI formula
        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        return float(psi)

    def check_prediction_drift(
        self, error_stream: np.ndarray, min_window: int = 30
    ) -> dict:
        """Check for concept drift in prediction errors using ADWIN.

        Uses a simplified ADWIN approach: sliding window comparison of
        error means with statistical significance testing.

        Args:
            error_stream: Array of prediction errors over time.
            min_window: Minimum window size for comparison.

        Returns:
            Dict with has_drift, change_point_index, p_value.
        """
        n = len(error_stream)
        if n < min_window * 2:
            return {
                "has_drift": False,
                "change_point_index": None,
                "mean_before": float(error_stream.mean()),
                "mean_after": float(error_stream.mean()),
            }

        # Scan for change point using max difference in window means
        best_score = 0.0
        best_idx = None

        for split in range(min_window, n - min_window):
            window_before = error_stream[max(0, split - min_window):split]
            window_after = error_stream[split:min(n, split + min_window)]

            mean_before = window_before.mean()
            mean_after = window_after.mean()
            pooled_var = (window_before.var() + window_after.var()) / 2 + 1e-10

            # Welch's t-statistic approximation
            score = abs(mean_after - mean_before) / np.sqrt(
                pooled_var * (1 / len(window_before) + 1 / len(window_after))
            )

            if score > best_score:
                best_score = score
                best_idx = split

        # Use t-statistic threshold based on delta
        # For delta=0.01, threshold ~ 2.58 (99% confidence)
        threshold = -np.log(self.adwin_delta) * 1.0
        has_drift = bool(best_score > threshold)

        mean_before = error_stream[:best_idx].mean() if best_idx else error_stream.mean()
        mean_after = error_stream[best_idx:].mean() if best_idx else error_stream.mean()

        return {
            "has_drift": has_drift,
            "change_point_index": int(best_idx) if has_drift else None,
            "score": float(best_score),
            "threshold": float(threshold),
            "mean_before": float(mean_before),
            "mean_after": float(mean_after),
        }

    def get_drift_status(
        self,
        reference_data: np.ndarray,
        current_data: np.ndarray,
        error_stream: np.ndarray,
    ) -> dict:
        """Get combined drift status.

        Args:
            reference_data: Reference feature data.
            current_data: Current feature data.
            error_stream: Prediction error stream.

        Returns:
            Combined drift status dict.
        """
        feature_result = self.check_feature_drift(reference_data, current_data)
        prediction_result = self.check_prediction_drift(error_stream)

        return {
            "feature_drift": feature_result,
            "prediction_drift": prediction_result,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
