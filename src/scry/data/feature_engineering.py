# Description: Feature engineering for the X-DEC model.
# Description: Handles metric pivoting, type classification, windowing, and normalization.

"""Feature engineering for the X-DEC model."""

import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from scry.config import FeatureConfig, get_profile

logger = logging.getLogger(__name__)

# Module-level feature lists for backward compatibility
# These default to the kubernetes profile but can be overridden
_active_config: FeatureConfig | None = None


def get_active_config() -> FeatureConfig:
    """Get the currently active feature configuration.

    Returns:
        Active FeatureConfig, defaults to kubernetes profile.
    """
    global _active_config
    if _active_config is None:
        _active_config = get_profile("kubernetes")
    return _active_config


def set_active_profile(profile_name: str) -> FeatureConfig:
    """Set the active feature profile.

    Args:
        profile_name: Name of the profile to activate.

    Returns:
        The newly activated FeatureConfig.
    """
    global _active_config
    _active_config = get_profile(profile_name)
    logger.info(
        "Activated profile '%s': %d numerical, %d categorical features",
        profile_name,
        _active_config.num_numerical,
        _active_config.num_categorical,
    )
    return _active_config


def get_numerical_features() -> list[str]:
    """Get list of numerical features for active profile.

    Returns:
        List of numerical feature names.
    """
    return get_active_config().numerical_features


def get_categorical_features() -> list[str]:
    """Get list of categorical features for active profile.

    Returns:
        List of categorical feature names.
    """
    return get_active_config().categorical_features


def get_all_features() -> list[str]:
    """Get combined list of all features for active profile.

    Returns:
        List of all feature names.
    """
    return get_active_config().all_features


# Backward compatibility: module-level constants that delegate to active config
# These are properties that look up the active config each time
class _FeatureList:
    """Lazy feature list that delegates to active config."""

    def __init__(self, feature_type: str) -> None:
        self._type = feature_type

    def _get_list(self) -> list[str]:
        if self._type == "numerical":
            return get_numerical_features()
        elif self._type == "categorical":
            return get_categorical_features()
        else:
            return get_all_features()

    def __iter__(self):
        return iter(self._get_list())

    def __len__(self):
        return len(self._get_list())

    def __contains__(self, item):
        return item in self._get_list()

    def __add__(self, other):
        if isinstance(other, _FeatureList):
            return self._get_list() + other._get_list()
        return self._get_list() + list(other)

    def __radd__(self, other):
        if isinstance(other, _FeatureList):
            return other._get_list() + self._get_list()
        return list(other) + self._get_list()


# Module-level constants for backward compatibility
NUMERICAL_FEATURES: list[str] = _FeatureList("numerical")  # type: ignore
CATEGORICAL_FEATURES: list[str] = _FeatureList("categorical")  # type: ignore
ALL_FEATURES: list[str] = _FeatureList("all")  # type: ignore


def pivot_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Convert long format metrics to wide format.

    Args:
        df: Long format DataFrame with columns:
            resource_id, timestamp, metric_name, value

    Returns:
        Wide format DataFrame with columns:
            resource_id, timestamp, metric1, metric2, ...
    """
    if df.empty:
        return pd.DataFrame(columns=["resource_id", "timestamp"])

    # Pivot the data
    pivoted = df.pivot_table(
        index=["resource_id", "timestamp"],
        columns="metric_name",
        values="value",
        aggfunc="first",  # Take first value if duplicates
    ).reset_index()

    # Flatten column names
    pivoted.columns.name = None

    return pivoted


def filter_target_metrics(
    df: pd.DataFrame,
    config: FeatureConfig | None = None,
) -> pd.DataFrame:
    """Filter DataFrame to keep only target metric columns.

    Args:
        df: Wide format DataFrame with metric columns.
        config: Optional feature config. Uses active config if not provided.

    Returns:
        DataFrame with only resource_id, timestamp, and target metrics.
    """
    if config is None:
        config = get_active_config()

    # Always keep these columns
    keep_columns = ["resource_id", "timestamp"]

    # Add any target metrics that exist in the DataFrame
    for col in config.all_features:
        if col in df.columns:
            keep_columns.append(col)

    return df[keep_columns]


def split_by_type(
    df: pd.DataFrame,
    config: FeatureConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split DataFrame into numerical and categorical feature DataFrames.

    Args:
        df: Wide format DataFrame with metric columns.
        config: Optional feature config. Uses active config if not provided.

    Returns:
        Tuple of (df_numerical, df_categorical).
        Both include resource_id and timestamp columns.
    """
    if config is None:
        config = get_active_config()

    # Base columns to keep in both
    base_columns = ["resource_id", "timestamp"]

    # Numerical columns present in DataFrame
    num_cols = base_columns.copy()
    for col in config.numerical_features:
        if col in df.columns:
            num_cols.append(col)

    # Categorical columns present in DataFrame
    cat_cols = base_columns.copy()
    for col in config.categorical_features:
        if col in df.columns:
            cat_cols.append(col)

    df_num = df[num_cols].copy()
    df_cat = df[cat_cols].copy()

    return df_num, df_cat


def create_sliding_windows(
    df: pd.DataFrame,
    window_size: int = 30,
    step: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Create sliding windows from time series DataFrame.

    Args:
        df: DataFrame with resource_id, timestamp, and feature columns.
        window_size: Number of time steps per window.
        step: Step size between windows.

    Returns:
        Tuple of:
            - windows: np.ndarray of shape (num_windows, window_size, num_features)
            - labels: np.ndarray of shape (num_windows, 2) with (resource_id, end_timestamp)
    """
    # Get feature columns (exclude resource_id and timestamp)
    feature_cols = [c for c in df.columns if c not in ["resource_id", "timestamp"]]

    windows_list = []
    labels_list = []

    # Process each resource separately
    for resource_id, group in df.groupby("resource_id"):
        # Sort by timestamp
        group = group.sort_values("timestamp")

        # Extract feature values
        values = group[feature_cols].values
        timestamps = group["timestamp"].values

        # Create windows
        num_windows = (len(values) - window_size) // step + 1

        for i in range(num_windows):
            start_idx = i * step
            end_idx = start_idx + window_size

            window = values[start_idx:end_idx]
            end_timestamp = timestamps[end_idx - 1]

            windows_list.append(window)
            labels_list.append([resource_id, end_timestamp])

    if not windows_list:
        # Return empty arrays with correct shape
        return (
            np.zeros((0, window_size, len(feature_cols))),
            np.zeros((0, 2), dtype=object),
        )

    windows = np.array(windows_list)
    labels = np.array(labels_list, dtype=object)

    return windows, labels


def create_dual_windows(
    df_num: pd.DataFrame,
    df_cat: pd.DataFrame,
    window_size: int = 30,
    step: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create aligned sliding windows for numerical and categorical data.

    Args:
        df_num: Numerical features DataFrame.
        df_cat: Categorical features DataFrame.
        window_size: Number of time steps per window.
        step: Step size between windows.

    Returns:
        Tuple of:
            - num_windows: (n, window_size, n_numerical)
            - cat_windows: (n, window_size, n_categorical)
            - labels: (n, 2) with (resource_id, end_timestamp)
    """
    # Create windows for each branch
    num_windows, labels = create_sliding_windows(df_num, window_size, step)
    cat_windows, _ = create_sliding_windows(df_cat, window_size, step)

    return num_windows, cat_windows, labels


def normalize_numerical(
    windows: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Z-score normalize numerical windows.

    Args:
        windows: Array of shape (n, window_size, n_features).

    Returns:
        Tuple of:
            - normalized: Normalized windows.
            - params: Dict with 'mean' and 'std' arrays per feature.
    """
    # Handle NaN by forward-fill then backward-fill
    windows = windows.copy()

    # Fill NaN for each sample
    for i in range(windows.shape[0]):
        for j in range(windows.shape[2]):
            series = pd.Series(windows[i, :, j])
            series = series.ffill().bfill()
            # If still NaN (all NaN), fill with 0
            series = series.fillna(0)
            windows[i, :, j] = series.values

    # Compute mean and std per feature (across all samples and time steps)
    # Reshape to (n * window_size, n_features)
    flat = windows.reshape(-1, windows.shape[2])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)

    # Prevent division by zero
    std = np.where(std == 0, 1.0, std)

    # Normalize
    normalized = (windows - mean) / std

    params = {
        "mean": mean,
        "std": std,
    }

    return normalized, params


def encode_categorical(
    windows: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Encode categorical windows to [0, 1] range.

    Binary features (0/1) are kept as-is.
    Multi-class features (like podConditionPhase 0-4) are normalized to [0, 1].

    Args:
        windows: Array of shape (n, window_size, n_features).

    Returns:
        Tuple of:
            - encoded: Windows with values in [0, 1].
            - params: Dict with 'min' and 'max' arrays per feature, so the same
              encoding can be reproduced at inference time.
    """
    windows = windows.copy()

    # Handle NaN by filling with 0 (most common default state)
    windows = np.nan_to_num(windows, nan=0.0)

    n_features = windows.shape[2]
    mins = np.zeros(n_features)
    maxs = np.zeros(n_features)

    # Normalize each feature to [0, 1]
    for i in range(n_features):
        feature_data = windows[:, :, i]
        min_val = feature_data.min()
        max_val = feature_data.max()
        mins[i] = min_val
        maxs[i] = max_val

        if max_val > min_val:
            windows[:, :, i] = (feature_data - min_val) / (max_val - min_val)
        else:
            # All same value, set to 0 or 1 depending on value
            windows[:, :, i] = 1.0 if max_val > 0 else 0.0

    return windows, {"min": mins, "max": maxs}


class XDECDataset(Dataset):
    """PyTorch Dataset for X-DEC model with dual branches.

    Returns (numerical_tensor, categorical_tensor) or
    (numerical_tensor, categorical_tensor, label) if labels provided.
    """

    def __init__(
        self,
        num_windows: np.ndarray,
        cat_windows: np.ndarray,
        labels: np.ndarray | None = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            num_windows: Numerical windows (n, seq_len, n_numerical).
            cat_windows: Categorical windows (n, seq_len, n_categorical).
            labels: Optional labels (n, 2) with (resource_id, timestamp).
        """
        self.num_windows = torch.tensor(num_windows, dtype=torch.float32)
        self.cat_windows = torch.tensor(cat_windows, dtype=torch.float32)
        self.labels = labels

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.num_windows)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """Get a sample.

        Args:
            idx: Sample index.

        Returns:
            (num_tensor, cat_tensor) or (num_tensor, cat_tensor, label).
        """
        if self.labels is not None:
            return self.num_windows[idx], self.cat_windows[idx], self.labels[idx]
        return self.num_windows[idx], self.cat_windows[idx]
