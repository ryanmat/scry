# Description: End-to-end feature pipeline for the X-DEC model.
# Description: Combines data extraction, transformation, and persistence.

"""Feature pipeline for the X-DEC model."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from scry.data.feature_engineering import (
    XDECDataset,
    create_dual_windows,
    encode_categorical,
    filter_target_metrics,
    get_active_config,
    normalize_numerical,
    pivot_metrics,
    split_by_type,
)
from scry.data.fetcher import DataFetcher
from scry.utils.config import ScryConfig


class XDECFeaturePipeline:
    """End-to-end feature pipeline for the X-DEC model.

    Builds windowed training tensors from a DataFetcher over object storage.
    """

    def __init__(self, fetcher: DataFetcher, config: ScryConfig) -> None:
        """Initialize the pipeline.

        Args:
            fetcher: DataFetcher instance.
            config: Scry configuration.
        """
        self._config = config
        self._fetcher = fetcher

    async def extract(
        self,
        start_time: datetime,
        end_time: datetime,
        profile: str | None = None,
    ) -> pd.DataFrame:
        """Extract raw metrics from data source.

        Args:
            start_time: Start of time range.
            end_time: End of time range.
            profile: Feature profile for filtering.

        Returns:
            DataFrame with raw metrics in long format.
        """
        return await self._fetcher.get_metrics_dataframe(
            start_time, end_time, profile=profile
        )

    def transform(self, df: pd.DataFrame) -> dict[str, Any]:
        """Transform raw metrics to X-DEC training data.

        Args:
            df: Raw metrics DataFrame in long format.

        Returns:
            Dictionary with:
                - num_windows: Normalized numerical windows (n, seq, n_num)
                - cat_windows: Encoded categorical windows (n, seq, n_cat)
                - labels: Window labels (n, 2) with (resource_id, timestamp)
                - num_norm_params: Normalization parameters {mean, std}
        """
        if df.empty:
            cfg = get_active_config()
            num_names = list(cfg.numerical_features)
            cat_names = list(cfg.categorical_features)
            n_num, n_cat = len(num_names), len(cat_names)
            return {
                "num_windows": np.zeros((0, self._config.sequence_length, n_num)),
                "cat_windows": np.zeros((0, self._config.sequence_length, n_cat)),
                "labels": np.zeros((0, 2), dtype=object),
                "num_norm_params": {"mean": np.zeros(n_num), "std": np.ones(n_num)},
                "cat_norm_params": {"min": np.zeros(n_cat), "max": np.ones(n_cat)},
                "feature_names": {"numerical": num_names, "categorical": cat_names},
                "profile": cfg.profile_name,
            }

        # Step 1: Pivot from long to wide format
        pivoted = pivot_metrics(df)

        # Step 2: Filter to target metrics
        filtered = filter_target_metrics(pivoted)

        # Step 3: Split into numerical and categorical
        df_num, df_cat = split_by_type(filtered)

        # Capture the ordered, present feature names (these are the exact window
        # columns) and the active profile, so the model can align by name later.
        base_cols = ("resource_id", "timestamp")
        num_names = [c for c in df_num.columns if c not in base_cols]
        cat_names = [c for c in df_cat.columns if c not in base_cols]
        profile = get_active_config().profile_name

        # Step 4: Create dual sliding windows
        num_windows, cat_windows, labels = create_dual_windows(
            df_num,
            df_cat,
            window_size=self._config.sequence_length,
            step=getattr(self._config, "window_step", 10),
        )

        # Handle empty windows
        if num_windows.shape[0] == 0:
            return {
                "num_windows": num_windows,
                "cat_windows": cat_windows,
                "labels": labels,
                "num_norm_params": {
                    "mean": np.zeros(len(num_names)),
                    "std": np.ones(len(num_names)),
                },
                "cat_norm_params": {
                    "min": np.zeros(len(cat_names)),
                    "max": np.ones(len(cat_names)),
                },
                "feature_names": {"numerical": num_names, "categorical": cat_names},
                "profile": profile,
            }

        # Step 5: Normalize numerical features
        num_normalized, num_params = normalize_numerical(num_windows)

        # Step 6: Encode categorical features
        cat_encoded, cat_params = encode_categorical(cat_windows)

        return {
            "num_windows": num_normalized,
            "cat_windows": cat_encoded,
            "labels": labels,
            "num_norm_params": num_params,
            "cat_norm_params": cat_params,
            "feature_names": {"numerical": num_names, "categorical": cat_names},
            "profile": profile,
        }

    async def run(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> dict[str, Any]:
        """Run full extraction and transformation pipeline.

        Args:
            start_time: Start of time range.
            end_time: End of time range.

        Returns:
            Transformed data dictionary.
        """
        df = await self.extract(start_time, end_time)
        return self.transform(df)

    def save_training_data(self, data: dict[str, Any], path: str) -> None:
        """Save training data to .npz file.

        Args:
            data: Dictionary from transform().
            path: Output file path.
        """
        np.savez(
            path,
            num_windows=data["num_windows"],
            cat_windows=data["cat_windows"],
            labels=data["labels"],
            num_norm_mean=data["num_norm_params"]["mean"],
            num_norm_std=data["num_norm_params"]["std"],
            cat_norm_min=data["cat_norm_params"]["min"],
            cat_norm_max=data["cat_norm_params"]["max"],
            num_feature_names=np.array(data["feature_names"]["numerical"], dtype=object),
            cat_feature_names=np.array(data["feature_names"]["categorical"], dtype=object),
            profile=data["profile"],
        )

    def load_training_data(self, path: str) -> dict[str, Any]:
        """Load training data from .npz file.

        Args:
            path: Input file path.

        Returns:
            Dictionary with same structure as transform().
        """
        loaded = np.load(path, allow_pickle=True)

        return {
            "num_windows": loaded["num_windows"],
            "cat_windows": loaded["cat_windows"],
            "labels": loaded["labels"],
            "num_norm_params": {
                "mean": loaded["num_norm_mean"],
                "std": loaded["num_norm_std"],
            },
            "cat_norm_params": {
                "min": loaded["cat_norm_min"],
                "max": loaded["cat_norm_max"],
            },
            "feature_names": {
                "numerical": [str(x) for x in loaded["num_feature_names"]],
                "categorical": [str(x) for x in loaded["cat_feature_names"]],
            },
            "profile": str(loaded["profile"]),
        }

    def create_dataset(self, data: dict[str, Any]) -> XDECDataset:
        """Create PyTorch dataset from transformed data.

        Args:
            data: Dictionary from transform() or load_training_data().

        Returns:
            XDECDataset ready for DataLoader.
        """
        return XDECDataset(
            num_windows=data["num_windows"],
            cat_windows=data["cat_windows"],
            labels=data["labels"],
        )
