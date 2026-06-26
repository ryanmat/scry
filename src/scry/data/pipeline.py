# Description: End-to-end feature pipeline for the X-DEC model.
# Description: Combines data extraction, transformation, and persistence.

"""Feature pipeline for the X-DEC model."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from scry.data.feature_engineering import (
    XDECDataset,
    create_dual_windows,
    encode_categorical,
    filter_target_metrics,
    normalize_numerical,
    pivot_metrics,
    split_by_type,
)
from scry.data.fetcher import DataFetcher
from scry.utils.config import ScryConfig

if TYPE_CHECKING:
    from scry.data.sources.http_ingest import HttpIngestClient


class XDECFeaturePipeline:
    """End-to-end feature pipeline for the X-DEC model.

    Builds windowed training tensors from any DataFetcher. Use the
    from_http_client() factory for the HttpIngest adapter, or construct a
    pipeline directly from a DataFetcher over object storage.
    """

    def __init__(self, fetcher: DataFetcher, config: ScryConfig) -> None:
        """Initialize the pipeline.

        Args:
            fetcher: DataFetcher instance (HTTP backed).
            config: Scry configuration.
        """
        self._config = config
        self._fetcher = fetcher

    @classmethod
    def from_http_client(
        cls, client: HttpIngestClient, config: ScryConfig
    ) -> XDECFeaturePipeline:
        """Create pipeline with HTTP data source (HttpIngest ML API).

        Args:
            client: Connected HttpIngestClient instance.
            config: Scry configuration.
        """
        return cls(DataFetcher.from_http_client(client), config)

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
            profile: Feature profile for filtering (HTTP mode only).

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
            return {
                "num_windows": np.zeros((0, self._config.sequence_length, 9)),
                "cat_windows": np.zeros((0, self._config.sequence_length, 8)),
                "labels": np.zeros((0, 2), dtype=object),
                "num_norm_params": {"mean": np.zeros(9), "std": np.ones(9)},
            }

        # Step 1: Pivot from long to wide format
        pivoted = pivot_metrics(df)

        # Step 2: Filter to target metrics
        filtered = filter_target_metrics(pivoted)

        # Step 3: Split into numerical and categorical
        df_num, df_cat = split_by_type(filtered)

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
                "num_norm_params": {"mean": np.zeros(9), "std": np.ones(9)},
            }

        # Step 5: Normalize numerical features
        num_normalized, num_params = normalize_numerical(num_windows)

        # Step 6: Encode categorical features
        cat_encoded = encode_categorical(cat_windows)

        return {
            "num_windows": num_normalized,
            "cat_windows": cat_encoded,
            "labels": labels,
            "num_norm_params": num_params,
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
