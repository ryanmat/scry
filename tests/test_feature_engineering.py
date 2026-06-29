# Description: Unit tests for the feature engineering module.
# Description: Tests metric pivoting, type classification, and window generation.

"""Tests for scry.data.feature_engineering module."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest


class TestFeatureLists:
    """Tests for feature list constants."""

    def test_numerical_features_not_empty(self) -> None:
        """NUMERICAL_FEATURES should have features from active profile."""
        from scry.data.feature_engineering import NUMERICAL_FEATURES

        assert len(NUMERICAL_FEATURES) > 0

    def test_categorical_features_not_empty(self) -> None:
        """CATEGORICAL_FEATURES should have features from active profile."""
        from scry.data.feature_engineering import CATEGORICAL_FEATURES

        assert len(CATEGORICAL_FEATURES) > 0

    def test_all_features_combined(self) -> None:
        """ALL_FEATURES should be numerical + categorical."""
        from scry.data.feature_engineering import (
            ALL_FEATURES,
            CATEGORICAL_FEATURES,
            NUMERICAL_FEATURES,
        )

        assert len(ALL_FEATURES) == len(NUMERICAL_FEATURES) + len(CATEGORICAL_FEATURES)
        assert set(ALL_FEATURES) == set(NUMERICAL_FEATURES + CATEGORICAL_FEATURES)

    def test_numerical_features_are_continuous_metrics(self) -> None:
        """NUMERICAL_FEATURES should contain expected continuous metrics."""
        from scry.data.feature_engineering import NUMERICAL_FEATURES

        expected = ["cpuUsageNanoCores", "memoryUsageBytes", "networkRxBytes"]
        for metric in expected:
            assert metric in NUMERICAL_FEATURES

    def test_categorical_features_are_discrete_metrics(self) -> None:
        """CATEGORICAL_FEATURES should contain expected discrete metrics."""
        from scry.data.feature_engineering import CATEGORICAL_FEATURES

        expected = ["kubePodStatusReady", "podConditionPhase", "kubePodContainerStatusRunning"]
        for metric in expected:
            assert metric in CATEGORICAL_FEATURES


class TestPivotMetrics:
    """Tests for pivot_metrics function."""

    @pytest.fixture
    def long_format_df(self) -> pd.DataFrame:
        """Sample long format DataFrame."""
        return pd.DataFrame(
            {
                "resource_id": [1, 1, 1, 2, 2, 2],
                "timestamp": [
                    datetime(2024, 12, 1, 10, 0, tzinfo=timezone.utc),
                    datetime(2024, 12, 1, 10, 0, tzinfo=timezone.utc),
                    datetime(2024, 12, 1, 10, 1, tzinfo=timezone.utc),
                    datetime(2024, 12, 1, 10, 0, tzinfo=timezone.utc),
                    datetime(2024, 12, 1, 10, 0, tzinfo=timezone.utc),
                    datetime(2024, 12, 1, 10, 1, tzinfo=timezone.utc),
                ],
                "metric_name": [
                    "cpuUsageNanoCores",
                    "memoryUsageBytes",
                    "cpuUsageNanoCores",
                    "cpuUsageNanoCores",
                    "memoryUsageBytes",
                    "cpuUsageNanoCores",
                ],
                "value": [45.5, 62.3, 48.2, 12.1, 35.8, 15.3],
            }
        )

    def test_pivot_converts_long_to_wide(self, long_format_df: pd.DataFrame) -> None:
        """pivot_metrics should convert long format to wide format."""
        from scry.data.feature_engineering import pivot_metrics

        result = pivot_metrics(long_format_df)

        # Should have metric columns
        assert "cpuUsageNanoCores" in result.columns
        assert "memoryUsageBytes" in result.columns
        # Should have fewer rows (grouped by resource_id + timestamp)
        assert len(result) < len(long_format_df)

    def test_pivot_preserves_resource_id_and_timestamp(
        self, long_format_df: pd.DataFrame
    ) -> None:
        """Pivoted DataFrame should preserve resource_id and timestamp."""
        from scry.data.feature_engineering import pivot_metrics

        result = pivot_metrics(long_format_df)

        assert "resource_id" in result.columns
        assert "timestamp" in result.columns

    def test_pivot_handles_missing_metrics(self) -> None:
        """pivot_metrics should fill missing metrics with NaN."""
        from scry.data.feature_engineering import pivot_metrics

        # Only has cpuUsageNanoCores, not memoryUsageBytes for resource 2
        df = pd.DataFrame(
            {
                "resource_id": [1, 1, 2],
                "timestamp": [
                    datetime(2024, 12, 1, 10, 0, tzinfo=timezone.utc),
                    datetime(2024, 12, 1, 10, 0, tzinfo=timezone.utc),
                    datetime(2024, 12, 1, 10, 0, tzinfo=timezone.utc),
                ],
                "metric_name": ["cpuUsageNanoCores", "memoryUsageBytes", "cpuUsageNanoCores"],
                "value": [45.5, 62.3, 12.1],
            }
        )

        result = pivot_metrics(df)

        # Resource 2 should have NaN for memoryUsageBytes
        resource_2 = result[result["resource_id"] == 2]
        assert pd.isna(resource_2["memoryUsageBytes"].iloc[0])

    def test_pivot_empty_dataframe(self) -> None:
        """pivot_metrics should handle empty DataFrame."""
        from scry.data.feature_engineering import pivot_metrics

        df = pd.DataFrame(columns=["resource_id", "timestamp", "metric_name", "value"])
        result = pivot_metrics(df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


class TestFilterTargetMetrics:
    """Tests for filter_target_metrics function."""

    def test_filter_keeps_only_target_columns(self) -> None:
        """filter_target_metrics should keep only target metric columns."""
        from scry.data.feature_engineering import (
            filter_target_metrics,
        )

        df = pd.DataFrame(
            {
                "resource_id": [1],
                "timestamp": [datetime(2024, 12, 1, tzinfo=timezone.utc)],
                "cpuUsageNanoCores": [45.5],
                "memoryUsageBytes": [62.3],
                "kubePodStatusReady": [1],
                "someOtherMetric": [999],  # Should be filtered out
            }
        )

        result = filter_target_metrics(df)

        assert "someOtherMetric" not in result.columns
        assert "cpuUsageNanoCores" in result.columns
        assert "resource_id" in result.columns
        assert "timestamp" in result.columns

    def test_filter_handles_missing_target_metrics(self) -> None:
        """filter_target_metrics should handle missing target metrics gracefully."""
        from scry.data.feature_engineering import filter_target_metrics

        df = pd.DataFrame(
            {
                "resource_id": [1],
                "timestamp": [datetime(2024, 12, 1, tzinfo=timezone.utc)],
                "cpuUsageNanoCores": [45.5],
                # Missing most target metrics
            }
        )

        # Should not raise, just include what's available
        result = filter_target_metrics(df)
        assert "cpuUsageNanoCores" in result.columns


class TestSplitByType:
    """Tests for split_by_type function."""

    def test_split_separates_numerical_and_categorical(self) -> None:
        """split_by_type should separate numerical and categorical columns."""
        from scry.data.feature_engineering import split_by_type

        df = pd.DataFrame(
            {
                "resource_id": [1],
                "timestamp": [datetime(2024, 12, 1, tzinfo=timezone.utc)],
                "cpuUsageNanoCores": [45.5],
                "memoryUsageBytes": [62.3],
                "kubePodStatusReady": [1],
                "podConditionPhase": [1],
            }
        )

        df_num, df_cat = split_by_type(df)

        # Numerical should have cpu and memory
        assert "cpuUsageNanoCores" in df_num.columns
        assert "memoryUsageBytes" in df_num.columns
        assert "kubePodStatusReady" not in df_num.columns

        # Categorical should have kubePodStatusReady and podConditionPhase
        assert "kubePodStatusReady" in df_cat.columns
        assert "podConditionPhase" in df_cat.columns
        assert "cpuUsageNanoCores" not in df_cat.columns

    def test_split_preserves_resource_id_and_timestamp(self) -> None:
        """Both split DataFrames should preserve resource_id and timestamp."""
        from scry.data.feature_engineering import split_by_type

        df = pd.DataFrame(
            {
                "resource_id": [1],
                "timestamp": [datetime(2024, 12, 1, tzinfo=timezone.utc)],
                "cpuUsageNanoCores": [45.5],
                "kubePodStatusReady": [1],
            }
        )

        df_num, df_cat = split_by_type(df)

        assert "resource_id" in df_num.columns
        assert "timestamp" in df_num.columns
        assert "resource_id" in df_cat.columns
        assert "timestamp" in df_cat.columns

    def test_split_handles_missing_features(self) -> None:
        """split_by_type should handle missing features gracefully."""
        from scry.data.feature_engineering import split_by_type

        df = pd.DataFrame(
            {
                "resource_id": [1],
                "timestamp": [datetime(2024, 12, 1, tzinfo=timezone.utc)],
                "cpuUsageNanoCores": [45.5],
                # No categorical features
            }
        )

        df_num, df_cat = split_by_type(df)

        assert "cpuUsageNanoCores" in df_num.columns
        # Categorical should still have resource_id and timestamp
        assert "resource_id" in df_cat.columns


class TestSlidingWindows:
    """Tests for sliding window generation."""

    @pytest.fixture
    def time_series_df(self) -> pd.DataFrame:
        """Sample time series DataFrame for one resource."""
        timestamps = pd.date_range(
            start="2024-12-01 10:00:00", periods=40, freq="1min", tz=timezone.utc
        )
        return pd.DataFrame(
            {
                "resource_id": [1] * 40,
                "timestamp": timestamps,
                "cpuUsageNanoCores": np.random.uniform(10, 90, 40),
                "memoryUsageBytes": np.random.uniform(20, 80, 40),
            }
        )

    def test_create_sliding_windows_shape(self, time_series_df: pd.DataFrame) -> None:
        """create_sliding_windows should return correct shape."""
        from scry.data.feature_engineering import create_sliding_windows

        windows, labels = create_sliding_windows(
            time_series_df, window_size=30, step=1
        )

        # With 40 rows, window_size=30, step=1: expect 11 windows
        assert windows.shape[0] == 11
        assert windows.shape[1] == 30  # window_size
        assert windows.shape[2] == 2  # num features (cpu and memory)

    def test_create_sliding_windows_labels(self, time_series_df: pd.DataFrame) -> None:
        """create_sliding_windows should return matching labels."""
        from scry.data.feature_engineering import create_sliding_windows

        windows, labels = create_sliding_windows(
            time_series_df, window_size=30, step=1
        )

        # Labels should match number of windows
        assert len(labels) == windows.shape[0]
        # Each label should have resource_id and end timestamp
        assert labels.shape[1] == 2

    def test_create_dual_windows_aligned(self) -> None:
        """create_dual_windows should return aligned numerical and categorical windows."""
        from scry.data.feature_engineering import create_dual_windows

        timestamps = pd.date_range(
            start="2024-12-01 10:00:00", periods=40, freq="1min", tz=timezone.utc
        )
        df_num = pd.DataFrame(
            {
                "resource_id": [1] * 40,
                "timestamp": timestamps,
                "cpuUsageNanoCores": np.random.uniform(10, 90, 40),
            }
        )
        df_cat = pd.DataFrame(
            {
                "resource_id": [1] * 40,
                "timestamp": timestamps,
                "kubePodStatusReady": np.ones(40),
            }
        )

        num_windows, cat_windows, labels = create_dual_windows(
            df_num, df_cat, window_size=30, step=1
        )

        # Both should have same number of samples
        assert num_windows.shape[0] == cat_windows.shape[0]
        assert num_windows.shape[0] == len(labels)


    def test_step_10_produces_expected_count(self) -> None:
        """With step=10, window count should be (n - window_size) // step + 1."""
        from scry.data.feature_engineering import create_sliding_windows

        timestamps = pd.date_range(
            start="2024-12-01 10:00:00", periods=100, freq="1min", tz=timezone.utc
        )
        df = pd.DataFrame(
            {
                "resource_id": [1] * 100,
                "timestamp": timestamps,
                "cpuUsageNanoCores": np.random.uniform(10, 90, 100),
            }
        )

        windows, labels = create_sliding_windows(df, window_size=30, step=10)

        expected_count = (100 - 30) // 10 + 1  # 8 windows
        assert windows.shape[0] == expected_count
        assert labels.shape[0] == expected_count

    def test_dual_windows_pass_step_through(self) -> None:
        """create_dual_windows should pass step parameter to both branches."""
        from scry.data.feature_engineering import create_dual_windows

        timestamps = pd.date_range(
            start="2024-12-01 10:00:00", periods=100, freq="1min", tz=timezone.utc
        )
        df_num = pd.DataFrame(
            {
                "resource_id": [1] * 100,
                "timestamp": timestamps,
                "cpuUsageNanoCores": np.random.uniform(10, 90, 100),
            }
        )
        df_cat = pd.DataFrame(
            {
                "resource_id": [1] * 100,
                "timestamp": timestamps,
                "kubePodStatusReady": np.ones(100),
            }
        )

        num_windows, cat_windows, labels = create_dual_windows(
            df_num, df_cat, window_size=30, step=10
        )

        expected_count = (100 - 30) // 10 + 1  # 8
        assert num_windows.shape[0] == expected_count
        assert cat_windows.shape[0] == expected_count
        assert labels.shape[0] == expected_count


class TestNormalization:
    """Tests for normalization functions."""

    def test_normalize_numerical_produces_standard_scale(self) -> None:
        """normalize_numerical should produce approximately mean=0, std=1."""
        from scry.data.feature_engineering import normalize_numerical

        # Create windows with known distribution
        windows = np.random.normal(100, 20, (50, 30, 3))

        normalized, params = normalize_numerical(windows)

        # Check approximate standardization per feature
        for i in range(3):
            feature_data = normalized[:, :, i].flatten()
            assert np.abs(feature_data.mean()) < 0.5  # Close to 0
            assert np.abs(feature_data.std() - 1.0) < 0.5  # Close to 1

    def test_normalize_numerical_returns_params(self) -> None:
        """normalize_numerical should return mean and std params."""
        from scry.data.feature_engineering import normalize_numerical

        windows = np.random.normal(100, 20, (50, 30, 3))

        normalized, params = normalize_numerical(windows)

        assert "mean" in params
        assert "std" in params
        assert len(params["mean"]) == 3
        assert len(params["std"]) == 3

    def test_normalize_numerical_handles_nan(self) -> None:
        """normalize_numerical should handle NaN values."""
        from scry.data.feature_engineering import normalize_numerical

        windows = np.random.normal(100, 20, (50, 30, 3))
        windows[0, 5, 0] = np.nan  # Introduce NaN

        normalized, params = normalize_numerical(windows)

        # Should not have NaN after forward/backward fill
        assert not np.isnan(normalized).any()


class TestCategoricalEncoding:
    """Tests for categorical encoding."""

    def test_encode_categorical_values_in_range(self) -> None:
        """encode_categorical should produce values in [0, 1]."""
        from scry.data.feature_engineering import encode_categorical

        # Categorical data with various values
        windows = np.array([[[0, 1, 2], [1, 0, 3]], [[1, 1, 1], [0, 0, 0]]])

        encoded, _ = encode_categorical(windows)

        assert encoded.min() >= 0
        assert encoded.max() <= 1

    def test_encode_categorical_handles_nan(self) -> None:
        """encode_categorical should handle NaN values."""
        from scry.data.feature_engineering import encode_categorical

        windows = np.array([[[1, np.nan], [0, 1]]], dtype=float)

        encoded, _ = encode_categorical(windows)

        assert not np.isnan(encoded).any()


class TestXDECDataset:
    """Tests for XDECDataset PyTorch dataset."""

    def test_dataset_length(self) -> None:
        """XDECDataset should return correct length."""
        from scry.data.feature_engineering import XDECDataset

        num_windows = np.random.rand(100, 30, 9)
        cat_windows = np.random.rand(100, 30, 8)

        dataset = XDECDataset(num_windows, cat_windows)

        assert len(dataset) == 100

    def test_dataset_getitem_returns_tuple(self) -> None:
        """XDECDataset __getitem__ should return (num_tensor, cat_tensor)."""
        import torch

        from scry.data.feature_engineering import XDECDataset

        num_windows = np.random.rand(100, 30, 9)
        cat_windows = np.random.rand(100, 30, 8)

        dataset = XDECDataset(num_windows, cat_windows)
        num_tensor, cat_tensor = dataset[0]

        assert isinstance(num_tensor, torch.Tensor)
        assert isinstance(cat_tensor, torch.Tensor)
        assert num_tensor.shape == (30, 9)
        assert cat_tensor.shape == (30, 8)

    def test_dataset_with_labels(self) -> None:
        """XDECDataset should include labels when provided."""
        from scry.data.feature_engineering import XDECDataset

        num_windows = np.random.rand(100, 30, 9)
        cat_windows = np.random.rand(100, 30, 8)
        labels = np.array([[i, i * 1000] for i in range(100)])

        dataset = XDECDataset(num_windows, cat_windows, labels=labels)
        num_tensor, cat_tensor, label = dataset[0]

        assert len(label) == 2
