# Description: Unit tests for the feature pipeline module.
# Description: Tests end-to-end feature extraction, transformation, and persistence.

"""Tests for scry.data.pipeline module."""

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest


class TestXDECFeaturePipeline:
    """Tests for XDECFeaturePipeline class."""

    @pytest.fixture
    def mock_fetcher(self) -> MagicMock:
        """Create a mock DataFetcher."""
        fetcher = MagicMock()
        fetcher.get_metrics_dataframe = AsyncMock()
        return fetcher

    @pytest.fixture
    def mock_config(self) -> MagicMock:
        """Create a mock configuration."""
        config = MagicMock()
        config.sequence_length = 30
        config.window_step = 10
        return config

    @pytest.fixture
    def sample_raw_data(self) -> pd.DataFrame:
        """Sample raw metric data in long format."""
        # Create 60 minutes of data for 2 resources
        records = []
        for resource_id in [1, 2]:
            for minute in range(60):
                timestamp = datetime(2024, 12, 1, 10, minute, tzinfo=timezone.utc)
                # Numerical metrics
                records.append(
                    {
                        "resource_id": resource_id,
                        "host_name": f"pod-{resource_id}",
                        "metric_name": "cpuUsageNanoCores",
                        "timestamp": timestamp,
                        "value": 30 + np.random.uniform(-5, 5) + minute * 0.5,
                        "datasource_instance": "container",
                        "datasource_name": "Kubernetes_KSM_Pods",
                    }
                )
                records.append(
                    {
                        "resource_id": resource_id,
                        "host_name": f"pod-{resource_id}",
                        "metric_name": "memoryUsageBytes",
                        "timestamp": timestamp,
                        "value": 50 + np.random.uniform(-10, 10),
                        "datasource_instance": "container",
                        "datasource_name": "Kubernetes_KSM_Pods",
                    }
                )
                # Categorical metric
                records.append(
                    {
                        "resource_id": resource_id,
                        "host_name": f"pod-{resource_id}",
                        "metric_name": "kubePodStatusReady",
                        "timestamp": timestamp,
                        "value": 1,
                        "datasource_instance": "container",
                        "datasource_name": "Kubernetes_KSM_Pods",
                    }
                )
        return pd.DataFrame(records)

    def test_transform_returns_expected_keys(
        self, mock_fetcher: MagicMock, mock_config: MagicMock, sample_raw_data: pd.DataFrame
    ) -> None:
        """transform should return dict with expected keys."""
        from scry.data.pipeline import XDECFeaturePipeline

        pipeline = XDECFeaturePipeline(mock_fetcher, mock_config)
        result = pipeline.transform(sample_raw_data)

        assert "num_windows" in result
        assert "cat_windows" in result
        assert "labels" in result
        assert "num_norm_params" in result

    def test_transform_produces_correct_shapes(
        self, mock_fetcher: MagicMock, mock_config: MagicMock, sample_raw_data: pd.DataFrame
    ) -> None:
        """transform should produce correctly shaped arrays."""
        from scry.data.pipeline import XDECFeaturePipeline

        pipeline = XDECFeaturePipeline(mock_fetcher, mock_config)
        result = pipeline.transform(sample_raw_data)

        # Should have windows for both resources
        assert result["num_windows"].shape[0] > 0
        assert result["cat_windows"].shape[0] > 0

        # Window size should match config
        assert result["num_windows"].shape[1] == 30
        assert result["cat_windows"].shape[1] == 30

        # Both branches should have same number of samples
        assert result["num_windows"].shape[0] == result["cat_windows"].shape[0]

    def test_transform_normalizes_numerical(
        self, mock_fetcher: MagicMock, mock_config: MagicMock, sample_raw_data: pd.DataFrame
    ) -> None:
        """transform should normalize numerical features."""
        from scry.data.pipeline import XDECFeaturePipeline

        pipeline = XDECFeaturePipeline(mock_fetcher, mock_config)
        result = pipeline.transform(sample_raw_data)

        # Should have normalization params
        assert "mean" in result["num_norm_params"]
        assert "std" in result["num_norm_params"]

        # Numerical data should be approximately normalized
        num_data = result["num_windows"].flatten()
        # Mean should be close to 0 (within reasonable tolerance)
        assert np.abs(num_data.mean()) < 1.0

    def test_transform_encodes_categorical(
        self, mock_fetcher: MagicMock, mock_config: MagicMock, sample_raw_data: pd.DataFrame
    ) -> None:
        """transform should encode categorical features to [0, 1]."""
        from scry.data.pipeline import XDECFeaturePipeline

        pipeline = XDECFeaturePipeline(mock_fetcher, mock_config)
        result = pipeline.transform(sample_raw_data)

        # Categorical data should be in [0, 1]
        cat_data = result["cat_windows"]
        assert cat_data.min() >= 0
        assert cat_data.max() <= 1

    def test_save_and_load_roundtrip(
        self, mock_fetcher: MagicMock, mock_config: MagicMock, sample_raw_data: pd.DataFrame
    ) -> None:
        """save_training_data and load_training_data should roundtrip correctly."""
        from scry.data.pipeline import XDECFeaturePipeline

        pipeline = XDECFeaturePipeline(mock_fetcher, mock_config)
        original = pipeline.transform(sample_raw_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_data.npz"

            pipeline.save_training_data(original, str(path))
            loaded = pipeline.load_training_data(str(path))

        # Check all arrays match
        np.testing.assert_array_almost_equal(
            original["num_windows"], loaded["num_windows"]
        )
        np.testing.assert_array_almost_equal(
            original["cat_windows"], loaded["cat_windows"]
        )
        np.testing.assert_array_almost_equal(
            original["num_norm_params"]["mean"], loaded["num_norm_params"]["mean"]
        )

        # Feature schema must round-trip order-exact, and categorical encoding
        # params must survive, so the trained model can align metrics by name.
        assert (
            loaded["feature_names"]["numerical"]
            == original["feature_names"]["numerical"]
        )
        assert (
            loaded["feature_names"]["categorical"]
            == original["feature_names"]["categorical"]
        )
        assert loaded["profile"] == original["profile"]
        np.testing.assert_array_almost_equal(
            original["cat_norm_params"]["min"], loaded["cat_norm_params"]["min"]
        )
        np.testing.assert_array_almost_equal(
            original["cat_norm_params"]["max"], loaded["cat_norm_params"]["max"]
        )

    def test_create_dataset_returns_valid_dataset(
        self, mock_fetcher: MagicMock, mock_config: MagicMock, sample_raw_data: pd.DataFrame
    ) -> None:
        """create_dataset should return valid XDECDataset."""
        from scry.data.feature_engineering import XDECDataset
        from scry.data.pipeline import XDECFeaturePipeline

        pipeline = XDECFeaturePipeline(mock_fetcher, mock_config)
        data = pipeline.transform(sample_raw_data)
        dataset = pipeline.create_dataset(data)

        assert isinstance(dataset, XDECDataset)
        assert len(dataset) == data["num_windows"].shape[0]

    def test_transform_handles_empty_data(
        self, mock_fetcher: MagicMock, mock_config: MagicMock
    ) -> None:
        """transform should handle empty DataFrame gracefully."""
        from scry.data.pipeline import XDECFeaturePipeline

        pipeline = XDECFeaturePipeline(mock_fetcher, mock_config)

        empty_df = pd.DataFrame(
            columns=[
                "resource_id",
                "host_name",
                "metric_name",
                "timestamp",
                "value",
                "datasource_instance",
                "datasource_name",
            ]
        )

        result = pipeline.transform(empty_df)

        assert result["num_windows"].shape[0] == 0
        assert result["cat_windows"].shape[0] == 0


    def test_transform_respects_window_step_from_config(
        self, mock_fetcher: MagicMock, sample_raw_data: pd.DataFrame
    ) -> None:
        """transform should use window_step from config for step parameter."""
        from scry.data.pipeline import XDECFeaturePipeline

        config = MagicMock()
        config.sequence_length = 30
        config.window_step = 10

        pipeline = XDECFeaturePipeline(mock_fetcher, config)
        result_step10 = pipeline.transform(sample_raw_data)

        # With step=10, should produce fewer windows than step=1
        config_step1 = MagicMock()
        config_step1.sequence_length = 30
        config_step1.window_step = 1

        pipeline_step1 = XDECFeaturePipeline(mock_fetcher, config_step1)
        result_step1 = pipeline_step1.transform(sample_raw_data)

        assert result_step10["num_windows"].shape[0] < result_step1["num_windows"].shape[0]


class TestXDECFeaturePipelineAsync:
    """Async tests for XDECFeaturePipeline."""

    @pytest.fixture
    def mock_client(self) -> MagicMock:
        """Create a mock HttpIngestClient."""
        client = MagicMock()
        client.get_training_data_time_chunked = AsyncMock(return_value=[])
        client.health_check = AsyncMock(return_value={"status": "healthy"})
        return client

    @pytest.fixture
    def mock_config(self) -> MagicMock:
        """Create a mock configuration."""
        config = MagicMock()
        config.sequence_length = 30
        return config

    async def test_extract_calls_fetcher(
        self, mock_client: MagicMock, mock_config: MagicMock
    ) -> None:
        """extract should call DataFetcher with correct time range."""
        from scry.data.pipeline import XDECFeaturePipeline

        pipeline = XDECFeaturePipeline.from_http_client(mock_client, mock_config)

        start = datetime(2024, 12, 1, tzinfo=timezone.utc)
        end = datetime(2024, 12, 2, tzinfo=timezone.utc)

        await pipeline.extract(start, end, profile="collector")

        # Should have called get_training_data_paginated
        mock_client.get_training_data_time_chunked.assert_called()

    async def test_run_returns_transformed_data(
        self, mock_client: MagicMock, mock_config: MagicMock
    ) -> None:
        """run should return transformed data dict."""
        from scry.data.pipeline import XDECFeaturePipeline

        # Setup mock with sample data
        records = []
        for minute in range(60):
            timestamp = f"2024-12-01T10:{minute:02d}:00+00:00"
            records.append(
                {
                    "resource_hash": "r1",
                    "metric_name": "cpuUsageNanoCores",
                    "timestamp": timestamp,
                    "value": 50.0,
                    "datasource_name": "Kubernetes_KSM_Pods",
                    "attributes": '{"host_name": "pod-1", "dataSourceInstanceName": "container"}',
                }
            )
        mock_client.get_training_data_time_chunked.return_value = records

        pipeline = XDECFeaturePipeline.from_http_client(mock_client, mock_config)

        start = datetime(2024, 12, 1, tzinfo=timezone.utc)
        end = datetime(2024, 12, 2, tzinfo=timezone.utc)

        result = await pipeline.run(start, end)

        assert "num_windows" in result
        assert "cat_windows" in result
