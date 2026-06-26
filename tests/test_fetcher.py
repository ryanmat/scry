# Description: Unit tests for the data fetcher service.
# Description: Tests DataFrame conversion, HTTP data source, and data type handling.

"""Tests for scry.data.fetcher module."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest


class TestDataFetcher:
    """Tests for DataFetcher with HTTP data source."""

    @pytest.fixture
    def mock_client(self) -> MagicMock:
        """Create a mock HttpIngestClient."""
        client = MagicMock()
        client.get_training_data_time_chunked = AsyncMock()
        client.get_inventory = AsyncMock()
        client.get_profile_coverage = AsyncMock()
        client.get_quality = AsyncMock()
        return client

    @pytest.fixture
    def sample_datalake_records(self) -> list:
        """Sample metric records as returned by HttpIngest ML API."""
        return [
            {
                "resource_hash": "abc123",
                "metric_name": "CpuUsage",
                "timestamp": "2024-12-01T10:00:00+00:00",
                "value": 45.5,
                "datasource_name": "LogicMonitor_Collector_ThreadCPUUsage",
                "attributes": '{"host_name": "collector-01", "dataSourceInstanceName": "thread-1"}',
            },
            {
                "resource_hash": "abc123",
                "metric_name": "ThreadCount",
                "timestamp": "2024-12-01T10:00:00+00:00",
                "value": 62.0,
                "datasource_name": "LogicMonitor_Collector_ThreadUsage",
                "attributes": '{"host_name": "collector-01", "dataSourceInstanceName": "main"}',
            },
            {
                "resource_hash": "xyz789",
                "metric_name": "CpuUsage",
                "timestamp": "2024-12-01T10:01:00+00:00",
                "value": 12.1,
                "datasource_name": "LogicMonitor_Collector_ThreadCPUUsage",
                "attributes": '{"host_name": "collector-02", "dataSourceInstanceName": "thread-1"}',
            },
        ]

    async def test_get_metrics_dataframe_returns_dataframe(
        self, mock_client: MagicMock, sample_datalake_records: list
    ) -> None:
        """get_metrics_dataframe should return a pandas DataFrame."""
        from scry.data.fetcher import DataFetcher

        mock_client.get_training_data_time_chunked.return_value = sample_datalake_records

        fetcher = DataFetcher.from_http_client(mock_client)
        start = datetime(2024, 12, 1, tzinfo=timezone.utc)
        end = datetime(2024, 12, 2, tzinfo=timezone.utc)

        result = await fetcher.get_metrics_dataframe(start, end)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    async def test_get_metrics_dataframe_has_correct_columns(
        self, mock_client: MagicMock, sample_datalake_records: list
    ) -> None:
        """DataFrame should have all required columns."""
        from scry.data.fetcher import DataFetcher

        mock_client.get_training_data_time_chunked.return_value = sample_datalake_records

        fetcher = DataFetcher.from_http_client(mock_client)
        start = datetime(2024, 12, 1, tzinfo=timezone.utc)
        end = datetime(2024, 12, 2, tzinfo=timezone.utc)

        result = await fetcher.get_metrics_dataframe(start, end)

        expected_columns = [
            "resource_id",
            "host_name",
            "metric_name",
            "timestamp",
            "value",
            "datasource_instance",
            "datasource_name",
        ]
        for col in expected_columns:
            assert col in result.columns

    async def test_get_metrics_dataframe_empty_returns_empty_df(
        self, mock_client: MagicMock
    ) -> None:
        """Empty results should return empty DataFrame, not error."""
        from scry.data.fetcher import DataFetcher

        mock_client.get_training_data_time_chunked.return_value = []

        fetcher = DataFetcher.from_http_client(mock_client)
        start = datetime(2024, 12, 1, tzinfo=timezone.utc)
        end = datetime(2024, 12, 2, tzinfo=timezone.utc)

        result = await fetcher.get_metrics_dataframe(start, end)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    async def test_get_metrics_dataframe_correct_data_types(
        self, mock_client: MagicMock, sample_datalake_records: list
    ) -> None:
        """DataFrame columns should have correct data types."""
        from scry.data.fetcher import DataFetcher

        mock_client.get_training_data_time_chunked.return_value = sample_datalake_records

        fetcher = DataFetcher.from_http_client(mock_client)
        start = datetime(2024, 12, 1, tzinfo=timezone.utc)
        end = datetime(2024, 12, 2, tzinfo=timezone.utc)

        result = await fetcher.get_metrics_dataframe(start, end)

        # Value should be numeric
        assert pd.api.types.is_numeric_dtype(result["value"])
        # Timestamp should be datetime
        assert pd.api.types.is_datetime64_any_dtype(result["timestamp"])

    async def test_get_metrics_dataframe_normalizes_datalake_schema(
        self, mock_client: MagicMock, sample_datalake_records: list
    ) -> None:
        """DataFrame should normalize Data Lake schema to standard format."""
        from scry.data.fetcher import DataFetcher

        mock_client.get_training_data_time_chunked.return_value = sample_datalake_records

        fetcher = DataFetcher.from_http_client(mock_client)
        start = datetime(2024, 12, 1, tzinfo=timezone.utc)
        end = datetime(2024, 12, 2, tzinfo=timezone.utc)

        result = await fetcher.get_metrics_dataframe(start, end)

        # resource_hash should be mapped to resource_id
        assert result.iloc[0]["resource_id"] == "abc123"
        # host_name should be extracted from attributes JSON
        assert result.iloc[0]["host_name"] == "collector-01"
        # datasource_instance should be extracted from attributes JSON
        assert result.iloc[0]["datasource_instance"] == "thread-1"

    async def test_get_metrics_dataframe_passes_profile(
        self, mock_client: MagicMock
    ) -> None:
        """get_metrics_dataframe should pass profile to HTTP client."""
        from scry.data.fetcher import DataFetcher

        mock_client.get_training_data_time_chunked.return_value = []

        fetcher = DataFetcher.from_http_client(mock_client)
        start = datetime(2024, 12, 1, tzinfo=timezone.utc)
        end = datetime(2024, 12, 2, tzinfo=timezone.utc)

        await fetcher.get_metrics_dataframe(start, end, profile="collector")

        mock_client.get_training_data_time_chunked.assert_called_once()
        call_kwargs = mock_client.get_training_data_time_chunked.call_args
        assert call_kwargs.kwargs.get("profile") == "collector" or \
            call_kwargs[1].get("profile") == "collector"

    async def test_get_resource_list_returns_list(
        self, mock_client: MagicMock
    ) -> None:
        """get_resource_list should return list of resource dicts."""
        from scry.data.fetcher import DataFetcher

        mock_client.get_inventory.return_value = {
            "resources": [
                {"resource_id": 1, "host_name": "collector-01"},
                {"resource_id": 2, "host_name": "collector-02"},
            ],
        }

        fetcher = DataFetcher.from_http_client(mock_client)
        result = await fetcher.get_resource_list()

        assert isinstance(result, list)
        assert len(result) == 2

    async def test_get_metric_names_returns_list(
        self, mock_client: MagicMock
    ) -> None:
        """get_metric_names should return list of metric name strings."""
        from scry.data.fetcher import DataFetcher

        mock_client.get_inventory.return_value = {
            "metrics": ["CpuUsage", "ThreadCount", "QueueSize"],
        }

        fetcher = DataFetcher.from_http_client(mock_client)
        result = await fetcher.get_metric_names()

        assert isinstance(result, list)
        assert len(result) == 3
        assert "CpuUsage" in result

    async def test_get_data_summary_returns_dict(
        self, mock_client: MagicMock
    ) -> None:
        """get_data_summary should return summary dict."""
        from scry.data.fetcher import DataFetcher

        mock_client.get_inventory.return_value = {
            "total_data_points": 50000,
            "resources": [{"id": 1}, {"id": 2}],
            "metrics": ["CpuUsage", "ThreadCount"],
            "time_range": {
                "start": "2024-12-01T00:00:00Z",
                "end": "2024-12-02T00:00:00Z",
            },
        }

        fetcher = DataFetcher.from_http_client(mock_client)
        result = await fetcher.get_data_summary()

        assert isinstance(result, dict)
        assert result["total_rows"] == 50000
        assert result["unique_resources"] == 2
        assert result["unique_metrics"] == 2
