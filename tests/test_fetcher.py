# Description: Unit tests for the data fetcher over the object store.
# Description: Tests DataFrame conversion, schema normalization, and summary helpers.

"""Tests for scry.data.fetcher module."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from scry.data.fetcher import DataFetcher


def _write_parquet(path: Path, df: pd.DataFrame) -> None:
    """Write a DataFrame to Parquet via DuckDB (no pyarrow dependency)."""
    con = duckdb.connect()
    try:
        con.register("rows", df)
        con.execute(f"COPY rows TO '{path}' (FORMAT PARQUET)")
    finally:
        con.close()


@pytest.fixture
def datalake_parquet(tmp_path: Path) -> Path:
    """A Parquet store in the object-store layout: resource_hash, value_double, attributes JSON."""
    rows = [
        {
            "resource_hash": "abc123",
            "metric_name": "CpuUsage",
            "timestamp": pd.Timestamp("2024-12-01T10:00:00", tz="UTC"),
            "value_double": 45.5,
            "datasource_name": "LogicMonitor_Collector_ThreadCPUUsage",
            "attributes": '{"host_name": "collector-01", "dataSourceInstanceName": "thread-1"}',
        },
        {
            "resource_hash": "abc123",
            "metric_name": "ThreadCount",
            "timestamp": pd.Timestamp("2024-12-01T10:00:00", tz="UTC"),
            "value_double": 62.0,
            "datasource_name": "LogicMonitor_Collector_ThreadUsage",
            "attributes": '{"host_name": "collector-01", "dataSourceInstanceName": "main"}',
        },
        {
            "resource_hash": "xyz789",
            "metric_name": "CpuUsage",
            "timestamp": pd.Timestamp("2024-12-01T10:01:00", tz="UTC"),
            "value_double": 12.1,
            "datasource_name": "LogicMonitor_Collector_ThreadCPUUsage",
            "attributes": '{"host_name": "collector-02", "dataSourceInstanceName": "thread-1"}',
        },
    ]
    path = tmp_path / "metrics.parquet"
    _write_parquet(path, pd.DataFrame(rows))
    return path


class TestDataFetcher:
    """Tests for DataFetcher over an object store."""

    async def test_get_metrics_dataframe_returns_dataframe(self, datalake_parquet: Path) -> None:
        """get_metrics_dataframe should return a pandas DataFrame with all rows."""
        fetcher = DataFetcher.from_object_store(str(datalake_parquet))
        start = datetime(2024, 12, 1, tzinfo=timezone.utc)
        end = datetime(2024, 12, 2, tzinfo=timezone.utc)

        result = await fetcher.get_metrics_dataframe(start, end)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    async def test_get_metrics_dataframe_has_canonical_columns(
        self, datalake_parquet: Path
    ) -> None:
        """DataFrame should have all canonical columns."""
        fetcher = DataFetcher.from_object_store(str(datalake_parquet))
        start = datetime(2024, 12, 1, tzinfo=timezone.utc)
        end = datetime(2024, 12, 2, tzinfo=timezone.utc)

        result = await fetcher.get_metrics_dataframe(start, end)

        for col in [
            "resource_id",
            "host_name",
            "metric_name",
            "timestamp",
            "value",
            "datasource_instance",
            "datasource_name",
        ]:
            assert col in result.columns

    async def test_get_metrics_dataframe_empty_returns_empty_df(
        self, datalake_parquet: Path
    ) -> None:
        """A range with no data should return an empty DataFrame, not an error."""
        fetcher = DataFetcher.from_object_store(str(datalake_parquet))
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        end = datetime(2025, 1, 2, tzinfo=timezone.utc)

        result = await fetcher.get_metrics_dataframe(start, end)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    async def test_get_metrics_dataframe_types(self, datalake_parquet: Path) -> None:
        """value should be numeric and timestamp should be datetime."""
        fetcher = DataFetcher.from_object_store(str(datalake_parquet))
        start = datetime(2024, 12, 1, tzinfo=timezone.utc)
        end = datetime(2024, 12, 2, tzinfo=timezone.utc)

        result = await fetcher.get_metrics_dataframe(start, end)

        assert pd.api.types.is_numeric_dtype(result["value"])
        assert pd.api.types.is_datetime64_any_dtype(result["timestamp"])

    async def test_get_metrics_dataframe_normalizes_object_store_schema(
        self, datalake_parquet: Path
    ) -> None:
        """resource_hash, value_double, and attributes JSON normalize to the canonical schema."""
        fetcher = DataFetcher.from_object_store(str(datalake_parquet))
        start = datetime(2024, 12, 1, tzinfo=timezone.utc)
        end = datetime(2024, 12, 2, tzinfo=timezone.utc)

        result = await fetcher.get_metrics_dataframe(start, end)
        row = result[
            (result["resource_id"] == "abc123") & (result["metric_name"] == "CpuUsage")
        ].iloc[0]

        assert row["resource_id"] == "abc123"  # resource_hash -> resource_id
        assert row["host_name"] == "collector-01"  # extracted from attributes JSON
        assert row["datasource_instance"] == "thread-1"  # extracted from attributes JSON
        assert pd.notna(row["value"])  # value_double -> value

    async def test_get_resource_list(self, datalake_parquet: Path) -> None:
        """get_resource_list should return the distinct resources."""
        fetcher = DataFetcher.from_object_store(str(datalake_parquet))
        result = await fetcher.get_resource_list()

        assert {r["resource_id"] for r in result} == {"abc123", "xyz789"}

    async def test_get_metric_names(self, datalake_parquet: Path) -> None:
        """get_metric_names should return the distinct metric names."""
        fetcher = DataFetcher.from_object_store(str(datalake_parquet))
        result = await fetcher.get_metric_names()

        assert set(result) == {"CpuUsage", "ThreadCount"}

    async def test_get_data_summary(self, datalake_parquet: Path) -> None:
        """get_data_summary should report row, resource, and metric counts."""
        fetcher = DataFetcher.from_object_store(str(datalake_parquet))
        result = await fetcher.get_data_summary()

        assert result["total_rows"] == 3
        assert result["unique_resources"] == 2
        assert result["unique_metrics"] == 2
