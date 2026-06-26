# Description: Data fetcher: turns a DataSource into pandas DataFrames for feature work.
# Description: Factory methods select the backend (object store, local files, HttpIngest).

"""Data fetcher service.

A thin layer over a :class:`~scry.data.sources.base.DataSource` that returns
pandas DataFrames in the canonical schema, ready for feature engineering.
"""

from datetime import datetime
from typing import Any

import pandas as pd

from scry.data.sources.base import METRICS_COLUMNS, DataSource


class DataFetcher:
    """Fetches metric data from a DataSource and shapes it for training/inference."""

    def __init__(self, source: DataSource) -> None:
        """Initialize with a concrete data source.

        Args:
            source: A DataSource implementation.
        """
        self._source = source

    @classmethod
    def from_object_store(
        cls,
        uri: str,
        *,
        hive_partitioning: bool = True,
        data_format: str | None = None,
        connection_string: str | None = None,
    ) -> "DataFetcher":
        """Create a fetcher that reads Parquet/CSV from a URI via DuckDB.

        Args:
            uri: Path or glob; scheme selects the backend (file/s3/gs/az).
            hive_partitioning: Treat ``key=value`` path segments as columns.
            data_format: ``"parquet"`` or ``"csv"``; inferred from the URI when omitted.
            connection_string: Optional Azure storage connection string.
        """
        from scry.data.sources.object_store import ObjectStoreSource

        return cls(
            ObjectStoreSource(
                uri,
                hive_partitioning=hive_partitioning,
                data_format=data_format,
                connection_string=connection_string,
            )
        )

    @classmethod
    def from_local(cls, path: str, *, data_format: str | None = None) -> "DataFetcher":
        """Create a fetcher for local Parquet/CSV files (a thin alias for object store)."""
        return cls.from_object_store(path, data_format=data_format)

    @classmethod
    def from_http_client(cls, client: Any) -> "DataFetcher":
        """Create a fetcher backed by an HttpIngest client (the LogicMonitor adapter).

        Requires the ``logicmonitor`` extra (httpx).
        """
        from scry.data.sources.http_ingest import HttpDataSource

        return cls(HttpDataSource(client))

    async def get_metrics_dataframe(
        self,
        start_time: datetime,
        end_time: datetime,
        profile: str | None = None,
    ) -> pd.DataFrame:
        """Fetch metrics in the time range as a canonical-schema DataFrame.

        Args:
            start_time: Start of the range (inclusive).
            end_time: End of the range (exclusive).
            profile: Optional feature profile to filter metric names.

        Returns:
            DataFrame with the canonical METRICS_COLUMNS, typed value and timestamp.
        """
        records = await self._source.fetch_metrics(start_time, end_time, profile)
        if not records:
            return pd.DataFrame(columns=METRICS_COLUMNS)

        df = pd.DataFrame(records)
        for col in METRICS_COLUMNS:
            if col not in df.columns:
                df[col] = None

        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601")
        return df[METRICS_COLUMNS]

    async def get_resource_list(self) -> list[dict[str, Any]]:
        """Fetch the list of resources."""
        return await self._source.fetch_resources()

    async def get_metric_names(self) -> list[str]:
        """Fetch the list of available metric names."""
        return await self._source.fetch_metric_names()

    async def get_data_summary(self) -> dict[str, Any]:
        """Fetch summary statistics about the available data."""
        return await self._source.fetch_summary()

    async def check_profile_coverage(self, min_coverage: float = 80.0) -> dict[str, Any]:
        """Check feature-profile coverage before training.

        Only supported by sources that expose coverage (the HttpIngest adapter).

        Args:
            min_coverage: Minimum acceptable coverage percentage.

        Returns:
            Dict with profiles, warnings, and a ready flag.
        """
        if not hasattr(self._source, "get_profile_coverage"):
            raise NotImplementedError(
                "profile coverage is only available for the HttpIngest source"
            )

        coverage = await self._source.get_profile_coverage()
        profiles = coverage.get("profiles", [])

        warnings = []
        for p in profiles:
            if p.get("coverage_percent", 0) < min_coverage:
                warnings.append(
                    f"{p['name']}: {p['coverage_percent']:.1f}% "
                    f"(missing: {', '.join(p.get('missing', [])[:3])}...)"
                )

        return {"profiles": profiles, "warnings": warnings, "ready": len(warnings) == 0}

    async def check_data_quality(
        self,
        profile: str | None = None,
        min_freshness: float = 50.0,
        min_gap_score: float = 80.0,
    ) -> dict[str, Any]:
        """Check data quality before training.

        Only supported by sources that expose quality metrics (the HttpIngest adapter).

        Args:
            profile: Optional profile to check.
            min_freshness: Minimum acceptable freshness score.
            min_gap_score: Minimum acceptable gap score.

        Returns:
            Dict with quality metrics, warnings, and a ready flag.
        """
        if not hasattr(self._source, "get_quality"):
            raise NotImplementedError(
                "data quality checks are only available for the HttpIngest source"
            )

        quality = await self._source.get_quality(profile=profile)
        summary = quality.get("summary", {})

        warnings = []
        freshness_score = summary.get("freshness_score", 0)
        gap_score = summary.get("gap_score", 100)

        if freshness_score < min_freshness:
            warnings.append(
                f"Data freshness low: {freshness_score:.1f}% (min: {min_freshness}%)"
            )
        if gap_score < min_gap_score:
            warnings.append(
                f"Data gaps detected: {gap_score:.1f}% coverage (min: {min_gap_score}%)"
            )

        return {
            "summary": summary,
            "freshness": quality.get("freshness", []),
            "gaps": quality.get("gaps", []),
            "warnings": warnings,
            "ready": len(warnings) == 0,
        }
