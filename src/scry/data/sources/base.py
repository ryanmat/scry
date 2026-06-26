# Description: Data-source abstraction and the canonical metric schema for Scry.
# Description: Every data source normalizes to one long-format table defined here.

"""Data-source interface and canonical schema.

Every source returns metric records normalized to ``METRICS_COLUMNS``: a
long-format table of (resource, metric, timestamp, value) plus optional labels.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

# Canonical long-format metric schema. Every source normalizes to this.
METRICS_COLUMNS = [
    "resource_id",
    "host_name",
    "metric_name",
    "timestamp",
    "value",
    "datasource_instance",
    "datasource_name",
]


def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    """Normalize a raw source record to the canonical metric schema.

    Accepts records that use the canonical field names directly, or the common
    object-store layout (``resource_hash`` plus a JSON ``attributes`` blob), and
    resolves the value across ``value`` / ``value_double`` / ``value_int``.

    Args:
        record: A single raw record from a data source.

    Returns:
        A dict with exactly the canonical ``METRICS_COLUMNS`` keys.
    """
    attributes: dict[str, Any] = {}
    raw_attrs = record.get("attributes")
    if isinstance(raw_attrs, str):
        try:
            attributes = json.loads(raw_attrs)
        except (json.JSONDecodeError, TypeError):
            attributes = {}
    elif isinstance(raw_attrs, dict):
        attributes = raw_attrs

    value = record.get("value")
    if value is None:
        value = record.get("value_double")
    if value is None:
        value = record.get("value_int")

    return {
        "resource_id": record.get("resource_id", record.get("resource_hash")),
        "host_name": attributes.get("host_name", record.get("host_name")),
        "metric_name": record.get("metric_name"),
        "timestamp": record.get("timestamp"),
        "value": value,
        "datasource_instance": attributes.get(
            "dataSourceInstanceName",
            record.get("datasource_instance"),
        ),
        "datasource_name": record.get("datasource_name"),
    }


class DataSource(ABC):
    """Abstract interface for metric data sources.

    Implementations read from somewhere (local files, object storage, a remote
    ingestion API) and return records normalized to the canonical schema.
    """

    @abstractmethod
    async def fetch_metrics(
        self,
        start_time: datetime,
        end_time: datetime,
        profile: str | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch metric records in [start_time, end_time), normalized to the canonical schema."""

    @abstractmethod
    async def fetch_resources(self) -> list[dict[str, Any]]:
        """Fetch the list of distinct resources (dicts with at least ``resource_id``)."""

    @abstractmethod
    async def fetch_metric_names(self) -> list[str]:
        """Fetch the list of distinct metric names."""

    @abstractmethod
    async def fetch_summary(self) -> dict[str, Any]:
        """Fetch summary statistics: row count, distinct resources/metrics, time range."""
