# Description: HTTP client for HttpIngest ML endpoints.
# Description: Provides async data fetching via REST API instead of direct SQL.

"""HTTP client for HttpIngest ML API endpoints."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any

import httpx

from scry.data.sources.base import DataSource, normalize_record
from scry.utils.config import get_config

logger = logging.getLogger(__name__)


class HttpIngestClientError(Exception):
    """Raised when HttpIngest API returns an error."""

    pass


class HttpIngestClient:
    """Async HTTP client for HttpIngest ML endpoints.

    Usage:
        async with HttpIngestClient() as client:
            data = await client.get_training_data(profile="kubernetes")
    """

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 600.0,
        max_retries: int = 5,
    ) -> None:
        """Initialize the HTTP client.

        Args:
            base_url: HttpIngest API base URL. Defaults to config value.
            timeout: Request timeout in seconds. Default 600s for slow backend queries.
            max_retries: Max retry attempts for transient errors (timeouts, 504s).
        """
        config = get_config()
        self._base_url = (base_url or config.httpingest_url).rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries
        self._client: httpx.AsyncClient | None = None

    async def connect(self) -> None:
        """Create the HTTP client."""
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(self._timeout),
            headers={"Accept": "application/json"},
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "HttpIngestClient":
        """Enter async context manager."""
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type | None,
        exc_val: Exception | None,
        exc_tb: Any | None,
    ) -> None:
        """Exit async context manager."""
        await self.close()

    def _ensure_connected(self) -> httpx.AsyncClient:
        """Ensure client is connected."""
        if self._client is None:
            raise HttpIngestClientError(
                "HTTP client not connected. Call connect() first or use as context manager."
            )
        return self._client

    async def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make an HTTP request with retry logic and return JSON response.

        Retries on transient errors (timeouts, connection resets) with
        exponential backoff. Non-retryable HTTP errors (4xx) raise immediately.

        Args:
            method: HTTP method (GET, POST, etc.)
            path: API path (e.g., /api/ml/inventory)
            params: Query parameters

        Returns:
            JSON response as dictionary

        Raises:
            HttpIngestClientError: On HTTP errors or exhausted retries
        """
        client = self._ensure_connected()
        last_error: Exception | None = None

        for attempt in range(self._max_retries):
            try:
                response = await client.request(method, path, params=params)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                # 4xx errors are not retryable
                if 400 <= e.response.status_code < 500:
                    raise HttpIngestClientError(
                        f"HTTP {e.response.status_code}: {e.response.text}"
                    ) from e
                # 5xx errors are retryable
                last_error = e
                logger.warning(
                    "HTTP %d on %s %s (attempt %d/%d)",
                    e.response.status_code,
                    method,
                    path,
                    attempt + 1,
                    self._max_retries,
                )
            except (httpx.TimeoutException, httpx.ConnectError) as e:
                last_error = e
                logger.warning(
                    "%s on %s %s (attempt %d/%d): %s",
                    type(e).__name__,
                    method,
                    path,
                    attempt + 1,
                    self._max_retries,
                    str(e),
                )
            except httpx.RequestError as e:
                raise HttpIngestClientError(f"Request failed: {e}") from e

            if attempt < self._max_retries - 1:
                backoff = 2**attempt * 10  # 10s, 20s, 40s, 80s, ...
                logger.info("Retrying in %ds...", backoff)
                await asyncio.sleep(backoff)

        raise HttpIngestClientError(
            f"Request failed after {self._max_retries} attempts: {last_error}"
        ) from last_error

    async def get_inventory(self) -> dict[str, Any]:
        """Get available metrics, resources, datasources, and time range.

        Returns:
            Dictionary with metrics, resources, datasources, time_range keys.
        """
        return await self._request("GET", "/api/ml/inventory")

    async def get_profiles(self) -> dict[str, Any]:
        """Get list of available feature profiles.

        Returns:
            Dictionary with profiles list and default_profile.
            Each profile has name, description, numerical_features,
            categorical_features.
        """
        return await self._request("GET", "/api/ml/profiles")

    async def get_profile_coverage(self) -> dict[str, Any]:
        """Get coverage percentage for each feature profile.

        Returns:
            Dictionary with profiles list, each containing:
            - name, description, coverage_percent
            - available, missing feature lists
            - total_expected, total_available counts
        """
        return await self._request("GET", "/api/ml/profile-coverage")

    async def get_quality(
        self,
        profile: str | None = None,
        lookback_hours: int = 24,
    ) -> dict[str, Any]:
        """Get data quality metrics.

        Args:
            profile: Optional profile name to filter by
            lookback_hours: Hours to look back for freshness check

        Returns:
            Dictionary with summary, freshness, gaps, ranges data.
        """
        params: dict[str, Any] = {"lookback_hours": lookback_hours}
        if profile:
            params["profile"] = profile
        return await self._request("GET", "/api/ml/quality", params=params)

    async def get_training_data(
        self,
        profile: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        resource_id: int | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Get training data with optional filtering (single page).

        Args:
            profile: Feature profile to filter metrics
            start_time: Start of time range
            end_time: End of time range
            resource_id: Filter to specific resource
            limit: Maximum rows to return

        Returns:
            List of metric data points with resource_id, host_name,
            metric_name, timestamp, value, datasource_instance.
        """
        params: dict[str, Any] = {}

        if profile:
            params["profile"] = profile
        if start_time:
            params["start_time"] = start_time.isoformat()
        if end_time:
            params["end_time"] = end_time.isoformat()
        if resource_id:
            params["resource_id"] = resource_id
        if limit:
            params["limit"] = limit

        response = await self._request("GET", "/api/ml/training-data", params=params)
        return response.get("data", [])

    async def get_training_data_paginated(
        self,
        profile: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        page_size: int = 10000,
        max_records: int | None = None,
        progress_callback: Any | None = None,
    ) -> list[dict[str, Any]]:
        """Get training data with automatic pagination.

        Fetches all matching records by paginating through the API.
        Use progress_callback to monitor progress during large fetches.

        Args:
            profile: Feature profile to filter metrics
            start_time: Start of time range
            end_time: End of time range
            page_size: Records per page (default 50000)
            max_records: Stop after this many records (None for all)
            progress_callback: Callable(fetched, total) called after each page

        Returns:
            List of all metric data points.
        """
        all_records: list[dict[str, Any]] = []
        offset = 0
        total = None

        while True:
            params: dict[str, Any] = {
                "limit": page_size,
                "offset": offset,
            }
            if profile:
                params["profile"] = profile
            if start_time:
                params["start_time"] = start_time.isoformat()
            if end_time:
                params["end_time"] = end_time.isoformat()

            response = await self._request(
                "GET", "/api/ml/training-data", params=params
            )
            page_data = response.get("data", [])
            meta = response.get("meta", {})

            if total is None:
                total = meta.get("total", 0)

            all_records.extend(page_data)

            if progress_callback and total:
                progress_callback(len(all_records), total)

            # Stop conditions
            if len(page_data) < page_size:
                break
            if max_records and len(all_records) >= max_records:
                all_records = all_records[:max_records]
                break

            offset += page_size

        return all_records

    async def get_training_data_time_chunked(
        self,
        profile: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        page_size: int = 50000,
        max_records: int | None = None,
        progress_callback: Any | None = None,
        chunk_hours: float = 1.0,
    ) -> list[dict[str, Any]]:
        """Get training data by splitting the time range into chunks.

        Avoids high OFFSET values in the backend by splitting the date range
        into smaller time windows (default 1 hour). Each chunk paginates
        independently starting at offset 0, keeping offsets low.

        Args:
            profile: Feature profile to filter metrics
            start_time: Start of time range (required)
            end_time: End of time range (required)
            page_size: Records per page (default 50000)
            max_records: Stop after this many records (None for all)
            progress_callback: Callable(fetched, total) called after each chunk
            chunk_hours: Size of each time chunk in hours (default 1.0)

        Returns:
            List of all metric data points across all chunks.

        Raises:
            ValueError: If start_time or end_time is missing
        """
        if start_time is None:
            raise ValueError("start_time is required for time-chunked pagination")
        if end_time is None:
            raise ValueError("end_time is required for time-chunked pagination")

        # Generate time chunk boundaries
        chunk_delta = timedelta(hours=chunk_hours)
        chunks: list[tuple] = []
        chunk_start = start_time
        while chunk_start < end_time:
            chunk_end = min(chunk_start + chunk_delta, end_time)
            chunks.append((chunk_start, chunk_end))
            chunk_start = chunk_end

        all_records: list[dict[str, Any]] = []

        for chunk_start, chunk_end in chunks:
            # Paginate within this chunk (offset resets to 0 for each chunk)
            offset = 0

            while True:
                params: dict[str, Any] = {
                    "limit": page_size,
                    "offset": offset,
                    "start_time": chunk_start.isoformat(),
                    "end_time": chunk_end.isoformat(),
                }
                if profile:
                    params["profile"] = profile

                response = await self._request(
                    "GET", "/api/ml/training-data", params=params
                )
                page_data = response.get("data", [])

                all_records.extend(page_data)

                # Stop paginating this chunk if we got fewer than page_size
                if len(page_data) < page_size:
                    break

                offset += page_size

                # Check max_records within pagination loop
                if max_records and len(all_records) >= max_records:
                    break

            # Report progress after each chunk
            if progress_callback:
                progress_callback(len(all_records), len(all_records))

            # Check max_records across chunks
            if max_records and len(all_records) >= max_records:
                all_records = all_records[:max_records]
                break

        return all_records

    async def health_check(self) -> dict[str, Any]:
        """Check HttpIngest health status.

        Returns:
            Dictionary with status, version, components.
        """
        return await self._request("GET", "/api/health")


class HttpDataSource(DataSource):
    """DataSource backed by an HttpIngest ML API (the LogicMonitor adapter).

    Wraps an :class:`HttpIngestClient` and normalizes its records to the
    canonical schema. Used when metrics are served by an HttpIngest-compatible
    endpoint rather than read directly from object storage.
    """

    def __init__(self, client: "HttpIngestClient") -> None:
        self._client = client

    async def fetch_metrics(
        self,
        start_time: datetime,
        end_time: datetime,
        profile: str | None = None,
    ) -> list[dict[str, Any]]:
        raw = await self._client.get_training_data_time_chunked(
            profile=profile,
            start_time=start_time,
            end_time=end_time,
            page_size=50000,
            chunk_hours=1.0,
        )
        return [normalize_record(r) for r in raw]

    async def fetch_resources(self) -> list[dict[str, Any]]:
        inventory = await self._client.get_inventory()
        return inventory.get("resources", [])

    async def fetch_metric_names(self) -> list[str]:
        inventory = await self._client.get_inventory()
        return inventory.get("metrics", [])

    async def fetch_summary(self) -> dict[str, Any]:
        inventory = await self._client.get_inventory()
        time_range = inventory.get("time_range", {})
        return {
            "total_rows": inventory.get("total_data_points", 0),
            "unique_resources": len(inventory.get("resources", [])),
            "unique_metrics": len(inventory.get("metrics", [])),
            "earliest_timestamp": time_range.get("start"),
            "latest_timestamp": time_range.get("end"),
        }

    async def get_profile_coverage(self) -> dict[str, Any]:
        """Feature-profile coverage report from the HttpIngest ML API."""
        return await self._client.get_profile_coverage()

    async def get_quality(
        self,
        profile: str | None = None,
        lookback_hours: int = 24,
    ) -> dict[str, Any]:
        """Data-quality metrics from the HttpIngest ML API."""
        return await self._client.get_quality(profile=profile, lookback_hours=lookback_hours)
