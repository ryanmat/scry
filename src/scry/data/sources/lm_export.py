# Description: LogicMonitor REST (LMv1) exporter that pulls device metrics into the canonical schema.
# Description: Walks devices -> datasources -> instances -> data and writes long-format Parquet/CSV.

"""Export LogicMonitor time-series metrics to Scry's canonical schema.

This is the bring-your-own-data LogicMonitor adapter: a generic LM user runs it
with their own read-only LMv1 API token to land their device metrics as a
canonical Parquet table that Scry trains on. It talks to the LogicMonitor REST
API v3 directly (no MCP, no proprietary server), authenticating with the
documented LMv1 scheme.

The canonical row shape is defined by
:data:`scry.data.sources.base.METRICS_COLUMNS`. One row per
(resource, metric, timestamp) is emitted; ``"No Data"`` samples are dropped.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import parse_qsl

import httpx

from scry.data.sources.base import METRICS_COLUMNS

logger = logging.getLogger(__name__)

# LogicMonitor returns this literal for gaps / not-yet-collected samples.
NO_DATA = "No Data"

# Endpoints that page on size/offset; the metric data endpoint pages on nextPageParams.
_DEFAULT_PAGE_SIZE = 1000
_MAX_DATA_PAGES = 200


@dataclass(frozen=True)
class LMCredentials:
    """LMv1 API credentials for one LogicMonitor portal."""

    access_id: str
    access_key: str
    company: str

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> LMCredentials:
        """Build credentials from ``LM_ACCESS_ID`` / ``LM_ACCESS_KEY`` / ``LM_COMPANY``.

        Raises:
            ValueError: if any of the three variables is missing, with a clear message.
        """
        src = env if env is not None else os.environ
        missing = [k for k in ("LM_ACCESS_ID", "LM_ACCESS_KEY", "LM_COMPANY") if not src.get(k)]
        if missing:
            raise ValueError(
                "Missing LogicMonitor credentials: "
                + ", ".join(missing)
                + ". Set them in the environment or a gitignored .env."
            )
        return cls(
            access_id=src["LM_ACCESS_ID"],
            access_key=src["LM_ACCESS_KEY"],
            company=src["LM_COMPANY"],
        )


class LMRestClient:
    """Minimal LMv1-signed client for the LogicMonitor REST API v3.

    Signs each request per the LMv1 scheme (HMAC-SHA256 over
    ``VERB + epoch_ms + body + resourcePath``, hex-digested then base64-encoded),
    retries transient 429/5xx with backoff, and raises on other errors rather than
    swallowing them.
    """

    def __init__(
        self,
        creds: LMCredentials,
        *,
        timeout: float = 30.0,
        max_retries: int = 4,
        rate_limit_pause: float = 0.2,
        client: httpx.Client | None = None,
    ) -> None:
        self._creds = creds
        self._base_url = f"https://{creds.company}.logicmonitor.com/santaba/rest"
        self._max_retries = max_retries
        self._rate_limit_pause = rate_limit_pause
        self._client = client or httpx.Client(timeout=timeout, headers={"X-Version": "3"})

    def __enter__(self) -> LMRestClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def _auth_header(self, verb: str, resource_path: str, body: str = "") -> str:
        epoch = str(int(time.time() * 1000))
        message = f"{verb}{epoch}{body}{resource_path}"
        digest = hmac.new(
            self._creds.access_key.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        signature = base64.b64encode(digest.encode("utf-8")).decode("utf-8")
        return f"LMv1 {self._creds.access_id}:{signature}:{epoch}"

    def _throttle(self, response: httpx.Response) -> None:
        """Adaptive pause after a successful GET, honoring LM rate-limit headers.

        The portal-wide LogicMonitor budget is ~500 GET/min shared across every
        token; each response advertises ``X-Rate-Limit-Remaining`` and
        ``X-Rate-Limit-Window`` (window in seconds) and returns 429 once the
        budget is spent. Spread the remaining budget evenly over the window so a
        sustained pull stays under the limit; if the budget is already spent,
        wait out the full window. When the headers are absent or unparseable,
        fall back to the fixed ``rate_limit_pause``.
        """
        sleep_for = self._rate_limit_pause
        remaining_raw = response.headers.get("X-Rate-Limit-Remaining")
        window_raw = response.headers.get("X-Rate-Limit-Window")
        if remaining_raw is not None and window_raw is not None:
            try:
                remaining = int(remaining_raw)
                window = float(window_raw)
            except ValueError:
                logger.debug(
                    "LM rate-limit headers unparseable (remaining=%r, window=%r); "
                    "using fixed pause",
                    remaining_raw,
                    window_raw,
                )
            else:
                if window > 0 and remaining > 0:
                    sleep_for = max(self._rate_limit_pause, window / remaining)
                elif window > 0 and remaining == 0:
                    sleep_for = window
        if sleep_for > 0:
            time.sleep(sleep_for)

    def get(self, resource_path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """GET a REST resource path (e.g. ``/device/devices``). Query params are not signed."""
        url = f"{self._base_url}{resource_path}"
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            headers = {"Authorization": self._auth_header("GET", resource_path)}
            try:
                resp = self._client.get(url, params=params, headers=headers)
            except httpx.HTTPError as exc:  # network/transport error
                last_exc = exc
                logger.warning("LM GET %s transport error (attempt %d): %s", resource_path, attempt, exc)
            else:
                if resp.status_code == 429 or resp.status_code >= 500:
                    last_exc = httpx.HTTPStatusError(
                        f"{resp.status_code} for {resource_path}", request=resp.request, response=resp
                    )
                    logger.warning(
                        "LM GET %s rate-limited/5xx %s (attempt %d)",
                        resource_path,
                        resp.status_code,
                        attempt,
                    )
                else:
                    resp.raise_for_status()
                    self._throttle(resp)
                    return resp.json()
            # backoff before retrying
            if attempt < self._max_retries:
                time.sleep(min(2.0 ** attempt, 30.0))
        assert last_exc is not None
        raise last_exc

    def get_paginated(
        self,
        resource_path: str,
        params: dict[str, Any] | None = None,
        *,
        page_size: int = _DEFAULT_PAGE_SIZE,
    ) -> Iterator[dict[str, Any]]:
        """Yield items across size/offset pages of a list endpoint."""
        offset = 0
        base_params = dict(params or {})
        while True:
            page_params = {**base_params, "size": page_size, "offset": offset}
            body = self.get(resource_path, page_params)
            items = body.get("items", [])
            yield from items
            total = body.get("total")
            offset += len(items)
            if not items or len(items) < page_size or (isinstance(total, int) and offset >= total):
                break


class LogicMonitorExporter:
    """Pull device metrics from LogicMonitor and write the canonical schema."""

    def __init__(self, client: LMRestClient) -> None:
        self._client = client

    # -- discovery --------------------------------------------------------

    def list_devices(
        self,
        *,
        ids: Sequence[int] | None = None,
        group_id: int | None = None,
        name_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Resolve the target devices by explicit ids, group membership, or display-name match."""
        fields = "id,displayName,name"
        if ids:
            devices = []
            for device_id in ids:
                devices.append(self._client.get(f"/device/devices/{device_id}", {"fields": fields}))
            return devices
        if group_id is not None:
            path = f"/device/groups/{group_id}/devices"
            return list(self._client.get_paginated(path, {"fields": fields}))
        params: dict[str, Any] = {"fields": fields}
        if name_filter:
            params["filter"] = f'displayName~"{name_filter}"'
        return list(self._client.get_paginated("/device/devices", params))

    def list_datasources(
        self, device_id: int, *, name_filters: Sequence[str] | None = None
    ) -> list[dict[str, Any]]:
        """List a device's applied datasources, optionally filtered by name substring(s)."""
        path = f"/device/devices/{device_id}/devicedatasources"
        items = list(
            self._client.get_paginated(path, {"fields": "id,dataSourceName,instanceNumber"})
        )
        items = [d for d in items if (d.get("instanceNumber") or 0) > 0]
        if name_filters:
            items = [
                d
                for d in items
                if any(f.lower() in (d.get("dataSourceName") or "").lower() for f in name_filters)
            ]
        return items

    def list_instances(self, device_id: int, hds_id: int) -> list[dict[str, Any]]:
        """List the monitored instances of one device-datasource."""
        path = f"/device/devices/{device_id}/devicedatasources/{hds_id}/instances"
        return list(self._client.get_paginated(path, {"fields": "id,name,displayName"}))

    def fetch_data(
        self, device_id: int, hds_id: int, instance_id: int, start: int, end: int
    ) -> dict[str, Any]:
        """Fetch one instance's time series in [start, end], following nextPageParams.

        Returns a dict with ``dataPoints`` (column names), ``time`` (epoch ms,
        ascending), and ``values`` (rows aligned to time and dataPoints).
        """
        path = f"/device/devices/{device_id}/devicedatasources/{hds_id}/instances/{instance_id}/data"
        params: dict[str, Any] = {"start": start, "end": end}
        datapoints: list[str] | None = None
        times: list[int] = []
        values: list[list[Any]] = []
        for _ in range(_MAX_DATA_PAGES):
            body = self._client.get(path, params)
            if datapoints is None:
                datapoints = body.get("dataPoints")
            times.extend(body.get("time") or [])
            values.extend(body.get("values") or [])
            next_params = body.get("nextPageParams")
            if not next_params:
                break
            params = dict(parse_qsl(next_params))
        # LogicMonitor returns newest-first; sort ascending and keep rows aligned.
        order = sorted(range(len(times)), key=lambda i: times[i])
        return {
            "dataPoints": datapoints or [],
            "time": [times[i] for i in order],
            "values": [values[i] for i in order],
        }

    # -- normalization ----------------------------------------------------

    @staticmethod
    def iter_canonical_rows(
        device: dict[str, Any],
        datasource_name: str,
        instance: dict[str, Any],
        data: dict[str, Any],
    ) -> Iterator[dict[str, Any]]:
        """Yield canonical rows for one instance's time series, dropping ``No Data`` samples."""
        resource_id = device.get("displayName") or device.get("name")
        host_name = device.get("name")
        instance_name = instance.get("name") or instance.get("displayName") or ""
        datapoints: list[str] = data.get("dataPoints") or []
        for ts_ms, row in zip(data.get("time") or [], data.get("values") or []):
            ts = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).isoformat()
            for col, raw in zip(datapoints, row):
                value = _to_float(raw)
                if value is None:
                    continue
                yield {
                    "resource_id": resource_id,
                    "host_name": host_name,
                    "metric_name": col,
                    "timestamp": ts,
                    "value": value,
                    "datasource_instance": instance_name,
                    "datasource_name": datasource_name,
                }

    # -- orchestration ----------------------------------------------------

    def collect_rows(
        self,
        *,
        start: int,
        end: int,
        ids: Sequence[int] | None = None,
        group_id: int | None = None,
        name_filter: str | None = None,
        datasource_filters: Sequence[str] | None = None,
        max_instances_per_datasource: int | None = None,
    ) -> list[dict[str, Any]]:
        """Walk the target devices and return all canonical rows in [start, end]."""
        devices = self.list_devices(ids=ids, group_id=group_id, name_filter=name_filter)
        logger.info("LM export: %d device(s) selected", len(devices))
        rows: list[dict[str, Any]] = []
        for device in devices:
            device_id = device["id"]
            for dds in self.list_datasources(device_id, name_filters=datasource_filters):
                hds_id, ds_name = dds["id"], dds.get("dataSourceName") or str(dds["id"])
                instances = self.list_instances(device_id, hds_id)
                if max_instances_per_datasource is not None:
                    instances = instances[:max_instances_per_datasource]
                for instance in instances:
                    data = self.fetch_data(device_id, hds_id, instance["id"], start, end)
                    rows.extend(self.iter_canonical_rows(device, ds_name, instance, data))
            logger.info(
                "LM export: device %s (%s) -> %d rows so far",
                device_id,
                device.get("displayName"),
                len(rows),
            )
        return rows

    def export(self, *, output: str, **kwargs: Any) -> dict[str, Any]:
        """Collect rows and write them to ``output`` (``.parquet`` or ``.csv``).

        Returns a small stats dict (row/resource/metric counts and the path).
        """
        rows = self.collect_rows(**kwargs)
        write_canonical(rows, output)
        resources = {r["resource_id"] for r in rows}
        metrics = {r["metric_name"] for r in rows}
        stats = {
            "output": output,
            "rows": len(rows),
            "resources": len(resources),
            "metrics": len(metrics),
        }
        logger.info("LM export complete: %s", stats)
        return stats


def _to_float(value: Any) -> float | None:
    """Coerce a LogicMonitor sample to float, mapping ``No Data``/blank/non-numeric to None."""
    if value is None or value == NO_DATA or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def write_canonical(rows: Iterable[dict[str, Any]], output: str) -> None:
    """Write canonical rows to Parquet or CSV via DuckDB (no pyarrow dependency)."""
    import duckdb
    import pandas as pd

    df = pd.DataFrame(list(rows), columns=METRICS_COLUMNS)
    fmt = "csv" if output.lower().endswith(".csv") else "parquet"
    con = duckdb.connect()
    try:
        con.register("rows", df)
        if fmt == "csv":
            con.execute(f"COPY rows TO '{output}' (FORMAT CSV, HEADER)")
        else:
            con.execute(f"COPY rows TO '{output}' (FORMAT PARQUET)")
    finally:
        con.close()
