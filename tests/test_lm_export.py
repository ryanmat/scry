# Description: Tests for the LogicMonitor LMv1 exporter (auth, paging, normalization, write).
# Description: Uses respx to mock the LM REST API; no real network calls are made.

"""Unit tests for :mod:`scry.data.sources.lm_export`.

The LogicMonitor REST API is mocked end-to-end with respx, so these tests never
touch the network. They cover credential loading, the LMv1 signing scheme, the
two pagination styles, time-series normalization, and the DuckDB Parquet writer.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
from pathlib import Path

import httpx
import pytest
import respx

import scry.data.sources.lm_export as lm_export
from scry.data.sources.base import METRICS_COLUMNS
from scry.data.sources.lm_export import (
    LMCredentials,
    LMRestClient,
    LogicMonitorExporter,
    write_canonical,
)

BASE_URL = "https://acme.logicmonitor.com/santaba/rest"


def _creds() -> LMCredentials:
    return LMCredentials(access_id="id123", access_key="secretkey", company="acme")


def _url(path: str) -> str:
    return f"{BASE_URL}{path}"


def _lmv1_signature(access_key: str, verb: str, epoch: str, body: str, resource_path: str) -> str:
    """Recompute an LMv1 signature: base64(hex(HMAC-SHA256(key, VERB+epoch+body+path)))."""
    message = f"{verb}{epoch}{body}{resource_path}"
    digest = hmac.new(access_key.encode("utf-8"), message.encode("utf-8"), hashlib.sha256).hexdigest()
    return base64.b64encode(digest.encode("utf-8")).decode("utf-8")


# -- LMCredentials.from_env -----------------------------------------------------


def test_from_env_builds_from_three_vars() -> None:
    env = {"LM_ACCESS_ID": "id123", "LM_ACCESS_KEY": "key456", "LM_COMPANY": "acme"}
    creds = LMCredentials.from_env(env)
    assert creds.access_id == "id123"
    assert creds.access_key == "key456"
    assert creds.company == "acme"


def test_from_env_missing_vars_raises_listing_them() -> None:
    env = {"LM_ACCESS_ID": "id123"}
    with pytest.raises(ValueError) as exc_info:
        LMCredentials.from_env(env)
    message = str(exc_info.value)
    assert "LM_ACCESS_KEY" in message
    assert "LM_COMPANY" in message
    assert "LM_ACCESS_ID" not in message


# -- LMv1 auth ------------------------------------------------------------------


def test_lmv1_auth_header_shape_and_excludes_query(respx_mock: respx.MockRouter) -> None:
    route = respx_mock.get(_url("/device/devices")).mock(
        return_value=httpx.Response(200, json={"items": [], "total": 0})
    )
    with LMRestClient(_creds(), rate_limit_pause=0.0) as client:
        client.get("/device/devices", {"size": 50, "offset": 0, "filter": 'displayName~"web"'})

    auth = route.calls.last.request.headers["Authorization"]
    scheme, payload = auth.split(" ", 1)
    access_id, signature, epoch = payload.split(":")
    assert scheme == "LMv1"
    assert access_id == "id123"

    # The signed resourcePath is the bare path; the query string is not signed.
    assert signature == _lmv1_signature("secretkey", "GET", epoch, "", "/device/devices")
    signed_with_query = _lmv1_signature(
        "secretkey", "GET", epoch, "", '/device/devices?size=50&offset=0&filter=displayName~"web"'
    )
    assert signature != signed_with_query


# -- get_paginated --------------------------------------------------------------


def test_get_paginated_walks_until_short_page(respx_mock: respx.MockRouter) -> None:
    route = respx_mock.get(_url("/device/devices")).mock(
        side_effect=[
            httpx.Response(200, json={"items": [{"id": 1}, {"id": 2}]}),
            httpx.Response(200, json={"items": [{"id": 3}]}),
        ]
    )
    with LMRestClient(_creds(), rate_limit_pause=0.0) as client:
        items = list(client.get_paginated("/device/devices", page_size=2))

    assert [item["id"] for item in items] == [1, 2, 3]
    assert route.call_count == 2
    first, second = route.calls[0].request, route.calls[1].request
    assert first.url.params["size"] == "2"
    assert first.url.params["offset"] == "0"
    assert second.url.params["offset"] == "2"


# -- fetch_data -----------------------------------------------------------------


def test_fetch_data_follows_pages_and_sorts_ascending(respx_mock: respx.MockRouter) -> None:
    data_path = "/device/devices/10/devicedatasources/20/instances/30/data"
    route = respx_mock.get(_url(data_path)).mock(
        side_effect=[
            # API returns newest-first within a page; nextPageParams chains to the next.
            httpx.Response(
                200,
                json={
                    "dataSourceName": "X",
                    "dataPoints": ["A", "B"],
                    "values": [[1.0, "No Data"], [2.0, 3.0]],
                    "time": [1782703680000, 1782703620000],
                    "nextPageParams": "start=1782703000&end=1782704000&offset=2",
                },
            ),
            httpx.Response(
                200,
                json={
                    "dataSourceName": "X",
                    "dataPoints": ["A", "B"],
                    "values": [[4.0, 5.0]],
                    "time": [1782703560000],
                    "nextPageParams": "",
                },
            ),
        ]
    )
    with LMRestClient(_creds(), rate_limit_pause=0.0) as client:
        data = LogicMonitorExporter(client).fetch_data(10, 20, 30, start=1782703000, end=1782704000)

    assert route.call_count == 2
    assert data["dataPoints"] == ["A", "B"]
    # Merged across pages and sorted ascending by time.
    assert data["time"] == [1782703560000, 1782703620000, 1782703680000]
    # Rows stay aligned to their timestamps: the "No Data" row rode along with ts 680.
    assert data["values"] == [[4.0, 5.0], [2.0, 3.0], [1.0, "No Data"]]


# -- iter_canonical_rows --------------------------------------------------------


def test_iter_canonical_rows_maps_fields_and_drops_non_numeric() -> None:
    device = {"id": 10, "displayName": "Web Server 01", "name": "web01.internal"}
    instance = {"id": 30, "name": "eth0", "displayName": "Ethernet 0"}
    data = {
        "dataPoints": ["A", "B", "C"],
        "time": [1782703620000, 1782703680000],
        "values": [[2.0, 3.0, None], [1.0, "No Data", "oops"]],
    }

    rows = list(LogicMonitorExporter.iter_canonical_rows(device, "Interfaces", instance, data))

    # 2 timestamps x 3 datapoints = 6 candidates; None, "No Data", "oops" drop -> 3 rows.
    assert len(rows) == 3
    for row in rows:
        assert set(row.keys()) == set(METRICS_COLUMNS)
        assert row["resource_id"] == "Web Server 01"  # device displayName
        assert row["host_name"] == "web01.internal"  # device name
        assert row["datasource_instance"] == "eth0"  # instance name
        assert row["datasource_name"] == "Interfaces"
        assert isinstance(row["value"], float)

    # No row survives for the all-bad datapoint C; B keeps only its numeric sample.
    assert all(row["metric_name"] != "C" for row in rows)
    b_rows = [row["value"] for row in rows if row["metric_name"] == "B"]
    assert b_rows == [3.0]


# -- write_canonical ------------------------------------------------------------


def test_write_canonical_parquet_roundtrips_with_double_value(tmp_path: Path) -> None:
    import duckdb

    rows = [
        {
            "resource_id": "Web Server 01",
            "host_name": "web01.internal",
            "metric_name": "A",
            "timestamp": "2026-06-29T00:00:00+00:00",
            "value": 2.0,
            "datasource_instance": "eth0",
            "datasource_name": "Interfaces",
        },
        {
            "resource_id": "Web Server 01",
            "host_name": "web01.internal",
            "metric_name": "A",
            "timestamp": "2026-06-29T00:01:00+00:00",
            "value": 1.5,
            "datasource_instance": "eth0",
            "datasource_name": "Interfaces",
        },
    ]
    out = tmp_path / "metrics.parquet"
    write_canonical(rows, str(out))
    assert out.is_file()

    con = duckdb.connect()
    try:
        described = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{out}')").fetchall()
        schema = {name: dtype for name, dtype, *_ in described}
        values = con.execute(
            f"SELECT value FROM read_parquet('{out}') ORDER BY timestamp"
        ).fetchall()
    finally:
        con.close()

    assert list(schema.keys()) == METRICS_COLUMNS
    assert schema["value"] == "DOUBLE"
    assert [row[0] for row in values] == [2.0, 1.5]


# -- adaptive rate-limit throttle (TASK 1 hardening) ----------------------------


@pytest.mark.parametrize(
    ("headers", "pause", "expected_sleep"),
    [
        # remaining>0, window>0: spread the budget over the window.
        ({"X-Rate-Limit-Remaining": "120", "X-Rate-Limit-Window": "60"}, 0.2, 0.5),
        # the fixed pause is a floor when the spread is smaller.
        ({"X-Rate-Limit-Remaining": "600", "X-Rate-Limit-Window": "60"}, 0.2, 0.2),
        # remaining==0: wait out the whole window.
        ({"X-Rate-Limit-Remaining": "0", "X-Rate-Limit-Window": "60"}, 0.2, 60.0),
        # missing headers: fall back to the fixed pause.
        ({}, 0.2, 0.2),
        # unparseable headers: fall back to the fixed pause.
        ({"X-Rate-Limit-Remaining": "abc", "X-Rate-Limit-Window": "xyz"}, 0.2, 0.2),
    ],
)
def test_adaptive_throttle_sleep_duration(
    headers: dict[str, str],
    pause: float,
    expected_sleep: float,
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slept: list[float] = []
    monkeypatch.setattr(lm_export.time, "sleep", slept.append)
    respx_mock.get(_url("/device/devices")).mock(
        return_value=httpx.Response(200, json={"items": []}, headers=headers)
    )
    with LMRestClient(_creds(), rate_limit_pause=pause) as client:
        client.get("/device/devices")

    assert slept == [pytest.approx(expected_sleep)]


def test_zero_pause_with_missing_headers_does_not_sleep(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    slept: list[float] = []
    monkeypatch.setattr(lm_export.time, "sleep", slept.append)
    respx_mock.get(_url("/device/devices")).mock(
        return_value=httpx.Response(200, json={"items": []})
    )
    with LMRestClient(_creds(), rate_limit_pause=0.0) as client:
        client.get("/device/devices")

    assert slept == []
