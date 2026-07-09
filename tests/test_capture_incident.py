# Description: Tests for the capture_incident wrapper (window math, labels, validate, export stage).
# Description: Exercises the real analyze path on synthetic captures; the LM REST API is respx-mocked.

"""Tests for :mod:`scripts.capture_incident`.

The --data path runs the real incident-validation analysis end to end on
synthetic captures (the proven ramp parameters from test_validate_incident).
The export stage is exercised through a respx-mocked LogicMonitor REST chain,
asserting the computed epoch window reaches the data request. No Scry internals
are mocked.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import capture_incident as ci
import httpx
import numpy as np
import pandas as pd
import pytest
import respx
from synth import CAT, SERIES, gen_capture, write_csv

import scry.data.feature_engineering as _fe

BASE_URL = "https://acme.logicmonitor.com/santaba/rest"


@pytest.fixture(autouse=True)
def _restore_active_profile():
    """analyze() sets the global active profile as part of its contract; snapshot and
    restore it so test modules that run after this one see the profile they expect."""
    prev = _fe._active_config
    yield
    _fe._active_config = prev


# -- window math and timestamp parsing -------------------------------------------


def test_export_window_brackets_lead_in_and_tail() -> None:
    onset = pd.Timestamp("2026-07-15T14:00:00Z")
    incident_end = pd.Timestamp("2026-07-15T14:45:00Z")
    start, end = ci.export_window(onset, incident_end, lead_in_hours=6.0, tail_minutes=30.0)
    assert start == int(pd.Timestamp("2026-07-15T08:00:00Z").timestamp())
    assert end == int(pd.Timestamp("2026-07-15T15:15:00Z").timestamp())


def test_parse_utc_accepts_iso_and_rejects_garbage() -> None:
    ts = ci.parse_utc("2026-07-15T14:00:00Z")
    assert ts.tzinfo is not None
    assert ts.isoformat() == "2026-07-15T14:00:00+00:00"
    with pytest.raises(argparse.ArgumentTypeError):
        ci.parse_utc("not-a-timestamp")


# -- argument validation ----------------------------------------------------------


def _base_args(tmp_path: Path, **overrides: str) -> list[str]:
    args = {
        "--onset": "2026-01-01T10:00:00Z",
        "--incident-end": "2026-01-01T11:39:00Z",
        "--resource-id": "node-a",
        "--type": "cpu_ramp",
        "--model": "unused.pt",
        "--out-dir": str(tmp_path / "out"),
    }
    args.update(overrides)
    return [item for pair in args.items() for item in pair]


def test_onset_after_end_is_rejected(tmp_path: Path) -> None:
    argv = _base_args(
        tmp_path, **{"--onset": "2026-01-01T12:00:00Z", "--incident-end": "2026-01-01T11:00:00Z"}
    ) + ["--data", "whatever.csv"]
    assert ci.main(argv) == 2


def test_data_and_export_target_together_are_rejected(tmp_path: Path) -> None:
    argv = _base_args(tmp_path) + ["--data", "whatever.csv", "--device-id", "42"]
    assert ci.main(argv) == 2


def test_no_data_and_no_target_is_rejected(tmp_path: Path) -> None:
    argv = _base_args(tmp_path)
    assert ci.main(argv) == 2


def test_unparseable_onset_exits_via_argparse(tmp_path: Path) -> None:
    argv = _base_args(tmp_path, **{"--onset": "yesterday-ish"}) + ["--data", "whatever.csv"]
    with pytest.raises(SystemExit) as exc_info:
        ci.main(argv)
    assert exc_info.value.code == 2


# -- the --data path: real analyze on synthetic captures ---------------------------


def test_ramp_capture_is_detected_with_positive_lead(
    keeper_path: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The proven gradual-precursor scenario reports a positive lead time end to end."""
    ref_df, _ = gen_capture("ref-node", 400, seed=10)
    reference = write_csv(ref_df, tmp_path / "reference.csv")
    capture_df, ts = gen_capture("node-a", 700, seed=2, ramp=(500, 700, 40.0))
    data = write_csv(capture_df, tmp_path / "capture.csv")

    out_dir = tmp_path / "out"
    argv = _base_args(
        tmp_path,
        **{"--onset": ts[600].isoformat(), "--incident-end": ts[699].isoformat()},
    ) + ["--data", data, "--model", keeper_path, "--reference", reference]
    assert ci.main(argv) == 0

    summary = json.loads((out_dir / "summary.json").read_text())
    (incident,) = summary["incidents"]
    assert incident["detected"] is True
    assert incident["lead_time_seconds"] > 0

    out = capsys.readouterr().out
    assert "DETECTED cpu_ramp on node-a" in out
    assert "lead_time_seconds=" in out


def test_healthy_capture_reports_not_detected_with_exit_zero(
    keeper_path: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """No detection is a valid outcome, reported plainly, not an error."""
    capture_df, ts = gen_capture("node-a", 560, seed=3)
    data = write_csv(capture_df, tmp_path / "capture.csv")

    out_dir = tmp_path / "out"
    argv = _base_args(
        tmp_path,
        **{"--onset": ts[500].isoformat(), "--incident-end": ts[559].isoformat()},
    ) + ["--data", data, "--model", keeper_path]
    assert ci.main(argv) == 0

    summary = json.loads((out_dir / "summary.json").read_text())
    (incident,) = summary["incidents"]
    assert incident["detected"] is False
    assert "NOT DETECTED cpu_ramp on node-a" in capsys.readouterr().out


def test_labels_sidecar_has_exactly_the_harness_schema(keeper_path: str, tmp_path: Path) -> None:
    """The labels file carries exactly {resource_id, type, start, end} with start = onset."""
    capture_df, ts = gen_capture("node-a", 560, seed=3)
    data = write_csv(capture_df, tmp_path / "capture.csv")

    out_dir = tmp_path / "out"
    argv = _base_args(
        tmp_path,
        **{"--onset": ts[500].isoformat(), "--incident-end": ts[559].isoformat()},
    ) + ["--data", data, "--model", keeper_path]
    assert ci.main(argv) == 0

    (label,) = json.loads((out_dir / "labels.json").read_text())
    assert set(label) == {"resource_id", "type", "start", "end"}
    assert label["resource_id"] == "node-a"
    assert label["type"] == "cpu_ramp"
    assert pd.Timestamp(label["start"]) == ts[500]
    assert pd.Timestamp(label["end"]) == ts[559]


def test_capture_too_short_for_windows_exits_one(
    keeper_path: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A capture below seq_len produces no windows: a clear error, not a fake result."""
    capture_df, ts = gen_capture("node-a", 10, seed=4)
    data = write_csv(capture_df, tmp_path / "capture.csv")

    argv = _base_args(
        tmp_path,
        **{"--onset": ts[5].isoformat(), "--incident-end": ts[9].isoformat()},
    ) + ["--data", data, "--model", keeper_path]
    assert ci.main(argv) == 1
    assert "no windows" in capsys.readouterr().err


def test_mismatched_resource_id_is_an_error_not_a_miss(
    keeper_path: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A label that matches nothing in the capture must not read as NOT DETECTED.

    The exporter keys resources by device displayName; a paste typo or the
    FQDN-for-displayName mistake would otherwise score zero windows and report a
    clean miss on a one-shot induced incident.
    """
    capture_df, ts = gen_capture("node-a", 560, seed=3)
    data = write_csv(capture_df, tmp_path / "capture.csv")

    argv = _base_args(
        tmp_path,
        **{
            "--onset": ts[500].isoformat(),
            "--incident-end": ts[559].isoformat(),
            "--resource-id": "node-a.internal",
        },
    ) + ["--data", data, "--model", keeper_path]
    assert ci.main(argv) == 1
    err = capsys.readouterr().err
    assert "matches no resource" in err
    assert "node-a" in err  # the available ids are named


def test_capture_without_numerical_coverage_is_an_error(
    keeper_path: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A capture whose numerical collection died must not read as NOT DETECTED.

    An induced failure can take out the numeric datapoints ('No Data' rows are
    dropped by the exporter) while the availability flags keep flowing; the
    reconstruction signal then has nothing to score.
    """
    capture_df, ts = gen_capture("node-a", 560, seed=3)
    cat_only = capture_df[capture_df["metric_name"].isin(CAT)]
    data = write_csv(cat_only, tmp_path / "capture.csv")

    argv = _base_args(
        tmp_path,
        **{"--onset": ts[500].isoformat(), "--incident-end": ts[559].isoformat()},
    ) + ["--data", data, "--model", keeper_path]
    assert ci.main(argv) == 1
    assert "numerical features" in capsys.readouterr().err


def test_incident_outside_the_capture_is_an_error_not_a_miss(
    keeper_path: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An incident span with no scored windows must not read as NOT DETECTED."""
    capture_df, _ = gen_capture("node-a", 560, seed=3)
    data = write_csv(capture_df, tmp_path / "capture.csv")

    argv = _base_args(
        tmp_path,
        **{"--onset": "2026-01-02T00:00:00Z", "--incident-end": "2026-01-02T01:00:00Z"},
    ) + ["--data", data, "--model", keeper_path]
    assert ci.main(argv) == 1
    assert "no scored windows" in capsys.readouterr().err


def test_report_formats_a_late_detection_without_a_double_negative(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A negative lead prints as '3.0 min after onset', not '-3.0 min after onset'."""
    summary = {
        "threshold": 0.1,
        "threshold_source": "healthy_split",
        "threshold_quantile": 0.99,
        "sustain": 3,
        "max_leadtime_seconds": 7200.0,
        "n_windows": 10,
        "n_threshold_windows": 5,
        "n_eval_windows": 3,
        "healthy_fpr": None,
        "incidents": [
            {
                "resource_id": "node-a",
                "type": "cpu",
                "detected": True,
                "first_detection_time": "2026-01-01T02:03:00+00:00",
                "lead_time_seconds": -180.0,
                "max_error_in_window": 5.0,
            }
        ],
    }
    assert ci.report(summary, tmp_path / "summary.json") == 0
    out = capsys.readouterr().out
    assert "lead_time_seconds=-180.0" in out
    assert "(3.0 min after onset)" in out


# -- the export stage: respx-mocked LogicMonitor REST chain ------------------------


def _mock_lm_chain(respx_mock: respx.MockRouter, timestamps: pd.DatetimeIndex) -> respx.Route:
    """Mock the device -> datasource -> instance -> data chain for one device."""
    rng = np.random.default_rng(7)
    n = len(timestamps)
    series = {
        "cpuUsageNanoCores": 1e8 + rng.normal(0, 3e6, n),
        "memoryUsageBytes": 5e8 + rng.normal(0, 5e6, n),
        "fsUsedBytes": 1e9 + rng.normal(0, 8e6, n),
        "ksmMetricsAvailable": np.ones(n),
        "summaryMetricsAvailable": np.ones(n),
    }
    datapoints = list(SERIES) + list(CAT)
    respx_mock.get(f"{BASE_URL}/device/devices/42").mock(
        return_value=httpx.Response(
            200, json={"id": 42, "displayName": "node-a", "name": "node-a.internal"}
        )
    )
    respx_mock.get(f"{BASE_URL}/device/devices/42/devicedatasources").mock(
        return_value=httpx.Response(
            200,
            json={
                "items": [{"id": 7, "dataSourceName": "Kubernetes_KSM_Nodes", "instanceNumber": 1}],
                "total": 1,
            },
        )
    )
    respx_mock.get(f"{BASE_URL}/device/devices/42/devicedatasources/7/instances").mock(
        return_value=httpx.Response(
            200,
            json={"items": [{"id": 9, "name": "kube-node", "displayName": "kube-node"}], "total": 1},
        )
    )
    return respx_mock.get(f"{BASE_URL}/device/devices/42/devicedatasources/7/instances/9/data").mock(
        return_value=httpx.Response(
            200,
            json={
                "dataPoints": datapoints,
                "time": [int(ts.timestamp() * 1000) for ts in timestamps],
                "values": [[float(series[dp][i]) for dp in datapoints] for i in range(n)],
            },
        )
    )


def test_export_stage_pulls_the_computed_window(
    keeper_path: str,
    tmp_path: Path,
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The mocked LM chain receives the bracketed epoch window and the pipeline completes."""
    monkeypatch.setenv("LM_ACCESS_ID", "id123")
    monkeypatch.setenv("LM_ACCESS_KEY", "secretkey")
    monkeypatch.setenv("LM_COMPANY", "acme")

    onset = pd.Timestamp("2026-01-01T03:00:00Z")
    incident_end = pd.Timestamp("2026-01-01T03:30:00Z")
    window_start = pd.Timestamp("2026-01-01T00:00:00Z")  # onset - 3h lead-in
    timestamps = pd.date_range(window_start, periods=240, freq="1min")
    data_route = _mock_lm_chain(respx_mock, timestamps)

    out_dir = tmp_path / "out"
    argv = _base_args(
        tmp_path,
        **{"--onset": onset.isoformat(), "--incident-end": incident_end.isoformat()},
    ) + [
        "--device-id", "42",
        "--lead-in", "3",
        "--tail", "30",
        "--rate-limit-pause", "0",
        "--env-file", str(tmp_path / "missing.env"),
        "--model", keeper_path,
    ]
    assert ci.main(argv) == 0

    params = data_route.calls.last.request.url.params
    expected_start, expected_end = ci.export_window(onset, incident_end, 3.0, 30.0)
    assert int(params["start"]) == expected_start
    assert int(params["end"]) == expected_end
    assert (out_dir / "capture.parquet").exists()
    assert (out_dir / "labels.json").exists()

    # The analysis genuinely engaged the labeled resource: windows were scored
    # inside the incident span, so the exported ids and the label line up.
    summary = json.loads((out_dir / "summary.json").read_text())
    (incident,) = summary["incidents"]
    assert incident["max_error_in_window"] is not None


def test_export_with_no_rows_exits_one(
    tmp_path: Path,
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    keeper_path: str,
) -> None:
    """An empty export is an actionable error before any analysis runs."""
    monkeypatch.setenv("LM_ACCESS_ID", "id123")
    monkeypatch.setenv("LM_ACCESS_KEY", "secretkey")
    monkeypatch.setenv("LM_COMPANY", "acme")
    respx_mock.get(f"{BASE_URL}/device/devices/42").mock(
        return_value=httpx.Response(
            200, json={"id": 42, "displayName": "node-a", "name": "node-a.internal"}
        )
    )
    respx_mock.get(f"{BASE_URL}/device/devices/42/devicedatasources").mock(
        return_value=httpx.Response(200, json={"items": [], "total": 0})
    )

    argv = _base_args(tmp_path) + [
        "--device-id", "42",
        "--rate-limit-pause", "0",
        "--env-file", str(tmp_path / "missing.env"),
        "--model", keeper_path,
    ]
    assert ci.main(argv) == 1
    assert "no rows" in capsys.readouterr().err


def test_missing_credentials_is_a_clear_error(
    keeper_path: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Missing LM credentials fail with the variables named, before any network call."""
    for var in ("LM_ACCESS_ID", "LM_ACCESS_KEY", "LM_COMPANY"):
        monkeypatch.delenv(var, raising=False)

    argv = _base_args(tmp_path) + [
        "--device-id", "42",
        "--env-file", str(tmp_path / "missing.env"),
        "--model", keeper_path,
    ]
    assert ci.main(argv) == 1
    err = capsys.readouterr().err
    assert "LM_ACCESS_ID" in err


def test_missing_model_fails_before_the_export(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A typo'd --model path fails in seconds, not after a rate-limited LM pull."""
    argv = _base_args(tmp_path) + ["--device-id", "42"]
    assert ci.main(argv) == 2
    assert "model checkpoint not found" in capsys.readouterr().err


def test_unknown_profile_fails_before_the_export(
    keeper_path: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An unknown --profile fails in seconds with the available profiles named."""
    argv = _base_args(tmp_path) + [
        "--device-id", "42",
        "--model", keeper_path,
        "--profile", "no_such_profile",
    ]
    assert ci.main(argv) == 2
    assert "no_such_profile" in capsys.readouterr().err


def test_format_is_rejected_with_an_export_target(
    keeper_path: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--format governs --data reads; the export always writes Parquet."""
    argv = _base_args(tmp_path) + ["--device-id", "42", "--model", keeper_path, "--format", "csv"]
    assert ci.main(argv) == 2
    assert "--format" in capsys.readouterr().err
