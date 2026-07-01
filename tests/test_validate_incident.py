# Description: Tests for the incident-validation harness (scripts/validate_incident.py).
# Description: Injects precursors/anomalies into synthetic captures and checks lead time and leakage.

"""Deterministic, mock-free tests for the incident-validation harness.

Each test runs a tiny keeper X-DEC (the shared ``keeper_path`` fixture) over
synthetic captures written as CSV (read through the real object-store path), and
exercises the harness end to end: genuine early warning from a precursor (positive
lead time), a step at onset that is not credited as early, out-of-sample
false-positive rate, multi-resource attribution, the reference-threshold path, and
the healthy-only threshold leakage guard. The capture generators and the keeper
fixture are shared through ``conftest.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import validate_incident as vi
from synth import PROFILE as _PROFILE
from synth import gen_capture as _gen_capture
from synth import make_incident as _incident
from synth import write_csv as _write_csv
from synth import write_labels as _write_labels


def test_precursor_yields_positive_lead_time(keeper_path: str, tmp_path: Path) -> None:
    """A gradual precursor before onset is detected with a positive, bounded lead time.

    The threshold uses a separate clean reference (a precursor is a pre-onset
    anomaly, so the temporal split would treat it as healthy). The detection must
    fire before onset and within the look-back horizon.
    """
    ref_df, _ = _gen_capture("ref-node", 400, seed=10)
    ref_csv = _write_csv(ref_df, tmp_path / "reference.csv")

    capture_df, ts = _gen_capture("node-a", 700, seed=2, ramp=(500, 700, 40.0))
    capture_csv = _write_csv(capture_df, tmp_path / "incident.csv")
    labels = _write_labels(
        tmp_path / "labels.json", [_incident("node-a", "cpu_ramp", ts[600], ts[699])]
    )

    summary = vi.analyze(
        keeper_path, capture_csv, labels, _PROFILE, reference=ref_csv, max_leadtime_seconds=7200.0
    )

    assert summary["threshold_source"] == "reference"
    incident = summary["incidents"][0]
    assert incident["detected"] is True
    # The alarm begins before onset (genuine early warning), within the horizon.
    assert incident["lead_time_seconds"] > 0
    assert incident["lead_time_seconds"] <= 7200.0


def test_step_at_onset_is_not_credited_as_early(keeper_path: str, tmp_path: Path) -> None:
    """A step exactly at onset is detected, but never with a positive lead time."""
    capture_df, ts = _gen_capture("node-a", 700, seed=2, spike=(600, 700, 40.0))
    capture_csv = _write_csv(capture_df, tmp_path / "step.csv")
    labels = _write_labels(
        tmp_path / "labels.json", [_incident("node-a", "cpu_step", ts[600], ts[699])]
    )

    summary = vi.analyze(keeper_path, capture_csv, labels, _PROFILE)

    assert summary["threshold_source"] == "healthy_split"
    incident = summary["incidents"][0]
    assert incident["detected"] is True
    # No precursor, so the model cannot honestly claim it saw this coming.
    assert incident["lead_time_seconds"] <= 0


def test_recovered_blip_is_not_credited_as_early_warning() -> None:
    """A recovered pre-onset blip plus a post-onset alarm must not yield positive lead.

    Exercises evaluate_incidents directly: a sustained blip at 00:10-00:13 that
    recovers, and the real alarm only from 02:01 (after a 02:00 onset). The blip
    is causally unrelated and must not be credited as early warning.
    """
    onset = pd.Timestamp("2026-01-01T02:00:00Z")
    end_times = pd.DatetimeIndex(pd.date_range("2026-01-01T00:00:00Z", periods=131, freq="1min"))
    resource_ids = np.array(["node-a"] * len(end_times), dtype=object)
    errors = np.zeros(len(end_times))
    errors[10:14] = 10.0  # recovered blip 00:10-00:13
    errors[121:126] = 10.0  # real alarm 02:01-02:05
    incidents = [vi.Incident("node-a", "cpu", onset, pd.Timestamp("2026-01-01T02:10:00Z"))]

    result = vi.evaluate_incidents(
        errors,
        resource_ids,
        end_times,
        incidents,
        threshold=1.0,
        sustain=3,
        max_leadtime=pd.Timedelta(hours=2),
    )[0]

    assert result["detected"] is True
    assert result["lead_time_seconds"] <= 0  # only the post-onset alarm counts


def test_run_spanning_onset_yields_its_start_as_lead() -> None:
    """A sustained run that spans onset is credited from its start (positive lead)."""
    onset = pd.Timestamp("2026-01-01T02:00:00Z")
    end_times = pd.DatetimeIndex(pd.date_range("2026-01-01T00:00:00Z", periods=131, freq="1min"))
    resource_ids = np.array(["node-a"] * len(end_times), dtype=object)
    errors = np.zeros(len(end_times))
    errors[115:126] = 10.0  # 01:55-02:05 spans onset
    incidents = [vi.Incident("node-a", "cpu", onset, pd.Timestamp("2026-01-01T02:10:00Z"))]

    result = vi.evaluate_incidents(
        errors,
        resource_ids,
        end_times,
        incidents,
        threshold=1.0,
        sustain=3,
        max_leadtime=pd.Timedelta(hours=2),
    )[0]

    assert result["detected"] is True
    assert result["lead_time_seconds"] == 300.0  # onset 02:00 - run start 01:55


def test_healthy_capture_has_no_detection_and_out_of_sample_fpr(
    keeper_path: str, tmp_path: Path
) -> None:
    """An all-healthy capture yields no detection and a measured out-of-sample FPR."""
    capture_df, ts = _gen_capture("node-a", 560, seed=3)
    capture_csv = _write_csv(capture_df, tmp_path / "healthy.csv")
    labels = _write_labels(
        tmp_path / "labels.json", [_incident("node-a", "none", ts[500], ts[559])]
    )

    summary = vi.analyze(keeper_path, capture_csv, labels, _PROFILE)

    # FPR is measured on a held-out healthy set disjoint from the fit set.
    assert summary["n_threshold_windows"] > 0
    assert summary["n_eval_windows"] > 0
    assert summary["healthy_fpr"] is not None
    assert 0.0 <= summary["healthy_fpr"] <= 0.5
    incident = summary["incidents"][0]
    assert incident["detected"] is False
    assert incident["lead_time_seconds"] is None


def test_huge_anomaly_does_not_inflate_threshold(keeper_path: str, tmp_path: Path) -> None:
    """The threshold is healthy-only: a huge spike cannot raise it, yet is detected."""
    small_df, ts = _gen_capture("node-a", 560, seed=2, spike=(500, 560, 40.0))
    huge_df, _ = _gen_capture("node-a", 560, seed=2, spike=(500, 560, 400.0))
    small_csv = _write_csv(small_df, tmp_path / "small.csv")
    huge_csv = _write_csv(huge_df, tmp_path / "huge.csv")
    labels = _write_labels(
        tmp_path / "labels.json", [_incident("node-a", "cpu_spike", ts[500], ts[559])]
    )

    small = vi.analyze(keeper_path, small_csv, labels, _PROFILE)
    huge = vi.analyze(keeper_path, huge_csv, labels, _PROFILE)

    # Identical healthy prefix (same seed) => identical healthy-only threshold,
    # independent of the incident magnitude. No leakage from the spike.
    assert small["threshold"] == pytest.approx(huge["threshold"], abs=1e-9)
    assert (
        huge["incidents"][0]["max_error_in_window"]
        > small["incidents"][0]["max_error_in_window"]
    )
    assert huge["incidents"][0]["detected"] is True


def test_multi_resource_detection_is_attributed_correctly(
    keeper_path: str, tmp_path: Path
) -> None:
    """With two resources, a spike on one is attributed to it; the healthy one is clean."""
    node_a, ts = _gen_capture("node-a", 560, seed=2, spike=(500, 560, 40.0))
    node_b, _ = _gen_capture("node-b", 560, seed=7)  # all healthy
    capture_csv = _write_csv(pd.concat([node_a, node_b], ignore_index=True), tmp_path / "two.csv")
    labels = _write_labels(
        tmp_path / "labels.json",
        [
            _incident("node-a", "cpu_spike", ts[500], ts[559]),
            _incident("node-b", "none", ts[500], ts[559]),
        ],
    )

    summary = vi.analyze(keeper_path, capture_csv, labels, _PROFILE)
    by_resource = {inc["resource_id"]: inc for inc in summary["incidents"]}

    assert by_resource["node-a"]["detected"] is True
    assert by_resource["node-b"]["detected"] is False


def test_multiple_incidents_evaluated_independently(keeper_path: str, tmp_path: Path) -> None:
    """Two spikes on one resource are each evaluated and detected independently."""
    first, ts = _gen_capture("node-a", 700, seed=2, spike=(300, 340, 40.0))
    second, _ = _gen_capture("node-a", 700, seed=2, spike=(600, 640, 40.0))
    # Merge the two cpu spikes by taking the elementwise max across the two series.
    merged = pd.concat([first, second], ignore_index=True)
    merged = merged.groupby(["resource_id", "metric_name", "timestamp"], as_index=False)[
        "value"
    ].max()
    capture_csv = _write_csv(merged, tmp_path / "two_incidents.csv")
    labels = _write_labels(
        tmp_path / "labels.json",
        [
            _incident("node-a", "spike_1", ts[300], ts[339]),
            _incident("node-a", "spike_2", ts[600], ts[639]),
        ],
    )

    summary = vi.analyze(keeper_path, capture_csv, labels, _PROFILE)
    assert len(summary["incidents"]) == 2
    assert all(inc["detected"] for inc in summary["incidents"])
    d0 = pd.to_datetime(summary["incidents"][0]["first_detection_time"])
    d1 = pd.to_datetime(summary["incidents"][1]["first_detection_time"])
    # Each incident is attributed to its own onset, not both to the first spike.
    assert d0 < d1
    assert d1 > ts[480]  # spike_2 detected in its own horizon, past spike_1's region


def test_incident_outside_capture_is_not_detected(keeper_path: str, tmp_path: Path) -> None:
    """An incident whose window falls entirely after the capture is reported undetected."""
    capture_df, ts = _gen_capture("node-a", 560, seed=2, spike=(500, 560, 40.0))
    capture_csv = _write_csv(capture_df, tmp_path / "incident.csv")
    far_start = ts[-1] + pd.Timedelta(days=1)
    far_end = far_start + pd.Timedelta(hours=1)
    labels = _write_labels(
        tmp_path / "labels.json", [_incident("node-a", "later", far_start, far_end)]
    )

    summary = vi.analyze(keeper_path, capture_csv, labels, _PROFILE)
    incident = summary["incidents"][0]
    assert incident["detected"] is False
    assert incident["lead_time_seconds"] is None


def test_zero_healthy_windows_errors_clearly(keeper_path: str, tmp_path: Path) -> None:
    """With every window at or after the incident start, the harness errors clearly."""
    capture_df, ts = _gen_capture("node-a", 560, seed=2, spike=(0, 60, 40.0))
    capture_csv = _write_csv(capture_df, tmp_path / "all_incident.csv")
    labels = _write_labels(
        tmp_path / "labels.json", [_incident("node-a", "cpu_spike", ts[0], ts[559])]
    )

    with pytest.raises(ValueError, match="No healthy windows"):
        vi.analyze(keeper_path, capture_csv, labels, _PROFILE)
