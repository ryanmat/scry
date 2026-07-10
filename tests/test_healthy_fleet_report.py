# Description: Tests for the per-resource healthy-baseline report (scripts/healthy_fleet_report.py).
# Description: Unit-tests the grouping/quantile/run math and exercises analyze + the CLI end to end.

"""Deterministic, mock-free tests for the healthy fleet report.

The math (resource grouping, own-quantile, window FPR, temporal sustained runs,
per-week normalization, and the pooled global rollup) is checked on hand-built
error arrays through ``compute_report``. The end-to-end path is exercised through
``analyze`` and ``main`` with the shared tiny keeper fixture over synthetic
multi-resource captures read via the real object-store path, verifying the JSON
schema and that a resource's own quantile matches an independent recomputation.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import healthy_fleet_report as fleet
import numpy as np
import pandas as pd
import pytest
import validate_incident as vi
from synth import PROFILE, gen_capture, write_csv

from scry.data.feature_engineering import set_active_profile
from scry.data.fetcher import fetch_full_capture
from scry.model.checkpoint import load_keeper
from scry.model.reconstruction import reconstruction_errors
from scry.utils.config import get_config


def _times(base: str, minutes: list[int]) -> pd.DatetimeIndex:
    """Build a UTC DatetimeIndex at the given minute offsets from ``base``."""
    origin = pd.Timestamp(base)
    return pd.DatetimeIndex([origin + pd.Timedelta(minutes=m) for m in minutes])


def test_compute_report_groups_quantile_fpr_and_runs() -> None:
    """Grouping, own-quantile, window FPR, sustained runs, and the global rollup."""
    resource_ids = np.array(["a"] * 5 + ["b"] * 4, dtype=object)
    end_times = _times("2026-01-01T00:00:00Z", [0, 1, 2, 3, 4] + [0, 1, 2, 3])
    # a: a sustained run of three over threshold 1.0; b: a run of only two.
    errors = np.array([0.0, 0.0, 10.0, 10.0, 10.0, 0.0, 5.0, 5.0, 0.0])

    report = fleet.compute_report(
        errors, resource_ids, end_times, thresholds=[1.0], sustain=3, quantile=0.99
    )
    res = report["resources"]

    assert set(res) == {"a", "b"}
    assert res["a"]["n_windows"] == 5
    assert res["b"]["n_windows"] == 4

    # own quantile matches a direct np.quantile of that resource's errors.
    assert res["a"]["own_q"] == pytest.approx(float(np.quantile(errors[:5], 0.99)))
    assert res["b"]["own_q"] == pytest.approx(float(np.quantile(errors[5:], 0.99)))

    # window FPR: fraction of that resource's windows over the threshold.
    assert res["a"]["thresholds"]["1.0"]["window_fpr"] == pytest.approx(3 / 5)
    assert res["b"]["thresholds"]["1.0"]["window_fpr"] == pytest.approx(2 / 4)

    # sustained runs: a has one run of length 3; b's run of length 2 does not qualify.
    assert res["a"]["thresholds"]["1.0"]["sustained_runs"] == 1
    assert res["b"]["thresholds"]["1.0"]["sustained_runs"] == 0

    # global rollup: pooled window count and FPR, runs summed from per-resource.
    glob = report["global"]
    assert glob["n_windows"] == 9
    assert glob["thresholds"]["1.0"]["window_fpr"] == pytest.approx(5 / 9)
    assert glob["thresholds"]["1.0"]["sustained_runs"] == 1
    assert glob["own_q"] == pytest.approx(float(np.quantile(errors, 0.99)))


def test_runs_per_week_normalizes_by_span() -> None:
    """runs_per_week is 7 * runs / span_days for the resource's own span."""
    resource_ids = np.array(["a"] * 5, dtype=object)
    end_times = _times("2026-01-01T00:00:00Z", [0, 1, 2, 3, 4])  # 4-minute span
    errors = np.array([0.0, 0.0, 10.0, 10.0, 10.0])

    report = fleet.compute_report(
        errors, resource_ids, end_times, thresholds=[1.0], sustain=3, quantile=0.99
    )
    stats = report["resources"]["a"]["thresholds"]["1.0"]

    span_days = 4.0 / 1440.0
    assert report["resources"]["a"]["span_days"] == pytest.approx(span_days)
    assert stats["sustained_runs"] == 1
    assert stats["runs_per_week"] == pytest.approx(7.0 * 1 / span_days)


def test_sustained_runs_use_temporal_order() -> None:
    """Runs are counted in end-time order, not window-emission order."""
    resource_ids = np.array(["a"] * 5, dtype=object)
    # Emission order scatters the over-threshold windows (no run of 3), but sorting
    # by end-time gathers them into one contiguous run of length 3.
    end_times = _times("2026-01-01T00:00:00Z", [0, 3, 1, 4, 2])
    errors = np.array([10.0, 0.0, 10.0, 0.0, 10.0])

    report = fleet.compute_report(
        errors, resource_ids, end_times, thresholds=[1.0], sustain=3, quantile=0.99
    )
    # Emission-order flags [T,F,T,F,T] give zero runs; time-order [T,T,T,F,F] gives one.
    assert report["resources"]["a"]["thresholds"]["1.0"]["sustained_runs"] == 1


def test_single_window_span_zero_guards_runs_per_week() -> None:
    """A single-window resource has span 0 and a None (undefined) per-week rate."""
    resource_ids = np.array(["solo"], dtype=object)
    end_times = _times("2026-01-01T00:00:00Z", [0])
    errors = np.array([5.0])

    report = fleet.compute_report(
        errors, resource_ids, end_times, thresholds=[1.0], sustain=1, quantile=0.99
    )
    stats = report["resources"]["solo"]

    assert stats["span_days"] == 0.0
    assert stats["thresholds"]["1.0"]["sustained_runs"] == 1  # one window over, sustain=1
    assert stats["thresholds"]["1.0"]["runs_per_week"] is None


def test_no_thresholds_reports_baselines_only() -> None:
    """With no thresholds, resources still carry baselines and an empty thresholds map."""
    resource_ids = np.array(["a"] * 3, dtype=object)
    end_times = _times("2026-01-01T00:00:00Z", [0, 1, 2])
    errors = np.array([1.0, 2.0, 3.0])

    report = fleet.compute_report(
        errors, resource_ids, end_times, thresholds=[], sustain=3, quantile=0.5
    )
    assert report["resources"]["a"]["thresholds"] == {}
    assert report["resources"]["a"]["own_q"] == pytest.approx(float(np.quantile(errors, 0.5)))
    assert report["global"]["thresholds"] == {}


def test_analyze_multi_resource_schema_and_own_quantile(keeper_path: str, tmp_path: Path) -> None:
    """End to end: two resources are grouped, the schema holds, own_q matches recomputation."""
    node_a, _ = gen_capture("node-a", 560, seed=2)
    node_b, _ = gen_capture("node-b", 560, seed=7)
    capture_csv = write_csv(pd.concat([node_a, node_b], ignore_index=True), tmp_path / "fleet.csv")

    summary = fleet.analyze(
        keeper_path, capture_csv, PROFILE, thresholds=[0.05], sustain=3, threshold_quantile=0.99
    )

    assert set(summary) == {
        "model",
        "data",
        "profile",
        "sustain",
        "threshold_quantile",
        "generated_from_span",
        "resources",
        "global",
    }
    res = summary["resources"]
    assert set(res) == {"node-a", "node-b"}
    for name in ("node-a", "node-b"):
        entry = res[name]
        assert set(entry) == {"n_windows", "own_q", "span_days", "thresholds"}
        assert entry["n_windows"] > 0
        assert isinstance(entry["own_q"], float)
        assert entry["span_days"] > 0.0
        assert set(entry["thresholds"]["0.05"]) == {
            "window_fpr",
            "sustained_runs",
            "runs_per_week",
        }
    # Grouping conserves windows: the global count is the sum of the resources'.
    assert summary["global"]["n_windows"] == res["node-a"]["n_windows"] + res["node-b"]["n_windows"]

    # node-a's own quantile matches an independent single-resource recomputation.
    keeper = load_keeper(keeper_path)
    set_active_profile(PROFILE)
    seq_len = int(keeper.config["seq_len"])
    step = int(get_config().window_step)
    a_csv = write_csv(node_a, tmp_path / "node_a.csv")
    a_df = asyncio.run(fetch_full_capture(a_csv, profile=PROFILE))
    a_windows = vi._windows_for_keeper(a_df, keeper, seq_len, step)
    a_errors = reconstruction_errors(keeper.model, a_windows.x_num, a_windows.x_cat, keeper.device)
    assert res["node-a"]["own_q"] == pytest.approx(float(np.quantile(a_errors, 0.99)), rel=1e-6)


def test_main_writes_json_report(keeper_path: str, tmp_path: Path) -> None:
    """The CLI writes a JSON report with the documented top-level and nested schema."""
    node_a, _ = gen_capture("node-a", 400, seed=2)
    capture_csv = write_csv(node_a, tmp_path / "cap.csv")
    out = tmp_path / "report.json"

    rc = fleet.main(
        [
            "--model",
            keeper_path,
            "--data",
            capture_csv,
            "--profile",
            PROFILE,
            "--threshold",
            "0.05",
            "--threshold",
            "0.1",
            "--sustain",
            "3",
            "--output",
            str(out),
        ]
    )
    assert rc == 0

    loaded = json.loads(out.read_text())
    assert set(loaded) == {
        "model",
        "data",
        "profile",
        "sustain",
        "threshold_quantile",
        "generated_from_span",
        "resources",
        "global",
    }
    entry = loaded["resources"]["node-a"]
    assert set(entry) == {"n_windows", "own_q", "span_days", "thresholds"}
    assert set(entry["thresholds"]) == {"0.05", "0.1"}
    for key in ("0.05", "0.1"):
        assert set(entry["thresholds"][key]) == {"window_fpr", "sustained_runs", "runs_per_week"}
    assert set(loaded["global"]) == {"n_windows", "own_q", "span_days", "thresholds"}


def test_nonpositive_threshold_is_rejected(keeper_path: str, tmp_path: Path) -> None:
    """A non-positive --threshold is rejected by argument parsing."""
    node_a, _ = gen_capture("node-a", 100, seed=2)
    capture_csv = write_csv(node_a, tmp_path / "cap.csv")
    with pytest.raises(SystemExit):
        fleet.main(
            [
                "--model",
                keeper_path,
                "--data",
                capture_csv,
                "--profile",
                PROFILE,
                "--threshold",
                "0",
            ]
        )
