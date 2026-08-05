# Description: Shared synthetic-capture generators and constants for scry tests.
# Description: Imported by conftest fixtures and by the harness, endpoint, and bake test modules.

"""Deterministic synthetic captures in the canonical long format.

A small subset of the aro_node profile: three numerical series with sinusoidal
baselines plus noise, and two always-available categorical flags. Optional spike
(step) or ramp (gradual precursor) injections drive the reconstruction error up
for anomaly tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scry.eval.labels import LabelCase, LabelSet, dump_labels

# A subset of the aro_node profile features; the synthetic capture supplies exactly these.
SERIES = ("cpuUsageNanoCores", "memoryUsageBytes", "fsUsedBytes")
CAT = ("ksmMetricsAvailable", "summaryMetricsAvailable")
PROFILE = "aro_node"
SEQ_LEN = 30


def gen_capture(
    resource: str,
    n: int,
    seed: int,
    *,
    spike: tuple[int, int, float] | None = None,
    ramp: tuple[int, int, float] | None = None,
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Generate a synthetic long-format capture for one resource.

    Args:
        resource: Resource id.
        n: Number of timesteps (1-minute cadence).
        seed: RNG seed for reproducibility.
        spike: Optional (lo, hi, multiplier) step injected into cpuUsageNanoCores.
        ramp: Optional (lo, hi, peak) linear precursor scaling cpu 1.0 -> peak.

    Returns:
        Tuple of (long-format DataFrame, timestamp index).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    cpu = 1e8 + 5e6 * np.sin(t / 15.0) + rng.normal(0, 3e6, n)
    mem = 5e8 + 1e7 * np.sin(t / 20.0) + rng.normal(0, 5e6, n)
    fs = 1e9 + rng.normal(0, 8e6, n)
    timestamps = pd.date_range("2026-01-01T00:00:00Z", periods=n, freq="1min")

    if ramp is not None:
        lo, hi, peak = ramp
        cpu[lo:hi] = cpu[lo:hi] * np.linspace(1.0, peak, hi - lo)
    if spike is not None:
        lo, hi, mult = spike
        cpu[lo:hi] = cpu[lo:hi] * mult

    series = {
        "cpuUsageNanoCores": cpu,
        "memoryUsageBytes": mem,
        "fsUsedBytes": fs,
        "ksmMetricsAvailable": np.ones(n),
        "summaryMetricsAvailable": np.ones(n),
    }
    rows = [
        {
            "resource_id": resource,
            "metric_name": name,
            "timestamp": timestamps[i].isoformat(),
            "value": float(values[i]),
        }
        for name, values in series.items()
        for i in range(n)
    ]
    return pd.DataFrame(rows), timestamps


def write_csv(df: pd.DataFrame, path: Path) -> str:
    """Write a capture DataFrame to CSV and return its path string."""
    df.to_csv(path, index=False)
    return str(path)


def gated_fleet_csv(tmp_path: Path) -> str:
    """Three resources: eligible, divergent-coverage, and under the window floor.

    node-a is clean (58 windows at step 10). node-b drops cpuUsageNanoCores,
    which the capture supplies elsewhere (gate 1). node-c has 140 samples, so
    exactly 12 windows (gate 2).
    """
    df_a, _ = gen_capture("node-a", 600, seed=51)
    df_b, _ = gen_capture("node-b", 600, seed=52)
    df_b = df_b[df_b["metric_name"] != "cpuUsageNanoCores"]
    df_c, _ = gen_capture("node-c", 140, seed=53)
    fleet = pd.concat([df_a, df_b, df_c], ignore_index=True)
    return write_csv(fleet, tmp_path / "gated_fleet.csv")


def write_labels(path: Path, entries: list[dict[str, str]]) -> str:
    """Write a labels JSON from a list of incident entries and return its path."""
    path.write_text(json.dumps(entries))
    return str(path)


def make_incident(
    resource: str, incident_type: str, start: pd.Timestamp, end: pd.Timestamp
) -> dict[str, str]:
    """Build one labels entry."""
    return {
        "resource_id": resource,
        "type": incident_type,
        "start": start.isoformat(),
        "end": end.isoformat(),
    }


# Kind-scoped gates carry allow_absent so a mixed suite evaluates: the
# detection gates and controls gate are absent on the healthy case, and
# alarm_fatigue is absent on the incident case.
RUN_SUITE_RUBRIC = {
    "version": 1,
    "profile": "aro_node",
    "detection": {
        "mode": "no_bridging",
        "onset_anchor": "T0",
        "sustain": 3,
        "max_leadtime_minutes": 120,
        "lead_in_hours": 1,
    },
    "grids": {
        "offline": {"step_samples": 10},
        "serving": {"step_samples": 1, "cadence_minutes": 10},
    },
    "headline_grid": "serving",
    "gates": {
        "no_pre_onset_bridging": {"required": True, "allow_absent": True},
        "detection_lead": {
            "required": True,
            "allow_absent": True,
            "min_lead_vs": "T2",
            "min_lead_seconds": 0,
        },
        "lead_in_fpr": {
            "required": True,
            "allow_absent": True,
            "max_fraction": 0.02,
            "min_eval_windows": 4,
        },
        "alarm_fatigue": {
            "required": True,
            "allow_absent": True,
            "grid": "serving",
            "sustain": 1,
            "max_time_in_alarm_fraction_per_resource": 0.05,
            "max_fleet_raises_per_week": 10,
        },
        "negative_controls_clean": {"required": True, "allow_absent": True},
        "coverage_integrity": {"required": True},
        "sanity": {"required": True},
    },
}


def write_run_suite_tree(tmp_path: Path, keeper_path: str) -> str:
    """A runnable two-case suite: a ramp incident with a control, and a pinned alarm.

    The incident capture ramps node-a's cpu 1x to 8x over minutes 400-700 with
    T0 at the ramp start (06:40) and T2 at 10:00, alongside a healthy control
    node-ctl. The healthy-reference capture is node-hot at a constant 50x cpu,
    the wall-to-wall alarm shape. The calibration fleet carries all three
    resources healthy so every hygiene verdict exists.
    """
    suite_dir = tmp_path / "run_suite"
    (suite_dir / "data").mkdir(parents=True)

    ramp_df, timestamps = gen_capture("node-a", 700, seed=101, ramp=(400, 700, 8.0))
    control_df, _ = gen_capture("node-ctl", 700, seed=102)
    write_csv(
        pd.concat([ramp_df, control_df], ignore_index=True), suite_dir / "data" / "incident.csv"
    )
    labels = LabelSet(
        version=2,
        capture=None,
        cases=[
            LabelCase(
                resource_id="node-a",
                role="incident",
                type="cpu",
                onsets={"T0": timestamps[400], "T2": timestamps[600]},
                primary_onset="T0",
                end=timestamps[699],
                notes=None,
            ),
            LabelCase(
                resource_id="node-ctl",
                role="negative_control",
                type=None,
                onsets={},
                primary_onset=None,
                end=None,
                notes=None,
            ),
        ],
    )
    dump_labels(labels, str(suite_dir / "data" / "labels_v2.json"))

    pinned_df, _ = gen_capture("node-hot", 700, seed=103, spike=(0, 700, 50.0))
    write_csv(pinned_df, suite_dir / "data" / "healthy.csv")

    calibration = pd.concat(
        [
            gen_capture(rid, 700, seed=110 + i)[0]
            for i, rid in enumerate(("node-a", "node-ctl", "node-hot"))
        ],
        ignore_index=True,
    )
    write_csv(calibration, suite_dir / "data" / "calibration.csv")

    (suite_dir / "rubric.yaml").write_text(yaml.safe_dump(RUN_SUITE_RUBRIC))
    suite = {
        "version": 1,
        "suite": "run_suite",
        "candidate": {"type": "reconstruction", "model": keeper_path, "profile": "aro_node"},
        "threshold_policy": {
            "type": "per_resource_margin",
            "calibration": "data/calibration.csv",
            "format": "csv",
            "quantile": 0.99,
            "margin": 2.0,
        },
        "rubric": "rubric.yaml",
        "cases": [
            {
                "name": "ramp_incident",
                "kind": "incident_capture",
                "capture": "data/incident.csv",
                "labels": "data/labels_v2.json",
                "format": "csv",
            },
            {
                "name": "pinned_alarm",
                "kind": "healthy_reference",
                "capture": "data/healthy.csv",
                "format": "csv",
            },
        ],
    }
    path = suite_dir / "suite.yaml"
    path.write_text(yaml.safe_dump(suite))
    return str(path)
