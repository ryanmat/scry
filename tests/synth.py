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
