# Description: Generates a small synthetic metrics dataset in Scry's canonical schema.
# Description: Fake Kubernetes-style data for demos and tests; train on your own metrics for real use.

"""Generate synthetic sample metrics for Scry.

Writes ``examples/sample_data/metrics.parquet`` in the canonical long schema
(resource_id, host_name, metric_name, timestamp, value, datasource_instance,
datasource_name) using metric names from the kubernetes feature profile. The
data is deterministic (fixed seed) and entirely synthetic.
"""

import math
import os
from datetime import datetime, timedelta, timezone

import duckdb
import numpy as np
import pandas as pd

OUT_DIR = os.path.join(os.path.dirname(__file__), "sample_data")
RESOURCES = ["web-app-1", "web-app-2", "web-app-3", "api-worker-1", "cache-1"]
N_STEPS = 240          # per resource, 1-minute spacing -> ~22 windows each at step=10
SEED = 7               # fixed for reproducible synthetic data

NUMERICAL_BASES = {
    "cpuUsageNanoCores": (2.0e8, 5.0e7, 0.0),
    "memoryUsageBytes": (5.0e8, 4.0e7, 1.0e5),
    "memoryWorkingSetBytes": (4.5e8, 3.5e7, 1.0e5),
    "memoryRssBytes": (4.0e8, 3.0e7, 0.0),
    "networkRxBytes": (1.0e6, 3.0e5, 0.0),
    "networkTxBytes": (8.0e5, 2.0e5, 0.0),
    "fsUsedBytes": (2.0e9, 1.0e7, 5.0e5),
    "kubeDeploymentStatusReplicasAvailable": (3.0, 0.0, 0.0),
}
CATEGORICAL = [
    "podConditionPhase",
    "kubePodStatusReady",
    "kubePodContainerStatusRunning",
    "kubePodContainerStatusWaiting",
    "kubePodContainerStatusTerminated",
    "kubePodContainerStatusReady",
    "kubeNodeStatusConditionReady",
    "status",
]


def _numeric_series(rng, base, amp, trend, degrade):
    """A daily-ish sine plus noise and slow trend; the degrading host ramps late."""
    t = np.arange(N_STEPS)
    vals = base + amp * np.sin(2 * math.pi * t / 60.0) + rng.normal(0, amp * 0.1, N_STEPS) + trend * t
    if degrade and amp > 0:
        tail = N_STEPS - int(N_STEPS * 0.7)
        vals[int(N_STEPS * 0.7):] += amp * 3.0 * np.linspace(0.0, 1.0, tail)
    return np.clip(vals, 0.0, None)


def main() -> None:
    rng = np.random.default_rng(SEED)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows: list[tuple] = []

    for idx, resource in enumerate(RESOURCES):
        degrade = idx == len(RESOURCES) - 1  # one host trends toward trouble, for cluster variety

        numeric = {
            name: _numeric_series(rng, base, amp, trend, degrade and name.startswith(("cpu", "memory")))
            for name, (base, amp, trend) in NUMERICAL_BASES.items()
        }
        numeric["cpuUsageNanoCores"] = _numeric_series(rng, 2.0e8, 5.0e7, 0.0, degrade)
        restarts = np.cumsum((rng.random(N_STEPS) < (0.06 if degrade else 0.005)).astype(float))
        numeric["kubePodContainerStatusRestartsTotal"] = restarts

        ready = (numeric["cpuUsageNanoCores"] < 4.0e8).astype(int)
        categorical = {
            "podConditionPhase": np.where(ready == 1, 2, 4),
            "kubePodStatusReady": ready,
            "kubePodContainerStatusRunning": ready,
            "kubePodContainerStatusWaiting": 1 - ready,
            "kubePodContainerStatusTerminated": np.zeros(N_STEPS, dtype=int),
            "kubePodContainerStatusReady": ready,
            "kubeNodeStatusConditionReady": np.ones(N_STEPS, dtype=int),
            "status": ready,
        }

        for i in range(N_STEPS):
            ts = (start + timedelta(minutes=i)).isoformat()
            for name, series in numeric.items():
                rows.append((resource, resource, name, ts, float(series[i]), f"{resource}-inst", "Kubernetes_KSM_Pods"))
            for name in CATEGORICAL:
                rows.append((resource, resource, name, ts, float(categorical[name][i]), f"{resource}-inst", "Kubernetes_KSM_Pods"))

    df = pd.DataFrame(
        rows,
        columns=[
            "resource_id", "host_name", "metric_name", "timestamp",
            "value", "datasource_instance", "datasource_name",
        ],
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "metrics.parquet")
    con = duckdb.connect()
    con.register("df", df)
    con.execute(f"COPY df TO '{out_path}' (FORMAT PARQUET)")
    con.close()
    print(f"wrote {len(df)} rows for {len(RESOURCES)} resources to {out_path}")


if __name__ == "__main__":
    main()
