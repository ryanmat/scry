# Description: Tests for the incident-validation harness (scripts/validate_incident.py).
# Description: Trains a tiny keeper, injects anomalies, and checks detection and leakage guards.

"""Deterministic, mock-free tests for the incident-validation harness.

Each test trains a tiny keeper X-DEC on synthetic normal data, writes synthetic
captures as CSV (read through the real object-store path), and exercises the
harness end to end: anomaly detection with a finite lead time, a clean
healthy-only run, and the leakage guard that the threshold is computed from
healthy windows only.
"""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F  # noqa: N812 -- PyTorch convention
import validate_incident as vi

from scry.data.feature_engineering import set_active_profile
from scry.data.fetcher import DataFetcher
from scry.data.pipeline import XDECFeaturePipeline
from scry.model.xdec import TemporalXDEC
from scry.utils.config import get_config

# A subset of the aro_node profile features; the capture supplies exactly these.
_SERIES = ("cpuUsageNanoCores", "memoryUsageBytes", "fsUsedBytes")
_CAT = ("ksmMetricsAvailable", "summaryMetricsAvailable")
_PROFILE = "aro_node"
_SEQ_LEN = 30


def _gen_capture(
    resource: str,
    n: int,
    seed: int,
    spike: tuple[int, int, float] | None = None,
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Generate a synthetic long-format capture for one resource.

    Args:
        resource: Resource id.
        n: Number of timesteps (1-minute cadence).
        seed: RNG seed for reproducibility.
        spike: Optional (lo, hi, multiplier) injected into cpuUsageNanoCores.

    Returns:
        Tuple of (long-format DataFrame, timestamp index).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    cpu = 1e8 + 5e6 * np.sin(t / 15.0) + rng.normal(0, 3e6, n)
    mem = 5e8 + 1e7 * np.sin(t / 20.0) + rng.normal(0, 5e6, n)
    fs = 1e9 + rng.normal(0, 8e6, n)
    timestamps = pd.date_range("2026-01-01T00:00:00Z", periods=n, freq="1min")

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


def _write_csv(df: pd.DataFrame, path: Path) -> str:
    """Write a capture DataFrame to CSV and return its path string."""
    df.to_csv(path, index=False)
    return str(path)


def _write_labels(
    path: Path,
    resource: str,
    incident_type: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> str:
    """Write a one-incident labels JSON and return its path string."""
    labels = [
        {
            "resource_id": resource,
            "type": incident_type,
            "start": start.isoformat(),
            "end": end.isoformat(),
        }
    ]
    path.write_text(json.dumps(labels))
    return str(path)


@pytest.fixture(scope="module")
def keeper_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Train a tiny keeper on synthetic normal data and save its checkpoint.

    The training data is windowed and normalized through the real feature
    pipeline so the checkpoint carries the same stored normalization the harness
    re-applies to incident windows.
    """
    tmp = tmp_path_factory.mktemp("keeper")
    train_df, _ = _gen_capture("train-node", 800, seed=1)
    train_csv = _write_csv(train_df, tmp / "train.csv")

    set_active_profile(_PROFILE)
    fetcher = DataFetcher.from_object_store(train_csv)
    pipeline = XDECFeaturePipeline(fetcher, get_config())
    start = pd.Timestamp("2025-01-01T00:00:00Z").to_pydatetime()
    end = pd.Timestamp("2027-01-01T00:00:00Z").to_pydatetime()
    raw = asyncio.run(pipeline.extract(start, end, profile=_PROFILE))
    data = pipeline.transform(raw)

    assert data["num_windows"].shape[0] > 0
    assert data["feature_names"]["numerical"] == list(_SERIES)

    torch.manual_seed(0)
    np.random.seed(0)
    model = TemporalXDEC(
        num_numerical=len(_SERIES),
        num_categorical=len(_CAT),
        seq_len=_SEQ_LEN,
        num_hidden=16,
        cat_hidden=8,
        latent_dim=4,
        n_clusters=3,
    )
    x_num = torch.tensor(data["num_windows"], dtype=torch.float32)
    x_cat = torch.tensor(data["cat_windows"], dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    model.train()
    for _ in range(500):
        optimizer.zero_grad()
        out = model.xvae(x_num, x_cat)
        loss = F.mse_loss(out["x_num_recon"], x_num) + F.mse_loss(out["x_cat_recon"], x_cat)
        loss.backward()
        optimizer.step()
    model.eval()

    ckpt_path = tmp / "keeper.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "num_numerical": len(_SERIES),
                "num_categorical": len(_CAT),
                "seq_len": _SEQ_LEN,
                "num_hidden": 16,
                "cat_hidden": 8,
                "latent_dim": 4,
                "n_clusters": 3,
            },
            "normalization": {
                "mean": data["num_norm_params"]["mean"],
                "std": data["num_norm_params"]["std"],
            },
            "categorical_normalization": {
                "min": data["cat_norm_params"]["min"],
                "max": data["cat_norm_params"]["max"],
            },
            "feature_schema": {
                "numerical": data["feature_names"]["numerical"],
                "categorical": data["feature_names"]["categorical"],
                "profile": _PROFILE,
            },
        },
        ckpt_path,
    )
    return str(ckpt_path)


def test_detects_injected_anomaly_with_finite_lead(
    keeper_path: str, tmp_path: Path
) -> None:
    """A clear injected spike is flagged with a finite lead time and low healthy FPR."""
    capture_df, timestamps = _gen_capture("node-a", 560, seed=2, spike=(500, 560, 40.0))
    capture_csv = _write_csv(capture_df, tmp_path / "incident.csv")
    labels = _write_labels(
        tmp_path / "labels.json", "node-a", "cpu_spike", timestamps[500], timestamps[559]
    )

    summary = vi.analyze(keeper_path, capture_csv, labels, _PROFILE)

    assert summary["threshold_source"] == "healthy_split"
    assert summary["n_healthy_windows"] >= 21
    assert summary["healthy_fpr"] <= 0.05

    incident = summary["incidents"][0]
    assert incident["detected"] is True
    assert incident["lead_time_seconds"] is not None
    assert math.isfinite(incident["lead_time_seconds"])
    # The spike dwarfs the healthy threshold.
    assert incident["max_error_in_window"] > summary["threshold"] * 100


def test_healthy_capture_has_no_sustained_detection(
    keeper_path: str, tmp_path: Path
) -> None:
    """An all-healthy capture yields no sustained detection and a low FPR."""
    capture_df, timestamps = _gen_capture("node-a", 560, seed=3, spike=None)
    capture_csv = _write_csv(capture_df, tmp_path / "healthy.csv")
    labels = _write_labels(
        tmp_path / "labels.json", "node-a", "none", timestamps[500], timestamps[559]
    )

    summary = vi.analyze(keeper_path, capture_csv, labels, _PROFILE)

    assert summary["healthy_fpr"] <= 0.05
    incident = summary["incidents"][0]
    assert incident["detected"] is False
    assert incident["lead_time_seconds"] is None


def test_huge_anomaly_does_not_inflate_threshold(
    keeper_path: str, tmp_path: Path
) -> None:
    """The threshold is healthy-only: a huge spike cannot raise it, yet is detected."""
    small_df, ts = _gen_capture("node-a", 560, seed=2, spike=(500, 560, 40.0))
    huge_df, _ = _gen_capture("node-a", 560, seed=2, spike=(500, 560, 400.0))
    small_csv = _write_csv(small_df, tmp_path / "small.csv")
    huge_csv = _write_csv(huge_df, tmp_path / "huge.csv")
    labels = _write_labels(
        tmp_path / "labels.json", "node-a", "cpu_spike", ts[500], ts[559]
    )

    small = vi.analyze(keeper_path, small_csv, labels, _PROFILE)
    huge = vi.analyze(keeper_path, huge_csv, labels, _PROFILE)

    # Identical healthy prefix (same seed) => identical healthy-only threshold,
    # independent of the incident magnitude. No leakage from the spike.
    assert small["threshold"] == pytest.approx(huge["threshold"], abs=1e-9)
    # The far larger spike produces a far larger in-window error.
    assert (
        huge["incidents"][0]["max_error_in_window"]
        > small["incidents"][0]["max_error_in_window"]
    )
    assert huge["incidents"][0]["detected"] is True


def test_zero_healthy_windows_errors_clearly(keeper_path: str, tmp_path: Path) -> None:
    """With every window at or after the incident start, the harness errors clearly."""
    capture_df, timestamps = _gen_capture("node-a", 560, seed=2, spike=(0, 60, 40.0))
    capture_csv = _write_csv(capture_df, tmp_path / "all_incident.csv")
    # Incident starts at the very first timestamp, so no window ends before it.
    labels = _write_labels(
        tmp_path / "labels.json", "node-a", "cpu_spike", timestamps[0], timestamps[559]
    )

    with pytest.raises(ValueError, match="No healthy windows"):
        vi.analyze(keeper_path, capture_csv, labels, _PROFILE)
