# Description: Tests for scripts/bake_serving_threshold.py (serving-threshold bake utility).
# Description: Bakes a threshold from a synthetic healthy capture and checks the serving block.

"""Deterministic tests for the serving-threshold bake utility."""

from __future__ import annotations

from pathlib import Path

import bake_serving_threshold as bake_mod
import pytest
import torch
from synth import PROFILE, gen_capture, write_csv


def _healthy_csv(tmp_path: Path, seed: int = 21) -> str:
    df, _ = gen_capture("cal-node", 600, seed=seed)
    return write_csv(df, tmp_path / "healthy.csv")


def test_bake_writes_serving_block(keeper_path: str, tmp_path: Path) -> None:
    """The serving block is written with sane fields and the checkpoint is preserved."""
    healthy = _healthy_csv(tmp_path)
    out = str(tmp_path / "keeper_serving.pt")
    serving = bake_mod.bake(keeper_path, healthy, profile=PROFILE, output=out)

    assert serving["threshold"] > 0.0
    assert serving["quantile"] == 0.99
    assert serving["recon_metric"] == "numerical_mse_from_mu"
    assert serving["n_calibration_windows"] > 0
    if serving["healthy_fpr"] is not None:
        assert 0.0 <= serving["healthy_fpr"] <= 1.0

    ckpt = torch.load(out, map_location="cpu", weights_only=False)
    assert ckpt["serving"] == serving
    # The bake preserves the rest of the checkpoint.
    assert "model_state_dict" in ckpt
    assert "feature_schema" in ckpt
    assert "config" in ckpt


def test_bake_is_deterministic(keeper_path: str, tmp_path: Path) -> None:
    """The same healthy capture and model yield the same threshold (mu is deterministic)."""
    healthy = _healthy_csv(tmp_path)
    a = bake_mod.bake(keeper_path, healthy, profile=PROFILE, output=str(tmp_path / "a.pt"))
    b = bake_mod.bake(keeper_path, healthy, profile=PROFILE, output=str(tmp_path / "b.pt"))
    assert a["threshold"] == b["threshold"]


def test_bake_quantile_monotonic(keeper_path: str, tmp_path: Path) -> None:
    """A higher healthy quantile never lowers the threshold."""
    healthy = _healthy_csv(tmp_path)
    low = bake_mod.bake(
        keeper_path, healthy, profile=PROFILE, quantile=0.5, output=str(tmp_path / "lo.pt")
    )
    high = bake_mod.bake(
        keeper_path, healthy, profile=PROFILE, quantile=0.99, output=str(tmp_path / "hi.pt")
    )
    assert high["threshold"] >= low["threshold"]


def test_bake_defaults_profile_from_checkpoint(keeper_path: str, tmp_path: Path) -> None:
    """With no explicit profile, the checkpoint's stored profile is used."""
    healthy = _healthy_csv(tmp_path)
    serving = bake_mod.bake(keeper_path, healthy, output=str(tmp_path / "def.pt"))
    assert serving["threshold"] > 0.0


def test_bake_empty_capture_errors_clearly(keeper_path: str, tmp_path: Path) -> None:
    """A capture too short to window fails with a clear error, not a silent NaN."""
    df, _ = gen_capture("cal-node", 10, seed=22)  # shorter than seq_len
    short_csv = write_csv(df, tmp_path / "short.csv")
    with pytest.raises(ValueError, match="no windows"):
        bake_mod.bake(keeper_path, short_csv, profile=PROFILE, output=str(tmp_path / "x.pt"))
