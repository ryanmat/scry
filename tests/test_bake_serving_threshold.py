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


# -- per-resource thresholds --


def _two_resource_csv(tmp_path: Path) -> str:
    import pandas as pd

    df_a, _ = gen_capture("node-a", 600, seed=31)
    df_b, _ = gen_capture("node-b", 600, seed=32)
    return write_csv(pd.concat([df_a, df_b], ignore_index=True), tmp_path / "fleet.csv")


def test_bake_per_resource_margin_writes_map(keeper_path: str, tmp_path: Path) -> None:
    """--per-resource-margin writes a per-resource threshold map plus the margin."""
    healthy = _two_resource_csv(tmp_path)
    out = str(tmp_path / "pr.pt")
    serving = bake_mod.bake(
        keeper_path, healthy, profile=PROFILE, per_resource_margin=2.0, output=out
    )

    assert serving["margin_multiplier"] == 2.0
    per_resource = serving["per_resource"]
    assert set(per_resource) == {"node-a", "node-b"}
    for threshold in per_resource.values():
        assert threshold > 0.0
    # The global fields are unchanged by the flag.
    plain = bake_mod.bake(
        keeper_path, healthy, profile=PROFILE, output=str(tmp_path / "plain.pt")
    )
    assert serving["threshold"] == plain["threshold"]

    ckpt = torch.load(out, map_location="cpu", weights_only=False)
    assert ckpt["serving"] == serving


def test_bake_per_resource_values_scale_with_margin(keeper_path: str, tmp_path: Path) -> None:
    """The margin multiplies each resource's own healthy quantile linearly."""
    healthy = _two_resource_csv(tmp_path)
    one = bake_mod.bake(
        keeper_path, healthy, profile=PROFILE, per_resource_margin=1.0,
        output=str(tmp_path / "m1.pt"),
    )
    two = bake_mod.bake(
        keeper_path, healthy, profile=PROFILE, per_resource_margin=2.0,
        output=str(tmp_path / "m2.pt"),
    )
    for rid in one["per_resource"]:
        assert two["per_resource"][rid] == pytest.approx(2.0 * one["per_resource"][rid])


def test_bake_default_has_no_per_resource(keeper_path: str, tmp_path: Path) -> None:
    """Without the flag the serving block keeps its original shape."""
    healthy = _healthy_csv(tmp_path)
    serving = bake_mod.bake(keeper_path, healthy, profile=PROFILE, output=str(tmp_path / "d.pt"))
    assert "per_resource" not in serving
    assert "margin_multiplier" not in serving


def test_bake_per_resource_skips_unwindowable_resource(keeper_path: str, tmp_path: Path) -> None:
    """A resource too short to window falls back to the global threshold (omitted from the map)."""
    import pandas as pd

    df_a, _ = gen_capture("node-a", 600, seed=33)
    df_short, _ = gen_capture("node-short", 10, seed=34)  # shorter than seq_len
    healthy = write_csv(pd.concat([df_a, df_short], ignore_index=True), tmp_path / "mix.csv")
    serving = bake_mod.bake(
        keeper_path, healthy, profile=PROFILE, per_resource_margin=2.0,
        output=str(tmp_path / "mix.pt"),
    )
    assert "node-a" in serving["per_resource"]
    assert "node-short" not in serving["per_resource"]


def test_bake_per_resource_omits_divergent_coverage(keeper_path: str, tmp_path: Path) -> None:
    """A resource missing a trained feature the capture supplies elsewhere is omitted.

    Its bake-time windows would carry that column at -mean/std (capture-wide absent
    masking) while the single-resource serving path fills it neutral, so its baked
    quantile would sit on a scale serving never produces.
    """
    import pandas as pd

    df_a, _ = gen_capture("node-a", 600, seed=41)
    df_b, _ = gen_capture("node-b", 600, seed=42)
    df_b = df_b[df_b["metric_name"] != "cpuUsageNanoCores"]
    healthy = write_csv(pd.concat([df_a, df_b], ignore_index=True), tmp_path / "hetero.csv")
    serving = bake_mod.bake(
        keeper_path, healthy, profile=PROFILE, per_resource_margin=2.0,
        output=str(tmp_path / "hetero.pt"),
    )
    assert "node-a" in serving["per_resource"]
    assert "node-b" not in serving["per_resource"]


def test_bake_rejects_non_finite_or_non_positive_margin(keeper_path: str, tmp_path: Path) -> None:
    """NaN, inf, zero, and negative margins are rejected before any windowing."""
    healthy = _healthy_csv(tmp_path)
    for bad in (float("nan"), float("inf"), 0.0, -2.0):
        with pytest.raises(ValueError, match="margin"):
            bake_mod.bake(
                keeper_path, healthy, profile=PROFILE, per_resource_margin=bad,
                output=str(tmp_path / "bad.pt"),
            )
