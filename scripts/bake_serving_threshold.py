#!/usr/bin/env python3
# Description: Bake a healthy reconstruction-error threshold into a keeper checkpoint's serving block.
# Description: Reuses the shared windowing/reconstruction core so the number matches validate_incident.

"""Bake a serving threshold into a trained keeper checkpoint.

Loads a keeper, scores an all-healthy capture with the same reconstruction math
the incident-validation harness uses, derives a quantile threshold from the
healthy windows (with a half held out from threshold fitting for the
false-positive rate), and writes a ``serving`` block into the checkpoint:
``{threshold, quantile, healthy_fpr, n_calibration_windows, recon_metric}``. The
serving API reads this block to turn a per-window reconstruction error into an
alertable ratio.

The holdout is from THRESHOLD FITTING only. When the healthy capture is also the
data the model trained on, ``healthy_fpr`` is in-sample for the model and will
understate the live rate; bake against a fresh healthy capture disjoint from the
training range for an operational number.

Example:
    python scripts/bake_serving_threshold.py \\
        --model models/aro_keeper_v1.pt \\
        --healthy data/captures/aro_healthy.parquet \\
        --profile aro_node
"""

from __future__ import annotations

import argparse
import asyncio
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from scry.data.fetcher import fetch_full_capture
from scry.data.quality import missing_features
from scry.data.windowing import build_windows
from scry.model.checkpoint import Keeper, load_keeper
from scry.model.reconstruction import healthy_threshold, reconstruction_errors
from scry.utils.config import get_config

# Identifies how the score was computed, so a serving block is never misread as a
# different metric (e.g. numerical+categorical error) at a different scale.
RECON_METRIC = "numerical_mse_from_mu"

# Floor on per-resource calibration size: a resource with fewer windows is omitted
# from the per-resource map. This screens near-empty resources only; a small-sample
# q99 still sits at or near the sample maximum and leans tight relative to the true
# quantile, and the margin multiplier is what carries that headroom.
MIN_PER_RESOURCE_WINDOWS = 50


def compute_serving_block(
    keeper: Keeper,
    df_long: pd.DataFrame,
    *,
    quantile: float,
    step: int,
    per_resource_margin: float | None = None,
) -> dict[str, Any]:
    """Derive the serving block from a keeper and an all-healthy capture.

    Args:
        keeper: The loaded keeper (model, schema, stored normalization).
        df_long: Canonical long-format healthy metrics.
        quantile: Healthy-window quantile for the threshold.
        step: Sliding-window step (from the config).
        per_resource_margin: When set, additionally write a ``per_resource`` map of
            ``margin x own_quantile`` thresholds, one per resource in the capture.
            The per-resource quantile is taken over all of that resource's windows
            (no holdout split); the margin carries the cross-day drift headroom.
            A resource whose windows are too few for a meaningful quantile is
            omitted (it serves the global threshold as fallback).

    Returns:
        The serving block dict.

    Raises:
        ValueError: If the capture produces no windows.
    """
    seq_len = int(keeper.config["seq_len"])
    windows = build_windows(
        df_long,
        numerical_features=keeper.numerical_features,
        categorical_features=keeper.categorical_features,
        normalization=keeper.normalization,
        cat_normalization=keeper.cat_normalization,
        seq_len=seq_len,
        step=step,
    )
    if windows.x_num.shape[0] == 0:
        raise ValueError(
            "Healthy capture produced no windows; check the profile, the time "
            "range, and that the metrics are present."
        )
    errors = reconstruction_errors(keeper.model, windows.x_num, windows.x_cat, keeper.device)
    # Drop a gap spanning the window overlap (ceil(seq_len / step) windows) so the
    # held-out FPR shares no raw samples with the fit half (mirrors validate_incident).
    overlap_gap = -(-seq_len // step)
    threshold, fit, eval_ = healthy_threshold(
        errors, windows.end_times, quantile=quantile, gap=overlap_gap
    )
    healthy_fpr = float(np.mean(eval_ > threshold)) if eval_.size else None
    block: dict[str, Any] = {
        "threshold": threshold,
        "quantile": quantile,
        "healthy_fpr": healthy_fpr,
        "n_calibration_windows": int(fit.size),
        "recon_metric": RECON_METRIC,
    }
    if per_resource_margin is not None:
        if not math.isfinite(per_resource_margin) or per_resource_margin <= 0:
            raise ValueError("per_resource_margin must be a positive finite number")
        # A feature absent for one resource but supplied elsewhere in the capture is
        # filled at -mean/std by the capture-wide windowing here, while the serving
        # path (which windows the resource alone) fills it neutral. A per-resource
        # quantile from such windows sits on a scale serving never produces, so the
        # affected resource is omitted and falls back to the global threshold.
        trained = set(keeper.numerical_features)
        capture_features = set(df_long["metric_name"].unique()) & trained
        features_by_resource = {
            str(rid): set(group["metric_name"].unique())
            for rid, group in df_long.groupby("resource_id")
        }
        per_resource: dict[str, float] = {}
        for rid in sorted(set(windows.resource_ids)):
            divergent = capture_features - features_by_resource.get(str(rid), set())
            if divergent:
                print(
                    f"warning: resource {rid!r} lacks {len(divergent)} trained "
                    f"feature(s) the capture supplies elsewhere "
                    f"({', '.join(sorted(divergent))}); its bake-time windows are "
                    "filled at -mean/std where serving fills neutral, so it is "
                    "omitted from the per-resource map and serves the global "
                    "threshold.",
                    file=sys.stderr,
                )
                continue
            rerr = errors[windows.resource_ids == rid]
            if rerr.size < MIN_PER_RESOURCE_WINDOWS:
                print(
                    f"warning: resource {rid!r} has only {rerr.size} window(s) "
                    f"(< {MIN_PER_RESOURCE_WINDOWS}); omitting it from the "
                    "per-resource map, it will serve the global threshold.",
                    file=sys.stderr,
                )
                continue
            own_quantile = float(np.quantile(rerr, quantile))
            if own_quantile <= 0:
                print(
                    f"warning: resource {rid!r} has a non-positive healthy quantile; "
                    "omitting it from the per-resource map.",
                    file=sys.stderr,
                )
                continue
            per_resource[str(rid)] = per_resource_margin * own_quantile
        block["per_resource"] = per_resource
        block["margin_multiplier"] = per_resource_margin
    return block


def bake(
    model_path: str,
    healthy_data: str,
    *,
    profile: str | None = None,
    quantile: float = 0.99,
    data_format: str | None = None,
    output: str | None = None,
    per_resource_margin: float | None = None,
) -> dict[str, Any]:
    """Bake a serving threshold into a checkpoint and write it out.

    Args:
        model_path: Path to the keeper checkpoint.
        healthy_data: All-healthy capture URI or path.
        profile: Feature profile; defaults to the checkpoint's stored profile.
        quantile: Healthy-window quantile for the threshold.
        data_format: Optional explicit file format override.
        output: Where to write the updated checkpoint; defaults to in place.
        per_resource_margin: When set, also bake per-resource thresholds of
            ``margin x own_quantile`` (see :func:`compute_serving_block`).

    Returns:
        The serving block that was written.
    """
    keeper = load_keeper(model_path)
    profile = profile or keeper.profile
    step = int(get_config().window_step)
    df_long = asyncio.run(
        fetch_full_capture(healthy_data, profile=profile, data_format=data_format)
    )
    # A profile that no longer lists a trained feature strips it from the
    # fetch; the threshold would then be calibrated on neutral-filled windows.
    missing = missing_features(df_long, keeper.numerical_features)
    if missing:
        print(
            f"warning: {healthy_data} lacks {len(missing)} feature(s) the "
            f"checkpoint was trained on ({', '.join(missing)}); the baked "
            "threshold will not match what the model sees on complete data.",
            file=sys.stderr,
        )
    serving = compute_serving_block(
        keeper, df_long, quantile=quantile, step=step, per_resource_margin=per_resource_margin
    )

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    checkpoint["serving"] = serving
    out = output or model_path
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, out)
    return serving


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Bake a serving reconstruction threshold into a keeper checkpoint."
    )
    parser.add_argument("--model", required=True, help="Path to the keeper checkpoint (.pt).")
    parser.add_argument("--healthy", required=True, help="All-healthy capture URI or path.")
    parser.add_argument(
        "--profile", default=None, help="Feature profile (defaults to the checkpoint's)."
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.99,
        help="Healthy-window quantile for the threshold (default: 0.99).",
    )
    parser.add_argument(
        "--format", choices=["parquet", "csv"], default=None, help="Override the file format."
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Where to write the updated checkpoint (default: in place).",
    )
    parser.add_argument(
        "--per-resource-margin",
        type=float,
        default=None,
        help=(
            "Also bake per-resource thresholds of MARGIN x each resource's own "
            "healthy quantile; unknown resources serve the global threshold."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    serving = bake(
        args.model,
        args.healthy,
        profile=args.profile,
        quantile=args.quantile,
        data_format=args.format,
        output=args.output,
        per_resource_margin=args.per_resource_margin,
    )
    fpr = serving["healthy_fpr"]
    fpr_str = f"{fpr:.4f}" if fpr is not None else "n/a"
    print(
        f"serving threshold={serving['threshold']:.6f} q={serving['quantile']} "
        f"healthy_fpr={fpr_str} n_calibration_windows={serving['n_calibration_windows']}"
    )
    if "per_resource" in serving:
        print(
            f"per-resource thresholds: {len(serving['per_resource'])} resource(s) "
            f"at margin {serving['margin_multiplier']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
