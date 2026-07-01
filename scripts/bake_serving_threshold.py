#!/usr/bin/env python3
# Description: Bake a healthy reconstruction-error threshold into a keeper checkpoint's serving block.
# Description: Reuses the shared windowing/reconstruction core so the number matches validate_incident.

"""Bake a serving threshold into a trained keeper checkpoint.

Loads a keeper, scores an all-healthy capture with the same reconstruction math
the incident-validation harness uses, derives a quantile threshold from the
healthy windows (with a held-out half for the false-positive rate), and writes a
``serving`` block into the checkpoint:
``{threshold, quantile, healthy_fpr, n_calibration_windows, recon_metric}``. The
serving API reads this block to turn a per-window reconstruction error into an
alertable ratio.

Example:
    python scripts/bake_serving_threshold.py \\
        --model models/aro_keeper_v1.pt \\
        --healthy data/captures/aro_healthy.parquet \\
        --profile aro_node
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from scry.data.fetcher import fetch_full_capture
from scry.data.windowing import build_windows
from scry.model.checkpoint import Keeper, load_keeper
from scry.model.reconstruction import healthy_threshold, reconstruction_errors
from scry.utils.config import get_config

# Identifies how the score was computed, so a serving block is never misread as a
# different metric (e.g. numerical+categorical error) at a different scale.
RECON_METRIC = "numerical_mse_from_mu"


def compute_serving_block(
    keeper: Keeper, df_long: pd.DataFrame, *, quantile: float, step: int
) -> dict[str, Any]:
    """Derive the serving block from a keeper and an all-healthy capture.

    Args:
        keeper: The loaded keeper (model, schema, stored normalization).
        df_long: Canonical long-format healthy metrics.
        quantile: Healthy-window quantile for the threshold.
        step: Sliding-window step (from the config).

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
    return {
        "threshold": threshold,
        "quantile": quantile,
        "healthy_fpr": healthy_fpr,
        "n_calibration_windows": int(fit.size),
        "recon_metric": RECON_METRIC,
    }


def bake(
    model_path: str,
    healthy_data: str,
    *,
    profile: str | None = None,
    quantile: float = 0.99,
    data_format: str | None = None,
    output: str | None = None,
) -> dict[str, Any]:
    """Bake a serving threshold into a checkpoint and write it out.

    Args:
        model_path: Path to the keeper checkpoint.
        healthy_data: All-healthy capture URI or path.
        profile: Feature profile; defaults to the checkpoint's stored profile.
        quantile: Healthy-window quantile for the threshold.
        data_format: Optional explicit file format override.
        output: Where to write the updated checkpoint; defaults to in place.

    Returns:
        The serving block that was written.
    """
    keeper = load_keeper(model_path)
    profile = profile or keeper.profile
    step = int(get_config().window_step)
    df_long = asyncio.run(
        fetch_full_capture(healthy_data, profile=profile, data_format=data_format)
    )
    serving = compute_serving_block(keeper, df_long, quantile=quantile, step=step)

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
    )
    fpr = serving["healthy_fpr"]
    fpr_str = f"{fpr:.4f}" if fpr is not None else "n/a"
    print(
        f"serving threshold={serving['threshold']:.6f} q={serving['quantile']} "
        f"healthy_fpr={fpr_str} n_calibration_windows={serving['n_calibration_windows']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
