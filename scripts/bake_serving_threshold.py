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

The setup every bake starts with -- the keeper, the capture, and the window
step, with a warning when the capture lacks a trained feature -- is
``scry.eval.bake.prepare_bake``, shared with the rebake path so the two
cannot bake from differently assembled inputs.

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
import json
from pathlib import Path
from typing import Any

import torch

from scry.eval.bake import compute_serving_block, prepare_bake


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
    keeper, df_long, step = prepare_bake(
        model_path, healthy_data, profile=profile, data_format=data_format
    )
    serving, eligibility = compute_serving_block(
        keeper, df_long, quantile=quantile, step=step, per_resource_margin=per_resource_margin
    )
    if eligibility is not None:
        reasons_map = {rid: verdict.reasons for rid, verdict in eligibility.items()}
        print(f"per-resource eligibility: {json.dumps(reasons_map)}")

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
