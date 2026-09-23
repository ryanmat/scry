#!/usr/bin/env python3
# Description: CLI over scry.eval.sweep.run_sweep: the corrected margin sweep as a results JSON.
# Description: Holds the input paths run_sweep cannot see and records them in its provenance.

"""Run the corrected margin sweep over a healthy week and a capture.

A thin wrapper over ``run_sweep``: it scores both captures with the keeper
through a ``ReconstructionCandidate`` and writes the results JSON -- the named
baselines, the three rate arms, detection at BOTH baselines, and the
DESIGN-addendum caveats -- to ``--output``.

Both captures are scored on ``SERVING_GRID``: the sweep picks the margin the
deployed threshold will serve at, so it measures on the deployed latest-window
refresh emulated offline, at native window resolution rather than the coarse
stride the original sweep's flat detection lead turned out to be an artifact of.

``run_sweep`` is handed DataFrames, so its provenance cannot name the files
they came from; this CLI holds those paths and records them in the provenance
``cases`` block, leaving the result's top-level keys exactly the contract.

Example:
    python scripts/sweep_margin.py --model models/keeper.pt \\
        --healthy captures/healthy_week.csv --capture captures/incident_day.csv \\
        --labels captures/incident_labels.json --profile aro_node \\
        --margins 1.5,2.0,2.5 --output sweep_results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scry.eval.candidate import ReconstructionCandidate
from scry.eval.labels import load_labels
from scry.eval.provenance import build_provenance
from scry.eval.scoring import SERVING_GRID
from scry.eval.suite import load_capture
from scry.eval.sweep import run_sweep


def margin_list(value: str) -> list[float]:
    """Parse ``--margins`` as a comma-separated list of threshold multipliers.

    Raises:
        argparse.ArgumentTypeError: If an entry is not a number, so a typo is a
            usage error rather than a sweep of the wrong margins.
    """
    try:
        return [float(part) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--margins takes a comma-separated float list: {exc}")


def record_inputs(provenance: dict[str, Any], args: argparse.Namespace) -> None:
    """Record the three input files in ``provenance["cases"]``, in place.

    run_sweep is handed frames, so it leaves the cases block empty for the
    caller holding the paths. Only that block is filled in, and through the
    same builder that made the rest of the provenance, so the recorded
    identity cannot drift from the shape the rest of the harness records.
    """
    provenance["cases"] = build_provenance(
        model_path=args.model,
        profile=provenance["profile"],
        seq_len=provenance["seq_len"],
        grids={SERVING_GRID.label: SERVING_GRID},
        sustains=provenance["sustains"],
        detection_mode=provenance["detection_mode"],
        policy_description=provenance["threshold_policy"],
        rubric_path=provenance["rubric_path"],
        rubric_version=provenance["rubric_version"],
        case_data_paths={
            "healthy": args.healthy,
            "capture": args.capture,
            "labels": args.labels,
        },
    )["cases"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the corrected margin sweep and write its results JSON.",
    )
    parser.add_argument("--model", required=True, help="Path to the keeper checkpoint (.pt).")
    parser.add_argument("--healthy", required=True, help="All-healthy week capture URI or path.")
    parser.add_argument("--capture", required=True, help="Capture under evaluation, URI or path.")
    parser.add_argument("--labels", required=True, help="Labels sidecar JSON for the capture.")
    parser.add_argument("--profile", required=True, help="Feature profile both captures load on.")
    parser.add_argument(
        "--margins", required=True, type=margin_list,
        help="Comma-separated threshold multipliers to sweep, e.g. 1.5,2.0,2.5.",
    )
    parser.add_argument(
        "--quantile", type=float, default=0.99,
        help="Healthy-window quantile behind every baseline (default: 0.99).",
    )
    parser.add_argument(
        "--sustain", type=int, default=3,
        help="Windows a run must hold to count as sustained (default: 3).",
    )
    parser.add_argument("--output", required=True, help="Where to write the results JSON.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    result = run_sweep(
        ReconstructionCandidate(args.model, profile=args.profile),
        load_capture(args.healthy, args.profile),
        load_capture(args.capture, args.profile),
        load_labels(args.labels),
        args.margins,
        quantile=args.quantile,
        sustain=args.sustain,
        grid=SERVING_GRID,
    )
    record_inputs(result["provenance"], args)

    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"swept margins {args.margins} over "
        f"{len(result['per_resource_baselines'])} resource(s) -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
