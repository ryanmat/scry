#!/usr/bin/env python3
# Description: Per-resource healthy-baseline report for a trained keeper over a healthy capture.
# Description: Groups reconstruction errors by resource and compares per-resource vs global thresholds.

"""Report per-resource healthy baselines for a keeper X-DEC model.

Windows a healthy capture exactly as the incident-validation harness does (the
model's own sequence length and stored normalization, the checkpoint feature
order), scores every window's reconstruction error once, then groups the windows
by resource. For each resource it reports the window count, the resource's own
healthy quantile of the error, and the data span; for each supplied threshold it
reports the window false-positive rate, the number of sustained anomalous runs
(computed per resource in time order, so runs are temporal), and those runs
normalized to a per-week rate. The same rollup pooled across all resources sits
alongside, so a global threshold (set by the noisiest resource's tail) can be
read against each resource's own baseline in one place.

Example:
    python scripts/healthy_fleet_report.py \\
        --model models/aro_keeper_v2.pt \\
        --data data/captures/aro_healthy_week.parquet \\
        --profile aro_node \\
        --threshold 0.1932 --threshold 0.25
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import validate_incident
except ModuleNotFoundError:  # running from outside scripts/ without an install
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import validate_incident

from scry.data.feature_engineering import set_active_profile
from scry.data.fetcher import fetch_full_capture
from scry.model.checkpoint import load_keeper
from scry.model.reconstruction import reconstruction_errors
from scry.utils.config import get_config


def _threshold_key(threshold: float) -> str:
    """The JSON/table key for a threshold (str of the float, matching the CLI value)."""
    return str(threshold)


def _span_days(end_times: pd.DatetimeIndex) -> float:
    """Span in days between the earliest and latest window end-time (0 for <2 windows)."""
    if len(end_times) == 0:
        return 0.0
    span = end_times.max() - end_times.min()
    return span.total_seconds() / 86400.0


def _resource_baseline(
    errors: np.ndarray,
    end_times: pd.DatetimeIndex,
    thresholds: list[float],
    sustain: int,
    quantile: float,
) -> dict[str, Any]:
    """Healthy-baseline stats for one resource's windows.

    Sustained runs are counted on the errors in end-time order, so a run is a
    temporal streak within the resource, not an artifact of window emission order.

    Args:
        errors: Per-window reconstruction errors for the resource.
        end_times: Matching per-window end timestamps.
        thresholds: Thresholds to score (may be empty).
        sustain: Consecutive over-threshold windows required for a run.
        quantile: Quantile reported as the resource's own healthy baseline.

    Returns:
        A per-resource stats dict: n_windows, own_q, span_days, thresholds.
    """
    order = np.argsort(end_times.values, kind="stable")
    ordered = errors[order]
    span_days = _span_days(end_times)
    own_q = float(np.quantile(errors, quantile)) if errors.size else None

    thresh_stats: dict[str, Any] = {}
    for threshold in thresholds:
        runs = len(validate_incident._anomaly_runs(ordered > threshold, sustain))
        window_fpr = float(np.mean(errors > threshold)) if errors.size else None
        runs_per_week = (7.0 * runs / span_days) if span_days > 0 else None
        thresh_stats[_threshold_key(threshold)] = {
            "window_fpr": window_fpr,
            "sustained_runs": runs,
            "runs_per_week": runs_per_week,
        }

    return {
        "n_windows": int(errors.size),
        "own_q": own_q,
        "span_days": span_days,
        "thresholds": thresh_stats,
    }


def _global_baseline(
    errors: np.ndarray,
    end_times: pd.DatetimeIndex,
    resources: dict[str, dict[str, Any]],
    thresholds: list[float],
    sustain: int,
    quantile: float,
) -> dict[str, Any]:
    """Pooled healthy-baseline stats across all resources.

    The window count, own quantile, span, and window false-positive rate are over
    the pooled errors. Sustained runs are summed from the per-resource counts
    rather than recomputed on the pooled, time-sorted errors: a run is a
    within-resource temporal streak, and interleaving resources by global
    end-time would fabricate cross-resource runs that never fire in serving (each
    resource is scored independently). The per-week rate normalizes that summed
    count by the pooled span.

    Args:
        errors: All per-window errors.
        end_times: All per-window end timestamps.
        resources: The already-computed per-resource baselines.
        thresholds: Thresholds to score (may be empty).
        sustain: Consecutive over-threshold windows required for a run.
        quantile: Quantile reported as the pooled healthy baseline.

    Returns:
        A rollup stats dict of the same shape as a per-resource entry.
    """
    span_days = _span_days(end_times)
    own_q = float(np.quantile(errors, quantile)) if errors.size else None

    thresh_stats: dict[str, Any] = {}
    for threshold in thresholds:
        key = _threshold_key(threshold)
        total_runs = sum(res["thresholds"][key]["sustained_runs"] for res in resources.values())
        window_fpr = float(np.mean(errors > threshold)) if errors.size else None
        runs_per_week = (7.0 * total_runs / span_days) if span_days > 0 else None
        thresh_stats[key] = {
            "window_fpr": window_fpr,
            "sustained_runs": total_runs,
            "runs_per_week": runs_per_week,
        }

    return {
        "n_windows": int(errors.size),
        "own_q": own_q,
        "span_days": span_days,
        "thresholds": thresh_stats,
    }


def compute_report(
    errors: np.ndarray,
    resource_ids: np.ndarray,
    end_times: pd.DatetimeIndex,
    thresholds: list[float],
    sustain: int,
    quantile: float,
) -> dict[str, Any]:
    """Group per-window errors by resource and build the per-resource and global stats.

    Args:
        errors: Per-window reconstruction errors.
        resource_ids: Per-window resource ids.
        end_times: Per-window end timestamps.
        thresholds: Thresholds to score (may be empty).
        sustain: Consecutive over-threshold windows required for a run.
        quantile: Quantile reported as each resource's own healthy baseline.

    Returns:
        Dict with ``resources`` (one entry per resource id) and ``global``.
    """
    ids = resource_ids.astype(str)
    resources: dict[str, dict[str, Any]] = {}
    for rid in sorted(set(ids.tolist())):
        mask = ids == rid
        resources[rid] = _resource_baseline(
            errors[mask], end_times[mask], thresholds, sustain, quantile
        )
    global_stats = _global_baseline(errors, end_times, resources, thresholds, sustain, quantile)
    return {"resources": resources, "global": global_stats}


def analyze(
    model_path: str,
    data: str,
    profile: str,
    *,
    thresholds: list[float],
    sustain: int = 3,
    threshold_quantile: float = 0.99,
    data_format: str | None = None,
) -> dict[str, Any]:
    """Load the keeper, score the capture, and return the per-resource healthy report.

    Args:
        model_path: Path to the keeper checkpoint.
        data: Healthy capture URI or path.
        profile: Feature profile for the capture.
        thresholds: Thresholds to score (may be empty; duplicates are collapsed).
        sustain: Consecutive over-threshold windows required for a run.
        threshold_quantile: Per-resource quantile reported as the own baseline.
        data_format: Optional explicit file format override.

    Returns:
        The report dict (also suitable for JSON serialization).

    Raises:
        ValueError: If the capture produces no windows.
    """
    keeper = load_keeper(model_path)

    if keeper.profile and keeper.profile != profile:
        print(
            f"warning: --profile '{profile}' differs from the checkpoint profile "
            f"'{keeper.profile}'; using the checkpoint feature order for alignment.",
            file=sys.stderr,
        )

    set_active_profile(profile)
    seq_len = int(keeper.config["seq_len"])
    step = int(get_config().window_step)

    df_long = asyncio.run(fetch_full_capture(data, profile=profile, data_format=data_format))
    validate_incident._warn_missing_model_features(df_long, keeper, data)
    windows = validate_incident._windows_for_keeper(df_long, keeper, seq_len, step)
    if windows.x_num.shape[0] == 0:
        raise ValueError(
            f"Capture {data} produced no windows for profile '{profile}'. "
            "Check the time range, profile, and that the metrics are present."
        )
    errors = reconstruction_errors(keeper.model, windows.x_num, windows.x_cat, keeper.device)

    unique_thresholds = list(dict.fromkeys(thresholds))
    report = compute_report(
        errors,
        windows.resource_ids,
        windows.end_times,
        unique_thresholds,
        sustain,
        threshold_quantile,
    )

    return {
        "model": model_path,
        "data": data,
        "profile": profile,
        "sustain": sustain,
        "threshold_quantile": threshold_quantile,
        "generated_from_span": report["global"]["span_days"],
        "resources": report["resources"],
        "global": report["global"],
    }


def _fmt(value: float | None, spec: str) -> str:
    """Format an optional float for the table, rendering None as ``n/a``."""
    return "n/a" if value is None else format(value, spec)


def print_report(summary: dict[str, Any]) -> None:
    """Print the human-readable per-resource baseline and per-threshold tables."""
    resources: dict[str, Any] = summary["resources"]
    global_stats: dict[str, Any] = summary["global"]
    names = list(resources.keys())
    width = max(len("resource_id"), len("GLOBAL"), *(len(name) for name in names))
    q = summary["threshold_quantile"]

    print("Healthy fleet report")
    print(f"model:   {summary['model']}")
    print(f"data:    {summary['data']}")
    print(f"profile: {summary['profile']}   sustain: {summary['sustain']}   quantile: {q}")
    print(
        f"span:    {summary['generated_from_span']:.3f} days across {len(names)} "
        f"resource(s), {global_stats['n_windows']} windows"
    )
    print()

    print(f"per-resource baseline (own q{q} reconstruction error):")
    print(f"  {'resource_id':<{width}}  {'n_windows':>10}  {'span_days':>10}  {'own_q':>12}")
    for name in names:
        res = resources[name]
        print(
            f"  {name:<{width}}  {res['n_windows']:>10}  {res['span_days']:>10.3f}  "
            f"{_fmt(res['own_q'], '.6f'):>12}"
        )
    print(
        f"  {'GLOBAL':<{width}}  {global_stats['n_windows']:>10}  "
        f"{global_stats['span_days']:>10.3f}  {_fmt(global_stats['own_q'], '.6f'):>12}"
    )

    thresholds = list(global_stats["thresholds"].keys())
    if not thresholds:
        print()
        print("no --threshold supplied; baselines only.")
        return

    for key in thresholds:
        print()
        print(
            f"threshold = {key} (window_fpr = fraction over it; "
            f"sustained_runs = temporal runs >= {summary['sustain']}):"
        )
        print(
            f"  {'resource_id':<{width}}  {'window_fpr':>10}  {'sustained_runs':>15}  "
            f"{'runs_per_week':>14}"
        )
        for name in names:
            stats = resources[name]["thresholds"][key]
            print(
                f"  {name:<{width}}  {_fmt(stats['window_fpr'], '.4f'):>10}  "
                f"{stats['sustained_runs']:>15}  {_fmt(stats['runs_per_week'], '.2f'):>14}"
            )
        gstats = global_stats["thresholds"][key]
        print(
            f"  {'GLOBAL':<{width}}  {_fmt(gstats['window_fpr'], '.4f'):>10}  "
            f"{gstats['sustained_runs']:>15}  {_fmt(gstats['runs_per_week'], '.2f'):>14}"
        )


def _positive_float(value: str) -> float:
    """Argparse type for a threshold: a float strictly greater than zero."""
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not a float: {value!r}") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"threshold must be > 0, got {value}")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Per-resource healthy-baseline report for a keeper X-DEC model."
    )
    parser.add_argument("--model", required=True, help="Path to the keeper checkpoint (.pt).")
    parser.add_argument("--data", required=True, help="Healthy capture URI or path.")
    parser.add_argument(
        "--profile", required=True, help="Feature profile (see config/features.yaml)."
    )
    parser.add_argument(
        "--threshold",
        type=_positive_float,
        action="append",
        default=None,
        help="Anomaly threshold to score (must be > 0; repeatable; omit for baselines only).",
    )
    parser.add_argument(
        "--sustain",
        type=int,
        default=3,
        help="Consecutive anomalous windows required for a sustained run (default: 3).",
    )
    parser.add_argument(
        "--threshold-quantile",
        type=float,
        default=0.99,
        help="Per-resource quantile reported as the own healthy baseline (default: 0.99).",
    )
    parser.add_argument(
        "--format", choices=["parquet", "csv"], default=None, help="Override the file format."
    )
    parser.add_argument("--output", default=None, help="Optional path to write the JSON report.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    thresholds = list(dict.fromkeys(args.threshold or []))

    try:
        summary = analyze(
            args.model,
            args.data,
            args.profile,
            thresholds=thresholds,
            sustain=args.sustain,
            threshold_quantile=args.threshold_quantile,
            data_format=args.format,
        )
    except (ValueError, FileNotFoundError) as exc:  # no windows / missing model or capture
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print_report(summary)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(summary, indent=2))
        print(f"summary written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
