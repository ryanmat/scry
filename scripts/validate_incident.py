#!/usr/bin/env python3
# Description: Incident-validation harness for a trained keeper X-DEC model.
# Description: Scores per-window reconstruction error and reports detection lead time.

"""Validate a keeper (normal-behavior) X-DEC model against an incident capture.

The harness loads a trained keeper model, windows an incident capture in the
model's own feature order, normalizes those windows with the checkpoint's stored
parameters (never re-fitting on incident data), and computes a per-window
reconstruction error from the deterministic latent mean. A detection threshold is
derived from healthy windows only -- either the windows that end strictly before
the earliest labeled incident, or a separate reference capture -- so a large
incident can never inflate its own threshold, and the false-positive rate is
measured on a healthy set held out from the threshold fit (when the model also
trained on that capture, the FPR remains in-sample for the model; an incident
capture the model never saw gives a fully out-of-sample result). For each
labeled incident the harness reports
the sustained anomaly that leads into onset within a bounded look-back horizon and
its lead time, where a positive value means the alarm began before onset.

Examples:
    python scripts/validate_incident.py \\
        --model models/aro_keeper.pt \\
        --data data/captures/aro_incident.parquet \\
        --profile aro_node \\
        --labels incidents.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scry.data.feature_engineering import set_active_profile
from scry.data.fetcher import fetch_full_capture
from scry.data.quality import missing_features
from scry.data.windowing import WindowSet, build_windows
from scry.model.checkpoint import Keeper, load_keeper
from scry.model.reconstruction import reconstruction_errors, time_split
from scry.utils.config import get_config


@dataclass
class Incident:
    """A labeled incident window for one resource."""

    resource_id: str
    type: str
    start: pd.Timestamp
    end: pd.Timestamp


def load_incidents(labels_path: str) -> list[Incident]:
    """Load and parse the labeled incidents from a JSON file.

    Args:
        labels_path: Path to a JSON list of ``{resource_id, type, start, end}``.

    Returns:
        Parsed incidents with UTC timestamps.

    Raises:
        ValueError: If the file is not a non-empty list of well-formed entries.
    """
    with open(labels_path) as handle:
        raw = json.load(handle)
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"Labels file {labels_path} must be a non-empty JSON list.")

    incidents: list[Incident] = []
    for entry in raw:
        missing = [k for k in ("resource_id", "type", "start", "end") if k not in entry]
        if missing:
            raise ValueError(f"Incident entry missing keys {missing}: {entry}")
        incidents.append(
            Incident(
                resource_id=str(entry["resource_id"]),
                type=str(entry["type"]),
                start=pd.to_datetime(entry["start"], utc=True),
                end=pd.to_datetime(entry["end"], utc=True),
            )
        )
    return incidents


def _anomaly_runs(flags: np.ndarray, sustain: int) -> list[tuple[int, int]]:
    """Maximal runs of consecutive True flags with length >= ``sustain``.

    Args:
        flags: Boolean array of per-window anomaly flags, in time order.
        sustain: Minimum run length to qualify.

    Returns:
        List of (start_index, end_index) inclusive, one per qualifying run.
    """
    runs: list[tuple[int, int]] = []
    i, n = 0, len(flags)
    while i < n:
        if flags[i]:
            j = i
            while j + 1 < n and flags[j + 1]:
                j += 1
            if j - i + 1 >= sustain:
                runs.append((i, j))
            i = j + 1
        else:
            i += 1
    return runs


def _select_detection(
    spans: list[tuple[pd.Timestamp, pd.Timestamp]],
    scan_ts: pd.DatetimeIndex,
    onset: pd.Timestamp,
) -> pd.Timestamp | None:
    """Pick the detection time for the alarm that leads into onset, if any.

    A sustained run counts as leading into onset only when it is still active at
    onset: it starts at or before onset and its last window ends within one
    inter-window step of onset, so a run that recovers before onset is not
    credited. Such a run is traced back to its start. If no run reaches onset but
    one extends into or past it, that is a late detection. A run that both starts
    and ends before onset (a recovered blip) is ignored.

    Args:
        spans: Sustained anomalous runs as (start_time, end_time), time-ordered.
        scan_ts: The scanned window end-times (for the inter-window step).
        onset: The incident onset.

    Returns:
        The detection start-time, or None when no run leads into or follows onset.
    """
    if not spans:
        return None
    step = pd.Timedelta(np.median(np.diff(scan_ts.values))) if len(scan_ts) > 1 else pd.Timedelta(0)
    leading = [span for span in spans if span[0] <= onset and span[1] >= onset - step]
    if leading:
        return min(leading, key=lambda span: span[0])[0]
    after = [span for span in spans if span[1] >= onset]
    if after:
        return min(after, key=lambda span: span[0])[0]
    return None


def evaluate_incidents(
    errors: np.ndarray,
    resource_ids: np.ndarray,
    end_times: pd.DatetimeIndex,
    incidents: list[Incident],
    threshold: float,
    sustain: int,
    max_leadtime: pd.Timedelta,
) -> list[dict[str, Any]]:
    """Detect each incident and compute the lead time of the alarm that leads into it.

    For each incident, only that resource's windows in the look-back horizon
    ``[onset - max_leadtime, incident.end]`` are scanned. Detection is the
    sustained anomalous run (>= ``sustain`` consecutive windows over
    ``threshold``) that is still active at onset -- it starts at or before onset
    and ends within one inter-window step of it -- traced back to its start; lead
    time is then the incident start minus that start (positive means the alarm
    began before onset). A run that recovers before onset is a causally-unrelated
    blip and is not credited. If no run reaches onset but one extends into or past
    it, that is a late detection (non-positive lead time). See ``_select_detection``.

    Args:
        errors: Per-window reconstruction errors.
        resource_ids: Per-window resource ids.
        end_times: Per-window end timestamps (UTC).
        incidents: Labeled incidents.
        threshold: Anomaly threshold.
        sustain: Sustained-anomaly run length.
        max_leadtime: Look-back horizon before onset to consider for detection.

    Returns:
        One result dict per incident.
    """
    results: list[dict[str, Any]] = []
    for incident in incidents:
        resource_mask = resource_ids == incident.resource_id
        # A run beginning before the horizon is clamped to start at horizon_start
        # within the scan, so it errs toward under-detection (lead capped at
        # max_leadtime), never toward a fabricated early warning.
        horizon_start = incident.start - max_leadtime
        scan_mask = resource_mask & (end_times >= horizon_start) & (end_times <= incident.end)

        result: dict[str, Any] = {
            "resource_id": incident.resource_id,
            "type": incident.type,
            "detected": False,
            "first_detection_time": None,
            "lead_time_seconds": None,
            "max_error_in_window": None,
        }

        if scan_mask.any():
            scan_ts = end_times[scan_mask]
            order = np.argsort(scan_ts.values, kind="stable")
            scan_ts = scan_ts[order]
            flags = errors[scan_mask][order] > threshold
            spans = [(scan_ts[s], scan_ts[e]) for s, e in _anomaly_runs(flags, sustain)]
            detection_time = _select_detection(spans, scan_ts, incident.start)
            if detection_time is not None:
                result["detected"] = True
                result["first_detection_time"] = detection_time.isoformat()
                result["lead_time_seconds"] = (incident.start - detection_time).total_seconds()

        in_window = resource_mask & (end_times >= incident.start) & (end_times <= incident.end)
        if in_window.any():
            result["max_error_in_window"] = float(errors[in_window].max())

        results.append(result)
    return results


def compute_threshold(
    errors: np.ndarray,
    end_times: pd.DatetimeIndex,
    incidents: list[Incident],
    reference_errors: np.ndarray | None,
    quantile: float,
    gap: int,
) -> tuple[float, str, np.ndarray, np.ndarray]:
    """Derive the anomaly threshold from healthy windows, held out for the FPR.

    The threshold is fit on a healthy "fit" set and the false-positive rate is
    measured on a disjoint healthy "eval" set, so the reported FPR is
    out-of-sample for the threshold fit (it is out-of-sample for the model too
    only when the model never trained on the capture supplying the eval
    windows). With a ``--reference`` capture the threshold is fit on the
    reference and evaluated on the main capture's pre-incident windows. Otherwise
    the pre-incident windows (those ending strictly before the earliest incident)
    are time-split: the earlier half fits the threshold and the later half
    evaluates it, with a ``gap`` of windows dropped between them so the two sets
    share no raw samples. The incident windows are never used, so a large incident cannot
    inflate its own threshold. The temporal split assumes the pre-onset windows
    are healthy; a precursor that begins before the labeled onset would leak into
    it, so prefer ``--reference`` when pre-onset drift is expected.

    Args:
        errors: Per-window errors for the incident capture.
        end_times: Per-window end timestamps.
        incidents: Labeled incidents (for the temporal split).
        reference_errors: Optional healthy errors from a reference capture.
        quantile: Threshold quantile over the fit errors.
        gap: Windows to drop between the fit and eval halves (the window overlap).

    Returns:
        Tuple of (threshold, source, fit_errors, eval_errors). ``eval_errors`` may
        be empty when no held-out healthy windows are available.

    Raises:
        ValueError: If there are no healthy windows to fit the threshold.
    """
    earliest = min(incident.start for incident in incidents)
    pre_mask = end_times < earliest
    pre_errors = errors[pre_mask]
    pre_times = end_times[pre_mask]

    if reference_errors is not None:
        source = "reference"
        fit = reference_errors
        eval_ = pre_errors  # out-of-sample relative to the reference; may be empty
    else:
        source = "healthy_split"
        if pre_errors.size == 0:
            raise ValueError(
                "No healthy windows available to fit the threshold: no capture "
                "windows end before the earliest incident start. Provide a "
                "--reference healthy capture instead."
            )
        fit, eval_ = time_split(pre_errors, pre_times, gap=gap)

    if fit.size == 0:
        raise ValueError("No healthy windows available to fit the threshold.")

    threshold = float(np.quantile(fit, quantile))
    return threshold, source, fit, eval_


def _windows_for_keeper(
    df_long: pd.DataFrame, keeper: Keeper, seq_len: int, step: int
) -> WindowSet:
    """Window a capture for a loaded keeper, passing its schema and stored normalization."""
    return build_windows(
        df_long,
        numerical_features=keeper.numerical_features,
        categorical_features=keeper.categorical_features,
        normalization=keeper.normalization,
        cat_normalization=keeper.cat_normalization,
        seq_len=seq_len,
        step=step,
    )


def _warn_missing_model_features(df_long: pd.DataFrame, keeper: Keeper, source: str) -> None:
    """Warn when a capture lacks features the checkpoint trained on.

    The profile filter strips metrics the live profile no longer lists, and
    windowing fills absent features with neutral values, so a checkpoint from
    an older profile definition scores silently differently. A name-only
    profile comparison cannot catch this.
    """
    missing = missing_features(df_long, keeper.numerical_features)
    if missing:
        print(
            f"warning: {source} lacks {len(missing)} feature(s) the checkpoint was "
            f"trained on ({', '.join(missing)}); they window as neutral values, so "
            "scores are not comparable to the checkpoint's calibration.",
            file=sys.stderr,
        )


def analyze(
    model_path: str,
    data: str,
    labels_path: str,
    profile: str,
    *,
    threshold_quantile: float = 0.99,
    sustain: int = 3,
    max_leadtime_seconds: float = 7200.0,
    reference: str | None = None,
    data_format: str | None = None,
) -> dict[str, Any]:
    """Run the full incident-validation analysis and return the summary.

    Args:
        model_path: Path to the keeper checkpoint.
        data: Incident capture URI or path.
        labels_path: Path to the incident labels JSON.
        profile: Feature profile for the capture.
        threshold_quantile: Healthy-window quantile for the threshold.
        sustain: Consecutive anomalous windows required for a detection.
        max_leadtime_seconds: Look-back horizon before onset for detection.
        reference: Optional healthy reference capture URI/path.
        data_format: Optional explicit file format override.

    Returns:
        The summary dict (also suitable for JSON serialization).
    """
    keeper = load_keeper(model_path)
    incidents = load_incidents(labels_path)

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
    _warn_missing_model_features(df_long, keeper, data)
    windows = _windows_for_keeper(df_long, keeper, seq_len, step)
    if windows.x_num.shape[0] == 0:
        raise ValueError(
            f"Capture {data} produced no windows for profile '{profile}'. "
            "Check the time range, profile, and that the metrics are present."
        )
    errors = reconstruction_errors(keeper.model, windows.x_num, windows.x_cat, keeper.device)

    reference_errors: np.ndarray | None = None
    if reference is not None:
        ref_long = asyncio.run(
            fetch_full_capture(reference, profile=profile, data_format=data_format)
        )
        _warn_missing_model_features(ref_long, keeper, reference)
        ref_windows = _windows_for_keeper(ref_long, keeper, seq_len, step)
        if ref_windows.x_num.shape[0] == 0:
            raise ValueError(
                f"Reference capture {reference} produced no windows for profile '{profile}'."
            )
        reference_errors = reconstruction_errors(
            keeper.model, ref_windows.x_num, ref_windows.x_cat, keeper.device
        )

    # Drop a gap spanning the window overlap (ceil(seq_len / step) windows) between
    # the fit and eval halves so the held-out FPR shares no raw samples with the fit.
    overlap_gap = -(-seq_len // step)
    threshold, source, fit_errors, eval_errors = compute_threshold(
        errors, windows.end_times, incidents, reference_errors, threshold_quantile, overlap_gap
    )
    healthy_fpr = float(np.mean(eval_errors > threshold)) if eval_errors.size else None

    incident_results = evaluate_incidents(
        errors,
        windows.resource_ids,
        windows.end_times,
        incidents,
        threshold,
        sustain,
        pd.Timedelta(seconds=max_leadtime_seconds),
    )

    return {
        "threshold": threshold,
        "threshold_source": source,
        "threshold_quantile": threshold_quantile,
        "sustain": sustain,
        "max_leadtime_seconds": max_leadtime_seconds,
        "n_windows": int(errors.size),
        "n_threshold_windows": int(fit_errors.size),
        "n_eval_windows": int(eval_errors.size),
        "healthy_fpr": healthy_fpr,
        "incidents": incident_results,
    }


def write_plot(
    plot_path: str,
    errors: np.ndarray,
    end_times: pd.DatetimeIndex,
    threshold: float,
    incidents: list[Incident],
) -> bool:
    """Write a reconstruction-error timeline with the threshold and incident spans.

    Degrades gracefully: if matplotlib is not installed, prints a note and
    returns False instead of raising.

    Args:
        plot_path: Output image path.
        errors: Per-window errors.
        end_times: Per-window end timestamps.
        threshold: Anomaly threshold.
        incidents: Labeled incidents to shade.

    Returns:
        True if the figure was written, else False.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("note: matplotlib not installed; skipping --plot.", file=sys.stderr)
        return False

    order = np.argsort(end_times.values, kind="stable")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(end_times[order], errors[order], linewidth=0.8, label="reconstruction error")
    ax.axhline(threshold, color="red", linestyle="--", label="threshold")
    for incident in incidents:
        ax.axvspan(incident.start, incident.end, color="orange", alpha=0.2)
    ax.set_xlabel("window end time (UTC)")
    ax.set_ylabel("MSE reconstruction error")
    ax.set_title("Incident validation: per-window reconstruction error")
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    return True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate a keeper X-DEC model against an incident capture."
    )
    parser.add_argument("--model", required=True, help="Path to the keeper checkpoint (.pt).")
    parser.add_argument("--data", required=True, help="Incident capture URI or path.")
    parser.add_argument(
        "--labels",
        required=True,
        help="JSON list of incidents: {resource_id, type, start, end} (UTC ISO8601).",
    )
    parser.add_argument(
        "--profile", required=True, help="Feature profile (see config/features.yaml)."
    )
    parser.add_argument(
        "--threshold-quantile",
        type=float,
        default=0.99,
        help="Healthy-window quantile for the anomaly threshold (default: 0.99).",
    )
    parser.add_argument(
        "--sustain",
        type=int,
        default=3,
        help="Consecutive anomalous windows required for a detection (default: 3).",
    )
    parser.add_argument(
        "--max-leadtime",
        type=float,
        default=7200.0,
        help="Look-back horizon in seconds before onset to credit a detection (default: 7200).",
    )
    parser.add_argument(
        "--reference",
        default=None,
        help="Optional healthy reference capture for the threshold instead of a temporal split.",
    )
    parser.add_argument(
        "--format", choices=["parquet", "csv"], default=None, help="Override the file format."
    )
    parser.add_argument("--plot", default=None, help="Optional path to write a timeline figure.")
    parser.add_argument("--output", default=None, help="Optional path to write the JSON summary.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)

    summary = analyze(
        args.model,
        args.data,
        args.labels,
        args.profile,
        threshold_quantile=args.threshold_quantile,
        sustain=args.sustain,
        max_leadtime_seconds=args.max_leadtime,
        reference=args.reference,
        data_format=args.format,
    )

    payload = json.dumps(summary, indent=2)
    print(payload)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(payload)

    if args.plot:
        keeper = load_keeper(args.model)
        set_active_profile(args.profile)
        seq_len = int(keeper.config["seq_len"])
        step = int(get_config().window_step)
        df_long = asyncio.run(
            fetch_full_capture(args.data, profile=args.profile, data_format=args.format)
        )
        windows = _windows_for_keeper(df_long, keeper, seq_len, step)
        errors = reconstruction_errors(keeper.model, windows.x_num, windows.x_cat, keeper.device)
        incidents = load_incidents(args.labels)
        write_plot(args.plot, errors, windows.end_times, summary["threshold"], incidents)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
