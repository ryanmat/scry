# Description: Detection primitives for the evaluation harness: sustained anomaly runs.
# Description: Torch-free; promoted from scripts/validate_incident.py so both share one implementation.

"""Sustained-run detection primitives.

``anomaly_runs`` is the single implementation of run extraction over per-window
anomaly flags; the incident-validation harness aliases it. ``select_detection``
turns time-ordered sustained runs into a detection verdict against an incident
onset under either selection mode. Everything here is importable without torch.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

SUSTAIN_DEFAULT: int = 3

DetectionMode = Literal["lookback", "no_bridging"]


@dataclass(frozen=True)
class DetectionResult:
    """Detection verdict for one incident onset.

    ``lead_seconds`` is onset minus detection time; positive means early.
    ``bridged`` reports a sustained run that starts pre-onset and is still
    active at onset, whichever mode selected the detection.
    """

    detected: bool
    detection_time: pd.Timestamp | None
    lead_seconds: float | None
    bridged: bool
    n_runs_pre_onset: int
    n_runs_at_or_after: int


def select_detection(
    spans: list[tuple[pd.Timestamp, pd.Timestamp]],
    scan_ts: pd.DatetimeIndex,
    onset: pd.Timestamp,
    mode: DetectionMode,
) -> DetectionResult:
    """Select the detection for ``onset`` from time-ordered sustained runs.

    mode="lookback" (legacy harness semantics): a run is leading when it starts
    at or before onset and its last window ends within one median inter-window
    step of onset; the earliest-starting leading run is credited from its START.
    With no leading run, the earliest run ending at or after onset is a late
    detection with non-positive lead. ``detected=True`` therefore does not imply
    early warning.

    mode="no_bridging": runs partition strictly by START (pre_onset when start
    < onset, at_or_after when start >= onset); detected iff at_or_after is
    non-empty, credited at the first at_or_after start. A run that begins
    pre-onset and continues past onset is never a detection; it sets
    ``bridged``, which is reported, not silently skipped.

    Args:
        spans: Sustained anomalous runs as (start_time, end_time), time-ordered.
        scan_ts: The scanned window end-times (for the inter-window step).
        onset: The incident onset.
        mode: Selection semantics.

    Returns:
        The detection verdict; ``detection_time`` and ``lead_seconds`` are None
        when nothing qualifies.

    Raises:
        ValueError: On an unknown ``mode``.
    """
    if mode not in ("lookback", "no_bridging"):
        raise ValueError(f"unknown detection mode {mode!r}")

    n_pre = sum(1 for span in spans if span[0] < onset)
    n_after = len(spans) - n_pre
    bridged = any(span[0] < onset and span[1] >= onset for span in spans)

    detection_time: pd.Timestamp | None = None
    if mode == "lookback":
        step = (
            pd.Timedelta(np.median(np.diff(scan_ts.values)))
            if len(scan_ts) > 1
            else pd.Timedelta(0)
        )
        leading = [span for span in spans if span[0] <= onset and span[1] >= onset - step]
        if leading:
            detection_time = min(leading, key=lambda span: span[0])[0]
        else:
            after = [span for span in spans if span[1] >= onset]
            if after:
                detection_time = min(after, key=lambda span: span[0])[0]
    else:
        at_or_after = [span for span in spans if span[0] >= onset]
        if at_or_after:
            detection_time = min(at_or_after, key=lambda span: span[0])[0]

    if detection_time is None:
        return DetectionResult(False, None, None, bridged, n_pre, n_after)
    return DetectionResult(
        detected=True,
        detection_time=detection_time,
        lead_seconds=float((onset - detection_time).total_seconds()),
        bridged=bridged,
        n_runs_pre_onset=n_pre,
        n_runs_at_or_after=n_after,
    )


def slice_stats(
    errors: np.ndarray,
    end_times: pd.DatetimeIndex,
    start: pd.Timestamp,
    end: pd.Timestamp,
    thresholds: Sequence[float],
    sustain: int,
) -> dict[str, Any]:
    """Error statistics and sustained-run counts for one resource over a slice.

    Operates on ONE resource's arrays. Windows are ordered by end time with a
    stable sort, so duplicate end times keep input order and the run counts are
    deterministic. The slice is inclusive on both bounds.

    Returns:
        ``n_windows_in_slice``, ``max_error``/``mean_error`` (None on an empty
        slice, never NaN), and one ``over_{threshold:.4f}`` entry per requested
        threshold with ``windows_over``, ``frac_over``, and ``sustained_runs``.
        Every key is present for every threshold regardless of slice content.
    """
    order = np.argsort(end_times.values, kind="stable")
    ordered_errors = np.asarray(errors)[order]
    ordered_ends = end_times[order]
    in_slice = (ordered_ends >= start) & (ordered_ends <= end)
    slice_errors = ordered_errors[in_slice]

    stats: dict[str, Any] = {
        "n_windows_in_slice": int(in_slice.sum()),
        "max_error": float(slice_errors.max()) if slice_errors.size else None,
        "mean_error": float(slice_errors.mean()) if slice_errors.size else None,
    }
    for threshold in thresholds:
        flags = slice_errors > threshold
        runs = anomaly_runs(flags, sustain) if slice_errors.size else []
        stats[f"over_{threshold:.4f}"] = {
            "windows_over": int(flags.sum()) if slice_errors.size else 0,
            "frac_over": float(flags.mean()) if slice_errors.size else 0.0,
            "sustained_runs": len(runs),
        }
    return stats


def anomaly_runs(flags: np.ndarray, sustain: int) -> list[tuple[int, int]]:
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
