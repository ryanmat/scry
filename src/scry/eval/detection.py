# Description: Detection primitives for the evaluation harness: sustained anomaly runs.
# Description: Torch-free; promoted from scripts/validate_incident.py so both share one implementation.

"""Sustained-run detection primitives.

``anomaly_runs`` is the single implementation of run extraction over per-window
anomaly flags; the incident-validation harness aliases it. ``select_detection``
turns time-ordered sustained runs into a detection verdict against an incident
onset under either selection mode. Everything here is importable without torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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
