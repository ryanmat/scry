# Description: The corrected margin sweep: named baselines, per-week rates, the arms, and run_sweep.
# Description: Torch-free at import; the gapped split is deferred to scry.eval.scoring inside the call.

"""Margin-sweep baselines, per-week arithmetic, and the sweep run.

The corrected sweep never interchanges its two thresholds. ``q_deploy`` is the
quantile over every healthy window -- exactly what the bake persists, in-sample
within the week. ``q_fit`` is the quantile over the gapped earlier half, whose
held-out later half is the only out-of-sample false-positive estimate inside one
week. ``compute_baselines`` reports both plus ``q_eval`` (the held-out half's
quantile) and the two drift ratios, and computes the fit/eval halves through the
shared ``per_resource_time_split`` primitive so a single resource's baselines are
bit-identical to the pooled ``time_split``.

Per-week rates come from the OBSERVED span, ``7.0 * count / span_days``, never a
hardcoded weekly extrapolation, so a capture shorter or longer than seven days
is scaled by its own duration.

``ARM_BASELINES`` names which baseline each sweep arm reads and ``sweep_arms``
is the code path for it: an arm's per-resource threshold is the margin times
that named baseline, scanned over the healthy holdout, the whole healthy week
in sample, or the capture day. ``run_sweep`` composes the whole sweep and
reports detection at BOTH baselines, so a reader can never mistake the
in-sample deployed threshold for the out-of-sample fit one, and carries
``CAVEATS`` in every result: the sweep's own measured defects are part of its
output, not a footnote in a README.

This module does pure numpy/pandas arithmetic over already-computed errors and
imports torch-free; the split primitive lives in the torch-heavy scoring module,
so it is imported lazily inside the functions that need it, keeping ``import
scry.eval`` from pulling torch.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from scry.eval.detection import DetectionMode, anomaly_runs, select_detection

if TYPE_CHECKING:
    from scry.eval.candidate import Candidate, ScoreSet
    from scry.eval.labels import LabelSet
    from scry.eval.scoring import ScoringGrid

ARM_BASELINES: dict[str, str] = {
    "healthy_holdout": "q_fit",
    "healthy_insample": "q_deploy",
    "capture_day": "q_deploy",
    "detection_at_deploy": "q_deploy",
    "detection_at_fit": "q_fit",
}
"""Which baseline each sweep arm reads. The detection arms are consumed by ``run_sweep``."""

DETECTION_ARMS: tuple[str, ...] = tuple(
    arm for arm in ARM_BASELINES if arm.startswith("detection_at_")
)
"""The arms ``run_sweep`` reports detection for, one per baseline."""

DETECTION_MODE: DetectionMode = "no_bridging"
"""The sweep's selection mode: a run bridging the onset is reported, never credited."""

CAVEATS: tuple[str, ...] = (
    "At the deployed baseline (q_deploy) the held-out false-positive rate is 0.00 runs/wk "
    "at both m=1.75 and m=2.0, erasing the knee the original margin selection turned on.",
    "Alert raises are non-monotone in m (4.02/wk at m=2.5 vs 9.06/wk at m=3.0 on Jul 21-27): "
    "a higher threshold fragments one long excursion into more accounted raises.",
    "The flat +12 min detection lead across m in [1.5, 4.0] is a 20-minute-stride artifact; "
    "at the native 2-minute resolution lead is monotone in m (28.0 min at m=2.0, 18.0 at m=4.0).",
)
"""The DESIGN-addendum caveats every sweep result carries."""


def compute_baselines(
    errors: np.ndarray,
    end_times: pd.DatetimeIndex,
    *,
    quantile: float,
    gap: int,
) -> dict[str, float]:
    """Named healthy-error baselines for one resource.

    Args:
        errors: One resource's per-window errors.
        end_times: Matching per-window end timestamps.
        quantile: The threshold quantile (e.g. 0.99).
        gap: Windows dropped between the fit and eval halves; callers pass
            ``ceil(seq_len / step)`` so the held-out half shares no raw samples
            with the fit half.

    Returns:
        Dict with exactly ``q_deploy`` (quantile over every window), ``q_fit``
        (quantile over the gapped earlier half), ``q_eval`` (quantile over the
        held-out later half), ``drift_x`` = ``q_eval / q_fit``, and
        ``deploy_over_fit_x`` = ``q_deploy / q_fit``.
    """
    from scry.eval.scoring import per_resource_time_split

    errors = np.asarray(errors, dtype=np.float64)
    resource_ids = np.zeros(errors.shape[0], dtype=np.int64)
    fit, held_out = per_resource_time_split(errors, end_times, resource_ids, gap=gap)

    q_fit = float(np.quantile(fit, quantile))
    q_eval = float(np.quantile(held_out, quantile))
    q_deploy = float(np.quantile(errors, quantile))
    return {
        "q_fit": q_fit,
        "q_eval": q_eval,
        "q_deploy": q_deploy,
        "drift_x": q_eval / q_fit,
        "deploy_over_fit_x": q_deploy / q_fit,
    }


def per_week(count: int, end_times: pd.DatetimeIndex) -> float:
    """Events per seven-day week from the OBSERVED span of ``end_times``.

    ``7.0 * count / span_days`` where ``span_days`` is the wall-clock span of the
    capture (last end time minus first), not a hardcoded seven-day assumption.

    Args:
        count: Number of events (e.g. sustained runs or alert raises) observed.
        end_times: The scanned window end-times whose min/max bound the span.

    Returns:
        The per-week rate scaled by the observed span.
    """
    span_days = (end_times.max() - end_times.min()) / pd.Timedelta(days=1)
    return 7.0 * count / span_days


def _ordered(
    errors: np.ndarray, end_times: pd.DatetimeIndex
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """One resource's windows in time order, stably sorted as the split primitive sorts."""
    order = np.argsort(end_times.values, kind="stable")
    return np.asarray(errors, dtype=np.float64)[order], end_times[order]


def _holdout(
    errors: np.ndarray, end_times: pd.DatetimeIndex, gap: int
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """One resource's gapped held-out half, from ``per_resource_time_split``.

    Its end times are the tail of the resource's time-ordered ends, sized by
    the length the primitive returns.
    """
    from scry.eval.scoring import per_resource_time_split

    ordered_errors, ordered_ends = _ordered(errors, end_times)
    resource_ids = np.zeros(ordered_errors.shape[0], dtype=np.int64)
    _, held_out = per_resource_time_split(ordered_errors, ordered_ends, resource_ids, gap=gap)
    return held_out, ordered_ends[len(ordered_ends) - len(held_out) :]


def _threshold(baselines: Mapping[str, Mapping[str, float]], resource_id: str, arm: str) -> float:
    """The unscaled baseline this arm reads for one resource."""
    if resource_id not in baselines:
        raise ValueError(
            f"no baselines for scanned resource {resource_id!r}; "
            f"per_resource_baselines covers {sorted(baselines)}"
        )
    return float(baselines[resource_id][ARM_BASELINES[arm]])


def sweep_arms(
    healthy: Mapping[str, tuple[np.ndarray, pd.DatetimeIndex]],
    capture: Mapping[str, tuple[np.ndarray, pd.DatetimeIndex]],
    *,
    per_resource_baselines: Mapping[str, Mapping[str, float]],
    margins: Sequence[float],
    sustain: int,
    gap: int,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Sustained runs and raises per margin for the three rate arms.

    Every scored resource is scanned regardless of role, at its own threshold:
    the margin times the baseline ``ARM_BASELINES`` names for the arm. Runs are
    maximal over-threshold stretches of at least ``sustain`` windows; raises are
    the same count at accounting 1, so the two rates differ where an excursion
    fragments.

    Args:
        healthy: Per-resource already-scored ``(errors, end_times)`` for the
            healthy week, keyed by resource id.
        capture: Per-resource ``(errors, end_times)`` for the capture day.
        per_resource_baselines: Resource id -> the ``compute_baselines`` dict.
        margins: Threshold multipliers; each is keyed ``str(float(margin))``.
        sustain: Windows a run must hold to count.
        gap: Windows dropped between the healthy fit and holdout halves.

    Returns:
        Arm name -> margin key -> ``{n_runs, runs_per_week, raises_per_week,
        span_days, threshold_by_resource}``. An arm spanning no time reports
        both rates as None with ``span_days`` 0.0 rather than dividing by zero.

    Raises:
        ValueError: A scanned resource has no entry in ``per_resource_baselines``.
    """
    scans = {
        "healthy_holdout": {rid: _holdout(e, t, gap) for rid, (e, t) in healthy.items()},
        "healthy_insample": {rid: _ordered(e, t) for rid, (e, t) in healthy.items()},
        "capture_day": {rid: _ordered(e, t) for rid, (e, t) in capture.items()},
    }
    arms: dict[str, dict[str, dict[str, Any]]] = {}
    for arm, scanned in scans.items():
        end_values = [t.values for _, t in scanned.values()]
        pooled_ends = pd.DatetimeIndex(np.concatenate(end_values) if end_values else [])
        # An arm that scanned nothing, or one timestamp, spans no time.
        span = pooled_ends.max() - pooled_ends.min() if len(pooled_ends) else pd.Timedelta(0)
        span_days = float(span / pd.Timedelta(days=1))
        by_margin: dict[str, dict[str, Any]] = {}
        for margin in margins:
            thresholds = {
                rid: float(margin) * _threshold(per_resource_baselines, rid, arm) for rid in scanned
            }
            n_runs = 0
            n_raises = 0
            for rid, (errors, _) in scanned.items():
                flags = errors > thresholds[rid]
                n_runs += len(anomaly_runs(flags, sustain))
                n_raises += len(anomaly_runs(flags, 1))
            by_margin[str(float(margin))] = {
                "n_runs": n_runs,
                "runs_per_week": None if span_days == 0.0 else per_week(n_runs, pooled_ends),
                "raises_per_week": None if span_days == 0.0 else per_week(n_raises, pooled_ends),
                "span_days": span_days,
                "threshold_by_resource": thresholds,
            }
        arms[arm] = by_margin
    return arms


def _by_resource(scores: ScoreSet) -> dict[str, tuple[np.ndarray, pd.DatetimeIndex]]:
    """Slice a ScoreSet into one time-ordered ``(errors, end_times)`` pair per resource."""
    errors = np.asarray(scores.errors, dtype=np.float64)
    resource_ids = np.asarray(scores.resource_ids).astype(str)
    sliced: dict[str, tuple[np.ndarray, pd.DatetimeIndex]] = {}
    for rid in sorted(set(resource_ids.tolist())):
        mask = resource_ids == rid
        sliced[rid] = _ordered(errors[mask], scores.end_times[mask])
    return sliced


def _detect(
    errors: np.ndarray,
    end_times: pd.DatetimeIndex,
    *,
    threshold: float,
    onset: pd.Timestamp,
    end: pd.Timestamp,
    sustain: int,
) -> dict[str, Any]:
    """One resource's no-bridging detection verdict, as JSON-ready fields.

    The scan starts at the resource's first scored capture window -- the sweep
    asks when the threshold would have fired, with no lead-time horizon clamp
    -- and is BOUNDED at the case's labeled ``end``, keeping the windows whose
    end time is at or before it exactly as the suite's scan does (``ends <=
    case.end``, ``scry.eval.metrics``). A run is therefore measured within the
    labeled window, and an excursion after the incident is over is never
    credited as its detection.
    """
    scanned = end_times <= end
    scan_ends = end_times[scanned]
    spans = [
        (scan_ends[i], scan_ends[j]) for i, j in anomaly_runs(errors[scanned] > threshold, sustain)
    ]
    verdict = select_detection(spans, scan_ends, onset, DETECTION_MODE)
    moment = verdict.detection_time
    return {
        "detected": verdict.detected,
        "detection_time": None if moment is None else moment.isoformat().replace("+00:00", "Z"),
        "lead_seconds": verdict.lead_seconds,
        "bridged": verdict.bridged,
        "n_runs_pre_onset": verdict.n_runs_pre_onset,
        "n_runs_at_or_after": verdict.n_runs_at_or_after,
    }


def run_sweep(
    candidate: Candidate,
    healthy: pd.DataFrame,
    capture: pd.DataFrame,
    labels: LabelSet,
    margins: Sequence[float],
    *,
    quantile: float,
    sustain: int,
    grid: ScoringGrid,
) -> dict[str, Any]:
    """Run the corrected margin sweep over one healthy week and one capture.

    The candidate scores both frames on ``grid``, each sliced into per-resource
    ``(errors, end_times)``. Baselines are fit per resource on the healthy week
    at ``gap = ceil(seq_len / step)`` from the ScoreSet meta, so the holdout
    half shares no raw samples with the fit half; the rate arms come from
    ``sweep_arms``; detection is reported for every labeled incident at BOTH
    baselines, per margin, anchored at the case's ``primary_onset`` (falling
    back to ``T0``) and scanned only to the case's labeled ``end``, as the
    suite's scan is.

    Args:
        candidate: The scorer; its ScoreSet meta supplies the model path,
            profile, ``seq_len``, and ``step`` the provenance and gap read.
        healthy: Canonical long-format healthy week.
        capture: Canonical long-format capture under evaluation.
        labels: The capture's labels; its incident cases drive detection.
        margins: Threshold multipliers, recorded verbatim and keyed
            ``str(float(margin))``.
        quantile: The baseline quantile (e.g. 0.99).
        sustain: Windows a run must hold to count.
        grid: The one scoring grid both frames are scored on.

    Returns:
        The spec-11 result dict with exactly ``provenance``, ``grid``,
        ``quantile``, ``sustain``, ``margins``, ``per_resource_baselines``,
        ``arm_baselines``, ``arms``, ``detection`` (arm -> margin key ->
        resource id -> the DetectionResult fields, ``detection_time`` as an
        ISO-8601 Z string), and ``caveats``. Provenance records no rubric
        (path ``""``, version 0) and no case data paths: run_sweep is handed
        frames, so the caller holding the paths records them.

    Raises:
        ValueError: A labeled incident resource has no capture windows, or a
            scanned resource has no healthy baselines (via ``sweep_arms``).
        IndexError: From ``compute_baselines`` when a healthy resource is too
            short to leave a held-out half (``n <= 2 * gap`` windows).
        ZeroDivisionError: From ``compute_baselines`` when a healthy
            resource's fit half is all zero. run_sweep guards neither case.
    """
    from scry.eval.provenance import build_provenance

    healthy_scores = candidate.score(healthy, grid)
    capture_scores = candidate.score(capture, grid)
    meta = healthy_scores.meta
    gap = math.ceil(int(meta["seq_len"]) / int(meta["step"]))

    healthy_by_resource = _by_resource(healthy_scores)
    capture_by_resource = _by_resource(capture_scores)
    per_resource_baselines = {
        rid: compute_baselines(errors, end_times, quantile=quantile, gap=gap)
        for rid, (errors, end_times) in healthy_by_resource.items()
    }

    # Anchors first: an incident with no capture windows is an error whatever
    # the margins are, not a silently empty detection entry.
    anchors: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for case in labels.incidents():
        if case.resource_id not in capture_by_resource:
            raise ValueError(
                f"no capture windows for incident resource {case.resource_id!r}; "
                f"the capture covers {sorted(capture_by_resource)}"
            )
        anchors[case.resource_id] = (case.onsets[case.primary_onset or "T0"], case.end)

    detection: dict[str, dict[str, dict[str, Any]]] = {}
    for arm in DETECTION_ARMS:
        detection[arm] = {
            str(float(margin)): {
                rid: _detect(
                    *capture_by_resource[rid],
                    threshold=float(margin) * _threshold(per_resource_baselines, rid, arm),
                    onset=onset,
                    end=end,
                    sustain=sustain,
                )
                for rid, (onset, end) in anchors.items()
            }
            for margin in margins
        }

    # The sweep evaluates no rubric and is handed frames, not paths.
    provenance = build_provenance(
        model_path=meta["model_path"],
        profile=meta["profile"],
        seq_len=int(meta["seq_len"]),
        grids={grid.label: grid},
        sustains=[sustain],
        detection_mode=DETECTION_MODE,
        policy_description={
            "type": "MarginSweep",
            "quantile": quantile,
            "sustain": sustain,
            "margins": list(margins),
            "arm_baselines": dict(ARM_BASELINES),
        },
        rubric_path="",
        rubric_version=0,
        case_data_paths={},
    )
    return {
        "provenance": provenance,
        # The grid block is the provenance entry, so the two can never disagree.
        "grid": provenance["grids"][grid.label],
        "quantile": quantile,
        "sustain": sustain,
        "margins": list(margins),
        "per_resource_baselines": per_resource_baselines,
        "arm_baselines": dict(ARM_BASELINES),
        "arms": sweep_arms(
            healthy_by_resource,
            capture_by_resource,
            per_resource_baselines=per_resource_baselines,
            margins=margins,
            sustain=sustain,
            gap=gap,
        ),
        "detection": detection,
        "caveats": list(CAVEATS),
    }
