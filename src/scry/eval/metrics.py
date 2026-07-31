# Description: Per-case metric bundles for the evaluation harness: schemas and their producer.
# Description: Torch-free; the sole producer of the bundles rubric.py evaluates.

"""Metric bundle schema and producer for the evaluation harness.

``ResourceMetrics`` carries one resource's numbers for one grid, grouped by
role: the incident group (detection and per-onset leads, lead-in FPR,
incident coverage), the healthy-reference group (duration-weighted
alarm-fatigue rates), and the negative-control group (slice stats at the
control's own and each incident resource's threshold). ``GridMetrics`` adds
the pooled and fleet aggregates for one grid; ``CaseMetrics`` keys
GridMetrics by grid label and carries the ``context`` conduit the
coverage_integrity and sanity gates read by documented key (unknown keys are
ignored, so producers may add keys without breaking an older rubric).

Validation runs in ``__post_init__``: every populated sustain-keyed dict
carries both the sustain-3 and the sustain-1 accounting (the two accountings
are fields, never variants), the incident and negative-control field groups
populate only under their role, and every grid key equals its grid's label.
The healthy-reference group is case-kind-driven, not role-gated: a
healthy-reference capture takes no labels, so its resources carry whatever
labels role applies.

``compute_case_metrics`` is the producer: it consumes ScoreSets, a LabelSet,
and resolved per-resource thresholds. Incident cases get one detection per
resource anchored at the rubric's onset anchor with arithmetic leads against
every named onset, lead-in FPR at each resource's own threshold for both
accountings with the pooled_ window-pooling aggregate, the amendment-5
incident-coverage numbers from alarm spans carrying the
one-effective-grid-step wall-duration rule, and declared controls sliced
over the incident's span at their own and the incident resource's
thresholds. Healthy-reference and control-only cases get the
duration-weighted alarm-fatigue metrics (the sweep's span_days and per_week
semantics) with fleet aggregates summing rates (None as 0.0) and dividing
summed alarm seconds by summed span seconds. Everything here is importable
without torch.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from scry.eval.detection import (
    DetectionMode,
    DetectionResult,
    anomaly_runs,
    select_detection,
    slice_stats,
)
from scry.eval.labels import ROLES, LabelCase, LabelSet

if TYPE_CHECKING:
    from scry.eval.candidate import ScoreSet
    from scry.eval.scoring import ScoringGrid

CASE_KINDS: tuple[str, ...] = ("incident", "healthy_reference", "negative_control")

# The two accountings every populated sustain-keyed dict must carry: the
# rubric's sustain-3 and the deployed sustain-1.
SUSTAIN_ACCOUNTINGS: tuple[int, ...] = (3, 1)


def _require_both_accountings(owner: str, name: str, values: Mapping[int, Any]) -> None:
    """Reject a populated sustain-keyed dict missing either accounting."""
    if not values:
        return
    missing = [sustain for sustain in SUSTAIN_ACCOUNTINGS if sustain not in values]
    if missing:
        raise ValueError(
            f"{owner} {name} carries sustains {sorted(values)} but every populated "
            "sustain-keyed dict must carry both sustain accountings 3 and 1"
        )


@dataclass(frozen=True)
class ResourceMetrics:
    """One resource's metrics for one grid; ``role`` gates the field groups."""

    resource_id: str
    role: str
    threshold: float
    threshold_source: str
    n_eval_windows: int
    # incident role
    detection: DetectionResult | None
    detection_time: pd.Timestamp | None
    lead_seconds_by_onset: dict[str, float | None]
    lead_in_fpr: dict[int, float | None]
    n_lead_in_windows: int
    coverage_fraction: float | None
    clear_lead_vs_end_s: float | None
    alarm_seconds_in_incident: float | None
    # healthy-reference group (case-kind-driven; allowed under any role)
    time_in_alarm_fraction: dict[int, float | None]
    raises_per_week: dict[int, float | None]
    runs_per_week: dict[int, float | None]
    sustained_run_counts: dict[int, int]
    observed_span_days: float | None
    # negative-control role
    slice_stats_by_threshold: dict[str, dict[str, Any]]
    exceedances_by_threshold: dict[str, int]

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"unknown role {self.role!r}; valid roles: {', '.join(ROLES)}")
        for name in (
            "lead_in_fpr",
            "time_in_alarm_fraction",
            "raises_per_week",
            "runs_per_week",
            "sustained_run_counts",
        ):
            _require_both_accountings(f"resource {self.resource_id!r}", name, getattr(self, name))
        if self.role != "incident":
            incident_values = {
                "detection": self.detection,
                "detection_time": self.detection_time,
                "lead_seconds_by_onset": self.lead_seconds_by_onset or None,
                "lead_in_fpr": self.lead_in_fpr or None,
                "n_lead_in_windows": self.n_lead_in_windows or None,
                "coverage_fraction": self.coverage_fraction,
                "clear_lead_vs_end_s": self.clear_lead_vs_end_s,
                "alarm_seconds_in_incident": self.alarm_seconds_in_incident,
            }
            populated = [name for name, value in incident_values.items() if value is not None]
            if populated:
                raise ValueError(
                    f"resource {self.resource_id!r} with role {self.role!r} populates "
                    f"incident field(s) {', '.join(populated)}; role is authoritative"
                )
        if self.role != "negative_control":
            populated = [
                name
                for name in ("slice_stats_by_threshold", "exceedances_by_threshold")
                if getattr(self, name)
            ]
            if populated:
                raise ValueError(
                    f"resource {self.resource_id!r} with role {self.role!r} populates "
                    f"negative-control field(s) {', '.join(populated)}; role is authoritative"
                )


@dataclass(frozen=True)
class GridMetrics:
    """One grid's per-resource metrics plus the pooled and fleet aggregates."""

    grid: ScoringGrid
    per_resource: dict[str, ResourceMetrics]
    pooled_lead_in_fpr: dict[int, float | None]
    fleet_time_in_alarm_fraction: dict[int, float | None]
    fleet_raises_per_week: dict[int, float | None]
    fleet_runs_per_week: dict[int, float | None]
    n_eval_windows: int
    vus_pr: dict[str, Any]

    def __post_init__(self) -> None:
        for name in (
            "pooled_lead_in_fpr",
            "fleet_time_in_alarm_fraction",
            "fleet_raises_per_week",
            "fleet_runs_per_week",
        ):
            _require_both_accountings(f"grid {self.grid.label!r}", name, getattr(self, name))


@dataclass(frozen=True)
class CaseMetrics:
    """One case's grid-keyed metric bundles; the only input rubric.py accepts.

    ``context`` is the flat conduit for what the coverage_integrity and sanity
    gates need and no other field carries (hygiene population name,
    per-resource eligibility verdicts, keeper profile, labelled resources
    present in the capture). Nothing computed in the metric definitions goes
    in it, and it serializes under the case's ``reported`` block, never under
    ``per_resource``.
    """

    case_id: str
    case_kind: str
    grids: dict[str, GridMetrics]
    context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.case_kind not in CASE_KINDS:
            raise ValueError(
                f"unknown case_kind {self.case_kind!r}; valid kinds: {', '.join(CASE_KINDS)}"
            )
        for label, grid_metrics in self.grids.items():
            if not label:
                raise ValueError(
                    f"case {self.case_id!r} keys a grid with an empty label; "
                    "grids are keyed by grid label"
                )
            if grid_metrics.grid.label != label:
                raise ValueError(
                    f"case {self.case_id!r} grid key {label!r} does not match the grid's "
                    f"label {grid_metrics.grid.label!r}"
                )


def _effective_step(grid: ScoringGrid, ends: pd.DatetimeIndex) -> pd.Timedelta:
    """One effective grid step: the cadence when the grid defines one, else the
    median inter-window spacing (zero when a single window leaves no spacing
    evidence, erring toward under-counting)."""
    if grid.cadence is not None:
        return grid.cadence
    if len(ends) > 1:
        return pd.Timedelta(np.median(np.diff(ends.values)))
    return pd.Timedelta(0)


def _covered_windows(flags: np.ndarray, sustain: int) -> int:
    """Windows inside maximal over-threshold runs of length >= ``sustain``."""
    return sum(end - start + 1 for start, end in anomaly_runs(flags, sustain))


def _vus_pr_placeholder() -> dict[str, Any]:
    """The vus_pr field until scry/eval/vus.py lands: null value, stated reason."""
    return {"value": None, "reason": "not implemented"}


def _resolved_threshold(thresholds: Mapping[str, tuple[float, str]], rid: str) -> tuple[float, str]:
    if rid not in thresholds:
        raise ValueError(f"no resolved threshold for resource {rid!r}")
    return thresholds[rid]


def _sorted_slice(
    errors: np.ndarray, resource_ids: np.ndarray, end_times: pd.DatetimeIndex, rid: str
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """One resource's errors and end times, stably ordered by end time."""
    mask = resource_ids == rid
    ends = end_times[mask]
    order = np.argsort(ends.values, kind="stable")
    return errors[mask][order], ends[order]


def _alarm_seconds(
    flags: np.ndarray, ends: pd.DatetimeIndex, sustain: int, step: pd.Timedelta
) -> float:
    """Total alarm wall duration: each run lasts last end - first end + one step."""
    return float(
        sum(((ends[j] - ends[i]) + step).total_seconds() for i, j in anomaly_runs(flags, sustain))
    )


def _incident_coverage(
    alarm_spans: list[tuple[pd.Timestamp, pd.Timestamp]],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[float | None, float, float | None]:
    """Coverage of ``[start, end]`` by alarm spans (amendment 5).

    Returns (coverage_fraction, alarm_seconds_in_incident, clear_lead_vs_end_s).
    ``clear_lead_vs_end_s`` is ``end`` minus the last clear among alarms
    overlapping the incident: positive means the alarm cleared early, negative
    means it outlived the incident; None when no alarm overlaps at all.
    """
    denominator = (end - start).total_seconds()
    alarm_seconds = 0.0
    last_clear: pd.Timestamp | None = None
    for raised, cleared in alarm_spans:
        overlap = (min(cleared, end) - max(raised, start)).total_seconds()
        if overlap <= 0:
            continue
        alarm_seconds += overlap
        if last_clear is None or cleared > last_clear:
            last_clear = cleared
    coverage = alarm_seconds / denominator if denominator > 0 else None
    clear_lead = float((end - last_clear).total_seconds()) if last_clear is not None else None
    return coverage, float(alarm_seconds), clear_lead


def _incident_resource_metrics(
    case: LabelCase,
    errors: np.ndarray,
    ends: pd.DatetimeIndex,
    threshold: float,
    source: str,
    step: pd.Timedelta,
    *,
    mode: DetectionMode,
    onset_anchor: str,
    sustain: int,
    lead_in: pd.Timedelta,
    max_leadtime: pd.Timedelta,
) -> tuple[ResourceMetrics, dict[int, int], int]:
    """One incident resource's metrics plus its lead-in pooling contributions.

    Returns (metrics, covered lead-in windows by sustain, lead-in window count).
    """
    if onset_anchor not in case.onsets:
        present = ", ".join(case.onsets) or "none"
        raise ValueError(
            f"onset_anchor {onset_anchor!r} absent for {case.resource_id!r}; "
            f"present onsets: {present}"
        )
    if case.primary_onset is None:
        raise ValueError(
            f"incident case {case.resource_id!r} has no primary_onset; the lead-in window needs one"
        )
    anchor_ts = case.onsets[onset_anchor]
    primary_ts = case.onsets[case.primary_onset]

    # Detection: EXACTLY ONE select_detection call, anchored at the rubric's
    # onset_anchor, scanning [anchor - max_leadtime, end] (the horizon clamp
    # errs toward under-detection, mirroring evaluate_incidents).
    scan = (ends >= anchor_ts - max_leadtime) & (ends <= case.end)
    scan_ts = ends[scan]
    scan_flags = errors[scan] > threshold
    spans = [(scan_ts[i], scan_ts[j]) for i, j in anomaly_runs(scan_flags, sustain)]
    detection = select_detection(spans, scan_ts, anchor_ts, mode)
    detection_time = detection.detection_time if detection.detected else None

    # Leads vs every named onset, ARITHMETICALLY from the one detection time.
    leads = {
        name: float((ts - detection_time).total_seconds()) if detection_time is not None else None
        for name, ts in case.onsets.items()
    }

    # Lead-in FPR over the resource's own threshold in [primary - lead_in,
    # primary): for accounting s, the fraction of lead-in windows inside
    # over-threshold runs of length >= s, with run length measured within the
    # slice (s=1 is the raw fraction; a boundary-straddling run counts only
    # its inside remnant, erring toward under-counting).
    lead_mask = (ends >= primary_ts - lead_in) & (ends < primary_ts)
    lead_flags = errors[lead_mask] > threshold
    n_lead_in = int(lead_mask.sum())
    covered_by_sustain: dict[int, int] = {}
    lead_in_fpr: dict[int, float | None] = {}
    for accounting in SUSTAIN_ACCOUNTINGS:
        covered = _covered_windows(lead_flags, accounting)
        covered_by_sustain[accounting] = covered
        lead_in_fpr[accounting] = covered / n_lead_in if n_lead_in else None

    # Incident coverage from alarm spans over the full scored series at the
    # rubric's sustain; a run spans [first end, last end + one grid step).
    full_flags = errors > threshold
    alarm_spans = [(ends[i], ends[j] + step) for i, j in anomaly_runs(full_flags, sustain)]
    coverage_fraction, alarm_seconds, clear_lead = _incident_coverage(
        alarm_spans, primary_ts, case.end
    )

    metrics = ResourceMetrics(
        resource_id=case.resource_id,
        role="incident",
        threshold=threshold,
        threshold_source=source,
        n_eval_windows=int(ends.size),
        detection=detection,
        detection_time=detection_time,
        lead_seconds_by_onset=leads,
        lead_in_fpr=lead_in_fpr,
        n_lead_in_windows=n_lead_in,
        coverage_fraction=coverage_fraction,
        clear_lead_vs_end_s=clear_lead,
        alarm_seconds_in_incident=alarm_seconds,
        time_in_alarm_fraction={},
        raises_per_week={},
        runs_per_week={},
        sustained_run_counts={
            accounting: len(anomaly_runs(full_flags, accounting))
            for accounting in SUSTAIN_ACCOUNTINGS
        },
        observed_span_days=None,
        slice_stats_by_threshold={},
        exceedances_by_threshold={},
    )
    return metrics, covered_by_sustain, n_lead_in


def _control_resource_metrics(
    control: LabelCase,
    incident: LabelCase,
    errors: np.ndarray,
    ends: pd.DatetimeIndex,
    threshold: float,
    source: str,
    incident_threshold: float,
    *,
    sustain: int,
    lead_in: pd.Timedelta,
) -> ResourceMetrics:
    """One control's metrics, sliced over the incident's ``[primary - lead_in, end]`` span.

    The control is scored with ``slice_stats`` at its own resolved threshold
    (key ``"own"``) and at the incident resource's (keyed by that resource id),
    so a control that would have alarmed on a neighbour's threshold is visible.
    """
    primary_ts = incident.onsets[incident.primary_onset]
    start = primary_ts - lead_in
    slice_stats_by_threshold: dict[str, dict[str, Any]] = {}
    exceedances: dict[str, int] = {}
    for key, at_threshold in (("own", threshold), (incident.resource_id, incident_threshold)):
        stats = slice_stats(errors, ends, start, incident.end, [at_threshold], sustain)
        slice_stats_by_threshold[key] = stats
        exceedances[key] = stats[f"over_{at_threshold:.4f}"]["windows_over"]

    full_flags = errors > threshold
    return ResourceMetrics(
        resource_id=control.resource_id,
        role="negative_control",
        threshold=threshold,
        threshold_source=source,
        n_eval_windows=int(ends.size),
        detection=None,
        detection_time=None,
        lead_seconds_by_onset={},
        lead_in_fpr={},
        n_lead_in_windows=0,
        coverage_fraction=None,
        clear_lead_vs_end_s=None,
        alarm_seconds_in_incident=None,
        time_in_alarm_fraction={},
        raises_per_week={},
        runs_per_week={},
        sustained_run_counts={
            accounting: len(anomaly_runs(full_flags, accounting))
            for accounting in SUSTAIN_ACCOUNTINGS
        },
        observed_span_days=None,
        slice_stats_by_threshold=slice_stats_by_threshold,
        exceedances_by_threshold=exceedances,
    )


def _incident_grid_metrics(
    scores: ScoreSet,
    labels: LabelSet,
    thresholds: Mapping[str, tuple[float, str]],
    *,
    mode: DetectionMode,
    onset_anchor: str,
    sustain: int,
    lead_in: pd.Timedelta,
    max_leadtime: pd.Timedelta,
) -> GridMetrics:
    """One grid's incident-case metrics: incident and control bundles plus the pooled lead-in FPR."""
    errors = np.asarray(scores.errors)
    resource_ids = np.asarray(scores.resource_ids).astype(str)
    incidents = labels.incidents()
    controls = labels.negative_controls()
    if controls and len(incidents) > 1:
        raise NotImplementedError(
            "negative-control slicing is defined against a single incident case; "
            f"the labels declare {len(incidents)}"
        )

    per_resource: dict[str, ResourceMetrics] = {}
    pooled_covered = dict.fromkeys(SUSTAIN_ACCOUNTINGS, 0)
    pooled_lead_in_windows = 0
    for case in incidents:
        rid = case.resource_id
        threshold, source = _resolved_threshold(thresholds, rid)
        errs_r, ends_r = _sorted_slice(errors, resource_ids, scores.end_times, rid)
        metrics, covered_by_sustain, n_lead_in = _incident_resource_metrics(
            case,
            errs_r,
            ends_r,
            threshold,
            source,
            _effective_step(scores.grid, ends_r),
            mode=mode,
            onset_anchor=onset_anchor,
            sustain=sustain,
            lead_in=lead_in,
            max_leadtime=max_leadtime,
        )
        per_resource[rid] = metrics
        for accounting in SUSTAIN_ACCOUNTINGS:
            pooled_covered[accounting] += covered_by_sustain[accounting]
        pooled_lead_in_windows += n_lead_in

    for control in controls:
        rid = control.resource_id
        threshold, source = _resolved_threshold(thresholds, rid)
        errs_r, ends_r = _sorted_slice(errors, resource_ids, scores.end_times, rid)
        incident = incidents[0]
        per_resource[rid] = _control_resource_metrics(
            control,
            incident,
            errs_r,
            ends_r,
            threshold,
            source,
            _resolved_threshold(thresholds, incident.resource_id)[0],
            sustain=sustain,
            lead_in=lead_in,
        )

    # Pooled = pooled WINDOWS over the incident resources, each judged at its
    # own threshold (spec 7.4: every resource scores against its OWN
    # threshold; pooled aggregates are labeled pooled_ with the per-resource
    # breakdown beside).
    pooled_lead_in_fpr = {
        accounting: (
            pooled_covered[accounting] / pooled_lead_in_windows if pooled_lead_in_windows else None
        )
        for accounting in SUSTAIN_ACCOUNTINGS
    }
    return GridMetrics(
        grid=scores.grid,
        per_resource=per_resource,
        pooled_lead_in_fpr=pooled_lead_in_fpr,
        fleet_time_in_alarm_fraction={},
        fleet_raises_per_week={},
        fleet_runs_per_week={},
        n_eval_windows=sum(metrics.n_eval_windows for metrics in per_resource.values()),
        vus_pr=_vus_pr_placeholder(),
    )


def _healthy_resource_metrics(
    rid: str,
    role: str,
    errors: np.ndarray,
    ends: pd.DatetimeIndex,
    threshold: float,
    source: str,
    step: pd.Timedelta,
) -> tuple[ResourceMetrics, dict[int, float], float]:
    """One resource's duration-weighted alarm-fatigue metrics on a healthy span.

    The observed span is the sweep's ``span_days``: last end minus first end,
    zero below two windows (rates and fractions are then None). Alarm
    durations carry the +one-step wall rule, so a wall-to-wall alarm reads
    slightly above 1.0. Returns (metrics, alarm seconds by sustain,
    span seconds) for the fleet aggregation.
    """
    flags = errors > threshold
    span_seconds = float((ends.max() - ends.min()).total_seconds()) if len(ends) >= 2 else 0.0
    span_days = span_seconds / 86400.0

    run_counts: dict[int, int] = {}
    alarm_by_sustain: dict[int, float] = {}
    rates: dict[int, float | None] = {}
    time_in_alarm: dict[int, float | None] = {}
    for accounting in SUSTAIN_ACCOUNTINGS:
        n_runs = len(anomaly_runs(flags, accounting))
        run_counts[accounting] = n_runs
        alarm_by_sustain[accounting] = _alarm_seconds(flags, ends, accounting, step)
        rates[accounting] = 7.0 * n_runs / span_days if span_days > 0 else None
        time_in_alarm[accounting] = (
            alarm_by_sustain[accounting] / span_seconds if span_seconds > 0 else None
        )

    metrics = ResourceMetrics(
        resource_id=rid,
        role=role,
        threshold=threshold,
        threshold_source=source,
        n_eval_windows=int(ends.size),
        detection=None,
        detection_time=None,
        lead_seconds_by_onset={},
        lead_in_fpr={},
        n_lead_in_windows=0,
        coverage_fraction=None,
        clear_lead_vs_end_s=None,
        alarm_seconds_in_incident=None,
        time_in_alarm_fraction=time_in_alarm,
        # A raise is the start of a maximal over-threshold run, so at
        # accounting s the raise count equals the run count by construction.
        raises_per_week=dict(rates),
        runs_per_week=dict(rates),
        sustained_run_counts=run_counts,
        observed_span_days=span_days,
        slice_stats_by_threshold={},
        exceedances_by_threshold={},
    )
    return metrics, alarm_by_sustain, span_seconds


def _healthy_grid_metrics(
    scores: ScoreSet,
    labels: LabelSet,
    thresholds: Mapping[str, tuple[float, str]],
) -> GridMetrics:
    """One grid's healthy-reference (or control-only) metrics with fleet aggregates."""
    errors = np.asarray(scores.errors)
    resource_ids = np.asarray(scores.resource_ids).astype(str)

    per_resource: dict[str, ResourceMetrics] = {}
    alarm_totals = dict.fromkeys(SUSTAIN_ACCOUNTINGS, 0.0)
    rate_totals = dict.fromkeys(SUSTAIN_ACCOUNTINGS, 0.0)
    span_total = 0.0
    for rid in sorted(set(resource_ids)):
        threshold, source = _resolved_threshold(thresholds, rid)
        errs_r, ends_r = _sorted_slice(errors, resource_ids, scores.end_times, rid)
        metrics, alarm_by_sustain, span_seconds = _healthy_resource_metrics(
            rid,
            labels.role_for(rid),
            errs_r,
            ends_r,
            threshold,
            source,
            _effective_step(scores.grid, ends_r),
        )
        per_resource[rid] = metrics
        span_total += span_seconds
        for accounting in SUSTAIN_ACCOUNTINGS:
            alarm_totals[accounting] += alarm_by_sustain[accounting]
            rate_totals[accounting] += metrics.runs_per_week[accounting] or 0.0

    # Fleet rates sum per-resource rates (None as 0.0); the fleet fraction is
    # duration-weighted: sum of alarm seconds over sum of span seconds.
    fleet_time = {
        accounting: (alarm_totals[accounting] / span_total if span_total > 0 else None)
        for accounting in SUSTAIN_ACCOUNTINGS
    }
    fleet_rates = {accounting: rate_totals[accounting] for accounting in SUSTAIN_ACCOUNTINGS}
    return GridMetrics(
        grid=scores.grid,
        per_resource=per_resource,
        pooled_lead_in_fpr={},
        fleet_time_in_alarm_fraction=fleet_time,
        fleet_raises_per_week=dict(fleet_rates),
        fleet_runs_per_week=dict(fleet_rates),
        n_eval_windows=sum(metrics.n_eval_windows for metrics in per_resource.values()),
        vus_pr=_vus_pr_placeholder(),
    )


def compute_case_metrics(
    case_id: str,
    scores_by_grid: Mapping[str, ScoreSet],
    labels: LabelSet,
    thresholds: Mapping[str, tuple[float, str]],
    *,
    mode: DetectionMode,
    onset_anchor: str,
    sustain: int,
    lead_in: pd.Timedelta,
    max_leadtime: pd.Timedelta,
    context: dict[str, Any] | None = None,
) -> CaseMetrics:
    """Compute one case's metric bundle from scored grids and labels.

    The case kind follows the labels: declared incidents make an incident
    case, declared controls with no incident a negative_control case, and no
    labels at all a healthy_reference case (every resource treated as
    negative/healthy).

    On an incident case, for each grid and each incident resource, exactly
    one detection is selected at the rubric's ``onset_anchor`` over
    ``[anchor - max_leadtime, end]``; leads against every other named onset
    are arithmetic from that single detection time. Lead-in FPR is computed
    per resource at its OWN resolved threshold over ``[primary_onset -
    lead_in, primary_onset)`` for both sustain accountings, with the pooled_
    aggregate pooling windows across resources, never thresholds. Declared
    controls in the same capture are sliced over the incident's span at their
    own and the incident resource's thresholds.

    On healthy_reference and negative_control cases, every scored resource
    gets the duration-weighted alarm-fatigue metrics with the fleet
    aggregates on the grid.

    Args:
        case_id: The suite case name.
        scores_by_grid: ScoreSets keyed by grid label.
        labels: The case's labels (roles, onsets, ends).
        thresholds: Resolved per-resource ``(threshold, source)`` pairs.
        mode: Detection-selection semantics.
        onset_anchor: The onset name detection anchors on.
        sustain: The rubric's sustained-run length.
        lead_in: Lead-in window length before the primary onset.
        max_leadtime: Look-back horizon before the anchor for detection.
        context: Optional gate-conduit entries, passed through to CaseMetrics.

    Raises:
        ValueError: On a missing threshold, an absent anchor onset, or an
            incident case without a primary onset.
        NotImplementedError: For controls declared alongside more than one
            incident case (the flat "own" slice key is defined against a
            single incident).
    """
    if labels.incidents():
        case_kind = "incident"
        grids = {
            label: _incident_grid_metrics(
                scores,
                labels,
                thresholds,
                mode=mode,
                onset_anchor=onset_anchor,
                sustain=sustain,
                lead_in=lead_in,
                max_leadtime=max_leadtime,
            )
            for label, scores in scores_by_grid.items()
        }
    else:
        case_kind = "negative_control" if labels.negative_controls() else "healthy_reference"
        grids = {
            label: _healthy_grid_metrics(scores, labels, thresholds)
            for label, scores in scores_by_grid.items()
        }
    return CaseMetrics(
        case_id=case_id, case_kind=case_kind, grids=grids, context=dict(context or {})
    )
