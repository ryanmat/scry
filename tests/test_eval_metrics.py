# Description: Tests for scry.eval.metrics: the metric bundle dataclasses and their producer.
# Description: Pins both-accountings, role rules, grid keying, incident computation, torch-free.

"""Tests for the metric bundle schema and producer.

``ResourceMetrics``/``GridMetrics``/``CaseMetrics`` are pinned through
full-field construction and frozenness, the both-accountings invariant (every
populated sustain-keyed dict carries keys 3 and 1 or raises), the role rules
(the incident and negative-control field groups populate only under their
role; the healthy-reference group is case-kind-driven and allowed under any
role), grid keying (empty labels rejected, key must equal the grid's label),
the case_kind enumeration, the context conduit default, and the torch-free
import contract. ``compute_case_metrics`` is pinned through the multi-onset
single-detection arithmetic (one select_detection call, anchored at T0, T2
lead arithmetic from that time), the no-detection all-leads-None case, the
phase-4 coverage miniature with exact fractions, the single-window
one-grid-step alarm duration, the two-threshold lead-in FPR shape with the
pooled_ aggregate differing from every per-resource number, incident grid
assembly, the one-effective-step rule on both the cadence and the measured
median-spacing grids, the control-inside-incident slice at own and incident
thresholds, the multi-incident-plus-controls NotImplementedError, the
healthy-reference sweep-semantics rates and duration-weighted fleet
aggregation (including the master-2 wall-to-wall shape and the zero-span
None rules), the negative_control-only case kind, the vus_pr placeholder
and both-accountings sweep across all three kinds, and the eager package
exports.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from scry.eval.candidate import ScoreSet
from scry.eval.detection import DetectionResult
from scry.eval.labels import LabelCase, LabelSet
from scry.eval.metrics import CaseMetrics, GridMetrics, ResourceMetrics, compute_case_metrics
from scry.eval.scoring import ScoringGrid


def _incident_detection() -> DetectionResult:
    return DetectionResult(
        detected=True,
        detection_time=pd.Timestamp("2026-01-01T01:00:00Z"),
        lead_seconds=720.0,
        bridged=False,
        n_runs_pre_onset=0,
        n_runs_at_or_after=1,
    )


def _resource(**overrides) -> ResourceMetrics:
    """An incident-role ResourceMetrics with the full 8.2 field set."""
    kwargs = dict(
        resource_id="node-a",
        role="incident",
        threshold=0.19,
        threshold_source="per_resource",
        n_eval_windows=250,
        detection=_incident_detection(),
        detection_time=pd.Timestamp("2026-01-01T01:00:00Z"),
        lead_seconds_by_onset={"T0": 720.0, "T2": 510.0},
        lead_in_fpr={3: 0.014, 1: 0.035},
        n_lead_in_windows=209,
        coverage_fraction=0.214,
        clear_lead_vs_end_s=5280.0,
        alarm_seconds_in_incident=1800.0,
        time_in_alarm_fraction={},
        raises_per_week={},
        runs_per_week={},
        sustained_run_counts={3: 1, 1: 4},
        observed_span_days=None,
        slice_stats_by_threshold={},
        exceedances_by_threshold={},
    )
    kwargs.update(overrides)
    return ResourceMetrics(**kwargs)


def _healthy_resource(**overrides) -> ResourceMetrics:
    """A healthy-reference-shaped resource: role from labels, healthy dicts populated."""
    kwargs = dict(
        resource_id="node-b",
        role="excluded",
        threshold=0.19,
        threshold_source="global",
        n_eval_windows=2016,
        detection=None,
        detection_time=None,
        lead_seconds_by_onset={},
        lead_in_fpr={},
        n_lead_in_windows=0,
        coverage_fraction=None,
        clear_lead_vs_end_s=None,
        alarm_seconds_in_incident=None,
        time_in_alarm_fraction={3: 0.002, 1: 0.007},
        raises_per_week={3: 1.4, 1: 4.9},
        runs_per_week={3: 1.4, 1: 4.9},
        sustained_run_counts={3: 2, 1: 7},
        observed_span_days=7.0,
        slice_stats_by_threshold={},
        exceedances_by_threshold={},
    )
    kwargs.update(overrides)
    return ResourceMetrics(**kwargs)


def _control_resource(**overrides) -> ResourceMetrics:
    """A negative-control resource sliced against one incident case."""
    kwargs = dict(
        resource_id="node-c",
        role="negative_control",
        threshold=0.21,
        threshold_source="per_resource",
        n_eval_windows=250,
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
        sustained_run_counts={3: 0, 1: 0},
        observed_span_days=None,
        slice_stats_by_threshold={
            "own": {"n_windows_in_slice": 250},
            "node-a": {"n_windows_in_slice": 250},
        },
        exceedances_by_threshold={"own": 0, "node-a": 2},
    )
    kwargs.update(overrides)
    return ResourceMetrics(**kwargs)


def _grid_metrics(label: str = "serving", **overrides) -> GridMetrics:
    kwargs = dict(
        grid=ScoringGrid(label=label, step_samples=1, cadence=pd.Timedelta(minutes=10)),
        per_resource={"node-a": _resource()},
        pooled_lead_in_fpr={3: 0.215, 1: 0.31},
        fleet_time_in_alarm_fraction={},
        fleet_raises_per_week={},
        fleet_runs_per_week={},
        n_eval_windows=250,
        vus_pr={"value": None, "reason": "not implemented"},
    )
    kwargs.update(overrides)
    return GridMetrics(**kwargs)


class TestConstructionAndFrozen:
    def test_incident_resource_constructs_with_full_field_set(self) -> None:
        resource = _resource()
        assert resource.role == "incident"
        assert resource.detection is not None and resource.detection.detected
        assert resource.lead_seconds_by_onset["T2"] == 510.0
        assert resource.lead_in_fpr == {3: 0.014, 1: 0.035}
        with pytest.raises(AttributeError):
            resource.threshold = 0.5  # type: ignore[misc]

    def test_grid_and_case_construct(self) -> None:
        grid_metrics = _grid_metrics()
        case = CaseMetrics(
            case_id="phase4_75cpl",
            case_kind="incident",
            grids={"serving": grid_metrics},
            context={"hygiene_population": "lead_in", "profile": "aro_node"},
        )
        assert case.grids["serving"].pooled_lead_in_fpr[3] == 0.215
        assert case.context["profile"] == "aro_node"
        with pytest.raises(AttributeError):
            case.case_kind = "healthy_reference"  # type: ignore[misc]
        with pytest.raises(AttributeError):
            grid_metrics.n_eval_windows = 0  # type: ignore[misc]

    def test_context_defaults_empty(self) -> None:
        case = CaseMetrics(case_id="week", case_kind="healthy_reference", grids={})
        assert case.context == {}


class TestBothAccountings:
    def test_resource_sustain_dicts_require_both_keys(self) -> None:
        cases = (
            ("lead_in_fpr", _resource, 0.5),
            ("time_in_alarm_fraction", _healthy_resource, 0.5),
            ("raises_per_week", _healthy_resource, 0.5),
            ("runs_per_week", _healthy_resource, 0.5),
            ("sustained_run_counts", _resource, 1),
        )
        for field_name, make, value in cases:
            for lone in ({3: value}, {1: value}):
                with pytest.raises(ValueError, match="both sustain accountings"):
                    make(**{field_name: lone})
            make(**{field_name: {3: value, 1: value}})

    def test_grid_sustain_dicts_require_both_keys(self) -> None:
        for field_name in (
            "pooled_lead_in_fpr",
            "fleet_time_in_alarm_fraction",
            "fleet_raises_per_week",
            "fleet_runs_per_week",
        ):
            for lone in ({3: 0.1}, {1: 0.1}):
                with pytest.raises(ValueError, match="both sustain accountings"):
                    _grid_metrics(**{field_name: lone})
            _grid_metrics(**{field_name: {3: 0.1, 1: 0.1}})

    def test_empty_sustain_dict_means_not_applicable_and_passes(self) -> None:
        assert _resource(time_in_alarm_fraction={}).time_in_alarm_fraction == {}
        assert _grid_metrics(fleet_runs_per_week={}).fleet_runs_per_week == {}


class TestRoleRules:
    def test_unknown_role_enumerates_valid_roles(self) -> None:
        with pytest.raises(ValueError, match="valid roles: incident, negative_control, excluded"):
            _resource(role="observer")

    def test_non_incident_role_rejects_incident_fields(self) -> None:
        with pytest.raises(ValueError, match="detection_time.*role is authoritative"):
            _control_resource(detection_time=pd.Timestamp("2026-01-01T01:00:00Z"))
        with pytest.raises(ValueError, match="lead_in_fpr.*role is authoritative"):
            _healthy_resource(lead_in_fpr={3: 0.0, 1: 0.0})
        with pytest.raises(ValueError, match="coverage_fraction"):
            _healthy_resource(coverage_fraction=0.0)

    def test_non_control_role_rejects_control_maps(self) -> None:
        with pytest.raises(ValueError, match="slice_stats_by_threshold"):
            _resource(slice_stats_by_threshold={"own": {}})
        with pytest.raises(ValueError, match="exceedances_by_threshold"):
            _healthy_resource(exceedances_by_threshold={"own": 0})

    def test_control_resource_constructs_with_populated_maps(self) -> None:
        control = _control_resource()
        assert control.exceedances_by_threshold["node-a"] == 2
        assert set(control.slice_stats_by_threshold) == {"own", "node-a"}

    def test_healthy_group_allowed_under_any_role(self) -> None:
        # A healthy-reference capture takes no labels, so its resources carry
        # whatever labels role applies; the healthy group is case-kind-driven.
        assert _healthy_resource().observed_span_days == 7.0
        control_shaped = _healthy_resource(role="negative_control", resource_id="node-c")
        assert control_shaped.runs_per_week[1] == 4.9


class TestCaseKeying:
    def test_empty_grid_label_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty label"):
            CaseMetrics(case_id="c", case_kind="incident", grids={"": _grid_metrics()})

    def test_grid_key_must_match_grid_label(self) -> None:
        with pytest.raises(ValueError, match="does not match"):
            CaseMetrics(
                case_id="c", case_kind="incident", grids={"serving": _grid_metrics(label="offline")}
            )

    def test_unknown_case_kind_enumerates_valid_kinds(self) -> None:
        with pytest.raises(
            ValueError, match="valid kinds: incident, healthy_reference, negative_control"
        ):
            CaseMetrics(case_id="c", case_kind="weird", grids={})


_TEN_MIN_GRID = ScoringGrid(label="serving", step_samples=1, cadence=pd.Timedelta(minutes=10))


def _score_set(errors: np.ndarray, ends: pd.DatetimeIndex, ids: list[str]) -> ScoreSet:
    return ScoreSet(
        errors=np.asarray(errors, dtype=np.float64),
        end_times=ends,
        resource_ids=np.asarray(ids),
        grid=_TEN_MIN_GRID,
        meta={},
    )


def _incident_labels(
    resource_id: str, onsets: dict[str, str], end: str, primary: str = "T0"
) -> LabelSet:
    case = LabelCase(
        resource_id=resource_id,
        role="incident",
        type="cpu",
        onsets={name: pd.Timestamp(value) for name, value in onsets.items()},
        primary_onset=primary,
        end=pd.Timestamp(end),
        notes=None,
    )
    return LabelSet(version=2, capture=None, cases=[case])


def _phase4_miniature(over: bool = True) -> tuple[ScoreSet, LabelSet]:
    """27 ten-minute windows 10:00..14:20; optionally a 3-window excursion 12:10-12:30.

    Labels: T0 12:00 (primary), T2 12:24, end 14:00. With the excursion, the
    sustained run starts between T0 and T2 and its alarm span [12:10, 12:40)
    covers 1800 s of the 7200 s incident, clearing 4800 s early -- the phase-4
    shape (partial coverage, alarm clears well before end) in miniature.
    """
    ends = pd.date_range("2026-01-01T10:00:00Z", periods=27, freq="10min")
    errors = np.full(27, 0.1)
    if over:
        excursion = (ends >= pd.Timestamp("2026-01-01T12:10:00Z")) & (
            ends <= pd.Timestamp("2026-01-01T12:30:00Z")
        )
        errors[excursion] = 2.0
    scores = _score_set(errors, ends, ["node-a"] * 27)
    labels = _incident_labels(
        "node-a",
        {"T0": "2026-01-01T12:00:00Z", "T2": "2026-01-01T12:24:00Z"},
        "2026-01-01T14:00:00Z",
    )
    return scores, labels


def _compute(scores: ScoreSet, labels: LabelSet, thresholds: dict, **overrides) -> CaseMetrics:
    params = dict(
        mode="no_bridging",
        onset_anchor="T0",
        sustain=3,
        lead_in=pd.Timedelta(hours=1),
        max_leadtime=pd.Timedelta(hours=2),
    )
    params.update(overrides)
    return compute_case_metrics("miniature", {"serving": scores}, labels, thresholds, **params)


class TestComputeIncidentCaseMetrics:
    def test_multi_onset_single_detection_with_arithmetic_leads(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Exactly ONE select_detection call, anchored at T0; the T2 lead is
        # arithmetic from that detection_time, never a second detection run
        # (a re-run at T2 under no_bridging would find nothing and lead None).
        import scry.eval.metrics as metrics_module
        from scry.eval.detection import select_detection as real_select_detection

        calls: list[pd.Timestamp] = []

        def counting(spans, scan_ts, onset, mode):
            calls.append(onset)
            return real_select_detection(spans, scan_ts, onset, mode)

        monkeypatch.setattr(metrics_module, "select_detection", counting)

        scores, labels = _phase4_miniature()
        case = _compute(scores, labels, {"node-a": (1.0, "override")})

        assert calls == [pd.Timestamp("2026-01-01T12:00:00Z")]
        resource = case.grids["serving"].per_resource["node-a"]
        assert resource.detection is not None and resource.detection.detected
        assert resource.detection_time == pd.Timestamp("2026-01-01T12:10:00Z")
        assert resource.lead_seconds_by_onset == {"T0": -600.0, "T2": 840.0}
        assert resource.threshold == 1.0
        assert resource.threshold_source == "override"

    def test_no_detection_has_all_leads_none(self) -> None:
        scores, labels = _phase4_miniature(over=False)
        case = _compute(scores, labels, {"node-a": (1.0, "override")})

        resource = case.grids["serving"].per_resource["node-a"]
        assert resource.detection is not None and not resource.detection.detected
        assert resource.detection_time is None
        assert resource.lead_seconds_by_onset == {"T0": None, "T2": None}
        assert resource.coverage_fraction == 0.0
        assert resource.alarm_seconds_in_incident == 0.0
        assert resource.clear_lead_vs_end_s is None

    def test_incident_coverage_phase4_miniature(self) -> None:
        # Alarm span [12:10, 12:40) inside incident [12:00, 14:00]: 1800 of
        # 7200 s covered, cleared 4800 s before end.
        scores, labels = _phase4_miniature()
        case = _compute(scores, labels, {"node-a": (1.0, "override")})

        resource = case.grids["serving"].per_resource["node-a"]
        assert resource.coverage_fraction == 0.25
        assert resource.alarm_seconds_in_incident == 1800.0
        assert resource.clear_lead_vs_end_s == 4800.0
        assert resource.sustained_run_counts == {3: 1, 1: 1}
        assert resource.n_eval_windows == 27
        assert resource.n_lead_in_windows == 6
        assert resource.lead_in_fpr == {3: 0.0, 1: 0.0}

    def test_single_window_alarm_has_one_grid_step_duration(self) -> None:
        # Alarm run wall duration = last end - first end + one effective grid
        # step: a single-window run on the 10-minute cadence lasts 600 s.
        ends = pd.date_range("2026-01-01T11:00:00Z", periods=13, freq="10min")
        errors = np.full(13, 0.1)
        errors[ends == pd.Timestamp("2026-01-01T12:30:00Z")] = 2.0
        scores = _score_set(errors, ends, ["node-a"] * 13)
        labels = _incident_labels("node-a", {"T0": "2026-01-01T12:00:00Z"}, "2026-01-01T13:00:00Z")
        case = _compute(scores, labels, {"node-a": (1.0, "override")}, sustain=1)

        resource = case.grids["serving"].per_resource["node-a"]
        assert resource.alarm_seconds_in_incident == 600.0
        assert resource.coverage_fraction == 600.0 / 3600.0
        assert resource.clear_lead_vs_end_s == 1200.0

    def test_lead_in_fpr_both_accountings_and_pooled_differ(self) -> None:
        # The 4b.4 two-threshold shape: each resource's lead-in FPR at its OWN
        # threshold for both accountings; the pooled_ number pools windows and
        # differs from every per-resource number at both accountings.
        ends = pd.date_range("2026-01-01T10:20:00Z", periods=17, freq="10min")
        quiet = np.full(17, 0.1)
        noisy = np.full(17, 1.5)
        minute = {ts: i for i, ts in enumerate(ends)}
        for stamp in ("11:00", "11:10", "11:20", "11:50"):
            quiet[minute[pd.Timestamp(f"2026-01-01T{stamp}:00Z")]] = 0.3
        for stamp in ("11:00", "11:10", "11:50"):
            noisy[minute[pd.Timestamp(f"2026-01-01T{stamp}:00Z")]] = 2.5
        scores = _score_set(
            np.concatenate([quiet, noisy]), ends.append(ends), ["quiet"] * 17 + ["noisy"] * 17
        )
        cases = [
            LabelCase(
                resource_id=rid,
                role="incident",
                type="cpu",
                onsets={"T0": pd.Timestamp("2026-01-01T12:00:00Z")},
                primary_onset="T0",
                end=pd.Timestamp("2026-01-01T13:00:00Z"),
                notes=None,
            )
            for rid in ("quiet", "noisy")
        ]
        labels = LabelSet(version=2, capture=None, cases=cases)
        thresholds = {"quiet": (0.2, "per_resource"), "noisy": (2.0, "per_resource")}
        case = _compute(scores, labels, thresholds, lead_in=pd.Timedelta(minutes=100))

        grid = case.grids["serving"]
        assert grid.per_resource["quiet"].lead_in_fpr == {3: 0.3, 1: 0.4}
        assert grid.per_resource["noisy"].lead_in_fpr == {3: 0.0, 1: 0.3}
        assert grid.per_resource["quiet"].n_lead_in_windows == 10
        assert grid.per_resource["noisy"].n_lead_in_windows == 10
        assert grid.pooled_lead_in_fpr == {3: 0.15, 1: 0.35}
        for sustain in (3, 1):
            for resource in grid.per_resource.values():
                assert grid.pooled_lead_in_fpr[sustain] != resource.lead_in_fpr[sustain]

    def test_incident_case_grid_assembly(self) -> None:
        scores, labels = _phase4_miniature()
        case = _compute(scores, labels, {"node-a": (1.0, "override")})

        assert case.case_id == "miniature"
        assert case.case_kind == "incident"
        assert case.context == {}
        grid = case.grids["serving"]
        assert grid.grid is _TEN_MIN_GRID
        assert grid.n_eval_windows == 27
        assert grid.fleet_time_in_alarm_fraction == {}
        assert grid.fleet_raises_per_week == {}
        assert grid.fleet_runs_per_week == {}
        assert grid.vus_pr == {"value": None, "reason": "not implemented"}

    def test_median_spacing_grid_alarm_duration(self) -> None:
        # A cadence-less grid measures its effective step as the median
        # inter-window spacing: the same single-window run on 10-minute-spaced
        # windows still lasts 600 s.
        offline = ScoringGrid(label="offline", step_samples=10)
        ends = pd.date_range("2026-01-01T11:00:00Z", periods=13, freq="10min")
        errors = np.full(13, 0.1)
        errors[ends == pd.Timestamp("2026-01-01T12:30:00Z")] = 2.0
        scores = ScoreSet(
            errors=errors,
            end_times=ends,
            resource_ids=np.array(["node-a"] * 13),
            grid=offline,
            meta={},
        )
        labels = _incident_labels("node-a", {"T0": "2026-01-01T12:00:00Z"}, "2026-01-01T13:00:00Z")
        case = compute_case_metrics(
            "miniature",
            {"offline": scores},
            labels,
            {"node-a": (1.0, "override")},
            mode="no_bridging",
            onset_anchor="T0",
            sustain=1,
            lead_in=pd.Timedelta(hours=1),
            max_leadtime=pd.Timedelta(hours=2),
        )
        assert case.grids["offline"].per_resource["node-a"].alarm_seconds_in_incident == 600.0

    def test_control_inside_incident_capture_sliced_at_both_thresholds(self) -> None:
        # The phase-4 controls check in miniature: the control's windows over
        # the incident's [T0 - lead_in, end] span, scored at its own resolved
        # threshold AND at the incident resource's -- keys "own" + incident id.
        incident_scores, labels = _phase4_miniature()
        ends = incident_scores.end_times
        control_errors = np.full(27, 0.2)
        for stamp, value in (("11:30", 0.6), ("11:40", 0.6), ("12:50", 1.2)):
            control_errors[ends == pd.Timestamp(f"2026-01-01T{stamp}:00Z")] = value
        scores = _score_set(
            np.concatenate([np.asarray(incident_scores.errors), control_errors]),
            ends.append(ends),
            ["node-a"] * 27 + ["node-ctl"] * 27,
        )
        control_case = LabelCase(
            resource_id="node-ctl",
            role="negative_control",
            type=None,
            onsets={},
            primary_onset=None,
            end=None,
            notes=None,
        )
        mixed = LabelSet(version=2, capture=None, cases=list(labels.cases) + [control_case])
        thresholds = {"node-a": (1.0, "override"), "node-ctl": (0.5, "per_resource")}
        case = _compute(scores, mixed, thresholds)

        assert case.case_kind == "incident"
        grid = case.grids["serving"]
        assert set(grid.per_resource) == {"node-a", "node-ctl"}
        control = grid.per_resource["node-ctl"]
        assert control.role == "negative_control"
        assert set(control.slice_stats_by_threshold) == {"own", "node-a"}
        assert control.exceedances_by_threshold == {"own": 3, "node-a": 1}
        own_stats = control.slice_stats_by_threshold["own"]
        assert own_stats["n_windows_in_slice"] == 19  # ends in [11:00, 14:00]
        assert own_stats["over_0.5000"]["windows_over"] == 3
        assert own_stats["over_0.5000"]["sustained_runs"] == 0  # 2-run + isolated at sustain 3
        assert control.slice_stats_by_threshold["node-a"]["over_1.0000"]["windows_over"] == 1
        assert control.sustained_run_counts == {3: 0, 1: 2}
        assert control.detection is None
        assert control.lead_in_fpr == {}
        assert control.time_in_alarm_fraction == {}
        # The incident resource is unchanged beside it; pooling stays incident-only.
        assert grid.per_resource["node-a"].coverage_fraction == 0.25
        assert grid.pooled_lead_in_fpr[1] is not None

    def test_control_with_multiple_incidents_not_implemented(self) -> None:
        scores, labels = _phase4_miniature()
        second_incident = LabelCase(
            resource_id="node-b",
            role="incident",
            type="cpu",
            onsets={"T0": pd.Timestamp("2026-01-01T12:30:00Z")},
            primary_onset="T0",
            end=pd.Timestamp("2026-01-01T14:00:00Z"),
            notes=None,
        )
        control_case = LabelCase(
            resource_id="node-ctl",
            role="negative_control",
            type=None,
            onsets={},
            primary_onset=None,
            end=None,
            notes=None,
        )
        mixed = LabelSet(
            version=2, capture=None, cases=[*labels.cases, second_incident, control_case]
        )
        with pytest.raises(NotImplementedError, match="single incident"):
            _compute(scores, mixed, {"node-a": (1.0, "override"), "node-b": (1.0, "override")})


_HEALTHY_THRESHOLDS = {"node-hot": (0.5, "global"), "node-quiet": (0.5, "global")}


def _healthy_fleet_scores() -> ScoreSet:
    """Two resources, 505 ten-minute windows each: a 3.5-day observed span.

    node-hot is over threshold wall to wall (the master-2 shape: one run, the
    time-in-alarm fraction shows what the run count hides). node-quiet has
    five isolated single-window exceedances plus one 3-window run.
    """
    ends = pd.date_range("2026-01-01T00:00:00Z", periods=505, freq="10min")
    hot = np.full(505, 2.0)
    quiet = np.full(505, 0.1)
    for index in (50, 100, 150, 200, 250):
        quiet[index] = 0.7
    quiet[300:303] = 0.7
    return _score_set(
        np.concatenate([hot, quiet]), ends.append(ends), ["node-hot"] * 505 + ["node-quiet"] * 505
    )


class TestHealthyAndControlCases:
    def test_healthy_reference_rates_and_fractions(self) -> None:
        # Span = 504 ten-minute steps = 3.5 days (the sweep's span_days: last
        # minus first, no step added). Alarm durations carry the +one-step
        # wall rule, so the wall-to-wall alarm reads slightly above 1.0.
        empty = LabelSet(version=2, capture=None, cases=[])
        case = _compute(_healthy_fleet_scores(), empty, _HEALTHY_THRESHOLDS)

        assert case.case_kind == "healthy_reference"
        grid = case.grids["serving"]
        hot = grid.per_resource["node-hot"]
        assert hot.role == "excluded"
        assert hot.observed_span_days == 3.5
        assert hot.sustained_run_counts == {3: 1, 1: 1}
        assert hot.runs_per_week == {3: 2.0, 1: 2.0}
        assert hot.raises_per_week == {3: 2.0, 1: 2.0}
        assert hot.time_in_alarm_fraction == {3: 303000.0 / 302400.0, 1: 303000.0 / 302400.0}
        assert hot.detection is None
        assert hot.lead_in_fpr == {}
        assert hot.slice_stats_by_threshold == {}

        quiet = grid.per_resource["node-quiet"]
        assert quiet.sustained_run_counts == {3: 1, 1: 6}
        assert quiet.runs_per_week == {3: 2.0, 1: 12.0}
        assert quiet.raises_per_week == {3: 2.0, 1: 12.0}
        assert quiet.time_in_alarm_fraction == {3: 1800.0 / 302400.0, 1: 4800.0 / 302400.0}

    def test_fleet_aggregation_is_duration_weighted(self) -> None:
        empty = LabelSet(version=2, capture=None, cases=[])
        grid = _compute(_healthy_fleet_scores(), empty, _HEALTHY_THRESHOLDS).grids["serving"]

        assert grid.fleet_runs_per_week == {3: 4.0, 1: 14.0}
        assert grid.fleet_raises_per_week == {3: 4.0, 1: 14.0}
        assert grid.fleet_time_in_alarm_fraction == {
            3: 304800.0 / 604800.0,
            1: 307800.0 / 604800.0,
        }
        assert grid.pooled_lead_in_fpr == {}
        assert grid.n_eval_windows == 1010

    def test_zero_span_resource_rates_none_and_fleet_sums_none_as_zero(self) -> None:
        # A single-window resource on a cadence-less grid has no observed
        # span: its rates and fraction are None, and the fleet sums treat
        # None as 0.0 (rates) and contribute nothing to either duration sum.
        offline = ScoringGrid(label="offline", step_samples=10)
        ends_x = pd.date_range("2026-01-01T00:00:00Z", periods=8, freq="10min")
        errors_x = np.full(8, 0.1)
        errors_x[2] = 0.9
        errors_x[5] = 0.9
        solo_end = pd.DatetimeIndex([pd.Timestamp("2026-01-01T02:00:00Z")])
        scores = ScoreSet(
            errors=np.concatenate([errors_x, [0.9]]),
            end_times=ends_x.append(solo_end),
            resource_ids=np.array(["node-x"] * 8 + ["node-solo"]),
            grid=offline,
            meta={},
        )
        empty = LabelSet(version=2, capture=None, cases=[])
        thresholds = {"node-x": (0.5, "global"), "node-solo": (0.5, "global")}
        case = compute_case_metrics(
            "fleet",
            {"offline": scores},
            empty,
            thresholds,
            mode="no_bridging",
            onset_anchor="T0",
            sustain=3,
            lead_in=pd.Timedelta(hours=1),
            max_leadtime=pd.Timedelta(hours=2),
        )

        grid = case.grids["offline"]
        solo = grid.per_resource["node-solo"]
        assert solo.observed_span_days == 0.0
        assert solo.runs_per_week == {3: None, 1: None}
        assert solo.raises_per_week == {3: None, 1: None}
        assert solo.time_in_alarm_fraction == {3: None, 1: None}
        assert solo.sustained_run_counts == {3: 0, 1: 1}
        assert grid.fleet_runs_per_week == {3: 0.0, 1: 7.0 * 2 / (4200.0 / 86400.0)}
        assert grid.fleet_time_in_alarm_fraction == {3: 0.0, 1: 1200.0 / 4200.0}

    def test_negative_control_only_case_kind(self) -> None:
        ends = pd.date_range("2026-01-01T00:00:00Z", periods=8, freq="10min")
        declared = np.full(8, 0.1)
        declared[3:6] = 0.9
        unlabeled = np.full(8, 0.1)
        scores = _score_set(
            np.concatenate([declared, unlabeled]),
            ends.append(ends),
            ["node-ctl"] * 8 + ["node-b"] * 8,
        )
        control_case = LabelCase(
            resource_id="node-ctl",
            role="negative_control",
            type=None,
            onsets={},
            primary_onset=None,
            end=None,
            notes=None,
        )
        labels = LabelSet(version=2, capture=None, cases=[control_case])
        thresholds = {"node-ctl": (0.5, "per_resource"), "node-b": (0.5, "global")}
        case = _compute(scores, labels, thresholds)

        assert case.case_kind == "negative_control"
        grid = case.grids["serving"]
        control = grid.per_resource["node-ctl"]
        assert control.role == "negative_control"
        assert control.sustained_run_counts == {3: 1, 1: 1}
        assert set(control.runs_per_week) == {3, 1}
        assert control.slice_stats_by_threshold == {}  # no incident to slice against
        assert grid.per_resource["node-b"].role == "excluded"
        assert set(grid.fleet_runs_per_week) == {3, 1}

    def test_vus_pr_placeholder_and_both_accountings_on_every_kind(self) -> None:
        incident_scores, incident_labels_set = _phase4_miniature()
        control_case = LabelCase(
            resource_id="node-a",
            role="negative_control",
            type=None,
            onsets={},
            primary_onset=None,
            end=None,
            notes=None,
        )
        produced = [
            _compute(incident_scores, incident_labels_set, {"node-a": (1.0, "override")}),
            _compute(
                _healthy_fleet_scores(),
                LabelSet(version=2, capture=None, cases=[]),
                _HEALTHY_THRESHOLDS,
            ),
            _compute(
                incident_scores,
                LabelSet(version=2, capture=None, cases=[control_case]),
                {"node-a": (0.5, "per_resource")},
            ),
        ]
        assert [case.case_kind for case in produced] == [
            "incident",
            "healthy_reference",
            "negative_control",
        ]
        for case in produced:
            for grid in case.grids.values():
                assert grid.vus_pr == {"value": None, "reason": "not implemented"}
                for sustain_dict in (
                    grid.pooled_lead_in_fpr,
                    grid.fleet_time_in_alarm_fraction,
                    grid.fleet_raises_per_week,
                    grid.fleet_runs_per_week,
                ):
                    assert sustain_dict == {} or set(sustain_dict) >= {3, 1}
                for resource in grid.per_resource.values():
                    for sustain_dict in (
                        resource.lead_in_fpr,
                        resource.time_in_alarm_fraction,
                        resource.raises_per_week,
                        resource.runs_per_week,
                        resource.sustained_run_counts,
                    ):
                        assert sustain_dict == {} or set(sustain_dict) >= {3, 1}
                    assert set(resource.sustained_run_counts) >= {3, 1}


class TestEagerExports:
    def test_metrics_members_export_eagerly(self) -> None:
        import scry.eval
        from scry.eval import metrics as metrics_module

        names = (
            "CASE_KINDS",
            "SUSTAIN_ACCOUNTINGS",
            "CaseMetrics",
            "GridMetrics",
            "ResourceMetrics",
            "compute_case_metrics",
        )
        for name in names:
            assert getattr(scry.eval, name) is getattr(metrics_module, name)
            assert name in scry.eval.__all__
            assert name in vars(scry.eval)  # eager, not resolved via __getattr__


class TestTorchFreeImport:
    def test_metrics_module_imports_without_torch(self) -> None:
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "import scry.eval.metrics, sys; assert 'torch' not in sys.modules",
            ],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
