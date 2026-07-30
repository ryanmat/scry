# Description: Tests for scry.eval.metrics: the per-case metric bundle dataclasses.
# Description: Pins both-accountings, role rules, grid keying, case kinds, and the torch-free import.

"""Tests for the metric bundle schema.

``ResourceMetrics``/``GridMetrics``/``CaseMetrics`` are pinned through
full-field construction and frozenness, the both-accountings invariant (every
populated sustain-keyed dict carries keys 3 and 1 or raises), the role rules
(the incident and negative-control field groups populate only under their
role; the healthy-reference group is case-kind-driven and allowed under any
role), grid keying (empty labels rejected, key must equal the grid's label),
the case_kind enumeration, the context conduit default, and the torch-free
import contract.
"""

from __future__ import annotations

import subprocess
import sys

import pandas as pd
import pytest

from scry.eval.detection import DetectionResult
from scry.eval.metrics import CaseMetrics, GridMetrics, ResourceMetrics
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
