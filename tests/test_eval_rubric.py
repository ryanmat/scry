# Description: Tests for scry.eval.rubric: rubric loading, result dataclasses, and SpecError.
# Description: Pins schema validation, the typed exception, eager exports, and the torch-free import.

"""Tests for rubric loading and the gate-result types.

``load_rubric`` is pinned through a valid round-trip, the enumerated
unknown-gate error, the missing-version error, the non-mapping guard, the
grids-well-formed rule, headline_grid and per-gate grid references to
declared grids, and the free-form reported/known_divergences blocks.
``GateResult``/``RubricResult`` are pinned as frozen with exactly their
declared fields. ``SpecError`` is pinned as a typed ValueError subclass
importable from ``scry.eval``, the rubric members as eager package exports,
and the module as torch-free. ``evaluate_rubric`` is pinned through every
gate driven to both verdicts from synthetic bundles (including the
underpowered-lead-in FAIL, the master-2 alarm-fatigue regression, the
context-conduit reads with the hygiene population named in observed, and
the sanity checks), the headline_grid flip fixture, gate-declared grids
beating the headline, the missing-bound-grid SpecError, allow_absent
semantics, the rolled-up required-only verdict, and the unknown-gate guard
on raw mappings.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import pandas as pd
import pytest
import yaml

from scry.eval.detection import DetectionResult
from scry.eval.metrics import CaseMetrics, GridMetrics, ResourceMetrics
from scry.eval.rubric import (
    GATE_NAMES,
    GateResult,
    RubricResult,
    SpecError,
    evaluate_rubric,
    load_rubric,
)
from scry.eval.scoring import ScoringGrid


def _base_rubric() -> dict:
    return {
        "version": 1,
        "profile": "aro_node",
        "detection": {
            "mode": "no_bridging",
            "onset_anchor": "T0",
            "sustain": 3,
            "max_leadtime_minutes": 120,
            "lead_in_hours": 24,
        },
        "grids": {
            "offline": {"step_samples": 10},
            "serving": {"step_samples": 1, "cadence_minutes": 10},
        },
        "headline_grid": "serving",
        "gates": {"no_pre_onset_bridging": {"required": True}},
        "reported": {"vus_pr": {}},
        "known_divergences": ["serving is sustain-1 single-window scoring"],
    }


def _write_rubric(tmp_path: Path, rubric: dict) -> str:
    path = tmp_path / "rubric.yaml"
    path.write_text(yaml.safe_dump(rubric))
    return str(path)


class TestLoadRubric:
    def test_valid_rubric_round_trips(self, tmp_path: Path) -> None:
        rubric = load_rubric(_write_rubric(tmp_path, _base_rubric()))
        assert rubric["version"] == 1
        assert rubric["headline_grid"] == "serving"
        assert set(rubric["grids"]) == {"offline", "serving"}
        assert rubric["known_divergences"]

    def test_unknown_gate_name_enumerates_valid_names(self, tmp_path: Path) -> None:
        bad = _base_rubric()
        bad["gates"]["bogus_gate"] = {"required": True}
        with pytest.raises(ValueError, match="unknown gate 'bogus_gate'; valid gates: "):
            load_rubric(_write_rubric(tmp_path, bad))
        with pytest.raises(SpecError, match=", ".join(GATE_NAMES)):
            load_rubric(_write_rubric(tmp_path, bad))

    def test_missing_version_errors(self, tmp_path: Path) -> None:
        bad = _base_rubric()
        del bad["version"]
        with pytest.raises(SpecError, match="missing 'version'"):
            load_rubric(_write_rubric(tmp_path, bad))

    def test_non_mapping_rubric_errors(self, tmp_path: Path) -> None:
        path = tmp_path / "rubric.yaml"
        path.write_text(yaml.safe_dump(["not", "a", "mapping"]))
        with pytest.raises(SpecError, match="YAML mapping"):
            load_rubric(str(path))

    def test_grids_must_be_declared_and_well_formed(self, tmp_path: Path) -> None:
        no_grids = _base_rubric()
        del no_grids["grids"]
        with pytest.raises(SpecError, match="at least one grid"):
            load_rubric(_write_rubric(tmp_path, no_grids))
        malformed = _base_rubric()
        malformed["grids"]["offline"] = {}
        with pytest.raises(SpecError, match="'step_samples'"):
            load_rubric(_write_rubric(tmp_path, malformed))

    def test_headline_grid_must_name_a_declared_grid(self, tmp_path: Path) -> None:
        bad = _base_rubric()
        bad["headline_grid"] = "native"
        with pytest.raises(SpecError, match="headline_grid 'native' does not name a declared grid"):
            load_rubric(_write_rubric(tmp_path, bad))

    def test_gate_grid_must_name_a_declared_grid(self, tmp_path: Path) -> None:
        bad = _base_rubric()
        bad["gates"]["alarm_fatigue"] = {"required": True, "grid": "native", "sustain": 1}
        with pytest.raises(
            SpecError, match="gate 'alarm_fatigue' grid 'native' does not name a declared grid"
        ):
            load_rubric(_write_rubric(tmp_path, bad))

    def test_reported_and_divergences_are_free_form(self, tmp_path: Path) -> None:
        rubric = _base_rubric()
        rubric["reported"]["future_metric"] = {"anything": [1, 2]}
        rubric["known_divergences"].append("an arbitrary free-form note")
        loaded = load_rubric(_write_rubric(tmp_path, rubric))
        assert loaded["reported"]["future_metric"] == {"anything": [1, 2]}
        assert "an arbitrary free-form note" in loaded["known_divergences"]


class TestResultTypes:
    def test_gate_result_declared_fields_and_frozen(self) -> None:
        assert [f.name for f in fields(GateResult)] == [
            "name",
            "required",
            "passed",
            "grid",
            "observed",
            "detail",
        ]
        result = GateResult(
            name="detection_lead",
            required=True,
            passed=True,
            grid="serving",
            observed={"lead_seconds": 720.0},
            detail="detected 720 s before T2",
        )
        with pytest.raises(AttributeError):
            result.passed = False  # type: ignore[misc]

    def test_rubric_result_declared_fields_and_frozen(self) -> None:
        assert [f.name for f in fields(RubricResult)] == [
            "rubric_version",
            "gates",
            "reported",
            "passed",
        ]
        result = RubricResult(rubric_version=1, gates=[], reported={}, passed=True)
        with pytest.raises(AttributeError):
            result.passed = False  # type: ignore[misc]


class TestSpecError:
    def test_typed_value_error_subclass_importable_from_package(self) -> None:
        import scry.eval

        assert scry.eval.SpecError is SpecError
        assert issubclass(SpecError, ValueError)
        assert SpecError is not ValueError


class TestEagerExports:
    def test_rubric_members_export_eagerly(self) -> None:
        import scry.eval
        from scry.eval import rubric as rubric_module

        for name in ("GateResult", "RubricResult", "SpecError", "evaluate_rubric", "load_rubric"):
            assert getattr(scry.eval, name) is getattr(rubric_module, name)
            assert name in scry.eval.__all__
            assert name in vars(scry.eval)  # eager, not resolved via __getattr__


def _detection(
    detected: bool = True, bridged: bool = False, lead: float = -600.0
) -> DetectionResult:
    return DetectionResult(
        detected=detected,
        detection_time=pd.Timestamp("2026-01-01T12:10:00Z") if detected else None,
        lead_seconds=lead if detected else None,
        bridged=bridged,
        n_runs_pre_onset=1 if bridged else 0,
        n_runs_at_or_after=1,
    )


def _resource_metrics(role: str = "incident", **overrides) -> ResourceMetrics:
    base = dict(
        resource_id="node-a",
        role=role,
        threshold=0.19,
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
        slice_stats_by_threshold={},
        exceedances_by_threshold={},
    )
    base.update(overrides)
    return ResourceMetrics(**base)


def _incident_resource(lead_vs_t2: float | None, **overrides) -> ResourceMetrics:
    detection = _detection(detected=lead_vs_t2 is not None)
    leads = {"T0": -600.0, "T2": lead_vs_t2} if lead_vs_t2 is not None else {"T0": None, "T2": None}
    base = dict(
        detection=detection,
        detection_time=detection.detection_time,
        lead_seconds_by_onset=leads,
        lead_in_fpr={3: 0.014, 1: 0.035},
        n_lead_in_windows=209,
    )
    base.update(overrides)
    return _resource_metrics(role="incident", **base)


def _grid_bundle(
    label: str = "serving", resources: dict[str, ResourceMetrics] | None = None, **overrides
) -> GridMetrics:
    resources = resources or {}
    base = dict(
        grid=ScoringGrid(label=label, step_samples=1, cadence=pd.Timedelta(minutes=10)),
        per_resource=resources,
        pooled_lead_in_fpr={},
        fleet_time_in_alarm_fraction={},
        fleet_raises_per_week={},
        fleet_runs_per_week={},
        n_eval_windows=sum(r.n_eval_windows for r in resources.values()),
        vus_pr={"value": None, "reason": "not implemented"},
    )
    base.update(overrides)
    return GridMetrics(**base)


def _case_bundle(
    kind: str = "incident",
    grids: dict[str, GridMetrics] | None = None,
    context: dict | None = None,
) -> CaseMetrics:
    return CaseMetrics(case_id="case", case_kind=kind, grids=grids or {}, context=context or {})


def _rubric(gates: dict, headline: str = "serving", profile: str = "aro_node") -> dict:
    rubric = _base_rubric()
    rubric["gates"] = gates
    rubric["headline_grid"] = headline
    rubric["profile"] = profile
    return rubric


_LEAD_GATE = {"detection_lead": {"required": True, "min_lead_vs": "T2", "min_lead_seconds": 0}}
_FPR_GATE = {"lead_in_fpr": {"required": True, "max_fraction": 0.02, "min_eval_windows": 150}}
_FATIGUE_GATE = {
    "alarm_fatigue": {
        "required": True,
        "grid": "serving",
        "sustain": 1,
        "max_time_in_alarm_fraction_per_resource": 0.05,
        "max_fleet_raises_per_week": 10,
    }
}


class TestGateVerdicts:
    def test_no_pre_onset_bridging_both_verdicts(self) -> None:
        for bridged, expected in ((False, True), (True, False)):
            resource = _incident_resource(840.0, detection=_detection(bridged=bridged))
            case = _case_bundle(grids={"serving": _grid_bundle(resources={"node-a": resource})})
            result = evaluate_rubric(_rubric({"no_pre_onset_bridging": {"required": True}}), case)
            gate = result.gates[0]
            assert gate.name == "no_pre_onset_bridging"
            assert gate.passed is expected
            assert gate.grid == "serving"
            assert gate.observed["bridged"] == {"node-a": bridged}
            assert gate.detail
            assert result.passed is expected
            assert result.rubric_version == 1

    def test_detection_lead_both_verdicts_and_undetected(self) -> None:
        for lead, expected in ((840.0, True), (-120.0, False), (None, False)):
            resource = _incident_resource(lead)
            case = _case_bundle(grids={"serving": _grid_bundle(resources={"node-a": resource})})
            gate = evaluate_rubric(_rubric(dict(_LEAD_GATE)), case).gates[0]
            assert gate.passed is expected, lead
            assert gate.grid == "serving"
            assert gate.observed["lead_seconds"] == {"node-a": lead}
            assert gate.observed["min_lead_vs"] == "T2"

    def test_lead_in_fpr_verdicts_and_underpowered_fails(self) -> None:
        cases = (
            ({3: 0.014, 1: 0.035}, 209, True),
            ({3: 0.05, 1: 0.10}, 209, False),
            ({3: 0.0, 1: 0.0}, 48, False),  # underpowered is a FAIL, never a vacuous pass
        )
        for fractions, n_lead_in, expected in cases:
            resource = _incident_resource(840.0, lead_in_fpr=fractions, n_lead_in_windows=n_lead_in)
            case = _case_bundle(grids={"serving": _grid_bundle(resources={"node-a": resource})})
            gate = evaluate_rubric(_rubric(dict(_FPR_GATE)), case).gates[0]
            assert gate.passed is expected, (fractions, n_lead_in)
            assert gate.observed["n_lead_in_windows"] == {"node-a": n_lead_in}
            assert gate.observed["sustain"] == 3
        underpowered = evaluate_rubric(
            _rubric(dict(_FPR_GATE)),
            _case_bundle(
                grids={
                    "serving": _grid_bundle(
                        resources={
                            "node-a": _incident_resource(
                                840.0, lead_in_fpr={3: 0.0, 1: 0.0}, n_lead_in_windows=48
                            )
                        }
                    )
                }
            ),
        ).gates[0]
        assert "underpowered" in underpowered.detail

    def test_alarm_fatigue_verdicts_and_master2_regression(self) -> None:
        def healthy_case(fraction: dict, fleet_raises: dict) -> CaseMetrics:
            resource = _resource_metrics(
                role="excluded",
                time_in_alarm_fraction=fraction,
                raises_per_week={3: 1.0, 1: 1.4},
                runs_per_week={3: 1.0, 1: 1.4},
                observed_span_days=7.0,
            )
            grid = _grid_bundle(
                resources={"node-a": resource},
                fleet_time_in_alarm_fraction={3: 0.001, 1: 0.002},
                fleet_raises_per_week=fleet_raises,
                fleet_runs_per_week=fleet_raises,
            )
            return _case_bundle(kind="healthy_reference", grids={"serving": grid})

        clean = healthy_case({3: 0.001, 1: 0.002}, {3: 1.0, 1: 3.0})
        gate = evaluate_rubric(_rubric(dict(_FATIGUE_GATE)), clean).gates[0]
        assert gate.passed is True
        assert gate.observed["sustain"] == 1

        # The master-2 regression in miniature: the run count is small while
        # the time-in-alarm fraction is ~1.0; the gate must fail on the fraction.
        pinned = healthy_case({3: 0.99, 1: 0.997}, {3: 1.0, 1: 1.4})
        gate = evaluate_rubric(_rubric(dict(_FATIGUE_GATE)), pinned).gates[0]
        assert gate.passed is False
        assert gate.observed["time_in_alarm_fraction"] == {"node-a": 0.997}

        noisy_fleet = healthy_case({3: 0.001, 1: 0.002}, {3: 8.0, 1: 22.0})
        gate = evaluate_rubric(_rubric(dict(_FATIGUE_GATE)), noisy_fleet).gates[0]
        assert gate.passed is False
        assert gate.observed["fleet_raises_per_week"] == 22.0

    def test_negative_controls_clean_both_verdicts(self) -> None:
        def control(runs: int) -> ResourceMetrics:
            return _resource_metrics(
                role="negative_control",
                resource_id="node-c",
                threshold=0.5,
                slice_stats_by_threshold={
                    "own": {
                        "n_windows_in_slice": 19,
                        "over_0.5000": {
                            "windows_over": runs,
                            "frac_over": 0.0,
                            "sustained_runs": runs,
                        },
                    },
                    "node-a": {"n_windows_in_slice": 19},
                },
                exceedances_by_threshold={"own": runs, "node-a": 0},
            )

        for runs, expected in ((0, True), (2, False)):
            resources = {"node-a": _incident_resource(840.0), "node-c": control(runs)}
            case = _case_bundle(grids={"serving": _grid_bundle(resources=resources)})
            gate = evaluate_rubric(
                _rubric({"negative_controls_clean": {"required": True}}), case
            ).gates[0]
            assert gate.passed is expected
            assert gate.observed["sustained_runs"] == {"node-c": runs}

    def test_coverage_integrity_reads_context_and_names_population(self) -> None:
        resource = _incident_resource(840.0)
        grids = {"serving": _grid_bundle(resources={"node-a": resource})}
        clean = _case_bundle(
            grids=grids,
            context={"hygiene_population": "pre_onset", "eligibility": {"node-a": []}},
        )
        gate = evaluate_rubric(_rubric({"coverage_integrity": {"required": True}}), clean).gates[0]
        assert gate.passed is True
        assert gate.observed["hygiene_population"] == "pre_onset"

        dirty = _case_bundle(
            grids=grids,
            context={
                "hygiene_population": "calibration",
                "eligibility": {"node-a": ["divergent-coverage:cpuUsageNanoCores"]},
            },
        )
        gate = evaluate_rubric(_rubric({"coverage_integrity": {"required": True}}), dirty).gates[0]
        assert gate.passed is False
        assert gate.observed["hygiene_population"] == "calibration"
        assert "divergent-coverage" in gate.detail

        unverdicted = _case_bundle(
            grids=grids, context={"hygiene_population": "pre_onset", "eligibility": {}}
        )
        gate = evaluate_rubric(
            _rubric({"coverage_integrity": {"required": True}}), unverdicted
        ).gates[0]
        assert gate.passed is False  # a resource without a verdict cannot be confirmed

    def test_sanity_both_verdicts(self) -> None:
        sound = _case_bundle(
            grids={"serving": _grid_bundle(resources={"node-a": _incident_resource(840.0)})},
            context={"profile": "aro_node", "labels_resources_present": {"node-a": True}},
        )
        gate = evaluate_rubric(_rubric({"sanity": {"required": True}}), sound).gates[0]
        assert gate.passed is True

        bad_threshold = _case_bundle(
            grids={
                "serving": _grid_bundle(
                    resources={"node-a": _incident_resource(840.0, threshold=0.0)}
                )
            },
            context={"profile": "aro_node", "labels_resources_present": {"node-a": True}},
        )
        gate = evaluate_rubric(_rubric({"sanity": {"required": True}}), bad_threshold).gates[0]
        assert gate.passed is False
        assert "threshold" in gate.detail

        wrong_profile = _case_bundle(
            grids={"serving": _grid_bundle(resources={"node-a": _incident_resource(840.0)})},
            context={"profile": "exathlon", "labels_resources_present": {"node-a": True}},
        )
        gate = evaluate_rubric(_rubric({"sanity": {"required": True}}), wrong_profile).gates[0]
        assert gate.passed is False
        assert "profile" in gate.detail

        absent_resource = _case_bundle(
            grids={"serving": _grid_bundle(resources={"node-a": _incident_resource(840.0)})},
            context={"profile": "aro_node", "labels_resources_present": {"node-b": False}},
        )
        gate = evaluate_rubric(_rubric({"sanity": {"required": True}}), absent_resource).gates[0]
        assert gate.passed is False


class TestBindingAndRollup:
    def test_headline_grid_flip_changes_the_verdict(self) -> None:
        # Detection passes sustain-3 on the serving grid, fails on offline:
        # flipping headline_grid flips the detection_lead verdict; one
        # RubricResult per case regardless of grid count.
        serving = _grid_bundle(label="serving", resources={"node-a": _incident_resource(840.0)})
        offline_grid = GridMetrics(
            grid=ScoringGrid(label="offline", step_samples=10),
            per_resource={"node-a": _incident_resource(-300.0)},
            pooled_lead_in_fpr={},
            fleet_time_in_alarm_fraction={},
            fleet_raises_per_week={},
            fleet_runs_per_week={},
            n_eval_windows=250,
            vus_pr={"value": None, "reason": "not implemented"},
        )
        case = _case_bundle(grids={"serving": serving, "offline": offline_grid})

        on_serving = evaluate_rubric(_rubric(dict(_LEAD_GATE), headline="serving"), case)
        assert on_serving.gates[0].passed is True
        assert on_serving.gates[0].grid == "serving"

        on_offline = evaluate_rubric(_rubric(dict(_LEAD_GATE), headline="offline"), case)
        assert on_offline.gates[0].passed is False
        assert on_offline.gates[0].grid == "offline"

        assert len(on_serving.gates) == len(on_offline.gates) == 1

    def test_gate_declared_grid_beats_headline(self) -> None:
        healthy = _resource_metrics(
            role="excluded",
            time_in_alarm_fraction={3: 0.001, 1: 0.002},
            raises_per_week={3: 1.0, 1: 1.4},
            runs_per_week={3: 1.0, 1: 1.4},
            observed_span_days=7.0,
        )
        serving = _grid_bundle(
            resources={"node-a": healthy}, fleet_raises_per_week={3: 1.0, 1: 3.0}
        )
        offline_grid = GridMetrics(
            grid=ScoringGrid(label="offline", step_samples=10),
            per_resource={},
            pooled_lead_in_fpr={},
            fleet_time_in_alarm_fraction={},
            fleet_raises_per_week={},
            fleet_runs_per_week={},
            n_eval_windows=0,
            vus_pr={"value": None, "reason": "not implemented"},
        )
        case = _case_bundle(
            kind="healthy_reference", grids={"serving": serving, "offline": offline_grid}
        )
        result = evaluate_rubric(_rubric(dict(_FATIGUE_GATE), headline="offline"), case)
        assert result.gates[0].grid == "serving"  # the gate's own grid, not the headline
        assert result.gates[0].passed is True

    def test_missing_bound_grid_is_a_spec_error(self) -> None:
        case = _case_bundle(grids={"offline": _grid_bundle(label="offline")})
        with pytest.raises(SpecError, match="binds to grid 'serving'"):
            evaluate_rubric(_rubric(dict(_LEAD_GATE), headline="serving"), case)

    def test_allow_absent_semantics(self) -> None:
        # An incident case with no control-role resources: the required gate
        # raises SpecError; with allow_absent it passes vacuously.
        case = _case_bundle(
            grids={"serving": _grid_bundle(resources={"node-a": _incident_resource(840.0)})}
        )
        with pytest.raises(SpecError, match="no applicable cases"):
            evaluate_rubric(_rubric({"negative_controls_clean": {"required": True}}), case)

        result = evaluate_rubric(
            _rubric({"negative_controls_clean": {"required": True, "allow_absent": True}}), case
        )
        gate = result.gates[0]
        assert gate.passed is True
        assert gate.detail == "no applicable cases"
        assert gate.observed == {}
        assert result.passed is True

    def test_rolled_up_verdict_ignores_non_required_failures(self) -> None:
        resource = _incident_resource(840.0, detection=_detection(bridged=True))
        case = _case_bundle(grids={"serving": _grid_bundle(resources={"node-a": resource})})
        gates = {
            "no_pre_onset_bridging": {"required": False},
            "detection_lead": {"required": True, "min_lead_vs": "T2", "min_lead_seconds": 0},
        }
        result = evaluate_rubric(_rubric(gates), case)
        assert [gate.name for gate in result.gates] == list(gates)
        assert result.gates[0].passed is False
        assert result.gates[1].passed is True
        assert result.passed is True

    def test_unknown_gate_in_raw_mapping_is_a_spec_error(self) -> None:
        case = _case_bundle(grids={"serving": _grid_bundle()})
        with pytest.raises(SpecError, match="unknown gate 'bogus_gate'"):
            evaluate_rubric(_rubric({"bogus_gate": {"required": True}}), case)


class TestTorchFreeImport:
    def test_rubric_module_imports_without_torch(self) -> None:
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "import scry.eval.rubric, sys; assert 'torch' not in sys.modules",
            ],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
