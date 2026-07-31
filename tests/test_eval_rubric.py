# Description: Tests for scry.eval.rubric: rubric loading, result dataclasses, and SpecError.
# Description: Pins schema validation, the typed exception, eager exports, and the torch-free import.

"""Tests for rubric loading and the gate-result types.

``load_rubric`` is pinned through a valid round-trip, the enumerated
unknown-gate error, the missing-version error, the non-mapping guard, the
grids-well-formed rule, headline_grid and per-gate grid references to
declared grids, and the free-form reported/known_divergences blocks.
``GateResult``/``RubricResult`` are pinned as frozen with exactly their
declared fields. ``SpecError`` is pinned as a typed ValueError subclass
importable from ``scry.eval``, ``evaluate_rubric`` as declared but not
implemented, the rubric members as eager package exports, and the module as
torch-free.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import pytest
import yaml

from scry.eval.metrics import CaseMetrics
from scry.eval.rubric import (
    GATE_NAMES,
    GateResult,
    RubricResult,
    SpecError,
    evaluate_rubric,
    load_rubric,
)


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

    def test_evaluate_rubric_declared_but_not_implemented(self, tmp_path: Path) -> None:
        rubric = load_rubric(_write_rubric(tmp_path, _base_rubric()))
        bundle = CaseMetrics(case_id="c", case_kind="healthy_reference", grids={})
        with pytest.raises(NotImplementedError):
            evaluate_rubric(rubric, bundle)


class TestEagerExports:
    def test_rubric_members_export_eagerly(self) -> None:
        import scry.eval
        from scry.eval import rubric as rubric_module

        for name in ("GateResult", "RubricResult", "SpecError", "evaluate_rubric", "load_rubric"):
            assert getattr(scry.eval, name) is getattr(rubric_module, name)
            assert name in scry.eval.__all__
            assert name in vars(scry.eval)  # eager, not resolved via __getattr__


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
