# Description: Tests for scry.eval.provenance and scry.eval.suite: run provenance and suite loading.
# Description: Pins provenance fields, suite validation with collected errors, and torch-free imports.

"""Tests for the provenance builder and the suite loader.

``build_provenance`` is pinned through every required field: git_rev (a real
revision inside the repo, "unknown" with a non-repo working directory) plus
the dirty flag, the mandatory model sha256 against an independently computed
hash, per-case data path with byte size and Z-suffix mtime, profile, seq_len,
every grid definition, sustains, detection mode, the threshold policy
describe() passthrough with resolved per-resource thresholds, rubric path and
version, the scryml version, and a Z-suffix UTC generated_at. The output is
pinned JSON-serializable, the member as an eager package export, and the
module as torch-free. ``load_suite`` is pinned through a valid suite with
every path resolved against the suite file's directory, the
incident-requires-labels and healthy-takes-none rules, the collected
one-SpecError report naming every problem at once, the calibration loading
through the one fetch_full_capture loader with the case-style format key, and
the rubric-validation propagation; ``scry.eval.suite`` imports torch-free.
``run_suite`` is pinned end to end on the tiny keeper: one serialized case
entry per case on every rubric-declared grid, per-resource policy resolution
with controls populated inside the incident case, the context conduit under
each case's reported block (never in per_resource), the ramp fixture passing
the detection gates and the pinned alarm failing alarm_fatigue, the rubric
evaluated exactly once per case, case-name filtering, and the lazy package
export. The report is pinned through its top-level shape and both verdicts,
the exact 9.3 grid and 20-field per-resource key sets, sustain fields as
string-keyed objects with both accountings always present (empty dicts as
nulls), the validator rejecting a stripped accounting, JSON
serializability, and determinism modulo generated_at.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest
import yaml
from synth import (
    PROFILE,
    gen_capture,
    make_incident,
    write_csv,
    write_labels,
    write_run_suite_tree,
)

from scry.eval.provenance import build_provenance
from scry.eval.rubric import SpecError
from scry.eval.scoring import ScoringGrid
from scry.eval.suite import load_capture, load_suite, run_suite


def _provenance(tmp_path: Path, **overrides) -> tuple[dict, Path, Path]:
    model = tmp_path / "keeper.pt"
    model.write_bytes(b"model-bytes")
    capture = tmp_path / "capture.csv"
    capture.write_text("resource_id,metric_name,timestamp,value\n")
    kwargs = dict(
        model_path=str(model),
        profile="aro_node",
        seq_len=30,
        grids={
            "offline": ScoringGrid(label="offline", step_samples=10),
            "serving": ScoringGrid(
                label="serving", step_samples=1, cadence=pd.Timedelta(minutes=10)
            ),
        },
        sustains=(3, 1),
        detection_mode="no_bridging",
        policy_description={
            "type": "PerResourceMargin",
            "margin": 2.0,
            "fallback": 0.1932,
            "per_resource": {"node-a": 0.31},
        },
        rubric_path="config/rubrics/aro_node_v1.yaml",
        rubric_version=1,
        case_data_paths={"phase4": str(capture)},
    )
    kwargs.update(overrides)
    return build_provenance(**kwargs), model, capture


class TestBuildProvenance:
    def test_required_fields(self, tmp_path: Path) -> None:
        provenance, model, capture = _provenance(tmp_path)
        assert re.fullmatch(r"[0-9a-f]{40}", provenance["git_rev"])  # runs inside the repo
        assert isinstance(provenance["git_dirty"], bool)
        assert provenance["model_path"] == str(model)
        assert provenance["model_sha256"] == hashlib.sha256(b"model-bytes").hexdigest()
        assert provenance["profile"] == "aro_node"
        assert provenance["seq_len"] == 30
        assert provenance["grids"] == {
            "offline": {"label": "offline", "step_samples": 10, "cadence_seconds": None},
            "serving": {"label": "serving", "step_samples": 1, "cadence_seconds": 600.0},
        }
        assert provenance["sustains"] == [3, 1]
        assert provenance["detection_mode"] == "no_bridging"
        assert provenance["threshold_policy"]["per_resource"] == {"node-a": 0.31}
        assert provenance["rubric_path"] == "config/rubrics/aro_node_v1.yaml"
        assert provenance["rubric_version"] == 1
        assert provenance["scryml_version"]
        assert provenance["generated_at"].endswith("Z")
        parsed = datetime.fromisoformat(provenance["generated_at"].replace("Z", "+00:00"))
        assert parsed.tzinfo is not None

    def test_per_case_data_identity(self, tmp_path: Path) -> None:
        provenance, _, capture = _provenance(tmp_path)
        case = provenance["cases"]["phase4"]
        assert case["data_path"] == str(capture)
        assert case["bytes"] == capture.stat().st_size
        expected_mtime = datetime.fromtimestamp(capture.stat().st_mtime, tz=timezone.utc)
        assert case["mtime"] == expected_mtime.isoformat().replace("+00:00", "Z")

    def test_git_rev_unknown_outside_a_repo(self, tmp_path: Path) -> None:
        provenance, _, _ = _provenance(tmp_path, repo_dir=str(tmp_path))
        assert provenance["git_rev"] == "unknown"
        assert provenance["git_dirty"] is None

    def test_output_is_json_serializable(self, tmp_path: Path) -> None:
        provenance, _, _ = _provenance(tmp_path)
        json.dumps(provenance)


class TestEagerExports:
    def test_build_provenance_exports_eagerly(self) -> None:
        import scry.eval
        from scry.eval import provenance as provenance_module

        assert scry.eval.build_provenance is provenance_module.build_provenance
        assert "build_provenance" in scry.eval.__all__
        assert "build_provenance" in vars(scry.eval)


_MINIMAL_RUBRIC = {
    "version": 1,
    "profile": "aro_node",
    "detection": {"mode": "no_bridging", "onset_anchor": "T0", "sustain": 3},
    "grids": {"serving": {"step_samples": 1, "cadence_minutes": 10}},
    "headline_grid": "serving",
    "gates": {"sanity": {"required": True}},
}


def _write_suite_tree(tmp_path: Path) -> tuple[Path, dict]:
    """A valid suite in a subdirectory, every path relative to the suite file."""
    suite_dir = tmp_path / "suitedir"
    (suite_dir / "data").mkdir(parents=True)
    (suite_dir / "keeper.pt").write_bytes(b"checkpoint-bytes")
    incident_df, _ = gen_capture("node-a", 120, seed=91)
    write_csv(incident_df, suite_dir / "data" / "incident.csv")
    healthy_df, _ = gen_capture("node-b", 120, seed=92)
    write_csv(healthy_df, suite_dir / "data" / "healthy.csv")
    calibration_df, _ = gen_capture("node-a", 120, seed=93)
    write_csv(calibration_df, suite_dir / "data" / "calibration.csv")
    start = pd.Timestamp("2026-01-01T01:00:00Z")
    write_labels(
        suite_dir / "data" / "labels.json",
        [make_incident("node-a", "cpu", start, start + pd.Timedelta(minutes=30))],
    )
    (suite_dir / "rubric.yaml").write_text(yaml.safe_dump(_MINIMAL_RUBRIC))
    suite = {
        "version": 1,
        "suite": "tmp_suite",
        "candidate": {"type": "reconstruction", "model": "keeper.pt", "profile": "aro_node"},
        "threshold_policy": {
            "type": "per_resource_margin",
            "calibration": "data/calibration.csv",
            "format": "csv",
            "quantile": 0.99,
            "margin": 2.0,
        },
        "rubric": "rubric.yaml",
        "cases": [
            {
                "name": "incident_case",
                "kind": "incident_capture",
                "capture": "data/incident.csv",
                "labels": "data/labels.json",
                "format": "csv",
            },
            {
                "name": "healthy_case",
                "kind": "healthy_reference",
                "capture": "data/healthy.csv",
                "format": "csv",
            },
        ],
    }
    return suite_dir, suite


def _dump_suite(suite_dir: Path, suite: dict) -> str:
    path = suite_dir / "suite.yaml"
    path.write_text(yaml.safe_dump(suite))
    return str(path)


class TestLoadSuite:
    def test_valid_suite_loads_with_paths_resolved(self, tmp_path: Path) -> None:
        suite_dir, suite = _write_suite_tree(tmp_path)
        loaded = load_suite(_dump_suite(suite_dir, suite))
        assert loaded["suite"] == "tmp_suite"
        assert loaded["candidate"]["model"] == str(suite_dir / "keeper.pt")
        assert loaded["threshold_policy"]["calibration"] == str(
            suite_dir / "data" / "calibration.csv"
        )
        assert loaded["rubric"] == str(suite_dir / "rubric.yaml")
        incident, healthy = loaded["cases"]
        assert incident["capture"] == str(suite_dir / "data" / "incident.csv")
        assert incident["labels"] == str(suite_dir / "data" / "labels.json")
        assert healthy["capture"] == str(suite_dir / "data" / "healthy.csv")
        assert "labels" not in healthy

    def test_incident_requires_labels_and_healthy_takes_none(self, tmp_path: Path) -> None:
        suite_dir, suite = _write_suite_tree(tmp_path)
        del suite["cases"][0]["labels"]
        with pytest.raises(SpecError, match="requires 'labels'"):
            load_suite(_dump_suite(suite_dir, suite))

        suite_dir, suite = _write_suite_tree(tmp_path / "second")
        suite["cases"][1]["labels"] = "data/labels.json"
        with pytest.raises(SpecError, match="takes no labels"):
            load_suite(_dump_suite(suite_dir, suite))

    def test_spec_errors_are_collected_into_one_error(self, tmp_path: Path) -> None:
        suite_dir, suite = _write_suite_tree(tmp_path)
        suite["extra_knob"] = True
        suite["threshold_policy"]["type"] = "bogus_policy"
        suite["cases"][0]["capture"] = "data/nope.csv"
        with pytest.raises(SpecError) as excinfo:
            load_suite(_dump_suite(suite_dir, suite))
        message = str(excinfo.value)
        assert "extra_knob" in message
        assert "bogus_policy" in message
        assert "nope.csv" in message

    def test_calibration_loads_through_the_one_loader(self, tmp_path: Path) -> None:
        import scry.data.fetcher
        import scry.eval.suite as suite_module

        assert suite_module.fetch_full_capture is scry.data.fetcher.fetch_full_capture
        suite_dir, suite = _write_suite_tree(tmp_path)
        loaded = load_suite(_dump_suite(suite_dir, suite))
        policy = loaded["threshold_policy"]
        calibration = load_capture(
            policy["calibration"], profile=PROFILE, data_format=policy["format"]
        )
        assert not calibration.empty
        assert set(calibration["resource_id"]) == {"node-a"}

    def test_rubric_validation_failure_propagates(self, tmp_path: Path) -> None:
        suite_dir, suite = _write_suite_tree(tmp_path)
        broken = dict(_MINIMAL_RUBRIC, gates={"bogus_gate": {"required": True}})
        (suite_dir / "rubric.yaml").write_text(yaml.safe_dump(broken))
        with pytest.raises(SpecError, match="rubric validation"):
            load_suite(_dump_suite(suite_dir, suite))

    def test_calibration_required_for_calibration_policies(self, tmp_path: Path) -> None:
        suite_dir, suite = _write_suite_tree(tmp_path)
        del suite["threshold_policy"]["calibration"]
        with pytest.raises(SpecError, match="requires 'calibration'"):
            load_suite(_dump_suite(suite_dir, suite))


@pytest.fixture(scope="module")
def run_result(keeper_path: str, tmp_path_factory: pytest.TempPathFactory) -> dict:
    # Module-scoped fixtures run outside the per-test active-profile restore,
    # so snapshot and restore around the run like the keeper_path fixture.
    import scry.data.feature_engineering as fe

    prev_profile = fe._active_config
    suite_path = write_run_suite_tree(tmp_path_factory.mktemp("run"), keeper_path)
    result = run_suite(load_suite(suite_path))
    fe._active_config = prev_profile
    return result


_RESOURCE_FIELDS = (
    "resource_id",
    "role",
    "threshold",
    "threshold_source",
    "n_eval_windows",
    "detection",
    "detection_time",
    "lead_seconds_by_onset",
    "lead_in_fpr",
    "n_lead_in_windows",
    "coverage_fraction",
    "clear_lead_vs_end_s",
    "alarm_seconds_in_incident",
    "time_in_alarm_fraction",
    "raises_per_week",
    "runs_per_week",
    "sustained_run_counts",
    "observed_span_days",
    "slice_stats_by_threshold",
    "exceedances_by_threshold",
)


_TRACKED_RUBRIC = Path(__file__).resolve().parent.parent / "config" / "rubrics" / "aro_node_v1.yaml"


class TestRunSuite:
    def test_tracked_rubric_can_evaluate_a_mixed_suite(
        self, keeper_path: str, tmp_path: Path
    ) -> None:
        """The shipped rubric must be runnable on the case kinds it describes.

        A required gate whose population is empty raises SpecError unless the
        gate declares allow_absent, and the kind-scoped gates are empty on the
        kind they do not apply to: the detection gates on a healthy reference,
        alarm_fatigue on an incident. Without allow_absent the tracked rubric
        evaluates no suite at all, which would make the harness unusable with
        the very rubric it ships.
        """
        suite = load_suite(write_run_suite_tree(tmp_path, keeper_path))
        suite["rubric"] = str(_TRACKED_RUBRIC)

        report = run_suite(suite)

        assert {case["kind"] for case in report["cases"]} == {"incident", "healthy_reference"}
        for case in report["cases"]:
            names = {gate["name"] for gate in case["rubric"]["gates"]}
            assert "alarm_fatigue" in names
            assert "no_pre_onset_bridging" in names

    def test_one_case_entry_per_case_on_every_declared_grid(self, run_result: dict) -> None:
        assert [case["name"] for case in run_result["cases"]] == ["ramp_incident", "pinned_alarm"]
        incident, healthy = run_result["cases"]
        assert incident["kind"] == "incident"
        assert healthy["kind"] == "healthy_reference"
        for case in (incident, healthy):
            assert set(case["grids"]) == {"offline", "serving"}
            for label, grid in case["grids"].items():
                assert grid["grid"]["label"] == label

    def test_policy_resolved_per_resource_and_controls_populated(self, run_result: dict) -> None:
        grid = run_result["cases"][0]["grids"]["serving"]
        assert set(grid["per_resource"]) == {"node-a", "node-ctl"}
        assert grid["per_resource"]["node-a"]["role"] == "incident"
        assert grid["per_resource"]["node-a"]["threshold_source"] == "per_resource"
        control = grid["per_resource"]["node-ctl"]
        assert control["role"] == "negative_control"
        assert set(control["slice_stats_by_threshold"]) == {"own", "node-a"}

    def test_context_conduit_serializes_under_reported(self, run_result: dict) -> None:
        reported = run_result["cases"][0]["reported"]
        assert reported["hygiene_population"] == "calibration"
        assert reported["profile"] == PROFILE
        assert reported["labels_resources_present"] == {"node-a": True, "node-ctl": True}
        eligibility = reported["eligibility"]
        assert {"node-a", "node-ctl", "node-hot"} <= set(eligibility)
        assert eligibility["node-a"] == []
        # The conduit never leaks into per_resource: its enumeration stays closed.
        for grid in run_result["cases"][0]["grids"].values():
            for resource in grid["per_resource"].values():
                assert set(resource) == set(_RESOURCE_FIELDS)

    def test_ramp_passes_detection_gates(self, run_result: dict) -> None:
        gates = {gate["name"]: gate for gate in run_result["cases"][0]["rubric"]["gates"]}
        assert gates["no_pre_onset_bridging"]["passed"] is True
        assert gates["detection_lead"]["passed"] is True
        assert gates["detection_lead"]["observed"]["lead_seconds"]["node-a"] > 0
        assert gates["detection_lead"]["grid"] == "serving"

    def test_pinned_alarm_fails_alarm_fatigue(self, run_result: dict) -> None:
        healthy = run_result["cases"][1]
        gates = {gate["name"]: gate for gate in healthy["rubric"]["gates"]}
        assert gates["alarm_fatigue"]["passed"] is False
        assert healthy["rubric"]["passed"] is False

    def test_rubric_evaluated_once_per_case(
        self, keeper_path: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import scry.eval.suite as suite_module
        from scry.eval.rubric import evaluate_rubric as real_evaluate_rubric

        calls: list[str] = []

        def counting(rubric, case_metrics):
            calls.append(case_metrics.case_id)
            return real_evaluate_rubric(rubric, case_metrics)

        monkeypatch.setattr(suite_module, "evaluate_rubric", counting)
        suite_path = write_run_suite_tree(tmp_path, keeper_path)
        run_suite(load_suite(suite_path))
        assert calls == ["ramp_incident", "pinned_alarm"]

    def test_case_names_filter(self, keeper_path: str, tmp_path: Path) -> None:
        suite_path = write_run_suite_tree(tmp_path, keeper_path)
        suite = load_suite(suite_path)
        result = run_suite(suite, case_names=["pinned_alarm"])
        assert [case["name"] for case in result["cases"]] == ["pinned_alarm"]
        with pytest.raises(SpecError, match="unknown case"):
            run_suite(suite, case_names=["nope"])

    def test_run_suite_is_a_lazy_package_export(self) -> None:
        import scry.eval

        assert "run_suite" not in vars(scry.eval)  # lazy, not eager
        assert scry.eval.run_suite is run_suite
        assert "run_suite" in scry.eval.__all__


class TestReport:
    def test_top_level_shape_and_fail_verdict(self, run_result: dict) -> None:
        assert set(run_result) == {"provenance", "suite", "cases", "verdict", "exit_code"}
        assert run_result["suite"] == "run_suite"
        assert run_result["verdict"] == "FAIL"  # the pinned alarm fails a required gate
        assert run_result["exit_code"] == 1
        provenance = run_result["provenance"]
        assert re.fullmatch(r"[0-9a-f]{64}", provenance["model_sha256"])
        assert set(provenance["grids"]) == {"offline", "serving"}
        assert provenance["threshold_policy"]["type"] == "PerResourceMargin"
        assert set(provenance["cases"]) == {"ramp_incident", "pinned_alarm"}
        for case in run_result["cases"]:
            assert set(case) == {"name", "kind", "grids", "rubric", "reported"}
            for gate in case["rubric"]["gates"]:
                assert set(gate) == {"name", "required", "passed", "grid", "observed", "detail"}

    def test_pass_verdict_and_exit_zero(self, keeper_path: str, tmp_path: Path) -> None:
        suite_path = write_run_suite_tree(tmp_path, keeper_path)
        report = run_suite(load_suite(suite_path), case_names=["ramp_incident"])
        assert report["verdict"] == "PASS"
        assert report["exit_code"] == 0

    def test_grid_serialization_carries_exactly_the_schema_keys(self, run_result: dict) -> None:
        grid = run_result["cases"][0]["grids"]["serving"]
        assert set(grid) == {
            "grid",
            "per_resource",
            "pooled_lead_in_fpr",
            "fleet_time_in_alarm_fraction",
            "fleet_raises_per_week",
            "fleet_runs_per_week",
            "n_eval_windows",
            "vus_pr",
        }
        assert set(grid["grid"]) == {"label", "step_samples", "cadence_seconds"}
        assert grid["grid"] == {"label": "serving", "step_samples": 1, "cadence_seconds": 600.0}
        resource = grid["per_resource"]["node-a"]
        assert set(resource) == set(_RESOURCE_FIELDS)
        assert set(resource["detection"]) == {
            "detected",
            "detection_time",
            "lead_seconds",
            "bridged",
            "n_runs_pre_onset",
            "n_runs_at_or_after",
        }
        assert resource["detection_time"].endswith("Z")

    def test_sustain_fields_serialize_with_both_string_keys(self, run_result: dict) -> None:
        for case in run_result["cases"]:
            for grid in case["grids"].values():
                for field_name in (
                    "pooled_lead_in_fpr",
                    "fleet_time_in_alarm_fraction",
                    "fleet_raises_per_week",
                    "fleet_runs_per_week",
                ):
                    assert set(grid[field_name]) == {"3", "1"}
                for resource in grid["per_resource"].values():
                    for field_name in (
                        "lead_in_fpr",
                        "time_in_alarm_fraction",
                        "raises_per_week",
                        "runs_per_week",
                        "sustained_run_counts",
                    ):
                        assert set(resource[field_name]) == {"3", "1"}
        # A role-neutral empty dict still serializes with both keys, as nulls.
        incident_resource = run_result["cases"][0]["grids"]["serving"]["per_resource"]["node-a"]
        assert incident_resource["time_in_alarm_fraction"] == {"3": None, "1": None}

    def test_validator_rejects_a_stripped_accounting(self, run_result: dict) -> None:
        from copy import deepcopy

        from scry.eval.suite import validate_report

        validate_report(run_result)
        stripped = deepcopy(run_result)
        del stripped["cases"][0]["grids"]["serving"]["per_resource"]["node-a"]["lead_in_fpr"]["3"]
        with pytest.raises(SpecError, match="lead_in_fpr"):
            validate_report(stripped)

    def test_report_is_json_serializable(self, run_result: dict) -> None:
        json.dumps(run_result)

    def test_determinism_modulo_generated_at(self, keeper_path: str, tmp_path: Path) -> None:
        suite_path = write_run_suite_tree(tmp_path, keeper_path)
        first = json.loads(json.dumps(run_suite(load_suite(suite_path))))
        second = json.loads(json.dumps(run_suite(load_suite(suite_path))))
        first["provenance"].pop("generated_at")
        second["provenance"].pop("generated_at")
        assert first == second


class TestTorchFreeImport:
    def test_provenance_module_imports_without_torch(self) -> None:
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "import scry.eval.provenance, sys; assert 'torch' not in sys.modules",
            ],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr

    def test_suite_module_imports_without_torch(self) -> None:
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "import scry.eval.suite, sys; assert 'torch' not in sys.modules",
            ],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
