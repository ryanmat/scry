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
from synth import PROFILE, gen_capture, make_incident, write_csv, write_labels

from scry.eval.provenance import build_provenance
from scry.eval.rubric import SpecError
from scry.eval.scoring import ScoringGrid
from scry.eval.suite import load_capture, load_suite


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
