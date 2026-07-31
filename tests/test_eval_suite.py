# Description: Tests for scry.eval.provenance: the run-provenance block of the report.
# Description: Pins every required field, JSON-serializability, the eager export, and torch-free import.

"""Tests for the provenance builder.

``build_provenance`` is pinned through every required field: git_rev (a real
revision inside the repo, "unknown" with a non-repo working directory) plus
the dirty flag, the mandatory model sha256 against an independently computed
hash, per-case data path with byte size and Z-suffix mtime, profile, seq_len,
every grid definition, sustains, detection mode, the threshold policy
describe() passthrough with resolved per-resource thresholds, rubric path and
version, the scryml version, and a Z-suffix UTC generated_at. The output is
pinned JSON-serializable, the member as an eager package export, and the
module as torch-free.
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

from scry.eval.provenance import build_provenance
from scry.eval.scoring import ScoringGrid


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
