# Description: Tests for scry.eval.labels: constants, validation, load/dump, and v1 back-compat.
# Description: Enforces the torch-free import contract for the labels module via a subprocess check.

"""Tests for the eval label model.

``LabelCase`` validation is pinned through its enumerated error messages
(unknown onset, unknown role, absent primary_onset target, naive timestamp,
incident without T0, incident without end) and its UTC normalization of aware
non-UTC timestamps. ``LabelSet`` accessors are pinned on a three-role fixture.
``load_labels``/``dump_labels`` are pinned by v2 round-trip equality, the
top-level dispatch rules, and the v1 (single-onset list) back-compat mapping.
A subprocess test enforces the spec's torch-free acceptance criterion for
``scry.eval.labels``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from synth import make_incident, write_labels

from scry.eval.labels import (
    ONSET_NAMES,
    ROLES,
    LabelCase,
    LabelSet,
    dump_labels,
    from_v1,
    load_labels,
)

_T0 = pd.Timestamp("2026-01-01T02:00:00Z")
_T1 = pd.Timestamp("2026-01-01T02:30:00Z")
_END = pd.Timestamp("2026-01-01T04:00:00Z")


def make_case(**overrides: Any) -> LabelCase:
    """A valid incident case; overrides drive each validation rule."""
    base: dict[str, Any] = {
        "resource_id": "node-a",
        "role": "incident",
        "type": "cpu_ramp",
        "onsets": {"T0": _T0},
        "primary_onset": "T0",
        "end": _END,
        "notes": None,
    }
    base.update(overrides)
    return LabelCase(**base)


class TestConstants:
    def test_onset_names(self) -> None:
        assert ONSET_NAMES == ("T0", "T1", "T2", "T2b")

    def test_roles(self) -> None:
        assert ROLES == ("incident", "negative_control", "excluded")


class TestLabelCaseValidation:
    def test_unknown_onset_message_enumerates_valid_names(self) -> None:
        with pytest.raises(ValueError) as exc:
            make_case(onsets={"T0": _T0, "T3": _T1})
        message = str(exc.value)
        assert "unknown onset 'T3'" in message
        assert "valid onset names: T0, T1, T2, T2b" in message

    def test_unknown_role_message_enumerates_valid_roles(self) -> None:
        with pytest.raises(ValueError) as exc:
            make_case(role="control")
        message = str(exc.value)
        for role in ("incident", "negative_control", "excluded"):
            assert role in message

    def test_primary_onset_naming_absent_onset_rejected(self) -> None:
        with pytest.raises(ValueError, match="primary_onset"):
            make_case(primary_onset="T2")

    def test_naive_onset_timestamp_rejected(self) -> None:
        with pytest.raises(ValueError, match="naive"):
            make_case(onsets={"T0": pd.Timestamp("2026-01-01T02:00:00")})

    def test_naive_end_timestamp_rejected(self) -> None:
        with pytest.raises(ValueError, match="naive"):
            make_case(end=pd.Timestamp("2026-01-01T04:00:00"))

    def test_aware_non_utc_onset_converts_to_utc(self) -> None:
        case = make_case(onsets={"T0": "2026-01-01T04:00:00+02:00"})
        assert case.onsets["T0"] == _T0
        assert str(case.onsets["T0"].tz) == "UTC"

    def test_aware_non_utc_end_converts_to_utc(self) -> None:
        case = make_case(end="2026-01-01T06:00:00+02:00")
        assert case.end == _END
        assert str(case.end.tz) == "UTC"

    def test_incident_without_t0_rejected(self) -> None:
        with pytest.raises(ValueError, match="T0"):
            make_case(onsets={"T1": _T1}, primary_onset="T1")

    def test_incident_without_end_rejected(self) -> None:
        with pytest.raises(ValueError, match="end"):
            make_case(end=None)

    def test_minimal_negative_control_constructs(self) -> None:
        case = make_case(
            role="negative_control", type=None, onsets={}, primary_onset=None, end=None
        )
        assert case.role == "negative_control"
        assert case.onsets == {}


class TestLabelSetAccessors:
    def _label_set(self) -> LabelSet:
        return LabelSet(
            version=2,
            capture="synthetic",
            cases=[
                make_case(resource_id="node-a"),
                make_case(
                    resource_id="node-b",
                    role="negative_control",
                    type=None,
                    onsets={},
                    primary_onset=None,
                    end=None,
                ),
                make_case(
                    resource_id="node-c",
                    role="excluded",
                    type=None,
                    onsets={},
                    primary_onset=None,
                    end=None,
                ),
            ],
        )

    def test_incidents_filters_by_role(self) -> None:
        labels = self._label_set()
        assert [case.resource_id for case in labels.incidents()] == ["node-a"]

    def test_negative_controls_filters_by_role(self) -> None:
        labels = self._label_set()
        assert [case.resource_id for case in labels.negative_controls()] == ["node-b"]

    def test_role_for_present_and_absent_resources(self) -> None:
        labels = self._label_set()
        assert labels.role_for("node-a") == "incident"
        assert labels.role_for("node-b") == "negative_control"
        assert labels.role_for("node-c") == "excluded"
        assert labels.role_for("node-z") == "excluded"


def _v2_set() -> LabelSet:
    """A v2 set spanning all three roles, with a multi-onset incident."""
    return LabelSet(
        version=2,
        capture="synthetic",
        cases=[
            make_case(
                resource_id="node-a",
                onsets={
                    "T0": _T0,
                    "T1": _T1,
                    "T2": pd.Timestamp("2026-01-01T03:00:00Z"),
                    "T2b": pd.Timestamp("2026-01-01T03:30:00Z"),
                },
                notes="synthetic multi-onset incident",
            ),
            make_case(
                resource_id="node-b",
                role="negative_control",
                type=None,
                onsets={},
                primary_onset=None,
                end=None,
            ),
            make_case(
                resource_id="node-c",
                role="excluded",
                type=None,
                onsets={},
                primary_onset=None,
                end=None,
            ),
        ],
    )


class TestV1BackCompat:
    def _v1_entries(self) -> list[dict[str, str]]:
        return [
            make_incident("node-a", "cpu_spike", _T0, _END),
            make_incident("node-b", "cpu_ramp", _T1, _END),
        ]

    def test_from_v1_maps_entries_to_incident_cases(self) -> None:
        labels = from_v1(self._v1_entries())
        assert labels.version == 1
        assert labels.capture is None
        assert [case.resource_id for case in labels.cases] == ["node-a", "node-b"]
        first = labels.cases[0]
        assert first.role == "incident"
        assert first.type == "cpu_spike"
        assert first.onsets == {"T0": _T0}
        assert first.primary_onset == "T0"
        assert first.end == _END

    def test_load_labels_reads_v1_file(self, tmp_path: Path) -> None:
        path = write_labels(tmp_path / "labels_v1.json", self._v1_entries())
        labels = load_labels(path)
        assert labels == from_v1(self._v1_entries())


class TestLoadDumpDispatch:
    def test_dict_with_version_2_is_v2(self, tmp_path: Path) -> None:
        path = str(tmp_path / "labels_v2.json")
        dump_labels(_v2_set(), path)
        assert load_labels(path).version == 2

    def test_unsupported_version_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "labels_v3.json"
        path.write_text(json.dumps({"version": 3, "cases": []}))
        with pytest.raises(ValueError, match="version"):
            load_labels(str(path))

    def test_non_list_non_dict_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "labels_scalar.json"
        path.write_text("42")
        with pytest.raises(ValueError):
            load_labels(str(path))


class TestRoundTrip:
    def test_v2_round_trip_equality(self, tmp_path: Path) -> None:
        original = _v2_set()
        path = str(tmp_path / "labels_v2.json")
        dump_labels(original, path)
        loaded = load_labels(path)
        assert loaded == original
        assert loaded.role_for("node-z") == "excluded"

    def test_dump_always_writes_v2(self, tmp_path: Path) -> None:
        v1_path = write_labels(
            tmp_path / "labels_v1.json", [make_incident("node-a", "cpu_spike", _T0, _END)]
        )
        out_path = str(tmp_path / "labels_dumped.json")
        dump_labels(load_labels(v1_path), out_path)
        raw = json.loads(Path(out_path).read_text())
        assert raw["version"] == 2
        assert load_labels(out_path).version == 2

    def test_dumped_timestamps_use_z_suffix(self, tmp_path: Path) -> None:
        path = str(tmp_path / "labels_v2.json")
        dump_labels(_v2_set(), path)
        raw = json.loads(Path(path).read_text())
        incident = raw["cases"][0]
        assert all(value.endswith("Z") for value in incident["onsets"].values())
        assert incident["end"].endswith("Z")


class TestEagerExports:
    def test_labels_members_export_eagerly_from_package_root(self) -> None:
        import scry.eval

        for name in (
            "ONSET_NAMES",
            "ROLES",
            "LabelCase",
            "LabelSet",
            "load_labels",
            "dump_labels",
            "from_v1",
        ):
            assert name in scry.eval.__all__
            assert getattr(scry.eval, name) is getattr(sys.modules["scry.eval.labels"], name)


class TestTorchFreeImport:
    def test_labels_module_imports_without_torch(self) -> None:
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "import scry.eval.labels, sys; assert 'torch' not in sys.modules",
            ],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
