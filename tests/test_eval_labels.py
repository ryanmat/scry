# Description: Tests for scry.eval.labels: constants, case validation, and timestamp normalization.
# Description: Enforces the torch-free import contract for the labels module via a subprocess check.

"""Tests for the eval label model.

``LabelCase`` validation is pinned through its enumerated error messages
(unknown onset, unknown role, absent primary_onset target, naive timestamp,
incident without T0, incident without end) and its UTC normalization of aware
non-UTC timestamps. ``LabelSet`` accessors are pinned on a three-role fixture.
A subprocess test enforces the spec's torch-free acceptance criterion for
``scry.eval.labels``.
"""

from __future__ import annotations

import subprocess
import sys
from typing import Any

import pandas as pd
import pytest

from scry.eval.labels import ONSET_NAMES, ROLES, LabelCase, LabelSet

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
