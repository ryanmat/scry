# Description: Versioned incident labels: named onsets and per-resource roles for the eval harness.
# Description: Torch-free; loads v1 or v2 JSON, always dumps v2, and validates on construction.

"""Incident label model with named onsets and per-resource roles.

A ``LabelCase`` assigns one resource a role (incident, negative_control, or
excluded) and, for incidents, named onsets drawn from ``ONSET_NAMES`` plus a
required ``end``. Validation runs in ``__post_init__`` so every construction
path enforces the same rules; error messages enumerate the valid alternatives.
Timestamps normalize to UTC; naive timestamps are rejected.

``load_labels`` reads either schema version, dispatching on the top-level JSON
type: a list is the v1 single-onset incident format, an object with
``"version": 2`` is the versioned multi-onset format. ``dump_labels`` always
writes v2. Everything here is importable without torch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd

ONSET_NAMES: tuple[str, ...] = ("T0", "T1", "T2", "T2b")
ROLES: tuple[str, ...] = ("incident", "negative_control", "excluded")


def _utc_timestamp(value: str | pd.Timestamp, context: str) -> pd.Timestamp:
    """Parse ``value`` as an aware UTC timestamp; reject naive input."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        raise ValueError(
            f"naive timestamp {value!r} for {context}; timestamps must be timezone-aware UTC"
        )
    return ts.tz_convert("UTC")


@dataclass(frozen=True)
class LabelCase:
    """One resource's role and, for incidents, its named onsets and end."""

    resource_id: str
    role: str
    type: str | None
    onsets: dict[str, pd.Timestamp]
    primary_onset: str | None
    end: pd.Timestamp | None
    notes: str | None

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"unknown role {self.role!r}; valid roles: {', '.join(ROLES)}")
        for name in self.onsets:
            if name not in ONSET_NAMES:
                raise ValueError(
                    f"unknown onset {name!r}; valid onset names: {', '.join(ONSET_NAMES)}"
                )
        onsets = {
            name: _utc_timestamp(value, f"onset {name!r}") for name, value in self.onsets.items()
        }
        object.__setattr__(self, "onsets", onsets)
        if self.end is not None:
            object.__setattr__(self, "end", _utc_timestamp(self.end, "'end'"))
        if self.primary_onset is not None and self.primary_onset not in self.onsets:
            present = ", ".join(self.onsets) if self.onsets else "none"
            raise ValueError(
                f"primary_onset {self.primary_onset!r} names an absent onset; "
                f"present onsets: {present}"
            )
        if self.role == "incident":
            if "T0" not in self.onsets:
                raise ValueError(f"incident case {self.resource_id!r} requires onset T0")
            if self.end is None:
                raise ValueError(f"incident case {self.resource_id!r} requires 'end'")


@dataclass(frozen=True)
class LabelSet:
    """A versioned collection of label cases for one capture."""

    version: int
    capture: str | None
    cases: list[LabelCase]

    def incidents(self) -> list[LabelCase]:
        """Cases with role ``incident``."""
        return [case for case in self.cases if case.role == "incident"]

    def negative_controls(self) -> list[LabelCase]:
        """Cases with role ``negative_control``."""
        return [case for case in self.cases if case.role == "negative_control"]

    def role_for(self, resource_id: str) -> str:
        """The resource's labeled role; resources absent from cases are excluded."""
        for case in self.cases:
            if case.resource_id == resource_id:
                return case.role
        return "excluded"


def from_v1(entries: list[dict]) -> LabelSet:
    """Map v1 entries ({resource_id, type, start, end}) to a version-1 LabelSet.

    Every v1 entry is an incident: ``start`` becomes onset T0 and the primary
    onset, ``end`` stays ``end``. The set keeps ``version=1`` to record its
    source format; ``dump_labels`` writes v2 regardless.
    """
    cases = [
        LabelCase(
            resource_id=entry["resource_id"],
            role="incident",
            type=entry.get("type"),
            onsets={"T0": entry["start"]},
            primary_onset="T0",
            end=entry["end"],
            notes=None,
        )
        for entry in entries
    ]
    return LabelSet(version=1, capture=None, cases=cases)


def load_labels(path: str) -> LabelSet:
    """Load a labels file, dispatching on the top-level type (v1 list or v2 object)."""
    with open(path) as handle:
        raw = json.load(handle)
    if isinstance(raw, list):
        return from_v1(raw)
    if isinstance(raw, dict):
        version = raw.get("version")
        if version != 2:
            raise ValueError(
                f"unsupported labels version {version!r} in {path}; "
                "supported: a JSON list (v1) or an object with version 2"
            )
        cases = [
            LabelCase(
                resource_id=entry["resource_id"],
                role=entry["role"],
                type=entry.get("type"),
                onsets=dict(entry.get("onsets") or {}),
                primary_onset=entry.get("primary_onset"),
                end=entry.get("end"),
                notes=entry.get("notes"),
            )
            for entry in raw.get("cases", [])
        ]
        return LabelSet(version=2, capture=raw.get("capture"), cases=cases)
    raise ValueError(
        f"labels file {path} must be a JSON list (v1) or an object with version 2; "
        f"got {type(raw).__name__}"
    )


def _iso_z(ts: pd.Timestamp) -> str:
    """ISO-8601 UTC with a Z suffix, the v2 on-disk timestamp form."""
    return ts.isoformat().replace("+00:00", "Z")


def dump_labels(labels: LabelSet, path: str) -> None:
    """Write ``labels`` as v2 JSON regardless of the source format."""
    payload = {
        "version": 2,
        "capture": labels.capture,
        "cases": [
            {
                "resource_id": case.resource_id,
                "role": case.role,
                "type": case.type,
                "onsets": {name: _iso_z(value) for name, value in case.onsets.items()},
                "primary_onset": case.primary_onset,
                "end": _iso_z(case.end) if case.end is not None else None,
                "notes": case.notes,
            }
            for case in labels.cases
        ],
    }
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
