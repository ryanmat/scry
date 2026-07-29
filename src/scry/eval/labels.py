# Description: Versioned incident labels: named onsets and per-resource roles for the eval harness.
# Description: Torch-free; validation runs on construction and enumerates the valid alternatives.

"""Incident label model with named onsets and per-resource roles.

A ``LabelCase`` assigns one resource a role (incident, negative_control, or
excluded) and, for incidents, named onsets drawn from ``ONSET_NAMES`` plus a
required ``end``. Validation runs in ``__post_init__`` so every construction
path enforces the same rules; error messages enumerate the valid alternatives.
Timestamps normalize to UTC; naive timestamps are rejected. Everything here is
importable without torch.
"""

from __future__ import annotations

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
