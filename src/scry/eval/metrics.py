# Description: Per-case metric bundles for the evaluation harness: resource, grid, and case schemas.
# Description: Torch-free; the sole producer of the bundles rubric.py evaluates.

"""Metric bundle schema for the evaluation harness.

``ResourceMetrics`` carries one resource's numbers for one grid, grouped by
role: the incident group (detection and per-onset leads, lead-in FPR,
incident coverage), the healthy-reference group (duration-weighted
alarm-fatigue rates), and the negative-control group (slice stats at the
control's own and each incident resource's threshold). ``GridMetrics`` adds
the pooled and fleet aggregates for one grid; ``CaseMetrics`` keys
GridMetrics by grid label and carries the ``context`` conduit the
coverage_integrity and sanity gates read by documented key (unknown keys are
ignored, so producers may add keys without breaking an older rubric).

Validation runs in ``__post_init__``: every populated sustain-keyed dict
carries both the sustain-3 and the sustain-1 accounting (the two accountings
are fields, never variants), the incident and negative-control field groups
populate only under their role, and every grid key equals its grid's label.
The healthy-reference group is case-kind-driven, not role-gated: a
healthy-reference capture takes no labels, so its resources carry whatever
labels role applies. Everything here is importable without torch.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd

from scry.eval.detection import DetectionResult
from scry.eval.labels import ROLES

if TYPE_CHECKING:
    from scry.eval.scoring import ScoringGrid

CASE_KINDS: tuple[str, ...] = ("incident", "healthy_reference", "negative_control")

# The two accountings every populated sustain-keyed dict must carry: the
# rubric's sustain-3 and the deployed sustain-1.
SUSTAIN_ACCOUNTINGS: tuple[int, ...] = (3, 1)


def _require_both_accountings(owner: str, name: str, values: Mapping[int, Any]) -> None:
    """Reject a populated sustain-keyed dict missing either accounting."""
    if not values:
        return
    missing = [sustain for sustain in SUSTAIN_ACCOUNTINGS if sustain not in values]
    if missing:
        raise ValueError(
            f"{owner} {name} carries sustains {sorted(values)} but every populated "
            "sustain-keyed dict must carry both sustain accountings 3 and 1"
        )


@dataclass(frozen=True)
class ResourceMetrics:
    """One resource's metrics for one grid; ``role`` gates the field groups."""

    resource_id: str
    role: str
    threshold: float
    threshold_source: str
    n_eval_windows: int
    # incident role
    detection: DetectionResult | None
    detection_time: pd.Timestamp | None
    lead_seconds_by_onset: dict[str, float | None]
    lead_in_fpr: dict[int, float | None]
    n_lead_in_windows: int
    coverage_fraction: float | None
    clear_lead_vs_end_s: float | None
    alarm_seconds_in_incident: float | None
    # healthy-reference group (case-kind-driven; allowed under any role)
    time_in_alarm_fraction: dict[int, float | None]
    raises_per_week: dict[int, float | None]
    runs_per_week: dict[int, float | None]
    sustained_run_counts: dict[int, int]
    observed_span_days: float | None
    # negative-control role
    slice_stats_by_threshold: dict[str, dict[str, Any]]
    exceedances_by_threshold: dict[str, int]

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"unknown role {self.role!r}; valid roles: {', '.join(ROLES)}")
        for name in (
            "lead_in_fpr",
            "time_in_alarm_fraction",
            "raises_per_week",
            "runs_per_week",
            "sustained_run_counts",
        ):
            _require_both_accountings(f"resource {self.resource_id!r}", name, getattr(self, name))
        if self.role != "incident":
            incident_values = {
                "detection": self.detection,
                "detection_time": self.detection_time,
                "lead_seconds_by_onset": self.lead_seconds_by_onset or None,
                "lead_in_fpr": self.lead_in_fpr or None,
                "n_lead_in_windows": self.n_lead_in_windows or None,
                "coverage_fraction": self.coverage_fraction,
                "clear_lead_vs_end_s": self.clear_lead_vs_end_s,
                "alarm_seconds_in_incident": self.alarm_seconds_in_incident,
            }
            populated = [name for name, value in incident_values.items() if value is not None]
            if populated:
                raise ValueError(
                    f"resource {self.resource_id!r} with role {self.role!r} populates "
                    f"incident field(s) {', '.join(populated)}; role is authoritative"
                )
        if self.role != "negative_control":
            populated = [
                name
                for name in ("slice_stats_by_threshold", "exceedances_by_threshold")
                if getattr(self, name)
            ]
            if populated:
                raise ValueError(
                    f"resource {self.resource_id!r} with role {self.role!r} populates "
                    f"negative-control field(s) {', '.join(populated)}; role is authoritative"
                )


@dataclass(frozen=True)
class GridMetrics:
    """One grid's per-resource metrics plus the pooled and fleet aggregates."""

    grid: ScoringGrid
    per_resource: dict[str, ResourceMetrics]
    pooled_lead_in_fpr: dict[int, float | None]
    fleet_time_in_alarm_fraction: dict[int, float | None]
    fleet_raises_per_week: dict[int, float | None]
    fleet_runs_per_week: dict[int, float | None]
    n_eval_windows: int
    vus_pr: dict[str, Any]

    def __post_init__(self) -> None:
        for name in (
            "pooled_lead_in_fpr",
            "fleet_time_in_alarm_fraction",
            "fleet_raises_per_week",
            "fleet_runs_per_week",
        ):
            _require_both_accountings(f"grid {self.grid.label!r}", name, getattr(self, name))


@dataclass(frozen=True)
class CaseMetrics:
    """One case's grid-keyed metric bundles; the only input rubric.py accepts.

    ``context`` is the flat conduit for what the coverage_integrity and sanity
    gates need and no other field carries (hygiene population name,
    per-resource eligibility verdicts, keeper profile, labelled resources
    present in the capture). Nothing computed in the metric definitions goes
    in it, and it serializes under the case's ``reported`` block, never under
    ``per_resource``.
    """

    case_id: str
    case_kind: str
    grids: dict[str, GridMetrics]
    context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.case_kind not in CASE_KINDS:
            raise ValueError(
                f"unknown case_kind {self.case_kind!r}; valid kinds: {', '.join(CASE_KINDS)}"
            )
        for label, grid_metrics in self.grids.items():
            if not label:
                raise ValueError(
                    f"case {self.case_id!r} keys a grid with an empty label; "
                    "grids are keyed by grid label"
                )
            if grid_metrics.grid.label != label:
                raise ValueError(
                    f"case {self.case_id!r} grid key {label!r} does not match the grid's "
                    f"label {grid_metrics.grid.label!r}"
                )
