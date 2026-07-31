# Description: Rubric loading, gate-result types, and the SpecError for the evaluation harness.
# Description: Torch-free; a rubric is data, evaluated against the CaseMetrics bundles metrics.py produces.

"""Versioned pass criteria for the evaluation harness.

A rubric is a YAML document of gates and their parameters; ``load_rubric``
reads an explicit path (no environment search) and validates the schema with
enumerated error messages: known gate names only, version present, grids
well-formed, ``headline_grid`` and every gate's optional ``grid:`` naming a
declared grid. The ``reported`` and ``known_divergences`` blocks are
free-form. Validation failures raise ``SpecError``, the typed ValueError
subclass shared by the suite and report layers for unevaluable-as-written
input (their exit-2 path). ``GateResult`` and ``RubricResult`` carry gate
verdicts with the grid each evaluated on; ``evaluate_rubric`` reads a
CaseMetrics bundle and never computes a metric. Everything here is
importable without torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import yaml

from scry.eval.metrics import CaseMetrics

# The seven gates a rubric may declare, in the spec's order.
GATE_NAMES: tuple[str, ...] = (
    "no_pre_onset_bridging",
    "detection_lead",
    "lead_in_fpr",
    "alarm_fatigue",
    "negative_controls_clean",
    "coverage_integrity",
    "sanity",
)


class SpecError(ValueError):
    """The rubric or suite is unevaluable as written; maps to exit code 2."""


@dataclass(frozen=True)
class GateResult:
    """One gate's verdict: the grid it evaluated on and the numbers behind it."""

    name: str
    required: bool
    passed: bool
    grid: str
    observed: dict[str, Any]
    detail: str


@dataclass(frozen=True)
class RubricResult:
    """One case's gate verdicts and the rolled-up pass (all required gates)."""

    rubric_version: int
    gates: list[GateResult]
    reported: dict[str, Any]
    passed: bool


def load_rubric(path: str) -> dict[str, Any]:
    """Load and validate a rubric from an explicit YAML path.

    Args:
        path: The rubric file; no environment or directory search.

    Returns:
        The rubric mapping, unchanged beyond validation.

    Raises:
        SpecError: On a non-mapping document, a missing version, missing or
            malformed grids, an unknown gate name, or a ``headline_grid`` or
            per-gate ``grid:`` that does not name a declared grid.
    """
    with open(path) as handle:
        rubric = yaml.safe_load(handle)
    if not isinstance(rubric, dict):
        raise SpecError(f"rubric {path} must be a YAML mapping; got {type(rubric).__name__}")
    if "version" not in rubric:
        raise SpecError(f"rubric {path} is missing 'version'")
    grids = rubric.get("grids")
    if not isinstance(grids, dict) or not grids:
        raise SpecError(f"rubric {path} must declare at least one grid under 'grids'")
    for grid_name, declared in grids.items():
        if not isinstance(declared, dict) or "step_samples" not in declared:
            raise SpecError(f"rubric grid {grid_name!r} must be a mapping with 'step_samples'")
    headline = rubric.get("headline_grid")
    if headline not in grids:
        raise SpecError(
            f"headline_grid {headline!r} does not name a declared grid; "
            f"declared grids: {', '.join(grids)}"
        )
    for gate_name, gate in (rubric.get("gates") or {}).items():
        if gate_name not in GATE_NAMES:
            raise SpecError(f"unknown gate {gate_name!r}; valid gates: {', '.join(GATE_NAMES)}")
        gate_grid = (gate or {}).get("grid")
        if gate_grid is not None and gate_grid not in grids:
            raise SpecError(
                f"gate {gate_name!r} grid {gate_grid!r} does not name a declared grid; "
                f"declared grids: {', '.join(grids)}"
            )
    return rubric


def evaluate_rubric(rubric: dict[str, Any], case_metrics: CaseMetrics) -> RubricResult:
    """Evaluate the rubric's gates against one case's metric bundle."""
    raise NotImplementedError("gate evaluation is not implemented")
