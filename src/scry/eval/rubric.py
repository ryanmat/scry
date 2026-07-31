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
verdicts with the grid each evaluated on. ``evaluate_rubric`` dispatches the
seven gates against a CaseMetrics bundle, binding each to the headline grid
unless the gate declares its own; it never computes a metric, only reads the
bundle and the context conduit. Everything here is importable without
torch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import yaml

from scry.eval.detection import SUSTAIN_DEFAULT
from scry.eval.metrics import CaseMetrics, GridMetrics

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


def _require_params(gate_name: str, config: dict[str, Any], names: tuple[str, ...]) -> None:
    missing = [name for name in names if config.get(name) is None]
    if missing:
        raise SpecError(f"gate {gate_name!r} requires parameter(s): {', '.join(missing)}")


def _role_resources(grid: GridMetrics, role: str) -> dict[str, Any]:
    return {rid: r for rid, r in grid.per_resource.items() if r.role == role}


def _gate_no_pre_onset_bridging(
    config: dict[str, Any], grid: GridMetrics, case: CaseMetrics, sustain: int, rubric: dict
) -> tuple[bool, dict[str, Any], str] | None:
    incidents = {
        rid: r for rid, r in _role_resources(grid, "incident").items() if r.detection is not None
    }
    if not incidents:
        return None
    bridged = {rid: r.detection.bridged for rid, r in incidents.items()}
    offenders = sorted(rid for rid, is_bridged in bridged.items() if is_bridged)
    if offenders:
        detail = f"pre-onset run bridges the onset on: {', '.join(offenders)}"
    else:
        detail = "no pre-onset run bridges an onset"
    return not offenders, {"bridged": bridged}, detail


def _gate_detection_lead(
    config: dict[str, Any], grid: GridMetrics, case: CaseMetrics, sustain: int, rubric: dict
) -> tuple[bool, dict[str, Any], str] | None:
    incidents = {
        rid: r for rid, r in _role_resources(grid, "incident").items() if r.detection is not None
    }
    if not incidents:
        return None
    anchor = (rubric.get("detection") or {}).get("onset_anchor", "T0")
    vs = config.get("min_lead_vs", anchor)
    floor = float(config.get("min_lead_seconds", 0.0))
    leads = {rid: r.lead_seconds_by_onset.get(vs) for rid, r in incidents.items()}
    offenders = sorted(rid for rid, lead in leads.items() if lead is None or lead < floor)
    observed = {"min_lead_vs": vs, "min_lead_seconds": floor, "lead_seconds": leads}
    if offenders:
        detail = f"lead vs {vs} under {floor} s (or undetected) on: {', '.join(offenders)}"
    else:
        detail = f"every incident resource leads {vs} by at least {floor} s"
    return not offenders, observed, detail


def _gate_lead_in_fpr(
    config: dict[str, Any], grid: GridMetrics, case: CaseMetrics, sustain: int, rubric: dict
) -> tuple[bool, dict[str, Any], str] | None:
    incidents = _role_resources(grid, "incident")
    if not incidents:
        return None
    _require_params("lead_in_fpr", config, ("max_fraction", "min_eval_windows"))
    max_fraction = float(config["max_fraction"])
    min_windows = int(config["min_eval_windows"])
    fractions = {rid: r.lead_in_fpr.get(sustain) for rid, r in incidents.items()}
    n_windows = {rid: r.n_lead_in_windows for rid, r in incidents.items()}
    underpowered = sorted(rid for rid, n in n_windows.items() if n < min_windows)
    over = sorted(
        rid for rid, fraction in fractions.items() if fraction is None or fraction > max_fraction
    )
    observed = {
        "sustain": sustain,
        "max_fraction": max_fraction,
        "min_eval_windows": min_windows,
        "lead_in_fpr": fractions,
        "n_lead_in_windows": n_windows,
    }
    if underpowered:
        # An underpowered lead-in FAILS the gate; it never passes vacuously.
        detail = f"underpowered lead-in (< {min_windows} windows) on: {', '.join(underpowered)}"
        return False, observed, detail
    if over:
        return False, observed, f"lead-in FPR over {max_fraction} on: {', '.join(over)}"
    return True, observed, f"lead-in FPR within {max_fraction} on every incident resource"


def _gate_alarm_fatigue(
    config: dict[str, Any], grid: GridMetrics, case: CaseMetrics, sustain: int, rubric: dict
) -> tuple[bool, dict[str, Any], str] | None:
    population = {rid: r for rid, r in grid.per_resource.items() if r.time_in_alarm_fraction}
    if not population:
        return None
    _require_params(
        "alarm_fatigue",
        config,
        ("max_time_in_alarm_fraction_per_resource", "max_fleet_raises_per_week"),
    )
    gate_sustain = int(config.get("sustain", sustain))
    max_fraction = float(config["max_time_in_alarm_fraction_per_resource"])
    max_raises = float(config["max_fleet_raises_per_week"])
    fractions = {rid: r.time_in_alarm_fraction.get(gate_sustain) for rid, r in population.items()}
    over = sorted(
        rid
        for rid, fraction in fractions.items()
        if fraction is not None and fraction > max_fraction
    )
    fleet_raises = grid.fleet_raises_per_week.get(gate_sustain)
    observed = {
        "sustain": gate_sustain,
        "max_time_in_alarm_fraction_per_resource": max_fraction,
        "max_fleet_raises_per_week": max_raises,
        "time_in_alarm_fraction": fractions,
        "fleet_raises_per_week": fleet_raises,
    }
    if over:
        return False, observed, f"time-in-alarm fraction over {max_fraction} on: {', '.join(over)}"
    if fleet_raises is None:
        return False, observed, "fleet raises_per_week is unevaluable (no observed span)"
    if fleet_raises > max_raises:
        return False, observed, f"fleet raises {fleet_raises} per week over {max_raises}"
    return True, observed, "alarm-fatigue budget holds per resource and fleet-wide"


def _gate_negative_controls_clean(
    config: dict[str, Any], grid: GridMetrics, case: CaseMetrics, sustain: int, rubric: dict
) -> tuple[bool, dict[str, Any], str] | None:
    controls = _role_resources(grid, "negative_control")
    if not controls:
        return None
    runs_by_control: dict[str, int] = {}
    for rid, resource in controls.items():
        own = resource.slice_stats_by_threshold.get("own")
        if own is not None:
            over_entries = [value for key, value in own.items() if key.startswith("over_")]
            runs_by_control[rid] = sum(entry["sustained_runs"] for entry in over_entries)
        else:
            # A control-only capture has no incident slice; the full-series
            # run count at the rubric's sustain is the own-threshold record.
            runs_by_control[rid] = resource.sustained_run_counts.get(sustain, 0)
    offenders = sorted(rid for rid, runs in runs_by_control.items() if runs != 0)
    observed = {"sustained_runs": runs_by_control}
    if offenders:
        detail = f"sustained runs at own thresholds on: {', '.join(offenders)}"
    else:
        detail = "zero sustained runs on every control at its own threshold"
    return not offenders, observed, detail


def _gate_coverage_integrity(
    config: dict[str, Any], grid: GridMetrics, case: CaseMetrics, sustain: int, rubric: dict
) -> tuple[bool, dict[str, Any], str] | None:
    if not grid.per_resource:
        return None
    population = case.context.get("hygiene_population", "unstated")
    eligibility = case.context.get("eligibility") or {}
    spans_lead_in = bool(case.context.get("capture_spans_lead_in", True))
    unverdicted = sorted(rid for rid in grid.per_resource if rid not in eligibility)
    dirty = {rid: reasons for rid, reasons in eligibility.items() if reasons}
    observed = {
        "hygiene_population": population,
        "eligibility": {rid: list(reasons) for rid, reasons in eligibility.items()},
        "capture_spans_lead_in": spans_lead_in,
    }
    problems: list[str] = []
    if unverdicted:
        problems.append(f"no hygiene verdict for: {', '.join(unverdicted)}")
    for rid, reasons in sorted(dirty.items()):
        problems.append(f"{rid} fails hygiene: {', '.join(reasons)}")
    if not spans_lead_in:
        problems.append("capture does not span [primary_onset - lead_in, end]")
    if problems:
        return False, observed, "; ".join(problems)
    return True, observed, f"every resource passes hygiene over the {population} population"


def _finite_metric_values(resource: Any) -> list[float]:
    values: list[float] = []
    for sustain_dict in (
        resource.lead_in_fpr,
        resource.time_in_alarm_fraction,
        resource.raises_per_week,
        resource.runs_per_week,
    ):
        values.extend(value for value in sustain_dict.values() if value is not None)
    for scalar in (
        resource.coverage_fraction,
        resource.clear_lead_vs_end_s,
        resource.alarm_seconds_in_incident,
        resource.observed_span_days,
    ):
        if scalar is not None:
            values.append(scalar)
    return values


def _gate_sanity(
    config: dict[str, Any], grid: GridMetrics, case: CaseMetrics, sustain: int, rubric: dict
) -> tuple[bool, dict[str, Any], str] | None:
    problems: list[str] = []
    for rid, resource in grid.per_resource.items():
        if not (math.isfinite(resource.threshold) and resource.threshold > 0):
            problems.append(
                f"threshold {resource.threshold!r} for {rid} is not finite and positive"
            )
        if any(not math.isfinite(value) for value in _finite_metric_values(resource)):
            problems.append(f"non-finite metric value for {rid}")
    if grid.n_eval_windows <= 0:
        problems.append("n_eval_windows is 0")
    profile = case.context.get("profile")
    rubric_profile = rubric.get("profile")
    if profile != rubric_profile:
        problems.append(
            f"keeper profile {profile!r} does not match rubric profile {rubric_profile!r}"
        )
    missing = sorted(
        rid
        for rid, present in (case.context.get("labels_resources_present") or {}).items()
        if not present
    )
    if missing:
        problems.append(f"labelled resource(s) absent from the capture: {', '.join(missing)}")
    observed = {
        "n_eval_windows": grid.n_eval_windows,
        "profile": profile,
        "problems": list(problems),
    }
    if problems:
        return False, observed, "; ".join(problems)
    return True, observed, "all sanity checks hold"


_GATE_EVALUATORS = {
    "no_pre_onset_bridging": _gate_no_pre_onset_bridging,
    "detection_lead": _gate_detection_lead,
    "lead_in_fpr": _gate_lead_in_fpr,
    "alarm_fatigue": _gate_alarm_fatigue,
    "negative_controls_clean": _gate_negative_controls_clean,
    "coverage_integrity": _gate_coverage_integrity,
    "sanity": _gate_sanity,
}


def evaluate_rubric(rubric: dict[str, Any], case_metrics: CaseMetrics) -> RubricResult:
    """Evaluate the rubric's gates against one case's metric bundle.

    Every gate evaluates on the rubric's ``headline_grid`` unless it declares
    its own ``grid:``; each GateResult records the grid label it used. A gate
    whose target population is empty in this case (no incident resource for
    the detection gates, no control for negative_controls_clean, no healthy
    metrics for alarm_fatigue) raises SpecError when required, unless the
    gate declares ``allow_absent: true``, in which case it passes vacuously
    with detail "no applicable cases". The rolled-up ``passed`` is the
    conjunction of the required gates. No metric is computed here; every
    number comes from the bundle.

    Raises:
        SpecError: On an unknown gate name, a bound grid the case does not
            carry, a missing required gate parameter, or a required gate with
            no applicable cases and no ``allow_absent``.
    """
    detection_cfg = rubric.get("detection") or {}
    default_sustain = int(detection_cfg.get("sustain", SUSTAIN_DEFAULT))
    headline = rubric["headline_grid"]

    gate_results: list[GateResult] = []
    for name, config in (rubric.get("gates") or {}).items():
        config = config or {}
        if name not in _GATE_EVALUATORS:
            raise SpecError(f"unknown gate {name!r}; valid gates: {', '.join(GATE_NAMES)}")
        bound = config.get("grid", headline)
        if bound not in case_metrics.grids:
            carried = ", ".join(case_metrics.grids) or "none"
            raise SpecError(
                f"gate {name!r} binds to grid {bound!r} but case "
                f"{case_metrics.case_id!r} carries: {carried}"
            )
        required = bool(config.get("required", False))
        outcome = _GATE_EVALUATORS[name](
            config, case_metrics.grids[bound], case_metrics, default_sustain, rubric
        )
        if outcome is None:
            if required and not config.get("allow_absent"):
                raise SpecError(
                    f"required gate {name!r} has no applicable cases in case "
                    f"{case_metrics.case_id!r}"
                )
            gate_results.append(
                GateResult(
                    name=name,
                    required=required,
                    passed=True,
                    grid=bound,
                    observed={},
                    detail="no applicable cases",
                )
            )
            continue
        passed, observed, detail = outcome
        gate_results.append(
            GateResult(
                name=name,
                required=required,
                passed=passed,
                grid=bound,
                observed=observed,
                detail=detail,
            )
        )

    return RubricResult(
        rubric_version=int(rubric["version"]),
        gates=gate_results,
        reported=dict(rubric.get("reported") or {}),
        passed=all(gate.passed for gate in gate_results if gate.required),
    )
