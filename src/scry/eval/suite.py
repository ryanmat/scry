# Description: Suite loading and validation for the evaluation harness: cases, policy, rubric, paths.
# Description: Torch-free at import; spec errors are collected and raised together as one SpecError.

"""Suite schema loading for the evaluation harness.

A suite is an operator-authored YAML naming the candidate checkpoint, the
threshold policy, the rubric, and the cases. ``load_suite`` reads an explicit
path, resolves every path field relative to the suite file's directory, and
validates the schema: version and suite name present, the reconstruction
candidate with an existing model, a known threshold-policy type (calibration
required for the policies with a calibration population and loaded through
the same ``fetch_full_capture`` loader the cases use), a rubric that passes
``load_rubric``, and cases of kind incident_capture (labels required) or
healthy_reference (takes no labels). Spec errors are COLLECTED and raised
together as one SpecError naming every problem, the exit-2 signal.
``load_capture`` is the one loader for case captures and policy calibrations.

``run_suite`` orchestrates a run: grids built from the rubric's declarations
(labeled by grid name), every case scored on every grid, the threshold
policy resolved per resource (the calibration scored on the headline grid;
healthy_split fits per case), one CaseMetrics per case with the context
conduit filled, and the rubric evaluated exactly once per case. It returns
the full report -- provenance, serialized cases with the conduit under each
case's ``reported`` block, the PASS/FAIL verdict, and the exit code --
validated against the schema enumeration by ``validate_report``, with every
sustain-keyed field carrying both string-keyed accountings. Importing this
module never pulls torch; the candidate and model stacks are imported lazily
inside the orchestration functions.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import yaml

from scry.data.fetcher import fetch_full_capture
from scry.eval.detection import SUSTAIN_DEFAULT, DetectionResult
from scry.eval.hygiene import per_resource_eligibility
from scry.eval.labels import LabelSet, load_labels
from scry.eval.metrics import (
    SUSTAIN_ACCOUNTINGS,
    CaseMetrics,
    GridMetrics,
    ResourceMetrics,
    compute_case_metrics,
)
from scry.eval.provenance import build_provenance
from scry.eval.rubric import RubricResult, SpecError, evaluate_rubric, load_rubric

if TYPE_CHECKING:
    from scry.eval.candidate import ScoreSet, ThresholdPolicy
    from scry.eval.scoring import ScoringGrid
    from scry.model.checkpoint import Keeper

POLICY_TYPES: tuple[str, ...] = (
    "global_override",
    "reference_quantile",
    "healthy_split",
    "per_resource_margin",
    "serving_block",
)

# Policies whose threshold is fit on a separate calibration capture (the 8.3
# hygiene rule's "calibration population"); the others need none.
CALIBRATION_POLICY_TYPES: tuple[str, ...] = ("reference_quantile", "per_resource_margin")

CASE_KINDS: tuple[str, ...] = ("incident_capture", "healthy_reference")

_FORMATS: tuple[str, ...] = ("parquet", "csv")
_TOP_LEVEL_KEYS = frozenset(
    {"version", "suite", "candidate", "threshold_policy", "rubric", "cases"}
)
_CASE_KEYS = frozenset({"name", "kind", "capture", "labels", "format"})


def load_capture(path: str, profile: str, data_format: str | None = None) -> pd.DataFrame:
    """The one loader for case captures and policy calibrations."""
    return asyncio.run(fetch_full_capture(path, profile=profile, data_format=data_format))


def _resolve(base: Path, value: str) -> str:
    return str((base / value).resolve())


def _check_format(owner: str, block: dict[str, Any], problems: list[str]) -> None:
    declared = block.get("format")
    if declared is not None and declared not in _FORMATS:
        problems.append(f"{owner} format {declared!r} is not one of: {', '.join(_FORMATS)}")


def _resolve_existing(
    base: Path, owner: str, block: dict[str, Any], key: str, problems: list[str]
) -> None:
    resolved = _resolve(base, block[key])
    if not Path(resolved).exists():
        problems.append(f"{owner} {key} not found: {resolved}")
    else:
        block[key] = resolved


def load_suite(path: str) -> dict[str, Any]:
    """Load and validate a suite from an explicit YAML path.

    Every path field in the returned mapping is resolved relative to the
    suite file's directory.

    Raises:
        SpecError: One error naming every collected problem: unknown keys,
            missing files, a bad candidate or policy type, a missing
            calibration on a calibration policy, case rules (incident_capture
            requires labels, healthy_reference takes none), or a rubric that
            fails validation.
    """
    suite_path = Path(path).resolve()
    base = suite_path.parent
    with open(suite_path) as handle:
        suite = yaml.safe_load(handle)
    if not isinstance(suite, dict):
        raise SpecError(f"suite {path} must be a YAML mapping; got {type(suite).__name__}")

    problems: list[str] = []
    unknown = sorted(set(suite) - _TOP_LEVEL_KEYS)
    if unknown:
        problems.append(f"unknown key(s): {', '.join(unknown)}")
    if "version" not in suite:
        problems.append("missing 'version'")
    if not suite.get("suite"):
        problems.append("missing 'suite' name")

    candidate = suite.get("candidate")
    if not isinstance(candidate, dict):
        problems.append("missing 'candidate' block")
    else:
        if candidate.get("type") != "reconstruction":
            problems.append(
                f"unknown candidate type {candidate.get('type')!r}; valid types: reconstruction"
            )
        if not candidate.get("profile"):
            problems.append("candidate is missing 'profile'")
        if not candidate.get("model"):
            problems.append("candidate is missing 'model'")
        else:
            _resolve_existing(base, "candidate", candidate, "model", problems)

    policy = suite.get("threshold_policy")
    if not isinstance(policy, dict):
        problems.append("missing 'threshold_policy' block")
    else:
        policy_type = policy.get("type")
        if policy_type not in POLICY_TYPES:
            problems.append(
                f"unknown threshold_policy type {policy_type!r}; "
                f"valid types: {', '.join(POLICY_TYPES)}"
            )
        _check_format("threshold_policy", policy, problems)
        if policy.get("calibration") is not None:
            _resolve_existing(base, "threshold_policy", policy, "calibration", problems)
        elif policy_type in CALIBRATION_POLICY_TYPES:
            problems.append(f"threshold_policy type {policy_type!r} requires 'calibration'")

    rubric = suite.get("rubric")
    if not rubric:
        problems.append("missing 'rubric'")
    else:
        resolved = _resolve(base, rubric)
        if not Path(resolved).exists():
            problems.append(f"rubric not found: {resolved}")
        else:
            try:
                load_rubric(resolved)
                suite["rubric"] = resolved
            except SpecError as error:
                problems.append(f"rubric validation: {error}")

    cases = suite.get("cases")
    if not isinstance(cases, list) or not cases:
        problems.append("missing 'cases'")
    else:
        for case in cases:
            name = case.get("name") or "<unnamed>"
            unknown_case = sorted(set(case) - _CASE_KEYS)
            if unknown_case:
                problems.append(f"case {name!r} has unknown key(s): {', '.join(unknown_case)}")
            kind = case.get("kind")
            if kind not in CASE_KINDS:
                problems.append(
                    f"case {name!r} has unknown kind {kind!r}; valid kinds: {', '.join(CASE_KINDS)}"
                )
            _check_format(f"case {name!r}", case, problems)
            if not case.get("capture"):
                problems.append(f"case {name!r} is missing 'capture'")
            else:
                _resolve_existing(base, f"case {name!r}", case, "capture", problems)
            if kind == "incident_capture":
                if not case.get("labels"):
                    problems.append(f"incident_capture case {name!r} requires 'labels'")
                else:
                    _resolve_existing(base, f"case {name!r}", case, "labels", problems)
            elif kind == "healthy_reference" and case.get("labels"):
                problems.append(f"healthy_reference case {name!r} takes no labels")

    if problems:
        raise SpecError(f"suite {path} is unevaluable: " + "; ".join(problems))
    return suite


def _grids_from_rubric(rubric: dict[str, Any]) -> dict[str, ScoringGrid]:
    """ScoringGrid instances from the rubric's grids block, labeled by grid name."""
    from scry.eval.scoring import ScoringGrid  # imports the model stack; keep lazy

    grids: dict[str, ScoringGrid] = {}
    for name, declared in rubric["grids"].items():
        cadence = declared.get("cadence_minutes")
        grids[name] = ScoringGrid(
            label=name,
            step_samples=int(declared["step_samples"]),
            cadence=pd.Timedelta(minutes=cadence) if cadence is not None else None,
        )
    return grids


def _features_by_resource(df_long: pd.DataFrame) -> dict[str, set[str]]:
    return {
        str(rid): set(group["metric_name"].unique())
        for rid, group in df_long.groupby("resource_id")
    }


def _build_policy(
    policy_cfg: dict[str, Any],
    model_path: str,
    keeper: Keeper,
    calibration_df: pd.DataFrame | None,
    calibration_scores: ScoreSet | None,
) -> ThresholdPolicy | None:
    """The suite's threshold policy; None for healthy_split (built per case)."""
    from scry.eval.candidate import (
        GlobalOverride,
        PerResourceMargin,
        ReferenceQuantile,
        ServingBlock,
    )

    policy_type = policy_cfg["type"]
    quantile = float(policy_cfg.get("quantile", 0.99))
    if policy_type == "global_override":
        threshold = policy_cfg.get("threshold")
        if threshold is None:
            raise SpecError("threshold_policy type 'global_override' requires 'threshold'")
        return GlobalOverride(float(threshold))
    if policy_type == "serving_block":
        return ServingBlock(model_path)
    if policy_type == "reference_quantile":
        return ReferenceQuantile(calibration_scores, quantile)
    if policy_type == "per_resource_margin":
        # The global fallback for ineligible and unknown resources is the
        # pooled calibration quantile, the harness's own global convention.
        return PerResourceMargin(
            calibration_scores,
            margin=float(policy_cfg["margin"]),
            fallback=float(np.quantile(np.asarray(calibration_scores.errors), quantile)),
            trained_features=keeper.numerical_features,
            features_by_resource=_features_by_resource(calibration_df),
            quantile=quantile,
        )
    return None  # healthy_split fits on each case's own capture


def _hygiene_context(
    population: str,
    keeper: Keeper,
    features_by_resource: dict[str, set[str]],
    scores: ScoreSet,
    quantile: float,
) -> dict[str, Any]:
    verdicts = per_resource_eligibility(
        trained_features=keeper.numerical_features,
        features_by_resource=features_by_resource,
        resource_ids=np.asarray(scores.resource_ids),
        errors=np.asarray(scores.errors),
        quantile=quantile,
    )
    return {
        "hygiene_population": population,
        "eligibility": {rid: list(verdict.reasons) for rid, verdict in verdicts.items()},
    }


# The 9.3 report enumeration: serialized field sets, exact and closed.
_RESOURCE_FIELDS: tuple[str, ...] = (
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
_RESOURCE_SUSTAIN_FIELDS: tuple[str, ...] = (
    "lead_in_fpr",
    "time_in_alarm_fraction",
    "raises_per_week",
    "runs_per_week",
    "sustained_run_counts",
)
_GRID_FIELDS: tuple[str, ...] = (
    "grid",
    "per_resource",
    "pooled_lead_in_fpr",
    "fleet_time_in_alarm_fraction",
    "fleet_raises_per_week",
    "fleet_runs_per_week",
    "n_eval_windows",
    "vus_pr",
)
_GRID_SUSTAIN_FIELDS: tuple[str, ...] = (
    "pooled_lead_in_fpr",
    "fleet_time_in_alarm_fraction",
    "fleet_raises_per_week",
    "fleet_runs_per_week",
)
_CASE_ENTRY_FIELDS: tuple[str, ...] = ("name", "kind", "grids", "rubric", "reported")
_GATE_FIELDS: tuple[str, ...] = ("name", "required", "passed", "grid", "observed", "detail")


def _iso_or_none(ts: pd.Timestamp | None) -> str | None:
    return ts.isoformat().replace("+00:00", "Z") if ts is not None else None


def _sustain_object(values: dict[int, Any]) -> dict[str, Any]:
    """Sustain-keyed dict as a string-keyed object with both accountings always
    present; a role-neutral empty dict serializes as nulls under both keys."""
    return {str(sustain): values.get(sustain) for sustain in SUSTAIN_ACCOUNTINGS}


def _serialize_detection(detection: DetectionResult | None) -> dict[str, Any] | None:
    if detection is None:
        return None
    return {
        "detected": detection.detected,
        "detection_time": _iso_or_none(detection.detection_time),
        "lead_seconds": detection.lead_seconds,
        "bridged": detection.bridged,
        "n_runs_pre_onset": detection.n_runs_pre_onset,
        "n_runs_at_or_after": detection.n_runs_at_or_after,
    }


def _serialize_resource(resource: ResourceMetrics) -> dict[str, Any]:
    serialized = {
        "resource_id": resource.resource_id,
        "role": resource.role,
        "threshold": resource.threshold,
        "threshold_source": resource.threshold_source,
        "n_eval_windows": resource.n_eval_windows,
        "detection": _serialize_detection(resource.detection),
        "detection_time": _iso_or_none(resource.detection_time),
        "lead_seconds_by_onset": dict(resource.lead_seconds_by_onset),
        "n_lead_in_windows": resource.n_lead_in_windows,
        "coverage_fraction": resource.coverage_fraction,
        "clear_lead_vs_end_s": resource.clear_lead_vs_end_s,
        "alarm_seconds_in_incident": resource.alarm_seconds_in_incident,
        "observed_span_days": resource.observed_span_days,
        "slice_stats_by_threshold": dict(resource.slice_stats_by_threshold),
        "exceedances_by_threshold": dict(resource.exceedances_by_threshold),
    }
    for field_name in _RESOURCE_SUSTAIN_FIELDS:
        serialized[field_name] = _sustain_object(getattr(resource, field_name))
    return {name: serialized[name] for name in _RESOURCE_FIELDS}


def _serialize_grid(grid_metrics: GridMetrics) -> dict[str, Any]:
    grid = grid_metrics.grid
    serialized: dict[str, Any] = {
        "grid": {
            "label": grid.label,
            "step_samples": int(grid.step_samples),
            "cadence_seconds": (
                float(grid.cadence.total_seconds()) if grid.cadence is not None else None
            ),
        },
        "per_resource": {
            rid: _serialize_resource(resource)
            for rid, resource in grid_metrics.per_resource.items()
        },
        "n_eval_windows": grid_metrics.n_eval_windows,
        "vus_pr": dict(grid_metrics.vus_pr),
    }
    for field_name in _GRID_SUSTAIN_FIELDS:
        serialized[field_name] = _sustain_object(getattr(grid_metrics, field_name))
    return {name: serialized[name] for name in _GRID_FIELDS}


def _serialize_rubric_result(result: RubricResult) -> dict[str, Any]:
    return {
        "rubric_version": result.rubric_version,
        "gates": [
            {
                "name": gate.name,
                "required": gate.required,
                "passed": gate.passed,
                "grid": gate.grid,
                "observed": gate.observed,
                "detail": gate.detail,
            }
            for gate in result.gates
        ],
        "reported": dict(result.reported),
        "passed": result.passed,
    }


def _serialize_case(name: str, metrics: CaseMetrics, rubric_result: RubricResult) -> dict[str, Any]:
    return {
        "name": name,
        "kind": metrics.case_kind,
        "grids": {label: _serialize_grid(grid) for label, grid in metrics.grids.items()},
        "rubric": _serialize_rubric_result(rubric_result),
        "reported": dict(metrics.context),
    }


def validate_report(report: dict[str, Any]) -> None:
    """Check a report against the 9.3 schema enumeration.

    Raises:
        SpecError: On a missing or extra top-level, case, grid, per-resource,
            or gate field, or a sustain-keyed object missing either of the
            string keys "3" and "1".
    """
    problems: list[str] = []
    if set(report) != {"provenance", "suite", "cases", "verdict", "exit_code"}:
        problems.append(f"top-level keys are {sorted(report)}")
    for case in report.get("cases", []):
        name = case.get("name", "<unnamed>")
        if set(case) != set(_CASE_ENTRY_FIELDS):
            problems.append(f"case {name!r} keys are {sorted(case)}")
            continue
        for gate in case["rubric"]["gates"]:
            if set(gate) != set(_GATE_FIELDS):
                problems.append(f"case {name!r} gate keys are {sorted(gate)}")
        for label, grid in case["grids"].items():
            if set(grid) != set(_GRID_FIELDS):
                problems.append(f"case {name!r} grid {label!r} keys are {sorted(grid)}")
                continue
            for field_name in _GRID_SUSTAIN_FIELDS:
                if set(grid[field_name]) != {"3", "1"}:
                    problems.append(
                        f"case {name!r} grid {label!r} {field_name} is missing an accounting"
                    )
            for rid, resource in grid["per_resource"].items():
                if set(resource) != set(_RESOURCE_FIELDS):
                    problems.append(f"case {name!r} resource {rid!r} keys are {sorted(resource)}")
                    continue
                for field_name in _RESOURCE_SUSTAIN_FIELDS:
                    if set(resource[field_name]) != {"3", "1"}:
                        problems.append(
                            f"case {name!r} resource {rid!r} {field_name} is missing an accounting"
                        )
    if problems:
        raise SpecError("report violates the schema: " + "; ".join(problems))


def run_suite(suite: dict[str, Any] | str, case_names: list[str] | None = None) -> dict[str, Any]:
    """Run every case of a suite: score, resolve thresholds, bundle, evaluate.

    Each case is scored on every rubric-declared grid (grid labels are the
    rubric's grid names), the threshold policy resolves per resource, one
    CaseMetrics is produced per case with the context conduit filled (hygiene
    population per the 8.3 rule: the policy's calibration population where it
    has one, else the case's pre-onset slice for incident cases and the whole
    capture for healthy ones; keeper profile; labelled resources present),
    and the rubric is evaluated exactly once per case. The calibration is
    scored on the headline grid, the declared reading convention.

    Args:
        suite: A validated suite mapping from ``load_suite``, or a path.
        case_names: Optional case-name filter; unknown names are a SpecError.

    Returns:
        The full report: provenance, suite name, one serialized case entry per
        case ({name, kind, grids, rubric, reported} with the context conduit
        under ``reported``), the PASS/FAIL verdict, and the exit code (0 when
        every required gate of every case passed, else 1). Validated against
        the report schema before returning.

    Raises:
        SpecError: On unknown case names or a policy missing its parameters.
    """
    from scry.eval.candidate import HealthySplitQuantile, ReconstructionCandidate
    from scry.model.checkpoint import load_keeper

    if isinstance(suite, str):
        suite = load_suite(suite)
    rubric = load_rubric(suite["rubric"])
    detection_cfg = rubric.get("detection") or {}
    mode = detection_cfg.get("mode", "no_bridging")
    onset_anchor = detection_cfg.get("onset_anchor", "T0")
    sustain = int(detection_cfg.get("sustain", SUSTAIN_DEFAULT))
    lead_in = pd.Timedelta(hours=float(detection_cfg.get("lead_in_hours", 24)))
    max_leadtime = pd.Timedelta(minutes=float(detection_cfg.get("max_leadtime_minutes", 120)))
    grids = _grids_from_rubric(rubric)
    headline = rubric["headline_grid"]

    candidate_cfg = suite["candidate"]
    profile = candidate_cfg["profile"]
    model_path = candidate_cfg["model"]
    candidate = ReconstructionCandidate(model_path, profile=profile)
    keeper = load_keeper(model_path)
    quantile = float(suite["threshold_policy"].get("quantile", 0.99))

    policy_cfg = suite["threshold_policy"]
    calibration_df = None
    calibration_scores = None
    calibration_context = None
    if policy_cfg.get("calibration"):
        calibration_df = load_capture(policy_cfg["calibration"], profile, policy_cfg.get("format"))
        calibration_scores = candidate.score(calibration_df, grids[headline])
        calibration_context = _hygiene_context(
            "calibration",
            keeper,
            _features_by_resource(calibration_df),
            calibration_scores,
            quantile,
        )
    policy = _build_policy(policy_cfg, model_path, keeper, calibration_df, calibration_scores)

    cases = suite["cases"]
    if case_names is not None:
        known = {case["name"] for case in cases}
        unknown = sorted(set(case_names) - known)
        if unknown:
            raise SpecError(f"unknown case name(s): {', '.join(unknown)}")
        cases = [case for case in cases if case["name"] in case_names]

    results: list[dict[str, Any]] = []
    for case in cases:
        df_long = load_capture(case["capture"], profile, case.get("format"))
        if case["kind"] == "incident_capture":
            labels = load_labels(case["labels"])
        else:
            labels = LabelSet(version=2, capture=None, cases=[])
        scores_by_grid = {name: candidate.score(df_long, grid) for name, grid in grids.items()}

        case_policy = policy
        if case_policy is None:
            case_policy = HealthySplitQuantile(scores_by_grid[headline], quantile)

        scored_ids = sorted(
            {str(rid) for scores in scores_by_grid.values() for rid in scores.resource_ids}
        )
        thresholds = {rid: case_policy.resolve(rid) for rid in scored_ids}

        if calibration_context is not None:
            context = dict(calibration_context)
        else:
            headline_scores = scores_by_grid[headline]
            ends = headline_scores.end_times
            incidents = labels.incidents()
            if incidents:
                primary = min(incident.onsets[incident.primary_onset] for incident in incidents)
                mask = (ends >= primary - lead_in) & (ends < primary)
                population = "pre_onset"
            else:
                mask = np.ones(len(ends), dtype=bool)
                population = "capture"
            sliced = type(headline_scores)(
                errors=np.asarray(headline_scores.errors)[mask],
                end_times=ends[mask],
                resource_ids=np.asarray(headline_scores.resource_ids)[mask],
                grid=headline_scores.grid,
                meta=dict(headline_scores.meta),
            )
            context = _hygiene_context(
                population, keeper, _features_by_resource(df_long), sliced, quantile
            )
        context["profile"] = keeper.profile or profile
        context["labels_resources_present"] = {
            label_case.resource_id: label_case.resource_id in scored_ids
            for label_case in labels.cases
        }

        metrics = compute_case_metrics(
            case["name"],
            scores_by_grid,
            labels,
            thresholds,
            mode=mode,
            onset_anchor=onset_anchor,
            sustain=sustain,
            lead_in=lead_in,
            max_leadtime=max_leadtime,
            context=context,
        )
        rubric_result = evaluate_rubric(rubric, metrics)
        results.append(_serialize_case(case["name"], metrics, rubric_result))

    if policy is not None:
        policy_description = policy.describe()
    else:
        # healthy_split fits per case; the config-level identity is the
        # reproducible statement.
        policy_description = {"type": "healthy_split", "quantile": quantile}
    provenance = build_provenance(
        model_path=model_path,
        profile=keeper.profile or profile,
        seq_len=int(keeper.config["seq_len"]),
        grids=grids,
        sustains=SUSTAIN_ACCOUNTINGS,
        detection_mode=mode,
        policy_description=policy_description,
        rubric_path=suite["rubric"],
        rubric_version=int(rubric["version"]),
        case_data_paths={case["name"]: case["capture"] for case in cases},
    )
    passed = all(case["rubric"]["passed"] for case in results)
    report = {
        "provenance": provenance,
        "suite": suite["suite"],
        "cases": results,
        "verdict": "PASS" if passed else "FAIL",
        "exit_code": 0 if passed else 1,
    }
    validate_report(report)
    return report
