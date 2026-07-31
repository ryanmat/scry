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
conduit filled, and the rubric evaluated exactly once per case. Importing
this module never pulls torch; the candidate and model stacks are imported
lazily inside the orchestration functions.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import yaml

from scry.data.fetcher import fetch_full_capture
from scry.eval.detection import SUSTAIN_DEFAULT
from scry.eval.hygiene import per_resource_eligibility
from scry.eval.labels import LabelSet, load_labels
from scry.eval.metrics import compute_case_metrics
from scry.eval.rubric import SpecError, evaluate_rubric, load_rubric

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
        Dict with the suite name, the rubric version, and one entry per case
        carrying its name, kind, CaseMetrics, and RubricResult.

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
        results.append(
            {
                "name": case["name"],
                "kind": metrics.case_kind,
                "metrics": metrics,
                "rubric_result": rubric_result,
            }
        )

    return {"suite": suite["suite"], "rubric_version": int(rubric["version"]), "cases": results}
