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
Importing this module never pulls torch; the candidate stack is imported
lazily by the orchestration layer.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from scry.data.fetcher import fetch_full_capture
from scry.eval.rubric import SpecError, load_rubric

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
