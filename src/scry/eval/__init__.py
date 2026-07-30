# Description: Evaluation harness package: detection primitives, labels, scoring, and metrics.
# Description: Torch-free members import eagerly; torch-dependent members import lazily.

"""Evaluation harness for Scry.

Torch-free members (detection, labels, hygiene) import eagerly. Torch-dependent
members import lazily via PEP 562 ``__getattr__`` so torch-free consumers can
use this package without pulling torch. Every name in ``__all__`` still
resolves.
"""

from scry.eval.detection import (
    SUSTAIN_DEFAULT,
    DetectionMode,
    DetectionResult,
    anomaly_runs,
    select_detection,
    slice_stats,
)
from scry.eval.hygiene import (
    MIN_PER_RESOURCE_WINDOWS,
    REASON_DIVERGENT,
    REASON_NONPOSITIVE_QUANTILE,
    REASON_TOO_FEW_WINDOWS,
    ResourceEligibility,
    per_resource_eligibility,
)
from scry.eval.labels import (
    ONSET_NAMES,
    ROLES,
    LabelCase,
    LabelSet,
    dump_labels,
    from_v1,
    load_labels,
)

# name -> source module, resolved on first access by __getattr__ below.
_LAZY_EXPORTS: dict[str, str] = {
    "SERVING_GRID": "scry.eval.scoring",
    "Candidate": "scry.eval.candidate",
    "GlobalOverride": "scry.eval.candidate",
    "HealthySplitQuantile": "scry.eval.candidate",
    "PerResourceMargin": "scry.eval.candidate",
    "ReconstructionCandidate": "scry.eval.candidate",
    "ReferenceQuantile": "scry.eval.candidate",
    "ScoreSet": "scry.eval.candidate",
    "ScoringGrid": "scry.eval.scoring",
    "ServingBlock": "scry.eval.candidate",
    "ThresholdPolicy": "scry.eval.candidate",
    "per_resource_time_split": "scry.eval.scoring",
    "warn_missing_model_features": "scry.eval.scoring",
    "windows_for_keeper": "scry.eval.scoring",
}


def __getattr__(name: str):
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_path), name)


def __dir__() -> list[str]:
    return sorted([*globals().keys(), *_LAZY_EXPORTS])


__all__ = [
    "MIN_PER_RESOURCE_WINDOWS",
    "ONSET_NAMES",
    "REASON_DIVERGENT",
    "REASON_NONPOSITIVE_QUANTILE",
    "REASON_TOO_FEW_WINDOWS",
    "ROLES",
    "SERVING_GRID",
    "SUSTAIN_DEFAULT",
    "Candidate",
    "DetectionMode",
    "DetectionResult",
    "GlobalOverride",
    "HealthySplitQuantile",
    "LabelCase",
    "LabelSet",
    "PerResourceMargin",
    "ReconstructionCandidate",
    "ReferenceQuantile",
    "ResourceEligibility",
    "ScoreSet",
    "ScoringGrid",
    "ServingBlock",
    "ThresholdPolicy",
    "anomaly_runs",
    "dump_labels",
    "from_v1",
    "load_labels",
    "per_resource_eligibility",
    "per_resource_time_split",
    "select_detection",
    "slice_stats",
    "warn_missing_model_features",
    "windows_for_keeper",
]
