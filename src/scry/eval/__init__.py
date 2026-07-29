# Description: Evaluation harness package: detection primitives, labels, scoring, and metrics.
# Description: Torch-free members import eagerly; torch-dependent members import lazily.

"""Evaluation harness for Scry.

Torch-free members (the detection primitives) import eagerly. Torch-dependent
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
_LAZY_EXPORTS: dict[str, str] = {}


def __getattr__(name: str):
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_path), name)


def __dir__() -> list[str]:
    return sorted([*globals().keys(), *_LAZY_EXPORTS])


__all__ = [
    "ONSET_NAMES",
    "ROLES",
    "SUSTAIN_DEFAULT",
    "DetectionMode",
    "DetectionResult",
    "LabelCase",
    "LabelSet",
    "anomaly_runs",
    "dump_labels",
    "from_v1",
    "load_labels",
    "select_detection",
    "slice_stats",
]
