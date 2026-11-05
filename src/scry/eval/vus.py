# Description: VUS-PR entry point over TheDatumOrg/VUS, consumed as an optional dependency.
# Description: Torch-free; a missing or unimportable backend degrades to null with a reason.

"""VUS-PR: volume under the precision-recall surface (spec section 12).

VUS-PR scores a detector across a range of temporal tolerances instead of at
one best threshold, which keeps scry's numbers comparable to the time-series
anomaly-detection literature without adopting single-threshold scoring. The
implementation it is measured against is TheDatumOrg/VUS, consumed as an
OPTIONAL DEPENDENCY rather than vendored into this tree: upstream declares
Apache-2.0 in its packaging metadata but publishes no LICENSE and no NOTICE
file, while Apache-2.0 sections 4(a) and 4(d) oblige a redistributor to carry
both, so vendoring the scoring code would mean supplying license text upstream
never wrote. docs/evaluation.md records the verdict and its evidence.

An optional dependency is one most hosts will not have, so the backend is
imported where it is used and never at module scope, and ``score_vus_pr``
answers with the same value-and-reason object either way: the metric when the
backend is there, null with the reason stated when it is not. Nothing here
raises for a missing backend. VUS-PR is reported and never gates (section
8.1), so a suite run on a host without the backend must still produce a report
rather than die on an import. Everything here is importable without torch.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from types import ModuleType
from typing import Any

import numpy as np

REASON_BACKEND_UNAVAILABLE = (
    "VUS-PR unavailable: the optional 'vus' backend (TheDatumOrg/VUS) is not installed"
)
REASON_BACKEND_UNIMPORTABLE = (
    "VUS-PR unavailable: the optional 'vus' backend is installed but importing "
    "vus.metrics failed: "
)
REASON_SCORING_FAILED = "VUS-PR unavailable: the scoring call raised "
"""The reason for a backend that is present and fails anyway, with the error.

Produced by ``scry.eval.metrics``, not here: ``score_vus_pr`` lets everything
but a failed import through, and the wiring catches it where the field is
filled, because that is where "reported, never gating" (section 8.1) is
enforced. The wording is here with its siblings so one module owns the
vocabulary.
"""
REASON_NO_POSITIVE_WINDOWS = "VUS-PR undefined: the scored series has no positive window"
"""The reason for a series the metric is not defined on.

A precision-recall surface needs at least one positive; upstream raises on an
all-negative label array rather than answering. ``score_vus_pr`` answers this
after the import probe and before the backend is called, so with the backend
present a case with no incident labels reads as undefined rather than as a
failed call, and with it absent the backend's own reason wins.
"""

MAX_TOLERANCE_WINDOWS = 4
"""The top of the tolerance range VUS-PR is averaged over: w = 0, 1, ... 4.

VUS-PR is the mean of the average precisions at each tolerance in that range.
A tolerance of w extends each labelled region by w // 2 windows on each side,
with partial credit sqrt(1 - d / w) at distance d (upstream 0.0.6's
``extend_postive_range``), so w = 0 and w = 1 extend nothing and agree, and
the range is part of what the number means rather than a tuning knob: on the
fixture tests/test_eval_vus.py pins, w = 0..4 gives 0.908040 and w = 0..1 gives
0.833333. Upstream sweeps 0 to its ``slidingWindow`` argument inclusive.
"""


def _load_backend() -> tuple[ModuleType | None, str | None]:
    """The backend's ``vus.metrics``, or None and the reason it is unavailable.

    Deferred rather than imported at module scope so that importing this
    module, and the eval package that exports it, never depends on the
    optional package being present.

    The probe is two-stage because upstream's ``vus/__init__.py`` is empty:
    ``import vus`` succeeding proves only that the package is on the path, and
    a host that installed it without the dependencies its scoring module needs
    (statsmodels and arch among them) still cannot score. Those two states get
    their own reasons, so an operator who has the package installed is not sent
    to install it again. Both stages go through ``importlib`` because the
    import statements for them would bind the same name twice.
    """
    try:
        importlib.import_module("vus")
    except ImportError:
        return None, REASON_BACKEND_UNAVAILABLE
    try:
        return importlib.import_module("vus.metrics"), None
    except ImportError as exc:
        return None, f"{REASON_BACKEND_UNIMPORTABLE}{exc}"


def score_vus_pr(scores: Sequence[float], labels: Sequence[int]) -> dict[str, Any]:
    """VUS-PR over one window series, as a value-and-reason object.

    The backend returns several metrics in one dict; ``metric="vus"`` asks for
    the VUS family, ``VUS_PR`` is the entry read back, and ``slidingWindow`` is
    upstream's name for the top of the tolerance range. The backend does array
    arithmetic on its inputs (``-np.sort(-score)``), so the sequences reach it
    as numpy arrays, and the numpy scalar it answers with comes back a plain
    float so the report serializes.

    Args:
        scores: per-window anomaly scores, ordered by window end.
        labels: per-window binary ground truth, aligned with ``scores``.

    Returns:
        ``{"value": float, "reason": None}`` when the metric was computed, and
        ``{"value": None, "reason": str}`` naming what stopped it when it was
        not. A backend that is missing, or installed but unimportable, is one
        of those reasons and not an exception: the field degrades, the run
        continues. A series with no positive label is another: the metric is
        undefined on it and the backend is not called.

    Raises:
        Exception: the backend's own scoring errors (a malformed input, say)
            pass through; only the import and the positive-free series
            degrade here. ``scry.eval.metrics`` catches them under
            ``REASON_SCORING_FAILED`` where the field is filled.
    """
    metrics, reason = _load_backend()
    if metrics is None:
        return {"value": None, "reason": reason}
    truth = np.asarray(labels, dtype=int)
    if not truth.any():
        return {"value": None, "reason": REASON_NO_POSITIVE_WINDOWS}
    scored = metrics.get_metrics(
        np.asarray(scores, dtype=float),
        truth,
        metric="vus",
        slidingWindow=MAX_TOLERANCE_WINDOWS,
    )
    return {"value": float(scored["VUS_PR"]), "reason": None}
