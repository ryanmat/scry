# Description: VUS-PR entry point over TheDatumOrg/VUS, consumed as an optional dependency.
# Description: Torch-free; degrades to a null value with a stated reason, never raising.

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
answers with the same value-and-reason object either way: a float once the
scoring path lands, null with the reason stated when it cannot be computed.
Nothing here raises for a missing backend. VUS-PR is reported and never gates
(section 8.1), so a suite run on a host without the backend must still produce
a report rather than die on an import. Everything here is importable without
torch.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import ModuleType
from typing import Any

REASON_BACKEND_UNAVAILABLE = (
    "VUS-PR unavailable: the optional 'vus' backend (TheDatumOrg/VUS) is not installed"
)
REASON_SCORING_PATH_PENDING = (
    "VUS-PR unavailable: the 'vus' backend is installed but the scoring path has not landed"
)


def _load_backend() -> ModuleType | None:
    """The optional ``vus`` backend module, or None when it is not installed.

    Deferred rather than imported at module scope so that importing this
    module, and the eval package that exports it, never depends on the
    optional package being present.
    """
    try:
        import vus
    except ImportError:
        return None
    return vus


def score_vus_pr(scores: Sequence[float], labels: Sequence[int]) -> dict[str, Any]:
    """VUS-PR over one window series, as a value-and-reason object.

    Args:
        scores: per-window anomaly scores, ordered by window end.
        labels: per-window binary ground truth, aligned with ``scores``.

    Returns:
        ``{"value": float, "reason": None}`` when the metric was computed, and
        ``{"value": None, "reason": str}`` naming what stopped it when it was
        not. A missing backend is one of those reasons, not an exception: the
        field degrades, the run continues. The computation over the backend,
        and with it the range of tolerances it sweeps, is not written yet; an
        installed backend therefore yields its own stated reason rather than
        being mistaken for a missing one.
    """
    if _load_backend() is None:
        return {"value": None, "reason": REASON_BACKEND_UNAVAILABLE}
    return {"value": None, "reason": REASON_SCORING_PATH_PENDING}
