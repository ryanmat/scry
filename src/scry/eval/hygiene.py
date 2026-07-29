# Description: Per-resource eligibility gates for per-resource threshold baking.
# Description: Torch-free; divergent coverage, window floor, and non-positive quantile verdicts.

"""Per-resource eligibility gates.

One implementation of the three gates that decide whether a resource's own
healthy quantile is safe to bake: divergent coverage (the resource lacks a
trained feature the capture supplies elsewhere, so capture-wide windowing
fills it on a scale serving never produces), the minimum-window floor, and a
non-positive quantile. Unlike a bake loop that skips a resource at its first
failing gate, every gate is evaluated so a verdict lists all of its reasons,
formatted ``"{REASON}:{detail}"``. Everything here is importable without
torch.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

# Floor on per-resource calibration size, the same floor the bake enforces: a
# resource with fewer windows serves the global threshold. This screens
# near-empty resources only; a small-sample quantile still leans tight and the
# margin multiplier is what carries the headroom.
MIN_PER_RESOURCE_WINDOWS: int = 50

REASON_DIVERGENT = "divergent-coverage"
REASON_TOO_FEW_WINDOWS = "insufficient-windows"
REASON_NONPOSITIVE_QUANTILE = "non-positive-quantile"


@dataclass(frozen=True)
class ResourceEligibility:
    """One resource's per-resource-threshold eligibility verdict."""

    resource_id: str
    eligible: bool
    reasons: list[str]
    n_windows: int
    own_quantile: float | None
    missing_features: list[str]


def per_resource_eligibility(
    *,
    trained_features: Sequence[str],
    features_by_resource: Mapping[str, set[str]],
    resource_ids: np.ndarray,
    errors: np.ndarray,
    quantile: float,
    min_windows: int = MIN_PER_RESOURCE_WINDOWS,
) -> dict[str, ResourceEligibility]:
    """Evaluate the three eligibility gates for every resource in the capture.

    Verdict-identical to the per-resource bake gates: capture features are the
    union of features present anywhere in the capture intersected with the
    trained features (a feature absent from the whole capture makes nobody
    divergent); iteration follows ``sorted(set(resource_ids))`` with
    stringified keys.

    Args:
        trained_features: The keeper's trained (numerical) feature names.
        features_by_resource: Per-resource metric_name sets from the capture,
            keyed by stringified resource id.
        resource_ids: Per-window resource ids, aligned with ``errors``.
        errors: Per-window reconstruction errors.
        quantile: The healthy quantile the bake would take per resource.
        min_windows: Floor on per-resource window count.

    Returns:
        Stringified resource id -> ``ResourceEligibility``, in raw sort order.
    """
    trained = set(trained_features)
    capture_features = set().union(*features_by_resource.values()) & trained
    resource_ids = np.asarray(resource_ids)
    errors = np.asarray(errors)

    verdicts: dict[str, ResourceEligibility] = {}
    for rid in sorted(set(resource_ids)):
        reasons: list[str] = []
        missing = sorted(capture_features - features_by_resource.get(str(rid), set()))
        if missing:
            reasons.append(f"{REASON_DIVERGENT}:{','.join(missing)}")
        resource_errors = errors[resource_ids == rid]
        n_windows = int(resource_errors.size)
        if n_windows < min_windows:
            reasons.append(f"{REASON_TOO_FEW_WINDOWS}:{n_windows}<{min_windows}")
        own_quantile = float(np.quantile(resource_errors, quantile)) if n_windows else None
        if own_quantile is not None and own_quantile <= 0:
            reasons.append(f"{REASON_NONPOSITIVE_QUANTILE}:{own_quantile}")
        verdicts[str(rid)] = ResourceEligibility(
            resource_id=str(rid),
            eligible=not reasons,
            reasons=reasons,
            n_windows=n_windows,
            own_quantile=own_quantile,
            missing_features=missing,
        )
    return verdicts
