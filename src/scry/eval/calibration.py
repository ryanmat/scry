# Description: Guarded serving-threshold recalibration: the per-week band and its verdicts.
# Description: Torch-free; pure arithmetic over a previous/proposed threshold map.

"""Band guards for a recalibrated serving threshold.

A rebake may only move a threshold so far per week of age, and the band is
exponential in that age: a value may grow by at most ``MAX_WEEKLY_GROWTH **
weeks`` and may shrink by at most a factor ``MAX_WEEKLY_SHRINK ** weeks``.
Downward is the dangerous direction -- a threshold that collapses turns healthy
windows into alerts -- so its base is held tighter than the growth base.

``guard_value`` applies that band to ONE value: a per-resource threshold, or
the global threshold riding as ``resource_id=None``. It returns the value that
is kept beside the ``GuardVerdict`` that records the decision. A proposal
inside the band is ``accepted`` and kept; one outside it is ``REJECTED-grew``
or ``REJECTED-shrank`` and the previous value is kept instead. A non-finite
proposal never reaches the band at all: it is ``REJECTED-nonfinite``, because
every band comparison against a NaN is False and an unguarded band would
therefore accept it and serve it. Every verdict carries ``old``, ``proposed``,
the observed ``ratio``, and the ``limit`` that ratio was measured against --
the grow limit for a value that grew, the shrink floor for one that shrank --
so a report reader can re-derive the decision from the verdict alone.

``check_guards`` applies that guard to a whole rebake: the global threshold and
every proposed per-resource threshold. The global is guarded too -- it used to
ship wholesale while only per-resource entries met the band, and the measured
2026-07-28 jump 0.193205 -> 0.417031 landed above the serving grid's peak on
the one threshold the POST endpoint actually resolves.

Age is not read here: callers pass ``weeks`` already anchored and floored (one
day, i.e. ``weeks = 1/7``, is the youngest band a rebake ever claims). Pure
arithmetic, no I/O and no clock.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from scry.eval.hygiene import ResourceEligibility

MAX_WEEKLY_GROWTH: float = 1.5
"""Most a threshold may grow per week of age."""

MAX_WEEKLY_SHRINK: float = 1.35
"""Most a threshold may shrink per week of age; tighter, shrinking is the FP direction."""

VERDICT_ACCEPTED = "accepted"
VERDICT_ACCEPTED_ALLOW_DRIFT = "accepted-allow-drift"
VERDICT_REJECTED_GREW = "REJECTED-grew"
VERDICT_REJECTED_SHRANK = "REJECTED-shrank"
VERDICT_REJECTED_NONFINITE = "REJECTED-nonfinite"

_BAND_REJECTIONS = frozenset({VERDICT_REJECTED_GREW, VERDICT_REJECTED_SHRANK})
"""The verdicts ``allow_drift`` may override: a band decision, and only that."""


@dataclass(frozen=True)
class GuardVerdict:
    """One guarded value's decision, as the rebake report records it."""

    resource_id: str | None
    """The resource the value belongs to; ``None`` is the global threshold."""

    verdict: str
    old: float | None
    proposed: float | None
    ratio: float | None
    limit: float | None
    """The band bound ``ratio`` was measured against, in the direction it moved."""


def guard_value(
    resource_id: str | None,
    old: float,
    proposed: float,
    weeks: float,
    *,
    max_weekly_growth: float = MAX_WEEKLY_GROWTH,
    max_weekly_shrink: float = MAX_WEEKLY_SHRINK,
) -> tuple[float, GuardVerdict]:
    """Guard one proposed threshold against the per-week band.

    Args:
        resource_id: The resource the value belongs to, or ``None`` for the
            global threshold.
        old: The previous (serving or seeded) value. Must be positive; a
            non-positive previous threshold is an eligibility question
            (``scry.eval.hygiene``), not a band one.
        proposed: The freshly baked value.
        weeks: Age of the previous value in weeks, already anchored and
            floored by the caller.
        max_weekly_growth: Per-week base of the grow limit.
        max_weekly_shrink: Per-week base of the shrink floor.

    Returns:
        ``(kept, verdict)``: the value that stays in force -- ``proposed`` when
        it is inside the band, ``old`` when it is not -- and the
        ``GuardVerdict`` recording old, proposed, ratio, and the limit that
        ratio was measured against. A ratio exactly on a bound is accepted. A
        non-finite ``proposed`` is ``REJECTED-nonfinite`` with ``old`` kept and
        no ratio or limit, since no band comparison was made.
    """
    if not math.isfinite(proposed):
        return old, GuardVerdict(
            resource_id=resource_id,
            verdict=VERDICT_REJECTED_NONFINITE,
            old=old,
            proposed=proposed,
            ratio=None,
            limit=None,
        )

    ratio = proposed / old
    grow_limit = max_weekly_growth**weeks
    shrink_floor = 1.0 / (max_weekly_shrink**weeks)

    if ratio > grow_limit:
        kept, verdict, limit = old, VERDICT_REJECTED_GREW, grow_limit
    elif ratio < shrink_floor:
        kept, verdict, limit = old, VERDICT_REJECTED_SHRANK, shrink_floor
    else:
        kept, verdict = proposed, VERDICT_ACCEPTED
        limit = grow_limit if ratio >= 1.0 else shrink_floor

    return kept, GuardVerdict(
        resource_id=resource_id,
        verdict=verdict,
        old=old,
        proposed=proposed,
        ratio=ratio,
        limit=limit,
    )


def _guard_one(
    resource_id: str | None,
    old: float,
    proposed: float,
    weeks: float,
    *,
    allow_drift: bool,
    max_weekly_growth: float,
    max_weekly_shrink: float,
) -> tuple[float, GuardVerdict]:
    """``guard_value`` plus the ``allow_drift`` override of a band rejection.

    The override is of the band decision only: a proposal the band rejected is
    kept and its verdict restamped ``accepted-allow-drift``, with the ratio and
    the limit it was measured against left on the verdict so the report shows
    what was overridden. A proposal that was inside the band keeps the plain
    ``accepted`` verdict, so the stamp names exactly the values that drifted.
    ``REJECTED-nonfinite`` is not a band decision -- no ratio was formed -- and
    is never overridden.
    """
    kept, verdict = guard_value(
        resource_id,
        old,
        proposed,
        weeks,
        max_weekly_growth=max_weekly_growth,
        max_weekly_shrink=max_weekly_shrink,
    )
    if allow_drift and verdict.verdict in _BAND_REJECTIONS:
        return proposed, replace(verdict, verdict=VERDICT_ACCEPTED_ALLOW_DRIFT)
    return kept, verdict


def check_guards(
    old_global: float | None,
    old: Mapping[str, float],
    new_global: float,
    new: Mapping[str, float],
    weeks: float,
    *,
    eligibility: Mapping[str, ResourceEligibility] | None = None,
    allow_drift: bool = False,
    max_weekly_growth: float = MAX_WEEKLY_GROWTH,
    max_weekly_shrink: float = MAX_WEEKLY_SHRINK,
) -> tuple[float, dict[str, float], list[GuardVerdict]]:
    """Guard a whole rebake: the global threshold and every proposed resource.

    A pure function of ``(old, new, weeks)``. Resolving where ``old`` came from
    -- the live serving block or an explicit seed -- happens upstream, and so
    does stamping the verdicts whose ``old`` came from a seed.

    Args:
        old_global: The previous global threshold. Must be a positive float:
            the seed requirement resolves a first bake's missing global
            upstream, so ``None`` never reaches the band.
        old: The previous per-resource thresholds. Must cover every resource in
            ``new`` -- an unseeded proposal has no band to measure against, and
            accepting it is the unguarded first bake this module prevents.
        new_global: The freshly baked global threshold.
        new: The freshly baked per-resource thresholds.
        weeks: Age of the previous values in weeks, already anchored and
            floored by the caller.
        eligibility: The bake's eligibility map. Accepted for the omission
            provenance of resources present in ``old`` and absent from ``new``;
            not read yet.
        allow_drift: Keep values the band rejected, restamped
            ``accepted-allow-drift``. Bypasses the band and nothing else:
            neither the seed requirement above nor the non-finite rejection.
        max_weekly_growth: Per-week base of the grow limit.
        max_weekly_shrink: Per-week base of the shrink floor.

    Returns:
        ``(kept_global, kept, verdicts)``: the global threshold that stays in
        force, the per-resource map that stays in force (an accepted proposal,
        or the previous value where the proposal was rejected), and one verdict
        per guarded value -- the global first as ``resource_id=None``, then the
        resources in sorted order. A resource in ``old`` with no proposal in
        ``new`` is neither guarded nor carried here; distinguishing a
        hygiene-gate omission from absence from the capture is what
        ``eligibility`` is for.
    """
    kept_global, global_verdict = _guard_one(
        None,
        old_global,
        new_global,
        weeks,
        allow_drift=allow_drift,
        max_weekly_growth=max_weekly_growth,
        max_weekly_shrink=max_weekly_shrink,
    )
    verdicts = [global_verdict]
    kept: dict[str, float] = {}
    for resource_id in sorted(new):
        kept[resource_id], verdict = _guard_one(
            resource_id,
            old[resource_id],
            new[resource_id],
            weeks,
            allow_drift=allow_drift,
            max_weekly_growth=max_weekly_growth,
            max_weekly_shrink=max_weekly_shrink,
        )
        verdicts.append(verdict)

    return kept_global, kept, verdicts
