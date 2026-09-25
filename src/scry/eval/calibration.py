# Description: Guarded serving-threshold recalibration: the per-week band and its verdicts.
# Description: Torch-free; pure arithmetic over one previous/proposed threshold pair.

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

Age is not read here: callers pass ``weeks`` already anchored and floored (one
day, i.e. ``weeks = 1/7``, is the youngest band a rebake ever claims). Pure
arithmetic, no I/O and no clock.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

MAX_WEEKLY_GROWTH: float = 1.5
"""Most a threshold may grow per week of age."""

MAX_WEEKLY_SHRINK: float = 1.35
"""Most a threshold may shrink per week of age; tighter, shrinking is the FP direction."""

VERDICT_ACCEPTED = "accepted"
VERDICT_REJECTED_GREW = "REJECTED-grew"
VERDICT_REJECTED_SHRANK = "REJECTED-shrank"
VERDICT_REJECTED_NONFINITE = "REJECTED-nonfinite"


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
