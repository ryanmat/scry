# Description: Guarded serving-threshold recalibration: the per-week band and its verdicts.
# Description: Reads the seed file, importing torch lazily; the band itself is pure arithmetic.

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

``resolve_old_thresholds`` runs upstream of both and answers what ``old`` is.
A value the band can measure against has to be present, finite, and positive;
anything else -- an empty serving map on a first bake, a NaN that turns the band
off, an infinity that keeps itself in force forever, a zero that divides by
zero, a negative that is no threshold at all -- counts as missing and must be
supplied by an explicit ``--seed``. Absent that, the run is a spec error rather
than an unguarded bake. That resolution is the only I/O in this module (it reads
the seed file); the guard itself stays pure.

Age is not read here: callers pass ``weeks`` already anchored and floored (one
day, i.e. ``weeks = 1/7``, is the youngest band a rebake ever claims). Pure
arithmetic, no clock.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from scry.eval.rubric import SpecError

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from typing import Any

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

_ZIP_MAGIC = b"PK\x03\x04"
"""Leading bytes of a zip archive, and so of a torch checkpoint; a JSON seed is text."""


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


def _usable(value: float | None) -> float | None:
    """``value`` as a previous threshold the band can measure against, else ``None``.

    Usable means present, finite, and strictly positive. The band divides by
    the previous value and compares the ratio, so a NaN turns every comparison
    False and accepts whatever was proposed, an infinity can never be moved off
    by any finite proposal, and a zero divides by zero; a negative threshold is
    not a reconstruction error to begin with. Each of those counts as missing.

    Args:
        value: The recorded value, or ``None`` where the map had no entry. A
            number or ``None``; anything else raises ``TypeError`` at ``float``.
    """
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        return None
    return number


def _read_seed(seed_path: str | Path) -> tuple[float | None, Mapping[str, float]]:
    """Read a seed file in either accepted form into ``(global, per_resource)``.

    The form is chosen by content rather than by suffix, since ``--seed`` is an
    operator-supplied path: a torch checkpoint is a zip archive, so a file
    starting with the zip magic is read as one and its serving block carries the
    two values under ``threshold`` and ``per_resource``; anything else is the
    JSON form ``{"global": float, "per_resource": {rid: float}}``. Values are
    returned as recorded -- usability is decided per key by the caller, so a
    seed that carries a key uselessly is reported against that key.

    Raises:
        SpecError: If a seed that is not a checkpoint is not valid JSON either.
            The message names the seed path and carries the decode error, which
            is chained as the cause. Choosing the branch by the magic rather
            than by a failed parse is what keeps a hand-written JSON typo from
            being unpickled and reported as a torch failure instead.
    """
    raw = Path(seed_path).read_bytes()
    if raw.startswith(_ZIP_MAGIC):
        import torch  # local: this module stays importable without torch

        serving = torch.load(seed_path, map_location="cpu", weights_only=False).get("serving") or {}
        return serving.get("threshold"), serving.get("per_resource") or {}
    try:
        payload = json.loads(raw)
    except ValueError as exc:
        raise SpecError(
            f"the seed {seed_path!s} is not a zip-format torch checkpoint (it does not start "
            f"with the zip magic) and is not valid JSON: {exc}. A JSON seed is "
            '{"global": float, "per_resource": {rid: float}}.'
        ) from exc
    return payload.get("global"), payload.get("per_resource") or {}


def _resolve_one(
    key: str, current: float | None, seeded: float | None, seed_path: str | Path | None
) -> float:
    """The previous value for one key: the current one if usable, else the seeded one."""
    resolved = _usable(current)
    if resolved is None:
        resolved = _usable(seeded)
    if resolved is None:
        from_seed = (
            "no seed was given" if seed_path is None else f"the seed {seed_path!s} has none either"
        )
        raise SpecError(
            f"no usable previous threshold for {key}: the current serving value is absent or "
            f"unusable (non-finite, zero, or negative) and {from_seed}. Pass --seed PATH "
            f"carrying {key} -- a checkpoint whose serving block has a per_resource map and a "
            "global threshold, or a JSON map of global and per_resource. An unguarded first "
            "bake is an error, not a default."
        )
    return resolved


def resolve_old_thresholds(
    serving: Mapping[str, Any],
    resource_ids: Iterable[str],
    *,
    seed_path: str | Path | None = None,
) -> tuple[float, dict[str, float]]:
    """Resolve the ``(old_global, old)`` a rebake is guarded against.

    Requirement 1, upstream of ``check_guards``: every value the guard needs
    comes from the live serving block where that block has a usable one, and
    from an explicit seed where it does not. The seed is a fallback, never an
    override -- a usable serving value is what the rebake is measured against.

    Args:
        serving: The current serving block (``{"threshold": float,
            "per_resource": {rid: float}, ...}``), as a checkpoint carries it.
            Empty for a checkpoint with no serving block at all.
        resource_ids: The resources the fresh bake proposes a threshold for.
            Passing the bake's ``new`` map works: only its keys are read.
        seed_path: The ``--seed`` file: a checkpoint whose serving block
            carries a per_resource map and a global threshold, or a JSON
            ``{"global": float, "per_resource": {rid: float}}`` map.

    Returns:
        ``(old_global, old)``, the inputs ``check_guards`` takes: a positive
        finite global, and a per-resource map covering every id in
        ``resource_ids`` plus every other usable entry of the serving map (a
        resource the bake did not propose is an omission for the report to
        record, not one to drop here).

    Raises:
        SpecError: If any needed value is neither usable in ``serving`` nor
            usably supplied by the seed. The message names ``--seed`` and the
            key -- ``global``, or the resource id. Also if the seed file itself
            is neither a checkpoint nor valid JSON, naming the path and the
            decode error.
    """
    current_per_resource: Mapping[str, float] = serving.get("per_resource") or {}
    seed_global, seed_per_resource = (None, {}) if seed_path is None else _read_seed(seed_path)

    old_global = _resolve_one("global", serving.get("threshold"), seed_global, seed_path)
    old: dict[str, float] = {
        resource_id: _resolve_one(
            resource_id,
            current_per_resource.get(resource_id),
            seed_per_resource.get(resource_id),
            seed_path,
        )
        for resource_id in resource_ids
    }
    for resource_id, value in current_per_resource.items():
        if resource_id not in old:
            usable = _usable(value)
            if usable is not None:
                old[resource_id] = usable

    return old_global, old
