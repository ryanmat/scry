# Description: Guarded serving-threshold recalibration: the per-week band and its verdicts.
# Description: Reads the seed and checkpoint and runs the bake lazily; the band is pure arithmetic.

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
than an unguarded bake. That resolution reads the seed file; the guard itself
stays pure.

``guard_rebake`` composes the three: resolve, guard, then stamp the provenance
the band cannot know. A threshold measured against a seed is ``accepted-seeded``
-- inside the band, but against a value that was never in production -- while
one measured against the live serving block stays plainly ``accepted``. A
serving resource the bake proposed nothing for keeps its threshold and records
why: ``kept-gate-omitted:{reason}`` when the bake's eligibility map shows a
hygiene gate dropped it, ``kept-absent-from-capture`` when the capture did not
contain it at all. The two are different operator actions, which is why one
verdict cannot serve for both. A serving entry the bake proposed nothing for
whose value the predictor would refuse -- non-finite, zero, negative, or null --
is ``kept-global-unusable:{kind}``: ``Predictor._resolve_per_resource_thresholds``
drops such an entry and serves the global, so the map that stays in force
leaves it out the same way and its row names the global as what serves.

``weeks_since_rebake`` supplies the ``weeks`` all of the above take, and reads
the clock for that age. Age anchors on the stamp the serving
block carries, because the mtime of the serving checkpoint is the staged write
time the swap's copy/move chain left behind -- a file minutes old can carry a
block baked weeks ago, and a band anchored on the file widens to match the copy
rather than the bake. The mtime is the fallback for a block with no stamp, and
either way the age floors at one day (``weeks = 1/7``), the narrowest band a
rebake ever claims. The guard and the report are pure arithmetic over the
``weeks`` it returns.

``build_rebake_report`` turns all of that into the document an operator reads:
one row per guarded value, carrying the verdict, the numbers it was decided
from, and the threshold that stays in force afterwards. That last field is why
a row exists at all rather than a verdict string -- a ``kept-gate-omitted``
resource goes on serving its own previous threshold, and a report that showed
only the verdict would leave a reader to assume it fell back to the global.
The report is strict JSON: a non-finite proposal, which ``json.dumps`` would
otherwise write as a bare ``NaN`` that no strict reader accepts, is recorded as
``"nan"``, ``"inf"``, or ``"-inf"``, so the one artifact that records a
non-finite rejection can always be written.

``run_calibration`` is the run itself, composed from all of the above, and the
module's I/O: it loads the checkpoint, fetches the capture, runs the bake's own
arithmetic against the base model, reads the clock once for ``generated_at``,
then applies the age anchor, the guard, and the report -- which carries the
run's inputs and the bake's warnings beside the verdicts, so the document says
what was baked, from what, and how much of it was refused. It returns that
report and the serving block a swap would put in force, both in memory: the
checkpoint an operator is serving cannot change before anyone has read what the
rebake decided. A kept global keeps the provenance of the fit that produced it
(``quantile``, ``healthy_fpr``, ``n_calibration_windows``), not the discarded
fit's.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
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
VERDICT_ACCEPTED_SEEDED = "accepted-seeded"
VERDICT_ACCEPTED_ALLOW_DRIFT = "accepted-allow-drift"
VERDICT_REJECTED_GREW = "REJECTED-grew"
VERDICT_REJECTED_SHRANK = "REJECTED-shrank"
VERDICT_REJECTED_NONFINITE = "REJECTED-nonfinite"
VERDICT_KEPT_GATE_OMITTED = "kept-gate-omitted"
"""Prefix; the full verdict is ``kept-gate-omitted:{reason}``, the hygiene gate's own."""

VERDICT_KEPT_ABSENT_FROM_CAPTURE = "kept-absent-from-capture"

VERDICT_KEPT_GLOBAL_UNUSABLE = "kept-global-unusable"
"""Prefix; the full verdict is ``kept-global-unusable:{kind}``, the kind of unusable value.

A serving entry the bake proposed nothing for and whose recorded value the
predictor refuses (``nan``, ``inf``, ``-inf``, ``zero``, ``negative``, ``null``).
The predictor serves the global for it, so the map that stays in force leaves
it out, and the row's ``kept`` is the global.
"""

_GLOBAL_PROVENANCE = ("quantile", "healthy_fpr", "n_calibration_windows")
"""The serving-block keys that describe the fit a global threshold came from."""

_BAND_REJECTIONS = frozenset({VERDICT_REJECTED_GREW, VERDICT_REJECTED_SHRANK})
"""The verdicts ``allow_drift`` may override: a band decision, and only that."""

_REJECTIONS = _BAND_REJECTIONS | {VERDICT_REJECTED_NONFINITE}
"""Every refusal, band or non-finite; the report counts them at the top."""

_ZIP_MAGIC = b"PK\x03\x04"
"""Leading bytes of a zip archive, and so of a torch checkpoint; a JSON seed is text."""

DEFAULT_QUANTILE = 0.99
"""Healthy-window quantile the bake takes, as ``scripts/bake_serving_threshold.py`` has it."""

DEFAULT_MARGIN = 2.0
"""Per-resource margin on each resource's own healthy quantile; the sweep's selection."""

AGE_FLOOR = timedelta(days=1)
"""Youngest age a rebake may claim, however recently the previous one ran."""

_ONE_WEEK = timedelta(days=7)
"""The unit ``weeks`` is measured in; the band's exponent is in weeks."""


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


def _omission_verdict(
    resource_id: str,
    old: float,
    eligibility: Mapping[str, ResourceEligibility] | None,
) -> GuardVerdict:
    """The verdict for a serving resource the bake proposed no threshold for.

    Requirement 4: the previous report could not tell a hygiene-gate omission
    from a resource the capture never contained, which are different operator
    actions -- fix the gate, or retire the threshold. The bake's eligibility
    map is what distinguishes them, and the gate's own reason rides on the
    verdict, so the rebake report names the gate rather than the symptom. A
    resource the map lists as eligible cannot reach here (the bake proposes a
    threshold for every eligible resource) and reads as absent from the
    capture, as it does when no eligibility map was supplied at all.
    """
    verdict_for = None if eligibility is None else eligibility.get(resource_id)
    if verdict_for is None or verdict_for.eligible:
        verdict = VERDICT_KEPT_ABSENT_FROM_CAPTURE
    else:
        # Reasons are recorded in gate order, so the first one names the gate
        # that omitted the resource -- the convention the bake's warnings use.
        verdict = f"{VERDICT_KEPT_GATE_OMITTED}:{verdict_for.reasons[0]}"
    return GuardVerdict(
        resource_id=resource_id,
        verdict=verdict,
        old=old,
        proposed=None,
        ratio=None,
        limit=None,
    )


def check_guards(
    old_global: float | None,
    old: Mapping[str, float | None],
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
        eligibility: The bake's eligibility map, read for the omission
            provenance of a resource present in ``old`` and absent from
            ``new``: a hygiene gate dropped it, or the capture never had it.
        allow_drift: Keep values the band rejected, restamped
            ``accepted-allow-drift``. Bypasses the band and nothing else:
            neither the seed requirement above nor the non-finite rejection.
        max_weekly_growth: Per-week base of the grow limit.
        max_weekly_shrink: Per-week base of the shrink floor.

    Returns:
        ``(kept_global, kept, verdicts)``: the global threshold that stays in
        force, the per-resource map that stays in force (an accepted proposal,
        or the previous value where the proposal was rejected or where the bake
        proposed nothing), and one verdict per value -- the global first as
        ``resource_id=None``, then every resource of either map in sorted
        order. A resource in ``old`` with no proposal in ``new`` keeps its
        previous threshold, since dropping it from the map would fall the
        resource back to the global with no verdict saying so, and its verdict
        records why it was omitted (``eligibility``). Where that previous value
        is one the predictor refuses (non-finite, zero, negative, or null) the
        resource is left out of ``kept``, as the predictor leaves it out, and
        its verdict is ``kept-global-unusable:{kind}``.
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
    for resource_id in sorted({*new, *old}):
        if resource_id in new:
            kept[resource_id], verdict = _guard_one(
                resource_id,
                old[resource_id],
                new[resource_id],
                weeks,
                allow_drift=allow_drift,
                max_weekly_growth=max_weekly_growth,
                max_weekly_shrink=max_weekly_shrink,
            )
        elif _usable(old[resource_id]) is None:
            # The predictor drops a per-resource threshold it cannot serve and
            # falls the resource back to the global; the map that stays in
            # force does the same, and the row says which value it was.
            verdict = GuardVerdict(
                resource_id=resource_id,
                verdict=f"{VERDICT_KEPT_GLOBAL_UNUSABLE}:{_unusable_kind(old[resource_id])}",
                old=old[resource_id],
                proposed=None,
                ratio=None,
                limit=None,
            )
        else:
            kept[resource_id] = old[resource_id]
            verdict = _omission_verdict(resource_id, old[resource_id], eligibility)
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


def _unusable_kind(value: float | None) -> str:
    """Why ``_usable`` refused ``value``, as the verdict suffix names it.

    One of ``null``, ``nan``, ``inf``, ``-inf``, ``zero``, or ``negative``; a
    usable value raises ``ValueError``, since no verdict names one.
    """
    if value is None:
        return "null"
    number = float(value)
    if math.isnan(number):
        return "nan"
    if math.isinf(number):
        return "inf" if number > 0 else "-inf"
    if number == 0.0:
        return "zero"
    if number < 0.0:
        return "negative"
    raise ValueError(f"{value!r} is a usable threshold")


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
) -> tuple[float, dict[str, float | None]]:
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
        finite global, a value the band can measure against for every id in
        ``resource_ids``, and every other entry of the serving map as it
        stands, usable or not -- a resource the bake did not propose is an
        omission for the report to record, not one to drop here.

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
    old: dict[str, float | None] = {
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
            # Carried as recorded, usable or not: nothing proposed a threshold
            # for this resource, so no band measures against the value, while
            # screening it out here would drop its omission verdict with it and
            # fall the resource back to the global threshold with nothing in
            # the report saying so (requirement 4). check_guards decides what
            # an unusable one means.
            old[resource_id] = value

    return old_global, old


def _stamp_seeded(verdict: GuardVerdict, serving: Mapping[str, Any]) -> GuardVerdict:
    """``accepted-seeded`` where the value the band measured against came from the seed.

    Requirement 1 leaves the stamp to the caller, since ``check_guards`` is a
    pure function of ``(old, new, weeks)`` and cannot know where ``old`` came
    from; here the serving block is in hand, so the test is the same usability
    rule the resolution applied to it. The stamp replaces the plain
    ``accepted`` verdict only: a band rejection, a non-finite rejection, an
    allow-drift bypass, and a kept omission each name a decision, and the
    provenance must not overwrite one -- a rejection measured against a seed
    is still a rejection, and its seeded ``old`` is recorded on the verdict.
    """
    if verdict.verdict != VERDICT_ACCEPTED:
        return verdict
    if verdict.resource_id is None:
        current = serving.get("threshold")
    else:
        current = (serving.get("per_resource") or {}).get(verdict.resource_id)
    if _usable(current) is not None:
        return verdict
    return replace(verdict, verdict=VERDICT_ACCEPTED_SEEDED)


def guard_rebake(
    serving: Mapping[str, Any],
    new_global: float,
    new: Mapping[str, float],
    weeks: float,
    *,
    eligibility: Mapping[str, ResourceEligibility] | None = None,
    seed_path: str | Path | None = None,
    allow_drift: bool = False,
    max_weekly_growth: float = MAX_WEEKLY_GROWTH,
    max_weekly_shrink: float = MAX_WEEKLY_SHRINK,
) -> tuple[float, dict[str, float], list[GuardVerdict]]:
    """Guard a rebake against the serving block it would replace.

    The composed path: resolve what ``old`` is (the live serving block, or the
    seed where that block has nothing usable), run the band over it, then stamp
    the provenance the band cannot know. ``check_guards`` stays the pure
    function of ``(old, new, weeks)`` it is; this is its caller.

    Args:
        serving: The current serving block the rebake would replace.
        new_global: The freshly baked global threshold.
        new: The freshly baked per-resource thresholds.
        weeks: Age of the serving block in weeks, already anchored and floored.
        eligibility: The bake's eligibility map, for omission provenance.
        seed_path: The ``--seed`` file, for values ``serving`` cannot supply.
        allow_drift: Keep values the band rejected, restamped.
        max_weekly_growth: Per-week base of the grow limit.
        max_weekly_shrink: Per-week base of the shrink floor.

    Returns:
        ``(kept_global, kept, verdicts)`` as ``check_guards`` returns them,
        with every verdict measured against a seeded ``old`` restamped
        ``accepted-seeded`` -- the report then says which thresholds were never
        in production, rather than reading as if the band had confirmed them.

    Raises:
        SpecError: If a value the band needs is neither usable in ``serving``
            nor usably supplied by the seed.
    """
    old_global, old = resolve_old_thresholds(serving, new, seed_path=seed_path)
    kept_global, kept, verdicts = check_guards(
        old_global,
        old,
        new_global,
        new,
        weeks,
        eligibility=eligibility,
        allow_drift=allow_drift,
        max_weekly_growth=max_weekly_growth,
        max_weekly_shrink=max_weekly_shrink,
    )
    return kept_global, kept, [_stamp_seeded(verdict, serving) for verdict in verdicts]


def _parse_stamp(stamp: str) -> datetime:
    """An ISO-8601 serving stamp as an aware UTC moment.

    The repo writes UTC with a ``Z`` suffix (``scry.eval.provenance._iso_z``),
    which ``fromisoformat`` only learned to read in 3.11 while this package
    supports 3.10, so the suffix is spelled out as an offset first. A stamp
    carrying no offset at all is read as UTC, as ``scripts/extract_features.py``
    reads one.
    """
    if stamp.endswith("Z"):
        stamp = f"{stamp[:-1]}+00:00"
    parsed = datetime.fromisoformat(stamp)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def weeks_since_rebake(serving: Mapping[str, Any], checkpoint_path: str | Path) -> float:
    """Age of the serving block in weeks, as the band's exponent takes it.

    Requirement 3. The age used to come from the serving checkpoint's mtime,
    which the swap's copy/move chain sets to the staged write time: the file is
    then as young as the last swap rather than as old as the bake it carries,
    and the band -- exponential in this number -- narrows to match the copy. So
    the stamp the serving block itself carries is the anchor, and the mtime is
    the fallback for a block written before anything stamped one.

    Args:
        serving: The current serving block. ``rebaked_at`` is read if present:
            an ISO-8601 moment, as the staged block records it; one with no
            offset is read as UTC. A stamp that is not an ISO-8601 string
            raises rather than falling back, since a block that records its age
            wrongly is not a block to guess the age of.
        checkpoint_path: The checkpoint carrying ``serving``, whose mtime is the
            fallback anchor. Stat'ed, never read.

    Returns:
        Age in weeks, floored at one day (``1/7``) -- the floor the previous
        implementation had, and what keeps a rebake minutes after another from
        claiming a band so narrow that no threshold could move, or a clock skew
        that puts the stamp in the future from claiming a negative one.
    """
    stamp = serving.get("rebaked_at")
    if stamp is None:
        anchor = datetime.fromtimestamp(Path(checkpoint_path).stat().st_mtime, tz=timezone.utc)
    else:
        anchor = _parse_stamp(stamp)
    return max(datetime.now(timezone.utc) - anchor, AGE_FLOOR) / _ONE_WEEK


def _json_number(value: float | None) -> float | str | None:
    """One number as strict JSON takes it: a non-finite one under its name.

    ``json.dumps`` writes NaN and the infinities as the bare literals ``NaN``,
    ``Infinity``, and ``-Infinity``, which are not JSON -- and ``allow_nan=
    False`` refuses them outright rather than writing them. A rebake that
    proposed a non-finite threshold would then be unable to write the report
    recording that rejection, the one place the proposal is visible at all. So
    the value is written as ``"nan"``, ``"inf"``, or ``"-inf"``: still legible,
    and a string no consumer can mistake for a number to compute with.
    """
    if value is None:
        return None
    number = float(value)
    if math.isfinite(number):
        return number
    if math.isnan(number):
        return "nan"
    return "inf" if number > 0.0 else "-inf"


def _verdict_row(verdict: GuardVerdict, kept: float) -> dict[str, Any]:
    """One verdict as a report row, naming the threshold that stays in force.

    The verdict records the decision; ``kept`` records its consequence, and the
    two are not the same reading. An ``accepted`` row serves its proposal, a
    rejected one serves the previous value, and an omitted one serves the
    previous value as well -- which is the row that most needs saying so, since
    a reader who sees no threshold for a resource would otherwise take it to
    have fallen back to the global one. The one omitted row that does fall back,
    ``kept-global-unusable``, says so by carrying the global as ``kept``.
    """
    return {
        "verdict": verdict.verdict,
        "old": _json_number(verdict.old),
        "proposed": _json_number(verdict.proposed),
        "kept": _json_number(kept),
        "ratio": _json_number(verdict.ratio),
        "limit": _json_number(verdict.limit),
    }


def build_rebake_report(
    kept_global: float,
    kept: Mapping[str, float],
    verdicts: Iterable[GuardVerdict],
    weeks: float,
    *,
    dry_run: bool,
    staged_path: str | Path | None = None,
) -> dict[str, Any]:
    """Assemble the rebake report from a guarded bake.

    One row per guarded value -- the global its own row, the resources a map of
    them -- carrying the verdict, the numbers it was decided from, and the
    threshold the row keeps; around them the age the band was measured over,
    where the bake was staged, and ``dry_run`` as a field of the document
    rather than a property of the exit code (requirement 5 -- one code can no
    longer mean both a dry run and a live swap that partially held, so the
    report has to say which this was). ``run_calibration`` adds the run-level
    keys: what was baked, from what, and how much of it was refused.

    A pure function of its arguments: no clock, no file, nothing read. Every
    number goes through the strict-JSON encoding above, so the result always
    survives ``json.dumps(..., allow_nan=False)`` and reads back equal.

    Args:
        kept_global: The global threshold that stays in force.
        kept: The per-resource thresholds that stay in force, as
            ``check_guards`` returns it: every resource in ``verdicts`` except
            a ``kept-global-unusable`` one, whose row carries ``kept_global``.
        verdicts: The verdicts of this rebake, from ``check_guards`` or
            ``guard_rebake``: the global as ``resource_id=None``, then the
            resources in sorted order, which is the order published here.
        weeks: Age of the previous values in weeks, the band's own exponent.
        dry_run: Whether this run was to stop short of a live swap.
        staged_path: Where the staged checkpoint was written, if one was.

    Returns:
        ``{"dry_run", "weeks", "staged_path", "global", "per_resource"}``,
        where ``global`` is one row and ``per_resource`` maps each resource id
        to one. A row is ``{"verdict", "old", "proposed", "kept", "ratio",
        "limit"}``: the decision, the two values it was made from, the
        threshold that serves afterwards, and the comparison the band made
        (``ratio`` and ``limit`` are ``None`` where it made none -- an omitted
        resource, or a non-finite proposal that never formed a ratio).
    """
    global_row: dict[str, Any] | None = None
    per_resource: dict[str, dict[str, Any]] = {}
    for verdict in verdicts:
        if verdict.resource_id is None:
            global_row = _verdict_row(verdict, kept_global)
        else:
            per_resource[verdict.resource_id] = _verdict_row(
                verdict, kept.get(verdict.resource_id, kept_global)
            )

    return {
        "dry_run": bool(dry_run),
        "weeks": _json_number(weeks),
        "staged_path": None if staged_path is None else str(staged_path),
        "global": global_row,
        "per_resource": per_resource,
    }


def _bake(
    model_path: str | Path,
    calibration: str,
    *,
    profile: str | None,
    quantile: float,
    margin: float,
) -> tuple[dict[str, Any], Mapping[str, ResourceEligibility] | None, list[str]]:
    """Bake against the BASE model and take the serving block it would write.

    The arithmetic is ``scripts/bake_serving_threshold.py``'s own rather than a
    second copy of it here to drift from it -- the reason the bake delegates its
    gates to ``scry.eval.hygiene`` instead of keeping its own. Its CLI writes
    the block into a checkpoint; this takes the block itself, so a proposal can
    be guarded before anything at all is written. Returned beside it: the
    hygiene verdicts behind its ``per_resource`` map (omission provenance,
    requirement 4), and the warning lines it printed, which name the resources
    it could not propose for and are re-emitted on stderr as well as reported.
    """
    import asyncio

    # local: load_keeper pulls in torch, and this module stays importable
    # without it, as _serving_block's import says.
    from scry.data.fetcher import fetch_full_capture
    from scry.model.checkpoint import load_keeper
    from scry.utils.config import get_config

    # The scripts directory is not a package: it is already importable when the
    # rebake CLI runs beside it (and under the test suite), and is added from
    # this checkout otherwise.
    scripts = Path(__file__).resolve().parents[3] / "scripts"
    if scripts.is_dir() and str(scripts) not in sys.path:
        sys.path.append(str(scripts))
    import bake_serving_threshold

    keeper = load_keeper(str(model_path))
    df_long = asyncio.run(fetch_full_capture(calibration, profile=profile or keeper.profile))
    stderr = io.StringIO()
    try:
        with contextlib.redirect_stderr(stderr):
            block, eligibility = bake_serving_threshold.compute_serving_block(
                keeper,
                df_long,
                quantile=quantile,
                step=int(get_config().window_step),
                per_resource_margin=margin,
            )
    finally:
        # Re-emitted whether or not the bake finished: what it printed before
        # raising is the operator's only trace of why.
        sys.stderr.write(stderr.getvalue())
    printed = stderr.getvalue().splitlines()
    return block, eligibility, [line for line in printed if line.startswith("warning: ")]


def _serving_block(model_path: str | Path) -> dict[str, Any]:
    """The serving block the checkpoint carries today, or ``{}`` where it has none."""
    import torch  # local: this module stays importable without torch

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    return dict(checkpoint.get("serving") or {})


def run_calibration(
    model_path: str | Path,
    calibration: str,
    *,
    profile: str | None = None,
    quantile: float = DEFAULT_QUANTILE,
    margin: float = DEFAULT_MARGIN,
    calibration_days: float | None = None,
    seed_path: str | Path | None = None,
    allow_drift: bool = False,
    dry_run: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bake, guard, and report a recalibration, in memory.

    Age the serving block the checkpoint carries, bake a fresh one against that
    same checkpoint, put every proposed value through the band, and assemble
    the report. Nothing is written -- not the input, not a staged copy -- so the
    thresholds an operator is serving cannot move before the report of what
    would move them exists.

    Args:
        model_path: The keeper being rebaked. Its serving block is both what
            the band measures against and what the age is anchored on.
        calibration: The all-healthy capture to bake from.
        profile: Feature profile; defaults to the checkpoint's stored one.
        quantile: Healthy-window quantile for the thresholds.
        margin: Per-resource margin on each resource's own healthy quantile.
        calibration_days: Days of healthy data behind this bake, recorded on
            the report and on the block for the next run to read.
        seed_path: The ``--seed`` file, for values the serving block has
            nothing usable for (requirement 1).
        allow_drift: Keep values the band rejected, restamped.
        dry_run: Whether this run stops short of a live swap. A report field
            rather than a property of the exit code (requirement 5).

    Returns:
        ``(report, block)``: ``build_rebake_report``'s document plus the
        run-level keys -- what was baked, from what, which resources it left
        out, how many values it refused -- and the serving block that would go
        into force, which is the baked block carrying the GUARDED values, the
        margin, the calibration age, and the stamp the next run's age anchor
        reads. Of the run-level keys, ``omitted`` lists the resources no
        proposal reached (the ``kept-*`` rows), ``global_kept`` says
        whether the previous global is what stays in force, and when it is,
        the block keeps that global's own ``quantile``, ``healthy_fpr``, and
        ``n_calibration_windows`` rather than the discarded fit's.

    Raises:
        SpecError: If a value the band needs is neither usable in the serving
            block nor usably supplied by the seed.
        ValueError: If the capture produces no windows to bake from.
    """
    serving = _serving_block(model_path)
    weeks = weeks_since_rebake(serving, model_path)
    block, eligibility, warnings = _bake(
        model_path, calibration, profile=profile, quantile=quantile, margin=margin
    )
    kept_global, kept, verdicts = guard_rebake(
        serving,
        block["threshold"],
        block.get("per_resource") or {},
        weeks,
        eligibility=eligibility,
        seed_path=seed_path,
        allow_drift=allow_drift,
    )
    # The stamp form provenance writes and _parse_stamp reads, so the block
    # stamped here is one the next run can age.
    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    global_kept = kept_global != block["threshold"]
    report = build_rebake_report(kept_global, kept, verdicts, weeks, dry_run=dry_run)
    report.update(
        generated_at=generated_at,
        calibration=str(calibration),
        calibration_days=_json_number(calibration_days),
        margin=_json_number(margin),
        quantile=_json_number(quantile),
        allow_drift=bool(allow_drift),
        omitted=[v.resource_id for v in verdicts if v.resource_id and v.proposed is None],
        global_kept=global_kept,
        bake_warnings=warnings,
        rejected_count=sum(v.verdict in _REJECTIONS for v in verdicts),
    )
    # A kept global describes the fit that produced it; the bake's quantile,
    # fpr, and window count belong to the fit whose threshold was refused.
    provenance = (
        {key: serving[key] for key in _GLOBAL_PROVENANCE if key in serving} if global_kept else {}
    )
    return report, {
        **block,
        **provenance,
        "threshold": kept_global,
        "per_resource": kept,
        "margin_multiplier": margin,
        "calibration_days": calibration_days,
        "rebaked_at": generated_at,
    }
