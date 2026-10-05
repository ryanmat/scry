# Description: Tests for scry.eval.calibration: the per-week band guard and its composition.
# Description: Pins the exponential bands, the one-day age floor, and the measured global jump.

"""Tests for the recalibration guard, on single values and over a whole bake.

The band is pinned as arithmetic a reader can check by hand: at two weeks of
age a value may grow by ``1.5**2 == 2.25`` and may shrink by at most a factor
``1.35**2 == 1.8225``, and the one-day age floor (``weeks = max(age, 1 day) /
7``) leaves a grow limit of ``1.5**(1/7) == 1.059634``, tight enough to reject
a 1.1x move that a week of age would accept.

The measured 2026-07-28 global jump ``0.193205 -> 0.417031`` at one week of age
is pinned as the regression it is: ratio 2.158 over the 1.5 band, so the
verdict is ``REJECTED-grew`` and the previous value is what the guard keeps.
Each verdict kind is pinned on one value, with both rejected kinds keeping the
old value, and ``GuardVerdict`` is pinned to the spec's fields, frozen.

The band's edges are pinned too: a ratio sitting exactly on either bound is
inside the band, both bases are parameters rather than welded-in constants, and
a non-finite proposal -- which would otherwise slip past two ``ratio``
comparisons that are False for NaN and be kept as the serving threshold -- is
rejected with the old value kept.

``check_guards`` composes that guard over a whole rebake: the global threshold
rides the same band as the per-resource map (the measured jump is rejected
through the composed path too, not only through ``guard_value``), and
``--allow-drift`` bypasses the band -- and only the band -- with stamped
verdicts.

Upstream of all of it, ``resolve_old_thresholds`` decides what ``old`` even is.
The measured 2026-07-28 vacuous first bake -- an empty serving per_resource map,
every proposal accepted as new, exit 0 -- is pinned as the spec error it should
have been, and so is every current value that cannot serve as a band's ``old``:
a NaN one turns the band off, an infinite one keeps itself in force forever, a
zero one divides by zero. Each resolves through an explicit seed, in either
accepted form, or the run stops naming ``--seed`` and the key it wanted.

The seed reader picks its form by the zip magic a torch checkpoint starts with,
so a hand-written JSON seed with a typo in it reports its own parse error rather
than being unpickled and blamed on torch.

``guard_rebake`` is the composed path: resolution, then the band, then the
provenance stamps. A verdict measured against a seeded ``old`` says so
(``accepted-seeded``), while one measured against the live serving block stays
plainly accepted -- and ``check_guards``, a pure function of (old, new, weeks),
is not what knows the difference. A resource the bake proposed nothing for
keeps its serving threshold with the reason it was omitted on its verdict: the
hygiene gate that dropped it (``kept-gate-omitted:insufficient-windows:12<50``)
or its absence from the capture altogether (``kept-absent-from-capture``).

``weeks_since_rebake`` is where the band's ``weeks`` comes from, and the
measured defect it fixes is that a swap's copy/move chain sets the serving
checkpoint's mtime to the staged write time: a file written minutes ago can
carry a serving block baked weeks ago, and a band anchored on the file widens
to match the copy rather than the bake. The stamp the block carries is the
anchor; the mtime is the fallback for a block that has none. Either way the age
floors at one day, the narrowest band a same-day rebake can claim.

``run_calibration`` is the whole of it composed against a real keeper and a
real capture: the bake proposes, the band decides, the age anchor supplies the
weeks, and what comes back is the report beside the serving block that would be
staged -- in memory, with the base checkpoint byte-identical afterwards, since
writing anything at all is the staging step's and the swap's business.

``build_rebake_report`` assembles what an operator reads afterwards, and is
pinned on the two things a report of a rejection has to survive. It is strict
JSON: the proposal a ``REJECTED-nonfinite`` verdict rejected is written as
``"nan"``, ``"inf"``, or ``"-inf"``, because ``json.dumps`` spells a bare NaN
in a form no strict reader accepts and the report is the only record that the
rejection happened. And every row names the threshold that stays in force, so
a gate-omitted resource reads as keeping its own threshold rather than as
falling back to the global one.

``stage_checkpoint`` writes the only file a rebake writes, and it is not the
one in force: a copy of the base checkpoint beside it, carrying the guarded
serving block, so the thresholds an operator is serving cannot move until a
swap promotes the copy. The input is pinned by its bytes and by its mtime --
the age anchor's fallback -- and the staged block is pinned as a replacement of
the previous one rather than a merge over it.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
from dataclasses import fields
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import bake_serving_threshold as bake_mod
import numpy as np
import pytest
from synth import PROFILE, gated_fleet_csv

from scry.eval.calibration import (
    GuardVerdict,
    build_rebake_report,
    check_guards,
    guard_rebake,
    guard_value,
    resolve_old_thresholds,
    run_calibration,
    stage_checkpoint,
    weeks_since_rebake,
)
from scry.eval.hygiene import ResourceEligibility, per_resource_eligibility
from scry.eval.rubric import SpecError
from scry.utils.config import get_config

ONE_DAY_IN_WEEKS = 1.0 / 7.0
"""The age floor of one day expressed in weeks: ``max(age, 1 day) / 7``."""

GUARD_WEEKS = 1.0
"""One week of age: grow limit 1.5, shrink floor 1 / 1.35 == 0.7407..."""

OLD_PER_RESOURCE = {"node-a": 0.20, "node-b": 0.20, "node-c": 0.20}
"""The serving per-resource map the three-resource fixtures guard against."""

NEW_PER_RESOURCE = {"node-a": 0.24, "node-b": 0.40, "node-c": 0.10}
"""Proposals at ratios 1.2 (accepted), 2.0 (over 1.5), and 0.5 (under 0.7407)."""

SEED_GLOBAL = 0.19
"""The global threshold the seed fixtures carry, distinct from every current one."""

SEED_PER_RESOURCE = {"node-a": 0.21, "node-b": 0.22}
"""The per-resource map the seed fixtures carry, distinct from every current one."""

USABLE_A = {"node-a": 0.20}
"""A current per-resource map the band can measure against, needing no seed."""


def _write_seed(
    tmp_path: Path,
    form: str,
    *,
    global_threshold: float = SEED_GLOBAL,
    per_resource: dict[str, float] | None = None,
) -> str:
    """Write a seed in one of the two accepted forms and return its path.

    ``"checkpoint"`` is a checkpoint whose serving block carries ``threshold``
    and a non-empty ``per_resource`` map -- the same block ``ServingBlock``
    reads; ``"json"`` is the ``{"global", "per_resource"}`` map.
    """
    if per_resource is None:
        per_resource = SEED_PER_RESOURCE
    if form == "json":
        path = tmp_path / "seed.json"
        path.write_text(json.dumps({"global": global_threshold, "per_resource": per_resource}))
        return str(path)

    import torch  # local: the test module, like calibration.py, stays torch-free to import

    path = tmp_path / "seed.pt"
    torch.save({"serving": {"threshold": global_threshold, "per_resource": per_resource}}, path)
    return str(path)


def _gated_eligibility() -> dict[str, ResourceEligibility]:
    """A real hygiene verdict map for the capture behind the omission tests.

    node-a passes every gate and is baked; node-b has 12 windows against the
    50-window floor; node-c's windows are all NaN, so its own quantile is NaN
    and the non-finite gate drops it. node-z is not in the capture at all and
    so is absent from this map. The map comes from the gates themselves rather
    than hand-built verdicts: the reason strings the rebake report publishes
    are hygiene's, and inventing them here would not notice a format change.
    """
    ids = np.array(["node-a"] * 60 + ["node-b"] * 12 + ["node-c"] * 60)
    errors = np.concatenate([np.full(60, 0.05), np.full(12, 0.05), np.full(60, np.nan)])
    return per_resource_eligibility(
        trained_features=("cpu",),
        features_by_resource={rid: {"cpu"} for rid in ("node-a", "node-b", "node-c")},
        resource_ids=ids,
        errors=errors,
        quantile=0.99,
    )


def _torch_that_must_not_load() -> ModuleType:
    """A stand-in ``torch`` whose ``load`` fails the test if the seed reader calls it."""
    module = ModuleType("torch")

    def _load(*args: object, **kwargs: object) -> object:
        raise AssertionError("torch.load called for a seed that is not a checkpoint")

    module.load = _load  # type: ignore[attr-defined]
    return module


AGE_TOLERANCE_WEEKS = 1e-4
"""About a minute of slack on an asserted age (1e-4 week is 60.5 s): no test turns on less."""


def _aged_file(tmp_path: Path, mtime: datetime) -> str:
    """A stand-in serving checkpoint with ``mtime`` as its modification time.

    The age function stats the path and never reads it, so the bytes are
    irrelevant and a real checkpoint would only pin that it does not load one.
    """
    path = tmp_path / "serving.pt"
    path.write_bytes(b"")
    os.utime(path, (mtime.timestamp(), mtime.timestamp()))
    return str(path)


def _stamp(moment: datetime, suffix: str = "Z") -> str:
    """``moment`` as a serving block records it: ISO-8601 UTC, Z suffix by default."""
    return moment.isoformat().replace("+00:00", suffix)


STALE_WEEKS = 26.0
"""Age of the serving block the run test carries.

Half a year of it puts the band (grow ``1.5**26``, shrink ``1/1.35**26``)
around any threshold the tiny keeper bakes, so the composed run turns on what
it composes rather than on arithmetic ``TestBandMath`` already pins.
"""

BASE_GLOBAL = 0.20
"""The global threshold the base checkpoint of the run test is serving."""

TINY_THRESHOLD = 1e-6
"""A served threshold orders of magnitude under any the tiny keeper bakes.

Usable (finite, positive), so nothing is seeded for it, and so far under a
real proposal that no band the age floor allows -- grow ``1.5 ** (1 / 7)`` --
can accept the proposal: the rejected direction of the run, on demand.
"""

BASE_PROVENANCE = {"quantile": 0.5, "healthy_fpr": 0.25, "n_calibration_windows": 7}
"""The fit the base checkpoint's global came from, as its serving block records it.

Values no real bake of the fixture produces (the bake takes 0.99 and sees
dozens of windows), so a block that carries them took them from the serving
block, and one that does not took them from the bake.
"""


def _base_checkpoint(
    keeper_path: str,
    tmp_path: Path,
    per_resource: dict[str, float],
    *,
    threshold: float = BASE_GLOBAL,
    age: timedelta = timedelta(weeks=STALE_WEEKS),
) -> str:
    """The keeper plus an ``age``-old serving block: what a rebake reads."""
    import torch  # local: the test module, like calibration.py, stays torch-free to import

    checkpoint = torch.load(keeper_path, map_location="cpu", weights_only=False)
    checkpoint["serving"] = {
        "threshold": threshold,
        **BASE_PROVENANCE,
        "per_resource": per_resource,
        "rebaked_at": _stamp(datetime.now(timezone.utc) - age),
    }
    path = tmp_path / "serving.pt"
    torch.save(checkpoint, path)
    return str(path)


def _digest(path: str | Path) -> str:
    """The file's bytes as one hash: what "the input is untouched" is measured by."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _checkpoint(path: str | Path) -> dict[str, Any]:
    """A saved checkpoint as it comes back: weights, config, schema, serving block."""
    import torch  # local: the test module, like calibration.py, stays torch-free to import

    return torch.load(path, map_location="cpu", weights_only=False)


def _same(left: object, right: object) -> bool:
    """Deep equality over a checkpoint's mixed contents: tensors, arrays, plain data.

    A checkpoint holds model weights beside plain dicts, and neither ``==`` nor
    ``torch.equal`` alone compares the whole of it: ``==`` on two tensors is a
    tensor of elementwise comparisons, which is not a truth value.
    """
    import torch  # local: the test module, like calibration.py, stays torch-free to import

    arrays = (torch.Tensor, np.ndarray)
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(_same(left[key], right[key]) for key in left)
    if isinstance(left, arrays) or isinstance(right, arrays):
        return np.array_equal(np.asarray(left), np.asarray(right))
    return bool(left == right)


STAGED_BLOCK = {
    "threshold": 0.31,
    "quantile": 0.99,
    "healthy_fpr": 0.01,
    "n_calibration_windows": 58,
    "recon_metric": "numerical_mse_from_mu",
    "per_resource": {"node-a": 0.24},
    "margin_multiplier": 2.0,
    "calibration_days": 7.0,
    "rebaked_at": "2026-10-05T12:00:00Z",
}
"""A guarded block as ``run_calibration`` returns one, for the staging tests.

The bake's own fields plus the four the rebake patches on (``per_resource``,
``margin_multiplier``, ``calibration_days``, ``rebaked_at``). Every value of it
differs from the base checkpoint's serving block (``BASE_GLOBAL`` and
``BASE_PROVENANCE``), so a staged file carrying the base's block, or merging it
over this one, differs key by key rather than coincidentally matching.
"""


# One case per unusable kind: the current serving block, and the key it leaves
# with no usable previous value. The other key of each case is usable.
UNUSABLE_CURRENT_VALUES = [
    pytest.param({"threshold": math.nan, "per_resource": USABLE_A}, "global", id="nan-global"),
    pytest.param({"per_resource": USABLE_A}, "global", id="absent-global"),
    pytest.param({"threshold": 0.30, "per_resource": {"node-a": math.inf}}, "node-a", id="inf"),
    pytest.param({"threshold": 0.30, "per_resource": {"node-a": 0.0}}, "node-a", id="zero"),
    pytest.param({"threshold": 0.30, "per_resource": {"node-a": -0.2}}, "node-a", id="negative"),
    pytest.param({"threshold": 0.30}, "node-a", id="new-to-the-bake"),
]


class TestBandMath:
    def test_bands_scale_exponentially_with_weeks(self) -> None:
        # Two weeks of age: the grow limit is 1.5**2 == 2.25 and the shrink
        # floor is 1 / 1.35**2 == 1 / 1.8225 (downward is the dangerous
        # direction, so its base is the tighter one). 2.0x and 0.9x sit inside
        # the band; 2.3x is over it and 0.5x is under it.
        kept_grown, grown = guard_value("node-a", 1.0, 2.0, 2.0)
        _, over = guard_value("node-a", 1.0, 2.3, 2.0)
        kept_shrunk, shrunk = guard_value("node-a", 1.0, 0.9, 2.0)
        _, under = guard_value("node-a", 1.0, 0.5, 2.0)

        assert kept_grown == 2.0
        assert grown.verdict == "accepted"
        assert grown.limit == pytest.approx(2.25)
        assert over.verdict == "REJECTED-grew"
        assert over.limit == pytest.approx(1.5**2)

        assert kept_shrunk == 0.9
        assert shrunk.verdict == "accepted"
        assert 1.0 / shrunk.limit == pytest.approx(1.8225)
        assert under.verdict == "REJECTED-shrank"
        assert 1.0 / under.limit == pytest.approx(1.35**2)

        # The two bases are parameters, not constants welded into the band.
        _, widened = guard_value("node-a", 1.0, 3.5, 2.0, max_weekly_growth=2.0)
        assert widened.verdict == "accepted"
        assert widened.limit == pytest.approx(4.0)

    def test_one_day_age_floor_leaves_a_seventh_of_a_week(self) -> None:
        # The age anchor floors at one day, so the widest band a same-day
        # rebake can claim is weeks = 1/7 -> grow limit 1.5**(1/7) == 1.059634.
        # A 1.05x move fits inside it; 1.1x -- which a full week would accept --
        # does not.
        _, inside = guard_value(None, 0.2, 0.21, ONE_DAY_IN_WEEKS)
        _, outside = guard_value(None, 0.2, 0.22, ONE_DAY_IN_WEEKS)

        assert round(inside.limit, 6) == 1.059634
        assert inside.verdict == "accepted"
        assert round(outside.limit, 6) == 1.059634
        assert outside.verdict == "REJECTED-grew"
        assert guard_value(None, 0.2, 0.22, 1.0)[1].verdict == "accepted"


class TestBandEdges:
    def test_ratio_exactly_on_either_bound_is_accepted(self) -> None:
        # The band is closed at both ends: the documented maximum move is by
        # definition still within the documented band. At one week of age the
        # bounds are exactly 1.5 and exactly 1 / 1.35, so these two proposals
        # land on them with no floating-point slack, and both are kept.
        kept_grown, on_grow_limit = guard_value("node-a", 1.0, 1.5, 1.0)
        kept_shrunk, on_shrink_floor = guard_value("node-a", 1.0, 1.0 / 1.35, 1.0)

        assert on_grow_limit.ratio == 1.5 == on_grow_limit.limit
        assert on_grow_limit.verdict == "accepted"
        assert kept_grown == 1.5

        assert on_shrink_floor.ratio == 1.0 / 1.35 == on_shrink_floor.limit
        assert on_shrink_floor.verdict == "accepted"
        assert kept_shrunk == 1.0 / 1.35

    def test_shrink_base_is_a_parameter(self) -> None:
        # Mirror of the growth-base case above: a caller widening the shrink
        # base to 2.0 at one week of age gets a floor of 0.5, so a 0.6x
        # proposal is inside the band -- while the default 1.35 base, whose
        # floor is 1 / 1.35 == 0.7407..., rejects the same proposal.
        kept, widened = guard_value("node-a", 1.0, 0.6, 1.0, max_weekly_shrink=2.0)
        kept_default, default = guard_value("node-a", 1.0, 0.6, 1.0)

        assert widened.verdict == "accepted"
        assert widened.limit == pytest.approx(0.5)
        assert kept == 0.6

        assert default.verdict == "REJECTED-shrank"
        assert kept_default == 1.0


class TestUnguardedGlobalPin:
    def test_measured_global_jump_is_rejected_and_previous_kept(self) -> None:
        # The 2026-07-28 measurement: a freshly baked global 0.417031 against
        # the serving 0.193205 at one week of age. It shipped wholesale because
        # only per-resource entries passed the band; through the band it is a
        # 2.158x jump over a 1.5 limit and the previous value is kept.
        kept, verdict = guard_value(None, 0.193205, 0.417031, 1.0)

        assert kept == 0.193205
        assert verdict.verdict == "REJECTED-grew"
        assert round(verdict.ratio, 3) == 2.158
        assert verdict.ratio > 1.5
        assert verdict.resource_id is None
        assert verdict.old == 0.193205
        assert verdict.proposed == 0.417031
        assert verdict.limit == pytest.approx(1.5)


class TestVerdictKinds:
    @pytest.mark.parametrize(
        ("proposed", "expected_verdict", "expected_kept"),
        [
            (1.2, "accepted", 1.2),  # inside both bands at one week
            (1.6, "REJECTED-grew", 1.0),  # over 1.5**1
            (0.7, "REJECTED-shrank", 1.0),  # under 1 / 1.35**1 == 0.7407...
        ],
    )
    def test_each_kind_on_one_value(
        self, proposed: float, expected_verdict: str, expected_kept: float
    ) -> None:
        kept, verdict = guard_value("node-a", 1.0, proposed, 1.0)

        assert verdict.verdict == expected_verdict
        assert kept == expected_kept  # a rejected verdict keeps the old value
        assert verdict.resource_id == "node-a"
        assert verdict.old == 1.0
        assert verdict.proposed == proposed
        assert verdict.ratio == pytest.approx(proposed)

    def test_verdict_declared_fields_and_frozen(self) -> None:
        assert [f.name for f in fields(GuardVerdict)] == [
            "resource_id",
            "verdict",
            "old",
            "proposed",
            "ratio",
            "limit",
        ]
        _, verdict = guard_value("node-a", 1.0, 1.2, 1.0)

        with pytest.raises(AttributeError):
            verdict.verdict = "accepted"  # type: ignore[misc]


class TestNonFiniteProposals:
    @pytest.mark.parametrize("proposed", [math.nan, math.inf, -math.inf])
    def test_non_finite_proposal_is_rejected_and_old_kept(self, proposed: float) -> None:
        # A non-finite proposal has no ratio to compare: every band comparison
        # against NaN is False, so an unguarded band accepts it and a NaN goes
        # on to serve as the threshold. It is a rejection, and the old value is
        # what stays in force.
        kept, verdict = guard_value("node-a", 0.2, proposed, 1.0)

        assert kept == 0.2
        assert verdict.verdict == "REJECTED-nonfinite"
        assert verdict.resource_id == "node-a"
        assert verdict.old == 0.2
        assert verdict.proposed == pytest.approx(proposed, nan_ok=True)
        assert verdict.ratio is None  # no band comparison was made
        assert verdict.limit is None


class TestCheckGuards:
    def test_each_verdict_kind_over_the_per_resource_map(self) -> None:
        # One rebake, three resources, one verdict kind each at one week of
        # age. What the band accepts is what serves; what it rejects leaves the
        # previous value in force, so a bad bake holds rather than ships.
        kept_global, guarded, verdicts = check_guards(
            0.30, OLD_PER_RESOURCE, 0.33, NEW_PER_RESOURCE, GUARD_WEEKS
        )

        by_resource = {verdict.resource_id: verdict for verdict in verdicts}
        assert by_resource["node-a"].verdict == "accepted"
        assert by_resource["node-b"].verdict == "REJECTED-grew"
        assert by_resource["node-c"].verdict == "REJECTED-shrank"
        assert guarded == {"node-a": 0.24, "node-b": 0.20, "node-c": 0.20}

        # The global is one more guarded value: 0.33 / 0.30 == 1.1 is inside
        # the band, so the fresh global is the one that serves.
        assert kept_global == 0.33
        assert by_resource[None].verdict == "accepted"
        assert by_resource[None].ratio == pytest.approx(1.1)
        assert len(verdicts) == 4  # the global plus one per resource

    def test_global_rides_as_resource_id_none_and_is_guarded(self) -> None:
        # Requirement 2, composed: the 2026-07-28 global jump 0.193205 ->
        # 0.417031 shipped wholesale because only per-resource entries met the
        # band. Through check_guards at the same one week of age the ratio is
        # 2.158 against a 1.5 limit, so the returned global is the previous one
        # and the rejection is in the verdict list under resource_id None.
        kept_global, guarded, verdicts = check_guards(
            0.193205, {"node-a": 0.20}, 0.417031, {"node-a": 0.22}, GUARD_WEEKS
        )

        assert kept_global == 0.193205
        global_verdicts = [verdict for verdict in verdicts if verdict.resource_id is None]
        assert len(global_verdicts) == 1
        (global_verdict,) = global_verdicts
        assert global_verdict.verdict == "REJECTED-grew"
        assert global_verdict.old == 0.193205
        assert global_verdict.proposed == 0.417031
        assert round(global_verdict.ratio, 3) == 2.158
        assert global_verdict.limit == pytest.approx(1.5)

        # A held global does not hold back an in-band per-resource entry.
        assert guarded == {"node-a": 0.22}

    def test_allow_drift_stamps_the_values_that_left_the_band(self) -> None:
        # --allow-drift keeps what the band rejected, stamped so the report
        # says the value drifted rather than fit. The global is bypassed on the
        # same terms (0.60 / 0.30 == 2.0, over the 1.5 limit).
        kept_global, guarded, verdicts = check_guards(
            0.30, OLD_PER_RESOURCE, 0.60, NEW_PER_RESOURCE, GUARD_WEEKS, allow_drift=True
        )

        by_resource = {verdict.resource_id: verdict for verdict in verdicts}
        assert by_resource[None].verdict == "accepted-allow-drift"
        assert by_resource["node-b"].verdict == "accepted-allow-drift"
        assert by_resource["node-c"].verdict == "accepted-allow-drift"
        assert kept_global == 0.60
        assert guarded == {"node-a": 0.24, "node-b": 0.40, "node-c": 0.10}

        # A proposal that never needed the bypass is still plainly accepted,
        # and the comparison the bypass overrode is still on the verdict.
        assert by_resource["node-a"].verdict == "accepted"
        assert by_resource["node-b"].ratio == pytest.approx(2.0)
        assert by_resource["node-b"].limit == pytest.approx(1.5)

    def test_allow_drift_does_not_invent_a_missing_previous_value(self) -> None:
        # The bypass is of the band, never of the seed requirement: a resource
        # with no previous value has no band to bypass, and accepting it here
        # is the unguarded first bake this module exists to prevent. The typed
        # spec error for an empty map belongs to the seed resolution upstream.
        with pytest.raises(KeyError, match="node-c"):
            check_guards(
                0.30,
                {"node-a": 0.20, "node-b": 0.20},
                0.33,
                NEW_PER_RESOURCE,
                GUARD_WEEKS,
                allow_drift=True,
            )

    def test_allow_drift_does_not_bypass_the_non_finite_rejection(self) -> None:
        # A non-finite proposal -- a NaN global, an infinite resource here --
        # is not a drifted threshold, it is not a threshold at all: no ratio
        # was formed, so there is no band decision to override. Serving either
        # one silences the detector, which --allow-drift must not be able to
        # ask for.
        kept_global, guarded, verdicts = check_guards(
            0.30,
            {"node-a": 0.20},
            math.nan,
            {"node-a": math.inf},
            GUARD_WEEKS,
            allow_drift=True,
        )

        by_resource = {verdict.resource_id: verdict for verdict in verdicts}
        assert by_resource[None].verdict == "REJECTED-nonfinite"
        assert by_resource["node-a"].verdict == "REJECTED-nonfinite"
        assert kept_global == 0.30
        assert guarded == {"node-a": 0.20}
        # The composition records the proposal verbatim, NaN included, exactly
        # as guard_value does: a report writer still owns the JSON encoding.
        assert math.isnan(by_resource[None].proposed)


class TestSeedResolution:
    def test_vacuous_first_bake_without_a_seed_is_a_spec_error(self) -> None:
        # The measured 2026-07-28 failure: with an empty serving per_resource
        # map every proposal was accepted as new and the run exited 0, shipping
        # an excursion-baked 0.3766 for lp7tj and a pinned-state 1.0124 for
        # master-2. An unguarded first bake is an error, not a default: the
        # resolution stops here, so nothing downstream runs.
        with pytest.raises(SpecError) as excinfo:
            resolve_old_thresholds({"threshold": 0.30, "per_resource": {}}, ["node-a", "node-b"])

        assert "--seed" in str(excinfo.value)
        assert "node-a" in str(excinfo.value)

    @pytest.mark.parametrize("form", ["checkpoint", "json"])
    def test_a_seed_in_either_form_resolves_the_old_inputs(self, tmp_path: Path, form: str) -> None:
        # The two accepted seed forms carry the same two things -- a global
        # threshold and a non-empty per-resource map -- and resolve identically
        # into the (old_global, old) that check_guards takes.
        old_global, old = resolve_old_thresholds(
            {"per_resource": {}},
            ["node-a", "node-b"],
            seed_path=_write_seed(tmp_path, form),
        )

        assert old_global == SEED_GLOBAL
        assert old == SEED_PER_RESOURCE

    @pytest.mark.parametrize(("serving", "key"), UNUSABLE_CURRENT_VALUES)
    def test_an_unusable_current_value_needs_the_seed_or_stops_the_run(
        self, tmp_path: Path, serving: dict, key: str
    ) -> None:
        # guard_value finiteness-checks the proposal only, so a NaN old turns
        # the band off, an infinite old keeps itself in force forever and a zero
        # old divides by zero; a negative one is no threshold at all, and a
        # resource new to the bake has nothing to measure against. Each counts
        # as missing: the seed's value becomes the old, while the key that was
        # usable keeps its current value -- the seed is a fallback, not an
        # override. With no seed the run stops on the typed spec error, naming
        # the flag that fixes it and the key it wanted, so an operator can act
        # on the message without reading the serving block. The key has to be
        # named AS the key: `key in message` passes on any message carrying the
        # word "global", which the fixed text of every one of these does, so a
        # renamed key would go unnoticed. Matching "threshold for <key>:" is the
        # position in the sentence where the key is reported.
        seeded_a = {"node-a": SEED_PER_RESOURCE["node-a"]}  # node-b is seeded but unneeded
        expected = (SEED_GLOBAL, USABLE_A) if key == "global" else (0.30, seeded_a)

        seeded = resolve_old_thresholds(
            serving, ["node-a"], seed_path=_write_seed(tmp_path, "json")
        )
        with pytest.raises(SpecError, match=rf"threshold for {re.escape(key)}:") as excinfo:
            resolve_old_thresholds(serving, ["node-a"])

        assert seeded == expected
        assert "--seed" in str(excinfo.value)

    def test_a_seed_without_the_key_is_the_same_spec_error(self, tmp_path: Path) -> None:
        # A seed is not a free pass over the requirement: it has to carry the
        # key that is missing. One that covers node-b does nothing for node-a.
        seed_path = _write_seed(tmp_path, "json", per_resource={"node-b": 0.22})

        with pytest.raises(SpecError) as excinfo:
            resolve_old_thresholds(
                {"threshold": 0.30, "per_resource": {}}, ["node-a"], seed_path=seed_path
            )

        assert "--seed" in str(excinfo.value)
        assert "node-a" in str(excinfo.value)

    def test_an_unusable_seed_value_is_the_same_spec_error(self, tmp_path: Path) -> None:
        # What is resolved is a usable value, not a present key: a seed global
        # of 0.0 would divide by zero in the band exactly as a serving 0.0
        # would, so it is refused on the same terms.
        seed_path = _write_seed(tmp_path, "json", global_threshold=0.0)

        with pytest.raises(SpecError, match=r"threshold for global:") as excinfo:
            resolve_old_thresholds(
                {"per_resource": {"node-a": 0.20}}, ["node-a"], seed_path=seed_path
            )

        assert "--seed" in str(excinfo.value)

    def test_a_malformed_json_seed_reports_the_json_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A hand-written seed with a trailing comma used to be misreported as a
        # checkpoint failure: the JSONDecodeError was swallowed and torch.load
        # unpickled the text, so the operator saw `UnpicklingError: invalid load
        # key, '{'` pointing at torch with no mention of the syntax error. The
        # form is chosen by the zip magic a checkpoint starts with, so a text
        # seed is only ever a JSON seed: it reports its own parse error, naming
        # the seed path and carrying the decode error's message and position,
        # with that error chained as the cause. torch.load is never reached --
        # the stand-in below fails the test if it is called.
        seed_path = tmp_path / "seed.json"
        seed_path.write_text('{"global": 0.19, "per_resource": {"node-a": 0.21},}')
        monkeypatch.setitem(sys.modules, "torch", _torch_that_must_not_load())

        with pytest.raises(SpecError) as excinfo:
            resolve_old_thresholds({"per_resource": {}}, ["node-a"], seed_path=str(seed_path))

        message = str(excinfo.value)
        cause = excinfo.value.__cause__
        assert str(seed_path) in message
        assert isinstance(cause, json.JSONDecodeError)
        # The interpreter's own wording is what the message embeds, whichever version prints it:
        # 3.12 says "Expecting property name enclosed in double quotes", 3.13 and later "Illegal
        # trailing comma before end of object", so the assertion reads the chained error itself.
        assert str(cause) in message
        assert "line 1 column" in message  # the decode error's position survives

    def test_a_seed_that_is_neither_zip_nor_text_reports_the_decode_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # torch's pre-zip serialization starts with the pickle protocol header, not the zip
        # magic, and is not UTF-8 text either, so it goes down the JSON branch and the decode
        # failure is the typed spec error with the UnicodeDecodeError chained; torch.load is
        # never reached for it. Every scry writer saves the zip format, so this seed shape is a
        # mistaken --seed, and the message says which form is expected.
        seed_path = tmp_path / "seed.pkl"
        seed_path.write_bytes(b"\x80\x02}q.")
        monkeypatch.setitem(sys.modules, "torch", _torch_that_must_not_load())

        with pytest.raises(SpecError) as excinfo:
            resolve_old_thresholds({"per_resource": {}}, ["node-a"], seed_path=str(seed_path))

        assert isinstance(excinfo.value.__cause__, UnicodeDecodeError)
        assert str(seed_path) in str(excinfo.value)

    def test_serving_resources_the_bake_did_not_propose_are_carried(self) -> None:
        # check_guards takes the whole previous map, not only the keys the bake
        # proposed: node-z, dropped by this capture, is an omission for the
        # report to record, and dropping it here would hide it. Nothing needs a
        # seed when every current value is usable -- and node-y's unusable 0.0
        # is carried as it stands, since no band will measure against a
        # resource nothing was proposed for, while dropping it would drop the
        # one record that it is still serving that value.
        old_global, old = resolve_old_thresholds(
            {"threshold": 0.30, "per_resource": {"node-a": 0.20, "node-z": 0.25, "node-y": 0.0}},
            ["node-a"],
        )

        assert old_global == 0.30
        assert old == {"node-a": 0.20, "node-z": 0.25, "node-y": 0.0}


class TestComposedGuardPath:
    def test_the_seeded_old_values_are_what_the_band_measures(self, tmp_path: Path) -> None:
        # Requirement 1's mechanism, composed: with an empty serving
        # per_resource map the seed carries the only previous values there are,
        # and they are what the band measures against -- node-a's 1.05x move
        # against the seeded 0.21 is inside the band, node-b's 2.0x move
        # against the seeded 0.22 is not, and the seeded value is what stays in
        # force for it. A composition that resolved the seed and then guarded
        # something else would report another old; one that never resolved it
        # would stop on the spec error instead.
        kept_global, kept, verdicts = guard_rebake(
            {"per_resource": {}},
            SEED_GLOBAL * 1.1,
            {
                "node-a": SEED_PER_RESOURCE["node-a"] * 1.05,
                "node-b": SEED_PER_RESOURCE["node-b"] * 2.0,
            },
            GUARD_WEEKS,
            seed_path=_write_seed(tmp_path, "checkpoint"),
        )

        by_resource = {verdict.resource_id: verdict for verdict in verdicts}
        assert by_resource[None].old == SEED_GLOBAL
        assert by_resource["node-a"].old == SEED_PER_RESOURCE["node-a"]
        assert by_resource["node-b"].old == SEED_PER_RESOURCE["node-b"]
        assert by_resource["node-b"].verdict == "REJECTED-grew"
        assert kept["node-b"] == SEED_PER_RESOURCE["node-b"]
        assert kept["node-a"] == pytest.approx(SEED_PER_RESOURCE["node-a"] * 1.05)
        assert kept_global == pytest.approx(SEED_GLOBAL * 1.1)

    def test_only_the_verdicts_whose_old_came_from_the_seed_are_stamped(
        self, tmp_path: Path
    ) -> None:
        # The stamp records provenance, not a decision: node-b had no usable
        # serving value and was measured against the seed, so its verdict says
        # so, while the global and node-a were measured against the live
        # serving block and stay plainly accepted. An operator reading
        # `accepted-seeded` knows that threshold was never in production.
        # check_guards is not what stamps it: given the same resolved inputs it
        # returns the plain verdict, because it is a pure function of (old,
        # new, weeks) and knows nothing about where old came from.
        serving = {"threshold": 0.30, "per_resource": {"node-a": 0.20}}
        new = {"node-a": 0.22, "node-b": 0.23}

        _, _, verdicts = guard_rebake(
            serving, 0.33, new, GUARD_WEEKS, seed_path=_write_seed(tmp_path, "json")
        )
        _, _, unstamped = check_guards(
            0.30, {"node-a": 0.20, "node-b": SEED_PER_RESOURCE["node-b"]}, 0.33, new, GUARD_WEEKS
        )

        by_resource = {verdict.resource_id: verdict for verdict in verdicts}
        assert by_resource["node-b"].verdict == "accepted-seeded"
        assert by_resource["node-b"].old == SEED_PER_RESOURCE["node-b"]
        assert by_resource[None].verdict == "accepted"
        assert by_resource["node-a"].verdict == "accepted"
        assert [verdict.verdict for verdict in unstamped] == ["accepted"] * 3

    def test_omitted_resources_keep_their_threshold_and_say_why(self) -> None:
        # Requirement 4: the old report could not tell a hygiene-gate omission
        # from a resource the capture never contained, so an operator could not
        # tell a fixable gate failure from a decommissioned node. The bake's
        # eligibility map is what distinguishes them, and each verdict carries
        # the gate's own reason. Both kinds KEEP the serving threshold: dropping
        # it from the map would silently fall the resource back to the global,
        # a live threshold change nobody asked for and no verdict would show.
        serving = {
            "threshold": 0.30,
            "per_resource": {"node-z": 0.25, "node-c": 0.22, "node-a": 0.20, "node-b": 0.21},
        }

        kept_global, kept, verdicts = guard_rebake(
            serving, 0.33, {"node-a": 0.22}, GUARD_WEEKS, eligibility=_gated_eligibility()
        )

        by_resource = {verdict.resource_id: verdict for verdict in verdicts}
        assert by_resource["node-b"].verdict == "kept-gate-omitted:insufficient-windows:12<50"
        assert by_resource["node-c"].verdict == "kept-gate-omitted:non-finite-quantile:nan"
        assert by_resource["node-z"].verdict == "kept-absent-from-capture"
        assert kept == {"node-a": 0.22, "node-b": 0.21, "node-c": 0.22, "node-z": 0.25}
        assert kept_global == 0.33

        omitted = by_resource["node-b"]
        assert omitted.old == 0.21  # the value that stays in force
        assert omitted.proposed is None  # nothing was proposed, so there is no band
        assert omitted.ratio is None
        assert omitted.limit is None

        # The report is a deterministic artifact: the global leads, then every
        # resource in sorted order whether it was guarded or omitted (the
        # serving map above is deliberately in another order).
        assert [verdict.resource_id for verdict in verdicts] == [
            None,
            "node-a",
            "node-b",
            "node-c",
            "node-z",
        ]

    def test_an_eligible_resource_with_no_proposal_is_not_blamed_on_a_gate(self) -> None:
        # The bake bakes a threshold for every resource its gates pass, so a
        # resource the eligibility map calls eligible should have a proposal.
        # Where the two disagree there is no gate reason to name, and the
        # verdict falls back to the absence one rather than publishing an
        # empty reason or failing on one. node-a is eligible in the map below
        # and this bake proposed nothing at all.
        _, kept, verdicts = guard_rebake(
            {"threshold": 0.30, "per_resource": {"node-a": 0.20}},
            0.33,
            {},
            GUARD_WEEKS,
            eligibility=_gated_eligibility(),
        )

        by_resource = {verdict.resource_id: verdict for verdict in verdicts}
        assert _gated_eligibility()["node-a"].eligible is True
        assert by_resource["node-a"].verdict == "kept-absent-from-capture"
        assert kept == {"node-a": 0.20}

    def test_an_omission_with_no_eligibility_map_is_absent_from_capture(self) -> None:
        # Without the bake's map there is nothing to attribute an omission to,
        # and claiming a gate failed would be an invention. node-b keeps its
        # threshold either way.
        _, kept, verdicts = guard_rebake(
            {"threshold": 0.30, "per_resource": {"node-a": 0.20, "node-b": 0.21}},
            0.33,
            {"node-a": 0.22},
            GUARD_WEEKS,
        )

        by_resource = {verdict.resource_id: verdict for verdict in verdicts}
        assert by_resource["node-b"].verdict == "kept-absent-from-capture"
        assert kept == {"node-a": 0.22, "node-b": 0.21}

    @pytest.mark.parametrize(
        ("value", "written", "kind"),
        [
            (math.nan, "nan", "nan"),
            (math.inf, "inf", "inf"),
            (-math.inf, "-inf", "-inf"),
            (0.0, 0.0, "zero"),
            (-0.2, -0.2, "negative"),
            (None, None, "null"),
        ],
        ids=["nan", "inf", "-inf", "zero", "negative", "null"],
    )
    def test_an_unusable_serving_value_with_no_proposal_falls_to_the_global_on_the_record(
        self, value: float | None, written: float | str | None, kind: str
    ) -> None:
        # The last silent omission under requirement 4: a serving entry whose
        # value the predictor refuses -- and that this bake proposed nothing
        # for -- was dropped during resolution, before the guard could see it,
        # so the report carried no row for it while the resource served the
        # global. The predictor still serves the global for it (it drops the
        # entry, as the parity assertion below pins), so the map that stays in
        # force leaves it out the same way, and what the report adds is the
        # record: a row naming the unusable value, why it is unusable, and the
        # global as what serves. A row that said the resource kept NaN would
        # be the misreading `kept` exists to prevent.
        from scry.api.predictor import Predictor  # local: pulls in torch

        serving = {"threshold": 0.30, "per_resource": {"node-a": 0.20, "node-z": value}}
        kept_global, kept, verdicts = guard_rebake(serving, 0.33, {"node-a": 0.22}, GUARD_WEEKS)
        report = build_rebake_report(kept_global, kept, verdicts, GUARD_WEEKS, dry_run=True)

        assert kept == {"node-a": 0.22}  # node-z is not staged, as the predictor would not serve it
        assert set(kept) == set(
            Predictor._resolve_per_resource_thresholds({"per_resource": {**kept, "node-z": value}})
        )
        row = report["per_resource"]["node-z"]
        assert row["verdict"] == f"kept-global-unusable:{kind}"
        assert row["old"] == written
        assert row["kept"] == report["global"]["kept"] == 0.33  # the global is what serves
        assert report["per_resource"]["node-a"]["verdict"] == "accepted"  # the guarded one stands
        assert json.loads(json.dumps(report, allow_nan=False)) == report

    def test_a_seeded_global_and_an_unusable_serving_value_are_stamped(
        self, tmp_path: Path
    ) -> None:
        # The stamp reads the same usability rule the resolution applied, not
        # mere presence: a serving global of NaN and a serving node-b of zero are
        # present but unusable, so both were measured against the seed and say
        # so. The global is requirement 1's own case -- the first bake whose
        # previous global was never in production -- and must not read as a
        # plainly accepted threshold. node-a had a usable serving value.
        serving = {"threshold": float("nan"), "per_resource": {"node-a": 0.20, "node-b": 0.0}}

        _, _, verdicts = guard_rebake(
            serving,
            SEED_GLOBAL * 1.05,
            {"node-a": 0.22, "node-b": SEED_PER_RESOURCE["node-b"] * 1.05},
            GUARD_WEEKS,
            seed_path=_write_seed(tmp_path, "json"),
        )

        by_resource = {verdict.resource_id: verdict for verdict in verdicts}
        assert by_resource[None].verdict == "accepted-seeded"
        assert by_resource[None].old == SEED_GLOBAL
        assert by_resource["node-b"].verdict == "accepted-seeded"
        assert by_resource["node-b"].old == SEED_PER_RESOURCE["node-b"]
        assert by_resource["node-a"].verdict == "accepted"

    def test_allow_drift_reaches_the_guard_and_keeps_its_own_stamp(self, tmp_path: Path) -> None:
        # --allow-drift passes through the composed path to the band, and the
        # seed stamp does not overwrite the decision it records: node-b's 2.0x
        # move against its seeded 0.22 is over the band, kept anyway, and stamped
        # accepted-allow-drift; node-a inside the band is accepted-seeded.
        _, kept, verdicts = guard_rebake(
            {"threshold": 0.30, "per_resource": {}},
            0.33,
            {
                "node-a": SEED_PER_RESOURCE["node-a"] * 1.05,
                "node-b": SEED_PER_RESOURCE["node-b"] * 2.0,
            },
            GUARD_WEEKS,
            seed_path=_write_seed(tmp_path, "json"),
            allow_drift=True,
        )

        by_resource = {verdict.resource_id: verdict for verdict in verdicts}
        assert by_resource["node-b"].verdict == "accepted-allow-drift"
        assert kept["node-b"] == pytest.approx(SEED_PER_RESOURCE["node-b"] * 2.0)
        assert by_resource["node-a"].verdict == "accepted-seeded"
        assert by_resource[None].verdict == "accepted"

    def test_the_band_bases_reach_the_guard(self) -> None:
        # A 1.4x grow and a 1/1.3 shrink are inside the default one-week band
        # (1.5 and 1/1.35) and outside a band of 1.3 and 1/1.2; the composed
        # path guards with the bases it was given.
        _, _, verdicts = guard_rebake(
            {"threshold": 0.30, "per_resource": {"node-a": 0.20}},
            0.30 / 1.3,
            {"node-a": 0.20 * 1.4},
            GUARD_WEEKS,
            max_weekly_growth=1.3,
            max_weekly_shrink=1.2,
        )

        by_resource = {verdict.resource_id: verdict for verdict in verdicts}
        assert by_resource[None].verdict == "REJECTED-shrank"
        assert by_resource["node-a"].verdict == "REJECTED-grew"

    def test_an_omission_names_the_first_gate_that_failed(self) -> None:
        # node-b has 12 windows of zero error, so it fails the window floor and
        # the non-positive gate both; the verdict names the first in gate order,
        # the convention the bake's own warning follows.
        ids = np.array(["node-a"] * 60 + ["node-b"] * 12)
        errors = np.concatenate([np.full(60, 0.05), np.zeros(12)])
        eligibility = per_resource_eligibility(
            trained_features=("cpu",),
            features_by_resource={rid: {"cpu"} for rid in ("node-a", "node-b")},
            resource_ids=ids,
            errors=errors,
            quantile=0.99,
        )

        _, _, verdicts = guard_rebake(
            {"threshold": 0.30, "per_resource": {"node-a": 0.20, "node-b": 0.21}},
            0.33,
            {"node-a": 0.22},
            GUARD_WEEKS,
            eligibility=eligibility,
        )

        assert eligibility["node-b"].reasons == [
            "insufficient-windows:12<50",
            "non-positive-quantile:0.0",
        ]
        by_resource = {verdict.resource_id: verdict for verdict in verdicts}
        assert by_resource["node-b"].verdict == "kept-gate-omitted:insufficient-windows:12<50"

    def test_guarded_and_omitted_resources_interleave_in_sorted_order(self) -> None:
        # The guarded resource sorts between two omitted ones, so a verdict list
        # that appended omissions after the guarded resources would differ.
        _, _, verdicts = guard_rebake(
            {"threshold": 0.30, "per_resource": {"node-c": 0.22, "node-a": 0.20, "node-b": 0.21}},
            0.33,
            {"node-b": 0.22},
            GUARD_WEEKS,
        )

        assert [verdict.resource_id for verdict in verdicts] == [None, "node-a", "node-b", "node-c"]


class TestAgeAnchor:
    @pytest.mark.parametrize("suffix", ["Z", "+00:00", ""], ids=["z", "offset", "naive"])
    def test_the_serving_stamp_anchors_the_age_not_the_file_mtime(
        self, tmp_path: Path, suffix: str
    ) -> None:
        # Requirement 3, the measured defect: the swap's copy/move chain sets
        # the serving checkpoint's mtime to the staged write time, so the file
        # here is brand new while the block it carries was baked three weeks
        # ago. Anchoring on the mtime would claim an age of one day (the floor)
        # and hand the band a grow limit of 1.059634 instead of 1.5**3 == 3.375
        # -- the age is read off the block, which is what records the bake.
        # The three spellings are the same instant: the repo writes the Z form
        # (scry.eval.provenance._iso_z) and a stamp with no offset at all is
        # read as UTC, as scripts/extract_features.py reads one.
        now = datetime.now(timezone.utc)
        path = _aged_file(tmp_path, now)

        weeks = weeks_since_rebake({"rebaked_at": _stamp(now - timedelta(weeks=3), suffix)}, path)

        assert weeks == pytest.approx(3.0, abs=AGE_TOLERANCE_WEEKS)

    def test_the_file_mtime_is_the_fallback_without_a_stamp(self, tmp_path: Path) -> None:
        # A serving block baked before this module stamped anything has no
        # anchor of its own, and the file's own age is the best evidence left.
        # Two weeks of mtime age is two weeks of band.
        now = datetime.now(timezone.utc)
        path = _aged_file(tmp_path, now - timedelta(weeks=2))

        weeks = weeks_since_rebake({"threshold": 0.30, "per_resource": USABLE_A}, path)

        assert weeks == pytest.approx(2.0, abs=AGE_TOLERANCE_WEEKS)

    def test_an_age_under_a_day_floors_at_one_day(self, tmp_path: Path) -> None:
        # A rebake an hour after the last one does not get an hour-wide band:
        # the age floors at one day, which is weeks = 1/7 exactly -- the band
        # TestBandMath pins, grow limit 1.5**(1/7) == 1.059634, where a 1.1x
        # move is rejected. Without the floor the hour would give weeks =
        # 1/168 and a limit of 1.0024, and the rebake would be unable to move
        # a threshold at all.
        now = datetime.now(timezone.utc)
        path = _aged_file(tmp_path, now)

        weeks = weeks_since_rebake({"rebaked_at": _stamp(now - timedelta(hours=1))}, path)

        assert weeks == ONE_DAY_IN_WEEKS
        _, verdict = guard_value(None, 0.2, 0.22, weeks)
        assert round(verdict.limit, 6) == 1.059634
        assert verdict.verdict == "REJECTED-grew"

    def test_a_stamp_in_the_future_floors_instead_of_going_negative(self, tmp_path: Path) -> None:
        # A clock skew between the host that baked the block and this one can
        # stamp it ahead of now, and the elapsed time is then negative. The
        # band is exponential in this number, so a negative age turns it inside
        # out: at weeks = -1 the grow limit 1.5**-1 == 0.667 sits BELOW the
        # shrink floor 1.35, the band holds nothing, and every threshold is
        # rejected -- a whole rebake silently held, reported as a fleet that
        # drifted. The floor the hour-ago case lands on is what keeps the skew
        # harmless, and the file below is two weeks old to show the mtime
        # fallback is not what answers here.
        now = datetime.now(timezone.utc)
        path = _aged_file(tmp_path, now - timedelta(weeks=2))

        weeks = weeks_since_rebake({"rebaked_at": _stamp(now + timedelta(weeks=1))}, path)

        assert weeks == ONE_DAY_IN_WEEKS

    def test_an_unreadable_stamp_raises_instead_of_falling_back_to_the_mtime(
        self, tmp_path: Path
    ) -> None:
        # The mtime is the fallback for a block with NO stamp, never for one
        # whose stamp cannot be read. A block that records its own age in a
        # form nothing can parse is not a block to guess the age of: guessing
        # would hand the band an age measured from the staged write time, which
        # is exactly the defect requirement 3 exists to fix, and it would do it
        # silently. The file below is two weeks old, so a fallback would return
        # a plausible 2.0 rather than fail.
        path = _aged_file(tmp_path, datetime.now(timezone.utc) - timedelta(weeks=2))

        with pytest.raises(ValueError, match="not-a-date"):
            weeks_since_rebake({"rebaked_at": "not-a-date"}, path)


class TestReportAssembly:
    def test_the_report_carries_the_whole_guarded_bake_and_round_trips(
        self, tmp_path: Path
    ) -> None:
        # One rebake with every verdict kind in it, as the report publishes it:
        # the measured global jump 0.193205 -> 0.417031 rejected, node-a inside
        # the band, node-s measured against the seed, node-b dropped by a
        # hygiene gate and node-z absent from the capture. The report is a
        # pure function of those inputs -- no clock, no file -- and strict
        # JSON: json.dumps(allow_nan=False) accepts it and reading it back
        # gives the same document, which is what makes it an artifact a later
        # run or a reviewer can diff.
        staged = tmp_path / "staged.pt"
        serving = {
            "threshold": 0.193205,
            "per_resource": {"node-a": 0.20, "node-b": 0.21, "node-z": 0.25},
        }

        report = build_rebake_report(
            *guard_rebake(
                serving,
                0.417031,
                {"node-a": 0.22, "node-s": 0.21},
                GUARD_WEEKS,
                eligibility=_gated_eligibility(),
                seed_path=_write_seed(tmp_path, "json", per_resource={"node-s": 0.20}),
            ),
            GUARD_WEEKS,
            dry_run=True,
            staged_path=staged,
        )

        assert set(report) == {"dry_run", "weeks", "staged_path", "global", "per_resource"}
        assert report["dry_run"] is True  # a report FIELD, not only an exit code
        assert report["weeks"] == GUARD_WEEKS
        assert report["staged_path"] == str(staged)

        assert set(report["global"]) == {"verdict", "old", "proposed", "kept", "ratio", "limit"}
        assert report["global"]["verdict"] == "REJECTED-grew"
        assert report["global"]["old"] == 0.193205
        assert report["global"]["proposed"] == 0.417031
        assert report["global"]["kept"] == 0.193205  # the rejected jump does not serve
        assert round(report["global"]["ratio"], 3) == 2.158
        assert report["global"]["limit"] == pytest.approx(1.5)

        assert list(report["per_resource"]) == ["node-a", "node-b", "node-s", "node-z"]
        assert {rid: row["verdict"] for rid, row in report["per_resource"].items()} == {
            "node-a": "accepted",
            "node-b": "kept-gate-omitted:insufficient-windows:12<50",
            "node-s": "accepted-seeded",
            "node-z": "kept-absent-from-capture",
        }
        assert report["per_resource"]["node-a"]["kept"] == 0.22  # accepted: the fresh value serves
        assert report["per_resource"]["node-s"]["old"] == 0.20  # the seeded previous value

        assert json.loads(json.dumps(report, allow_nan=False)) == report

    @pytest.mark.parametrize(
        ("proposed", "written"),
        [(math.nan, "nan"), (math.inf, "inf"), (-math.inf, "-inf")],
        ids=["nan", "inf", "-inf"],
    )
    def test_a_rejected_non_finite_proposal_is_written_as_its_name(
        self, proposed: float, written: str
    ) -> None:
        # The guard records a non-finite proposal verbatim, so the number the
        # report has to publish is one json.dumps writes as bare NaN,
        # Infinity, or -Infinity -- none of them JSON, and all three refused
        # outright under allow_nan=False. The report would then be unwritable
        # for exactly the bake whose rejection most needs recording, so the
        # value is written as its name instead and the document stays strict.
        report = build_rebake_report(
            *guard_rebake(
                {"threshold": 0.30, "per_resource": {"node-a": 0.20}},
                proposed,
                {"node-a": proposed},
                GUARD_WEEKS,
            ),
            GUARD_WEEKS,
            dry_run=True,
        )

        row = report["per_resource"]["node-a"]
        assert report["global"]["verdict"] == "REJECTED-nonfinite"
        assert report["global"]["proposed"] == written
        assert row["proposed"] == written
        assert row["old"] == 0.20
        assert row["kept"] == 0.20  # the finite previous value is what serves
        assert row["ratio"] is None  # no band comparison was made
        assert json.loads(json.dumps(report, allow_nan=False)) == report

    def test_a_kept_row_names_the_threshold_that_resource_keeps(self) -> None:
        # The report must not read as if an omitted resource serves the global
        # threshold: node-b was dropped by a hygiene gate and node-z was not in
        # the capture, and each one goes on serving its own previous value,
        # which is neither the proposal it never got nor the fresh global. The
        # verdict alone does not say that -- the row names the kept threshold.
        report = build_rebake_report(
            *guard_rebake(
                {
                    "threshold": 0.30,
                    "per_resource": {"node-a": 0.20, "node-b": 0.21, "node-z": 0.25},
                },
                0.33,
                {"node-a": 0.22},
                GUARD_WEEKS,
                eligibility=_gated_eligibility(),
            ),
            GUARD_WEEKS,
            dry_run=False,
        )

        gated = report["per_resource"]["node-b"]
        absent = report["per_resource"]["node-z"]
        assert gated["verdict"] == "kept-gate-omitted:insufficient-windows:12<50"
        assert gated["kept"] == 0.21 == gated["old"]
        assert gated["proposed"] is None  # nothing was proposed, so nothing was guarded
        assert absent["verdict"] == "kept-absent-from-capture"
        assert absent["kept"] == 0.25 == absent["old"]
        assert report["global"]["kept"] == 0.33  # what the fresh global moved to
        assert gated["kept"] != report["global"]["kept"]

        assert report["dry_run"] is False
        assert report["staged_path"] is None  # nothing staged, and the report says so


class TestStaging:
    def test_staging_writes_beside_the_input_and_never_touches_it(
        self, keeper_path: str, tmp_path: Path
    ) -> None:
        # The rebake's one write, and it is not the file in force: the staged
        # copy goes beside the live checkpoint, which keeps serving the
        # thresholds the report was measured against until the swap chain
        # (requirement 6, --apply) promotes the copy. The input is checked by
        # its bytes AND its mtime: a rewrite in place would both move the live
        # thresholds and, since the mtime is the age anchor's fallback, leave
        # the checkpoint looking freshly baked to the next run's band.
        model = _base_checkpoint(keeper_path, tmp_path, {"node-a": 0.20})
        digest, mtime = _digest(model), os.stat(model).st_mtime

        staged = stage_checkpoint(model, STAGED_BLOCK)

        assert _digest(model) == digest  # the live checkpoint is byte-identical
        assert os.stat(model).st_mtime == mtime  # and not even restated
        assert staged == Path(model).with_name("serving.staged.pt")
        assert staged.parent == Path(model).parent  # beside the input, not off in a temp dir
        assert staged.is_file()

    def test_the_staged_block_is_the_guarded_one_and_the_rest_carries_through(
        self, keeper_path: str, tmp_path: Path
    ) -> None:
        # A rebake moves thresholds and nothing else about the model, so the
        # staged checkpoint is the base one with its serving block replaced:
        # the weights, the config, the stored normalization, and the feature
        # schema are carried through, and the block that was in force --
        # BASE_PROVENANCE's fit, the previous per-resource map, the previous
        # stamp -- is gone rather than merged under the new one, which would
        # leave a refused value in force behind the staged one.
        model = _base_checkpoint(keeper_path, tmp_path, {"node-a": 0.20})
        base = _checkpoint(model)

        staged = _checkpoint(stage_checkpoint(model, STAGED_BLOCK))

        assert set(staged) == set(base)
        assert {"model_state_dict", "feature_schema"} <= set(base)  # the carry-through is real
        assert {key for key in base if not _same(staged[key], base[key])} == {"serving"}
        assert staged["serving"] == STAGED_BLOCK
        assert staged["serving"]["per_resource"] == {"node-a": 0.24}
        assert staged["serving"]["margin_multiplier"] == 2.0
        assert staged["serving"]["calibration_days"] == 7.0
        assert staged["serving"]["quantile"] == 0.99 != BASE_PROVENANCE["quantile"]


class TestCalibrationRun:
    def test_the_run_bakes_guards_and_reports_without_writing_anything(
        self, keeper_path: str, tmp_path: Path
    ) -> None:
        # The composed run against a real keeper and a real capture: the bake
        # proposes (its own arithmetic, run against the BASE model), the band
        # decides through guard_rebake so a seeded old is stamped, the age
        # anchor supplies the weeks, and the report comes back beside the
        # serving block that would be staged. Nothing is written and the base
        # checkpoint is byte-identical afterwards: a run that moved the live
        # thresholds before anyone read its report is the failure the
        # stage-then-swap separation exists to prevent. The gated fleet makes
        # the bake propose for node-a alone -- node-b loses a trained feature
        # the capture supplies elsewhere, node-c has 12 windows -- and node-a
        # is the resource the serving block has nothing for, so the seed is
        # what its band measures against.
        model = _base_checkpoint(keeper_path, tmp_path, {"node-b": 0.21, "node-c": 0.22})
        healthy = gated_fleet_csv(tmp_path)
        seed = _write_seed(tmp_path, "json", per_resource={"node-a": 0.20})
        digest, listing = _digest(model), sorted(path.name for path in tmp_path.iterdir())

        report, block = run_calibration(
            model, healthy, profile=PROFILE, seed_path=seed, calibration_days=7.0
        )

        assert report["weeks"] == pytest.approx(STALE_WEEKS, abs=AGE_TOLERANCE_WEEKS)
        assert report["global"]["verdict"] == "accepted"
        assert report["global"]["old"] == BASE_GLOBAL  # the serving global, not the seed's
        assert report["global"]["proposed"] > 0.0  # a real bake of the capture above
        assert {rid: row["verdict"] for rid, row in report["per_resource"].items()} == {
            "node-a": "accepted-seeded",  # its old came from the seed, not from serving
            "node-b": "kept-gate-omitted:divergent-coverage:cpuUsageNanoCores",
            "node-c": "kept-gate-omitted:insufficient-windows:12<50",
        }

        assert block["threshold"] == report["global"]["kept"]
        assert block["per_resource"] == {
            rid: row["kept"] for rid, row in report["per_resource"].items()
        }
        assert block["margin_multiplier"] == 2.0  # the default margin, on the staged block
        assert block["calibration_days"] == 7.0
        assert block["rebaked_at"] == report["generated_at"]
        assert block["recon_metric"] == "numerical_mse_from_mu"  # the bake's own fields survive
        # An accepted global carries the fit that produced it: this bake's.
        assert block["quantile"] == 0.99
        assert block["n_calibration_windows"] > BASE_PROVENANCE["n_calibration_windows"]
        assert block["healthy_fpr"] != BASE_PROVENANCE["healthy_fpr"]

        assert _digest(model) == digest  # the base checkpoint is untouched
        assert sorted(path.name for path in tmp_path.iterdir()) == listing  # and nothing is staged
        assert json.loads(json.dumps(report, allow_nan=False)) == report

    def test_the_run_stages_the_block_it_would_serve_and_the_report_names_it(
        self, keeper_path: str, tmp_path: Path
    ) -> None:
        # The same composed run as above, asked to stage. What it wrote is the
        # block it returned -- the GUARDED one, not the bake's proposals -- and
        # the report names the file, which is how the CLI and the swap chain
        # find out what a later --apply would promote. The input stays
        # byte-identical: staging is a copy beside it, never a write over it.
        model = _base_checkpoint(keeper_path, tmp_path, {"node-b": 0.21, "node-c": 0.22})
        healthy = gated_fleet_csv(tmp_path)
        seed = _write_seed(tmp_path, "json", per_resource={"node-a": 0.20})
        digest = _digest(model)

        report, block = run_calibration(
            model, healthy, profile=PROFILE, seed_path=seed, calibration_days=7.0, stage=True
        )

        staged = Path(model).with_name("serving.staged.pt")
        assert report["staged_path"] == str(staged)
        assert staged.is_file()
        serving = _checkpoint(staged)["serving"]
        assert serving == block  # the guarded block, as the run returned it
        assert serving["threshold"] == report["global"]["kept"]
        # ISO UTC, so the next rebake ages the staged block from its stamp
        # rather than from the file: freshly staged, it is at the one-day floor.
        assert serving["rebaked_at"].endswith("Z")
        assert weeks_since_rebake(serving, staged) == pytest.approx(
            ONE_DAY_IN_WEEKS, abs=AGE_TOLERANCE_WEEKS
        )
        assert _digest(model) == digest  # the checkpoint in force is untouched
        assert json.loads(json.dumps(report, allow_nan=False)) == report

    def test_the_report_carries_the_run_level_keys_by_name(
        self, keeper_path: str, tmp_path: Path
    ) -> None:
        # Everything the untracked rebake report recorded is recorded here too
        # (Ryan's ruling of 2026-10-01): the run's own inputs, the warnings the
        # bake printed, and the two counts a reader scans first -- what this
        # rebake left out, and how many proposals it refused.
        model = _base_checkpoint(
            keeper_path, tmp_path, {"node-a": 0.20, "node-b": 0.21, "node-c": 0.22}
        )
        healthy = gated_fleet_csv(tmp_path)

        report, _ = run_calibration(
            model,
            healthy,
            profile=PROFILE,
            margin=2.5,
            quantile=0.95,
            calibration_days=7.0,
            allow_drift=True,
            dry_run=False,
        )

        assert set(report) == {
            "dry_run",
            "weeks",
            "staged_path",
            "global",
            "per_resource",
            "generated_at",
            "calibration",
            "calibration_days",
            "margin",
            "quantile",
            "allow_drift",
            "omitted",
            "global_kept",
            "bake_warnings",
            "rejected_count",
        }
        assert report["calibration"] == healthy
        assert report["calibration_days"] == 7.0
        assert report["margin"] == 2.5
        assert report["quantile"] == 0.95
        assert report["allow_drift"] is True
        assert report["dry_run"] is False
        assert report["omitted"] == ["node-b", "node-c"]  # no proposal reached the band
        assert report["global_kept"] is False  # the fresh global is what serves
        assert report["rejected_count"] == 0
        assert len(report["bake_warnings"]) == 2  # the bake's stderr, carried into the report
        assert all(line.startswith("warning: resource ") for line in report["bake_warnings"])

        stamped = datetime.fromisoformat(report["generated_at"].replace("Z", "+00:00"))
        assert abs((datetime.now(timezone.utc) - stamped).total_seconds()) < 300
        assert json.loads(json.dumps(report, allow_nan=False)) == report

    def test_a_rejected_proposal_keeps_the_old_value_and_the_report_counts_it(
        self, keeper_path: str, tmp_path: Path
    ) -> None:
        # The direction the two runs above never take. A serving block stamped
        # just now ages to the one-day floor, where the band is about six
        # percent either way, and TINY_THRESHOLD sits orders of magnitude under
        # anything a real bake of the gated fleet proposes: the global is
        # REJECTED-grew, and node-a, served at the same value with a real
        # proposal against it, is refused the same way. What matters is what
        # comes back beside the report -- the block that would be staged
        # carries the GUARDED values, the old global and the old node-a, not
        # the bake's -- and that the report says so: global_kept True, two
        # refusals counted. No seed: every old value is usable, so nothing is
        # seeded, and node-b and node-c, with neither a previous threshold nor
        # a proposal, have no row.
        model = _base_checkpoint(
            keeper_path,
            tmp_path,
            {"node-a": TINY_THRESHOLD},
            threshold=TINY_THRESHOLD,
            age=timedelta(0),
        )
        healthy = gated_fleet_csv(tmp_path)

        report, block = run_calibration(model, healthy, profile=PROFILE)

        assert report["weeks"] == pytest.approx(ONE_DAY_IN_WEEKS, abs=AGE_TOLERANCE_WEEKS)
        assert report["global"]["verdict"] == "REJECTED-grew"
        assert report["global"]["kept"] == report["global"]["old"] == TINY_THRESHOLD
        assert report["global"]["proposed"] > TINY_THRESHOLD  # a real bake, refused
        assert {rid: row["verdict"] for rid, row in report["per_resource"].items()} == {
            "node-a": "REJECTED-grew"
        }
        assert report["global_kept"] is True  # the previous global is what stays in force
        assert report["rejected_count"] == 2  # the global and node-a
        assert report["omitted"] == []  # nothing was omitted that had a previous threshold

        assert block["threshold"] == TINY_THRESHOLD  # the guarded global, not the bake's
        assert block["per_resource"] == {"node-a": TINY_THRESHOLD}  # likewise per resource
        # A kept global keeps its own provenance, not the refused fit's.
        assert {key: block[key] for key in BASE_PROVENANCE} == BASE_PROVENANCE
        assert json.loads(json.dumps(report, allow_nan=False)) == report

    @pytest.mark.parametrize(
        ("proposed", "allow_drift", "verdict", "rejected", "global_kept"),
        [
            (math.nan, False, "REJECTED-nonfinite", 2, True),
            (10 * BASE_GLOBAL, True, "accepted-allow-drift", 0, False),
        ],
        ids=["nonfinite-refused", "drift-allowed"],
    )
    def test_the_run_hands_the_bake_its_parameters_and_the_guard_its_flag(
        self,
        keeper_path: str,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
        proposed: float,
        allow_drift: bool,
        verdict: str,
        rejected: int,
        global_kept: bool,
    ) -> None:
        # The plumbing the composed runs above cannot see. A real bake of the
        # fixture never proposes a non-finite value, and the other runs take a
        # band that accepts whatever it proposes, so a run that ignored its
        # quantile, its margin, or its allow_drift, or that reported inputs the
        # bake never used, passed them all. The bake is replaced by a recorder
        # returning a block in the real bake's shape (its keys are the ones
        # compute_serving_block writes); everything around it is real: the
        # checkpoint, the capture fetch, the guard, the report.
        seen: dict[str, object] = {}

        def recording_bake(keeper, df_long, *, quantile, step, per_resource_margin):
            seen.update(quantile=quantile, step=step, per_resource_margin=per_resource_margin)
            print(
                "warning: resource 'node-b' failed the non-finite-quantile:nan gate.",
                file=sys.stderr,
            )
            block = {
                "threshold": proposed,
                "quantile": quantile,
                "healthy_fpr": 0.01,
                "n_calibration_windows": 60,
                "recon_metric": "numerical_mse_from_mu",
                "per_resource": {"node-a": proposed},
                "margin_multiplier": per_resource_margin,
            }
            return block, {}

        monkeypatch.setattr(bake_mod, "compute_serving_block", recording_bake)
        model = _base_checkpoint(keeper_path, tmp_path, {"node-a": BASE_GLOBAL}, age=timedelta(0))
        healthy = gated_fleet_csv(tmp_path)

        report, block = run_calibration(
            model, healthy, profile=PROFILE, quantile=0.95, margin=2.5, allow_drift=allow_drift
        )

        assert seen == {
            "quantile": 0.95,
            "step": int(get_config().window_step),
            "per_resource_margin": 2.5,
        }
        assert report["quantile"] == 0.95 and report["margin"] == 2.5
        assert report["global"]["verdict"] == verdict
        assert report["per_resource"]["node-a"]["verdict"] == verdict
        assert report["rejected_count"] == rejected
        assert report["global_kept"] is global_kept
        assert report["bake_warnings"] == [
            "warning: resource 'node-b' failed the non-finite-quantile:nan gate."
        ]
        assert report["bake_warnings"][0] in capsys.readouterr().err  # re-emitted, not swallowed
        # The block's provenance follows the global that stays in force.
        expected = (
            BASE_PROVENANCE
            if global_kept
            else {"quantile": 0.95, "healthy_fpr": 0.01, "n_calibration_windows": 60}
        )
        assert {key: block[key] for key in BASE_PROVENANCE} == expected
        assert block["threshold"] == report["global"]["kept"]
        assert json.loads(json.dumps(report, allow_nan=False)) == report


class TestPackageExports:
    def test_guard_surface_resolves_from_scry_eval(self) -> None:
        # The three names are package exports, resolved lazily: calibration.py
        # turns torch-heavy once the bake lands, and import scry.eval stays
        # torch-free either way.
        import scry.eval
        from scry.eval import GuardVerdict as PackageGuardVerdict
        from scry.eval import check_guards as package_check_guards
        from scry.eval import guard_value as package_guard_value

        assert package_check_guards is check_guards
        assert package_guard_value is guard_value
        assert PackageGuardVerdict is GuardVerdict
        for name in ("GuardVerdict", "check_guards", "guard_value"):
            assert name in scry.eval.__all__
            assert name in dir(scry.eval)
