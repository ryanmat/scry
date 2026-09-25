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
"""

from __future__ import annotations

import math
from dataclasses import fields

import pytest

from scry.eval.calibration import GuardVerdict, check_guards, guard_value

ONE_DAY_IN_WEEKS = 1.0 / 7.0
"""The age floor of one day expressed in weeks: ``max(age, 1 day) / 7``."""

GUARD_WEEKS = 1.0
"""One week of age: grow limit 1.5, shrink floor 1 / 1.35 == 0.7407..."""

OLD_PER_RESOURCE = {"node-a": 0.20, "node-b": 0.20, "node-c": 0.20}
"""The serving per-resource map the three-resource fixtures guard against."""

NEW_PER_RESOURCE = {"node-a": 0.24, "node-b": 0.40, "node-c": 0.10}
"""Proposals at ratios 1.2 (accepted), 2.0 (over 1.5), and 0.5 (under 0.7407)."""


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
