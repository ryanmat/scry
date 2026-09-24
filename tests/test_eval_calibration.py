# Description: Tests for scry.eval.calibration: the per-week band guard on a single value.
# Description: Pins the exponential bands, the one-day age floor, and the measured global jump.

"""Tests for the single-value recalibration guard.

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
"""

from __future__ import annotations

from dataclasses import fields

import pytest

from scry.eval.calibration import GuardVerdict, guard_value

ONE_DAY_IN_WEEKS = 1.0 / 7.0
"""The age floor of one day expressed in weeks: ``max(age, 1 day) / 7``."""


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
