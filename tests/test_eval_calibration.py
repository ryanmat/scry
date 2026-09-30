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
"""

from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import fields
from pathlib import Path
from types import ModuleType

import pytest

from scry.eval.calibration import GuardVerdict, check_guards, guard_value, resolve_old_thresholds
from scry.eval.rubric import SpecError

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


def _torch_that_must_not_load() -> ModuleType:
    """A stand-in ``torch`` whose ``load`` fails the test if the seed reader calls it."""
    module = ModuleType("torch")

    def _load(*args: object, **kwargs: object) -> object:
        raise AssertionError("torch.load called for a seed that is not a checkpoint")

    module.load = _load  # type: ignore[attr-defined]
    return module


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
        # seed when every current value is usable. What does not survive is an
        # unusable entry -- every value in the resolved map is one the band
        # could measure against -- and node-y has no proposal to guard anyway.
        old_global, old = resolve_old_thresholds(
            {"threshold": 0.30, "per_resource": {"node-a": 0.20, "node-z": 0.25, "node-y": 0.0}},
            ["node-a"],
        )

        assert old_global == 0.30
        assert old == {"node-a": 0.20, "node-z": 0.25}


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
