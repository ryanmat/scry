# Description: Tests for scry.eval.hygiene: the three per-resource eligibility gates.
# Description: Pins the bake's delegation to them and to scry.eval.bake, and the torch-free import.

"""Tests for the per-resource eligibility gates.

Each gate is pinned through its reason string: divergent coverage (a trained
feature the capture supplies elsewhere is missing for the resource), the
window floor (exact ``insufficient-windows:12<50`` format), non-positive
quantile, and non-finite quantile -- the last of which no ``<= 0`` screen can
catch, since every comparison against a NaN is False. The capture-feature
arithmetic, multi-gate accumulation, stringified key iteration order, and the
torch-free import contract are pinned separately.
The bake-delegation tests run the real bake on a gated synthetic fleet and pin
the printed eligibility map, its agreement with the per_resource map, the
stderr warning text (each warning naming the gate that omitted the resource
and what the resource serves afterwards), and the compute_serving_block
exposure. They also pin what the bake script takes from the package rather than
holding itself: its setup (``scry.eval.bake.prepare_bake``) and the two
constants that name the metric and what an omitted resource serves, which the
script imports rather than defines, so the package and the script cannot carry
two wordings of either.

``prepare_bake`` is the setup the script and the rebake path share, pinned on
what it hands the bake -- the keeper, the long-format capture, and the window
step -- on the device override it applies to the keeper's model, and on the
warning it prints when the capture lacks a feature the checkpoint was trained
on, which reaches the rebake path through it.
"""

from __future__ import annotations

import inspect
import json
import re
import subprocess
import sys
from pathlib import Path

import bake_serving_threshold as bake_mod
import numpy as np
import pytest
from synth import MISSING_FEATURE, PROFILE, SERIES, gated_fleet_csv, missing_feature_csv

from scry.data.fetcher import fetch_full_capture
from scry.eval import bake as bake_pkg
from scry.eval.bake import prepare_bake
from scry.eval.calibration import DEFAULT_QUANTILE
from scry.eval.hygiene import (
    MIN_PER_RESOURCE_WINDOWS,
    REASON_DIVERGENT,
    REASON_NONFINITE_QUANTILE,
    REASON_NONPOSITIVE_QUANTILE,
    REASON_TOO_FEW_WINDOWS,
    ResourceEligibility,
    per_resource_eligibility,
)
from scry.model.checkpoint import load_keeper
from scry.utils.config import get_config

_TRAINED = ("cpu", "mem", "fs")


def _arrays(counts: dict[str, int], zero: set[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Per-window resource_ids and positive errors; ``zero`` names all-zero resources."""
    rng = np.random.default_rng(7)
    ids: list[str] = []
    errs: list[float] = []
    for rid, n in counts.items():
        ids.extend([rid] * n)
        if zero and rid in zero:
            errs.extend([0.0] * n)
        else:
            errs.extend(rng.uniform(0.01, 0.2, n).tolist())
    return np.array(ids), np.array(errs)


def _full_coverage(*rids: str) -> dict[str, set[str]]:
    return {rid: set(_TRAINED) for rid in rids}


class TestConstants:
    def test_min_per_resource_windows(self) -> None:
        assert MIN_PER_RESOURCE_WINDOWS == 50

    def test_reason_constants(self) -> None:
        assert REASON_DIVERGENT == "divergent-coverage"
        assert REASON_TOO_FEW_WINDOWS == "insufficient-windows"
        assert REASON_NONPOSITIVE_QUANTILE == "non-positive-quantile"
        assert REASON_NONFINITE_QUANTILE == "non-finite-quantile"

    def test_reason_constants_are_package_exports(self) -> None:
        # The four reasons are one vocabulary: a consumer reading three of them
        # off the package and the fourth off the module would not notice the
        # newest gate at all.
        import scry.eval

        assert scry.eval.REASON_NONFINITE_QUANTILE == REASON_NONFINITE_QUANTILE
        assert "REASON_NONFINITE_QUANTILE" in scry.eval.__all__


class TestDivergentCoverageGate:
    def test_missing_trained_feature_present_elsewhere_is_divergent(self) -> None:
        ids, errs = _arrays({"node-a": 60, "node-b": 60})
        features = {"node-a": set(_TRAINED), "node-b": {"cpu", "mem"}}
        result = per_resource_eligibility(
            trained_features=_TRAINED,
            features_by_resource=features,
            resource_ids=ids,
            errors=errs,
            quantile=0.99,
        )
        verdict = result["node-b"]
        assert verdict.eligible is False
        assert verdict.missing_features == ["fs"]
        assert any(r.startswith(f"{REASON_DIVERGENT}:") for r in verdict.reasons)
        assert result["node-a"].eligible is True
        assert result["node-a"].reasons == []

    def test_feature_absent_from_whole_capture_is_not_divergent(self) -> None:
        ids, errs = _arrays({"node-a": 60, "node-b": 60})
        features = {"node-a": {"cpu", "mem"}, "node-b": {"cpu", "mem"}}
        result = per_resource_eligibility(
            trained_features=(*_TRAINED, "netio"),
            features_by_resource=features,
            resource_ids=ids,
            errors=errs,
            quantile=0.99,
        )
        assert result["node-a"].eligible is True
        assert result["node-b"].eligible is True

    def test_untrained_capture_feature_does_not_count(self) -> None:
        ids, errs = _arrays({"node-a": 60, "node-b": 60})
        features = {"node-a": {*_TRAINED, "extra_metric"}, "node-b": set(_TRAINED)}
        result = per_resource_eligibility(
            trained_features=_TRAINED,
            features_by_resource=features,
            resource_ids=ids,
            errors=errs,
            quantile=0.99,
        )
        assert result["node-b"].eligible is True
        assert result["node-b"].missing_features == []


class TestWindowFloorGate:
    def test_reason_format_is_exact(self) -> None:
        ids, errs = _arrays({"node-a": 60, "node-b": 12})
        result = per_resource_eligibility(
            trained_features=_TRAINED,
            features_by_resource=_full_coverage("node-a", "node-b"),
            resource_ids=ids,
            errors=errs,
            quantile=0.99,
        )
        verdict = result["node-b"]
        assert verdict.eligible is False
        assert verdict.reasons == ["insufficient-windows:12<50"]
        assert verdict.n_windows == 12

    def test_min_windows_override(self) -> None:
        ids, errs = _arrays({"node-b": 12})
        result = per_resource_eligibility(
            trained_features=_TRAINED,
            features_by_resource=_full_coverage("node-b"),
            resource_ids=ids,
            errors=errs,
            quantile=0.99,
            min_windows=10,
        )
        assert result["node-b"].eligible is True


class TestNonPositiveQuantileGate:
    def test_all_zero_errors_fail_with_quantile_recorded(self) -> None:
        ids, errs = _arrays({"node-a": 60, "node-b": 60}, zero={"node-b"})
        result = per_resource_eligibility(
            trained_features=_TRAINED,
            features_by_resource=_full_coverage("node-a", "node-b"),
            resource_ids=ids,
            errors=errs,
            quantile=0.99,
        )
        verdict = result["node-b"]
        assert verdict.eligible is False
        assert any(r.startswith(f"{REASON_NONPOSITIVE_QUANTILE}:") for r in verdict.reasons)
        assert verdict.own_quantile == 0.0


class TestNonFiniteQuantileGate:
    @pytest.mark.parametrize(
        ("errors_for_node_b", "expected_detail"),
        [
            pytest.param(np.full(60, np.nan), "nan", id="nan"),
            pytest.param(np.concatenate([np.full(59, 0.05), [np.inf]]), "inf", id="inf"),
        ],
    )
    def test_non_finite_quantile_is_ineligible_on_its_own_reason(
        self, errors_for_node_b: np.ndarray, expected_detail: str
    ) -> None:
        # A NaN quantile compares False against the non-positive gate's `<= 0`,
        # so without a gate of its own it passes as eligible and the bake
        # serves `margin * nan` -- a threshold no error can ever exceed, which
        # silences the resource instead of protecting it. An infinite quantile
        # is the same defect in the other direction: nothing can move it. Both
        # are reported on their own reason, not folded into the non-positive
        # one, so the verdict names what is wrong with the value. (A -inf
        # quantile takes the same `not isfinite` branch; numpy's interpolation
        # cannot produce one without a RuntimeWarning, so it is unpinned here.)
        ids = np.array(["node-a"] * 60 + ["node-b"] * 60)
        errors = np.concatenate([np.full(60, 0.05), errors_for_node_b])
        result = per_resource_eligibility(
            trained_features=_TRAINED,
            features_by_resource=_full_coverage("node-a", "node-b"),
            resource_ids=ids,
            errors=errors,
            quantile=0.99,
        )

        verdict = result["node-b"]
        assert verdict.eligible is False
        assert verdict.reasons == [f"{REASON_NONFINITE_QUANTILE}:{expected_detail}"]
        assert result["node-a"].eligible is True


class TestGateAccumulation:
    def test_resource_failing_two_gates_lists_both_reasons(self) -> None:
        ids, errs = _arrays({"node-a": 60, "node-b": 12}, zero={"node-b"})
        result = per_resource_eligibility(
            trained_features=_TRAINED,
            features_by_resource=_full_coverage("node-a", "node-b"),
            resource_ids=ids,
            errors=errs,
            quantile=0.99,
        )
        verdict = result["node-b"]
        assert verdict.eligible is False
        assert "insufficient-windows:12<50" in verdict.reasons
        assert any(r.startswith(f"{REASON_NONPOSITIVE_QUANTILE}:") for r in verdict.reasons)
        assert len(verdict.reasons) == 2

    def test_eligible_resource_verdict_fields(self) -> None:
        ids, errs = _arrays({"node-a": 60})
        result = per_resource_eligibility(
            trained_features=_TRAINED,
            features_by_resource=_full_coverage("node-a"),
            resource_ids=ids,
            errors=errs,
            quantile=0.99,
        )
        verdict = result["node-a"]
        assert isinstance(verdict, ResourceEligibility)
        assert verdict.eligible is True
        assert verdict.reasons == []
        assert verdict.missing_features == []
        assert verdict.n_windows == 60
        assert verdict.own_quantile == float(np.quantile(errs, 0.99))


class TestIterationAndKeys:
    def test_keys_stringified_and_ordered_by_raw_sort(self) -> None:
        # Numeric ids sort numerically BEFORE stringification: 7 < 101, though
        # "101" < "7" as strings. Keys are str; order follows the raw sort.
        ids = np.array([101, 7] * 30)
        errs = np.full(60, 0.05)
        result = per_resource_eligibility(
            trained_features=_TRAINED,
            features_by_resource={"7": set(_TRAINED), "101": set(_TRAINED)},
            resource_ids=ids,
            errors=errs,
            quantile=0.99,
        )
        assert list(result.keys()) == ["7", "101"]
        assert all(isinstance(key, str) for key in result)
        assert result["7"].resource_id == "7"


class TestBakeDelegation:
    def _printed_map(self, stdout: str) -> dict[str, list[str]]:
        prefix = "per-resource eligibility: "
        line = next(ln for ln in stdout.splitlines() if ln.startswith(prefix))
        return json.loads(line.removeprefix(prefix))

    def test_eligibility_map_agrees_with_per_resource_keys(
        self, keeper_path: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        healthy = gated_fleet_csv(tmp_path)
        serving = bake_mod.bake(
            keeper_path,
            healthy,
            profile=PROFILE,
            per_resource_margin=2.0,
            output=str(tmp_path / "gated.pt"),
        )
        eligibility = self._printed_map(capsys.readouterr().out)
        assert set(eligibility) == {"node-a", "node-b", "node-c"}
        eligible = {rid for rid, reasons in eligibility.items() if not reasons}
        assert eligible == set(serving["per_resource"]) == {"node-a"}
        assert eligibility["node-b"] == ["divergent-coverage:cpuUsageNanoCores"]
        assert eligibility["node-c"] == ["insufficient-windows:12<50"]

    def test_gate_warning_stderr_text_unchanged(
        self, keeper_path: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        healthy = gated_fleet_csv(tmp_path)
        bake_mod.bake(
            keeper_path,
            healthy,
            profile=PROFILE,
            per_resource_margin=2.0,
            output=str(tmp_path / "warned.pt"),
        )
        err = capsys.readouterr().err
        # A re-bake keeps an omitted resource's previous threshold rather than
        # falling it back to the global one, which the old wording promised it
        # would; every warning describes both outcomes now.
        assert (
            "lacks 1 trained feature(s) the capture supplies elsewhere "
            "(cpuUsageNanoCores); its bake-time windows are filled at -mean/std "
            "where serving fills neutral. It is omitted from the per-resource "
            "map: it keeps its previous threshold where it has one, and serves "
            "the global threshold where it does not."
        ) in err
        assert (
            "has only 12 window(s) (< 50). It is omitted from the per-resource "
            "map: it keeps its previous threshold where it has one, and serves "
            "the global threshold where it does not."
        ) in err
        assert err.count("warning: resource") == 2

    async def test_a_non_finite_quantile_warning_names_its_gate(
        self,
        keeper_path: str,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # A NaN own-quantile fails the non-finite gate, which no `<= 0` screen
        # catches, and the warning for it used to call it "a non-positive
        # healthy quantile" -- sending an operator to look for a negative
        # number that is not there. The gate's own reason is what the warning
        # names now. The verdict map is substituted because no healthy capture
        # produces a NaN quantile on demand; the branch it drives is the real
        # one, printed by the real bake.
        healthy = gated_fleet_csv(tmp_path)
        keeper = load_keeper(keeper_path)
        df_long = await fetch_full_capture(healthy, profile=PROFILE)
        monkeypatch.setattr(
            bake_pkg,
            "per_resource_eligibility",
            lambda **kwargs: {
                rid: ResourceEligibility(
                    resource_id=rid,
                    eligible=False,
                    reasons=[f"{REASON_NONFINITE_QUANTILE}:nan"],
                    n_windows=60,
                    own_quantile=float("nan"),
                    missing_features=[],
                )
                for rid in ("node-a", "node-b", "node-c")
            },
        )

        serving, _ = bake_mod.compute_serving_block(
            keeper, df_long, quantile=0.99, step=10, per_resource_margin=2.0
        )

        err = capsys.readouterr().err
        assert serving["per_resource"] == {}  # every resource failed the gate
        assert err.count(f"failed the {REASON_NONFINITE_QUANTILE}:nan gate") == 3
        assert "non-positive healthy quantile" not in err
        assert "it keeps its previous threshold where it has one" in err

    def test_the_rebake_takes_the_quantile_the_bake_takes(self) -> None:
        # calibration.DEFAULT_QUANTILE is written out rather than imported,
        # since the bake is a script the package reaches only lazily; this
        # pins the copy to bake()'s own default so the two cannot drift apart.
        assert DEFAULT_QUANTILE == inspect.signature(bake_mod.bake).parameters["quantile"].default

    def test_bake_module_delegates_to_hygiene(self) -> None:
        # The gates are the bake's, so the delegation sits wherever the bake
        # does: scry.eval.bake, since compute_serving_block moved there.
        assert bake_pkg.per_resource_eligibility is per_resource_eligibility
        assert bake_pkg.MIN_PER_RESOURCE_WINDOWS == MIN_PER_RESOURCE_WINDOWS
        # The constant is imported, never redefined locally (equality alone
        # would not catch a shadowing `MIN_PER_RESOURCE_WINDOWS = 50`).
        source = Path(bake_pkg.__file__).read_text()
        assert not re.search(r"^MIN_PER_RESOURCE_WINDOWS\s*=", source, re.MULTILINE)

    def test_bake_module_takes_its_setup_and_constants_from_the_package(self) -> None:
        # The script is a CLI over package code, not a second copy of it: the
        # setup and the bake it shares with the rebake path live in
        # scry.eval.bake and the script calls them, so the two cannot drift; a
        # second copy of the setup is how one path loses the missing-feature
        # warning, and a second copy of the bake is how they stop agreeing on
        # the number itself.
        assert bake_mod.prepare_bake is prepare_bake
        assert bake_mod.compute_serving_block is bake_pkg.compute_serving_block

        source = Path(bake_mod.__file__).read_text()
        assert "prepare_bake(" in source  # called, not merely imported
        assert not re.search(r"^def compute_serving_block", source, re.MULTILINE)
        # The script keeps no feature check of its own and no second wording
        # of the warning that check prints.
        assert "missing_features(" not in source
        assert "the checkpoint was trained on" not in source
        # Both constants are imported, never redefined locally (equality alone
        # would not catch a shadowing definition, as above).
        for name in ("RECON_METRIC", "OMITTED_FALLBACK"):
            assert not re.search(rf"^{name}\s*=", source, re.MULTILINE)

    async def test_compute_serving_block_exposes_eligibility(
        self, keeper_path: str, tmp_path: Path
    ) -> None:
        healthy = gated_fleet_csv(tmp_path)
        keeper = load_keeper(keeper_path)
        df_long = await fetch_full_capture(healthy, profile=PROFILE)

        serving, eligibility = bake_mod.compute_serving_block(
            keeper, df_long, quantile=0.99, step=10, per_resource_margin=2.0
        )
        assert eligibility is not None
        assert {rid for rid, v in eligibility.items() if v.eligible} == set(serving["per_resource"])
        assert isinstance(eligibility["node-b"], ResourceEligibility)

        _, without_margin = bake_mod.compute_serving_block(keeper, df_long, quantile=0.99, step=10)
        assert without_margin is None


class TestPrepareBake:
    def test_it_returns_the_keeper_the_capture_and_the_window_step(
        self, keeper_path: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # The three things a bake needs before it can score anything,
        # assembled once for the script and the rebake path alike. The
        # step comes from the config rather than from a caller, which is why
        # it is returned at all.
        healthy = gated_fleet_csv(tmp_path)

        keeper, df_long, step = prepare_bake(keeper_path, healthy, profile=PROFILE)

        assert keeper.numerical_features == list(SERIES)  # the checkpoint's own schema
        assert set(df_long["resource_id"].unique()) == {"node-a", "node-b", "node-c"}
        assert step == int(get_config().window_step)
        assert capsys.readouterr().err == ""  # a complete capture warns about nothing

    def test_it_warns_when_the_capture_lacks_a_trained_feature(
        self, keeper_path: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # bake()'s missing-feature warning, in its wording, printed for every
        # caller of the setup: a capture missing a trained feature is windowed
        # with that feature filled neutral, so the threshold it produces is
        # not the one the model sees on complete data.
        healthy = missing_feature_csv(tmp_path)

        keeper, df_long, _ = prepare_bake(keeper_path, healthy, profile=PROFILE)

        assert capsys.readouterr().err == (
            f"warning: {healthy} lacks 1 feature(s) the checkpoint was trained "
            f"on ({MISSING_FEATURE}); the baked threshold will not match what "
            "the model sees on complete data.\n"
        )
        assert MISSING_FEATURE in keeper.numerical_features  # trained on it
        assert MISSING_FEATURE not in set(df_long["metric_name"])  # and it is not there

    def test_it_moves_the_keeper_onto_the_device_it_is_given(
        self, keeper_path: str, tmp_path: Path
    ) -> None:
        # ``meta`` is the device for this, because every torch build has it
        # and it allocates nothing -- on a CPU-only host a run asked for
        # ``cpu`` and a run that ignored its device are indistinguishable.
        # The weights move, not just the label the keeper carries, since the
        # label is what the scoring call passes and the weights are what it
        # runs on.
        healthy = gated_fleet_csv(tmp_path)

        moved, _, _ = prepare_bake(keeper_path, healthy, profile=PROFILE, device="meta")
        default, _, _ = prepare_bake(keeper_path, healthy, profile=PROFILE)

        assert moved.device == "meta"
        assert {p.device.type for p in moved.model.parameters()} == {"meta"}
        # No device named leaves the keeper's own detection in force.
        assert default.device != "meta"
        assert "meta" not in {p.device.type for p in default.model.parameters()}


class TestTorchFreeImport:
    def test_hygiene_module_imports_without_torch(self) -> None:
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "import scry.eval.hygiene, sys; assert 'torch' not in sys.modules",
            ],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr

    def test_bake_module_imports_without_torch(self) -> None:
        # The bake's setup lives in the package so the image can reach it
        # (`pip install .` copies src alone), and the package stays importable
        # where torch is not: load_keeper is reached inside prepare_bake.
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "import scry.eval.bake, sys; assert 'torch' not in sys.modules",
            ],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
