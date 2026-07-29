# Description: Tests for scry.eval.hygiene: the three per-resource eligibility gates.
# Description: Enforces the torch-free import contract for the hygiene module via a subprocess check.

"""Tests for the per-resource eligibility gates.

Each gate is pinned through its reason string: divergent coverage (a trained
feature the capture supplies elsewhere is missing for the resource), the
window floor (exact ``insufficient-windows:12<50`` format), and non-positive
quantile. The capture-feature arithmetic, multi-gate accumulation, stringified
key iteration order, and the torch-free import contract are pinned separately.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np

from scry.eval.hygiene import (
    MIN_PER_RESOURCE_WINDOWS,
    REASON_DIVERGENT,
    REASON_NONPOSITIVE_QUANTILE,
    REASON_TOO_FEW_WINDOWS,
    ResourceEligibility,
    per_resource_eligibility,
)

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
