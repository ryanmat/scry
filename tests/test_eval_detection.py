# Description: Tests for scry.eval.detection: anomaly_runs pins, differential parity, and properties.
# Description: Enforces the torch-free import contract for the eval package via subprocess checks.

"""Tests for the eval detection primitives.

``anomaly_runs`` is pinned three ways: exact hand-computed cases, differential
parity against the harness implementation it was promoted from
(``validate_incident._anomaly_runs``), and run-shape invariants over seeded
random inputs. Two subprocess tests enforce the spec's torch-free acceptance
criterion for ``scry.eval`` and ``scry.eval.detection``.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import validate_incident as vi

from scry.eval.detection import SUSTAIN_DEFAULT, anomaly_runs


def _flags(*bits: int) -> np.ndarray:
    return np.array(bits, dtype=bool)


class TestAnomalyRunsPins:
    def test_sustain_three_keeps_only_long_run(self) -> None:
        assert anomaly_runs(_flags(0, 1, 1, 1, 0, 1, 1), sustain=3) == [(1, 3)]

    def test_sustain_two_keeps_both_runs(self) -> None:
        assert anomaly_runs(_flags(0, 1, 1, 1, 0, 1, 1), sustain=2) == [(1, 3), (5, 6)]

    def test_sustain_one_matches_sustain_two_here(self) -> None:
        assert anomaly_runs(_flags(0, 1, 1, 1, 0, 1, 1), sustain=1) == [(1, 3), (5, 6)]

    def test_all_true_single_run(self) -> None:
        assert anomaly_runs(_flags(1, 1, 1, 1), sustain=3) == [(0, 3)]

    def test_empty_flags(self) -> None:
        assert anomaly_runs(np.array([], dtype=bool), sustain=3) == []

    def test_short_run_dropped_entirely(self) -> None:
        assert anomaly_runs(_flags(1, 1), sustain=3) == []

    def test_sustain_default_is_three(self) -> None:
        assert SUSTAIN_DEFAULT == 3


class TestAnomalyRunsDifferential:
    def test_matches_validate_incident_on_seeded_arrays(self) -> None:
        """Promoted code must agree with the origin on 20 seeded random inputs."""
        for seed in range(20):
            rng = np.random.default_rng(seed)
            n = int(rng.integers(0, 60))
            flags = rng.random(n) < rng.uniform(0.2, 0.8)
            sustain = int(rng.integers(1, 5))
            assert anomaly_runs(flags, sustain) == vi._anomaly_runs(flags, sustain), (
                f"divergence at seed={seed} sustain={sustain}"
            )


class TestAnomalyRunsProperties:
    def _cases(self) -> list[tuple[np.ndarray, int]]:
        cases = []
        for seed in range(40, 60):
            rng = np.random.default_rng(seed)
            n = int(rng.integers(1, 80))
            flags = rng.random(n) < rng.uniform(0.1, 0.9)
            cases.append((flags, int(rng.integers(1, 5))))
        return cases

    def test_runs_disjoint_and_time_ordered(self) -> None:
        for flags, sustain in self._cases():
            runs = anomaly_runs(flags, sustain)
            for (s0, e0), (s1, _e1) in zip(runs, runs[1:]):
                assert s0 <= e0
                assert s1 > e0 + 1, "adjacent runs would have merged; not disjoint-maximal"

    def test_run_length_at_least_sustain(self) -> None:
        for flags, sustain in self._cases():
            for start, end in anomaly_runs(flags, sustain):
                assert end - start + 1 >= sustain

    def test_flags_all_true_inside_each_run(self) -> None:
        for flags, sustain in self._cases():
            for start, end in anomaly_runs(flags, sustain):
                assert flags[start : end + 1].all()

    def test_runs_maximal_on_both_sides(self) -> None:
        for flags, sustain in self._cases():
            for start, end in anomaly_runs(flags, sustain):
                assert start == 0 or not flags[start - 1]
                assert end == len(flags) - 1 or not flags[end + 1]


class TestTorchFreeImport:
    def _assert_torch_free(self, module: str) -> None:
        proc = subprocess.run(
            [sys.executable, "-c", f"import {module}, sys; assert 'torch' not in sys.modules"],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr

    def test_detection_module_imports_without_torch(self) -> None:
        self._assert_torch_free("scry.eval.detection")

    def test_eval_package_root_imports_without_torch(self) -> None:
        self._assert_torch_free("scry.eval")
