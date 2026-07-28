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
import pandas as pd
import pytest
import validate_incident as vi

from scry.eval.detection import SUSTAIN_DEFAULT, anomaly_runs, select_detection


def _flags(*bits: int) -> np.ndarray:
    return np.array(bits, dtype=bool)


def _ts(hhmm: str) -> pd.Timestamp:
    return pd.Timestamp(f"2026-01-01 {hhmm}:00+00:00")


# A 5-minute scan grid spanning 00:00-03:55; onset fixed at 02:00 throughout.
_GRID = pd.date_range("2026-01-01 00:00:00+00:00", periods=48, freq="5min")
_ONSET = _ts("02:00")


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


class TestSelectDetectionLookback:
    def test_precursor_ramp_detected_with_positive_lead(self) -> None:
        """(a) A sustained run starting before onset and reaching it leads."""
        result = select_detection([(_ts("01:40"), _ts("02:05"))], _GRID, _ONSET, mode="lookback")
        assert result.detected
        assert result.detection_time == _ts("01:40")
        assert result.lead_seconds == 1200.0
        assert result.bridged
        assert (result.n_runs_pre_onset, result.n_runs_at_or_after) == (1, 0)

    def test_step_exactly_at_onset_leads_with_zero_lead(self) -> None:
        """(b) A run starting exactly at onset IS leading, lead 0."""
        result = select_detection([(_ts("02:00"), _ts("02:15"))], _GRID, _ONSET, mode="lookback")
        assert result.detected
        assert result.detection_time == _ONSET
        assert result.lead_seconds == 0.0
        assert not result.bridged
        assert (result.n_runs_pre_onset, result.n_runs_at_or_after) == (0, 1)

    def test_recovered_blip_not_credited_late_alarm_detected(self) -> None:
        """(c) A blip recovered more than one step pre-onset never counts."""
        spans = [(_ts("01:20"), _ts("01:35")), (_ts("02:10"), _ts("02:25"))]
        result = select_detection(spans, _GRID, _ONSET, mode="lookback")
        assert result.detected
        assert result.detection_time == _ts("02:10")
        assert result.lead_seconds == -600.0
        assert not result.bridged
        assert (result.n_runs_pre_onset, result.n_runs_at_or_after) == (1, 1)

    def test_run_spanning_onset_credited_from_start(self) -> None:
        """(d) A run spanning onset is credited from its START, lead 300.0."""
        result = select_detection([(_ts("01:55"), _ts("02:05"))], _GRID, _ONSET, mode="lookback")
        assert result.detected
        assert result.detection_time == _ts("01:55")
        assert result.lead_seconds == 300.0
        assert result.bridged

    def test_run_recovered_within_one_step_of_onset_still_leads(self) -> None:
        """The bridge tolerance is one median inter-window step, not exact contact."""
        result = select_detection([(_ts("01:45"), _ts("01:55"))], _GRID, _ONSET, mode="lookback")
        assert result.detected
        assert result.detection_time == _ts("01:45")
        assert result.lead_seconds == 900.0

    def test_degenerate_grid_uses_zero_step(self) -> None:
        """step is 0 when len(scan_ts) <= 1: no bridge tolerance at all."""
        one_ts = pd.DatetimeIndex([_ONSET])
        at_onset = select_detection([(_ts("02:00"), _ts("02:00"))], one_ts, _ONSET, mode="lookback")
        assert at_onset.detected
        assert at_onset.lead_seconds == 0.0
        just_before = select_detection(
            [(_ts("01:50"), _ts("01:59"))], one_ts, _ONSET, mode="lookback"
        )
        assert not just_before.detected

    def test_no_qualifying_run_yields_none_fields(self) -> None:
        result = select_detection([(_ts("01:00"), _ts("01:10"))], _GRID, _ONSET, mode="lookback")
        assert not result.detected
        assert result.detection_time is None
        assert result.lead_seconds is None
        assert not result.bridged
        assert (result.n_runs_pre_onset, result.n_runs_at_or_after) == (1, 0)

    def test_empty_spans_yield_none_fields(self) -> None:
        result = select_detection([], _GRID, _ONSET, mode="lookback")
        assert not result.detected
        assert result.detection_time is None
        assert result.lead_seconds is None
        assert not result.bridged
        assert (result.n_runs_pre_onset, result.n_runs_at_or_after) == (0, 0)

    def test_unknown_mode_rejected(self) -> None:
        with pytest.raises(ValueError, match="mode"):
            select_detection([], _GRID, _ONSET, mode="sideways")


class TestSelectDetectionNoBridging:
    def test_lone_bridging_run_is_not_a_detection(self) -> None:
        """A run beginning pre-onset and continuing past onset is reported, not credited."""
        result = select_detection(
            [(_ts("01:50"), _ts("02:10"))], _GRID, _ONSET, mode="no_bridging"
        )
        assert not result.detected
        assert result.detection_time is None
        assert result.lead_seconds is None
        assert result.bridged
        assert (result.n_runs_pre_onset, result.n_runs_at_or_after) == (1, 0)

    def test_post_onset_run_after_bridging_run_detects_at_post_onset_start(self) -> None:
        spans = [(_ts("01:50"), _ts("02:10")), (_ts("02:20"), _ts("02:35"))]
        result = select_detection(spans, _GRID, _ONSET, mode="no_bridging")
        assert result.detected
        assert result.detection_time == _ts("02:20")
        assert result.lead_seconds == -1200.0
        assert result.bridged
        assert (result.n_runs_pre_onset, result.n_runs_at_or_after) == (1, 1)

    def test_run_starting_exactly_at_onset_counts_as_at_or_after(self) -> None:
        result = select_detection(
            [(_ts("02:00"), _ts("02:15"))], _GRID, _ONSET, mode="no_bridging"
        )
        assert result.detected
        assert result.detection_time == _ONSET
        assert result.lead_seconds == 0.0
        assert not result.bridged

    def test_partition_is_strictly_by_start(self) -> None:
        """Three-run fixture: recovered blip, bridging run, post-onset run."""
        spans = [
            (_ts("01:00"), _ts("01:10")),
            (_ts("01:55"), _ts("02:05")),
            (_ts("02:30"), _ts("02:45")),
        ]
        result = select_detection(spans, _GRID, _ONSET, mode="no_bridging")
        assert result.detected
        assert result.detection_time == _ts("02:30")
        assert result.bridged
        assert (result.n_runs_pre_onset, result.n_runs_at_or_after) == (2, 1)


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
