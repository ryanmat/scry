# Description: Tests for scry.eval.sweep baselines and per-week arithmetic: the named-baseline contract.
# Description: Pins the hand-checkable quantiles, the full-capture-quantile regression, and observed-span rates.

"""Tests for the margin-sweep primitives.

``compute_baselines`` is pinned three ways: the exact key set and each
baseline's value on a hand-checkable array (median split of ``1..8``); the
full-capture-quantile regression from spec section 11, where drift injected
into the held-out half makes ``q_deploy`` diverge from ``q_fit`` so the
held-out over-threshold count computed at the two baselines differs (fitting
``q_fit`` on the whole capture instead of its gapped earlier half collapses the
two counts and fails); and the eager-export / torch-free contract that keeps
``import scry.eval`` from pulling torch while exposing ``compute_baselines``.
``per_week`` is pinned to ``7.0 * count / span_days`` derived from the OBSERVED
span, so a 3.5-day fixture with two events reads 4.0 runs/week and no hardcoded
7-day extrapolation can survive.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest


class TestComputeBaselines:
    def test_exact_keys_and_hand_checkable_pin(self) -> None:
        # errors ascend 1..8 with matching ascending ends, so the single-resource
        # split is time_split: fit = [1,2,3,4], held-out = [5,6,7,8], q_deploy over
        # all eight. At the median: q_fit=2.5, q_eval=6.5, q_deploy=4.5, so
        # drift_x=6.5/2.5=2.6 and deploy_over_fit_x=4.5/2.5=1.8.
        from scry.eval.sweep import compute_baselines

        errors = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        ends = pd.date_range("2026-01-01T00:00:00Z", periods=8, freq="5min")

        result = compute_baselines(errors, ends, quantile=0.5, gap=0)

        assert set(result) == {"q_fit", "q_eval", "q_deploy", "drift_x", "deploy_over_fit_x"}
        assert result["q_fit"] == 2.5
        assert result["q_eval"] == 6.5
        assert result["q_deploy"] == 4.5
        assert result["drift_x"] == pytest.approx(2.6)
        assert result["deploy_over_fit_x"] == pytest.approx(1.8)

    def test_full_capture_quantile_counts_differ(self) -> None:
        # A synthetic healthy week: a flat-low first half and a drifted second
        # half. q_fit is the quantile of the gapped earlier (low) half; q_deploy
        # is the quantile over every window and so absorbs the drift. The
        # held-out over-threshold count read at q_fit far exceeds the count read
        # at q_deploy; fitting q_fit on the whole capture would collapse them.
        from scry.eval.scoring import per_resource_time_split
        from scry.eval.sweep import compute_baselines

        low = np.full(100, 0.1)
        drift = np.linspace(1.0, 2.0, 100)
        errors = np.concatenate([low, drift])
        ends = pd.date_range("2026-01-01T00:00:00Z", periods=200, freq="50min")
        gap = 6

        baselines = compute_baselines(errors, ends, quantile=0.99, gap=gap)

        # Drift makes the deployed baseline measurably larger than the fit baseline.
        assert baselines["q_deploy"] > baselines["q_fit"]
        assert baselines["q_deploy"] > 5.0 * baselines["q_fit"]

        _, held_out = per_resource_time_split(
            errors, ends, np.zeros(len(errors), dtype=int), gap=gap
        )
        count_at_fit = int(np.count_nonzero(held_out > baselines["q_fit"]))
        count_at_deploy = int(np.count_nonzero(held_out > baselines["q_deploy"]))

        assert count_at_fit != count_at_deploy
        assert count_at_fit > count_at_deploy


class TestPerWeek:
    def test_rate_from_observed_span(self) -> None:
        # Ends span exactly 3.5 days; two events read 7.0 * 2 / 3.5 == 4.0 per
        # week. A hardcoded 7-day span would read 2.0 and fail.
        from scry.eval.sweep import per_week

        ends = pd.date_range("2026-01-01T00:00:00Z", "2026-01-04T12:00:00Z", periods=8)

        assert per_week(2, ends) == 4.0


class TestExportsAndTorchFree:
    def test_compute_baselines_exports_eagerly(self) -> None:
        import scry.eval
        from scry.eval import sweep as sweep_module

        assert scry.eval.compute_baselines is sweep_module.compute_baselines
        assert "compute_baselines" in scry.eval.__all__
        assert "compute_baselines" in vars(scry.eval)  # eager, not resolved via __getattr__

    def test_sweep_module_imports_without_torch(self) -> None:
        proc = subprocess.run(
            [sys.executable, "-c", "import scry.eval.sweep, sys; assert 'torch' not in sys.modules"],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
