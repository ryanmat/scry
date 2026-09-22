# Description: Tests for scry.eval.sweep baselines, per-week arithmetic, and the rate arms.
# Description: Pins the hand-checkable quantiles, the two spec-11 regressions, and observed-span rates.

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

``sweep_arms`` is pinned on the same drift fixture plus small hand-built
excursions: ``ARM_BASELINES`` is the code path (patching an entry moves the
thresholds), the holdout arm counts at ``q_fit``, and the gapped split drops a
boundary leak that an ungapped split would count.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest


def drift_week() -> tuple[np.ndarray, pd.DatetimeIndex]:
    """The spec-11 healthy week: a flat-low first half and a drifted second half.

    The drift lands entirely in the held-out half, so ``q_deploy`` (every
    window) is measurably larger than ``q_fit`` (the gapped earlier half).
    """
    errors = np.concatenate([np.full(100, 0.1), np.linspace(1.0, 2.0, 100)])
    return errors, pd.date_range("2026-01-01T00:00:00Z", periods=200, freq="50min")


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

        errors, ends = drift_week()
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


UNIT = {"a": {"q_fit": 1.0, "q_deploy": 1.0}}  # both baselines 1.0: the margin IS the threshold


class TestSweepArms:
    def test_arm_baselines_pin_is_the_code_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from scry.eval import sweep

        assert sweep.ARM_BASELINES == {
            "healthy_holdout": "q_fit",
            "healthy_insample": "q_deploy",
            "capture_day": "q_deploy",
            "detection_at_deploy": "q_deploy",
            "detection_at_fit": "q_fit",
        }
        ends = pd.date_range("2026-01-01T00:00:00Z", periods=8, freq="1h")
        scored = {"a": (np.full(8, 0.1), ends)}
        args = {"per_resource_baselines": {"a": {"q_fit": 1.0, "q_deploy": 4.0}}, "margins": [2.0]}

        arms = sweep.sweep_arms(scored, scored, sustain=3, gap=0, **args)

        # Each arm's threshold is margin * baselines[rid][ARM_BASELINES[arm]].
        assert arms["healthy_holdout"]["2.0"]["threshold_by_resource"] == {"a": 2.0}
        assert arms["healthy_insample"]["2.0"]["threshold_by_resource"] == {"a": 8.0}
        assert arms["capture_day"]["2.0"]["threshold_by_resource"] == {"a": 8.0}

        # Changing an entry changes the numbers, so the mapping is not just a record.
        patched = {**sweep.ARM_BASELINES, "healthy_insample": "q_fit"}
        monkeypatch.setattr(sweep, "ARM_BASELINES", patched)
        swapped = sweep.sweep_arms(scored, scored, sustain=3, gap=0, **args)
        assert swapped["healthy_insample"]["2.0"]["threshold_by_resource"] == {"a": 2.0}

    def test_holdout_arm_counts_at_the_fit_baseline(self) -> None:
        # Spec 11 full-capture-quantile regression through the arm: the holdout
        # run count is the one read at q_fit, never the one read at q_deploy.
        from scry.eval.detection import anomaly_runs
        from scry.eval.scoring import per_resource_time_split
        from scry.eval.sweep import compute_baselines, sweep_arms

        errors, ends = drift_week()
        gap = 6
        baselines = compute_baselines(errors, ends, quantile=0.99, gap=gap)
        ids = np.zeros(len(errors), dtype=int)
        _, held_out = per_resource_time_split(errors, ends, ids, gap=gap)
        at_fit = len(anomaly_runs(held_out > baselines["q_fit"], 3))
        at_deploy = len(anomaly_runs(held_out > baselines["q_deploy"], 3))
        assert at_fit != at_deploy

        arms = sweep_arms(
            {"a": (errors, ends)}, {}, per_resource_baselines={"a": baselines},
            margins=[1.0], sustain=3, gap=gap,
        )

        assert arms["healthy_holdout"]["1.0"]["n_runs"] == at_fit

    def test_gapped_split_drops_the_boundary_leak(self) -> None:
        # 40 flat-low windows with a 4-window spike occupying exactly the gap
        # between the fit half (0..19) and the gapped holdout (24..39): only
        # boundary-leaked windows are over threshold.
        from scry.eval.sweep import sweep_arms

        errors = np.full(40, 0.1)
        errors[20:24] = 10.0
        ends = pd.date_range("2026-01-01T00:00:00Z", periods=40, freq="1h")
        healthy = {"a": (errors, ends)}
        args = {"per_resource_baselines": UNIT, "margins": [1.5], "sustain": 3}

        gapped = sweep_arms(healthy, {}, gap=4, **args)["healthy_holdout"]["1.5"]
        ungapped = sweep_arms(healthy, {}, gap=0, **args)["healthy_holdout"]["1.5"]

        assert gapped["n_runs"] == 0
        assert ungapped["n_runs"] == 1
        # The holdout spans windows 24..39, 15 hours; an over-wide gap leaves 11.
        assert gapped["span_days"] == pytest.approx(15 / 24)

    def test_arm_values_carry_exactly_the_five_fields(self) -> None:
        from scry.eval.sweep import sweep_arms

        errors = np.full(20, 0.1)
        errors[2:5] = 5.0
        errors[10:14] = 5.0
        errors[17] = 5.0  # a raise too short to sustain
        ends = pd.date_range("2026-01-01T00:00:00Z", periods=20, freq="1h")

        arms = sweep_arms(
            {}, {"a": (errors, ends)}, per_resource_baselines=UNIT,
            margins=[1.5, 2], sustain=3, gap=0,
        )

        assert set(arms) == {"healthy_holdout", "healthy_insample", "capture_day"}
        day = arms["capture_day"]
        assert set(day) == {"1.5", "2.0"}  # margins keyed str(float(m))
        rates = {"n_runs", "runs_per_week", "raises_per_week", "span_days"}
        assert set(day["1.5"]) == rates | {"threshold_by_resource"}
        # Two sustained runs, three raises: the two accountings differ.
        assert day["1.5"]["n_runs"] == 2
        assert day["1.5"]["raises_per_week"] == pytest.approx(1.5 * day["1.5"]["runs_per_week"])

    def test_rates_come_from_the_observed_span(self) -> None:
        # 15 six-hourly ends span exactly 3.5 days with two sustained runs and
        # one short raise: 7 * 2 / 3.5 == 4.0 and 7 * 3 / 3.5 == 6.0. A lone
        # window spans no time and reports None instead of dividing by zero.
        from scry.eval.sweep import sweep_arms

        errors = np.full(15, 0.1)
        errors[0:3] = 5.0
        errors[5:9] = 5.0
        errors[12] = 5.0
        ends = pd.date_range("2026-01-01T00:00:00Z", periods=15, freq="6h")
        args = {"per_resource_baselines": UNIT, "margins": [1.5], "gap": 0}

        day = sweep_arms({}, {"a": (errors, ends)}, sustain=3, **args)["capture_day"]["1.5"]

        assert day["span_days"] == pytest.approx(3.5)
        assert day["runs_per_week"] == pytest.approx(4.0)
        assert day["raises_per_week"] == pytest.approx(6.0)

        flat = sweep_arms({}, {"a": (np.array([5.0]), ends[:1])}, sustain=1, **args)
        lone = flat["capture_day"]["1.5"]

        assert lone["n_runs"] == 1  # scanned, not skipped
        assert lone["span_days"] == 0.0
        assert lone["runs_per_week"] is None
        assert lone["raises_per_week"] is None

    def test_capture_resource_without_baselines_raises(self) -> None:
        from scry.eval.sweep import sweep_arms

        ends = pd.date_range("2026-01-01T00:00:00Z", periods=4, freq="1h")
        capture = {"a": (np.full(4, 0.1), ends), "b": (np.full(4, 0.1), ends)}

        with pytest.raises(ValueError, match="'b'"):
            sweep_arms({}, capture, per_resource_baselines=UNIT, margins=[1.5], sustain=3, gap=0)


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
