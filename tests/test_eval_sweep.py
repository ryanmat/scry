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

``run_sweep`` is pinned through a Candidate stub whose frames state their
errors outright: the spec-11 top-level key set (``arms`` equal to
``sweep_arms``, the gap read from the ScoreSet meta), detection at BOTH
baselines so swapping either moves the credited time, the no-bridging rule
that a pre-onset run is never credited, the three DESIGN-addendum caveats, the
provenance block's empty rubric and case paths beside the MarginSweep policy,
and the ValueError for an incident resource with no capture windows.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest

if TYPE_CHECKING:
    from scry.eval.candidate import ScoreSet
    from scry.eval.scoring import ScoringGrid


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


# The healthy week run_sweep composes: ten windows at 1.0 then ten at 2.0, so
# the fit half reads q_fit 1.0 and the whole week reads q_deploy 2.0.
HEALTHY = (
    np.concatenate([np.full(10, 1.0), np.full(10, 2.0)]),
    pd.date_range("2026-01-01T00:00:00Z", periods=20, freq="1h"),
)


def _capture_day() -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Two 1.5 excursions and one 2.5 excursion, hourly, around T0 at 06:00.

    At q_fit (1.0) the 04:00-06:00 run starts before T0 and is still over
    threshold at it, and the first run starting at or after T0 is the 08:00
    one; at q_deploy (2.0) only the 14:00 run clears at all.
    """
    errors = np.full(18, 0.5)
    errors[[4, 5, 6, 8, 9, 10]] = 1.5
    errors[[14, 15, 16]] = 2.5
    return errors, pd.date_range("2026-01-02T00:00:00Z", periods=18, freq="1h")


CAPTURE = _capture_day()


def _frame(series: dict[str, tuple[np.ndarray, pd.DatetimeIndex]]) -> pd.DataFrame:
    """A capture frame whose ``value`` column IS the error the stub scores back out."""
    return pd.DataFrame(
        [
            {"resource_id": rid, "timestamp": ts, "value": float(err)}
            for rid, (errors, ends) in series.items()
            for err, ts in zip(errors, ends, strict=True)
        ]
    )


class StatedErrors:
    """A Candidate stub: it reads a frame's stated errors back out as a ScoreSet.

    ``seq_len`` and ``step`` are the meta fields run_sweep's
    ``gap = ceil(seq_len / step)`` reads, and ``model_path`` is a real file so
    the provenance sha256 is computable.
    """

    def __init__(self, model_path: str, seq_len: int = 4, step: int = 2) -> None:
        self.meta = dict(model_path=model_path, profile="aro_node", seq_len=seq_len, step=step)

    def score(self, df_long: pd.DataFrame, grid: ScoringGrid) -> ScoreSet:
        from scry.eval.candidate import ScoreSet

        return ScoreSet(
            errors=df_long["value"].to_numpy(dtype=float),
            end_times=pd.DatetimeIndex(df_long["timestamp"]),
            resource_ids=df_long["resource_id"].to_numpy(),
            grid=grid,
            meta=dict(self.meta),
        )


def _sweep(
    tmp_path: Path, *, margins: Sequence[float] = (1.0,), primary: str | None = "T0",
    seq_len: int = 4, uncaptured: str | None = None,
) -> dict[str, Any]:
    """run_sweep over HEALTHY and CAPTURE for node-a, T0 at 06:00 and T2 at 12:00.

    ``uncaptured`` names a second incident resource added to the healthy week
    and the labels but NOT to the capture frame.
    """
    from scry.eval.labels import LabelCase, LabelSet
    from scry.eval.scoring import ScoringGrid
    from scry.eval.sweep import run_sweep

    model = tmp_path / "keeper.pt"
    model.write_bytes(b"keeper")
    ends = CAPTURE[1]
    resources = ["node-a"] + ([uncaptured] if uncaptured is not None else [])
    onsets = {"T0": ends[6], "T2": ends[12]}
    # LabelCase fields: resource_id, role, type, onsets, primary_onset, end, notes.
    cases = [
        LabelCase(rid, "incident", "cpu", onsets, primary, ends[-1], None) for rid in resources
    ]
    return run_sweep(
        StatedErrors(str(model), seq_len=seq_len),
        _frame({rid: HEALTHY for rid in resources}),
        _frame({"node-a": CAPTURE}),
        LabelSet(version=2, capture=None, cases=cases),
        list(margins),
        quantile=0.99,
        sustain=3,
        grid=ScoringGrid(label="sweep", step_samples=2),
    )


class TestRunSweep:
    def test_result_carries_exactly_the_contract_keys(self, tmp_path: Path) -> None:
        from scry.eval.sweep import ARM_BASELINES, compute_baselines, sweep_arms

        result = _sweep(tmp_path, margins=[1.0, 2.0])

        assert set(result) == {
            "provenance", "grid", "quantile", "sustain", "margins",
            "per_resource_baselines", "arm_baselines", "arms", "detection", "caveats",
        }
        assert result["arm_baselines"] == ARM_BASELINES
        assert result["quantile"] == 0.99
        assert result["sustain"] == 3
        assert result["margins"] == [1.0, 2.0]
        assert result["grid"] == {"label": "sweep", "step_samples": 2, "cadence_seconds": None}
        baselines = compute_baselines(*HEALTHY, quantile=0.99, gap=2)
        assert result["per_resource_baselines"] == {"node-a": baselines}
        assert result["arms"] == sweep_arms(
            {"node-a": HEALTHY}, {"node-a": CAPTURE},
            per_resource_baselines={"node-a": baselines}, margins=[1.0, 2.0], sustain=3, gap=2,
        )
        # gap = ceil(seq_len / step) from the ScoreSet meta: at seq_len 4 / step 2
        # the holdout drops 2 of the 10 later windows and spans 12:00..19:00; at
        # seq_len 5 it drops 3 and spans 13:00..19:00.
        assert result["arms"]["healthy_holdout"]["1.0"]["span_days"] == pytest.approx(7 / 24)
        wider = _sweep(tmp_path, seq_len=5)["arms"]["healthy_holdout"]["1.0"]
        assert wider["span_days"] == pytest.approx(6 / 24)

    def test_detection_reported_at_both_baselines(self, tmp_path: Path) -> None:
        detection = _sweep(tmp_path, margins=[1.0, 2.0])["detection"]

        assert set(detection) == {"detection_at_deploy", "detection_at_fit"}
        assert set(detection["detection_at_fit"]) == {"1.0", "2.0"}
        # At m=1.0 the fit baseline (1.0) sees the 1.5 excursions and credits the
        # first run at or after T0, 08:00; the deploy baseline (2.0) waits for
        # the 2.5 run at 14:00. Swapping the two baselines moves both times.
        at_fit = detection["detection_at_fit"]["1.0"]["node-a"]
        at_deploy = detection["detection_at_deploy"]["1.0"]["node-a"]
        assert at_fit["detection_time"] == "2026-01-02T08:00:00Z"
        assert at_fit["lead_seconds"] == -7200.0
        assert at_deploy["detection_time"] == "2026-01-02T14:00:00Z"
        assert at_deploy["lead_seconds"] == -28800.0
        # m=2.0 lifts the fit threshold onto the 2.5 run and the deploy
        # threshold (4.0) above every window: undetected, with every field.
        fit_at_2 = detection["detection_at_fit"]["2.0"]["node-a"]
        assert fit_at_2["detection_time"] == "2026-01-02T14:00:00Z"
        assert detection["detection_at_deploy"]["2.0"]["node-a"] == {
            "detected": False, "detection_time": None, "lead_seconds": None,
            "bridged": False, "n_runs_pre_onset": 0, "n_runs_at_or_after": 0,
        }

    def test_pre_onset_run_is_never_credited(self, tmp_path: Path) -> None:
        # No primary_onset falls back to T0 (06:00); anchoring at the T2 this
        # fixture also carries would credit 14:00 instead of 08:00.
        at_fit = _sweep(tmp_path, primary=None)["detection"]["detection_at_fit"]["1.0"]["node-a"]

        # The 04:00-06:00 run starts pre-onset and is still over threshold at
        # T0: no_bridging reports it and credits the 08:00 run instead.
        assert at_fit["bridged"] is True
        assert at_fit["n_runs_pre_onset"] == 1
        assert at_fit["n_runs_at_or_after"] == 2
        assert at_fit["detected"] is True
        assert at_fit["detection_time"] == "2026-01-02T08:00:00Z"

    def test_detection_anchors_at_the_primary_onset(self, tmp_path: Path) -> None:
        # primary_onset T2 (12:00) anchors detection there: both 1.5 runs start
        # before it and neither is over threshold at it, so the 14:00 run is
        # credited unbridged. Anchoring at T0 instead credits 08:00.
        at_fit = _sweep(tmp_path, primary="T2")["detection"]["detection_at_fit"]["1.0"]["node-a"]

        assert at_fit["detection_time"] == "2026-01-02T14:00:00Z"
        assert at_fit["bridged"] is False
        assert at_fit["n_runs_pre_onset"] == 2
        assert at_fit["n_runs_at_or_after"] == 1

    def test_detection_thresholds_are_per_resource(self, tmp_path: Path) -> None:
        from scry.eval.labels import LabelCase, LabelSet
        from scry.eval.scoring import ScoringGrid
        from scry.eval.sweep import run_sweep

        # node-b's healthy week runs at twice node-a's (q_fit 2.0), so at m=1.0
        # the same capture credits node-a at 08:00 and node-b only at the 2.5
        # run, 14:00. A threshold pooled across resources moves one of them.
        model = tmp_path / "keeper.pt"
        model.write_bytes(b"keeper")
        ends = CAPTURE[1]
        cases = [
            LabelCase(rid, "incident", "cpu", {"T0": ends[6]}, "T0", ends[-1], None)
            for rid in ("node-a", "node-b")
        ]
        result = run_sweep(
            StatedErrors(str(model)),
            _frame({"node-a": HEALTHY, "node-b": (2 * HEALTHY[0], HEALTHY[1])}),
            _frame({"node-a": CAPTURE, "node-b": CAPTURE}),
            LabelSet(version=2, capture=None, cases=cases),
            [1.0],
            quantile=0.99,
            sustain=3,
            grid=ScoringGrid(label="sweep", step_samples=2),
        )

        at_fit = result["detection"]["detection_at_fit"]["1.0"]
        assert result["per_resource_baselines"]["node-b"]["q_fit"] == pytest.approx(2.0)
        assert at_fit["node-a"]["detection_time"] == "2026-01-02T08:00:00Z"
        assert at_fit["node-b"]["detection_time"] == "2026-01-02T14:00:00Z"

    def test_caveats_embed_the_three_design_statements(self, tmp_path: Path) -> None:
        from scry.eval.sweep import CAVEATS

        caveats = _sweep(tmp_path)["caveats"]

        assert caveats == list(CAVEATS)
        assert len(caveats) == 3
        # (a) held-out FP flatness at the deployed baseline, (b) raises
        # non-monotone in m, (c) the stride artifact behind the flat lead.
        assert any("0.00 runs/wk" in caveat for caveat in caveats)
        assert any("non-monotone" in caveat for caveat in caveats)
        assert any("stride" in caveat for caveat in caveats)

    def test_provenance_records_the_margin_sweep_policy(self, tmp_path: Path) -> None:
        from scry.eval.sweep import ARM_BASELINES

        provenance = _sweep(tmp_path, margins=[1.0, 2.0])["provenance"]

        # The sweep evaluates no rubric and receives DataFrames, not paths.
        assert provenance["rubric_path"] == ""
        assert provenance["rubric_version"] == 0
        assert provenance["cases"] == {}
        assert provenance["threshold_policy"] == {
            "type": "MarginSweep", "quantile": 0.99, "sustain": 3,
            "margins": [1.0, 2.0], "arm_baselines": ARM_BASELINES,
        }
        assert provenance["detection_mode"] == "no_bridging"
        assert provenance["sustains"] == [3]
        # Model identity and window geometry come from the ScoreSet meta.
        assert provenance["model_path"] == str(tmp_path / "keeper.pt")
        assert provenance["model_sha256"] == hashlib.sha256(b"keeper").hexdigest()
        assert provenance["profile"] == "aro_node"
        assert provenance["seq_len"] == 4

    def test_incident_resource_without_capture_windows_raises(self, tmp_path: Path) -> None:
        # node-b has healthy baselines but no capture windows to detect over.
        with pytest.raises(ValueError, match="no capture windows.*'node-b'"):
            _sweep(tmp_path, uncaptured="node-b")


class TestExportsAndTorchFree:
    def test_sweep_members_export_eagerly(self) -> None:
        import scry.eval
        from scry.eval import sweep as sweep_module

        for name in ("compute_baselines", "run_sweep"):
            assert getattr(scry.eval, name) is getattr(sweep_module, name)
            assert name in scry.eval.__all__
            assert name in vars(scry.eval)  # eager, not resolved via __getattr__

    def test_sweep_module_imports_without_torch(self) -> None:
        proc = subprocess.run(
            [sys.executable, "-c", "import scry.eval.sweep, sys; assert 'torch' not in sys.modules"],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
