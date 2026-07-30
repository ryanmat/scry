# Description: Tests for scry.eval scoring and candidates: keeper windowing, grids, policies.
# Description: Pins the validate_incident alias, the lazy exports, the cadence rule, and resolution.

"""Tests for the eval scoring primitives, candidates, and threshold policies.

``windows_for_keeper`` is pinned through the per-resource window-count
arithmetic on a synthetic capture, the alias identity that keeps
``validate_incident._windows_for_keeper`` working, and the lazy-export
contract: the name resolves from ``scry.eval`` while a bare package import
still leaves torch unimported. ``ScoringGrid`` is pinned through the serving
preset, identity selection without a cadence, the wall-clock (floored, not
span-anchored) tick lattice, per-resource per-tick greatest-end selection,
and single-keep dedup when one window is the greatest for several ticks.
``ReconstructionCandidate`` is pinned through ScoreSet provenance meta, the
missing-model contract, the two-grid count divergence, and the empty capture.
The threshold policies are pinned through per-source resolution, the
per-resource-split fit half against a pooled-split counterfactual,
JSON-serializable describe() provenance, and the empty-fit ValueError that
keeps the dead fallback dead. ``PerResourceMargin`` is additionally pinned
bit-identically (float.hex) against the real bake on one calibration, through
the gated-fleet hygiene fallback routing, the eligibility reasons in
describe(), and the positive-finite margin guard. ``ServingBlock`` is pinned
through the baked-global resolution, the patched per_resource map beating the
global for the mapped id, additive-key tolerance, the missing-block error,
the two-scale per-resource-vs-pooled exceedance miniature, and the
lazy-export completeness of every candidate member.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import bake_serving_threshold as bake_mod
import numpy as np
import pandas as pd
import pytest
import torch
import validate_incident as vi
from synth import PROFILE, SEQ_LEN, gated_fleet_csv, gen_capture, write_csv

from scry.data.fetcher import fetch_full_capture
from scry.model.checkpoint import load_keeper


@pytest.fixture(scope="module")
def fleet_df(tmp_path_factory: pytest.TempPathFactory) -> pd.DataFrame:
    tmp = tmp_path_factory.mktemp("scoring_fleet")
    node_a, _ = gen_capture("node-a", 200, seed=61)
    node_b, _ = gen_capture("node-b", 200, seed=62)
    csv = write_csv(pd.concat([node_a, node_b], ignore_index=True), tmp / "fleet.csv")

    async def _fetch() -> pd.DataFrame:
        return await fetch_full_capture(csv, profile=PROFILE)

    return asyncio.run(_fetch())


class TestWindowsForKeeper:
    def test_window_count_matches_arithmetic_per_resource(
        self, keeper_path: str, fleet_df: pd.DataFrame
    ) -> None:
        from scry.eval.scoring import windows_for_keeper

        keeper = load_keeper(keeper_path)
        step = 10
        windows = windows_for_keeper(fleet_df, keeper, SEQ_LEN, step)

        expected_per_resource = (200 - SEQ_LEN) // step + 1
        ids = windows.resource_ids.astype(str)
        assert (ids == "node-a").sum() == expected_per_resource
        assert (ids == "node-b").sum() == expected_per_resource
        assert len(windows.end_times) == 2 * expected_per_resource
        assert windows.x_num.shape[0] == 2 * expected_per_resource

    def test_alias_is_the_eval_implementation(self) -> None:
        from scry.eval import scoring

        assert vi._windows_for_keeper is scoring.windows_for_keeper


class TestLazyExport:
    def test_windows_for_keeper_resolves_from_package_root(self) -> None:
        import scry.eval
        from scry.eval import scoring, windows_for_keeper

        assert windows_for_keeper is scoring.windows_for_keeper
        assert "windows_for_keeper" in dir(scry.eval)
        assert "windows_for_keeper" in scry.eval.__all__

    def test_package_root_import_stays_torch_free(self) -> None:
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "import scry.eval, sys; assert 'torch' not in sys.modules",
            ],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr


class TestValidateIncidentSource:
    def test_private_definition_removed_from_script(self) -> None:
        # The script keeps only the alias assignment, not a local implementation.
        source = Path(vi.__file__).read_text()
        assert "def _windows_for_keeper" not in source
        assert "_windows_for_keeper = windows_for_keeper" in source


def _minute_ends(start: str, n: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="1min")


class TestScoringGrid:
    def test_frozen_dataclass(self) -> None:
        from scry.eval.scoring import ScoringGrid

        grid = ScoringGrid(label="offline-20min", step_samples=10)
        with pytest.raises(AttributeError):
            grid.label = "other"  # type: ignore[misc]
        assert grid.cadence is None

    def test_serving_preset(self) -> None:
        from scry.eval.scoring import SERVING_GRID, ScoringGrid

        assert SERVING_GRID == ScoringGrid(
            label="serving-10min", step_samples=1, cadence=pd.Timedelta(minutes=10)
        )

    def test_no_cadence_keeps_every_window(self) -> None:
        from scry.eval.scoring import ScoringGrid

        ends = _minute_ends("2026-01-01T16:07:00Z", 30)
        ids = np.array(["node-a"] * 15 + ["node-b"] * 15)
        grid = ScoringGrid(label="native", step_samples=1)
        assert grid.select_indices(ends, ids).tolist() == list(range(30))

    def test_wall_clock_tick_anchoring(self) -> None:
        # Ends every minute from 16:07; the tick lattice floors to 16:00, so the
        # kept windows sit on 16:10/16:20/16:30, never on the 16:07 span anchor.
        from scry.eval.scoring import ScoringGrid

        ends = _minute_ends("2026-01-01T16:07:00Z", 30)  # 16:07 .. 16:36
        ids = np.array(["node-a"] * 30)
        grid = ScoringGrid(label="serving-10min", step_samples=1, cadence=pd.Timedelta(minutes=10))
        kept = ends[grid.select_indices(ends, ids)]
        assert list(kept) == [
            pd.Timestamp("2026-01-01T16:10:00Z"),
            pd.Timestamp("2026-01-01T16:20:00Z"),
            pd.Timestamp("2026-01-01T16:30:00Z"),
        ]

    def test_lattice_invariant_to_span_offset(self) -> None:
        # Shifting the span start by a non-cadence offset (+3 minutes) leaves
        # the tick lattice unchanged: both captures keep the same wall-clock ends.
        from scry.eval.scoring import ScoringGrid

        grid = ScoringGrid(label="serving-10min", step_samples=1, cadence=pd.Timedelta(minutes=10))
        ids = np.array(["node-a"] * 30)
        base = _minute_ends("2026-01-01T16:07:00Z", 30)
        shifted = _minute_ends("2026-01-01T16:10:00Z", 30)
        kept_base = set(base[grid.select_indices(base, ids)])
        kept_shifted = set(shifted[grid.select_indices(shifted, ids)])
        lattice = {
            pd.Timestamp("2026-01-01T16:10:00Z"),
            pd.Timestamp("2026-01-01T16:20:00Z"),
            pd.Timestamp("2026-01-01T16:30:00Z"),
        }
        assert lattice <= kept_base
        assert lattice <= kept_shifted

    def test_per_resource_per_tick_greatest_end(self) -> None:
        # node-a ends on even minutes 16:00..16:18, node-b on odd 16:01..16:19,
        # interleaved in emission order. Ticks 16:00 and 16:10 each keep, per
        # resource, only the single greatest end <= tick.
        from scry.eval.scoring import ScoringGrid

        ends = _minute_ends("2026-01-01T16:00:00Z", 20)
        ids = np.array(["node-a", "node-b"] * 10)
        grid = ScoringGrid(label="serving-10min", step_samples=1, cadence=pd.Timedelta(minutes=10))
        idx = grid.select_indices(ends, ids)
        kept_a = {ends[i] for i in idx if ids[i] == "node-a"}
        kept_b = {ends[i] for i in idx if ids[i] == "node-b"}
        assert kept_a == {
            pd.Timestamp("2026-01-01T16:00:00Z"),
            pd.Timestamp("2026-01-01T16:10:00Z"),
        }
        assert kept_b == {pd.Timestamp("2026-01-01T16:09:00Z")}

    def test_sparse_window_kept_once_across_ticks(self) -> None:
        # One window is the greatest end for ticks 16:10, 16:20, and 16:30; it
        # is kept exactly once (the same stale window served at several polls
        # is still one scored window offline).
        from scry.eval.scoring import ScoringGrid

        ends = pd.DatetimeIndex(
            [pd.Timestamp("2026-01-01T16:05:00Z"), pd.Timestamp("2026-01-01T16:35:00Z")]
        )
        ids = np.array(["node-a", "node-a"])
        grid = ScoringGrid(label="serving-10min", step_samples=1, cadence=pd.Timedelta(minutes=10))
        assert grid.select_indices(ends, ids).tolist() == [0]


class TestPerResourceTimeSplit:
    # Windows are seq_len=30 one-minute samples emitted every 5 minutes, so two
    # windows of one resource share raw samples iff their ends differ by less
    # than 30 minutes. Error values encode (resource, end-minutes) so the split
    # halves can be decoded back to windows.
    _SEQ_MINUTES = 30
    _STEP_MINUTES = 5
    _GAP = -(-30 // 5)  # ceil(seq_len / step) with seq_len=30, step=5

    def test_gap_expression_matches_ceil(self) -> None:
        assert -(-30 // 10) == 3
        assert -(-30 // 5) == 6
        assert self._GAP == 6

    def _interleaved_fixture(self) -> tuple[np.ndarray, pd.DatetimeIndex, np.ndarray]:
        base = pd.Timestamp("2026-01-01T00:00:00Z")
        minutes = [m for step in range(24) for m in (step * 5, step * 5)]
        ends = pd.DatetimeIndex([base + pd.Timedelta(minutes=m) for m in minutes])
        ids = np.array(["node-a", "node-b"] * 24)
        codes = {"node-a": 10_000.0, "node-b": 20_000.0}
        errors = np.array([codes[rid] + m for rid, m in zip(ids, minutes)])
        return errors, ends, ids

    def _decode(self, values: np.ndarray, resource: str) -> set[float]:
        lo = 10_000.0 if resource == "node-a" else 20_000.0
        return {v - lo for v in values if lo <= v < lo + 10_000.0}

    def test_single_resource_bit_identity(self) -> None:
        from scry.eval.scoring import per_resource_time_split
        from scry.model.reconstruction import time_split

        rng = np.random.default_rng(71)
        errors = rng.uniform(0.01, 0.5, 40)
        ends = pd.date_range("2026-01-01T00:00:00Z", periods=40, freq="5min")
        shuffle = rng.permutation(40)

        fit_ref, eval_ref = time_split(errors[shuffle], ends[shuffle], gap=self._GAP)
        fit, eval_ = per_resource_time_split(
            errors[shuffle], ends[shuffle], np.array(["solo"] * 40), gap=self._GAP
        )
        assert np.array_equal(fit, fit_ref)
        assert np.array_equal(eval_, eval_ref)
        assert fit.dtype == fit_ref.dtype
        assert eval_.dtype == eval_ref.dtype

    def test_pooled_split_leaks_and_per_resource_does_not(self) -> None:
        # The defect and the fix in one test: the pooled gap spans only
        # ~gap/n_resources distinct time steps, so pooled fit and eval windows
        # of one resource still share raw samples; the per-resource split
        # separates every resource's halves by more than a window length.
        from scry.eval.scoring import per_resource_time_split
        from scry.model.reconstruction import time_split

        errors, ends, ids = self._interleaved_fixture()

        pooled_fit, pooled_eval = time_split(errors, ends, gap=self._GAP)
        pooled_leaks = False
        for resource in ("node-a", "node-b"):
            fit_ends = self._decode(pooled_fit, resource)
            eval_ends = self._decode(pooled_eval, resource)
            if fit_ends and eval_ends:
                pooled_leaks |= min(eval_ends) - max(fit_ends) < self._SEQ_MINUTES
        assert pooled_leaks

        fit, eval_ = per_resource_time_split(errors, ends, ids, gap=self._GAP)
        for resource in ("node-a", "node-b"):
            fit_ends = self._decode(fit, resource)
            eval_ends = self._decode(eval_, resource)
            assert len(fit_ends) == 12
            assert len(eval_ends) == 6
            assert min(eval_ends) - max(fit_ends) >= self._SEQ_MINUTES


class TestReconstructionCandidate:
    def test_score_returns_scoreset_with_provenance_meta(
        self, keeper_path: str, fleet_df: pd.DataFrame
    ) -> None:
        from scry.eval.candidate import ReconstructionCandidate, ScoreSet
        from scry.eval.scoring import ScoringGrid

        grid = ScoringGrid(label="offline-20min", step_samples=10)
        score_set = ReconstructionCandidate(keeper_path, profile=PROFILE).score(fleet_df, grid)

        assert isinstance(score_set, ScoreSet)
        assert score_set.errors.dtype == np.float64
        assert score_set.errors.ndim == 1
        n = score_set.errors.shape[0]
        assert len(score_set.end_times) == n
        assert score_set.resource_ids.shape[0] == n
        assert n == 2 * ((200 - SEQ_LEN) // 10 + 1)
        assert score_set.grid is grid

        expected_sha = hashlib.sha256(Path(keeper_path).read_bytes()).hexdigest()
        keeper = load_keeper(keeper_path)
        assert score_set.meta["model_path"] == keeper_path
        assert score_set.meta["model_sha256"] == expected_sha
        assert score_set.meta["profile"] == PROFILE
        assert score_set.meta["seq_len"] == int(keeper.config["seq_len"])
        assert score_set.meta["step"] == 10
        assert score_set.meta["device"] == keeper.device

    def test_missing_model_path_raises_file_not_found(self) -> None:
        from scry.eval.candidate import ReconstructionCandidate
        from scry.eval.scoring import ScoringGrid

        candidate = ReconstructionCandidate("models/does_not_exist.pt", profile=PROFILE)
        with pytest.raises(FileNotFoundError):
            candidate.score(pd.DataFrame(), ScoringGrid(label="native", step_samples=1))

    def test_grid_changes_over_threshold_counts(self, keeper_path: str, tmp_path: Path) -> None:
        # The 2-vs-3 finding in miniature: one excursion, two grids, different
        # over-threshold window counts, each keyed by its grid label. The spike
        # spans 55 samples so the over-window end interval [600, 683] holds 8
        # offline windows (ends on 9 mod 10) vs 9 serving-lattice windows (ends
        # on 0 mod 10); the q99 threshold additionally leaves one healthy
        # quantile-tail window over per grid, so the counts differ by one
        # structurally either way.
        from scry.eval.candidate import ReconstructionCandidate
        from scry.eval.scoring import SERVING_GRID, ScoringGrid

        df, _ = gen_capture("node-a", 700, seed=81, spike=(600, 655, 40.0))
        csv = write_csv(df, tmp_path / "excursion.csv")
        df_long = asyncio.run(fetch_full_capture(csv, profile=PROFILE))

        candidate = ReconstructionCandidate(keeper_path, profile=PROFILE)
        offline = ScoringGrid(label="offline-20min", step_samples=10)

        counts: dict[str, int] = {}
        thresholds: dict[str, float] = {}
        for grid in (offline, SERVING_GRID):
            score_set = candidate.score(df_long, grid)
            spike_start = df_long["timestamp"].min() + pd.Timedelta(minutes=600)
            healthy = score_set.errors[score_set.end_times < spike_start]
            thresholds[grid.label] = float(np.quantile(healthy, 0.99))
            counts[grid.label] = int((score_set.errors > thresholds[grid.label]).sum())

        assert set(counts) == {"offline-20min", "serving-10min"}
        assert counts["offline-20min"] != counts["serving-10min"]

    def test_empty_input_scores_to_empty_arrays(self, keeper_path: str) -> None:
        from scry.eval.candidate import ReconstructionCandidate
        from scry.eval.scoring import ScoringGrid

        empty = pd.DataFrame(columns=["resource_id", "metric_name", "timestamp", "value"])
        score_set = ReconstructionCandidate(keeper_path, profile=PROFILE).score(
            empty, ScoringGrid(label="native", step_samples=1)
        )
        assert score_set.errors.shape == (0,)
        assert len(score_set.end_times) == 0
        assert score_set.resource_ids.shape == (0,)


def _policy_score_set(errors: np.ndarray, ends: pd.DatetimeIndex, ids: np.ndarray):
    # A hand-built ScoreSet for policy tests; meta carries the seq_len=30 /
    # step=5 the gap convention derives from (ceil(30 / 5) = 6).
    from scry.eval.candidate import ScoreSet
    from scry.eval.scoring import ScoringGrid

    return ScoreSet(
        errors=np.asarray(errors, dtype=np.float64),
        end_times=ends,
        resource_ids=np.asarray(ids),
        grid=ScoringGrid(label="native", step_samples=5),
        meta={"seq_len": 30, "step": 5},
    )


class TestThresholdPolicies:
    def test_global_override_resolves_every_resource(self) -> None:
        from scry.eval.candidate import GlobalOverride

        policy = GlobalOverride(0.25)
        assert policy.resolve("anything") == (0.25, "override")
        assert policy.resolve("worker-1") == (0.25, "override")
        described = policy.describe()
        assert described["type"] == "GlobalOverride"
        assert described["threshold"] == 0.25

    def test_reference_quantile_pools_the_full_reference(self) -> None:
        # Errors increase over time, so an accidental fit/eval split inside the
        # policy would drag the q99 threshold far below the full-set quantile.
        from scry.eval.candidate import ReferenceQuantile

        errors = np.arange(1.0, 101.0)
        ends = pd.date_range("2026-01-01T00:00:00Z", periods=100, freq="5min")
        ids = np.array(["node-a", "node-b"] * 50)
        reference = _policy_score_set(errors, ends, ids)

        median = ReferenceQuantile(reference, quantile=0.5)
        assert median.resolve("node-a") == (50.5, "reference")
        tail = ReferenceQuantile(reference, quantile=0.99)
        assert tail.resolve("node-b") == (float(np.quantile(errors, 0.99)), "reference")
        assert tail.resolve("node-a") == tail.resolve("node-b")

    def test_healthy_split_fits_on_the_per_resource_fit_half(self) -> None:
        # Drift in each resource's second half, with node-b's span shifted 60
        # minutes later: a pooled time_split's first-half fit would absorb
        # node-a's drifted windows (ends 60..85), so only the per-resource
        # split resolves the all-low fit-half quantile pinned here.
        from scry.eval.candidate import HealthySplitQuantile

        low_a = np.linspace(0.10, 0.21, 12)
        high_a = np.linspace(10.0, 11.1, 12)
        low_b = np.linspace(0.30, 0.41, 12)
        high_b = np.linspace(20.0, 21.1, 12)
        base = pd.Timestamp("2026-01-01T00:00:00Z")
        ends_a = [base + pd.Timedelta(minutes=m * 5) for m in range(24)]
        ends_b = [base + pd.Timedelta(minutes=60 + m * 5) for m in range(24)]
        ends = pd.DatetimeIndex(ends_a + ends_b)
        ids = np.array(["node-a"] * 24 + ["node-b"] * 24)
        errors = np.concatenate([low_a, high_a, low_b, high_b])
        scores = _policy_score_set(errors, ends, ids)

        policy = HealthySplitQuantile(scores, quantile=0.99)
        threshold, source = policy.resolve("node-a")
        assert source == "healthy_split"
        fit_expected = np.concatenate([low_a, low_b])
        assert threshold == float(np.quantile(fit_expected, 0.99))
        assert threshold != float(np.quantile(errors, 0.99))
        assert policy.resolve("node-b") == (threshold, "healthy_split")

    def test_describe_is_json_serializable_with_resolved_parameters(self) -> None:
        from scry.eval.candidate import GlobalOverride, HealthySplitQuantile, ReferenceQuantile

        reference = _policy_score_set(
            np.linspace(0.1, 0.5, 60),
            pd.date_range("2026-01-01T00:00:00Z", periods=60, freq="5min"),
            np.array(["node-a"] * 60),
        )
        policies = [
            GlobalOverride(0.25),
            ReferenceQuantile(reference, quantile=0.99),
            HealthySplitQuantile(reference, quantile=0.99),
        ]
        for policy in policies:
            described = policy.describe()
            json.dumps(described)
            assert described["type"] == type(policy).__name__
            assert described["threshold"] == policy.resolve("node-a")[0]

        assert policies[1].describe()["quantile"] == 0.99
        assert policies[1].describe()["n_reference_windows"] == 60
        assert policies[1].describe()["grid"] == "native"
        assert policies[2].describe()["quantile"] == 0.99
        assert policies[2].describe()["gap"] == 6
        assert policies[2].describe()["n_fit_windows"] == 30
        assert policies[2].describe()["grid"] == "native"

    def test_empty_fit_raises_instead_of_falling_back(self) -> None:
        # The dead empty-fit fallback in healthy_threshold stays dead: an
        # empty calibration population is an error, never fit-on-everything.
        from scry.eval.candidate import HealthySplitQuantile, ReferenceQuantile

        empty = _policy_score_set(
            np.zeros(0), pd.DatetimeIndex([], tz="UTC"), np.array([], dtype=str)
        )
        with pytest.raises(ValueError, match="No healthy windows"):
            ReferenceQuantile(empty, quantile=0.99)
        with pytest.raises(ValueError, match="No healthy windows"):
            HealthySplitQuantile(empty, quantile=0.99)

    def test_policy_names_resolve_from_package_root(self) -> None:
        import scry.eval
        from scry.eval import candidate

        names = (
            "ThresholdPolicy",
            "GlobalOverride",
            "ReferenceQuantile",
            "HealthySplitQuantile",
            "PerResourceMargin",
        )
        for name in names:
            assert getattr(scry.eval, name) is getattr(candidate, name)
            assert name in scry.eval.__all__


class TestPerResourceMargin:
    def test_bit_identical_to_the_bake(self, keeper_path: str, fleet_df: pd.DataFrame) -> None:
        # Same calibration, same quantile, same margin: every policy-resolved
        # per-resource threshold reproduces the baked map bit for bit
        # (float.hex). Step 1 keeps both 200-sample nodes over the 50-window
        # floor, so the map covers the whole fleet.
        from scry.eval.candidate import PerResourceMargin, ReconstructionCandidate
        from scry.eval.scoring import ScoringGrid

        keeper = load_keeper(keeper_path)
        block, _ = bake_mod.compute_serving_block(
            keeper, fleet_df, quantile=0.99, step=1, per_resource_margin=2.0
        )
        assert set(block["per_resource"]) == {"node-a", "node-b"}

        scores = ReconstructionCandidate(keeper_path, profile=PROFILE).score(
            fleet_df, ScoringGrid(label="native", step_samples=1)
        )
        features = {
            str(rid): set(group["metric_name"].unique())
            for rid, group in fleet_df.groupby("resource_id")
        }
        policy = PerResourceMargin(
            scores,
            margin=2.0,
            fallback=block["threshold"],
            trained_features=keeper.numerical_features,
            features_by_resource=features,
            quantile=0.99,
        )
        for rid, baked in block["per_resource"].items():
            threshold, source = policy.resolve(rid)
            assert source == "per_resource"
            assert threshold.hex() == float(baked).hex()

    def test_hygiene_gates_route_to_fallback(self, keeper_path: str, tmp_path: Path) -> None:
        # The gated fleet: node-a eligible, node-b divergent-coverage, node-c
        # under the window floor at step 10; a ghost id never scored at all.
        # Everything but node-a serves the fallback.
        from scry.eval.candidate import PerResourceMargin, ReconstructionCandidate
        from scry.eval.scoring import ScoringGrid

        csv = gated_fleet_csv(tmp_path)
        df_long = asyncio.run(fetch_full_capture(csv, profile=PROFILE))
        keeper = load_keeper(keeper_path)
        scores = ReconstructionCandidate(keeper_path, profile=PROFILE).score(
            df_long, ScoringGrid(label="offline-step10", step_samples=10)
        )
        features = {
            str(rid): set(group["metric_name"].unique())
            for rid, group in df_long.groupby("resource_id")
        }
        policy = PerResourceMargin(
            scores,
            margin=2.0,
            fallback=0.5,
            trained_features=keeper.numerical_features,
            features_by_resource=features,
            quantile=0.99,
        )
        own = scores.errors[scores.resource_ids.astype(str) == "node-a"]
        assert policy.resolve("node-a") == (2.0 * float(np.quantile(own, 0.99)), "per_resource")
        assert policy.resolve("node-b") == (0.5, "per_resource_fallback")
        assert policy.resolve("node-c") == (0.5, "per_resource_fallback")
        assert policy.resolve("node-ghost") == (0.5, "per_resource_fallback")

    def test_describe_carries_map_fallback_and_eligibility(self) -> None:
        from scry.eval.candidate import PerResourceMargin

        errors = np.concatenate([np.linspace(0.1, 0.3, 60), np.linspace(0.2, 0.4, 12)])
        ends = pd.date_range("2026-01-01T00:00:00Z", periods=72, freq="5min")
        ids = np.array(["node-a"] * 60 + ["node-b"] * 12)
        calibration = _policy_score_set(errors, ends, ids)
        policy = PerResourceMargin(
            calibration,
            margin=2.0,
            fallback=0.5,
            trained_features=("cpu",),
            features_by_resource={"node-a": {"cpu"}, "node-b": {"cpu"}},
            quantile=0.99,
        )
        described = policy.describe()
        json.dumps(described)
        assert described["type"] == "PerResourceMargin"
        assert described["margin"] == 2.0
        assert described["quantile"] == 0.99
        assert described["fallback"] == 0.5
        assert set(described["per_resource"]) == {"node-a"}
        assert described["per_resource"]["node-a"] == policy.resolve("node-a")[0]
        assert described["eligibility"]["node-a"] == []
        assert described["eligibility"]["node-b"] == ["insufficient-windows:12<50"]
        assert described["grid"] == "native"

    def test_invalid_margin_raises(self) -> None:
        from scry.eval.candidate import PerResourceMargin

        calibration = _policy_score_set(
            np.linspace(0.1, 0.3, 60),
            pd.date_range("2026-01-01T00:00:00Z", periods=60, freq="5min"),
            np.array(["node-a"] * 60),
        )
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            with pytest.raises(ValueError, match="margin must be a positive finite number"):
                PerResourceMargin(
                    calibration,
                    margin=bad,
                    fallback=0.5,
                    trained_features=("cpu",),
                    features_by_resource={"node-a": {"cpu"}},
                )


def _patched_serving_keeper(source_path: str, tmp_path: Path, **serving_keys: object) -> str:
    # The test_reconstruction_endpoint patch pattern: copy the serving keeper
    # with extra keys merged into its serving block.
    ckpt = torch.load(source_path, map_location="cpu", weights_only=False)
    ckpt["serving"] = dict(ckpt["serving"], **serving_keys)
    out = str(tmp_path / "patched_keeper.pt")
    torch.save(ckpt, out)
    return out


class TestServingBlock:
    def test_resolves_baked_global_without_map(self, serving_keeper_path: str) -> None:
        from scry.eval.candidate import ServingBlock

        baked = torch.load(serving_keeper_path, map_location="cpu", weights_only=False)["serving"]
        policy = ServingBlock(serving_keeper_path)
        assert policy.resolve("any-node") == (baked["threshold"], "global")
        described = policy.describe()
        json.dumps(described)
        assert described["type"] == "ServingBlock"
        assert described["model_path"] == serving_keeper_path
        assert described["serving"]["threshold"] == baked["threshold"]
        assert described["serving"]["quantile"] == baked["quantile"]
        assert described["serving"]["recon_metric"] == baked["recon_metric"]

    def test_per_resource_entry_beats_global_for_mapped_id(
        self, serving_keeper_path: str, tmp_path: Path
    ) -> None:
        # A per_resource map injected into a copy of the serving keeper: the
        # mapped id resolves its entry, everyone else the global -- the
        # predictor's order absent the env override.
        from scry.eval.candidate import ServingBlock

        global_threshold = torch.load(
            serving_keeper_path, map_location="cpu", weights_only=False
        )["serving"]["threshold"]
        path = _patched_serving_keeper(
            serving_keeper_path, tmp_path, per_resource={"node-a": 0.007}, margin_multiplier=2.0
        )

        policy = ServingBlock(path)
        assert policy.resolve("node-a") == (0.007, "per_resource")
        assert policy.resolve("node-b") == (global_threshold, "global")

    def test_tolerates_additive_serving_keys(
        self, serving_keeper_path: str, tmp_path: Path
    ) -> None:
        from scry.eval.candidate import ServingBlock

        path = _patched_serving_keeper(
            serving_keeper_path,
            tmp_path,
            calibration_days=14.0,
            rebaked_at="2026-07-26T12:00:00Z",
            some_future_key="tolerated",
        )

        policy = ServingBlock(path)
        _, source = policy.resolve("any-node")
        assert source == "global"
        described = policy.describe()
        json.dumps(described)
        assert described["serving"]["calibration_days"] == 14.0
        assert described["serving"]["rebaked_at"] == "2026-07-26T12:00:00Z"
        assert described["serving"]["some_future_key"] == "tolerated"

    def test_missing_serving_block_raises(self, keeper_path: str) -> None:
        from scry.eval.candidate import ServingBlock

        with pytest.raises(ValueError, match="no serving block"):
            ServingBlock(keeper_path)

    def test_two_resource_lead_in_exceedance_per_resource_vs_pooled(self, tmp_path: Path) -> None:
        # The phase-4 defect in miniature: a quiet and a noisy resource, each
        # counted at its OWN resolved threshold; the pooled count at the global
        # threshold (deliberately fed the whole fleet as one slice) is labeled
        # pooled_ and misses quiet's exceedances while flooding on noisy's
        # baseline. Metrics packaging lands in PR 5; this exercises
        # slice_stats + the policy directly.
        from scry.eval.candidate import ServingBlock
        from scry.eval.detection import slice_stats

        serving = {
            "threshold": 1.0,
            "quantile": 0.99,
            "healthy_fpr": 0.0,
            "n_calibration_windows": 200,
            "recon_metric": "numerical_mse_from_mu",
            "per_resource": {"quiet": 0.2, "noisy": 2.0},
            "margin_multiplier": 2.0,
        }
        path = str(tmp_path / "two_scale.pt")
        torch.save({"serving": serving}, path)
        policy = ServingBlock(path)

        quiet = np.full(20, 0.1)
        quiet[5:8] = 0.3  # one sustained excursion over quiet's own 0.2
        quiet[15] = 0.3  # plus one isolated exceedance
        noisy = np.full(20, 1.5)  # baseline over the pooled 1.0, under its own 2.0
        noisy[10:13] = 2.5  # one sustained excursion over noisy's own 2.0
        ends = pd.date_range("2026-01-01T00:00:00Z", periods=20, freq="5min")

        lead_in_windows_over: dict[str, int] = {}
        for rid, errors in (("quiet", quiet), ("noisy", noisy)):
            threshold, source = policy.resolve(rid)
            assert source == "per_resource"
            stats = slice_stats(errors, ends, ends[0], ends[-1], [threshold], sustain=3)
            over = stats[f"over_{threshold:.4f}"]
            assert over["sustained_runs"] == 1
            lead_in_windows_over[rid] = over["windows_over"]
        assert lead_in_windows_over == {"quiet": 4, "noisy": 3}

        pooled_threshold, pooled_source = policy.resolve("fleet-as-a-whole")
        assert pooled_source == "global"
        pooled_stats = slice_stats(
            np.concatenate([quiet, noisy]),
            ends.append(ends),
            ends[0],
            ends[-1],
            [pooled_threshold],
            sustain=3,
        )
        pooled_lead_in_windows_over = pooled_stats[f"over_{pooled_threshold:.4f}"]["windows_over"]
        assert pooled_lead_in_windows_over == 20
        assert all(pooled_lead_in_windows_over != n for n in lead_in_windows_over.values())

    def test_lazy_exports_cover_all_candidate_members(self) -> None:
        import scry.eval
        from scry.eval import candidate

        public = {
            name
            for name, obj in vars(candidate).items()
            if not name.startswith("_")
            and getattr(obj, "__module__", None) == "scry.eval.candidate"
        }
        assert "ServingBlock" in public
        assert public <= set(scry.eval.__all__)
        for name in public:
            assert getattr(scry.eval, name) is getattr(candidate, name)
