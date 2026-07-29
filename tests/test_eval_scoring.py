# Description: Tests for scry.eval.scoring: keeper windowing and the explicit scoring grids.
# Description: Pins the validate_incident alias, the lazy exports, and the wall-clock cadence rule.

"""Tests for the eval scoring primitives.

``windows_for_keeper`` is pinned through the per-resource window-count
arithmetic on a synthetic capture, the alias identity that keeps
``validate_incident._windows_for_keeper`` working, and the lazy-export
contract: the name resolves from ``scry.eval`` while a bare package import
still leaves torch unimported. ``ScoringGrid`` is pinned through the serving
preset, identity selection without a cadence, the wall-clock (floored, not
span-anchored) tick lattice, per-resource per-tick greatest-end selection,
and single-keep dedup when one window is the greatest for several ticks.
"""

from __future__ import annotations

import asyncio
import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import validate_incident as vi
from synth import PROFILE, SEQ_LEN, gen_capture, write_csv

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
