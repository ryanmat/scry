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
