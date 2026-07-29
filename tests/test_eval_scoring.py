# Description: Tests for scry.eval.scoring: keeper windowing promoted from the harness.
# Description: Pins the validate_incident alias, the lazy export, and the torch-free package root.

"""Tests for the eval scoring primitives.

``windows_for_keeper`` is pinned through the per-resource window-count
arithmetic on a synthetic capture, the alias identity that keeps
``validate_incident._windows_for_keeper`` working, and the lazy-export
contract: the name resolves from ``scry.eval`` while a bare package import
still leaves torch unimported.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

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
