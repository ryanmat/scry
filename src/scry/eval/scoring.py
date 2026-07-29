# Description: Keeper-aware window building and explicit scoring grids for the evaluation harness.
# Description: Torch-heavy; exposed through the lazy side of the eval package exports.

"""Scoring primitives that touch the model stack.

``windows_for_keeper`` windows a canonical long-format capture with the
checkpoint's own feature schema and stored normalization, through the same
``build_windows``/``WindowSet`` primitives the predictor uses. ``ScoringGrid``
makes the scoring stride and cadence explicit: the stride never comes from
ambient config inside ``scry.eval``, and every sustain-bearing number is keyed
by the grid label it was computed on. This module imports the model stack, so
the eval package exposes it lazily; importing ``scry.eval`` alone stays
torch-free.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from scry.data.windowing import WindowSet, build_windows
from scry.model.checkpoint import Keeper


@dataclass(frozen=True)
class ScoringGrid:
    """An explicit scoring grid: window stride plus optional wall-clock cadence.

    ``step_samples`` is the ``build_windows`` stride in samples. When
    ``cadence`` is set, the grid keeps, for each cadence tick over the span,
    the single window with the greatest end time at or before the tick, per
    resource -- the serving latest-window refresh emulated offline. Ticks are
    wall-clock aligned: boundaries come from flooring the span to the cadence,
    so a capture whose first window ends at 16:07 ticks at 16:00, 16:10, ...,
    and two captures with different start offsets land on the same lattice.
    """

    label: str
    step_samples: int
    cadence: pd.Timedelta | None = None

    def select_indices(self, end_times: pd.DatetimeIndex, resource_ids: np.ndarray) -> np.ndarray:
        """Indices of the windows the grid keeps, in time-stable input order.

        Args:
            end_times: Per-window end timestamps.
            resource_ids: Per-window resource ids, aligned with ``end_times``.

        Returns:
            Sorted array of kept positions; every position when ``cadence`` is
            None (identity selection). Ticks cover the aligned boundaries
            inside the span, so windows ending after the last covered boundary
            (up to one cadence at the capture tail) select nothing.
        """
        if self.cadence is None or len(end_times) == 0:
            return np.arange(len(end_times))
        resource_ids = np.asarray(resource_ids)
        first_tick = end_times.min().floor(self.cadence)
        last_tick = end_times.max().floor(self.cadence)
        ticks = pd.date_range(first_tick, last_tick, freq=self.cadence)

        kept: list[int] = []
        for rid in np.unique(resource_ids):
            positions = np.flatnonzero(resource_ids == rid)
            order = np.argsort(end_times[positions].values, kind="stable")
            sorted_ends = end_times[positions[order]]
            locations = sorted_ends.searchsorted(ticks, side="right") - 1
            for location in np.unique(locations[locations >= 0]):
                kept.append(int(positions[order[location]]))
        return np.array(sorted(kept), dtype=int)


# Serving-grid emulation: step-1 windows subsampled at the 10-minute wall-clock
# cadence the deployed latest-window refresh advances on.
SERVING_GRID = ScoringGrid(label="serving-10min", step_samples=1, cadence=pd.Timedelta(minutes=10))


def windows_for_keeper(df_long: pd.DataFrame, keeper: Keeper, seq_len: int, step: int) -> WindowSet:
    """Window a capture for a loaded keeper, passing its schema and stored normalization."""
    return build_windows(
        df_long,
        numerical_features=keeper.numerical_features,
        categorical_features=keeper.categorical_features,
        normalization=keeper.normalization,
        cat_normalization=keeper.cat_normalization,
        seq_len=seq_len,
        step=step,
    )
