# Description: Keeper-aware window building and explicit scoring grids for the evaluation harness.
# Description: Torch-heavy; exposed through the lazy side of the eval package exports.

"""Scoring primitives that touch the model stack.

``windows_for_keeper`` windows a canonical long-format capture with the
checkpoint's own feature schema and stored normalization, through the same
``build_windows``/``WindowSet`` primitives the predictor uses;
``warn_missing_model_features`` flags a capture that lacks trained features
before those windows silently score on neutral fills. ``ScoringGrid``
makes the scoring stride and cadence explicit: the stride never comes from
ambient config inside ``scry.eval``, and every sustain-bearing number is keyed
by the grid label it was computed on. This module imports the model stack, so
the eval package exposes it lazily; importing ``scry.eval`` alone stays
torch-free.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from scry.data.quality import missing_features
from scry.data.windowing import WindowSet, build_windows
from scry.model.checkpoint import Keeper
from scry.model.reconstruction import time_split


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


def per_resource_time_split(
    errors: np.ndarray,
    end_times: pd.DatetimeIndex,
    resource_ids: np.ndarray,
    gap: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split each resource independently through ``time_split``, then pool the halves.

    The pooled ``time_split`` drops a flat window count between its halves; in
    a multi-resource capture that gap spans only ~gap/n_resources distinct time
    steps, so one resource's fit and eval windows can still share raw samples.
    Splitting per resource keeps the holdout guarantee true for every resource.
    A single-resource input produces bit-identical output to ``time_split``.

    Args:
        errors: Per-window errors.
        end_times: Matching per-window end timestamps.
        resource_ids: Per-window resource ids, aligned with ``errors``.
        gap: Windows dropped between each resource's halves; callers pass
            ``ceil(seq_len / step)``.

    Returns:
        Tuple of (pooled fit halves, pooled eval halves), resources in sorted
        id order.
    """
    errors = np.asarray(errors)
    resource_ids = np.asarray(resource_ids)
    fit_halves: list[np.ndarray] = []
    eval_halves: list[np.ndarray] = []
    for rid in sorted(set(resource_ids)):
        mask = resource_ids == rid
        fit, eval_ = time_split(errors[mask], end_times[mask], gap=gap)
        fit_halves.append(fit)
        eval_halves.append(eval_)
    if not fit_halves:
        empty = np.zeros(0, dtype=errors.dtype)
        return empty, empty
    return np.concatenate(fit_halves), np.concatenate(eval_halves)


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


def warn_missing_model_features(df_long: pd.DataFrame, keeper: Keeper, source: str) -> None:
    """Warn when a capture lacks features the checkpoint trained on.

    The profile filter strips metrics the live profile no longer lists, and
    windowing fills absent features with neutral values, so a checkpoint from
    an older profile definition scores silently differently. A name-only
    profile comparison cannot catch this.
    """
    missing = missing_features(df_long, keeper.numerical_features)
    if missing:
        print(
            f"warning: {source} lacks {len(missing)} feature(s) the checkpoint was "
            f"trained on ({', '.join(missing)}); they window as neutral values, so "
            "scores are not comparable to the checkpoint's calibration.",
            file=sys.stderr,
        )
