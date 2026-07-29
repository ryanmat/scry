# Description: Keeper-aware window building for the evaluation harness.
# Description: Torch-heavy; exposed through the lazy side of the eval package exports.

"""Scoring primitives that touch the model stack.

``windows_for_keeper`` windows a canonical long-format capture with the
checkpoint's own feature schema and stored normalization, through the same
``build_windows``/``WindowSet`` primitives the predictor uses. This module
imports the model stack, so the eval package exposes it lazily; importing
``scry.eval`` alone stays torch-free.
"""

from __future__ import annotations

import pandas as pd

from scry.data.windowing import WindowSet, build_windows
from scry.model.checkpoint import Keeper


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
