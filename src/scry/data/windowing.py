# Description: Window a canonical long-format capture into normalized model-order tensors.
# Description: Applies a checkpoint's stored normalization so serving and validation share one path.

"""Shared windowing for X-DEC reconstruction scoring.

Turns a canonical long-format metrics frame into per-window numerical and
categorical tensors in a model's feature order, normalized with the model's own
stored parameters (never re-fit on new data). The incident-validation harness,
the serving reconstruction path, and the serving-threshold bake utility all use
this so they window and normalize identically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch

from scry.data.feature_engineering import create_dual_windows, pivot_metrics


@dataclass
class WindowSet:
    """Normalized windows ready for scoring, with their resource and end-time labels."""

    x_num: torch.Tensor
    x_cat: torch.Tensor
    resource_ids: np.ndarray
    end_times: pd.DatetimeIndex


def align_frames(
    df_long: pd.DataFrame,
    numerical_features: list[str],
    categorical_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Pivot the long capture and align columns to the model's feature order.

    Features the model expects but the capture lacks are inserted as all-NaN
    columns so the window tensors keep the model's shape and column order. The
    returned masks flag those absent numerical and categorical features so they
    can be mapped to the neutral normalized value (0) after scaling, mirroring
    the predictor.

    Args:
        df_long: Canonical long-format metrics.
        numerical_features: Model numerical features, in model order.
        categorical_features: Model categorical features, in model order.

    Returns:
        Tuple of (df_num, df_cat, absent_numerical_mask, absent_categorical_mask).
    """
    pivoted = pivot_metrics(df_long)

    df_num = pivoted[["resource_id", "timestamp"]].copy()
    absent_num = np.zeros(len(numerical_features), dtype=bool)
    for i, name in enumerate(numerical_features):
        if name in pivoted.columns:
            df_num[name] = pivoted[name]
        else:
            df_num[name] = np.nan
            absent_num[i] = True

    df_cat = pivoted[["resource_id", "timestamp"]].copy()
    absent_cat = np.zeros(len(categorical_features), dtype=bool)
    for i, name in enumerate(categorical_features):
        if name in pivoted.columns:
            df_cat[name] = pivoted[name]
        else:
            df_cat[name] = np.nan
            absent_cat[i] = True

    return df_num, df_cat, absent_num, absent_cat


def apply_numerical_normalization(
    windows: np.ndarray,
    normalization: dict[str, Any],
    absent: np.ndarray,
) -> np.ndarray:
    """Apply a checkpoint's stored z-score normalization to numerical windows.

    NaNs are filled forward then backward then with zero (the training rule), the
    stored per-feature mean/std are applied, and features the capture never
    supplied are set to the neutral normalized value (0).

    Args:
        windows: Raw numerical windows (n, seq, n_num).
        normalization: Stored ``{mean, std}`` arrays from the checkpoint.
        absent: Boolean mask of features missing from the capture.

    Returns:
        Normalized windows.
    """
    out = windows.astype(np.float32).copy()
    for i in range(out.shape[0]):
        for j in range(out.shape[2]):
            series = pd.Series(out[i, :, j]).ffill().bfill().fillna(0.0)
            out[i, :, j] = series.to_numpy(dtype=np.float32)

    if normalization.get("mean") is not None:
        mean = np.asarray(normalization["mean"], dtype=np.float32)
        std = np.asarray(normalization["std"], dtype=np.float32)
        std = np.where(std == 0, 1.0, std)
        out = (out - mean) / std

    if absent.any():
        out[:, :, absent] = 0.0
    return out


def apply_categorical_encoding(
    windows: np.ndarray,
    cat_normalization: dict[str, Any] | None,
    absent: np.ndarray,
) -> np.ndarray:
    """Apply a checkpoint's stored min-max encoding to categorical windows.

    Mirrors :func:`scry.data.feature_engineering.encode_categorical` and the
    predictor: NaNs become 0, each feature is scaled by its stored ``[min, max]``,
    and a degenerate range falls back to 1.0 when the training value was positive
    else 0.0. Features the capture never supplied are then set to the neutral
    value 0.0, matching the predictor, which never scales an absent categorical.

    Args:
        windows: Raw categorical windows (n, seq, n_cat).
        cat_normalization: Stored ``{min, max}`` arrays, or None.
        absent: Boolean mask of categorical features missing from the capture.

    Returns:
        Encoded windows in [0, 1].
    """
    out = np.nan_to_num(windows.astype(np.float32), nan=0.0)
    if cat_normalization is not None:
        cat_min = np.asarray(cat_normalization["min"], dtype=np.float32)
        cat_max = np.asarray(cat_normalization["max"], dtype=np.float32)
        for j in range(out.shape[2]):
            lo, hi = cat_min[j], cat_max[j]
            if hi > lo:
                out[:, :, j] = np.clip((out[:, :, j] - lo) / (hi - lo), 0.0, 1.0)
            else:
                out[:, :, j] = 1.0 if hi > 0 else 0.0

    if absent.any():
        out[:, :, absent] = 0.0
    return out


def build_windows(
    df_long: pd.DataFrame,
    *,
    numerical_features: list[str],
    categorical_features: list[str],
    normalization: dict[str, Any],
    cat_normalization: dict[str, Any] | None,
    seq_len: int,
    step: int,
) -> WindowSet:
    """Window a capture in model order and normalize with stored parameters.

    Absent-feature masking is capture-wide: a feature is treated as absent only
    when no resource in ``df_long`` reports it. This assumes homogeneous per-resource
    coverage within a capture (true for a single-resource serving frame and for a
    fleet of like resources). If one resource in a multi-resource capture omits a
    feature that others report, that resource's windows carry the feature at its
    normalized zero (-mean/std) here but at neutral 0 when the resource is served
    alone; reconcile coverage before mixing heterogeneous resources in one capture.

    Args:
        df_long: Canonical long-format metrics for the capture.
        numerical_features: Model numerical features, in model order.
        categorical_features: Model categorical features, in model order.
        normalization: Stored ``{mean, std}`` numerical params from the checkpoint.
        cat_normalization: Stored ``{min, max}`` categorical params, or None.
        seq_len: Window length (the model's sequence length).
        step: Sliding-window step.

    Returns:
        A :class:`WindowSet` of normalized tensors with labels, possibly empty.
    """
    df_num, df_cat, absent_num, absent_cat = align_frames(
        df_long, numerical_features, categorical_features
    )
    num_windows, cat_windows, labels = create_dual_windows(
        df_num, df_cat, window_size=seq_len, step=step
    )

    if num_windows.shape[0] == 0:
        empty_num = torch.zeros((0, seq_len, len(numerical_features)), dtype=torch.float32)
        empty_cat = torch.zeros((0, seq_len, len(categorical_features)), dtype=torch.float32)
        return WindowSet(
            x_num=empty_num,
            x_cat=empty_cat,
            resource_ids=np.zeros(0, dtype=object),
            end_times=pd.DatetimeIndex([], tz="UTC"),
        )

    norm_num = apply_numerical_normalization(num_windows, normalization, absent_num)
    norm_cat = apply_categorical_encoding(cat_windows, cat_normalization, absent_cat)

    end_times = pd.to_datetime(labels[:, 1], utc=True)
    return WindowSet(
        x_num=torch.tensor(norm_num, dtype=torch.float32),
        x_cat=torch.tensor(norm_cat, dtype=torch.float32),
        resource_ids=labels[:, 0].astype(object),
        end_times=pd.DatetimeIndex(end_times),
    )
