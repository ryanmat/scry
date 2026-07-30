# Description: Scoring candidates and threshold policies for the evaluation harness.
# Description: Torch-heavy; exposed through the lazy side of the eval package exports.

"""Candidate scorers and threshold policies.

A ``Candidate`` turns a canonical long-format capture into a ``ScoreSet`` of
per-window errors on an explicit ``ScoringGrid``. ``ReconstructionCandidate``
is the X-DEC keeper reconstruction-error scorer: seq_len comes from the
checkpoint config and the stride from the grid, never from ambient config, and
the ScoreSet meta carries the provenance (model path and sha256, profile,
seq_len, step, device) the report layer records. This module imports the model
stack, so the eval package exposes it lazily.

A ``ThresholdPolicy`` resolves the anomaly threshold a resource is scored
against, as ``(threshold, source)``. ``GlobalOverride`` states one number,
``ReferenceQuantile`` fits the pooled quantile of a separate healthy reference
ScoreSet (no split; the capture under evaluation is disjoint from the
reference), and ``HealthySplitQuantile`` fits on the pooled fit halves of the
per-resource time split, holding the eval halves out. An empty fit population
raises ValueError; no policy falls back to fitting on everything.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scry.data.feature_engineering import set_active_profile
from scry.eval.scoring import ScoringGrid, per_resource_time_split, windows_for_keeper
from scry.model.checkpoint import load_keeper
from scry.model.reconstruction import reconstruction_errors


@dataclass(frozen=True)
class ScoreSet:
    """Per-window scores from one candidate on one grid."""

    errors: np.ndarray
    end_times: pd.DatetimeIndex
    resource_ids: np.ndarray
    grid: ScoringGrid
    meta: dict[str, Any]


class Candidate(ABC):
    """A scorer that produces per-window errors on an explicit grid."""

    @abstractmethod
    def score(self, df_long: pd.DataFrame, grid: ScoringGrid) -> ScoreSet:
        """Score a canonical long-format capture on ``grid``."""


class ReconstructionCandidate(Candidate):
    """Reconstruction-error scoring through a keeper checkpoint.

    Args:
        model_path: Path to the keeper checkpoint; a missing path raises
            FileNotFoundError at score time (the ``load_keeper`` contract).
        profile: Feature profile; defaults to the checkpoint's stored profile.
        device: Torch device override; defaults to the keeper's detected device.
    """

    def __init__(self, model_path: str, profile: str | None = None, device: str | None = None):
        self._model_path = model_path
        self._profile = profile
        self._device = device

    def score(self, df_long: pd.DataFrame, grid: ScoringGrid) -> ScoreSet:
        """Score the capture: window with the keeper schema, subsample per the grid."""
        keeper = load_keeper(self._model_path)
        profile = self._profile or keeper.profile
        if profile:
            set_active_profile(profile)
        device = self._device or keeper.device
        if device != keeper.device:
            keeper.model.to(device)

        seq_len = int(keeper.config["seq_len"])
        windows = windows_for_keeper(df_long, keeper, seq_len, grid.step_samples)
        errors = reconstruction_errors(keeper.model, windows.x_num, windows.x_cat, device)
        kept = grid.select_indices(windows.end_times, windows.resource_ids)

        meta = {
            "model_path": self._model_path,
            "model_sha256": hashlib.sha256(Path(self._model_path).read_bytes()).hexdigest(),
            "profile": profile,
            "seq_len": seq_len,
            "step": grid.step_samples,
            "device": device,
        }
        return ScoreSet(
            errors=errors[kept],
            end_times=windows.end_times[kept],
            resource_ids=np.asarray(windows.resource_ids)[kept],
            grid=grid,
            meta=meta,
        )


class ThresholdPolicy(ABC):
    """Resolves the anomaly threshold a resource is scored against."""

    @abstractmethod
    def resolve(self, resource_id: str) -> tuple[float, str]:
        """Return ``(threshold, source)`` for ``resource_id``."""

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """JSON-serializable type, parameters, and resolved thresholds, for provenance."""


class GlobalOverride(ThresholdPolicy):
    """One explicit threshold for every resource; source ``"override"``.

    Args:
        threshold: The threshold every resource resolves to.
    """

    def __init__(self, threshold: float):
        self._threshold = float(threshold)

    def resolve(self, resource_id: str) -> tuple[float, str]:
        return self._threshold, "override"

    def describe(self) -> dict[str, Any]:
        return {"type": "GlobalOverride", "threshold": self._threshold}


class ReferenceQuantile(ThresholdPolicy):
    """Pooled quantile of a healthy reference ScoreSet; source ``"reference"``.

    The whole reference fits the threshold, with no split: held-out evaluation
    comes from the capture under evaluation, which is disjoint from the
    reference by construction (the ``--reference`` path of validate_incident's
    ``compute_threshold``, as a policy).

    Args:
        reference: ScoreSet of an all-healthy reference capture.
        quantile: Threshold quantile over the reference errors.

    Raises:
        ValueError: If the reference holds no windows.
    """

    def __init__(self, reference: ScoreSet, quantile: float = 0.99):
        if reference.errors.size == 0:
            raise ValueError("No healthy windows available to fit the threshold.")
        self._quantile = quantile
        self._threshold = float(np.quantile(reference.errors, quantile))
        self._n_reference_windows = int(reference.errors.size)
        self._grid_label = reference.grid.label

    def resolve(self, resource_id: str) -> tuple[float, str]:
        return self._threshold, "reference"

    def describe(self) -> dict[str, Any]:
        return {
            "type": "ReferenceQuantile",
            "quantile": self._quantile,
            "threshold": self._threshold,
            "n_reference_windows": self._n_reference_windows,
            "grid": self._grid_label,
        }


class HealthySplitQuantile(ThresholdPolicy):
    """Per-resource-split fit-half quantile; source ``"healthy_split"``.

    Each resource's windows split independently through
    ``per_resource_time_split`` with ``gap = ceil(seq_len / step)`` from the
    ScoreSet meta; the pooled fit halves fit the threshold and the eval halves
    stay held out for downstream false-positive measurement. This is the
    leak-free harness variant of the temporal-split threshold (the shipped
    bake keeps its pooled split; the divergence is documented).

    Args:
        scores: ScoreSet whose fit halves calibrate the threshold; callers
            pass a healthy capture, or a case capture pre-sliced to its
            healthy span.
        quantile: Threshold quantile over the pooled fit halves.

    Raises:
        ValueError: If the split leaves no fit windows.
    """

    def __init__(self, scores: ScoreSet, quantile: float = 0.99):
        gap = -(-int(scores.meta["seq_len"]) // int(scores.meta["step"]))
        fit, eval_ = per_resource_time_split(
            scores.errors, scores.end_times, scores.resource_ids, gap=gap
        )
        if fit.size == 0:
            raise ValueError("No healthy windows available to fit the threshold.")
        self._quantile = quantile
        self._gap = gap
        self._threshold = float(np.quantile(fit, quantile))
        self._n_fit_windows = int(fit.size)
        self._n_eval_windows = int(eval_.size)
        self._grid_label = scores.grid.label

    def resolve(self, resource_id: str) -> tuple[float, str]:
        return self._threshold, "healthy_split"

    def describe(self) -> dict[str, Any]:
        return {
            "type": "HealthySplitQuantile",
            "quantile": self._quantile,
            "threshold": self._threshold,
            "gap": self._gap,
            "n_fit_windows": self._n_fit_windows,
            "n_eval_windows": self._n_eval_windows,
            "grid": self._grid_label,
        }
