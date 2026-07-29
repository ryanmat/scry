# Description: Scoring candidates for the evaluation harness: model wrappers producing ScoreSets.
# Description: Torch-heavy; exposed through the lazy side of the eval package exports.

"""Candidate scorers.

A ``Candidate`` turns a canonical long-format capture into a ``ScoreSet`` of
per-window errors on an explicit ``ScoringGrid``. ``ReconstructionCandidate``
is the X-DEC keeper reconstruction-error scorer: seq_len comes from the
checkpoint config and the stride from the grid, never from ambient config, and
the ScoreSet meta carries the provenance (model path and sha256, profile,
seq_len, step, device) the report layer records. This module imports the model
stack, so the eval package exposes it lazily.
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
from scry.eval.scoring import ScoringGrid, windows_for_keeper
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
