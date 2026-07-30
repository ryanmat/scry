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
reference), ``HealthySplitQuantile`` fits on the pooled fit halves of the
per-resource time split, holding the eval halves out,
``PerResourceMargin`` bakes margin-times-own-quantile per resource behind the
hygiene gates, falling back to a global threshold for ineligible and unknown
resources, and ``ServingBlock`` reads a checkpoint's baked serving block and
resolves per_resource map hit else global -- the predictor's order absent the
env override. The quantile-fit policies raise ValueError on an empty fit
population; none falls back to fitting on everything.
"""

from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from scry.data.feature_engineering import set_active_profile
from scry.eval.hygiene import per_resource_eligibility
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


class PerResourceMargin(ThresholdPolicy):
    """Margin times own-quantile per resource; source ``"per_resource"`` or
    ``"per_resource_fallback"``.

    Mirrors the bake's per-resource arithmetic: pooled windows sliced by
    resource, no holdout split, the margin carrying the cross-day drift
    headroom -- so resolving on the calibration that fed the bake reproduces
    the baked ``per_resource`` map bit-identically. The ``scry.eval.hygiene``
    gates decide eligibility; an ineligible or unknown resource resolves the
    global fallback. Gate verdicts land in ``describe()`` rather than on
    stderr (the bake's CLI concern).

    Args:
        calibration: ScoreSet of an all-healthy calibration capture.
        margin: Multiplier on each resource's own healthy quantile.
        fallback: Global threshold served to ineligible and unknown resources.
        trained_features: The keeper's trained numerical feature names.
        features_by_resource: Per-resource metric_name sets from the
            calibration capture, keyed by stringified resource id.
        quantile: Healthy quantile taken per resource.

    Raises:
        ValueError: If ``margin`` is not a positive finite number.
    """

    def __init__(
        self,
        calibration: ScoreSet,
        *,
        margin: float,
        fallback: float,
        trained_features: Sequence[str],
        features_by_resource: Mapping[str, set[str]],
        quantile: float = 0.99,
    ):
        if not math.isfinite(margin) or margin <= 0:
            raise ValueError("margin must be a positive finite number")
        eligibility = per_resource_eligibility(
            trained_features=trained_features,
            features_by_resource=features_by_resource,
            resource_ids=calibration.resource_ids,
            errors=calibration.errors,
            quantile=quantile,
        )
        per_resource: dict[str, float] = {}
        for rid, verdict in eligibility.items():
            if verdict.eligible:
                per_resource[rid] = margin * verdict.own_quantile
        self._margin = margin
        self._quantile = quantile
        self._fallback = float(fallback)
        self._per_resource = per_resource
        self._eligibility = {rid: list(v.reasons) for rid, v in eligibility.items()}
        self._grid_label = calibration.grid.label

    def resolve(self, resource_id: str) -> tuple[float, str]:
        threshold = self._per_resource.get(resource_id)
        if threshold is None:
            return self._fallback, "per_resource_fallback"
        return threshold, "per_resource"

    def describe(self) -> dict[str, Any]:
        return {
            "type": "PerResourceMargin",
            "margin": self._margin,
            "quantile": self._quantile,
            "fallback": self._fallback,
            "per_resource": dict(self._per_resource),
            "eligibility": {rid: list(reasons) for rid, reasons in self._eligibility.items()},
            "grid": self._grid_label,
        }


class ServingBlock(ThresholdPolicy):
    """Thresholds from a checkpoint's baked serving block; source
    ``"per_resource"`` or ``"global"``.

    Reads ``checkpoint["serving"]`` -- ``{threshold, quantile, healthy_fpr,
    n_calibration_windows, recon_metric}`` plus the optional ``{per_resource,
    margin_multiplier}`` and any additive key (the deployed re-bake adds
    ``calibration_days`` and ``rebaked_at``) -- and resolves the way the
    predictor does absent the env override: per_resource map hit, else the
    global threshold. The whole block passes through ``describe()`` so
    additive keys stay visible in provenance.

    Args:
        model_path: Path to a checkpoint carrying a baked serving block.

    Raises:
        FileNotFoundError: If the checkpoint path does not exist.
        ValueError: If the checkpoint has no serving block.
    """

    def __init__(self, model_path: str):
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        serving = checkpoint.get("serving")
        if not serving:
            raise ValueError(
                f"checkpoint {model_path!r} has no serving block; bake one with "
                "scripts/bake_serving_threshold.py first"
            )
        self._model_path = model_path
        self._serving: dict[str, Any] = dict(serving)
        self._per_resource = {
            str(rid): float(value) for rid, value in (serving.get("per_resource") or {}).items()
        }
        self._threshold = float(serving["threshold"])

    def resolve(self, resource_id: str) -> tuple[float, str]:
        threshold = self._per_resource.get(str(resource_id))
        if threshold is not None:
            return threshold, "per_resource"
        return self._threshold, "global"

    def describe(self) -> dict[str, Any]:
        return {
            "type": "ServingBlock",
            "model_path": self._model_path,
            "serving": dict(self._serving),
        }
