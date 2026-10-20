# Description: The serving bake: its setup (keeper, capture, window step) and the block it derives.
# Description: Torch-free to import; the torch-backed imports are inside the functions.

"""What a serving-threshold bake needs, and the block it derives.

``prepare_bake`` loads the keeper, resolves the feature profile from the
checkpoint when the caller named none, fetches the whole healthy capture, and
reports the window step the config carries -- the setup every bake begins
with, and the place the capture is checked against the features the checkpoint
was trained on. A profile that no longer lists a trained feature strips it from
the fetch, and windowing then fills it neutral, so a threshold baked from such
a capture is not the one the model produces on complete data: that is a warning
on stderr rather than an error, because the bake is still the best number
available and the operator is the one who decides.

``compute_serving_block`` is the arithmetic over that setup: it windows the
capture, scores it against the keeper, and derives the block a checkpoint
serves -- the global threshold at a healthy quantile with a held-out half for
the false-positive rate, and, where a margin is given, a per-resource
threshold for every resource the hygiene gates leave it to propose for.

Both live in the package and not in ``scripts/bake_serving_threshold.py``
because two callers need them: the script's own ``bake``, and
``scry.eval.calibration``'s rebake, which cannot import the script in the
serving image (``pip install .`` copies ``src`` alone). One setup and one
arithmetic for both callers is what puts the missing-feature warning on both
paths and keeps the two from reaching different numbers from the same inputs.

Importable without torch, like the rest of the torch-free half of this package:
the torch-backed imports are inside the functions, so a consumer that imports
this module for its constants pays for no model stack.
"""

from __future__ import annotations

import asyncio
import math
import sys
from typing import TYPE_CHECKING, Any

import numpy as np

from scry.eval.hygiene import (
    MIN_PER_RESOURCE_WINDOWS,
    REASON_DIVERGENT,
    REASON_TOO_FEW_WINDOWS,
    ResourceEligibility,
    per_resource_eligibility,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

    from scry.model.checkpoint import Keeper

# Identifies how the score was computed, so a serving block is never misread as a
# different metric (e.g. numerical+categorical error) at a different scale.
RECON_METRIC = "numerical_mse_from_mu"

# What an omitted resource serves, in both situations: a first bake has nothing
# for it and it falls back to the global threshold, while a re-bake
# (scry.eval.calibration) keeps the threshold it is already serving rather than
# moving it, which the old "it will serve the global threshold" wording promised
# it would not do.
OMITTED_FALLBACK = (
    "It is omitted from the per-resource map: it keeps its previous threshold "
    "where it has one, and serves the global threshold where it does not."
)


def prepare_bake(
    model_path: str | Path,
    healthy_data: str,
    *,
    profile: str | None = None,
    data_format: str | None = None,
    device: str | None = None,
) -> tuple[Keeper, pd.DataFrame, int]:
    """Load the keeper and the capture a bake scores, and say what step to window at.

    Args:
        model_path: Path to the keeper checkpoint.
        healthy_data: All-healthy capture URI or path.
        profile: Feature profile; defaults to the checkpoint's stored profile,
            which is the profile the model was trained under.
        data_format: Optional explicit file format override.
        device: Torch device to score on. A keeper detects its own; this is the
            operator's override of that detection, and the model is moved onto
            it here, because the device the keeper carries is the one the
            scoring runs on.

    Returns:
        ``(keeper, df_long, step)``: the loaded keeper, the canonical
        long-format capture, and the sliding-window step from the config.

    Prints a warning to stderr when the capture lacks a feature the checkpoint
    was trained on: windowing fills such a feature neutral, so the threshold
    this bake produces will not match what the model sees on complete data.
    """
    # local: load_keeper pulls in torch, and this module stays importable
    # without it, as the module docstring says.
    from scry.data.fetcher import fetch_full_capture
    from scry.data.quality import missing_features
    from scry.model.checkpoint import load_keeper
    from scry.utils.config import get_config

    keeper = load_keeper(str(model_path))
    if device is not None and device != keeper.device:
        keeper.model.to(device)
        keeper.device = device
    df_long = asyncio.run(
        fetch_full_capture(healthy_data, profile=profile or keeper.profile, data_format=data_format)
    )
    # A profile that no longer lists a trained feature strips it from the
    # fetch; the threshold would then be calibrated on neutral-filled windows.
    missing = missing_features(df_long, keeper.numerical_features)
    if missing:
        print(
            f"warning: {healthy_data} lacks {len(missing)} feature(s) the "
            f"checkpoint was trained on ({', '.join(missing)}); the baked "
            "threshold will not match what the model sees on complete data.",
            file=sys.stderr,
        )
    return keeper, df_long, int(get_config().window_step)


def compute_serving_block(
    keeper: Keeper,
    df_long: pd.DataFrame,
    *,
    quantile: float,
    step: int,
    per_resource_margin: float | None = None,
) -> tuple[dict[str, Any], dict[str, ResourceEligibility] | None]:
    """Derive the serving block from a keeper and an all-healthy capture.

    Args:
        keeper: The loaded keeper (model, schema, stored normalization).
        df_long: Canonical long-format healthy metrics.
        quantile: Healthy-window quantile for the threshold.
        step: Sliding-window step (from the config).
        per_resource_margin: When set, additionally write a ``per_resource`` map of
            ``margin x own_quantile`` thresholds, one per resource in the capture.
            The per-resource quantile is taken over all of that resource's windows
            (no holdout split); the margin carries the cross-day drift headroom.
            A resource whose windows are too few for a meaningful quantile is
            omitted (see ``OMITTED_FALLBACK`` for what it serves instead).

    Returns:
        Tuple of the serving block dict and the per-resource eligibility map
        (``scry.eval.hygiene`` verdicts, so a downstream re-bake can distinguish
        gate-omitted from absent-from-capture); the map is None when per-resource
        baking is off.

    Raises:
        ValueError: If the capture produces no windows.
    """
    # local: both pull in torch, and this module stays importable without it,
    # as the module docstring says.
    from scry.data.windowing import build_windows
    from scry.model.reconstruction import healthy_threshold, reconstruction_errors

    seq_len = int(keeper.config["seq_len"])
    windows = build_windows(
        df_long,
        numerical_features=keeper.numerical_features,
        categorical_features=keeper.categorical_features,
        normalization=keeper.normalization,
        cat_normalization=keeper.cat_normalization,
        seq_len=seq_len,
        step=step,
    )
    if windows.x_num.shape[0] == 0:
        raise ValueError(
            "Healthy capture produced no windows; check the profile, the time "
            "range, and that the metrics are present."
        )
    errors = reconstruction_errors(keeper.model, windows.x_num, windows.x_cat, keeper.device)
    # Drop a gap spanning the window overlap (ceil(seq_len / step) windows) so the
    # held-out FPR shares no raw samples with the fit half (mirrors validate_incident).
    overlap_gap = -(-seq_len // step)
    threshold, fit, eval_ = healthy_threshold(
        errors, windows.end_times, quantile=quantile, gap=overlap_gap
    )
    healthy_fpr = float(np.mean(eval_ > threshold)) if eval_.size else None
    block: dict[str, Any] = {
        "threshold": threshold,
        "quantile": quantile,
        "healthy_fpr": healthy_fpr,
        "n_calibration_windows": int(fit.size),
        "recon_metric": RECON_METRIC,
    }
    eligibility: dict[str, ResourceEligibility] | None = None
    if per_resource_margin is not None:
        if not math.isfinite(per_resource_margin) or per_resource_margin <= 0:
            raise ValueError("per_resource_margin must be a positive finite number")
        # A feature absent for one resource but supplied elsewhere in the capture is
        # filled at -mean/std by the capture-wide windowing here, while the serving
        # path (which windows the resource alone) fills it neutral. A per-resource
        # quantile from such windows sits on a scale serving never produces, so the
        # affected resource is omitted and falls back to the global threshold.
        # The gate verdicts come from scry.eval.hygiene, the shared implementation.
        features_by_resource = {
            str(rid): set(group["metric_name"].unique())
            for rid, group in df_long.groupby("resource_id")
        }
        eligibility = per_resource_eligibility(
            trained_features=keeper.numerical_features,
            features_by_resource=features_by_resource,
            resource_ids=windows.resource_ids,
            errors=errors,
            quantile=quantile,
        )
        per_resource: dict[str, float] = {}
        for rid in sorted(set(windows.resource_ids)):
            verdict = eligibility[str(rid)]
            if verdict.eligible:
                per_resource[str(rid)] = per_resource_margin * verdict.own_quantile
                continue
            # Reasons are recorded in gate order, so the first one names the
            # gate that omits the resource; only that warning is printed.
            reason = verdict.reasons[0]
            if reason.startswith(REASON_DIVERGENT):
                print(
                    f"warning: resource {rid!r} lacks {len(verdict.missing_features)} trained "
                    f"feature(s) the capture supplies elsewhere "
                    f"({', '.join(verdict.missing_features)}); its bake-time windows are "
                    f"filled at -mean/std where serving fills neutral. {OMITTED_FALLBACK}",
                    file=sys.stderr,
                )
            elif reason.startswith(REASON_TOO_FEW_WINDOWS):
                print(
                    f"warning: resource {rid!r} has only {verdict.n_windows} window(s) "
                    f"(< {MIN_PER_RESOURCE_WINDOWS}). {OMITTED_FALLBACK}",
                    file=sys.stderr,
                )
            else:
                # The quantile gates name themselves: non-finite-quantile or
                # non-positive-quantile. Calling both "non-positive" sent an
                # operator looking for a negative number that is not there --
                # a NaN quantile fails no `<= 0` comparison at all.
                print(
                    f"warning: resource {rid!r} failed the {reason} gate. {OMITTED_FALLBACK}",
                    file=sys.stderr,
                )
        block["per_resource"] = per_resource
        block["margin_multiplier"] = per_resource_margin
    return block, eligibility
