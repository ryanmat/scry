# Description: The serving bake's setup: the keeper, the healthy capture, and the window step.
# Description: Torch-free to import; load_keeper is reached inside prepare_bake, as calibration is.

"""What a serving-threshold bake needs before it can score anything.

``prepare_bake`` loads the keeper, resolves the feature profile from the
checkpoint when the caller named none, fetches the whole healthy capture, and
reports the window step the config carries -- the setup every bake begins
with, and the place the capture is checked against the features the checkpoint
was trained on. A profile that no longer lists a trained feature strips it from
the fetch, and windowing then fills it neutral, so a threshold baked from such
a capture is not the one the model produces on complete data: that is a warning
on stderr rather than an error, because the bake is still the best number
available and the operator is the one who decides.

It lives in the package and not in ``scripts/bake_serving_threshold.py``
because two callers need it: the script's own ``bake``, and
``scry.eval.calibration``'s rebake, which cannot import the script in the
serving image (``pip install .`` copies ``src`` alone). One setup for both
callers is what puts the missing-feature warning on both paths.

Importable without torch, like the rest of the torch-free half of this package:
``load_keeper`` is imported inside the function, so a consumer that only wants
the constants pays for no model stack.
"""

from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING

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
