# Description: Load a keeper X-DEC checkpoint into a model plus its schema and normalization.
# Description: Shared loader for the incident-validation harness and the serving-threshold bake utility.

"""Keeper checkpoint loading.

Reconstructs a :class:`~scry.model.xdec.TemporalXDEC` from a saved checkpoint and
carries the stored normalization and feature schema, so a capture can be aligned
and scaled exactly as in training. The API predictor keeps its own loader with
serving-specific validation; this module is for the offline scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from scry.model.xdec import TemporalXDEC


@dataclass
class Keeper:
    """A loaded keeper model plus the schema and normalization it was trained with."""

    model: TemporalXDEC
    device: str
    config: dict[str, Any]
    normalization: dict[str, Any]
    cat_normalization: dict[str, Any] | None
    numerical_features: list[str]
    categorical_features: list[str]
    profile: str | None


def detect_device() -> str:
    """Detect the best available torch device."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_keeper(model_path: str) -> Keeper:
    """Load the keeper checkpoint and reconstruct the model.

    Mirrors the predictor's load path: reconstruct ``TemporalXDEC`` from the saved
    config, load the weights, and carry the stored normalization and feature
    schema so a capture can be aligned and scaled exactly as in training.

    Args:
        model_path: Path to the saved checkpoint.

    Returns:
        A populated :class:`Keeper`.

    Raises:
        FileNotFoundError: If the checkpoint does not exist.
        ValueError: If the checkpoint has no usable feature schema.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    device = detect_device()
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    config = checkpoint["config"]
    normalization = checkpoint.get("normalization") or {"mean": None, "std": None}
    cat_normalization = checkpoint.get("categorical_normalization")

    schema = checkpoint.get("feature_schema")
    if not schema or "numerical" not in schema or "categorical" not in schema:
        raise ValueError(
            "Model checkpoint has no feature_schema; a capture cannot be aligned "
            "by name. Retrain with the current scripts/train_model.py."
        )
    numerical_features = [str(x) for x in schema["numerical"]]
    categorical_features = [str(x) for x in schema["categorical"]]

    if len(numerical_features) != config["num_numerical"]:
        raise ValueError(
            f"feature_schema numerical count ({len(numerical_features)}) does not "
            f"match model num_numerical ({config['num_numerical']}). Retrain the model."
        )
    if len(categorical_features) != config["num_categorical"]:
        raise ValueError(
            f"feature_schema categorical count ({len(categorical_features)}) does not "
            f"match model num_categorical ({config['num_categorical']}). Retrain the model."
        )

    model = TemporalXDEC(
        num_numerical=config["num_numerical"],
        num_categorical=config["num_categorical"],
        seq_len=config["seq_len"],
        num_hidden=config["num_hidden"],
        cat_hidden=config["cat_hidden"],
        latent_dim=config["latent_dim"],
        n_clusters=config["n_clusters"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return Keeper(
        model=model,
        device=device,
        config=config,
        normalization=normalization,
        cat_normalization=cat_normalization,
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        profile=schema.get("profile"),
    )
