# Description: Model export utilities for ONNX and TorchScript formats.
# Description: Enables deployment of trained models to production environments.

"""Model export utilities for ONNX and TorchScript formats."""

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from scry.model.xdec import TemporalXDEC

logger = logging.getLogger(__name__)


class InferenceWrapper(nn.Module):
    """Wrapper for inference-only forward pass.

    Simplifies the model output to only return cluster predictions
    and confidence scores, suitable for production deployment.

    Args:
        model: Trained TemporalXDEC model.
    """

    def __init__(self, model: TemporalXDEC) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, x_num: torch.Tensor, x_cat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inference forward pass returning cluster ID and confidence.

        Args:
            x_num: Numerical features (batch, seq_len, num_numerical).
            x_cat: Categorical features (batch, seq_len, num_categorical).

        Returns:
            Tuple of (cluster_ids, confidences) where:
                - cluster_ids: Predicted cluster (batch,) as long tensor.
                - confidences: Max soft assignment probability (batch,).
        """
        q = self.model.get_cluster_assignments(x_num, x_cat)
        cluster_ids = q.argmax(dim=1)
        confidences = q.max(dim=1).values
        return cluster_ids, confidences


def export_to_torchscript(
    model: TemporalXDEC,
    output_path: str,
    example_batch_size: int = 1,
    use_inference_wrapper: bool = True,
) -> str:
    """Export model to TorchScript format.

    TorchScript enables running PyTorch models in C++ or other
    environments without Python dependencies.

    Args:
        model: Trained TemporalXDEC model.
        output_path: Path to save the TorchScript model.
        example_batch_size: Batch size for tracing (default: 1).
        use_inference_wrapper: If True, wrap model for simpler inference output.

    Returns:
        Path to the saved TorchScript model.
    """
    model.eval()

    if use_inference_wrapper:
        export_model = InferenceWrapper(model)
    else:
        export_model = model

    export_model.eval()

    # Create example inputs for tracing
    x_num = torch.randn(
        example_batch_size, model.seq_len, model.num_numerical
    )
    x_cat = torch.randn(
        example_batch_size, model.seq_len, model.num_categorical
    )

    # Trace the model
    traced_model = torch.jit.trace(export_model, (x_num, x_cat))

    # Save the traced model
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    traced_model.save(str(output_path))
    logger.info("Exported TorchScript model to %s", output_path)

    return str(output_path)


def export_to_onnx(
    model: TemporalXDEC,
    output_path: str,
    example_batch_size: int = 1,
    use_inference_wrapper: bool = True,
    opset_version: int = 14,
    dynamic_batch: bool = True,
) -> str:
    """Export model to ONNX format.

    ONNX enables running the model in various runtime environments
    including ONNX Runtime, TensorRT, and others.

    Args:
        model: Trained TemporalXDEC model.
        output_path: Path to save the ONNX model.
        example_batch_size: Batch size for export (default: 1).
        use_inference_wrapper: If True, wrap model for simpler inference output.
        opset_version: ONNX opset version (default: 14).
        dynamic_batch: If True, allow dynamic batch size (default: True).

    Returns:
        Path to the saved ONNX model.
    """
    try:
        import onnx  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "ONNX export requires the 'onnx' and 'onnxscript' packages. "
            "Install them with: pip install onnx onnxscript"
        ) from e

    model.eval()

    if use_inference_wrapper:
        export_model = InferenceWrapper(model)
        output_names = ["cluster_ids", "confidences"]
    else:
        export_model = model
        output_names = ["z", "mu", "logvar", "x_num_recon", "x_cat_recon", "q"]

    export_model.eval()

    # Create example inputs
    x_num = torch.randn(
        example_batch_size, model.seq_len, model.num_numerical
    )
    x_cat = torch.randn(
        example_batch_size, model.seq_len, model.num_categorical
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Set up dynamic axes for batch dimension
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "x_num": {0: "batch_size"},
            "x_cat": {0: "batch_size"},
        }
        for name in output_names:
            dynamic_axes[name] = {0: "batch_size"}

    # Export to ONNX
    torch.onnx.export(
        export_model,
        (x_num, x_cat),
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["x_num", "x_cat"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    logger.info("Exported ONNX model to %s", output_path)

    return str(output_path)


def get_model_metadata(model: TemporalXDEC) -> dict[str, Any]:
    """Get model metadata for export documentation.

    Args:
        model: TemporalXDEC model.

    Returns:
        Dictionary containing model metadata.
    """
    return {
        "model_type": "TemporalXDEC",
        "num_numerical": model.num_numerical,
        "num_categorical": model.num_categorical,
        "seq_len": model.seq_len,
        "latent_dim": model.latent_dim,
        "n_clusters": model.n_clusters,
        "input_shapes": {
            "x_num": f"(batch, {model.seq_len}, {model.num_numerical})",
            "x_cat": f"(batch, {model.seq_len}, {model.num_categorical})",
        },
        "output_shapes_inference": {
            "cluster_ids": "(batch,)",
            "confidences": "(batch,)",
        },
        "cluster_names": [
            "NORMAL",
            "PRE_SCALE",
            "PRE_FAILURE",
            "ACTIVE_DEGRADATION",
            "ANOMALY",
        ],
    }


def export_model(
    model: TemporalXDEC,
    output_dir: str,
    model_name: str = "scry_model",
    formats: tuple[str, ...] | None = None,
) -> dict[str, str]:
    """Export model to multiple formats.

    Args:
        model: Trained TemporalXDEC model.
        output_dir: Directory to save exported models.
        model_name: Base name for exported files (default: "scry_model").
        formats: Formats to export. Defaults to ("torchscript",) so the
            offline core needs no extra packages; "onnx" additionally
            requires the onnx and onnxscript packages.
                 Defaults to both if None.

    Returns:
        Dictionary mapping format names to output paths.
    """
    if formats is None:
        formats = ("torchscript",)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exported = {}

    if "torchscript" in formats:
        ts_path = output_dir / f"{model_name}.pt"
        exported["torchscript"] = export_to_torchscript(model, str(ts_path))

    if "onnx" in formats:
        onnx_path = output_dir / f"{model_name}.onnx"
        exported["onnx"] = export_to_onnx(model, str(onnx_path))

    # Save metadata alongside exports
    metadata = get_model_metadata(model)
    metadata_path = output_dir / f"{model_name}_metadata.json"

    import json
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    exported["metadata"] = str(metadata_path)
    logger.info("Exported model to %d formats in %s", len(formats), output_dir)

    return exported


def load_torchscript(model_path: str) -> nn.Module:
    """Load a TorchScript model for inference.

    Args:
        model_path: Path to the TorchScript model.

    Returns:
        Loaded TorchScript module.
    """
    return torch.jit.load(model_path)


def verify_onnx_export(onnx_path: str) -> bool:
    """Verify an ONNX model is valid.

    Args:
        onnx_path: Path to the ONNX model.

    Returns:
        True if model is valid.

    Raises:
        ImportError: If onnx package is not installed.
        onnx.checker.ValidationError: If model is invalid.
    """
    try:
        import onnx
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        logger.info("ONNX model verification passed: %s", onnx_path)
        return True
    except ImportError:
        logger.warning("onnx package not installed, skipping verification")
        raise
