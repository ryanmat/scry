# Description: Temporal X-DEC model architecture including GRU encoders,
# Description: decoders, XVAE, and DEC clustering layer.

"""Temporal X-DEC model for mixed numerical/categorical metric clustering."""

from scry.model.clustering import DECLayer, compute_target_distribution
from scry.model.decoders import (
    CategoricalDecoder,
    NumericalDecoder,
    TemporalDecoder,
)
from scry.model.encoders import (
    CategoricalEncoder,
    NumericalEncoder,
    TemporalAttention,
    TemporalEncoder,
)
from scry.model.evaluate import (
    cluster_summary,
    evaluate_clustering,
    get_embeddings,
    visualize_clusters,
)
from scry.model.export import (
    InferenceWrapper,
    export_model,
    export_to_onnx,
    export_to_torchscript,
    get_model_metadata,
    load_torchscript,
    verify_onnx_export,
)
from scry.model.losses import (
    XDECLoss,
    dec_clustering_loss,
    reconstruction_loss_categorical,
    reconstruction_loss_numerical,
    vae_kl_loss,
)
from scry.model.trainer import XDECTrainer
from scry.model.xdec import TemporalXDEC
from scry.model.xvae import TemporalXVAE

__all__ = [
    # Encoders
    "TemporalAttention",
    "TemporalEncoder",
    "NumericalEncoder",
    "CategoricalEncoder",
    # Decoders
    "TemporalDecoder",
    "NumericalDecoder",
    "CategoricalDecoder",
    # XVAE
    "TemporalXVAE",
    # Losses
    "reconstruction_loss_numerical",
    "reconstruction_loss_categorical",
    "vae_kl_loss",
    "dec_clustering_loss",
    "XDECLoss",
    # Clustering
    "DECLayer",
    "compute_target_distribution",
    # Complete model
    "TemporalXDEC",
    # Training
    "XDECTrainer",
    # Evaluation
    "evaluate_clustering",
    "visualize_clusters",
    "cluster_summary",
    "get_embeddings",
    # Export
    "InferenceWrapper",
    "export_to_torchscript",
    "export_to_onnx",
    "export_model",
    "get_model_metadata",
    "load_torchscript",
    "verify_onnx_export",
]
