# Description: Temporal X-DEC model for infrastructure operational state discovery.
# Description: Combines XVAE autoencoder with DEC clustering layer.

"""Temporal X-DEC model combining XVAE and DEC clustering."""

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from scry.model.clustering import DECLayer
from scry.model.xvae import TemporalXVAE

if TYPE_CHECKING:
    from scry.config import FeatureConfig


class TemporalXDEC(nn.Module):
    """Temporal X-DEC for mixed numerical/categorical infrastructure metrics.

    Architecture:
        Numerical → BiGRU → ┐
                            ├→ Merged → VAE Latent (z) → DEC Clustering
        Categorical → BiGRU → ┘
                                    │
                            ┌───────┴───────┐
                            ↓               ↓
                    Numerical Decoder   Categorical Decoder

    The model learns to:
    1. Encode time-series windows into latent representations
    2. Cluster latent space into operational states
    3. Reconstruct inputs for representation learning

    Args:
        num_numerical: Number of numerical features.
        num_categorical: Number of categorical features.
        seq_len: Sequence length for windows (default: 30).
        num_hidden: Hidden size for numerical GRU (default: 64).
        cat_hidden: Hidden size for categorical GRU (default: 32).
        latent_dim: Dimension of VAE latent space (default: 8).
        n_clusters: Number of operational state clusters (default: 5).
        alpha: Student's t-distribution degrees of freedom (default: 1.0).
    """

    def __init__(
        self,
        num_numerical: int,
        num_categorical: int,
        seq_len: int = 30,
        num_hidden: int = 64,
        cat_hidden: int = 32,
        latent_dim: int = 8,
        n_clusters: int = 5,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()

        self.num_numerical = num_numerical
        self.num_categorical = num_categorical
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.n_clusters = n_clusters

        # XVAE for encoding/decoding
        self.xvae = TemporalXVAE(
            num_numerical=num_numerical,
            num_categorical=num_categorical,
            seq_len=seq_len,
            num_hidden=num_hidden,
            cat_hidden=cat_hidden,
            latent_dim=latent_dim,
        )

        # DEC clustering layer
        self.dec_layer = DECLayer(
            n_clusters=n_clusters,
            latent_dim=latent_dim,
            alpha=alpha,
        )

    @classmethod
    def from_config(
        cls,
        config: "FeatureConfig",
        n_clusters: int = 5,
        alpha: float = 1.0,
    ) -> "TemporalXDEC":
        """Create model from feature configuration.

        Args:
            config: Feature configuration with numerical/categorical features.
            n_clusters: Number of operational state clusters (default: 5).
            alpha: Student's t-distribution degrees of freedom (default: 1.0).

        Returns:
            Configured TemporalXDEC model.
        """
        model_params = config.model_config

        return cls(
            num_numerical=config.num_numerical,
            num_categorical=config.num_categorical,
            seq_len=model_params.get("seq_len", 30),
            num_hidden=model_params.get("num_hidden", 64),
            cat_hidden=model_params.get("cat_hidden", 32),
            latent_dim=model_params.get("latent_dim", 8),
            n_clusters=n_clusters,
            alpha=alpha,
        )

    def forward(
        self, x_num: torch.Tensor, x_cat: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Forward pass through the complete X-DEC model.

        Args:
            x_num: Numerical features (batch, seq_len, num_numerical).
            x_cat: Categorical features (batch, seq_len, num_categorical).

        Returns:
            Dict containing z, mu, logvar, x_num_recon, x_cat_recon, q.
        """
        # XVAE forward pass
        xvae_output = self.xvae(x_num, x_cat)

        # DEC clustering on latent space
        q = self.dec_layer(xvae_output["z"])

        # Combine outputs
        return {
            "z": xvae_output["z"],
            "mu": xvae_output["mu"],
            "logvar": xvae_output["logvar"],
            "x_num_recon": xvae_output["x_num_recon"],
            "x_cat_recon": xvae_output["x_cat_recon"],
            "q": q,
        }

    def encode(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """Encode inputs to latent space.

        Args:
            x_num: Numerical features (batch, seq_len, num_numerical).
            x_cat: Categorical features (batch, seq_len, num_categorical).

        Returns:
            Latent embeddings z (batch, latent_dim).
        """
        z, _, _ = self.xvae.encode(x_num, x_cat)
        return z

    def get_cluster_assignments(
        self, x_num: torch.Tensor, x_cat: torch.Tensor
    ) -> torch.Tensor:
        """Get soft cluster assignments for inputs.

        Args:
            x_num: Numerical features (batch, seq_len, num_numerical).
            x_cat: Categorical features (batch, seq_len, num_categorical).

        Returns:
            Soft cluster assignments q (batch, n_clusters).
        """
        z = self.encode(x_num, x_cat)
        return self.dec_layer(z)

    def predict_cluster(
        self, x_num: torch.Tensor, x_cat: torch.Tensor
    ) -> torch.Tensor:
        """Predict hard cluster labels for inputs.

        Args:
            x_num: Numerical features (batch, seq_len, num_numerical).
            x_cat: Categorical features (batch, seq_len, num_categorical).

        Returns:
            Cluster labels (batch,) as long tensor.
        """
        q = self.get_cluster_assignments(x_num, x_cat)
        return q.argmax(dim=1)

    def initialize_centroids(
        self, x_num: torch.Tensor, x_cat: torch.Tensor
    ) -> None:
        """Initialize DEC centroids from data using k-means++.

        Should be called after pretraining the XVAE component.

        Args:
            x_num: Numerical features (n_samples, seq_len, num_numerical).
            x_cat: Categorical features (n_samples, seq_len, num_categorical).
        """
        with torch.no_grad():
            z = self.encode(x_num, x_cat)
            self.dec_layer.initialize_centroids(z)
