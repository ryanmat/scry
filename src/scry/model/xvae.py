# Description: Temporal X-shaped VAE with merged latent space.
# Description: Combines dual GRU encoders/decoders with VAE reparameterization.

"""Temporal XVAE module for mixed numerical/categorical infrastructure metrics."""


import torch
import torch.nn as nn

from scry.model.decoders import CategoricalDecoder, NumericalDecoder
from scry.model.encoders import CategoricalEncoder, NumericalEncoder


class TemporalXVAE(nn.Module):
    """X-shaped VAE with dual encoder/decoder branches.

    Architecture:
        Numerical → BiGRU → ┐
                            ├→ Merged → VAE Latent (z)
        Categorical → BiGRU → ┘
                                    │
                            ┌───────┴───────┐
                            ↓               ↓
                    Numerical Decoder   Categorical Decoder

    Args:
        num_numerical: Number of numerical features (default: 9).
        num_categorical: Number of categorical features (default: 8).
        seq_len: Sequence length for reconstruction (default: 30).
        num_hidden: Hidden size for numerical GRU (default: 64).
        cat_hidden: Hidden size for categorical GRU (default: 32).
        latent_dim: Dimension of VAE latent space (default: 8).
    """

    def __init__(
        self,
        num_numerical: int = 9,
        num_categorical: int = 8,
        seq_len: int = 30,
        num_hidden: int = 64,
        cat_hidden: int = 32,
        latent_dim: int = 8,
    ) -> None:
        super().__init__()

        self.num_numerical = num_numerical
        self.num_categorical = num_categorical
        self.seq_len = seq_len
        self.latent_dim = latent_dim

        # Dual encoders
        self.numerical_encoder = NumericalEncoder(
            input_dim=num_numerical,
            hidden_dim=num_hidden,
        )
        self.categorical_encoder = CategoricalEncoder(
            input_dim=num_categorical,
            hidden_dim=cat_hidden,
        )

        # Merged encoder output dimension (bidirectional: 2 * hidden for each)
        merged_dim = 2 * num_hidden + 2 * cat_hidden  # 128 + 64 = 192

        # VAE latent projections
        self.fc_mu = nn.Linear(merged_dim, latent_dim)
        self.fc_logvar = nn.Linear(merged_dim, latent_dim)

        # Dual decoders
        self.numerical_decoder = NumericalDecoder(
            latent_dim=latent_dim,
            output_dim=num_numerical,
            seq_len=seq_len,
            hidden_dim=num_hidden,
        )
        self.categorical_decoder = CategoricalDecoder(
            latent_dim=latent_dim,
            output_dim=num_categorical,
            seq_len=seq_len,
            hidden_dim=cat_hidden,
        )

    def encode(
        self, x_num: torch.Tensor, x_cat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode inputs to latent space.

        Args:
            x_num: Numerical features (batch, seq_len, num_numerical).
            x_cat: Categorical features (batch, seq_len, num_categorical).

        Returns:
            Tuple of (z, mu, logvar), each of shape (batch, latent_dim).
        """
        # Encode each branch
        h_num = self.numerical_encoder(x_num)  # (batch, 2*num_hidden)
        h_cat = self.categorical_encoder(x_cat)  # (batch, 2*cat_hidden)

        # Merge representations
        h_merged = torch.cat([h_num, h_cat], dim=1)  # (batch, merged_dim)

        # Project to latent parameters
        mu = self.fc_mu(h_merged)
        logvar = self.fc_logvar(h_merged)

        # Reparameterization trick
        z = self.reparameterize(mu, logvar)

        return z, mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Apply reparameterization trick: z = mu + std * epsilon.

        Args:
            mu: Mean of latent distribution (batch, latent_dim).
            logvar: Log variance of latent distribution (batch, latent_dim).

        Returns:
            Sampled latent vector z (batch, latent_dim).
        """
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mu + std * epsilon

    def decode(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode latent vector to reconstructions.

        Args:
            z: Latent vector (batch, latent_dim).

        Returns:
            Tuple of (x_num_recon, x_cat_recon).
        """
        x_num_recon = self.numerical_decoder(z)
        x_cat_recon = self.categorical_decoder(z)
        return x_num_recon, x_cat_recon

    def forward(
        self, x_num: torch.Tensor, x_cat: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Forward pass through the XVAE.

        Args:
            x_num: Numerical features (batch, seq_len, num_numerical).
            x_cat: Categorical features (batch, seq_len, num_categorical).

        Returns:
            Dict containing z, mu, logvar, x_num_recon, x_cat_recon.
        """
        z, mu, logvar = self.encode(x_num, x_cat)
        x_num_recon, x_cat_recon = self.decode(z)

        return {
            "z": z,
            "mu": mu,
            "logvar": logvar,
            "x_num_recon": x_num_recon,
            "x_cat_recon": x_cat_recon,
        }
