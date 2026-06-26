# Description: GRU decoder modules for the X-DEC model.
# Description: Separate decoders for numerical and categorical sequence reconstruction.

"""GRU decoders for X-DEC model sequence reconstruction."""

import torch
import torch.nn as nn


class TemporalDecoder(nn.Module):
    """GRU decoder for sequence reconstruction.

    Reconstructs sequences from latent representations. Supports
    both linear and sigmoid output activations for numerical and
    categorical feature reconstruction respectively.
    """

    def __init__(
        self,
        latent_dim: int = 8,
        hidden_dim: int = 64,
        output_dim: int = 9,
        seq_len: int = 30,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_sigmoid: bool = False,
    ) -> None:
        """Initialize temporal decoder.

        Args:
            latent_dim: Dimension of input latent vector.
            hidden_dim: GRU hidden dimension.
            output_dim: Number of output features.
            seq_len: Length of output sequence.
            num_layers: Number of GRU layers.
            dropout: Dropout probability.
            use_sigmoid: Whether to apply sigmoid to output (for categorical).
        """
        super().__init__()

        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.use_sigmoid = use_sigmoid

        # Project latent to hidden dimension
        self.fc_in = nn.Linear(latent_dim, hidden_dim)

        # GRU for sequence generation
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Output projection
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to sequence.

        Args:
            z: Latent vector of shape (batch, latent_dim).

        Returns:
            Reconstructed sequence of shape (batch, seq_len, output_dim).
        """
        # Project to hidden dimension
        h = self.fc_in(z)  # (batch, hidden_dim)
        h = torch.relu(h)

        # Repeat across sequence length
        h_repeated = h.unsqueeze(1).repeat(1, self.seq_len, 1)  # (batch, seq_len, hidden_dim)

        # GRU decoding
        gru_out, _ = self.gru(h_repeated)  # (batch, seq_len, hidden_dim)

        # Project to output dimension
        output = self.fc_out(gru_out)  # (batch, seq_len, output_dim)

        if self.use_sigmoid:
            output = torch.sigmoid(output)

        return output


class NumericalDecoder(TemporalDecoder):
    """Decoder for numerical features (default 9 features, 64 hidden, linear output)."""

    def __init__(
        self,
        latent_dim: int = 8,
        hidden_dim: int = 64,
        output_dim: int = 9,
        seq_len: int = 30,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            seq_len=seq_len,
            num_layers=num_layers,
            dropout=dropout,
            use_sigmoid=False,
        )


class CategoricalDecoder(TemporalDecoder):
    """Decoder for categorical features (default 8 features, 32 hidden, sigmoid output)."""

    def __init__(
        self,
        latent_dim: int = 8,
        hidden_dim: int = 32,
        output_dim: int = 8,
        seq_len: int = 30,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            seq_len=seq_len,
            num_layers=num_layers,
            dropout=dropout,
            use_sigmoid=True,
        )
