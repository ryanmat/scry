# Description: GRU encoder modules for the X-DEC model.
# Description: Separate encoders for numerical and categorical feature branches.

"""GRU encoders with temporal attention for X-DEC model."""

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812 -- PyTorch convention


class TemporalAttention(nn.Module):
    """Attention mechanism over temporal dimension.

    Computes weighted sum of hidden states across time steps,
    allowing the model to focus on relevant time points.
    """

    def __init__(self, hidden_dim: int) -> None:
        """Initialize temporal attention.

        Args:
            hidden_dim: Dimension of input hidden states.
        """
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute attention-weighted output.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim).

        Returns:
            Attended output of shape (batch, hidden_dim).
        """
        # Compute attention scores
        scores = self.attention(x)  # (batch, seq_len, 1)
        weights = F.softmax(scores, dim=1)  # (batch, seq_len, 1)

        # Weighted sum
        attended = (x * weights).sum(dim=1)  # (batch, hidden_dim)

        return attended

    def get_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Get attention weights for interpretability.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim).

        Returns:
            Attention weights of shape (batch, seq_len).
        """
        scores = self.attention(x)  # (batch, seq_len, 1)
        weights = F.softmax(scores, dim=1)  # (batch, seq_len, 1)
        return weights.squeeze(-1)  # (batch, seq_len)


class TemporalEncoder(nn.Module):
    """Bidirectional GRU encoder with temporal attention.

    Encodes sequences of features into fixed-size representations
    using GRU and temporal attention. Used for both numerical and
    categorical feature branches with different default configurations.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ) -> None:
        """Initialize temporal encoder.

        Args:
            input_dim: Number of input features.
            hidden_dim: GRU hidden dimension.
            num_layers: Number of GRU layers.
            dropout: Dropout probability.
            bidirectional: Whether to use bidirectional GRU.
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        self.layer_norm = nn.LayerNorm(hidden_dim * self.num_directions)

        self.attention = TemporalAttention(hidden_dim * self.num_directions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode sequence.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim).

        Returns:
            Encoded representation of shape (batch, hidden_dim * num_directions).
        """
        # GRU encoding
        gru_out, _ = self.gru(x)  # (batch, seq_len, hidden_dim * num_directions)

        # Layer normalization
        normalized = self.layer_norm(gru_out)

        # Temporal attention
        attended = self.attention(normalized)  # (batch, hidden_dim * num_directions)

        return attended


class NumericalEncoder(TemporalEncoder):
    """Encoder for numerical features with default config (9 features, 64 hidden)."""

    def __init__(
        self,
        input_dim: int = 9,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )


class CategoricalEncoder(TemporalEncoder):
    """Encoder for categorical features with default config (8 features, 32 hidden)."""

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )
