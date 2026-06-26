# Description: Unit tests for the GRU encoder modules.
# Description: Tests numerical and categorical encoders with temporal attention.

"""Tests for scry.model.encoders module."""

import torch


class TestTemporalAttention:
    """Tests for TemporalAttention module."""

    def test_attention_output_shape(self) -> None:
        """Attention should reduce time dimension."""
        from scry.model.encoders import TemporalAttention

        attention = TemporalAttention(hidden_dim=64)

        # Input: (batch, seq_len, hidden_dim)
        x = torch.randn(8, 30, 64)
        output = attention(x)

        # Output should be (batch, hidden_dim)
        assert output.shape == (8, 64)

    def test_attention_weights_sum_to_one(self) -> None:
        """Attention weights should sum to 1 across time dimension."""
        from scry.model.encoders import TemporalAttention

        attention = TemporalAttention(hidden_dim=64)

        x = torch.randn(8, 30, 64)
        weights = attention.get_weights(x)

        # Weights should be (batch, seq_len)
        assert weights.shape == (8, 30)

        # Should sum to 1 for each sample
        sums = weights.sum(dim=1)
        torch.testing.assert_close(sums, torch.ones(8), atol=1e-5, rtol=1e-5)

    def test_attention_is_differentiable(self) -> None:
        """Attention should support gradient computation."""
        from scry.model.encoders import TemporalAttention

        attention = TemporalAttention(hidden_dim=64)

        x = torch.randn(8, 30, 64, requires_grad=True)
        output = attention(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None


class TestNumericalEncoder:
    """Tests for NumericalEncoder module."""

    def test_encoder_output_shape(self) -> None:
        """Encoder should output correct shape for numerical features."""
        from scry.model.encoders import NumericalEncoder

        encoder = NumericalEncoder(input_dim=9, hidden_dim=64)

        # Input: (batch, seq_len, 9 numerical features)
        x = torch.randn(8, 30, 9)
        output = encoder(x)

        # Output should be (batch, 128) for bidirectional GRU
        assert output.shape == (8, 128)

    def test_encoder_handles_different_seq_lengths(self) -> None:
        """Encoder should handle different sequence lengths."""
        from scry.model.encoders import NumericalEncoder

        encoder = NumericalEncoder(input_dim=9, hidden_dim=64)

        for seq_len in [10, 30, 60]:
            x = torch.randn(4, seq_len, 9)
            output = encoder(x)
            assert output.shape == (4, 128)

    def test_encoder_bidirectional_doubles_output(self) -> None:
        """Bidirectional GRU should double the hidden dimension."""
        from scry.model.encoders import NumericalEncoder

        encoder = NumericalEncoder(input_dim=9, hidden_dim=64, bidirectional=True)
        x = torch.randn(4, 30, 9)
        output = encoder(x)

        # 64 * 2 = 128 for bidirectional
        assert output.shape == (4, 128)

    def test_encoder_is_differentiable(self) -> None:
        """Encoder should support gradient computation."""
        from scry.model.encoders import NumericalEncoder

        encoder = NumericalEncoder(input_dim=9, hidden_dim=64)

        x = torch.randn(4, 30, 9, requires_grad=True)
        output = encoder(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None

    def test_encoder_batch_size_one(self) -> None:
        """Encoder should handle batch size of 1."""
        from scry.model.encoders import NumericalEncoder

        encoder = NumericalEncoder(input_dim=9, hidden_dim=64)

        x = torch.randn(1, 30, 9)
        output = encoder(x)

        assert output.shape == (1, 128)


class TestCategoricalEncoder:
    """Tests for CategoricalEncoder module."""

    def test_encoder_output_shape(self) -> None:
        """Encoder should output correct shape for categorical features."""
        from scry.model.encoders import CategoricalEncoder

        encoder = CategoricalEncoder(input_dim=8, hidden_dim=32)

        # Input: (batch, seq_len, 8 categorical features)
        x = torch.randn(8, 30, 8)
        output = encoder(x)

        # Output should be (batch, 64) for bidirectional GRU
        assert output.shape == (8, 64)

    def test_encoder_handles_different_seq_lengths(self) -> None:
        """Encoder should handle different sequence lengths."""
        from scry.model.encoders import CategoricalEncoder

        encoder = CategoricalEncoder(input_dim=8, hidden_dim=32)

        for seq_len in [10, 30, 60]:
            x = torch.randn(4, seq_len, 8)
            output = encoder(x)
            assert output.shape == (4, 64)

    def test_encoder_bidirectional_doubles_output(self) -> None:
        """Bidirectional GRU should double the hidden dimension."""
        from scry.model.encoders import CategoricalEncoder

        encoder = CategoricalEncoder(input_dim=8, hidden_dim=32, bidirectional=True)
        x = torch.randn(4, 30, 8)
        output = encoder(x)

        # 32 * 2 = 64 for bidirectional
        assert output.shape == (4, 64)

    def test_encoder_is_differentiable(self) -> None:
        """Encoder should support gradient computation."""
        from scry.model.encoders import CategoricalEncoder

        encoder = CategoricalEncoder(input_dim=8, hidden_dim=32)

        x = torch.randn(4, 30, 8, requires_grad=True)
        output = encoder(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None

    def test_encoder_batch_size_one(self) -> None:
        """Encoder should handle batch size of 1."""
        from scry.model.encoders import CategoricalEncoder

        encoder = CategoricalEncoder(input_dim=8, hidden_dim=32)

        x = torch.randn(1, 30, 8)
        output = encoder(x)

        assert output.shape == (1, 64)


class TestEncoderConsistency:
    """Tests for encoder consistency between branches."""

    def test_encoders_produce_different_outputs(self) -> None:
        """Different encoders should produce different outputs for same input."""
        from scry.model.encoders import CategoricalEncoder, NumericalEncoder

        num_encoder = NumericalEncoder(input_dim=9, hidden_dim=64)
        cat_encoder = CategoricalEncoder(input_dim=8, hidden_dim=32)

        # Use same random seed for reproducibility
        torch.manual_seed(42)
        x_num = torch.randn(4, 30, 9)
        x_cat = torch.randn(4, 30, 8)

        out_num = num_encoder(x_num)
        out_cat = cat_encoder(x_cat)

        # Outputs should have different shapes
        assert out_num.shape[1] != out_cat.shape[1]

    def test_encoders_can_be_combined(self) -> None:
        """Encoder outputs should be concatenatable."""
        from scry.model.encoders import CategoricalEncoder, NumericalEncoder

        num_encoder = NumericalEncoder(input_dim=9, hidden_dim=64)
        cat_encoder = CategoricalEncoder(input_dim=8, hidden_dim=32)

        x_num = torch.randn(4, 30, 9)
        x_cat = torch.randn(4, 30, 8)

        out_num = num_encoder(x_num)  # (4, 128)
        out_cat = cat_encoder(x_cat)  # (4, 64)

        # Should be concatenatable
        combined = torch.cat([out_num, out_cat], dim=1)
        assert combined.shape == (4, 192)
