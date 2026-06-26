# Description: Unit tests for the GRU decoder modules.
# Description: Tests numerical and categorical decoders for sequence reconstruction.

"""Tests for scry.model.decoders module."""

import torch


class TestNumericalDecoder:
    """Tests for NumericalDecoder module."""

    def test_decoder_output_shape(self) -> None:
        """Decoder should output correct shape for numerical reconstruction."""
        from scry.model.decoders import NumericalDecoder

        decoder = NumericalDecoder(latent_dim=8, hidden_dim=64, output_dim=9, seq_len=30)

        # Input: latent vector (batch, latent_dim)
        z = torch.randn(8, 8)
        output = decoder(z)

        # Output should be (batch, seq_len, output_dim)
        assert output.shape == (8, 30, 9)

    def test_decoder_handles_different_latent_dims(self) -> None:
        """Decoder should handle different latent dimensions."""
        from scry.model.decoders import NumericalDecoder

        for latent_dim in [4, 8, 16]:
            decoder = NumericalDecoder(
                latent_dim=latent_dim, hidden_dim=64, output_dim=9, seq_len=30
            )
            z = torch.randn(4, latent_dim)
            output = decoder(z)
            assert output.shape == (4, 30, 9)

    def test_decoder_linear_activation(self) -> None:
        """Numerical decoder should use linear activation (unbounded outputs)."""
        from scry.model.decoders import NumericalDecoder

        decoder = NumericalDecoder(latent_dim=8, hidden_dim=64, output_dim=9, seq_len=30)

        z = torch.randn(8, 8) * 10  # Large inputs
        output = decoder(z)

        # Should be able to produce values outside [0, 1]
        # With random weights, some outputs should be negative or > 1
        assert output.min() < 0 or output.max() > 1

    def test_decoder_is_differentiable(self) -> None:
        """Decoder should support gradient computation."""
        from scry.model.decoders import NumericalDecoder

        decoder = NumericalDecoder(latent_dim=8, hidden_dim=64, output_dim=9, seq_len=30)

        z = torch.randn(4, 8, requires_grad=True)
        output = decoder(z)
        loss = output.sum()
        loss.backward()

        assert z.grad is not None

    def test_decoder_batch_size_one(self) -> None:
        """Decoder should handle batch size of 1."""
        from scry.model.decoders import NumericalDecoder

        decoder = NumericalDecoder(latent_dim=8, hidden_dim=64, output_dim=9, seq_len=30)

        z = torch.randn(1, 8)
        output = decoder(z)

        assert output.shape == (1, 30, 9)


class TestCategoricalDecoder:
    """Tests for CategoricalDecoder module."""

    def test_decoder_output_shape(self) -> None:
        """Decoder should output correct shape for categorical reconstruction."""
        from scry.model.decoders import CategoricalDecoder

        decoder = CategoricalDecoder(latent_dim=8, hidden_dim=32, output_dim=8, seq_len=30)

        # Input: latent vector (batch, latent_dim)
        z = torch.randn(8, 8)
        output = decoder(z)

        # Output should be (batch, seq_len, output_dim)
        assert output.shape == (8, 30, 8)

    def test_decoder_sigmoid_activation(self) -> None:
        """Categorical decoder should use sigmoid activation (outputs in [0, 1])."""
        from scry.model.decoders import CategoricalDecoder

        decoder = CategoricalDecoder(latent_dim=8, hidden_dim=32, output_dim=8, seq_len=30)

        z = torch.randn(8, 8) * 10  # Large inputs
        output = decoder(z)

        # All outputs should be in [0, 1]
        assert output.min() >= 0
        assert output.max() <= 1

    def test_decoder_handles_different_latent_dims(self) -> None:
        """Decoder should handle different latent dimensions."""
        from scry.model.decoders import CategoricalDecoder

        for latent_dim in [4, 8, 16]:
            decoder = CategoricalDecoder(
                latent_dim=latent_dim, hidden_dim=32, output_dim=8, seq_len=30
            )
            z = torch.randn(4, latent_dim)
            output = decoder(z)
            assert output.shape == (4, 30, 8)

    def test_decoder_is_differentiable(self) -> None:
        """Decoder should support gradient computation."""
        from scry.model.decoders import CategoricalDecoder

        decoder = CategoricalDecoder(latent_dim=8, hidden_dim=32, output_dim=8, seq_len=30)

        z = torch.randn(4, 8, requires_grad=True)
        output = decoder(z)
        loss = output.sum()
        loss.backward()

        assert z.grad is not None

    def test_decoder_batch_size_one(self) -> None:
        """Decoder should handle batch size of 1."""
        from scry.model.decoders import CategoricalDecoder

        decoder = CategoricalDecoder(latent_dim=8, hidden_dim=32, output_dim=8, seq_len=30)

        z = torch.randn(1, 8)
        output = decoder(z)

        assert output.shape == (1, 30, 8)


class TestDecoderConsistency:
    """Tests for decoder consistency with encoders."""

    def test_decoders_reconstruct_correct_dimensions(self) -> None:
        """Decoders should reconstruct original input dimensions."""
        from scry.model.decoders import CategoricalDecoder, NumericalDecoder

        num_decoder = NumericalDecoder(latent_dim=8, hidden_dim=64, output_dim=9, seq_len=30)
        cat_decoder = CategoricalDecoder(latent_dim=8, hidden_dim=32, output_dim=8, seq_len=30)

        z = torch.randn(4, 8)

        num_recon = num_decoder(z)
        cat_recon = cat_decoder(z)

        # Should match original input dimensions
        assert num_recon.shape == (4, 30, 9)  # Same as numerical input
        assert cat_recon.shape == (4, 30, 8)  # Same as categorical input
