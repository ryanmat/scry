# Description: Unit tests for the Temporal XVAE module.
# Description: Tests the X-shaped VAE with merged latent space.

"""Tests for scry.model.xvae module."""

import torch


class TestTemporalXVAE:
    """Tests for TemporalXVAE module."""

    def test_forward_returns_expected_keys(self) -> None:
        """forward should return dict with all expected keys."""
        from scry.model.xvae import TemporalXVAE

        xvae = TemporalXVAE()

        x_num = torch.randn(8, 30, 9)
        x_cat = torch.randn(8, 30, 8)

        output = xvae(x_num, x_cat)

        assert "z" in output
        assert "mu" in output
        assert "logvar" in output
        assert "x_num_recon" in output
        assert "x_cat_recon" in output

    def test_latent_dimensions(self) -> None:
        """Latent variables should have correct dimensions."""
        from scry.model.xvae import TemporalXVAE

        xvae = TemporalXVAE(latent_dim=8)

        x_num = torch.randn(4, 30, 9)
        x_cat = torch.randn(4, 30, 8)

        output = xvae(x_num, x_cat)

        assert output["z"].shape == (4, 8)
        assert output["mu"].shape == (4, 8)
        assert output["logvar"].shape == (4, 8)

    def test_reconstruction_dimensions(self) -> None:
        """Reconstructions should match input dimensions."""
        from scry.model.xvae import TemporalXVAE

        xvae = TemporalXVAE(num_numerical=9, num_categorical=8, seq_len=30)

        x_num = torch.randn(4, 30, 9)
        x_cat = torch.randn(4, 30, 8)

        output = xvae(x_num, x_cat)

        assert output["x_num_recon"].shape == (4, 30, 9)
        assert output["x_cat_recon"].shape == (4, 30, 8)

    def test_encode_returns_latent(self) -> None:
        """encode should return z, mu, logvar."""
        from scry.model.xvae import TemporalXVAE

        xvae = TemporalXVAE(latent_dim=8)

        x_num = torch.randn(4, 30, 9)
        x_cat = torch.randn(4, 30, 8)

        z, mu, logvar = xvae.encode(x_num, x_cat)

        assert z.shape == (4, 8)
        assert mu.shape == (4, 8)
        assert logvar.shape == (4, 8)

    def test_decode_returns_reconstructions(self) -> None:
        """decode should return both reconstructions."""
        from scry.model.xvae import TemporalXVAE

        xvae = TemporalXVAE()

        z = torch.randn(4, 8)
        x_num_recon, x_cat_recon = xvae.decode(z)

        assert x_num_recon.shape == (4, 30, 9)
        assert x_cat_recon.shape == (4, 30, 8)

    def test_reparameterization_is_stochastic(self) -> None:
        """Reparameterization should produce different samples."""
        from scry.model.xvae import TemporalXVAE

        xvae = TemporalXVAE()

        mu = torch.zeros(4, 8)
        logvar = torch.zeros(4, 8)  # std = 1

        z1 = xvae.reparameterize(mu, logvar)
        z2 = xvae.reparameterize(mu, logvar)

        # Should be different due to random sampling
        assert not torch.allclose(z1, z2)

    def test_deterministic_limit(self) -> None:
        """With very small variance, z should approximately equal mu."""
        from scry.model.xvae import TemporalXVAE

        xvae = TemporalXVAE()

        mu = torch.randn(4, 8)
        # Very small variance: std = exp(-15) ~ 3e-7, so z = mu + eps*std stays
        # within tolerance of mu for any standard-normal eps, not just a lucky draw.
        logvar = torch.full((4, 8), -30.0)

        z = xvae.reparameterize(mu, logvar)

        torch.testing.assert_close(z, mu, atol=1e-4, rtol=1e-4)

    def test_model_is_differentiable(self) -> None:
        """Model should support gradient computation."""
        from scry.model.xvae import TemporalXVAE

        xvae = TemporalXVAE()

        x_num = torch.randn(4, 30, 9, requires_grad=True)
        x_cat = torch.randn(4, 30, 8, requires_grad=True)

        output = xvae(x_num, x_cat)
        loss = output["z"].sum() + output["x_num_recon"].sum()
        loss.backward()

        assert x_num.grad is not None
        assert x_cat.grad is not None

    def test_batch_size_one(self) -> None:
        """Model should handle batch size of 1."""
        from scry.model.xvae import TemporalXVAE

        xvae = TemporalXVAE()

        x_num = torch.randn(1, 30, 9)
        x_cat = torch.randn(1, 30, 8)

        output = xvae(x_num, x_cat)

        assert output["z"].shape == (1, 8)


class TestNumericalOnlyXVAE:
    """Tests for the optional categorical branch (num_categorical=0)."""

    def test_constructor_drops_categorical_branch(self) -> None:
        """With num_categorical=0 the categorical modules are not built."""
        from scry.model.xvae import TemporalXVAE

        xvae = TemporalXVAE(num_numerical=9, num_categorical=0, num_hidden=64)

        assert xvae.categorical_encoder is None
        assert xvae.categorical_decoder is None
        # merged_dim collapses to the numerical branch only (2 * num_hidden).
        assert xvae.fc_mu.in_features == 2 * 64
        assert xvae.fc_logvar.in_features == 2 * 64

    def test_encode_numerical_only(self) -> None:
        """encode should work with a zero-width categorical input."""
        from scry.model.xvae import TemporalXVAE

        xvae = TemporalXVAE(num_numerical=9, num_categorical=0, latent_dim=8)

        x_num = torch.randn(4, 30, 9)
        x_cat = torch.empty(4, 30, 0)

        z, mu, logvar = xvae.encode(x_num, x_cat)

        assert z.shape == (4, 8)
        assert mu.shape == (4, 8)
        assert logvar.shape == (4, 8)
        assert torch.isfinite(z).all()

    def test_forward_numerical_only(self) -> None:
        """forward should return a numerical reconstruction and zero-width x_cat_recon."""
        from scry.model.xvae import TemporalXVAE

        xvae = TemporalXVAE(num_numerical=9, num_categorical=0, seq_len=30)

        x_num = torch.randn(4, 30, 9)
        x_cat = torch.empty(4, 30, 0)

        output = xvae(x_num, x_cat)

        assert output["x_num_recon"].shape == (4, 30, 9)
        assert output["x_cat_recon"].shape == (4, 30, 0)
        assert torch.isfinite(output["x_num_recon"]).all()

    def test_decode_numerical_only(self) -> None:
        """decode should produce a zero-width categorical reconstruction."""
        from scry.model.xvae import TemporalXVAE

        xvae = TemporalXVAE(num_numerical=9, num_categorical=0, seq_len=30)

        z = torch.randn(4, 8)
        x_num_recon, x_cat_recon = xvae.decode(z)

        assert x_num_recon.shape == (4, 30, 9)
        assert x_cat_recon.shape[-1] == 0
        assert x_cat_recon.shape == (4, 30, 0)
        assert x_cat_recon.dtype == x_num_recon.dtype

    def test_numerical_only_is_differentiable(self) -> None:
        """Numerical-only model should support gradient computation."""
        from scry.model.xvae import TemporalXVAE

        xvae = TemporalXVAE(num_numerical=9, num_categorical=0)

        x_num = torch.randn(4, 30, 9, requires_grad=True)
        x_cat = torch.empty(4, 30, 0)

        output = xvae(x_num, x_cat)
        loss = output["z"].sum() + output["x_num_recon"].sum()
        loss.backward()

        assert x_num.grad is not None

    def test_categorical_branch_unchanged_when_present(self) -> None:
        """With num_categorical>0 the merged dim and forward are unchanged."""
        from scry.model.xvae import TemporalXVAE

        xvae = TemporalXVAE(
            num_numerical=9, num_categorical=8, num_hidden=64, cat_hidden=32
        )

        assert xvae.categorical_encoder is not None
        assert xvae.categorical_decoder is not None
        # merged_dim == 2 * num_hidden + 2 * cat_hidden.
        assert xvae.fc_mu.in_features == 2 * 64 + 2 * 32
        assert xvae.fc_logvar.in_features == 2 * 64 + 2 * 32

        x_num = torch.randn(4, 30, 9)
        x_cat = torch.randn(4, 30, 8)
        output = xvae(x_num, x_cat)

        assert output["x_num_recon"].shape == (4, 30, 9)
        assert output["x_cat_recon"].shape == (4, 30, 8)

    def test_optimization_iterations_finite_loss(self) -> None:
        """A few XDECLoss optimization steps on numerical-only data stay finite."""
        from torch.optim import Adam

        from scry.model.losses import XDECLoss
        from scry.model.xvae import TemporalXVAE

        torch.manual_seed(0)
        xvae = TemporalXVAE(num_numerical=9, num_categorical=0, seq_len=30)
        loss_fn = XDECLoss(beta=1.0)
        optimizer = Adam(xvae.parameters(), lr=1e-3)

        x_num = torch.randn(16, 30, 9)
        x_cat = torch.empty(16, 30, 0)

        losses = []
        for _ in range(5):
            optimizer.zero_grad()
            outputs = xvae(x_num, x_cat)
            result = loss_fn(outputs, x_num, x_cat)
            loss = result["loss"]
            loss.backward()
            optimizer.step()

            assert torch.isfinite(loss)
            assert result["recon_cat"].item() == 0.0
            losses.append(loss.item())

        assert all(torch.isfinite(torch.tensor(value)) for value in losses)
