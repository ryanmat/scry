# Description: Unit tests for the loss function modules.
# Description: Tests reconstruction losses, KL divergences, and combined X-DEC loss.

"""Tests for scry.model.losses module."""

import torch


class TestReconstructionLosses:
    """Tests for reconstruction loss functions."""

    def test_numerical_loss_is_non_negative(self) -> None:
        """Numerical reconstruction loss should be non-negative."""
        from scry.model.losses import reconstruction_loss_numerical

        x = torch.randn(8, 30, 9)
        x_recon = torch.randn(8, 30, 9)

        loss = reconstruction_loss_numerical(x, x_recon)

        assert loss >= 0

    def test_numerical_loss_zero_for_perfect_reconstruction(self) -> None:
        """Numerical loss should be zero for perfect reconstruction."""
        from scry.model.losses import reconstruction_loss_numerical

        x = torch.randn(8, 30, 9)

        loss = reconstruction_loss_numerical(x, x)

        assert loss == 0

    def test_categorical_loss_is_non_negative(self) -> None:
        """Categorical reconstruction loss should be non-negative."""
        from scry.model.losses import reconstruction_loss_categorical

        x = torch.rand(8, 30, 8)  # Values in [0, 1]
        x_recon = torch.rand(8, 30, 8)

        loss = reconstruction_loss_categorical(x, x_recon)

        assert loss >= 0

    def test_categorical_loss_lower_for_better_reconstruction(self) -> None:
        """Categorical loss should be lower for better reconstruction."""
        from scry.model.losses import reconstruction_loss_categorical

        x = torch.rand(8, 30, 8)
        x_good = x + torch.randn_like(x) * 0.01  # Small perturbation
        x_good = torch.clamp(x_good, 0, 1)
        x_bad = torch.rand(8, 30, 8)  # Random reconstruction

        loss_good = reconstruction_loss_categorical(x, x_good)
        loss_bad = reconstruction_loss_categorical(x, x_bad)

        assert loss_good < loss_bad

    def test_categorical_loss_zero_for_zero_width(self) -> None:
        """Categorical loss should be a zero scalar when the tensor is zero-width."""
        from scry.model.losses import reconstruction_loss_categorical

        x = torch.empty(8, 30, 0)
        x_recon = torch.empty(8, 30, 0)

        loss = reconstruction_loss_categorical(x, x_recon)

        assert loss.shape == ()
        assert loss.item() == 0.0


class TestKLDivergence:
    """Tests for KL divergence loss functions."""

    def test_vae_kl_loss_is_non_negative(self) -> None:
        """VAE KL loss should be non-negative."""
        from scry.model.losses import vae_kl_loss

        mu = torch.randn(8, 8)
        logvar = torch.randn(8, 8)

        loss = vae_kl_loss(mu, logvar)

        assert loss >= 0

    def test_vae_kl_loss_zero_for_standard_normal(self) -> None:
        """VAE KL loss should be zero when q = N(0, 1)."""
        from scry.model.losses import vae_kl_loss

        mu = torch.zeros(8, 8)
        logvar = torch.zeros(8, 8)  # std = 1

        loss = vae_kl_loss(mu, logvar)

        assert loss < 0.01  # Should be approximately zero

    def test_dec_clustering_loss_is_non_negative(self) -> None:
        """DEC clustering loss should be non-negative."""
        from scry.model.losses import dec_clustering_loss

        # Soft assignments (must sum to 1 per sample)
        q = torch.softmax(torch.randn(8, 5), dim=1)
        p = torch.softmax(torch.randn(8, 5), dim=1)

        loss = dec_clustering_loss(q, p)

        assert loss >= 0


class TestXDECLoss:
    """Tests for combined XDECLoss class."""

    def test_loss_returns_dict(self) -> None:
        """XDECLoss should return dict with component losses."""
        from scry.model.losses import XDECLoss

        loss_fn = XDECLoss()

        outputs = {
            "x_num_recon": torch.randn(8, 30, 9),
            "x_cat_recon": torch.sigmoid(torch.randn(8, 30, 8)),
            "mu": torch.randn(8, 8),
            "logvar": torch.randn(8, 8),
        }
        x_num = torch.randn(8, 30, 9)
        x_cat = torch.rand(8, 30, 8)

        result = loss_fn(outputs, x_num, x_cat)

        assert "loss" in result
        assert "recon_num" in result
        assert "recon_cat" in result
        assert "kl_vae" in result

    def test_loss_combines_components(self) -> None:
        """Total loss should combine component losses."""
        from scry.model.losses import XDECLoss

        loss_fn = XDECLoss(beta=1.0)

        outputs = {
            "x_num_recon": torch.randn(8, 30, 9),
            "x_cat_recon": torch.sigmoid(torch.randn(8, 30, 8)),
            "mu": torch.randn(8, 8),
            "logvar": torch.randn(8, 8),
        }
        x_num = torch.randn(8, 30, 9)
        x_cat = torch.rand(8, 30, 8)

        result = loss_fn(outputs, x_num, x_cat)

        # Total should be sum of components (approximately)
        expected = result["recon_num"] + result["recon_cat"] + result["kl_vae"]
        torch.testing.assert_close(result["loss"], expected)

    def test_loss_with_clustering(self) -> None:
        """XDECLoss should include clustering loss when q and p provided."""
        from scry.model.losses import XDECLoss

        loss_fn = XDECLoss(beta=1.0, lambda_cluster=0.1)

        outputs = {
            "x_num_recon": torch.randn(8, 30, 9),
            "x_cat_recon": torch.sigmoid(torch.randn(8, 30, 8)),
            "mu": torch.randn(8, 8),
            "logvar": torch.randn(8, 8),
        }
        x_num = torch.randn(8, 30, 9)
        x_cat = torch.rand(8, 30, 8)
        q = torch.softmax(torch.randn(8, 5), dim=1)
        p = torch.softmax(torch.randn(8, 5), dim=1)

        result = loss_fn(outputs, x_num, x_cat, q=q, p=p)

        assert "kl_cluster" in result
        assert result["kl_cluster"] >= 0

    def test_loss_is_differentiable(self) -> None:
        """Loss should support gradient computation."""
        from scry.model.losses import XDECLoss

        loss_fn = XDECLoss()

        x_num_recon = torch.randn(8, 30, 9, requires_grad=True)
        outputs = {
            "x_num_recon": x_num_recon,
            "x_cat_recon": torch.sigmoid(torch.randn(8, 30, 8)),
            "mu": torch.randn(8, 8),
            "logvar": torch.randn(8, 8),
        }
        x_num = torch.randn(8, 30, 9)
        x_cat = torch.rand(8, 30, 8)

        result = loss_fn(outputs, x_num, x_cat)
        result["loss"].backward()

        assert x_num_recon.grad is not None

    def test_loss_recon_cat_zero_for_numerical_only(self) -> None:
        """recon_cat contributes zero when the categorical tensor is zero-width."""
        from scry.model.losses import XDECLoss

        loss_fn = XDECLoss(beta=1.0)

        outputs = {
            "x_num_recon": torch.randn(8, 30, 9),
            "x_cat_recon": torch.empty(8, 30, 0),
            "mu": torch.randn(8, 8),
            "logvar": torch.randn(8, 8),
        }
        x_num = torch.randn(8, 30, 9)
        x_cat = torch.empty(8, 30, 0)

        result = loss_fn(outputs, x_num, x_cat)

        assert result["recon_cat"].item() == 0.0
        # Total equals the numerical reconstruction plus KL, with no cat term.
        expected = result["recon_num"] + result["kl_vae"]
        torch.testing.assert_close(result["loss"], expected)


class TestClusterBalanceRegularization:
    """Tests for cluster balance entropy regularization."""

    def _make_outputs(self):
        """Helper to create dummy model outputs."""
        return {
            "x_num_recon": torch.randn(8, 30, 9),
            "x_cat_recon": torch.sigmoid(torch.randn(8, 30, 8)),
            "mu": torch.randn(8, 8),
            "logvar": torch.randn(8, 8),
        }

    def test_xdec_loss_accepts_lambda_balance(self) -> None:
        """XDECLoss should accept lambda_balance parameter."""
        from scry.model.losses import XDECLoss

        loss_fn = XDECLoss(lambda_balance=0.5)
        assert loss_fn.lambda_balance == 0.5

    def test_lambda_balance_default_zero(self) -> None:
        """lambda_balance should default to 0.0."""
        from scry.model.losses import XDECLoss

        loss_fn = XDECLoss()
        assert loss_fn.lambda_balance == 0.0

    def test_balance_entropy_included_when_q_provided(self) -> None:
        """Loss should include balance_entropy when q is provided and lambda_balance > 0."""
        from scry.model.losses import XDECLoss

        loss_fn = XDECLoss(lambda_balance=0.5, lambda_cluster=0.1)

        outputs = self._make_outputs()
        x_num = torch.randn(8, 30, 9)
        x_cat = torch.rand(8, 30, 8)
        q = torch.softmax(torch.randn(8, 5), dim=1)
        p = torch.softmax(torch.randn(8, 5), dim=1)

        result = loss_fn(outputs, x_num, x_cat, q=q, p=p)

        assert "balance_entropy" in result

    def test_uniform_q_has_max_entropy(self) -> None:
        """Uniform q should produce higher entropy (lower loss) than collapsed q."""
        from scry.model.losses import XDECLoss

        loss_fn = XDECLoss(lambda_balance=1.0, lambda_cluster=0.1)

        outputs = self._make_outputs()
        x_num = torch.randn(8, 30, 9)
        x_cat = torch.rand(8, 30, 8)
        p = torch.softmax(torch.randn(8, 5), dim=1)

        # Uniform q: all clusters equally likely
        q_uniform = torch.ones(8, 5) / 5.0

        # Collapsed q: most mass on cluster 0
        q_collapsed = torch.zeros(8, 5)
        q_collapsed[:, 0] = 0.96
        q_collapsed[:, 1:] = 0.01

        result_uniform = loss_fn(outputs, x_num, x_cat, q=q_uniform, p=p)
        result_collapsed = loss_fn(outputs, x_num, x_cat, q=q_collapsed, p=p)

        # Uniform has higher entropy, so balance term subtracts more -> lower total
        assert result_uniform["balance_entropy"] > result_collapsed["balance_entropy"]

    def test_balance_entropy_is_differentiable(self) -> None:
        """Balance entropy term should be differentiable through q."""
        from scry.model.losses import XDECLoss

        loss_fn = XDECLoss(lambda_balance=0.5, lambda_cluster=0.1)

        outputs = self._make_outputs()
        x_num = torch.randn(8, 30, 9)
        x_cat = torch.rand(8, 30, 8)
        # Use a leaf tensor for q so grad is populated
        q_logits = torch.randn(8, 5, requires_grad=True)
        q = torch.softmax(q_logits, dim=1)
        p = torch.softmax(torch.randn(8, 5), dim=1)

        result = loss_fn(outputs, x_num, x_cat, q=q, p=p)
        result["loss"].backward()

        # Gradient flows back to the leaf logits
        assert q_logits.grad is not None
