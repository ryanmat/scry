# Description: Unit tests for the DEC clustering layer.
# Description: Tests soft assignments, target distribution, and centroid initialization.

"""Tests for scry.model.clustering module."""

import torch


class TestDECLayer:
    """Tests for DEC clustering layer."""

    def test_forward_returns_soft_assignments(self) -> None:
        """Forward should return soft cluster assignments."""
        from scry.model.clustering import DECLayer

        layer = DECLayer(n_clusters=5, latent_dim=8)

        z = torch.randn(16, 8)
        q = layer(z)

        assert q.shape == (16, 5)

    def test_soft_assignments_sum_to_one(self) -> None:
        """Soft assignments should sum to 1 for each sample."""
        from scry.model.clustering import DECLayer

        layer = DECLayer(n_clusters=5, latent_dim=8)

        z = torch.randn(16, 8)
        q = layer(z)

        row_sums = q.sum(dim=1)
        torch.testing.assert_close(row_sums, torch.ones(16))

    def test_soft_assignments_are_positive(self) -> None:
        """Soft assignments should be in [0, 1]."""
        from scry.model.clustering import DECLayer

        layer = DECLayer(n_clusters=5, latent_dim=8)

        z = torch.randn(16, 8)
        q = layer(z)

        assert (q >= 0).all()
        assert (q <= 1).all()

    def test_centroids_are_learnable(self) -> None:
        """Cluster centroids should be learnable parameters."""
        from scry.model.clustering import DECLayer

        layer = DECLayer(n_clusters=5, latent_dim=8)

        assert hasattr(layer, "centroids")
        assert layer.centroids.requires_grad
        assert layer.centroids.shape == (5, 8)

    def test_forward_is_differentiable(self) -> None:
        """Forward pass should support gradient computation."""
        from scry.model.clustering import DECLayer

        layer = DECLayer(n_clusters=5, latent_dim=8)

        z = torch.randn(16, 8, requires_grad=True)
        q = layer(z)
        q.sum().backward()

        assert z.grad is not None


class TestTargetDistribution:
    """Tests for target distribution computation."""

    def test_target_distribution_shape(self) -> None:
        """Target distribution should have same shape as input."""
        from scry.model.clustering import compute_target_distribution

        q = torch.softmax(torch.randn(16, 5), dim=1)
        p = compute_target_distribution(q)

        assert p.shape == q.shape

    def test_target_distribution_sums_to_one(self) -> None:
        """Target distribution should sum to 1 for each sample."""
        from scry.model.clustering import compute_target_distribution

        q = torch.softmax(torch.randn(16, 5), dim=1)
        p = compute_target_distribution(q)

        row_sums = p.sum(dim=1)
        torch.testing.assert_close(row_sums, torch.ones(16))

    def test_target_is_sharper_than_input(self) -> None:
        """Target distribution should have higher confidence (sharper)."""
        from scry.model.clustering import compute_target_distribution

        q = torch.softmax(torch.randn(16, 5), dim=1)
        p = compute_target_distribution(q)

        # Max probability should be higher in p (sharper distribution)
        q_max = q.max(dim=1).values.mean()
        p_max = p.max(dim=1).values.mean()

        assert p_max >= q_max


class TestCentroidInitialization:
    """Tests for centroid initialization."""

    def test_initialize_centroids_from_data(self) -> None:
        """Centroids should be initialized from data samples."""
        from scry.model.clustering import DECLayer

        layer = DECLayer(n_clusters=5, latent_dim=8)
        z = torch.randn(100, 8)

        layer.initialize_centroids(z)

        assert layer.centroids.shape == (5, 8)
        # Centroids should be within the data range
        z_min = z.min(dim=0).values
        z_max = z.max(dim=0).values
        assert (layer.centroids >= z_min - 1).all()
        assert (layer.centroids <= z_max + 1).all()

    def test_alpha_parameter_affects_sharpness(self) -> None:
        """Alpha parameter should affect distribution sharpness."""
        from scry.model.clustering import DECLayer

        torch.manual_seed(42)
        layer_soft = DECLayer(n_clusters=5, latent_dim=8, alpha=0.5)
        layer_sharp = DECLayer(n_clusters=5, latent_dim=8, alpha=2.0)
        # Share centroids so alpha is the only difference between the two layers;
        # otherwise their independent random centroids confound the comparison and
        # can flip it by a hair on an unlucky draw (it did, on CI for 3.10).
        layer_sharp.centroids.data.copy_(layer_soft.centroids.data)
        z = torch.randn(16, 8)
        q_soft = layer_soft(z)
        q_sharp = layer_sharp(z)

        # Higher alpha = sharper distribution = higher max probability.
        assert q_sharp.max(dim=1).values.mean() >= q_soft.max(dim=1).values.mean()

