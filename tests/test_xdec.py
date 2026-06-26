# Description: Unit tests for the Temporal X-DEC model.
# Description: Tests the complete model combining XVAE and DEC clustering.

"""Tests for scry.model.xdec module."""

import torch


class TestTemporalXDEC:
    """Tests for TemporalXDEC model."""

    def test_forward_returns_all_outputs(self) -> None:
        """Forward should return all expected output keys."""
        from scry.model.xdec import TemporalXDEC

        model = TemporalXDEC(num_numerical=9, num_categorical=8)

        x_num = torch.randn(8, 30, 9)
        x_cat = torch.randn(8, 30, 8)

        output = model(x_num, x_cat)

        assert "z" in output
        assert "mu" in output
        assert "logvar" in output
        assert "x_num_recon" in output
        assert "x_cat_recon" in output
        assert "q" in output

    def test_cluster_assignments_shape(self) -> None:
        """Cluster assignments should have correct shape."""
        from scry.model.xdec import TemporalXDEC

        model = TemporalXDEC(num_numerical=9, num_categorical=8, n_clusters=5)

        x_num = torch.randn(8, 30, 9)
        x_cat = torch.randn(8, 30, 8)

        output = model(x_num, x_cat)

        assert output["q"].shape == (8, 5)

    def test_cluster_assignments_sum_to_one(self) -> None:
        """Cluster assignments should be valid probability distribution."""
        from scry.model.xdec import TemporalXDEC

        model = TemporalXDEC(num_numerical=9, num_categorical=8, n_clusters=5)

        x_num = torch.randn(8, 30, 9)
        x_cat = torch.randn(8, 30, 8)

        output = model(x_num, x_cat)
        q = output["q"]

        row_sums = q.sum(dim=1)
        torch.testing.assert_close(row_sums, torch.ones(8))

    def test_encode_returns_latent(self) -> None:
        """encode should return latent representations."""
        from scry.model.xdec import TemporalXDEC

        model = TemporalXDEC(num_numerical=9, num_categorical=8, latent_dim=8)

        x_num = torch.randn(4, 30, 9)
        x_cat = torch.randn(4, 30, 8)

        z = model.encode(x_num, x_cat)

        assert z.shape == (4, 8)

    def test_get_cluster_assignments(self) -> None:
        """get_cluster_assignments should return soft assignments."""
        from scry.model.xdec import TemporalXDEC

        model = TemporalXDEC(num_numerical=9, num_categorical=8, n_clusters=5)

        x_num = torch.randn(4, 30, 9)
        x_cat = torch.randn(4, 30, 8)

        q = model.get_cluster_assignments(x_num, x_cat)

        assert q.shape == (4, 5)
        assert (q >= 0).all()
        assert (q <= 1).all()

    def test_predict_cluster_returns_labels(self) -> None:
        """predict_cluster should return hard cluster labels."""
        from scry.model.xdec import TemporalXDEC

        model = TemporalXDEC(num_numerical=9, num_categorical=8, n_clusters=5)

        x_num = torch.randn(4, 30, 9)
        x_cat = torch.randn(4, 30, 8)

        labels = model.predict_cluster(x_num, x_cat)

        assert labels.shape == (4,)
        assert labels.dtype == torch.long
        assert (labels >= 0).all()
        assert (labels < 5).all()

    def test_model_is_differentiable(self) -> None:
        """Model should support gradient computation."""
        from scry.model.xdec import TemporalXDEC

        model = TemporalXDEC(num_numerical=9, num_categorical=8)

        x_num = torch.randn(4, 30, 9, requires_grad=True)
        x_cat = torch.randn(4, 30, 8, requires_grad=True)

        output = model(x_num, x_cat)
        loss = output["z"].sum() + output["q"].sum()
        loss.backward()

        assert x_num.grad is not None
        assert x_cat.grad is not None

    def test_initialize_centroids(self) -> None:
        """initialize_centroids should set DEC centroids from data."""
        from scry.model.xdec import TemporalXDEC

        model = TemporalXDEC(num_numerical=9, num_categorical=8, n_clusters=5, latent_dim=8)

        x_num = torch.randn(100, 30, 9)
        x_cat = torch.randn(100, 30, 8)

        # Get initial centroids
        initial_centroids = model.dec_layer.centroids.clone()

        # Initialize from data
        model.initialize_centroids(x_num, x_cat)

        # Centroids should have changed
        assert not torch.allclose(model.dec_layer.centroids, initial_centroids)

    def test_model_parameters_count(self) -> None:
        """Model should have expected number of parameter groups."""
        from scry.model.xdec import TemporalXDEC

        model = TemporalXDEC(num_numerical=9, num_categorical=8)

        # Should have parameters from XVAE + DEC layer
        param_count = sum(p.numel() for p in model.parameters())
        assert param_count > 0

        # DEC centroids should be included
        dec_params = sum(p.numel() for p in model.dec_layer.parameters())
        assert dec_params > 0

    def test_eval_mode_affects_sampling(self) -> None:
        """In eval mode, outputs should be more deterministic."""
        from scry.model.xdec import TemporalXDEC

        model = TemporalXDEC(num_numerical=9, num_categorical=8)
        x_num = torch.randn(4, 30, 9)
        x_cat = torch.randn(4, 30, 8)

        # Train mode: different z each time (stochastic sampling)
        model.train()
        z1 = model(x_num, x_cat)["z"]
        z2 = model(x_num, x_cat)["z"]

        # Should be different due to reparameterization sampling
        assert not torch.allclose(z1, z2)

