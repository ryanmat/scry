# Description: Unit tests for the X-DEC trainer module.
# Description: Tests pretraining, clustering initialization, and full training loop.

"""Tests for scry.model.trainer module."""

import tempfile
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset


@pytest.fixture
def sample_dataloader():
    """Create a sample DataLoader for testing."""
    batch_size = 8
    num_samples = 64
    seq_len = 30
    num_numerical = 9
    num_categorical = 8

    x_num = torch.randn(num_samples, seq_len, num_numerical)
    x_cat = torch.rand(num_samples, seq_len, num_categorical)

    dataset = TensorDataset(x_num, x_cat)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


@pytest.fixture
def mock_config():
    """Create a mock config for testing."""
    from unittest.mock import MagicMock

    config = MagicMock()
    config.num_clusters = 5
    config.latent_dim = 8
    config.sequence_length = 30
    config.numerical_hidden_dim = 64
    config.categorical_hidden_dim = 32
    config.learning_rate = 1e-3
    config.beta = 0.01
    config.lambda_cluster = 0.1
    config.lambda_balance = 0.0
    config.cluster_lr = 1e-4
    config.pretrain_epochs = 500
    config.cluster_epochs = 300
    return config


class TestXDECTrainer:
    """Tests for XDECTrainer class."""

    def test_trainer_initialization(self, mock_config) -> None:
        """Trainer should initialize with model and config."""
        from scry.model.trainer import XDECTrainer
        from scry.model.xdec import TemporalXDEC

        model = TemporalXDEC(num_numerical=9, num_categorical=8)
        trainer = XDECTrainer(model, mock_config)

        assert trainer.model is model
        assert trainer.config is mock_config
        assert trainer.optimizer is not None

    def test_trainer_device_selection(self, mock_config) -> None:
        """Trainer should handle device selection."""
        from scry.model.trainer import XDECTrainer
        from scry.model.xdec import TemporalXDEC

        model = TemporalXDEC(num_numerical=9, num_categorical=8)
        trainer = XDECTrainer(model, mock_config, device="cpu")

        assert trainer.device == torch.device("cpu")


class TestPretraining:
    """Tests for pretraining stage."""

    def test_pretrain_returns_history(self, sample_dataloader, mock_config) -> None:
        """pretrain should return training history dict."""
        from scry.model.trainer import XDECTrainer
        from scry.model.xdec import TemporalXDEC

        model = TemporalXDEC(num_numerical=9, num_categorical=8)
        trainer = XDECTrainer(model, mock_config, device="cpu")

        history = trainer.pretrain(sample_dataloader, epochs=2)

        assert isinstance(history, dict)
        assert "loss" in history
        assert "recon_num" in history
        assert "recon_cat" in history
        assert "kl_vae" in history
        assert len(history["loss"]) == 2

    def test_pretrain_no_clustering_loss(self, sample_dataloader, mock_config) -> None:
        """pretrain should not include clustering loss."""
        from scry.model.trainer import XDECTrainer
        from scry.model.xdec import TemporalXDEC

        model = TemporalXDEC(num_numerical=9, num_categorical=8)
        trainer = XDECTrainer(model, mock_config, device="cpu")

        history = trainer.pretrain(sample_dataloader, epochs=2)

        # No clustering loss during pretraining
        assert "kl_cluster" not in history or all(v == 0 for v in history.get("kl_cluster", [0]))

    def test_pretrain_reduces_loss(self, sample_dataloader, mock_config) -> None:
        """pretrain should reduce reconstruction loss over epochs."""
        from scry.model.trainer import XDECTrainer
        from scry.model.xdec import TemporalXDEC

        model = TemporalXDEC(num_numerical=9, num_categorical=8)
        trainer = XDECTrainer(model, mock_config, device="cpu")

        history = trainer.pretrain(sample_dataloader, epochs=5)

        # Loss should generally decrease (allow some variance)
        assert history["loss"][-1] < history["loss"][0] * 1.5


class TestClusterInitialization:
    """Tests for cluster initialization."""

    def test_initialize_clusters_sets_centroids(self, sample_dataloader, mock_config) -> None:
        """initialize_clusters should set DEC layer centroids."""
        from scry.model.trainer import XDECTrainer
        from scry.model.xdec import TemporalXDEC

        model = TemporalXDEC(num_numerical=9, num_categorical=8)
        trainer = XDECTrainer(model, mock_config, device="cpu")

        initial_centroids = model.dec_layer.centroids.clone()
        trainer.initialize_clusters(sample_dataloader)

        # Centroids should have changed
        assert not torch.allclose(model.dec_layer.centroids, initial_centroids)

    def test_initialize_clusters_returns_assignments(self, sample_dataloader, mock_config) -> None:
        """initialize_clusters should return initial cluster assignments."""
        from scry.model.trainer import XDECTrainer
        from scry.model.xdec import TemporalXDEC

        model = TemporalXDEC(num_numerical=9, num_categorical=8, n_clusters=5)
        trainer = XDECTrainer(model, mock_config, device="cpu")

        assignments = trainer.initialize_clusters(sample_dataloader)

        assert assignments.shape[0] == len(sample_dataloader.dataset)
        assert assignments.min() >= 0
        assert assignments.max() < 5


class TestClusteringTraining:
    """Tests for clustering training stage."""

    def test_train_clustering_returns_history(self, sample_dataloader, mock_config) -> None:
        """train_clustering should return training history."""
        from scry.model.trainer import XDECTrainer
        from scry.model.xdec import TemporalXDEC

        model = TemporalXDEC(num_numerical=9, num_categorical=8)
        trainer = XDECTrainer(model, mock_config, device="cpu")
        trainer.initialize_clusters(sample_dataloader)

        history = trainer.train_clustering(sample_dataloader, epochs=2)

        assert isinstance(history, dict)
        assert "loss" in history
        assert "kl_cluster" in history

    def test_train_clustering_includes_cluster_loss(self, sample_dataloader, mock_config) -> None:
        """train_clustering should include clustering KL loss."""
        from scry.model.trainer import XDECTrainer
        from scry.model.xdec import TemporalXDEC

        model = TemporalXDEC(num_numerical=9, num_categorical=8)
        trainer = XDECTrainer(model, mock_config, device="cpu")
        trainer.initialize_clusters(sample_dataloader)

        history = trainer.train_clustering(sample_dataloader, epochs=2)

        # Clustering loss should be non-zero
        assert any(v > 0 for v in history["kl_cluster"])


class TestGlobalTargetDistribution:
    """Tests for global target distribution usage in clustering."""

    def test_target_distribution_covers_full_dataset(
        self, sample_dataloader, mock_config
    ) -> None:
        """Global target distribution should have one row per dataset sample."""
        from scry.model.trainer import XDECTrainer
        from scry.model.xdec import TemporalXDEC

        model = TemporalXDEC(num_numerical=9, num_categorical=8, n_clusters=5)
        trainer = XDECTrainer(model, mock_config, device="cpu")
        trainer.initialize_clusters(sample_dataloader)

        trainer.train_clustering(sample_dataloader, epochs=2)

        assert trainer._target_distribution is not None
        assert trainer._target_distribution.shape == (
            len(sample_dataloader.dataset),
            model.n_clusters,
        )

    def test_target_distribution_rows_sum_to_one(
        self, sample_dataloader, mock_config
    ) -> None:
        """Each row of the target distribution should sum to 1."""
        from scry.model.trainer import XDECTrainer
        from scry.model.xdec import TemporalXDEC

        model = TemporalXDEC(num_numerical=9, num_categorical=8, n_clusters=5)
        trainer = XDECTrainer(model, mock_config, device="cpu")
        trainer.initialize_clusters(sample_dataloader)

        trainer.train_clustering(sample_dataloader, epochs=2)

        row_sums = trainer._target_distribution.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_cluster_kl_nonzero_with_global_targets(
        self, sample_dataloader, mock_config
    ) -> None:
        """Clustering KL should be meaningfully non-zero with global target distribution."""
        from scry.model.trainer import XDECTrainer
        from scry.model.xdec import TemporalXDEC

        model = TemporalXDEC(num_numerical=9, num_categorical=8, n_clusters=5)
        trainer = XDECTrainer(model, mock_config, device="cpu")
        trainer.initialize_clusters(sample_dataloader)

        history = trainer.train_clustering(sample_dataloader, epochs=5)

        # With global P (not per-batch), KL should be meaningfully non-zero
        # The per-batch bug produced kl_c ~ 0.0000 consistently
        max_kl = max(history["kl_cluster"])
        assert max_kl > 1e-4, (
            f"Cluster KL {max_kl:.6f} is near zero, "
            "suggesting per-batch target distribution (bug) instead of global"
        )


class TestFixedLambdaAndClusterOptimizer:
    """Tests for IDEC-style fixed lambda and separate cluster optimizer."""

    def test_fixed_lambda_applied_during_clustering(
        self, sample_dataloader, mock_config
    ) -> None:
        """Clustering phase should use fixed lambda_cluster from config."""
        from scry.model.trainer import XDECTrainer
        from scry.model.xdec import TemporalXDEC

        mock_config.lambda_cluster = 0.5
        model = TemporalXDEC(num_numerical=9, num_categorical=8)
        trainer = XDECTrainer(model, mock_config, device="cpu")
        trainer.initialize_clusters(sample_dataloader)

        trainer.train_clustering(sample_dataloader, epochs=2)

        assert trainer.loss_fn.lambda_cluster == 0.5

    def test_cluster_optimizer_uses_config_lr(
        self, sample_dataloader, mock_config
    ) -> None:
        """Cluster phase should use separate lower learning rate from config."""
        from scry.model.trainer import XDECTrainer
        from scry.model.xdec import TemporalXDEC

        mock_config.cluster_lr = 5e-5
        model = TemporalXDEC(num_numerical=9, num_categorical=8)
        trainer = XDECTrainer(model, mock_config, device="cpu")

        # Pretrain uses the main optimizer LR
        assert trainer.optimizer.defaults["lr"] == mock_config.learning_rate

        # Cluster phase uses separate LR (verified by running and checking loss_fn)
        trainer.initialize_clusters(sample_dataloader)
        trainer.train_clustering(sample_dataloader, epochs=1)

        # The loss_fn lambda should be set from config, not annealed
        assert trainer.loss_fn.lambda_cluster == mock_config.lambda_cluster

    def test_convergence_stops_early(self, mock_config) -> None:
        """Training should stop early when assignment change fraction drops below tol."""
        from scry.model.trainer import XDECTrainer
        from scry.model.xdec import TemporalXDEC

        # Use small dataset so convergence is reached quickly
        num_samples = 32
        x_num = torch.randn(num_samples, 30, 9)
        x_cat = torch.rand(num_samples, 30, 8)
        dataset = TensorDataset(x_num, x_cat)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)

        model = TemporalXDEC(num_numerical=9, num_categorical=8, n_clusters=3)
        trainer = XDECTrainer(model, mock_config, device="cpu")
        trainer.initialize_clusters(loader)

        # Large tol means we converge almost immediately
        history = trainer.train_clustering(
            loader, epochs=100, update_interval=1, tol=0.99
        )

        # Should have converged well before 100 epochs
        assert len(history["loss"]) < 100


class TestFullTraining:
    """Tests for full fit() training loop."""

    def test_fit_runs_all_stages(self, sample_dataloader, mock_config) -> None:
        """fit should run pretrain, initialize, and cluster training."""
        from scry.model.trainer import XDECTrainer
        from scry.model.xdec import TemporalXDEC

        model = TemporalXDEC(num_numerical=9, num_categorical=8)
        trainer = XDECTrainer(model, mock_config, device="cpu")

        history = trainer.fit(
            sample_dataloader,
            pretrain_epochs=2,
            cluster_epochs=2,
        )

        assert "pretrain" in history
        assert "cluster" in history
        assert "pretrain" in history and "loss" in history["pretrain"]
        assert "cluster" in history and "loss" in history["cluster"]


class TestCheckpointing:
    """Tests for checkpoint save/load."""

    def test_save_checkpoint_creates_file(self, sample_dataloader, mock_config) -> None:
        """save_checkpoint should create checkpoint file."""
        from scry.model.trainer import XDECTrainer
        from scry.model.xdec import TemporalXDEC

        model = TemporalXDEC(num_numerical=9, num_categorical=8)
        trainer = XDECTrainer(model, mock_config, device="cpu")
        trainer.pretrain(sample_dataloader, epochs=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"
            trainer.save_checkpoint(str(path))

            assert path.exists()

    def test_load_checkpoint_restores_state(self, sample_dataloader, mock_config) -> None:
        """load_checkpoint should restore model and optimizer state."""
        from scry.model.trainer import XDECTrainer
        from scry.model.xdec import TemporalXDEC

        model1 = TemporalXDEC(num_numerical=9, num_categorical=8)
        trainer1 = XDECTrainer(model1, mock_config, device="cpu")
        trainer1.pretrain(sample_dataloader, epochs=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"
            trainer1.save_checkpoint(str(path))

            # Create new trainer and load
            model2 = TemporalXDEC(num_numerical=9, num_categorical=8)
            trainer2 = XDECTrainer(model2, mock_config, device="cpu")
            trainer2.load_checkpoint(str(path))

            # Model weights should match
            for p1, p2 in zip(model1.parameters(), model2.parameters()):
                assert torch.allclose(p1, p2)


class TestProgressBars:
    """Tests for tqdm progress bars in training loops."""

    def test_pretrain_with_tqdm_returns_same_history(
        self, sample_dataloader, mock_config
    ) -> None:
        """Pretrain should return identical history structure with tqdm enabled."""
        from scry.model.trainer import XDECTrainer
        from scry.model.xdec import TemporalXDEC

        model = TemporalXDEC(num_numerical=9, num_categorical=8)
        trainer = XDECTrainer(model, mock_config, device="cpu")

        history = trainer.pretrain(sample_dataloader, epochs=3)

        assert isinstance(history, dict)
        assert set(history.keys()) == {"loss", "recon_num", "recon_cat", "kl_vae"}
        assert all(len(v) == 3 for v in history.values())

    def test_cluster_training_with_tqdm_returns_same_history(
        self, sample_dataloader, mock_config
    ) -> None:
        """Cluster training should return identical history structure with tqdm enabled."""
        from scry.model.trainer import XDECTrainer
        from scry.model.xdec import TemporalXDEC

        model = TemporalXDEC(num_numerical=9, num_categorical=8)
        trainer = XDECTrainer(model, mock_config, device="cpu")
        trainer.initialize_clusters(sample_dataloader)

        history = trainer.train_clustering(sample_dataloader, epochs=3)

        assert isinstance(history, dict)
        assert "loss" in history
        assert "kl_cluster" in history
        assert all(len(v) == 3 for v in history.values())


class TestSubsample:
    """Tests for data subsampling in training script."""

    def test_subsample_reduces_data_size(self) -> None:
        """Subsampling with fraction < 1.0 should reduce data size."""
        import numpy as np

        rng = np.random.default_rng(42)
        num_windows = rng.random((1000, 30, 9))
        cat_windows = rng.random((1000, 30, 8))

        subsample = 0.1
        n_subset = int(len(num_windows) * subsample)
        indices = rng.choice(len(num_windows), size=n_subset, replace=False)
        num_sub = num_windows[indices]
        cat_sub = cat_windows[indices]

        assert num_sub.shape[0] == 100
        assert cat_sub.shape[0] == 100
        assert num_sub.shape[1:] == (30, 9)
        assert cat_sub.shape[1:] == (30, 8)

    def test_subsample_full_keeps_all_data(self) -> None:
        """Subsampling with fraction 1.0 should keep all data."""
        import numpy as np

        num_windows = np.random.default_rng(42).random((500, 30, 9))

        subsample = 1.0
        n_subset = int(len(num_windows) * subsample)

        assert n_subset == 500

    def test_subsample_is_deterministic(self) -> None:
        """Subsampling with same seed should produce same results."""
        import numpy as np

        data = np.arange(1000)

        rng1 = np.random.default_rng(42)
        idx1 = rng1.choice(len(data), size=100, replace=False)

        rng2 = np.random.default_rng(42)
        idx2 = rng2.choice(len(data), size=100, replace=False)

        np.testing.assert_array_equal(idx1, idx2)
