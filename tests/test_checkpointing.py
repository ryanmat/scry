# Description: Tests for periodic checkpoint saving and resume from checkpoint.
# Description: Verifies preemption-safe training with stage tracking and epoch resume.

"""Tests for checkpoint save/resume during pretrain and cluster stages."""

import tempfile
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset


@pytest.fixture
def small_dataloader():
    """Create a small DataLoader for fast checkpoint tests."""
    x_num = torch.randn(32, 30, 9)
    x_cat = torch.rand(32, 30, 8)
    dataset = TensorDataset(x_num, x_cat)
    return DataLoader(dataset, batch_size=16, shuffle=True)


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


class TestPeriodicCheckpointing:
    """Tests for periodic checkpoint saves during training."""

    def test_pretrain_saves_checkpoints_at_interval(
        self, small_dataloader, mock_config
    ) -> None:
        """pretrain should save checkpoints every checkpoint_interval epochs."""
        from scry.model.trainer import XDECTrainer
        from scry.model.xdec import TemporalXDEC

        model = TemporalXDEC(num_numerical=9, num_categorical=8)
        trainer = XDECTrainer(model, mock_config, device="cpu")

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.pretrain(
                small_dataloader,
                epochs=6,
                checkpoint_dir=tmpdir,
                checkpoint_interval=3,
            )

            # Should have checkpoints at epoch 3 and 6
            files = sorted(Path(tmpdir).glob("checkpoint_*.pt"))
            assert len(files) >= 2
            assert any("pretrain" in f.name for f in files)

    def test_cluster_saves_checkpoints_at_interval(
        self, small_dataloader, mock_config
    ) -> None:
        """train_clustering should save checkpoints every checkpoint_interval epochs."""
        from scry.model.trainer import XDECTrainer
        from scry.model.xdec import TemporalXDEC

        model = TemporalXDEC(num_numerical=9, num_categorical=8)
        trainer = XDECTrainer(model, mock_config, device="cpu")
        trainer.initialize_clusters(small_dataloader)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.train_clustering(
                small_dataloader,
                epochs=6,
                checkpoint_dir=tmpdir,
                checkpoint_interval=3,
            )

            files = sorted(Path(tmpdir).glob("checkpoint_*.pt"))
            assert len(files) >= 2
            assert any("cluster" in f.name for f in files)

    def test_no_checkpoints_without_dir(
        self, small_dataloader, mock_config
    ) -> None:
        """No checkpoints should be saved when checkpoint_dir is None."""
        from scry.model.trainer import XDECTrainer
        from scry.model.xdec import TemporalXDEC

        model = TemporalXDEC(num_numerical=9, num_categorical=8)
        trainer = XDECTrainer(model, mock_config, device="cpu")

        # Should not raise even without checkpoint_dir
        history = trainer.pretrain(small_dataloader, epochs=2)
        assert len(history["loss"]) == 2


class TestCheckpointStageTracking:
    """Tests for stage and epoch tracking in checkpoints."""

    def test_pretrain_checkpoint_has_stage_and_epoch(
        self, small_dataloader, mock_config
    ) -> None:
        """Pretrain checkpoint should store stage='pretrain' and epoch number."""
        from scry.model.trainer import XDECTrainer
        from scry.model.xdec import TemporalXDEC

        model = TemporalXDEC(num_numerical=9, num_categorical=8)
        trainer = XDECTrainer(model, mock_config, device="cpu")

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.pretrain(
                small_dataloader,
                epochs=4,
                checkpoint_dir=tmpdir,
                checkpoint_interval=4,
            )

            files = list(Path(tmpdir).glob("checkpoint_*.pt"))
            assert len(files) >= 1

            ckpt = torch.load(files[0], map_location="cpu", weights_only=False)
            assert ckpt["stage"] == "pretrain"
            assert ckpt["epoch"] == 4

    def test_cluster_checkpoint_has_stage_and_epoch(
        self, small_dataloader, mock_config
    ) -> None:
        """Cluster checkpoint should store stage='cluster' and epoch number."""
        from scry.model.trainer import XDECTrainer
        from scry.model.xdec import TemporalXDEC

        model = TemporalXDEC(num_numerical=9, num_categorical=8)
        trainer = XDECTrainer(model, mock_config, device="cpu")
        trainer.initialize_clusters(small_dataloader)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.train_clustering(
                small_dataloader,
                epochs=4,
                checkpoint_dir=tmpdir,
                checkpoint_interval=4,
            )

            files = list(Path(tmpdir).glob("checkpoint_*.pt"))
            assert len(files) >= 1

            ckpt = torch.load(files[0], map_location="cpu", weights_only=False)
            assert ckpt["stage"] == "cluster"
            assert ckpt["epoch"] == 4

    def test_cluster_checkpoint_includes_target_distribution(
        self, small_dataloader, mock_config
    ) -> None:
        """Cluster checkpoint should include the target distribution tensor."""
        from scry.model.trainer import XDECTrainer
        from scry.model.xdec import TemporalXDEC

        model = TemporalXDEC(num_numerical=9, num_categorical=8, n_clusters=5)
        trainer = XDECTrainer(model, mock_config, device="cpu")
        trainer.initialize_clusters(small_dataloader)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.train_clustering(
                small_dataloader,
                epochs=2,
                checkpoint_dir=tmpdir,
                checkpoint_interval=2,
            )

            files = list(Path(tmpdir).glob("checkpoint_*.pt"))
            ckpt = torch.load(files[0], map_location="cpu", weights_only=False)

            assert "target_distribution" in ckpt
            assert ckpt["target_distribution"].shape == (32, 5)


class TestResumeFromCheckpoint:
    """Tests for resuming training from a checkpoint."""

    def test_resume_pretrain_from_checkpoint(
        self, small_dataloader, mock_config
    ) -> None:
        """Resuming pretrain should continue from saved epoch, not epoch 0."""
        from scry.model.trainer import XDECTrainer
        from scry.model.xdec import TemporalXDEC

        model = TemporalXDEC(num_numerical=9, num_categorical=8)
        trainer = XDECTrainer(model, mock_config, device="cpu")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Run 4 epochs, saving at epoch 4
            trainer.pretrain(
                small_dataloader,
                epochs=4,
                checkpoint_dir=tmpdir,
                checkpoint_interval=4,
            )

            ckpt_path = list(Path(tmpdir).glob("checkpoint_*.pt"))[0]

            # Resume: new trainer loads checkpoint and continues
            model2 = TemporalXDEC(num_numerical=9, num_categorical=8)
            trainer2 = XDECTrainer(model2, mock_config, device="cpu")
            resume_info = trainer2.load_checkpoint(str(ckpt_path))

            assert resume_info["stage"] == "pretrain"
            assert resume_info["epoch"] == 4

            # Continue pretrain from epoch 4 to 8
            history = trainer2.pretrain(
                small_dataloader,
                epochs=8,
                start_epoch=resume_info["epoch"],
            )

            # Should have 4 more epochs of history (epochs 4-7)
            assert len(history["loss"]) == 4

    def test_resume_cluster_from_checkpoint(
        self, small_dataloader, mock_config
    ) -> None:
        """Resuming cluster should continue from saved epoch with target distribution."""
        from scry.model.trainer import XDECTrainer
        from scry.model.xdec import TemporalXDEC

        model = TemporalXDEC(num_numerical=9, num_categorical=8)
        trainer = XDECTrainer(model, mock_config, device="cpu")
        trainer.initialize_clusters(small_dataloader)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Run 4 cluster epochs
            trainer.train_clustering(
                small_dataloader,
                epochs=4,
                checkpoint_dir=tmpdir,
                checkpoint_interval=4,
            )

            ckpt_path = list(Path(tmpdir).glob("checkpoint_*.pt"))[0]

            # Resume
            model2 = TemporalXDEC(num_numerical=9, num_categorical=8)
            trainer2 = XDECTrainer(model2, mock_config, device="cpu")
            resume_info = trainer2.load_checkpoint(str(ckpt_path))

            assert resume_info["stage"] == "cluster"
            assert resume_info["epoch"] == 4
            assert trainer2._target_distribution is not None

            # Continue from epoch 4 to 8
            history = trainer2.train_clustering(
                small_dataloader,
                epochs=8,
                start_epoch=resume_info["epoch"],
            )

            assert len(history["loss"]) == 4

    def test_fit_resumes_from_pretrain_checkpoint(
        self, small_dataloader, mock_config
    ) -> None:
        """fit() with resume_checkpoint in pretrain stage should skip completed epochs."""
        from scry.model.trainer import XDECTrainer
        from scry.model.xdec import TemporalXDEC

        model = TemporalXDEC(num_numerical=9, num_categorical=8)
        trainer = XDECTrainer(model, mock_config, device="cpu")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Run partial pretrain
            trainer.pretrain(
                small_dataloader,
                epochs=3,
                checkpoint_dir=tmpdir,
                checkpoint_interval=3,
            )

            ckpt_path = str(list(Path(tmpdir).glob("checkpoint_*.pt"))[0])

            # Resume via fit() - should continue pretrain then do clustering
            model2 = TemporalXDEC(num_numerical=9, num_categorical=8)
            trainer2 = XDECTrainer(model2, mock_config, device="cpu")

            history = trainer2.fit(
                small_dataloader,
                pretrain_epochs=6,
                cluster_epochs=2,
                checkpoint_dir=tmpdir,
                resume_checkpoint=ckpt_path,
            )

            assert "pretrain" in history
            assert "cluster" in history
            # Pretrain should have only 3 remaining epochs (3 to 6)
            assert len(history["pretrain"]["loss"]) == 3
            assert len(history["cluster"]["loss"]) == 2

    def test_fit_resumes_from_cluster_checkpoint(
        self, small_dataloader, mock_config
    ) -> None:
        """fit() with resume_checkpoint in cluster stage should skip pretrain entirely."""
        from scry.model.trainer import XDECTrainer
        from scry.model.xdec import TemporalXDEC

        model = TemporalXDEC(num_numerical=9, num_categorical=8)
        trainer = XDECTrainer(model, mock_config, device="cpu")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Do full pretrain + partial cluster
            trainer.pretrain(small_dataloader, epochs=3)
            trainer.initialize_clusters(small_dataloader)
            trainer.train_clustering(
                small_dataloader,
                epochs=3,
                checkpoint_dir=tmpdir,
                checkpoint_interval=3,
            )

            ckpt_path = str(list(Path(tmpdir).glob("checkpoint_*.pt"))[0])

            # Resume via fit() - should skip pretrain, skip cluster init
            model2 = TemporalXDEC(num_numerical=9, num_categorical=8)
            trainer2 = XDECTrainer(model2, mock_config, device="cpu")

            history = trainer2.fit(
                small_dataloader,
                pretrain_epochs=3,
                cluster_epochs=6,
                resume_checkpoint=ckpt_path,
            )

            # Pretrain should be empty (skipped)
            assert history["pretrain"] is None
            # Cluster should have 3 remaining epochs (3 to 6)
            assert len(history["cluster"]["loss"]) == 3
