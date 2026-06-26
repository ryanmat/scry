# Description: Training infrastructure for Temporal X-DEC model.
# Description: Handles pretraining, cluster initialization, and joint training.

"""XDECTrainer for training the Temporal X-DEC model."""

import logging
from typing import Any

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from scry.model.clustering import compute_target_distribution
from scry.model.losses import XDECLoss
from scry.model.xdec import TemporalXDEC

logger = logging.getLogger(__name__)


class _IndexedDataset(Dataset):
    """Wraps a dataset to return sample indices alongside data.

    Enables mapping shuffled batch samples back to their position in a
    globally-computed target distribution tensor.

    Args:
        dataset: Underlying dataset to wrap.
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        return (*self.dataset[idx], idx)

    def __len__(self):
        return len(self.dataset)


class XDECTrainer:
    """Trainer for Temporal X-DEC model.

    Implements a two-stage training procedure:
    1. Pretrain: Train XVAE for reconstruction (no clustering)
    2. Cluster: Joint training with clustering loss and lambda annealing

    Args:
        model: TemporalXDEC model instance.
        config: Configuration with training hyperparameters.
        device: Device to train on ("cpu", "cuda", or "auto").
    """

    def __init__(
        self,
        model: TemporalXDEC,
        config: Any,
        device: str = "auto",
    ) -> None:
        self.model = model
        self.config = config

        # Device selection
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = self.model.to(self.device)

        # Optimizer
        self.optimizer = Adam(
            self.model.parameters(),
            lr=getattr(config, "learning_rate", 1e-3),
        )

        # Loss function
        self.loss_fn = XDECLoss(
            beta=getattr(config, "beta", 1.0),
            lambda_cluster=0.0,  # Start with no clustering loss
            lambda_balance=getattr(config, "lambda_balance", 0.0),
        )

        # Training history
        self.history: dict[str, list] = {}

        # Current target distribution for clustering
        self._target_distribution: torch.Tensor | None = None

    def pretrain(
        self,
        dataloader: DataLoader,
        epochs: int = 500,
        start_epoch: int = 0,
        checkpoint_dir: str | None = None,
        checkpoint_interval: int = 50,
    ) -> dict[str, list]:
        """Pretrain XVAE for reconstruction only.

        Args:
            dataloader: Training DataLoader yielding (x_num, x_cat) batches.
            epochs: Number of pretraining epochs.
            start_epoch: Epoch to resume from (0-indexed). Epochs before this
                are skipped, allowing resume after checkpoint load.
            checkpoint_dir: Directory for periodic checkpoint saves. No
                checkpoints are saved when None.
            checkpoint_interval: Save a checkpoint every N epochs.

        Returns:
            Training history with loss components per epoch.
        """
        self.model.train()
        history = {
            "loss": [],
            "recon_num": [],
            "recon_cat": [],
            "kl_vae": [],
        }

        pbar = tqdm(range(start_epoch, epochs), desc="Pretrain", unit="epoch")
        for epoch in pbar:
            epoch_losses = {k: 0.0 for k in history}
            num_batches = 0

            for batch in dataloader:
                x_num, x_cat = batch
                x_num = x_num.to(self.device)
                x_cat = x_cat.to(self.device)

                self.optimizer.zero_grad()

                # Forward pass (XVAE only, no clustering)
                outputs = self.model(x_num, x_cat)

                # Compute loss without clustering
                loss_dict = self.loss_fn(outputs, x_num, x_cat)
                loss = loss_dict["loss"]

                # Backward pass
                loss.backward()
                self.optimizer.step()

                # Accumulate losses
                for key in history:
                    epoch_losses[key] += loss_dict[key].item()
                num_batches += 1

            # Average losses for epoch
            for key in history:
                history[key].append(epoch_losses[key] / max(num_batches, 1))

            pbar.set_postfix(loss=f"{history['loss'][-1]:.4f}")

            # Periodic checkpoint
            if checkpoint_dir and (epoch + 1) % checkpoint_interval == 0:
                self.save_checkpoint(
                    f"{checkpoint_dir}/checkpoint_pretrain_e{epoch + 1}.pt",
                    stage="pretrain",
                    epoch=epoch + 1,
                )

        return history

    def initialize_clusters(self, dataloader: DataLoader) -> np.ndarray:
        """Initialize DEC centroids using k-means++ on embeddings.

        Args:
            dataloader: DataLoader to get embeddings from.

        Returns:
            Initial cluster assignments for all samples.
        """
        self.model.eval()
        embeddings = []

        with torch.no_grad():
            for batch in dataloader:
                x_num, x_cat = batch
                x_num = x_num.to(self.device)
                x_cat = x_cat.to(self.device)

                z = self.model.encode(x_num, x_cat)
                embeddings.append(z.cpu())

        if not embeddings:
            logger.warning("initialize_clusters: empty dataloader, returning empty assignments")
            return np.array([], dtype=np.intp)

        embeddings = torch.cat(embeddings, dim=0)

        # Initialize centroids from CPU embeddings
        self.model.dec_layer.initialize_centroids(embeddings)

        # Get initial assignments (both embeddings and centroids on CPU here)
        q = self.model.dec_layer(embeddings)
        assignments = q.argmax(dim=1).numpy()

        # Move centroids to model device for subsequent GPU training
        self.model.dec_layer.centroids.data = (
            self.model.dec_layer.centroids.data.to(self.device)
        )

        return assignments

    def train_clustering(
        self,
        dataloader: DataLoader,
        epochs: int = 300,
        update_interval: int = 140,
        tol: float = 0.001,
        start_epoch: int = 0,
        checkpoint_dir: str | None = None,
        checkpoint_interval: int = 50,
    ) -> dict[str, list]:
        """Train with full X-DEC loss including clustering.

        Uses a globally-computed target distribution P rather than per-batch P.
        The global P is computed over the entire training set and held fixed for
        update_interval iterations, then recomputed. Batch samples are mapped
        back to their global P rows via dataset indices.

        Follows IDEC approach: joint reconstruction + clustering with a fixed
        lambda weight (no annealing). Uses a separate lower learning rate to
        avoid destroying pretrained representations.

        Training stops early if fewer than tol fraction of cluster assignments
        change between target distribution updates (per DEC paper).

        Args:
            dataloader: Training DataLoader.
            epochs: Maximum number of clustering epochs.
            update_interval: Iterations between target distribution updates.
            tol: Convergence threshold. Stop when fraction of changed
                assignments falls below this value. Set to 0 to disable.
            start_epoch: Epoch to resume from (0-indexed).
            checkpoint_dir: Directory for periodic checkpoint saves.
            checkpoint_interval: Save a checkpoint every N epochs.

        Returns:
            Training history with all loss components.
        """
        self.model.train()
        history = {
            "loss": [],
            "recon_num": [],
            "recon_cat": [],
            "kl_vae": [],
            "kl_cluster": [],
            "balance_entropy": [],
        }

        # Separate optimizer for cluster phase with lower LR
        cluster_lr = getattr(self.config, "cluster_lr", 1e-4)
        cluster_optimizer = Adam(
            self.model.parameters(),
            lr=cluster_lr,
        )

        # Fixed lambda for clustering loss (IDEC-style, no annealing)
        lambda_cluster = getattr(self.config, "lambda_cluster", 0.1)
        self.loss_fn.lambda_cluster = lambda_cluster

        # Indexed DataLoader for mapping batch samples to global P rows
        indexed_dataset = _IndexedDataset(dataloader.dataset)
        indexed_loader = DataLoader(
            indexed_dataset,
            batch_size=dataloader.batch_size,
            shuffle=True,
            drop_last=dataloader.drop_last,
        )

        # Sequential loader for deterministic global target computation
        sequential_loader = DataLoader(
            dataloader.dataset,
            batch_size=dataloader.batch_size,
            shuffle=False,
        )

        # Compute global target distribution (reuse existing if resuming)
        if self._target_distribution is not None and start_epoch > 0:
            prev_assignments = self._target_distribution.argmax(dim=1).cpu().numpy()
        else:
            prev_assignments = self._update_target_distribution(sequential_loader)

        iteration = start_epoch * max(len(indexed_loader), 1)

        pbar = tqdm(range(start_epoch, epochs), desc="Cluster", unit="epoch")
        for epoch in pbar:
            epoch_losses = {k: 0.0 for k in history}
            num_batches = 0

            for batch in indexed_loader:
                x_num, x_cat, indices = batch
                x_num = x_num.to(self.device)
                x_cat = x_cat.to(self.device)

                cluster_optimizer.zero_grad()

                # Forward pass
                outputs = self.model(x_num, x_cat)
                q = outputs["q"]

                # Look up pre-computed global target distribution for this batch
                p = self._target_distribution[indices].to(self.device)

                # Compute full loss with clustering
                loss_dict = self.loss_fn(outputs, x_num, x_cat, q=q, p=p)
                loss = loss_dict["loss"]

                # Backward pass
                loss.backward()
                cluster_optimizer.step()

                # Accumulate losses
                for key in history:
                    if key in loss_dict:
                        epoch_losses[key] += loss_dict[key].item()
                num_batches += 1
                iteration += 1

                # Update target distribution periodically
                if iteration % update_interval == 0:
                    new_assignments = self._update_target_distribution(
                        sequential_loader
                    )
                    # Check convergence: fraction of assignments that changed
                    if tol > 0 and prev_assignments is not None:
                        changed = (new_assignments != prev_assignments).sum()
                        change_frac = changed / max(len(new_assignments), 1)
                        if change_frac < tol:
                            pbar.write(
                                f"Converged at epoch {epoch}: "
                                f"{change_frac:.4%} assignments changed "
                                f"(< {tol:.2%} threshold)"
                            )
                            # Record final epoch losses before breaking
                            for key in history:
                                history[key].append(
                                    epoch_losses[key] / max(num_batches, 1)
                                )
                            return history
                    prev_assignments = new_assignments

            # Average losses for epoch
            for key in history:
                history[key].append(epoch_losses[key] / max(num_batches, 1))

            pbar.set_postfix(
                loss=f"{history['loss'][-1]:.4f}",
                kl_c=f"{history['kl_cluster'][-1]:.4f}",
                lam=f"{lambda_cluster:.3f}",
            )

            # Periodic checkpoint
            if checkpoint_dir and (epoch + 1) % checkpoint_interval == 0:
                self.save_checkpoint(
                    f"{checkpoint_dir}/checkpoint_cluster_e{epoch + 1}.pt",
                    stage="cluster",
                    epoch=epoch + 1,
                    cluster_optimizer_state_dict=cluster_optimizer.state_dict(),
                    target_distribution=self._target_distribution,
                )

        return history

    def fit(
        self,
        dataloader: DataLoader,
        pretrain_epochs: int = 500,
        cluster_epochs: int = 300,
        checkpoint_dir: str | None = None,
        checkpoint_interval: int = 50,
        resume_checkpoint: str | None = None,
    ) -> dict[str, dict[str, list]]:
        """Full training procedure: pretrain, initialize, cluster.

        Supports resuming from a checkpoint saved during a previous run.
        When resume_checkpoint points to a pretrain checkpoint, pretraining
        continues from that epoch then proceeds to clustering. When it points
        to a cluster checkpoint, pretraining and cluster init are skipped.

        Optionally logs metrics to MLflow if available.

        Args:
            dataloader: Training DataLoader.
            pretrain_epochs: Epochs for pretraining stage.
            cluster_epochs: Epochs for clustering stage.
            checkpoint_dir: Directory for periodic checkpoint saves.
            checkpoint_interval: Save a checkpoint every N epochs.
            resume_checkpoint: Path to checkpoint file to resume from.

        Returns:
            Combined training history from all stages.
        """
        mlflow = self._try_import_mlflow()

        if mlflow is not None:
            mlflow.log_params({
                "beta": getattr(self.config, "beta", 1.0),
                "lambda_cluster": getattr(self.config, "lambda_cluster", 0.1),
                "lambda_balance": getattr(self.config, "lambda_balance", 0.0),
                "cluster_lr": getattr(self.config, "cluster_lr", 1e-4),
                "learning_rate": getattr(self.config, "learning_rate", 1e-3),
                "pretrain_epochs": pretrain_epochs,
                "cluster_epochs": cluster_epochs,
                "num_clusters": self.model.n_clusters,
                "latent_dim": self.model.latent_dim,
            })

        # Determine resume state
        resume_stage = None
        resume_epoch = 0
        if resume_checkpoint:
            resume_info = self.load_checkpoint(resume_checkpoint)
            resume_stage = resume_info.get("stage", "pretrain")
            resume_epoch = resume_info.get("epoch", 0)
            logger.info(
                "Resuming from checkpoint: stage=%s, epoch=%d",
                resume_stage, resume_epoch,
            )

        pretrain_history = None
        cluster_history = None

        # Stage 1: Pretrain XVAE (skip if resuming from cluster stage)
        if resume_stage != "cluster":
            pretrain_start = resume_epoch if resume_stage == "pretrain" else 0
            pretrain_history = self.pretrain(
                dataloader,
                epochs=pretrain_epochs,
                start_epoch=pretrain_start,
                checkpoint_dir=checkpoint_dir,
                checkpoint_interval=checkpoint_interval,
            )

            if mlflow is not None:
                for epoch, loss in enumerate(pretrain_history["loss"]):
                    mlflow.log_metric(
                        "pretrain_loss", loss, step=pretrain_start + epoch
                    )

            # Save checkpoint after pretrain completes (before cluster init)
            if checkpoint_dir:
                self.save_checkpoint(
                    f"{checkpoint_dir}/checkpoint_pretrain_done.pt",
                    stage="pretrain_done",
                    epoch=pretrain_epochs,
                )

            # Stage 2: Initialize clusters
            self.initialize_clusters(dataloader)

        # Stage 3: Train with clustering
        cluster_start = resume_epoch if resume_stage == "cluster" else 0
        cluster_history = self.train_clustering(
            dataloader,
            epochs=cluster_epochs,
            start_epoch=cluster_start,
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval=checkpoint_interval,
        )

        if mlflow is not None:
            for epoch, loss in enumerate(cluster_history["loss"]):
                mlflow.log_metric("cluster_loss", loss, step=cluster_start + epoch)
            for epoch, kl in enumerate(cluster_history["kl_cluster"]):
                mlflow.log_metric("kl_cluster", kl, step=cluster_start + epoch)
            for epoch, ent in enumerate(cluster_history.get("balance_entropy", [])):
                mlflow.log_metric("balance_entropy", ent, step=cluster_start + epoch)

        return {
            "pretrain": pretrain_history,
            "cluster": cluster_history,
        }

    @staticmethod
    def _try_import_mlflow():
        """Try to import mlflow. Returns mlflow module or None.

        Only activates if MLFLOW_TRACKING_URI is set or an active run exists.
        This prevents accidental MLflow side effects during testing.
        """
        import os

        if not os.environ.get("MLFLOW_TRACKING_URI"):
            return None

        try:
            import mlflow

            if mlflow.active_run() is None:
                mlflow.start_run()
            return mlflow
        except ImportError:
            logger.debug("MLflow not available, skipping metric logging")
            return None

    def save_checkpoint(self, path: str, stage: str = "unknown",
                        epoch: int = 0, **extra) -> None:
        """Save training checkpoint with stage and epoch tracking.

        Args:
            path: Path to save checkpoint file.
            stage: Current training stage ("pretrain", "cluster", etc.).
            epoch: Current epoch number (1-indexed, i.e. completed epochs).
            **extra: Additional state to save (e.g. cluster_optimizer_state_dict,
                target_distribution).
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
            "stage": stage,
            "epoch": epoch,
            **extra,
        }
        torch.save(checkpoint, path)
        logger.info("Checkpoint saved: %s (stage=%s, epoch=%d)", path, stage, epoch)

    def load_checkpoint(self, path: str) -> dict:
        """Load training checkpoint and restore model state.

        Restores model weights, optimizer state, and (for cluster checkpoints)
        the target distribution tensor. Returns the full checkpoint dict so
        callers can inspect stage and epoch for resume logic.

        Args:
            path: Path to checkpoint file.

        Returns:
            Checkpoint dict with keys: stage, epoch, model_state_dict, etc.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint.get("history", {})

        # Restore target distribution if present (cluster stage resume)
        if "target_distribution" in checkpoint:
            self._target_distribution = checkpoint["target_distribution"]

        logger.info(
            "Checkpoint loaded: %s (stage=%s, epoch=%d)",
            path, checkpoint.get("stage", "unknown"), checkpoint.get("epoch", 0),
        )
        return checkpoint

    def _update_target_distribution(self, dataloader: DataLoader) -> np.ndarray:
        """Update target distribution P from current soft assignments Q.

        Args:
            dataloader: DataLoader to compute assignments over.

        Returns:
            Hard cluster assignments for convergence checking.
        """
        self.model.eval()
        all_q = []

        with torch.no_grad():
            for batch in dataloader:
                x_num, x_cat = batch
                x_num = x_num.to(self.device)
                x_cat = x_cat.to(self.device)

                q = self.model.get_cluster_assignments(x_num, x_cat)
                all_q.append(q.cpu())

        if not all_q:
            logger.warning("_update_target_distribution: empty dataloader, returning empty assignments")
            self._target_distribution = torch.empty(0)
            self.model.train()
            return np.array([], dtype=np.intp)

        all_q = torch.cat(all_q, dim=0)
        self._target_distribution = compute_target_distribution(all_q)
        assignments = all_q.argmax(dim=1).numpy()
        self.model.train()
        return assignments
