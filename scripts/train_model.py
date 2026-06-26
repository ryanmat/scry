#!/usr/bin/env python3
# Description: Training script for Temporal X-DEC model.
# Description: Handles data loading, training, evaluation, and model saving.

"""Training script for the Temporal X-DEC model."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scry.model.evaluate import cluster_summary, evaluate_clustering, visualize_clusters
from scry.model.trainer import XDECTrainer
from scry.model.xdec import TemporalXDEC


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Temporal X-DEC model for K8s operational state clustering"
    )

    # Data
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to training data .npz file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/xdec_model.pt",
        help="Path to save trained model",
    )
    parser.add_argument(
        "--validate-split",
        type=float,
        default=0.1,
        help="Fraction of data for validation",
    )

    # Model architecture
    parser.add_argument(
        "--num-clusters",
        type=int,
        default=5,
        help="Number of operational state clusters",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=8,
        help="Dimension of latent embedding space",
    )
    parser.add_argument(
        "--num-hidden",
        type=int,
        default=64,
        help="Hidden dimension for numerical GRU",
    )
    parser.add_argument(
        "--cat-hidden",
        type=int,
        default=32,
        help="Hidden dimension for categorical GRU",
    )

    # Training
    parser.add_argument(
        "--pretrain-epochs",
        type=int,
        default=500,
        help="Epochs for XVAE pretraining",
    )
    parser.add_argument(
        "--cluster-epochs",
        type=int,
        default=300,
        help="Epochs for clustering training",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Adam learning rate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to train on",
    )

    # DEC hyperparameters
    parser.add_argument(
        "--beta",
        type=float,
        default=0.01,
        help="VAE KL weight. Lower values preserve cluster structure. Default: 0.01",
    )
    parser.add_argument(
        "--lambda-cluster",
        type=float,
        default=0.1,
        help="Fixed clustering loss weight (IDEC-style). Default: 0.1",
    )
    parser.add_argument(
        "--cluster-lr",
        type=float,
        default=1e-4,
        help="Separate learning rate for cluster phase. Default: 1e-4",
    )
    parser.add_argument(
        "--lambda-balance",
        type=float,
        default=0.5,
        help="Cluster balance entropy weight. Penalizes imbalanced assignments. Default: 0.5",
    )

    # Subsampling
    parser.add_argument(
        "--subsample",
        type=float,
        default=1.0,
        help="Fraction of data to use for training (0.0-1.0). Default: 1.0 (all data)",
    )

    # Checkpointing
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Resume training from checkpoint file",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory for periodic checkpoint saves",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=50,
        help="Save checkpoint every N epochs. Default: 50",
    )

    # Evaluation
    parser.add_argument(
        "--sweep-clusters",
        action="store_true",
        help="Run silhouette sweep over k=2..10, print recommendation, and exit",
    )

    # Output
    parser.add_argument(
        "--plot-clusters",
        type=str,
        default=None,
        help="Path to save cluster visualization",
    )

    return parser.parse_args()


class TrainingConfig:
    """Configuration object for trainer."""

    def __init__(self, args: argparse.Namespace):
        self.num_clusters = args.num_clusters
        self.latent_dim = args.latent_dim
        self.sequence_length = 30  # Fixed window size
        self.numerical_hidden_dim = args.num_hidden
        self.categorical_hidden_dim = args.cat_hidden
        self.learning_rate = args.learning_rate
        self.beta = args.beta
        self.lambda_cluster = args.lambda_cluster
        self.lambda_balance = args.lambda_balance
        self.cluster_lr = args.cluster_lr
        self.pretrain_epochs = args.pretrain_epochs
        self.cluster_epochs = args.cluster_epochs


def load_data(data_path: str) -> tuple:
    """Load training data from .npz file.

    Args:
        data_path: Path to .npz file with num_windows, cat_windows.

    Returns:
        Tuple of (num_windows, cat_windows, normalization_params).
    """
    print(f"Loading data from {data_path}...")
    data = np.load(data_path, allow_pickle=True)

    num_windows = data["num_windows"]
    cat_windows = data["cat_windows"]

    # Load normalization params for inference
    norm_params = None
    if "num_norm_mean" in data and "num_norm_std" in data:
        norm_params = {
            "mean": data["num_norm_mean"],
            "std": data["num_norm_std"],
        }

    print(f"  Numerical windows: {num_windows.shape}")
    print(f"  Categorical windows: {cat_windows.shape}")

    return num_windows, cat_windows, norm_params


def create_dataloaders(
    num_windows: np.ndarray,
    cat_windows: np.ndarray,
    batch_size: int,
    val_split: float,
) -> tuple:
    """Create train and validation DataLoaders.

    Args:
        num_windows: Numerical feature windows.
        cat_windows: Categorical feature windows.
        batch_size: Batch size.
        val_split: Validation split fraction.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    # Convert to tensors
    x_num = torch.tensor(num_windows, dtype=torch.float32)
    x_cat = torch.tensor(cat_windows, dtype=torch.float32)

    dataset = TensorDataset(x_num, x_cat)

    # Split
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val

    train_dataset, val_dataset = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    print(f"  Train samples: {n_train}")
    print(f"  Validation samples: {n_val}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader


def main():
    """Main training entry point."""
    args = parse_args()

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    num_windows, cat_windows, norm_params = load_data(args.data)

    # Subsample if requested
    if args.subsample < 1.0:
        rng = np.random.default_rng(42)
        n_total = len(num_windows)
        n_subset = max(1, int(n_total * args.subsample))
        indices = rng.choice(n_total, size=n_subset, replace=False)
        num_windows = num_windows[indices]
        cat_windows = cat_windows[indices]
        print(f"  Subsampled: {n_subset:,} / {n_total:,} samples ({args.subsample:.0%})")

    # Infer dimensions from data
    seq_len, num_numerical = num_windows.shape[1], num_windows.shape[2]
    num_categorical = cat_windows.shape[2]

    print("\nModel configuration:")
    print(f"  Sequence length: {seq_len}")
    print(f"  Numerical features: {num_numerical}")
    print(f"  Categorical features: {num_categorical}")
    print(f"  Latent dimension: {args.latent_dim}")
    print(f"  Number of clusters: {args.num_clusters}")
    print(f"  Beta (VAE KL):    {args.beta}")
    print(f"  Lambda cluster:   {args.lambda_cluster}")
    print(f"  Lambda balance:   {args.lambda_balance}")
    print(f"  Cluster LR:       {args.cluster_lr}")

    # Cluster sweep mode: evaluate k values and exit
    if args.sweep_clusters:
        from scry.model.evaluate import sweep_clusters

        print("\nRunning cluster count sweep (k=2..10)...")
        sweep_loader = DataLoader(
            TensorDataset(
                torch.tensor(num_windows, dtype=torch.float32),
                torch.tensor(cat_windows, dtype=torch.float32),
            ),
            batch_size=args.batch_size,
            shuffle=False,
        )
        results = sweep_clusters(
            sweep_loader,
            k_range=range(2, 11),
            latent_dim=args.latent_dim,
            pretrain_epochs=min(50, args.pretrain_epochs),
            device=args.device,
        )

        print("\nSweep Results:")
        print(f"  {'k':>3}  {'Silhouette':>12}  Distribution")
        print(f"  {'---':>3}  {'----------':>12}  {'------------'}")
        best_k = max(results, key=lambda k: results[k]["silhouette_score"])
        for k in sorted(results):
            sil = results[k]["silhouette_score"]
            dist = results[k]["cluster_distribution"]
            marker = " <-- best" if k == best_k else ""
            print(f"  {k:>3}  {sil:>12.4f}  {dist}{marker}")

        print(f"\nRecommendation: k={best_k} (silhouette={results[best_k]['silhouette_score']:.4f})")
        return

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        num_windows,
        cat_windows,
        args.batch_size,
        args.validate_split,
    )

    # Create model
    print("\nInitializing model...")
    model = TemporalXDEC(
        num_numerical=num_numerical,
        num_categorical=num_categorical,
        seq_len=seq_len,
        num_hidden=args.num_hidden,
        cat_hidden=args.cat_hidden,
        latent_dim=args.latent_dim,
        n_clusters=args.num_clusters,
    )

    # Create trainer
    config = TrainingConfig(args)
    trainer = XDECTrainer(model, config, device=args.device)

    # Checkpoint directory setup
    checkpoint_dir = args.checkpoint_dir
    if checkpoint_dir:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Training
    print(f"\nStarting training on {trainer.device}...")
    print(f"  Pretrain epochs: {args.pretrain_epochs}")
    print(f"  Cluster epochs: {args.cluster_epochs}")
    if args.checkpoint:
        print(f"  Resuming from: {args.checkpoint}")
    if checkpoint_dir:
        print(f"  Checkpoint dir: {checkpoint_dir}")
        print(f"  Checkpoint interval: {args.checkpoint_interval}")

    history = trainer.fit(
        train_loader,
        pretrain_epochs=args.pretrain_epochs,
        cluster_epochs=args.cluster_epochs,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        resume_checkpoint=args.checkpoint,
    )

    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    metrics = evaluate_clustering(model, val_loader)

    print("\nValidation Results:")
    print(f"  Silhouette Score: {metrics['silhouette_score']:.4f}")
    print(f"  Cluster Distribution: {metrics['cluster_distribution']}")

    # Cluster summary
    summary = cluster_summary(model, val_loader)
    print("\nCluster Summary:")
    print(summary.to_string(index=False))

    # Save model
    print(f"\nSaving model to {args.output}...")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "num_numerical": num_numerical,
            "num_categorical": num_categorical,
            "seq_len": seq_len,
            "num_hidden": args.num_hidden,
            "cat_hidden": args.cat_hidden,
            "latent_dim": args.latent_dim,
            "n_clusters": args.num_clusters,
        },
        "normalization": norm_params,
        "history": history,
        "metrics": {
            "silhouette_score": metrics["silhouette_score"],
            "cluster_distribution": metrics["cluster_distribution"],
        },
    }, args.output)

    # Visualize clusters if requested
    if args.plot_clusters:
        print(f"Saving cluster visualization to {args.plot_clusters}...")
        visualize_clusters(
            metrics["embeddings"],
            metrics["labels"],
            save_path=args.plot_clusters,
        )

    print("\nTraining complete!")

    # Print final loss values
    print("\nFinal training losses:")
    if history.get("pretrain"):
        print(f"  Pretrain loss: {history['pretrain']['loss'][-1]:.4f}")
    else:
        print("  Pretrain: skipped (resumed from cluster checkpoint)")
    print(f"  Cluster loss: {history['cluster']['loss'][-1]:.4f}")


if __name__ == "__main__":
    main()
