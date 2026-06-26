# Description: Evaluation utilities for trained X-DEC models.
# Description: Includes clustering metrics, visualization, and summary statistics.

"""Evaluation utilities for Temporal X-DEC model."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader

from scry.model.xdec import TemporalXDEC

# sklearn's silhouette_score is O(n^2 * d). At n=25,000 on a single CPU core
# this takes ~50 minutes and looks exactly like a training hang (no stdout
# output because sklearn doesn't stream progress). Training runs on 2026-04-22
# burned 45+ min silently on this after all torch epochs had already finished
# on GPU. Cap at 2000 via sklearn's sample_size parameter -- random-subset
# silhouette stays statistically representative at this size while bounding
# compute to ~0.3 s regardless of input length. Below the cap we pass the
# full set for exact behavior parity with the pre-cap code path.
SILHOUETTE_SAMPLE_CAP = 2000
SILHOUETTE_RANDOM_STATE = 0


def _silhouette_score_capped(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Compute silhouette_score with a sample-size cap to bound O(n^2) cost."""
    if len(np.unique(labels)) < 2:
        return 0.0
    kwargs: dict = {}
    if len(embeddings) > SILHOUETTE_SAMPLE_CAP:
        kwargs["sample_size"] = SILHOUETTE_SAMPLE_CAP
        kwargs["random_state"] = SILHOUETTE_RANDOM_STATE
    return float(silhouette_score(embeddings, labels, **kwargs))


def get_embeddings(
    model: TemporalXDEC,
    dataloader: DataLoader,
    device: str | None = None,
) -> np.ndarray:
    """Extract latent embeddings from model.

    Args:
        model: Trained TemporalXDEC model.
        dataloader: DataLoader with (x_num, x_cat) batches.
        device: Device to run on (default: model's device).

    Returns:
        Embeddings array of shape (n_samples, latent_dim).
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    embeddings = []

    with torch.no_grad():
        for batch in dataloader:
            x_num, x_cat = batch
            x_num = x_num.to(device)
            x_cat = x_cat.to(device)

            z = model.encode(x_num, x_cat)
            embeddings.append(z.cpu().numpy())

    return np.concatenate(embeddings, axis=0)


def evaluate_clustering(
    model: TemporalXDEC,
    dataloader: DataLoader,
    device: str | None = None,
) -> dict:
    """Evaluate clustering quality of trained model.

    Args:
        model: Trained TemporalXDEC model.
        dataloader: DataLoader with (x_num, x_cat) batches.
        device: Device to run on.

    Returns:
        Dict with metrics:
            - silhouette_score: Clustering quality (-1 to 1)
            - cluster_distribution: Count per cluster
            - embeddings: Latent embeddings
            - labels: Cluster assignments
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for batch in dataloader:
            x_num, x_cat = batch
            x_num = x_num.to(device)
            x_cat = x_cat.to(device)

            z = model.encode(x_num, x_cat)
            cluster_labels = model.predict_cluster(x_num, x_cat)

            embeddings.append(z.cpu().numpy())
            labels.append(cluster_labels.cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)

    sil_score = _silhouette_score_capped(embeddings, labels)

    # Cluster distribution
    unique, counts = np.unique(labels, return_counts=True)
    distribution = {int(k): int(v) for k, v in zip(unique, counts)}

    return {
        "silhouette_score": float(sil_score),
        "cluster_distribution": distribution,
        "embeddings": embeddings,
        "labels": labels,
    }


def sweep_clusters(
    dataloader: DataLoader,
    k_range: Sequence[int] = range(2, 11),
    latent_dim: int = 8,
    pretrain_epochs: int = 50,
    device: str | None = None,
) -> dict[int, dict]:
    """Sweep over cluster counts and evaluate each with silhouette score.

    For each k, creates a fresh TemporalXDEC model, pretrains the XVAE,
    initializes k-means centroids, and computes silhouette score on the
    resulting cluster assignments.

    Args:
        dataloader: DataLoader with (x_num, x_cat) batches.
        k_range: Range of cluster counts to evaluate.
        latent_dim: Latent dimension for the XVAE.
        pretrain_epochs: Epochs for pretraining each model.
        device: Device to train on (auto-detected if not provided).

    Returns:
        Dict mapping k to metrics dict with silhouette_score and
        cluster_distribution.
    """
    from scry.model.trainer import XDECTrainer

    # Infer dimensions from first batch
    sample_batch = next(iter(dataloader))
    x_num_sample, x_cat_sample = sample_batch
    seq_len = x_num_sample.shape[1]
    num_numerical = x_num_sample.shape[2]
    num_categorical = x_cat_sample.shape[2]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    results = {}

    for k in k_range:
        model = TemporalXDEC(
            num_numerical=num_numerical,
            num_categorical=num_categorical,
            seq_len=seq_len,
            latent_dim=latent_dim,
            n_clusters=k,
        )

        # Minimal config for pretraining
        class _SweepConfig:
            learning_rate = 1e-3
            beta = 0.01
            lambda_cluster = 0.0
            lambda_balance = 0.0
            cluster_lr = 1e-4

        trainer = XDECTrainer(model, _SweepConfig(), device=device)
        trainer.pretrain(dataloader, epochs=pretrain_epochs)
        trainer.initialize_clusters(dataloader)

        # Evaluate
        metrics = evaluate_clustering(model, dataloader, device=device)
        results[k] = {
            "silhouette_score": metrics["silhouette_score"],
            "cluster_distribution": metrics["cluster_distribution"],
        }

    return results


def visualize_clusters(
    embeddings: np.ndarray,
    labels: np.ndarray,
    save_path: str | None = None,
    show: bool = False,
) -> None:
    """Visualize cluster embeddings using t-SNE.

    Args:
        embeddings: Latent embeddings (n_samples, latent_dim).
        labels: Cluster labels (n_samples,).
        save_path: Path to save plot (optional).
        show: Whether to display plot interactively.
    """
    import matplotlib
    from sklearn.manifold import TSNE

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Reduce to 2D with t-SNE
    if embeddings.shape[1] > 2:
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1))
        embeddings_2d = tsne.fit_transform(embeddings)
    else:
        embeddings_2d = embeddings

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=labels,
        cmap="tab10",
        alpha=0.7,
        s=50,
    )

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("Cluster Visualization")

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Cluster")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)


def cluster_summary(
    model: TemporalXDEC,
    dataloader: DataLoader,
    device: str | None = None,
) -> pd.DataFrame:
    """Generate summary statistics per cluster.

    Args:
        model: Trained TemporalXDEC model.
        dataloader: DataLoader with (x_num, x_cat) batches.
        device: Device to run on.

    Returns:
        DataFrame with cluster statistics.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    all_labels = []
    all_embeddings = []

    with torch.no_grad():
        for batch in dataloader:
            x_num, x_cat = batch
            x_num = x_num.to(device)
            x_cat = x_cat.to(device)

            z = model.encode(x_num, x_cat)
            labels = model.predict_cluster(x_num, x_cat)

            all_embeddings.append(z.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # Compute per-cluster statistics
    n_clusters = model.n_clusters
    summary_data = []

    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        count = mask.sum()

        if count > 0:
            cluster_embeddings = embeddings[mask]
            centroid = model.dec_layer.centroids[cluster_id].detach().cpu().numpy()

            # Compute distance to centroid
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            avg_distance = distances.mean()
            std_distance = distances.std()

            # Embedding statistics
            std_embedding = cluster_embeddings.std(axis=0).mean()
        else:
            avg_distance = 0.0
            std_distance = 0.0
            std_embedding = 0.0

        summary_data.append({
            "cluster": cluster_id,
            "count": int(count),
            "percentage": float(count / len(labels) * 100),
            "avg_distance_to_centroid": float(avg_distance),
            "std_distance": float(std_distance),
            "embedding_variance": float(std_embedding),
        })

    return pd.DataFrame(summary_data)
