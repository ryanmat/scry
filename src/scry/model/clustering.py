# Description: DEC clustering layer for operational state discovery.
# Description: Uses Student's t-distribution for soft cluster assignments.

"""DEC clustering layer for Temporal X-DEC."""

import torch
import torch.nn as nn


class DECLayer(nn.Module):
    """Deep Embedded Clustering layer.

    Uses Student's t-distribution to compute soft cluster assignments
    based on distance between latent embeddings and cluster centroids.

    q_ij = (1 + ||z_i - μ_j||² / α)^(-(α+1)/2) / Σ_j' ...

    Args:
        n_clusters: Number of clusters (default: 5).
        latent_dim: Dimension of latent space (default: 8).
        alpha: Degrees of freedom for Student's t-distribution (default: 1.0).
    """

    def __init__(
        self,
        n_clusters: int = 5,
        latent_dim: int = 8,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.n_clusters = n_clusters
        self.latent_dim = latent_dim
        self.alpha = alpha

        # Cluster centroids (learnable)
        self.centroids = nn.Parameter(torch.randn(n_clusters, latent_dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute soft cluster assignments using Student's t-distribution.

        Args:
            z: Latent embeddings (batch, latent_dim).

        Returns:
            Soft cluster assignments q (batch, n_clusters).
        """
        # Compute squared distances: ||z_i - μ_j||²
        # z: (batch, latent_dim), centroids: (n_clusters, latent_dim)
        # Expand for broadcasting
        z_expanded = z.unsqueeze(1)  # (batch, 1, latent_dim)
        centroids_expanded = self.centroids.unsqueeze(0)  # (1, n_clusters, latent_dim)

        sq_distances = torch.sum(
            (z_expanded - centroids_expanded) ** 2, dim=2
        )  # (batch, n_clusters)

        # Student's t-distribution kernel
        # q_ij = (1 + d²/α)^(-(α+1)/2)
        power = -(self.alpha + 1) / 2
        numerator = (1 + sq_distances / self.alpha) ** power

        # Normalize to get probabilities
        q = numerator / numerator.sum(dim=1, keepdim=True)

        return q

    def initialize_centroids(self, z: torch.Tensor) -> None:
        """Initialize centroids using k-means++ style initialization.

        Args:
            z: Latent embeddings to initialize from (n_samples, latent_dim).
        """
        with torch.no_grad():
            n_samples = z.shape[0]
            indices = []

            # First centroid: random sample
            idx = torch.randint(0, n_samples, (1,)).item()
            indices.append(idx)

            # Remaining centroids: proportional to squared distance
            for _ in range(1, self.n_clusters):
                # Compute distances to nearest centroid
                centroids_selected = z[indices]  # (k, latent_dim)
                distances = torch.cdist(z, centroids_selected)  # (n_samples, k)
                min_distances = distances.min(dim=1).values  # (n_samples,)

                # Sample proportional to squared distance
                probs = min_distances ** 2
                probs = probs / probs.sum()
                idx = torch.multinomial(probs, 1).item()
                indices.append(idx)

            # Set centroids
            self.centroids.data = z[indices].clone()


def compute_target_distribution(q: torch.Tensor) -> torch.Tensor:
    """Compute target distribution P from soft assignments Q.

    The target distribution sharpens Q to encourage confident assignments:
    p_ij = (q_ij² / f_j) / Σ_j' (q_ij'² / f_j')
    where f_j = Σ_i q_ij

    Args:
        q: Soft cluster assignments (batch, n_clusters).

    Returns:
        Target distribution p (batch, n_clusters).
    """
    # f_j = sum over samples for each cluster
    f = q.sum(dim=0, keepdim=True)  # (1, n_clusters)

    # p_ij = q_ij² / f_j (unnormalized)
    numerator = q ** 2 / (f + 1e-10)

    # Normalize per sample
    p = numerator / numerator.sum(dim=1, keepdim=True)

    return p
