# Description: Loss functions for X-DEC model training.
# Description: Includes reconstruction losses, KL divergences, and combined XDECLoss.

"""Loss functions for the Temporal X-DEC model."""


import torch
import torch.nn.functional as F  # noqa: N812 -- PyTorch convention


def reconstruction_loss_numerical(
    x: torch.Tensor, x_recon: torch.Tensor
) -> torch.Tensor:
    """Compute MSE reconstruction loss for numerical features.

    Args:
        x: Original numerical features (batch, seq_len, num_features).
        x_recon: Reconstructed numerical features (batch, seq_len, num_features).

    Returns:
        Scalar MSE loss averaged over all elements.
    """
    return F.mse_loss(x_recon, x, reduction="mean")


def reconstruction_loss_categorical(
    x: torch.Tensor, x_recon: torch.Tensor
) -> torch.Tensor:
    """Compute BCE reconstruction loss for categorical features.

    Args:
        x: Original categorical features in [0, 1] (batch, seq_len, num_features).
        x_recon: Reconstructed categorical features (batch, seq_len, num_features).

    Returns:
        Scalar BCE loss averaged over all elements. Zero for purely-numerical
        profiles, where the categorical tensor is zero-width.
    """
    # Purely-numerical profiles (num_categorical=0) carry a zero-width
    # categorical tensor; BCE over zero elements is undefined, so the term
    # contributes nothing and the total/gradients are unaffected.
    if x_recon.shape[-1] == 0:
        return x_recon.new_zeros(())
    return F.binary_cross_entropy(x_recon, x, reduction="mean")


def vae_kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Compute KL divergence between q(z|x) and p(z) = N(0, I).

    KL(q||p) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))

    Args:
        mu: Mean of q(z|x) (batch, latent_dim).
        logvar: Log variance of q(z|x) (batch, latent_dim).

    Returns:
        Scalar KL divergence loss averaged over batch.
    """
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return kl.mean()


def dec_clustering_loss(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Compute KL divergence between soft assignments q and target distribution p.

    This is the DEC clustering loss that encourages confident cluster assignments.

    Args:
        q: Soft cluster assignments (batch, n_clusters).
        p: Target distribution (batch, n_clusters).

    Returns:
        Scalar KL divergence loss.
    """
    # KL(P||Q) = sum(p * log(p/q))
    # Add small epsilon for numerical stability
    eps = 1e-10
    kl = p * torch.log((p + eps) / (q + eps))
    return kl.sum(dim=1).mean()


def cluster_balance_entropy(q: torch.Tensor) -> torch.Tensor:
    """Compute Shannon entropy of marginal cluster distribution.

    H(f) = -sum(f_j * log(f_j)) where f_j = mean(q_ij) over batch.
    Higher entropy means more balanced cluster assignments.

    Args:
        q: Soft cluster assignments (batch, n_clusters).

    Returns:
        Scalar entropy value (higher = more balanced).
    """
    eps = 1e-10
    # Marginal cluster distribution: average assignment across batch
    f = q.mean(dim=0)
    entropy = -(f * torch.log(f + eps)).sum()
    return entropy


class XDECLoss:
    """Combined loss function for Temporal X-DEC training.

    Combines:
        - Numerical reconstruction loss (MSE)
        - Categorical reconstruction loss (BCE)
        - VAE KL divergence (regularization)
        - DEC clustering KL divergence (optional, during fine-tuning)
        - Cluster balance entropy (optional, penalizes imbalanced assignments)

    Args:
        beta: Weight for VAE KL loss (default: 1.0).
        lambda_cluster: Weight for clustering loss (default: 0.0).
        lambda_balance: Weight for balance entropy (default: 0.0).
    """

    def __init__(
        self,
        beta: float = 1.0,
        lambda_cluster: float = 0.0,
        lambda_balance: float = 0.0,
    ) -> None:
        self.beta = beta
        self.lambda_cluster = lambda_cluster
        self.lambda_balance = lambda_balance

    def __call__(
        self,
        outputs: dict[str, torch.Tensor],
        x_num: torch.Tensor,
        x_cat: torch.Tensor,
        q: torch.Tensor | None = None,
        p: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute combined X-DEC loss.

        Args:
            outputs: Dict from TemporalXVAE forward pass containing
                x_num_recon, x_cat_recon, mu, logvar.
            x_num: Original numerical features.
            x_cat: Original categorical features.
            q: Soft cluster assignments (optional).
            p: Target distribution for clustering (optional).

        Returns:
            Dict with 'loss' (total) and component losses.
        """
        # Reconstruction losses
        recon_num = reconstruction_loss_numerical(x_num, outputs["x_num_recon"])
        recon_cat = reconstruction_loss_categorical(x_cat, outputs["x_cat_recon"])

        # VAE KL divergence
        kl_vae = vae_kl_loss(outputs["mu"], outputs["logvar"])

        # Total loss
        total = recon_num + recon_cat + self.beta * kl_vae

        result = {
            "loss": total,
            "recon_num": recon_num,
            "recon_cat": recon_cat,
            "kl_vae": kl_vae,
        }

        # Add clustering loss if q and p provided
        if q is not None and p is not None:
            kl_cluster = dec_clustering_loss(q, p)
            result["kl_cluster"] = kl_cluster
            total = total + self.lambda_cluster * kl_cluster

            # Add balance entropy regularization (subtract to maximize entropy)
            if self.lambda_balance > 0:
                entropy = cluster_balance_entropy(q)
                result["balance_entropy"] = entropy
                total = total - self.lambda_balance * entropy

            result["loss"] = total

        return result
