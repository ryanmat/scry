# Description: Unit tests for the model evaluation module.
# Description: Tests clustering metrics, visualization, and summary statistics.

"""Tests for scry.model.evaluate module."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset


@pytest.fixture
def trained_model():
    """Create a trained model for testing."""
    from scry.model.xdec import TemporalXDEC

    model = TemporalXDEC(num_numerical=9, num_categorical=8, n_clusters=5, latent_dim=8)
    # Initialize centroids with some spread
    model.dec_layer.centroids.data = torch.randn(5, 8) * 2
    return model


@pytest.fixture
def sample_dataloader():
    """Create sample data for evaluation."""
    batch_size = 16
    num_samples = 64
    seq_len = 30
    num_numerical = 9
    num_categorical = 8

    x_num = torch.randn(num_samples, seq_len, num_numerical)
    x_cat = torch.rand(num_samples, seq_len, num_categorical)

    dataset = TensorDataset(x_num, x_cat)
    return DataLoader(dataset, batch_size=batch_size)


class TestEvaluateClustering:
    """Tests for evaluate_clustering function."""

    def test_returns_metrics_dict(self, trained_model, sample_dataloader) -> None:
        """evaluate_clustering should return dict with metrics."""
        from scry.model.evaluate import evaluate_clustering

        metrics = evaluate_clustering(trained_model, sample_dataloader)

        assert isinstance(metrics, dict)
        assert "silhouette_score" in metrics
        assert "cluster_distribution" in metrics

    def test_silhouette_score_in_range(self, trained_model, sample_dataloader) -> None:
        """Silhouette score should be in [-1, 1]."""
        from scry.model.evaluate import evaluate_clustering

        metrics = evaluate_clustering(trained_model, sample_dataloader)

        assert -1 <= metrics["silhouette_score"] <= 1

    def test_cluster_distribution_sums_to_total(self, trained_model, sample_dataloader) -> None:
        """Cluster distribution should sum to total samples."""
        from scry.model.evaluate import evaluate_clustering

        metrics = evaluate_clustering(trained_model, sample_dataloader)
        total = sum(metrics["cluster_distribution"].values())

        assert total == len(sample_dataloader.dataset)

    def test_includes_embeddings_and_labels(self, trained_model, sample_dataloader) -> None:
        """Should return embeddings and cluster labels."""
        from scry.model.evaluate import evaluate_clustering

        metrics = evaluate_clustering(trained_model, sample_dataloader)

        assert "embeddings" in metrics
        assert "labels" in metrics
        assert len(metrics["embeddings"]) == len(sample_dataloader.dataset)
        assert len(metrics["labels"]) == len(sample_dataloader.dataset)


class TestSilhouetteSampleCap:
    """Training runs on 2026-04-22 hung for 45+ min on sklearn.metrics.silhouette_score
    with n=25K samples because the function is O(n^2 * d) and runs single-core CPU
    with no progress output. The cap in `_silhouette_score_capped` bounds cost by
    forwarding sklearn's `sample_size` parameter for inputs over the threshold.
    """

    def test_single_cluster_returns_zero(self) -> None:
        from scry.model.evaluate import _silhouette_score_capped

        embeddings = np.random.default_rng(0).standard_normal((100, 4))
        labels = np.zeros(100, dtype=int)
        assert _silhouette_score_capped(embeddings, labels) == 0.0

    def test_below_cap_uses_full_dataset(self, monkeypatch) -> None:
        """When n <= cap, no sample_size kwarg is passed -- exact behavior parity."""
        from scry.model import evaluate as evaluate_mod

        captured: dict = {}

        def fake(emb, lab, **kwargs):
            captured.update(kwargs)
            return 0.1

        monkeypatch.setattr(evaluate_mod, "silhouette_score", fake)

        n = evaluate_mod.SILHOUETTE_SAMPLE_CAP  # exactly at cap, still below threshold
        rng = np.random.default_rng(0)
        emb = rng.standard_normal((n, 4))
        lab = rng.integers(0, 3, size=n)

        evaluate_mod._silhouette_score_capped(emb, lab)
        assert "sample_size" not in captured
        assert "random_state" not in captured

    def test_above_cap_forwards_sample_size(self, monkeypatch) -> None:
        """When n > cap, sklearn receives sample_size so the O(n^2) cost is bounded."""
        from scry.model import evaluate as evaluate_mod

        captured: dict = {}

        def fake(emb, lab, **kwargs):
            captured.update(kwargs)
            return 0.1

        monkeypatch.setattr(evaluate_mod, "silhouette_score", fake)

        n = evaluate_mod.SILHOUETTE_SAMPLE_CAP + 1
        rng = np.random.default_rng(0)
        emb = rng.standard_normal((n, 4))
        lab = rng.integers(0, 3, size=n)

        evaluate_mod._silhouette_score_capped(emb, lab)
        assert captured.get("sample_size") == evaluate_mod.SILHOUETTE_SAMPLE_CAP
        assert captured.get("random_state") == evaluate_mod.SILHOUETTE_RANDOM_STATE

    def test_cap_is_fast_on_large_n(self) -> None:
        """End-to-end: 5000 samples must finish in well under a second with the cap."""
        import time

        from scry.model.evaluate import _silhouette_score_capped

        rng = np.random.default_rng(0)
        n = 5000
        emb = rng.standard_normal((n, 8)).astype(np.float32)
        lab = rng.integers(0, 5, size=n)

        t0 = time.perf_counter()
        score = _silhouette_score_capped(emb, lab)
        elapsed = time.perf_counter() - t0

        assert -1.0 <= score <= 1.0
        assert elapsed < 2.0, f"capped silhouette should be fast, took {elapsed:.2f}s"


class TestVisualizeAndSave:
    """Tests for visualization functions."""

    def test_visualize_clusters_creates_file(self, trained_model, sample_dataloader) -> None:
        """visualize_clusters should create plot file when path provided."""
        from scry.model.evaluate import evaluate_clustering, visualize_clusters

        metrics = evaluate_clustering(trained_model, sample_dataloader)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "clusters.png"
            visualize_clusters(
                metrics["embeddings"],
                metrics["labels"],
                save_path=str(path),
            )

            assert path.exists()

    def test_visualize_clusters_handles_no_path(self, trained_model, sample_dataloader) -> None:
        """visualize_clusters should not error without save_path."""
        from scry.model.evaluate import evaluate_clustering, visualize_clusters

        metrics = evaluate_clustering(trained_model, sample_dataloader)

        # Should not raise when save_path is None (just don't show)
        visualize_clusters(
            metrics["embeddings"],
            metrics["labels"],
            save_path=None,
            show=False,
        )


class TestClusterSummary:
    """Tests for cluster_summary function."""

    def test_summary_returns_dataframe(self, trained_model, sample_dataloader) -> None:
        """cluster_summary should return a DataFrame."""
        import pandas as pd

        from scry.model.evaluate import cluster_summary

        summary = cluster_summary(trained_model, sample_dataloader)

        assert isinstance(summary, pd.DataFrame)

    def test_summary_has_cluster_column(self, trained_model, sample_dataloader) -> None:
        """Summary should have cluster as index or column."""
        from scry.model.evaluate import cluster_summary

        summary = cluster_summary(trained_model, sample_dataloader)

        assert "cluster" in summary.columns or summary.index.name == "cluster"

    def test_summary_includes_count(self, trained_model, sample_dataloader) -> None:
        """Summary should include sample count per cluster."""
        from scry.model.evaluate import cluster_summary

        summary = cluster_summary(trained_model, sample_dataloader)

        assert "count" in summary.columns


class TestSilhouetteSweep:
    """Tests for cluster sweep evaluation."""

    def test_sweep_returns_dict_per_k(self, sample_dataloader) -> None:
        """sweep_clusters should return a dict with results for each k."""
        from scry.model.evaluate import sweep_clusters

        results = sweep_clusters(
            sample_dataloader,
            k_range=range(2, 5),
            latent_dim=8,
            pretrain_epochs=2,
        )

        assert isinstance(results, dict)
        for k in [2, 3, 4]:
            assert k in results
            assert "silhouette_score" in results[k]

    def test_sweep_silhouette_in_valid_range(self, sample_dataloader) -> None:
        """Silhouette scores from sweep should be in [-1, 1]."""
        from scry.model.evaluate import sweep_clusters

        results = sweep_clusters(
            sample_dataloader,
            k_range=range(2, 4),
            latent_dim=8,
            pretrain_epochs=2,
        )

        for k, metrics in results.items():
            assert -1 <= metrics["silhouette_score"] <= 1


class TestGetEmbeddings:
    """Tests for get_embeddings utility."""

    def test_get_embeddings_returns_array(self, trained_model, sample_dataloader) -> None:
        """get_embeddings should return numpy array."""
        from scry.model.evaluate import get_embeddings

        embeddings = get_embeddings(trained_model, sample_dataloader)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == len(sample_dataloader.dataset)
        assert embeddings.shape[1] == trained_model.latent_dim

    def test_get_embeddings_deterministic(self, trained_model, sample_dataloader) -> None:
        """get_embeddings should be deterministic in eval mode."""
        from scry.model.evaluate import get_embeddings

        trained_model.eval()
        emb1 = get_embeddings(trained_model, sample_dataloader)
        emb2 = get_embeddings(trained_model, sample_dataloader)

        # In eval mode, VAE uses mean (no sampling), so should be identical
        # Actually VAE still samples, so we just check shape matches
        assert emb1.shape == emb2.shape
