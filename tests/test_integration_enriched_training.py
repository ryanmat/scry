# Description: Integration tests for end-to-end enriched feature training pipeline.
# Description: Tests extract → enrich → train → evaluate → drift detection flow.

"""Integration tests for the enriched training pipeline.

Exercises the full flow: base data → feature enrichment → X-DEC training →
silhouette evaluation → drift detection. Uses synthetic data and the
fallback (no-forecaster) enrichment path to avoid Chronos dependency.
"""

import numpy as np
import pytest
import torch


class TestEnrichedTrainingPipeline:
    """End-to-end tests for the enriched training flow."""

    @pytest.fixture
    def base_data(self) -> dict:
        """Synthetic base training data matching extract_features output format."""
        rng = np.random.default_rng(42)
        n_samples = 100
        seq_len = 30
        n_num = 9
        n_cat = 8

        return {
            "num_windows": rng.standard_normal((n_samples, seq_len, n_num)).astype(
                np.float32
            ),
            "cat_windows": rng.integers(0, 2, (n_samples, seq_len, n_cat)).astype(
                np.float32
            ),
            "labels": np.array(
                [[i, i * 1000] for i in range(n_samples)], dtype=object
            ),
            "num_norm_params": {
                "mean": np.zeros(n_num, dtype=np.float32),
                "std": np.ones(n_num, dtype=np.float32),
            },
        }

    @pytest.fixture
    def enriched_data(self, base_data) -> dict:
        """Enriched data using fallback (no-forecaster) pipeline."""
        from scry.model.forecasting.enriched_pipeline import (
            EnrichedFeaturePipeline,
        )

        pipeline = EnrichedFeaturePipeline(
            n_metrics=9,
            horizons=[15, 60],
            forecaster=None,
        )
        return pipeline.enrich(base_data)

    def test_enrich_then_train_xvae(self, enriched_data) -> None:
        """Enriched data should train through XVAE pretraining without errors."""
        from scry.data.feature_engineering import XDECDataset
        from scry.model.losses import XDECLoss
        from scry.model.xvae import TemporalXVAE

        num_windows = torch.tensor(enriched_data["num_windows"], dtype=torch.float32)
        cat_windows = torch.tensor(enriched_data["cat_windows"], dtype=torch.float32)

        dataset = XDECDataset(num_windows, cat_windows)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

        enriched_num_dim = enriched_data["num_windows"].shape[2]
        cat_dim = enriched_data["cat_windows"].shape[2]

        model = TemporalXVAE(
            num_numerical=enriched_num_dim,
            num_categorical=cat_dim,
            latent_dim=8,
            seq_len=30,
        )
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = XDECLoss(beta=1.0, lambda_cluster=0.0)

        # Run 2 epochs to verify gradient flow
        for _epoch in range(2):
            for num_batch, cat_batch in loader:
                optimizer.zero_grad()
                outputs = model(num_batch, cat_batch)
                loss_dict = loss_fn(outputs, num_batch, cat_batch)
                loss_dict["loss"].backward()
                optimizer.step()

        # Model should produce valid latent representations
        model.eval()
        with torch.no_grad():
            sample_num, sample_cat = next(iter(loader))
            output = model(sample_num, sample_cat)
            assert output["z"].shape == (sample_num.shape[0], 8)
            assert torch.isfinite(output["z"]).all()

    def test_enrich_then_train_full_xdec(self, enriched_data) -> None:
        """Full X-DEC should train on enriched data with clustering."""
        from scry.data.feature_engineering import XDECDataset
        from scry.model.clustering import compute_target_distribution
        from scry.model.losses import XDECLoss
        from scry.model.xdec import TemporalXDEC

        num_windows = torch.tensor(enriched_data["num_windows"], dtype=torch.float32)
        cat_windows = torch.tensor(enriched_data["cat_windows"], dtype=torch.float32)

        dataset = XDECDataset(num_windows, cat_windows)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

        enriched_num_dim = enriched_data["num_windows"].shape[2]
        cat_dim = enriched_data["cat_windows"].shape[2]

        model = TemporalXDEC(
            num_numerical=enriched_num_dim,
            num_categorical=cat_dim,
            latent_dim=8,
            n_clusters=5,
            seq_len=30,
        )

        # Initialize cluster centroids using the full data
        model.initialize_centroids(num_windows, cat_windows)

        # Run clustering training with loss function
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        loss_fn = XDECLoss(beta=0.01, lambda_cluster=0.1, lambda_balance=0.5)

        for _epoch in range(2):
            for num_batch, cat_batch in loader:
                optimizer.zero_grad()
                outputs = model(num_batch, cat_batch)
                p = compute_target_distribution(outputs["q"])
                loss_dict = loss_fn(
                    outputs, num_batch, cat_batch, q=outputs["q"], p=p
                )
                loss_dict["loss"].backward()
                optimizer.step()

        # Model should assign clusters
        model.eval()
        with torch.no_grad():
            sample_num, sample_cat = next(iter(loader))
            output = model(sample_num, sample_cat)
            q = output["q"]
            assert q.shape[1] == 5
            # Each sample should have cluster assignments that sum to 1
            sums = q.sum(dim=1)
            np.testing.assert_allclose(sums.numpy(), 1.0, atol=1e-5)

    def test_enriched_silhouette_evaluation(self, enriched_data) -> None:
        """Silhouette score should be computable on enriched data clusters."""
        from sklearn.metrics import silhouette_score

        from scry.data.feature_engineering import XDECDataset
        from scry.model.xdec import TemporalXDEC

        num_windows = torch.tensor(enriched_data["num_windows"], dtype=torch.float32)
        cat_windows = torch.tensor(enriched_data["cat_windows"], dtype=torch.float32)

        dataset = XDECDataset(num_windows, cat_windows)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

        enriched_num_dim = enriched_data["num_windows"].shape[2]
        cat_dim = enriched_data["cat_windows"].shape[2]

        model = TemporalXDEC(
            num_numerical=enriched_num_dim,
            num_categorical=cat_dim,
            latent_dim=8,
            n_clusters=3,
            seq_len=30,
        )

        # Initialize clusters using full data
        model.initialize_centroids(num_windows, cat_windows)

        # Get cluster assignments
        model.eval()
        with torch.no_grad():
            all_q = []
            all_embeddings = []
            for num_batch, cat_batch in loader:
                output = model(num_batch, cat_batch)
                all_q.append(output["q"])
                all_embeddings.append(output["z"])

        all_q = torch.cat(all_q, dim=0)
        all_embeddings = torch.cat(all_embeddings, dim=0)
        cluster_labels = all_q.argmax(dim=1).numpy()

        # Silhouette score should be computable (may be negative for untrained model)
        n_unique = len(np.unique(cluster_labels))
        if n_unique >= 2:
            score = silhouette_score(all_embeddings.numpy(), cluster_labels)
            assert -1.0 <= score <= 1.0

    def test_drift_detection_on_enriched_features(self, enriched_data) -> None:
        """Drift detector should work with enriched feature dimensions."""
        from scry.model.drift import DriftDetector

        # Use only the base (non-synthetic) features for drift detection
        # to avoid false positives from zero-padded enrichment features
        n_base = 9
        feature_names = [f"feature_{i}" for i in range(n_base)]

        detector = DriftDetector(n_features=n_base, feature_names=feature_names)

        # Use first half as reference, second half as current (same distribution)
        n = enriched_data["num_windows"].shape[0]
        mid = n // 2

        # Flatten sequence dimension, use only base features
        reference = enriched_data["num_windows"][:mid, :, :n_base].mean(axis=1)
        current = enriched_data["num_windows"][mid:, :, :n_base].mean(axis=1)

        result = detector.check_feature_drift(reference, current)

        assert "has_drift" in result
        assert "psi_per_feature" in result
        assert len(result["psi_per_feature"]) == n_base

    def test_drift_detection_accepts_full_enriched_dim(self, enriched_data) -> None:
        """Drift detector should accept any feature dimension including enriched."""
        from scry.model.drift import DriftDetector

        enriched_num_dim = enriched_data["num_windows"].shape[2]
        feature_names = [f"feature_{i}" for i in range(enriched_num_dim)]

        detector = DriftDetector(
            n_features=enriched_num_dim, feature_names=feature_names
        )

        # Create two batches from same RNG seed to get matching distributions
        rng = np.random.default_rng(99)
        reference = rng.standard_normal((50, enriched_num_dim)).astype(np.float32)
        current = rng.standard_normal((50, enriched_num_dim)).astype(np.float32)

        result = detector.check_feature_drift(reference, current)

        assert "psi_per_feature" in result
        assert len(result["psi_per_feature"]) == enriched_num_dim

    def test_anomaly_detection_on_enriched_features(self, enriched_data) -> None:
        """Anomaly detector should work with enriched feature dimensions."""
        from scry.model.forecasting.anomaly_detector import (
            ForecastAnomalyDetector,
        )

        enriched_num_dim = enriched_data["num_windows"].shape[2]
        metric_names = [f"metric_{i}" for i in range(enriched_num_dim)]

        detector = ForecastAnomalyDetector(metric_names=metric_names)

        # Use one sample's mean across time as actuals
        sample = enriched_data["num_windows"][0].mean(axis=0)

        # Create forecast around the actual values
        forecast = {
            "median": sample,
            "lower": sample - 1.0,
            "upper": sample + 1.0,
        }

        result = detector.detect(sample, forecast)
        assert "is_anomaly" in result
        assert "anomaly_score" in result
        # Actuals should be within the interval
        assert result["is_anomaly"] is False


class TestEnrichedPipelineNpzRoundTrip:
    """Tests for saving and loading enriched data in .npz format."""

    def test_npz_save_load_roundtrip(self, tmp_path) -> None:
        """Enriched data should survive .npz save/load cycle."""
        from scry.model.forecasting.enriched_pipeline import (
            EnrichedFeaturePipeline,
        )

        rng = np.random.default_rng(42)
        base_data = {
            "num_windows": rng.standard_normal((20, 30, 9)).astype(np.float32),
            "cat_windows": rng.integers(0, 2, (20, 30, 8)).astype(np.float32),
            "labels": np.array([[i, i * 1000] for i in range(20)], dtype=object),
            "num_norm_params": {
                "mean": np.zeros(9, dtype=np.float32),
                "std": np.ones(9, dtype=np.float32),
            },
        }

        pipeline = EnrichedFeaturePipeline(
            n_metrics=9, horizons=[15, 60], forecaster=None
        )
        enriched = pipeline.enrich(base_data)

        # Save to .npz
        output_path = tmp_path / "enriched_data.npz"
        np.savez(
            str(output_path),
            num_windows=enriched["num_windows"],
            cat_windows=enriched["cat_windows"],
            labels=enriched["labels"],
            num_norm_mean=enriched["num_norm_params"]["mean"],
            num_norm_std=enriched["num_norm_params"]["std"],
        )

        # Load back
        loaded = np.load(str(output_path), allow_pickle=True)

        np.testing.assert_array_equal(enriched["num_windows"], loaded["num_windows"])
        np.testing.assert_array_equal(enriched["cat_windows"], loaded["cat_windows"])
        assert loaded["num_windows"].shape[2] > 9  # Enriched features present
