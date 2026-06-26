# Description: Unit tests for the enriched feature pipeline with Chronos-2 forecasts.
# Description: Tests feature enrichment, backward compatibility, and shape consistency.

"""Tests for scry.model.forecasting.enriched_pipeline module."""

import numpy as np
import pytest


class TestEnrichedFeaturePipeline:
    """Tests for EnrichedFeaturePipeline class."""

    @pytest.fixture
    def base_data(self) -> dict:
        """Sample base training data (9 numerical, 8 categorical features)."""
        n_samples = 50
        seq_len = 30
        return {
            "num_windows": np.random.randn(n_samples, seq_len, 9),
            "cat_windows": np.random.rand(n_samples, seq_len, 8),
            "labels": np.array([[i, i * 1000] for i in range(n_samples)], dtype=object),
            "num_norm_params": {
                "mean": np.zeros(9),
                "std": np.ones(9),
            },
        }

    @pytest.fixture
    def enriched_data(self, base_data) -> dict:
        """Enriched data with forecast-derived features."""
        from scry.model.forecasting.enriched_pipeline import EnrichedFeaturePipeline

        pipeline = EnrichedFeaturePipeline(
            n_metrics=9,
            horizons=[15, 60],
        )

        return pipeline.enrich(base_data)

    def test_enriched_transform_adds_features(self, base_data, enriched_data) -> None:
        """Enriched data should have more numerical features than base."""
        base_num_features = base_data["num_windows"].shape[2]
        enriched_num_features = enriched_data["num_windows"].shape[2]

        # 9 base + 9*2 residuals + 9*2 uncertainties = 9 + 18 + 18 = 45
        assert enriched_num_features > base_num_features

    def test_cat_windows_unchanged(self, base_data, enriched_data) -> None:
        """Categorical windows should be unchanged after enrichment."""
        np.testing.assert_array_equal(
            base_data["cat_windows"], enriched_data["cat_windows"]
        )

    def test_labels_match_base(self, base_data, enriched_data) -> None:
        """Labels should be unchanged after enrichment."""
        np.testing.assert_array_equal(
            base_data["labels"], enriched_data["labels"]
        )

    def test_fallback_without_forecaster(self, base_data) -> None:
        """Without Chronos installed, enrichment should fall back to base features."""
        from scry.model.forecasting.enriched_pipeline import EnrichedFeaturePipeline

        pipeline = EnrichedFeaturePipeline(
            n_metrics=9,
            horizons=[15, 60],
            forecaster=None,
        )

        result = pipeline.enrich(base_data)

        # Without forecaster, uses synthetic residuals (zeros + random uncertainty)
        # but still adds the feature dimensions
        assert result["num_windows"].shape[0] == base_data["num_windows"].shape[0]
        assert result["num_windows"].shape[2] > base_data["num_windows"].shape[2]

    def test_enriched_sample_count_preserved(self, base_data, enriched_data) -> None:
        """Number of samples should be unchanged after enrichment."""
        assert enriched_data["num_windows"].shape[0] == base_data["num_windows"].shape[0]

    def test_enriched_sequence_length_preserved(self, base_data, enriched_data) -> None:
        """Sequence length should be unchanged after enrichment."""
        assert enriched_data["num_windows"].shape[1] == base_data["num_windows"].shape[1]


class TestFourHorizonEnrichment:
    """Tests for production 4-horizon enrichment (t+15m, t+1h, t+4h, t+24h)."""

    PRODUCTION_HORIZONS = [15, 60, 240, 1440]

    @pytest.fixture
    def base_data(self) -> dict:
        """Sample base training data (9 numerical, 8 categorical features)."""
        n_samples = 20
        seq_len = 30
        return {
            "num_windows": np.random.randn(n_samples, seq_len, 9),
            "cat_windows": np.random.rand(n_samples, seq_len, 8),
            "labels": np.array([[i, i * 1000] for i in range(n_samples)], dtype=object),
            "num_norm_params": {
                "mean": np.zeros(9),
                "std": np.ones(9),
            },
        }

    def test_four_horizon_feature_count(self, base_data) -> None:
        """4-horizon enrichment should produce 81 numerical features."""
        from scry.model.forecasting.enriched_pipeline import EnrichedFeaturePipeline

        pipeline = EnrichedFeaturePipeline(
            n_metrics=9,
            horizons=self.PRODUCTION_HORIZONS,
            forecaster=None,
        )

        result = pipeline.enrich(base_data)

        # 9 base + 9*4 residuals + 9*4 uncertainties = 9 + 36 + 36 = 81
        assert result["num_windows"].shape[2] == 81

    def test_four_horizon_enriched_num_features_property(self) -> None:
        """enriched_num_features should report 81 for 9 metrics x 4 horizons."""
        from scry.model.forecasting.enriched_pipeline import EnrichedFeaturePipeline

        pipeline = EnrichedFeaturePipeline(
            n_metrics=9,
            horizons=self.PRODUCTION_HORIZONS,
            forecaster=None,
        )

        assert pipeline.enriched_num_features == 81

    def test_four_horizon_preserves_samples_and_seqlen(self, base_data) -> None:
        """4-horizon enrichment should not alter sample count or sequence length."""
        from scry.model.forecasting.enriched_pipeline import EnrichedFeaturePipeline

        pipeline = EnrichedFeaturePipeline(
            n_metrics=9,
            horizons=self.PRODUCTION_HORIZONS,
            forecaster=None,
        )

        result = pipeline.enrich(base_data)

        assert result["num_windows"].shape[0] == base_data["num_windows"].shape[0]
        assert result["num_windows"].shape[1] == base_data["num_windows"].shape[1]

    def test_four_horizon_norm_params_extended(self, base_data) -> None:
        """Norm params should be extended to match 81 enriched features."""
        from scry.model.forecasting.enriched_pipeline import EnrichedFeaturePipeline

        pipeline = EnrichedFeaturePipeline(
            n_metrics=9,
            horizons=self.PRODUCTION_HORIZONS,
            forecaster=None,
        )

        result = pipeline.enrich(base_data)

        assert len(result["num_norm_params"]["mean"]) == 81
        assert len(result["num_norm_params"]["std"]) == 81
        # Original 9 means should be preserved
        np.testing.assert_array_equal(
            result["num_norm_params"]["mean"][:9],
            base_data["num_norm_params"]["mean"],
        )

    def test_default_horizons_are_production(self) -> None:
        """Default horizons should include all 4 production horizons."""
        from scry.model.forecasting.enriched_pipeline import EnrichedFeaturePipeline

        pipeline = EnrichedFeaturePipeline(n_metrics=9, forecaster=None)

        assert pipeline.horizons == [15, 60, 240, 1440]
