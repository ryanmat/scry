# Description: Tests for the prediction service.
# Description: Tests model loading, preprocessing, and inference.

"""Tests for the prediction service."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from scry.model import TemporalXDEC


@pytest.fixture
def temp_model_path():
    """Create a temporary model file for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_model.pt"

        # Create and save a model
        model = TemporalXDEC(
            num_numerical=9,
            num_categorical=8,
            seq_len=30,
            num_hidden=64,
            cat_hidden=32,
            latent_dim=8,
            n_clusters=5,
        )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": {
                    "num_numerical": 9,
                    "num_categorical": 8,
                    "seq_len": 30,
                    "num_hidden": 64,
                    "cat_hidden": 32,
                    "latent_dim": 8,
                    "n_clusters": 5,
                },
                "normalization": {
                    "mean": np.zeros(9),
                    "std": np.ones(9),
                },
            },
            model_path,
        )

        yield model_path


class TestPredictorInit:
    """Tests for Predictor initialization."""

    def test_load_model(self, temp_model_path):
        """Test loading a model from disk."""
        from scry.api.predictor import Predictor

        predictor = Predictor(model_path=str(temp_model_path))

        assert predictor.model is not None
        assert predictor.is_loaded

    def test_load_model_not_found(self):
        """Test loading nonexistent model raises error."""
        from scry.api.predictor import Predictor

        with pytest.raises(FileNotFoundError):
            Predictor(model_path="/nonexistent/model.pt")

    def test_model_in_eval_mode(self, temp_model_path):
        """Test model is set to eval mode after loading."""
        from scry.api.predictor import Predictor

        predictor = Predictor(model_path=str(temp_model_path))

        assert not predictor.model.training

    def test_device_detection(self, temp_model_path):
        """Test device is properly detected."""
        from scry.api.predictor import Predictor

        predictor = Predictor(model_path=str(temp_model_path))

        # Should be cpu, cuda, or mps
        assert predictor.device in ["cpu", "cuda", "mps"]


class TestPreprocessing:
    """Tests for metric preprocessing."""

    def test_preprocess_numerical_metrics(self, temp_model_path):
        """Test preprocessing numerical metrics."""
        from scry.api.predictor import Predictor

        predictor = Predictor(model_path=str(temp_model_path))

        numerical = {
            "cpuUsageNanoCores": [1000000.0] * 30,
            "memoryUsageBytes": [50000000.0] * 30,
        }

        result = predictor._preprocess_numerical(numerical)

        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 1  # batch size
        assert result.shape[1] == 30  # sequence length
        assert result.shape[2] == 9  # num features (padded)

    def test_preprocess_categorical_metrics(self, temp_model_path):
        """Test preprocessing categorical metrics."""
        from scry.api.predictor import Predictor

        predictor = Predictor(model_path=str(temp_model_path))

        categorical = {
            "kubePodStatusReady": [1] * 30,
            "podConditionPhase": [1] * 30,
        }

        result = predictor._preprocess_categorical(categorical)

        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 1  # batch size
        assert result.shape[1] == 30  # sequence length
        assert result.shape[2] == 8  # num features (padded)

    def test_pad_short_sequence(self, temp_model_path):
        """Test padding short sequences to seq_len."""
        from scry.api.predictor import Predictor

        predictor = Predictor(model_path=str(temp_model_path))

        # Only 10 timesteps, should be padded to 30
        numerical = {"cpuUsageNanoCores": [1000000.0] * 10}

        result = predictor._preprocess_numerical(numerical)

        assert result.shape[1] == 30  # padded to seq_len

    def test_truncate_long_sequence(self, temp_model_path):
        """Test truncating long sequences to seq_len."""
        from scry.api.predictor import Predictor

        predictor = Predictor(model_path=str(temp_model_path))

        # 50 timesteps, should be truncated to 30 (most recent)
        numerical = {"cpuUsageNanoCores": list(range(50))}

        result = predictor._preprocess_numerical(numerical)

        assert result.shape[1] == 30  # truncated to seq_len


class TestPrediction:
    """Tests for prediction functionality."""

    def test_predict_returns_dict(self, temp_model_path):
        """Test predict returns expected keys."""
        from scry.api.predictor import Predictor

        predictor = Predictor(model_path=str(temp_model_path))

        numerical = {"cpuUsageNanoCores": [1000000.0] * 30}
        categorical = {"kubePodStatusReady": [1] * 30}

        result = predictor.predict(numerical, categorical)

        assert "cluster_id" in result
        assert "cluster_name" in result
        assert "confidence" in result
        assert "action" in result
        assert "priority" in result

    def test_predict_cluster_id_range(self, temp_model_path):
        """Test cluster_id is in valid range."""
        from scry.api.predictor import Predictor

        predictor = Predictor(model_path=str(temp_model_path))

        numerical = {"cpuUsageNanoCores": [1000000.0] * 30}
        categorical = {"kubePodStatusReady": [1] * 30}

        result = predictor.predict(numerical, categorical)

        assert 0 <= result["cluster_id"] <= 4

    def test_predict_confidence_range(self, temp_model_path):
        """Test confidence is in valid range."""
        from scry.api.predictor import Predictor

        predictor = Predictor(model_path=str(temp_model_path))

        numerical = {"cpuUsageNanoCores": [1000000.0] * 30}
        categorical = {"kubePodStatusReady": [1] * 30}

        result = predictor.predict(numerical, categorical)

        assert 0.0 <= result["confidence"] <= 1.0

    def test_predict_valid_cluster_name(self, temp_model_path):
        """Test cluster_name is valid."""
        from scry.api.predictor import Predictor

        predictor = Predictor(model_path=str(temp_model_path))

        numerical = {"cpuUsageNanoCores": [1000000.0] * 30}
        categorical = {"kubePodStatusReady": [1] * 30}

        result = predictor.predict(numerical, categorical)

        valid_names = ["NORMAL", "PRE_SCALE", "PRE_FAILURE", "ACTIVE_DEGRADATION", "ANOMALY"]
        assert result["cluster_name"] in valid_names

    def test_predict_valid_action(self, temp_model_path):
        """Test action is valid."""
        from scry.api.predictor import Predictor

        predictor = Predictor(model_path=str(temp_model_path))

        numerical = {"cpuUsageNanoCores": [1000000.0] * 30}
        categorical = {"kubePodStatusReady": [1] * 30}

        result = predictor.predict(numerical, categorical)

        valid_actions = ["NONE", "SCALE", "DIAGNOSTIC", "REMEDIATE", "ALERT"]
        assert result["action"] in valid_actions

    def test_predict_valid_priority(self, temp_model_path):
        """Test priority is valid."""
        from scry.api.predictor import Predictor

        predictor = Predictor(model_path=str(temp_model_path))

        numerical = {"cpuUsageNanoCores": [1000000.0] * 30}
        categorical = {"kubePodStatusReady": [1] * 30}

        result = predictor.predict(numerical, categorical)

        valid_priorities = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        assert result["priority"] in valid_priorities

    def test_predict_no_gradient(self, temp_model_path):
        """Test prediction doesn't require gradients."""
        from scry.api.predictor import Predictor

        predictor = Predictor(model_path=str(temp_model_path))

        numerical = {"cpuUsageNanoCores": [1000000.0] * 30}
        categorical = {"kubePodStatusReady": [1] * 30}

        # Should not raise any gradient-related errors
        result = predictor.predict(numerical, categorical)

        assert result is not None


class TestGetEmbedding:
    """Tests for getting latent embeddings."""

    def test_get_embedding_shape(self, temp_model_path):
        """Test embedding has correct shape."""
        from scry.api.predictor import Predictor

        predictor = Predictor(model_path=str(temp_model_path))

        numerical = {"cpuUsageNanoCores": [1000000.0] * 30}
        categorical = {"kubePodStatusReady": [1] * 30}

        embedding = predictor.get_embedding(numerical, categorical)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (8,)  # latent_dim

    def test_get_embedding_deterministic(self, temp_model_path):
        """Test embedding is deterministic in eval mode."""
        from scry.api.predictor import Predictor

        predictor = Predictor(model_path=str(temp_model_path))

        numerical = {"cpuUsageNanoCores": [1000000.0] * 30}
        categorical = {"kubePodStatusReady": [1] * 30}

        emb1 = predictor.get_embedding(numerical, categorical)
        emb2 = predictor.get_embedding(numerical, categorical)

        np.testing.assert_array_almost_equal(emb1, emb2)
