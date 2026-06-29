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
                "categorical_normalization": {
                    "min": np.zeros(8),
                    "max": np.ones(8),
                },
                "feature_schema": {
                    "numerical": [
                        "cpuUsageNanoCores", "memoryUsageBytes", "networkRxBytes",
                        "networkTxBytes", "fsUsedBytes", "cpuLimits", "memoryLimits",
                        "memoryRequests", "memoryWorkingSetBytes",
                    ],
                    "categorical": [
                        "kubePodStatusReady", "podConditionPhase", "kubePodStatusScheduled",
                        "kubePodContainerStatusRunning", "kubePodContainerStatusWaiting",
                        "kubePodContainerStatusTerminated", "kubePodContainerStatusReady",
                        "status",
                    ],
                    "profile": "test",
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


def _make_checkpoint(num_names, cat_names, *, include_schema=True):
    """Build a serving-checkpoint dict for the 9-numerical/8-categorical model."""
    model = TemporalXDEC(
        num_numerical=9,
        num_categorical=8,
        seq_len=30,
        num_hidden=64,
        cat_hidden=32,
        latent_dim=8,
        n_clusters=5,
    )
    ckpt = {
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
        "normalization": {"mean": np.zeros(9), "std": np.ones(9)},
    }
    if include_schema:
        ckpt["categorical_normalization"] = {"min": np.zeros(8), "max": np.ones(8)}
        ckpt["feature_schema"] = {
            "numerical": num_names,
            "categorical": cat_names,
            "profile": "test",
        }
    return ckpt


class TestSchemaContract:
    """Tests for by-name alignment and the feature-schema load contract."""

    def test_preprocess_numerical_is_name_aligned(self, temp_model_path):
        """A shuffled metrics dict yields the same input tensor (the core fix)."""
        from scry.api.predictor import Predictor

        predictor = Predictor(model_path=str(temp_model_path))
        a = {"cpuUsageNanoCores": [1.0] * 30, "memoryUsageBytes": [2.0] * 30}
        b = {"memoryUsageBytes": [2.0] * 30, "cpuUsageNanoCores": [1.0] * 30}

        ta = predictor._preprocess_numerical(a)
        tb = predictor._preprocess_numerical(b)

        assert torch.equal(ta, tb)
        # cpuUsageNanoCores is schema column 0, memoryUsageBytes column 1.
        assert ta[0, 0, 0].item() == 1.0
        assert ta[0, 0, 1].item() == 2.0

    def test_embedding_is_order_invariant(self, temp_model_path):
        """End-to-end (deterministic mu): embedding is independent of key order."""
        from scry.api.predictor import Predictor

        predictor = Predictor(model_path=str(temp_model_path))
        cat = {"kubePodStatusReady": [1] * 30}
        num_a = {"cpuUsageNanoCores": [1.0] * 30, "memoryUsageBytes": [2.0] * 30}
        num_b = {"memoryUsageBytes": [2.0] * 30, "cpuUsageNanoCores": [1.0] * 30}

        e1 = predictor.get_embedding(num_a, cat)
        e2 = predictor.get_embedding(num_b, cat)

        np.testing.assert_array_almost_equal(e1, e2)

    def test_missing_numerical_maps_to_zero(self, temp_model_path):
        """Schema features absent from the input become normalized 0 (the mean)."""
        from scry.api.predictor import Predictor

        predictor = Predictor(model_path=str(temp_model_path))
        # Only the first schema feature is supplied.
        t = predictor._preprocess_numerical({"cpuUsageNanoCores": [5.0] * 30})

        assert t.shape == (1, 30, 9)
        assert t[0, 0, 0].item() == 5.0  # mean 0, std 1 -> passes through
        assert t[0, :, 1:].abs().sum().item() == 0.0  # all missing columns are 0

    def test_missing_numerical_uses_mean_not_raw_zero(self, tmp_path):
        """A missing feature maps to the mean (normalized 0), not -mean/std."""
        from scry.api.predictor import Predictor

        ckpt = _make_checkpoint(
            [f"num{i}" for i in range(9)], [f"cat{i}" for i in range(8)]
        )
        # Non-zero mean so raw-zero and mean-impute give different normalized values.
        ckpt["normalization"] = {"mean": np.full(9, 10.0), "std": np.full(9, 2.0)}
        path = tmp_path / "nonzero_mean.pt"
        torch.save(ckpt, path)

        predictor = Predictor(model_path=str(path))
        # Supply only num0 with a value != mean so the present-feature assertion is
        # self-contained; every other feature is missing.
        t = predictor._preprocess_numerical({"num0": [14.0] * 30})

        # num0 normalizes to (14-10)/2 == 2.0; missing features -> 0, not (0-10)/2 == -5.
        assert t[0, 0, 0].item() == pytest.approx(2.0)
        assert t[0, :, 1:].abs().sum().item() == pytest.approx(0.0)

    def test_missing_categorical_maps_to_zero(self, temp_model_path):
        """Absent categorical features stay at 0, the default state."""
        from scry.api.predictor import Predictor

        predictor = Predictor(model_path=str(temp_model_path))
        t = predictor._preprocess_categorical({"kubePodStatusReady": [1] * 30})

        assert t.shape == (1, 30, 8)
        assert t[0, 0, 0].item() == 1.0  # min 0, max 1 -> (1-0)/1 == 1
        assert t[0, :, 1:].abs().sum().item() == 0.0

    def test_unknown_metric_is_ignored(self, temp_model_path):
        """A metric name not in the schema does not change the input tensor."""
        from scry.api.predictor import Predictor

        predictor = Predictor(model_path=str(temp_model_path))
        base = predictor._preprocess_numerical({"cpuUsageNanoCores": [5.0] * 30})
        with_unknown = predictor._preprocess_numerical(
            {"cpuUsageNanoCores": [5.0] * 30, "totallyUnknownMetric": [9.0] * 30}
        )

        assert torch.equal(base, with_unknown)

    def test_schema_less_checkpoint_rejected(self, tmp_path):
        """A checkpoint without a feature_schema is refused at load."""
        from scry.api.predictor import ModelSchemaError, Predictor

        path = tmp_path / "old_model.pt"
        torch.save(_make_checkpoint(None, None, include_schema=False), path)

        with pytest.raises(ModelSchemaError, match="feature_schema"):
            Predictor(model_path=str(path))

    def test_schema_dim_mismatch_rejected(self, tmp_path):
        """A schema whose length disagrees with the model dims is refused."""
        from scry.api.predictor import ModelSchemaError, Predictor

        path = tmp_path / "bad_schema.pt"
        # Only 8 numerical names for a 9-numerical model.
        torch.save(
            _make_checkpoint(
                [f"num{i}" for i in range(8)], [f"cat{i}" for i in range(8)]
            ),
            path,
        )

        with pytest.raises(ModelSchemaError, match="does not match"):
            Predictor(model_path=str(path))

    def test_categorical_schema_dim_mismatch_rejected(self, tmp_path):
        """A categorical schema length that disagrees with the model dims is refused."""
        from scry.api.predictor import ModelSchemaError, Predictor

        path = tmp_path / "bad_cat_schema.pt"
        # Correct 9 numerical names but only 7 categorical for an 8-categorical model.
        torch.save(
            _make_checkpoint(
                [f"num{i}" for i in range(9)], [f"cat{i}" for i in range(7)]
            ),
            path,
        )

        with pytest.raises(ModelSchemaError, match="categorical"):
            Predictor(model_path=str(path))

    def test_preprocess_categorical_is_name_aligned(self, temp_model_path):
        """Categorical placement is by name too: shuffled keys -> same tensor."""
        from scry.api.predictor import Predictor

        predictor = Predictor(model_path=str(temp_model_path))
        a = {"kubePodStatusReady": [1] * 30, "podConditionPhase": [0] * 30}
        b = {"podConditionPhase": [0] * 30, "kubePodStatusReady": [1] * 30}

        ta = predictor._preprocess_categorical(a)
        tb = predictor._preprocess_categorical(b)

        assert torch.equal(ta, tb)
        # kubePodStatusReady is schema column 0, podConditionPhase column 1.
        assert ta[0, 0, 0].item() == 1.0
        assert ta[0, 0, 1].item() == 0.0

    def test_categorical_min_max_scaling_applied(self, tmp_path):
        """Persisted categorical min/max are applied (not identity), with fallback+clip."""
        from scry.api.predictor import Predictor

        ckpt = _make_checkpoint(
            [f"num{i}" for i in range(9)], [f"cat{i}" for i in range(8)]
        )
        cmin, cmax = np.zeros(8), np.ones(8)
        cmin[0], cmax[0] = 0.0, 10.0  # cat0: input 5 -> 0.5, input 15 -> clipped 1.0
        cmin[1], cmax[1] = 3.0, 3.0  # cat1: degenerate hi==lo -> fallback 1.0 (hi>0)
        ckpt["categorical_normalization"] = {"min": cmin, "max": cmax}
        path = tmp_path / "cat_scale.pt"
        torch.save(ckpt, path)

        predictor = Predictor(model_path=str(path))
        t = predictor._preprocess_categorical({"cat0": [5.0] * 30, "cat1": [99.0] * 30})
        assert t[0, 0, 0].item() == pytest.approx(0.5)  # (5-0)/(10-0)
        assert t[0, 0, 1].item() == pytest.approx(1.0)  # degenerate fallback, ignores input

        clipped = predictor._preprocess_categorical({"cat0": [15.0] * 30})
        assert clipped[0, 0, 0].item() == pytest.approx(1.0)  # clipped to [0, 1]

    def test_nan_values_are_handled_at_serve(self, temp_model_path):
        """NaN inputs are filled (numerical ffill/bfill, categorical nan->0), never propagated."""
        from scry.api.predictor import Predictor

        predictor = Predictor(model_path=str(temp_model_path))
        # Leading NaNs backward-fill to the first real value; trailing handled too.
        vals = [float("nan")] * 5 + [1.0] * 25
        tn = predictor._preprocess_numerical({"cpuUsageNanoCores": vals})
        assert not torch.isnan(tn).any()

        tc = predictor._preprocess_categorical({"kubePodStatusReady": [float("nan")] * 30})
        assert not torch.isnan(tc).any()
