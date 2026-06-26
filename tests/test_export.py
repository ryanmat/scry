# Description: Tests for model export utilities.
# Description: Validates ONNX and TorchScript export functionality.

"""Tests for model export utilities."""

import json
import tempfile
from pathlib import Path

import pytest
import torch

from scry.model import TemporalXDEC
from scry.model.export import (
    InferenceWrapper,
    export_model,
    export_to_onnx,
    export_to_torchscript,
    get_model_metadata,
    load_torchscript,
)


@pytest.fixture
def model():
    """Create a test model."""
    return TemporalXDEC(
        num_numerical=9,
        num_categorical=8,
        seq_len=30,
        num_hidden=32,
        cat_hidden=16,
        latent_dim=8,
        n_clusters=5,
    )


@pytest.fixture
def sample_inputs(model):
    """Create sample inputs for testing."""
    batch_size = 4
    x_num = torch.randn(batch_size, model.seq_len, model.num_numerical)
    x_cat = torch.randn(batch_size, model.seq_len, model.num_categorical)
    return x_num, x_cat


class TestInferenceWrapper:
    """Tests for InferenceWrapper."""

    def test_forward_returns_tuple(self, model, sample_inputs):
        """Test wrapper returns cluster_ids and confidences."""
        wrapper = InferenceWrapper(model)
        x_num, x_cat = sample_inputs

        cluster_ids, confidences = wrapper(x_num, x_cat)

        assert cluster_ids.shape == (4,)
        assert confidences.shape == (4,)
        assert cluster_ids.dtype == torch.int64
        assert confidences.dtype == torch.float32

    def test_cluster_ids_in_valid_range(self, model, sample_inputs):
        """Test cluster IDs are within valid range."""
        wrapper = InferenceWrapper(model)
        x_num, x_cat = sample_inputs

        cluster_ids, _ = wrapper(x_num, x_cat)

        assert torch.all(cluster_ids >= 0)
        assert torch.all(cluster_ids < model.n_clusters)

    def test_confidences_in_valid_range(self, model, sample_inputs):
        """Test confidences are probabilities between 0 and 1."""
        wrapper = InferenceWrapper(model)
        x_num, x_cat = sample_inputs

        _, confidences = wrapper(x_num, x_cat)

        assert torch.all(confidences >= 0)
        assert torch.all(confidences <= 1)


class TestTorchScriptExport:
    """Tests for TorchScript export."""

    def test_export_creates_file(self, model):
        """Test TorchScript export creates a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.pt"

            result = export_to_torchscript(model, str(output_path))

            assert Path(result).exists()
            assert result == str(output_path)

    def test_exported_model_loads(self, model):
        """Test exported model can be loaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.pt"
            export_to_torchscript(model, str(output_path))

            loaded = load_torchscript(str(output_path))

            assert loaded is not None

    def test_exported_model_inference(self, model, sample_inputs):
        """Test exported model produces valid results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.pt"
            export_to_torchscript(model, str(output_path))

            loaded = load_torchscript(str(output_path))
            x_num, x_cat = sample_inputs

            # Get exported model output
            loaded.eval()
            with torch.no_grad():
                loaded_ids, loaded_conf = loaded(x_num, x_cat)

            # Validate output shapes and ranges
            assert loaded_ids.shape == (4,)
            assert loaded_conf.shape == (4,)
            assert torch.all(loaded_ids >= 0)
            assert torch.all(loaded_ids < model.n_clusters)
            assert torch.all(loaded_conf >= 0)
            assert torch.all(loaded_conf <= 1)

    @pytest.mark.skip(reason="TorchScript tracing doesn't support dict outputs")
    def test_export_without_wrapper(self, model, sample_inputs):
        """Test export without inference wrapper returns dict."""
        # TorchScript tracing requires tensor outputs, not dicts
        # Use use_inference_wrapper=True for production exports
        pass


class TestOnnxExport:
    """Tests for ONNX export.

    ONNX export is optional and needs the onnx/onnxscript packages, which the
    offline core does not ship. These tests skip cleanly when they are absent.
    """

    @pytest.fixture(autouse=True)
    def _require_onnx(self):
        pytest.importorskip("onnx")
        pytest.importorskip("onnxscript")

    def test_export_creates_file(self, model):
        """Test ONNX export creates a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.onnx"

            result = export_to_onnx(model, str(output_path))

            assert Path(result).exists()
            assert result == str(output_path)

    def test_export_with_dynamic_batch(self, model):
        """Test ONNX export with dynamic batch size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.onnx"

            result = export_to_onnx(
                model, str(output_path), dynamic_batch=True
            )

            assert Path(result).exists()

    def test_export_without_dynamic_batch(self, model):
        """Test ONNX export without dynamic batch size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.onnx"

            result = export_to_onnx(
                model, str(output_path), dynamic_batch=False
            )

            assert Path(result).exists()


class TestModelMetadata:
    """Tests for model metadata extraction."""

    def test_metadata_contains_required_fields(self, model):
        """Test metadata contains all required fields."""
        metadata = get_model_metadata(model)

        assert metadata["model_type"] == "TemporalXDEC"
        assert metadata["num_numerical"] == 9
        assert metadata["num_categorical"] == 8
        assert metadata["seq_len"] == 30
        assert metadata["latent_dim"] == 8
        assert metadata["n_clusters"] == 5

    def test_metadata_input_shapes(self, model):
        """Test metadata includes input shapes."""
        metadata = get_model_metadata(model)

        assert "input_shapes" in metadata
        assert "x_num" in metadata["input_shapes"]
        assert "x_cat" in metadata["input_shapes"]

    def test_metadata_cluster_names(self, model):
        """Test metadata includes cluster names."""
        metadata = get_model_metadata(model)

        assert "cluster_names" in metadata
        assert len(metadata["cluster_names"]) == 5
        assert "NORMAL" in metadata["cluster_names"]
        assert "PRE_FAILURE" in metadata["cluster_names"]


class TestExportModel:
    """Tests for combined export function."""

    def test_export_both_formats(self, model):
        """Test exporting to both formats."""
        pytest.importorskip("onnx")
        pytest.importorskip("onnxscript")
        with tempfile.TemporaryDirectory() as tmpdir:
            result = export_model(
                model,
                tmpdir,
                model_name="test_model",
                formats=("onnx", "torchscript"),
            )

            assert "onnx" in result
            assert "torchscript" in result
            assert "metadata" in result
            assert Path(result["onnx"]).exists()
            assert Path(result["torchscript"]).exists()
            assert Path(result["metadata"]).exists()

    def test_export_only_torchscript(self, model):
        """Test exporting only TorchScript."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = export_model(
                model,
                tmpdir,
                model_name="test_model",
                formats=("torchscript",),
            )

            assert "torchscript" in result
            assert "onnx" not in result

    def test_export_creates_metadata_file(self, model):
        """Test export creates valid metadata JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = export_model(
                model,
                tmpdir,
                model_name="test_model",
            )

            metadata_path = result["metadata"]
            with open(metadata_path) as f:
                metadata = json.load(f)

            assert metadata["model_type"] == "TemporalXDEC"
            assert metadata["n_clusters"] == 5

    def test_export_creates_directory(self, model):
        """Test export creates output directory if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_dir = Path(tmpdir) / "nested" / "output"

            result = export_model(
                model,
                str(nested_dir),
                model_name="test_model",
            )

            assert nested_dir.exists()
            assert Path(result["torchscript"]).exists()
