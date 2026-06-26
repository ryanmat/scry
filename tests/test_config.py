# Description: Unit tests for configuration management module.
# Description: Tests config loading, environment variable overrides, and validation.

"""Tests for scry.utils.config module."""

import os
from unittest.mock import patch

import pytest


class TestConfig:
    """Tests for Config Pydantic Settings class."""

    def test_config_loads_with_defaults(self) -> None:
        """Config should load with default values for non-required fields."""
        from scry.utils.config import Config

        config = Config()

        assert config.model_path == "models/xdec_model.pt"
        assert config.num_clusters == 5
        assert config.sequence_length == 30
        assert config.numerical_hidden_dim == 64
        assert config.categorical_hidden_dim == 32
        assert config.latent_dim == 8
        assert config.batch_size == 64

    def test_config_has_default_httpingest_url(self) -> None:
        """Config should default httpingest_url to localhost."""
        from scry.utils.config import Config

        config = Config(_env_file=None)

        assert config.httpingest_url == "http://localhost:8000"

    def test_config_reads_httpingest_from_environment(self) -> None:
        """Config should read HttpIngest URL from environment."""
        from scry.utils.config import Config

        env = {
            "HTTPINGEST_URL": "https://ingest.example.com",
        }
        with patch.dict(os.environ, env, clear=False):
            config = Config()

        assert config.httpingest_url == "https://ingest.example.com"

    def test_config_environment_overrides_defaults(self) -> None:
        """Environment variables should override default values."""
        from scry.utils.config import Config

        env = {
            "num_clusters": "7",
            "batch_size": "128",
            "latent_dim": "16",
        }
        with patch.dict(os.environ, env, clear=False):
            config = Config()

        assert config.num_clusters == 7
        assert config.batch_size == 128
        assert config.latent_dim == 16

    def test_config_validates_positive_integers(self) -> None:
        """Config should validate that numeric fields are positive."""
        from pydantic import ValidationError

        from scry.utils.config import Config

        env = {
            "num_clusters": "-1",
        }
        with patch.dict(os.environ, env, clear=False):
            with pytest.raises(ValidationError):
                Config()

    def test_config_training_defaults(self) -> None:
        """Config should have correct training hyperparameter defaults."""
        from scry.utils.config import Config

        config = Config()

        assert config.pretrain_epochs == 500
        assert config.cluster_epochs == 300
        assert config.learning_rate == 1e-3
        assert config.beta == 1.0
        assert config.lambda_cluster == 0.1
        assert config.lambda_balance == 0.0
        assert config.cluster_lr == 1e-4
        assert config.window_step == 10


class TestGetConfig:
    """Tests for get_config singleton function."""

    def test_get_config_returns_config_instance(self) -> None:
        """get_config should return a Config instance."""
        from scry.utils.config import Config, get_config, reset_config

        reset_config()
        config = get_config()

        assert isinstance(config, Config)

    def test_get_config_returns_same_instance(self) -> None:
        """get_config should return the same singleton instance on repeated calls."""
        from scry.utils.config import get_config, reset_config

        reset_config()
        config1 = get_config()
        config2 = get_config()

        assert config1 is config2

    def test_reset_config_clears_singleton(self) -> None:
        """reset_config should clear the singleton, creating a fresh instance."""
        from scry.utils.config import get_config, reset_config

        reset_config()
        config1 = get_config()
        reset_config()
        config2 = get_config()

        assert config1 is not config2


class TestConfigGracefulDegradation:
    """Tests for graceful degradation when optional config is missing."""

    def test_config_works_without_env_file(self) -> None:
        """Config should load successfully even without .env file."""
        from scry.utils.config import Config

        config = Config(_env_file=None)
        assert config.httpingest_url == "http://localhost:8000"

    def test_config_no_azure_urls_in_defaults(self) -> None:
        """Default config should not contain any Azure-specific URLs."""
        from scry.utils.config import Config

        config = Config(_env_file=None)
        assert "azurecontainerapps" not in config.httpingest_url
        assert "localhost" in config.httpingest_url

    def test_config_ignores_unknown_env_vars(self) -> None:
        """Config should ignore unknown environment variables (extra='ignore')."""
        from scry.utils.config import Config

        env = {"UNKNOWN_SCRY_VAR": "should_be_ignored"}
        with patch.dict(os.environ, env, clear=False):
            config = Config()
        assert config.httpingest_url is not None

    def test_config_model_path_has_relative_default(self) -> None:
        """MODEL_PATH should default to a relative path, not an absolute one."""
        from scry.utils.config import Config

        config = Config(_env_file=None)
        assert config.model_path == "models/xdec_model.pt"
        assert not config.model_path.startswith("/")
