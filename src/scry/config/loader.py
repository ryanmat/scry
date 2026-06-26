# Description: Configuration loader for feature profiles.
# Description: Loads domain-specific feature definitions from YAML config.

"""Configuration loader for feature profiles."""

import os
from pathlib import Path
from typing import Any

import yaml

# Default config path relative to project root
DEFAULT_CONFIG_PATH = "config/features.yaml"


class FeatureConfig:
    """Feature configuration for a domain profile.

    Attributes:
        profile_name: Name of the domain profile.
        description: Human-readable description of the profile.
        numerical_features: List of numerical feature names.
        categorical_features: List of categorical feature names.
        model_config: Optional model hyperparameter overrides.
    """

    def __init__(
        self,
        profile_name: str,
        description: str,
        numerical_features: list[str],
        categorical_features: list[str],
        model_config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize feature configuration.

        Args:
            profile_name: Name of the domain profile.
            description: Human-readable description.
            numerical_features: List of numerical feature names.
            categorical_features: List of categorical feature names.
            model_config: Optional model hyperparameter overrides.
        """
        self.profile_name = profile_name
        self.description = description
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.model_config = model_config or {}

    @property
    def num_numerical(self) -> int:
        """Return count of numerical features."""
        return len(self.numerical_features)

    @property
    def num_categorical(self) -> int:
        """Return count of categorical features."""
        return len(self.categorical_features)

    @property
    def all_features(self) -> list[str]:
        """Return combined list of all features."""
        return self.numerical_features + self.categorical_features


def find_config_path() -> Path:
    """Find the features.yaml config file.

    Searches in order:
    1. SCRY_CONFIG_PATH environment variable
    2. config/features.yaml relative to current working directory
    3. config/features.yaml relative to package location

    Returns:
        Path to the config file.

    Raises:
        FileNotFoundError: If config file cannot be found.
    """
    # Check environment variable first
    env_path = os.environ.get("SCRY_CONFIG_PATH")
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path
        raise FileNotFoundError(f"Config not found at SCRY_CONFIG_PATH: {env_path}")

    # Check relative to current working directory
    cwd_path = Path.cwd() / DEFAULT_CONFIG_PATH
    if cwd_path.exists():
        return cwd_path

    # Check relative to this file's location (src/scry/config/)
    package_path = Path(__file__).parent.parent.parent.parent / DEFAULT_CONFIG_PATH
    if package_path.exists():
        return package_path

    raise FileNotFoundError(
        f"Config file not found. Searched: {cwd_path}, {package_path}. "
        "Set SCRY_CONFIG_PATH environment variable to specify location."
    )


def load_config(config_path: str | None = None) -> dict[str, Any]:
    """Load the full configuration from YAML file.

    Args:
        config_path: Optional explicit path to config file.

    Returns:
        Parsed configuration dictionary.
    """
    if config_path:
        path = Path(config_path)
    else:
        path = find_config_path()

    with open(path) as f:
        return yaml.safe_load(f)


def get_profile(
    profile_name: str | None = None,
    config_path: str | None = None,
) -> FeatureConfig:
    """Get feature configuration for a domain profile.

    Args:
        profile_name: Name of profile to load. If None, uses default_profile.
        config_path: Optional explicit path to config file.

    Returns:
        FeatureConfig object for the requested profile.

    Raises:
        ValueError: If profile_name is not found in config.
    """
    config = load_config(config_path)

    # Use default profile if not specified
    if profile_name is None:
        profile_name = config.get("default_profile", "kubernetes")

    profiles = config.get("profiles", {})
    if profile_name not in profiles:
        available = list(profiles.keys())
        raise ValueError(
            f"Profile '{profile_name}' not found. Available: {available}"
        )

    profile_data = profiles[profile_name]

    # Get model config overrides if available
    model_configs = config.get("model_config", {})
    model_config = model_configs.get(profile_name, {})

    return FeatureConfig(
        profile_name=profile_name,
        description=profile_data.get("description", ""),
        numerical_features=profile_data.get("numerical_features", []),
        categorical_features=profile_data.get("categorical_features", []),
        model_config=model_config,
    )


def list_profiles(config_path: str | None = None) -> list[str]:
    """List available domain profiles.

    Args:
        config_path: Optional explicit path to config file.

    Returns:
        List of profile names.
    """
    config = load_config(config_path)
    return list(config.get("profiles", {}).keys())


def get_auto_discovery_settings(config_path: str | None = None) -> dict[str, Any]:
    """Get auto-discovery settings from config.

    Args:
        config_path: Optional explicit path to config file.

    Returns:
        Auto-discovery settings dictionary.
    """
    config = load_config(config_path)
    return config.get("auto_discovery", {
        "enabled": False,
        "min_numerical_features": 3,
        "min_categorical_features": 2,
        "max_numerical_features": 20,
        "max_categorical_features": 15,
    })
