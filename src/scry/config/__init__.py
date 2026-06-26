# Description: Configuration package for scry.
# Description: Exports feature profile loading and auto-discovery utilities.

"""Configuration package."""

from scry.config.auto_discovery import (
    classify_metric_type,
    discover_features_from_metrics,
    discover_from_dataframe,
    discover_from_http,
)
from scry.config.loader import (
    FeatureConfig,
    get_auto_discovery_settings,
    get_profile,
    list_profiles,
    load_config,
)

__all__ = [
    "FeatureConfig",
    "get_auto_discovery_settings",
    "get_profile",
    "list_profiles",
    "load_config",
    "classify_metric_type",
    "discover_features_from_metrics",
    "discover_from_dataframe",
    "discover_from_http",
]
