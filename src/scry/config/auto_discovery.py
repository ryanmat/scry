# Description: Auto-discovery of metric features from available data sources.
# Description: Analyzes available metrics and classifies them as numerical or categorical.

import logging
from typing import Any

from scry.config.loader import FeatureConfig, get_auto_discovery_settings

logger = logging.getLogger(__name__)


def classify_metric_type(
    metric_name: str,
    sample_values: list[float],
) -> str:
    """Classify a metric as numerical or categorical based on its values.

    Heuristics:
    - If all values are 0 or 1 → categorical (binary)
    - If unique values <= 10 and all integers → categorical
    - Otherwise → numerical

    Args:
        metric_name: Name of the metric.
        sample_values: Sample of metric values.

    Returns:
        "numerical" or "categorical".
    """
    if not sample_values:
        return "numerical"

    unique_values = set(sample_values)
    num_unique = len(unique_values)

    # Binary metrics (0/1) are categorical
    if unique_values <= {0.0, 1.0, 0, 1}:
        return "categorical"

    # Small number of unique integer values suggests categorical
    all_integers = all(float(v).is_integer() for v in sample_values)
    if all_integers and num_unique <= 10:
        return "categorical"

    # High cardinality or floating point → numerical
    return "numerical"


def discover_features_from_metrics(
    metrics_data: dict[str, list[float]],
    config_path: str | None = None,
) -> FeatureConfig:
    """Discover feature configuration from available metrics.

    Args:
        metrics_data: Dict mapping metric names to sample values.
        config_path: Optional path to config file for settings.

    Returns:
        Dynamically created FeatureConfig.

    Raises:
        ValueError: If insufficient features discovered.
    """
    settings = get_auto_discovery_settings(config_path)

    min_numerical = settings.get("min_numerical_features", 3)
    min_categorical = settings.get("min_categorical_features", 2)
    max_numerical = settings.get("max_numerical_features", 20)
    max_categorical = settings.get("max_categorical_features", 15)

    numerical_features = []
    categorical_features = []

    for metric_name, values in metrics_data.items():
        metric_type = classify_metric_type(metric_name, values)

        if metric_type == "numerical":
            if len(numerical_features) < max_numerical:
                numerical_features.append(metric_name)
        else:
            if len(categorical_features) < max_categorical:
                categorical_features.append(metric_name)

    # Validate minimum requirements
    if len(numerical_features) < min_numerical:
        logger.warning(
            "Auto-discovery found only %d numerical features (minimum: %d)",
            len(numerical_features),
            min_numerical,
        )

    if len(categorical_features) < min_categorical:
        logger.warning(
            "Auto-discovery found only %d categorical features (minimum: %d)",
            len(categorical_features),
            min_categorical,
        )

    if not numerical_features or not categorical_features:
        raise ValueError(
            f"Insufficient features discovered: "
            f"{len(numerical_features)} numerical, {len(categorical_features)} categorical"
        )

    logger.info(
        "Auto-discovered %d numerical and %d categorical features",
        len(numerical_features),
        len(categorical_features),
    )

    return FeatureConfig(
        profile_name="auto_discovered",
        description="Automatically discovered from available metrics",
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        model_config={
            "seq_len": 30,
            "num_hidden": 64,
            "cat_hidden": 32,
            "latent_dim": 8,
        },
    )


async def discover_from_http(
    client: Any,
    sample_size: int = 1000,
) -> FeatureConfig:
    """Discover feature configuration from HttpIngest ML API.

    Fetches available metrics from the HttpIngest ML API and samples
    values to classify each metric type.

    Args:
        client: HttpIngestClient instance (connected).
        sample_size: Number of samples to use for classification.

    Returns:
        Dynamically created FeatureConfig.
    """
    # Get inventory of available metrics
    inventory = await client.get_inventory()
    metric_names = inventory.get("metrics", [])

    logger.info("Found %d distinct metrics via HttpIngest", len(metric_names))

    # Fetch a sample of training data to classify metrics
    sample_data = await client.get_training_data(limit=sample_size)

    # Group values by metric name
    metrics_data: dict[str, list[float]] = {}
    for record in sample_data:
        name = record.get("metric_name")
        value = record.get("value")
        if name and value is not None:
            try:
                float_val = float(value)
                if name not in metrics_data:
                    metrics_data[name] = []
                metrics_data[name].append(float_val)
            except (ValueError, TypeError):
                pass

    return discover_features_from_metrics(metrics_data)


def discover_from_dataframe(
    df: Any,
    exclude_columns: list[str] | None = None,
) -> FeatureConfig:
    """Discover feature configuration from a pandas DataFrame.

    Args:
        df: pandas DataFrame with metric columns.
        exclude_columns: Columns to exclude from discovery.

    Returns:
        Dynamically created FeatureConfig.
    """
    exclude = set(exclude_columns or ["resource_id", "timestamp"])

    metrics_data = {}
    for col in df.columns:
        if col in exclude:
            continue

        values = df[col].dropna().tolist()
        if values:
            metrics_data[col] = values

    return discover_features_from_metrics(metrics_data)
