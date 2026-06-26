# Description: Data layer: sources, fetcher, feature engineering, and windowing pipeline.
# Description: Torch-free members import eagerly; torch-dependent and httpx members lazily.

"""Data layer for Scry.

Torch-free members (DataFetcher, DataSource, ObjectStoreSource) import eagerly.
Torch-dependent members (feature engineering, XDECFeaturePipeline) import lazily
via PEP 562 ``__getattr__`` so torch-free consumers can use this package without
pulling torch. The HttpIngest adapter also imports lazily so the ``logicmonitor``
extra (httpx) stays optional. Every name in ``__all__`` still resolves.
"""

from scry.data.fetcher import DataFetcher
from scry.data.sources.base import METRICS_COLUMNS, DataSource, normalize_record
from scry.data.sources.object_store import ObjectStoreSource

# name -> source module, resolved on first access by __getattr__ below.
_LAZY_EXPORTS = {
    "XDECFeaturePipeline": "scry.data.pipeline",
    "ALL_FEATURES": "scry.data.feature_engineering",
    "CATEGORICAL_FEATURES": "scry.data.feature_engineering",
    "NUMERICAL_FEATURES": "scry.data.feature_engineering",
    "XDECDataset": "scry.data.feature_engineering",
    "get_active_config": "scry.data.feature_engineering",
    "get_all_features": "scry.data.feature_engineering",
    "get_categorical_features": "scry.data.feature_engineering",
    "get_numerical_features": "scry.data.feature_engineering",
    "set_active_profile": "scry.data.feature_engineering",
    "HttpIngestClient": "scry.data.sources.http_ingest",
    "HttpDataSource": "scry.data.sources.http_ingest",
}


def __getattr__(name: str):
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_path), name)


def __dir__() -> list[str]:
    return sorted([*globals().keys(), *_LAZY_EXPORTS])


__all__ = [
    "DataFetcher",
    "DataSource",
    "ObjectStoreSource",
    "METRICS_COLUMNS",
    "normalize_record",
    "XDECFeaturePipeline",
    "XDECDataset",
    "NUMERICAL_FEATURES",
    "CATEGORICAL_FEATURES",
    "ALL_FEATURES",
    "get_active_config",
    "get_all_features",
    "get_categorical_features",
    "get_numerical_features",
    "set_active_profile",
    "HttpIngestClient",
    "HttpDataSource",
]
