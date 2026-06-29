# Description: Tests for the DuckDB-backed object-store source.
# Description: Covers profile coverage over Parquet/CSV, with an isolated features.yaml.

"""Tests for scry.data.sources.object_store (coverage)."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from scry.data.fetcher import DataFetcher
from scry.data.sources.object_store import ObjectStoreSource

# A self-contained profile config: a partially-covered, an empty, and a fully-absent profile.
_FEATURES_YAML = """
default_profile: p_present
profiles:
  p_present:
    description: "partially covered"
    numerical_features:
      - A
      - B
    categorical_features:
      - D
  p_empty:
    description: "no required features"
    numerical_features: []
    categorical_features: []
  p_absent:
    description: "nothing present"
    numerical_features:
      - X
      - "Y"
    categorical_features: []
"""


def _write(path: Path, df: pd.DataFrame) -> None:
    """Write a DataFrame to Parquet or CSV via DuckDB (no pyarrow dependency)."""
    con = duckdb.connect()
    try:
        con.register("rows", df)
        if str(path).lower().endswith(".csv"):
            con.execute(f"COPY rows TO '{path}' (FORMAT CSV, HEADER)")
        else:
            con.execute(f"COPY rows TO '{path}' (FORMAT PARQUET)")
    finally:
        con.close()


def _metrics_df(metric_names: list[str], resource_id: str = "r1") -> pd.DataFrame:
    """Build a minimal canonical frame with one row per metric name."""
    base = pd.Timestamp("2026-06-15T00:00:00")
    rows = [
        {
            "resource_id": resource_id,
            "metric_name": name,
            "timestamp": base + pd.Timedelta(minutes=i),
            "value": float(i),
        }
        for i, name in enumerate(metric_names)
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def features_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Write a small features.yaml and point SCRY_FEATURES_PATH at it."""
    path = tmp_path / "features.yaml"
    path.write_text(_FEATURES_YAML)
    monkeypatch.setenv("SCRY_FEATURES_PATH", str(path))
    return path


@pytest.fixture
def coverage_parquet(tmp_path: Path) -> Path:
    """A Parquet store whose metrics are {A, B, C, Z} (covers A,B of p_present; not D)."""
    path = tmp_path / "metrics.parquet"
    _write(path, _metrics_df(["A", "B", "C", "Z"]))
    return path


async def test_coverage_present_missing_percent(
    features_yaml: Path, coverage_parquet: Path
) -> None:
    """A partially-covered profile reports the right available/missing/percent."""
    source = ObjectStoreSource(str(coverage_parquet))
    result = await source.get_profile_coverage()
    by_name = {p["name"]: p for p in result["profiles"]}

    p = by_name["p_present"]
    assert p["available"] == ["A", "B"]
    assert p["missing"] == ["D"]
    assert p["total_expected"] == 3
    assert p["total_available"] == 2
    assert p["coverage_percent"] == pytest.approx(66.7)


async def test_coverage_empty_profile_is_full(
    features_yaml: Path, coverage_parquet: Path
) -> None:
    """A profile with no required features is vacuously fully covered."""
    source = ObjectStoreSource(str(coverage_parquet))
    by_name = {p["name"]: p for p in (await source.get_profile_coverage())["profiles"]}

    p = by_name["p_empty"]
    assert p["coverage_percent"] == 100.0
    assert p["available"] == []
    assert p["missing"] == []
    assert p["total_expected"] == 0


async def test_coverage_absent_profile_zero(
    features_yaml: Path, coverage_parquet: Path
) -> None:
    """A profile whose features are entirely absent reports 0% and lists them all missing."""
    source = ObjectStoreSource(str(coverage_parquet))
    by_name = {p["name"]: p for p in (await source.get_profile_coverage())["profiles"]}

    p = by_name["p_absent"]
    assert p["coverage_percent"] == 0.0
    assert p["available"] == []
    assert p["missing"] == ["X", "Y"]


async def test_coverage_extra_metric_not_in_available(
    features_yaml: Path, coverage_parquet: Path
) -> None:
    """Metrics present in data but not in a profile never appear in its available set."""
    source = ObjectStoreSource(str(coverage_parquet))
    for p in (await source.get_profile_coverage())["profiles"]:
        assert "C" not in p["available"]
        assert "Z" not in p["available"]


async def test_coverage_returns_all_profiles(
    features_yaml: Path, coverage_parquet: Path
) -> None:
    """Coverage reports every profile in the config, not just matched ones."""
    source = ObjectStoreSource(str(coverage_parquet))
    names = {p["name"] for p in (await source.get_profile_coverage())["profiles"]}
    assert names == {"p_present", "p_empty", "p_absent"}


async def test_coverage_csv(features_yaml: Path, tmp_path: Path) -> None:
    """Coverage works identically over a CSV store (read path is format-agnostic)."""
    path = tmp_path / "metrics.csv"
    _write(path, _metrics_df(["A", "B", "C", "Z"]))
    source = ObjectStoreSource(str(path))
    by_name = {p["name"]: p for p in (await source.get_profile_coverage())["profiles"]}
    assert by_name["p_present"]["coverage_percent"] == pytest.approx(66.7)


async def test_fetcher_check_profile_coverage_does_not_raise(
    features_yaml: Path, coverage_parquet: Path
) -> None:
    """DataFetcher.check_profile_coverage works over the object store (no NotImplementedError)."""
    fetcher = DataFetcher.from_object_store(str(coverage_parquet))
    result = await fetcher.check_profile_coverage()
    assert "profiles" in result
    assert "warnings" in result
    assert "ready" in result
    # p_absent is below the 80% threshold, so it must surface a warning.
    assert any("p_absent" in w for w in result["warnings"])
