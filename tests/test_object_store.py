# Description: Tests for the DuckDB-backed object-store source.
# Description: Covers profile coverage and freshness/gap quality over Parquet/CSV.

"""Tests for scry.data.sources.object_store (coverage and quality)."""

from __future__ import annotations

from datetime import datetime, timezone
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


# --- quality -----------------------------------------------------------------


def _utc_now_naive() -> pd.Timestamp:
    """Wall-clock now as a tz-naive UTC timestamp (matches the source's storage)."""
    return pd.Timestamp.now(tz="UTC").tz_localize(None)


def _series_df(specs: list, end: pd.Timestamp) -> pd.DataFrame:
    """Build a canonical frame from (resource_id, metric_name, [secs_before_end]) specs."""
    rows = [
        {
            "resource_id": rid,
            "metric_name": metric,
            "timestamp": end - pd.Timedelta(seconds=off),
            "value": 1.0,
        }
        for rid, metric, offsets in specs
        for off in offsets
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def quality_parquet(tmp_path: Path) -> tuple[Path, pd.Timestamp]:
    """Four series ending 14 days ago: healthy, dead (ends early), gappy, single-point."""
    end = _utc_now_naive() - pd.Timedelta(days=14)
    specs = [
        ("r1", "A", [60 * i for i in range(61)]),  # healthy: 61 pts at 60s, last at end
        ("r2", "A", [7200 + 60 * i for i in range(61)]),  # dead: last 7200s (120 intervals) early
        ("r3", "A", [0] + [3000 + 60 * i for i in range(11)]),  # gappy: 12 pts over a 1h span
        ("r4", "A", [0]),  # single point: no inferable interval
    ]
    path = tmp_path / "quality.parquet"
    _write(path, _series_df(specs, end))
    return path, end


async def test_quality_scores_and_listings(quality_parquet: tuple[Path, pd.Timestamp]) -> None:
    """The verified 4-series fixture yields freshness 66.7, gap 73.2, with the right listings."""
    path, _ = quality_parquet
    result = await ObjectStoreSource(str(path)).get_quality()
    s = result["summary"]

    assert s["total_series"] == 4
    assert s["series_analyzed"] == 3  # single-point r4 excluded
    assert s["freshness_score"] == pytest.approx(66.7, abs=0.1)  # r2 dead -> 2/3 live
    assert s["gap_score"] == pytest.approx(73.2, abs=0.1)  # mean(1.0, 1.0, 0.197)

    stale = {(f["resource_id"], f["metric_name"]) for f in result["freshness"]}
    gappy = {(g["resource_id"], g["metric_name"]) for g in result["gaps"]}
    assert ("r2", "A") in stale
    assert ("r3", "A") in gappy
    assert ("r1", "A") not in stale  # healthy series is neither
    assert ("r1", "A") not in gappy


async def test_quality_static_capture_reads_fresh(tmp_path: Path) -> None:
    """A 14-day-old capture of healthy series reads fully fresh, with lag reported separately."""
    end = _utc_now_naive() - pd.Timedelta(days=14)
    specs = [("r1", "A", [60 * i for i in range(61)]), ("r2", "B", [60 * i for i in range(61)])]
    path = tmp_path / "fresh.parquet"
    _write(path, _series_df(specs, end))

    s = (await ObjectStoreSource(str(path)).get_quality())["summary"]
    assert s["freshness_score"] == 100.0
    assert s["lag_seconds"] == pytest.approx(14 * 86400, abs=3600)  # ~14 days behind now


async def test_quality_reference_now_marks_old_data_stale(
    quality_parquet: tuple[Path, pd.Timestamp]
) -> None:
    """Anchoring freshness to wall-clock now makes a 14-day-old capture read fully stale."""
    path, _ = quality_parquet
    ref = datetime.now(timezone.utc)
    result = await ObjectStoreSource(str(path)).get_quality(reference_time=ref)
    assert result["summary"]["freshness_score"] == 0.0
    assert len(result["freshness"]) >= 1


async def test_quality_lookback_window_excludes_old_data(
    quality_parquet: tuple[Path, pd.Timestamp]
) -> None:
    """A 24h lookback before now contains none of the 14-day-old data and fails closed."""
    path, _ = quality_parquet
    ref = datetime.now(timezone.utc)
    s = (await ObjectStoreSource(str(path)).get_quality(reference_time=ref, lookback_hours=24))[
        "summary"
    ]
    assert s["total_series"] == 0
    assert s["freshness_score"] == 0.0
    assert s["gap_score"] == 0.0


async def test_quality_constant_timestamp_excluded(tmp_path: Path) -> None:
    """A series with one repeated timestamp has no interval and is excluded from scoring."""
    end = _utc_now_naive()
    specs = [("r1", "A", [60 * i for i in range(5)]), ("rc", "A", [0, 0, 0, 0, 0])]
    path = tmp_path / "const.parquet"
    _write(path, _series_df(specs, end))
    s = (await ObjectStoreSource(str(path)).get_quality())["summary"]
    assert s["total_series"] == 2
    assert s["series_analyzed"] == 1


async def test_quality_profile_filter(features_yaml: Path, tmp_path: Path) -> None:
    """A profile restricts quality to that profile's metric names."""
    end = _utc_now_naive()
    specs = [
        ("r1", "A", [60 * i for i in range(5)]),
        ("r2", "B", [60 * i for i in range(5)]),
        ("r3", "OTHER", [60 * i for i in range(5)]),  # not in p_present {A, B, D}
    ]
    path = tmp_path / "pf.parquet"
    _write(path, _series_df(specs, end))

    full = await ObjectStoreSource(str(path)).get_quality()
    filtered = await ObjectStoreSource(str(path)).get_quality(profile="p_present")
    assert full["summary"]["total_series"] == 3
    assert filtered["summary"]["total_series"] == 2  # OTHER excluded


async def test_quality_empty_dataset_fails_closed(tmp_path: Path) -> None:
    """An empty store yields zeroed scores and a null reference, not an error."""
    path = tmp_path / "empty.parquet"
    _write(
        path,
        pd.DataFrame(
            {
                "resource_id": pd.Series([], dtype="object"),
                "metric_name": pd.Series([], dtype="object"),
                "timestamp": pd.Series([], dtype="datetime64[ns]"),
                "value": pd.Series([], dtype="float64"),
            }
        ),
    )
    result = await ObjectStoreSource(str(path)).get_quality()
    s = result["summary"]
    assert s["total_series"] == 0
    assert s["freshness_score"] == 0.0
    assert s["gap_score"] == 0.0
    assert s["reference_time"] is None
    assert s["lag_seconds"] is None
    assert result["freshness"] == []
    assert result["gaps"] == []


async def test_quality_resource_hash_layout(tmp_path: Path) -> None:
    """The sibling layout (resource_hash, value_double, hive columns) is read correctly."""
    end = _utc_now_naive()
    rows = [
        {
            "resource_hash": "h1",
            "scope_name": "s",
            "metric_name": "A",
            "timestamp": end - pd.Timedelta(seconds=60 * i),
            "value_double": 1.0,
            "year": 2026,
            "month": 6,
            "day": 15,
            "hour": 0,
        }
        for i in range(5)
    ]
    path = tmp_path / "hash.parquet"
    _write(path, pd.DataFrame(rows))
    s = (await ObjectStoreSource(str(path)).get_quality())["summary"]
    assert s["total_series"] == 1  # grouped by resource_hash; extra columns are harmless
    assert s["series_analyzed"] == 1


async def test_quality_null_timestamps_skipped(tmp_path: Path) -> None:
    """Rows with a null timestamp are filtered and do not corrupt the series."""
    end = _utc_now_naive()
    rows = [
        {"resource_id": "r1", "metric_name": "A", "timestamp": end - pd.Timedelta(seconds=60 * i), "value": 1.0}
        for i in range(5)
    ]
    rows += [{"resource_id": "r1", "metric_name": "A", "timestamp": pd.NaT, "value": 9.0} for _ in range(2)]
    path = tmp_path / "nulls.parquet"
    _write(path, pd.DataFrame(rows))
    s = (await ObjectStoreSource(str(path)).get_quality())["summary"]
    assert s["total_series"] == 1
    assert s["series_analyzed"] == 1


async def test_fetcher_check_data_quality_does_not_raise(
    quality_parquet: tuple[Path, pd.Timestamp]
) -> None:
    """DataFetcher.check_data_quality works over the object store (no NotImplementedError)."""
    path, _ = quality_parquet
    fetcher = DataFetcher.from_object_store(str(path))
    result = await fetcher.check_data_quality(profile=None)
    assert "summary" in result
    assert "warnings" in result
    assert "ready" in result
