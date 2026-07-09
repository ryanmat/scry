# Description: Tests for the feature-sanity checks in scry.data.quality.
# Description: Covers counter shapes, gauges, constants, resets, duplicates, and instances.

"""Tests for cumulative-counter detection over canonical long-format metrics."""

import numpy as np
import pandas as pd

from scry.data.quality import (
    MIN_STEPS,
    find_monotone_features,
    judge_metrics,
    missing_features,
    monotone_feature_warnings,
)


def make_series(resource: str, metric: str, values, instance: str | None = None) -> pd.DataFrame:
    """Build a canonical long-format frame for one metric series."""
    timestamps = pd.date_range("2026-01-01", periods=len(values), freq="2min", tz="UTC")
    frame = pd.DataFrame(
        {
            "resource_id": resource,
            "metric_name": metric,
            "timestamp": timestamps,
            "value": values,
        }
    )
    if instance is not None:
        frame["datasource_instance"] = instance
    return frame


def counter(n: int = 200, start: float = 0.0) -> np.ndarray:
    """A strictly increasing since-boot-counter-shaped series."""
    return start + np.arange(n, dtype=float) * 120.0


def sinusoid(n: int = 200) -> np.ndarray:
    """A gauge-shaped series that rises and falls."""
    return 10.0 * np.sin(np.linspace(0.0, 20.0 * np.pi, n)) + 50.0


def flagged_names(df: pd.DataFrame, features=None) -> list[str]:
    return [f["metric_name"] for f in find_monotone_features(df, features)]


def test_counter_flagged_across_resources():
    df = pd.concat(
        [
            make_series("node-a", "networkRxBytes", counter()),
            make_series("node-b", "networkRxBytes", counter(start=5e9)),
        ]
    )
    results = find_monotone_features(df)
    assert len(results) == 1
    assert results[0]["metric_name"] == "networkRxBytes"
    assert results[0]["flagged_series"] == 2
    assert results[0]["judged_series"] == 2


def test_counter_with_reboot_reset_still_flagged():
    values = np.concatenate([counter(150), counter(60)])
    df = make_series("node-a", "nodeUptime", values)
    assert flagged_names(df) == ["nodeUptime"]


def test_counter_with_reset_near_window_end_still_flagged():
    values = np.concatenate([counter(200), counter(3)])
    df = make_series("node-a", "nodeUptime", values)
    assert flagged_names(df) == ["nodeUptime"]


def test_sparse_counter_still_flagged():
    rng = np.random.default_rng(11)
    increments = rng.random(200) < 0.1
    df = make_series("node-a", "memoryMajorPageFaults", np.cumsum(increments, dtype=float))
    assert flagged_names(df) == ["memoryMajorPageFaults"]


def test_single_step_change_not_flagged():
    values = np.concatenate([np.full(100, 4.0), np.full(100, 8.0)])
    df = make_series("node-a", "kubeNodeStatusCapacityCpu", values)
    assert flagged_names(df) == []


def test_duplicate_rows_do_not_hide_a_counter():
    df = make_series("node-a", "networkRxBytes", counter())
    duplicated = pd.concat([df, df]).reset_index(drop=True)
    assert flagged_names(duplicated) == ["networkRxBytes"]


def test_per_instance_counters_judged_separately():
    df = pd.concat(
        [
            make_series("node-a", "networkRxBytes", counter(start=1e6), instance="eth0"),
            make_series("node-a", "networkRxBytes", counter(start=5e4), instance="eth1"),
        ]
    )
    results = find_monotone_features(df)
    assert len(results) == 1
    assert results[0]["flagged_series"] == 2
    assert results[0]["judged_series"] == 2


def test_sinusoidal_gauge_not_flagged():
    df = make_series("node-a", "cpuUsagePercentage", sinusoid())
    assert flagged_names(df) == []


def test_constant_series_not_flagged():
    df = make_series("node-a", "kubeNodeStatusCapacityCpu", np.full(200, 4.0))
    assert flagged_names(df) == []


def test_wobbly_slow_growth_gauge_not_flagged():
    rng = np.random.default_rng(7)
    values = np.linspace(0.0, 50.0, 200) + rng.normal(0.0, 20.0, 200)
    df = make_series("node-a", "fsUsedBytes", values)
    assert flagged_names(df) == []


def test_short_series_reported_unjudged():
    df = make_series("node-a", "networkTxBytes", counter(MIN_STEPS))
    results = judge_metrics(df)
    assert len(results) == 1
    assert results[0]["judged_series"] == 0
    assert not results[0]["flagged"]
    assert flagged_names(df) == []


def test_mixed_resources_flag_at_half():
    df = pd.concat(
        [
            make_series("node-a", "memoryPageFaults", counter()),
            make_series("node-b", "memoryPageFaults", sinusoid()),
        ]
    )
    results = find_monotone_features(df)
    assert [r["metric_name"] for r in results] == ["memoryPageFaults"]
    assert results[0]["flagged_series"] == 1
    assert results[0]["judged_series"] == 2


def test_nan_values_dropped_before_judging():
    values = counter().astype(float)
    values[::9] = np.nan
    df = make_series("node-a", "cpuUsageCoreNanoSeconds", values)
    assert flagged_names(df) == ["cpuUsageCoreNanoSeconds"]


def test_features_filter_limits_evaluation():
    df = pd.concat(
        [
            make_series("node-a", "networkRxBytes", counter()),
            make_series("node-a", "networkTxBytes", counter()),
        ]
    )
    assert flagged_names(df, features=["networkTxBytes"]) == ["networkTxBytes"]


def test_empty_frame_returns_nothing():
    df = pd.DataFrame(columns=["resource_id", "metric_name", "timestamp", "value"])
    assert judge_metrics(df) == []
    assert find_monotone_features(df) == []
    assert monotone_feature_warnings(df) == []


def test_unsorted_timestamps_are_ordered_before_judging():
    df = make_series("node-a", "networkRxBytes", counter())
    shuffled = df.sample(frac=1.0, random_state=3).reset_index(drop=True)
    assert flagged_names(shuffled) == ["networkRxBytes"]


def test_warning_text_names_the_metric():
    df = make_series("node-a", "nodeUptime", counter())
    warnings = monotone_feature_warnings(df)
    assert len(warnings) == 1
    assert "nodeUptime" in warnings[0]
    assert "cumulative counter" in warnings[0]


def test_missing_features_lists_absent_metrics_in_order():
    df = make_series("node-a", "cpuUsageNanoCores", sinusoid())
    expected = ["nodeUptime", "cpuUsageNanoCores", "networkRxBytes"]
    assert missing_features(df, expected) == ["nodeUptime", "networkRxBytes"]


def test_missing_features_on_empty_frame_returns_all():
    df = pd.DataFrame(columns=["resource_id", "metric_name", "timestamp", "value"])
    assert missing_features(df, ["a", "b"]) == ["a", "b"]
