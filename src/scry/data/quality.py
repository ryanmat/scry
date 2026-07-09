# Description: Feature-sanity checks over canonical long-format metrics.
# Description: Flags cumulative-counter metrics that are invalid as model features.

"""Feature-sanity checks over canonical long-format metrics.

A reconstruction model normalizes features with statistics from a fixed
training window. A metric that increases monotonically for the lifetime of a
resource (a since-boot counter such as ``nodeUptime`` or ``networkRxBytes``)
drifts out of that normalized range as wall-clock time passes, so
reconstruction error on perfectly healthy data grows without bound. These
helpers detect that shape so it can be flagged before training or serving.

The check is advisory, not a gate, and has known blind spots: a counter that
resets more often than about once per hundred samples (a crash-looping pod)
reads non-monotone, and its bounded range also makes it far less of a drift
hazard than a long-lived counter.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# A series needs more than this many steps before it is judged.
MIN_STEPS = 50
# Fraction of non-decreasing steps at or above which a series reads monotone;
# high enough to require near-total monotonicity, low enough that occasional
# counter resets (a reboot zeroing a counter) do not hide the shape.
NONDECREASING_FRAC = 0.99
# Minimum fraction of strictly increasing steps; excludes constants and
# one-off step changes (a resized capacity figure) while catching sparse
# counters that tick up only occasionally.
STRICT_INCREASE_FRAC = 0.05


def judge_metrics(df: pd.DataFrame, features: list[str] | None = None) -> list[dict[str, Any]]:
    """Judge every metric's series for the cumulative-counter shape.

    Series are split per resource and, when the column is present, per
    ``datasource_instance``, so multi-instance metrics (per-NIC byte counters)
    are judged as separate series instead of one interleaved sequence.
    Duplicate timestamps within a series (overlapping exports) are dropped
    before the deltas are taken. A series is a counter hit when it is nearly
    non-decreasing (at least ``NONDECREASING_FRAC`` of steps) and strictly
    increases on at least ``STRICT_INCREASE_FRAC`` of steps; a late reset
    (a reboot near the end of the window) does not clear the shape.

    Args:
        df: Canonical long-format metrics with ``resource_id``,
            ``metric_name``, ``timestamp``, and ``value`` columns
            (``datasource_instance`` is honored when present).
        features: Metric names to evaluate; every metric in ``df`` when
            omitted.

    Returns:
        One dict per metric, ordered by metric name:
        ``{"metric_name", "flagged", "flagged_series", "judged_series"}``.
        A metric is flagged when at least half of its judged series hit.
    """
    if df.empty:
        return []
    frame = df if features is None else df[df["metric_name"].isin(features)]
    if frame.empty:
        return []

    series_keys = ["resource_id"]
    if "datasource_instance" in frame.columns:
        series_keys.append("datasource_instance")

    results: list[dict[str, Any]] = []
    for metric, metric_frame in frame.groupby("metric_name", sort=True):
        judged = 0
        hits = 0
        for _, series_frame in metric_frame.groupby(series_keys, dropna=False):
            values = (
                series_frame.sort_values("timestamp")
                .drop_duplicates(subset="timestamp")["value"]
                .astype(float)
                .dropna()
                .to_numpy()
            )
            if values.size <= MIN_STEPS:
                continue
            judged += 1
            deltas = np.diff(values)
            nondecreasing = float(np.mean(deltas >= 0))
            increasing = float(np.mean(deltas > 0))
            if nondecreasing >= NONDECREASING_FRAC and increasing >= STRICT_INCREASE_FRAC:
                hits += 1
        results.append(
            {
                "metric_name": str(metric),
                "flagged": judged > 0 and hits * 2 >= judged,
                "flagged_series": hits,
                "judged_series": judged,
            }
        )
    return results


def find_monotone_features(
    df: pd.DataFrame, features: list[str] | None = None
) -> list[dict[str, Any]]:
    """Return only the metrics :func:`judge_metrics` flags as counters.

    Args:
        df: Canonical long-format metrics.
        features: Metric names to evaluate; every metric in ``df`` when
            omitted.

    Returns:
        The flagged subset of :func:`judge_metrics`, ordered by metric name.
    """
    return [item for item in judge_metrics(df, features) if item["flagged"]]


def format_monotone_warning(item: dict[str, Any]) -> str:
    """Render one flagged metric from :func:`judge_metrics` as a warning."""
    return (
        f"'{item['metric_name']}' looks like a cumulative counter "
        f"(monotonically increasing on {item['flagged_series']}/"
        f"{item['judged_series']} series); counters drift out of a "
        f"fixed normalization range and should not be model features"
    )


def monotone_feature_warnings(df: pd.DataFrame, features: list[str] | None = None) -> list[str]:
    """Render :func:`find_monotone_features` results as warning strings.

    Args:
        df: Canonical long-format metrics.
        features: Metric names to evaluate; every metric in ``df`` when
            omitted.

    Returns:
        One human-readable warning per flagged metric.
    """
    return [format_monotone_warning(item) for item in find_monotone_features(df, features)]


def missing_features(df: pd.DataFrame, features: list[str]) -> list[str]:
    """List the features entirely absent from a long-format capture.

    A model checkpoint carries the feature names it was trained on; when the
    live profile no longer includes some of them, the fetch silently strips
    those metrics and windowing fills them with neutral values, so scores are
    no longer comparable to the checkpoint's calibration. Callers warn on a
    non-empty result.

    Args:
        df: Canonical long-format metrics.
        features: Feature names the model expects.

    Returns:
        The expected features with no rows in ``df``, in the given order.
    """
    present = set(df["metric_name"].unique()) if not df.empty else set()
    return [f for f in features if f not in present]
