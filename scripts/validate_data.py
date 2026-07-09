#!/usr/bin/env python3
# Description: Script to validate data pipeline connectivity and data availability.
# Description: Reports metrics, resources, coverage, and quality over an object store.

"""Validate data pipeline for Scry.

Usage:
    python scripts/validate_data.py --data "data/metrics/**/*.parquet" --profile collector
    SCRY_DATA_URI=s3://bucket/metrics/**/*.parquet python scripts/validate_data.py
"""

import argparse
import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone

from scry.data.fetcher import DataFetcher
from scry.data.quality import MIN_STEPS, format_monotone_warning, judge_metrics


def _data_uri(cli_value: str | None) -> str | None:
    """Resolve the object-store URI from the CLI flag, then ``SCRY_DATA_URI``."""
    return cli_value or os.environ.get("SCRY_DATA_URI")


async def _report(fetcher: DataFetcher, hours: int, verbose: bool, profile: str) -> None:
    """Print the data summary, coverage, metrics, resources, and a recent-data sample.

    Source-agnostic: works for any DataFetcher backend. Coverage degrades
    gracefully when the source does not expose it.
    """
    # Data summary
    print()
    print("Data Summary:")
    print("-" * 40)
    summary = await fetcher.get_data_summary()
    print(f"  Total data points: {summary['total_rows']:,}")
    print(f"  Unique resources:  {summary['unique_resources']}")
    print(f"  Unique metrics:    {summary['unique_metrics']}")

    if summary.get("earliest_timestamp"):
        print(f"  Earliest data:     {summary['earliest_timestamp']}")
        print(f"  Latest data:       {summary['latest_timestamp']}")

    # Profile coverage
    print()
    print("Profile Coverage:")
    print("-" * 40)
    try:
        coverage = await fetcher.check_profile_coverage()
        for p in coverage.get("profiles", []):
            marker = "[OK]" if p.get("coverage_percent", 0) >= 80 else "[!]"
            print(
                f"  {marker} {p['name']}: {p.get('coverage_percent', 0):.1f}% "
                f"({p.get('total_available', 0)}/{p.get('total_expected', 0)} features)"
            )
            if verbose and p.get("missing"):
                for m in p["missing"][:5]:
                    print(f"       missing: {m}")
                if len(p.get("missing", [])) > 5:
                    print(f"       ... and {len(p['missing']) - 5} more")
    except Exception as e:
        print(f"  [WARNING] Could not check coverage: {e}")

    # Data quality (freshness and gaps)
    print()
    print("Data Quality:")
    print("-" * 40)
    try:
        quality = await fetcher.check_data_quality(profile=profile)
        s = quality.get("summary", {})
        print(f"  Freshness score: {s.get('freshness_score', 0):.1f}%")
        print(f"  Gap score:       {s.get('gap_score', 0):.1f}%")
        lag = s.get("lag_seconds")
        if lag is not None:
            print(f"  Latest sample:   {lag / 3600:.1f} hours behind now")
        for w in quality.get("warnings", []):
            print(f"  [!] {w}")
        if verbose:
            for f in quality.get("freshness", [])[:5]:
                print(
                    f"       stale: {f['resource_id']}/{f['metric_name']} "
                    f"({f['intervals_behind']:.0f} intervals behind)"
                )
            for g in quality.get("gaps", [])[:5]:
                print(
                    f"       gappy: {g['resource_id']}/{g['metric_name']} "
                    f"({g['missing_pct']:.0f}% missing)"
                )
    except Exception as e:
        print(f"  [WARNING] Could not check quality: {e}")

    # Available metrics
    print()
    print("Available Metrics:")
    print("-" * 40)
    metric_names = await fetcher.get_metric_names()
    if metric_names:
        for name in metric_names[:20]:
            print(f"  - {name}")
        if len(metric_names) > 20:
            print(f"  ... and {len(metric_names) - 20} more")
    else:
        print("  [WARNING] No metrics found")

    # Available resources
    print()
    print("Available Resources:")
    print("-" * 40)
    resources = await fetcher.get_resource_list()
    if resources:
        for r in resources[:10]:
            host = r.get("host_name") or r.get("display_name") or r.get("resource_id") or "unknown"
            count = r.get("metric_count", 0)
            print(f"  - {host} ({count:,} metrics)")
        if len(resources) > 10:
            print(f"  ... and {len(resources) - 10} more")
    else:
        print("  [WARNING] No resources found")

    # Recent data for the profile
    print()
    print(f"Recent Data (last {hours} hours, profile={profile}):")
    print("-" * 40)
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(hours=hours)

    df = await fetcher.get_metrics_dataframe(start_time, end_time, profile=profile)
    print(f"  Rows fetched:      {len(df):,}")

    if len(df) > 0:
        unique_resources = df["resource_id"].nunique()
        unique_metrics = df["metric_name"].nunique()
        print(f"  Unique resources:  {unique_resources}")
        print(f"  Unique metrics:    {unique_metrics}")
        print(f"  Time range:        {df['timestamp'].min()} to {df['timestamp'].max()}")

        if verbose:
            print()
            print("Sample Data (first 5 rows):")
            print("-" * 40)
            sample = df.head(5)
            for _, row in sample.iterrows():
                host = str(row.get("host_name", ""))[:20]
                metric = str(row.get("metric_name", ""))[:25]
                value = row.get("value", 0)
                print(f"  {row['timestamp']} | {host:20} | {metric:25} | {value:.2f}")
    else:
        print(f"  [WARNING] No data in last {hours} hours for profile '{profile}'")

    # Feature sanity: cumulative counters drift under a fixed normalization.
    print()
    print("Feature Sanity:")
    print("-" * 40)
    judged = judge_metrics(df) if len(df) else []
    judged_series = sum(m["judged_series"] for m in judged)
    flagged = [m for m in judged if m["flagged"]]
    if len(df) == 0:
        print("  No data to judge.")
    elif judged_series == 0:
        print(
            f"  Nothing judged: no series exceeded {MIN_STEPS} samples in the "
            f"last {hours}h window. Increase --hours for a meaningful check."
        )
    elif flagged:
        for item in flagged:
            print(f"  [!] {format_monotone_warning(item)}")
    else:
        print(
            f"  {len(judged)} metrics judged ({judged_series} series); "
            "no cumulative-counter shapes flagged."
        )


async def main(
    hours: int,
    verbose: bool,
    profile: str,
    data_uri: str | None,
) -> int:
    """Validate data pipeline connectivity and data availability over the object store.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    print("=" * 60)
    print("Scry - Data Pipeline Validation")
    print("=" * 60)

    if not data_uri:
        print("[ERROR] No data source. Set --data or SCRY_DATA_URI to an object-store URI.")
        return 1

    try:
        print(f"Reading object store at {data_uri}...")
        fetcher = DataFetcher.from_object_store(data_uri)
        await _report(fetcher, hours, verbose, profile)
    except Exception as e:
        print(f"[ERROR] Pipeline validation failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    print()
    print("=" * 60)
    print("Validation complete")
    print("=" * 60)
    return 0


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate data pipeline connectivity and data availability."
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Object-store URI or local path/glob (Parquet or CSV). Default: SCRY_DATA_URI.",
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=1,
        help="Number of hours back to check for recent data (default: 1)",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="collector",
        help="Feature profile to validate (default: collector)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print sample data and missing features",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_uri = _data_uri(args.data)
    exit_code = asyncio.run(main(args.hours, args.verbose, args.profile, data_uri))
    sys.exit(exit_code)
