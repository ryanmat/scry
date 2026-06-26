#!/usr/bin/env python3
# Description: Script to validate data pipeline connectivity and data availability.
# Description: Connects to HttpIngest ML API and reports on available metrics and resources.

"""Validate data pipeline for Scry.

Usage:
    python scripts/validate_data.py --hours 24 --verbose
    python scripts/validate_data.py --profile collector
"""

import argparse
import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone

from scry.data.fetcher import DataFetcher
from scry.data.sources.http_ingest import HttpIngestClient


async def main(hours: int, verbose: bool, profile: str, httpingest_url: str) -> int:
    """Validate data pipeline connectivity and data availability.

    Args:
        hours: Number of hours back to check for data.
        verbose: If True, print sample data.
        profile: Feature profile to validate.
        httpingest_url: HttpIngest API base URL.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    print("=" * 60)
    print("Scry - Data Pipeline Validation")
    print("=" * 60)
    print()

    # Connect to HttpIngest
    print(f"Connecting to HttpIngest at {httpingest_url}...")
    try:
        async with HttpIngestClient(base_url=httpingest_url) as client:
            health = await client.health_check()
            print(f"[OK] HttpIngest {health.get('version', 'unknown')} - {health.get('status', 'unknown')}")

            fetcher = DataFetcher.from_http_client(client)

            # Get data summary
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

            # Check profile coverage
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

            # Get metric names
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

            # Get resource list
            print()
            print("Available Resources:")
            print("-" * 40)
            resources = await fetcher.get_resource_list()
            if resources:
                for r in resources[:10]:
                    host = r.get("host_name") or r.get("display_name") or "unknown"
                    count = r.get("metric_count", 0)
                    print(f"  - {host} ({count:,} metrics)")
                if len(resources) > 10:
                    print(f"  ... and {len(resources) - 10} more")
            else:
                print("  [WARNING] No resources found")

            # Fetch recent data for profile
            print()
            print(f"Recent Data (last {hours} hours, profile={profile}):")
            print("-" * 40)
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=hours)

            df = await fetcher.get_metrics_dataframe(
                start_time, end_time, profile=profile
            )
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
                        print(
                            f"  {row['timestamp']} | {host:20} | "
                            f"{metric:25} | {value:.2f}"
                        )
            else:
                print(f"  [WARNING] No data in last {hours} hours for profile '{profile}'")

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
        "--hours",
        type=int,
        default=1,
        help="Number of hours back to check for data (default: 1)",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="collector",
        help="Feature profile to validate (default: collector)",
    )
    parser.add_argument(
        "--httpingest-url",
        type=str,
        default=None,
        help="HttpIngest API URL. Default: HTTPINGEST_URL env var or http://localhost:8000",
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
    httpingest_url = args.httpingest_url or os.environ.get(
        "HTTPINGEST_URL",
        "http://localhost:8000",
    )
    exit_code = asyncio.run(
        main(args.hours, args.verbose, args.profile, httpingest_url)
    )
    sys.exit(exit_code)
