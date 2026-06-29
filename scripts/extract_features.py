#!/usr/bin/env python3
# Description: CLI to extract windowed X-DEC training data from a data source.
# Description: Reads metrics from the object store and writes a .npz training file.

"""Extract training data for the X-DEC model.

Examples:
    # Local Parquet/CSV (the default object-store path)
    python scripts/extract_features.py --data "data/metrics/**/*.parquet" --profile kubernetes

    # Remote object storage (S3 / GCS / Azure) via a URI
    python scripts/extract_features.py --data "s3://bucket/metrics/**/*.parquet" --start 7d
"""

import argparse
import asyncio
import os
import re
import sys
from datetime import datetime, timedelta, timezone

from scry.data.feature_engineering import set_active_profile
from scry.data.fetcher import DataFetcher
from scry.data.pipeline import XDECFeaturePipeline
from scry.utils.config import get_config

_REL_RE = re.compile(r"^(\d+)([smhdw])$")
_REL_UNITS = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days", "w": "weeks"}


def _parse_time(value: str, now: datetime) -> datetime:
    """Parse 'now', a relative offset ('7d', '24h'), or an ISO timestamp."""
    if value == "now":
        return now
    match = _REL_RE.match(value)
    if match:
        amount, unit = int(match.group(1)), match.group(2)
        return now - timedelta(**{_REL_UNITS[unit]: amount})
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


async def _extract(args: argparse.Namespace) -> tuple[XDECFeaturePipeline, dict]:
    """Build the pipeline for the chosen source and return (pipeline, training_data)."""
    config = get_config()
    now = datetime.now(timezone.utc)
    start = _parse_time(args.start, now)
    end = _parse_time(args.end, now)

    # Activate the profile before transform so the persisted feature schema and
    # the windowed column order both reflect --profile for every source.
    if args.profile:
        set_active_profile(args.profile)

    fetcher = DataFetcher.from_object_store(args.data, data_format=args.format)
    pipeline = XDECFeaturePipeline(fetcher, config)
    raw = await pipeline.extract(start, end, profile=args.profile)
    return pipeline, pipeline.transform(raw)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract X-DEC training data from a data source."
    )
    src = parser.add_argument_group("data source")
    src.add_argument("--data", help="Object-store URI or local path/glob (Parquet or CSV).")
    src.add_argument(
        "--format", choices=["parquet", "csv"], help="Override the inferred file format."
    )
    parser.add_argument(
        "--start", default="7d", help="Start time: ISO, 'now', or relative like '7d'/'24h'."
    )
    parser.add_argument("--end", default="now", help="End time: ISO or 'now'.")
    parser.add_argument(
        "--profile", default=None, help="Feature profile (see config/features.yaml)."
    )
    parser.add_argument(
        "--output", default="data/training_data.npz", help="Output .npz path."
    )
    args = parser.parse_args()

    if not args.data:
        parser.error("provide --data <uri-or-path>")

    pipeline, data = asyncio.run(_extract(args))

    n_windows = int(data["num_windows"].shape[0])
    if n_windows == 0:
        print(
            "warning: no training windows produced; check the time range, profile, and source.",
            file=sys.stderr,
        )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    pipeline.save_training_data(data, args.output)
    print(f"wrote {n_windows} windows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
