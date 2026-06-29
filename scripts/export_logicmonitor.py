# Description: CLI to export LogicMonitor device metrics to canonical Parquet/CSV.
# Description: Reads LMv1 credentials from env/.env; selects devices by id, group, or name.

"""Export LogicMonitor metrics to Scry's canonical schema.

Examples:
    # one device, specific datasources, last 14 days
    uv run python scripts/export_logicmonitor.py --device-id 274327 \
        --datasource-filter LogicMonitor_Collector_ --days 14 \
        --output data/captures/petclinic_collector.parquet

    # every member of a cluster device group, Kubernetes KSM only
    uv run python scripts/export_logicmonitor.py --group-id 2989 \
        --datasource-filter Kubernetes_KSM_ --days 7 \
        --output data/captures/aro_ksm.parquet
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

try:
    from scry.data.sources.lm_export import LMCredentials, LMRestClient, LogicMonitorExporter
except ModuleNotFoundError:  # running from a source checkout without an install
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from scry.data.sources.lm_export import LMCredentials, LMRestClient, LogicMonitorExporter


def load_dotenv(path: str) -> None:
    """Populate os.environ from a simple KEY=VALUE .env file (does not overwrite existing)."""
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export LogicMonitor metrics to canonical Parquet/CSV.")
    target = ap.add_argument_group("target selection (choose one)")
    target.add_argument("--device-id", type=int, action="append", dest="device_ids",
                        help="device id (repeatable)")
    target.add_argument("--group-id", type=int, help="capture all devices in this device group")
    target.add_argument("--name-filter", help="capture devices whose displayName contains this")

    ap.add_argument("--datasource-filter", action="append", dest="datasource_filters",
                    help="only datasources whose name contains this (repeatable)")
    ap.add_argument("--days", type=float, default=14.0, help="lookback window in days (default 14)")
    ap.add_argument("--start", type=int, help="window start, epoch seconds (overrides --days)")
    ap.add_argument("--end", type=int, help="window end, epoch seconds (default now)")
    ap.add_argument("--max-instances", type=int, dest="max_instances",
                    help="cap instances pulled per datasource")
    ap.add_argument("--rate-limit-pause", type=float, default=0.2,
                    help="seconds to pause between API calls (default 0.2)")
    ap.add_argument("--env-file", default=".env", help="path to the .env file (default ./.env)")
    ap.add_argument("--output", required=True, help="output .parquet or .csv path")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not (args.device_ids or args.group_id is not None or args.name_filter):
        print("error: choose a target with --device-id, --group-id, or --name-filter", file=sys.stderr)
        return 2

    load_dotenv(args.env_file)
    end = args.end or int(time.time())
    start = args.start if args.start is not None else int(end - args.days * 86400)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    creds = LMCredentials.from_env()
    with LMRestClient(creds, rate_limit_pause=args.rate_limit_pause) as client:
        exporter = LogicMonitorExporter(client)
        stats = exporter.export(
            output=args.output,
            start=start,
            end=end,
            ids=args.device_ids,
            group_id=args.group_id,
            name_filter=args.name_filter,
            datasource_filters=args.datasource_filters,
            max_instances_per_datasource=args.max_instances,
        )
    print("Export stats:", stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
