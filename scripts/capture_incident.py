#!/usr/bin/env python3
# Description: Turnkey incident capture-to-validate wrapper: export the window, write labels, report lead time.
# Description: One command after an induced incident; --data validates an already-exported capture instead.

"""Capture a labeled incident window from LogicMonitor and validate it in one command.

Given an induced incident's onset and end, this wrapper exports the metric window
(healthy lead-in through post-incident tail) with the LogicMonitor exporter,
writes the labels sidecar the incident-validation harness expects, runs the
harness, and prints the detection lead time. The operator's hands-on part is
"induce + note timestamps"; everything downstream is this one command. With
``--data`` the export step is skipped and an already-exported capture is
validated instead, so a failed induction can be re-analyzed without re-pulling.

Onset semantics follow the harness: the label's ``start`` IS the onset, the
moment the operator-visible degradation begins. Record it before looking at any
reconstruction error.

Examples:
    # export + validate an induced incident on one device
    python scripts/capture_incident.py \\
        --onset 2026-07-15T14:00:00Z --incident-end 2026-07-15T14:45:00Z \\
        --resource-id aro-node-1 --type cpu_ramp \\
        --device-id 274327 --lead-in 6 \\
        --model models/aro_keeper_v1.pt --out-dir data/captures/incident_0715

    # re-validate an already-exported capture (no LogicMonitor access needed)
    python scripts/capture_incident.py \\
        --onset 2026-07-15T14:00:00Z --incident-end 2026-07-15T14:45:00Z \\
        --resource-id aro-node-1 --type cpu_ramp \\
        --data data/captures/incident_0715/capture.parquet \\
        --model models/aro_keeper_v1.pt --out-dir data/captures/incident_0715
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

import pandas as pd

try:
    import validate_incident
except ModuleNotFoundError:  # running from outside scripts/ without an install
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import validate_incident

from scry.data.feature_engineering import set_active_profile
from scry.data.sources.object_store import ObjectStoreSource

# The aro_node profile's datasources (docs/incident-capture-runbook.md); used when
# exporting and no --datasource-filter is given.
DEFAULT_DATASOURCE_FILTERS = ["Kubernetes_KSM_Nodes", "Kubernetes_KSM_NodeSummary"]


def parse_utc(value: str) -> pd.Timestamp:
    """Parse a UTC ISO8601 wall-clock string into an aware Timestamp (argparse type)."""
    try:
        ts = pd.to_datetime(value, utc=True)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError(f"not a UTC ISO8601 timestamp: {value!r}") from exc
    if pd.isna(ts):
        raise argparse.ArgumentTypeError(f"not a UTC ISO8601 timestamp: {value!r}")
    return ts


def export_window(
    onset: pd.Timestamp,
    incident_end: pd.Timestamp,
    lead_in_hours: float,
    tail_minutes: float,
) -> tuple[int, int]:
    """Bracket the export window: healthy lead-in before onset through a tail after the end.

    Returns (start, end) as epoch seconds, the units the LogicMonitor exporter takes.
    """
    start = onset - pd.Timedelta(hours=lead_in_hours)
    end = incident_end + pd.Timedelta(minutes=tail_minutes)
    return int(start.timestamp()), int(end.timestamp())


def write_labels(
    path: Path, resource_id: str, incident_type: str, onset: pd.Timestamp, incident_end: pd.Timestamp
) -> str:
    """Write the labels sidecar the harness expects: one incident, start = onset."""
    labels = [
        {
            "resource_id": resource_id,
            "type": incident_type,
            "start": onset.isoformat(),
            "end": incident_end.isoformat(),
        }
    ]
    path.write_text(json.dumps(labels, indent=2))
    return str(path)


def run_export(args: argparse.Namespace, start: int, end: int, output: str) -> dict:
    """Pull the window from LogicMonitor into ``output``; returns the exporter stats."""
    try:
        from scry.data.sources.lm_export import LMCredentials, LMRestClient, LogicMonitorExporter
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "the LogicMonitor exporter needs the 'logicmonitor' extra "
            "(pip install 'scryml[logicmonitor]'); or validate an existing capture with --data"
        ) from exc

    from export_logicmonitor import load_dotenv

    load_dotenv(args.env_file)
    creds = LMCredentials.from_env()
    with LMRestClient(creds, rate_limit_pause=args.rate_limit_pause) as client:
        exporter = LogicMonitorExporter(client)
        return exporter.export(
            output=output,
            start=start,
            end=end,
            ids=args.device_ids,
            group_id=args.group_id,
            name_filter=args.name_filter,
            datasource_filters=args.datasource_filters or DEFAULT_DATASOURCE_FILTERS,
            max_instances_per_datasource=args.max_instances,
        )


def preflight_capture(
    data: str, resource_id: str, numerical_features: list[str], data_format: str | None
) -> str | None:
    """Check the capture engages the analysis before running it; returns an error or None.

    A label whose resource_id matches nothing in the capture, or a capture whose
    numerical collection died with the incident, would otherwise flow through the
    harness and read as a clean NOT DETECTED.
    """
    source = ObjectStoreSource(data, data_format=data_format)
    resources = {r["resource_id"] for r in asyncio.run(source.fetch_resources())}
    if resource_id not in resources:
        sample = ", ".join(sorted(resources)[:5]) or "none"
        return (
            f"--resource-id {resource_id!r} matches no resource in the capture "
            f"(the exporter uses the device displayName); present: {sample}"
        )
    present = set(asyncio.run(source.fetch_metric_names()))
    if not present & set(numerical_features):
        return (
            "the capture contains none of the profile's numerical features; "
            "metric collection likely failed during the incident, so the "
            "reconstruction signal has nothing to score"
        )
    return None


def report(summary: dict, summary_path: Path) -> int:
    """Print the human-facing result, write the full summary JSON, and return the exit code.

    An incident whose resource produced no scored windows inside its span is an
    error (exit 1), not a NOT DETECTED: the analysis never engaged it.
    """
    fpr = summary["healthy_fpr"]
    print(
        f"threshold={summary['threshold']:.6f} source={summary['threshold_source']} "
        f"q={summary['threshold_quantile']} "
        f"healthy_fpr={'n/a' if fpr is None else f'{fpr:.4f}'} "
        f"(windows={summary['n_windows']}, fit={summary['n_threshold_windows']}, "
        f"eval={summary['n_eval_windows']})"
    )
    exit_code = 0
    for inc in summary["incidents"]:
        label = f"{inc['type']} on {inc['resource_id']}"
        if inc["detected"]:
            lead = inc["lead_time_seconds"]
            print(
                f"DETECTED {label}: lead_time_seconds={lead:.1f} ({abs(lead) / 60.0:.1f} min "
                f"{'before' if lead >= 0 else 'after'} onset), "
                f"first_detection={inc['first_detection_time']}"
            )
        elif inc["max_error_in_window"] is None:
            print(
                f"error: no scored windows for {label} inside the incident span; "
                "the capture may not cover the incident window",
                file=sys.stderr,
            )
            exit_code = 1
        else:
            print(f"NOT DETECTED {label}: no sustained anomaly within the look-back horizon")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"summary written to {summary_path}")
    return exit_code


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Export a labeled incident window from LogicMonitor and validate it."
    )
    incident = ap.add_argument_group("incident (timestamps are UTC ISO8601)")
    incident.add_argument("--onset", type=parse_utc, required=True,
                          help="when the operator-visible degradation began (the label start)")
    incident.add_argument("--incident-end", type=parse_utc, required=True,
                          help="when the incident ended or the induction stopped")
    incident.add_argument("--resource-id", required=True,
                          help="device displayName the incident ran on")
    incident.add_argument("--type", default="induced", help="incident type label (default induced)")

    window = ap.add_argument_group("export window")
    window.add_argument("--lead-in", type=float, default=6.0,
                        help="hours of healthy context before onset (default 6; hours-to-days, "
                             "the lead-in is what makes lead time measurable)")
    window.add_argument("--tail", type=float, default=30.0,
                        help="minutes captured after the incident end (default 30)")

    source = ap.add_argument_group("data source (an export target, or --data to skip the export)")
    source.add_argument("--data", help="already-exported capture path/URI; skips the export step")
    source.add_argument("--device-id", type=int, action="append", dest="device_ids",
                        help="LogicMonitor device id (repeatable)")
    source.add_argument("--group-id", type=int, help="capture all devices in this device group")
    source.add_argument("--name-filter", help="capture devices whose displayName contains this")
    source.add_argument("--datasource-filter", action="append", dest="datasource_filters",
                        help="only datasources whose name contains this (repeatable; "
                             f"default {' + '.join(DEFAULT_DATASOURCE_FILTERS)})")
    source.add_argument("--max-instances", type=int, dest="max_instances",
                        help="cap instances pulled per datasource")
    source.add_argument("--rate-limit-pause", type=float, default=0.2,
                        help="seconds to pause between API calls (default 0.2)")
    source.add_argument("--env-file", default=".env", help="path to the .env file (default ./.env)")

    validation = ap.add_argument_group("validation")
    validation.add_argument("--model", required=True, help="keeper checkpoint (.pt)")
    validation.add_argument("--profile", default="aro_node",
                            help="feature profile for the capture (default aro_node)")
    validation.add_argument("--threshold-quantile", type=float, default=0.99,
                            help="healthy-window quantile for the threshold (default 0.99)")
    validation.add_argument("--sustain", type=int, default=3,
                            help="consecutive anomalous windows required (default 3)")
    validation.add_argument("--max-leadtime", type=float, default=7200.0,
                            help="look-back horizon before onset, seconds (default 7200)")
    threshold_group = validation.add_mutually_exclusive_group()
    threshold_group.add_argument("--reference", help="healthy reference capture for the threshold")
    threshold_group.add_argument("--threshold", type=float,
                                 help="explicit anomaly threshold; skips the fit and any reference")
    validation.add_argument("--format", choices=["parquet", "csv"], dest="data_format",
                            help="explicit capture file format override")

    ap.add_argument("--out-dir", required=True,
                    help="directory for the capture, labels, and summary artifacts")
    args = ap.parse_args(argv)
    if args.threshold is not None and args.threshold <= 0:
        ap.error("--threshold must be a positive reconstruction-error value")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.onset >= args.incident_end:
        print("error: --onset must be before --incident-end", file=sys.stderr)
        return 2
    if args.lead_in <= 0 or args.tail < 0:
        print("error: --lead-in must be positive and --tail non-negative", file=sys.stderr)
        return 2
    has_target = bool(args.device_ids or args.group_id is not None or args.name_filter)
    if args.data and has_target:
        print("error: choose --data or an export target, not both", file=sys.stderr)
        return 2
    if not args.data and not has_target:
        print(
            "error: choose a target with --device-id, --group-id, or --name-filter, "
            "or validate an existing capture with --data",
            file=sys.stderr,
        )
        return 2
    if args.data_format and not args.data:
        print(
            "error: --format applies to reading --data captures; the export always "
            "writes Parquet",
            file=sys.stderr,
        )
        return 2
    if not Path(args.model).is_file():
        print(f"error: model checkpoint not found: {args.model}", file=sys.stderr)
        return 2
    try:
        feature_config = set_active_profile(args.profile)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.lead_in < 1.0:
        print(
            "warning: less than an hour of healthy lead-in rarely yields enough "
            "pre-onset windows for the threshold",
            file=sys.stderr,
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    start, end = export_window(args.onset, args.incident_end, args.lead_in, args.tail)

    if args.data:
        data = args.data
    else:
        data = str(out_dir / "capture.parquet")
        try:
            stats = run_export(args, start, end, data)
        except (RuntimeError, ValueError) as exc:  # missing extra or missing credentials
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print("Export stats:", stats)
        if stats["rows"] == 0:
            print(
                "error: the export returned no rows; check the device target, the "
                "datasource filters, and the time window",
                file=sys.stderr,
            )
            return 1

    problem = preflight_capture(
        data, args.resource_id, feature_config.numerical_features, args.data_format
    )
    if problem:
        print(f"error: {problem}", file=sys.stderr)
        return 1

    labels_path = write_labels(
        out_dir / "labels.json", args.resource_id, args.type, args.onset, args.incident_end
    )

    try:
        summary = validate_incident.analyze(
            args.model,
            data,
            labels_path,
            args.profile,
            threshold_quantile=args.threshold_quantile,
            sustain=args.sustain,
            max_leadtime_seconds=args.max_leadtime,
            reference=args.reference,
            threshold=args.threshold,
            data_format=args.data_format,
        )
    except ValueError as exc:  # no windows / no healthy windows / bad labels
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return report(summary, out_dir / "summary.json")


if __name__ == "__main__":
    raise SystemExit(main())
