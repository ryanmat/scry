#!/usr/bin/env python3
# Description: CLI over scry.eval.suite.run_suite: run an eval suite and report its verdict.
# Description: Owns the 0/1/2 exit-code contract and the two report/summary output routings.

"""Run an evaluation suite and report its verdict.

A thin wrapper over ``run_suite``: it parses argv, runs every case of the
suite (or only the ``--case`` ones), writes the schema-validated report, and
prints a human summary of one line per case per required gate.

Exit codes:

    0   Every required gate of every case passed.
    1   At least one required gate failed. The report is still complete.
    2   The suite or rubric is invalid, or a named case does not exist. No
        scoring output is emitted, so a spec error can never be misread as
        a verdict.

Output routing (spec section 9.3) depends on ``--output``. With it, the
report JSON goes to the file and the summary to stdout. Without it, the
report JSON goes to stdout so it can be piped, and the summary goes to
stderr so it does not corrupt that JSON.

Example:
    python scripts/run_eval.py --suite suites/aro_node.yaml --output report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, TextIO

from scry.eval.rubric import SpecError
from scry.eval.suite import run_suite


def print_summary(report: dict[str, Any], stream: TextIO) -> None:
    """Print one PASS/FAIL line per case per required gate, then the verdict.

    Required gates only: an optional gate's result is in the report for the
    reader who wants it, but the summary is the pass/fail contract and a
    non-required gate cannot change it. Words only, no symbols, so the output
    stays greppable and terminal-safe.
    """
    for case in report["cases"]:
        for gate in case["rubric"]["gates"]:
            if not gate["required"]:
                continue
            verdict = "PASS" if gate["passed"] else "FAIL"
            print(f"{verdict}  {case['name']}  {gate['name']}", file=stream)
    print(f"VERDICT {report['verdict']}", file=stream)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run an evaluation suite and report its verdict.",
    )
    parser.add_argument("--suite", required=True, help="Path to the suite YAML.")
    parser.add_argument(
        "--output",
        help=(
            "Write the report JSON here and print the summary to stdout. "
            "Omit to write the report JSON to stdout and the summary to stderr."
        ),
    )
    parser.add_argument(
        "--case",
        action="append",
        metavar="NAME",
        help="Run only this case; repeatable. Omit to run every case in the suite.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)

    try:
        report = run_suite(args.suite, case_names=args.case or None)
    except SpecError as exc:
        print(f"spec error: {exc}", file=sys.stderr)
        return 2

    payload = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(payload + "\n")
        summary_stream = sys.stdout
    else:
        print(payload)
        summary_stream = sys.stderr
    print_summary(report, summary_stream)

    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
