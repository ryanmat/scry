# Description: Tests for scripts/run_eval.py, the CLI over scry.eval.suite.run_suite.
# Description: Pins the 0/1/2 exit-code contract, both output-routing modes, and the case filter.

"""Subprocess tests for the run_eval CLI.

The CLI is a thin wrapper over ``run_suite``, so these tests pin the contract
the library does not own: the exit code (0 when every required gate of every
case passes, 1 on a rubric failure, 2 on a spec error), where the report JSON
and the human summary each go in both output-routing modes, that a spec error
emits no partial scoring output that could be read as a verdict, and that
``--case`` filters. Every run is a real subprocess against the tiny keeper so
the argv surface and the exit status are exercised as an operator sees them.
The summary is pinned against the report it accompanies rather than a
hardcoded gate list, so the two cannot drift apart.
"""

from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from synth import RUN_SUITE_RUBRIC, gen_capture, write_csv, write_run_suite_tree

from scry.eval.labels import LabelCase, LabelSet, dump_labels
from scry.eval.rubric import SpecError
from scry.eval.suite import validate_report

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "run_eval.py"

# Every report field that carries a per-sustain accounting, by nesting level.
GRID_SUSTAIN_FIELDS = (
    "pooled_lead_in_fpr",
    "fleet_time_in_alarm_fraction",
    "fleet_raises_per_week",
    "fleet_runs_per_week",
)
RESOURCE_SUSTAIN_FIELDS = (
    "lead_in_fpr",
    "time_in_alarm_fraction",
    "raises_per_week",
    "runs_per_week",
    "sustained_run_counts",
)


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=REPO,
    )


def _gate_lines(summary: str) -> set[tuple[str, str, str]]:
    """Parse the ``VERDICT  case  gate`` summary lines into a comparable set."""
    parsed = set()
    for line in summary.splitlines():
        fields = line.split()
        if len(fields) == 3 and fields[0] in {"PASS", "FAIL"}:
            parsed.add((fields[0], fields[1], fields[2]))
    return parsed


def _expected_gate_lines(report: dict) -> set[tuple[str, str, str]]:
    """The lines the summary must carry: one per case per required gate."""
    return {
        ("PASS" if gate["passed"] else "FAIL", case["name"], gate["name"])
        for case in report["cases"]
        for gate in case["rubric"]["gates"]
        if gate["required"]
    }


def _tree_with(
    tmp_path: Path,
    keeper_path: str,
    *,
    keep_cases: set[str] | None = None,
    gates: dict | None = None,
    controls_only_case: bool = False,
) -> str:
    """Build an isolated suite tree, then edit its suite and rubric in place.

    The edits are the axes step 7.2 needs: which cases the suite declares
    (to make a required gate have no applicable case anywhere), a gate
    config override (to drop allow_absent), and an extra controls-only
    case, which is the only shape that produces the negative_control case
    kind -- labels carrying controls but no incident.
    """
    suite_path = Path(write_run_suite_tree(tmp_path, keeper_path))
    suite_dir = suite_path.parent
    suite = yaml.safe_load(suite_path.read_text())
    rubric = yaml.safe_load((suite_dir / "rubric.yaml").read_text())

    if keep_cases is not None:
        suite["cases"] = [case for case in suite["cases"] if case["name"] in keep_cases]
    if gates:
        rubric["gates"].update(gates)
    if controls_only_case:
        controls_df, _ = gen_capture("node-ctl2", 700, seed=104)
        write_csv(controls_df, suite_dir / "data" / "controls.csv")
        dump_labels(
            LabelSet(
                version=2,
                capture=None,
                cases=[
                    LabelCase(
                        resource_id="node-ctl2",
                        role="negative_control",
                        type=None,
                        onsets={},
                        primary_onset=None,
                        end=None,
                        notes=None,
                    )
                ],
            ),
            str(suite_dir / "data" / "controls_labels.json"),
        )
        suite["cases"].append(
            {
                "name": "controls_only",
                "kind": "incident_capture",
                "capture": "data/controls.csv",
                "labels": "data/controls_labels.json",
                "format": "csv",
            }
        )

    (suite_dir / "rubric.yaml").write_text(yaml.safe_dump(rubric))
    suite_path.write_text(yaml.safe_dump(suite))
    return str(suite_path)


@pytest.fixture(scope="module")
def suite_path(keeper_path: str, tmp_path_factory: pytest.TempPathFactory) -> str:
    return write_run_suite_tree(tmp_path_factory.mktemp("run_eval"), keeper_path)


@pytest.fixture(scope="module")
def passing_run(suite_path: str, tmp_path_factory: pytest.TempPathFactory) -> tuple:
    """The ramp case alone: every required gate passes, so exit 0."""
    out = tmp_path_factory.mktemp("pass") / "report.json"
    proc = _run("--suite", suite_path, "--case", "ramp_incident", "--output", str(out))
    return proc, out


@pytest.fixture(scope="module")
def failing_run(suite_path: str, tmp_path_factory: pytest.TempPathFactory) -> tuple:
    """Both cases: the pinned alarm fails alarm_fatigue, so exit 1."""
    out = tmp_path_factory.mktemp("fail") / "report.json"
    proc = _run("--suite", suite_path, "--output", str(out))
    return proc, out


class TestExitCodes:
    def test_exit_zero_when_every_required_gate_passes(self, passing_run: tuple) -> None:
        proc, out = passing_run
        assert proc.returncode == 0, proc.stderr
        report = json.loads(out.read_text())
        assert report["verdict"] == "PASS"
        assert report["exit_code"] == 0
        assert [case["name"] for case in report["cases"]] == ["ramp_incident"]
        assert all(gate["passed"] for case in report["cases"] for gate in case["rubric"]["gates"])

    def test_exit_one_on_rubric_failure_with_a_complete_report(self, failing_run: tuple) -> None:
        proc, out = failing_run
        assert proc.returncode == 1, proc.stderr
        report = json.loads(out.read_text())
        assert report["verdict"] == "FAIL"
        assert report["exit_code"] == 1
        # The report is still complete: full schema shape, both cases, and the
        # gate that failed is identifiable.
        assert set(report) == {"provenance", "suite", "cases", "verdict", "exit_code"}
        assert [case["name"] for case in report["cases"]] == ["ramp_incident", "pinned_alarm"]
        pinned = next(c for c in report["cases"] if c["name"] == "pinned_alarm")
        failed = [g["name"] for g in pinned["rubric"]["gates"] if not g["passed"]]
        assert "alarm_fatigue" in failed

    def test_exit_two_on_spec_error_emits_no_verdict(self, suite_path: str) -> None:
        suite = yaml.safe_load(Path(suite_path).read_text())
        suite["cases"][0]["capture"] = "data/does_not_exist.csv"
        broken = Path(suite_path).parent / "broken_suite.yaml"
        broken.write_text(yaml.safe_dump(suite))

        proc = _run("--suite", str(broken))

        assert proc.returncode == 2, proc.stdout
        # No partial scoring output that could be read as a verdict.
        assert not _gate_lines(proc.stdout)
        assert not _gate_lines(proc.stderr)
        assert "VERDICT" not in proc.stdout
        assert proc.stdout.strip() == ""
        assert "does_not_exist.csv" in proc.stderr

    def test_missing_suite_flag_is_a_usage_error(self) -> None:
        proc = _run()
        assert proc.returncode == 2
        assert "--suite" in proc.stderr


class TestOutputRouting:
    def test_with_output_report_to_file_and_summary_to_stdout(self, passing_run: tuple) -> None:
        proc, out = passing_run
        report = json.loads(out.read_text())
        assert _gate_lines(proc.stdout) == _expected_gate_lines(report)
        assert proc.stdout.strip().splitlines()[-1] == "VERDICT PASS"
        # The report went to the file, not to stdout.
        with pytest.raises(json.JSONDecodeError):
            json.loads(proc.stdout)

    def test_without_output_report_to_stdout_and_summary_to_stderr(self, suite_path: str) -> None:
        proc = _run("--suite", suite_path, "--case", "ramp_incident")
        assert proc.returncode == 0, proc.stderr
        report = json.loads(proc.stdout)
        assert set(report) == {"provenance", "suite", "cases", "verdict", "exit_code"}
        assert report["verdict"] == "PASS"
        assert _gate_lines(proc.stderr) == _expected_gate_lines(report)
        assert proc.stderr.strip().splitlines()[-1] == "VERDICT PASS"
        assert not _gate_lines(proc.stdout)

    def test_summary_is_plain_ascii_words_with_no_emojis(self, failing_run: tuple) -> None:
        proc, _ = failing_run
        assert proc.stdout.isascii()
        assert "VERDICT FAIL" in proc.stdout
        verdict_words = {line.split()[0] for line in proc.stdout.splitlines() if line.split()}
        assert verdict_words <= {"PASS", "FAIL", "VERDICT"}


class TestCaseFilter:
    def test_case_is_repeatable_and_runs_only_the_named_cases(self, suite_path: str) -> None:
        proc = _run("--suite", suite_path, "--case", "pinned_alarm", "--case", "ramp_incident")
        assert proc.returncode == 1, proc.stderr
        report = json.loads(proc.stdout)
        assert sorted(case["name"] for case in report["cases"]) == [
            "pinned_alarm",
            "ramp_incident",
        ]

    def test_unknown_case_name_is_a_spec_error(self, suite_path: str) -> None:
        proc = _run("--suite", suite_path, "--case", "nope")
        assert proc.returncode == 2, proc.stdout
        assert "nope" in proc.stderr
        assert proc.stdout.strip() == ""


class TestHelp:
    def test_help_documents_the_three_flags(self) -> None:
        proc = _run("--help")
        assert proc.returncode == 0, proc.stderr
        for flag in ("--suite", "--output", "--case"):
            assert flag in proc.stdout


class TestAbsentCases:
    """A required gate that never evaluates must be a spec error, not a pass."""

    def test_required_gate_with_no_applicable_case_exits_two_naming_it(
        self, keeper_path: str, tmp_path: Path
    ) -> None:
        strict = {k: v for k, v in RUN_SUITE_RUBRIC["gates"]["alarm_fatigue"].items()}
        del strict["allow_absent"]
        suite_path = _tree_with(
            tmp_path,
            keeper_path,
            keep_cases={"ramp_incident"},
            gates={"alarm_fatigue": strict},
        )

        proc = _run("--suite", suite_path)

        assert proc.returncode == 2, proc.stdout
        # Name the gate AND the reason, so this cannot pass on an unrelated error.
        assert "alarm_fatigue" in proc.stderr
        assert "no applicable cases" in proc.stderr
        # A gate that never ran must not be reported as a verdict.
        assert proc.stdout.strip() == ""
        assert not _gate_lines(proc.stderr)

    def test_allow_absent_passes_vacuously_with_the_documented_detail(
        self, keeper_path: str, tmp_path: Path
    ) -> None:
        # The tree's rubric already declares alarm_fatigue allow_absent.
        suite_path = _tree_with(tmp_path, keeper_path, keep_cases={"ramp_incident"})
        out = tmp_path / "report.json"

        proc = _run("--suite", suite_path, "--output", str(out))

        assert proc.returncode == 0, proc.stderr
        report = json.loads(out.read_text())
        assert report["verdict"] == "PASS"
        gate = next(
            g for g in report["cases"][0]["rubric"]["gates"] if g["name"] == "alarm_fatigue"
        )
        assert gate["passed"]
        assert gate["detail"] == "no applicable cases"
        assert gate["observed"] == {}


@pytest.fixture(scope="module")
def all_kinds_report(keeper_path: str, tmp_path_factory: pytest.TempPathFactory) -> dict:
    """A report spanning all three case kinds: incident, healthy, controls."""
    tmp = tmp_path_factory.mktemp("all_kinds")
    suite_path = _tree_with(tmp, keeper_path, controls_only_case=True)
    out = tmp / "report.json"
    proc = _run("--suite", suite_path, "--output", str(out))
    # The pinned alarm fails alarm_fatigue, so the verdict is FAIL; the
    # report is written and complete either way, which is what is pinned.
    assert proc.returncode in (0, 1), proc.stderr
    return json.loads(out.read_text())


class TestBothAccountings:
    def test_every_case_kind_is_represented(self, all_kinds_report: dict) -> None:
        kinds = {case["kind"] for case in all_kinds_report["cases"]}
        assert kinds == {"incident", "healthy_reference", "negative_control"}

    def test_both_accountings_on_every_sustain_bearing_field(self, all_kinds_report: dict) -> None:
        for case in all_kinds_report["cases"]:
            for grid_label, grid in case["grids"].items():
                where = f"{case['name']}/{grid_label}"
                for field_name in GRID_SUSTAIN_FIELDS:
                    assert set(grid[field_name]) == {"3", "1"}, f"{where}/{field_name}"
                for rid, resource in grid["per_resource"].items():
                    for field_name in RESOURCE_SUSTAIN_FIELDS:
                        assert set(resource[field_name]) == {"3", "1"}, (
                            f"{where}/{rid}/{field_name}"
                        )

    def test_validator_rejects_a_stripped_grid_accounting(self, all_kinds_report: dict) -> None:
        validate_report(all_kinds_report)
        stripped = deepcopy(all_kinds_report)
        del stripped["cases"][0]["grids"]["serving"]["fleet_raises_per_week"]["1"]
        with pytest.raises(SpecError, match="fleet_raises_per_week"):
            validate_report(stripped)

    def test_validator_rejects_a_stripped_resource_accounting(self, all_kinds_report: dict) -> None:
        stripped = deepcopy(all_kinds_report)
        controls = next(c for c in stripped["cases"] if c["kind"] == "negative_control")
        resource = next(iter(controls["grids"]["serving"]["per_resource"].values()))
        del resource["sustained_run_counts"]["3"]
        with pytest.raises(SpecError, match="sustained_run_counts"):
            validate_report(stripped)
