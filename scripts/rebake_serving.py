#!/usr/bin/env python3
# Description: The serving-checkpoint swap: back up, promote, restart, verify, roll back.
# Description: And the rebake CLI over it: bake, guard, stage beside the live checkpoint, report.

"""Rebake a serving checkpoint's thresholds, and promote the result on request.

The default run is stage-and-report and nothing else (spec section 10,
requirement 6). ``run_calibration`` bakes against the checkpoint in force, puts
every proposed threshold through the per-week band, and writes the guarded
block to a staged copy beside the live one; this script holds the operator's
inputs and writes the report of what was decided. The thresholds being served
do not move, because promoting that staged copy is the swap below -- a separate
run, behind ``--apply`` -- so there is a report of a change before there is a
change.

``swap_checkpoint`` is the chain ``--apply`` runs, and it is a function rather
than a stretch of an argparse body so that it can be driven by a test: the live
checkpoint, the staged copy, the restart command, and the health check are all
parameters, and what comes back is a ``SwapOutcome`` for the caller to map to
an exit code. It never exits the process itself. The injected health check is
what makes the failure path reachable at all -- the operator's is a curl
against the serving endpoint, a test's is ``/bin/false`` (spec section 10,
requirement 6).

The order is back up, promote, restart, verify, and that order is what makes
the swap recoverable. The backup is a copy rather than a move, so the live path
never stops existing underneath a process serving from it; the promote is a
rename onto that path, so no reader sees a half-written checkpoint and no later
``--apply`` can promote the same staged file twice; the service is restarted
only once the checkpoint it will load is the one in place; and the health check
is asked only once the restart has been. With nothing staged to promote, none
of it happens: that is a spec error raised before the live checkpoint is
touched, because a half-swap is worse than a swap that never started.

Recoverable is not enough on its own, so either link refusing rolls the swap
back from that backup before the outcome is returned: what an operator is left
with is the checkpoint that was in force, the staged copy they can promote
again once the service is fixed, and nothing else. A command that cannot be
executed at all counts as that link refusing, because it refuses after the
promote and an exception out of the chain there would leave the staged
checkpoint in force with no outcome to act on.

Guard, bake, and report logic stays in ``scry.eval.calibration``, pure and
tested there. This module is the one that holds the operator's inputs, moves
files, and runs commands.

Example:
    python scripts/rebake_serving.py --model models/serving.pt \\
        --calibration captures/healthy_week.csv --report rebake.json
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from scry.eval.calibration import run_calibration
from scry.eval.rubric import SpecError

if TYPE_CHECKING:
    from collections.abc import Sequence

BACKUP_INFIX = ".backup"
"""What the backup's name carries before the live checkpoint's suffix: ``serving.backup.pt``."""

OUTCOME_SWAPPED = "swapped"
"""The staged checkpoint is in force and the restarted service answered its health check."""

OUTCOME_ROLLED_BACK = "rolled-back"
"""The restart or the health check refused the staged checkpoint, and the swap was undone.

The live path holds the bytes and the mtime it held before, the staged copy is
back beside it to be promoted again once the service is fixed, and no backup is
left over: the restore is what consumes it. This is the outcome the CLI maps to
exit 3 (spec section 10, requirement 5).
"""


@dataclass(frozen=True)
class SwapOutcome:
    """What a swap did and what decided it."""

    outcome: str
    """``swapped`` or ``rolled-back``: whether the staged checkpoint is in force."""

    reason: str
    """The command that decided it, and what it exited with or why it could not run."""


def _refusal(what: str, command: Sequence[str]) -> str | None:
    """Run one link of the chain and say why it refused, or ``None`` if it passed.

    A command that cannot be executed at all -- ``argv[0]`` misspelled, absent
    from the operator's host, not executable -- is a refusal by this link and
    not an exception out of the chain. Both commands arrive from CLI flags, so
    that is an ordinary operator typo rather than a programming error, and it
    happens after the promote: letting the ``OSError`` escape would leave the
    staged checkpoint in force with no outcome and nothing rolled back.
    """
    try:
        code = subprocess.run(list(command), check=False).returncode
    except OSError as exc:
        return f"{what} command {shlex.join(command)} could not run: {exc}"
    return None if code == 0 else f"{what} command {shlex.join(command)} exited {code}"


def _roll_back(live: Path, staged: Path, backup: Path) -> None:
    """Undo a promote: the previous checkpoint back in force, the staged copy back beside it.

    The staged bytes are copied back to the staged path first and the backup is
    renamed over the live one second, which is the swap's own pair of
    operations run backwards and for the same reasons: the copy leaves the live
    path readable while it runs, and the rename is atomic, so a process reading
    the live path sees the staged checkpoint or the restored one and never a
    half-written file. The rename preserves the mtime ``copy2`` kept on the
    backup, which is the fallback the next run's age anchor needs. It is also
    what consumes the backup, so a second ``--apply`` after a rolled-back one
    cannot copy a refused checkpoint over the last known good one. Should
    either operation fail -- a full disk on the copy -- the ``OSError`` escapes
    with the state the restore started from still on disk, the staged
    checkpoint in force and the backup beside it: recoverable by hand, which is
    what the swap's ordering bought and what no reordering of these two could
    improve on.
    """
    shutil.copy2(live, staged)
    backup.replace(live)


def swap_checkpoint(
    live_path: str | Path,
    staged_path: str | Path,
    *,
    restart_command: Sequence[str],
    health_check_command: Sequence[str],
) -> SwapOutcome:
    """Put a staged checkpoint in force, restart the service, and check it.

    Args:
        live_path: The checkpoint being served, which this replaces.
        staged_path: The checkpoint to promote, as ``stage_checkpoint`` wrote
            it -- beside the live one, so the promote is a rename within one
            directory rather than a copy across filesystems.
        restart_command: Argument vector that restarts the service onto the
            promoted checkpoint. Run after the promote, never through a shell.
        health_check_command: Argument vector that asks the restarted service
            whether it is serving. A parameter so the chain can be tested
            against a command that fails; the operator's default is the CLI's.

    Returns:
        ``SwapOutcome``: ``swapped`` when the restart and the health check both
        succeeded, ``rolled-back`` -- the previous checkpoint restored and the
        staged copy left where it was -- when either refused, by a non-zero
        exit or by not running at all. Either way the backup is gone. A refusal
        is a value, not an exception and not an exit: which exit code it earns
        is the CLI's.

    Raises:
        SpecError: If there is nothing staged at ``staged_path``. Raised before
            the live checkpoint is copied, replaced, or restarted around, so a
            run with nothing to promote leaves everything as it found it.
    """
    live, staged = Path(live_path), Path(staged_path)
    if not staged.is_file():
        raise SpecError(f"nothing staged to promote at {staged}: stage a rebake first")
    backup = live.with_name(f"{live.stem}{BACKUP_INFIX}{live.suffix}")
    # copy2, not move: the live path stays readable throughout, and the backup
    # keeps the mtime a restored checkpoint's age anchor falls back to.
    shutil.copy2(live, backup)
    staged.replace(live)
    for what, command in (("restart", restart_command), ("health check", health_check_command)):
        refusal = _refusal(what, command)
        if refusal is not None:
            _roll_back(live, staged, backup)
            return SwapOutcome(outcome=OUTCOME_ROLLED_BACK, reason=refusal)
    backup.unlink()
    return SwapOutcome(
        outcome=OUTCOME_SWAPPED,
        reason=f"health check {shlex.join(health_check_command)} passed",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the rebake's inputs: what to rebake, from what, and where to report it."""
    parser = argparse.ArgumentParser(
        description=(
            "Rebake a serving checkpoint's thresholds, stage the guarded result beside "
            "it, and write the report. The checkpoint in force is not touched."
        ),
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Checkpoint in force: what is rebaked, and what the band measures against.",
    )
    parser.add_argument(
        "--calibration",
        required=True,
        help="All-healthy capture (URI or path) to bake the fresh thresholds from.",
    )
    parser.add_argument(
        "--report",
        required=True,
        help="Where to write the report JSON: every verdict, and the thresholds kept.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Stage a guarded rebake beside the checkpoint in force and report what it decided.

    The staged copy is written but never promoted: what an operator is left
    with is a report to read and a file a later ``--apply`` can put in force,
    which is the separation spec section 10, requirement 6 asks for.

    Returns:
        ``0``: the run completed (requirement 5, row 0). The report names the
        staged checkpoint, so the swap has something to be pointed at.
    """
    args = parse_args(argv)
    report, _ = run_calibration(args.model, args.calibration, stage=True)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"rebaked {args.model} against {args.calibration}: "
        f"staged {report['staged_path']}, reported to {args.report}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
