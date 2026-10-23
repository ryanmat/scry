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

Those operator inputs are the bake's own -- the margin, the quantile, and the
device it scores on -- plus the health check a later ``--apply`` runs, each a
flag defaulting to the constant the code already carries rather than a module
global welded in (requirement 6). Beside them are the two the operator decides
rather than the band: ``--exclude`` holds a resource out of the staged map
whatever the bake proposes for it, and ``--keep-global`` holds the global
already in force. What a completed run exits with is what the band decided: 0
with nothing refused, 1 with at least one threshold held back (requirement 5).
A refusal is not a failure, so both runs leave the same two things, a report to
read and a staged copy to promote; neither hold is a refusal either, so neither
moves the exit code.

``--allow-drift`` is the operator's answer to the band itself: what it refused
is staged at the freshly baked value instead, stamped ``accepted-allow-drift``
on its row, so the run that would have exited 1 holding those thresholds back
exits 0 having moved them. It never reaches the seed requirement, which
resolves upstream of the band -- a first bake with no previous thresholds and
no seed is refused under the flag exactly as it is without it.

That refusal is the exit code which is not the band's. 2 is a run that never
baked (requirement 5): a spec or config error, the missing seed above or a
calibration capture that cannot be read, and what distinguishes it from 1 for
an operator's cron is that nothing was decided -- no report to read and
nothing staged, where a 1 leaves both. The error is written to stderr naming
what failed, rather than escaping as a traceback and taking Python's own exit
1 with it, which is the code the table gives to a run that completed.

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

``--apply`` is what asks for all of that, and it takes the restart command
with it: no service name is the code's to assume, so a promote with nothing to
restart the service onto the new checkpoint with is a usage error rather than
a half-done swap. So is either command arriving empty, which is what
``shlex.split`` makes of a blank string and what ``subprocess`` raises
``IndexError`` on rather than refusing, after the promote.

What such a run exits with is the band's own code when the swap held -- 0 or
1, the two a stage-and-report run earns, since promoting what was staged does
not change what was decided -- and 3 when it did not: a swap or post-swap
health failure after the automatic rollback (requirement 5). The 3 outranks a
1, because thresholds held back are a report to read while a rolled-back swap
is a service to look at. On disk that rollback is complete, but the service
was restarted around the swap and the health check is what refused it, so what
the 3 says on stderr is that the process may still hold the checkpoint the
rollback has taken off disk. Which kind of run wrote a report is the report's
``dry_run`` field, where requirement 5 puts it, and not an exit code of its
own.

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
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb

from scry.eval.calibration import DEFAULT_MARGIN, DEFAULT_QUANTILE, run_calibration
from scry.eval.rubric import SpecError

if TYPE_CHECKING:
    from collections.abc import Sequence

_CONFIG_ERRORS: tuple[type[Exception], ...] = (OSError, ValueError, duckdb.Error)
"""What a run that never baked raises, and so what the CLI exits 2 on.

Requirement 5 gives code 2 to a spec or config error, which is every way the
operator's inputs can stop the run before it has decided anything.
``OSError`` is a checkpoint that is not there or cannot be read.
``ValueError`` is an input the bake cannot work with -- a capture with no
readable time range, one that yields no windows, a margin that is not a
positive finite number -- and it is also how the package's own refusal
arrives, since ``SpecError`` is a ``ValueError``: a first bake whose
thresholds neither the serving block nor a seed supplies, an ``--exclude``
naming a resource this rebake decided nothing about. Listing both would be
listing the same type twice, so the test that drives a seedless first bake
through this CLI is what holds that mapping down. ``duckdb.Error`` is the
capture itself, absent or unreadable where it is: fetching it is a query, and
the data layer raises its reader's error rather than wrapping it, so the CLI
names the reader to keep a mistyped path out of the exit code the band owns.

Argparse exits 2 on its own usage errors, so a flag that does not exist and a
flag naming a file that does not exist read alike to the cron that runs this:
nothing was baked.
"""

DEFAULT_HEALTH_CHECK = "curl -fsS http://localhost:8000/health"
"""What ``--apply`` asks the restarted service, unless the operator names something else.

The real invocation (spec section 10, requirement 6): ``-f`` is what makes it a
health check rather than a reachability check, since without it curl exits 0 on
an error page, and ``-sS`` keeps a passing check quiet while a failing one says
why. Held as the string an operator would type, because that is what ``--help``
prints and what ``shlex.split`` turns into the vector the chain runs.
"""

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
    """Parse the rebake's inputs: what to rebake, from what, how, and where to report it.

    The bake's own inputs -- the margin, the quantile, and the device it scores
    on -- are the operator's here rather than constants welded into a module,
    and so is the health check ``--apply`` runs (requirement 6). Each defaults
    to what the code already does, so a run that names none of them is the run
    this CLI made before they existed. The restart command is the one input
    with no default to fall back on, since the service to restart is the
    operator's and not this code's to guess, so ``--apply`` requires it.

    ``--exclude`` and ``--keep-global`` are of a different kind: standing policy
    on the serving host, applied after the band rather than weighed by it, and
    so they default to holding nothing and keeping nothing. ``--allow-drift``
    is a third kind again -- neither an input to the bake nor a hold over its
    result, but the band's own override -- and defaults to off, because the
    band is what this script exists to run.
    """
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
    parser.add_argument(
        "--margin",
        type=float,
        default=DEFAULT_MARGIN,
        help=(
            "Multiple of each resource's own healthy quantile its fresh threshold "
            "is taken at (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=DEFAULT_QUANTILE,
        help=(
            "Healthy-window quantile the fresh thresholds are taken at "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device the bake scores on (default: the device the keeper detects).",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="RESOURCE",
        help=(
            "Hold a resource out of the staged per-resource map whatever the bake "
            "proposes for it; it falls back to the global threshold. Repeatable."
        ),
    )
    parser.add_argument(
        "--keep-global",
        action="store_true",
        help=(
            "Hold the global threshold the checkpoint already serves in force, "
            "whatever the band decides about the freshly baked one."
        ),
    )
    parser.add_argument(
        "--allow-drift",
        action="store_true",
        help=(
            "Stage the freshly baked value where the band refused it, stamped "
            "accepted-allow-drift. The seed requirement is upstream of the band "
            "and is not bypassed."
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Promote the staged checkpoint once it is written: back up, promote, "
            "restart, health-check, and roll back if either command refuses. "
            "Requires --restart-command."
        ),
    )
    parser.add_argument(
        "--restart-command",
        type=shlex.split,
        help=(
            "Command --apply restarts the service onto the promoted checkpoint with. "
            "Required with --apply, since no service name is this code's to assume."
        ),
    )
    parser.add_argument(
        "--health-check-command",
        type=shlex.split,
        default=DEFAULT_HEALTH_CHECK,
        help=(
            "Command --apply asks whether the restarted service is serving "
            "(default: %(default)s)."
        ),
    )
    args = parser.parse_args(argv)
    if args.apply and not (args.restart_command and args.health_check_command):
        parser.error(
            "--apply needs a --restart-command and a --health-check-command it can run: "
            "an unrestarted service is not serving the checkpoint that was promoted, and "
            "a vector shlex.split left empty is not a command at all"
        )
    return args


def main(argv: list[str] | None = None) -> int:
    """Stage a guarded rebake beside the checkpoint in force, report it, and promote on request.

    Without ``--apply`` the staged copy is written and never promoted: what an
    operator is left with is a report to read and a file a later run can put in
    force, which is the separation spec section 10, requirement 6 asks for.
    With it, that later run is this one -- the same bake, staged and reported
    before any of it is promoted, and then put in force by the chain above.

    Returns:
        ``0`` when the run completed with nothing refused, ``1`` when it
        completed with at least one proposal refused by the band (requirement
        5, rows 0 and 1), counted off the run's own verdicts. A refusal is not
        a failure of the run: it stages and reports like any other run, and
        what the 1 says is that some of what is staged is the previous
        threshold rather than a fresh one -- the report's rows name which, and
        why. ``2`` is row 2, a run that never baked: one of
        ``_CONFIG_ERRORS``, reported on stderr, with no report written and
        nothing staged, since a run that decided nothing must leave nothing to
        read and nothing to promote. ``3`` is row 3 and only an ``--apply``
        run can earn it: the swap was rolled back, which outranks the band's 1
        because the report says what the band held while only stderr says what
        the service is left serving.
    """
    args = parse_args(argv)
    try:
        report, _ = run_calibration(
            args.model,
            args.calibration,
            quantile=args.quantile,
            margin=args.margin,
            device=args.device,
            allow_drift=args.allow_drift,
            exclude=args.exclude,
            keep_global=args.keep_global,
            dry_run=not args.apply,
            stage=True,
        )
    except _CONFIG_ERRORS as exc:
        # The table's own name for code 2, since the half of it that is a
        # config error -- a capture that is not there -- would send an
        # operator reading "spec error" to the wrong input.
        print(f"spec/config error: {exc}", file=sys.stderr)
        return 2
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"rebaked {args.model} against {args.calibration}: "
        f"staged {report['staged_path']}, reported to {args.report}"
    )
    rejected = report["rejected_count"]
    if rejected:
        print(f"refused {rejected} proposal(s): the previous thresholds stay in the staged block")
    staged = report["staged_path"]
    if args.apply:
        swap = swap_checkpoint(
            args.model,
            staged,
            restart_command=args.restart_command,
            health_check_command=args.health_check_command,
        )
        if swap.outcome == OUTCOME_ROLLED_BACK:
            print(
                f"rolled back: {swap.reason}. {args.model} is the checkpoint that was "
                f"in force and {staged} is beside it to promote again, but the service "
                "was restarted around the swap: check what it is serving.",
                file=sys.stderr,
            )
            return 3
        print(f"swapped {staged} into {args.model}: {swap.reason}")
    return 1 if rejected else 0


if __name__ == "__main__":
    raise SystemExit(main())
