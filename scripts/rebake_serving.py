#!/usr/bin/env python3
# Description: The serving-checkpoint swap: back up, promote the staged copy, restart, verify.
# Description: A function the CLI's --apply calls; guard, bake, and report stay in calibration.py.

"""Promote a staged serving checkpoint and verify what it put in force.

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

Guard, bake, and report logic stays in ``scry.eval.calibration``, pure and
tested there. This module is the one that moves files and runs commands.
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from scry.eval.rubric import SpecError

if TYPE_CHECKING:
    from collections.abc import Sequence

BACKUP_INFIX = ".backup"
"""What the backup's name carries before the live checkpoint's suffix: ``serving.backup.pt``."""

OUTCOME_SWAPPED = "swapped"
"""The staged checkpoint is in force and the restarted service answered its health check."""

OUTCOME_UNVERIFIED = "unverified"
"""The staged checkpoint is in force but the restart or the health check refused it.

The backup is still on disk, and restoring from it is the rollback half of this
chain, which is where the ``rolled-back`` outcome and its exit 3 come from.
Nothing in the repo calls this chain until that half lands.
"""


@dataclass(frozen=True)
class SwapOutcome:
    """What a swap did, what decided it, and what is left to undo it with."""

    outcome: str
    """``swapped`` or ``unverified``: whether what is in force is verified to serve."""

    reason: str
    """The command that decided it, and what it exited with."""

    backup: Path | None
    """Where the previous checkpoint is kept, or ``None`` once the swap cleaned it up."""


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
        ``SwapOutcome``: ``swapped`` with the backup cleaned up when the
        restart and the health check both succeeded, ``unverified`` with the
        backup kept when either refused. A refusal is a value, not an
        exception and not an exit: which exit code it earns is the CLI's.

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
        code = subprocess.run(list(command), check=False).returncode
        if code != 0:
            return SwapOutcome(
                outcome=OUTCOME_UNVERIFIED,
                reason=f"{what} command {shlex.join(command)} exited {code}",
                backup=backup,
            )
    backup.unlink()
    return SwapOutcome(
        outcome=OUTCOME_SWAPPED,
        reason=f"health check {shlex.join(health_check_command)} passed",
        backup=None,
    )
