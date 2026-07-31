# Description: Run provenance for the evaluation harness: model, data, config, and repo identity.
# Description: Torch-free; a report consumer can reproduce the run from this block plus the data files.

"""Provenance block for evaluation reports.

``build_provenance`` gathers everything a consumer needs to reproduce a run
from the report alone plus the data files: the git revision and dirty flag
("unknown" and None outside a repo, or when git is unavailable), the
mandatory model sha256, per-case data identity (path, byte size, mtime),
profile and seq_len from the checkpoint config, every grid definition, the
sustain accountings, detection mode, the threshold policy ``describe()``
including resolved per-resource thresholds, the rubric path and version, the
installed scryml version, and a Z-suffix UTC timestamp. Everything here is
importable without torch.
"""

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from scry.eval.scoring import ScoringGrid


def _iso_z(moment: datetime) -> str:
    """ISO-8601 UTC with a Z suffix, the on-disk timestamp form."""
    return moment.isoformat().replace("+00:00", "Z")


def _git_state(repo_dir: str | None) -> tuple[str, bool | None]:
    """The repo revision and dirty flag; ("unknown", None) outside a repo."""
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True, text=True
        )
    except OSError:
        return "unknown", None
    if rev.returncode != 0:
        return "unknown", None
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo_dir, capture_output=True, text=True
    )
    dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
    return rev.stdout.strip(), dirty


def _scryml_version() -> str:
    try:
        return metadata.version("scryml")
    except metadata.PackageNotFoundError:
        return "unknown"


def build_provenance(
    *,
    model_path: str,
    profile: str,
    seq_len: int,
    grids: Mapping[str, ScoringGrid],
    sustains: Sequence[int],
    detection_mode: str,
    policy_description: Mapping[str, Any],
    rubric_path: str,
    rubric_version: int,
    case_data_paths: Mapping[str, str],
    repo_dir: str | None = None,
) -> dict[str, Any]:
    """Build the report's provenance block.

    Args:
        model_path: The scored checkpoint; its sha256 is computed here and is
            mandatory (a missing file raises).
        profile: The keeper's feature profile.
        seq_len: Window length from the checkpoint config.
        grids: ScoringGrids keyed by the rubric's grid names.
        sustains: The sustain accountings the run reports.
        detection_mode: The rubric's detection-selection mode.
        policy_description: The threshold policy's ``describe()`` output,
            embedded verbatim (resolved per-resource thresholds included).
        rubric_path: The rubric file the run evaluated.
        rubric_version: The rubric's declared version.
        case_data_paths: Case name -> capture path; each file's byte size and
            mtime are recorded.
        repo_dir: Where to resolve the git revision; defaults to the working
            directory.

    Returns:
        A JSON-serializable dict with every provenance field.

    Raises:
        FileNotFoundError: If the model or a case data file does not exist.
    """
    git_rev, git_dirty = _git_state(repo_dir)
    cases: dict[str, dict[str, Any]] = {}
    for name, data_path in case_data_paths.items():
        stat = Path(data_path).stat()
        cases[name] = {
            "data_path": str(data_path),
            "bytes": int(stat.st_size),
            "mtime": _iso_z(datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)),
        }
    return {
        "git_rev": git_rev,
        "git_dirty": git_dirty,
        "model_path": str(model_path),
        "model_sha256": hashlib.sha256(Path(model_path).read_bytes()).hexdigest(),
        "profile": profile,
        "seq_len": int(seq_len),
        "grids": {
            name: {
                "label": grid.label,
                "step_samples": int(grid.step_samples),
                "cadence_seconds": (
                    float(grid.cadence.total_seconds()) if grid.cadence is not None else None
                ),
            }
            for name, grid in grids.items()
        },
        "sustains": [int(sustain) for sustain in sustains],
        "detection_mode": detection_mode,
        "threshold_policy": dict(policy_description),
        "rubric_path": str(rubric_path),
        "rubric_version": int(rubric_version),
        "cases": cases,
        "scryml_version": _scryml_version(),
        "generated_at": _iso_z(datetime.now(timezone.utc)),
    }
