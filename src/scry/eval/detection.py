# Description: Detection primitives for the evaluation harness: sustained anomaly runs.
# Description: Torch-free; promoted from scripts/validate_incident.py so both share one implementation.

"""Sustained-run detection primitives.

``anomaly_runs`` is the single implementation of run extraction over per-window
anomaly flags; the incident-validation harness aliases it. Everything here is
importable without torch.
"""

from __future__ import annotations

import numpy as np

SUSTAIN_DEFAULT: int = 3


def anomaly_runs(flags: np.ndarray, sustain: int) -> list[tuple[int, int]]:
    """Maximal runs of consecutive True flags with length >= ``sustain``.

    Args:
        flags: Boolean array of per-window anomaly flags, in time order.
        sustain: Minimum run length to qualify.

    Returns:
        List of (start_index, end_index) inclusive, one per qualifying run.
    """
    runs: list[tuple[int, int]] = []
    i, n = 0, len(flags)
    while i < n:
        if flags[i]:
            j = i
            while j + 1 < n and flags[j + 1]:
                j += 1
            if j - i + 1 >= sustain:
                runs.append((i, j))
            i = j + 1
        else:
            i += 1
    return runs
