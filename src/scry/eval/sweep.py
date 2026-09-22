# Description: Corrected margin-sweep primitives: named baselines, per-week rates, and the rate arms.
# Description: Torch-free at import; the gapped split is deferred to scry.eval.scoring inside the call.

"""Margin-sweep baselines and per-week arithmetic.

The corrected sweep never interchanges its two thresholds. ``q_deploy`` is the
quantile over every healthy window -- exactly what the bake persists, in-sample
within the week. ``q_fit`` is the quantile over the gapped earlier half, whose
held-out later half is the only out-of-sample false-positive estimate inside one
week. ``compute_baselines`` reports both plus ``q_eval`` (the held-out half's
quantile) and the two drift ratios, and computes the fit/eval halves through the
shared ``per_resource_time_split`` primitive so a single resource's baselines are
bit-identical to the pooled ``time_split``.

Per-week rates come from the OBSERVED span, ``7.0 * count / span_days``, never a
hardcoded weekly extrapolation, so a capture shorter or longer than seven days
is scaled by its own duration.

``ARM_BASELINES`` names which baseline each sweep arm reads and ``sweep_arms``
is the code path for it: an arm's per-resource threshold is the margin times
that named baseline, scanned over the healthy holdout, the whole healthy week
in sample, or the capture day.

This module does pure numpy/pandas arithmetic over already-computed errors and
imports torch-free; the split primitive lives in the torch-heavy scoring module,
so it is imported lazily inside the functions that need it, keeping ``import
scry.eval`` from pulling torch.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from scry.eval.detection import anomaly_runs

ARM_BASELINES: dict[str, str] = {
    "healthy_holdout": "q_fit",
    "healthy_insample": "q_deploy",
    "capture_day": "q_deploy",
    "detection_at_deploy": "q_deploy",
    "detection_at_fit": "q_fit",
}
"""Which baseline each sweep arm reads. The detection arms are consumed by ``run_sweep``."""


def compute_baselines(
    errors: np.ndarray,
    end_times: pd.DatetimeIndex,
    *,
    quantile: float,
    gap: int,
) -> dict[str, float]:
    """Named healthy-error baselines for one resource.

    Args:
        errors: One resource's per-window errors.
        end_times: Matching per-window end timestamps.
        quantile: The threshold quantile (e.g. 0.99).
        gap: Windows dropped between the fit and eval halves; callers pass
            ``ceil(seq_len / step)`` so the held-out half shares no raw samples
            with the fit half.

    Returns:
        Dict with exactly ``q_deploy`` (quantile over every window), ``q_fit``
        (quantile over the gapped earlier half), ``q_eval`` (quantile over the
        held-out later half), ``drift_x`` = ``q_eval / q_fit``, and
        ``deploy_over_fit_x`` = ``q_deploy / q_fit``.
    """
    from scry.eval.scoring import per_resource_time_split

    errors = np.asarray(errors, dtype=np.float64)
    resource_ids = np.zeros(errors.shape[0], dtype=np.int64)
    fit, held_out = per_resource_time_split(errors, end_times, resource_ids, gap=gap)

    q_fit = float(np.quantile(fit, quantile))
    q_eval = float(np.quantile(held_out, quantile))
    q_deploy = float(np.quantile(errors, quantile))
    return {
        "q_fit": q_fit,
        "q_eval": q_eval,
        "q_deploy": q_deploy,
        "drift_x": q_eval / q_fit,
        "deploy_over_fit_x": q_deploy / q_fit,
    }


def per_week(count: int, end_times: pd.DatetimeIndex) -> float:
    """Events per seven-day week from the OBSERVED span of ``end_times``.

    ``7.0 * count / span_days`` where ``span_days`` is the wall-clock span of the
    capture (last end time minus first), not a hardcoded seven-day assumption.

    Args:
        count: Number of events (e.g. sustained runs or alert raises) observed.
        end_times: The scanned window end-times whose min/max bound the span.

    Returns:
        The per-week rate scaled by the observed span.
    """
    span_days = (end_times.max() - end_times.min()) / pd.Timedelta(days=1)
    return 7.0 * count / span_days


def _ordered(
    errors: np.ndarray, end_times: pd.DatetimeIndex
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """One resource's windows in time order, stably sorted as the split primitive sorts."""
    order = np.argsort(end_times.values, kind="stable")
    return np.asarray(errors, dtype=np.float64)[order], end_times[order]


def _holdout(
    errors: np.ndarray, end_times: pd.DatetimeIndex, gap: int
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """One resource's gapped held-out half, from ``per_resource_time_split``.

    Its end times are the tail of the resource's time-ordered ends, sized by
    the length the primitive returns.
    """
    from scry.eval.scoring import per_resource_time_split

    ordered_errors, ordered_ends = _ordered(errors, end_times)
    resource_ids = np.zeros(ordered_errors.shape[0], dtype=np.int64)
    _, held_out = per_resource_time_split(ordered_errors, ordered_ends, resource_ids, gap=gap)
    return held_out, ordered_ends[len(ordered_ends) - len(held_out) :]


def _threshold(baselines: Mapping[str, Mapping[str, float]], resource_id: str, arm: str) -> float:
    """The unscaled baseline this arm reads for one resource."""
    if resource_id not in baselines:
        raise ValueError(
            f"no baselines for scanned resource {resource_id!r}; "
            f"per_resource_baselines covers {sorted(baselines)}"
        )
    return float(baselines[resource_id][ARM_BASELINES[arm]])


def sweep_arms(
    healthy: Mapping[str, tuple[np.ndarray, pd.DatetimeIndex]],
    capture: Mapping[str, tuple[np.ndarray, pd.DatetimeIndex]],
    *,
    per_resource_baselines: Mapping[str, Mapping[str, float]],
    margins: Sequence[float],
    sustain: int,
    gap: int,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Sustained runs and raises per margin for the three rate arms.

    Every scored resource is scanned regardless of role, at its own threshold:
    the margin times the baseline ``ARM_BASELINES`` names for the arm. Runs are
    maximal over-threshold stretches of at least ``sustain`` windows; raises are
    the same count at accounting 1, so the two rates differ where an excursion
    fragments.

    Args:
        healthy: Per-resource already-scored ``(errors, end_times)`` for the
            healthy week, keyed by resource id.
        capture: Per-resource ``(errors, end_times)`` for the capture day.
        per_resource_baselines: Resource id -> the ``compute_baselines`` dict.
        margins: Threshold multipliers; each is keyed ``str(float(margin))``.
        sustain: Windows a run must hold to count.
        gap: Windows dropped between the healthy fit and holdout halves.

    Returns:
        Arm name -> margin key -> ``{n_runs, runs_per_week, raises_per_week,
        span_days, threshold_by_resource}``. An arm spanning no time reports
        both rates as None with ``span_days`` 0.0 rather than dividing by zero.

    Raises:
        ValueError: A scanned resource has no entry in ``per_resource_baselines``.
    """
    scans = {
        "healthy_holdout": {rid: _holdout(e, t, gap) for rid, (e, t) in healthy.items()},
        "healthy_insample": {rid: _ordered(e, t) for rid, (e, t) in healthy.items()},
        "capture_day": {rid: _ordered(e, t) for rid, (e, t) in capture.items()},
    }
    arms: dict[str, dict[str, dict[str, Any]]] = {}
    for arm, scanned in scans.items():
        end_values = [t.values for _, t in scanned.values()]
        pooled_ends = pd.DatetimeIndex(np.concatenate(end_values) if end_values else [])
        # An arm that scanned nothing, or one timestamp, spans no time.
        span = pooled_ends.max() - pooled_ends.min() if len(pooled_ends) else pd.Timedelta(0)
        span_days = float(span / pd.Timedelta(days=1))
        by_margin: dict[str, dict[str, Any]] = {}
        for margin in margins:
            thresholds = {
                rid: float(margin) * _threshold(per_resource_baselines, rid, arm) for rid in scanned
            }
            n_runs = 0
            n_raises = 0
            for rid, (errors, _) in scanned.items():
                flags = errors > thresholds[rid]
                n_runs += len(anomaly_runs(flags, sustain))
                n_raises += len(anomaly_runs(flags, 1))
            by_margin[str(float(margin))] = {
                "n_runs": n_runs,
                "runs_per_week": None if span_days == 0.0 else per_week(n_runs, pooled_ends),
                "raises_per_week": None if span_days == 0.0 else per_week(n_raises, pooled_ends),
                "span_days": span_days,
                "threshold_by_resource": thresholds,
            }
        arms[arm] = by_margin
    return arms
