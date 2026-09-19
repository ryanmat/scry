# Description: Corrected margin-sweep primitives for the evaluation harness: named baselines and per-week rates.
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

This module does pure numpy/pandas arithmetic over already-computed errors and
imports torch-free; the split primitive lives in the torch-heavy scoring module,
so it is imported lazily inside ``compute_baselines`` to keep ``import
scry.eval`` from pulling torch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


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
