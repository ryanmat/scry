# Description: Per-window reconstruction error and healthy-threshold derivation for the keeper model.
# Description: Shared by the serving anomaly endpoint, the incident-validation harness, and the bake utility.

"""Reconstruction-error scoring for the X-DEC keeper model.

The anomaly signal is the per-window numerical reconstruction error from the
deterministic latent mean (mu): encode a window, take mu (no sampling), decode,
and take the mean squared error between the normalized numerical input and its
reconstruction. A healthy threshold is a quantile of that error over healthy
windows. Serving and validation import this module so their scores are on the
same scale as any baked threshold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from scry.model.xdec import TemporalXDEC


def reconstruction_errors(
    model: TemporalXDEC,
    x_num: torch.Tensor,
    x_cat: torch.Tensor,
    device: str,
    chunk_size: int = 512,
) -> np.ndarray:
    """Per-window numerical reconstruction error from the deterministic latent mean.

    Encodes each window, takes the latent mean ``mu`` (no sampling), decodes, and
    returns the mean squared error between the normalized numerical input and its
    reconstruction over (seq_len, num_features). The categorical branch informs
    the encoding but is not part of the error, matching the incident-validation
    harness so serving scores share the baked threshold's scale.

    Args:
        model: The keeper model.
        x_num: Normalized numerical windows (n, seq, n_num).
        x_cat: Encoded categorical windows (n, seq, n_cat).
        device: Torch device string.
        chunk_size: Batch size for inference.

    Returns:
        Array of shape (n,) with one error per window.
    """
    errors: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, x_num.shape[0], chunk_size):
            xn = x_num[start : start + chunk_size].to(device)
            xc = x_cat[start : start + chunk_size].to(device)
            _, mu, _ = model.xvae.encode(xn, xc)
            x_num_recon, _ = model.xvae.decode(mu)
            err = ((xn - x_num_recon) ** 2).mean(dim=(1, 2))
            errors.append(err.cpu().numpy())
    if not errors:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate(errors).astype(np.float64)


def time_split(
    errors: np.ndarray, end_times: pd.DatetimeIndex, gap: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Split errors into an earlier (fit) and later (eval) half by end-time.

    A ``gap`` of windows is dropped between the halves so the eval windows share
    no raw samples with the fit windows (sliding windows overlap), keeping the
    false-positive rate genuinely out-of-sample.

    Args:
        errors: Per-window errors.
        end_times: Matching per-window end timestamps.
        gap: Number of windows to drop between the fit and eval halves.

    Returns:
        Tuple of (earlier_half, later_half). The later half is empty when there
        are too few windows to leave any after the gap.
    """
    if errors.size < 2:
        return errors, np.zeros(0, dtype=errors.dtype)
    order = np.argsort(end_times.values, kind="stable")
    ordered = errors[order]
    mid = ordered.size // 2
    return ordered[:mid], ordered[mid + gap :]


def healthy_threshold(
    errors: np.ndarray,
    end_times: pd.DatetimeIndex,
    *,
    quantile: float,
    gap: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Derive a serving threshold from an all-healthy error set, held out for the FPR.

    The threshold is fit on the earlier half of the healthy windows and the later
    half (with a ``gap`` of windows dropped between them so the two share no raw
    samples) is left as an out-of-sample healthy set for measuring the
    false-positive rate. When there are too few windows to leave a held-out half,
    the threshold is fit on the whole set and the eval set is empty.

    Args:
        errors: Per-window errors for an all-healthy capture.
        end_times: Per-window end timestamps.
        quantile: Threshold quantile over the fit errors.
        gap: Windows to drop between the fit and eval halves (the window overlap).

    Returns:
        Tuple of (threshold, fit_errors, eval_errors). ``eval_errors`` may be
        empty when no held-out healthy windows are available.

    Raises:
        ValueError: If there are no healthy windows to fit the threshold.
    """
    if errors.size == 0:
        raise ValueError("No healthy windows available to fit the serving threshold.")

    fit, eval_ = time_split(errors, end_times, gap=gap)
    if fit.size == 0:
        # Too few windows to hold out a healthy eval set; fit on the whole set.
        fit, eval_ = errors, np.zeros(0, dtype=errors.dtype)

    threshold = float(np.quantile(fit, quantile))
    return threshold, fit, eval_
