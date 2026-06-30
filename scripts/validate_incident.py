#!/usr/bin/env python3
# Description: Incident-validation harness for a trained keeper X-DEC model.
# Description: Scores per-window reconstruction error and reports detection lead time.

"""Validate a keeper (normal-behavior) X-DEC model against an incident capture.

The harness loads a trained keeper model, windows an incident capture in the
model's own feature order, normalizes those windows with the checkpoint's stored
parameters (never re-fitting on incident data), and computes a per-window
reconstruction error from the deterministic latent mean. A detection threshold is
derived from healthy windows only -- either the windows that end strictly before
the earliest labeled incident, or a separate reference capture -- so a large
incident can never inflate its own threshold. For each labeled incident the
harness reports whether a sustained anomaly was detected and the lead time of that
detection relative to the incident onset.

Examples:
    python scripts/validate_incident.py \\
        --model models/aro_keeper.pt \\
        --data data/captures/aro_incident.parquet \\
        --profile aro_node \\
        --labels incidents.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from scry.data.feature_engineering import (
    create_dual_windows,
    pivot_metrics,
    set_active_profile,
)
from scry.data.fetcher import DataFetcher
from scry.model.xdec import TemporalXDEC
from scry.utils.config import get_config


@dataclass
class Keeper:
    """A loaded keeper model plus the schema and normalization it was trained with."""

    model: TemporalXDEC
    device: str
    config: dict[str, Any]
    normalization: dict[str, Any]
    cat_normalization: dict[str, Any] | None
    numerical_features: list[str]
    categorical_features: list[str]
    profile: str | None


@dataclass
class Incident:
    """A labeled incident window for one resource."""

    resource_id: str
    type: str
    start: pd.Timestamp
    end: pd.Timestamp


@dataclass
class WindowSet:
    """Normalized windows ready for scoring, with their resource and end-time labels."""

    x_num: torch.Tensor
    x_cat: torch.Tensor
    resource_ids: np.ndarray
    end_times: pd.DatetimeIndex


def _detect_device() -> str:
    """Detect the best available torch device."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_keeper(model_path: str) -> Keeper:
    """Load the keeper checkpoint and reconstruct the model.

    Mirrors the predictor's load path: reconstruct ``TemporalXDEC`` from the saved
    config, load the weights, and carry the stored normalization and feature
    schema so incident windows can be aligned and scaled exactly as in training.

    Args:
        model_path: Path to the saved checkpoint.

    Returns:
        A populated :class:`Keeper`.

    Raises:
        FileNotFoundError: If the checkpoint does not exist.
        ValueError: If the checkpoint has no usable feature schema.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    device = _detect_device()
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    config = checkpoint["config"]
    normalization = checkpoint.get("normalization") or {"mean": None, "std": None}
    cat_normalization = checkpoint.get("categorical_normalization")

    schema = checkpoint.get("feature_schema")
    if not schema or "numerical" not in schema or "categorical" not in schema:
        raise ValueError(
            "Model checkpoint has no feature_schema; incident windows cannot be "
            "aligned by name. Retrain with the current scripts/train_model.py."
        )
    numerical_features = [str(x) for x in schema["numerical"]]
    categorical_features = [str(x) for x in schema["categorical"]]

    if len(numerical_features) != config["num_numerical"]:
        raise ValueError(
            f"feature_schema numerical count ({len(numerical_features)}) does not "
            f"match model num_numerical ({config['num_numerical']}). Retrain the model."
        )
    if len(categorical_features) != config["num_categorical"]:
        raise ValueError(
            f"feature_schema categorical count ({len(categorical_features)}) does not "
            f"match model num_categorical ({config['num_categorical']}). Retrain the model."
        )

    model = TemporalXDEC(
        num_numerical=config["num_numerical"],
        num_categorical=config["num_categorical"],
        seq_len=config["seq_len"],
        num_hidden=config["num_hidden"],
        cat_hidden=config["cat_hidden"],
        latent_dim=config["latent_dim"],
        n_clusters=config["n_clusters"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return Keeper(
        model=model,
        device=device,
        config=config,
        normalization=normalization,
        cat_normalization=cat_normalization,
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        profile=schema.get("profile"),
    )


def load_incidents(labels_path: str) -> list[Incident]:
    """Load and parse the labeled incidents from a JSON file.

    Args:
        labels_path: Path to a JSON list of ``{resource_id, type, start, end}``.

    Returns:
        Parsed incidents with UTC timestamps.

    Raises:
        ValueError: If the file is not a non-empty list of well-formed entries.
    """
    with open(labels_path) as handle:
        raw = json.load(handle)
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"Labels file {labels_path} must be a non-empty JSON list.")

    incidents: list[Incident] = []
    for entry in raw:
        missing = [k for k in ("resource_id", "type", "start", "end") if k not in entry]
        if missing:
            raise ValueError(f"Incident entry missing keys {missing}: {entry}")
        incidents.append(
            Incident(
                resource_id=str(entry["resource_id"]),
                type=str(entry["type"]),
                start=pd.to_datetime(entry["start"], utc=True),
                end=pd.to_datetime(entry["end"], utc=True),
            )
        )
    return incidents


async def _fetch_long(
    data_uri: str,
    profile: str | None,
    data_format: str | None,
) -> pd.DataFrame:
    """Fetch the full capture as a canonical long-format DataFrame.

    The time range is derived from the source summary so the entire capture is
    windowed regardless of wall-clock time.
    """
    fetcher = DataFetcher.from_object_store(data_uri, data_format=data_format)
    summary = await fetcher.get_data_summary()
    earliest = pd.to_datetime(summary["earliest_timestamp"], utc=True)
    latest = pd.to_datetime(summary["latest_timestamp"], utc=True)
    if pd.isna(earliest) or pd.isna(latest):
        raise ValueError(f"Capture {data_uri} has no readable timestamp range.")
    # fetch_metrics is [start, end); add a margin so the last sample is included.
    start: datetime = earliest.to_pydatetime()
    end: datetime = latest.to_pydatetime() + timedelta(seconds=1)
    return await fetcher.get_metrics_dataframe(start, end, profile=profile)


def _align_frames(
    df_long: pd.DataFrame,
    numerical_features: list[str],
    categorical_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Pivot the long capture and align columns to the model's feature order.

    Features the model expects but the capture lacks are inserted as all-NaN
    columns so the window tensors keep the model's shape and column order. The
    returned mask flags those absent numerical features so they can be mapped to
    the neutral normalized value (0) after scaling, mirroring the predictor.

    Args:
        df_long: Canonical long-format metrics.
        numerical_features: Model numerical features, in model order.
        categorical_features: Model categorical features, in model order.

    Returns:
        Tuple of (df_num, df_cat, absent_numerical_mask).
    """
    pivoted = pivot_metrics(df_long)

    df_num = pivoted[["resource_id", "timestamp"]].copy()
    absent = np.zeros(len(numerical_features), dtype=bool)
    for i, name in enumerate(numerical_features):
        if name in pivoted.columns:
            df_num[name] = pivoted[name]
        else:
            df_num[name] = np.nan
            absent[i] = True

    df_cat = pivoted[["resource_id", "timestamp"]].copy()
    for name in categorical_features:
        df_cat[name] = pivoted[name] if name in pivoted.columns else np.nan

    return df_num, df_cat, absent


def _normalize_numerical(
    windows: np.ndarray,
    normalization: dict[str, Any],
    absent: np.ndarray,
) -> np.ndarray:
    """Apply the checkpoint's stored z-score normalization to numerical windows.

    NaNs are filled forward then backward then with zero (the training rule), the
    stored per-feature mean/std are applied, and features the capture never
    supplied are set to the neutral normalized value (0).

    Args:
        windows: Raw numerical windows (n, seq, n_num).
        normalization: Stored ``{mean, std}`` arrays from the checkpoint.
        absent: Boolean mask of features missing from the capture.

    Returns:
        Normalized windows.
    """
    out = windows.astype(np.float32).copy()
    for i in range(out.shape[0]):
        for j in range(out.shape[2]):
            series = pd.Series(out[i, :, j]).ffill().bfill().fillna(0.0)
            out[i, :, j] = series.to_numpy(dtype=np.float32)

    if normalization.get("mean") is not None:
        mean = np.asarray(normalization["mean"], dtype=np.float32)
        std = np.asarray(normalization["std"], dtype=np.float32)
        std = np.where(std == 0, 1.0, std)
        out = (out - mean) / std

    if absent.any():
        out[:, :, absent] = 0.0
    return out


def _normalize_categorical(
    windows: np.ndarray,
    cat_normalization: dict[str, Any] | None,
) -> np.ndarray:
    """Apply the checkpoint's stored min-max encoding to categorical windows.

    Mirrors :func:`encode_categorical` / the predictor: NaNs become 0, each
    feature is scaled by its stored ``[min, max]``, and a degenerate range falls
    back to 1.0 when the training value was positive else 0.0.

    Args:
        windows: Raw categorical windows (n, seq, n_cat).
        cat_normalization: Stored ``{min, max}`` arrays, or None.

    Returns:
        Encoded windows in [0, 1].
    """
    out = np.nan_to_num(windows.astype(np.float32), nan=0.0)
    if cat_normalization is None:
        return out

    cat_min = np.asarray(cat_normalization["min"], dtype=np.float32)
    cat_max = np.asarray(cat_normalization["max"], dtype=np.float32)
    for j in range(out.shape[2]):
        lo, hi = cat_min[j], cat_max[j]
        if hi > lo:
            out[:, :, j] = np.clip((out[:, :, j] - lo) / (hi - lo), 0.0, 1.0)
        else:
            out[:, :, j] = 1.0 if hi > 0 else 0.0
    return out


def build_windows(
    df_long: pd.DataFrame,
    keeper: Keeper,
    seq_len: int,
    step: int,
) -> WindowSet:
    """Window a capture in model order and normalize with the stored parameters.

    Args:
        df_long: Canonical long-format metrics for the capture.
        keeper: The loaded keeper model and its schema/normalization.
        seq_len: Window length (the model's sequence length).
        step: Sliding-window step.

    Returns:
        A :class:`WindowSet` of normalized tensors with labels, possibly empty.
    """
    df_num, df_cat, absent = _align_frames(
        df_long, keeper.numerical_features, keeper.categorical_features
    )
    num_windows, cat_windows, labels = create_dual_windows(
        df_num, df_cat, window_size=seq_len, step=step
    )

    if num_windows.shape[0] == 0:
        empty_num = torch.zeros((0, seq_len, len(keeper.numerical_features)), dtype=torch.float32)
        empty_cat = torch.zeros((0, seq_len, len(keeper.categorical_features)), dtype=torch.float32)
        return WindowSet(
            x_num=empty_num,
            x_cat=empty_cat,
            resource_ids=np.zeros(0, dtype=object),
            end_times=pd.DatetimeIndex([], tz="UTC"),
        )

    norm_num = _normalize_numerical(num_windows, keeper.normalization, absent)
    norm_cat = _normalize_categorical(cat_windows, keeper.cat_normalization)

    end_times = pd.to_datetime(labels[:, 1], utc=True)
    return WindowSet(
        x_num=torch.tensor(norm_num, dtype=torch.float32),
        x_cat=torch.tensor(norm_cat, dtype=torch.float32),
        resource_ids=labels[:, 0].astype(object),
        end_times=pd.DatetimeIndex(end_times),
    )


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
    reconstruction over (seq_len, num_features).

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


def _first_sustained(flags: np.ndarray, sustain: int) -> int | None:
    """Index of the first window starting a run of >= ``sustain`` consecutive True.

    Args:
        flags: Boolean array of per-window anomaly flags, in time order.
        sustain: Required run length.

    Returns:
        Start index of the first qualifying run, or None.
    """
    run = 0
    for i, flag in enumerate(flags):
        if flag:
            run += 1
            if run >= sustain:
                return i - sustain + 1
        else:
            run = 0
    return None


def evaluate_incidents(
    errors: np.ndarray,
    resource_ids: np.ndarray,
    end_times: pd.DatetimeIndex,
    incidents: list[Incident],
    threshold: float,
    sustain: int,
) -> list[dict[str, Any]]:
    """Detect each incident and compute its lead time.

    For each incident, the resource's windows up to the incident end are scanned
    in time order for the first sustained anomaly (>= ``sustain`` consecutive
    windows over ``threshold``). Lead time is the incident start minus that first
    sustained window's end-time (positive means detected before onset).

    Args:
        errors: Per-window reconstruction errors.
        resource_ids: Per-window resource ids.
        end_times: Per-window end timestamps (UTC).
        incidents: Labeled incidents.
        threshold: Anomaly threshold.
        sustain: Sustained-anomaly run length.

    Returns:
        One result dict per incident.
    """
    end_values = end_times.values
    results: list[dict[str, Any]] = []
    for incident in incidents:
        resource_mask = resource_ids == incident.resource_id
        scan_mask = resource_mask & (end_times <= incident.end)

        result: dict[str, Any] = {
            "resource_id": incident.resource_id,
            "type": incident.type,
            "detected": False,
            "first_detection_time": None,
            "lead_time_seconds": None,
            "max_error_in_window": None,
        }

        if scan_mask.any():
            order = np.argsort(end_values[scan_mask], kind="stable")
            scan_err = errors[scan_mask][order]
            scan_ts = end_times[scan_mask][order]
            flags = scan_err > threshold
            idx = _first_sustained(flags, sustain)
            if idx is not None:
                detection_time = scan_ts[idx]
                result["detected"] = True
                result["first_detection_time"] = detection_time.isoformat()
                result["lead_time_seconds"] = (
                    incident.start - detection_time
                ).total_seconds()

        in_window = (
            resource_mask
            & (end_times >= incident.start)
            & (end_times <= incident.end)
        )
        if in_window.any():
            result["max_error_in_window"] = float(errors[in_window].max())

        results.append(result)
    return results


def compute_threshold(
    errors: np.ndarray,
    end_times: pd.DatetimeIndex,
    incidents: list[Incident],
    reference_errors: np.ndarray | None,
    quantile: float,
) -> tuple[float, str, np.ndarray]:
    """Derive the anomaly threshold from healthy windows only.

    Healthy windows come from a separate reference capture when one is given,
    otherwise from the capture windows that end strictly before the earliest
    incident start. The incident windows themselves are never used, so a large
    incident cannot inflate its own threshold.

    Args:
        errors: Per-window errors for the incident capture.
        end_times: Per-window end timestamps.
        incidents: Labeled incidents (for the temporal split).
        reference_errors: Optional healthy errors from a reference capture.
        quantile: Threshold quantile over healthy errors.

    Returns:
        Tuple of (threshold, source, healthy_errors).

    Raises:
        ValueError: If there are no healthy windows to threshold from.
    """
    if reference_errors is not None:
        healthy = reference_errors
        source = "reference"
    else:
        earliest = min(incident.start for incident in incidents)
        healthy_mask = end_times < earliest
        healthy = errors[healthy_mask]
        source = "healthy_split"

    if healthy.size == 0:
        raise ValueError(
            "No healthy windows available to compute the threshold. "
            + (
                "The reference capture produced no windows."
                if source == "reference"
                else "No capture windows end before the earliest incident start; "
                "provide a --reference healthy capture instead."
            )
        )

    threshold = float(np.quantile(healthy, quantile))
    return threshold, source, healthy


def analyze(
    model_path: str,
    data: str,
    labels_path: str,
    profile: str,
    *,
    threshold_quantile: float = 0.99,
    sustain: int = 3,
    reference: str | None = None,
    data_format: str | None = None,
) -> dict[str, Any]:
    """Run the full incident-validation analysis and return the summary.

    Args:
        model_path: Path to the keeper checkpoint.
        data: Incident capture URI or path.
        labels_path: Path to the incident labels JSON.
        profile: Feature profile for the capture.
        threshold_quantile: Healthy-window quantile for the threshold.
        sustain: Consecutive anomalous windows required for a detection.
        reference: Optional healthy reference capture URI/path.
        data_format: Optional explicit file format override.

    Returns:
        The summary dict (also suitable for JSON serialization).
    """
    keeper = load_keeper(model_path)
    incidents = load_incidents(labels_path)

    if keeper.profile and keeper.profile != profile:
        print(
            f"warning: --profile '{profile}' differs from the checkpoint profile "
            f"'{keeper.profile}'; using the checkpoint feature order for alignment.",
            file=sys.stderr,
        )

    set_active_profile(profile)
    seq_len = int(keeper.config["seq_len"])
    step = int(get_config().window_step)

    df_long = asyncio.run(_fetch_long(data, profile, data_format))
    windows = build_windows(df_long, keeper, seq_len, step)
    if windows.x_num.shape[0] == 0:
        raise ValueError(
            f"Capture {data} produced no windows for profile '{profile}'. "
            "Check the time range, profile, and that the metrics are present."
        )
    errors = reconstruction_errors(keeper.model, windows.x_num, windows.x_cat, keeper.device)

    reference_errors: np.ndarray | None = None
    if reference is not None:
        ref_long = asyncio.run(_fetch_long(reference, profile, data_format))
        ref_windows = build_windows(ref_long, keeper, seq_len, step)
        if ref_windows.x_num.shape[0] == 0:
            raise ValueError(
                f"Reference capture {reference} produced no windows for profile '{profile}'."
            )
        reference_errors = reconstruction_errors(
            keeper.model, ref_windows.x_num, ref_windows.x_cat, keeper.device
        )

    threshold, source, healthy = compute_threshold(
        errors, windows.end_times, incidents, reference_errors, threshold_quantile
    )
    healthy_fpr = float(np.mean(healthy > threshold)) if healthy.size else 0.0

    incident_results = evaluate_incidents(
        errors, windows.resource_ids, windows.end_times, incidents, threshold, sustain
    )

    return {
        "threshold": threshold,
        "threshold_source": source,
        "threshold_quantile": threshold_quantile,
        "sustain": sustain,
        "n_windows": int(errors.size),
        "n_healthy_windows": int(healthy.size),
        "healthy_fpr": healthy_fpr,
        "incidents": incident_results,
    }


def write_plot(
    plot_path: str,
    errors: np.ndarray,
    end_times: pd.DatetimeIndex,
    threshold: float,
    incidents: list[Incident],
) -> bool:
    """Write a reconstruction-error timeline with the threshold and incident spans.

    Degrades gracefully: if matplotlib is not installed, prints a note and
    returns False instead of raising.

    Args:
        plot_path: Output image path.
        errors: Per-window errors.
        end_times: Per-window end timestamps.
        threshold: Anomaly threshold.
        incidents: Labeled incidents to shade.

    Returns:
        True if the figure was written, else False.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("note: matplotlib not installed; skipping --plot.", file=sys.stderr)
        return False

    order = np.argsort(end_times.values, kind="stable")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(end_times[order], errors[order], linewidth=0.8, label="reconstruction error")
    ax.axhline(threshold, color="red", linestyle="--", label="threshold")
    for incident in incidents:
        ax.axvspan(incident.start, incident.end, color="orange", alpha=0.2)
    ax.set_xlabel("window end time (UTC)")
    ax.set_ylabel("MSE reconstruction error")
    ax.set_title("Incident validation: per-window reconstruction error")
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    return True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate a keeper X-DEC model against an incident capture."
    )
    parser.add_argument("--model", required=True, help="Path to the keeper checkpoint (.pt).")
    parser.add_argument("--data", required=True, help="Incident capture URI or path.")
    parser.add_argument(
        "--labels",
        required=True,
        help="JSON list of incidents: {resource_id, type, start, end} (UTC ISO8601).",
    )
    parser.add_argument(
        "--profile", required=True, help="Feature profile (see config/features.yaml)."
    )
    parser.add_argument(
        "--threshold-quantile",
        type=float,
        default=0.99,
        help="Healthy-window quantile for the anomaly threshold (default: 0.99).",
    )
    parser.add_argument(
        "--sustain",
        type=int,
        default=3,
        help="Consecutive anomalous windows required for a detection (default: 3).",
    )
    parser.add_argument(
        "--reference",
        default=None,
        help="Optional healthy reference capture for the threshold instead of a temporal split.",
    )
    parser.add_argument(
        "--format", choices=["parquet", "csv"], default=None, help="Override the file format."
    )
    parser.add_argument("--plot", default=None, help="Optional path to write a timeline figure.")
    parser.add_argument("--output", default=None, help="Optional path to write the JSON summary.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)

    summary = analyze(
        args.model,
        args.data,
        args.labels,
        args.profile,
        threshold_quantile=args.threshold_quantile,
        sustain=args.sustain,
        reference=args.reference,
        data_format=args.format,
    )

    payload = json.dumps(summary, indent=2)
    print(payload)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(payload)

    if args.plot:
        keeper = load_keeper(args.model)
        set_active_profile(args.profile)
        seq_len = int(keeper.config["seq_len"])
        step = int(get_config().window_step)
        df_long = asyncio.run(_fetch_long(args.data, args.profile, args.format))
        windows = build_windows(df_long, keeper, seq_len, step)
        errors = reconstruction_errors(
            keeper.model, windows.x_num, windows.x_cat, keeper.device
        )
        incidents = load_incidents(args.labels)
        write_plot(args.plot, errors, windows.end_times, summary["threshold"], incidents)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
