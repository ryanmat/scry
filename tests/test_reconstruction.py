# Description: Tests for the shared reconstruction core (windowing, per-window error, threshold).
# Description: Exercises scry.data.windowing and scry.model.reconstruction against a tiny keeper.

"""Deterministic unit tests for the reconstruction scoring core."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from synth import gen_capture

from scry.data.windowing import build_windows
from scry.model.checkpoint import load_keeper
from scry.model.reconstruction import healthy_threshold, reconstruction_errors, time_split
from scry.utils.config import get_config


def _as_long(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce the synthetic capture's string timestamps to UTC datetimes."""
    return df.assign(timestamp=pd.to_datetime(df["timestamp"], utc=True))


def _windows(keeper, df: pd.DataFrame):
    """Window a capture in the keeper's feature order with its stored normalization."""
    return build_windows(
        _as_long(df),
        numerical_features=keeper.numerical_features,
        categorical_features=keeper.categorical_features,
        normalization=keeper.normalization,
        cat_normalization=keeper.cat_normalization,
        seq_len=keeper.config["seq_len"],
        step=int(get_config().window_step),
    )


def _errors(keeper, df: pd.DataFrame) -> np.ndarray:
    w = _windows(keeper, df)
    return reconstruction_errors(keeper.model, w.x_num, w.x_cat, keeper.device)


def test_reconstruction_errors_shape_nonneg_finite(keeper_path: str) -> None:
    """One non-negative, finite error per window."""
    keeper = load_keeper(keeper_path)
    w = _windows(keeper, gen_capture("node-a", 200, seed=5)[0])
    errors = reconstruction_errors(keeper.model, w.x_num, w.x_cat, keeper.device)
    assert w.x_num.shape[0] > 0
    assert errors.shape == (w.x_num.shape[0],)
    assert np.all(errors >= 0.0)
    assert np.all(np.isfinite(errors))


def test_reconstruction_errors_are_deterministic(keeper_path: str) -> None:
    """Scoring uses the latent mean (no sampling), so repeated calls are identical."""
    keeper = load_keeper(keeper_path)
    a = _errors(keeper, gen_capture("node-a", 120, seed=6)[0])
    b = _errors(keeper, gen_capture("node-a", 120, seed=6)[0])
    np.testing.assert_allclose(a, b)


def test_spike_scores_higher_than_healthy(keeper_path: str) -> None:
    """A large step drives the per-window error above the healthy maximum."""
    keeper = load_keeper(keeper_path)
    healthy = _errors(keeper, gen_capture("node-a", 200, seed=7)[0])
    spiked = _errors(keeper, gen_capture("node-a", 200, seed=7, spike=(150, 200, 50.0))[0])
    assert spiked.max() > healthy.max()


def test_absent_numerical_feature_maps_to_neutral(keeper_path: str) -> None:
    """A feature missing from the capture keeps model width and is the neutral value 0."""
    keeper = load_keeper(keeper_path)
    missing = keeper.numerical_features[0]
    df, _ = gen_capture("node-a", 60, seed=8)
    df = df[df["metric_name"] != missing]

    w = _windows(keeper, df)
    assert w.x_num.shape[2] == len(keeper.numerical_features)
    col = keeper.numerical_features.index(missing)
    assert np.allclose(w.x_num[:, :, col].numpy(), 0.0)


def test_build_windows_too_short_is_empty(keeper_path: str) -> None:
    """A capture shorter than seq_len yields no windows and an empty error array."""
    keeper = load_keeper(keeper_path)
    w = _windows(keeper, gen_capture("node-a", 10, seed=9)[0])
    assert w.x_num.shape[0] == 0
    errors = reconstruction_errors(keeper.model, w.x_num, w.x_cat, keeper.device)
    assert errors.shape == (0,)


def test_healthy_threshold_monotonic_in_quantile() -> None:
    """A higher quantile never lowers the threshold."""
    errors = np.linspace(0.0, 1.0, 100)
    ts = pd.DatetimeIndex(pd.date_range("2026-01-01T00:00:00Z", periods=100, freq="1min"))
    t50, _, _ = healthy_threshold(errors, ts, quantile=0.5, gap=0)
    t99, _, _ = healthy_threshold(errors, ts, quantile=0.99, gap=0)
    assert t99 >= t50


def test_healthy_threshold_holds_out_eval_with_gap() -> None:
    """The fit is the earlier half; the eval is the later half minus the gap."""
    errors = np.arange(100, dtype=float)
    ts = pd.DatetimeIndex(pd.date_range("2026-01-01T00:00:00Z", periods=100, freq="1min"))
    _, fit, eval_ = healthy_threshold(errors, ts, quantile=0.9, gap=5)
    assert fit.size == 50
    assert eval_.size == 100 - 50 - 5


def test_healthy_threshold_raises_on_empty() -> None:
    """No healthy windows is a clear error, not a silent NaN threshold."""
    with pytest.raises(ValueError, match="No healthy windows"):
        healthy_threshold(np.zeros(0), pd.DatetimeIndex([], tz="UTC"), quantile=0.99, gap=0)


def test_time_split_drops_the_gap() -> None:
    """time_split leaves a gap of windows between the fit and eval halves."""
    errors = np.arange(20, dtype=float)
    ts = pd.DatetimeIndex(pd.date_range("2026-01-01T00:00:00Z", periods=20, freq="1min"))
    fit, eval_ = time_split(errors, ts, gap=3)
    assert fit.size == 10
    assert eval_.size == 20 - 10 - 3
