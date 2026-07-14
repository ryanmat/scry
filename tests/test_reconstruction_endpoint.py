# Description: Tests for the reconstruction-anomaly endpoints (/anomaly/reconstruction[/lookup]).
# Description: Covers threshold resolution, the dict-vs-capture scale guard, and lookup plumbing.

"""Tests for the reconstruction-anomaly serving path.

The load-bearing case is ``test_dict_and_capture_paths_agree``: the serving
endpoint builds its window from request series (the dict path) while the baked
threshold is derived from the capture path; if those two disagree on the error of
one identical window, the ratio would compare a numerator and a threshold on
different scales.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from scry.api.predictor import Predictor, _severity_from_ratio
from scry.data.windowing import build_windows
from scry.model.checkpoint import load_keeper
from scry.model.reconstruction import reconstruction_errors
from scry.utils.config import get_config

_KEYS = (
    "resource_id",
    "reconstruction_error",
    "threshold",
    "ratio",
    "is_anomaly",
    "severity",
    "coverage",
    "timestamp",
)


def _make_client(model_path: str) -> TestClient:
    from scry.api.main import create_app

    return TestClient(create_app(model_path=model_path))


def _body(resource_id: str = "node-a") -> dict:
    return {
        "resource_id": resource_id,
        "numerical_metrics": {"cpuUsageNanoCores": [1e8] * 30},
        "categorical_metrics": {},
    }


# -- POST /anomaly/reconstruction --


def test_post_happy_returns_all_keys_and_ratio(serving_keeper_path: str, monkeypatch) -> None:
    """With a baked threshold, the POST returns every key and a computed ratio."""
    monkeypatch.delenv("SCRY_RECON_THRESHOLD", raising=False)
    client = _make_client(serving_keeper_path)
    resp = client.post("/anomaly/reconstruction", json=_body())
    assert resp.status_code == 200
    data = resp.json()
    for key in _KEYS:
        assert key in data, f"missing key {key}"
    assert data["threshold"] is not None
    assert data["ratio"] is not None
    assert data["reconstruction_error"] >= 0.0
    assert 1 <= data["severity"] <= 4
    assert 0.0 <= data["coverage"] <= 1.0


def test_post_severity_matches_ratio(serving_keeper_path: str, monkeypatch) -> None:
    """Severity band is consistent with the returned ratio."""
    monkeypatch.delenv("SCRY_RECON_THRESHOLD", raising=False)
    client = _make_client(serving_keeper_path)
    data = client.post("/anomaly/reconstruction", json=_body()).json()
    ratio, severity = data["ratio"], data["severity"]
    expected = 4 if ratio > 2.0 else 3 if ratio > 1.5 else 2 if ratio > 1.0 else 1
    assert severity == expected
    assert data["is_anomaly"] is (ratio > 1.0)


def test_post_model_not_loaded_returns_503(tmp_path: Path) -> None:
    """A missing checkpoint degrades to unhealthy; the endpoint returns 503."""
    client = _make_client(str(tmp_path / "missing.pt"))
    resp = client.post("/anomaly/reconstruction", json=_body())
    assert resp.status_code == 503


def test_post_empty_numerical_metrics_returns_422(serving_keeper_path: str) -> None:
    """numerical_metrics is required and non-empty."""
    client = _make_client(serving_keeper_path)
    resp = client.post(
        "/anomaly/reconstruction",
        json={"resource_id": "n", "numerical_metrics": {}, "categorical_metrics": {}},
    )
    assert resp.status_code == 422


# -- Threshold resolution --


def test_no_serving_block_degrades_gracefully(keeper_path: str, monkeypatch) -> None:
    """A checkpoint with no serving block returns the raw error, null ratio, not anomalous."""
    monkeypatch.delenv("SCRY_RECON_THRESHOLD", raising=False)
    client = _make_client(keeper_path)
    data = client.post("/anomaly/reconstruction", json=_body()).json()
    assert data["threshold"] is None
    assert data["ratio"] is None
    assert data["is_anomaly"] is False
    assert data["reconstruction_error"] >= 0.0


def test_env_override_wins(serving_keeper_path: str, monkeypatch) -> None:
    """SCRY_RECON_THRESHOLD overrides the baked serving threshold."""
    monkeypatch.setenv("SCRY_RECON_THRESHOLD", "0.5")
    client = _make_client(serving_keeper_path)
    data = client.post("/anomaly/reconstruction", json=_body()).json()
    assert data["threshold"] == 0.5


def test_invalid_env_override_falls_back_to_serving(serving_keeper_path: str, monkeypatch) -> None:
    """A non-numeric env override is ignored; the serving block threshold is used."""
    monkeypatch.setenv("SCRY_RECON_THRESHOLD", "not-a-number")
    client = _make_client(serving_keeper_path)
    data = client.post("/anomaly/reconstruction", json=_body()).json()
    assert data["threshold"] is not None
    assert data["threshold"] != 0.0


# -- The scale-consistency guard --


def test_dict_and_capture_paths_agree(serving_keeper_path: str, monkeypatch) -> None:
    """The serving (dict) path and the bake/validation (capture) path score one
    identical full window to the same reconstruction error."""
    monkeypatch.delenv("SCRY_RECON_THRESHOLD", raising=False)
    predictor = Predictor(serving_keeper_path)
    keeper = load_keeper(serving_keeper_path)
    seq_len = int(keeper.config["seq_len"])

    rng = np.random.default_rng(0)
    numerical = {
        name: (1e8 + rng.normal(0, 1e6, seq_len)).tolist() for name in keeper.numerical_features
    }
    categorical = {name: [1] * seq_len for name in keeper.categorical_features}

    error_dict = predictor.reconstruction_error(numerical, categorical)["reconstruction_error"]

    ts = pd.date_range("2026-01-01T00:00:00Z", periods=seq_len, freq="1min")
    rows = [
        {"resource_id": "n", "metric_name": name, "timestamp": ts[i], "value": float(v)}
        for name, values in {**numerical, **categorical}.items()
        for i, v in enumerate(values)
    ]
    windows = build_windows(
        pd.DataFrame(rows),
        numerical_features=keeper.numerical_features,
        categorical_features=keeper.categorical_features,
        normalization=keeper.normalization,
        cat_normalization=keeper.cat_normalization,
        seq_len=seq_len,
        step=int(get_config().window_step),
    )
    assert windows.x_num.shape[0] == 1
    error_capture = float(
        reconstruction_errors(keeper.model, windows.x_num, windows.x_cat, keeper.device)[0]
    )

    assert error_dict == pytest.approx(error_capture, rel=1e-4, abs=1e-8)


# -- GET /anomaly/reconstruction/lookup --


def test_lookup_happy_path(serving_keeper_path: str) -> None:
    """A configured source with usable metrics yields a 200 score."""
    df = pd.DataFrame({"resource_id": ["my-node"]})
    canned = {
        "reconstruction_error": 0.1,
        "threshold": 0.12,
        "ratio": 0.83,
        "is_anomaly": False,
        "severity": 1,
        "coverage": 1.0,
    }
    client = _make_client(serving_keeper_path)
    with (
        patch("scry.api.main._resource_metrics", new=AsyncMock(return_value=df)),
        patch.object(Predictor, "score_reconstruction", return_value=canned),
    ):
        resp = client.get("/anomaly/reconstruction/lookup?resource_id=my-node")
    assert resp.status_code == 200
    data = resp.json()
    assert data["resource_id"] == "my-node"
    assert data["ratio"] == 0.83
    for key in _KEYS:
        assert key in data


def test_lookup_no_source_returns_503(serving_keeper_path: str) -> None:
    """With no data source configured, the endpoint returns an honest 503."""
    client = _make_client(serving_keeper_path)
    with patch("scry.api.main._resource_metrics", new=AsyncMock(return_value=None)):
        resp = client.get("/anomaly/reconstruction/lookup?resource_id=n")
    assert resp.status_code == 503
    assert "No data source configured" in resp.json()["detail"]


def test_lookup_not_found_returns_404(serving_keeper_path: str) -> None:
    """A configured source with no matching metrics returns 404."""
    client = _make_client(serving_keeper_path)
    with patch(
        "scry.api.main._resource_metrics", new=AsyncMock(return_value=pd.DataFrame())
    ):
        resp = client.get("/anomaly/reconstruction/lookup?resource_id=ghost")
    assert resp.status_code == 404


def test_lookup_ambiguous_returns_409_without_scoring(serving_keeper_path: str) -> None:
    """An ambiguous resource id is refused (409) before any scoring."""
    df = pd.DataFrame(
        {
            "resource_id": ["node-a", "node-b"],
            "host_name": ["h", "h"],
            "metric_name": ["m", "m"],
            "value": [1.0, 1.0],
        }
    )
    client = _make_client(serving_keeper_path)
    with (
        patch("scry.api.main._resource_metrics", new=AsyncMock(return_value=df)),
        patch.object(Predictor, "score_reconstruction") as mock_score,
    ):
        resp = client.get("/anomaly/reconstruction/lookup?resource_id=node")
    assert resp.status_code == 409
    assert len(resp.json()["detail"]["candidates"]) == 2
    mock_score.assert_not_called()


def test_lookup_missing_resource_id_returns_422(serving_keeper_path: str) -> None:
    """The lookup requires a resource_id query parameter."""
    client = _make_client(serving_keeper_path)
    resp = client.get("/anomaly/reconstruction/lookup")
    assert resp.status_code == 422


# -- Short / heterogeneous windows (the divergence class the review caught) --


def test_short_window_is_not_scored_not_a_false_anomaly(
    serving_keeper_path: str, monkeypatch
) -> None:
    """A sub-seq_len window is reported not-scored, never a padded false critical."""
    monkeypatch.delenv("SCRY_RECON_THRESHOLD", raising=False)
    client = _make_client(serving_keeper_path)
    resp = client.post(
        "/anomaly/reconstruction",
        json={
            "resource_id": "n",
            "numerical_metrics": {"cpuUsageNanoCores": [1e8, 1e8, 1e8]},
            "categorical_metrics": {},
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["reconstruction_error"] is None
    assert data["ratio"] is None
    assert data["is_anomaly"] is False
    assert data["severity"] == 1


def test_insufficient_distinct_timestamps_not_scored(serving_keeper_path: str) -> None:
    """A frame with fewer than seq_len distinct timestamps scores as not-scored."""
    predictor = Predictor(serving_keeper_path)
    seq_len = int(predictor.config["seq_len"])
    ts = pd.date_range("2026-01-01T00:00:00Z", periods=seq_len - 5, freq="1min")
    rows = [
        {"resource_id": "n", "metric_name": "cpuUsageNanoCores", "timestamp": t, "value": 1e8}
        for t in ts
    ]
    result = predictor.score_reconstruction(pd.DataFrame(rows))
    assert result["reconstruction_error"] is None
    assert result["ratio"] is None
    assert result["is_anomaly"] is False


def test_lookup_frame_scored_on_shared_grid(serving_keeper_path: str) -> None:
    """Heterogeneous per-metric cadence is windowed on one shared grid (build_windows),
    not tail-sliced per metric; the score matches windowing the recent slice directly."""
    predictor = Predictor(serving_keeper_path)
    seq_len = int(predictor.config["seq_len"])
    ts = pd.date_range("2026-01-01T00:00:00Z", periods=seq_len + 20, freq="1min")
    rows = []
    for i, t in enumerate(ts):
        rows.append(
            {"resource_id": "n", "metric_name": "cpuUsageNanoCores", "timestamp": t, "value": 1e8 + i}
        )
        if i % 5 == 0:  # fsUsedBytes only every 5 minutes: a genuine cadence gap
            rows.append(
                {"resource_id": "n", "metric_name": "fsUsedBytes", "timestamp": t, "value": 1e9}
            )
    df = pd.DataFrame(rows)

    result = predictor.score_reconstruction(df)
    assert result["reconstruction_error"] is not None
    assert result["ratio"] is not None

    ts_all = pd.to_datetime(df["timestamp"], utc=True)
    recent = set(np.sort(ts_all.unique())[-seq_len:])
    df_recent = df[ts_all.isin(recent)]
    w = build_windows(
        df_recent,
        numerical_features=predictor.numerical_features,
        categorical_features=predictor.categorical_features,
        normalization=predictor.normalization,
        cat_normalization=predictor.cat_normalization,
        seq_len=seq_len,
        step=1,
    )
    expected = float(
        reconstruction_errors(predictor.model, w.x_num, w.x_cat, predictor.device)[
            int(np.argmax(w.end_times.values))
        ]
    )
    assert result["reconstruction_error"] == pytest.approx(expected, rel=1e-6, abs=1e-9)


# -- Severity bands, coverage, threshold hygiene --


def test_severity_from_ratio_band_boundaries() -> None:
    """The severity bands use strict '>' at 1.0 / 1.5 / 2.0."""
    assert _severity_from_ratio(0.5) == 1
    assert _severity_from_ratio(1.0) == 1  # at threshold is not yet anomalous
    assert _severity_from_ratio(1.0001) == 2
    assert _severity_from_ratio(1.5) == 2
    assert _severity_from_ratio(1.5001) == 3
    assert _severity_from_ratio(2.0) == 3
    assert _severity_from_ratio(2.0001) == 4


def test_coverage_is_exact_fraction(serving_keeper_path: str, monkeypatch) -> None:
    """Coverage is the fraction of the model's numerical features present (1 of 3 here)."""
    monkeypatch.delenv("SCRY_RECON_THRESHOLD", raising=False)
    client = _make_client(serving_keeper_path)
    data = client.post(
        "/anomaly/reconstruction",
        json={
            "resource_id": "n",
            "numerical_metrics": {"cpuUsageNanoCores": [1e8] * 30},
            "categorical_metrics": {},
        },
    ).json()
    assert data["coverage"] == pytest.approx(1 / 3)


def test_zero_numerical_coverage_is_not_scored(serving_keeper_path: str) -> None:
    """A window with only categorical data (no numerical) is not-scored, not fake-healthy.

    This is the collection-outage case: an all-neutral numerical branch would score
    as healthy, so it must be reported not-scored with coverage 0 instead.
    """
    predictor = Predictor(serving_keeper_path)
    seq_len = int(predictor.config["seq_len"])
    cat = predictor.categorical_features[0]
    ts = pd.date_range("2026-01-01T00:00:00Z", periods=seq_len + 5, freq="1min")
    rows = [{"resource_id": "n", "metric_name": cat, "timestamp": t, "value": 1.0} for t in ts]

    result = predictor.score_reconstruction(pd.DataFrame(rows))
    assert result["reconstruction_error"] is None
    assert result["ratio"] is None
    assert result["is_anomaly"] is False
    assert result["coverage"] == 0.0


def test_coverage_reflects_scored_window_not_whole_frame(serving_keeper_path: str) -> None:
    """A numerical feature that stops reporting before the scored window counts as absent."""
    predictor = Predictor(serving_keeper_path)
    seq_len = int(predictor.config["seq_len"])
    num = predictor.numerical_features
    ts = pd.date_range("2026-01-01T00:00:00Z", periods=seq_len + 10, freq="1min")
    rows = []
    for i, t in enumerate(ts):
        rows.append({"resource_id": "n", "metric_name": num[0], "timestamp": t, "value": 1e8})
        if i < 5:  # a second feature only in the oldest rows, outside the last seq_len window
            rows.append({"resource_id": "n", "metric_name": num[1], "timestamp": t, "value": 5e8})

    result = predictor.score_reconstruction(pd.DataFrame(rows))
    assert result["reconstruction_error"] is not None
    # Only num[0] falls inside the scored window, so coverage is 1 / n_numerical,
    # not the 2 / n_numerical the full frame would report.
    assert result["coverage"] == pytest.approx(1 / len(num))


def test_nonpositive_env_threshold_is_rejected(keeper_path: str, monkeypatch) -> None:
    """A non-positive env threshold is rejected (no serving block to fall back to);
    health and scoring agree the threshold is unset rather than 0."""
    monkeypatch.setenv("SCRY_RECON_THRESHOLD", "0")
    client = _make_client(keeper_path)
    assert client.get("/health/detailed").json()["recon_threshold"] is None
    data = client.post("/anomaly/reconstruction", json=_body()).json()
    assert data["threshold"] is None
    assert data["ratio"] is None


def test_missing_threshold_warns_at_load(keeper_path: str, monkeypatch, caplog) -> None:
    """Serving a model with no threshold logs a WARN so the disarmed state is visible."""
    import logging

    monkeypatch.delenv("SCRY_RECON_THRESHOLD", raising=False)
    with caplog.at_level(logging.WARNING, logger="scry.api.predictor"):
        Predictor(keeper_path)
    assert any("no reconstruction threshold configured" in r.message for r in caplog.records)


# -- Datasource contract --


def test_response_keys_cover_datasource(serving_keeper_path: str, monkeypatch) -> None:
    """The keys the repointed Scry_Anomaly Groovy parses are present."""
    monkeypatch.delenv("SCRY_RECON_THRESHOLD", raising=False)
    client = _make_client(serving_keeper_path)
    data = client.post("/anomaly/reconstruction", json=_body()).json()
    for key in ("ratio", "is_anomaly", "severity"):
        assert key in data


# -- per-resource threshold resolution --


def _keeper_with_per_resource(serving_keeper_path: str, tmp_path: Path, per_resource: dict) -> str:
    """Copy the serving keeper with a per_resource map injected into its serving block."""
    import torch

    ckpt = torch.load(serving_keeper_path, map_location="cpu", weights_only=False)
    ckpt["serving"] = dict(ckpt["serving"], per_resource=per_resource, margin_multiplier=2.0)
    out = str(tmp_path / "per_resource_keeper.pt")
    torch.save(ckpt, out)
    return out


def _scoreable_frame(predictor: Predictor, resource_id: str) -> pd.DataFrame:
    seq_len = int(predictor.config["seq_len"])
    ts = pd.date_range("2026-01-01T00:00:00Z", periods=seq_len + 5, freq="1min")
    rows = [
        {"resource_id": resource_id, "metric_name": "cpuUsageNanoCores", "timestamp": t, "value": 1e8}
        for t in ts
    ]
    return pd.DataFrame(rows)


def test_per_resource_threshold_used_for_known_resource(
    serving_keeper_path: str, tmp_path: Path, monkeypatch
) -> None:
    """A resource in the per_resource map is scored against its own threshold."""
    monkeypatch.delenv("SCRY_RECON_THRESHOLD", raising=False)
    path = _keeper_with_per_resource(serving_keeper_path, tmp_path, {"node-a": 0.007})
    predictor = Predictor(path)
    result = predictor.score_reconstruction(
        _scoreable_frame(predictor, "node-a"), resource_id="node-a"
    )
    assert result["threshold"] == pytest.approx(0.007)
    assert result["ratio"] == pytest.approx(result["reconstruction_error"] / 0.007)


def test_unknown_resource_falls_back_to_global(
    serving_keeper_path: str, tmp_path: Path, monkeypatch
) -> None:
    """A resource missing from the map is scored against the global serving threshold."""
    monkeypatch.delenv("SCRY_RECON_THRESHOLD", raising=False)
    path = _keeper_with_per_resource(serving_keeper_path, tmp_path, {"node-a": 0.007})
    predictor = Predictor(path)
    global_threshold = predictor.recon_threshold
    result = predictor.score_reconstruction(
        _scoreable_frame(predictor, "node-z"), resource_id="node-z"
    )
    assert result["threshold"] == pytest.approx(global_threshold)


def test_env_override_beats_per_resource(
    serving_keeper_path: str, tmp_path: Path, monkeypatch
) -> None:
    """SCRY_RECON_THRESHOLD overrides the per-resource map as well as the global."""
    monkeypatch.setenv("SCRY_RECON_THRESHOLD", "0.5")
    path = _keeper_with_per_resource(serving_keeper_path, tmp_path, {"node-a": 0.007})
    predictor = Predictor(path)
    result = predictor.score_reconstruction(
        _scoreable_frame(predictor, "node-a"), resource_id="node-a"
    )
    assert result["threshold"] == pytest.approx(0.5)


def test_no_resource_context_uses_global(
    serving_keeper_path: str, tmp_path: Path, monkeypatch
) -> None:
    """The body path (no resolved resource identity) keeps the global threshold."""
    monkeypatch.delenv("SCRY_RECON_THRESHOLD", raising=False)
    path = _keeper_with_per_resource(serving_keeper_path, tmp_path, {"node-a": 0.007})
    predictor = Predictor(path)
    global_threshold = predictor.recon_threshold
    result = predictor.score_reconstruction(_scoreable_frame(predictor, "node-a"))
    assert result["threshold"] == pytest.approx(global_threshold)


def test_invalid_per_resource_entries_dropped(
    serving_keeper_path: str, tmp_path: Path, monkeypatch, caplog
) -> None:
    """Non-numeric and non-positive per-resource entries are dropped with a warning."""
    monkeypatch.delenv("SCRY_RECON_THRESHOLD", raising=False)
    path = _keeper_with_per_resource(
        serving_keeper_path, tmp_path, {"node-a": -1.0, "node-b": "junk"}
    )
    with caplog.at_level("WARNING", logger="scry.api.predictor"):
        predictor = Predictor(path)
    assert predictor.recon_thresholds_per_resource == {}
    assert "per-resource" in caplog.text.lower()


def test_lookup_passes_resolved_resource_id(serving_keeper_path: str) -> None:
    """The lookup endpoint scores with the canonical resolved id, not the query string."""
    df = pd.DataFrame({"resource_id": ["full-canonical-node-name"]})
    canned = {
        "reconstruction_error": 0.1,
        "threshold": 0.12,
        "ratio": 0.83,
        "is_anomaly": False,
        "severity": 1,
        "coverage": 1.0,
    }
    client = _make_client(serving_keeper_path)
    with (
        patch("scry.api.main._resource_metrics", new=AsyncMock(return_value=df)),
        patch.object(Predictor, "score_reconstruction", return_value=canned) as mock_score,
    ):
        resp = client.get("/anomaly/reconstruction/lookup?resource_id=canonical")
    assert resp.status_code == 200
    assert mock_score.call_args.kwargs["resource_id"] == "full-canonical-node-name"


def test_non_finite_per_resource_entries_dropped(
    serving_keeper_path: str, tmp_path: Path, monkeypatch, caplog
) -> None:
    """An inf per-resource threshold is dropped (it would silently disable detection)."""
    monkeypatch.delenv("SCRY_RECON_THRESHOLD", raising=False)
    path = _keeper_with_per_resource(serving_keeper_path, tmp_path, {"node-a": float("inf")})
    with caplog.at_level("WARNING", logger="scry.api.predictor"):
        predictor = Predictor(path)
    assert predictor.recon_thresholds_per_resource == {}


def test_invalid_env_warns_once_not_per_request(
    serving_keeper_path: str, tmp_path: Path, monkeypatch, caplog
) -> None:
    """An invalid env override warns once at init, not on every scored request."""
    monkeypatch.setenv("SCRY_RECON_THRESHOLD", "not-a-number")
    path = _keeper_with_per_resource(serving_keeper_path, tmp_path, {"node-a": 0.007})
    with caplog.at_level("WARNING", logger="scry.api.predictor"):
        predictor = Predictor(path)
        frame = _scoreable_frame(predictor, "node-a")
        for _ in range(3):
            result = predictor.score_reconstruction(frame, resource_id="node-a")
    env_warnings = [r for r in caplog.records if "SCRY_RECON_THRESHOLD" in r.getMessage()]
    assert len(env_warnings) == 1
    # The invalid override is treated as unset, so the per-resource threshold applies.
    assert result["threshold"] == pytest.approx(0.007)
