# Description: FastAPI application for the prediction service.
# Description: Exposes /health, /predict(/lookup), /anomaly/reconstruction(/lookup), /clusters, /forecast, /drift, /anomaly, /accuracy.

"""FastAPI application for cluster prediction."""

import atexit
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import FastAPI, HTTPException, Query

from scry.api.forecaster import Forecaster
from scry.api.predictor import ModelSchemaError, Predictor
from scry.api.schemas import (
    ClusterInfo,
    DetailedHealthResponse,
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    MetricForecast,
    PredictionRequest,
    PredictionResponse,
    ReconstructionRequest,
    ReconstructionResponse,
    get_cluster_info,
)
from scry.utils.config import get_config
from scry.utils.tracing import get_tracer, setup_tracing, shutdown_tracing

tracer = get_tracer(__name__)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# API version
VERSION = "0.1.0"

# Lookback for the reconstruction lookup: only the most recent seq_len samples are
# scored, so a short window keeps the object-store fetch small.
_RECON_LOOKBACK_DAYS = 7


def _data_uri() -> str | None:
    """The configured object-store data URI, if any (env SCRY_DATA_URI / config)."""
    return getattr(get_config(), "data_uri", None) or os.environ.get("SCRY_DATA_URI")


def _datasource_descriptor() -> str | None:
    """Describe the configured data source for diagnostics, or None if unconfigured."""
    uri = _data_uri()
    if uri:
        return f"object-store: {uri}"
    return None


def _resolve_rows(df: Any, resource_id: str) -> Any:
    """Select the rows for a resource: exact resource_id, then exact host_name,
    then a substring fallback. Case-insensitive and literal (non-regex).

    Non-raising. Returns a possibly-empty DataFrame. The caller is responsible
    for treating a match that spans more than one distinct resource as ambiguous.
    """
    needle = str(resource_id).lower()
    rid = df["resource_id"].astype(str).str.lower()

    exact_rid = df[rid == needle]
    if not exact_rid.empty:
        return exact_rid

    host = df["host_name"].astype(str).str.lower() if "host_name" in df.columns else None
    if host is not None:
        exact_host = df[host == needle]
        if not exact_host.empty:
            return exact_host

    mask = rid.str.contains(needle, regex=False, na=False)
    if host is not None:
        mask = mask | host.str.contains(needle, regex=False, na=False)
    return df[mask]


async def _resource_metrics(
    resource_id: str, lookback_days: int = 30, profile: str | None = None
) -> Any:
    """Fetch recent metrics for a resource from the configured object store.

    Returns a canonical-schema DataFrame filtered to the resource, or None when no
    object store is configured (``SCRY_DATA_URI``).

    ``profile`` selects which metric names to pull; when None it falls back to the
    ``SCRY_PROFILE`` env var (and then the features.yaml default).
    """
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)
    profile_name = profile if profile is not None else os.environ.get("SCRY_PROFILE")

    uri = _data_uri()
    if not uri:
        return None

    from scry.data import DataFetcher

    fetcher = DataFetcher.from_object_store(uri)
    df = await fetcher.get_metrics_dataframe(start, end, profile_name)

    if df.empty:
        return df

    return _resolve_rows(df, resource_id)


def _split_by_profile(
    df: Any,
    profile_name: str | None = None,
) -> tuple[dict[str, list[float]], dict[str, list[int]]]:
    """Group canonical metric rows into numerical/categorical series per the active profile."""
    from scry.config.loader import get_profile

    try:
        profile = get_profile(profile_name)
    except (FileNotFoundError, ValueError):
        return {}, {}

    num_set = set(profile.numerical_features)
    cat_set = set(profile.categorical_features)
    numerical: dict[str, list[float]] = {}
    categorical: dict[str, list[int]] = {}

    ordered = df.sort_values("timestamp")
    for name, group in ordered.groupby("metric_name"):
        values = group["value"].dropna().tolist()
        if not values:
            continue
        if name in num_set:
            numerical[name] = [float(v) for v in values]
        elif name in cat_set:
            categorical[name] = [int(v) for v in values]
    return numerical, categorical


def create_app(model_path: str | None = None) -> FastAPI:
    """Create the FastAPI application.

    Args:
        model_path: Path to the model file. If None, uses the MODEL_PATH env var.

    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(
        title="Scry Predictive API",
        description="Predicts infrastructure failure states from a stream of metrics",
        version=VERSION,
    )

    # Optional tracing (no-op unless the 'otel' extra is installed and enabled).
    setup_tracing(app)
    atexit.register(shutdown_tracing)

    # Load model
    if model_path is None:
        model_path = os.environ.get("MODEL_PATH", "models/xdec_model.pt")

    app.state.started_at = time.monotonic()
    app.state.model_path = model_path
    app.state.model_load_time_ms = None

    try:
        t0 = time.monotonic()
        predictor = Predictor(model_path=model_path)
        app.state.model_load_time_ms = (time.monotonic() - t0) * 1000.0
        app.state.predictor = predictor
        app.state.model_loaded = True
    except (FileNotFoundError, ModelSchemaError) as e:
        logger.warning("model not loaded: %s", e)
        app.state.predictor = None
        app.state.model_loaded = False

    # Forecaster is lazy-loaded on first /forecast request, independent of X-DEC.
    forecast_model_id = os.environ.get("FORECAST_MODEL_ID", "amazon/chronos-bolt-tiny")
    forecast_device = os.environ.get("FORECAST_DEVICE", "cpu")
    app.state.forecaster = Forecaster(
        model_id=forecast_model_id,
        device=forecast_device,
    )

    @app.get("/")
    def root() -> dict:
        """Root endpoint with API info."""
        return {"name": "Scry Predictive API", "version": VERSION, "docs": "/docs"}

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """Liveness/readiness check."""
        status = "healthy" if app.state.model_loaded else "unhealthy"
        return HealthResponse(
            status=status,
            model_loaded=app.state.model_loaded,
            version=VERSION,
        )

    @app.get("/health/detailed", response_model=DetailedHealthResponse)
    def health_detailed() -> DetailedHealthResponse:
        """Detailed health check: model metadata, configured data source, uptime."""
        status = "healthy" if app.state.model_loaded else "unhealthy"
        uptime = time.monotonic() - app.state.started_at

        model_version = None
        recon_threshold = None
        if app.state.predictor is not None:
            cfg = app.state.predictor.config
            model_version = f"xdec-k{cfg['n_clusters']}-d{cfg['latent_dim']}"
            recon_threshold = app.state.predictor.recon_threshold
        env_version = os.environ.get("MODEL_VERSION")
        if env_version:
            model_version = f"{model_version}-{env_version}" if model_version else env_version

        return DetailedHealthResponse(
            status=status,
            model_loaded=app.state.model_loaded,
            version=VERSION,
            model_load_time_ms=app.state.model_load_time_ms,
            model_version=model_version,
            model_path=app.state.model_path,
            datasource=_datasource_descriptor(),
            chronos_loaded=app.state.forecaster.is_loaded,
            recon_threshold=recon_threshold,
            uptime_seconds=round(uptime, 2),
        )

    @app.post("/predict", response_model=PredictionResponse)
    def predict(request: PredictionRequest) -> PredictionResponse:
        """Predict the operational state for metrics supplied in the request body.

        Raises:
            HTTPException: 503 if the model is not loaded.
        """
        if not app.state.model_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")

        result = app.state.predictor.predict(
            numerical_metrics=request.numerical_metrics,
            categorical_metrics=request.categorical_metrics,
        )

        return PredictionResponse(
            resource_id=request.resource_id,
            cluster_id=result["cluster_id"],
            cluster_name=result["cluster_name"],
            confidence=result["confidence"],
            action=result["action"],
            priority=result["priority"],
        )

    @app.get("/clusters", response_model=list[ClusterInfo])
    def clusters() -> list[ClusterInfo]:
        """Get all operational state (cluster) definitions."""
        return get_cluster_info()

    @app.get("/predict/lookup", response_model=PredictionResponse)
    async def predict_lookup(
        resource_id: str = Query(..., description="Resource id or hostname to look up"),
    ) -> PredictionResponse:
        """Look up a resource's recent metrics from the configured object store and predict.

        The object store is selected by ``SCRY_DATA_URI``.

        Raises:
            HTTPException: 503 if the model is not loaded or no source is configured,
                404 if the resource has no usable recent metrics, 502 on a source error.
        """
        with tracer.start_as_current_span("predict_lookup") as span:
            span.set_attribute("resource_id", resource_id)

            if not app.state.model_loaded:
                raise HTTPException(status_code=503, detail="Model not loaded")

            # The model's own training profile is authoritative for the num/cat
            # split, so a mismatched SCRY_PROFILE cannot silently misalign it.
            model_profile = app.state.predictor.feature_schema.get("profile")
            profile_name = model_profile or os.environ.get("SCRY_PROFILE")

            try:
                df = await _resource_metrics(resource_id, profile=profile_name)
            except Exception as e:
                logger.error("lookup data fetch failed: %s: %s", type(e).__name__, e)
                raise HTTPException(status_code=502, detail=f"data source error: {e}") from e

            if df is None:
                raise HTTPException(
                    status_code=503,
                    detail="No data source configured. Set SCRY_DATA_URI to an object-store URI.",
                )
            if df.empty:
                raise HTTPException(
                    status_code=404,
                    detail=f"No recent metrics found for resource '{resource_id}'",
                )

            # Refuse to silently pool multiple resources into one prediction.
            ids = sorted(df["resource_id"].astype(str).unique())
            if len(ids) > 1:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "message": (
                            f"Ambiguous resource '{resource_id}': matched {len(ids)} "
                            "resources. Specify the exact resource_id."
                        ),
                        "candidates": ids[:20],
                    },
                )

            numerical, categorical = _split_by_profile(df, profile_name)
            if not numerical:
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"No usable metrics for resource '{resource_id}' "
                        "under the active profile"
                    ),
                )

            span.set_attribute("numerical_metrics.count", len(numerical))
            span.set_attribute("categorical_metrics.count", len(categorical))
            result = app.state.predictor.predict(
                numerical_metrics=numerical,
                categorical_metrics=categorical,
            )
            return PredictionResponse(
                resource_id=resource_id,
                cluster_id=result["cluster_id"],
                cluster_name=result["cluster_name"],
                confidence=result["confidence"],
                action=result["action"],
                priority=result["priority"],
            )

    @app.post("/anomaly/reconstruction", response_model=ReconstructionResponse)
    def reconstruction(request: ReconstructionRequest) -> ReconstructionResponse:
        """Score the reconstruction anomaly for metrics supplied in the request body.

        This is the keeper's validated signal: the per-window reconstruction error
        as a ratio against the persisted healthy threshold. Portable; needs no data
        source.

        Raises:
            HTTPException: 503 if the model is not loaded.
        """
        if not app.state.model_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")

        result = app.state.predictor.reconstruction_error(
            numerical_metrics=request.numerical_metrics,
            categorical_metrics=request.categorical_metrics,
        )
        return ReconstructionResponse(
            resource_id=request.resource_id,
            reconstruction_error=result["reconstruction_error"],
            threshold=result["threshold"],
            ratio=result["ratio"],
            is_anomaly=result["is_anomaly"],
            severity=result["severity"],
            coverage=result["coverage"],
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    @app.get("/anomaly/reconstruction/lookup", response_model=ReconstructionResponse)
    async def reconstruction_lookup(
        resource_id: str = Query(..., description="Resource id or hostname to look up"),
    ) -> ReconstructionResponse:
        """Look up a resource's recent metrics from the object store and score reconstruction.

        Mirrors ``/predict/lookup``: pulls the resource's recent window through the
        object-store seam (``SCRY_DATA_URI``) and scores it. This is the path the
        LogicMonitor ``Scry_Anomaly`` DataSource polls.

        Raises:
            HTTPException: 503 if the model is not loaded or no source is configured,
                404 if the resource has no usable recent metrics, 409 if the id is
                ambiguous, 502 on a source error.
        """
        with tracer.start_as_current_span("reconstruction_lookup") as span:
            span.set_attribute("resource_id", resource_id)

            if not app.state.model_loaded:
                raise HTTPException(status_code=503, detail="Model not loaded")

            # The model's own training profile is authoritative for the num/cat split.
            model_profile = app.state.predictor.feature_schema.get("profile")
            profile_name = model_profile or os.environ.get("SCRY_PROFILE")

            try:
                df = await _resource_metrics(
                    resource_id, lookback_days=_RECON_LOOKBACK_DAYS, profile=profile_name
                )
            except Exception as e:
                logger.error(
                    "reconstruction lookup fetch failed: %s: %s", type(e).__name__, e
                )
                raise HTTPException(status_code=502, detail=f"data source error: {e}") from e

            if df is None:
                raise HTTPException(
                    status_code=503,
                    detail="No data source configured. Set SCRY_DATA_URI to an object-store URI.",
                )
            if df.empty:
                raise HTTPException(
                    status_code=404,
                    detail=f"No recent metrics found for resource '{resource_id}'",
                )

            # Refuse to silently pool multiple resources into one score.
            ids = sorted(df["resource_id"].astype(str).unique())
            if len(ids) > 1:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "message": (
                            f"Ambiguous resource '{resource_id}': matched {len(ids)} "
                            "resources. Specify the exact resource_id."
                        ),
                        "candidates": ids[:20],
                    },
                )

            # Score the fetched long-format frame directly through the shared
            # windowing (the same path the threshold was baked on), so the metrics
            # are aligned on one timestamp grid rather than flattened per metric.
            result = app.state.predictor.score_reconstruction(df)
            span.set_attribute("reconstruction.is_anomaly", result["is_anomaly"])
            return ReconstructionResponse(
                resource_id=resource_id,
                reconstruction_error=result["reconstruction_error"],
                threshold=result["threshold"],
                ratio=result["ratio"],
                is_anomaly=result["is_anomaly"],
                severity=result["severity"],
                coverage=result["coverage"],
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

    @app.post("/forecast", response_model=ForecastResponse)
    def forecast(request: ForecastRequest) -> ForecastResponse:
        """Forecast metric values at the requested horizons using Chronos.

        Independent of the X-DEC cluster model. Requires the ``forecast`` extra.
        """
        forecaster: Forecaster = app.state.forecaster

        if request.horizons != forecaster.horizons:
            forecaster.horizons = request.horizons

        metric_forecasts = forecaster.forecast_metrics(request.metrics)

        return ForecastResponse(
            resource_id=request.resource_id,
            forecasts=[MetricForecast(**mf) for mf in metric_forecasts],
            model_id=forecaster.model_id,
        )

    @app.get("/drift")
    def drift_status() -> dict:
        """Get current drift detection status (PSI feature drift, ADWIN prediction drift)."""
        if not hasattr(app.state, "drift_detector"):
            return {
                "feature_drift": {"has_drift": False, "message": "No reference data configured"},
                "prediction_drift": {"has_drift": False, "message": "No error history available"},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        detector = app.state.drift_detector
        return detector.get_drift_status(
            app.state.reference_data,
            app.state.current_data,
            app.state.error_stream,
        )

    @app.get("/anomaly")
    def anomaly_status() -> dict:
        """Get current forecast-based anomaly detection status."""
        if not hasattr(app.state, "anomaly_detector"):
            return {
                "is_anomaly": False,
                "anomaly_score": 0.0,
                "violated_metrics": [],
                "severity": "low",
                "metric_count": 0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        detector = app.state.anomaly_detector
        actuals = app.state.last_actuals
        forecast = app.state.last_forecast

        result = detector.detect(actuals, forecast)

        return {
            "is_anomaly": bool(result["is_anomaly"]),
            "anomaly_score": float(result["anomaly_score"]),
            "violated_metrics": result["violated_metrics"],
            "severity": result["severity"],
            "metric_count": len(detector.metric_names),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @app.get("/accuracy")
    def accuracy_status() -> dict:
        """Get forecast accuracy and cluster stability metrics as flat key=value pairs."""
        start_ms = time.monotonic_ns() // 1_000_000

        if not hasattr(app.state, "accuracy_tracker"):
            elapsed = (time.monotonic_ns() // 1_000_000) - start_ms
            fallback: dict = {}
            for metric in ["Picp", "Mae", "Mase", "Mpiw"]:
                for horizon in ["15m", "1h", "4h", "24h"]:
                    fallback[f"{metric}{horizon}"] = 0.0
            fallback["TransitionRate"] = 0.0
            fallback["ConfidenceStd"] = 0.0
            fallback["DominantClusterPct"] = 0.0
            fallback["ObservationCount"] = 0
            fallback["ApiStatus"] = 1
            fallback["ApiLatencyMs"] = elapsed
            fallback["timestamp"] = datetime.now(timezone.utc).isoformat()
            return fallback

        tracker = app.state.accuracy_tracker
        metrics = tracker.compute_metrics()
        elapsed = (time.monotonic_ns() // 1_000_000) - start_ms

        result: dict = {}
        for metric_key, flat_prefix in [
            ("picp", "Picp"),
            ("mae", "Mae"),
            ("mase", "Mase"),
            ("mpiw", "Mpiw"),
        ]:
            for horizon in metrics["horizons"]:
                val = metrics["horizons"][horizon][metric_key]
                if isinstance(val, float) and (val != val):  # NaN check
                    val = 0.0
                result[f"{flat_prefix}{horizon}"] = round(val, 4)

        result["TransitionRate"] = round(metrics["stability"]["transition_rate"], 4)
        result["ConfidenceStd"] = round(metrics["stability"]["confidence_std"], 4)
        result["DominantClusterPct"] = round(metrics["stability"]["dominant_cluster_pct"], 1)
        result["ObservationCount"] = metrics["observation_count"]
        result["ApiStatus"] = 1
        result["ApiLatencyMs"] = elapsed
        result["timestamp"] = metrics["timestamp"]

        return result

    return app


# Default app instance for uvicorn
app = create_app()
