# Description: Optional OpenTelemetry tracing for the Scry API.
# Description: Degrades to no-ops when the 'otel' extra is absent or tracing is disabled.

"""Optional OpenTelemetry tracing.

Active only when the ``otel`` extra is installed and ``OTEL_TRACING_ENABLED`` is
not "false". Without OpenTelemetry installed, every helper here is a no-op so the
rest of the app runs unchanged. Configure via ``OTEL_SERVICE_NAME``,
``OTEL_EXPORTER_TYPE`` (otlp|console), ``OTEL_EXPORTER_OTLP_ENDPOINT``,
``OTEL_EXPORTER_OTLP_HEADERS``, and ``OTEL_TRACES_SAMPLER_ARG``.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)

try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    from opentelemetry.instrumentation.logging import LoggingInstrumentor
    from opentelemetry.sdk.resources import SERVICE_NAME, SERVICE_VERSION, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False

SERVICE_VERSION_VALUE = "0.1.0"


class _NoopSpan:
    """A span that records nothing, used when tracing is unavailable."""

    def set_attribute(self, *args: Any, **kwargs: Any) -> None:
        pass

    def record_exception(self, *args: Any, **kwargs: Any) -> None:
        pass

    def is_recording(self) -> bool:
        return False


class _NoopTracer:
    """A tracer whose spans do nothing."""

    @contextmanager
    def start_as_current_span(
        self, name: str, *args: Any, **kwargs: Any
    ) -> Iterator[_NoopSpan]:
        yield _NoopSpan()


_NOOP_TRACER = _NoopTracer()


def get_tracer(name: str = __name__):
    """Return a real tracer, or a no-op tracer when OpenTelemetry is not installed."""
    if _OTEL_AVAILABLE:
        return trace.get_tracer(name)
    return _NOOP_TRACER


def _parse_headers(raw: str) -> dict:
    """Parse comma-separated key=value pairs into a dict."""
    headers: dict = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair or "=" not in pair:
            continue
        key, _, value = pair.partition("=")
        if key.strip():
            headers[key.strip()] = value.strip()
    return headers


def setup_tracing(app: Any = None):
    """Initialize tracing if OpenTelemetry is installed and enabled; otherwise no-op.

    Args:
        app: Optional FastAPI application to auto-instrument.

    Returns:
        The TracerProvider when tracing is active, otherwise None.
    """
    if not _OTEL_AVAILABLE:
        logger.debug(
            "OpenTelemetry not installed; tracing disabled. Install the 'otel' extra to enable."
        )
        return None

    if os.getenv("OTEL_TRACING_ENABLED", "true").lower() == "false":
        logger.info("Tracing disabled via OTEL_TRACING_ENABLED=false")
        return None

    service_name = os.getenv("OTEL_SERVICE_NAME", "scry")
    sample_rate = float(os.getenv("OTEL_TRACES_SAMPLER_ARG", "1.0"))
    exporter_type = os.getenv("OTEL_EXPORTER_TYPE", "otlp").lower()

    resource = Resource.create(
        {
            SERVICE_NAME: service_name,
            SERVICE_VERSION: os.getenv("OTEL_SERVICE_VERSION", SERVICE_VERSION_VALUE),
            "service.namespace": os.getenv("OTEL_SERVICE_NAMESPACE", "scry"),
            "deployment.environment": os.getenv("ENVIRONMENT", "production"),
        }
    )
    provider = TracerProvider(resource=resource, sampler=TraceIdRatioBased(sample_rate))

    exporter = None
    if exporter_type == "console":
        exporter = ConsoleSpanExporter()
    else:
        endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
        if endpoint:
            headers = _parse_headers(os.getenv("OTEL_EXPORTER_OTLP_HEADERS", ""))
            exporter = OTLPSpanExporter(endpoint=endpoint, headers=headers or None)
        else:
            logger.warning(
                "OTEL_EXPORTER_OTLP_ENDPOINT not set; no span exporter configured"
            )

    if exporter is not None:
        provider.add_span_processor(BatchSpanProcessor(exporter))

    trace.set_tracer_provider(provider)

    if app is not None:
        FastAPIInstrumentor.instrument_app(app, excluded_urls="health,metrics")
    for instrumentor in (HTTPXClientInstrumentor, LoggingInstrumentor):
        try:
            instrumentor().instrument()
        except Exception as e:  # instrumentation is best-effort
            logger.warning("Failed to instrument %s: %s", instrumentor.__name__, e)

    logger.info("Tracing enabled: service=%s exporter=%s", service_name, exporter_type)
    return provider


def add_span_attributes(attributes: dict) -> None:
    """Add attributes to the current span (no-op if tracing is unavailable)."""
    if not _OTEL_AVAILABLE:
        return
    span = trace.get_current_span()
    if span and span.is_recording():
        for key, value in attributes.items():
            span.set_attribute(key, value)


def record_exception(exception: Exception, attributes: dict | None = None) -> None:
    """Record an exception on the current span (no-op if tracing is unavailable)."""
    if not _OTEL_AVAILABLE:
        return
    span = trace.get_current_span()
    if span and span.is_recording():
        span.record_exception(exception)
        for key, value in (attributes or {}).items():
            span.set_attribute(key, value)


def shutdown_tracing() -> None:
    """Flush and shut down tracing (no-op if tracing is unavailable)."""
    if not _OTEL_AVAILABLE:
        return
    provider = trace.get_tracer_provider()
    if hasattr(provider, "shutdown"):
        provider.shutdown()
        logger.info("Tracing shutdown complete")
