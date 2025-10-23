"""
OpenTelemetry Tracing Setup for Fractal VSM

Configures distributed tracing with OpenTelemetry, exporting to OTLP collector.
Provides utilities for creating spans and propagating trace context.

Usage:
    from fractal_agent.observability.tracing import get_tracer, init_tracing

    # Initialize once at app startup
    init_tracing(service_name="fractal-agent")

    # Use in code
    tracer = get_tracer(__name__)
    with tracer.start_as_current_span("my_operation"):
        # Work happens here
        pass

Author: BMad
Date: 2025-01-20
"""

import os
import logging
from typing import Optional
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.trace import Status, StatusCode

logger = logging.getLogger(__name__)

# Global tracer provider
_tracer_provider: Optional[TracerProvider] = None
_initialized = False


def init_tracing(
    service_name: str = "fractal-agent",
    otlp_endpoint: str = "http://localhost:4317",
    enable_console_export: bool = False
) -> TracerProvider:
    """
    Initialize OpenTelemetry tracing.

    Args:
        service_name: Name of the service (appears in Jaeger)
        otlp_endpoint: OTLP collector endpoint
        enable_console_export: If True, also print spans to console (debug mode)

    Returns:
        Configured TracerProvider

    Note:
        Only needs to be called once at application startup.
        Safe to call multiple times - will return existing provider.
    """
    global _tracer_provider, _initialized

    if _initialized and _tracer_provider is not None:
        return _tracer_provider

    try:
        # Create resource with service identification
        resource = Resource(attributes={
            SERVICE_NAME: service_name,
            "service.namespace": "fractal-vsm",
            "deployment.environment": os.getenv("ENVIRONMENT", "development")
        })

        # Create tracer provider
        _tracer_provider = TracerProvider(resource=resource)

        # Configure OTLP exporter
        otlp_exporter = OTLPSpanExporter(
            endpoint=otlp_endpoint,
            insecure=True  # Use TLS in production
        )

        # Add batch processor for efficiency
        _tracer_provider.add_span_processor(
            BatchSpanProcessor(otlp_exporter)
        )

        # Optional: Console exporter for debugging
        if enable_console_export:
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter
            _tracer_provider.add_span_processor(
                BatchSpanProcessor(ConsoleSpanExporter())
            )

        # Set as global tracer provider
        trace.set_tracer_provider(_tracer_provider)

        _initialized = True
        logger.info(f"OpenTelemetry tracing initialized: {service_name} -> {otlp_endpoint}")

        return _tracer_provider

    except Exception as e:
        logger.error(f"Failed to initialize OpenTelemetry tracing: {e}")
        # Create a no-op provider so app doesn't crash
        _tracer_provider = TracerProvider()
        trace.set_tracer_provider(_tracer_provider)
        return _tracer_provider


def get_tracer(name: str = __name__) -> trace.Tracer:
    """
    Get a tracer instance for creating spans.

    Args:
        name: Name of the tracer (typically __name__ of module)

    Returns:
        Tracer instance

    Note:
        If tracing not initialized, will auto-initialize with defaults.
    """
    global _initialized

    if not _initialized:
        logger.warning("Tracing not initialized, auto-initializing with defaults")
        init_tracing()

    return trace.get_tracer(name)


def record_exception(span: trace.Span, exception: Exception):
    """
    Record an exception in the current span.

    Args:
        span: The span to record exception in
        exception: The exception to record
    """
    span.record_exception(exception)
    span.set_status(Status(StatusCode.ERROR, str(exception)))


def set_span_attribute(key: str, value):
    """
    Set an attribute on the current span (if any).

    Args:
        key: Attribute key
        value: Attribute value

    Note:
        Safe to call even if no span is active.
    """
    span = trace.get_current_span()
    if span and span.is_recording():
        span.set_attribute(key, value)


def set_span_attributes(attributes: dict):
    """
    Set multiple attributes on the current span.

    Args:
        attributes: Dictionary of key-value pairs to set
    """
    span = trace.get_current_span()
    if span and span.is_recording():
        for key, value in attributes.items():
            span.set_attribute(key, value)


def add_span_event(name: str, attributes: Optional[dict] = None):
    """
    Add an event to the current span.

    Args:
        name: Event name
        attributes: Optional event attributes

    Note:
        Events are timestamped points within a span (e.g., "cache_hit", "llm_call_started")
    """
    span = trace.get_current_span()
    if span and span.is_recording():
        span.add_event(name, attributes=attributes or {})


def shutdown_tracing():
    """
    Shutdown tracing and flush remaining spans.

    Call this before application exit to ensure all spans are exported.
    """
    global _tracer_provider, _initialized

    if _tracer_provider:
        _tracer_provider.shutdown()
        # Note: Don't log here - file handles may be closed during Python shutdown

    _initialized = False
    _tracer_provider = None


# Auto-initialize if OTLP endpoint is available
# This allows tracing to work without explicit init() call
if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    service = os.getenv("OTEL_SERVICE_NAME", "fractal-agent")
    init_tracing(service_name=service, otlp_endpoint=endpoint)
