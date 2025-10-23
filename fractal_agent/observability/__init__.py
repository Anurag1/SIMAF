"""
Fractal VSM Observability Framework

Enterprise-grade observability with:
- Distributed tracing (OpenTelemetry → Jaeger)
- Structured logging (file + console with correlation IDs)
- Event sourcing (immutable audit trail in PostgreSQL)
- Context propagation (correlation IDs across tiers)

Quick Start:
    from fractal_agent.observability import (
        init_tracing, get_tracer, get_logger,
        init_context, get_event_store, VSMEvent
    )

    # Initialize at application startup
    init_tracing(service_name="fractal-agent")

    # Use in code
    tracer = get_tracer(__name__)
    logger = get_logger(__name__)
    event_store = get_event_store()

    with init_context() as ctx:
        with tracer.start_as_current_span("my_operation"):
            logger.info("Operation started")
            event_store.append(VSMEvent(
                tier="System4",
                event_type="operation_started",
                data={"operation": "my_operation"}
            ))

Author: BMad
Date: 2025-01-20
"""

# Context management
from .context import (
    get_correlation_id,
    set_correlation_id,
    get_trace_id,
    set_trace_id,
    init_context,
    clear_context,
    ContextManager
)

# Tracing
from .tracing import (
    init_tracing,
    get_tracer,
    record_exception,
    set_span_attribute,
    set_span_attributes,
    add_span_event,
    shutdown_tracing
)

# Logging
from .logging import (
    configure_logging,
    get_logger,
    log_info,
    log_warning,
    log_error
)

# Event sourcing
from .events import (
    VSMEvent,
    EventStore,
    get_event_store
)

# LLM instrumentation
from .llm_instrumentation import (
    InstrumentedUnifiedLM,
    instrument_llm
)

# Metrics
from .metrics import (
    record_llm_call,
    record_task_completion,
    record_agent_operation,
    record_research_stage,
    record_intelligence_analysis,
    record_event_stored,
    push_metrics
)

__all__ = [
    # Context
    'get_correlation_id',
    'set_correlation_id',
    'get_trace_id',
    'set_trace_id',
    'init_context',
    'clear_context',
    'ContextManager',

    # Tracing
    'init_tracing',
    'get_tracer',
    'record_exception',
    'set_span_attribute',
    'set_span_attributes',
    'add_span_event',
    'shutdown_tracing',

    # Logging
    'configure_logging',
    'get_logger',
    'log_info',
    'log_warning',
    'log_error',

    # Events
    'VSMEvent',
    'EventStore',
    'get_event_store',

    # LLM Instrumentation
    'InstrumentedUnifiedLM',
    'instrument_llm',

    # Metrics
    'record_llm_call',
    'record_task_completion',
    'record_agent_operation',
    'record_research_stage',
    'record_intelligence_analysis',
    'record_event_stored',
    'push_metrics',
]

# Version
__version__ = '1.0.0'


# ============================================================================
# AUTO-INITIALIZATION (runs once on module import)
# ============================================================================
# This ensures observability starts automatically when ANY part of the
# system imports from the observability module. Runs exactly ONCE.
# ============================================================================

_observability_initialized = False

def _auto_init_observability():
    """
    Auto-initialize observability on first import of this module.

    - Starts OpenTelemetry tracing (exports to Jaeger via OTEL Collector)
    - Starts Prometheus metrics server (port 9100)
    - Gracefully degrades if services unavailable

    This runs exactly ONCE, no matter how many modules import observability.
    """
    global _observability_initialized

    if _observability_initialized:
        return

    try:
        from .metrics_server import start_metrics_server
        import atexit
        import logging

        logger = logging.getLogger(__name__)

        # Initialize OpenTelemetry tracing
        init_tracing(
            service_name="fractal-agent",
            otlp_endpoint="http://localhost:4317"
        )
        logger.info("✓ OpenTelemetry tracing initialized (endpoint: localhost:4317)")

        # Start Prometheus metrics server (background thread)
        start_metrics_server(port=9100)
        logger.info("✓ Prometheus metrics server started (port: 9100)")

        # Register shutdown handler
        atexit.register(shutdown_tracing)

        _observability_initialized = True
        logger.info("✓ Observability system fully initialized")

    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Observability auto-start failed (non-fatal): {e}")
        logger.warning("System will continue without observability. To enable:")
        logger.warning("  1. Start observability stack: docker-compose -f docker-compose.observability.yml up -d")
        logger.warning("  2. Ensure ports 4317 (OTLP) and 9100 (metrics) are available")

# Auto-initialize on module import
_auto_init_observability()
