"""
Correlation ID and Context Management for Fractal VSM

Provides thread-safe context propagation for correlation IDs and trace context.
Enables end-to-end request tracing across all VSM tiers.

Usage:
    from fractal_agent.observability.context import set_correlation_id, get_correlation_id

    # Start of user request
    set_correlation_id()  # Auto-generates UUID

    # Anywhere in the call stack
    corr_id = get_correlation_id()  # Returns same ID

Author: BMad
Date: 2025-01-20
"""

import uuid
from contextvars import ContextVar
from typing import Optional
from datetime import datetime

# Thread-safe context variables
correlation_id_var: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)
trace_id_var: ContextVar[Optional[str]] = ContextVar('trace_id', default=None)
session_start_time_var: ContextVar[Optional[datetime]] = ContextVar('session_start_time', default=None)


def set_correlation_id(correlation_id: Optional[str] = None) -> str:
    """
    Set correlation ID for current context.

    Args:
        correlation_id: Optional correlation ID. If None, generates UUID.

    Returns:
        The correlation ID that was set
    """
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())
    correlation_id_var.set(correlation_id)
    return correlation_id


def get_correlation_id() -> str:
    """
    Get correlation ID from current context.

    Auto-generates and sets a new correlation ID if none exists.
    This ensures correlation_id is NEVER None, which fixes OpenTelemetry
    attribute type warnings.

    Returns:
        Correlation ID (always a string, never None)
    """
    corr_id = correlation_id_var.get()
    if corr_id is None:
        # Auto-generate if not set
        corr_id = set_correlation_id()
    return corr_id


def set_trace_id(trace_id: Optional[str] = None) -> str:
    """
    Set trace ID for current context (for OpenTelemetry compatibility).

    Args:
        trace_id: Optional trace ID. If None, generates UUID.

    Returns:
        The trace ID that was set
    """
    if trace_id is None:
        trace_id = str(uuid.uuid4())
    trace_id_var.set(trace_id)
    return trace_id


def get_trace_id() -> str:
    """
    Get trace ID from current context.

    Auto-generates and sets a new trace ID if none exists.

    Returns:
        Trace ID (always a string, never None)
    """
    t_id = trace_id_var.get()
    if t_id is None:
        # Auto-generate if not set
        t_id = set_trace_id()
    return t_id


def set_session_start_time(start_time: Optional[datetime] = None) -> datetime:
    """
    Set session start time for duration tracking.

    Args:
        start_time: Optional start time. If None, uses current time.

    Returns:
        The start time that was set
    """
    if start_time is None:
        start_time = datetime.now()
    session_start_time_var.set(start_time)
    return start_time


def get_session_start_time() -> Optional[datetime]:
    """
    Get session start time from current context.

    Returns:
        Session start time if set, None otherwise
    """
    return session_start_time_var.get()


def get_session_duration_seconds() -> Optional[float]:
    """
    Calculate session duration in seconds.

    Returns:
        Duration in seconds if session start time is set, None otherwise
    """
    start_time = get_session_start_time()
    if start_time is None:
        return None
    return (datetime.now() - start_time).total_seconds()


def clear_context():
    """
    Clear all context variables.
    Useful for cleaning up between tests or starting fresh request.
    """
    correlation_id_var.set(None)
    trace_id_var.set(None)
    session_start_time_var.set(None)


def init_context(correlation_id: Optional[str] = None, trace_id: Optional[str] = None) -> dict:
    """
    Initialize full context for a new request/workflow.

    Args:
        correlation_id: Optional correlation ID. Auto-generated if None.
        trace_id: Optional trace ID. Auto-generated if None.

    Returns:
        Dictionary with correlation_id, trace_id, start_time
    """
    corr_id = set_correlation_id(correlation_id)
    tr_id = set_trace_id(trace_id)
    start_time = set_session_start_time()

    return {
        'correlation_id': corr_id,
        'trace_id': tr_id,
        'start_time': start_time.isoformat()
    }


class ContextManager:
    """
    Context manager for automatic context initialization and cleanup.

    Usage:
        with ContextManager() as ctx:
            # Work happens here
            # Context automatically initialized
            print(f"Trace ID: {ctx['trace_id']}")
        # Context automatically cleared
    """

    def __enter__(self):
        return init_context()

    def __exit__(self, exc_type, exc_val, exc_tb):
        clear_context()
        return False  # Don't suppress exceptions
