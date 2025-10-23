"""
Event Sourcing for Fractal VSM

Provides event store interface for immutable audit trail.
Events are written to PostgreSQL for querying and analysis.

Usage:
    from fractal_agent.observability.events import EventStore, VSMEvent

    event_store = EventStore()
    event = VSMEvent(
        tier="System4",
        event_type="task_started",
        data={"task": "Implement export.py"}
    )
    event_store.append(event)

Author: BMad
Date: 2025-01-20
"""

import uuid
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Any, Optional
import psycopg2
from psycopg2.extras import RealDictCursor
import logging
from .context import get_correlation_id, get_trace_id

logger = logging.getLogger(__name__)


@dataclass
class VSMEvent:
    """
    Immutable event in the VSM system.

    Attributes:
        tier: VSM tier (System1, System2, System3, System4)
        event_type: Type of event (task_started, llm_called, etc.)
        data: Event payload (JSONB in database)
        event_id: Unique event ID (auto-generated)
        timestamp: Event timestamp (auto-generated)
        trace_id: OpenTelemetry trace ID (from context)
        span_id: OpenTelemetry span ID
        correlation_id: Request correlation ID (from context)
    """
    tier: str
    event_type: str
    data: Dict[str, Any]
    event_id: Optional[uuid.UUID] = None
    timestamp: Optional[datetime] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    correlation_id: Optional[str] = None

    def __post_init__(self):
        # Auto-generate fields if not provided
        if self.event_id is None:
            self.event_id = uuid.uuid4()
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.trace_id is None:
            self.trace_id = get_trace_id()
        if self.correlation_id is None:
            self.correlation_id = get_correlation_id()


class EventStore:
    """
    PostgreSQL-backed event store for VSM events.

    Provides append-only event log with querying capabilities.
    """

    def __init__(self, connection_string: Optional[str] = None):
        """
        Initialize event store.

        Args:
            connection_string: PostgreSQL connection string.
                             Defaults to localhost:5433 (observability DB)
        """
        if connection_string is None:
            connection_string = (
                "host=localhost port=5433 dbname=fractal_observability "
                "user=fractal password=fractal_dev_password"
            )

        self.connection_string = connection_string
        self._conn = None

    def connect(self):
        """Establish database connection"""
        if self._conn is None or self._conn.closed:
            try:
                self._conn = psycopg2.connect(self.connection_string)
                logger.info("Connected to event store")
            except Exception as e:
                logger.error(f"Failed to connect to event store: {e}")
                self._conn = None

    def append(self, event: VSMEvent):
        """
        Append event to store.

        Args:
            event: Event to append

        Note:
            Safe to call even if DB is unavailable - will log error but not crash.
        """
        try:
            self.connect()
            if self._conn is None:
                logger.warning("Event store unavailable, event not persisted")
                return

            with self._conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO vsm_events (
                        event_id, timestamp, trace_id, span_id, tier,
                        event_type, data, correlation_id
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s::jsonb, %s
                    )
                """, (
                    str(event.event_id),
                    event.timestamp,
                    event.trace_id,
                    event.span_id,
                    event.tier,
                    event.event_type,
                    json.dumps(event.data),  # Convert dict to valid JSON string for PostgreSQL JSONB
                    event.correlation_id
                ))

            self._conn.commit()
            logger.debug(f"Event appended: {event.event_type} ({event.tier})")

        except Exception as e:
            logger.error(f"Failed to append event: {e}", exc_info=True)
            if self._conn:
                self._conn.rollback()

    def close(self):
        """Close database connection"""
        if self._conn and not self._conn.closed:
            self._conn.close()
            logger.info("Event store connection closed")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# Global event store instance (lazy-initialized)
_global_event_store: Optional[EventStore] = None


def get_event_store() -> EventStore:
    """
    Get global event store instance.

    Returns:
        EventStore instance
    """
    global _global_event_store

    if _global_event_store is None:
        _global_event_store = EventStore()

    return _global_event_store
