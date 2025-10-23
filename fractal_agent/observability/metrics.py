"""
Metrics Module - Stub for LLM Observability

This is a minimal stub to allow imports to work.
Full metrics implementation to be restored.

Author: BMad
Date: 2025-10-22
"""

import logging
from typing import Dict, Any, Optional
from prometheus_client import CollectorRegistry

logger = logging.getLogger(__name__)

# Proper Prometheus registry (not just a dict!)
registry = CollectorRegistry()


def record_llm_call(
    model: str,
    input_tokens: int,
    output_tokens: int,
    latency_ms: float,
    success: bool,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Record LLM call metrics (stub implementation).

    Args:
        model: Model name
        input_tokens: Input token count
        output_tokens: Output token count
        latency_ms: Latency in milliseconds
        success: Whether call succeeded
        metadata: Additional metadata
    """
    logger.debug(
        f"LLM call: model={model}, input_tokens={input_tokens}, "
        f"output_tokens={output_tokens}, latency_ms={latency_ms:.2f}, success={success}"
    )


def record_task_completion(task_id: str, duration_ms: float, success: bool, metadata: Optional[Dict[str, Any]] = None):
    """Stub for task completion metrics"""
    logger.debug(f"Task {task_id}: duration={duration_ms:.2f}ms, success={success}")


def record_agent_operation(agent_type: str, operation: str, duration_ms: float, metadata: Optional[Dict[str, Any]] = None):
    """Stub for agent operation metrics"""
    logger.debug(f"Agent {agent_type}.{operation}: duration={duration_ms:.2f}ms")


def record_research_stage(stage: str, duration_ms: float, metadata: Optional[Dict[str, Any]] = None):
    """Stub for research stage metrics"""
    logger.debug(f"Research stage {stage}: duration={duration_ms:.2f}ms")


def record_intelligence_analysis(analysis_type: str, duration_ms: float, metadata: Optional[Dict[str, Any]] = None):
    """Stub for intelligence analysis metrics"""
    logger.debug(f"Intelligence analysis {analysis_type}: duration={duration_ms:.2f}ms")


def record_event_stored(event_type: str, tier: Optional[str] = None):
    """Stub for event storage metrics"""
    logger.debug(f"Event stored: type={event_type}, tier={tier}")


def push_metrics():
    """Stub for pushing metrics"""
    logger.debug("Metrics push (stub - no-op)")


def get_metrics_summary() -> Dict[str, Any]:
    """
    Get summary of recorded metrics (stub).

    Returns:
        Empty dict (stub implementation)
    """
    return {}


def reset_metrics():
    """Reset all metrics (stub)"""
    pass


def record_task_completion(
    task_id: str,
    agent_type: str,
    duration_seconds: float,
    success: bool,
    metadata: Optional[Dict[str, Any]] = None
):
    """Record task completion metrics (stub)"""
    logger.debug(f"Task completion: task_id={task_id}, agent_type={agent_type}, duration={duration_seconds:.2f}s, success={success}")


def record_agent_operation(
    agent_type: str,
    operation: str,
    duration_seconds: float,
    metadata: Optional[Dict[str, Any]] = None
):
    """Record agent operation metrics (stub)"""
    logger.debug(f"Agent operation: agent_type={agent_type}, operation={operation}, duration={duration_seconds:.2f}s")


def record_research_stage(
    stage: str,
    duration_seconds: float,
    metadata: Optional[Dict[str, Any]] = None
):
    """Record research stage metrics (stub)"""
    logger.debug(f"Research stage: stage={stage}, duration={duration_seconds:.2f}s")


def record_intelligence_analysis(
    analysis_type: str,
    duration_seconds: float,
    metadata: Optional[Dict[str, Any]] = None
):
    """Record intelligence analysis metrics (stub)"""
    logger.debug(f"Intelligence analysis: type={analysis_type}, duration={duration_seconds:.2f}s")


def record_event_stored(event_type: str, metadata: Optional[Dict[str, Any]] = None):
    """Record event storage metrics (stub)"""
    logger.debug(f"Event stored: type={event_type}")


def push_metrics():
    """Push metrics to backend (stub)"""
    pass
