#!/usr/bin/env python3
"""
Test script to generate sample metrics for the Fractal VSM observability stack.

This script simulates LLM calls and agent operations to generate test data
that can be visualized in Grafana.

Usage:
    # In one terminal, start the metrics server:
    python -m fractal_agent.observability.metrics_server

    # In another terminal, run this test script:
    python test_metrics.py

Author: BMad
Date: 2025-01-20
"""

import time
import random
from fractal_agent.observability.metrics import (
    record_llm_call,
    record_task_completion,
    record_agent_operation,
    record_research_stage,
    record_intelligence_analysis,
    record_event_stored
)

# Sample data for realistic metrics
TIERS = ['System1_Research', 'System2_Analyst', 'System3', 'System4']
PROVIDERS = ['anthropic', 'openai']
MODELS = {
    'anthropic': ['claude-sonnet-4.5', 'claude-haiku-3.5'],
    'openai': ['gpt-4', 'gpt-3.5-turbo']
}
TASK_TYPES = ['decomposition', 'planning', 'execution', 'review']
AGENT_TYPES = ['ResearchAgent', 'AnalystAgent', 'PMAgent', 'IntelligenceAgent']
OPERATIONS = ['planning', 'execution', 'analysis', 'synthesis']
RESEARCH_STAGES = ['initialization', 'research', 'analysis', 'synthesis', 'validation']
EVENT_TYPES = ['task_started', 'task_completed', 'decision_made', 'error_occurred']


def simulate_llm_call():
    """Simulate an LLM API call with realistic metrics."""
    tier = random.choice(TIERS)
    provider = random.choice(PROVIDERS)
    model = random.choice(MODELS[provider])

    # Simulate realistic token counts and latencies
    tokens = random.randint(500, 5000)
    latency_ms = random.uniform(500, 3000)

    # Calculate cost (simplified pricing)
    if 'sonnet' in model:
        cost = tokens * 0.000003
    elif 'haiku' in model:
        cost = tokens * 0.000001
    elif 'gpt-4' in model:
        cost = tokens * 0.00003
    else:
        cost = tokens * 0.000002

    # 10% cache hit rate
    cache_hit = random.random() < 0.1

    # 95% success rate
    success = random.random() < 0.95
    error_type = None if success else random.choice(['timeout', 'rate_limit', 'api_error'])

    record_llm_call(
        tier=tier,
        provider=provider,
        model=model,
        tokens=tokens,
        latency_ms=latency_ms,
        cost=cost,
        cache_hit=cache_hit,
        success=success,
        error_type=error_type
    )

    return success


def simulate_task():
    """Simulate a VSM tier task completion."""
    tier = random.choice(TIERS)
    task_type = random.choice(TASK_TYPES)
    duration_seconds = random.uniform(1.0, 30.0)
    success = random.random() < 0.97

    record_task_completion(
        tier=tier,
        task_type=task_type,
        duration_seconds=duration_seconds,
        success=success
    )


def simulate_agent_operation():
    """Simulate an agent operation."""
    agent_type = random.choice(AGENT_TYPES)
    operation = random.choice(OPERATIONS)
    duration_seconds = random.uniform(0.5, 15.0)
    success = random.random() < 0.98

    record_agent_operation(
        agent_type=agent_type,
        operation=operation,
        duration_seconds=duration_seconds,
        success=success
    )


def simulate_research():
    """Simulate research stage completion."""
    stage = random.choice(RESEARCH_STAGES)
    success = random.random() < 0.96

    record_research_stage(stage=stage, success=success)


def simulate_intelligence():
    """Simulate intelligence analysis."""
    trigger_type = random.choice(['scheduled', 'threshold', 'manual', 'anomaly'])
    duration_seconds = random.uniform(5.0, 60.0)
    success = random.random() < 0.99

    record_intelligence_analysis(
        trigger_type=trigger_type,
        duration_seconds=duration_seconds,
        success=success
    )


def simulate_event():
    """Simulate event storage."""
    tier = random.choice(TIERS)
    event_type = random.choice(EVENT_TYPES)
    success = random.random() < 0.995

    record_event_stored(
        tier=tier,
        event_type=event_type,
        success=success
    )


def main():
    """Run continuous metric generation."""
    print("Starting metrics generation...")
    print("Metrics will be available at http://localhost:8000/metrics")
    print("Press Ctrl+C to stop")
    print()

    iteration = 0
    try:
        while True:
            iteration += 1

            # Generate various metrics with different frequencies
            # LLM calls are most frequent
            for _ in range(random.randint(3, 8)):
                simulate_llm_call()

            # Tasks and agent operations
            if iteration % 2 == 0:
                for _ in range(random.randint(1, 3)):
                    simulate_task()
                    simulate_agent_operation()

            # Research and intelligence less frequent
            if iteration % 5 == 0:
                simulate_research()

            if iteration % 10 == 0:
                simulate_intelligence()

            # Events very frequent
            for _ in range(random.randint(2, 5)):
                simulate_event()

            # Print status every 10 iterations
            if iteration % 10 == 0:
                print(f"Generated {iteration * 20} metrics... (iteration {iteration})")

            # Wait a bit between iterations (0.5-2 seconds)
            time.sleep(random.uniform(0.5, 2.0))

    except KeyboardInterrupt:
        print("\nMetrics generation stopped")


if __name__ == '__main__':
    main()
