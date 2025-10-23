"""
Auto-Instrumentation for Fractal Agent LLM Calls

Automatically instruments all LLM calls in the fractal agent system with
production monitoring. This module patches the UnifiedLM class to add
observability without requiring code changes.

Features:
- Automatic instrumentation of all LLM calls
- Zero-code-change integration
- Production monitoring with accurate cost tracking
- Budget alerts and cost tracking
- OpenTelemetry tracing integration
- Event sourcing for audit trail

Usage:
    # Import at application startup
    from fractal_agent.observability.auto_instrumentation import enable_auto_instrumentation

    # Enable auto-instrumentation (call once at startup)
    enable_auto_instrumentation(
        hourly_budget_usd=100.0,
        daily_budget_usd=1000.0
    )

    # All subsequent LLM calls will be automatically instrumented
    from fractal_agent.utils.llm_provider import UnifiedLM
    lm = UnifiedLM(...)
    result = lm(messages=[...])  # Automatically tracked!

Author: BMad
Date: 2025-01-20
"""

import time
import logging
from typing import Dict, Any, List, Optional, Callable
from functools import wraps
import inspect

from .production_monitoring import get_production_monitor, initialize_production_monitoring
from .metrics_server_enhanced import start_metrics_server

logger = logging.getLogger(__name__)


# Track whether auto-instrumentation is enabled
_auto_instrumentation_enabled = False
_original_unifiedlm_call: Optional[Callable] = None


def enable_auto_instrumentation(
    hourly_budget_usd: float = 100.0,
    daily_budget_usd: float = 1000.0,
    metrics_port: int = 9100,
    start_metrics_server_flag: bool = True
):
    """
    Enable automatic instrumentation of all LLM calls.

    This function patches the UnifiedLM class to add observability to all calls.
    Call this once at application startup.

    Args:
        hourly_budget_usd: Hourly spending limit for alerts
        daily_budget_usd: Daily spending limit for alerts
        metrics_port: Port for Prometheus metrics server
        start_metrics_server_flag: Whether to start metrics server

    Example:
        from fractal_agent.observability.auto_instrumentation import enable_auto_instrumentation

        # At application startup
        enable_auto_instrumentation(
            hourly_budget_usd=100.0,
            daily_budget_usd=1000.0
        )
    """
    global _auto_instrumentation_enabled, _original_unifiedlm_call

    if _auto_instrumentation_enabled:
        logger.warning("Auto-instrumentation already enabled")
        return

    logger.info("Enabling auto-instrumentation for LLM calls...")

    # Initialize production monitoring
    initialize_production_monitoring(
        hourly_budget_usd=hourly_budget_usd,
        daily_budget_usd=daily_budget_usd
    )

    # Start metrics server
    if start_metrics_server_flag:
        try:
            start_metrics_server(port=metrics_port)
        except Exception as e:
            logger.warning(f"Failed to start metrics server: {e}")

    # Patch UnifiedLM
    try:
        from fractal_agent.utils.llm_provider import UnifiedLM

        # Save original __call__ method
        _original_unifiedlm_call = UnifiedLM.__call__

        # Replace with instrumented version
        UnifiedLM.__call__ = _create_instrumented_call(_original_unifiedlm_call)

        _auto_instrumentation_enabled = True
        logger.info("✓ Auto-instrumentation enabled successfully")

    except ImportError as e:
        logger.error(f"Failed to import UnifiedLM: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to enable auto-instrumentation: {e}")
        raise


def disable_auto_instrumentation():
    """
    Disable automatic instrumentation.

    Restores original UnifiedLM behavior.
    """
    global _auto_instrumentation_enabled, _original_unifiedlm_call

    if not _auto_instrumentation_enabled:
        logger.warning("Auto-instrumentation not enabled")
        return

    try:
        from fractal_agent.utils.llm_provider import UnifiedLM

        # Restore original __call__ method
        if _original_unifiedlm_call is not None:
            UnifiedLM.__call__ = _original_unifiedlm_call
            _original_unifiedlm_call = None

        _auto_instrumentation_enabled = False
        logger.info("Auto-instrumentation disabled")

    except Exception as e:
        logger.error(f"Failed to disable auto-instrumentation: {e}")
        raise


def _create_instrumented_call(original_call: Callable) -> Callable:
    """
    Create instrumented version of UnifiedLM.__call__

    Args:
        original_call: Original __call__ method

    Returns:
        Instrumented __call__ method
    """

    @wraps(original_call)
    def instrumented_call(
        self,
        messages: Optional[List[Dict[str, Any]]] = None,
        prompt: Optional[str] = None,
        system: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Instrumented LLM call with production monitoring"""

        monitor = get_production_monitor()
        start_time = time.time()

        # Determine tier from call context
        tier = _infer_tier_from_context()

        try:
            # Call original method
            result = original_call(self, messages=messages, prompt=prompt, system=system, **kwargs)

            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000

            # Extract metrics from result
            provider = result.get('provider', 'unknown')
            model = result.get('model', 'unknown')
            input_tokens = result.get('input_tokens', result.get('prompt_tokens', 0))
            output_tokens = result.get('output_tokens', result.get('completion_tokens', 0))
            cache_hit = result.get('cache_hit', False)

            # If tokens not separated, estimate 60/40 split (typical for most tasks)
            if input_tokens == 0 and output_tokens == 0:
                total_tokens = result.get('tokens_used', result.get('total_tokens', 0))
                if total_tokens > 0:
                    input_tokens = int(total_tokens * 0.6)
                    output_tokens = int(total_tokens * 0.4)

            # Record with production monitor
            cost_info = monitor.record_llm_call(
                tier=tier,
                provider=provider,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                cache_hit=cache_hit,
                success=True
            )

            # Enrich result with cost information
            result['cost_info'] = cost_info
            result['latency_ms'] = latency_ms

            # Log cost if significant
            if cost_info['total_cost_usd'] > 0.01:  # Log if > 1 cent
                logger.info(
                    f"LLM call: {provider}/{model} - "
                    f"{input_tokens} in, {output_tokens} out - "
                    f"${cost_info['total_cost_usd']:.4f} "
                    f"({latency_ms:.0f}ms)"
                )

            return result

        except Exception as e:
            # Calculate latency even on failure
            latency_ms = (time.time() - start_time) * 1000

            # Record failure
            monitor.record_llm_call(
                tier=tier,
                provider='unknown',
                model='unknown',
                input_tokens=0,
                output_tokens=0,
                latency_ms=latency_ms,
                cache_hit=False,
                success=False,
                error_type=type(e).__name__
            )

            logger.error(f"LLM call failed: {e}")
            raise

    return instrumented_call


def _infer_tier_from_context() -> str:
    """
    Infer VSM tier from call stack context.

    Examines the call stack to determine which agent/tier is making the call.

    Returns:
        Inferred tier name (e.g., "System1_Research")
    """
    try:
        # Walk up the call stack looking for agent modules
        frame = inspect.currentframe()
        while frame is not None:
            frame_info = inspect.getframeinfo(frame)
            filename = frame_info.filename.lower()

            # Check for agent modules
            if 'research_agent' in filename:
                return 'System1_Research'
            elif 'developer_agent' in filename:
                return 'System2_Developer'
            elif 'coordination_agent' in filename:
                return 'System3_Coordination'
            elif 'intelligence' in filename:
                return 'System4_Intelligence'
            elif 'memory' in filename:
                return 'Memory'

            frame = frame.f_back

    except Exception as e:
        logger.debug(f"Could not infer tier from context: {e}")

    # Default tier
    return 'LLM_General'


def get_instrumentation_status() -> Dict[str, Any]:
    """
    Get auto-instrumentation status.

    Returns:
        Dict with instrumentation status and metrics
    """
    monitor = get_production_monitor()

    return {
        'enabled': _auto_instrumentation_enabled,
        'budget_status': monitor.get_budget_status() if _auto_instrumentation_enabled else None,
        'cost_breakdown': monitor.get_cost_breakdown() if _auto_instrumentation_enabled else None,
        'metrics_endpoint': 'http://localhost:9100/metrics'
    }


def print_instrumentation_status():
    """Print instrumentation status to console"""
    status = get_instrumentation_status()

    print("\n" + "=" * 80)
    print("Fractal Agent Auto-Instrumentation Status")
    print("=" * 80)

    if status['enabled']:
        print("Status: ENABLED ✓")
        print()

        budget = status['budget_status']
        if budget:
            print("Budget Status:")
            print(f"  Hourly: ${budget['hourly']['rate_usd']:.2f}/hr "
                  f"({budget['hourly']['utilization_pct']:.1f}% of "
                  f"${budget['hourly']['budget_usd']:.2f})")
            print(f"  Daily:  ${budget['daily']['total_usd']:.2f} "
                  f"({budget['daily']['utilization_pct']:.1f}% of "
                  f"${budget['daily']['budget_usd']:.2f})")

            if budget['alert_triggered']:
                print("\n  ⚠️  BUDGET ALERT: Spending limit exceeded!")

        print()
        print(f"Metrics: {status['metrics_endpoint']}")

    else:
        print("Status: DISABLED")
        print()
        print("To enable:")
        print("  from fractal_agent.observability.auto_instrumentation import enable_auto_instrumentation")
        print("  enable_auto_instrumentation()")

    print("=" * 80 + "\n")


# Auto-enable on import if environment variable set
import os
if os.getenv('FRACTAL_AUTO_INSTRUMENTATION', '').lower() in ('1', 'true', 'yes'):
    logger.info("Auto-instrumentation enabled via environment variable")
    try:
        enable_auto_instrumentation(
            hourly_budget_usd=float(os.getenv('FRACTAL_HOURLY_BUDGET', '100.0')),
            daily_budget_usd=float(os.getenv('FRACTAL_DAILY_BUDGET', '1000.0'))
        )
    except Exception as e:
        logger.error(f"Failed to auto-enable instrumentation: {e}")
