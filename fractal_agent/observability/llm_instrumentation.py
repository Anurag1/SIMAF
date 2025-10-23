"""
LLM Instrumentation - Observable Wrapper for UnifiedLM

Provides full observability for all LLM API calls:
- OpenTelemetry distributed tracing spans
- Event sourcing for audit trail
- Structured logging with correlation IDs
- Metrics tracking (tokens, latency, cost, cache hits)

Usage:
    from fractal_agent.utils.llm_provider import UnifiedLM
    from fractal_agent.observability.llm_instrumentation import InstrumentedUnifiedLM

    # Wrap any UnifiedLM instance
    base_lm = UnifiedLM(providers=[...])
    instrumented_lm = InstrumentedUnifiedLM(base_lm)

    # Use exactly like UnifiedLM - observability is automatic
    result = instrumented_lm(messages=[...])

Author: BMad
Date: 2025-01-20
"""

import time
from typing import Dict, Any, List, Optional
from ..observability import (
    get_correlation_id, get_tracer, get_logger,
    get_event_store, VSMEvent, set_span_attributes
)
from .metrics import record_llm_call as metrics_record_llm_call

# Use observability-aware structured logger
logger = get_logger(__name__)


class InstrumentedUnifiedLM:
    """
    Observable wrapper for UnifiedLM that adds full observability instrumentation.

    Wraps any UnifiedLM instance and adds:
    - OpenTelemetry spans for distributed tracing
    - Event emissions for audit trail
    - Structured logging with correlation IDs
    - Automatic metrics collection (tokens, latency, cost, cache hits)

    This wrapper is transparent - it implements the same interface as UnifiedLM.
    """

    def __init__(self, base_lm: Any, tier: Optional[str] = None):
        """
        Initialize instrumented LLM wrapper.

        Args:
            base_lm: The UnifiedLM instance to wrap
            tier: Optional tier name for categorizing calls (e.g., "System1_Research")
        """
        self.base_lm = base_lm
        self.raw_tier = tier or "LLM"

        # Map tier names to VSM system tiers for database constraint compatibility
        # Database only accepts: 'System1', 'System2', 'System3', 'System4'
        self.tier = self._map_to_vsm_tier(self.raw_tier)

        # Observability components
        self.tracer = get_tracer(__name__)
        self.event_store = get_event_store()

        # Metrics
        self.total_calls = 0
        self.total_tokens = 0
        self.total_cache_hits = 0
        self.total_failures = 0

        logger.info(f"Initialized InstrumentedUnifiedLM wrapper for tier: {self.raw_tier} (mapped to {self.tier})")

    def _map_to_vsm_tier(self, tier: str) -> str:
        """
        Map arbitrary tier names to VSM system tiers for database compatibility.

        Database constraint only accepts: System1, System2, System3, System4

        Mapping logic:
        - Anything with "cheap", "FractalDSpy_cheap", or System1 → System1 (Operational)
        - Anything with "balanced" or System2 → System2 (Coordination)
        - Anything with "expensive" or System3 → System3 (Intelligence)
        - Anything with "premium" or System4 → System4 (Strategic)
        - Default: System1

        Args:
            tier: Raw tier name (e.g., "FractalDSpy_cheap", "System1_Research")

        Returns:
            VSM system tier (System1, System2, System3, or System4)
        """
        tier_lower = tier.lower()

        if "cheap" in tier_lower or tier == "System1":
            return "System1"
        elif "balanced" in tier_lower or tier == "System2":
            return "System2"
        elif "expensive" in tier_lower or tier == "System3":
            return "System3"
        elif "premium" in tier_lower or tier == "System4":
            return "System4"
        else:
            # Default to System1 for unknown tiers
            logger.warning(f"Unknown tier '{tier}', defaulting to System1")
            return "System1"

    def __call__(
        self,
        messages: Optional[List[Dict[str, Any]]] = None,
        prompt: Optional[str] = None,
        system: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Call LLM with full observability instrumentation.

        This method wraps the base UnifiedLM call with:
        - OpenTelemetry span
        - Event emissions (llm_call_started, llm_call_completed/failed)
        - Structured logging
        - Automatic metrics collection

        Args:
            messages: List of message dicts (chat format)
            prompt: Simple string prompt (converted to messages)
            system: System message
            **kwargs: Additional arguments (max_tokens, temperature, etc.)

        Returns:
            Dict with: {text, tokens_used, cache_hit, provider, model, latency_ms, cost}
        """
        correlation_id = get_correlation_id()
        start_time = time.time()

        # Extract prompt for logging (first 100 chars)
        if prompt:
            prompt_preview = prompt[:100]
        elif messages:
            # Get last user message
            user_msgs = [m for m in messages if m.get("role") == "user"]
            prompt_preview = user_msgs[-1].get("content", "")[:100] if user_msgs else ""
        else:
            prompt_preview = ""

        # Start observability span
        with self.tracer.start_as_current_span("llm_call") as span:
            # Set span attributes
            set_span_attributes({
                "llm.tier": self.tier,
                "llm.prompt_preview": prompt_preview,
                "llm.has_system": system is not None,
                "llm.max_tokens": kwargs.get("max_tokens", "default"),
                "llm.temperature": kwargs.get("temperature", "default"),
                "correlation_id": correlation_id
            })

            # Emit LLM call started event
            self.event_store.append(VSMEvent(
                tier=self.tier,
                event_type="llm_call_started",
                data={
                    "prompt_preview": prompt_preview,
                    "has_system": system is not None,
                    "max_tokens": kwargs.get("max_tokens"),
                    "temperature": kwargs.get("temperature"),
                    "correlation_id": correlation_id
                }
            ))

            logger.info(
                f"LLM call started: {prompt_preview}...",
                extra={
                    "correlation_id": correlation_id,
                    "tier": self.tier,
                    "prompt_preview": prompt_preview
                }
            )

            try:
                # Call base LLM
                result = self.base_lm(
                    messages=messages,
                    prompt=prompt,
                    system=system,
                    **kwargs
                )

                # Calculate latency
                latency_ms = int((time.time() - start_time) * 1000)
                result["latency_ms"] = latency_ms

                # Extract metrics
                provider = result.get("provider", "unknown")
                model = result.get("model", "unknown")
                tokens_used = result.get("tokens_used", 0)
                cache_hit = result.get("cache_hit", False)

                # Estimate cost (rough estimates - update with actual pricing)
                cost = self._estimate_cost(model, tokens_used, cache_hit)
                result["estimated_cost"] = cost

                # Update metrics
                self.total_calls += 1
                self.total_tokens += tokens_used
                if cache_hit:
                    self.total_cache_hits += 1

                # Update span with result attributes
                set_span_attributes({
                    "llm.provider": provider,
                    "llm.model": model,
                    "llm.tokens_used": tokens_used,
                    "llm.cache_hit": cache_hit,
                    "llm.latency_ms": latency_ms,
                    "llm.estimated_cost": cost,
                    "llm.success": True
                })

                # Emit LLM call completed event
                self.event_store.append(VSMEvent(
                    tier=self.tier,
                    event_type="llm_call_completed",
                    data={
                        "provider": provider,
                        "model": model,
                        "tokens_used": tokens_used,
                        "cache_hit": cache_hit,
                        "latency_ms": latency_ms,
                        "estimated_cost": cost,
                        "correlation_id": correlation_id
                    }
                ))

                logger.info(
                    f"LLM call completed: {provider}/{model} - {tokens_used} tokens, {latency_ms}ms, ${cost:.4f}, cache_hit={cache_hit}",
                    extra={
                        "correlation_id": correlation_id,
                        "tier": self.tier,
                        "provider": provider,
                        "model": model,
                        "tokens_used": tokens_used,
                        "cache_hit": cache_hit,
                        "latency_ms": latency_ms,
                        "estimated_cost": cost
                    }
                )

                # Record Prometheus metrics
                metrics_record_llm_call(
                    model=model,
                    input_tokens=tokens_used,  # Approximation - could be split if available
                    output_tokens=0,  # Not tracked separately in current response
                    latency_ms=latency_ms,
                    success=True,
                    metadata={
                        "tier": self.tier,
                        "provider": provider,
                        "cost": cost,
                        "cache_hit": cache_hit
                    }
                )

                return result

            except Exception as e:
                # Calculate latency even on failure
                latency_ms = int((time.time() - start_time) * 1000)

                # Update failure metrics
                self.total_failures += 1

                # Update span with error
                set_span_attributes({
                    "llm.success": False,
                    "llm.error": str(e),
                    "llm.latency_ms": latency_ms
                })

                # Emit LLM call failed event
                self.event_store.append(VSMEvent(
                    tier=self.tier,
                    event_type="llm_call_failed",
                    data={
                        "error": str(e),
                        "latency_ms": latency_ms,
                        "correlation_id": correlation_id
                    }
                ))

                logger.error(
                    f"LLM call failed: {str(e)}",
                    extra={
                        "correlation_id": correlation_id,
                        "tier": self.tier,
                        "error": str(e),
                        "latency_ms": latency_ms
                    }
                )

                # Record failure metrics
                metrics_record_llm_call(
                    model="unknown",
                    input_tokens=0,
                    output_tokens=0,
                    latency_ms=latency_ms,
                    success=False,
                    metadata={
                        "tier": self.tier,
                        "provider": "unknown",
                        "cost": 0.0,
                        "cache_hit": False,
                        "error_type": type(e).__name__
                    }
                )

                # Re-raise the exception
                raise

    def _estimate_cost(self, model: str, tokens: int, cache_hit: bool) -> float:
        """
        Estimate cost of LLM call based on model and tokens.

        Rough estimates - update with actual pricing from providers.
        Cache hits typically cost 10% of regular price.

        Args:
            model: Model name
            tokens: Total tokens used (input + output)
            cache_hit: Whether this was a cache hit

        Returns:
            Estimated cost in USD
        """
        # Rough pricing per 1M tokens (averaged input/output)
        # These are approximations - real pricing varies by input/output
        pricing = {
            # Anthropic Claude models
            "claude-sonnet-4.5": 3.00 / 1_000_000,  # ~$3/1M tokens
            "claude-sonnet-4.1": 3.00 / 1_000_000,
            "claude-haiku-4.5": 0.25 / 1_000_000,   # ~$0.25/1M tokens

            # Google Gemini models
            "gemini-2.0-flash-exp": 0.15 / 1_000_000,  # ~$0.15/1M tokens
            "gemini-1.5-pro": 1.25 / 1_000_000,        # ~$1.25/1M tokens
            "gemini-1.5-pro-002": 1.25 / 1_000_000,

            # Default fallback
            "default": 1.00 / 1_000_000
        }

        # Get per-token cost
        cost_per_token = pricing.get(model, pricing["default"])

        # Apply cache discount
        if cache_hit:
            cost_per_token *= 0.1  # Cache hits ~10% of full price

        return tokens * cost_per_token

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get aggregated metrics for this LLM instance.

        Returns:
            Dict with total_calls, total_tokens, cache_hit_rate, etc.
        """
        return {
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "total_cache_hits": self.total_cache_hits,
            "total_failures": self.total_failures,
            "cache_hit_rate": self.total_cache_hits / self.total_calls if self.total_calls > 0 else 0.0,
            "avg_tokens_per_call": self.total_tokens / self.total_calls if self.total_calls > 0 else 0.0
        }


# Convenience function for wrapping UnifiedLM instances
def instrument_llm(base_lm: Any, tier: Optional[str] = None) -> InstrumentedUnifiedLM:
    """
    Wrap a UnifiedLM instance with observability instrumentation.

    Usage:
        from fractal_agent.utils.llm_provider import UnifiedLM
        from fractal_agent.observability.llm_instrumentation import instrument_llm

        lm = UnifiedLM(providers=[...])
        instrumented_lm = instrument_llm(lm, tier="System1_Research")

    Args:
        base_lm: UnifiedLM instance to wrap
        tier: Optional tier name for categorizing calls

    Returns:
        InstrumentedUnifiedLM wrapper
    """
    return InstrumentedUnifiedLM(base_lm, tier=tier)
