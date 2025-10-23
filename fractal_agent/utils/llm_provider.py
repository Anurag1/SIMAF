"""
Unified LLM Provider for Fractal Agent Ecosystem

ALL LLM calls in the system route through this module.

Architecture:
- Extensible provider chain with automatic failover
- Primary: Anthropic Claude (leveraging max subscription)
- Fallback: Google Gemini (different infrastructure)
- Future: Add providers by implementing LLMProvider subclass

Author: BMad
Date: 2025-10-18
"""

import anthropic
import google.generativeai as genai
from claude_agent_sdk import query as claude_query
from claude_agent_sdk.types import ClaudeAgentOptions, AssistantMessage, TextBlock, ThinkingBlock
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv
import os
import asyncio
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod
import logging

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


# Base provider interface (for future extensibility)
class LLMProvider(ABC):
    """
    Base class for LLM providers.

    To add a new provider:
    1. Subclass this
    2. Implement _call_provider()
    3. Register in PROVIDER_REGISTRY
    """

    def __init__(self, model: str, **config):
        self.model = model
        self.config = config
        self.calls = 0
        self.tokens_used = 0

    @abstractmethod
    def _call_provider(
        self,
        messages: List[Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Provider-specific implementation.

        Must return: {
            "text": str,
            "tokens_used": int,
            "cache_hit": bool,
            "provider": str,
            "model": str
        }
        """
        pass

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def __call__(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """Call with automatic retry"""
        result = self._call_provider(messages, **kwargs)
        self.calls += 1
        self.tokens_used += result.get("tokens_used", 0)
        return result


class AnthropicProvider(LLMProvider):
    """
    Anthropic Claude provider using Claude Code subscription.

    Uses claude-agent-sdk which leverages Claude Code MAX subscription
    authentication automatically (NO API KEY needed).

    This provider uses the Claude SDK's query() function which communicates
    with Claude Code CLI to access your subscription.
    """

    def __init__(self, model: str, enable_caching: bool = True, **config):
        super().__init__(model, **config)
        self.enable_caching = enable_caching
        self.cache_hits = 0
        self.cache_misses = 0

        # Configure Claude Agent options
        # Set cwd to current working directory if not specified
        cwd = config.get("cwd", os.getcwd())
        self.options = ClaudeAgentOptions(
            cwd=cwd,
            permission_mode="bypassPermissions"  # Auto-allow for LLM operations
        )

        logger.info("Using Claude Code subscription via claude-agent-sdk")

    async def _call_provider_async(
        self,
        messages: List[Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        """Async implementation using claude-agent-sdk"""

        # Build prompt from messages
        # System messages go at the beginning, then conversation
        system_messages = [msg for msg in messages if msg.get("role") == "system"]
        conversation_messages = [msg for msg in messages if msg.get("role") != "system"]

        prompt_parts = []

        # Add system messages if present
        if system_messages:
            for msg in system_messages:
                prompt_parts.append(f"SYSTEM: {msg.get('content', '')}")

        # Add conversation messages
        for msg in conversation_messages:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")
            prompt_parts.append(f"{role}: {content}")

        prompt = "\n\n".join(prompt_parts)

        # Call Claude via SDK
        response_text = ""
        tokens_used = 0
        cache_hit = False

        try:
            async for message in claude_query(prompt=prompt, options=self.options):
                # Extract text from AssistantMessage
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            response_text += block.text
                        elif isinstance(block, ThinkingBlock):
                            # Include thinking blocks in response
                            response_text += f"\n[Thinking: {block.thinking}]\n"

                # Check for result message with token usage
                if hasattr(message, 'total_cost_usd'):
                    # ResultMessage - contains usage info
                    # Estimate tokens from cost (rough approximation)
                    # Claude Sonnet ~$3/M input, $15/M output
                    # Simplified: assume $10/M tokens average
                    tokens_used = int(message.total_cost_usd * 100000)  # Very rough estimate

        except Exception as e:
            logger.error(f"Claude SDK query failed: {e}")
            raise

        return {
            "text": response_text.strip(),
            "tokens_used": tokens_used,
            "cache_hit": cache_hit,
            "provider": "anthropic",
            "model": self.model
        }

    def _call_provider(
        self,
        messages: List[Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        """Synchronous wrapper around async implementation"""
        # Run async function in event loop
        try:
            loop = asyncio.get_running_loop()
            # If we're already in an async context, can't use run_until_complete
            raise RuntimeError("Cannot call sync wrapper from async context")
        except RuntimeError:
            # No event loop running - create new one
            return asyncio.run(self._call_provider_async(messages, **kwargs))


class GeminiProvider(LLMProvider):
    """Google Gemini provider"""

    def __init__(self, model: str, **config):
        super().__init__(model, **config)
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.client = genai.GenerativeModel(model)

    def _call_provider(
        self,
        messages: List[Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        # Convert messages to Gemini format
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt_parts.append(f"{role}: {content}")
        prompt = "\n\n".join(prompt_parts)

        # Convert Anthropic-style parameters to Gemini format
        generation_config = {}

        # Map max_tokens to max_output_tokens
        if "max_tokens" in kwargs:
            generation_config["max_output_tokens"] = kwargs.pop("max_tokens")

        # Map temperature (both use same name)
        if "temperature" in kwargs:
            generation_config["temperature"] = kwargs.pop("temperature")

        # Map top_p (both use same name)
        if "top_p" in kwargs:
            generation_config["top_p"] = kwargs.pop("top_p")

        # Build Gemini kwargs
        gemini_kwargs = {}
        if generation_config:
            gemini_kwargs["generation_config"] = generation_config

        # Pass any remaining kwargs (unlikely, but just in case)
        gemini_kwargs.update(kwargs)

        response = self.client.generate_content(prompt, **gemini_kwargs)

        # Rough token estimate for Gemini
        tokens_used = int(len(prompt.split()) * 1.3)

        return {
            "text": response.text,
            "tokens_used": tokens_used,
            "cache_hit": False,
            "provider": "gemini",
            "model": self.model
        }


# Provider registry for easy extension
PROVIDER_REGISTRY = {
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
    # Future: "openai": OpenAIProvider,
    # Future: "together": TogetherProvider,
    # etc.
}


class UnifiedLM:
    """
    Unified LLM provider for the entire Fractal Agent Ecosystem.

    ALL LLM calls route through this class.

    Supports priority-ordered provider chain for automatic failover.
    Extensible: add new providers without changing this class.

    Usage:
        # Current (Anthropic → Gemini)
        lm = UnifiedLM(
            providers=[
                ("anthropic", "claude-3-5-haiku-20241022"),
                ("gemini", "gemini-2.0-flash-exp")
            ]
        )

        # Future (Anthropic → Gemini → OpenAI)
        lm = UnifiedLM(
            providers=[
                ("anthropic", "claude-3-5-sonnet-20241022"),
                ("gemini", "gemini-2.0-flash-exp"),
                ("openai", "gpt-4o-mini")
            ]
        )

        # Simplified (use defaults)
        lm = UnifiedLM()  # Defaults to anthropic → gemini
    """

    def __init__(
        self,
        providers: Optional[List[tuple[str, str]]] = None,
        enable_caching: bool = True,
        **provider_configs
    ):
        """
        Initialize unified LLM with provider chain.

        Args:
            providers: List of (provider_name, model) tuples in priority order
                      Default: [("anthropic", "claude-3-5-sonnet"), ("gemini", "gemini-2.0-flash")]
            enable_caching: Enable prompt caching (where supported)
            **provider_configs: Additional config per provider
        """
        # Default provider chain
        # Primary: Anthropic Claude via CLI subscription (uses Claude Code MAX)
        # Fallback: Google Gemini (uses API key)
        if providers is None:
            providers = [
                ("anthropic", "claude-sonnet-4.5"),  # Latest Sonnet model
                ("gemini", "gemini-2.0-flash-exp")   # Fallback
            ]

        # Initialize provider instances in priority order
        self.provider_chain: List[LLMProvider] = []

        for provider_name, model in providers:
            provider_class = PROVIDER_REGISTRY.get(provider_name)
            if not provider_class:
                logger.warning(f"Unknown provider: {provider_name}, skipping")
                continue

            # Provider-specific config
            config = provider_configs.get(provider_name, {})

            # Add caching config for Anthropic
            if provider_name == "anthropic":
                config["enable_caching"] = enable_caching

            # Instantiate provider
            provider_instance = provider_class(model=model, **config)
            self.provider_chain.append(provider_instance)

        if not self.provider_chain:
            raise ValueError("No valid providers configured!")

        # Global metrics
        self.total_calls = 0
        self.total_failures = 0

    def __call__(
        self,
        messages: Optional[List[Dict[str, Any]]] = None,
        prompt: Optional[str] = None,
        system: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Call LLM with automatic failover through provider chain.

        Tries each provider in order until one succeeds.

        Args:
            messages: List of message dicts (chat format)
            prompt: Simple string prompt (converted to messages)
            system: System message (for cache optimization)
            **kwargs: max_tokens, temperature, etc.

        Returns:
            Dict with: {text, tokens_used, cache_hit, provider, model}

        Raises:
            Exception: If all providers in chain fail
        """
        # Convert prompt to messages if needed
        if prompt and not messages:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

        # Ensure we have messages
        if not messages:
            raise ValueError("Must provide either 'messages' or 'prompt'")

        # Try each provider in chain
        errors = []
        for i, provider in enumerate(self.provider_chain):
            try:
                logger.info(
                    f"Trying provider {i+1}/{len(self.provider_chain)}: "
                    f"{provider.model} ({provider.__class__.__name__})"
                )

                result = provider(messages, **kwargs)

                self.total_calls += 1
                logger.info(
                    f"Provider succeeded: {provider.model} "
                    f"(cache_hit={result.get('cache_hit', False)})"
                )

                return result

            except Exception as e:
                error_msg = f"{provider.model}: {str(e)}"
                errors.append(error_msg)
                logger.warning(f"Provider failed: {error_msg}")

                # If not last provider, continue to next
                if i < len(self.provider_chain) - 1:
                    logger.info(f"Trying next provider in chain...")
                continue

        # All providers failed
        self.total_failures += 1
        error_summary = "\n".join([f"  {i+1}. {e}" for i, e in enumerate(errors)])
        raise Exception(
            f"All {len(self.provider_chain)} providers failed:\n{error_summary}"
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive usage metrics."""
        # Aggregate metrics from all providers
        provider_metrics = []
        total_calls_per_provider = 0
        total_tokens = 0
        total_cache_hits = 0
        total_cache_misses = 0

        for provider in self.provider_chain:
            total_calls_per_provider += provider.calls
            total_tokens += provider.tokens_used

            # Cache metrics (if supported by provider)
            if hasattr(provider, 'cache_hits'):
                total_cache_hits += provider.cache_hits
                total_cache_misses += provider.cache_misses

            provider_metrics.append({
                "provider": provider.__class__.__name__.replace("Provider", "").lower(),
                "model": provider.model,
                "calls": provider.calls,
                "tokens_used": provider.tokens_used,
                "cache_hits": getattr(provider, 'cache_hits', 0),
                "cache_misses": getattr(provider, 'cache_misses', 0)
            })

        return {
            "total_calls": self.total_calls,
            "total_failures": self.total_failures,
            "failure_rate": self.total_failures / self.total_calls if self.total_calls > 0 else 0,
            "total_tokens": total_tokens,
            "cache_hits": total_cache_hits,
            "cache_misses": total_cache_misses,
            "cache_hit_rate": total_cache_hits / (total_cache_hits + total_cache_misses)
                if (total_cache_hits + total_cache_misses) > 0 else 0,
            "provider_chain": [
                {
                    "model": p.model,
                    "provider": p.__class__.__name__.replace("Provider", "").lower()
                }
                for p in self.provider_chain
            ],
            "per_provider": provider_metrics
        }


# Global metrics aggregator (optional)
class LLMMetricsAggregator:
    """Aggregate metrics across all UnifiedLM instances"""

    _instances: List[UnifiedLM] = []

    @classmethod
    def register(cls, instance: UnifiedLM):
        cls._instances.append(instance)

    @classmethod
    def get_global_metrics(cls) -> Dict[str, Any]:
        total_calls = sum(i.total_calls for i in cls._instances)
        total_failures = sum(i.total_failures for i in cls._instances)

        # Aggregate provider-specific metrics
        all_provider_metrics = {}
        for instance in cls._instances:
            for provider in instance.provider_chain:
                key = f"{provider.__class__.__name__}:{provider.model}"
                if key not in all_provider_metrics:
                    all_provider_metrics[key] = {
                        "calls": 0,
                        "tokens": 0,
                        "cache_hits": 0,
                        "cache_misses": 0
                    }
                all_provider_metrics[key]["calls"] += provider.calls
                all_provider_metrics[key]["tokens"] += provider.tokens_used
                all_provider_metrics[key]["cache_hits"] += getattr(provider, 'cache_hits', 0)
                all_provider_metrics[key]["cache_misses"] += getattr(provider, 'cache_misses', 0)

        return {
            "total_calls": total_calls,
            "total_failures": total_failures,
            "global_failure_rate": total_failures / total_calls if total_calls > 0 else 0,
            "active_instances": len(cls._instances),
            "by_provider": all_provider_metrics
        }


# Quick test
if __name__ == "__main__":
    import traceback

    # Initialize UnifiedLM
    lm = UnifiedLM()

    # Test call
    try:
        response = lm(prompt="What is the Viable System Model in one sentence?", max_tokens=100)
        print(f"✓ Response: {response['text'][:100]}...")
        print(f"✓ Provider: {response['provider']}")
        print(f"✓ Model: {response['model']}")
        print(f"✓ Cache hit: {response['cache_hit']}")

        # Check metrics
        metrics = lm.get_metrics()
        print(f"\n✓ Metrics: {metrics}")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        traceback.print_exc()
