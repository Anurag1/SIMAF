# Fractal Agent Ecosystem - Unified LLM Architecture

## Architectural Principle

**ALL LLM calls in the system use the `UnifiedLM` class.**

- Extensible provider chain (currently: Anthropic → Gemini, easily expandable)
- Model selection: Configuration parameter, not code variation
- Automatic failover through priority-ordered providers
- Centralized retry, metrics, and caching infrastructure
- Future-proof: Add providers without changing agent code

---

## Implementation

### Core Provider Class

```python
# src/fractal_agent/utils/llm_provider.py

import anthropic
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential
import os
from typing import Optional, Dict, Any, List, Callable
from abc import ABC, abstractmethod
import logging

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
    """Anthropic Claude provider"""

    def __init__(self, model: str, enable_caching: bool = True, **config):
        super().__init__(model, **config)
        self.client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        self.enable_caching = enable_caching
        self.cache_hits = 0
        self.cache_misses = 0

    def _call_provider(
        self,
        messages: List[Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        # Add cache control to system messages if enabled
        if self.enable_caching:
            for msg in messages:
                if msg.get("role") == "system" and "cache_control" not in msg:
                    msg["cache_control"] = {"type": "ephemeral"}

        response = self.client.messages.create(
            model=self.model,
            messages=messages,
            **kwargs
        )

        text = response.content[0].text
        tokens_used = response.usage.input_tokens + response.usage.output_tokens

        # Check cache hit
        cache_hit = False
        if hasattr(response.usage, 'cache_read_input_tokens'):
            cache_hit = response.usage.cache_read_input_tokens > 0
            if cache_hit:
                self.cache_hits += 1
            else:
                self.cache_misses += 1

        return {
            "text": text,
            "tokens_used": tokens_used,
            "cache_hit": cache_hit,
            "provider": "anthropic",
            "model": self.model
        }


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

        response = self.client.generate_content(prompt, **kwargs)

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
        if providers is None:
            providers = [
                ("anthropic", "claude-3-5-sonnet-20241022"),
                ("gemini", "gemini-2.0-flash-exp")
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
```

---

## Model Selection Strategy

### Recommended Model Mapping

```python
# src/fractal_agent/config/model_config.py

"""
Centralized model configuration for all agents.

Change models here, not in agent code.
"""

MODEL_CONFIG = {
    # Operational agents (System 1) - Fast, cost-efficient
    "operational": {
        "primary": "claude-3-5-haiku-20241022",
        "fallback": "gemini-2.0-flash-exp",
        "rationale": "High speed, low cost, good for simple tasks"
    },

    # Control agents (System 3) - Smart reasoning for decomposition
    "control": {
        "primary": "claude-3-7-sonnet-20250219",
        "fallback": "gemini-2.0-flash-thinking-exp",
        "rationale": "Strong reasoning for task decomposition"
    },

    # Intelligence agents (System 4) - Deep analysis and reflection
    "intelligence": {
        "primary": "claude-3-7-sonnet-20250219",
        "fallback": "gemini-exp-1206",
        "rationale": "Best quality for reflection and learning"
    },

    # Knowledge extraction - Accuracy critical
    "extraction": {
        "primary": "claude-3-5-sonnet-20241022",
        "fallback": "gemini-2.0-flash-exp",
        "rationale": "Balanced quality/cost for extraction tasks"
    },

    # Synthesis/reporting - Quality output
    "synthesis": {
        "primary": "claude-3-7-sonnet-20250219",
        "fallback": "gemini-exp-1206",
        "rationale": "High quality output for final reports"
    }
}


def get_llm_for_role(role: str, **kwargs) -> UnifiedLM:
    """
    Factory function: Get LLM instance for agent role.

    Args:
        role: Agent role (operational, control, intelligence, etc.)
        **kwargs: Override default config

    Returns:
        Configured UnifiedLM instance with appropriate provider chain
    """
    config = MODEL_CONFIG.get(role, MODEL_CONFIG["operational"])

    # Build provider chain
    providers = kwargs.get("providers")
    if providers is None:
        providers = [
            ("anthropic", config["primary"]),
            ("gemini", config["fallback"])
        ]

    return UnifiedLM(providers=providers, **kwargs)
```

---

## Usage Examples

### Example 1: Operational Agent

```python
# src/fractal_agent/agents/researcher.py

from fractal_agent.utils.llm_provider import UnifiedLM
from fractal_agent.config.model_config import get_llm_for_role

class ResearcherAgent:
    """VSM System 1: Operational agent for research tasks"""

    def __init__(self):
        # Get LLM configured for operational role
        self.llm = get_llm_for_role("operational")
        # Automatic chain: claude-3-5-haiku → gemini-2.0-flash

    def research(self, question: str, context: str = "") -> str:
        system = "You are a research agent. Provide accurate, concise answers."

        response = self.llm(
            prompt=question,
            system=system,  # Cached by Anthropic!
            max_tokens=2000
        )

        return response["text"]
```

### Example 2: Control Agent

```python
# src/fractal_agent/agents/control_agent.py

from fractal_agent.utils.llm_provider import UnifiedLM
from fractal_agent.config.model_config import get_llm_for_role

class ControlAgent:
    """VSM System 3: Decomposes tasks"""

    def __init__(self):
        # Get LLM configured for control role
        self.llm = get_llm_for_role("control")
        # Automatic chain: claude-3-7-sonnet → gemini-thinking

    def decompose(self, complex_task: str, available_agents: str) -> dict:
        system = """You are a control agent. Break down complex tasks into
        manageable subtasks. Think step-by-step."""

        prompt = f"""
        Task: {complex_task}
        Available agents: {available_agents}

        Decompose this task into subtasks.
        """

        response = self.llm(
            prompt=prompt,
            system=system,  # Cached!
            max_tokens=4000
        )

        return {"subtasks": response["text"]}
```

### Example 3: Reflection Agent

```python
# src/fractal_agent/agents/reflection_agent.py

from fractal_agent.utils.llm_provider import UnifiedLM
from fractal_agent.config.model_config import get_llm_for_role

class ReflectionAgent:
    """VSM System 4: Analyzes performance"""

    def __init__(self):
        # Get LLM configured for intelligence role
        self.llm = get_llm_for_role("intelligence")
        # Automatic chain: claude-3-7-sonnet → gemini-exp

    def reflect(self, task: str, trace: str, outcome: str) -> dict:
        system = """You are a reflection agent. Analyze task execution deeply.
        Identify patterns, suggest improvements, generate insights."""

        prompt = f"""
        Task: {task}
        Execution trace: {trace}
        Outcome: {outcome}

        Provide:
        1. Critique of performance
        2. Actionable suggestions
        3. Confidence in assessment (0-1)
        """

        response = self.llm(
            prompt=prompt,
            system=system,  # Cached!
            max_tokens=6000
        )

        return {"critique": response["text"]}
```

### Example 4: Knowledge Extraction

```python
# src/fractal_agent/pipelines/log_to_graph.py

from fractal_agent.utils.llm_provider import UnifiedLM
from fractal_agent.config.model_config import get_llm_for_role

class LogExtractor:
    """Extract knowledge from agent logs"""

    def __init__(self):
        self.llm = get_llm_for_role("extraction")
        # Automatic chain: claude-3-5-sonnet → gemini-flash

    def extract_triples(self, log_content: str) -> dict:
        system = """Extract entities and relationships from agent logs.
        Output format: JSON with entities and relationships arrays."""

        response = self.llm(
            prompt=f"Extract knowledge from:\n{log_content}",
            system=system,  # Cached!
            max_tokens=3000
        )

        return {"triples": response["text"]}
```

---

## Benefits of This Architecture

### 1. **Single Point of Control**

- ALL LLM calls go through one class
- Change failover logic once, benefits entire system
- Add features (rate limiting, cost caps) in one place

### 2. **Consistent Behavior**

- Same retry logic everywhere
- Same failover behavior everywhere
- Same metrics collection everywhere

### 3. **Model Selection = Configuration**

- Not a code change, just config
- Easy to experiment: swap models, compare performance
- A/B test model combinations

### 4. **Built-in Resilience**

- Every LLM call automatically has failover
- Every call has retry logic
- No single point of failure anywhere

### 5. **Comprehensive Metrics**

- Track usage per provider
- Monitor failover rates
- Measure cache hit rates
- Feed directly to Prometheus

### 6. **Cost Optimization**

- Prompt caching built-in (90% savings)
- Right model for right task (operational→cheap, control→smart)
- Metrics enable cost analysis

### 7. **Future-Proof**

- Want to add a third provider? Extend `DualProviderLM`
- Want smart routing by task type? Add to `get_llm_for_role()`
- Want cost-based selection? Add logic to `__call__()`

---

## Integration with DSPy

```python
# src/fractal_agent/utils/dspy_integration.py

import dspy
from fractal_agent.utils.llm_provider import UnifiedLM
from fractal_agent.config.model_config import get_llm_for_role

class FractalDSpyLM(dspy.LM):
    """
    DSPy-compatible LM wrapper for Fractal Agent system.

    ALL DSPy agents use this.
    Supports extensible provider chain with automatic failover.
    """

    def __init__(self, role: str = "operational", **kwargs):
        self.provider = get_llm_for_role(role, **kwargs)
        self.model = self.provider.provider_chain[0].model  # Primary model
        self.history = []

    def __call__(self, prompt=None, messages=None, **kwargs):
        result = self.provider(
            prompt=prompt,
            messages=messages,
            **kwargs
        )

        # Track for DSPy
        self.history.append({
            "prompt": prompt or messages,
            "response": result["text"],
            "provider": result["provider"],
            "model": result["model"],
            "cache_hit": result["cache_hit"],
            "tokens_used": result["tokens_used"]
        })

        return result["text"]

    def inspect_history(self, n: int = 1):
        return self.history[-n:]


# Configure DSPy globally
def configure_dspy_for_role(role: str, **kwargs):
    """Set up DSPy with our unified LLM provider"""
    lm = FractalDSpyLM(role=role, **kwargs)
    dspy.configure(lm=lm)
    return lm
```

---

## Next Steps

1. **Phase 0**: Implement `UnifiedLM` class with extensible provider chain
2. **Phase 0**: Implement `AnthropicProvider` and `GeminiProvider`
3. **Phase 0**: Create `model_config.py` with role mappings
4. **Phase 1**: Migrate all agents to use `get_llm_for_role()`
5. **Phase 1**: Integrate with DSPy via `FractalDSpyLM`
6. **Phase 2**: Add Prometheus metrics export
7. **Phase 3**: Monitor cache hit rates, optimize prompt structure
8. **Phase 4**: Experiment with model selection, A/B test combinations
9. **Future**: Add new providers as needed (see below)

---

## Adding New Providers (Future Extensibility)

To add a third provider (e.g., OpenAI), follow this pattern:

```python
# 1. Implement provider class
class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider"""

    def __init__(self, model: str, **config):
        super().__init__(model, **config)
        from openai import OpenAI
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _call_provider(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )

        return {
            "text": response.choices[0].message.content,
            "tokens_used": response.usage.total_tokens,
            "cache_hit": False,  # OpenAI doesn't support caching (yet)
            "provider": "openai",
            "model": self.model
        }

# 2. Register in PROVIDER_REGISTRY
PROVIDER_REGISTRY["openai"] = OpenAIProvider

# 3. Use in provider chain
lm = UnifiedLM(
    providers=[
        ("anthropic", "claude-3-5-sonnet-20241022"),
        ("gemini", "gemini-2.0-flash-exp"),
        ("openai", "gpt-4o-mini")  # Third fallback!
    ]
)
```

**That's it!** All existing agents now have OpenAI as a third fallback, zero code changes needed.

---

## Configuration File (.env)

```bash
# Anthropic (Primary Provider)
ANTHROPIC_API_KEY=sk-ant-...

# Google Gemini (Fallback Provider)
GOOGLE_API_KEY=...

# Future: Additional providers
# OPENAI_API_KEY=sk-...
# TOGETHER_API_KEY=...
```

---

## Testing the Provider Chain

```python
# Quick test script
if __name__ == "__main__":
    from fractal_agent.utils.llm_provider import UnifiedLM

    # Create LLM with default chain
    lm = UnifiedLM()

    # Test call
    response = lm(prompt="What is the Viable System Model?")
    print(f"Response: {response['text'][:100]}...")
    print(f"Provider: {response['provider']}")
    print(f"Model: {response['model']}")
    print(f"Cache hit: {response['cache_hit']}")

    # Check metrics
    metrics = lm.get_metrics()
    print(f"\nMetrics: {metrics}")

    # Test failover (simulate Anthropic failure)
    # You'd need to actually break Anthropic to test this in practice
```

---

**Architecture Status**: ✅ **APPROVED & EXTENSIBLE**

This is the official LLM interface for the Fractal Agent Ecosystem.

- All agents, all tasks, all LLM calls → `UnifiedLM`
- Currently: Anthropic (primary) → Gemini (fallback)
- Future: Add providers by implementing `LLMProvider` subclass
- Zero agent code changes when adding providers
