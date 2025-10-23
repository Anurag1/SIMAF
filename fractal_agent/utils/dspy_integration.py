"""
DSPy Integration for Fractal Agent Ecosystem

Provides DSPy-compatible LM wrapper for UnifiedLM, enabling:
- Declarative signature-based prompting
- Self-optimizing agent modules
- Automatic prompt engineering
- Integration with tier-based model selection

DSPy (Declarative Self-improving Python) is a framework for building
self-improving language model programs through optimization.

Author: BMad
Date: 2025-10-18
"""

import dspy
from typing import Optional, List, Dict, Any, Literal
from .llm_provider import UnifiedLM
from .model_config import configure_lm, Tier
from ..observability import instrument_llm
import logging

logger = logging.getLogger(__name__)


class FractalDSpyLM(dspy.LM):
    """
    DSPy-compatible language model wrapper for UnifiedLM.

    This class bridges our UnifiedLM provider system with DSPy's
    declarative programming interface, enabling:
    - Signature-based prompting
    - Automatic prompt optimization
    - Self-improving agent modules
    - Tier-based model selection

    Usage:
        # Basic usage
        lm = FractalDSpyLM(tier="balanced")
        dspy.configure(lm=lm)

        # With custom config
        lm = FractalDSpyLM(
            tier="expensive",
            providers=["anthropic"],
            require_caching=True
        )

        # Use with DSPy signatures
        class QA(dspy.Signature):
            question = dspy.InputField()
            answer = dspy.OutputField()

        predictor = dspy.Predict(QA)
        result = predictor(question="What is VSM?")
    """

    def __init__(
        self,
        tier: Tier = "balanced",
        providers: Optional[List[str]] = None,
        require_vision: bool = False,
        require_caching: bool = False,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **unified_lm_kwargs
    ):
        """
        Initialize FractalDSpyLM.

        Args:
            tier: Model tier (cheap/balanced/expensive/premium)
            providers: List of providers (default: ["anthropic", "gemini"])
            require_vision: Require vision support
            require_caching: Require caching support
            max_tokens: Max tokens for generation (None = unlimited, use model defaults)
            temperature: Default temperature for generation
            **unified_lm_kwargs: Additional UnifiedLM arguments
        """
        # Initialize parent DSPy LM
        super().__init__(model=f"fractal-{tier}")

        # Store configuration
        self.tier = tier
        self.providers = providers
        self.require_vision = require_vision
        self.require_caching = require_caching
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Create UnifiedLM instance
        base_lm = configure_lm(
            tier=tier,
            providers=providers,
            require_vision=require_vision,
            require_caching=require_caching,
            **unified_lm_kwargs
        )

        # Wrap with observability instrumentation
        # This adds tracing, logging, events, and metrics to all LLM calls
        self.unified_lm = instrument_llm(base_lm, tier=f"FractalDSpy_{tier}")

        # Track history for optimization
        self.history: List[Dict[str, Any]] = []

        logger.info(f"Initialized FractalDSpyLM with tier={tier} (instrumented)")

    def basic_request(
        self,
        prompt: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        DSPy-compatible basic request method.

        This method is called by DSPy's predict/chain/react modules.

        Args:
            prompt: The prompt to send
            **kwargs: Generation parameters (max_tokens, temperature, etc.)

        Returns:
            Dictionary with response and metadata
        """
        # Merge default params with kwargs
        generation_params = {}

        # Only add max_tokens if it's specified (not None)
        max_tokens_value = kwargs.get("max_tokens", self.max_tokens)
        if max_tokens_value is not None:
            generation_params["max_tokens"] = max_tokens_value

        # Always include temperature
        generation_params["temperature"] = kwargs.get("temperature", self.temperature)

        # Add any other kwargs
        for key, value in kwargs.items():
            if key not in ["max_tokens", "temperature"]:
                generation_params[key] = value

        # Call UnifiedLM
        try:
            response = self.unified_lm(
                prompt=prompt,
                **generation_params
            )

            # Store in history
            self.history.append({
                "prompt": prompt,
                "response": response['text'],
                "provider": response['provider'],
                "model": response['model'],
                "tokens_used": response['tokens_used'],
                **generation_params
            })

            # Return in DSPy format
            return {
                "choices": [{
                    "text": response['text'],
                    "finish_reason": "stop"
                }],
                "usage": {
                    "total_tokens": response['tokens_used']
                },
                "metadata": {
                    "provider": response['provider'],
                    "model": response['model'],
                    "tier": self.tier,
                    "cache_hit": response.get('cache_hit', False)
                }
            }

        except Exception as e:
            logger.error(f"Error in basic_request: {e}")
            raise

    def __call__(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> List[str]:
        """
        DSPy-compatible call method.

        Args:
            prompt: Simple prompt string
            messages: Chat-style messages
            **kwargs: Generation parameters

        Returns:
            List of generated completions (usually just one)
        """
        # Convert to prompt if messages provided
        if messages and not prompt:
            # Format messages as prompt
            prompt_parts = []
            for msg in messages:
                role = msg.get("role", "user").upper()
                content = msg.get("content", "")
                prompt_parts.append(f"{role}: {content}")
            prompt = "\n\n".join(prompt_parts)

        if not prompt:
            raise ValueError("Either prompt or messages must be provided")

        # Call basic_request
        response = self.basic_request(prompt, **kwargs)

        # Extract completions
        completions = [choice["text"] for choice in response["choices"]]

        return completions

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get usage metrics for this LM instance.

        Returns:
            Dictionary with metrics including:
            - Total calls
            - Total tokens used
            - Provider distribution
            - Average tokens per call
        """
        if not self.history:
            return {
                "total_calls": 0,
                "total_tokens": 0,
                "provider_distribution": {},
                "avg_tokens_per_call": 0
            }

        total_calls = len(self.history)
        total_tokens = sum(h["tokens_used"] for h in self.history)

        # Provider distribution
        provider_dist = {}
        for h in self.history:
            provider = h["provider"]
            provider_dist[provider] = provider_dist.get(provider, 0) + 1

        return {
            "total_calls": total_calls,
            "total_tokens": total_tokens,
            "provider_distribution": provider_dist,
            "avg_tokens_per_call": total_tokens / total_calls if total_calls > 0 else 0,
            "tier": self.tier
        }

    def clear_history(self):
        """Clear the history (useful for testing/optimization)"""
        self.history = []

    def __deepcopy__(self, memo):
        """
        Custom deepcopy implementation for MIPRO compatibility.

        MIPRO tries to deepcopy the LM instance, but provider clients
        and other state can't be serialized. We create a fresh instance
        with the same configuration instead.
        """
        import copy

        # Create a new instance with same config (fresh clients)
        new_instance = FractalDSpyLM(
            tier=self.tier,
            providers=copy.deepcopy(self.providers, memo) if self.providers else None,
            require_vision=self.require_vision,
            require_caching=self.require_caching,
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )

        # Don't copy history (each instance should have its own)
        # memo tracks what we've copied to avoid infinite recursion
        memo[id(self)] = new_instance

        return new_instance


def configure_dspy(
    tier: Tier = "balanced",
    providers: Optional[List[str]] = None,
    **kwargs
) -> FractalDSpyLM:
    """
    Configure DSPy with FractalDSpyLM and set as global LM.

    This is a convenience function that:
    1. Creates FractalDSpyLM instance
    2. Sets it as the global DSPy LM via dspy.configure()
    3. Returns the instance for further use

    Args:
        tier: Model tier (cheap/balanced/expensive/premium)
        providers: List of providers
        **kwargs: Additional FractalDSpyLM arguments

    Returns:
        Configured FractalDSpyLM instance

    Example:
        >>> lm = configure_dspy("balanced")
        >>> # Now DSPy modules will use this LM
        >>> predictor = dspy.Predict("question -> answer")
        >>> result = predictor(question="What is VSM?")
    """
    lm = FractalDSpyLM(tier=tier, providers=providers, **kwargs)
    dspy.configure(lm=lm)
    logger.info(f"Configured DSPy with FractalDSpyLM (tier={tier})")
    return lm


# Convenience functions for each tier
def configure_dspy_cheap(**kwargs) -> FractalDSpyLM:
    """Configure DSPy with cheap tier models"""
    return configure_dspy(tier="cheap", **kwargs)


def configure_dspy_balanced(**kwargs) -> FractalDSpyLM:
    """Configure DSPy with balanced tier models"""
    return configure_dspy(tier="balanced", **kwargs)


def configure_dspy_expensive(**kwargs) -> FractalDSpyLM:
    """Configure DSPy with expensive tier models"""
    return configure_dspy(tier="expensive", **kwargs)


def configure_dspy_premium(**kwargs) -> FractalDSpyLM:
    """Configure DSPy with premium tier models"""
    return configure_dspy(tier="premium", **kwargs)


# Quick test
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("=" * 80)
    print("DSPy Integration Test")
    print("=" * 80)
    print()

    # Test 1: Basic FractalDSpyLM
    print("Test 1: Basic FractalDSpyLM")
    print("-" * 80)

    lm = FractalDSpyLM(tier="balanced", max_tokens=100)
    completions = lm(prompt="What is 2+2? Answer with just the number.")

    print(f"Prompt: What is 2+2?")
    print(f"Response: {completions[0]}")
    print(f"Metrics: {lm.get_metrics()}")
    print()

    # Test 2: DSPy Signature
    print("Test 2: DSPy Signature-based prompting")
    print("-" * 80)

    # Configure DSPy globally
    configure_dspy("balanced", max_tokens=150)

    # Define a signature
    class SimpleMath(dspy.Signature):
        """Solve simple math problems"""
        problem = dspy.InputField(desc="A simple math problem")
        answer = dspy.OutputField(desc="The numeric answer")

    # Create predictor
    math_solver = dspy.Predict(SimpleMath)

    # Test it
    result = math_solver(problem="What is 15 + 27?")
    print(f"Problem: What is 15 + 27?")
    print(f"Answer: {result.answer}")
    print()

    # Test 3: Chain of Thought
    print("Test 3: DSPy Chain of Thought")
    print("-" * 80)

    class ReasoningTask(dspy.Signature):
        """Solve problems with reasoning"""
        question = dspy.InputField()
        answer = dspy.OutputField()

    # Use ChainOfThought module
    reasoner = dspy.ChainOfThought(ReasoningTask)

    result = reasoner(question="If a train travels 60 mph for 2.5 hours, how far does it go?")
    print(f"Question: If a train travels 60 mph for 2.5 hours, how far does it go?")
    print(f"Answer: {result.answer}")
    if hasattr(result, 'rationale'):
        print(f"Rationale: {result.rationale}")
    print()

    # Test 4: Different tiers
    print("Test 4: Testing different tiers")
    print("-" * 80)

    for tier in ["cheap", "balanced"]:
        lm = FractalDSpyLM(tier=tier, max_tokens=50)
        response = lm(prompt="Say 'hello' in French")
        print(f"{tier.upper():10} → {response[0][:50]}...")
        print(f"           Metrics: {lm.get_metrics()}")

    print()
    print("=" * 80)
    print("✓ DSPy integration test complete!")
    print("=" * 80)
