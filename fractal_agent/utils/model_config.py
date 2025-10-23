"""
Model Configuration with Tier-Based Selection

Provides semantic tier-based model selection instead of hardcoded model names.
Integrates ModelRegistry with UnifiedLM for dynamic provider chain configuration.

Usage:
    # Simple tier-based selection
    lm = configure_lm("balanced")  # Returns UnifiedLM with balanced models

    # With custom config
    lm = configure_lm("cheap", require_vision=True, fallback_provider="gemini")

    # Get provider chain only (for custom UnifiedLM init)
    chain = get_provider_chain("expensive", providers=["anthropic", "gemini"])

Author: BMad
Date: 2025-10-18
"""

from typing import List, Tuple, Optional, Literal
from .model_registry import ModelRegistry, get_registry, Tier, Provider
from .llm_provider import UnifiedLM
import logging

logger = logging.getLogger(__name__)


def get_provider_chain(
    tier: Tier,
    providers: Optional[List[Provider]] = None,
    require_vision: bool = False,
    require_caching: bool = False,
    fallback_tiers: bool = True
) -> List[Tuple[Provider, str]]:
    """
    Get a provider chain for the specified tier.

    Args:
        tier: Desired tier (cheap/balanced/expensive/premium)
        providers: List of providers to use (default: ["anthropic", "gemini"])
        require_vision: Only select models with vision support
        require_caching: Only select models with caching support
        fallback_tiers: If no model found in tier, try lower tiers

    Returns:
        List of (provider_name, model_id) tuples

    Example:
        >>> chain = get_provider_chain("balanced", providers=["anthropic", "gemini"])
        >>> # Returns: [("anthropic", "claude-sonnet-4.5"), ("gemini", "gemini-1.5-pro")]
    """
    if providers is None:
        providers = ["anthropic", "gemini"]

    registry = get_registry()
    chain = []

    # Try each provider in order
    for provider in providers:
        model = registry.get_model_by_tier(
            tier=tier,
            provider=provider,
            require_vision=require_vision,
            require_caching=require_caching
        )

        if model:
            chain.append((provider, model.model_id))
            logger.debug(f"Selected {model.model_id} for tier={tier}, provider={provider}")
        else:
            logger.warning(
                f"No {tier} model found for {provider} "
                f"(vision={require_vision}, caching={require_caching})"
            )

            # Try fallback tiers if enabled
            if fallback_tiers:
                fallback_model = _get_fallback_model(
                    provider=provider,
                    tier=tier,
                    require_vision=require_vision,
                    require_caching=require_caching,
                    registry=registry
                )
                if fallback_model:
                    chain.append((provider, fallback_model.model_id))
                    logger.info(
                        f"Using fallback {fallback_model.model_id} "
                        f"(tier={fallback_model.tier}) for {provider}"
                    )

    if not chain:
        raise ValueError(
            f"No models found for tier={tier}, providers={providers}. "
            f"Check ModelRegistry configuration."
        )

    return chain


def _get_fallback_model(
    provider: Provider,
    tier: Tier,
    require_vision: bool,
    require_caching: bool,
    registry: ModelRegistry
):
    """Try to find a model in a lower tier as fallback"""
    tier_order: List[Tier] = ["cheap", "balanced", "expensive", "premium"]

    # Get tiers below the requested tier
    try:
        current_index = tier_order.index(tier)
        fallback_tiers = tier_order[:current_index][::-1]  # Reverse for closest match
    except ValueError:
        return None

    # Try each fallback tier
    for fallback_tier in fallback_tiers:
        model = registry.get_model_by_tier(
            tier=fallback_tier,
            provider=provider,
            require_vision=require_vision,
            require_caching=require_caching
        )
        if model:
            return model

    return None


def configure_lm(
    tier: Tier = "balanced",
    providers: Optional[List[Provider]] = None,
    require_vision: bool = False,
    require_caching: bool = False,
    fallback_tiers: bool = True,
    **unified_lm_kwargs
) -> UnifiedLM:
    """
    Create a UnifiedLM instance with tier-based model selection.

    This is the main entry point for creating LLM instances with semantic tier names
    instead of hardcoded model IDs.

    Args:
        tier: Desired tier (cheap/balanced/expensive/premium)
        providers: List of providers to use (default: ["anthropic", "gemini"])
        require_vision: Only select models with vision support
        require_caching: Only select models with caching support
        fallback_tiers: Allow fallback to lower tiers if no match
        **unified_lm_kwargs: Additional arguments for UnifiedLM (e.g., enable_caching)

    Returns:
        Configured UnifiedLM instance

    Examples:
        >>> # Balanced tier with default providers
        >>> lm = configure_lm("balanced")

        >>> # Cheap tier with vision support
        >>> lm = configure_lm("cheap", require_vision=True)

        >>> # Premium tier, Anthropic only
        >>> lm = configure_lm("premium", providers=["anthropic"])

        >>> # Custom config
        >>> lm = configure_lm(
        ...     tier="expensive",
        ...     providers=["anthropic", "gemini"],
        ...     require_caching=True,
        ...     enable_caching=True  # Pass to UnifiedLM
        ... )
    """
    # Get provider chain
    chain = get_provider_chain(
        tier=tier,
        providers=providers,
        require_vision=require_vision,
        require_caching=require_caching,
        fallback_tiers=fallback_tiers
    )

    logger.info(
        f"Configuring UnifiedLM with tier={tier}: "
        f"{' → '.join([f'{p}:{m}' for p, m in chain])}"
    )

    # Create UnifiedLM instance
    return UnifiedLM(providers=chain, **unified_lm_kwargs)


# Convenience functions for each tier
def configure_cheap_lm(**kwargs) -> UnifiedLM:
    """
    Create UnifiedLM with cheap tier models.

    Best for: High-volume simple tasks, classification, simple Q&A

    Example:
        >>> lm = configure_cheap_lm()
        >>> response = lm(prompt="Is this sentiment positive or negative: 'I love this!'")
    """
    return configure_lm(tier="cheap", **kwargs)


def configure_balanced_lm(**kwargs) -> UnifiedLM:
    """
    Create UnifiedLM with balanced tier models.

    Best for: Most production workloads, code generation, analysis

    Example:
        >>> lm = configure_balanced_lm()
        >>> response = lm(prompt="Write a Python function to parse JSON")
    """
    return configure_lm(tier="balanced", **kwargs)


def configure_expensive_lm(**kwargs) -> UnifiedLM:
    """
    Create UnifiedLM with expensive tier models.

    Best for: Complex reasoning, advanced code review, deep research

    Example:
        >>> lm = configure_expensive_lm()
        >>> response = lm(prompt="Design a distributed system architecture for...")
    """
    return configure_lm(tier="expensive", **kwargs)


def configure_premium_lm(**kwargs) -> UnifiedLM:
    """
    Create UnifiedLM with premium tier models.

    Best for: Mission-critical decisions, strategic planning, expert reasoning

    Example:
        >>> lm = configure_premium_lm()
        >>> response = lm(prompt="Analyze the viability of this business strategy...")
    """
    return configure_lm(tier="premium", **kwargs)


def get_model_recommendations(task_type: str) -> Tier:
    """
    Get recommended tier based on task type.

    Args:
        task_type: Type of task (e.g., "classification", "code_generation", etc.)

    Returns:
        Recommended tier

    Example:
        >>> tier = get_model_recommendations("code_generation")
        >>> lm = configure_lm(tier)
    """
    recommendations = {
        # Cheap tier tasks
        "classification": "cheap",
        "sentiment_analysis": "cheap",
        "simple_qa": "cheap",
        "content_moderation": "cheap",
        "syntax_validation": "cheap",
        "keyword_extraction": "cheap",

        # Balanced tier tasks
        "code_generation": "balanced",
        "document_analysis": "balanced",
        "research_synthesis": "balanced",
        "agent_workflows": "balanced",
        "code_review": "balanced",
        "technical_writing": "balanced",
        "data_transformation": "balanced",

        # Expensive tier tasks
        "complex_reasoning": "expensive",
        "architecture_design": "expensive",
        "advanced_code_review": "expensive",
        "deep_research": "expensive",
        "problem_solving": "expensive",
        "system_design": "expensive",

        # Premium tier tasks
        "strategic_planning": "premium",
        "critical_decisions": "premium",
        "expert_analysis": "premium",
        "complex_system_design": "premium",
        "risk_assessment": "premium"
    }

    return recommendations.get(task_type.lower(), "balanced")


def list_available_configs() -> dict:
    """
    List all available model configurations by tier.

    Returns:
        Dictionary mapping tiers to available models per provider

    Example:
        >>> configs = list_available_configs()
        >>> print(configs["balanced"]["anthropic"])
        >>> # "claude-sonnet-4.5"
    """
    registry = get_registry()
    configs = {}

    for tier in ["cheap", "balanced", "expensive", "premium"]:
        configs[tier] = {}
        for provider in ["anthropic", "gemini"]:
            model = registry.get_model_by_tier(tier, provider=provider)
            if model:
                configs[tier][provider] = {
                    "model_id": model.model_id,
                    "input_cost": model.input_cost_per_mtok,
                    "output_cost": model.output_cost_per_mtok,
                    "context_window": model.context_window,
                    "supports_vision": model.supports_vision,
                    "supports_caching": model.supports_caching
                }

    return configs


# Quick test
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("=" * 80)
    print("Model Configuration Test")
    print("=" * 80)
    print()

    # Test provider chain generation
    print("Testing provider chain generation:")
    print("-" * 80)

    for tier in ["cheap", "balanced", "expensive", "premium"]:
        try:
            chain = get_provider_chain(tier)
            print(f"{tier.upper():10} → {chain}")
        except ValueError as e:
            print(f"{tier.upper():10} → ERROR: {e}")

    print()

    # Test UnifiedLM configuration
    print("Testing UnifiedLM configuration:")
    print("-" * 80)

    try:
        # Balanced tier
        lm = configure_balanced_lm()
        response = lm(
            prompt="What is 7+5? Answer with just the number.",
            max_tokens=10
        )
        print(f"Balanced LM test: {response['text']}")
        print(f"Provider used: {response['provider']}")
        print(f"Model: {response['model']}")

    except Exception as e:
        print(f"Error: {e}")

    print()

    # Test task-based recommendations
    print("Task-based recommendations:")
    print("-" * 80)

    tasks = [
        "classification",
        "code_generation",
        "complex_reasoning",
        "strategic_planning"
    ]

    for task in tasks:
        tier = get_model_recommendations(task)
        print(f"{task:25} → {tier}")

    print()

    # List all available configs
    print("Available configurations:")
    print("-" * 80)

    configs = list_available_configs()
    for tier, providers in configs.items():
        print(f"\n{tier.upper()}:")
        for provider, config in providers.items():
            print(f"  {provider:10} → {config['model_id']:30} "
                  f"(${config['input_cost']:.2f}/${config['output_cost']:.2f} per Mtok)")

    print()
    print("=" * 80)
    print("✓ Model configuration test complete!")
    print("=" * 80)
