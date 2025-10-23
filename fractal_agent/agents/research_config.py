"""
ResearchAgent Configuration for DSPy Optimization

Defines tier selection for each research stage.
DSPy optimizes prompts and demonstrations automatically.

Author: BMad
Date: 2025-10-18
"""

from dataclasses import dataclass
from typing import Optional
from ..utils.model_config import Tier


@dataclass
class ResearchConfig:
    """
    Configuration for ResearchAgent tier selection.

    DSPy will optimize prompts and demonstrations.
    We manually select tiers (cheap/balanced/expensive/premium) per stage.

    Default tiers:
    - Planning (expensive): Better strategic decomposition
    - Research (cheap): Well-planned questions may not need expensive models
    - Synthesis (balanced): Good baseline for quality synthesis
    - Validation (balanced): Needs intelligence to catch issues

    Token limits and temperature are NOT restricted by default.
    Models will generate complete responses using their full context.

    Usage:
        # Use defaults
        config = ResearchConfig()

        # Custom tiers
        config = ResearchConfig(
            planning_tier="premium",
            research_tier="balanced"
        )

        # With optional token limits
        config = ResearchConfig(
            planning_tier="expensive",
            max_tokens=4000  # Optional constraint
        )

        # Use with agent
        agent = ResearchAgent(config=config)
    """
    planning_tier: Tier = "expensive"
    research_tier: Tier = "cheap"
    synthesis_tier: Tier = "balanced"
    validation_tier: Tier = "balanced"

    # Optional constraints (NOT enforced by default)
    max_tokens: Optional[int] = None  # None = unlimited
    temperature: float = 0.7  # Reasonable default for creative tasks

    def __str__(self) -> str:
        """Human-readable string representation"""
        token_str = f"max_tokens={self.max_tokens}" if self.max_tokens else "unlimited tokens"
        return (
            f"ResearchConfig(\n"
            f"  planning={self.planning_tier}\n"
            f"  research={self.research_tier}\n"
            f"  synthesis={self.synthesis_tier}\n"
            f"  validation={self.validation_tier}\n"
            f"  {token_str}, temp={self.temperature}\n"
            f")"
        )


# Predefined configurations for common use cases
class PresetConfigs:
    """Common configuration presets"""

    @staticmethod
    def default() -> ResearchConfig:
        """Default balanced configuration (recommended)"""
        return ResearchConfig()

    @staticmethod
    def cost_optimized() -> ResearchConfig:
        """Minimize cost (all cheap/balanced tiers)"""
        return ResearchConfig(
            planning_tier="balanced",
            research_tier="cheap",
            synthesis_tier="cheap",
            validation_tier="cheap"
        )

    @staticmethod
    def quality_optimized() -> ResearchConfig:
        """Maximize quality (expensive/premium tiers)"""
        return ResearchConfig(
            planning_tier="premium",
            research_tier="expensive",
            synthesis_tier="premium",
            validation_tier="expensive"
        )

    @staticmethod
    def fast_iteration() -> ResearchConfig:
        """Fast testing with token limits"""
        return ResearchConfig(
            planning_tier="cheap",
            research_tier="cheap",
            synthesis_tier="balanced",
            validation_tier="cheap",
            max_tokens=2000  # Constrained for speed
        )


# Quick test
if __name__ == "__main__":
    print("=" * 80)
    print("ResearchConfig Test")
    print("=" * 80)
    print()

    # Test default config
    print("1. Default configuration (unlimited tokens):")
    print("-" * 80)
    config = ResearchConfig()
    print(config)
    print()

    # Test custom tiers
    print("2. Custom tier selection:")
    print("-" * 80)
    config = ResearchConfig(
        planning_tier="premium",
        research_tier="balanced"
    )
    print(config)
    print()

    # Test with token limit
    print("3. With optional token limit:")
    print("-" * 80)
    config = ResearchConfig(
        planning_tier="expensive",
        max_tokens=4000
    )
    print(config)
    print()

    # Test presets
    print("4. Preset configurations:")
    print("-" * 80)

    print("Default:")
    print(PresetConfigs.default())
    print()

    print("Cost optimized:")
    print(PresetConfigs.cost_optimized())
    print()

    print("Quality optimized:")
    print(PresetConfigs.quality_optimized())
    print()

    print("Fast iteration (limited tokens):")
    print(PresetConfigs.fast_iteration())
    print()

    print("=" * 80)
    print("✓ ResearchConfig test complete!")
    print("=" * 80)
