"""
IntelligenceAgent Configuration

System 4 (Intelligence) - Performance reflection and learning

Defines tier selection for intelligence/reflection operations.
Uses expensive models for deep analysis and pattern detection.

Author: BMad
Date: 2025-10-19
"""

from dataclasses import dataclass
from typing import Optional
from ..utils.model_config import Tier


@dataclass
class IntelligenceConfig:
    """
    Configuration for IntelligenceAgent tier selection.

    System 4 (Intelligence) in VSM performs reflection, learning,
    and pattern detection. Uses expensive models for quality analysis.

    Default tiers:
    - Analysis (expensive): Deep performance analysis
    - Pattern Detection (expensive): Identify improvement patterns
    - Insight Generation (expensive): High-quality insights
    - Prioritization (balanced): Action prioritization

    Usage:
        # Use defaults (expensive models for quality)
        config = IntelligenceConfig()

        # Custom tiers
        config = IntelligenceConfig(
            analysis_tier="premium",
            pattern_tier="expensive"
        )

        # With analysis parameters
        config = IntelligenceConfig(
            min_session_size=10,
            lookback_days=30
        )

        # Use with agent
        agent = IntelligenceAgent(config=config)
    """
    # Model tiers for each stage
    analysis_tier: Tier = "expensive"       # Deep performance analysis
    pattern_tier: Tier = "expensive"        # Pattern detection
    insight_tier: Tier = "expensive"        # Insight generation
    prioritization_tier: Tier = "balanced"  # Action prioritization

    # Analysis parameters
    min_session_size: int = 5               # Minimum tasks before analysis
    lookback_days: int = 7                  # Days of history to analyze
    insight_threshold: float = 0.7          # Confidence threshold for insights

    # Reflection triggers
    analyze_on_failure: bool = True         # Trigger on task failure
    analyze_on_schedule: bool = True        # Periodic analysis
    analyze_on_cost_spike: bool = True      # Trigger on cost anomaly
    cost_spike_threshold: float = 2.0       # 2x average cost = spike

    # Output configuration
    max_recommendations: int = 5            # Top N recommendations
    include_examples: bool = True           # Include example tasks
    verbose: bool = False

    # Optional constraints
    max_tokens: Optional[int] = None        # None = unlimited
    temperature: float = 0.7                # Balanced creativity

    def __str__(self) -> str:
        """Human-readable string representation"""
        token_str = f"max_tokens={self.max_tokens}" if self.max_tokens else "unlimited tokens"
        return (
            f"IntelligenceConfig(\n"
            f"  analysis={self.analysis_tier}\n"
            f"  pattern={self.pattern_tier}\n"
            f"  insight={self.insight_tier}\n"
            f"  prioritization={self.prioritization_tier}\n"
            f"  min_session_size={self.min_session_size}\n"
            f"  lookback_days={self.lookback_days}\n"
            f"  {token_str}, temp={self.temperature}\n"
            f")"
        )


# Predefined configurations for common use cases
class PresetIntelligenceConfigs:
    """Common configuration presets for Intelligence agent"""

    @staticmethod
    def default() -> IntelligenceConfig:
        """
        Default configuration - expensive models for quality analysis.

        Suitable for:
        - Regular performance reflection
        - Weekly deep analysis
        - Production systems
        """
        return IntelligenceConfig()

    @staticmethod
    def quick_analysis() -> IntelligenceConfig:
        """
        Fast analysis with balanced models.

        Suitable for:
        - Rapid feedback during development
        - Frequent analysis (daily)
        - Cost-sensitive environments
        """
        return IntelligenceConfig(
            analysis_tier="balanced",
            pattern_tier="balanced",
            insight_tier="balanced",
            prioritization_tier="cheap",
            min_session_size=3,
            lookback_days=1,
            max_recommendations=3
        )

    @staticmethod
    def deep_analysis() -> IntelligenceConfig:
        """
        Comprehensive analysis with premium models.

        Suitable for:
        - Monthly strategic reviews
        - Critical system improvements
        - Research and optimization
        """
        return IntelligenceConfig(
            analysis_tier="premium",
            pattern_tier="expensive",
            insight_tier="expensive",
            prioritization_tier="expensive",
            min_session_size=20,
            lookback_days=30,
            max_recommendations=10,
            include_examples=True,
            verbose=True
        )

    @staticmethod
    def failure_analysis() -> IntelligenceConfig:
        """
        Focused analysis triggered by failures.

        Suitable for:
        - Post-mortem analysis
        - Debugging production issues
        - Error pattern detection
        """
        return IntelligenceConfig(
            analysis_tier="expensive",
            pattern_tier="expensive",
            insight_tier="balanced",
            prioritization_tier="balanced",
            min_session_size=1,  # Analyze even single failures
            lookback_days=1,
            analyze_on_failure=True,
            analyze_on_schedule=False,
            max_recommendations=3
        )

    @staticmethod
    def cost_optimization() -> IntelligenceConfig:
        """
        Analysis focused on cost reduction.

        Suitable for:
        - Cost spike investigation
        - Token usage optimization
        - Budget management
        """
        return IntelligenceConfig(
            analysis_tier="balanced",
            pattern_tier="balanced",
            insight_tier="balanced",
            prioritization_tier="cheap",
            min_session_size=5,
            lookback_days=7,
            analyze_on_cost_spike=True,
            cost_spike_threshold=1.5,  # 1.5x average
            max_recommendations=5
        )


if __name__ == "__main__":
    # Demo: Show all preset configurations
    print("=" * 80)
    print("Intelligence Agent Configurations")
    print("=" * 80)
    print()

    print("1. Default Configuration:")
    print(PresetIntelligenceConfigs.default())
    print()

    print("2. Quick Analysis:")
    print(PresetIntelligenceConfigs.quick_analysis())
    print()

    print("3. Deep Analysis:")
    print(PresetIntelligenceConfigs.deep_analysis())
    print()

    print("4. Failure Analysis:")
    print(PresetIntelligenceConfigs.failure_analysis())
    print()

    print("5. Cost Optimization:")
    print(PresetIntelligenceConfigs.cost_optimization())
    print()
