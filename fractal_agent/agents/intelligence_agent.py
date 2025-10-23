"""
IntelligenceAgent - Performance Reflection and Learning

System 4 (Intelligence) in the VSM hierarchy.

Performs:
1. Performance Analysis - Analyze session logs and metrics
2. Pattern Detection - Identify recurring issues and opportunities
3. Insight Generation - Generate actionable insights
4. Prioritization - Prioritize improvement recommendations

Uses expensive tier models for deep analysis and quality insights.

Author: BMad
Date: 2025-10-19
"""

import dspy
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from ..utils.dspy_integration import FractalDSpyLM
from ..utils.model_config import Tier
from .intelligence_config import IntelligenceConfig
import logging
import json

logger = logging.getLogger(__name__)


# ============================================================================
# DSPy Signatures for Intelligence Workflow
# ============================================================================

class PerformanceAnalysis(dspy.Signature):
    """
    Analyze performance metrics and session logs.

    Identify patterns in task success/failure, cost, and latency.
    Provide statistical insights and anomaly detection.
    """
    session_logs = dspy.InputField(desc="JSON-formatted session logs with task history")
    performance_metrics = dspy.InputField(desc="Performance metrics: accuracy, cost, latency, cache_hit_rate")
    analysis = dspy.OutputField(desc="Detailed performance analysis with statistics and patterns")


class PatternDetection(dspy.Signature):
    """
    Detect patterns and recurring issues in performance data.

    Identify:
    - Common failure modes
    - Cost drivers
    - Latency bottlenecks
    - Success patterns
    """
    performance_analysis = dspy.InputField(desc="Performance analysis from previous stage")
    session_logs = dspy.InputField(desc="Session logs for pattern matching")
    patterns = dspy.OutputField(desc="List of detected patterns with confidence scores and examples")


class InsightGeneration(dspy.Signature):
    """
    Generate actionable insights from detected patterns.

    Transform patterns into specific recommendations for improvement.
    Focus on high-impact, actionable changes.
    """
    patterns = dspy.InputField(desc="Detected patterns from analysis")
    performance_metrics = dspy.InputField(desc="Current performance metrics")
    insights = dspy.OutputField(desc="Actionable insights with specific improvement recommendations")


class RecommendationPrioritization(dspy.Signature):
    """
    Prioritize improvement recommendations.

    Rank recommendations by:
    - Impact (potential improvement magnitude)
    - Effort (implementation difficulty)
    - Confidence (how certain we are this will help)
    """
    insights = dspy.InputField(desc="Generated insights and recommendations")
    performance_metrics = dspy.InputField(desc="Current performance metrics")
    action_plan = dspy.OutputField(desc="Prioritized action plan with top recommendations")


# ============================================================================
# Intelligence Result
# ============================================================================

@dataclass
class IntelligenceResult:
    """
    Result of intelligence agent reflection.

    Attributes:
        session_id: Session being analyzed
        analysis: Performance analysis
        patterns: Detected patterns
        insights: Generated insights
        action_plan: Prioritized recommendations
        metadata: Additional metadata (tokens used, models used, etc.)
    """
    session_id: str
    analysis: str
    patterns: str
    insights: str
    action_plan: str
    metadata: Dict[str, Any]

    def __str__(self) -> str:
        """Human-readable string representation"""
        output = []
        output.append("=" * 80)
        output.append("INTELLIGENCE REPORT (System 4)")
        output.append("=" * 80)
        output.append(f"\nSession: {self.session_id}")
        output.append(f"\nGenerated: {self.metadata.get('timestamp', 'N/A')}")
        output.append(f"\n{'-' * 80}")
        output.append("\nPERFORMANCE ANALYSIS:")
        output.append(self.analysis)
        output.append(f"\n{'-' * 80}")
        output.append("\nDETECTED PATTERNS:")
        output.append(self.patterns)
        output.append(f"\n{'-' * 80}")
        output.append("\nINSIGHTS:")
        output.append(self.insights)
        output.append(f"\n{'-' * 80}")
        output.append("\nACTION PLAN:")
        output.append(self.action_plan)
        output.append(f"\n{'-' * 80}")
        output.append(f"\nMetadata: {self.metadata}")
        output.append("=" * 80)
        return "\n".join(output)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "session_id": self.session_id,
            "analysis": self.analysis,
            "patterns": self.patterns,
            "insights": self.insights,
            "action_plan": self.action_plan,
            "metadata": self.metadata
        }


# ============================================================================
# Intelligence Agent
# ============================================================================

class IntelligenceAgent(dspy.Module):
    """
    System 4 (Intelligence) - Performance reflection and learning agent.

    Analyzes session logs and performance metrics to identify improvement
    opportunities. Generates prioritized action plans for optimization.

    Uses expensive tier models for deep analysis and quality insights.

    Usage:
        >>> agent = IntelligenceAgent()
        >>> result = agent(
        ...     session_logs=json.dumps(logs),
        ...     performance_metrics={
        ...         "accuracy": 0.85,
        ...         "cost": 12.50,
        ...         "latency": 2.3,
        ...         "cache_hit_rate": 0.75
        ...     }
        ... )
        >>> print(result)

        >>> # With custom config
        >>> config = IntelligenceConfig(
        ...     analysis_tier="premium",
        ...     max_recommendations=10
        ... )
        >>> agent = IntelligenceAgent(config=config)
    """

    def __init__(
        self,
        config: Optional[IntelligenceConfig] = None
    ):
        """
        Initialize IntelligenceAgent.

        Args:
            config: IntelligenceConfig with tier selection and parameters
        """
        super().__init__()  # Important for dspy.Module

        # Use provided config or create default
        self.config = config if config is not None else IntelligenceConfig()

        # Create LM instances for each stage
        lm_kwargs = {}
        if self.config.max_tokens is not None:
            lm_kwargs["max_tokens"] = self.config.max_tokens
        if self.config.temperature is not None:
            lm_kwargs["temperature"] = self.config.temperature

        self.analysis_lm = FractalDSpyLM(tier=self.config.analysis_tier, **lm_kwargs)
        self.pattern_lm = FractalDSpyLM(tier=self.config.pattern_tier, **lm_kwargs)
        self.insight_lm = FractalDSpyLM(tier=self.config.insight_tier, **lm_kwargs)
        self.prioritization_lm = FractalDSpyLM(tier=self.config.prioritization_tier, **lm_kwargs)

        # Configure default LM (analysis LM) so predictors can be created
        dspy.configure(lm=self.analysis_lm)

        # Create DSPy modules for each stage
        self.analyzer = dspy.ChainOfThought(PerformanceAnalysis)
        self.pattern_detector = dspy.ChainOfThought(PatternDetection)
        self.insight_generator = dspy.ChainOfThought(InsightGeneration)
        self.prioritizer = dspy.ChainOfThought(RecommendationPrioritization)

        logger.info(
            f"Initialized IntelligenceAgent with tiers: "
            f"analysis={self.config.analysis_tier}, pattern={self.config.pattern_tier}, "
            f"insight={self.config.insight_tier}, prioritization={self.config.prioritization_tier}"
        )

    def forward(
        self,
        session_logs: str,
        performance_metrics: Dict[str, float],
        session_id: str = None,
        verbose: bool = True
    ) -> IntelligenceResult:
        """
        Perform intelligence reflection on session data.

        This is the DSPy Module forward() method.
        Call the agent instance directly: agent(session_logs=..., performance_metrics=...)

        Args:
            session_logs: JSON-formatted session logs
            performance_metrics: Dict with accuracy, cost, latency, cache_hit_rate, etc.
            session_id: Optional session identifier
            verbose: Whether to print progress updates

        Returns:
            IntelligenceResult with analysis, patterns, insights, and action plan
        """
        if verbose:
            print(f"\n{'=' * 80}")
            print(f"Intelligence Agent (System 4) - Performance Reflection")
            print(f"{'=' * 80}\n")

        # Generate session ID if not provided
        if session_id is None:
            session_id = f"intelligence_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Format performance metrics for analysis
        metrics_str = json.dumps(performance_metrics, indent=2)

        # Stage 1: Performance Analysis
        if verbose:
            print(f"Stage 1: Performance Analysis... (tier={self.config.analysis_tier})")

        dspy.configure(lm=self.analysis_lm)
        analysis_result = self.analyzer(
            session_logs=session_logs,
            performance_metrics=metrics_str
        )
        analysis = analysis_result.analysis

        if verbose:
            print(f"✓ Analysis complete")
            print(f"  Preview: {analysis[:150]}...\n")

        # Stage 2: Pattern Detection
        if verbose:
            print(f"Stage 2: Pattern Detection... (tier={self.config.pattern_tier})")

        dspy.configure(lm=self.pattern_lm)
        pattern_result = self.pattern_detector(
            performance_analysis=analysis,
            session_logs=session_logs
        )
        patterns = pattern_result.patterns

        if verbose:
            print(f"✓ Patterns detected")
            print(f"  Preview: {patterns[:150]}...\n")

        # Stage 3: Insight Generation
        if verbose:
            print(f"Stage 3: Insight Generation... (tier={self.config.insight_tier})")

        dspy.configure(lm=self.insight_lm)
        insight_result = self.insight_generator(
            patterns=patterns,
            performance_metrics=metrics_str
        )
        insights = insight_result.insights

        if verbose:
            print(f"✓ Insights generated")
            print(f"  Preview: {insights[:150]}...\n")

        # Stage 4: Prioritization
        if verbose:
            print(f"Stage 4: Prioritization... (tier={self.config.prioritization_tier})")

        dspy.configure(lm=self.prioritization_lm)
        prioritization_result = self.prioritizer(
            insights=insights,
            performance_metrics=metrics_str
        )
        action_plan = prioritization_result.action_plan

        if verbose:
            print(f"✓ Action plan created")
            print(f"  Preview: {action_plan[:150]}...\n")

        # Calculate metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "config": str(self.config),
            "tiers": {
                "analysis": self.config.analysis_tier,
                "pattern": self.config.pattern_tier,
                "insight": self.config.insight_tier,
                "prioritization": self.config.prioritization_tier
            },
            "input_metrics": performance_metrics,
            "lookback_days": self.config.lookback_days,
            "min_session_size": self.config.min_session_size
        }

        # Create result
        result = IntelligenceResult(
            session_id=session_id,
            analysis=analysis,
            patterns=patterns,
            insights=insights,
            action_plan=action_plan,
            metadata=metadata
        )

        if verbose:
            print(f"{'=' * 80}")
            print(f"Intelligence reflection complete!")
            print(f"{'=' * 80}\n")

        return result

    def should_trigger_analysis(
        self,
        performance_metrics: Dict[str, float],
        session_size: int,
        last_analysis_days_ago: int = None
    ) -> tuple[bool, str]:
        """
        Determine if intelligence analysis should be triggered.

        Args:
            performance_metrics: Current performance metrics
            session_size: Number of tasks in session
            last_analysis_days_ago: Days since last analysis

        Returns:
            (should_trigger, reason) tuple
        """
        # Check session size
        if session_size < self.config.min_session_size:
            return False, f"Session too small ({session_size} < {self.config.min_session_size})"

        # Check for failure trigger
        if self.config.analyze_on_failure:
            accuracy = performance_metrics.get("accuracy", 1.0)
            if accuracy < 0.5:  # More than 50% failure rate
                return True, "High failure rate detected"

        # Check for cost spike trigger
        if self.config.analyze_on_cost_spike:
            current_cost = performance_metrics.get("cost", 0)
            avg_cost = performance_metrics.get("avg_cost", current_cost)
            if current_cost > avg_cost * self.config.cost_spike_threshold:
                return True, f"Cost spike detected ({current_cost:.2f} > {avg_cost * self.config.cost_spike_threshold:.2f})"

        # Check for scheduled trigger
        if self.config.analyze_on_schedule and last_analysis_days_ago is not None:
            if last_analysis_days_ago >= self.config.lookback_days:
                return True, f"Scheduled analysis due ({last_analysis_days_ago} days since last)"

        return False, "No trigger conditions met"


# ============================================================================
# Main Demo
# ============================================================================

if __name__ == "__main__":
    # Demo: Run intelligence agent with sample data
    from .intelligence_config import PresetIntelligenceConfigs

    print("=" * 80)
    print("Intelligence Agent Demo")
    print("=" * 80)
    print()

    # Sample session logs (simplified)
    sample_logs = json.dumps({
        "session_id": "session_20251019_001",
        "tasks": [
            {
                "task_id": "task_001",
                "agent_type": "research",
                "status": "completed",
                "duration_seconds": 15.3,
                "tokens_used": 2500,
                "cost": 0.05
            },
            {
                "task_id": "task_002",
                "agent_type": "research",
                "status": "failed",
                "duration_seconds": 8.1,
                "tokens_used": 1200,
                "cost": 0.02,
                "error": "Context limit exceeded"
            },
            {
                "task_id": "task_003",
                "agent_type": "control",
                "status": "completed",
                "duration_seconds": 25.7,
                "tokens_used": 4200,
                "cost": 0.12
            }
        ]
    }, indent=2)

    # Sample performance metrics
    sample_metrics = {
        "accuracy": 0.67,  # 2/3 success rate
        "cost": 0.19,
        "latency": 16.4,  # avg seconds
        "cache_hit_rate": 0.45,
        "failed_tasks": 1,
        "avg_cost": 0.063  # per task
    }

    # Create agent with quick analysis config
    config = PresetIntelligenceConfigs.quick_analysis()
    agent = IntelligenceAgent(config=config)

    # Check if analysis should trigger
    should_trigger, reason = agent.should_trigger_analysis(
        performance_metrics=sample_metrics,
        session_size=3,
        last_analysis_days_ago=8
    )

    print(f"Should trigger analysis: {should_trigger}")
    print(f"Reason: {reason}")
    print()

    # Run intelligence reflection
    print("Running intelligence reflection...")
    print("-" * 80)

    result = agent(
        session_logs=sample_logs,
        performance_metrics=sample_metrics,
        verbose=True
    )

    print("\n" + str(result))
