"""
Context Improvement Agent - Self-Improving Context Preparation

This agent analyzes failures in context preparation and generates
improvement strategies for future attempts. It creates a feedback loop
that enables continuous learning and adaptation.

Capabilities:
1. Analyzes why context preparation failed (low precision or completeness)
2. Identifies patterns across multiple failures
3. Generates actionable improvement strategies
4. Updates context preparation parameters
5. Learns which sources work best for which tasks

This is VSM System 3 (Optimization) for the context preparation system.

Author: BMad
Date: 2025-10-22
"""

import logging
import dspy
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import json
from pathlib import Path
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)


# ============================================================================
# DSPy Signatures
# ============================================================================

class FailureAnalysis(dspy.Signature):
    """
    Analyze why context preparation failed.

    Given a failed attempt, identify the root causes and what was missing.
    """
    user_task = dspy.InputField(desc="What the user asked for")
    agent_type = dspy.InputField(desc="Type of agent (research, developer, coordination)")

    # Context that was prepared
    sources_used = dspy.InputField(desc="Which sources were queried")
    domain_knowledge = dspy.InputField(desc="Domain knowledge provided")
    recent_examples = dspy.InputField(desc="Recent examples provided")
    constraints = dspy.InputField(desc="Constraints provided")

    # Validation results
    attribution_precision = dspy.InputField(desc="How much context was actually used (0.0-1.0)")
    completeness_score = dspy.InputField(desc="How complete the context was (0.0-1.0)")

    # What went wrong
    root_causes = dspy.OutputField(desc="List of 2-5 specific root causes for failure")
    missing_information = dspy.OutputField(desc="What information was missing or inadequate")
    unused_context = dspy.OutputField(desc="Why provided context wasn't used")


class ImprovementStrategy(dspy.Signature):
    """
    Generate improvement strategy based on failure analysis.

    Given failure patterns, recommend specific changes to improve
    future context preparation.
    """
    failure_patterns = dspy.InputField(desc="Common patterns across failures")
    task_type = dspy.InputField(desc="Type of task (research, development, etc)")
    current_sources = dspy.InputField(desc="Current sources being used")

    # Generate improvements
    recommended_sources = dspy.OutputField(desc="Which sources to prioritize for this task type")
    query_improvements = dspy.OutputField(desc="How to improve queries to sources")
    iteration_strategy = dspy.OutputField(desc="When to iterate vs when context is sufficient")
    confidence_threshold = dspy.OutputField(desc="Recommended confidence threshold for this task type")


class PatternIdentification(dspy.Signature):
    """
    Identify patterns across multiple failures.

    Find common themes that indicate systematic issues.
    """
    failures = dspy.InputField(desc="JSON array of failure summaries")

    # Identify patterns
    common_themes = dspy.OutputField(desc="3-5 common themes across failures")
    task_type_patterns = dspy.OutputField(desc="Patterns specific to research vs developer vs coordination")
    source_issues = dspy.OutputField(desc="Which sources consistently underperform")


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class ImprovementRecommendation:
    """
    Specific recommendation for improving context preparation.
    """
    recommendation_id: str
    timestamp: str

    # What to improve
    area: str  # "source_selection", "query_strategy", "iteration_logic", "confidence_threshold"
    description: str
    rationale: str

    # How to apply
    parameters: Dict[str, Any]

    # Tracking
    applied: bool = False
    impact_precision: Optional[float] = None
    impact_completeness: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "recommendation_id": self.recommendation_id,
            "timestamp": self.timestamp,
            "area": self.area,
            "description": self.description,
            "rationale": self.rationale,
            "parameters": self.parameters,
            "applied": self.applied,
            "impact_precision": self.impact_precision,
            "impact_completeness": self.impact_completeness
        }


# ============================================================================
# Context Improvement Agent
# ============================================================================

class ContextImprovementAgent:
    """
    Self-improving agent that learns from context preparation failures.

    This agent analyzes failures, identifies patterns, and generates
    improvement strategies that can be applied to future context preparation.

    Usage:
        >>> from fractal_agent.validation.learning_tracker import LearningTracker
        >>>
        >>> tracker = LearningTracker()
        >>> improvement_agent = ContextImprovementAgent()
        >>>
        >>> # Analyze recent failures
        >>> failures = tracker.get_failures(min_precision=0.6, min_completeness=0.7)
        >>> recommendations = improvement_agent.analyze_and_improve(failures)
        >>>
        >>> # Apply recommendations to context preparation
        >>> for rec in recommendations:
        ...     if rec.area == "source_selection":
        ...         context_prep_agent.update_source_weights(rec.parameters)
    """

    def __init__(
        self,
        recommendations_dir: str = "logs/improvements",
        min_failures_for_pattern: int = 3
    ):
        """
        Initialize context improvement agent.

        Args:
            recommendations_dir: Where to store improvement recommendations
            min_failures_for_pattern: Minimum failures needed to identify patterns
        """
        self.recommendations_dir = Path(recommendations_dir)
        self.recommendations_dir.mkdir(parents=True, exist_ok=True)

        self.min_failures_for_pattern = min_failures_for_pattern
        self.recommendations_file = self.recommendations_dir / "recommendations.jsonl"

        # DSPy modules
        self.failure_analyzer = dspy.ChainOfThought(FailureAnalysis)
        self.pattern_identifier = dspy.ChainOfThought(PatternIdentification)
        self.strategy_generator = dspy.ChainOfThought(ImprovementStrategy)

        logger.info(f"Initialized ContextImprovementAgent: {self.recommendations_dir}")

    def analyze_failure(
        self,
        failure_attempt,  # ContextPreparationAttempt from LearningTracker
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze a single failure to understand what went wrong.

        Args:
            failure_attempt: ContextPreparationAttempt that failed validation
            verbose: Print detailed analysis

        Returns:
            Dict with root_causes, missing_information, unused_context
        """
        if verbose:
            logger.info(f"Analyzing failure: {failure_attempt.user_task[:50]}...")

        # Run failure analysis
        analysis = self.failure_analyzer(
            user_task=failure_attempt.user_task,
            agent_type=failure_attempt.agent_type,
            sources_used=", ".join(failure_attempt.sources_used),
            domain_knowledge="Provided" if failure_attempt.confidence > 0 else "None",
            recent_examples="Provided" if failure_attempt.iterations > 0 else "None",
            constraints="Provided",
            attribution_precision=str(failure_attempt.attribution_precision),
            completeness_score=str(failure_attempt.completeness_score)
        )

        result = {
            "root_causes": analysis.root_causes,
            "missing_information": analysis.missing_information,
            "unused_context": analysis.unused_context
        }

        if verbose:
            logger.info(f"Root causes: {result['root_causes']}")

        return result

    def identify_patterns(
        self,
        failures: List,  # List[ContextPreparationAttempt]
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Identify patterns across multiple failures.

        Args:
            failures: List of ContextPreparationAttempt that failed
            verbose: Print detailed analysis

        Returns:
            Dict with common_themes, task_type_patterns, source_issues
        """
        if len(failures) < self.min_failures_for_pattern:
            logger.warning(
                f"Only {len(failures)} failures - need at least {self.min_failures_for_pattern} to identify patterns"
            )
            return {
                "common_themes": "Insufficient data",
                "task_type_patterns": "Insufficient data",
                "source_issues": "Insufficient data"
            }

        if verbose:
            logger.info(f"Identifying patterns across {len(failures)} failures...")

        # Summarize failures for LLM
        failure_summaries = []
        for f in failures:
            failure_summaries.append({
                "task": f.user_task[:100],
                "agent_type": f.agent_type,
                "precision": f.attribution_precision,
                "completeness": f.completeness_score,
                "sources": f.sources_used,
                "iterations": f.iterations
            })

        # Run pattern identification
        patterns = self.pattern_identifier(
            failures=json.dumps(failure_summaries, indent=2)
        )

        result = {
            "common_themes": patterns.common_themes,
            "task_type_patterns": patterns.task_type_patterns,
            "source_issues": patterns.source_issues
        }

        if verbose:
            logger.info(f"Common themes: {result['common_themes']}")

        return result

    def generate_improvements(
        self,
        patterns: Dict[str, Any],
        task_type: str,
        current_sources: List[str],
        verbose: bool = False
    ) -> List[ImprovementRecommendation]:
        """
        Generate improvement recommendations based on patterns.

        Args:
            patterns: Patterns identified from failures
            task_type: Type of task to optimize for
            current_sources: Currently used sources
            verbose: Print detailed recommendations

        Returns:
            List of ImprovementRecommendation
        """
        if verbose:
            logger.info(f"Generating improvements for {task_type} tasks...")

        # Run strategy generation
        strategy = self.strategy_generator(
            failure_patterns=json.dumps(patterns, indent=2),
            task_type=task_type,
            current_sources=", ".join(current_sources)
        )

        recommendations = []

        # Create recommendation for source selection
        recommendations.append(ImprovementRecommendation(
            recommendation_id=self._generate_rec_id(),
            timestamp=self._get_timestamp(),
            area="source_selection",
            description=f"Prioritize sources for {task_type} tasks",
            rationale=strategy.recommended_sources,
            parameters={
                "recommended_sources": self._parse_sources(strategy.recommended_sources),
                "task_type": task_type
            }
        ))

        # Create recommendation for query improvements
        recommendations.append(ImprovementRecommendation(
            recommendation_id=self._generate_rec_id(),
            timestamp=self._get_timestamp(),
            area="query_strategy",
            description=f"Improve query formulation for {task_type}",
            rationale=strategy.query_improvements,
            parameters={
                "query_improvements": strategy.query_improvements,
                "task_type": task_type
            }
        ))

        # Create recommendation for iteration strategy
        recommendations.append(ImprovementRecommendation(
            recommendation_id=self._generate_rec_id(),
            timestamp=self._get_timestamp(),
            area="iteration_logic",
            description=f"Adjust iteration strategy for {task_type}",
            rationale=strategy.iteration_strategy,
            parameters={
                "iteration_strategy": strategy.iteration_strategy,
                "task_type": task_type
            }
        ))

        # Create recommendation for confidence threshold
        try:
            confidence_value = float(strategy.confidence_threshold.split()[0]) if strategy.confidence_threshold else 0.8
        except:
            confidence_value = 0.8

        recommendations.append(ImprovementRecommendation(
            recommendation_id=self._generate_rec_id(),
            timestamp=self._get_timestamp(),
            area="confidence_threshold",
            description=f"Adjust confidence threshold for {task_type}",
            rationale=strategy.confidence_threshold,
            parameters={
                "min_confidence": confidence_value,
                "task_type": task_type
            }
        ))

        # Save recommendations
        for rec in recommendations:
            self._save_recommendation(rec)

        if verbose:
            logger.info(f"Generated {len(recommendations)} improvement recommendations")

        return recommendations

    def analyze_and_improve(
        self,
        failures: List,  # List[ContextPreparationAttempt]
        verbose: bool = False
    ) -> List[ImprovementRecommendation]:
        """
        Complete workflow: analyze failures and generate improvements.

        Args:
            failures: List of ContextPreparationAttempt that failed validation
            verbose: Print detailed analysis

        Returns:
            List of ImprovementRecommendation
        """
        if not failures:
            logger.warning("No failures to analyze")
            return []

        # Step 1: Analyze individual failures
        if verbose:
            print("\n" + "=" * 80)
            print("FAILURE ANALYSIS")
            print("=" * 80)
            print()

        failure_analyses = []
        for i, failure in enumerate(failures[:10], 1):  # Analyze top 10
            if verbose:
                print(f"\nFailure {i}: {failure.user_task[:60]}...")
                print(f"  Precision: {failure.attribution_precision:.2f}")
                print(f"  Completeness: {failure.completeness_score:.2f}")

            analysis = self.analyze_failure(failure, verbose=False)
            failure_analyses.append(analysis)

            if verbose:
                print(f"  Root causes: {analysis['root_causes'][:100]}...")

        # Step 2: Identify patterns
        if verbose:
            print("\n" + "=" * 80)
            print("PATTERN IDENTIFICATION")
            print("=" * 80)
            print()

        patterns = self.identify_patterns(failures, verbose=verbose)

        # Step 3: Generate improvements by task type
        if verbose:
            print("\n" + "=" * 80)
            print("IMPROVEMENT RECOMMENDATIONS")
            print("=" * 80)
            print()

        all_recommendations = []

        # Group by task type
        by_task_type = defaultdict(list)
        for failure in failures:
            by_task_type[failure.agent_type].append(failure)

        for task_type, task_failures in by_task_type.items():
            if verbose:
                print(f"\n{task_type.upper()} Tasks ({len(task_failures)} failures):")
                print("-" * 80)

            # Get common sources
            all_sources = set()
            for f in task_failures:
                all_sources.update(f.sources_used)

            # Generate recommendations
            recommendations = self.generate_improvements(
                patterns=patterns,
                task_type=task_type,
                current_sources=list(all_sources),
                verbose=False
            )

            all_recommendations.extend(recommendations)

            if verbose:
                for rec in recommendations:
                    print(f"  • {rec.description}")
                    print(f"    Rationale: {rec.rationale[:80]}...")
                    print()

        if verbose:
            print("=" * 80)
            print(f"Generated {len(all_recommendations)} total recommendations")
            print("=" * 80)
            print()

        return all_recommendations

    def get_recommendations(
        self,
        area: Optional[str] = None,
        task_type: Optional[str] = None,
        unapplied_only: bool = False
    ) -> List[ImprovementRecommendation]:
        """
        Load recommendations with optional filtering.

        Args:
            area: Filter by area (source_selection, query_strategy, etc)
            task_type: Filter by task type
            unapplied_only: Only return recommendations that haven't been applied

        Returns:
            List of ImprovementRecommendation
        """
        if not self.recommendations_file.exists():
            return []

        recommendations = []
        with open(self.recommendations_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    rec = ImprovementRecommendation(**data)

                    # Apply filters
                    if area and rec.area != area:
                        continue
                    if task_type and rec.parameters.get('task_type') != task_type:
                        continue
                    if unapplied_only and rec.applied:
                        continue

                    recommendations.append(rec)
                except Exception as e:
                    logger.warning(f"Failed to parse recommendation: {e}")

        return recommendations

    def _generate_rec_id(self) -> str:
        """Generate unique recommendation ID"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"rec_{timestamp}"

    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()

    def _parse_sources(self, sources_text: str) -> List[str]:
        """Parse source names from LLM output"""
        sources = []
        for word in sources_text.lower().split():
            if "graphrag" in word:
                sources.append("graphrag")
            elif "memory" in word or "task" in word:
                sources.append("recent_tasks")
            elif "obsidian" in word or "note" in word:
                sources.append("obsidian")
            elif "web" in word or "search" in word:
                sources.append("web")
        return list(set(sources)) or ["graphrag", "recent_tasks", "obsidian"]

    def _save_recommendation(self, rec: ImprovementRecommendation):
        """Save recommendation to JSONL file"""
        with open(self.recommendations_file, 'a') as f:
            f.write(json.dumps(rec.to_dict()) + '\n')


# ============================================================================
# Module Testing
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Context Improvement Agent Test")
    print("=" * 80)
    print()

    # Create mock failures
    from fractal_agent.validation.learning_tracker import ContextPreparationAttempt

    failures = [
        ContextPreparationAttempt(
            attempt_id="test_1",
            timestamp="2025-10-22T12:00:00",
            user_task="Research VSM System 2 coordination",
            agent_type="research",
            confidence=0.6,
            sources_used=["graphrag", "recent_tasks"],
            total_tokens=150,
            preparation_time=1.2,
            iterations=1,
            attribution_precision=0.45,
            attribution_used=2,
            attribution_total=5,
            completeness_score=0.65,
            completeness_is_complete=False,
            source_precision={"graphrag": 0.5, "recent_tasks": 0.4}
        ),
        ContextPreparationAttempt(
            attempt_id="test_2",
            timestamp="2025-10-22T12:05:00",
            user_task="Implement Python coordination class",
            agent_type="developer",
            confidence=0.7,
            sources_used=["recent_tasks", "obsidian"],
            total_tokens=180,
            preparation_time=1.5,
            iterations=2,
            attribution_precision=0.50,
            attribution_used=3,
            attribution_total=6,
            completeness_score=0.68,
            completeness_is_complete=False,
            source_precision={"recent_tasks": 0.5, "obsidian": 0.5}
        ),
        ContextPreparationAttempt(
            attempt_id="test_3",
            timestamp="2025-10-22T12:10:00",
            user_task="Research cybernetic control loops",
            agent_type="research",
            confidence=0.65,
            sources_used=["graphrag", "web"],
            total_tokens=200,
            preparation_time=1.8,
            iterations=2,
            attribution_precision=0.40,
            attribution_used=2,
            attribution_total=5,
            completeness_score=0.60,
            completeness_is_complete=False,
            source_precision={"graphrag": 0.4, "web": 0.4}
        )
    ]

    print(f"Mock failures created: {len(failures)}")
    print()

    # Initialize improvement agent (without LLM for quick test)
    improvement_agent = ContextImprovementAgent(
        recommendations_dir="logs/improvements_test"
    )

    print("Note: Full test requires LLM access via DSPy")
    print("Run integration tests to see complete workflow")
    print()

    print("=" * 80)
    print("Context Improvement Agent Test Complete")
    print("=" * 80)
