"""
Learning Tracker - Continuous Improvement System for Context Preparation

Tracks context preparation effectiveness over time to enable:
1. Measuring improvement trends
2. Identifying what works (and what doesn't)
3. Learning which sources are most effective
4. Understanding iteration patterns
5. Optimizing context preparation strategies

This is the feedback loop that enables self-improvement.

Author: BMad
Date: 2025-10-22
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)


@dataclass
class ContextPreparationAttempt:
    """
    Record of a single context preparation attempt with its outcome.

    This captures everything we need to learn from experience.
    """
    # Identity
    attempt_id: str
    timestamp: str

    # Input
    user_task: str
    agent_type: str

    # Context preparation
    confidence: float
    sources_used: List[str]
    total_tokens: int
    preparation_time: float
    iterations: int

    # Validation results
    attribution_precision: float  # 0.0-1.0
    attribution_used: int
    attribution_total: int
    completeness_score: float  # 0.0-1.0
    completeness_is_complete: bool

    # Breakdown by source
    source_precision: Dict[str, float] = field(default_factory=dict)

    # Agent outcome (if available)
    agent_succeeded: Optional[bool] = None
    agent_error: Optional[str] = None

    # Learning metadata
    tags: List[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContextPreparationAttempt':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class LearningMetrics:
    """
    Aggregated learning metrics over time.

    Provides trend analysis and improvement insights.
    """
    total_attempts: int

    # Precision trends (attribution)
    avg_precision: float
    precision_trend: str  # "improving", "stable", "declining"
    precision_history: List[float]

    # Completeness trends
    avg_completeness: float
    completeness_rate: float  # % of attempts that were complete
    completeness_trend: str
    completeness_history: List[float]

    # Efficiency metrics
    avg_iterations: float
    avg_prep_time: float
    avg_tokens: float

    # Source effectiveness
    source_stats: Dict[str, Dict[str, Any]]  # source -> {attempts, avg_precision, success_rate}

    # Recommendations
    insights: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


class LearningTracker:
    """
    Tracks context preparation attempts and learns from outcomes.

    Provides continuous improvement by:
    1. Recording every context preparation attempt
    2. Tracking validation results (attribution + completeness)
    3. Analyzing trends over time
    4. Identifying patterns of success/failure
    5. Recommending improvements

    Usage:
        >>> tracker = LearningTracker(log_dir="logs/learning")
        >>>
        >>> # After context preparation and validation
        >>> attempt = tracker.record_attempt(
        ...     user_task="Research VSM System 2",
        ...     agent_type="research",
        ...     context_package=context_package,
        ...     attribution_result=attribution_result,
        ...     completeness_result=completeness_result
        ... )
        >>>
        >>> # Analyze learning
        >>> metrics = tracker.get_learning_metrics(last_n=100)
        >>> print(f"Precision trend: {metrics.precision_trend}")
        >>> print(f"Top insight: {metrics.insights[0]}")
    """

    def __init__(self, log_dir: str = "logs/learning"):
        """
        Initialize learning tracker.

        Args:
            log_dir: Directory to store learning data
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.attempts_file = self.log_dir / "attempts.jsonl"
        self.metrics_file = self.log_dir / "latest_metrics.json"

        logger.info(f"Initialized LearningTracker: {self.log_dir}")

    def record_attempt(
        self,
        user_task: str,
        agent_type: str,
        context_package,  # ContextPackage
        attribution_result,  # AttributionResult
        completeness_result,  # CompletenessResult
        agent_succeeded: Optional[bool] = None,
        agent_error: Optional[str] = None,
        tags: Optional[List[str]] = None,
        notes: str = ""
    ) -> ContextPreparationAttempt:
        """
        Record a context preparation attempt with its validation results.

        Args:
            user_task: Original user task
            agent_type: Type of agent (research, developer, coordination)
            context_package: ContextPackage that was prepared
            attribution_result: Attribution validation result
            completeness_result: Completeness validation result
            agent_succeeded: Whether the agent succeeded (if known)
            agent_error: Error message if agent failed
            tags: Optional tags for categorization
            notes: Optional notes

        Returns:
            ContextPreparationAttempt record
        """
        # Create attempt record
        attempt = ContextPreparationAttempt(
            attempt_id=self._generate_attempt_id(),
            timestamp=datetime.now().isoformat(),
            user_task=user_task,
            agent_type=agent_type,
            confidence=context_package.confidence,
            sources_used=context_package.sources_used,
            total_tokens=context_package.total_tokens,
            preparation_time=context_package.preparation_time,
            iterations=context_package.iterations,
            attribution_precision=attribution_result.precision,
            attribution_used=attribution_result.used_pieces,
            attribution_total=attribution_result.total_pieces,
            completeness_score=completeness_result.completeness_score,
            completeness_is_complete=completeness_result.is_complete,
            source_precision={
                source: stats['precision']
                for source, stats in attribution_result.by_source.items()
            },
            agent_succeeded=agent_succeeded,
            agent_error=agent_error,
            tags=tags or [],
            notes=notes
        )

        # Append to log file (JSONL format)
        self._append_attempt(attempt)

        logger.info(
            f"Recorded attempt {attempt.attempt_id}: "
            f"precision={attempt.attribution_precision:.2f}, "
            f"completeness={attempt.completeness_score:.2f}"
        )

        return attempt

    def get_learning_metrics(
        self,
        last_n: Optional[int] = None,
        agent_type: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> LearningMetrics:
        """
        Get aggregated learning metrics.

        Args:
            last_n: Only analyze last N attempts (None = all)
            agent_type: Filter by agent type (None = all)
            tags: Filter by tags (None = all)

        Returns:
            LearningMetrics with trends and insights
        """
        # Load attempts
        attempts = self._load_attempts(last_n=last_n, agent_type=agent_type, tags=tags)

        if not attempts:
            logger.warning("No attempts found for analysis")
            return LearningMetrics(
                total_attempts=0,
                avg_precision=0.0,
                precision_trend="no_data",
                precision_history=[],
                avg_completeness=0.0,
                completeness_rate=0.0,
                completeness_trend="no_data",
                completeness_history=[],
                avg_iterations=0.0,
                avg_prep_time=0.0,
                avg_tokens=0.0,
                source_stats={},
                insights=["No learning data available yet"]
            )

        # Calculate metrics
        precision_history = [a.attribution_precision for a in attempts]
        completeness_history = [a.completeness_score for a in attempts]

        avg_precision = statistics.mean(precision_history)
        avg_completeness = statistics.mean(completeness_history)
        completeness_rate = sum(1 for a in attempts if a.completeness_is_complete) / len(attempts)

        avg_iterations = statistics.mean(a.iterations for a in attempts)
        avg_prep_time = statistics.mean(a.preparation_time for a in attempts)
        avg_tokens = statistics.mean(a.total_tokens for a in attempts)

        # Analyze trends
        precision_trend = self._analyze_trend(precision_history)
        completeness_trend = self._analyze_trend(completeness_history)

        # Source effectiveness
        source_stats = self._calculate_source_stats(attempts)

        # Generate insights
        insights = self._generate_insights(
            attempts=attempts,
            avg_precision=avg_precision,
            avg_completeness=avg_completeness,
            precision_trend=precision_trend,
            completeness_trend=completeness_trend,
            source_stats=source_stats
        )

        metrics = LearningMetrics(
            total_attempts=len(attempts),
            avg_precision=avg_precision,
            precision_trend=precision_trend,
            precision_history=precision_history[-20:],  # Last 20 for visualization
            avg_completeness=avg_completeness,
            completeness_rate=completeness_rate,
            completeness_trend=completeness_trend,
            completeness_history=completeness_history[-20:],
            avg_iterations=avg_iterations,
            avg_prep_time=avg_prep_time,
            avg_tokens=avg_tokens,
            source_stats=source_stats,
            insights=insights
        )

        # Save latest metrics
        self._save_metrics(metrics)

        return metrics

    def get_failures(self, min_precision: float = 0.5, min_completeness: float = 0.7) -> List[ContextPreparationAttempt]:
        """
        Get attempts that failed validation thresholds.

        Args:
            min_precision: Minimum acceptable precision
            min_completeness: Minimum acceptable completeness

        Returns:
            List of failed attempts
        """
        attempts = self._load_attempts()

        failures = [
            a for a in attempts
            if a.attribution_precision < min_precision or a.completeness_score < min_completeness
        ]

        return failures

    def get_successes(self, min_precision: float = 0.7, min_completeness: float = 0.8) -> List[ContextPreparationAttempt]:
        """
        Get highly successful attempts to learn from.

        Args:
            min_precision: Minimum precision for success
            min_completeness: Minimum completeness for success

        Returns:
            List of successful attempts
        """
        attempts = self._load_attempts()

        successes = [
            a for a in attempts
            if a.attribution_precision >= min_precision and a.completeness_score >= min_completeness
        ]

        return successes

    def _generate_attempt_id(self) -> str:
        """Generate unique attempt ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"attempt_{timestamp}"

    def _append_attempt(self, attempt: ContextPreparationAttempt):
        """Append attempt to JSONL log file"""
        with open(self.attempts_file, 'a') as f:
            f.write(json.dumps(attempt.to_dict()) + '\n')

    def _load_attempts(
        self,
        last_n: Optional[int] = None,
        agent_type: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[ContextPreparationAttempt]:
        """Load attempts from JSONL file with optional filtering"""
        if not self.attempts_file.exists():
            return []

        attempts = []
        with open(self.attempts_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    attempt = ContextPreparationAttempt.from_dict(data)

                    # Apply filters
                    if agent_type and attempt.agent_type != agent_type:
                        continue
                    if tags and not any(tag in attempt.tags for tag in tags):
                        continue

                    attempts.append(attempt)
                except Exception as e:
                    logger.warning(f"Failed to parse attempt line: {e}")

        # Apply last_n filter
        if last_n:
            attempts = attempts[-last_n:]

        return attempts

    def _analyze_trend(self, history: List[float], window: int = 10) -> str:
        """
        Analyze trend in metric history.

        Args:
            history: List of values over time
            window: Window size for trend analysis

        Returns:
            "improving", "stable", or "declining"
        """
        if len(history) < window:
            return "insufficient_data"

        # Compare recent window to earlier window
        recent = history[-window:]
        earlier = history[-2*window:-window] if len(history) >= 2*window else history[:-window]

        if not earlier:
            return "insufficient_data"

        recent_avg = statistics.mean(recent)
        earlier_avg = statistics.mean(earlier)

        # Calculate relative change
        change = (recent_avg - earlier_avg) / earlier_avg if earlier_avg > 0 else 0

        if change > 0.05:  # 5% improvement
            return "improving"
        elif change < -0.05:  # 5% decline
            return "declining"
        else:
            return "stable"

    def _calculate_source_stats(self, attempts: List[ContextPreparationAttempt]) -> Dict[str, Dict[str, Any]]:
        """Calculate statistics by source"""
        source_data = defaultdict(lambda: {"attempts": 0, "precisions": []})

        for attempt in attempts:
            for source, precision in attempt.source_precision.items():
                source_data[source]["attempts"] += 1
                source_data[source]["precisions"].append(precision)

        # Calculate averages
        source_stats = {}
        for source, data in source_data.items():
            source_stats[source] = {
                "attempts": data["attempts"],
                "avg_precision": statistics.mean(data["precisions"]) if data["precisions"] else 0.0,
                "success_rate": sum(1 for p in data["precisions"] if p >= 0.7) / len(data["precisions"]) if data["precisions"] else 0.0
            }

        return source_stats

    def _generate_insights(
        self,
        attempts: List[ContextPreparationAttempt],
        avg_precision: float,
        avg_completeness: float,
        precision_trend: str,
        completeness_trend: str,
        source_stats: Dict[str, Dict[str, Any]]
    ) -> List[str]:
        """Generate actionable insights from learning data"""
        insights = []

        # Overall performance
        if avg_precision >= 0.8 and avg_completeness >= 0.9:
            insights.append("✅ Excellent context preparation performance overall")
        elif avg_precision >= 0.6 and avg_completeness >= 0.8:
            insights.append("✓ Good context preparation performance")
        else:
            insights.append("⚠️  Context preparation needs improvement")

        # Trends
        if precision_trend == "improving":
            insights.append("📈 Attribution precision is improving over time")
        elif precision_trend == "declining":
            insights.append("📉 Attribution precision is declining - investigate recent changes")

        if completeness_trend == "improving":
            insights.append("📈 Context completeness is improving over time")
        elif completeness_trend == "declining":
            insights.append("📉 Context completeness is declining - may need more iterations")

        # Source effectiveness
        if source_stats:
            best_source = max(source_stats.items(), key=lambda x: x[1]["avg_precision"])
            worst_source = min(source_stats.items(), key=lambda x: x[1]["avg_precision"])

            if best_source[1]["avg_precision"] >= 0.7:
                insights.append(f"⭐ Best source: {best_source[0]} ({best_source[1]['avg_precision']*100:.0f}% avg precision)")

            if worst_source[1]["avg_precision"] < 0.5:
                insights.append(f"⚠️  Weakest source: {worst_source[0]} ({worst_source[1]['avg_precision']*100:.0f}% avg precision) - consider improving retrieval")

        # Iteration patterns
        recent_iterations = [a.iterations for a in attempts[-20:]]
        if recent_iterations:
            avg_recent_iterations = statistics.mean(recent_iterations)
            if avg_recent_iterations > 2:
                insights.append(f"⚠️  High iteration count ({avg_recent_iterations:.1f} avg) - initial retrieval may be insufficient")
            elif avg_recent_iterations <= 1:
                insights.append("✅ Low iteration count - efficient context preparation")

        # Specific recommendations
        failures = [a for a in attempts if not a.completeness_is_complete]
        if len(failures) > len(attempts) * 0.3:  # >30% failure rate
            insights.append(f"🔧 High incompleteness rate ({len(failures)/len(attempts)*100:.0f}%) - consider raising max_iterations or improving sources")

        return insights[:10]  # Top 10 insights

    def _save_metrics(self, metrics: LearningMetrics):
        """Save latest metrics to file"""
        with open(self.metrics_file, 'w') as f:
            json.dump(metrics.to_dict(), f, indent=2)

    def export_report(self, output_file: str, format: str = "markdown") -> str:
        """
        Export learning report.

        Args:
            output_file: Path to output file
            format: "markdown" or "json"

        Returns:
            Path to exported file
        """
        metrics = self.get_learning_metrics()

        if format == "json":
            with open(output_file, 'w') as f:
                json.dump(metrics.to_dict(), f, indent=2)

        elif format == "markdown":
            report = self._generate_markdown_report(metrics)
            with open(output_file, 'w') as f:
                f.write(report)

        logger.info(f"Exported learning report to {output_file}")
        return output_file

    def _generate_markdown_report(self, metrics: LearningMetrics) -> str:
        """Generate markdown learning report"""
        report = f"""# Context Preparation Learning Report

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Summary

- **Total Attempts**: {metrics.total_attempts}
- **Average Precision**: {metrics.avg_precision*100:.1f}% ({metrics.precision_trend})
- **Average Completeness**: {metrics.avg_completeness*100:.1f}% ({metrics.completeness_trend})
- **Completeness Rate**: {metrics.completeness_rate*100:.1f}% of attempts
- **Average Iterations**: {metrics.avg_iterations:.2f}
- **Average Prep Time**: {metrics.avg_prep_time:.2f}s
- **Average Tokens**: {metrics.avg_tokens:.0f}

## Key Insights

"""
        for insight in metrics.insights:
            report += f"- {insight}\n"

        report += "\n## Source Effectiveness\n\n"
        for source, stats in sorted(metrics.source_stats.items(), key=lambda x: x[1]['avg_precision'], reverse=True):
            report += f"- **{source}**: {stats['avg_precision']*100:.1f}% precision, {stats['success_rate']*100:.1f}% success rate ({stats['attempts']} attempts)\n"

        report += "\n## Trends\n\n"
        report += f"- **Precision**: {metrics.precision_trend}\n"
        report += f"- **Completeness**: {metrics.completeness_trend}\n"

        return report


# ============================================================================
# Helper Functions
# ============================================================================

def quick_track(
    user_task: str,
    agent_type: str,
    context_package,
    attribution_result,
    completeness_result,
    log_dir: str = "logs/learning"
) -> ContextPreparationAttempt:
    """
    Quick tracking with default settings.

    Args:
        user_task: Original user task
        agent_type: Type of agent
        context_package: ContextPackage that was prepared
        attribution_result: Attribution validation result
        completeness_result: Completeness validation result
        log_dir: Directory for learning logs

    Returns:
        ContextPreparationAttempt record
    """
    tracker = LearningTracker(log_dir=log_dir)
    return tracker.record_attempt(
        user_task=user_task,
        agent_type=agent_type,
        context_package=context_package,
        attribution_result=attribution_result,
        completeness_result=completeness_result
    )


# ============================================================================
# Module Testing
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Learning Tracker Test")
    print("=" * 80)
    print()

    # Create mock data
    from fractal_agent.memory.context_package import ContextPackage
    from fractal_agent.validation.context_attribution import AttributionResult
    from fractal_agent.validation.context_completeness import CompletenessResult

    # Initialize tracker
    tracker = LearningTracker(log_dir="logs/learning_test")

    # Simulate 10 attempts with improving performance
    print("Simulating 10 context preparation attempts...")
    for i in range(10):
        # Create mock context package
        context_package = ContextPackage(
            confidence=0.7 + (i * 0.02),  # Improving confidence
            sources_used=["graphrag", "recent_tasks"],
            total_tokens=200 + (i * 10),
            preparation_time=1.0 - (i * 0.05),  # Getting faster
            iterations=2 - (i // 5)  # Fewer iterations over time
        )

        # Create mock attribution result
        attribution_result = AttributionResult(
            total_pieces=5,
            used_pieces=3 + (i // 3),  # Improving usage
            unused_pieces=2 - (i // 5),
            precision=0.6 + (i * 0.03),  # Improving precision
            by_source={
                "graphrag": {"total": 3, "used": 2 + (i // 4), "precision": 0.6 + (i * 0.03)},
                "recent_tasks": {"total": 2, "used": 1 + (i // 5), "precision": 0.5 + (i * 0.04)}
            }
        )

        # Create mock completeness result
        completeness_result = CompletenessResult(
            completeness_score=0.7 + (i * 0.02),  # Improving completeness
            is_complete=(i >= 5)  # Complete after attempt 5
        )

        # Record attempt
        tracker.record_attempt(
            user_task=f"Test task {i+1}",
            agent_type="research",
            context_package=context_package,
            attribution_result=attribution_result,
            completeness_result=completeness_result,
            tags=["test"]
        )

    print(f"✓ Recorded 10 attempts")
    print()

    # Analyze metrics
    print("Analyzing learning metrics...")
    metrics = tracker.get_learning_metrics()

    print()
    print(f"Total Attempts: {metrics.total_attempts}")
    print(f"Average Precision: {metrics.avg_precision*100:.1f}%")
    print(f"Precision Trend: {metrics.precision_trend}")
    print(f"Average Completeness: {metrics.avg_completeness*100:.1f}%")
    print(f"Completeness Trend: {metrics.completeness_trend}")
    print(f"Completeness Rate: {metrics.completeness_rate*100:.1f}%")
    print()

    print("Key Insights:")
    for insight in metrics.insights[:5]:
        print(f"  • {insight}")
    print()

    # Export report
    report_file = "logs/learning_test/learning_report.md"
    tracker.export_report(report_file, format="markdown")
    print(f"✓ Exported report to {report_file}")
    print()

    print("=" * 80)
    print("Learning Tracker Test Complete")
    print("=" * 80)
