"""
Integration Test: Learning Tracker with Context Preparation Workflow

Tests that the learning tracker correctly records and analyzes
context preparation attempts over time.

Author: BMad
Date: 2025-10-22
"""

import logging
import sys
from pathlib import Path
import tempfile

# Add to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fractal_agent.memory.context_package import ContextPackage, create_context_piece
from fractal_agent.validation.context_attribution import ContextAttributionAnalyzer
from fractal_agent.validation.context_completeness import CompletenessEvaluator
from fractal_agent.validation.learning_tracker import LearningTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_learning_tracker_integration():
    """
    Test learning tracker with complete context preparation workflow.
    """
    print("\n" + "=" * 80)
    print("INTEGRATION TEST: Learning Tracker with Context Preparation")
    print("=" * 80)
    print()

    with tempfile.TemporaryDirectory() as temp_dir:
        learning_dir = Path(temp_dir) / "learning"

        # Initialize components
        tracker = LearningTracker(log_dir=str(learning_dir))
        attribution_analyzer = ContextAttributionAnalyzer(similarity_threshold=0.6)
        completeness_evaluator = CompletenessEvaluator(completeness_threshold=0.8)

        print("✓ Initialized learning tracker and validators")
        print()

        # ====================================================================
        # Simulate multiple attempts with varying quality
        # ====================================================================
        print("Simulating 5 context preparation attempts...")
        print("-" * 80)

        attempts_data = [
            {
                "task": "Research VSM System 1 operational patterns",
                "agent_type": "research",
                "context_quality": "excellent",
                "precision_target": 0.9,
                "completeness_target": 0.95
            },
            {
                "task": "Implement Python class for coordination",
                "agent_type": "developer",
                "context_quality": "good",
                "precision_target": 0.75,
                "completeness_target": 0.85
            },
            {
                "task": "Research System 2 coordination mechanisms",
                "agent_type": "research",
                "context_quality": "poor",
                "precision_target": 0.45,
                "completeness_target": 0.60
            },
            {
                "task": "Implement VSM System 3 optimizer",
                "agent_type": "developer",
                "context_quality": "good",
                "precision_target": 0.80,
                "completeness_target": 0.90
            },
            {
                "task": "Research cybernetic control theory",
                "agent_type": "research",
                "context_quality": "excellent",
                "precision_target": 0.92,
                "completeness_target": 0.98
            }
        ]

        for i, attempt_data in enumerate(attempts_data, 1):
            print(f"\nAttempt {i}: {attempt_data['task'][:50]}...")
            print(f"  Quality: {attempt_data['context_quality']}")

            # Create mock context package
            context_package = ContextPackage(
                domain_knowledge=f"Domain knowledge for {attempt_data['task']}",
                recent_examples=[{"task": "previous task", "outcome": "success"}],
                constraints="Follow best practices",
                current_info="Current information",
                confidence=attempt_data['completeness_target'],
                sources_used=["graphrag", "recent_tasks", "obsidian"],
                total_tokens=150 + (i * 20),
                preparation_time=0.8 + (i * 0.1),
                iterations=1 if attempt_data['context_quality'] == "excellent" else 2
            )

            # Add context pieces
            context_package.context_pieces = [
                create_context_piece(
                    content=f"VSM concept related to {attempt_data['task']}",
                    source="graphrag",
                    source_id=f"entity_{i}_1",
                    relevance_score=0.9,
                    piece_type="domain_knowledge"
                ),
                create_context_piece(
                    content=f"Previous work on {attempt_data['agent_type']} tasks",
                    source="recent_tasks",
                    source_id=f"task_{i}_prev",
                    relevance_score=0.75,
                    piece_type="example"
                ),
                create_context_piece(
                    content=f"Guidelines for {attempt_data['agent_type']} agents",
                    source="obsidian",
                    source_id=f"note_{i}",
                    relevance_score=0.85,
                    piece_type="constraint"
                )
            ]

            # Create mock agent output with varying quality
            if attempt_data['context_quality'] == "excellent":
                # Uses all context pieces
                mock_output = f"""
                Based on VSM concepts, {attempt_data['task']} involves understanding
                the operational patterns. Following previous work on {attempt_data['agent_type']} tasks,
                we apply the guidelines to ensure quality. The approach is comprehensive and well-supported.
                """
            elif attempt_data['context_quality'] == "good":
                # Uses some context pieces
                mock_output = f"""
                The task {attempt_data['task']} requires understanding some concepts.
                Following previous work, we can implement a solution. The approach is reasonable.
                """
            else:
                # Poor context usage
                mock_output = f"""
                This task might involve some concepts, but I'm not entirely sure.
                Assuming we follow standard practices, we could probably implement something.
                More information needed to be certain about the best approach.
                """

            # Run validation
            attribution_result = attribution_analyzer.analyze(
                context_package=context_package,
                agent_output=mock_output,
                method="hybrid",
                verbose=False
            )

            completeness_result = completeness_evaluator.evaluate(
                context_package=context_package,
                agent_output=mock_output,
                user_task=attempt_data['task'],
                verbose=False
            )

            print(f"  Attribution: {attribution_result.precision*100:.1f}% precision")
            print(f"  Completeness: {completeness_result.completeness_score*100:.1f}%")

            # Record in learning tracker
            recorded_attempt = tracker.record_attempt(
                user_task=attempt_data['task'],
                agent_type=attempt_data['agent_type'],
                context_package=context_package,
                attribution_result=attribution_result,
                completeness_result=completeness_result,
                agent_succeeded=(completeness_result.is_complete and attribution_result.precision >= 0.7),
                tags=[attempt_data['context_quality'], attempt_data['agent_type']]
            )

            print(f"  ✓ Recorded as {recorded_attempt.attempt_id}")

        print()
        print("=" * 80)
        print()

        # ====================================================================
        # Analyze learning metrics
        # ====================================================================
        print("LEARNING METRICS ANALYSIS")
        print("=" * 80)
        print()

        # Overall metrics
        print("Overall Metrics:")
        print("-" * 80)
        metrics = tracker.get_learning_metrics()

        print(f"Total Attempts: {metrics.total_attempts}")
        print(f"Average Precision: {metrics.avg_precision*100:.1f}%")
        print(f"Average Completeness: {metrics.avg_completeness*100:.1f}%")
        print(f"Completeness Rate: {metrics.completeness_rate*100:.1f}%")
        print(f"Average Iterations: {metrics.avg_iterations:.2f}")
        print(f"Average Prep Time: {metrics.avg_prep_time:.2f}s")
        print()

        # Source effectiveness
        print("Source Effectiveness:")
        print("-" * 80)
        for source, stats in sorted(metrics.source_stats.items(), key=lambda x: x[1]['avg_precision'], reverse=True):
            print(f"{source:15} {stats['avg_precision']*100:5.1f}% precision  {stats['success_rate']*100:5.1f}% success  ({stats['attempts']} attempts)")
        print()

        # Key insights
        print("Key Insights:")
        print("-" * 80)
        for i, insight in enumerate(metrics.insights, 1):
            print(f"{i}. {insight}")
        print()

        # ====================================================================
        # Analyze by agent type
        # ====================================================================
        print("By Agent Type:")
        print("-" * 80)

        for agent_type in ["research", "developer"]:
            agent_metrics = tracker.get_learning_metrics(agent_type=agent_type)
            print(f"{agent_type.capitalize():12} {agent_metrics.total_attempts} attempts  "
                  f"precision={agent_metrics.avg_precision*100:.1f}%  "
                  f"completeness={agent_metrics.avg_completeness*100:.1f}%")
        print()

        # ====================================================================
        # Identify failures and successes
        # ====================================================================
        print("Failures and Successes:")
        print("-" * 80)

        failures = tracker.get_failures(min_precision=0.6, min_completeness=0.7)
        successes = tracker.get_successes(min_precision=0.7, min_completeness=0.8)

        print(f"Failures: {len(failures)} attempts")
        for failure in failures:
            print(f"  • {failure.user_task[:50]} (precision={failure.attribution_precision:.2f}, completeness={failure.completeness_score:.2f})")

        print()
        print(f"Successes: {len(successes)} attempts")
        for success in successes:
            print(f"  • {success.user_task[:50]} (precision={success.attribution_precision:.2f}, completeness={success.completeness_score:.2f})")
        print()

        # ====================================================================
        # Export report
        # ====================================================================
        print("Exporting Learning Report:")
        print("-" * 80)

        report_file = learning_dir / "learning_report.md"
        tracker.export_report(str(report_file), format="markdown")

        print(f"✓ Exported to {report_file}")
        print()

        # Display excerpt
        with open(report_file, 'r') as f:
            content = f.read()
            lines = content.split('\n')
            print("Report Excerpt:")
            for line in lines[:20]:
                print(f"  {line}")
        print()

        # ====================================================================
        # Validation
        # ====================================================================
        print("=" * 80)
        print("VALIDATION")
        print("=" * 80)
        print()

        # Check that learning data was recorded correctly
        assert metrics.total_attempts == 5, f"Expected 5 attempts, got {metrics.total_attempts}"
        assert 0.0 <= metrics.avg_precision <= 1.0, "Precision out of range"
        assert 0.0 <= metrics.avg_completeness <= 1.0, "Completeness out of range"
        assert len(metrics.source_stats) == 3, f"Expected 3 sources, got {len(metrics.source_stats)}"
        assert len(metrics.insights) > 0, "No insights generated"
        assert failures, "Should have at least one failure"
        assert successes, "Should have at least one success"

        print("✅ All assertions passed")
        print()

        print("=" * 80)
        print("✅ INTEGRATION TEST PASSED")
        print("=" * 80)
        print()
        print("Learning Tracker is WORKING:")
        print("  1. ✓ Records context preparation attempts")
        print("  2. ✓ Tracks attribution and completeness results")
        print("  3. ✓ Analyzes trends and patterns")
        print("  4. ✓ Identifies source effectiveness")
        print("  5. ✓ Generates actionable insights")
        print("  6. ✓ Exports learning reports")
        print()
        print("System has continuous learning capability.")
        print()

        return True


if __name__ == "__main__":
    try:
        success = test_learning_tracker_integration()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Test failed with exception: {e}", exc_info=True)
        print()
        print("=" * 80)
        print("❌ TEST FAILED WITH EXCEPTION")
        print("=" * 80)
        print(f"Error: {e}")
        print()
        sys.exit(1)
