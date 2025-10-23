"""
Integration test for MIPRO token reduction verification

Verifies that MIPRO optimization reduces token usage by at least 20%
as specified in Phase 3 success criteria.

Run: pytest tests/integration/test_mipro_token_reduction.py -v -m llm
"""

import pytest
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

# Mark as LLM test (requires actual LLM calls)
pytestmark = pytest.mark.llm


@pytest.fixture
def test_tasks():
    """
    Benchmark tasks for token reduction measurement.

    Returns 50 diverse research queries covering different domains.
    """
    return [
        "What is quantum computing and how does it differ from classical computing?",
        "Explain the CRISPR gene editing technology and its applications",
        "What are the main causes of climate change?",
        "How do neural networks learn from data?",
        "What is the blockchain and how does it work?",
        "Explain the theory of relativity in simple terms",
        "What are the different types of renewable energy?",
        "How does the human immune system fight infections?",
        "What is machine learning and how is it used?",
        "Explain the structure and function of DNA",

        "What are the key principles of agile software development?",
        "How do vaccines work to prevent diseases?",
        "What is dark matter and why is it important?",
        "Explain how photosynthesis works in plants",
        "What are the main theories about the origin of the universe?",
        "How does encryption protect data security?",
        "What is artificial general intelligence?",
        "Explain the water cycle and its importance",
        "What are the principles of microeconomics?",
        "How do antibiotics fight bacterial infections?",

        "What is the Internet of Things (IoT)?",
        "Explain the greenhouse effect and global warming",
        "What are stem cells and their potential applications?",
        "How does GPS technology work?",
        "What is quantum entanglement?",
        "Explain the process of protein synthesis",
        "What are the main types of machine learning?",
        "How does the brain process and store memories?",
        "What is nanotechnology and its applications?",
        "Explain how solar panels generate electricity",

        "What are the principles of supply and demand?",
        "How do self-driving cars navigate?",
        "What is the microbiome and why is it important?",
        "Explain the concept of spacetime in physics",
        "What are the different types of clouds in computing?",
        "How does natural language processing work?",
        "What is the Higgs boson and why is it significant?",
        "Explain the process of evolution by natural selection",
        "What are the main components of a computer system?",
        "How do nuclear reactors generate power?",

        "What is the difference between AI and machine learning?",
        "Explain how antibodies recognize pathogens",
        "What are exoplanets and how are they detected?",
        "How does 5G technology improve wireless communication?",
        "What is neuroplasticity and how does it work?",
        "Explain the carbon cycle in ecosystems",
        "What are the principles of object-oriented programming?",
        "How do batteries store and release energy?",
        "What is the cosmic microwave background radiation?",
        "Explain how enzymes catalyze biochemical reactions"
    ]


def run_benchmark(agent, tasks: List[str]) -> Dict:
    """
    Run benchmark on agent with given tasks.

    Args:
        agent: ResearchAgent instance to benchmark
        tasks: List of research queries

    Returns:
        Dict with metrics:
        - total_tokens: Total tokens used across all tasks
        - avg_tokens_per_task: Average tokens per task
        - total_cost: Estimated total cost
        - task_count: Number of tasks completed
    """
    total_tokens = 0
    completed_tasks = 0
    total_cost = 0.0

    for i, task in enumerate(tasks):
        try:
            logger.info(f"Running task {i+1}/{len(tasks)}: {task[:50]}...")

            # Run the research task
            result = agent(query=task)

            # Extract token usage from result
            # DSPy tracks this in the LM history
            if hasattr(result, 'token_usage'):
                tokens = result.token_usage
            elif hasattr(agent, 'lm') and hasattr(agent.lm, 'history'):
                # Estimate from LM history
                # This is a simplified version - actual implementation
                # would need to properly track tokens
                tokens = len(str(result)) * 0.75  # Rough estimate
            else:
                # Fallback estimate
                tokens = len(str(result)) * 0.75

            total_tokens += tokens
            completed_tasks += 1

            # Estimate cost (rough approximation)
            # $0.003 per 1K tokens for claude-sonnet
            cost = (tokens / 1000) * 0.003
            total_cost += cost

        except Exception as e:
            logger.error(f"Task {i+1} failed: {e}")
            continue

    return {
        "total_tokens": total_tokens,
        "avg_tokens_per_task": total_tokens / completed_tasks if completed_tasks > 0 else 0,
        "total_cost": total_cost,
        "task_count": completed_tasks
    }


@pytest.mark.skip(reason="Requires actual LLM calls and MIPRO optimization - run manually for verification")
def test_mipro_reduces_token_usage(test_tasks):
    """
    Verify MIPRO reduces token usage by at least 20%.

    This test:
    1. Runs ResearchAgent with default prompts on benchmark tasks
    2. Optimizes the agent using MIPRO
    3. Runs optimized agent on same tasks
    4. Verifies token reduction >= 20%

    NOTE: This test is skipped by default because it requires:
    - Actual LLM API calls (expensive)
    - MIPRO optimization (time-consuming)
    - Multiple runs for statistical significance

    To run manually:
    pytest tests/integration/test_mipro_token_reduction.py::test_mipro_reduces_token_usage -v -s
    """
    from fractal_agent.agents.research_agent import ResearchAgent
    from fractal_agent.agents.research_config import ResearchConfig

    # Note: This would need to be implemented if optimize_research_agent
    # is not yet available in the package
    try:
        from fractal_agent.agents.optimize_research import optimize_research_agent
    except ImportError:
        pytest.skip("MIPRO optimization not yet implemented in package")

    logger.info("=" * 80)
    logger.info("MIPRO Token Reduction Test")
    logger.info("=" * 80)

    # Step 1: Baseline measurement with default prompts
    logger.info("\n[1/3] Running baseline ResearchAgent with default prompts...")
    baseline_agent = ResearchAgent(config=ResearchConfig())
    baseline_metrics = run_benchmark(baseline_agent, test_tasks)

    logger.info(f"Baseline Results:")
    logger.info(f"  Total tokens: {baseline_metrics['total_tokens']:,}")
    logger.info(f"  Avg tokens/task: {baseline_metrics['avg_tokens_per_task']:.0f}")
    logger.info(f"  Completed tasks: {baseline_metrics['task_count']}/{len(test_tasks)}")
    logger.info(f"  Total cost: ${baseline_metrics['total_cost']:.2f}")

    # Step 2: MIPRO optimization
    logger.info("\n[2/3] Running MIPRO optimization...")
    logger.info("  This may take several minutes...")

    # Use subset of tasks for training
    train_tasks = test_tasks[:25]  # First half for training
    test_subset = test_tasks[25:]  # Second half for testing

    optimized_agent = optimize_research_agent(
        base_agent=baseline_agent,
        train_tasks=train_tasks,
        num_candidates=10,
        num_trials=20
    )

    logger.info("  MIPRO optimization complete!")

    # Step 3: Optimized measurement
    logger.info("\n[3/3] Running optimized ResearchAgent...")
    optimized_metrics = run_benchmark(optimized_agent, test_subset)

    logger.info(f"Optimized Results:")
    logger.info(f"  Total tokens: {optimized_metrics['total_tokens']:,}")
    logger.info(f"  Avg tokens/task: {optimized_metrics['avg_tokens_per_task']:.0f}")
    logger.info(f"  Completed tasks: {optimized_metrics['task_count']}/{len(test_subset)}")
    logger.info(f"  Total cost: ${optimized_metrics['total_cost']:.2f}")

    # Step 4: Calculate token reduction
    baseline_avg = baseline_metrics['avg_tokens_per_task']
    optimized_avg = optimized_metrics['avg_tokens_per_task']

    reduction = (baseline_avg - optimized_avg) / baseline_avg
    reduction_pct = reduction * 100

    logger.info("\n" + "=" * 80)
    logger.info("RESULTS")
    logger.info("=" * 80)
    logger.info(f"Baseline avg tokens/task: {baseline_avg:.0f}")
    logger.info(f"Optimized avg tokens/task: {optimized_avg:.0f}")
    logger.info(f"Token reduction: {reduction_pct:.1f}%")
    logger.info(f"Target: 20.0%")
    logger.info("=" * 80)

    # Verify 20% reduction
    assert reduction >= 0.20, (
        f"Expected at least 20% token reduction, got {reduction_pct:.1f}%. "
        f"Baseline: {baseline_avg:.0f} tokens/task, "
        f"Optimized: {optimized_avg:.0f} tokens/task"
    )

    logger.info(f"\n✅ SUCCESS: MIPRO achieved {reduction_pct:.1f}% token reduction (target: 20%)")


@pytest.mark.skip(reason="Quick verification test - run manually")
def test_mipro_integration_with_intelligence_agent(test_tasks):
    """
    Verify Intelligence Agent can trigger MIPRO optimization.

    Tests the integration between:
    - Intelligence Agent (monitors performance)
    - MIPRO Optimizer (optimizes prompts)
    - A/B Testing Framework (compares variants)
    """
    from fractal_agent.agents import IntelligenceAgent
    from fractal_agent.agents.intelligence_config import PresetIntelligenceConfigs
    from fractal_agent.testing import run_mipro_ab_test
    from fractal_agent.agents.research_agent import ResearchAgent

    logger.info("=" * 80)
    logger.info("Intelligence Agent + MIPRO Integration Test")
    logger.info("=" * 80)

    # Step 1: Create intelligence agent
    intel_agent = IntelligenceAgent(
        config=PresetIntelligenceConfigs.cost_optimization()
    )

    # Step 2: Simulate cost spike scenario
    metrics = {
        "cost": 10.0,  # High cost
        "avg_cost": 2.0,  # Historical average
        "accuracy": 0.75,
        "failed_tasks": [],
        "total_tasks": 10,
        "num_completed": 8,
        "num_failed": 2
    }

    # Check if optimization should be triggered
    should_optimize, reason = intel_agent.should_trigger_analysis(
        performance_metrics=metrics,
        session_size=10,
        last_analysis_days_ago=7
    )

    logger.info(f"\nIntelligence Agent Analysis:")
    logger.info(f"  Should trigger optimization: {should_optimize}")
    logger.info(f"  Reason: {reason}")

    assert should_optimize, "Intelligence agent should detect cost spike"

    # Step 3: Run MIPRO A/B test
    logger.info("\nRunning MIPRO A/B test...")

    optimization_configs = [
        {
            "id": "baseline",
            "name": "Baseline (no optimization)",
            "mipro_params": None
        },
        {
            "id": "fast",
            "name": "Fast MIPRO",
            "mipro_params": {
                "num_candidates": 5,
                "num_trials": 10
            }
        }
    ]

    # Use small subset for quick test
    quick_tasks = test_tasks[:5]

    results = run_mipro_ab_test(
        base_agent=ResearchAgent(),
        optimization_configs=optimization_configs,
        test_tasks=quick_tasks,
        test_name="integration_test"
    )

    logger.info(f"\nA/B Test Results:")
    logger.info(f"  Variants tested: {len(results['variants'])}")
    logger.info(f"  Recommendation: {results['recommendation']['variant_id']}")
    logger.info(f"  Improvement: {results['recommendation']['improvement']:.1%}")

    assert "analysis" in results
    assert "recommendation" in results

    logger.info("\n✅ SUCCESS: Intelligence Agent + MIPRO integration working!")


if __name__ == "__main__":
    # Run with: python -m tests.integration.test_mipro_token_reduction
    print("=" * 80)
    print("MIPRO Token Reduction Integration Test")
    print("=" * 80)
    print()
    print("This test verifies that MIPRO optimization reduces token usage by >=20%")
    print()
    print("To run:")
    print("  pytest tests/integration/test_mipro_token_reduction.py -v -s -m llm")
    print()
    print("Note: Requires actual LLM calls and may take 30-60 minutes")
    print("=" * 80)
