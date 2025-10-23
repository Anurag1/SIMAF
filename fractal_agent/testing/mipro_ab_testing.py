"""
MIPRO + A/B Testing Integration - Phase 3

Integration layer for running A/B tests of different MIPRO optimization strategies
to determine which optimization approach produces the best results.

Author: BMad
Date: 2025-10-19
"""

from typing import List, Dict, Any, Optional, Callable
import dspy
from datetime import datetime
import logging

from fractal_agent.testing.ab_testing import (
    ABTestFramework,
    Variant,
    VariantType,
    ABTestResult
)

logger = logging.getLogger(__name__)


def run_mipro_ab_test(
    base_agent: dspy.Module,
    optimization_configs: List[Dict[str, Any]],
    test_tasks: List[Any],
    metric_fn: Optional[Callable] = None,
    test_name: str = "mipro_optimization_comparison",
    results_dir: str = "./ab_tests"
) -> Dict:
    """
    Run A/B test of different MIPRO optimization strategies.

    Compares different MIPRO configurations to determine which
    optimization strategy produces the best-performing agent.

    Args:
        base_agent: Base dspy.Module to optimize
        optimization_configs: List of MIPRO configs to test, each containing:
            - id: Unique identifier for this config
            - name: Human-readable name
            - mipro_params: Dict of MIPRO parameters
              (num_candidates, num_trials, metric, etc.)
            - traffic_percentage: Optional, defaults to equal split
        test_tasks: List of tasks to evaluate optimized agents on
        metric_fn: Optional custom metric function for MIPRO
                   Signature: (gold, pred, trace=None) -> float
        test_name: Name for this A/B test
        results_dir: Directory to save results

    Returns:
        Dict containing:
        - variants: List of tested variants
        - results: Raw A/B test results
        - analysis: Statistical analysis of results
        - comparison: Comparison against baseline
        - recommendation: Which variant performed best

    Example:
        >>> from fractal_agent.agents.research_agent import ResearchAgent
        >>> from fractal_agent.agents.research_config import ResearchConfig
        >>>
        >>> # Define base agent
        >>> base_agent = ResearchAgent(config=ResearchConfig())
        >>>
        >>> # Define optimization strategies to test
        >>> optimization_configs = [
        ...     {
        ...         "id": "baseline",
        ...         "name": "No Optimization (Baseline)",
        ...         "mipro_params": None,  # Skip optimization
        ...         "traffic_percentage": 25.0
        ...     },
        ...     {
        ...         "id": "fast",
        ...         "name": "Fast MIPRO",
        ...         "mipro_params": {
        ...             "num_candidates": 5,
        ...             "num_trials": 10,
        ...             "max_bootstrapped_demos": 2
        ...         },
        ...         "traffic_percentage": 25.0
        ...     },
        ...     {
        ...         "id": "balanced",
        ...         "name": "Balanced MIPRO",
        ...         "mipro_params": {
        ...             "num_candidates": 10,
        ...             "num_trials": 20,
        ...             "max_bootstrapped_demos": 4
        ...         },
        ...         "traffic_percentage": 25.0
        ...     },
        ...     {
        ...         "id": "thorough",
        ...         "name": "Thorough MIPRO",
        ...         "mipro_params": {
        ...             "num_candidates": 20,
        ...             "num_trials": 50,
        ...             "max_bootstrapped_demos": 8
        ...         },
        ...         "traffic_percentage": 25.0
        ...     }
        ... ]
        >>>
        >>> # Define test tasks
        >>> test_tasks = [
        ...     {"query": "What is quantum computing?", "expected_quality": 0.8},
        ...     {"query": "Explain machine learning", "expected_quality": 0.8},
        ...     # ... more tasks
        ... ]
        >>>
        >>> # Run A/B test
        >>> results = run_mipro_ab_test(
        ...     base_agent=base_agent,
        ...     optimization_configs=optimization_configs,
        ...     test_tasks=test_tasks
        ... )
        >>>
        >>> # View recommendation
        >>> print(f"Best variant: {results['recommendation']['variant_id']}")
        >>> print(f"Improvement: {results['recommendation']['improvement']:.1%}")
    """
    logger.info(f"Starting MIPRO A/B test with {len(optimization_configs)} configurations")

    # Create variants from optimization configs
    num_configs = len(optimization_configs)
    equal_traffic = 100.0 / num_configs

    variants = []
    for config in optimization_configs:
        variant = Variant(
            id=config["id"],
            name=config.get("name", config["id"]),
            variant_type=VariantType.CONFIG,
            config=config,
            traffic_percentage=config.get("traffic_percentage", equal_traffic)
        )
        variants.append(variant)

    # Agent factory: creates agent with specified MIPRO optimization
    def agent_factory(variant_config: Dict) -> dspy.Module:
        """Create agent with specified MIPRO optimization"""
        mipro_params = variant_config.get("mipro_params")

        if mipro_params is None:
            # Baseline: no optimization
            logger.debug(f"Creating baseline agent (no MIPRO optimization)")
            return base_agent

        # Run MIPRO optimization
        logger.debug(f"Optimizing agent with MIPRO params: {mipro_params}")

        try:
            from dspy.teleprompt import MIPRO

            # Create MIPRO optimizer
            optimizer = MIPRO(
                metric=metric_fn,
                **mipro_params
            )

            # Optimize agent (using training subset of tasks)
            # Use first 50% of tasks for training, last 50% for testing
            train_size = len(test_tasks) // 2
            train_set = test_tasks[:train_size]

            optimized_agent = optimizer.compile(
                student=base_agent,
                trainset=train_set
            )

            return optimized_agent

        except Exception as e:
            logger.error(f"MIPRO optimization failed: {e}")
            # Fallback to baseline
            return base_agent

    # Task function: runs task on agent and measures performance
    def task_fn(agent: dspy.Module) -> Dict:
        """Run tasks on agent and collect metrics"""
        # Use test subset (second 50% of tasks)
        test_size = len(test_tasks) // 2
        test_set = test_tasks[test_size:]

        successes = 0
        total_cost = 0.0
        total_latency = 0.0
        accuracies = []

        for task in test_set:
            try:
                start_time = datetime.now()

                # Run task
                if isinstance(task, dict):
                    query = task.get("query", task.get("question", ""))
                    result = agent(query=query)
                else:
                    result = agent(task)

                end_time = datetime.now()
                latency = (end_time - start_time).total_seconds() * 1000  # ms

                # Evaluate result
                if metric_fn and isinstance(task, dict):
                    expected = task.get("expected", task.get("answer", ""))
                    accuracy = metric_fn(expected, result)
                else:
                    # Default: assume result has accuracy attribute
                    accuracy = getattr(result, 'accuracy', 0.8)

                accuracies.append(accuracy)
                total_latency += latency

                # Estimate cost (rough approximation based on token usage)
                # This should be customized based on actual LM provider
                estimated_cost = 0.01  # Placeholder
                total_cost += estimated_cost

                # Consider success if accuracy > threshold
                if accuracy > 0.7:
                    successes += 1

            except Exception as e:
                logger.error(f"Task execution failed: {e}")
                # Count as failure
                accuracies.append(0.0)

        # Calculate metrics
        num_tasks = len(test_set)
        success_rate = successes / num_tasks if num_tasks > 0 else 0.0
        avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
        avg_cost = total_cost / num_tasks if num_tasks > 0 else 0.0
        avg_latency = total_latency / num_tasks if num_tasks > 0 else 0.0

        return {
            "success": success_rate > 0.7,
            "metrics": {
                "accuracy": avg_accuracy,
                "success_rate": success_rate,
                "cost": avg_cost,
                "latency": avg_latency
            }
        }

    # Create and run A/B test
    ab_test = ABTestFramework(
        test_name=test_name,
        variants=variants,
        results_dir=results_dir
    )

    # Run test (1 trial per variant since each trial runs multiple tasks)
    num_trials_per_variant = 3  # Run each variant multiple times for stability
    results = ab_test.run_test(
        agent_factory=agent_factory,
        task_fn=task_fn,
        num_trials=num_trials_per_variant * len(variants)
    )

    # Analyze results
    analysis = ab_test.analyze_results(results)

    # Find baseline variant
    baseline_id = next(
        (v.id for v in variants if v.config.get("mipro_params") is None),
        variants[0].id  # Fallback to first variant
    )

    # Compare against baseline
    comparison = ab_test.compare_variants(
        results=results,
        baseline_variant_id=baseline_id,
        metric="avg_accuracy"
    )

    # Determine best variant
    best_variant_id = max(
        analysis.keys(),
        key=lambda vid: analysis[vid].get("avg_accuracy", 0)
    )

    best_analysis = analysis[best_variant_id]
    baseline_analysis = analysis[baseline_id]

    improvement = (
        best_analysis["avg_accuracy"] - baseline_analysis["avg_accuracy"]
    ) / baseline_analysis["avg_accuracy"] if baseline_analysis["avg_accuracy"] > 0 else 0.0

    recommendation = {
        "variant_id": best_variant_id,
        "variant_name": next(v.name for v in variants if v.id == best_variant_id),
        "accuracy": best_analysis["avg_accuracy"],
        "improvement": improvement,
        "cost_impact": best_analysis["avg_cost"] - baseline_analysis["avg_cost"],
        "recommendation": (
            f"Use '{best_variant_id}' variant for {improvement:.1%} accuracy improvement"
            if improvement > 0.05
            else "Baseline performs similarly to optimized variants"
        )
    }

    logger.info(
        f"MIPRO A/B test complete. Best variant: {best_variant_id} "
        f"({improvement:+.1%} improvement)"
    )

    return {
        "test_name": test_name,
        "variants": [v.id for v in variants],
        "results": results,
        "analysis": analysis,
        "comparison": comparison,
        "recommendation": recommendation
    }


def compare_mipro_presets(
    base_agent: dspy.Module,
    test_tasks: List[Any],
    metric_fn: Optional[Callable] = None,
    presets: Optional[List[str]] = None
) -> Dict:
    """
    Quick comparison of common MIPRO preset configurations.

    Args:
        base_agent: Base agent to optimize
        test_tasks: Tasks for evaluation
        metric_fn: Optional custom metric function
        presets: List of preset names to test
                 Options: ["baseline", "fast", "balanced", "thorough"]
                 Defaults to all presets

    Returns:
        A/B test results with recommendation

    Example:
        >>> results = compare_mipro_presets(
        ...     base_agent=my_agent,
        ...     test_tasks=my_tasks
        ... )
        >>> print(results['recommendation'])
    """
    presets = presets or ["baseline", "fast", "balanced", "thorough"]

    # Define preset configurations
    preset_configs = {
        "baseline": {
            "id": "baseline",
            "name": "No Optimization (Baseline)",
            "mipro_params": None
        },
        "fast": {
            "id": "fast",
            "name": "Fast MIPRO (5 candidates, 10 trials)",
            "mipro_params": {
                "num_candidates": 5,
                "num_trials": 10,
                "max_bootstrapped_demos": 2
            }
        },
        "balanced": {
            "id": "balanced",
            "name": "Balanced MIPRO (10 candidates, 20 trials)",
            "mipro_params": {
                "num_candidates": 10,
                "num_trials": 20,
                "max_bootstrapped_demos": 4
            }
        },
        "thorough": {
            "id": "thorough",
            "name": "Thorough MIPRO (20 candidates, 50 trials)",
            "mipro_params": {
                "num_candidates": 20,
                "num_trials": 50,
                "max_bootstrapped_demos": 8
            }
        }
    }

    # Select requested presets
    configs = [preset_configs[p] for p in presets if p in preset_configs]

    if not configs:
        raise ValueError(f"No valid presets selected. Available: {list(preset_configs.keys())}")

    # Run A/B test
    return run_mipro_ab_test(
        base_agent=base_agent,
        optimization_configs=configs,
        test_tasks=test_tasks,
        metric_fn=metric_fn,
        test_name="mipro_preset_comparison"
    )


# Demo
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("=" * 80)
    print("MIPRO + A/B Testing Integration Demo - Phase 3")
    print("=" * 80)
    print()

    # Mock setup for demo
    class MockAgent(dspy.Module):
        def __init__(self):
            super().__init__()
            self.optimized = False

        def forward(self, query):
            # Mock result with random accuracy
            import random
            accuracy = random.uniform(0.7, 0.9) + (0.05 if self.optimized else 0.0)
            return type('Result', (), {'accuracy': accuracy})

    def mock_metric(gold, pred, trace=None):
        """Mock metric function"""
        return pred.accuracy if hasattr(pred, 'accuracy') else 0.8

    # Create base agent
    print("[1/3] Creating base agent...")
    base_agent = MockAgent()
    print("   Base agent created")
    print()

    # Create test tasks
    print("[2/3] Preparing test tasks...")
    test_tasks = [
        {"query": f"Test query {i}", "expected": f"Expected answer {i}"}
        for i in range(20)
    ]
    print(f"   Created {len(test_tasks)} test tasks")
    print()

    # Run preset comparison
    print("[3/3] Running MIPRO preset comparison...")
    print("   NOTE: This is a demo with mocked agents")
    print()

    try:
        results = compare_mipro_presets(
            base_agent=base_agent,
            test_tasks=test_tasks,
            metric_fn=mock_metric,
            presets=["baseline", "fast"]  # Limited for demo
        )

        print()
        print("=" * 80)
        print("MIPRO A/B Test Results:")
        print("=" * 80)

        # Show analysis
        for variant_id, metrics in results["analysis"].items():
            print(f"\nVariant: {variant_id}")
            print(f"  Trials: {metrics['num_trials']}")
            print(f"  Avg Accuracy: {metrics['avg_accuracy']:.2%}")
            print(f"  Avg Cost: ${metrics['avg_cost']:.4f}")

        # Show recommendation
        print()
        print("=" * 80)
        print("Recommendation:")
        print("=" * 80)
        rec = results["recommendation"]
        print(f"\nBest Variant: {rec['variant_name']}")
        print(f"Accuracy: {rec['accuracy']:.2%}")
        print(f"Improvement: {rec['improvement']:+.1%}")
        print(f"\n{rec['recommendation']}")

    except Exception as e:
        print(f"\n⚠️  Demo limitation: {e}")
        print("   Full functionality requires dspy.teleprompt.MIPRO")

    print()
    print("=" * 80)
    print("MIPRO + A/B Testing Integration Demo Complete!")
    print("=" * 80)
