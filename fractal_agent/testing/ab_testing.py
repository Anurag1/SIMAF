"""
A/B Testing Framework - Phase 3

Framework for comparing agent variants with different configurations,
prompts, models, or parameters to determine optimal performance.

Author: BMad
Date: 2025-10-19
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Callable, Any, Optional
from enum import Enum
from datetime import datetime
import random
import json
import logging
from pathlib import Path
import math

logger = logging.getLogger(__name__)


class VariantType(Enum):
    """Types of agent variants for A/B testing"""
    PROMPT = "prompt"           # Different prompts
    MODEL = "model"             # Different model tiers
    TEMPERATURE = "temperature" # Different temperatures
    CONFIG = "config"           # Different configurations


@dataclass
class Variant:
    """
    A/B test variant definition.

    Attributes:
        id: Unique identifier for this variant
        name: Human-readable name
        variant_type: Type of variation being tested
        config: Configuration dictionary for this variant
        traffic_percentage: Percentage of traffic allocated (0-100)
    """
    id: str
    name: str
    variant_type: VariantType
    config: Dict[str, Any]
    traffic_percentage: float = 50.0  # % of traffic to this variant

    def __post_init__(self):
        """Validate variant configuration"""
        if not 0 <= self.traffic_percentage <= 100:
            raise ValueError(f"Traffic percentage must be 0-100, got {self.traffic_percentage}")

        # Convert string to enum if needed
        if isinstance(self.variant_type, str):
            self.variant_type = VariantType(self.variant_type)


@dataclass
class ABTestResult:
    """
    Results from a single A/B test trial.

    Attributes:
        variant_id: ID of variant that was tested
        task_id: Unique identifier for the task
        success: Whether the task succeeded
        metrics: Performance metrics (accuracy, cost, latency, etc.)
        timestamp: When the test was run
    """
    variant_id: str
    task_id: str
    success: bool
    metrics: Dict[str, float]
    timestamp: str


class ABTestFramework:
    """
    A/B testing framework for agent variants.

    Allows testing different configurations, prompts, or models
    to determine which performs better on real tasks.

    Usage:
        >>> # Define variants
        >>> variants = [
        ...     Variant(
        ...         id="baseline",
        ...         name="Baseline Agent",
        ...         variant_type=VariantType.PROMPT,
        ...         config={"prompt": "Original prompt"},
        ...         traffic_percentage=50.0
        ...     ),
        ...     Variant(
        ...         id="optimized",
        ...         name="Optimized Agent",
        ...         variant_type=VariantType.PROMPT,
        ...         config={"prompt": "MIPRO-optimized prompt"},
        ...         traffic_percentage=50.0
        ...     )
        ... ]
        >>>
        >>> # Create test framework
        >>> ab_test = ABTestFramework(
        ...     test_name="prompt_optimization_v1",
        ...     variants=variants
        ... )
        >>>
        >>> # Run test
        >>> results = ab_test.run_test(
        ...     agent_factory=lambda cfg: create_agent(cfg),
        ...     task_fn=lambda agent: run_task(agent),
        ...     num_trials=100
        ... )
        >>>
        >>> # Analyze results
        >>> analysis = ab_test.analyze_results(results)
        >>> print(analysis)
    """

    def __init__(
        self,
        test_name: str,
        variants: List[Variant],
        results_dir: str = "./ab_tests"
    ):
        """
        Initialize A/B test framework.

        Args:
            test_name: Name of this A/B test
            variants: List of variants to test
            results_dir: Directory to save results

        Raises:
            ValueError: If traffic percentages don't sum to 100
        """
        self.test_name = test_name
        self.variants = variants
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Validate traffic percentages sum to 100
        total_traffic = sum(v.traffic_percentage for v in variants)
        if abs(total_traffic - 100.0) > 0.01:
            raise ValueError(
                f"Traffic percentages must sum to 100, got {total_traffic}. "
                f"Variants: {[(v.id, v.traffic_percentage) for v in variants]}"
            )

        logger.info(
            f"Initialized A/B test '{test_name}' with {len(variants)} variants: "
            f"{[v.id for v in variants]}"
        )

    def select_variant(self) -> Variant:
        """
        Select a variant based on traffic allocation.

        Uses weighted random selection based on traffic_percentage.

        Returns:
            Selected variant
        """
        rand = random.random() * 100
        cumulative = 0

        for variant in self.variants:
            cumulative += variant.traffic_percentage
            if rand < cumulative:
                return variant

        # Fallback (should not happen with valid percentages)
        return self.variants[-1]

    def run_test(
        self,
        agent_factory: Callable[[Dict], Any],
        task_fn: Callable[[Any], Dict],
        num_trials: int = 100
    ) -> Dict[str, List[ABTestResult]]:
        """
        Run A/B test across variants.

        Args:
            agent_factory: Function that creates agent from config
                          Signature: (config: Dict) -> Agent
            task_fn: Function that runs task on agent and returns result
                     Signature: (agent: Any) -> Dict with keys:
                                - "success": bool
                                - "metrics": Dict[str, float]
            num_trials: Number of tasks to run

        Returns:
            Dict mapping variant_id to list of results

        Example:
            >>> def create_agent(config):
            ...     return ResearchAgent(prompt=config.get("prompt"))
            ...
            >>> def run_task(agent):
            ...     result = agent.run("What is quantum computing?")
            ...     return {
            ...         "success": result.accuracy > 0.8,
            ...         "metrics": {
            ...             "accuracy": result.accuracy,
            ...             "cost": result.cost,
            ...             "latency": result.latency_ms
            ...         }
            ...     }
            ...
            >>> results = ab_test.run_test(
            ...     agent_factory=create_agent,
            ...     task_fn=run_task,
            ...     num_trials=100
            ... )
        """
        logger.info(f"Starting A/B test '{self.test_name}' with {num_trials} trials")

        results = {v.id: [] for v in self.variants}

        for i in range(num_trials):
            # Select variant based on traffic allocation
            variant = self.select_variant()

            try:
                # Create agent with variant config
                agent = agent_factory(variant.config)

                # Run task
                result = task_fn(agent)

                # Record result
                ab_result = ABTestResult(
                    variant_id=variant.id,
                    task_id=f"{self.test_name}_{i}",
                    success=result.get("success", False),
                    metrics=result.get("metrics", {}),
                    timestamp=datetime.now().isoformat()
                )
                results[variant.id].append(ab_result)

                if (i + 1) % 10 == 0:
                    logger.info(f"Completed {i + 1}/{num_trials} trials")

            except Exception as e:
                logger.error(f"Trial {i} failed for variant {variant.id}: {e}")
                # Record failure
                ab_result = ABTestResult(
                    variant_id=variant.id,
                    task_id=f"{self.test_name}_{i}",
                    success=False,
                    metrics={"error": 1.0},
                    timestamp=datetime.now().isoformat()
                )
                results[variant.id].append(ab_result)

        # Save results
        self.save_results(results)

        logger.info(f"A/B test '{self.test_name}' completed")
        return results

    def save_results(self, results: Dict[str, List[ABTestResult]]):
        """
        Save test results to JSON.

        Args:
            results: Dict mapping variant_id to list of results
        """
        output_file = self.results_dir / f"{self.test_name}_results.json"

        with open(output_file, 'w') as f:
            json.dump(
                {
                    "test_name": self.test_name,
                    "timestamp": datetime.now().isoformat(),
                    "variants": [
                        {
                            "id": v.id,
                            "name": v.name,
                            "variant_type": v.variant_type.value,
                            "config": v.config,
                            "traffic_percentage": v.traffic_percentage
                        }
                        for v in self.variants
                    ],
                    "results": {
                        vid: [asdict(r) for r in results_list]
                        for vid, results_list in results.items()
                    }
                },
                f,
                indent=2
            )

        logger.info(f"Results saved to {output_file}")

    def load_results(self, test_name: Optional[str] = None) -> Dict[str, List[ABTestResult]]:
        """
        Load previously saved test results.

        Args:
            test_name: Name of test to load (defaults to self.test_name)

        Returns:
            Dict mapping variant_id to list of results
        """
        test_name = test_name or self.test_name
        results_file = self.results_dir / f"{test_name}_results.json"

        if not results_file.exists():
            raise FileNotFoundError(f"No results found for test '{test_name}'")

        with open(results_file, 'r') as f:
            data = json.load(f)

        # Reconstruct ABTestResult objects
        results = {}
        for vid, results_list in data["results"].items():
            results[vid] = [
                ABTestResult(**result_data)
                for result_data in results_list
            ]

        return results

    def analyze_results(self, results: Dict[str, List[ABTestResult]]) -> Dict[str, Dict]:
        """
        Analyze A/B test results with statistical metrics.

        Args:
            results: Dict mapping variant_id to list of results

        Returns:
            Dict with analysis for each variant including:
            - num_trials: Number of trials run
            - success_rate: Percentage of successful trials
            - avg_cost: Average cost per trial
            - avg_latency: Average latency per trial
            - total_cost: Total cost across all trials
            - std_dev_cost: Standard deviation of cost
            - confidence_interval: 95% confidence interval for success rate
        """
        analysis = {}

        for variant_id, results_list in results.items():
            if not results_list:
                analysis[variant_id] = {
                    "num_trials": 0,
                    "success_rate": 0.0,
                    "error": "No results for this variant"
                }
                continue

            # Basic metrics
            num_trials = len(results_list)
            successes = sum(r.success for r in results_list)
            success_rate = successes / num_trials

            # Cost metrics
            costs = [r.metrics.get("cost", 0) for r in results_list]
            avg_cost = sum(costs) / num_trials
            total_cost = sum(costs)

            # Compute standard deviation
            cost_variance = sum((c - avg_cost) ** 2 for c in costs) / num_trials
            std_dev_cost = math.sqrt(cost_variance)

            # Latency metrics
            latencies = [r.metrics.get("latency", 0) for r in results_list]
            avg_latency = sum(latencies) / num_trials

            # Accuracy metrics (if available)
            accuracies = [r.metrics.get("accuracy", 0) for r in results_list]
            avg_accuracy = sum(accuracies) / num_trials if accuracies else 0.0

            # 95% confidence interval for success rate
            # Using normal approximation: p ± 1.96 * sqrt(p(1-p)/n)
            std_error = math.sqrt(success_rate * (1 - success_rate) / num_trials)
            confidence_interval = (
                max(0, success_rate - 1.96 * std_error),
                min(1, success_rate + 1.96 * std_error)
            )

            analysis[variant_id] = {
                "num_trials": num_trials,
                "success_rate": success_rate,
                "avg_cost": avg_cost,
                "avg_latency": avg_latency,
                "avg_accuracy": avg_accuracy,
                "total_cost": total_cost,
                "std_dev_cost": std_dev_cost,
                "confidence_interval": {
                    "lower": confidence_interval[0],
                    "upper": confidence_interval[1]
                }
            }

        return analysis

    def compare_variants(
        self,
        results: Dict[str, List[ABTestResult]],
        baseline_variant_id: str,
        metric: str = "success_rate"
    ) -> Dict[str, Dict]:
        """
        Compare all variants against a baseline.

        Args:
            results: Dict mapping variant_id to list of results
            baseline_variant_id: ID of baseline variant to compare against
            metric: Metric to compare (success_rate, avg_cost, avg_latency)

        Returns:
            Dict with comparison data for each variant:
            - absolute_diff: Absolute difference from baseline
            - relative_diff: Percentage difference from baseline
            - is_better: Whether this variant is better than baseline
        """
        analysis = self.analyze_results(results)

        if baseline_variant_id not in analysis:
            raise ValueError(f"Baseline variant '{baseline_variant_id}' not found in results")

        baseline_value = analysis[baseline_variant_id].get(metric, 0)

        # Determine if higher is better
        higher_is_better = metric in ["success_rate", "avg_accuracy"]

        comparisons = {}
        for variant_id, variant_analysis in analysis.items():
            if variant_id == baseline_variant_id:
                comparisons[variant_id] = {
                    "is_baseline": True,
                    "absolute_diff": 0.0,
                    "relative_diff": 0.0,
                    "is_better": False
                }
                continue

            variant_value = variant_analysis.get(metric, 0)
            absolute_diff = variant_value - baseline_value
            relative_diff = (absolute_diff / baseline_value * 100) if baseline_value != 0 else 0

            is_better = (
                (absolute_diff > 0) if higher_is_better else (absolute_diff < 0)
            )

            comparisons[variant_id] = {
                "is_baseline": False,
                "absolute_diff": absolute_diff,
                "relative_diff": relative_diff,
                "is_better": is_better,
                "baseline_value": baseline_value,
                "variant_value": variant_value
            }

        return comparisons


# Convenience function for quick A/B tests
def quick_ab_test(
    test_name: str,
    variant_configs: List[Dict[str, Any]],
    agent_factory: Callable[[Dict], Any],
    task_fn: Callable[[Any], Dict],
    num_trials: int = 100
) -> Dict:
    """
    Run a quick A/B test with minimal setup.

    Args:
        test_name: Name of the test
        variant_configs: List of variant configurations, each with:
                        - id: str
                        - name: str
                        - config: Dict
                        - traffic_percentage: float (optional, defaults to equal split)
        agent_factory: Function to create agent from config
        task_fn: Function to run task on agent
        num_trials: Number of trials to run

    Returns:
        Dict with "results" and "analysis" keys

    Example:
        >>> result = quick_ab_test(
        ...     test_name="temperature_test",
        ...     variant_configs=[
        ...         {"id": "low", "name": "Low Temp", "config": {"temperature": 0.3}},
        ...         {"id": "high", "name": "High Temp", "config": {"temperature": 0.9}}
        ...     ],
        ...     agent_factory=create_agent,
        ...     task_fn=run_task,
        ...     num_trials=50
        ... )
    """
    # Create variants with equal traffic if not specified
    num_variants = len(variant_configs)
    equal_traffic = 100.0 / num_variants

    variants = []
    for config in variant_configs:
        variants.append(Variant(
            id=config["id"],
            name=config.get("name", config["id"]),
            variant_type=VariantType.CONFIG,
            config=config.get("config", {}),
            traffic_percentage=config.get("traffic_percentage", equal_traffic)
        ))

    # Run test
    ab_test = ABTestFramework(test_name=test_name, variants=variants)
    results = ab_test.run_test(
        agent_factory=agent_factory,
        task_fn=task_fn,
        num_trials=num_trials
    )
    analysis = ab_test.analyze_results(results)

    return {
        "results": results,
        "analysis": analysis,
        "test_name": test_name
    }


# Demo
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("=" * 80)
    print("A/B Testing Framework Demo - Phase 3")
    print("=" * 80)
    print()

    # Mock agent and task for demo
    class MockAgent:
        def __init__(self, config):
            self.config = config
            self.temperature = config.get("temperature", 0.7)

    def create_agent(config):
        return MockAgent(config)

    def run_task(agent):
        # Simulate task with temperature-dependent success
        success_prob = 0.7 if agent.temperature < 0.5 else 0.8
        success = random.random() < success_prob

        return {
            "success": success,
            "metrics": {
                "accuracy": random.uniform(0.7, 0.95),
                "cost": random.uniform(0.01, 0.05),
                "latency": random.uniform(100, 500)
            }
        }

    # Define variants
    print("[1/3] Defining variants...")
    variants = [
        Variant(
            id="low_temp",
            name="Low Temperature (0.3)",
            variant_type=VariantType.TEMPERATURE,
            config={"temperature": 0.3},
            traffic_percentage=50.0
        ),
        Variant(
            id="high_temp",
            name="High Temperature (0.9)",
            variant_type=VariantType.TEMPERATURE,
            config={"temperature": 0.9},
            traffic_percentage=50.0
        )
    ]
    print(f"   Created {len(variants)} variants")
    print()

    # Create A/B test
    print("[2/3] Running A/B test...")
    ab_test = ABTestFramework(
        test_name="temperature_optimization_demo",
        variants=variants
    )

    results = ab_test.run_test(
        agent_factory=create_agent,
        task_fn=run_task,
        num_trials=50
    )
    print()

    # Analyze results
    print("[3/3] Analyzing results...")
    analysis = ab_test.analyze_results(results)

    print()
    print("=" * 80)
    print("A/B Test Results:")
    print("=" * 80)
    for variant_id, metrics in analysis.items():
        print(f"\nVariant: {variant_id}")
        print(f"  Trials: {metrics['num_trials']}")
        print(f"  Success Rate: {metrics['success_rate']:.2%}")
        print(f"  Avg Cost: ${metrics['avg_cost']:.4f}")
        print(f"  Avg Latency: {metrics['avg_latency']:.0f}ms")
        print(f"  95% CI: [{metrics['confidence_interval']['lower']:.2%}, "
              f"{metrics['confidence_interval']['upper']:.2%}]")

    # Compare variants
    print()
    print("=" * 80)
    print("Variant Comparison (vs low_temp baseline):")
    print("=" * 80)
    comparisons = ab_test.compare_variants(results, baseline_variant_id="low_temp")
    for variant_id, comparison in comparisons.items():
        if comparison.get("is_baseline"):
            print(f"\n{variant_id}: BASELINE")
        else:
            print(f"\n{variant_id}:")
            print(f"  Success Rate Diff: {comparison['relative_diff']:+.1f}%")
            print(f"  Better than baseline: {'YES' if comparison['is_better'] else 'NO'}")

    print()
    print("=" * 80)
    print("A/B Testing Framework Demo Complete!")
    print("=" * 80)
