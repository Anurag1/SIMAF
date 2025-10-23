"""
Unit tests for A/B Testing Framework

Tests variant selection, test execution, results analysis,
and statistical significance calculations.
"""

import pytest
import random
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from fractal_agent.testing.ab_testing import (
    ABTestFramework,
    Variant,
    VariantType,
    ABTestResult,
    quick_ab_test
)


class TestVariant:
    """Test Variant dataclass"""

    def test_variant_creation(self):
        """Test creating a valid variant"""
        variant = Variant(
            id="test_variant",
            name="Test Variant",
            variant_type=VariantType.PROMPT,
            config={"prompt": "Test prompt"},
            traffic_percentage=50.0
        )

        assert variant.id == "test_variant"
        assert variant.name == "Test Variant"
        assert variant.variant_type == VariantType.PROMPT
        assert variant.config == {"prompt": "Test prompt"}
        assert variant.traffic_percentage == 50.0

    def test_variant_type_string_conversion(self):
        """Test variant_type converts from string to enum"""
        variant = Variant(
            id="test",
            name="Test",
            variant_type="prompt",  # String instead of enum
            config={},
            traffic_percentage=100.0
        )

        assert variant.variant_type == VariantType.PROMPT

    def test_variant_invalid_traffic_percentage(self):
        """Test validation of traffic percentage"""
        with pytest.raises(ValueError, match="Traffic percentage must be 0-100"):
            Variant(
                id="test",
                name="Test",
                variant_type=VariantType.PROMPT,
                config={},
                traffic_percentage=150.0  # Invalid: > 100
            )

        with pytest.raises(ValueError, match="Traffic percentage must be 0-100"):
            Variant(
                id="test",
                name="Test",
                variant_type=VariantType.PROMPT,
                config={},
                traffic_percentage=-10.0  # Invalid: < 0
            )


class TestABTestFramework:
    """Test ABTestFramework core functionality"""

    @pytest.fixture
    def temp_results_dir(self):
        """Create temporary directory for test results"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def simple_variants(self):
        """Create simple test variants"""
        return [
            Variant(
                id="control",
                name="Control",
                variant_type=VariantType.CONFIG,
                config={"value": 1},
                traffic_percentage=50.0
            ),
            Variant(
                id="treatment",
                name="Treatment",
                variant_type=VariantType.CONFIG,
                config={"value": 2},
                traffic_percentage=50.0
            )
        ]

    def test_init_valid_variants(self, simple_variants, temp_results_dir):
        """Test initialization with valid variants"""
        ab_test = ABTestFramework(
            test_name="test_ab_test",
            variants=simple_variants,
            results_dir=temp_results_dir
        )

        assert ab_test.test_name == "test_ab_test"
        assert len(ab_test.variants) == 2
        assert ab_test.results_dir == Path(temp_results_dir)

    def test_init_invalid_traffic_percentages(self, temp_results_dir):
        """Test initialization fails with invalid traffic percentages"""
        invalid_variants = [
            Variant(
                id="v1",
                name="V1",
                variant_type=VariantType.CONFIG,
                config={},
                traffic_percentage=40.0  # Total = 90%, not 100%
            ),
            Variant(
                id="v2",
                name="V2",
                variant_type=VariantType.CONFIG,
                config={},
                traffic_percentage=50.0
            )
        ]

        with pytest.raises(ValueError, match="Traffic percentages must sum to 100"):
            ABTestFramework(
                test_name="test",
                variants=invalid_variants,
                results_dir=temp_results_dir
            )

    def test_select_variant_distribution(self, simple_variants, temp_results_dir):
        """Test variant selection follows traffic allocation"""
        ab_test = ABTestFramework(
            test_name="test",
            variants=simple_variants,
            results_dir=temp_results_dir
        )

        # Run many selections and check distribution
        random.seed(42)  # For reproducibility
        selections = [ab_test.select_variant().id for _ in range(1000)]

        control_count = selections.count("control")
        treatment_count = selections.count("treatment")

        # Should be approximately 50/50 (within 10% tolerance)
        assert 400 <= control_count <= 600
        assert 400 <= treatment_count <= 600

    def test_select_variant_unequal_distribution(self, temp_results_dir):
        """Test variant selection with unequal traffic allocation"""
        variants = [
            Variant(
                id="v1",
                name="V1",
                variant_type=VariantType.CONFIG,
                config={},
                traffic_percentage=25.0  # 25%
            ),
            Variant(
                id="v2",
                name="V2",
                variant_type=VariantType.CONFIG,
                config={},
                traffic_percentage=75.0  # 75%
            )
        ]

        ab_test = ABTestFramework(
            test_name="test",
            variants=variants,
            results_dir=temp_results_dir
        )

        random.seed(42)
        selections = [ab_test.select_variant().id for _ in range(1000)]

        v1_count = selections.count("v1")
        v2_count = selections.count("v2")

        # V2 should get ~3x more traffic than V1
        # V1: ~250, V2: ~750 (within tolerance)
        assert 200 <= v1_count <= 300
        assert 700 <= v2_count <= 800

    def test_run_test_basic(self, simple_variants, temp_results_dir):
        """Test basic A/B test execution"""
        ab_test = ABTestFramework(
            test_name="test",
            variants=simple_variants,
            results_dir=temp_results_dir
        )

        # Mock agent and task
        def agent_factory(config):
            return {"value": config.get("value", 0)}

        def task_fn(agent):
            return {
                "success": True,
                "metrics": {
                    "accuracy": 0.8,
                    "cost": 0.01,
                    "latency": 100.0
                }
            }

        # Run test
        results = ab_test.run_test(
            agent_factory=agent_factory,
            task_fn=task_fn,
            num_trials=10
        )

        # Verify results structure
        assert "control" in results
        assert "treatment" in results
        assert len(results["control"]) + len(results["treatment"]) == 10

        # Verify all results are ABTestResult objects
        for variant_id, variant_results in results.items():
            for result in variant_results:
                assert isinstance(result, ABTestResult)
                assert result.variant_id == variant_id
                assert result.success is True
                assert "accuracy" in result.metrics

    def test_run_test_with_failures(self, simple_variants, temp_results_dir):
        """Test A/B test handles task failures gracefully"""
        ab_test = ABTestFramework(
            test_name="test",
            variants=simple_variants,
            results_dir=temp_results_dir
        )

        def agent_factory(config):
            return {"value": config.get("value", 0)}

        # Task that sometimes fails
        call_count = [0]

        def task_fn(agent):
            call_count[0] += 1
            if call_count[0] % 3 == 0:
                raise Exception("Task failure")

            return {
                "success": True,
                "metrics": {"accuracy": 0.8}
            }

        # Run test
        results = ab_test.run_test(
            agent_factory=agent_factory,
            task_fn=task_fn,
            num_trials=10
        )

        # Should have results for all trials (some marked as failures)
        total_results = sum(len(r) for r in results.values())
        assert total_results == 10

        # Check that some failures were recorded
        all_results = [r for results_list in results.values() for r in results_list]
        failures = [r for r in all_results if not r.success]
        assert len(failures) > 0

    def test_save_and_load_results(self, simple_variants, temp_results_dir):
        """Test saving and loading test results"""
        ab_test = ABTestFramework(
            test_name="test_save_load",
            variants=simple_variants,
            results_dir=temp_results_dir
        )

        # Create mock results
        results = {
            "control": [
                ABTestResult(
                    variant_id="control",
                    task_id="task_1",
                    success=True,
                    metrics={"accuracy": 0.8, "cost": 0.01},
                    timestamp="2025-10-19T10:00:00"
                )
            ],
            "treatment": [
                ABTestResult(
                    variant_id="treatment",
                    task_id="task_2",
                    success=True,
                    metrics={"accuracy": 0.9, "cost": 0.02},
                    timestamp="2025-10-19T10:00:01"
                )
            ]
        }

        # Save results
        ab_test.save_results(results)

        # Verify file was created
        results_file = Path(temp_results_dir) / "test_save_load_results.json"
        assert results_file.exists()

        # Load results
        loaded_results = ab_test.load_results()

        # Verify loaded results match original
        assert "control" in loaded_results
        assert "treatment" in loaded_results
        assert len(loaded_results["control"]) == 1
        assert len(loaded_results["treatment"]) == 1

        control_result = loaded_results["control"][0]
        assert control_result.variant_id == "control"
        assert control_result.success is True
        assert control_result.metrics["accuracy"] == 0.8

    def test_analyze_results(self, simple_variants, temp_results_dir):
        """Test results analysis with statistical metrics"""
        ab_test = ABTestFramework(
            test_name="test",
            variants=simple_variants,
            results_dir=temp_results_dir
        )

        # Create mock results
        results = {
            "control": [
                ABTestResult(
                    variant_id="control",
                    task_id=f"task_{i}",
                    success=i % 2 == 0,  # 50% success rate
                    metrics={"accuracy": 0.7, "cost": 0.01, "latency": 100.0},
                    timestamp="2025-10-19T10:00:00"
                )
                for i in range(10)
            ],
            "treatment": [
                ABTestResult(
                    variant_id="treatment",
                    task_id=f"task_{i}",
                    success=i % 3 != 0,  # ~66% success rate
                    metrics={"accuracy": 0.8, "cost": 0.02, "latency": 150.0},
                    timestamp="2025-10-19T10:00:00"
                )
                for i in range(10)
            ]
        }

        # Analyze results
        analysis = ab_test.analyze_results(results)

        # Verify analysis structure
        assert "control" in analysis
        assert "treatment" in analysis

        control_analysis = analysis["control"]
        assert control_analysis["num_trials"] == 10
        assert control_analysis["success_rate"] == 0.5
        assert control_analysis["avg_cost"] == 0.01
        assert control_analysis["avg_latency"] == 100.0
        assert "confidence_interval" in control_analysis

        treatment_analysis = analysis["treatment"]
        assert treatment_analysis["num_trials"] == 10
        assert 0.6 <= treatment_analysis["success_rate"] <= 0.7  # ~66%
        assert treatment_analysis["avg_cost"] == 0.02

    def test_compare_variants(self, simple_variants, temp_results_dir):
        """Test variant comparison against baseline"""
        ab_test = ABTestFramework(
            test_name="test",
            variants=simple_variants,
            results_dir=temp_results_dir
        )

        # Create mock results with clear difference
        results = {
            "control": [
                ABTestResult(
                    variant_id="control",
                    task_id=f"task_{i}",
                    success=True,
                    metrics={"accuracy": 0.7, "cost": 0.01},
                    timestamp="2025-10-19T10:00:00"
                )
                for i in range(10)
            ],
            "treatment": [
                ABTestResult(
                    variant_id="treatment",
                    task_id=f"task_{i}",
                    success=True,
                    metrics={"accuracy": 0.9, "cost": 0.02},  # Better accuracy, higher cost
                    timestamp="2025-10-19T10:00:00"
                )
                for i in range(10)
            ]
        }

        # Compare on success_rate (higher is better)
        comparison = ab_test.compare_variants(
            results=results,
            baseline_variant_id="control",
            metric="success_rate"
        )

        assert "control" in comparison
        assert "treatment" in comparison

        # Control should be baseline
        assert comparison["control"]["is_baseline"] is True
        assert comparison["control"]["absolute_diff"] == 0.0

        # Treatment should show no difference in success rate (both 100%)
        assert comparison["treatment"]["is_baseline"] is False
        assert comparison["treatment"]["absolute_diff"] == 0.0

        # Compare on avg_accuracy (higher is better)
        comparison_acc = ab_test.compare_variants(
            results=results,
            baseline_variant_id="control",
            metric="avg_accuracy"
        )

        # Treatment should be better
        assert comparison_acc["treatment"]["is_better"] is True
        assert comparison_acc["treatment"]["absolute_diff"] > 0

        # Compare on avg_cost (lower is better)
        comparison_cost = ab_test.compare_variants(
            results=results,
            baseline_variant_id="control",
            metric="avg_cost"
        )

        # Treatment should be worse (higher cost)
        assert comparison_cost["treatment"]["is_better"] is False
        assert comparison_cost["treatment"]["absolute_diff"] > 0


class TestQuickABTest:
    """Test quick_ab_test convenience function"""

    def test_quick_ab_test_basic(self):
        """Test quick A/B test with minimal setup"""
        # Mock agent and task
        def agent_factory(config):
            return {"value": config.get("value", 0)}

        def task_fn(agent):
            return {
                "success": True,
                "metrics": {"accuracy": 0.8}
            }

        variant_configs = [
            {"id": "v1", "name": "V1", "config": {"value": 1}},
            {"id": "v2", "name": "V2", "config": {"value": 2}}
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            # Run quick test
            result = quick_ab_test(
                test_name="quick_test",
                variant_configs=variant_configs,
                agent_factory=agent_factory,
                task_fn=task_fn,
                num_trials=20
            )

            # Verify result structure
            assert "results" in result
            assert "analysis" in result
            assert "test_name" in result
            assert result["test_name"] == "quick_test"

            # Verify results
            assert "v1" in result["results"]
            assert "v2" in result["results"]

            # Verify analysis
            assert "v1" in result["analysis"]
            assert "v2" in result["analysis"]

    def test_quick_ab_test_equal_traffic_default(self):
        """Test that quick_ab_test defaults to equal traffic allocation"""
        def agent_factory(config):
            return {}

        def task_fn(agent):
            return {"success": True, "metrics": {}}

        variant_configs = [
            {"id": "v1", "config": {}},
            {"id": "v2", "config": {}},
            {"id": "v3", "config": {}}
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            result = quick_ab_test(
                test_name="equal_traffic_test",
                variant_configs=variant_configs,
                agent_factory=agent_factory,
                task_fn=task_fn,
                num_trials=30
            )

            # Each variant should have gotten approximately 1/3 of traffic
            for variant_id in ["v1", "v2", "v3"]:
                variant_results = result["results"][variant_id]
                # Should be around 10 trials each (30 / 3)
                assert 5 <= len(variant_results) <= 15  # Allow some variance


class TestABTestResultDataclass:
    """Test ABTestResult dataclass"""

    def test_ab_test_result_creation(self):
        """Test creating ABTestResult"""
        result = ABTestResult(
            variant_id="test_variant",
            task_id="task_123",
            success=True,
            metrics={"accuracy": 0.9, "cost": 0.01},
            timestamp="2025-10-19T10:00:00"
        )

        assert result.variant_id == "test_variant"
        assert result.task_id == "task_123"
        assert result.success is True
        assert result.metrics["accuracy"] == 0.9
        assert result.timestamp == "2025-10-19T10:00:00"


class TestStatisticalMetrics:
    """Test statistical calculations in analysis"""

    def test_confidence_interval_calculation(self, temp_results_dir):
        """Test 95% confidence interval calculation"""
        variants = [
            Variant(
                id="test",
                name="Test",
                variant_type=VariantType.CONFIG,
                config={},
                traffic_percentage=100.0
            )
        ]

        ab_test = ABTestFramework(
            test_name="test",
            variants=variants,
            results_dir=temp_results_dir
        )

        # Create results with 80% success rate
        results = {
            "test": [
                ABTestResult(
                    variant_id="test",
                    task_id=f"task_{i}",
                    success=i < 80,  # 80 successes out of 100
                    metrics={},
                    timestamp="2025-10-19T10:00:00"
                )
                for i in range(100)
            ]
        }

        analysis = ab_test.analyze_results(results)

        # Verify confidence interval
        ci = analysis["test"]["confidence_interval"]
        assert "lower" in ci
        assert "upper" in ci

        # For 80% success rate with n=100:
        # CI should be approximately [0.72, 0.88]
        assert 0.70 <= ci["lower"] <= 0.75
        assert 0.85 <= ci["upper"] <= 0.90

        # Success rate should be in the interval
        success_rate = analysis["test"]["success_rate"]
        assert ci["lower"] <= success_rate <= ci["upper"]

    @pytest.fixture
    def temp_results_dir(self):
        """Create temporary directory for test results"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
