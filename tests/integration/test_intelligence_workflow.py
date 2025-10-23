"""
Integration tests for Intelligence Agent workflow

Tests Intelligence Agent with real ShortTermMemory integration.
Uses actual LLM calls to verify end-to-end functionality.
"""

import pytest
import json
import tempfile
from pathlib import Path
from fractal_agent.agents.intelligence_agent import IntelligenceAgent, IntelligenceResult
from fractal_agent.agents.intelligence_config import IntelligenceConfig, PresetIntelligenceConfigs
from fractal_agent.memory.short_term import ShortTermMemory


@pytest.mark.integration
class TestIntelligenceWorkflow:
    """Integration tests for Intelligence Agent workflow"""

    @pytest.fixture
    def temp_log_dir(self):
        """Create temporary log directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def memory_with_tasks(self, temp_log_dir):
        """Create ShortTermMemory with sample tasks"""
        memory = ShortTermMemory(log_dir=temp_log_dir)

        # Task 1: Completed successfully
        task1 = memory.start_task(
            agent_id="research_agent_001",
            agent_type="operational",
            task_description="Research quantum computing basics",
            inputs={"topic": "quantum computing"}
        )
        memory.end_task(
            task_id=task1,
            outputs={"report": "Quantum computing uses qubits..."},
            metadata={
                "cost": 0.05,
                "tokens": 2500,
                "cache_hit": True,
                "model": "claude-sonnet-4.5"
            }
        )

        # Task 2: Failed (context limit)
        task2 = memory.start_task(
            agent_id="research_agent_002",
            agent_type="operational",
            task_description="Analyze large dataset",
            inputs={"dataset_size": "1GB"}
        )
        memory.tasks[task2]["status"] = "failed"
        memory.tasks[task2]["duration_seconds"] = 8.1
        memory.tasks[task2]["metadata"] = {
            "cost": 0.02,
            "tokens": 1200,
            "cache_hit": False,
            "error": "Context limit exceeded",
            "model": "claude-sonnet-4.5"
        }

        # Task 3: Completed successfully (expensive)
        task3 = memory.start_task(
            agent_id="control_agent_001",
            agent_type="control",
            task_description="Coordinate multi-agent research",
            inputs={"subtasks": 5}
        )
        memory.end_task(
            task_id=task3,
            outputs={"coordination_result": "All subtasks completed"},
            metadata={
                "cost": 0.12,
                "tokens": 4200,
                "cache_hit": False,
                "model": "claude-sonnet-4.5"
            }
        )

        return memory

    def test_intelligence_agent_with_real_memory(self, memory_with_tasks):
        """Test Intelligence Agent with real ShortTermMemory"""
        # Get performance metrics
        metrics = memory_with_tasks.get_performance_metrics()

        # Verify metrics are correct
        assert metrics["accuracy"] == pytest.approx(2/3)  # 2 out of 3 succeeded
        assert metrics["total_tasks"] == 3
        assert metrics["num_failed"] == 1
        assert len(metrics["failed_tasks"]) == 1

        # Format session logs for Intelligence Agent
        session_logs = json.dumps({
            "session_id": memory_with_tasks.session_id,
            "tasks": list(memory_with_tasks.tasks.values())
        }, indent=2)

        # Create Intelligence Agent with quick analysis config (for speed)
        config = PresetIntelligenceConfigs.quick_analysis()
        agent = IntelligenceAgent(config=config)

        # Check if analysis should trigger
        should_trigger, reason = agent.should_trigger_analysis(
            performance_metrics=metrics,
            session_size=3,
            last_analysis_days_ago=8
        )

        # Should trigger due to high failure rate (33%) or scheduled analysis
        assert should_trigger is True

        # Run intelligence reflection
        result = agent(
            session_logs=session_logs,
            performance_metrics=metrics,
            session_id=memory_with_tasks.session_id,
            verbose=True
        )

        # Verify result structure
        assert isinstance(result, IntelligenceResult)
        assert result.session_id == memory_with_tasks.session_id
        assert len(result.analysis) > 0
        assert len(result.patterns) > 0
        assert len(result.insights) > 0
        assert len(result.action_plan) > 0

        # Verify metadata
        assert "timestamp" in result.metadata
        assert "tiers" in result.metadata
        assert result.metadata["tiers"]["analysis"] == "balanced"

        # Result should be JSON serializable
        result_dict = result.to_dict()
        assert "session_id" in result_dict
        assert "analysis" in result_dict

    def test_trigger_on_high_failure_rate(self, memory_with_tasks):
        """Test Intelligence Agent triggers on high failure rate"""
        # Add more failed tasks to increase failure rate
        for i in range(3):
            task_id = memory_with_tasks.start_task(
                f"agent_{i}",
                "operational",
                f"Failed task {i}",
                {}
            )
            memory_with_tasks.tasks[task_id]["status"] = "failed"
            memory_with_tasks.tasks[task_id]["duration_seconds"] = 5.0
            memory_with_tasks.tasks[task_id]["metadata"] = {"cost": 0.01}

        # Now we have 4 failed, 2 succeeded = 33% success rate
        metrics = memory_with_tasks.get_performance_metrics()
        assert metrics["num_failed"] == 4
        assert metrics["num_completed"] == 2

        config = IntelligenceConfig()
        agent = IntelligenceAgent(config=config)

        should_trigger, reason = agent.should_trigger_analysis(
            performance_metrics=metrics,
            session_size=6
        )

        # Should trigger on high failure rate
        assert should_trigger is True
        assert "High failure rate" in reason or "failure" in reason.lower()

    def test_trigger_on_cost_spike(self, memory_with_tasks):
        """Test Intelligence Agent triggers on cost spike"""
        # Add a very expensive task
        expensive_task = memory_with_tasks.start_task(
            "expensive_agent",
            "operational",
            "Very expensive task",
            {}
        )
        memory_with_tasks.end_task(
            expensive_task,
            outputs={},
            metadata={"cost": 5.00, "tokens": 100000}  # 5x more expensive than average
        )

        metrics = memory_with_tasks.get_performance_metrics()

        # Current total cost: 0.05 + 0.02 + 0.12 + 5.00 = 5.19
        # Average cost: 5.19 / 4 = 1.2975

        config = IntelligenceConfig(
            cost_spike_threshold=2.0,
            analyze_on_cost_spike=True,
            analyze_on_failure=False,  # Disable failure trigger to isolate cost spike test
            analyze_on_schedule=False,  # Disable schedule trigger
            min_session_size=1  # Allow small sessions to test cost spike
        )
        agent = IntelligenceAgent(config=config)

        # Simulate cost spike: current cost (5.00) > avg_cost (1.5) * threshold (2.0) = 3.0
        spike_metrics = {
            "accuracy": 0.75,
            "cost": 5.00,  # Current high cost
            "avg_cost": 1.5,  # Historical average: 5.00 > 1.5 * 2.0 = 3.0, triggers spike!
            "latency": 10.0,
            "cache_hit_rate": 0.5,
            "failed_tasks": [],
            "total_tasks": 4,
            "total_tokens": 100000,
            "num_completed": 3,
            "num_failed": 1
        }

        should_trigger, reason = agent.should_trigger_analysis(
            performance_metrics=spike_metrics,
            session_size=4,
            last_analysis_days_ago=1
        )

        # Should trigger on cost spike
        assert should_trigger is True
        assert "Cost spike" in reason

    def test_no_trigger_on_good_performance(self):
        """Test Intelligence Agent doesn't trigger on good performance"""
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = ShortTermMemory(log_dir=tmpdir)

            # Create only successful tasks
            for i in range(10):
                task_id = memory.start_task(
                    f"agent_{i}",
                    "operational",
                    f"Task {i}",
                    {}
                )
                memory.end_task(
                    task_id,
                    outputs={"result": "success"},
                    metadata={"cost": 0.05, "tokens": 1000, "cache_hit": True}
                )

            metrics = memory.get_performance_metrics()

            # All tasks succeeded, no cost spike
            assert metrics["accuracy"] == 1.0
            assert metrics["num_failed"] == 0

            config = IntelligenceConfig(
                analyze_on_failure=True,
                analyze_on_cost_spike=True,
                analyze_on_schedule=True,
                cost_spike_threshold=1000.0,  # Very high threshold to prevent false triggers
                lookback_days=7  # Explicitly set to match test expectation
            )
            agent = IntelligenceAgent(config=config)

            should_trigger, reason = agent.should_trigger_analysis(
                performance_metrics=metrics,
                session_size=10,
                last_analysis_days_ago=3  # Recent analysis (< 7 day lookback)
            )

            # Should not trigger because:
            # - Accuracy is 100% (no failures, requires < 50%)
            # - Cost spike threshold is very high (no spike detection)
            # - Last analysis was 3 days ago (< 7 day lookback, no scheduled trigger)
            assert should_trigger is False
            assert "No trigger conditions met" in reason

    @pytest.mark.llm
    def test_full_intelligence_workflow_end_to_end(self, memory_with_tasks):
        """
        Full end-to-end test of Intelligence workflow.

        This test uses real LLM calls and validates the complete workflow
        from ShortTermMemory → metrics → Intelligence Agent → insights.
        """
        # Step 1: Get performance metrics from memory
        metrics = memory_with_tasks.get_performance_metrics()
        session_logs = json.dumps({
            "session_id": memory_with_tasks.session_id,
            "tasks": list(memory_with_tasks.tasks.values())
        }, indent=2)

        # Step 2: Create Intelligence Agent
        config = PresetIntelligenceConfigs.quick_analysis()
        agent = IntelligenceAgent(config=config)

        # Step 3: Run intelligence reflection
        result = agent(
            session_logs=session_logs,
            performance_metrics=metrics,
            verbose=True
        )

        # Step 4: Validate result contains actionable insights
        assert isinstance(result, IntelligenceResult)

        # Analysis should mention the failure
        assert len(result.analysis) > 100  # Substantial analysis

        # Patterns should identify the context limit issue
        assert len(result.patterns) > 50

        # Insights should be actionable
        assert len(result.insights) > 50

        # Action plan should prioritize recommendations
        assert len(result.action_plan) > 50

        # Step 5: Verify result is human-readable
        result_str = str(result)
        assert "INTELLIGENCE REPORT" in result_str
        assert "System 4" in result_str
        assert "PERFORMANCE ANALYSIS" in result_str
        assert "ACTION PLAN" in result_str

        # Step 6: Verify result can be saved
        result_dict = result.to_dict()
        result_json = json.dumps(result_dict, indent=2)
        assert len(result_json) > 0

        # Should be able to parse it back
        parsed = json.loads(result_json)
        assert parsed["session_id"] == result.session_id


@pytest.mark.integration
class TestIntelligenceMemoryIntegration:
    """Test Intelligence Agent integration with ShortTermMemory API"""

    def test_memory_metrics_format_compatible_with_agent(self):
        """Test that ShortTermMemory metrics format works with Intelligence Agent"""
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = ShortTermMemory(log_dir=tmpdir)

            # Add sample tasks
            task1 = memory.start_task("agent1", "operational", "Task 1", {})
            memory.end_task(task1, outputs={}, metadata={"cost": 0.05, "tokens": 1000})

            task2 = memory.start_task("agent2", "operational", "Task 2", {})
            memory.tasks[task2]["status"] = "failed"
            memory.tasks[task2]["metadata"] = {"cost": 0.02}

            # Get metrics
            metrics = memory.get_performance_metrics()

            # Verify all required keys are present
            required_keys = [
                "accuracy", "cost", "latency", "cache_hit_rate",
                "failed_tasks", "avg_cost", "total_tasks", "total_tokens"
            ]
            for key in required_keys:
                assert key in metrics, f"Missing required key: {key}"

            # Metrics should be valid for Intelligence Agent
            config = IntelligenceConfig()
            agent = IntelligenceAgent(config=config)

            # This should not raise an error
            should_trigger, reason = agent.should_trigger_analysis(
                performance_metrics=metrics,
                session_size=2
            )

            assert isinstance(should_trigger, bool)
            assert isinstance(reason, str)

    def test_session_logs_format_compatible_with_agent(self):
        """Test that ShortTermMemory session format works with Intelligence Agent"""
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = ShortTermMemory(log_dir=tmpdir)

            # Add tasks
            task_id = memory.start_task("agent", "operational", "Task", {})
            memory.end_task(task_id, outputs={"result": "success"}, metadata={})

            # Format session logs
            session_logs = json.dumps({
                "session_id": memory.session_id,
                "tasks": list(memory.tasks.values())
            })

            # Should be valid JSON
            parsed = json.loads(session_logs)
            assert "session_id" in parsed
            assert "tasks" in parsed
            assert len(parsed["tasks"]) == 1

            # Should have all required task fields
            task = parsed["tasks"][0]
            required_task_fields = [
                "task_id", "agent_id", "agent_type", "task_description",
                "status", "timestamp_start"
            ]
            for field in required_task_fields:
                assert field in task, f"Missing task field: {field}"
