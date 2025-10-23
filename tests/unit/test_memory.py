"""
Unit tests for Short-Term Memory

Tests task logging, session management, and task tree operations.
"""

import pytest
import json
import tempfile
from pathlib import Path
from fractal_agent.memory.short_term import ShortTermMemory


class TestShortTermMemory:
    """Test short-term memory operations"""

    @pytest.fixture
    def temp_log_dir(self):
        """Create temporary log directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def memory(self, temp_log_dir):
        """Create ShortTermMemory instance"""
        return ShortTermMemory(log_dir=temp_log_dir)

    def test_memory_initialization(self, memory, temp_log_dir):
        """Test memory system initialization"""
        assert memory.log_dir == Path(temp_log_dir)
        assert memory.retention_days == 30
        assert memory.session_id.startswith("session_")
        assert len(memory.tasks) == 0

    def test_start_task(self, memory):
        """Test starting a new task"""
        task_id = memory.start_task(
            agent_id="test_agent_001",
            agent_type="operational",
            task_description="Test task",
            inputs={"param": "value"}
        )

        assert task_id.startswith("task_")
        assert task_id in memory.tasks

        task = memory.tasks[task_id]
        assert task["agent_id"] == "test_agent_001"
        assert task["agent_type"] == "operational"
        assert task["task_description"] == "Test task"
        assert task["inputs"] == {"param": "value"}
        assert task["status"] == "in_progress"
        assert task["parent_task_id"] is None

    def test_start_task_with_parent(self, memory):
        """Test starting a subtask with parent"""
        parent_id = memory.start_task(
            agent_id="parent_agent",
            agent_type="control",
            task_description="Parent task",
            inputs={}
        )

        child_id = memory.start_task(
            agent_id="child_agent",
            agent_type="operational",
            task_description="Child task",
            inputs={},
            parent_task_id=parent_id
        )

        child_task = memory.tasks[child_id]
        assert child_task["parent_task_id"] == parent_id

    def test_end_task(self, memory):
        """Test ending a task"""
        task_id = memory.start_task(
            agent_id="test_agent",
            agent_type="operational",
            task_description="Test task",
            inputs={}
        )

        memory.end_task(
            task_id=task_id,
            outputs={"result": "success"},
            metadata={"tokens": 1000}
        )

        task = memory.tasks[task_id]
        assert task["status"] == "completed"
        assert task["outputs"] == {"result": "success"}
        assert task["metadata"] == {"tokens": 1000}
        assert task["duration_seconds"] is not None
        assert task["duration_seconds"] > 0

    def test_get_task(self, memory):
        """Test retrieving a task"""
        task_id = memory.start_task(
            agent_id="test_agent",
            agent_type="operational",
            task_description="Test task",
            inputs={}
        )

        retrieved_task = memory.get_task(task_id)
        assert retrieved_task is not None
        assert retrieved_task["task_id"] == task_id

        # Test non-existent task
        assert memory.get_task("nonexistent") is None

    def test_get_task_tree(self, memory):
        """Test task tree retrieval"""
        # Create parent task
        parent_id = memory.start_task(
            agent_id="control_agent",
            agent_type="control",
            task_description="Parent task",
            inputs={}
        )

        # Create child tasks
        child1_id = memory.start_task(
            agent_id="operational_1",
            agent_type="operational",
            task_description="Child 1",
            inputs={},
            parent_task_id=parent_id
        )

        child2_id = memory.start_task(
            agent_id="operational_2",
            agent_type="operational",
            task_description="Child 2",
            inputs={},
            parent_task_id=parent_id
        )

        # Get task tree
        tree = memory.get_task_tree(parent_id)

        assert len(tree) == 3  # Parent + 2 children
        assert tree[0]["task_id"] == parent_id
        assert tree[1]["task_id"] == child1_id
        assert tree[2]["task_id"] == child2_id

    def test_save_and_load_session(self, memory, temp_log_dir):
        """Test session persistence"""
        # Create some tasks
        task_id = memory.start_task(
            agent_id="test_agent",
            agent_type="operational",
            task_description="Test task",
            inputs={"param": "value"}
        )

        memory.end_task(
            task_id=task_id,
            outputs={"result": "success"},
            metadata={"tokens": 1000}
        )

        # Save session
        memory.save_session()

        # Verify file exists
        assert memory.session_file.exists()

        # Load session in new memory instance
        new_memory = ShortTermMemory(log_dir=temp_log_dir)
        new_memory.load_session(str(memory.session_file))

        # Verify loaded data
        assert len(new_memory.tasks) == 1
        assert task_id in new_memory.tasks
        loaded_task = new_memory.tasks[task_id]
        assert loaded_task["task_description"] == "Test task"
        assert loaded_task["status"] == "completed"

    def test_get_session_summary(self, memory):
        """Test session summary statistics"""
        # Create tasks
        task1 = memory.start_task(
            agent_id="agent1",
            agent_type="operational",
            task_description="Task 1",
            inputs={}
        )

        task2 = memory.start_task(
            agent_id="agent2",
            agent_type="operational",
            task_description="Task 2",
            inputs={}
        )

        # Complete one task
        memory.end_task(task1, outputs={}, metadata={})

        # Get summary
        summary = memory.get_session_summary()

        assert summary["session_id"] == memory.session_id
        assert summary["total_tasks"] == 2
        assert summary["completed_tasks"] == 1
        assert summary["in_progress_tasks"] == 1
        assert "avg_duration_seconds" in summary
        assert "total_duration_seconds" in summary

    def test_session_file_format(self, memory, temp_log_dir):
        """Test session file JSON structure"""
        task_id = memory.start_task(
            agent_id="test_agent",
            agent_type="operational",
            task_description="Test task",
            inputs={"test": "data"}
        )

        memory.end_task(
            task_id=task_id,
            outputs={"result": "complete"},
            metadata={"tokens": 500}
        )

        memory.save_session()

        # Load and verify JSON structure
        with open(memory.session_file, 'r') as f:
            session_data = json.load(f)

        assert "session_id" in session_data
        assert "timestamp" in session_data
        assert "num_tasks" in session_data
        assert "tasks" in session_data
        assert isinstance(session_data["tasks"], list)
        assert len(session_data["tasks"]) == 1

    def test_memory_initialization_with_default_log_dir(self):
        """Test memory initialization with default log_dir (None)"""
        memory = ShortTermMemory(log_dir=None)

        assert memory.log_dir == Path("./logs/sessions")
        assert memory.log_dir.exists()

    def test_end_task_with_nonexistent_task_id(self, memory):
        """Test end_task with task_id that doesn't exist"""
        # This should not raise an error, just log and return
        memory.end_task(
            task_id="nonexistent_task_id",
            outputs={"result": "test"},
            metadata={}
        )

        # Task should not be in memory
        assert "nonexistent_task_id" not in memory.tasks

    def test_load_session_nonexistent_file(self, memory):
        """Test load_session with file that doesn't exist"""
        # This should not raise an error, just log and return
        memory.load_session("/path/to/nonexistent/file.json")

        # Memory should still be empty
        assert len(memory.tasks) == 0

    def test_cleanup_old_sessions(self, temp_log_dir):
        """Test cleanup of old session files"""
        import time
        from datetime import datetime, timedelta

        # Use retention_days=1 so we don't accidentally delete the current session
        memory = ShortTermMemory(log_dir=temp_log_dir, retention_days=1)

        # Create an "old" session file manually (2 days old)
        old_timestamp = (datetime.now() - timedelta(days=2)).strftime("%Y%m%d_%H%M%S")
        old_session_file = Path(temp_log_dir) / f"session_{old_timestamp}.json"
        old_session_file.write_text(json.dumps({
            "session_id": f"session_{old_timestamp}",
            "timestamp": old_timestamp,
            "num_tasks": 0,
            "tasks": []
        }))

        # Create current session file
        current_session_file = memory.session_file
        memory.start_task("agent", "operational", "Test task", {})
        memory.save_session()

        # Verify both files exist
        assert old_session_file.exists()
        assert current_session_file.exists()

        # Cleanup old sessions
        memory.cleanup_old_sessions()

        # Old file should be deleted (older than 1 day)
        assert not old_session_file.exists()
        # Current session should still exist (created just now)
        assert current_session_file.exists()

    def test_cleanup_old_sessions_handles_invalid_filenames(self, temp_log_dir):
        """Test cleanup_old_sessions handles invalid filename formats gracefully"""
        memory = ShortTermMemory(log_dir=temp_log_dir, retention_days=7)

        # Create a file with invalid name format
        invalid_file = Path(temp_log_dir) / "session_invalid_format.json"
        invalid_file.write_text("{}")

        # This should not raise an error
        memory.cleanup_old_sessions()

        # Invalid file should still exist (not deleted)
        assert invalid_file.exists()

    def test_get_performance_metrics_empty_session(self, memory):
        """Test get_performance_metrics with no tasks"""
        metrics = memory.get_performance_metrics()

        assert metrics["accuracy"] == 1.0
        assert metrics["cost"] == 0.0
        assert metrics["latency"] == 0.0
        assert metrics["cache_hit_rate"] == 0.0
        assert metrics["failed_tasks"] == []
        assert metrics["avg_cost"] == 0.0
        assert metrics["total_tasks"] == 0
        assert metrics["total_tokens"] == 0

    def test_get_performance_metrics_single_completed_task(self, memory):
        """Test get_performance_metrics with one completed task"""
        task_id = memory.start_task(
            agent_id="test_agent",
            agent_type="operational",
            task_description="Test task",
            inputs={}
        )

        memory.end_task(
            task_id=task_id,
            outputs={"result": "success"},
            metadata={"cost": 0.05, "tokens": 1000, "cache_hit": True}
        )

        metrics = memory.get_performance_metrics()

        assert metrics["accuracy"] == 1.0  # 100% success
        assert metrics["cost"] == 0.05
        assert metrics["total_tokens"] == 1000
        assert metrics["cache_hit_rate"] == 1.0  # 100% cache hit
        assert len(metrics["failed_tasks"]) == 0
        assert metrics["avg_cost"] == 0.05
        assert metrics["total_tasks"] == 1
        assert metrics["num_completed"] == 1
        assert metrics["num_failed"] == 0

    def test_get_performance_metrics_mixed_tasks(self, memory):
        """Test get_performance_metrics with completed and failed tasks"""
        # Task 1: Completed
        task1 = memory.start_task("agent1", "operational", "Task 1", {})
        memory.end_task(
            task1,
            outputs={"result": "success"},
            metadata={"cost": 0.05, "tokens_used": 1000, "cache_hit": True}
        )

        # Task 2: Failed
        task2 = memory.start_task("agent2", "operational", "Task 2", {})
        memory.tasks[task2]["status"] = "failed"
        memory.tasks[task2]["duration_seconds"] = 5.0
        memory.tasks[task2]["metadata"] = {"cost": 0.02, "tokens": 500, "cache_hit": False}

        # Task 3: Completed
        task3 = memory.start_task("agent3", "operational", "Task 3", {})
        memory.end_task(
            task3,
            outputs={"result": "success"},
            metadata={"cost": 0.08, "tokens": 1500, "cache_hit": False}
        )

        metrics = memory.get_performance_metrics()

        assert metrics["accuracy"] == pytest.approx(2/3)  # 2 out of 3 succeeded
        assert metrics["cost"] == pytest.approx(0.15)  # 0.05 + 0.02 + 0.08
        assert metrics["total_tokens"] == 3000  # 1000 + 500 + 1500
        assert metrics["cache_hit_rate"] == pytest.approx(1/3)  # 1 hit out of 3
        assert len(metrics["failed_tasks"]) == 1
        assert task2 in metrics["failed_tasks"]
        assert metrics["avg_cost"] == pytest.approx(0.05)  # 0.15 / 3
        assert metrics["total_tasks"] == 3
        assert metrics["num_completed"] == 2
        assert metrics["num_failed"] == 1

    def test_get_performance_metrics_latency_calculation(self, memory):
        """Test latency calculation in performance metrics"""
        # Create tasks with different durations
        task1 = memory.start_task("agent1", "operational", "Fast task", {})
        memory.end_task(task1, outputs={}, metadata={})
        memory.tasks[task1]["duration_seconds"] = 5.0

        task2 = memory.start_task("agent2", "operational", "Slow task", {})
        memory.end_task(task2, outputs={}, metadata={})
        memory.tasks[task2]["duration_seconds"] = 15.0

        metrics = memory.get_performance_metrics()

        # Average latency should be (5 + 15) / 2 = 10.0
        assert metrics["latency"] == 10.0

    def test_get_performance_metrics_no_cache_data(self, memory):
        """Test metrics when cache_hit not tracked"""
        task_id = memory.start_task("agent", "operational", "Task", {})
        memory.end_task(
            task_id,
            outputs={},
            metadata={"cost": 0.05, "tokens": 1000}  # No cache_hit
        )

        metrics = memory.get_performance_metrics()

        # Cache hit rate should be 0 when no cache data tracked
        assert metrics["cache_hit_rate"] == 0.0

    def test_get_performance_metrics_tokens_vs_tokens_used(self, memory):
        """Test metrics handle both 'tokens' and 'tokens_used' keys"""
        # Task 1 with 'tokens'
        task1 = memory.start_task("agent1", "operational", "Task 1", {})
        memory.end_task(task1, outputs={}, metadata={"tokens": 1000})

        # Task 2 with 'tokens_used'
        task2 = memory.start_task("agent2", "operational", "Task 2", {})
        memory.end_task(task2, outputs={}, metadata={"tokens_used": 2000})

        metrics = memory.get_performance_metrics()

        # Should aggregate both
        assert metrics["total_tokens"] == 3000
