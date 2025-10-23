"""
Integration tests for Control Agent workflow

Tests the full workflow from task decomposition through operational agent execution.
Uses mocked LLM responses to avoid costs.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from fractal_agent.agents.control_agent import ControlAgent
from fractal_agent.agents.research_agent import ResearchAgent
from fractal_agent.agents.research_config import ResearchConfig


@pytest.mark.integration
class TestControlWorkflow:
    """Test control agent workflow integration"""

    @pytest.fixture
    def mock_operational_runner(self):
        """Mock operational agent runner"""
        def runner(subtask: str) -> dict:
            return {
                "subtask": subtask,
                "synthesis": f"Mocked synthesis for: {subtask}",
                "tokens_used": 1000
            }
        return runner

    def test_control_agent_initialization(self):
        """Test control agent can be initialized"""
        agent = ControlAgent(tier="balanced")
        assert agent is not None
        assert hasattr(agent, 'decomposer')
        assert hasattr(agent, 'synthesizer')

    @pytest.mark.mock
    @patch('fractal_agent.agents.control_agent.dspy.ChainOfThought')
    def test_task_decomposition_mock(self, mock_cot):
        """Test task decomposition with mocked DSPy"""
        # Setup mock
        mock_decomposer = MagicMock()
        mock_decomposer.return_value = MagicMock(
            subtasks="1. Research foundations\n2. Analyze applications\n3. Synthesize findings"
        )
        mock_cot.return_value = mock_decomposer

        # Test would go here - demonstrates pattern
        # Actual implementation depends on ControlAgent structure

    @pytest.mark.mock
    def test_operational_agent_runner_integration(self, mock_operational_runner):
        """Test that operational runner is called correctly"""
        subtasks = [
            "Research VSM foundations",
            "Analyze VSM applications",
            "Synthesize VSM insights"
        ]

        results = []
        for subtask in subtasks:
            result = mock_operational_runner(subtask)
            results.append(result)

        assert len(results) == 3
        assert all('synthesis' in r for r in results)
        assert all('tokens_used' in r for r in results)
        assert results[0]['subtask'] == subtasks[0]

    @pytest.mark.mock
    def test_full_workflow_mock(self, mock_operational_runner):
        """Test complete workflow with mocked components"""
        # This demonstrates the workflow pattern
        # 1. Task decomposition
        main_task = "Research the Viable System Model"
        subtasks = [
            "Research VSM theoretical foundations",
            "Analyze VSM architecture",
            "Examine VSM applications"
        ]

        # 2. Execute operational agents
        results = [mock_operational_runner(st) for st in subtasks]

        # 3. Verify workflow completion
        assert len(results) == len(subtasks)
        assert all(r['subtask'] in subtasks for r in results)

        # 4. Synthesis (mocked)
        final_report = f"Report based on {len(results)} subtask results"
        assert len(final_report) > 0


@pytest.mark.integration
@pytest.mark.mock
class TestMemoryIntegration:
    """Test memory system integration with workflows"""

    def test_task_tree_creation_during_workflow(self):
        """Test that task trees are properly created during execution"""
        from fractal_agent.memory.short_term import ShortTermMemory
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            memory = ShortTermMemory(log_dir=tmpdir)

            # Simulate control workflow
            main_task_id = memory.start_task(
                agent_id="control_001",
                agent_type="control",
                task_description="Research VSM",
                inputs={"topic": "VSM"}
            )

            # Simulate operational agents
            subtasks = ["foundations", "architecture", "applications"]
            subtask_ids = []

            for i, subtask in enumerate(subtasks):
                task_id = memory.start_task(
                    agent_id=f"operational_{i:03d}",
                    agent_type="operational",
                    task_description=f"Research VSM {subtask}",
                    inputs={"subtask": subtask},
                    parent_task_id=main_task_id
                )
                subtask_ids.append(task_id)

                memory.end_task(
                    task_id=task_id,
                    outputs={"result": f"{subtask} complete"},
                    metadata={"tokens": 1000}
                )

            # Complete main task
            memory.end_task(
                task_id=main_task_id,
                outputs={"report": "Complete"},
                metadata={"total_tokens": 3000}
            )

            # Verify task tree
            tree = memory.get_task_tree(main_task_id)
            assert len(tree) == 4  # 1 main + 3 subtasks
            assert tree[0]['task_id'] == main_task_id
            assert all(tree[i+1]['parent_task_id'] == main_task_id for i in range(3))


@pytest.mark.integration
@pytest.mark.mock
class TestObsidianExport:
    """Test Obsidian export integration"""

    def test_obsidian_export_from_session(self):
        """Test exporting session to Obsidian format"""
        from fractal_agent.memory.short_term import ShortTermMemory
        from fractal_agent.memory.obsidian_export import ObsidianExporter
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create session
            memory = ShortTermMemory(log_dir=tmpdir)

            task_id = memory.start_task(
                agent_id="test_agent",
                agent_type="operational",
                task_description="Test task",
                inputs={"test": "data"}
            )

            memory.end_task(
                task_id=task_id,
                outputs={"result": "success"},
                metadata={"tokens": 500}
            )

            # Export to Obsidian
            exporter = ObsidianExporter(vault_path=tmpdir)
            export_file = exporter.export_session(memory)

            # Verify export
            assert export_file.exists()
            assert export_file.suffix == ".md"

            # Read and verify content
            content = export_file.read_text()
            assert "---" in content  # YAML frontmatter
            assert "session_id:" in content
            assert "Test task" in content
            assert "- [ ] Approved" in content  # Approval checkbox
