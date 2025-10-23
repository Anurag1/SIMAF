"""
Unit tests for workflow modules

Tests workflow node functions, graph building, and state management.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from fractal_agent.workflows.research_workflow import (
    WorkflowState,
    research_node,
    analyze_node,
    report_node,
    create_research_workflow,
    run_research_workflow
)
from fractal_agent.agents.research_agent import ResearchResult


class TestWorkflowState:
    """Test WorkflowState TypedDict"""

    def test_workflow_state_structure(self):
        """Test WorkflowState can be created with expected keys"""
        state: WorkflowState = {
            "topic": "Test topic",
            "research_result": None,
            "analysis": None,
            "report": None
        }

        assert state["topic"] == "Test topic"
        assert state["research_result"] is None
        assert state["analysis"] is None
        assert state["report"] is None

    def test_workflow_state_with_data(self):
        """Test WorkflowState with populated data"""
        mock_result = Mock(spec=ResearchResult)
        mock_result.topic = "VSM"
        mock_result.synthesis = "Test synthesis"

        state: WorkflowState = {
            "topic": "VSM",
            "research_result": mock_result,
            "analysis": "Test analysis",
            "report": "Test report"
        }

        assert state["topic"] == "VSM"
        assert state["research_result"] is not None
        assert state["analysis"] == "Test analysis"
        assert state["report"] == "Test report"


class TestResearchNode:
    """Test research_node function"""

    @patch('fractal_agent.workflows.research_workflow.ResearchAgent')
    def test_research_node_creates_agent(self, mock_agent_class):
        """Test research_node creates ResearchAgent"""
        # Setup mock
        mock_result = Mock(spec=ResearchResult)
        mock_agent = Mock()
        mock_agent.return_value = mock_result
        mock_agent_class.return_value = mock_agent

        # Create state
        state: WorkflowState = {
            "topic": "Test topic",
            "research_result": None,
            "analysis": None,
            "report": None
        }

        # Execute
        result_state = research_node(state)

        # Verify agent was created
        mock_agent_class.assert_called_once()
        call_kwargs = mock_agent_class.call_args[1]
        assert "config" in call_kwargs
        assert "max_research_questions" in call_kwargs
        assert call_kwargs["max_research_questions"] == 2

    @patch('fractal_agent.workflows.research_workflow.ResearchAgent')
    def test_research_node_calls_agent(self, mock_agent_class):
        """Test research_node calls agent with topic"""
        # Setup mock
        mock_result = Mock(spec=ResearchResult)
        mock_agent = Mock()
        mock_agent.return_value = mock_result
        mock_agent_class.return_value = mock_agent

        # Create state
        state: WorkflowState = {
            "topic": "VSM Study",
            "research_result": None,
            "analysis": None,
            "report": None
        }

        # Execute
        result_state = research_node(state)

        # Verify agent was called with correct topic
        mock_agent.assert_called_once()
        call_kwargs = mock_agent.call_args[1]
        assert call_kwargs["topic"] == "VSM Study"
        assert call_kwargs["verbose"] is True

    @patch('fractal_agent.workflows.research_workflow.ResearchAgent')
    def test_research_node_updates_state(self, mock_agent_class):
        """Test research_node updates state with result"""
        # Setup mock
        mock_result = Mock(spec=ResearchResult)
        mock_result.topic = "Test"
        mock_agent = Mock()
        mock_agent.return_value = mock_result
        mock_agent_class.return_value = mock_agent

        # Create state
        state: WorkflowState = {
            "topic": "Test",
            "research_result": None,
            "analysis": None,
            "report": None
        }

        # Execute
        result_state = research_node(state)

        # Verify state was updated
        assert result_state["research_result"] == mock_result
        assert result_state["topic"] == "Test"


class TestAnalyzeNode:
    """Test analyze_node function"""

    def test_analyze_node_processes_research_result(self):
        """Test analyze_node creates analysis from research result"""
        # Create mock research result
        mock_result = Mock(spec=ResearchResult)
        mock_result.topic = "VSM"
        mock_result.synthesis = "Systems thinking framework"
        mock_result.validation = "Complete and accurate"
        mock_result.metadata = {
            "total_tokens": 1500,
            "num_questions": 3
        }

        # Create state
        state: WorkflowState = {
            "topic": "VSM",
            "research_result": mock_result,
            "analysis": None,
            "report": None
        }

        # Execute
        result_state = analyze_node(state)

        # Verify analysis was created
        assert result_state["analysis"] is not None
        analysis = result_state["analysis"]
        assert "VSM" in analysis
        assert "Systems thinking framework" in analysis
        assert "Complete and accurate" in analysis
        assert "1500" in analysis
        assert "3" in analysis

    def test_analyze_node_includes_metadata(self):
        """Test analyze_node includes metadata in analysis"""
        mock_result = Mock(spec=ResearchResult)
        mock_result.topic = "Test"
        mock_result.synthesis = "Test synthesis"
        mock_result.validation = "Valid"
        mock_result.metadata = {
            "total_tokens": 500,
            "num_questions": 2
        }

        state: WorkflowState = {
            "topic": "Test",
            "research_result": mock_result,
            "analysis": None,
            "report": None
        }

        result_state = analyze_node(state)

        analysis = result_state["analysis"]
        assert "500" in analysis  # total_tokens
        assert "2" in analysis    # num_questions

    def test_analyze_node_preserves_other_state(self):
        """Test analyze_node preserves other state fields"""
        mock_result = Mock(spec=ResearchResult)
        mock_result.topic = "Test"
        mock_result.synthesis = "Synthesis"
        mock_result.validation = "Valid"
        mock_result.metadata = {"total_tokens": 100, "num_questions": 1}

        state: WorkflowState = {
            "topic": "Test Topic",
            "research_result": mock_result,
            "analysis": None,
            "report": None
        }

        result_state = analyze_node(state)

        # Other fields should be preserved
        assert result_state["topic"] == "Test Topic"
        assert result_state["research_result"] == mock_result


class TestReportNode:
    """Test report_node function"""

    def test_report_node_generates_report(self):
        """Test report_node creates final report"""
        # Create mock data
        mock_result = Mock(spec=ResearchResult)
        mock_result.topic = "VSM"
        mock_result.research_plan = "Study history and applications"
        mock_result.findings = [
            {"question": "What is VSM?", "answer": "A systems model"},
            {"question": "Who created it?", "answer": "Stafford Beer"}
        ]

        state: WorkflowState = {
            "topic": "VSM",
            "research_result": mock_result,
            "analysis": "# Analysis\n\nKey insights about VSM",
            "report": None
        }

        # Execute
        result_state = report_node(state)

        # Verify report was created
        assert result_state["report"] is not None
        report = result_state["report"]
        assert "RESEARCH WORKFLOW REPORT" in report
        assert "VSM" in report
        assert "Key insights about VSM" in report
        assert "Study history and applications" in report
        assert "What is VSM?" in report
        assert "A systems model" in report

    def test_report_node_includes_all_findings(self):
        """Test report_node includes all research findings"""
        mock_result = Mock(spec=ResearchResult)
        mock_result.topic = "Test"
        mock_result.research_plan = "Plan"
        mock_result.findings = [
            {"question": "Q1", "answer": "A1"},
            {"question": "Q2", "answer": "A2"},
            {"question": "Q3", "answer": "A3"}
        ]

        state: WorkflowState = {
            "topic": "Test",
            "research_result": mock_result,
            "analysis": "Analysis",
            "report": None
        }

        result_state = report_node(state)
        report = result_state["report"]

        # All findings should be in report
        assert "Q1" in report
        assert "A1" in report
        assert "Q2" in report
        assert "A2" in report
        assert "Q3" in report
        assert "A3" in report

    def test_report_node_formats_correctly(self):
        """Test report_node uses correct formatting"""
        mock_result = Mock(spec=ResearchResult)
        mock_result.topic = "Test"
        mock_result.research_plan = "Plan"
        mock_result.findings = [{"question": "Q", "answer": "A"}]

        state: WorkflowState = {
            "topic": "Test",
            "research_result": mock_result,
            "analysis": "Analysis",
            "report": None
        }

        result_state = report_node(state)
        report = result_state["report"]

        # Check for section headers
        assert "RESEARCH WORKFLOW REPORT" in report
        assert "RESEARCH PLAN" in report
        assert "DETAILED FINDINGS" in report
        assert "WORKFLOW COMPLETE" in report


class TestCreateResearchWorkflow:
    """Test create_research_workflow function"""

    @patch('fractal_agent.workflows.research_workflow.StateGraph')
    def test_create_research_workflow_creates_graph(self, mock_state_graph_class):
        """Test create_research_workflow creates StateGraph"""
        # Setup mock
        mock_graph = Mock()
        mock_graph.compile.return_value = Mock()
        mock_state_graph_class.return_value = mock_graph

        # Execute
        result = create_research_workflow()

        # Verify StateGraph was created
        mock_state_graph_class.assert_called_once()

    @patch('fractal_agent.workflows.research_workflow.StateGraph')
    def test_create_research_workflow_adds_nodes(self, mock_state_graph_class):
        """Test create_research_workflow adds all nodes"""
        # Setup mock
        mock_graph = Mock()
        mock_graph.compile.return_value = Mock()
        mock_state_graph_class.return_value = mock_graph

        # Execute
        create_research_workflow()

        # Verify nodes were added
        assert mock_graph.add_node.call_count == 3

        # Check node names
        call_args_list = [call[0] for call in mock_graph.add_node.call_args_list]
        node_names = [args[0] for args in call_args_list]
        assert "research" in node_names
        assert "analyze" in node_names
        assert "report" in node_names

    @patch('fractal_agent.workflows.research_workflow.StateGraph')
    def test_create_research_workflow_sets_entry_point(self, mock_state_graph_class):
        """Test create_research_workflow sets entry point"""
        # Setup mock
        mock_graph = Mock()
        mock_graph.compile.return_value = Mock()
        mock_state_graph_class.return_value = mock_graph

        # Execute
        create_research_workflow()

        # Verify entry point was set
        mock_graph.set_entry_point.assert_called_once_with("research")

    @patch('fractal_agent.workflows.research_workflow.StateGraph')
    def test_create_research_workflow_adds_edges(self, mock_state_graph_class):
        """Test create_research_workflow creates edge flow"""
        # Setup mock
        mock_graph = Mock()
        mock_graph.compile.return_value = Mock()
        mock_state_graph_class.return_value = mock_graph

        # Execute
        create_research_workflow()

        # Verify edges were added
        assert mock_graph.add_edge.call_count == 3

    @patch('fractal_agent.workflows.research_workflow.StateGraph')
    def test_create_research_workflow_compiles_graph(self, mock_state_graph_class):
        """Test create_research_workflow compiles and returns graph"""
        # Setup mock
        mock_graph = Mock()
        mock_compiled = Mock()
        mock_graph.compile.return_value = mock_compiled
        mock_state_graph_class.return_value = mock_graph

        # Execute
        result = create_research_workflow()

        # Verify compile was called and result returned
        mock_graph.compile.assert_called_once()
        assert result == mock_compiled


class TestRunResearchWorkflow:
    """Test run_research_workflow function"""

    @patch('fractal_agent.workflows.research_workflow.create_research_workflow')
    def test_run_research_workflow_creates_workflow(self, mock_create):
        """Test run_research_workflow creates workflow"""
        # Setup mock
        mock_app = Mock()
        mock_app.invoke.return_value = {"report": "Test"}
        mock_create.return_value = mock_app

        # Execute
        run_research_workflow("Test topic")

        # Verify workflow was created
        mock_create.assert_called_once()

    @patch('fractal_agent.workflows.research_workflow.create_research_workflow')
    def test_run_research_workflow_invokes_with_topic(self, mock_create):
        """Test run_research_workflow invokes with correct initial state"""
        # Setup mock
        mock_app = Mock()
        mock_app.invoke.return_value = {"report": "Test"}
        mock_create.return_value = mock_app

        # Execute
        run_research_workflow("VSM Study")

        # Verify invoke was called with correct state
        mock_app.invoke.assert_called_once()
        call_args = mock_app.invoke.call_args[0][0]
        assert call_args["topic"] == "VSM Study"
        assert call_args["research_result"] is None
        assert call_args["analysis"] is None
        assert call_args["report"] is None

    @patch('fractal_agent.workflows.research_workflow.create_research_workflow')
    def test_run_research_workflow_returns_final_state(self, mock_create):
        """Test run_research_workflow returns final state"""
        # Setup mock
        mock_final_state = {
            "topic": "Test",
            "research_result": Mock(),
            "analysis": "Analysis",
            "report": "Report"
        }
        mock_app = Mock()
        mock_app.invoke.return_value = mock_final_state
        mock_create.return_value = mock_app

        # Execute
        result = run_research_workflow("Test")

        # Verify final state was returned
        assert result == mock_final_state
        assert result["topic"] == "Test"
        assert result["report"] == "Report"
