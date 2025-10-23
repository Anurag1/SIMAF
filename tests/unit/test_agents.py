"""
Unit tests for agent modules

Tests agent initialization and basic functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from fractal_agent.agents.control_agent import ControlAgent, ControlResult, TaskDecomposition, SynthesisCoordination
from fractal_agent.agents.research_agent import ResearchAgent, ResearchResult
from fractal_agent.agents.research_config import ResearchConfig


class TestControlResult:
    """Test ControlResult dataclass"""

    def test_control_result_creation(self):
        """Test creating ControlResult"""
        result = ControlResult(
            main_task="Research VSM",
            subtasks=["Task 1", "Task 2", "Task 3"],
            subtask_results=[{"result": "Result 1"}, {"result": "Result 2"}],
            final_report="Complete research on VSM",
            metadata={"tokens": 500}
        )

        assert result.main_task == "Research VSM"
        assert len(result.subtasks) == 3
        assert len(result.subtask_results) == 2
        assert result.final_report == "Complete research on VSM"
        assert result.metadata["tokens"] == 500

    def test_control_result_str_representation(self):
        """Test ControlResult string representation"""
        result = ControlResult(
            main_task="Research Topic",
            subtasks=["Subtask A", "Subtask B"],
            subtask_results=[{"data": "result1"}],
            final_report="Final synthesis",
            metadata={"duration": 10}
        )

        str_repr = str(result)

        assert "CONTROL AGENT REPORT" in str_repr
        assert "Research Topic" in str_repr
        assert "Subtask A" in str_repr
        assert "Subtask B" in str_repr
        assert "Final synthesis" in str_repr
        assert "duration" in str_repr

    def test_control_result_empty_subtasks(self):
        """Test ControlResult with empty subtasks"""
        result = ControlResult(
            main_task="Task",
            subtasks=[],
            subtask_results=[],
            final_report="Report",
            metadata={}
        )

        assert len(result.subtasks) == 0
        assert len(result.subtask_results) == 0


class TestControlAgent:
    """Test ControlAgent functionality"""

    def test_control_agent_initialization(self):
        """Test control agent can be initialized"""
        with patch('fractal_agent.agents.control_agent.configure_dspy'):
            agent = ControlAgent(tier="balanced")

            assert agent is not None
            assert hasattr(agent, 'decomposer')
            assert hasattr(agent, 'synthesizer')

    def test_control_agent_with_different_tiers(self):
        """Test control agent with different tier configurations"""
        tiers = ["cheap", "balanced", "expensive"]

        with patch('fractal_agent.agents.control_agent.configure_dspy'):
            for tier in tiers:
                agent = ControlAgent(tier=tier)
                assert agent is not None

    def test_control_agent_has_required_attributes(self):
        """Test control agent has required attributes"""
        with patch('fractal_agent.agents.control_agent.configure_dspy'):
            agent = ControlAgent(tier="balanced")

            required_attrs = ['decomposer', 'synthesizer']
            for attr in required_attrs:
                assert hasattr(agent, attr)

    def test_control_agent_decomposer_type(self):
        """Test decomposer is correct DSPy module type"""
        with patch('fractal_agent.agents.control_agent.configure_dspy'):
            agent = ControlAgent(tier="balanced")

            # Decomposer should be a DSPy ChainOfThought module
            assert agent.decomposer is not None
            assert hasattr(agent.decomposer, '__call__')

    def test_control_agent_synthesizer_type(self):
        """Test synthesizer is correct DSPy module type"""
        with patch('fractal_agent.agents.control_agent.configure_dspy'):
            agent = ControlAgent(tier="balanced")

            # Synthesizer should be a DSPy ChainOfThought module
            assert agent.synthesizer is not None
            assert hasattr(agent.synthesizer, '__call__')

    @pytest.mark.skip(reason="Complex mocking of DSPy internals - covered by integration tests")
    def test_control_agent_forward_with_mocked_lm(self):
        """Test control agent forward method with mocked LM"""
        pass

    @pytest.mark.skip(reason="Complex mocking of DSPy internals - covered by integration tests")
    def test_control_agent_parse_subtasks(self):
        """Test subtask parsing from decomposer output"""
        pass


class TestResearchAgent:
    """Test ResearchAgent functionality"""

    def test_research_agent_initialization(self):
        """Test research agent can be initialized"""
        with patch('fractal_agent.agents.research_agent.configure_dspy'):
            config = ResearchConfig()
            agent = ResearchAgent(config=config)

            assert agent is not None
            assert hasattr(agent, 'config')

    def test_research_agent_with_custom_config(self):
        """Test research agent with custom configuration"""
        with patch('fractal_agent.agents.research_agent.configure_dspy'):
            config = ResearchConfig(
                planning_tier="expensive",
                research_tier="cheap",
                synthesis_tier="balanced",
                validation_tier="cheap"
            )
            agent = ResearchAgent(config=config)

            assert agent.config == config
            assert agent.config.planning_tier == "expensive"

    def test_research_agent_has_required_components(self):
        """Test research agent has required components"""
        with patch('fractal_agent.agents.research_agent.configure_dspy'):
            config = ResearchConfig()
            agent = ResearchAgent(config=config)

            # Verify agent has planning, research, synthesis, validation capabilities
            assert hasattr(agent, 'config')
            assert agent.config.planning_tier is not None
            assert agent.config.research_tier is not None
            assert agent.config.synthesis_tier is not None
            assert agent.config.validation_tier is not None

    def test_research_agent_with_default_config(self):
        """Test research agent with default config"""
        with patch('fractal_agent.agents.research_agent.configure_dspy'):
            agent = ResearchAgent()

            assert agent is not None
            assert hasattr(agent, 'config')

    @patch('fractal_agent.agents.research_agent.configure_dspy')
    def test_research_agent_has_dspy_modules(self, mock_configure):
        """Test research agent initializes DSPy modules"""
        mock_lm = Mock()
        mock_configure.return_value = mock_lm

        agent = ResearchAgent()

        # Should have research modules
        assert hasattr(agent, 'planner')
        assert hasattr(agent, 'gatherer')  # It's 'gatherer', not 'researcher'
        assert hasattr(agent, 'synthesizer')
        assert hasattr(agent, 'validator')

    @pytest.mark.skip(reason="Complex DSPy mocking - forward() execution covered by integration tests")
    def test_research_agent_forward_method(self):
        """Test research agent forward method"""
        # This test requires complex mocking of DSPy internals and dspy.configure()
        # The actual forward() execution is tested in integration tests
        pass

    @patch('fractal_agent.agents.research_agent.FractalDSpyLM')
    def test_research_agent_config_tiers_applied(self, mock_dspy_lm):
        """Test that config tiers are applied to LM instances"""
        # Track what tiers are used
        tier_calls = []

        def track_tier(tier, **kwargs):
            tier_calls.append(tier)
            return Mock()

        mock_dspy_lm.side_effect = track_tier

        config = ResearchConfig(
            planning_tier="expensive",
            research_tier="balanced",
            synthesis_tier="balanced",
            validation_tier="cheap"
        )

        agent = ResearchAgent(config=config)

        # Verify different tiers were used to create LMs
        assert "expensive" in tier_calls
        assert "balanced" in tier_calls
        assert "cheap" in tier_calls
        # Should have 4 LM instances (planning, research, synthesis, validation)
        assert len(tier_calls) == 4


class TestResearchResult:
    """Test ResearchResult dataclass"""

    def test_research_result_creation(self):
        """Test creating ResearchResult"""
        result = ResearchResult(
            topic="VSM",
            research_plan="Plan to research VSM",
            findings=[
                {"question": "What is VSM?", "answer": "A systems model"},
                {"question": "Who created it?", "answer": "Stafford Beer"}
            ],
            synthesis="VSM is a powerful framework",
            validation="Complete and accurate",
            metadata={"total_tokens": 1500, "num_questions": 2}
        )

        assert result.topic == "VSM"
        assert result.research_plan == "Plan to research VSM"
        assert len(result.findings) == 2
        assert result.synthesis == "VSM is a powerful framework"
        assert result.validation == "Complete and accurate"
        assert result.metadata["total_tokens"] == 1500

    def test_research_result_str_representation(self):
        """Test ResearchResult __str__ method"""
        result = ResearchResult(
            topic="Test Topic",
            research_plan="Research plan details",
            findings=[
                {"question": "Question 1", "answer": "This is a long answer that needs to be truncated because it exceeds 200 characters limit and we want to test that the truncation works correctly in the string representation method of the ResearchResult dataclass when displaying findings"},
                {"question": "Question 2", "answer": "Short answer"}
            ],
            synthesis="Synthesis of findings",
            validation="Valid research",
            metadata={"total_tokens": 500}
        )

        str_repr = str(result)

        # Check for section headers
        assert "RESEARCH REPORT" in str_repr
        assert "Test Topic" in str_repr
        assert "RESEARCH PLAN:" in str_repr
        assert "Research plan details" in str_repr
        assert "FINDINGS:" in str_repr
        assert "Question 1" in str_repr
        assert "Question 2" in str_repr
        assert "SYNTHESIS:" in str_repr
        assert "Synthesis of findings" in str_repr
        assert "VALIDATION:" in str_repr
        assert "Valid research" in str_repr
        assert "Metadata:" in str_repr
        assert "500" in str_repr

    def test_research_result_str_truncates_long_answers(self):
        """Test that __str__ truncates long answers to 200 chars"""
        long_answer = "A" * 300  # 300 character answer
        result = ResearchResult(
            topic="Test",
            research_plan="Plan",
            findings=[{"question": "Q", "answer": long_answer}],
            synthesis="Synth",
            validation="Valid",
            metadata={}
        )

        str_repr = str(result)

        # Answer should be truncated with ...
        assert "..." in str_repr
        # Full answer should not appear
        assert long_answer not in str_repr

    def test_research_result_with_empty_findings(self):
        """Test ResearchResult with no findings"""
        result = ResearchResult(
            topic="Test",
            research_plan="Plan",
            findings=[],
            synthesis="Synth",
            validation="Valid",
            metadata={}
        )

        str_repr = str(result)

        assert "FINDINGS:" in str_repr
        assert len(result.findings) == 0

    def test_research_result_with_multiple_findings(self):
        """Test ResearchResult with many findings"""
        findings = [
            {"question": f"Q{i}", "answer": f"A{i}"}
            for i in range(10)
        ]

        result = ResearchResult(
            topic="Multi-finding test",
            research_plan="Plan",
            findings=findings,
            synthesis="Synth",
            validation="Valid",
            metadata={}
        )

        str_repr = str(result)

        # All findings should be enumerated
        for i in range(1, 11):
            assert f"{i}." in str_repr


class TestResearchAgentHelpers:
    """Test ResearchAgent helper methods"""

    @patch('fractal_agent.agents.research_agent.FractalDSpyLM')
    def test_extract_questions_numbered_list(self, mock_lm):
        """Test _extract_questions with numbered list format"""
        # Setup mock
        mock_lm.return_value = Mock()

        agent = ResearchAgent(max_research_questions=3)

        research_plan = """
        1. What are the key principles of VSM?
        2. How does VSM apply to organizations?
        3. What are the limitations of VSM?
        4. Who are the main contributors?
        """

        questions = agent._extract_questions(research_plan)

        # Should extract first 3 questions (max_research_questions=3)
        assert len(questions) == 3
        assert "key principles" in questions[0]
        assert "apply to organizations" in questions[1]
        assert "limitations" in questions[2]

    @patch('fractal_agent.agents.research_agent.FractalDSpyLM')
    def test_extract_questions_bullet_points(self, mock_lm):
        """Test _extract_questions with bullet point format"""
        mock_lm.return_value = Mock()

        agent = ResearchAgent(max_research_questions=2)

        research_plan = """
        - What is the history of VSM?
        - How is VSM used in practice?
        - What are the benefits?
        """

        questions = agent._extract_questions(research_plan)

        assert len(questions) == 2
        assert "history" in questions[0]
        assert "used in practice" in questions[1]

    @patch('fractal_agent.agents.research_agent.FractalDSpyLM')
    def test_extract_questions_mixed_format(self, mock_lm):
        """Test _extract_questions with mixed bullet formats"""
        mock_lm.return_value = Mock()

        agent = ResearchAgent(max_research_questions=4)

        research_plan = """
        * First question with asterisk?
        • Second question with bullet?
        - Third question with dash?
        1. Fourth question numbered?
        5. Fifth question also numbered?
        """

        questions = agent._extract_questions(research_plan)

        # Should extract up to max_research_questions
        assert len(questions) <= 4
        assert len(questions) >= 3  # At least some questions found

    @patch('fractal_agent.agents.research_agent.FractalDSpyLM')
    def test_extract_questions_fallback_generic(self, mock_lm):
        """Test _extract_questions fallback to generic questions"""
        mock_lm.return_value = Mock()

        agent = ResearchAgent(max_research_questions=3)

        # Plan with no clear question structure
        research_plan = "This is a general research plan without specific questions."

        questions = agent._extract_questions(research_plan)

        # Should return generic fallback questions
        assert len(questions) == 3
        assert "key concepts" in questions[0].lower()
        assert "applications" in questions[1].lower() or "use cases" in questions[1].lower()
        assert "benefits" in questions[2].lower() or "challenges" in questions[2].lower()

    @patch('fractal_agent.agents.research_agent.FractalDSpyLM')
    def test_extract_questions_filters_short_lines(self, mock_lm):
        """Test that _extract_questions filters out very short lines"""
        mock_lm.return_value = Mock()

        agent = ResearchAgent(max_research_questions=5)

        research_plan = """
        1. Short
        2. This is a proper research question about the topic?
        3. Ok
        4. Another good question that is long enough to be valid?
        """

        questions = agent._extract_questions(research_plan)

        # Should only include questions longer than 10 characters
        for question in questions:
            assert len(question) > 10

    @patch('fractal_agent.agents.research_agent.FractalDSpyLM')
    def test_extract_questions_respects_max_limit(self, mock_lm):
        """Test that _extract_questions respects max_research_questions limit"""
        mock_lm.return_value = Mock()

        agent = ResearchAgent(max_research_questions=2)

        research_plan = """
        1. Question one that is long enough to be included?
        2. Question two that is long enough to be included?
        3. Question three that is long enough to be included?
        4. Question four that is long enough to be included?
        5. Question five that is long enough to be included?
        """

        questions = agent._extract_questions(research_plan)

        # Should return exactly max_research_questions
        assert len(questions) == 2
