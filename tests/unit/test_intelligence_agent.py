"""
Unit tests for Intelligence Agent (System 4)

Tests intelligence configuration, agent initialization, and core functionality.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from fractal_agent.agents.intelligence_agent import (
    IntelligenceAgent,
    IntelligenceResult,
    PerformanceAnalysis,
    PatternDetection,
    InsightGeneration,
    RecommendationPrioritization
)
from fractal_agent.agents.intelligence_config import (
    IntelligenceConfig,
    PresetIntelligenceConfigs
)


class TestIntelligenceConfig:
    """Test IntelligenceConfig"""

    def test_config_initialization_default(self):
        """Test default configuration initialization"""
        config = IntelligenceConfig()

        assert config.analysis_tier == "expensive"
        assert config.pattern_tier == "expensive"
        assert config.insight_tier == "expensive"
        assert config.prioritization_tier == "balanced"
        assert config.min_session_size == 5
        assert config.lookback_days == 7
        assert config.insight_threshold == 0.7
        assert config.analyze_on_failure is True
        assert config.analyze_on_schedule is True
        assert config.analyze_on_cost_spike is True
        assert config.cost_spike_threshold == 2.0
        assert config.max_recommendations == 5
        assert config.include_examples is True
        assert config.verbose is False
        assert config.max_tokens is None
        assert config.temperature == 0.7

    def test_config_initialization_custom(self):
        """Test custom configuration initialization"""
        config = IntelligenceConfig(
            analysis_tier="premium",
            pattern_tier="balanced",
            min_session_size=10,
            lookback_days=30,
            max_recommendations=10,
            verbose=True
        )

        assert config.analysis_tier == "premium"
        assert config.pattern_tier == "balanced"
        assert config.min_session_size == 10
        assert config.lookback_days == 30
        assert config.max_recommendations == 10
        assert config.verbose is True

    def test_config_string_representation(self):
        """Test configuration string representation"""
        config = IntelligenceConfig()
        config_str = str(config)

        assert "IntelligenceConfig" in config_str
        assert "analysis=expensive" in config_str
        assert "pattern=expensive" in config_str
        assert "insight=expensive" in config_str
        assert "prioritization=balanced" in config_str
        assert "min_session_size=5" in config_str
        assert "lookback_days=7" in config_str

    def test_preset_default(self):
        """Test default preset configuration"""
        config = PresetIntelligenceConfigs.default()

        assert config.analysis_tier == "expensive"
        assert config.min_session_size == 5
        assert config.lookback_days == 7

    def test_preset_quick_analysis(self):
        """Test quick analysis preset"""
        config = PresetIntelligenceConfigs.quick_analysis()

        assert config.analysis_tier == "balanced"
        assert config.pattern_tier == "balanced"
        assert config.insight_tier == "balanced"
        assert config.prioritization_tier == "cheap"
        assert config.min_session_size == 3
        assert config.lookback_days == 1
        assert config.max_recommendations == 3

    def test_preset_deep_analysis(self):
        """Test deep analysis preset"""
        config = PresetIntelligenceConfigs.deep_analysis()

        assert config.analysis_tier == "premium"
        assert config.pattern_tier == "expensive"
        assert config.min_session_size == 20
        assert config.lookback_days == 30
        assert config.max_recommendations == 10
        assert config.include_examples is True
        assert config.verbose is True

    def test_preset_failure_analysis(self):
        """Test failure analysis preset"""
        config = PresetIntelligenceConfigs.failure_analysis()

        assert config.analysis_tier == "expensive"
        assert config.pattern_tier == "expensive"
        assert config.min_session_size == 1  # Analyze even single failures
        assert config.lookback_days == 1
        assert config.analyze_on_failure is True
        assert config.analyze_on_schedule is False

    def test_preset_cost_optimization(self):
        """Test cost optimization preset"""
        config = PresetIntelligenceConfigs.cost_optimization()

        assert config.analyze_on_cost_spike is True
        assert config.cost_spike_threshold == 1.5
        assert config.max_recommendations == 5


class TestIntelligenceResult:
    """Test IntelligenceResult dataclass"""

    @pytest.fixture
    def sample_result(self):
        """Create sample intelligence result"""
        return IntelligenceResult(
            session_id="test_session_001",
            analysis="Test analysis text",
            patterns="Test patterns text",
            insights="Test insights text",
            action_plan="Test action plan",
            metadata={
                "timestamp": "2025-10-19T00:00:00",
                "config": "test_config",
                "tiers": {
                    "analysis": "expensive",
                    "pattern": "expensive",
                    "insight": "expensive",
                    "prioritization": "balanced"
                }
            }
        )

    def test_result_initialization(self, sample_result):
        """Test result initialization"""
        assert sample_result.session_id == "test_session_001"
        assert sample_result.analysis == "Test analysis text"
        assert sample_result.patterns == "Test patterns text"
        assert sample_result.insights == "Test insights text"
        assert sample_result.action_plan == "Test action plan"
        assert "timestamp" in sample_result.metadata

    def test_result_to_dict(self, sample_result):
        """Test result to_dict serialization"""
        result_dict = sample_result.to_dict()

        assert result_dict["session_id"] == "test_session_001"
        assert result_dict["analysis"] == "Test analysis text"
        assert result_dict["patterns"] == "Test patterns text"
        assert result_dict["insights"] == "Test insights text"
        assert result_dict["action_plan"] == "Test action plan"
        assert "metadata" in result_dict

    def test_result_string_representation(self, sample_result):
        """Test result string representation"""
        result_str = str(sample_result)

        assert "INTELLIGENCE REPORT" in result_str
        assert "System 4" in result_str
        assert "test_session_001" in result_str
        assert "PERFORMANCE ANALYSIS" in result_str
        assert "DETECTED PATTERNS" in result_str
        assert "INSIGHTS" in result_str
        assert "ACTION PLAN" in result_str


class TestIntelligenceAgent:
    """Test IntelligenceAgent"""

    @pytest.fixture
    def mock_lm(self):
        """Create mock language model"""
        mock = Mock()
        mock.model = "test-model"
        mock.calls = 0
        mock.tokens_used = 0
        return mock

    def test_agent_initialization_default(self):
        """Test agent initialization with default config"""
        with patch('fractal_agent.agents.intelligence_agent.FractalDSpyLM'):
            with patch('fractal_agent.agents.intelligence_agent.dspy.configure'):
                agent = IntelligenceAgent()

                assert agent.config.analysis_tier == "expensive"
                assert agent.config.pattern_tier == "expensive"
                assert agent.config.insight_tier == "expensive"
                assert agent.config.prioritization_tier == "balanced"

    def test_agent_initialization_custom_config(self):
        """Test agent initialization with custom config"""
        config = IntelligenceConfig(
            analysis_tier="premium",
            min_session_size=10
        )

        with patch('fractal_agent.agents.intelligence_agent.FractalDSpyLM'):
            with patch('fractal_agent.agents.intelligence_agent.dspy.configure'):
                agent = IntelligenceAgent(config=config)

                assert agent.config.analysis_tier == "premium"
                assert agent.config.min_session_size == 10

    def test_agent_has_required_attributes(self):
        """Test agent has all required DSPy modules"""
        with patch('fractal_agent.agents.intelligence_agent.FractalDSpyLM'):
            with patch('fractal_agent.agents.intelligence_agent.dspy.configure'):
                with patch('fractal_agent.agents.intelligence_agent.dspy.ChainOfThought'):
                    agent = IntelligenceAgent()

                    assert hasattr(agent, 'config')
                    assert hasattr(agent, 'analysis_lm')
                    assert hasattr(agent, 'pattern_lm')
                    assert hasattr(agent, 'insight_lm')
                    assert hasattr(agent, 'prioritization_lm')
                    assert hasattr(agent, 'analyzer')
                    assert hasattr(agent, 'pattern_detector')
                    assert hasattr(agent, 'insight_generator')
                    assert hasattr(agent, 'prioritizer')

    def test_should_trigger_analysis_session_too_small(self):
        """Test trigger analysis - session too small"""
        with patch('fractal_agent.agents.intelligence_agent.FractalDSpyLM'):
            with patch('fractal_agent.agents.intelligence_agent.dspy.configure'):
                agent = IntelligenceAgent()

                metrics = {"accuracy": 0.9, "cost": 10.0}
                should_trigger, reason = agent.should_trigger_analysis(
                    performance_metrics=metrics,
                    session_size=3,  # Less than min_session_size (5)
                    last_analysis_days_ago=1
                )

                assert should_trigger is False
                assert "Session too small" in reason

    def test_should_trigger_analysis_high_failure_rate(self):
        """Test trigger analysis - high failure rate"""
        with patch('fractal_agent.agents.intelligence_agent.FractalDSpyLM'):
            with patch('fractal_agent.agents.intelligence_agent.dspy.configure'):
                agent = IntelligenceAgent()

                metrics = {"accuracy": 0.4, "cost": 10.0}  # 40% success = high failure
                should_trigger, reason = agent.should_trigger_analysis(
                    performance_metrics=metrics,
                    session_size=10,
                    last_analysis_days_ago=1
                )

                assert should_trigger is True
                assert "High failure rate" in reason

    def test_should_trigger_analysis_cost_spike(self):
        """Test trigger analysis - cost spike"""
        with patch('fractal_agent.agents.intelligence_agent.FractalDSpyLM'):
            with patch('fractal_agent.agents.intelligence_agent.dspy.configure'):
                agent = IntelligenceAgent()

                metrics = {
                    "accuracy": 0.9,
                    "cost": 100.0,
                    "avg_cost": 10.0  # Current cost is 10x average
                }
                should_trigger, reason = agent.should_trigger_analysis(
                    performance_metrics=metrics,
                    session_size=10,
                    last_analysis_days_ago=1
                )

                assert should_trigger is True
                assert "Cost spike detected" in reason

    def test_should_trigger_analysis_scheduled(self):
        """Test trigger analysis - scheduled"""
        with patch('fractal_agent.agents.intelligence_agent.FractalDSpyLM'):
            with patch('fractal_agent.agents.intelligence_agent.dspy.configure'):
                agent = IntelligenceAgent()

                metrics = {"accuracy": 0.9, "cost": 10.0}
                should_trigger, reason = agent.should_trigger_analysis(
                    performance_metrics=metrics,
                    session_size=10,
                    last_analysis_days_ago=10  # More than lookback_days (7)
                )

                assert should_trigger is True
                assert "Scheduled analysis due" in reason

    def test_should_trigger_analysis_no_triggers(self):
        """Test trigger analysis - no triggers met"""
        with patch('fractal_agent.agents.intelligence_agent.FractalDSpyLM'):
            with patch('fractal_agent.agents.intelligence_agent.dspy.configure'):
                agent = IntelligenceAgent()

                metrics = {"accuracy": 0.9, "cost": 10.0, "avg_cost": 10.0}
                should_trigger, reason = agent.should_trigger_analysis(
                    performance_metrics=metrics,
                    session_size=10,
                    last_analysis_days_ago=3  # Less than lookback_days
                )

                assert should_trigger is False
                assert "No trigger conditions met" in reason

    @pytest.mark.skip(reason="Complex mocking - forward() tested via integration tests")
    def test_forward_method(self):
        """Test forward method with mocked LLM responses"""
        # This would require complex mocking of all 4 DSPy stages
        # Better suited for integration tests
        pass

    def test_dspy_signatures_defined(self):
        """Test that all DSPy signatures are properly defined"""
        import dspy

        # Test that signatures are dspy.Signature subclasses
        assert issubclass(PerformanceAnalysis, dspy.Signature)
        assert issubclass(PatternDetection, dspy.Signature)
        assert issubclass(InsightGeneration, dspy.Signature)
        assert issubclass(RecommendationPrioritization, dspy.Signature)

        # Test signatures have docstrings
        assert PerformanceAnalysis.__doc__ is not None
        assert PatternDetection.__doc__ is not None
        assert InsightGeneration.__doc__ is not None
        assert RecommendationPrioritization.__doc__ is not None


class TestIntelligenceAgentIntegration:
    """Integration-style tests (but still unit tests with mocks)"""

    def test_sample_data_format(self):
        """Test that sample data formats are correct"""
        # Sample session logs
        sample_logs = json.dumps({
            "session_id": "test_001",
            "tasks": [
                {
                    "task_id": "task_001",
                    "status": "completed",
                    "duration_seconds": 10.0,
                    "metadata": {"cost": 0.05}
                }
            ]
        })

        # Should be valid JSON
        parsed = json.loads(sample_logs)
        assert "session_id" in parsed
        assert "tasks" in parsed
        assert len(parsed["tasks"]) == 1

        # Sample metrics
        sample_metrics = {
            "accuracy": 0.85,
            "cost": 12.50,
            "latency": 2.3,
            "cache_hit_rate": 0.75
        }

        # Should have all required keys
        assert "accuracy" in sample_metrics
        assert "cost" in sample_metrics
        assert "latency" in sample_metrics
        assert "cache_hit_rate" in sample_metrics

    def test_config_with_optional_parameters(self):
        """Test config with optional max_tokens and temperature"""
        config = IntelligenceConfig(
            max_tokens=4000,
            temperature=0.5
        )

        assert config.max_tokens == 4000
        assert config.temperature == 0.5

    def test_agent_with_max_tokens_config(self):
        """Test agent initialization with max_tokens in config"""
        config = IntelligenceConfig(max_tokens=4000)

        with patch('fractal_agent.agents.intelligence_agent.FractalDSpyLM') as MockLM:
            with patch('fractal_agent.agents.intelligence_agent.dspy.configure'):
                agent = IntelligenceAgent(config=config)

                # Verify LMs were created with max_tokens
                # Check that FractalDSpyLM was called with max_tokens
                call_kwargs = MockLM.call_args_list[0][1]
                assert "max_tokens" in call_kwargs
                assert call_kwargs["max_tokens"] == 4000
