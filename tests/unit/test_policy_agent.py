"""
Unit Tests for PolicyAgent (VSM System 5)

Tests ethical boundary detection, strategic guidance, policy validation,
and tier verification capabilities.

Author: BMad
Date: 2025-10-20
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from fractal_agent.agents.policy_agent import (
    PolicyAgent,
    PolicyResult,
    PolicyDecision,
    PolicyEvaluation,
    StrategicAssessment,
    StrategicPriority
)
from fractal_agent.agents.policy_config import (
    PolicyConfig,
    PolicyMode,
    ResourceLimits,
    PresetPolicyConfigs,
    EthicalBoundary
)
from fractal_agent.verification import Goal, TierVerificationResult


# ============================================================================
# PolicyConfig Tests
# ============================================================================

class TestPolicyConfig:
    """Test PolicyConfig and ResourceLimits"""

    def test_default_config(self):
        """Test default configuration"""
        config = PolicyConfig()

        assert config.mode == PolicyMode.STRICT
        assert config.ethical_tier == "premium"
        assert config.confidence_threshold == 0.8
        assert len(config.prohibited_topics) > 0

    def test_resource_limits_validation(self):
        """Test resource limits validation"""
        # Valid limits
        limits = ResourceLimits(
            max_cost_per_task=10.0,
            max_duration_minutes=30,
            max_tokens_per_task=100000,
            max_llm_calls_per_task=50
        )
        assert limits.max_cost_per_task == 10.0

        # Invalid cost
        with pytest.raises(ValueError, match="max_cost_per_task must be > 0"):
            ResourceLimits(max_cost_per_task=-1)

        # Invalid duration
        with pytest.raises(ValueError, match="max_duration_minutes must be > 0"):
            ResourceLimits(max_duration_minutes=0)

    def test_confidence_threshold_validation(self):
        """Test confidence threshold validation"""
        # Valid threshold
        config = PolicyConfig(confidence_threshold=0.5)
        assert config.confidence_threshold == 0.5

        # Invalid threshold (too low)
        with pytest.raises(ValueError, match="confidence_threshold must be between 0 and 1"):
            PolicyConfig(confidence_threshold=-0.1)

        # Invalid threshold (too high)
        with pytest.raises(ValueError, match="confidence_threshold must be between 0 and 1"):
            PolicyConfig(confidence_threshold=1.5)

    def test_is_topic_prohibited(self):
        """Test prohibited topic checking"""
        config = PolicyConfig(
            prohibited_topics=["violence", "hate speech", "illegal activities"]
        )

        assert config.is_topic_prohibited("violent content")
        assert config.is_topic_prohibited("hate speech example")
        assert config.is_topic_prohibited("illegal activities guide")
        assert not config.is_topic_prohibited("peaceful meditation")

    def test_is_domain_allowed(self):
        """Test domain allowlist"""
        # No allowlist (all allowed)
        config1 = PolicyConfig(allowed_domains=None)
        assert config1.is_domain_allowed("example.com")
        assert config1.is_domain_allowed("any-domain.org")

        # With allowlist
        config2 = PolicyConfig(allowed_domains=["example.com", "trusted.org"])
        assert config2.is_domain_allowed("example.com")
        assert config2.is_domain_allowed("EXAMPLE.COM")  # Case insensitive
        assert config2.is_domain_allowed("trusted.org")
        assert not config2.is_domain_allowed("untrusted.com")

    def test_check_resource_limits(self):
        """Test resource limit checking"""
        config = PolicyConfig(
            resource_limits=ResourceLimits(
                max_cost_per_task=10.0,
                max_duration_minutes=30,
                max_tokens_per_task=100000,
                max_llm_calls_per_task=50
            )
        )

        # Within limits
        result = config.check_resource_limits(
            cost=5.0,
            duration_minutes=20,
            tokens=50000,
            llm_calls=25
        )
        assert result["within_limits"] is True
        assert len(result["violations"]) == 0

        # Cost violation
        result = config.check_resource_limits(cost=15.0)
        assert result["within_limits"] is False
        assert len(result["violations"]) == 1
        assert "Cost" in result["violations"][0]

        # Duration violation
        result = config.check_resource_limits(duration_minutes=45)
        assert result["within_limits"] is False
        assert "Duration" in result["violations"][0]

        # Tokens violation
        result = config.check_resource_limits(tokens=200000)
        assert result["within_limits"] is False
        assert "Tokens" in result["violations"][0]

        # LLM calls violation
        result = config.check_resource_limits(llm_calls=100)
        assert result["within_limits"] is False
        assert "LLM calls" in result["violations"][0]

        # Multiple violations
        result = config.check_resource_limits(
            cost=20.0,
            duration_minutes=60,
            tokens=300000,
            llm_calls=100
        )
        assert result["within_limits"] is False
        assert len(result["violations"]) == 4


# ============================================================================
# Preset Configurations Tests
# ============================================================================

class TestPresetPolicyConfigs:
    """Test preset policy configurations"""

    def test_production_preset(self):
        """Test production preset"""
        config = PresetPolicyConfigs.production()

        assert config.mode == PolicyMode.STRICT
        assert config.ethical_tier == "premium"
        assert config.confidence_threshold == 0.8
        assert config.resource_limits.max_cost_per_task == 10.0
        assert config.enable_audit_logging is True
        assert config.require_verification is True

    def test_development_preset(self):
        """Test development preset"""
        config = PresetPolicyConfigs.development()

        assert config.mode == PolicyMode.PERMISSIVE
        assert config.ethical_tier == "expensive"
        assert config.confidence_threshold == 0.6
        assert config.resource_limits.max_cost_per_task == 50.0
        assert config.enable_audit_logging is True

    def test_research_preset(self):
        """Test research preset"""
        config = PresetPolicyConfigs.research()

        assert config.mode == PolicyMode.FLEXIBLE
        assert config.ethical_tier == "balanced"
        assert config.confidence_threshold == 0.5
        assert config.resource_limits.max_cost_per_task == 100.0
        assert len(config.prohibited_topics) == 0  # No prohibitions
        assert config.enable_audit_logging is False

    def test_human_in_loop_preset(self):
        """Test human-in-loop preset"""
        config = PresetPolicyConfigs.human_in_loop()

        assert config.mode == PolicyMode.STRICT
        assert config.ethical_tier == "premium"
        assert config.confidence_threshold == 0.9
        assert config.escalate_to_human is True
        assert config.resource_limits.max_cost_per_task == 5.0


# ============================================================================
# PolicyAgent Tests (with mocked LLM)
# ============================================================================

class TestPolicyAgent:
    """Test PolicyAgent with mocked LLM responses"""

    @pytest.fixture
    def mock_llm_responses(self):
        """Mock LLM responses for testing"""
        return {
            "ethical_ethical": Mock(
                is_ethical="true",
                confidence="0.95",
                reasoning="Task is ethical and safe",
                violations="none",
                recommendations="none"
            ),
            "ethical_unethical": Mock(
                is_ethical="false",
                confidence="0.90",
                reasoning="Task contains harmful content",
                violations="harmful_content, violence",
                recommendations="Remove violent elements"
            ),
            "strategic": Mock(
                priority="high",
                recommended_approach="Implement with security best practices",
                resource_allocation="2 developers, 1 week, $500 budget",
                risks="Security vulnerabilities if not properly tested",
                success_criteria="Tests pass, security audit complete, documentation written"
            ),
            "validation_approved": Mock(
                is_valid="true",
                validation_issues="none",
                final_decision="approved",
                audit_summary="Task approved after ethical and strategic review"
            ),
            "validation_blocked": Mock(
                is_valid="false",
                validation_issues="ethical_violation",
                final_decision="blocked",
                audit_summary="Task blocked due to ethical concerns"
            )
        }

    @pytest.fixture
    def agent(self):
        """Create PolicyAgent with production config"""
        config = PresetPolicyConfigs.production()
        return PolicyAgent(config=config)

    def test_agent_initialization(self, agent):
        """Test agent initialization"""
        assert agent.config.mode == PolicyMode.STRICT
        assert agent.config.ethical_tier == "premium"
        assert agent.ethical_evaluator is not None
        assert agent.strategic_advisor is not None
        assert agent.policy_validator is not None

    @patch('fractal_agent.agents.policy_agent.dspy.ChainOfThought')
    @patch('fractal_agent.agents.policy_agent.dspy.Predict')
    @patch('fractal_agent.agents.policy_agent.configure_dspy')
    def test_evaluate_ethical_task(self, mock_configure, mock_predict, mock_cot, agent, mock_llm_responses):
        """Test evaluation of ethical task"""
        # Mock DSPy module responses
        ethical_mock = Mock()
        ethical_mock.return_value = mock_llm_responses["ethical_ethical"]
        strategic_mock = Mock()
        strategic_mock.return_value = mock_llm_responses["strategic"]
        validation_mock = Mock()
        validation_mock.return_value = mock_llm_responses["validation_approved"]

        agent.ethical_evaluator = ethical_mock
        agent.strategic_advisor = strategic_mock
        agent.policy_validator = validation_mock

        # Evaluate task
        result = agent.evaluate_task(
            task_description="Implement user authentication",
            context={"user": "developer"}
        )

        # Verify ethical evaluation was called
        assert ethical_mock.called

        # Verify result
        assert isinstance(result, PolicyResult)
        assert result.decision == PolicyDecision.APPROVED
        assert result.ethical_evaluation.is_ethical is True
        assert result.ethical_evaluation.confidence >= 0.8

    @patch('fractal_agent.agents.policy_agent.dspy.ChainOfThought')
    @patch('fractal_agent.agents.policy_agent.dspy.Predict')
    @patch('fractal_agent.agents.policy_agent.configure_dspy')
    def test_evaluate_unethical_task_strict_mode(self, mock_configure, mock_predict, mock_cot, agent, mock_llm_responses):
        """Test evaluation of unethical task in strict mode"""
        # Mock DSPy module responses
        ethical_mock = Mock()
        ethical_mock.return_value = mock_llm_responses["ethical_unethical"]
        validation_mock = Mock()
        validation_mock.return_value = mock_llm_responses["validation_blocked"]

        agent.ethical_evaluator = ethical_mock
        agent.policy_validator = validation_mock

        # Evaluate task
        result = agent.evaluate_task(
            task_description="Generate violent content",
            context={"user": "developer"}
        )

        # Verify result
        assert isinstance(result, PolicyResult)
        assert result.decision == PolicyDecision.BLOCKED
        assert result.ethical_evaluation.is_ethical is False
        assert len(result.ethical_evaluation.violations) > 0

    @patch('fractal_agent.agents.policy_agent.dspy.ChainOfThought')
    @patch('fractal_agent.agents.policy_agent.dspy.Predict')
    @patch('fractal_agent.agents.policy_agent.configure_dspy')
    def test_evaluate_unethical_task_permissive_mode(self, mock_configure, mock_predict, mock_cot, mock_llm_responses):
        """Test evaluation of unethical task in permissive mode"""
        # Create permissive config
        config = PresetPolicyConfigs.development()
        agent = PolicyAgent(config=config)

        # Mock DSPy module responses
        ethical_mock = Mock()
        ethical_mock.return_value = mock_llm_responses["ethical_unethical"]
        validation_mock = Mock()
        validation_mock.return_value = Mock(
            is_valid="true",
            validation_issues="none",
            final_decision="warning",
            audit_summary="Warning issued for ethical concerns"
        )

        agent.ethical_evaluator = ethical_mock
        agent.policy_validator = validation_mock

        # Evaluate task
        result = agent.evaluate_task(
            task_description="Generate violent content",
            context={"user": "developer"}
        )

        # In permissive mode, should warn but allow
        assert result.decision == PolicyDecision.WARNING

    @patch('fractal_agent.agents.policy_agent.dspy.ChainOfThought')
    @patch('fractal_agent.agents.policy_agent.dspy.Predict')
    @patch('fractal_agent.agents.policy_agent.configure_dspy')
    def test_low_confidence_escalation(self, mock_configure, mock_predict, mock_cot, agent, mock_llm_responses):
        """Test escalation on low confidence"""
        # Mock low confidence response
        low_conf_response = Mock(
            is_ethical="true",
            confidence="0.5",  # Below threshold (0.8)
            reasoning="Uncertain evaluation",
            violations="none",
            recommendations="none"
        )
        validation_mock = Mock()
        validation_mock.return_value = Mock(
            is_valid="true",
            validation_issues="low_confidence",
            final_decision="escalated",
            audit_summary="Escalated due to low confidence"
        )

        agent.ethical_evaluator = Mock(return_value=low_conf_response)
        agent.policy_validator = validation_mock

        result = agent.evaluate_task(
            task_description="Ambiguous task",
            context={}
        )

        # Should escalate due to low confidence in strict mode
        assert result.decision in [PolicyDecision.ESCALATED, PolicyDecision.WARNING]

    def test_parse_bool(self, agent):
        """Test boolean parsing"""
        assert agent._parse_bool(True) is True
        assert agent._parse_bool(False) is False
        assert agent._parse_bool("true") is True
        assert agent._parse_bool("True") is True
        assert agent._parse_bool("yes") is True
        assert agent._parse_bool("approved") is True
        assert agent._parse_bool("ethical") is True
        assert agent._parse_bool("false") is False
        assert agent._parse_bool("no") is False

    def test_parse_confidence(self, agent):
        """Test confidence parsing"""
        assert agent._parse_confidence("0.95") == 0.95
        assert agent._parse_confidence(0.8) == 0.8
        assert agent._parse_confidence("1.5") == 1.0  # Clamped to max
        assert agent._parse_confidence("-0.5") == 0.0  # Clamped to min
        assert agent._parse_confidence("invalid") == 0.5  # Default

    def test_parse_list(self, agent):
        """Test list parsing"""
        assert agent._parse_list("none") == []
        assert agent._parse_list("n/a") == []
        assert agent._parse_list("") == []
        assert agent._parse_list("item1, item2, item3") == ["item1", "item2", "item3"]
        assert agent._parse_list("item1\nitem2\nitem3") == ["item1", "item2", "item3"]
        assert agent._parse_list(["a", "b", "c"]) == ["a", "b", "c"]

    def test_parse_priority(self, agent):
        """Test priority parsing"""
        assert agent._parse_priority("critical") == StrategicPriority.CRITICAL
        assert agent._parse_priority("HIGH priority") == StrategicPriority.HIGH
        assert agent._parse_priority("medium") == StrategicPriority.MEDIUM
        assert agent._parse_priority("low") == StrategicPriority.LOW
        assert agent._parse_priority("defer") == StrategicPriority.DEFER
        assert agent._parse_priority("unknown") == StrategicPriority.MEDIUM

    def test_policy_result_to_dict(self):
        """Test PolicyResult serialization"""
        eval = PolicyEvaluation(
            is_ethical=True,
            confidence=0.95,
            reasoning="Task is safe",
            violations=[],
            recommendations=[]
        )
        assess = StrategicAssessment(
            priority=StrategicPriority.HIGH,
            recommended_approach="Implement carefully",
            resource_allocation="1 week",
            risks="None identified",
            success_criteria=["Tests pass", "Code reviewed"]
        )
        result = PolicyResult(
            decision=PolicyDecision.APPROVED,
            ethical_evaluation=eval,
            strategic_assessment=assess,
            audit_summary="Task approved"
        )

        dict_result = result.to_dict()

        assert dict_result["decision"] == "approved"
        assert dict_result["ethical_evaluation"]["is_ethical"] is True
        assert dict_result["strategic_assessment"]["priority"] == "high"
        assert "timestamp" in dict_result


# ============================================================================
# Tier Verification Tests
# ============================================================================

class TestPolicyAgentVerification:
    """Test PolicyAgent tier verification capabilities"""

    @pytest.fixture
    def agent(self):
        """Create PolicyAgent with verification enabled"""
        config = PresetPolicyConfigs.production()
        return PolicyAgent(config=config)

    def test_verify_system4_output_enabled(self, agent):
        """Test System 4 verification when enabled"""
        # Create mock goal and report
        goal = Goal(
            objective="Analyze performance data",
            success_criteria=["Data analyzed", "Insights generated"],
            required_artifacts=[]
        )

        system4_report = {
            "analysis_complete": True,
            "insights": ["Insight 1", "Insight 2"]
        }

        # Mock tier verifier
        mock_verifier = Mock()
        mock_verification = TierVerificationResult(
            goal_achieved=True,
            report_accurate=True,
            discrepancies=[],
            confidence=0.95,
            reasoning="System 4 output verified"
        )
        mock_verifier.verify_subordinate = Mock(return_value=mock_verification)
        agent.tier_verifier = mock_verifier

        # Verify
        result = agent.verify_system4_output(
            goal=goal,
            system4_report=system4_report
        )

        assert result.goal_achieved is True
        assert result.report_accurate is True
        assert mock_verifier.verify_subordinate.called

    def test_verify_system4_output_disabled(self):
        """Test System 4 verification when disabled"""
        config = PresetPolicyConfigs.research()
        config.require_verification = False
        agent = PolicyAgent(config=config)

        goal = Goal(
            objective="Test task",
            success_criteria=["Complete"],
            required_artifacts=[]
        )

        # Verification should be skipped
        result = agent.verify_system4_output(
            goal=goal,
            system4_report={"done": True}
        )

        assert result.goal_achieved is True
        assert result.metadata.get("verification_skipped") is True


# ============================================================================
# Integration Behavior Tests
# ============================================================================

class TestPolicyAgentBehavior:
    """Test PolicyAgent behavioral scenarios"""

    def test_human_in_loop_escalation(self):
        """Test human-in-loop mode escalates all decisions"""
        config = PresetPolicyConfigs.human_in_loop()
        agent = PolicyAgent(config=config)

        # Mock ethical evaluation as ethical
        ethical_mock = Mock()
        ethical_mock.return_value = Mock(
            is_ethical="true",
            confidence="0.95",
            reasoning="Task is ethical",
            violations="none",
            recommendations="none"
        )
        validation_mock = Mock()
        validation_mock.return_value = Mock(
            is_valid="true",
            validation_issues="none",
            final_decision="approved",
            audit_summary="Approved but flagged for human review"
        )

        agent.ethical_evaluator = ethical_mock
        agent.policy_validator = validation_mock

        result = agent.evaluate_task(
            task_description="Normal task",
            context={}
        )

        # Should escalate even though ethical
        assert result.decision == PolicyDecision.ESCALATED

    def test_strategic_guidance_disabled(self):
        """Test that strategic guidance can be disabled"""
        config = PresetPolicyConfigs.research()
        config.enable_strategic_guidance = False
        agent = PolicyAgent(config=config)

        # Mock ethical evaluation
        ethical_mock = Mock()
        ethical_mock.return_value = Mock(
            is_ethical="true",
            confidence="0.95",
            reasoning="Ethical",
            violations="none",
            recommendations="none"
        )
        validation_mock = Mock()
        validation_mock.return_value = Mock(
            is_valid="true",
            validation_issues="none",
            final_decision="approved",
            audit_summary="Approved without strategic guidance"
        )

        agent.ethical_evaluator = ethical_mock
        agent.policy_validator = validation_mock

        result = agent.evaluate_task(
            task_description="Task",
            context={}
        )

        # Strategic assessment should be None when disabled
        assert result.strategic_assessment is None
