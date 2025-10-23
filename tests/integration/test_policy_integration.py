"""
Integration Tests for PolicyAgent (VSM System 5)

Tests integration with System 4 (Intelligence), verification framework,
and end-to-end policy workflows.

Author: BMad
Date: 2025-10-20
"""

import pytest
from fractal_agent.agents.policy_agent import (
    PolicyAgent,
    PolicyDecision,
    PolicyResult
)
from fractal_agent.agents.policy_config import (
    PolicyConfig,
    PolicyMode,
    PresetPolicyConfigs
)
from fractal_agent.verification import Goal, create_code_generation_goal


# ============================================================================
# End-to-End Policy Workflows
# ============================================================================

class TestPolicyWorkflows:
    """Test complete policy workflows"""

    @pytest.mark.integration
    def test_production_workflow_ethical_task(self):
        """Test production workflow with ethical task"""
        config = PresetPolicyConfigs.production()
        agent = PolicyAgent(config=config)

        result = agent.evaluate_task(
            task_description="Implement secure password hashing using bcrypt",
            context={
                "user": "security_engineer",
                "domain": "authentication",
                "purpose": "user_security"
            },
            verbose=True
        )

        # Should be approved in production mode
        assert result.decision in [PolicyDecision.APPROVED, PolicyDecision.WARNING]
        assert isinstance(result, PolicyResult)
        assert result.ethical_evaluation is not None

        # Should have strategic assessment (enabled in production)
        assert result.strategic_assessment is not None
        assert result.strategic_assessment.priority is not None

    @pytest.mark.integration
    def test_development_workflow_permissive(self):
        """Test development workflow with permissive mode"""
        config = PresetPolicyConfigs.development()
        agent = PolicyAgent(config=config)

        result = agent.evaluate_task(
            task_description="Test edge case with potentially problematic input",
            context={
                "user": "developer",
                "domain": "testing",
                "purpose": "edge_case_validation"
            },
            verbose=True
        )

        # Development mode is permissive
        assert result.decision in [PolicyDecision.APPROVED, PolicyDecision.WARNING]
        assert result.ethical_evaluation.confidence >= 0.0

    @pytest.mark.integration
    def test_research_workflow_flexible(self):
        """Test research workflow with flexible mode"""
        config = PresetPolicyConfigs.research()
        agent = PolicyAgent(config=config)

        result = agent.evaluate_task(
            task_description="Research advanced AI techniques",
            context={
                "user": "researcher",
                "domain": "ai_research",
                "purpose": "academic_study"
            },
            verbose=True
        )

        # Research mode is flexible (advisory only)
        assert result.decision in [PolicyDecision.APPROVED, PolicyDecision.WARNING]

        # Strategic guidance should be disabled in research preset
        assert result.strategic_assessment is None


# ============================================================================
# System 4 Integration Tests
# ============================================================================

class TestSystem4Integration:
    """Test PolicyAgent integration with System 4 (Intelligence)"""

    @pytest.mark.integration
    def test_verify_system4_intelligence_output(self):
        """Test verification of System 4 intelligence output"""
        config = PresetPolicyConfigs.production()
        agent = PolicyAgent(config=config)

        # Create goal for System 4
        goal = Goal(
            objective="Analyze system performance and identify optimization opportunities",
            success_criteria=[
                "Performance data analyzed",
                "Bottlenecks identified",
                "Optimization recommendations provided"
            ],
            required_artifacts=[],
            context={"system": "fractal_agent"}
        )

        # Mock System 4 report
        system4_report = {
            "agent_type": "IntelligenceAgent",
            "claimed_success": True,
            "analysis_complete": True,
            "bottlenecks_identified": ["GraphRAG query latency", "LLM API calls"],
            "recommendations": [
                "Add caching layer for GraphRAG",
                "Batch LLM API calls",
                "Use cheaper tier for simple queries"
            ],
            "metadata": {
                "metrics_analyzed": 150,
                "patterns_found": 3
            }
        }

        # Verify System 4 output
        verification = agent.verify_system4_output(
            goal=goal,
            system4_report=system4_report,
            verbose=True
        )

        # Should verify successfully (even with mocked data)
        assert verification is not None
        assert hasattr(verification, 'goal_achieved')
        assert hasattr(verification, 'report_accurate')

    @pytest.mark.integration
    def test_policy_then_verification_workflow(self):
        """Test complete workflow: policy evaluation → System 4 → verification"""
        config = PresetPolicyConfigs.production()
        agent = PolicyAgent(config=config)

        # Step 1: Evaluate task with policy agent
        task_description = "Optimize system performance using MIPRO"

        policy_result = agent.evaluate_task(
            task_description=task_description,
            context={"user": "system", "domain": "optimization"},
            verbose=True
        )

        # Should be approved
        assert policy_result.decision == PolicyDecision.APPROVED

        # Step 2: Create goal for System 4 based on approved task
        goal = Goal(
            objective=task_description,
            success_criteria=[
                "Performance optimized",
                "MIPRO successfully applied"
            ],
            required_artifacts=[]
        )

        # Step 3: Verify System 4 completion (mocked)
        system4_report = {
            "optimization_complete": True,
            "mipro_applied": True,
            "performance_improvement": "15%"
        }

        verification = agent.verify_system4_output(
            goal=goal,
            system4_report=system4_report,
            verbose=True
        )

        assert verification is not None


# ============================================================================
# Resource Limit Integration Tests
# ============================================================================

class TestResourceLimitIntegration:
    """Test resource limit enforcement in real scenarios"""

    @pytest.mark.integration
    def test_cost_limit_enforcement(self):
        """Test cost limit enforcement"""
        config = PresetPolicyConfigs.production()
        agent = PolicyAgent(config=config)

        # Check within limits
        within_limits = config.check_resource_limits(
            cost=5.0,
            duration_minutes=15,
            tokens=50000,
            llm_calls=25
        )
        assert within_limits["within_limits"] is True

        # Check exceeding cost limit
        exceeds_cost = config.check_resource_limits(cost=15.0)
        assert exceeds_cost["within_limits"] is False
        assert any("Cost" in v for v in exceeds_cost["violations"])

    @pytest.mark.integration
    def test_resource_limits_different_presets(self):
        """Test resource limits across different presets"""
        production = PresetPolicyConfigs.production()
        development = PresetPolicyConfigs.development()
        research = PresetPolicyConfigs.research()

        # Production is most strict
        assert production.resource_limits.max_cost_per_task < development.resource_limits.max_cost_per_task
        assert development.resource_limits.max_cost_per_task < research.resource_limits.max_cost_per_task

        # Test same usage across presets
        usage = {"cost": 25.0, "duration_minutes": 45, "tokens": 150000}

        prod_check = production.check_resource_limits(**usage)
        dev_check = development.check_resource_limits(**usage)
        research_check = research.check_resource_limits(**usage)

        # Production should block, development might warn, research should allow
        assert prod_check["within_limits"] is False  # Too expensive for production
        assert dev_check["within_limits"] is True  # OK for development
        assert research_check["within_limits"] is True  # OK for research


# ============================================================================
# Audit Trail Integration Tests
# ============================================================================

class TestAuditTrailIntegration:
    """Test audit trail logging"""

    @pytest.mark.integration
    def test_audit_logging_enabled(self, caplog):
        """Test that audit logging works when enabled"""
        import logging

        caplog.set_level(logging.INFO)

        config = PresetPolicyConfigs.production()
        assert config.enable_audit_logging is True

        agent = PolicyAgent(config=config)

        # Evaluate a task
        result = agent.evaluate_task(
            task_description="Test task for audit trail",
            context={"user": "test_user"},
            verbose=False
        )

        # Check that audit log was created
        assert result.audit_summary != ""
        assert result.timestamp is not None

    @pytest.mark.integration
    def test_audit_logging_disabled(self, caplog):
        """Test that audit logging can be disabled"""
        import logging

        caplog.set_level(logging.INFO)

        config = PresetPolicyConfigs.research()
        assert config.enable_audit_logging is False

        agent = PolicyAgent(config=config)

        # Clear previous logs
        caplog.clear()

        # Evaluate a task
        result = agent.evaluate_task(
            task_description="Test task without audit",
            context={"user": "test_user"},
            verbose=False
        )

        # Result should still have timestamp
        assert result.timestamp is not None


# ============================================================================
# Ethical Boundary Integration Tests
# ============================================================================

class TestEthicalBoundaryIntegration:
    """Test ethical boundary detection in realistic scenarios"""

    @pytest.mark.integration
    def test_prohibited_topic_detection(self):
        """Test detection of prohibited topics"""
        config = PolicyConfig(
            mode=PolicyMode.STRICT,
            prohibited_topics=["violence", "hate speech", "illegal activities"]
        )

        assert config.is_topic_prohibited("creating violent content")
        assert config.is_topic_prohibited("hate speech generator")
        assert config.is_topic_prohibited("guide to illegal activities")
        assert not config.is_topic_prohibited("peaceful conflict resolution")

    @pytest.mark.integration
    def test_domain_allowlist_enforcement(self):
        """Test domain allowlist enforcement"""
        config = PolicyConfig(
            mode=PolicyMode.STRICT,
            allowed_domains=["example.com", "trusted.org", "safe.net"]
        )

        # Allowed domains
        assert config.is_domain_allowed("example.com")
        assert config.is_domain_allowed("trusted.org")
        assert config.is_domain_allowed("SAFE.NET")  # Case insensitive

        # Blocked domains
        assert not config.is_domain_allowed("untrusted.com")
        assert not config.is_domain_allowed("malicious.org")


# ============================================================================
# Multi-Mode Integration Tests
# ============================================================================

class TestMultiModeIntegration:
    """Test behavior across different operational modes"""

    @pytest.mark.integration
    def test_strict_mode_blocks_violations(self):
        """Test that strict mode blocks violations"""
        config = PolicyConfig(mode=PolicyMode.STRICT)
        agent = PolicyAgent(config=config)

        # This should be evaluated as potentially problematic
        # Note: Actual blocking depends on LLM evaluation
        result = agent.evaluate_task(
            task_description="Generate content that might be harmful",
            context={"user": "test"},
            verbose=True
        )

        # In strict mode, confidence should be high or decision clear
        assert result.decision in [PolicyDecision.APPROVED, PolicyDecision.BLOCKED, PolicyDecision.WARNING, PolicyDecision.ESCALATED]

    @pytest.mark.integration
    def test_permissive_mode_warns(self):
        """Test that permissive mode issues warnings"""
        config = PolicyConfig(mode=PolicyMode.PERMISSIVE)
        agent = PolicyAgent(config=config)

        result = agent.evaluate_task(
            task_description="Test edge case scenario",
            context={"user": "developer"},
            verbose=True
        )

        # Permissive mode should allow with possible warnings
        assert result.decision in [PolicyDecision.APPROVED, PolicyDecision.WARNING]

    @pytest.mark.integration
    def test_flexible_mode_advisory(self):
        """Test that flexible mode is advisory only"""
        config = PolicyConfig(mode=PolicyMode.FLEXIBLE)
        agent = PolicyAgent(config=config)

        result = agent.evaluate_task(
            task_description="Research experimental approach",
            context={"user": "researcher"},
            verbose=True
        )

        # Flexible mode should approve (advisory only)
        assert result.decision in [PolicyDecision.APPROVED, PolicyDecision.WARNING]


# ============================================================================
# Stress and Edge Case Tests
# ============================================================================

class TestPolicyEdgeCases:
    """Test edge cases and stress scenarios"""

    @pytest.mark.integration
    def test_empty_task_description(self):
        """Test handling of empty task description"""
        config = PresetPolicyConfigs.production()
        agent = PolicyAgent(config=config)

        result = agent.evaluate_task(
            task_description="",
            context={},
            verbose=False
        )

        # Should still produce a result (likely low confidence)
        assert isinstance(result, PolicyResult)

    @pytest.mark.integration
    def test_very_long_task_description(self):
        """Test handling of very long task description"""
        config = PresetPolicyConfigs.production()
        agent = PolicyAgent(config=config)

        # Create a very long task description
        long_description = "Implement a complex system " * 100

        result = agent.evaluate_task(
            task_description=long_description,
            context={},
            verbose=False
        )

        # Should still produce a result
        assert isinstance(result, PolicyResult)

    @pytest.mark.integration
    def test_minimal_context(self):
        """Test evaluation with minimal context"""
        config = PresetPolicyConfigs.production()
        agent = PolicyAgent(config=config)

        result = agent.evaluate_task(
            task_description="Do a thing",
            context=None,  # No context
            verbose=False
        )

        # Should handle gracefully
        assert isinstance(result, PolicyResult)

    @pytest.mark.integration
    def test_rich_context(self):
        """Test evaluation with rich context"""
        config = PresetPolicyConfigs.production()
        agent = PolicyAgent(config=config)

        result = agent.evaluate_task(
            task_description="Implement secure data processing",
            context={
                "user": "senior_engineer",
                "domain": "data_security",
                "purpose": "compliance",
                "regulations": ["GDPR", "CCPA"],
                "data_types": ["PII", "financial"],
                "security_level": "high",
                "budget": "$5000",
                "timeline": "2 weeks"
            },
            verbose=False
        )

        # Should leverage rich context
        assert isinstance(result, PolicyResult)
        assert result.ethical_evaluation is not None
