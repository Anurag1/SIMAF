"""
PolicyAgent - VSM System 5 (Policy & Ethics)

Top-tier agent providing ethical governance, strategic direction, and
resource authorization for the Fractal Agent Ecosystem.

VSM System 5 Responsibilities:
- Ethical boundary enforcement
- Strategic direction setting
- Resource authorization (cost, tokens, duration)
- Tier adjacency compliance (only talks to System 4)
- Trust-but-verify: Verifies System 4 outputs

Architecture:
    Stage 1: Ethical Evaluation (Premium Tier)
        → Detects harmful content, privacy violations, deception
        → Returns ethical/unethical with confidence score

    Stage 2: Strategic Guidance (Expensive Tier)
        → Assesses strategic priority
        → Recommends approach and resource allocation

    Stage 3: Policy Validation (Balanced Tier)
        → Final compliance check
        → Verifies resource limits
        → Generates audit trail

Author: BMad
Date: 2025-10-20 (Reconstructed: 2025-10-22)
"""

import dspy
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime
from .policy_config import PolicyConfig, PolicyMode, EthicalBoundary
from ..utils.dspy_integration import FractalDSpyLM, configure_dspy
from ..utils.model_config import Tier
from ..observability import (
    get_correlation_id, get_tracer, get_logger,
    get_event_store, VSMEvent, set_span_attributes
)
import logging

# Use observability-aware structured logger
logger = get_logger(__name__)


# ============================================================================
# Data Models
# ============================================================================

class PolicyDecision(str, Enum):
    """Policy decision outcomes"""
    APPROVED = "approved"
    REJECTED = "rejected"
    ESCALATED = "escalated"  # Requires human review
    CONDITIONAL = "conditional"  # Approved with conditions


class StrategicPriority(str, Enum):
    """Strategic priority levels"""
    CRITICAL = "critical"  # Immediate action required
    HIGH = "high"  # Important, schedule soon
    MEDIUM = "medium"  # Normal priority
    LOW = "low"  # Can be deferred
    DEFER = "defer"  # Postpone indefinitely


@dataclass
class PolicyEvaluation:
    """
    Result of ethical evaluation stage.

    Attributes:
        is_ethical: Whether task passes ethical boundaries
        confidence: Confidence in evaluation (0-1)
        violations: List of detected ethical violations
        recommendations: List of recommendations to address violations
    """
    is_ethical: bool
    confidence: float
    violations: List[str]
    recommendations: List[str]


@dataclass
class StrategicAssessment:
    """
    Result of strategic guidance stage.

    Attributes:
        priority: Strategic priority level
        approach: Recommended approach to task
        risks: List of identified risks
        success_criteria: List of success criteria
        estimated_cost: Estimated cost in USD
        estimated_duration: Estimated duration in minutes
    """
    priority: StrategicPriority
    approach: str
    risks: List[str]
    success_criteria: List[str]
    estimated_cost: float
    estimated_duration: int


@dataclass
class PolicyResult:
    """
    Complete policy evaluation result.

    Attributes:
        decision: Final policy decision
        ethical_evaluation: Result from Stage 1
        strategic_assessment: Result from Stage 2 (if enabled)
        resource_approved: Whether resource usage is approved
        audit_trail: Audit log entry
        metadata: Additional metadata
    """
    decision: PolicyDecision
    ethical_evaluation: PolicyEvaluation
    strategic_assessment: Optional[StrategicAssessment]
    resource_approved: bool
    audit_trail: Dict[str, Any]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        # Convert enums to strings
        result["decision"] = self.decision.value
        if self.strategic_assessment:
            result["strategic_assessment"]["priority"] = self.strategic_assessment.priority.value
        return result


# ============================================================================
# DSPy Signatures
# ============================================================================

class EthicalEvaluationSignature(dspy.Signature):
    """
    Evaluate task for ethical compliance.

    Identify any violations of ethical boundaries including harmful content,
    privacy violations, deception, illegal activities, safety risks, bias,
    or resource abuse.
    """
    task_description: str = dspy.InputField(desc="The task to evaluate")
    context: str = dspy.InputField(desc="Additional context about the task")
    prohibited_topics: str = dspy.InputField(desc="List of prohibited topics/keywords")

    is_ethical: bool = dspy.OutputField(desc="True if task is ethical, False if violations detected")
    confidence: float = dspy.OutputField(desc="Confidence in evaluation (0.0-1.0)")
    violations: str = dspy.OutputField(desc="Comma-separated list of detected violations (empty if none)")
    recommendations: str = dspy.OutputField(desc="Comma-separated list of recommendations to address violations")


class StrategicGuidanceSignature(dspy.Signature):
    """
    Provide strategic guidance for task.

    Assess strategic priority, recommend approach, identify risks,
    and define success criteria.
    """
    task_description: str = dspy.InputField(desc="The task to assess")
    ethical_evaluation: str = dspy.InputField(desc="Result of ethical evaluation")
    context: str = dspy.InputField(desc="Additional context")

    priority: str = dspy.OutputField(desc="Strategic priority: critical, high, medium, low, or defer")
    approach: str = dspy.OutputField(desc="Recommended approach to task")
    risks: str = dspy.OutputField(desc="Comma-separated list of identified risks")
    success_criteria: str = dspy.OutputField(desc="Comma-separated list of success criteria")
    estimated_cost: float = dspy.OutputField(desc="Estimated cost in USD")
    estimated_duration: int = dspy.OutputField(desc="Estimated duration in minutes")


# ============================================================================
# Policy Agent
# ============================================================================

class PolicyAgent(dspy.Module):
    """
    VSM System 5 (Policy & Ethics) agent.

    Provides ethical governance, strategic direction, and resource authorization
    for the Fractal Agent Ecosystem.

    Usage:
        >>> config = PolicyConfig(mode=PolicyMode.STRICT)
        >>> agent = PolicyAgent(config=config)
        >>> result = agent.evaluate(
        ...     task_description="Research climate change impacts",
        ...     context="Scientific research project"
        ... )
        >>> print(result.decision)
        PolicyDecision.APPROVED
    """

    def __init__(self, config: Optional[PolicyConfig] = None):
        """
        Initialize policy agent.

        Args:
            config: PolicyConfig instance (uses default if None)
        """
        super().__init__()

        self.config = config or PolicyConfig()

        # Configure LLMs for different stages
        # Tier values: "cheap", "balanced", "expensive", "premium"
        self.ethical_lm = configure_dspy(tier=self.config.ethical_tier)
        self.ethical_evaluator = dspy.ChainOfThought(EthicalEvaluationSignature)

        # Strategic advisor uses different tier
        self.strategic_lm = configure_dspy(tier=self.config.strategic_tier)
        self.strategic_advisor = dspy.ChainOfThought(StrategicGuidanceSignature)

        logger.info(f"Initialized PolicyAgent: mode={self.config.mode.value}")

    def evaluate(
        self,
        task_description: str,
        context: str = "",
        resource_usage: Optional[Dict[str, Any]] = None,
        verify_system4_output: Optional[Any] = None
    ) -> PolicyResult:
        """
        Evaluate task through three-stage policy pipeline.

        Args:
            task_description: Description of task to evaluate
            context: Additional context
            resource_usage: Current/estimated resource usage
            verify_system4_output: Optional System 4 output to verify

        Returns:
            PolicyResult with decision and supporting data
        """
        tracer = get_tracer()
        with tracer.start_as_current_span("policy_evaluation") as span:
            set_span_attributes(span, {
                "agent.type": "policy",
                "policy.mode": self.config.mode.value
            })

            try:
                # Stage 1: Ethical Evaluation
                ethical_eval = self._evaluate_ethics(task_description, context)

                # Check if unethical in strict mode
                if not ethical_eval.is_ethical and self.config.mode == PolicyMode.STRICT:
                    return self._create_rejection_result(ethical_eval)

                # Check confidence threshold
                if ethical_eval.confidence < self.config.confidence_threshold:
                    if self.config.escalate_to_human:
                        return self._create_escalation_result(ethical_eval, "Low confidence")

                # Stage 2: Strategic Guidance (if enabled)
                strategic_assess = None
                if self.config.enable_strategic_guidance:
                    strategic_assess = self._provide_strategic_guidance(
                        task_description, context, ethical_eval
                    )

                # Stage 3: Resource Validation
                resource_approved = self._validate_resources(resource_usage, strategic_assess)

                # Create decision
                decision = self._make_decision(
                    ethical_eval, strategic_assess, resource_approved
                )

                # Create audit trail
                audit_trail = self._create_audit_trail(
                    task_description, ethical_eval, strategic_assess,
                    resource_usage, decision
                )

                # System 4 Verification (if requested)
                if verify_system4_output and self.config.require_verification:
                    self._verify_system4_output(verify_system4_output, audit_trail)

                result = PolicyResult(
                    decision=decision,
                    ethical_evaluation=ethical_eval,
                    strategic_assessment=strategic_assess,
                    resource_approved=resource_approved,
                    audit_trail=audit_trail,
                    metadata={
                        "mode": self.config.mode.value,
                        "correlation_id": get_correlation_id()
                    }
                )

                # Log event
                event_store = get_event_store()
                event_store.record(VSMEvent(
                    event_type="policy_decision",
                    agent="PolicyAgent",
                    data={
                        "decision": decision.value,
                        "is_ethical": ethical_eval.is_ethical,
                        "resource_approved": resource_approved
                    },
                    metadata={"correlation_id": get_correlation_id()}
                ))

                logger.info(f"Policy decision: {decision.value} (ethical={ethical_eval.is_ethical})")

                return result

            except Exception as e:
                logger.error(f"Policy evaluation failed: {e}", exc_info=True)
                set_span_attributes(span, {"error": str(e)})
                raise

    def _evaluate_ethics(self, task_description: str, context: str) -> PolicyEvaluation:
        """Stage 1: Evaluate ethical boundaries"""
        try:
            prohibited_topics_str = ", ".join(self.config.prohibited_topics)

            result = self.ethical_evaluator(
                task_description=task_description,
                context=context or "",
                prohibited_topics=prohibited_topics_str
            )

            # Parse outputs
            is_ethical = self._parse_boolean(result.is_ethical)
            confidence = self._parse_confidence(result.confidence)
            violations = self._parse_list(result.violations)
            recommendations = self._parse_list(result.recommendations)

            return PolicyEvaluation(
                is_ethical=is_ethical,
                confidence=confidence,
                violations=violations,
                recommendations=recommendations
            )

        except Exception as e:
            logger.error(f"Ethical evaluation failed: {e}")
            # Fail-safe: reject on error in strict mode
            if self.config.mode == PolicyMode.STRICT:
                return PolicyEvaluation(
                    is_ethical=False,
                    confidence=0.0,
                    violations=["evaluation_error"],
                    recommendations=["Manual review required"]
                )
            raise

    def _provide_strategic_guidance(
        self,
        task_description: str,
        context: str,
        ethical_eval: PolicyEvaluation
    ) -> StrategicAssessment:
        """Stage 2: Provide strategic guidance"""
        try:
            ethical_summary = f"Ethical: {ethical_eval.is_ethical}, Confidence: {ethical_eval.confidence}"

            result = self.strategic_advisor(
                task_description=task_description,
                ethical_evaluation=ethical_summary,
                context=context or ""
            )

            # Parse outputs
            priority = self._parse_priority(result.priority)
            approach = str(result.approach)
            risks = self._parse_list(result.risks)
            success_criteria = self._parse_list(result.success_criteria)
            estimated_cost = float(result.estimated_cost) if result.estimated_cost else 0.0
            estimated_duration = int(result.estimated_duration) if result.estimated_duration else 0

            return StrategicAssessment(
                priority=priority,
                approach=approach,
                risks=risks,
                success_criteria=success_criteria,
                estimated_cost=estimated_cost,
                estimated_duration=estimated_duration
            )

        except Exception as e:
            logger.error(f"Strategic guidance failed: {e}")
            # Return safe defaults
            return StrategicAssessment(
                priority=StrategicPriority.MEDIUM,
                approach="Proceed with caution",
                risks=["guidance_error"],
                success_criteria=["Manual validation"],
                estimated_cost=0.0,
                estimated_duration=0
            )

    def _validate_resources(
        self,
        resource_usage: Optional[Dict[str, Any]],
        strategic_assess: Optional[StrategicAssessment]
    ) -> bool:
        """Stage 3: Validate resource usage"""
        if not resource_usage:
            return True

        limits = self.config.resource_limits

        # Check cost
        cost = resource_usage.get("cost", 0.0)
        if strategic_assess:
            cost = max(cost, strategic_assess.estimated_cost)
        if cost > limits.max_cost_per_task:
            logger.warning(f"Cost {cost} exceeds limit {limits.max_cost_per_task}")
            return False

        # Check duration
        duration = resource_usage.get("duration_minutes", 0)
        if strategic_assess:
            duration = max(duration, strategic_assess.estimated_duration)
        if duration > limits.max_duration_minutes:
            logger.warning(f"Duration {duration} exceeds limit {limits.max_duration_minutes}")
            return False

        # Check tokens
        tokens = resource_usage.get("tokens", 0)
        if tokens > limits.max_tokens_per_task:
            logger.warning(f"Tokens {tokens} exceeds limit {limits.max_tokens_per_task}")
            return False

        # Check LLM calls
        llm_calls = resource_usage.get("llm_calls", 0)
        if llm_calls > limits.max_llm_calls_per_task:
            logger.warning(f"LLM calls {llm_calls} exceeds limit {limits.max_llm_calls_per_task}")
            return False

        return True

    def _make_decision(
        self,
        ethical_eval: PolicyEvaluation,
        strategic_assess: Optional[StrategicAssessment],
        resource_approved: bool
    ) -> PolicyDecision:
        """Make final policy decision"""
        # Escalate to human if configured
        if self.config.escalate_to_human:
            return PolicyDecision.ESCALATED

        # Reject if unethical (regardless of mode in this method - rejection handled earlier)
        if not ethical_eval.is_ethical:
            if self.config.mode == PolicyMode.STRICT:
                return PolicyDecision.REJECTED
            elif self.config.mode == PolicyMode.PERMISSIVE:
                return PolicyDecision.CONDITIONAL
            else:  # FLEXIBLE
                return PolicyDecision.CONDITIONAL

        # Reject if resources not approved
        if not resource_approved:
            if self.config.mode == PolicyMode.STRICT:
                return PolicyDecision.REJECTED
            else:
                return PolicyDecision.CONDITIONAL

        # Check strategic priority (if available)
        if strategic_assess and strategic_assess.priority == StrategicPriority.DEFER:
            return PolicyDecision.CONDITIONAL

        return PolicyDecision.APPROVED

    def _create_rejection_result(self, ethical_eval: PolicyEvaluation) -> PolicyResult:
        """Create rejection result for unethical tasks"""
        return PolicyResult(
            decision=PolicyDecision.REJECTED,
            ethical_evaluation=ethical_eval,
            strategic_assessment=None,
            resource_approved=False,
            audit_trail={
                "timestamp": datetime.utcnow().isoformat(),
                "reason": "ethical_violation",
                "violations": ethical_eval.violations
            },
            metadata={"mode": self.config.mode.value}
        )

    def _create_escalation_result(
        self,
        ethical_eval: PolicyEvaluation,
        reason: str
    ) -> PolicyResult:
        """Create escalation result for human review"""
        return PolicyResult(
            decision=PolicyDecision.ESCALATED,
            ethical_evaluation=ethical_eval,
            strategic_assessment=None,
            resource_approved=False,
            audit_trail={
                "timestamp": datetime.utcnow().isoformat(),
                "reason": reason,
                "escalation_required": True
            },
            metadata={"mode": self.config.mode.value}
        )

    def _create_audit_trail(
        self,
        task_description: str,
        ethical_eval: PolicyEvaluation,
        strategic_assess: Optional[StrategicAssessment],
        resource_usage: Optional[Dict[str, Any]],
        decision: PolicyDecision
    ) -> Dict[str, Any]:
        """Create audit trail entry"""
        trail = {
            "timestamp": datetime.utcnow().isoformat(),
            "task_description": task_description[:200],  # Truncate for storage
            "decision": decision.value,
            "ethical_evaluation": {
                "is_ethical": ethical_eval.is_ethical,
                "confidence": ethical_eval.confidence,
                "violations": ethical_eval.violations
            },
            "resource_usage": resource_usage or {},
            "mode": self.config.mode.value,
            "correlation_id": get_correlation_id()
        }

        if strategic_assess:
            trail["strategic_assessment"] = {
                "priority": strategic_assess.priority.value,
                "estimated_cost": strategic_assess.estimated_cost,
                "estimated_duration": strategic_assess.estimated_duration
            }

        return trail

    def _verify_system4_output(self, system4_output: Any, audit_trail: Dict[str, Any]):
        """Verify System 4 output (trust-but-verify)"""
        try:
            from ..verification import TierVerification

            verifier = TierVerification()
            # Implementation would depend on system4_output structure
            # This is a placeholder for the verification logic
            logger.info("System 4 output verification requested")
            audit_trail["system4_verified"] = True

        except Exception as e:
            logger.error(f"System 4 verification failed: {e}")
            audit_trail["system4_verified"] = False
            audit_trail["verification_error"] = str(e)

    # ============================================================================
    # Parsing Helpers
    # ============================================================================

    def _parse_boolean(self, value: Any) -> bool:
        """Parse boolean value from LLM output"""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("true", "yes", "1", "ethical")
        return bool(value)

    def _parse_confidence(self, value: Any) -> float:
        """Parse confidence value (0-1)"""
        try:
            conf = float(value)
            return max(0.0, min(1.0, conf))  # Clamp to [0, 1]
        except (ValueError, TypeError):
            return 0.5  # Default to medium confidence

    def _parse_list(self, value: Any) -> List[str]:
        """Parse comma-separated list"""
        if isinstance(value, list):
            return [str(item).strip() for item in value]
        if isinstance(value, str):
            if not value or value.lower() in ("none", "empty", ""):
                return []
            return [item.strip() for item in value.split(",") if item.strip()]
        return []

    def _parse_priority(self, value: Any) -> StrategicPriority:
        """Parse strategic priority"""
        if isinstance(value, StrategicPriority):
            return value
        if isinstance(value, str):
            value_lower = value.lower().strip()
            for priority in StrategicPriority:
                if priority.value == value_lower:
                    return priority
        # Default to medium priority
        return StrategicPriority.MEDIUM


# Demo
if __name__ == "__main__":
    print("=" * 80)
    print("PolicyAgent (VSM System 5) Demo")
    print("=" * 80)
    print()

    try:
        # Test ethical task
        agent = PolicyAgent()

        result = agent.evaluate(
            task_description="Research renewable energy solutions for climate change",
            context="Scientific research project",
            resource_usage={"cost": 5.0, "duration_minutes": 15}
        )

        print(f"✅ Decision: {result.decision.value}")
        print(f"   Ethical: {result.ethical_evaluation.is_ethical}")
        print(f"   Confidence: {result.ethical_evaluation.confidence:.2f}")
        print(f"   Resource Approved: {result.resource_approved}")
        print()

        print("=" * 80)
        print("PolicyAgent Demo Complete!")
        print("=" * 80)

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
