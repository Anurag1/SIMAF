"""
Policy Agent Configuration

Configuration for the PolicyAgent - VSM System 5 (Policy & Ethics)
that provides ethical boundary detection, strategic guidance, and
resource authorization for the Fractal Agent Ecosystem.

Features:
- Ethical boundary enforcement
- Strategic direction setting
- Resource authorization (cost, tokens, duration)
- Audit trail logging
- Three operational presets: Production, Development, Research

Author: BMad
Date: 2025-10-20
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum
from ..utils.model_config import Tier


class PolicyMode(str, Enum):
    """Operational modes for policy enforcement"""
    STRICT = "strict"  # Block violations (production)
    PERMISSIVE = "permissive"  # Warn on violations (development)
    FLEXIBLE = "flexible"  # Advisory only (research)


class EthicalBoundary(str, Enum):
    """Categories of ethical boundaries"""
    HARMFUL_CONTENT = "harmful_content"  # Violence, hate speech, etc.
    PRIVACY_VIOLATION = "privacy_violation"  # PII exposure, data misuse
    DECEPTION = "deception"  # Misleading, fraud, manipulation
    ILLEGAL_ACTIVITY = "illegal_activity"  # Criminal activities
    SAFETY_RISK = "safety_risk"  # Physical or psychological harm
    BIAS_DISCRIMINATION = "bias_discrimination"  # Unfair treatment
    RESOURCE_ABUSE = "resource_abuse"  # Excessive cost, token usage


@dataclass
class ResourceLimits:
    """Resource authorization limits"""
    max_cost_per_task: float = 10.0  # USD
    max_duration_minutes: int = 30  # minutes
    max_tokens_per_task: int = 100000  # tokens
    max_llm_calls_per_task: int = 50  # LLM API calls

    def __post_init__(self):
        """Validate limits"""
        if self.max_cost_per_task <= 0:
            raise ValueError("max_cost_per_task must be > 0")
        if self.max_duration_minutes <= 0:
            raise ValueError("max_duration_minutes must be > 0")
        if self.max_tokens_per_task <= 0:
            raise ValueError("max_tokens_per_task must be > 0")
        if self.max_llm_calls_per_task <= 0:
            raise ValueError("max_llm_calls_per_task must be > 0")


@dataclass
class PolicyConfig:
    """
    Configuration for PolicyAgent (VSM System 5).

    The PolicyAgent is the top tier in the VSM hierarchy, providing
    ethical governance, strategic direction, and resource authorization.

    VSM System 5 Responsibilities:
    - Ethical boundary enforcement
    - Strategic direction setting
    - Resource authorization and limits
    - Tier adjacency compliance (only talks to System 4)
    - Trust-but-verify: Verifies System 4 outputs

    Attributes:
        mode: Operational mode (strict/permissive/flexible)
        ethical_tier: Model tier for ethical evaluation (default: premium)
        strategic_tier: Model tier for strategic guidance (default: expensive)
        validation_tier: Model tier for policy validation (default: balanced)

        resource_limits: Resource authorization limits
        confidence_threshold: Minimum confidence for policy decisions (0-1)

        prohibited_topics: List of prohibited topics/keywords
        allowed_domains: Optional allowlist of permitted domains

        enable_strategic_guidance: Provide strategic direction
        enable_audit_logging: Log all policy decisions
        escalate_to_human: Flag decisions for human review

        require_verification: Verify System 4 outputs using TierVerification
    """

    # Operational mode
    mode: PolicyMode = PolicyMode.STRICT

    # Model tiers for different decision types
    ethical_tier: Tier = "premium"  # Highest quality for ethical decisions
    strategic_tier: Tier = "expensive"  # High quality for strategy
    validation_tier: Tier = "balanced"  # Moderate for validation

    # Resource limits
    resource_limits: ResourceLimits = field(default_factory=ResourceLimits)

    # Decision thresholds
    confidence_threshold: float = 0.8  # Minimum confidence (0-1)

    # Ethical boundaries
    prohibited_topics: List[str] = field(default_factory=lambda: [
        "violence",
        "hate speech",
        "illegal activities",
        "personal attacks",
        "discrimination",
        "harassment"
    ])

    allowed_domains: Optional[List[str]] = None  # None = all allowed

    # Feature flags
    enable_strategic_guidance: bool = True
    enable_audit_logging: bool = True
    escalate_to_human: bool = False  # Flag for human review

    # Verification
    require_verification: bool = True  # Verify System 4 outputs

    def __post_init__(self):
        """Validate configuration"""
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")

        # Ensure prohibited_topics is not empty in strict mode
        if self.mode == PolicyMode.STRICT and not self.prohibited_topics:
            raise ValueError("prohibited_topics cannot be empty in strict mode")

    def is_topic_prohibited(self, topic: str) -> bool:
        """
        Check if a topic is prohibited.

        Args:
            topic: Topic to check

        Returns:
            True if prohibited, False otherwise
        """
        topic_lower = topic.lower()
        return any(
            prohibited.lower() in topic_lower
            for prohibited in self.prohibited_topics
        )

    def is_domain_allowed(self, domain: str) -> bool:
        """
        Check if a domain is allowed.

        Args:
            domain: Domain to check

        Returns:
            True if allowed, False otherwise
        """
        if self.allowed_domains is None:
            return True  # All domains allowed

        return domain.lower() in [d.lower() for d in self.allowed_domains]

    def check_resource_limits(
        self,
        cost: Optional[float] = None,
        duration_minutes: Optional[int] = None,
        tokens: Optional[int] = None,
        llm_calls: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Check if resource usage is within limits.

        Args:
            cost: Cost in USD
            duration_minutes: Duration in minutes
            tokens: Token count
            llm_calls: Number of LLM API calls

        Returns:
            Dict with 'within_limits' (bool) and 'violations' (list)
        """
        violations = []

        if cost is not None and cost > self.resource_limits.max_cost_per_task:
            violations.append(
                f"Cost ${cost:.2f} exceeds limit ${self.resource_limits.max_cost_per_task:.2f}"
            )

        if duration_minutes is not None and duration_minutes > self.resource_limits.max_duration_minutes:
            violations.append(
                f"Duration {duration_minutes}m exceeds limit {self.resource_limits.max_duration_minutes}m"
            )

        if tokens is not None and tokens > self.resource_limits.max_tokens_per_task:
            violations.append(
                f"Tokens {tokens} exceeds limit {self.resource_limits.max_tokens_per_task}"
            )

        if llm_calls is not None and llm_calls > self.resource_limits.max_llm_calls_per_task:
            violations.append(
                f"LLM calls {llm_calls} exceeds limit {self.resource_limits.max_llm_calls_per_task}"
            )

        return {
            "within_limits": len(violations) == 0,
            "violations": violations
        }


# ============================================================================
# Preset Configurations
# ============================================================================

class PresetPolicyConfigs:
    """Preset configurations for common policy scenarios."""

    @staticmethod
    def production() -> PolicyConfig:
        """
        Production mode - strict ethical enforcement.

        - Strict mode: Blocks violations
        - Premium tier for ethical decisions
        - Conservative resource limits
        - Full audit logging
        - Verification enabled
        """
        return PolicyConfig(
            mode=PolicyMode.STRICT,
            ethical_tier="premium",
            strategic_tier="expensive",
            validation_tier="balanced",
            resource_limits=ResourceLimits(
                max_cost_per_task=10.0,
                max_duration_minutes=30,
                max_tokens_per_task=100000,
                max_llm_calls_per_task=50
            ),
            confidence_threshold=0.8,
            enable_strategic_guidance=True,
            enable_audit_logging=True,
            escalate_to_human=False,
            require_verification=True
        )

    @staticmethod
    def development() -> PolicyConfig:
        """
        Development mode - permissive with warnings.

        - Permissive mode: Warns on violations
        - Balanced tier for cost savings
        - Relaxed resource limits
        - Audit logging enabled
        - Verification enabled
        """
        return PolicyConfig(
            mode=PolicyMode.PERMISSIVE,
            ethical_tier="expensive",
            strategic_tier="balanced",
            validation_tier="balanced",
            resource_limits=ResourceLimits(
                max_cost_per_task=50.0,  # Higher for development
                max_duration_minutes=60,
                max_tokens_per_task=200000,
                max_llm_calls_per_task=100
            ),
            confidence_threshold=0.6,
            enable_strategic_guidance=True,
            enable_audit_logging=True,
            escalate_to_human=False,
            require_verification=True
        )

    @staticmethod
    def research() -> PolicyConfig:
        """
        Research mode - flexible advisory only.

        - Flexible mode: Advisory only, no blocking
        - Balanced tier for efficiency
        - Very relaxed resource limits
        - Minimal logging
        - Verification optional
        """
        return PolicyConfig(
            mode=PolicyMode.FLEXIBLE,
            ethical_tier="balanced",
            strategic_tier="balanced",
            validation_tier="cheap",
            resource_limits=ResourceLimits(
                max_cost_per_task=100.0,  # Very high for research
                max_duration_minutes=120,
                max_tokens_per_task=500000,
                max_llm_calls_per_task=200
            ),
            confidence_threshold=0.5,
            prohibited_topics=[],  # No prohibitions in research
            enable_strategic_guidance=False,  # Skip for speed
            enable_audit_logging=False,  # Minimal logging
            escalate_to_human=False,
            require_verification=False  # Optional in research
        )

    @staticmethod
    def human_in_loop() -> PolicyConfig:
        """
        Human-in-loop mode - escalate all decisions.

        - Strict mode with human escalation
        - Premium tier for all decisions
        - Conservative limits
        - Full audit logging
        - All decisions flagged for human review
        """
        return PolicyConfig(
            mode=PolicyMode.STRICT,
            ethical_tier="premium",
            strategic_tier="premium",
            validation_tier="expensive",
            resource_limits=ResourceLimits(
                max_cost_per_task=5.0,  # Conservative
                max_duration_minutes=15,
                max_tokens_per_task=50000,
                max_llm_calls_per_task=25
            ),
            confidence_threshold=0.9,  # Very high bar
            enable_strategic_guidance=True,
            enable_audit_logging=True,
            escalate_to_human=True,  # All decisions escalated
            require_verification=True
        )
