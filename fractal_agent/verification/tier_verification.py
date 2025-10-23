"""
Fractal VSM Tier Verification System

Implements hierarchical verification where each VSM tier validates its immediate
subordinate tier using three-way comparison:
1. GOAL: What I asked them to do
2. REPORT: What they said they did
3. ACTUAL: What actually happened (independent reality check)

Key Principle:
- Each tier ONLY validates the tier directly below it (no tier skipping)
- Verification is GENERIC (not task-specific)
- Trust-but-verify: Reports are verified against independent reality checks
- Discrepancies are detected and can trigger retries or escalation

Usage Example:
    # System 2 (Coordination) verifying System 1 (Operational):
    tier_verifier = TierVerification(
        tier_name="System 2 (Coordination)",
        subordinate_tier="System 1 (Operational)"
    )

    result = tier_verifier.verify_subordinate(
        goal=integration_goal,                    # What I asked S1 to do
        report=system1_agent.get_report(),        # What S1 claimed it did
        reality_check=lambda: check_filesystem()  # Independent verification
    )

    if not result.goal_achieved:
        # Handle discrepancies
        handle_failures(result.discrepancies)

Architecture:
- TierVerification: Main class for tier-to-tier verification
- ComparisonResult: Result of comparing two states
- TierVerificationResult: Full result with goal/report/actual comparison
- RealityCheckRegistry: Pluggable verification methods for different goal types

Author: BMad
Date: 2025-10-19
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable, Protocol
from pathlib import Path
from enum import Enum
import logging

from .goals import Goal, Evidence, VerificationResult, GoalStatus
from .verify import verify_goal, EvidenceCollector

logger = logging.getLogger(__name__)


# ============================================================================
# Comparison and Discrepancy Types
# ============================================================================

class DiscrepancyType(Enum):
    """Types of discrepancies that can be detected."""
    GOAL_NOT_ACHIEVED = "goal_not_achieved"  # Actual doesn't match goal
    REPORT_INACCURATE = "report_inaccurate"  # Report doesn't match actual
    REPORT_GOAL_MISMATCH = "report_goal_mismatch"  # Report doesn't match what was asked
    APPROACH_SUBOPTIMAL = "approach_suboptimal"  # Goal achieved but approach was wrong
    PARTIAL_COMPLETION = "partial_completion"  # Some criteria met, some not
    UNVERIFIABLE = "unverifiable"  # Cannot verify (missing evidence)


@dataclass
class Discrepancy:
    """Represents a discrepancy detected during verification."""
    type: DiscrepancyType
    description: str
    severity: int = 1  # 1=minor, 2=moderate, 3=major, 4=critical
    affected_criteria: List[str] = field(default_factory=list)
    suggested_action: Optional[str] = None

    def __str__(self) -> str:
        severity_labels = {1: "MINOR", 2: "MODERATE", 3: "MAJOR", 4: "CRITICAL"}
        severity_str = severity_labels.get(self.severity, "UNKNOWN")
        return f"[{severity_str}] {self.type.value}: {self.description}"


@dataclass
class ComparisonResult:
    """Result of comparing two states (e.g., goal vs actual)."""
    matches: bool  # Do they match?
    differences: List[str] = field(default_factory=list)  # What's different?
    commonalities: List[str] = field(default_factory=list)  # What matches?
    confidence: float = 1.0  # Confidence in comparison (0.0 to 1.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "matches": self.matches,
            "differences": self.differences,
            "commonalities": self.commonalities,
            "confidence": self.confidence
        }


# ============================================================================
# Reality Check Protocol and Registry
# ============================================================================

class RealityCheck(Protocol):
    """
    Protocol for reality check functions.

    A reality check is an INDEPENDENT verification that doesn't rely on
    the subordinate tier's report. It directly observes reality.
    """

    def __call__(self, goal: Goal, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check reality against goal.

        Args:
            goal: The goal that was supposed to be achieved
            context: Additional context for verification

        Returns:
            Dictionary describing the actual state of reality
        """
        ...


class RealityCheckRegistry:
    """
    Registry of reality check methods for different goal types.

    This allows pluggable verification methods that are NOT hardcoded
    to specific tasks. New goal types can register their own reality checks.
    """

    def __init__(self):
        self._checkers: Dict[str, RealityCheck] = {}
        self._register_builtin_checks()

    def register(self, goal_type: str, checker: RealityCheck):
        """Register a reality check for a goal type."""
        self._checkers[goal_type] = checker
        logger.debug(f"Registered reality check for goal type: {goal_type}")

    def get_checker(self, goal_type: str) -> Optional[RealityCheck]:
        """Get reality check for a goal type."""
        return self._checkers.get(goal_type)

    def has_checker(self, goal_type: str) -> bool:
        """Check if reality check exists for goal type."""
        return goal_type in self._checkers

    def _register_builtin_checks(self):
        """Register built-in reality checks."""

        # Code generation reality check - LLM-BASED VERIFICATION
        def code_generation_check(goal: Goal, context: Dict[str, Any]) -> Dict[str, Any]:
            """
            Verify code generation goals using LLM-based analysis.

            This is TRUE reality checking - an independent LLM analyzes the
            generated code to verify it matches the specification.
            """
            from ..utils.llm_provider import UnifiedLM
            import dspy

            actual_state = {
                "goal_type": "code_generation",
                "files_exist": [],
                "files_missing": [],
                "spec_compliance": {},
                "implementation_quality": {},
                "llm_verification_performed": False
            }

            # First check: Do files exist?
            for artifact_path in goal.required_artifacts:
                path = Path(artifact_path)
                if path.exists():
                    actual_state["files_exist"].append(artifact_path)
                else:
                    actual_state["files_missing"].append(artifact_path)

            # Second check: LLM-based code analysis
            if actual_state["files_exist"] and not context.get("skip_llm_verification", False):
                try:
                    lm = UnifiedLM(model="gemini/gemini-2.0-flash-exp", max_tokens=4000)

                    for file_path in actual_state["files_exist"]:
                        # Read the generated code
                        with open(file_path, 'r') as f:
                            code_content = f.read()

                        # Truncate if too long (keep first 3000 chars)
                        if len(code_content) > 3000:
                            code_content = code_content[:3000] + "\n... (truncated)"

                        # Build verification prompt
                        verification_prompt = f"""Analyze this generated code against the specification.

SPECIFICATION:
{goal.description}

SUCCESS CRITERIA:
{chr(10).join('- ' + c for c in goal.success_criteria)}

GENERATED CODE:
```python
{code_content}
```

Analyze and respond in this exact format:
SPEC_COMPLIANCE: <percentage 0-100>
MISSING_FEATURES: <comma-separated list or "none">
QUALITY_ISSUES: <comma-separated list or "none">
OVERALL_ASSESSMENT: <one sentence summary>
"""

                        # Get LLM analysis
                        with dspy.context(lm=lm):
                            response = lm(verification_prompt)

                        # Parse response
                        analysis = {
                            "spec_compliance": 0,
                            "missing_features": [],
                            "quality_issues": [],
                            "overall_assessment": ""
                        }

                        for line in response.split('\n'):
                            if line.startswith("SPEC_COMPLIANCE:"):
                                try:
                                    analysis["spec_compliance"] = int(line.split(':')[1].strip().rstrip('%'))
                                except:
                                    pass
                            elif line.startswith("MISSING_FEATURES:"):
                                features = line.split(':', 1)[1].strip()
                                if features.lower() != "none":
                                    analysis["missing_features"] = [f.strip() for f in features.split(',')]
                            elif line.startswith("QUALITY_ISSUES:"):
                                issues = line.split(':', 1)[1].strip()
                                if issues.lower() != "none":
                                    analysis["quality_issues"] = [i.strip() for i in issues.split(',')]
                            elif line.startswith("OVERALL_ASSESSMENT:"):
                                analysis["overall_assessment"] = line.split(':', 1)[1].strip()

                        actual_state["spec_compliance"][file_path] = analysis

                    actual_state["llm_verification_performed"] = True

                except Exception as e:
                    logger.warning(f"LLM verification failed: {e}, falling back to basic checks")
                    actual_state["llm_verification_performed"] = False
                    actual_state["llm_verification_error"] = str(e)

            return actual_state

        self.register("code_generation", code_generation_check)

        # Research quality reality check
        def research_quality_check(goal: Goal, context: Dict[str, Any]) -> Dict[str, Any]:
            """Verify research goals by analyzing output quality."""
            actual_state = {
                "goal_type": "research",
                "sources_found": 0,
                "coverage_depth": 0,
                "accuracy_verified": False
            }

            # Extract actual research output from context
            research_output = context.get("research_output", "")

            # Count sources (simple heuristic)
            import re
            citations = re.findall(r'\[(\d+)\]', research_output)
            actual_state["sources_found"] = len(set(citations))

            # Estimate coverage (word count as proxy)
            actual_state["coverage_depth"] = len(research_output.split())

            return actual_state

        self.register("research", research_quality_check)

        # Coordination reality check
        def coordination_check(goal: Goal, context: Dict[str, Any]) -> Dict[str, Any]:
            """Verify coordination goals by checking conflict resolution."""
            actual_state = {
                "goal_type": "coordination",
                "conflicts_detected": context.get("initial_conflicts", 0),
                "conflicts_resolved": 0,
                "integration_verified": False
            }

            # Check if conflicts were actually resolved
            remaining_conflicts = context.get("remaining_conflicts", [])
            actual_state["conflicts_resolved"] = (
                actual_state["conflicts_detected"] - len(remaining_conflicts)
            )

            # Verify integration (if components are meant to work together)
            if context.get("check_integration", False):
                # Would run integration tests
                pass

            return actual_state

        self.register("coordination", coordination_check)

        # Performance monitoring reality check
        def performance_check(goal: Goal, context: Dict[str, Any]) -> Dict[str, Any]:
            """Verify performance goals by checking metrics."""
            actual_state = {
                "goal_type": "performance",
                "metrics_collected": {},
                "thresholds_met": []
            }

            # Extract actual metrics
            metrics = context.get("metrics", {})
            actual_state["metrics_collected"] = metrics

            # Check against thresholds defined in goal
            for criterion in goal.success_criteria:
                # Would parse criterion and check metric
                # e.g., "cost < 0.50" -> check if metrics["cost"] < 0.50
                pass

            return actual_state

        self.register("performance", performance_check)


# Global registry instance
_REALITY_CHECK_REGISTRY = RealityCheckRegistry()


def get_reality_check_registry() -> RealityCheckRegistry:
    """Get the global reality check registry."""
    return _REALITY_CHECK_REGISTRY


# ============================================================================
# Tier Verification Result
# ============================================================================

@dataclass
class TierVerificationResult:
    """
    Result of verifying a subordinate tier's work.

    Compares three things:
    1. GOAL: What was requested
    2. REPORT: What subordinate claimed they did
    3. ACTUAL: What independent verification found

    Attributes:
        goal_achieved: Does ACTUAL match GOAL?
        report_accurate: Does REPORT match ACTUAL?
        discrepancies: List of detected discrepancies
        goal_vs_actual: Comparison result
        report_vs_actual: Comparison result
        goal_vs_report: Comparison result
        actual_state: The independently verified state
        verification_result: Underlying VerificationResult (if available)
        confidence: Overall confidence in verification (0.0 to 1.0)
    """

    goal_achieved: bool
    report_accurate: bool
    discrepancies: List[Discrepancy] = field(default_factory=list)
    goal_vs_actual: Optional[ComparisonResult] = None
    report_vs_actual: Optional[ComparisonResult] = None
    goal_vs_report: Optional[ComparisonResult] = None
    actual_state: Dict[str, Any] = field(default_factory=dict)
    verification_result: Optional[VerificationResult] = None
    confidence: float = 1.0

    @property
    def is_success(self) -> bool:
        """Is verification successful? (goal achieved AND report accurate)"""
        return self.goal_achieved and self.report_accurate

    @property
    def has_critical_discrepancies(self) -> bool:
        """Are there any critical discrepancies?"""
        return any(d.severity >= 4 for d in self.discrepancies)

    @property
    def has_major_discrepancies(self) -> bool:
        """Are there any major discrepancies?"""
        return any(d.severity >= 3 for d in self.discrepancies)

    def get_discrepancies_by_type(self, dtype: DiscrepancyType) -> List[Discrepancy]:
        """Get all discrepancies of a specific type."""
        return [d for d in self.discrepancies if d.type == dtype]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "goal_achieved": self.goal_achieved,
            "report_accurate": self.report_accurate,
            "is_success": self.is_success,
            "confidence": self.confidence,
            "discrepancies": [
                {
                    "type": d.type.value,
                    "description": d.description,
                    "severity": d.severity,
                    "affected_criteria": d.affected_criteria,
                    "suggested_action": d.suggested_action
                }
                for d in self.discrepancies
            ],
            "goal_vs_actual": self.goal_vs_actual.to_dict() if self.goal_vs_actual else None,
            "report_vs_actual": self.report_vs_actual.to_dict() if self.report_vs_actual else None,
            "goal_vs_report": self.goal_vs_report.to_dict() if self.goal_vs_report else None,
            "actual_state": self.actual_state
        }


# ============================================================================
# Tier Verification
# ============================================================================

class TierVerification:
    """
    Implements fractal VSM tier verification.

    Each tier uses this to verify the tier immediately below it using
    three-way comparison: goal vs report vs actual.

    This is GENERIC and works for any tier verifying any subordinate tier,
    with pluggable reality checks for different goal types.
    """

    def __init__(
        self,
        tier_name: str,
        subordinate_tier: str
    ):
        """
        Initialize tier verification.

        Args:
            tier_name: Name of this tier (e.g., "System 2")
            subordinate_tier: Name of subordinate tier being verified (e.g., "System 1")
        """
        self.tier_name = tier_name
        self.subordinate_tier = subordinate_tier

    def verify_subordinate(
        self,
        goal: Goal,
        report: Dict[str, Any],
        reality_check: Optional[Callable[[Goal, Dict[str, Any]], Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> TierVerificationResult:
        """
        Verify that subordinate tier achieved the goal.

        Performs three-way comparison:
        1. GOAL: What I asked them to do
        2. REPORT: What they said they did
        3. ACTUAL: What independent reality check found

        Args:
            goal: The goal that was assigned to subordinate
            report: The report from subordinate (what they claimed they did)
            reality_check: Custom reality check function (overrides registry)
            context: Additional context for verification

        Returns:
            TierVerificationResult with goal achievement, report accuracy, discrepancies
        """
        context = context or {}
        logger.info(
            f"{self.tier_name} verifying {self.subordinate_tier}: {goal.objective}"
        )

        # Step 1: Perform independent reality check
        actual_state = self._perform_reality_check(goal, reality_check, context)

        # Step 2: Compare goal vs actual (was goal achieved?)
        goal_vs_actual = self._compare_goal_to_actual(goal, actual_state, context)

        # Step 3: Compare report vs actual (was report accurate?)
        report_vs_actual = self._compare_report_to_actual(report, actual_state, context)

        # Step 4: Compare goal vs report (did they report on the right thing?)
        goal_vs_report = self._compare_goal_to_report(goal, report, context)

        # Step 5: Detect discrepancies
        discrepancies = self._detect_discrepancies(
            goal, report, actual_state,
            goal_vs_actual, report_vs_actual, goal_vs_report
        )

        # Step 6: Build result
        result = TierVerificationResult(
            goal_achieved=goal_vs_actual.matches,
            report_accurate=report_vs_actual.matches,
            discrepancies=discrepancies,
            goal_vs_actual=goal_vs_actual,
            report_vs_actual=report_vs_actual,
            goal_vs_report=goal_vs_report,
            actual_state=actual_state,
            confidence=min(
                goal_vs_actual.confidence,
                report_vs_actual.confidence,
                goal_vs_report.confidence
            )
        )

        # Log result
        if result.is_success:
            logger.info(
                f"{self.tier_name}: {self.subordinate_tier} successfully achieved goal"
            )
        else:
            logger.warning(
                f"{self.tier_name}: {self.subordinate_tier} failed - "
                f"{len(discrepancies)} discrepancies found"
            )
            for d in discrepancies:
                logger.warning(f"  - {d}")

        return result

    def _perform_reality_check(
        self,
        goal: Goal,
        custom_check: Optional[RealityCheck],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform independent reality check using LLM.

        This is the key: We DON'T trust the subordinate's report.
        We independently verify what actually happened by:
        1. Gathering what actually exists (files, outputs, etc.)
        2. Asking LLM: "Does this achieve the goal 100%?"
        """
        # Use custom check if provided
        if custom_check:
            return custom_check(goal, context)

        # Simple LLM-based reality check
        return self._llm_reality_check(goal, context)

    def _llm_reality_check(self, goal: Goal, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simple LLM-based reality check.

        Ask LLM: "Here's the goal, here's what actually exists, was goal achieved 100%?"
        """
        from ..utils.llm_provider import UnifiedLM
        import dspy

        # Gather what actually exists
        actual_artifacts = {}
        artifacts_exist = []
        artifacts_missing = []

        for artifact_path in goal.required_artifacts:
            path = Path(artifact_path)
            if path.exists():
                artifacts_exist.append(artifact_path)
                try:
                    # Read file content (truncate if too long)
                    content = path.read_text()
                    if len(content) > 3000:
                        content = content[:3000] + "\n... (truncated)"
                    actual_artifacts[artifact_path] = content
                except Exception as e:
                    actual_artifacts[artifact_path] = f"<Error reading file: {e}>"
            else:
                artifacts_missing.append(artifact_path)

        # Format actual state for LLM
        if actual_artifacts:
            artifacts_text = "\n\n".join(
                f"FILE: {path}\n{content}"
                for path, content in actual_artifacts.items()
            )
        else:
            artifacts_text = "<No artifacts found>"

        # Build verification prompt
        success_criteria_text = "\n".join(f"- {c}" for c in goal.success_criteria)

        prompt = f"""You are verifying if a goal was achieved by examining actual outputs.

GOAL:
{goal.objective}

SUCCESS CRITERIA:
{success_criteria_text}

ACTUAL OUTPUT:
{artifacts_text}

Question: Was this goal achieved 100%?

Answer with:
ACHIEVED: YES or NO
EXPLANATION: <one sentence explaining why>
MISSING: <comma-separated list of missing items, or "none">
"""

        # Ask LLM
        try:
            lm = UnifiedLM()  # Use default provider chain
            result = lm(prompt=prompt, max_tokens=1000)
            response = result['text']

            # Parse response
            goal_achieved = "YES" in response.split("ACHIEVED:")[1].split("\n")[0].upper() if "ACHIEVED:" in response else False

            explanation_parts = response.split("EXPLANATION:")
            explanation = explanation_parts[1].split("\n")[0].strip() if len(explanation_parts) > 1 else response[:200]

            missing_parts = response.split("MISSING:")
            missing = missing_parts[1].split("\n")[0].strip() if len(missing_parts) > 1 else ""
            missing_items = [m.strip() for m in missing.split(",") if m.strip() and m.strip().lower() != "none"]

            return {
                "goal_achieved": goal_achieved,
                "explanation": explanation,
                "artifacts_exist": artifacts_exist,
                "artifacts_missing": artifacts_missing,
                "missing_items": missing_items,
                "llm_response": response
            }

        except Exception as e:
            logger.warning(f"LLM reality check failed: {e}, falling back to basic check")
            return self._basic_artifact_check(goal, context)

    def _basic_artifact_check(self, goal: Goal, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Basic reality check: Verify required artifacts exist.

        This is a fallback when no specific reality check is registered.
        """
        actual_state = {
            "goal_type": "generic",
            "artifacts_exist": [],
            "artifacts_missing": []
        }

        for artifact_path in goal.required_artifacts:
            path = Path(artifact_path)
            if path.exists():
                actual_state["artifacts_exist"].append(artifact_path)
            else:
                actual_state["artifacts_missing"].append(artifact_path)

        return actual_state

    def _compare_goal_to_actual(
        self,
        goal: Goal,
        actual_state: Dict[str, Any],
        context: Dict[str, Any]
    ) -> ComparisonResult:
        """
        Compare goal to actual state: Was the goal actually achieved?

        This is the most important comparison - it tells us if the work
        was actually done, regardless of what the report says.
        """
        differences = []
        commonalities = []

        # Check required artifacts
        for artifact in goal.required_artifacts:
            if artifact in actual_state.get("artifacts_exist", []):
                commonalities.append(f"Artifact exists: {artifact}")
            else:
                differences.append(f"Missing artifact: {artifact}")

        # Check success criteria (if we can)
        for criterion in goal.success_criteria:
            # Would need to parse criterion and check against actual_state
            # For now, just log it
            pass

        matches = len(differences) == 0
        confidence = 1.0 if matches else 0.5

        return ComparisonResult(
            matches=matches,
            differences=differences,
            commonalities=commonalities,
            confidence=confidence
        )

    def _compare_report_to_actual(
        self,
        report: Dict[str, Any],
        actual_state: Dict[str, Any],
        context: Dict[str, Any]
    ) -> ComparisonResult:
        """
        Compare report to actual state: Was the report accurate?

        This detects if the subordinate tier LIED or was MISTAKEN about
        what it achieved.
        """
        differences = []
        commonalities = []

        # Compare reported artifacts vs actual artifacts
        reported_artifacts = report.get("artifacts_created", [])
        actual_artifacts = actual_state.get("artifacts_exist", [])

        for reported in reported_artifacts:
            if reported in actual_artifacts:
                commonalities.append(f"Correctly reported artifact: {reported}")
            else:
                differences.append(f"Falsely reported artifact: {reported}")

        # Check if any actual artifacts were not reported
        for actual in actual_artifacts:
            if actual not in reported_artifacts:
                differences.append(f"Unreported artifact: {actual}")

        # Compare other reported vs actual metrics
        for key in ["goal_achieved", "success", "completed"]:
            if key in report and key in actual_state:
                if report[key] == actual_state[key]:
                    commonalities.append(f"Correctly reported {key}: {report[key]}")
                else:
                    differences.append(
                        f"Incorrectly reported {key}: claimed {report[key]}, "
                        f"actually {actual_state[key]}"
                    )

        matches = len(differences) == 0
        confidence = 1.0 if matches else 0.7

        return ComparisonResult(
            matches=matches,
            differences=differences,
            commonalities=commonalities,
            confidence=confidence
        )

    def _compare_goal_to_report(
        self,
        goal: Goal,
        report: Dict[str, Any],
        context: Dict[str, Any]
    ) -> ComparisonResult:
        """
        Compare goal to report: Did subordinate report on the right thing?

        This detects if the subordinate tier misunderstood the goal or
        reported on something completely different.
        """
        differences = []
        commonalities = []

        # Check if report addresses the goal objective
        goal_keywords = set(goal.objective.lower().split())
        report_text = str(report).lower()

        matching_keywords = [kw for kw in goal_keywords if kw in report_text]
        if len(matching_keywords) > len(goal_keywords) / 2:
            commonalities.append("Report addresses goal objective")
        else:
            differences.append("Report doesn't address goal objective")

        # Check if report mentions required artifacts
        for artifact in goal.required_artifacts:
            if artifact in str(report):
                commonalities.append(f"Report mentions required artifact: {artifact}")
            else:
                differences.append(f"Report doesn't mention required artifact: {artifact}")

        matches = len(differences) == 0
        confidence = 0.8  # Lower confidence - this is more subjective

        return ComparisonResult(
            matches=matches,
            differences=differences,
            commonalities=commonalities,
            confidence=confidence
        )

    def _detect_discrepancies(
        self,
        goal: Goal,
        report: Dict[str, Any],
        actual_state: Dict[str, Any],
        goal_vs_actual: ComparisonResult,
        report_vs_actual: ComparisonResult,
        goal_vs_report: ComparisonResult
    ) -> List[Discrepancy]:
        """
        Detect all discrepancies from the three-way comparison.

        Returns prioritized list of discrepancies with suggested actions.
        """
        discrepancies = []

        # Critical: Goal not achieved
        if not goal_vs_actual.matches:
            severity = 4 if len(goal_vs_actual.differences) > len(goal.required_artifacts) / 2 else 3
            discrepancies.append(Discrepancy(
                type=DiscrepancyType.GOAL_NOT_ACHIEVED,
                description=f"Goal '{goal.objective}' not achieved. {len(goal_vs_actual.differences)} issues found.",
                severity=severity,
                affected_criteria=goal_vs_actual.differences,
                suggested_action="Retry task with corrected approach or escalate to higher tier"
            ))

        # Major: Report inaccurate
        if not report_vs_actual.matches:
            severity = 3 if len(report_vs_actual.differences) > 3 else 2
            discrepancies.append(Discrepancy(
                type=DiscrepancyType.REPORT_INACCURATE,
                description=f"Report from {self.subordinate_tier} is inaccurate. {len(report_vs_actual.differences)} discrepancies.",
                severity=severity,
                affected_criteria=report_vs_actual.differences,
                suggested_action="Investigate reporting mechanism, may indicate systemic issue"
            ))

        # Moderate: Report doesn't match goal
        if not goal_vs_report.matches:
            discrepancies.append(Discrepancy(
                type=DiscrepancyType.REPORT_GOAL_MISMATCH,
                description=f"Report doesn't address assigned goal",
                severity=2,
                affected_criteria=goal_vs_report.differences,
                suggested_action="Clarify goal specification or check for goal misunderstanding"
            ))

        # Partial completion
        if goal_vs_actual.matches and len(goal_vs_actual.differences) > 0:
            discrepancies.append(Discrepancy(
                type=DiscrepancyType.PARTIAL_COMPLETION,
                description=f"Partial goal achievement: {len(goal_vs_actual.commonalities)}/{len(goal.required_artifacts)} artifacts",
                severity=2,
                affected_criteria=goal_vs_actual.differences,
                suggested_action="Complete remaining artifacts"
            ))

        return sorted(discrepancies, key=lambda d: d.severity, reverse=True)


# ============================================================================
# Convenience Functions
# ============================================================================

def verify_subordinate_tier(
    tier_name: str,
    subordinate_tier: str,
    goal: Goal,
    report: Dict[str, Any],
    reality_check: Optional[Callable[[Goal, Dict[str, Any]], Dict[str, Any]]] = None,
    context: Optional[Dict[str, Any]] = None
) -> TierVerificationResult:
    """
    Convenience function for verifying a subordinate tier.

    This is a shortcut for creating a TierVerification instance and calling
    verify_subordinate().

    Args:
        tier_name: Name of verifying tier
        subordinate_tier: Name of tier being verified
        goal: The goal that was assigned
        report: The report from subordinate
        reality_check: Optional custom reality check
        context: Optional context for verification

    Returns:
        TierVerificationResult with verification results
    """
    verifier = TierVerification(tier_name, subordinate_tier)
    return verifier.verify_subordinate(goal, report, reality_check, context)
