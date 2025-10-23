"""
Goal-Based Verification System - Core Data Structures

Provides Goal, Evidence, and VerificationResult dataclasses for implementing
self-verifying workflows where agents prove they achieved their objectives.

Design Principle:
- Goals are explicit, measurable objectives
- Evidence is proof of goal achievement (file paths, test results, etc.)
- Verification confirms goals were achieved (not just attempted)

Author: BMad
Date: 2025-10-19
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from pathlib import Path
from enum import Enum


class VerificationMethod(Enum):
    """Methods for verifying goal completion."""
    ARTIFACT_CHECK = "artifact_check"  # Check files/directories exist
    LLM_VERIFICATION = "llm_verification"  # Use DSPy to verify semantically
    TEST_EXECUTION = "test_execution"  # Run tests and check they pass
    HYBRID = "hybrid"  # Combine multiple methods


class GoalStatus(Enum):
    """Status of goal achievement."""
    ACHIEVED = "achieved"  # Goal fully completed with evidence
    PARTIALLY_ACHIEVED = "partially_achieved"  # Some success, some failures
    FAILED = "failed"  # Goal not achieved
    UNVERIFIABLE = "unverifiable"  # Cannot verify (missing evidence)


@dataclass
class Goal:
    """
    Represents a verifiable goal for an agent.

    A goal is NOT "generate code" - that's an action.
    A goal IS "create working Python module at path X with tests that pass".

    Attributes:
        objective: Clear, measurable goal statement
        success_criteria: List of conditions that must be true
        required_artifacts: Files/directories that must exist
        verification_method: How to verify completion
        context: Additional context for verification (e.g., test command)
    """

    objective: str
    success_criteria: List[str] = field(default_factory=list)
    required_artifacts: List[str] = field(default_factory=list)
    verification_method: VerificationMethod = VerificationMethod.HYBRID
    context: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate goal configuration."""
        if not self.objective:
            raise ValueError("Goal must have an objective")

        # Ensure at least one success criterion
        if not self.success_criteria and not self.required_artifacts:
            raise ValueError(
                "Goal must have either success_criteria or required_artifacts"
            )


@dataclass
class Evidence:
    """
    Evidence of goal completion.

    This is PROOF that the goal was achieved, not just attempted.

    Attributes:
        artifacts_created: List of file paths that were created
        artifacts_verified: List of file paths that exist and are valid
        test_results: Results from running tests (if applicable)
        llm_observations: Semantic verification from LLM
        metadata: Additional evidence (tokens used, errors encountered, etc.)
    """

    artifacts_created: List[str] = field(default_factory=list)
    artifacts_verified: List[str] = field(default_factory=list)
    test_results: Optional[Dict[str, Any]] = None
    llm_observations: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_artifact(self, file_path: str, verified: bool = False):
        """Add an artifact to evidence."""
        self.artifacts_created.append(file_path)
        if verified:
            self.artifacts_verified.append(file_path)

    def verify_artifact(self, file_path: str) -> bool:
        """
        Verify that an artifact exists and is valid.

        Returns:
            True if artifact exists, False otherwise
        """
        path = Path(file_path)
        if path.exists():
            if file_path not in self.artifacts_verified:
                self.artifacts_verified.append(file_path)
            return True
        return False

    def verify_all_artifacts(self) -> Dict[str, bool]:
        """
        Verify all claimed artifacts exist.

        Returns:
            Dict mapping file paths to existence status
        """
        results = {}
        for artifact in self.artifacts_created:
            exists = self.verify_artifact(artifact)
            results[artifact] = exists
        return results


@dataclass
class VerificationResult:
    """
    Result of verifying a goal against evidence.

    Attributes:
        goal: The goal that was verified
        evidence: The evidence collected
        status: Whether goal was achieved
        score: Confidence score (0.0 - 1.0)
        reasoning: Explanation of verification result
        failures: List of criteria that failed
        recommendations: Suggestions for fixing failures
    """

    goal: Goal
    evidence: Evidence
    status: GoalStatus
    score: float = 0.0  # 0.0 = total failure, 1.0 = perfect success
    reasoning: str = ""
    failures: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate verification result."""
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"Score must be 0.0-1.0, got {self.score}")

    @property
    def is_success(self) -> bool:
        """Whether the goal was achieved."""
        return self.status == GoalStatus.ACHIEVED

    @property
    def needs_retry(self) -> bool:
        """Whether the goal should be retried."""
        return self.status in [GoalStatus.FAILED, GoalStatus.PARTIALLY_ACHIEVED]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "objective": self.goal.objective,
            "status": self.status.value,
            "score": self.score,
            "reasoning": self.reasoning,
            "artifacts_created": self.evidence.artifacts_created,
            "artifacts_verified": self.evidence.artifacts_verified,
            "failures": self.failures,
            "recommendations": self.recommendations
        }


# ============================================================================
# Helper Functions
# ============================================================================

def create_code_generation_goal(
    specification: str,
    output_path: str,
    test_path: Optional[str] = None,
    language: str = "python"
) -> Goal:
    """
    Create a goal for code generation tasks.

    Args:
        specification: What code to generate
        output_path: Where code should be written
        test_path: Where tests should be written (optional)
        language: Programming language

    Returns:
        Goal configured for code generation
    """
    required_artifacts = [output_path]
    success_criteria = [
        f"Source code file exists at {output_path}",
        f"Code is syntactically valid {language}",
        "Code implements the specification requirements"
    ]

    if test_path:
        required_artifacts.append(test_path)
        success_criteria.append(f"Test file exists at {test_path}")
        success_criteria.append("Tests are executable and pass")

    return Goal(
        objective=f"Generate working {language} code: {specification}",
        success_criteria=success_criteria,
        required_artifacts=required_artifacts,
        verification_method=VerificationMethod.HYBRID,
        context={
            "language": language,
            "output_path": output_path,
            "test_path": test_path,
            "specification": specification
        }
    )


def create_research_goal(
    topic: str,
    min_sources: int = 3,
    min_synthesis_words: int = 200
) -> Goal:
    """
    Create a goal for research tasks.

    Args:
        topic: Research topic
        min_sources: Minimum number of sources to consult
        min_synthesis_words: Minimum length of synthesis

    Returns:
        Goal configured for research
    """
    return Goal(
        objective=f"Research and synthesize information about: {topic}",
        success_criteria=[
            f"Consulted at least {min_sources} sources",
            f"Synthesis is at least {min_synthesis_words} words",
            "Synthesis addresses the research topic",
            "Sources are credible and relevant"
        ],
        required_artifacts=[],  # Research doesn't create files
        verification_method=VerificationMethod.LLM_VERIFICATION,
        context={
            "topic": topic,
            "min_sources": min_sources,
            "min_synthesis_words": min_synthesis_words
        }
    )
