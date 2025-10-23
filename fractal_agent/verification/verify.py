"""
Goal-Based Verification System - Verification Logic

Implements DSPy-based verification that proves goals were achieved.

Key Components:
- GoalVerification: DSPy signature for semantic verification
- ArtifactChecker: Verifies files/directories exist and are valid
- EvidenceCollector: Gathers proof of goal completion
- verify_goal(): Main verification function

Author: BMad
Date: 2025-10-19
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import logging
import dspy

from .goals import (
    Goal,
    Evidence,
    VerificationResult,
    GoalStatus,
    VerificationMethod
)

logger = logging.getLogger(__name__)


# ============================================================================
# DSPy Verification Signature
# ============================================================================

class GoalVerification(dspy.Signature):
    """
    Verify that a goal was achieved based on evidence.

    This signature uses an LLM to semantically verify that:
    1. The evidence proves the goal was achieved
    2. All success criteria are met
    3. Any failures are identified with recommendations

    This is NOT "did the agent try?" - it's "did the agent SUCCEED?"
    """

    goal_objective = dspy.InputField(
        desc="The goal that should have been achieved"
    )
    success_criteria = dspy.InputField(
        desc="List of criteria that must be true for success"
    )
    evidence_summary = dspy.InputField(
        desc="Summary of evidence collected (artifacts, test results, etc.)"
    )

    # Outputs
    achieved = dspy.OutputField(
        desc="Whether the goal was achieved (true/false)"
    )
    confidence = dspy.OutputField(
        desc="Confidence score as decimal (0.0-1.0)"
    )
    reasoning = dspy.OutputField(
        desc="Detailed explanation of why goal was/wasn't achieved"
    )
    failures = dspy.OutputField(
        desc="List of success criteria that failed (comma-separated, or 'none')"
    )
    recommendations = dspy.OutputField(
        desc="Suggestions for fixing failures (comma-separated, or 'none')"
    )


# ============================================================================
# Artifact Checking
# ============================================================================

class ArtifactChecker:
    """
    Checks that required artifacts (files, directories) exist and are valid.
    """

    @staticmethod
    def check_file_exists(file_path: str) -> bool:
        """Check if a file exists."""
        path = Path(file_path)
        return path.exists() and path.is_file()

    @staticmethod
    def check_directory_exists(dir_path: str) -> bool:
        """Check if a directory exists."""
        path = Path(dir_path)
        return path.exists() and path.is_dir()

    @staticmethod
    def check_file_not_empty(file_path: str) -> bool:
        """Check if a file exists and is not empty."""
        path = Path(file_path)
        return path.exists() and path.is_file() and path.stat().st_size > 0

    @staticmethod
    def check_python_syntax(file_path: str) -> Dict[str, Any]:
        """
        Check if a Python file has valid syntax.

        Returns:
            Dict with 'valid' (bool) and 'error' (str or None)
        """
        import ast

        path = Path(file_path)
        if not path.exists():
            return {"valid": False, "error": f"File not found: {file_path}"}

        try:
            with open(path, "r") as f:
                code = f.read()
            ast.parse(code)
            return {"valid": True, "error": None}
        except SyntaxError as e:
            return {
                "valid": False,
                "error": f"Syntax error at line {e.lineno}: {e.msg}"
            }
        except Exception as e:
            return {"valid": False, "error": str(e)}

    @staticmethod
    def verify_artifacts(
        required_artifacts: List[str],
        language: str = "python"
    ) -> Dict[str, Any]:
        """
        Verify all required artifacts exist and are valid.

        Args:
            required_artifacts: List of file paths that should exist
            language: Programming language (for syntax checking)

        Returns:
            Dict with verification results
        """
        results = {
            "all_exist": True,
            "all_valid": True,
            "details": {},
            "missing": [],
            "invalid": []
        }

        for artifact in required_artifacts:
            # Check existence
            exists = ArtifactChecker.check_file_exists(artifact)
            not_empty = ArtifactChecker.check_file_not_empty(artifact) if exists else False

            detail = {
                "exists": exists,
                "not_empty": not_empty,
                "syntax_valid": None
            }

            if exists and language == "python" and artifact.endswith(".py"):
                # Check Python syntax
                syntax_result = ArtifactChecker.check_python_syntax(artifact)
                detail["syntax_valid"] = syntax_result["valid"]
                detail["syntax_error"] = syntax_result.get("error")

                if not syntax_result["valid"]:
                    results["all_valid"] = False
                    results["invalid"].append(artifact)

            if not exists:
                results["all_exist"] = False
                results["missing"].append(artifact)

            results["details"][artifact] = detail

        return results


# ============================================================================
# Evidence Collection
# ============================================================================

class EvidenceCollector:
    """
    Collects evidence of goal completion.
    """

    @staticmethod
    def collect_for_code_generation(
        output_path: Optional[str] = None,
        test_path: Optional[str] = None,
        language: str = "python",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Evidence:
        """
        Collect evidence for code generation goals.

        Args:
            output_path: Path to generated code
            test_path: Path to generated tests
            language: Programming language
            metadata: Additional metadata

        Returns:
            Evidence object with verification results
        """
        evidence = Evidence(metadata=metadata or {})

        # Collect artifacts
        if output_path:
            evidence.add_artifact(output_path)
        if test_path:
            evidence.add_artifact(test_path)

        # Verify all artifacts
        artifact_results = evidence.verify_all_artifacts()

        # Check syntax for Python files
        if language == "python":
            for artifact in evidence.artifacts_created:
                if artifact.endswith(".py"):
                    checker = ArtifactChecker()
                    syntax_result = checker.check_python_syntax(artifact)
                    evidence.metadata[f"syntax_{Path(artifact).name}"] = syntax_result

        # Store verification results
        evidence.metadata["artifact_verification"] = artifact_results

        return evidence

    @staticmethod
    def collect_for_research(
        synthesis: str,
        sources: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Evidence:
        """
        Collect evidence for research goals.

        Args:
            synthesis: Research synthesis text
            sources: List of sources consulted
            metadata: Additional metadata

        Returns:
            Evidence object
        """
        evidence = Evidence(metadata=metadata or {})

        # Research doesn't create files, but we track synthesis quality
        evidence.llm_observations = synthesis
        evidence.metadata["num_sources"] = len(sources)
        evidence.metadata["synthesis_word_count"] = len(synthesis.split())
        evidence.metadata["sources"] = sources

        return evidence


# ============================================================================
# Main Verification Function
# ============================================================================

def verify_goal(
    goal: Goal,
    evidence: Evidence,
    use_llm: bool = True
) -> VerificationResult:
    """
    Verify that a goal was achieved based on evidence.

    This is the main verification function that:
    1. Checks artifacts exist (if applicable)
    2. Uses DSPy to semantically verify goal completion
    3. Returns detailed VerificationResult

    Args:
        goal: The goal to verify
        evidence: Evidence collected
        use_llm: Whether to use LLM for semantic verification

    Returns:
        VerificationResult with status, score, reasoning, failures, recommendations
    """
    logger.info(f"Verifying goal: {goal.objective}")

    # Phase 1: Artifact verification (fast, deterministic)
    artifact_failures = []
    artifact_score = 1.0

    if goal.required_artifacts:
        artifact_results = ArtifactChecker.verify_artifacts(
            goal.required_artifacts,
            language=goal.context.get("language", "python")
        )

        if not artifact_results["all_exist"]:
            artifact_failures.append(
                f"Missing artifacts: {', '.join(artifact_results['missing'])}"
            )
            artifact_score -= 0.5

        if not artifact_results["all_valid"]:
            artifact_failures.append(
                f"Invalid artifacts: {', '.join(artifact_results['invalid'])}"
            )
            artifact_score -= 0.3

        # Store results in evidence
        evidence.metadata["artifact_verification"] = artifact_results

    # Phase 2: LLM-based semantic verification (if enabled)
    llm_failures = []
    llm_score = 1.0
    llm_reasoning = ""
    llm_recommendations = []

    if use_llm and goal.verification_method in [
        VerificationMethod.LLM_VERIFICATION,
        VerificationMethod.HYBRID
    ]:
        try:
            # Prepare evidence summary
            evidence_summary = _format_evidence_summary(evidence, goal)

            # Run DSPy verification
            verifier = dspy.Predict(GoalVerification)
            result = verifier(
                goal_objective=goal.objective,
                success_criteria="\n".join(
                    f"- {criterion}" for criterion in goal.success_criteria
                ),
                evidence_summary=evidence_summary
            )

            # Parse results
            llm_achieved = result.achieved.strip().lower() in ["true", "yes", "achieved"]
            llm_score = float(result.confidence) if result.confidence else 0.0
            llm_reasoning = result.reasoning

            # Parse failures
            failures_str = result.failures.strip().lower()
            if failures_str not in ["none", "n/a", ""]:
                llm_failures = [f.strip() for f in result.failures.split(",")]

            # Parse recommendations
            recommendations_str = result.recommendations.strip().lower()
            if recommendations_str not in ["none", "n/a", ""]:
                llm_recommendations = [r.strip() for r in result.recommendations.split(",")]

        except Exception as e:
            logger.error(f"LLM verification failed: {e}")
            llm_failures.append(f"LLM verification error: {e}")
            llm_score = 0.5  # Uncertain

    # Phase 3: Combine results
    all_failures = artifact_failures + llm_failures
    all_recommendations = llm_recommendations

    # Calculate final score
    if goal.verification_method == VerificationMethod.ARTIFACT_CHECK:
        final_score = artifact_score
    elif goal.verification_method == VerificationMethod.LLM_VERIFICATION:
        final_score = llm_score
    else:  # HYBRID or TEST_EXECUTION
        # Weight artifacts more heavily (60%) than LLM (40%)
        if use_llm:
            final_score = (artifact_score * 0.6) + (llm_score * 0.4)
        else:
            # When LLM disabled, rely solely on artifact verification
            final_score = artifact_score

    # Determine status
    if final_score >= 0.9:
        status = GoalStatus.ACHIEVED
    elif final_score >= 0.5:
        status = GoalStatus.PARTIALLY_ACHIEVED
    else:
        status = GoalStatus.FAILED

    # Build reasoning
    reasoning_parts = []
    if artifact_failures:
        reasoning_parts.append(f"Artifact check: {'; '.join(artifact_failures)}")
    elif goal.required_artifacts:
        reasoning_parts.append("All required artifacts exist and are valid")

    if llm_reasoning:
        reasoning_parts.append(f"LLM verification: {llm_reasoning}")

    final_reasoning = ". ".join(reasoning_parts)

    # Create result
    result = VerificationResult(
        goal=goal,
        evidence=evidence,
        status=status,
        score=final_score,
        reasoning=final_reasoning,
        failures=all_failures,
        recommendations=all_recommendations
    )

    logger.info(f"Verification result: {status.value} (score: {final_score:.2f})")

    return result


def _format_evidence_summary(evidence: Evidence, goal: Goal) -> str:
    """Format evidence into a summary for LLM verification."""
    parts = []

    # Artifacts
    if evidence.artifacts_created:
        parts.append(f"Artifacts claimed: {', '.join(evidence.artifacts_created)}")
        parts.append(f"Artifacts verified: {', '.join(evidence.artifacts_verified)}")

    # Test results
    if evidence.test_results:
        parts.append(f"Test results: {evidence.test_results}")

    # LLM observations
    if evidence.llm_observations:
        parts.append(f"Output: {evidence.llm_observations[:500]}")  # Truncate if long

    # Metadata
    if "artifact_verification" in evidence.metadata:
        av = evidence.metadata["artifact_verification"]
        parts.append(f"All artifacts exist: {av.get('all_exist', False)}")
        parts.append(f"All artifacts valid: {av.get('all_valid', False)}")

    return "\n".join(parts)


# ============================================================================
# Quick Test
# ============================================================================

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("=" * 80)
    print("TESTING GOAL-BASED VERIFICATION SYSTEM")
    print("=" * 80)
    print()

    # Test 1: Code generation goal with missing files
    print("Test 1: Code generation goal with missing files")
    print("-" * 80)

    from .goals import create_code_generation_goal

    goal = create_code_generation_goal(
        specification="Calculator class with add/subtract methods",
        output_path="/tmp/test_calculator.py",
        test_path="/tmp/test_calculator_test.py",
        language="python"
    )

    evidence = EvidenceCollector.collect_for_code_generation(
        output_path="/tmp/test_calculator.py",
        test_path="/tmp/test_calculator_test.py",
        language="python"
    )

    result = verify_goal(goal, evidence, use_llm=False)  # No LLM for now

    print(f"Status: {result.status.value}")
    print(f"Score: {result.score:.2f}")
    print(f"Reasoning: {result.reasoning}")
    print(f"Failures: {result.failures}")
    print()

    print("=" * 80)
    print("VERIFICATION SYSTEM TEST COMPLETE")
    print("=" * 80)
