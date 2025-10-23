"""
Goal-Based Verification System

Provides infrastructure for self-verifying workflows where agents prove
they achieved their objectives rather than just attempting them.

Core Components:
- Goal, Evidence, VerificationResult: Data structures for goals and verification
- verify_goal(): Main verification function using DSPy
- ArtifactChecker: Validates files/directories exist
- EvidenceCollector: Gathers proof of completion

Usage:
    from fractal_agent.verification import (
        Goal,
        Evidence,
        VerificationResult,
        verify_goal,
        create_code_generation_goal
    )

    # Create a goal
    goal = create_code_generation_goal(
        specification="Calculator class",
        output_path="/tmp/calculator.py",
        test_path="/tmp/test_calculator.py"
    )

    # Collect evidence
    evidence = EvidenceCollector.collect_for_code_generation(
        output_path="/tmp/calculator.py",
        test_path="/tmp/test_calculator.py"
    )

    # Verify goal was achieved
    result = verify_goal(goal, evidence)

    if result.is_success:
        print("Goal achieved!")
    else:
        print(f"Goal failed: {result.reasoning}")

Author: BMad
Date: 2025-10-19
"""

from .goals import (
    Goal,
    Evidence,
    VerificationResult,
    GoalStatus,
    VerificationMethod,
    create_code_generation_goal,
    create_research_goal
)

from .verify import (
    verify_goal,
    GoalVerification,
    ArtifactChecker,
    EvidenceCollector
)

from .tier_verification import (
    TierVerification,
    TierVerificationResult,
    Discrepancy,
    DiscrepancyType,
    ComparisonResult,
    RealityCheckRegistry,
    get_reality_check_registry,
    verify_subordinate_tier
)

__all__ = [
    # Data structures
    "Goal",
    "Evidence",
    "VerificationResult",
    "GoalStatus",
    "VerificationMethod",

    # Verification
    "verify_goal",
    "GoalVerification",
    "ArtifactChecker",
    "EvidenceCollector",

    # Tier Verification (NEW - Fractal VSM)
    "TierVerification",
    "TierVerificationResult",
    "Discrepancy",
    "DiscrepancyType",
    "ComparisonResult",
    "RealityCheckRegistry",
    "get_reality_check_registry",
    "verify_subordinate_tier",

    # Helpers
    "create_code_generation_goal",
    "create_research_goal"
]
