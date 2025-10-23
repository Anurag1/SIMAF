"""
Integration Test: Goal-Based Verification Fix

This test verifies that the system correctly reports when goals are NOT achieved
(e.g., when files aren't written to disk).

Before the fix:
- validation_passed = True (code string was syntactically valid)
- Files were NOT on disk
- System incorrectly reported success

After the fix:
- goal_achieved = False (required artifacts missing)
- Files are NOT on disk
- System correctly reports failure

Author: BMad
Date: 2025-10-19
"""

import os
import tempfile
from pathlib import Path

from fractal_agent.agents.developer_agent import DeveloperAgent
from fractal_agent.agents.developer_config import (
    PresetDeveloperConfigs,
    CodeGenerationTask
)


def test_goal_verification_detects_missing_files():
    """
    Test that goal verification correctly detects when files aren't written.

    This is the CORE BUG FIX:
    - Old system: Checked if code STRING was valid → always passed
    - New system: Checks if GOAL was achieved (files exist) → fails appropriately
    """
    print("=" * 80)
    print("TEST: Goal Verification Detects Missing Files")
    print("=" * 80)
    print()

    # Create temporary paths for code and tests
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "calculator.py"
        test_path = Path(tmpdir) / "test_calculator.py"

        print(f"Expected output path: {output_path}")
        print(f"Expected test path: {test_path}")
        print()

        # Create task WITHOUT file writing enabled (files won't be created)
        config = PresetDeveloperConfigs.quick_python()
        config.enable_file_writing = False  # Files will NOT be written
        config.max_iterations = 1  # Single iteration for speed

        agent = DeveloperAgent(config=config)

        task = CodeGenerationTask(
            specification="Create a Calculator class with add/subtract methods",
            language="python",
            output_path=str(output_path),
            test_path=str(test_path)
        )

        print("[1/3] Running DeveloperAgent WITHOUT file writing...")
        result = agent(task, verbose=False)

        print()
        print("=" * 80)
        print("VERIFICATION RESULTS")
        print("=" * 80)
        print(f"Code generated: {len(result.code)} characters")
        print(f"Tests generated: {'Yes' if result.tests else 'No'}")
        print()
        print(f"Files on disk:")
        print(f"  {output_path.name}: {'✅ EXISTS' if output_path.exists() else '❌ MISSING'}")
        print(f"  {test_path.name}: {'✅ EXISTS' if test_path.exists() else '❌ MISSING'}")
        print()

        # Check verification results
        print(f"Goal-Based Verification:")
        print(f"  goal_achieved: {result.goal_achieved}")
        print(f"  verification_score: {result.verification_result.score:.2f}")
        print(f"  status: {result.verification_result.status.value}")
        print()

        if result.verification_result.reasoning:
            print(f"  reasoning: {result.verification_result.reasoning}")

        if result.verification_result.failures:
            print(f"  failures:")
            for failure in result.verification_result.failures:
                print(f"    - {failure}")

        print()
        print("=" * 80)
        print("TEST ASSERTIONS")
        print("=" * 80)

        # CRITICAL ASSERTIONS - These verify the fix works
        assert not output_path.exists(), "File should NOT exist (file writing disabled)"
        assert not test_path.exists(), "Test file should NOT exist (file writing disabled)"

        print("✅ Files are NOT on disk (expected)")
        print()

        # OLD BEHAVIOR: validation_passed would be True (just checked syntax)
        # NEW BEHAVIOR: goal_achieved should be False (checks files exist)
        print(f"Old behavior (validation_passed): {result.validation_passed}")
        print(f"New behavior (goal_achieved): {result.goal_achieved}")
        print()

        if result.goal_achieved:
            print("❌ BUG NOT FIXED: goal_achieved=True when files don't exist!")
            print("   The system incorrectly reports success.")
            assert False, "Goal should NOT be achieved when files don't exist"
        else:
            print("✅ BUG FIXED: goal_achieved=False when files don't exist!")
            print("   The system correctly detects the failure.")

        # Verify evidence was collected
        assert result.evidence is not None, "Evidence should be collected"
        assert len(result.evidence.artifacts_created) > 0, "Should track expected artifacts"
        assert len(result.evidence.artifacts_verified) == 0, "No artifacts should be verified (none exist)"

        print(f"✅ Evidence collected: {len(result.evidence.artifacts_created)} artifacts tracked")
        print(f"✅ Verification correct: 0/{len(result.evidence.artifacts_created)} artifacts verified")

    print()
    print("=" * 80)
    print("TEST PASSED: Goal verification correctly detects missing files!")
    print("=" * 80)


def test_backward_compatibility():
    """
    Test that old code using validation_passed still works.

    The validation_passed property should be an alias for goal_achieved.
    """
    print("\n" + "=" * 80)
    print("TEST: Backward Compatibility")
    print("=" * 80)
    print()

    config = PresetDeveloperConfigs.quick_python()
    config.enable_file_writing = False
    config.max_iterations = 1

    agent = DeveloperAgent(config=config)

    with tempfile.TemporaryDirectory() as tmpdir:
        task = CodeGenerationTask(
            specification="Simple function",
            language="python",
            output_path=str(Path(tmpdir) / "test.py")
        )

        result = agent(task, verbose=False)

        # Verify backward compatibility
        assert result.validation_passed == result.goal_achieved, \
            "validation_passed should be an alias for goal_achieved"

        print(f"✅ validation_passed == goal_achieved: {result.validation_passed}")
        print(f"✅ Backward compatibility maintained")

    print()
    print("=" * 80)
    print("TEST PASSED: Backward compatibility works!")
    print("=" * 80)


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("GOAL-BASED VERIFICATION FIX - INTEGRATION TEST")
    print("=" * 80)
    print()
    print("This test verifies that the system now correctly reports when")
    print("goals are NOT achieved (e.g., when files aren't written to disk).")
    print()

    try:
        # Test 1: Verify goal verification detects missing files
        test_goal_verification_detects_missing_files()

        print("\n")

        # Test 2: Verify backward compatibility
        test_backward_compatibility()

        print("\n" + "=" * 80)
        print("ALL TESTS PASSED!")
        print("=" * 80)
        print()
        print("Summary:")
        print("  ✅ Goal verification correctly detects missing files")
        print("  ✅ System no longer reports false positives")
        print("  ✅ Backward compatibility maintained")
        print()
        print("The bug is FIXED!")
        print("=" * 80)

    except AssertionError as e:
        print("\n" + "=" * 80)
        print("TEST FAILED!")
        print("=" * 80)
        print(f"Error: {e}")
        print()
        print("The bug may not be fully fixed.")
        raise

    except Exception as e:
        print("\n" + "=" * 80)
        print("TEST ERROR!")
        print("=" * 80)
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        raise
