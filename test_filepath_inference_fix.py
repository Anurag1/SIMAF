#!/usr/bin/env python3
"""
Test FilePathInference Fix

This test verifies that the CoordinationAgent can automatically infer
file paths from task descriptions WITHOUT requiring explicit paths.

This proves the fix for the Phase 5 file persistence issue.
"""

import sys
from pathlib import Path

print("=" * 80)
print("TESTING: FilePathInference Fix")
print("=" * 80)
print()
print("Goal: Prove that file paths are automatically inferred from task descriptions")
print("Task: 'Implement a simple greeting utility'")
print("Expected: Code generated AND written to file WITHOUT explicit path in prompt")
print()
print("=" * 80)
print()

# Clean up any previous test file
test_file = Path("fractal_agent/utils/greeting_utility.py")
if test_file.exists():
    test_file.unlink()
    print(f"Cleaned up previous test file: {test_file}")
    print()

# Run the test
from fractal_agent.workflows.intelligence_workflow import run_user_task

result = run_user_task(
    user_task="""
    Implement a simple greeting utility.

    Requirements:
    - Create a function called greet(name: str) that returns a greeting message
    - Include a docstring explaining what the function does
    - Keep it simple and minimal
    """,
    verify_control=True,
    verbose=True
)

print()
print("=" * 80)
print("TEST RESULTS")
print("=" * 80)
print()

# Check results
print(f"Goal achieved: {result.get('goal_achieved', False)}")
print()

# Check if file was written
if test_file.exists():
    file_size = test_file.stat().st_size
    print(f"✅ SUCCESS: File written to {test_file}")
    print(f"   File size: {file_size} bytes")
    print()
    print("File contents:")
    print("-" * 80)
    print(test_file.read_text())
    print("-" * 80)
    print()
    print("✅ PROOF: FilePathInference fix works!")
    print("   - NO explicit file path in task description")
    print("   - CoordinationAgent inferred path: fractal_agent/utils/greeting_utility.py")
    print("   - DeveloperAgent wrote file successfully")
    sys.exit(0)
else:
    print(f"❌ FAILURE: File NOT written to {test_file}")
    print()
    print("Result details:")
    print(result)
    sys.exit(1)
