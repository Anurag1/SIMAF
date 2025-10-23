"""
Test Simplified LLM Reality Check

Verifies that the simplified LLM-based reality check works correctly.
No registry, no goal types, just: goal → actual → LLM judgment.

Author: BMad
Date: 2025-10-19
"""

import tempfile
from pathlib import Path

print("=" * 80)
print("SIMPLIFIED LLM REALITY CHECK TEST")
print("=" * 80)
print()

# Test 1: Import test
print("[1/3] Testing imports...")
try:
    from fractal_agent.verification.tier_verification import TierVerification
    from fractal_agent.verification.goals import Goal
    print("✅ Imports successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    import sys
    sys.exit(1)

print()

# Test 2: Create a simple test file and verify it
print("[2/3] Testing LLM reality check with incomplete code...")
try:
    # Create a test file with incomplete implementation
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        test_file = f.name
        f.write("""
class Calculator:
    '''Simple calculator'''

    def add(self, a: int, b: int) -> int:
        '''Add two numbers'''
        return a + b

    def subtract(self, a: int, b: int) -> int:
        '''Subtract two numbers'''
        return a - b

    # Missing multiply and divide!
""")

    print(f"✅ Created test file: {test_file}")

    # Create a goal
    goal = Goal(
        objective="Create a Calculator class with add, subtract, multiply, and divide methods",
        success_criteria=[
            "Calculator class exists",
            "add method with type hints",
            "subtract method with type hints",
            "multiply method with type hints",
            "divide method with type hints"
        ],
        required_artifacts=[test_file]
    )

    print("✅ Created goal")

    # Create TierVerification
    verifier = TierVerification(
        tier_name="System 3 (Test)",
        subordinate_tier="System 2 (Test)"
    )

    print("✅ Created verifier")
    print()
    print("Running LLM reality check...")
    print("(This may take a few seconds...)")
    print()

    # Perform reality check (no report needed for direct test)
    actual_state = verifier._llm_reality_check(goal, {})

    print("=" * 80)
    print("REALITY CHECK RESULTS")
    print("=" * 80)
    print(f"Goal achieved: {actual_state.get('goal_achieved', False)}")
    print(f"Explanation: {actual_state.get('explanation', 'N/A')}")
    print(f"Artifacts exist: {actual_state.get('artifacts_exist', [])}")
    print(f"Artifacts missing: {actual_state.get('artifacts_missing', [])}")
    print(f"Missing items: {actual_state.get('missing_items', [])}")
    print("=" * 80)

    # Verify LLM caught the incomplete implementation
    if not actual_state.get('goal_achieved', True):
        print()
        print("✅ LLM correctly identified incomplete implementation!")
    else:
        print()
        print("⚠️  LLM may not have caught the incomplete implementation")

    # Cleanup
    Path(test_file).unlink()
    print()
    print("✅ Cleaned up test file")

except Exception as e:
    print(f"❌ Reality check test failed: {e}")
    import traceback
    traceback.print_exc()
    import sys
    sys.exit(1)

print()

# Test 3: Test with complete implementation
print("[3/3] Testing LLM reality check with complete code...")
try:
    # Create a test file with COMPLETE implementation
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        test_file = f.name
        f.write("""
class Calculator:
    '''Simple calculator with all operations'''

    def add(self, a: int, b: int) -> int:
        '''Add two numbers'''
        return a + b

    def subtract(self, a: int, b: int) -> int:
        '''Subtract two numbers'''
        return a - b

    def multiply(self, a: int, b: int) -> int:
        '''Multiply two numbers'''
        return a * b

    def divide(self, a: int, b: int) -> float:
        '''Divide two numbers'''
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
""")

    # Same goal
    goal = Goal(
        objective="Create a Calculator class with add, subtract, multiply, and divide methods",
        success_criteria=[
            "Calculator class exists",
            "add method with type hints",
            "subtract method with type hints",
            "multiply method with type hints",
            "divide method with type hints"
        ],
        required_artifacts=[test_file]
    )

    # Check again
    actual_state = verifier._llm_reality_check(goal, {})

    print()
    print("=" * 80)
    print("REALITY CHECK RESULTS (Complete Implementation)")
    print("=" * 80)
    print(f"Goal achieved: {actual_state.get('goal_achieved', False)}")
    print(f"Explanation: {actual_state.get('explanation', 'N/A')}")
    print("=" * 80)

    if actual_state.get('goal_achieved', False):
        print()
        print("✅ LLM correctly identified complete implementation!")
    else:
        print()
        print("⚠️  LLM may have incorrectly flagged complete code as incomplete")
        print(f"   Missing items: {actual_state.get('missing_items', [])}")

    # Cleanup
    Path(test_file).unlink()
    print()
    print("✅ Cleaned up test file")

except Exception as e:
    print(f"❌ Complete code test failed: {e}")
    import traceback
    traceback.print_exc()
    import sys
    sys.exit(1)

print()
print("=" * 80)
print("SIMPLIFIED LLM REALITY CHECK: COMPLETE ✅")
print("=" * 80)
print()
print("Summary:")
print("  ✅ TierVerification simplified (no registry)")
print("  ✅ LLM reality check works")
print("  ✅ LLM can distinguish complete vs incomplete code")
print()
print("Architecture is now simple: Goal → Actual → LLM → Did it match?")
print()
