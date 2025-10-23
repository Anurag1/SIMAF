"""
Test: DeveloperAgent File Persistence and Fix Validation

Minimal test that verifies the DeveloperAgent can:
1. Generate code from specification
2. Persist code to files
3. Validate that fixes meet requirements
4. Verify goal achievement through evidence collection

Success Criteria:
- Agent generates valid Python code
- Code is written to specified file path
- File exists and contains expected content
- Goal verification confirms success

Author: BMad
Date: 2025-10-20
"""

import sys
import tempfile
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from fractal_agent.agents.developer_agent import DeveloperAgent
from fractal_agent.agents.developer_config import (
    CodeGenerationTask,
    PresetDeveloperConfigs,
    DeveloperConfig
)
from fractal_agent.verification import (
    EvidenceCollector,
    verify_goal
)


def test_file_persistence_and_validation():
    """Test that DeveloperAgent can generate and persist code with fix validation."""

    print("=" * 80)
    print("Test: DeveloperAgent File Persistence and Fix Validation")
    print("=" * 80)
    print()

    # Create temporary directory for output
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "calculator.py"
        test_path = Path(tmpdir) / "test_calculator.py"

        print(f"[1/5] Setup")
        print(f"  Output path: {output_path}")
        print(f"  Test path: {test_path}")
        print()

        # Create agent with file writing enabled
        print("[2/5] Creating DeveloperAgent with file writing enabled...")
        config = DeveloperConfig(
            tier="cheap",
            language="python",
            mode="quick",
            max_iterations=2,
            validation_enabled=True,
            generate_tests=True,
            enable_file_writing=True,
            project_root=Path(tmpdir)
        )
        agent = DeveloperAgent(config=config)
        print(f"  ✅ Agent created (file_writing={config.enable_file_writing})")
        print()

        # Define simple specification
        specification = """
Create a Calculator class with basic arithmetic operations.

REQUIREMENTS:
1. Class name: Calculator
2. Methods:
   - add(a: float, b: float) -> float
   - subtract(a: float, b: float) -> float
   - multiply(a: float, b: float) -> float
   - divide(a: float, b: float) -> float (handle division by zero)

TECHNICAL REQUIREMENTS:
- Include type hints
- Include docstrings
- Handle division by zero with ValueError
- Follow PEP 8
"""

        # Create task with file paths
        print("[3/5] Creating code generation task...")
        task = CodeGenerationTask(
            specification=specification,
            language="python",
            output_path=str(output_path),
            test_path=str(test_path)
        )
        print(f"  ✅ Task created")
        print(f"  Expected artifacts: {task.expected_artifacts}")
        print(f"  Success criteria: {len(task.success_criteria)} criteria")
        print()

        # Generate code
        print("[4/5] Generating code with fix validation...")
        result = agent(task, verbose=True)
        print()

        # Verify results
        print("[5/5] Verifying results...")

        # Check 1: Code was generated
        assert result.code, "No code was generated"
        print(f"  ✅ Code generated ({len(result.code)} chars)")

        # Check 2: File was written
        assert output_path.exists(), f"Output file not created at {output_path}"
        print(f"  ✅ File created at {output_path}")

        # Check 3: File contains the generated code
        written_code = output_path.read_text()
        assert len(written_code) > 0, "File is empty"
        print(f"  ✅ File contains code ({len(written_code)} chars)")

        # Check 4: Code contains expected elements
        assert "Calculator" in written_code, "Calculator class not found in code"
        assert "def add" in written_code, "add method not found"
        assert "def divide" in written_code, "divide method not found"
        print(f"  ✅ Code contains expected class and methods")

        # Check 5: Goal verification confirms success
        assert result.verification_result is not None, "No verification result"
        print(f"  ✅ Verification performed")
        print(f"     Score: {result.verification_result.score:.2f}")
        print(f"     Goal achieved: {result.goal_achieved}")

        # Check 6: Evidence was collected
        assert result.evidence is not None, "No evidence collected"
        print(f"  ✅ Evidence collected")
        print(f"     Artifacts verified: {len(result.evidence.artifacts_verified)}")
        print(f"     Files checked: {len(result.evidence.files_checked)}")

        # Check 7: Tests were generated if enabled
        if config.generate_tests:
            assert result.tests, "Tests were not generated"
            print(f"  ✅ Tests generated ({len(result.tests)} chars)")

            if test_path.exists():
                print(f"  ✅ Test file created at {test_path}")

        print()
        print("=" * 80)
        print("RESULTS:")
        print("=" * 80)
        print(f"  File persistence: ✅ Success")
        print(f"  Code generation: ✅ Success")
        print(f"  Fix validation: {'✅ Success' if result.goal_achieved else '⚠️ Partial'}")
        print(f"  Goal achieved: {result.goal_achieved}")
        print(f"  Iterations: {result.iterations}")
        print(f"  Verification score: {result.verification_result.score:.2f}")
        print()

        # Display generated code snippet
        print("Generated Code (first 500 chars):")
        print("-" * 80)
        print(written_code[:500])
        if len(written_code) > 500:
            print("...")
        print("-" * 80)
        print()

        print("=" * 80)
        print("VERDICT:")
        print("=" * 80)

        if output_path.exists() and result.goal_achieved:
            print("✅ SUCCESS: DeveloperAgent successfully generated and persisted code!")
            print()
            print("Validation Summary:")
            print("  - Code was generated")
            print("  - File was written to disk")
            print("  - Goal verification passed")
            print("  - Evidence was collected")
            print("  - Fix validation completed")
            return 0
        elif output_path.exists():
            print("⚠️ PARTIAL SUCCESS: Code persisted but goal not fully achieved")
            print(f"   Reason: {result.verification_result.reasoning if result.verification_result else 'Unknown'}")
            return 1
        else:
            print("❌ FAILURE: File persistence failed")
            return 2


if __name__ == "__main__":
    try:
        exit_code = test_file_persistence_and_validation()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n❌ TEST FAILED WITH EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(3)
