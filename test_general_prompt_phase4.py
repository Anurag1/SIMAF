"""
Test: General Prompt for Phase 4 Self-Implementation

This test provides a minimal, general prompt to see if the agents can:
1. Read PHASE4_PLAN.md themselves
2. Decompose the task appropriately
3. Understand "implement into project" means writing files
4. Actually integrate code into the fractal_agent project

No explicit instructions - testing agent self-organization capability.

Author: BMad
Date: 2025-10-19
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from fractal_agent.workflows.multi_agent_workflow import run_multi_agent_workflow


def test_general_prompt():
    """
    Test with minimal, general prompt.

    Let the agents figure out:
    - How to read the specification
    - What needs to be implemented
    - Where to put the code
    - How to integrate it
    """
    print("=" * 80)
    print("TEST: General Prompt for Phase 4 Self-Implementation")
    print("=" * 80)
    print()
    print("Approach: Give minimal instructions, let agents self-organize")
    print()

    # Simple, general prompt - just point to spec file
    task = """
Read the specification at PHASE4_PLAN.md and implement the ObsidianVault
component into the fractal_agent project.
"""

    print("Task prompt:")
    print(task)
    print()
    print("=" * 80)
    print()

    # Run workflow
    result = run_multi_agent_workflow(main_task=task)

    print()
    print("=" * 80)
    print("POST-EXECUTION VERIFICATION")
    print("=" * 80)
    print()

    # Check if files were created
    impl_file = Path(__file__).parent / "fractal_agent/memory/obsidian_vault.py"
    test_file = Path(__file__).parent / "tests/unit/test_obsidian_vault.py"

    impl_exists = impl_file.exists()
    test_exists = test_file.exists()

    print(f"Implementation file created: {'✅ Yes' if impl_exists else '❌ No'}")
    if impl_exists:
        print(f"  Path: {impl_file}")
        print(f"  Size: {impl_file.stat().st_size} bytes")

    print()
    print(f"Test file created: {'✅ Yes' if test_exists else '❌ No'}")
    if test_exists:
        print(f"  Path: {test_file}")
        print(f"  Size: {test_file.stat().st_size} bytes")

    print()
    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print()

    # Analyze agent behavior
    if impl_exists and test_exists:
        print("✅ SUCCESS: Agents self-organized and implemented!")
        print()
        print("Agents demonstrated ability to:")
        print("  - Read specification file themselves")
        print("  - Decompose into appropriate subtasks")
        print("  - Understand 'implement' means writing files")
        print("  - Create files in correct project locations")
        print()
        return 0

    elif result['control_result']:
        print("⚠️  PARTIAL: Agents executed but files not created")
        print()
        print("Let's review what the agents did:")
        print()

        for i, sr in enumerate(result['control_result'].subtask_results, 1):
            agent_type = sr['result'].get('agent_type', 'Unknown')
            subtask = sr['subtask']
            print(f"{i}. {agent_type}: {subtask[:80]}...")

        print()
        print("Possible explanations:")
        print("  - Agents generated code but didn't save files")
        print("  - File paths not inferred from general prompt")
        print("  - Need to add file-writing capability to DeveloperAgent")
        print()
        return 1

    else:
        print("❌ FAILED: Workflow did not execute properly")
        return 2


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("=" * 80)
    print("TESTING AGENT SELF-ORGANIZATION FROM GENERAL PROMPT")
    print("=" * 80)
    print()
    print("Question: Can agents implement Phase 4 from minimal instructions?")
    print()
    print("Previous approach: Overly specific, step-by-step instructions")
    print("Current approach: General prompt, let agents figure it out")
    print()
    print("This tests whether agents can parse and apply general requirements.")
    print()
    print("=" * 80)
    print()

    exit_code = test_general_prompt()

    exit(exit_code)
