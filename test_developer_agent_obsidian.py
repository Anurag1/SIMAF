"""
Test: DeveloperAgent Generates ObsidianVault Class for Phase 4

This test demonstrates the system can build itself by generating
a production-quality component for Phase 4 (Coordination & Human Review).

Test Case: Generate ObsidianVault class

Success Criteria:
- Generates valid Python code (parseable)
- Follows project patterns
- Includes docstrings and type hints
- Passes validation
- Generates comprehensive tests

Author: BMad
Date: 2025-10-19
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from fractal_agent.agents.developer_agent import DeveloperAgent
from fractal_agent.agents.developer_config import (
    CodeGenerationTask,
    PresetDeveloperConfigs
)


def load_context_code():
    """Load existing code as context for the agent."""
    # Read existing ObsidianExporter to provide pattern examples
    context_file = Path(__file__).parent / "fractal_agent/memory/obsidian_export.py"

    with open(context_file, 'r') as f:
        context = f.read()

    # Include first 2000 chars as example pattern
    return context[:2000]


def main():
    print("=" * 80)
    print("Testing DeveloperAgent on ObsidianVault Class (Phase 4 Component)")
    print("=" * 80)
    print()

    # Load context
    print("[1/3] Loading context code...")
    context_code = load_context_code()
    print(f"  ✅ Loaded {len(context_code)} chars from obsidian_export.py")
    print()

    # Create agent
    print("[2/3] Creating DeveloperAgent...")
    config = PresetDeveloperConfigs.thorough_python()
    agent = DeveloperAgent(config=config)
    print(f"  ✅ Agent created (mode={config.mode}, tier={config.tier})")
    print()

    # Create task
    specification = """
Create production-quality ObsidianVault class for Phase 4 human review workflow.

REQUIREMENTS:
1. Class name: ObsidianVault
2. Purpose: Manage Obsidian vault structure for coordinated human review
3. Core features:
   - __init__(vault_path: str) - Initialize vault at specified path
   - initialize_vault() -> None - Create folder structure:
     * Inbox/ (new items for review)
     * Knowledge/ (reviewed and organized)
     * Archive/ (completed items)
     * Templates/ (document templates)
   - create_from_template(template_name: str, content: Dict) -> Path
     Create file from template with metadata
   - move_to_reviewed(file_path: Path, category: str) -> Path
     Move file from Inbox to Knowledge with categorization
   - sync_to_git(commit_message: str) -> bool
     Sync vault to Git repository with commit
   - list_pending() -> List[Path]
     List all files in Inbox awaiting review
   - export_metadata(file_path: Path) -> Dict
     Extract frontmatter metadata from markdown file

TECHNICAL REQUIREMENTS:
- Use pathlib.Path for all file operations
- Include comprehensive Google-style docstrings
- Include type hints on all methods
- Follow PEP 8 style guidelines
- Handle errors gracefully with try/except
- Use logging for important operations
- Support YAML frontmatter in markdown files

DEPENDENCIES:
- pathlib (stdlib)
- logging (stdlib)
- subprocess (for Git operations)
- yaml (for frontmatter parsing)

Follow the patterns from the example code provided in context.
"""

    task = CodeGenerationTask(
        specification=specification,
        language="python",
        context=f"EXAMPLE CODE PATTERN:\n```python\n{context_code}\n```"
    )

    # Generate code
    print("[3/3] Generating ObsidianVault class...")
    print("  This will call Claude/Gemini with thorough mode...")
    print()

    result = agent(task, verbose=True)

    print()
    print("=" * 80)
    print("Generated Code:")
    print("=" * 80)
    print(result.code)
    print("=" * 80)
    print()

    if result.tests:
        print("Generated Tests:")
        print("=" * 80)
        print(result.tests)
        print("=" * 80)
        print()

    # Results summary
    print("=" * 80)
    print("RESULTS:")
    print("=" * 80)
    print(f"  Validation: {'✅ Passed' if result.validation_passed else '❌ Failed'}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Tests generated: {'✅ Yes' if result.tests else '❌ No'}")
    print(f"  Tier: {result.metadata.get('tier')}")
    print(f"  Mode: {result.metadata.get('mode')}")

    if result.validation_details:
        print(f"\n  Validation Details:")
        for key, value in result.validation_details.items():
            print(f"    {key}: {value}")

    print()
    print("=" * 80)
    print("VERDICT:")
    print("=" * 80)

    if result.validation_passed and result.tests:
        print("✅ SUCCESS: DeveloperAgent generated production-quality Phase 4 component!")
        print()
        print("Meta-Learning Demonstration:")
        print("  - System successfully built a component for its own Phase 4")
        print("  - Code passes validation")
        print("  - Comprehensive tests generated")
        print("  - Ready for integration into Phase 4 workflow")
        print()
        print("Next Steps:")
        print("  1. Review generated code")
        print("  2. Save to fractal_agent/memory/obsidian_vault.py")
        print("  3. Run generated tests")
        print("  4. Integrate into Phase 4 coordination workflow")
        return 0
    elif result.validation_passed:
        print("⚠️  PARTIAL: Code is valid but tests not generated")
        return 1
    else:
        print("❌ FAIL: Code validation failed")
        print(f"  Details: {result.validation_details}")
        return 2


if __name__ == "__main__":
    exit(main())
