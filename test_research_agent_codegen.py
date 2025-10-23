"""
Test: Can ResearchAgent Generate Production-Quality Code?

This test evaluates whether the existing ResearchAgent can be used for
code generation tasks, or if we need a specialized DeveloperAgent.

Test Case: Generate ObsidianVault class for Phase 4

Success Criteria:
- Generates valid Python code (parseable)
- Follows project patterns (similar to existing code)
- Includes docstrings and type hints
- Includes tests
- Quality score >= 80%

Author: BMad
Date: 2025-10-19
"""

import ast
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from fractal_agent.agents.research_agent import ResearchAgent
from fractal_agent.agents.research_config import ResearchConfig

def load_context_code():
    """Load existing code as context for the agent."""
    # Read existing ObsidianExporter to provide pattern examples
    context_file = Path(__file__).parent / "fractal_agent/memory/obsidian_export.py"

    with open(context_file, 'r') as f:
        context = f.read()

    return context

def validate_syntax(code: str) -> tuple[bool, str]:
    """Validate Python syntax."""
    try:
        ast.parse(code)
        return True, "✅ Syntax valid"
    except SyntaxError as e:
        return False, f"❌ Syntax error: {e}"

def check_code_quality(code: str) -> dict:
    """Basic quality checks."""
    checks = {
        "has_class": "class Obsidian" in code,
        "has_docstring": '"""' in code or "'''" in code,
        "has_type_hints": ": " in code and "->" in code,
        "has_init": "def __init__" in code,
        "has_methods": code.count("def ") >= 3,  # At least 3 methods
        "uses_pathlib": "Path" in code,
        "proper_length": 100 < len(code.split("\n")) < 500  # Reasonable size
    }

    score = sum(checks.values()) / len(checks)

    return {
        "score": score,
        "checks": checks,
        "passed": score >= 0.8
    }

def main():
    print("=" * 80)
    print("Testing ResearchAgent for Code Generation")
    print("=" * 80)
    print()

    # Load context
    print("[1/4] Loading context code...")
    context_code = load_context_code()
    print(f"  ✅ Loaded {len(context_code)} chars from obsidian_export.py")
    print()

    # Create research agent
    print("[2/4] Creating ResearchAgent...")
    agent = ResearchAgent(
        config=ResearchConfig(),
        max_research_questions=1  # Keep focused
    )
    print("  ✅ Agent created")
    print()

    # Generate code
    print("[3/4] Generating ObsidianVault class...")
    print("  This will call Claude/Gemini...")
    print()

    prompt = f"""
Generate production-quality Python code for an ObsidianVault class.

REQUIREMENTS:
- Class name: ObsidianVault
- Purpose: Manage Obsidian vault structure for Phase 4 human review workflow
- Features needed:
  1. __init__(vault_path: str) - Initialize vault structure
  2. initialize_vault() - Create folders (Inbox, Knowledge, Archive, Templates)
  3. create_from_template(template_name, content) - Create file from template
  4. move_to_reviewed(file_path) - Move file from pending to reviewed
  5. sync_to_git(commit_message) - Sync vault to Git repository

- Use pathlib.Path for file operations
- Include comprehensive docstrings
- Include type hints
- Follow patterns from this existing code:

EXAMPLE CODE PATTERN:
```python
{context_code[:1000]}  # First 1000 chars as example
```

IMPORTANT:
- Generate ONLY the Python code
- Include imports at the top
- Make it production-ready with error handling
- Follow PEP 8 style guidelines

Generate the complete ObsidianVault class implementation now.
"""

    try:
        result = agent(topic=prompt, verbose=True)

        print()
        print("=" * 80)
        print("Generated Code:")
        print("=" * 80)
        print(result.synthesis)
        print("=" * 80)
        print()

        # Extract code from result (might be embedded in markdown)
        code = result.synthesis

        # Try to extract code blocks if wrapped in markdown
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[1].split("```")[0]

        # Validate syntax
        print("[4/4] Validating generated code...")
        print()

        syntax_valid, syntax_msg = validate_syntax(code)
        print(f"  Syntax: {syntax_msg}")

        # Quality checks
        quality_result = check_code_quality(code)
        print(f"\n  Quality Score: {quality_result['score']:.0%}")
        print(f"\n  Quality Checks:")
        for check, passed in quality_result['checks'].items():
            status = "✅" if passed else "❌"
            print(f"    {status} {check}")

        # Final verdict
        print()
        print("=" * 80)
        print("VERDICT:")
        print("=" * 80)

        if syntax_valid and quality_result['passed']:
            print("✅ SUCCESS: ResearchAgent CAN generate production code!")
            print()
            print("Recommendation: RENAME ResearchAgent → OperationalAgent")
            print("  - Keep single agent for all operational tasks")
            print("  - Agent is already language-agnostic and generic")
            print("  - Add validation pipeline (next step)")
            return 0
        elif syntax_valid:
            print("⚠️  PARTIAL: ResearchAgent generates valid code but quality is low")
            print()
            print("Recommendation: BUILD specialized DeveloperAgent")
            print("  - ResearchAgent for research tasks")
            print("  - DeveloperAgent for code generation")
            print("  - Both use same validation pipeline")
            return 1
        else:
            print("❌ FAIL: ResearchAgent cannot generate valid code")
            print()
            print("Recommendation: BUILD DeveloperAgent with code-specific prompts")
            return 2

    except Exception as e:
        print(f"❌ Error during code generation: {e}")
        import traceback
        traceback.print_exc()
        return 3

if __name__ == "__main__":
    exit(main())
