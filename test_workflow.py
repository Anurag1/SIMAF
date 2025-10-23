#!/usr/bin/env python3
"""
Test Simple Research Workflow - Phase 0 Completion

Tests the linear LangGraph workflow: research → analyze → report
"""

from fractal_agent.workflows.research_workflow import run_research_workflow

print("=" * 80)
print("Phase 0: Simple Research Workflow Test")
print("=" * 80)
print()

# Run workflow on simple topic
result = run_research_workflow(
    topic="What is prompt caching?"
)

# Print final report
print(result["report"])

# Verify all stages completed
assert result["research_result"] is not None, "Research stage failed"
assert result["analysis"] is not None, "Analysis stage failed"
assert result["report"] is not None, "Report stage failed"

print()
print("=" * 80)
print("✓ Phase 0 Complete: All workflow stages executed successfully")
print("=" * 80)
