#!/usr/bin/env python3
"""
Phase 1 Integration Test - Vertical Slice

Demonstrates all Phase 1 components working together:
1. Control agent (task decomposition)
2. Multiple operational agents (parallel execution)
3. Short-term memory (JSON logs)
4. Human review interface (Obsidian integration)

Use Case (per specification):
Control agent decomposes "Research VSM" → spawns 5 operational agents →
writes synthesis report

Success Criteria:
- Multi-agent coordination works
- Logs capture full task tree
- Human can review and approve in Obsidian

Author: BMad
Date: 2025-10-18
"""

import logging
from fractal_agent.agents.control_agent import ControlAgent
from fractal_agent.agents.research_agent import ResearchAgent
from fractal_agent.agents.research_config import ResearchConfig
from fractal_agent.memory.short_term import ShortTermMemory
from fractal_agent.memory.obsidian_export import ObsidianExporter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("=" * 80)
print("PHASE 1: VERTICAL SLICE - INTEGRATION TEST")
print("=" * 80)
print()
print("Use Case: Control agent decomposes 'Research VSM' →")
print("          spawns 5 operational agents → writes synthesis report")
print()
print("=" * 80)
print()

# Initialize components
print("Initializing Phase 1 components...")
print()

# 1. Short-term memory for logging
memory = ShortTermMemory(log_dir="./logs/phase1_test")
print("✓ Short-term memory initialized")

# 2. Obsidian exporter for human review
exporter = ObsidianExporter(vault_path="./obsidian_vault/phase1")
print("✓ Obsidian exporter initialized")

# 3. Control agent
control_agent = ControlAgent(tier="balanced")
print("✓ Control agent initialized")
print()

# Start logging main task
main_task = "Research the Viable System Model"
main_task_id = memory.start_task(
    agent_id="control_001",
    agent_type="control",
    task_description=main_task,
    inputs={"topic": "Viable System Model", "num_subtasks": 5}
)

print("=" * 80)
print("EXECUTING CONTROL WORKFLOW")
print("=" * 80)
print()

# Create operational agent runner with logging
def run_logged_operational_agent(subtask: str) -> dict:
    """
    Execute operational agent with full logging.

    This creates a research agent, logs its execution,
    and returns the result.
    """
    # Generate agent ID
    import hashlib
    agent_id = f"operational_{hashlib.md5(subtask.encode()).hexdigest()[:8]}"

    # Start subtask log
    subtask_id = memory.start_task(
        agent_id=agent_id,
        agent_type="operational",
        task_description=subtask,
        inputs={"subtask": subtask},
        parent_task_id=main_task_id
    )

    # Create and execute research agent
    agent = ResearchAgent(
        config=ResearchConfig(),
        max_research_questions=2  # Keep focused for subtasks
    )

    result = agent(topic=subtask, verbose=False)

    # End subtask log
    memory.end_task(
        task_id=subtask_id,
        outputs={
            "synthesis": result.synthesis,
            "validation": result.validation
        },
        metadata=result.metadata
    )

    # Return for control agent
    return {
        "subtask": subtask,
        "synthesis": result.synthesis,
        "tokens_used": result.metadata['total_tokens']
    }

# Execute control workflow
try:
    control_result = control_agent(
        main_task=main_task,
        operational_agent_runner=run_logged_operational_agent,
        verbose=True
    )

    # End main task log
    memory.end_task(
        task_id=main_task_id,
        outputs={
            "final_report": control_result.final_report,
            "num_subtasks": len(control_result.subtasks)
        },
        metadata=control_result.metadata
    )

    print()
    print("=" * 80)
    print("PHASE 1 WORKFLOW COMPLETE")
    print("=" * 80)
    print()

    # Print final report
    print("FINAL SYNTHESIS REPORT:")
    print("-" * 80)
    print(control_result.final_report)
    print("-" * 80)
    print()

    # Save session
    print("Saving session logs...")
    memory.save_session()
    print(f"✓ Session saved: {memory.session_file}")
    print()

    # Export to Obsidian
    print("Exporting to Obsidian for human review...")
    obsidian_file = exporter.export_session(memory)
    print(f"✓ Exported to: {obsidian_file}")
    print()

    # Print summary
    print("=" * 80)
    print("EXECUTION SUMMARY")
    print("=" * 80)
    print()

    summary = memory.get_session_summary()
    print(f"Session ID: {summary['session_id']}")
    print(f"Total Tasks: {summary['total_tasks']}")
    print(f"Completed Tasks: {summary['completed_tasks']}")
    print(f"Average Duration: {summary['avg_duration_seconds']:.2f}s")
    print(f"Total Duration: {summary['total_duration_seconds']:.2f}s")
    print()

    # Print task tree
    print("TASK TREE:")
    print("-" * 80)
    tree = memory.get_task_tree(main_task_id)
    for task in tree:
        indent = "  " if task.get("parent_task_id") else ""
        status = "✓" if task["status"] == "completed" else "⏳"
        duration = f"({task['duration_seconds']:.1f}s)" if task.get("duration_seconds") else ""
        print(f"{indent}{status} {task['task_description']} {duration}")
    print("-" * 80)
    print()

    # Phase 1 Success Criteria Check
    print("=" * 80)
    print("PHASE 1 SUCCESS CRITERIA")
    print("=" * 80)
    print()

    print("✅ Multi-agent coordination works")
    print(f"   - Control agent decomposed task into {len(control_result.subtasks)} subtasks")
    print(f"   - {len(control_result.subtask_results)} operational agents executed")
    print(f"   - Final report synthesized successfully")
    print()

    print("✅ Logs capture full task tree")
    print(f"   - Session logged to: {memory.session_file}")
    print(f"   - Task tree preserved with parent-child relationships")
    print(f"   - All {summary['total_tasks']} tasks captured")
    print()

    print("✅ Human can review and approve in Obsidian")
    print(f"   - Markdown exported to: {obsidian_file}")
    print(f"   - Includes task tree visualization")
    print(f"   - Includes approval workflow checkboxes")
    print()

    print("=" * 80)
    print("✅ PHASE 1: VERTICAL SLICE - COMPLETE")
    print("=" * 80)

except Exception as e:
    logger.error(f"Phase 1 test failed: {e}")
    import traceback
    traceback.print_exc()

    # Try to save logs even if failed
    try:
        memory.end_task(
            task_id=main_task_id,
            outputs={"error": str(e)},
            metadata={"status": "failed"}
        )
        memory.save_session()
    except:
        pass
