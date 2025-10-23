"""
Multi-Agent Coordination Workflow - VSM Tier Hierarchy Compliant

REFACTORED FOR FRACTAL VSM ARCHITECTURE

Implements proper VSM tier adjacency:
- System 3 (Control) → System 2 (Coordination) → System 1 (Operational)
- Each tier only interacts with adjacent tiers
- Each tier verifies its immediate subordinate using TierVerification

Architecture:
1. ControlAgent (S3) decomposes tasks into subtasks
2. CoordinationAgent (S2) orchestrates System 1 agents and verifies results
3. System 1 agents (Developer, Research) execute tasks
4. Verification flows upward: S1 → S2 → S3

Key Changes from Previous Version:
- REMOVED: Direct Control → Operational agent routing (violated tier adjacency)
- ADDED: Control → Coordination → Operational flow (proper VSM hierarchy)
- ADDED: TierVerification at every level
- MOVED: Task classification and routing to CoordinationAgent

Author: BMad
Date: 2025-10-19 (Refactored for VSM compliance)
"""

from typing import TypedDict, List, Dict, Any, Optional
from pathlib import Path
from langgraph.graph import StateGraph, END
from ..agents.control_agent import ControlAgent, ControlResult
from ..agents.coordination_agent import CoordinationAgent, CoordinationConfig
from ..memory.short_term import ShortTermMemory
from ..observability import (
    get_correlation_id, get_tracer, get_logger,
    get_event_store, VSMEvent, set_span_attributes
)
import logging
import concurrent.futures
import dspy

# Use observability-aware structured logger
logger = get_logger(__name__)

# Note: Observability auto-initializes when fractal_agent.observability is imported
# See fractal_agent/observability/__init__.py for auto-initialization code


class MultiAgentState(TypedDict):
    """
    State for multi-agent coordination workflow.

    Attributes:
        main_task: The complex task to execute
        control_result: Output from control agent
        final_report: Final synthesized report
        memory: Short-term memory instance for logging
        workflow_task_id: Task ID for the main workflow task
    """
    main_task: str
    control_result: ControlResult | None
    final_report: str | None
    memory: ShortTermMemory | None
    workflow_task_id: str | None


def control_decomposition_node(state: MultiAgentState) -> MultiAgentState:
    """
    Node 1: Control Agent decomposes task and delegates to Coordination Layer.

    NEW VSM-COMPLIANT ARCHITECTURE:
    1. ControlAgent (S3) decomposes main task into subtasks
    2. ControlAgent delegates subtasks to CoordinationAgent (S2) - NOT directly to System 1!
    3. CoordinationAgent orchestrates System 1 agents and verifies results
    4. ControlAgent synthesizes verified results into final report

    This enforces tier adjacency: S3 → S2 → S1 (no tier skipping)
    """
    # OBSERVABILITY: Get tracer and event store
    tracer = get_tracer(__name__)
    event_store = get_event_store()
    correlation_id = get_correlation_id()

    with tracer.start_as_current_span("system3_control_decomposition") as span:
        # OBSERVABILITY: Set span attributes
        set_span_attributes({
            "vsm.tier": "System3_Control",
            "vsm.operation": "control_decomposition",
            "task.description": state['main_task'][:100],
            "correlation_id": correlation_id
        })

        # OBSERVABILITY: Emit control started event
        event_store.append(VSMEvent(
            tier="System3",
            event_type="control_decomposition_started",
            data={
                "main_task": state['main_task'],
                "correlation_id": correlation_id
            }
        ))

        logger.info(
            f"Control node (System 3): {state['main_task']}",
            extra={"correlation_id": correlation_id}
        )

        # Track control decomposition in memory if available
        memory = state.get("memory")
        control_task_id = None
        if memory:
            control_task_id = memory.start_task(
                agent_id="system3_control_agent",
                agent_type="control",
                task_description=f"Decompose and synthesize: {state['main_task']}",
                inputs={"main_task": state["main_task"]}
            )
            logger.info(
                f"Started control decomposition task: {control_task_id}",
                extra={"correlation_id": correlation_id, "task_id": control_task_id}
            )

        # Create control agent (System 3)
        control_agent = ControlAgent(tier="balanced")

        # Create coordination agent (System 2)
        coordination_config = CoordinationConfig(
            tier="balanced",
            require_verification=True,
            enable_consensus_building=True
        )
        coordination_agent = CoordinationAgent(config=coordination_config)

        # Create coordination runner that delegates to System 2
        def run_coordination_layer(subtask: str) -> Dict[str, Any]:
            """
            Delegate to System 2 (CoordinationAgent) for execution and verification.

            NEW BEHAVIOR:
            - System 3 no longer talks directly to System 1 agents
            - System 3 delegates to System 2 (CoordinationAgent)
            - System 2 handles routing, execution, and verification of System 1
            - System 3 receives verified results from System 2

            Args:
                subtask: Single subtask to execute

            Returns:
                Dict with coordination results including tier verification
            """
            # OBSERVABILITY: Emit delegation to S2 event
            event_store.append(VSMEvent(
                tier="System3",
                event_type="delegating_to_s2",
                data={
                    "subtask": subtask,
                    "correlation_id": correlation_id
                }
            ))

            logger.info(
                f"Coordination layer (System 2) handling: {subtask}",
                extra={"correlation_id": correlation_id, "subtask": subtask}
            )

            # Delegate to System 2 for orchestration and verification
            coordination_result = coordination_agent.orchestrate_subtasks(
                subtasks=[subtask],  # Single subtask per call (Control expects 1:1 mapping)
                context={},
                verbose=False
            )

            # Extract result from System 2's coordination
            if coordination_result["agent_outputs"]:
                agent_output = coordination_result["agent_outputs"][0]
                tier_verification = coordination_result["tier_verification_results"][0]

                # Build result summary for System 3
                goal_status = "✅ Achieved" if tier_verification.goal_achieved else "❌ Not Achieved"
                report_status = "accurate" if tier_verification.report_accurate else "inaccurate"

                return {
                    "subtask": subtask,
                    "agent_type": agent_output.agent_type,
                    "output": agent_output.content,
                    "metadata": agent_output.metadata,
                    "goal_achieved": tier_verification.goal_achieved,
                    "report_accurate": tier_verification.report_accurate,
                    "discrepancies": len(tier_verification.discrepancies),
                    "result_summary": f"{agent_output.agent_type} completed, goal: {goal_status}, report: {report_status}"
                }
            else:
                return {
                    "subtask": subtask,
                    "error": "No agent output from coordination layer"
                }

        # Execute control workflow (ControlAgent delegates to System 2, not System 1)
        control_result = control_agent(
            main_task=state["main_task"],
            operational_agent_runner=run_coordination_layer,  # Delegates to S2, not S1!
            verbose=True
        )

        # Track control completion in memory if available
        if memory and control_task_id:
            memory.end_task(
                task_id=control_task_id,
                outputs={
                    "num_subtasks": len(control_result.subtasks),
                    "final_report": control_result.final_report,
                    "all_goals_achieved": all(
                        sr.get("result", {}).get("goal_achieved", False)
                        for sr in control_result.subtask_results
                    )
                },
                metadata={
                    "control_tier": "balanced",
                    "subtask_count": len(control_result.subtasks),
                    "control_metrics": control_result.metadata
                }
            )
            logger.info(
                f"Completed control decomposition task: {control_task_id}",
                extra={"correlation_id": correlation_id, "task_id": control_task_id}
            )

        # OBSERVABILITY: Emit control completion event
        event_store.append(VSMEvent(
            tier="System3",
            event_type="control_decomposition_completed",
            data={
                "num_subtasks": len(control_result.subtasks),
                "final_report_length": len(control_result.final_report),
                "correlation_id": correlation_id
            }
        ))

        logger.info(
            f"System 3 completed: {len(control_result.subtasks)} subtasks processed",
            extra={"correlation_id": correlation_id, "num_subtasks": len(control_result.subtasks)}
        )

        # Update state
        state["control_result"] = control_result
        state["final_report"] = control_result.final_report

        return state


def create_multi_agent_workflow() -> StateGraph:
    """
    Create the multi-agent coordination workflow (VSM-compliant).

    NEW ARCHITECTURE:
    - System 3 (Control) → System 2 (Coordination) → System 1 (Operational)
    - Each tier verifies its immediate subordinate
    - No tier skipping

    Returns:
        Compiled LangGraph StateGraph
    """
    # Create graph
    workflow = StateGraph(MultiAgentState)

    # Add control node (handles decomposition, delegation, synthesis)
    workflow.add_node("control", control_decomposition_node)

    # Simple linear flow (control does everything)
    workflow.set_entry_point("control")
    workflow.add_edge("control", END)

    # Compile
    return workflow.compile()


def run_multi_agent_workflow(main_task: str, memory: Optional[ShortTermMemory] = None) -> Dict[str, Any]:
    """
    Run the multi-agent coordination workflow (VSM-compliant) with memory logging.

    NEW VSM ARCHITECTURE:
    1. System 3 (Control) decomposes task into subtasks
    2. System 2 (Coordination) orchestrates System 1 agents and verifies results
    3. System 1 (Operational) executes tasks (Developer, Research)
    4. Each tier verifies its subordinate using TierVerification

    MEMORY INTEGRATION:
    - Logs all task executions to ShortTermMemory
    - Creates session logs in ./logs/sessions/
    - Enables performance monitoring and Intelligence Layer analysis

    Args:
        main_task: Complex task requiring multi-agent collaboration
        memory: ShortTermMemory instance for logging (creates new if None)

    Returns:
        Final workflow state with synthesized report and tier verification results
    """
    print("=" * 80)
    print("MULTI-AGENT WORKFLOW - VSM TIER HIERARCHY COMPLIANT")
    print("=" * 80)
    print(f"\nMain Task: {main_task}")
    print(f"\nArchitecture: System 3 (Control) → System 2 (Coordination) → System 1 (Operational)")
    print("=" * 80)
    print()

    # Initialize memory if not provided
    if memory is None:
        memory = ShortTermMemory()
        logger.info(f"Created new ShortTermMemory session: {memory.session_id}")
    else:
        logger.info(f"Using existing ShortTermMemory session: {memory.session_id}")

    # Log workflow start
    workflow_task_id = memory.start_task(
        agent_id="system3_control_workflow",
        agent_type="control",
        task_description=main_task,
        inputs={"main_task": main_task}
    )
    logger.info(f"Started workflow task: {workflow_task_id}")

    # Create workflow
    app = create_multi_agent_workflow()

    # Initialize state with memory
    initial_state = {
        "main_task": main_task,
        "control_result": None,
        "final_report": None,
        "memory": memory,
        "workflow_task_id": workflow_task_id
    }

    # Execute workflow
    final_state = app.invoke(initial_state)

    # Log workflow completion
    memory.end_task(
        task_id=workflow_task_id,
        outputs={
            "final_report": final_state["final_report"],
            "num_subtasks": len(final_state["control_result"].subtasks) if final_state["control_result"] else 0
        },
        metadata={
            "workflow_type": "multi_agent",
            "control_metrics": final_state["control_result"].metadata if final_state["control_result"] else {}
        }
    )
    logger.info(f"Completed workflow task: {workflow_task_id}")

    # End session (saves to JSON and auto-exports to Obsidian if enabled)
    export_path = memory.end_session(
        async_export=True,  # Non-blocking background export
        include_approval=True
    )
    logger.info(f"Ended session: {memory.session_id}")

    # Log export status
    export_status = memory.get_export_status()
    if export_status['enabled']:
        logger.info(f"Obsidian export status: {export_status['status']}")

    print()
    print("=" * 80)
    print("WORKFLOW COMPLETE")
    print("=" * 80)
    print()
    print("FINAL REPORT:")
    print("=" * 80)
    print(final_state["final_report"])
    print("=" * 80)
    print()
    print(f"Session logged to: {memory.session_file}")
    print("=" * 80)

    return final_state


# Quick test
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("=" * 80)
    print("TESTING VSM TIER HIERARCHY (S3 → S2 → S1)")
    print("=" * 80)
    print()
    print("This test demonstrates:")
    print("- System 3 (ControlAgent) decomposing complex tasks")
    print("- System 2 (CoordinationAgent) orchestrating System 1 agents")
    print("- System 1 agents (Research/Developer) executing tasks")
    print("- TierVerification at each level (fractal VSM pattern)")
    print("- Proper tier adjacency (no tier skipping)")
    print()
    print("=" * 80)
    print()

    # Run a mixed task (research + implementation)
    result = run_multi_agent_workflow(
        main_task="Create a simple Calculator class in Python with add, subtract, multiply, divide methods. Include type hints and docstrings."
    )

    print()
    print("=" * 80)
    print("EXECUTION SUMMARY")
    print("=" * 80)
    print(f"Subtasks executed: {len(result['control_result'].subtasks)}")
    print(f"Total operational agents: {len(result['control_result'].subtask_results)}")
    print()
    print("Agent Routing:")
    for i, sr in enumerate(result['control_result'].subtask_results, 1):
        agent_type = sr['result'].get('agent_type', 'Unknown')
        summary = sr['result'].get('result_summary', 'No summary')
        print(f"  {i}. {agent_type}: {summary}")
    print("=" * 80)
