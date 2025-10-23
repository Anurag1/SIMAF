"""
Coordination Workflow - LangGraph Integration for System 2 (Coordination Layer)

FRACTAL VSM ARCHITECTURE - SYSTEM 2 COORDINATION

This workflow implements System 2 (Coordination) using LangGraph for orchestrating
multiple System 1 agents with conditional routing based on task complexity and conflicts.

Architecture:
- System 3 (Control) delegates subtasks to this workflow
- System 2 (Coordination) routes to System 1 agents (Developer, Research)
- Conditional routing based on:
  * Number of agents (single vs multi-agent)
  * Conflict detection (conflicts present or not)
  * Task complexity (simple vs complex coordination)
- TierVerification at coordination level (S2 verifies S1)

Workflow Nodes:
1. analyze_task: Analyze incoming task and determine routing strategy
2. execute_agents: Execute System 1 agents (single or multiple)
3. detect_conflicts: Check for conflicts between agent outputs (conditional)
4. resolve_conflicts: Resolve detected conflicts (conditional)
5. verify_results: Verify System 1 results using TierVerification
6. build_report: Generate final coordination report

Conditional Edges:
- analyze_task → execute_agents (always)
- execute_agents → detect_conflicts (if multi-agent) OR verify_results (if single)
- detect_conflicts → resolve_conflicts (if conflicts) OR verify_results (if no conflicts)
- resolve_conflicts → verify_results (always)
- verify_results → build_report (always)

Author: BMad
Date: 2025-10-19
"""

from typing import TypedDict, List, Dict, Any, Optional, Literal
from pathlib import Path
from langgraph.graph import StateGraph, END
from ..agents.coordination_agent import (
    CoordinationAgent,
    CoordinationConfig,
    AgentOutput,
    Conflict,
    Resolution,
    CoordinationResult
)
from ..memory.short_term import ShortTermMemory
from ..verification import TierVerification, TierVerificationResult, Goal
import logging
import json

logger = logging.getLogger(__name__)


class CoordinationWorkflowState(TypedDict):
    """
    State for coordination workflow.

    Attributes:
        subtasks: List of subtasks to coordinate
        context: Additional context (resources, dependencies, constraints)
        coordination_agent: CoordinationAgent instance
        agent_outputs: Outputs from System 1 agents
        conflicts_detected: Detected conflicts between agents
        resolutions: Proposed/implemented conflict resolutions
        tier_verification_results: Verification results for each System 1 agent
        coordination_result: Overall coordination result
        final_report: Final coordination report
        metadata: Execution metadata
        memory: ShortTermMemory instance for logging
        task_id: Task ID for memory logging
        routing_strategy: Strategy for routing (single, multi-agent, complex)
        requires_conflict_resolution: Whether conflict resolution is needed
    """
    subtasks: List[str]
    context: Dict[str, Any]
    coordination_agent: CoordinationAgent
    agent_outputs: List[AgentOutput]
    conflicts_detected: List[Conflict]
    resolutions: List[Resolution]
    tier_verification_results: List[TierVerificationResult]
    coordination_result: Optional[CoordinationResult]
    final_report: str
    metadata: Dict[str, Any]
    memory: Optional[ShortTermMemory]
    task_id: Optional[str]
    routing_strategy: str
    requires_conflict_resolution: bool


def analyze_task_node(state: CoordinationWorkflowState) -> CoordinationWorkflowState:
    """
    Node 1: Analyze Task and Determine Routing Strategy

    Analyzes incoming subtasks to determine optimal routing strategy:
    - single: Single subtask, direct execution
    - multi-agent: Multiple subtasks, parallel execution
    - complex: Complex dependencies, requires careful coordination
    """
    logger.info(f"Analyze task: {len(state['subtasks'])} subtask(s)")

    num_subtasks = len(state['subtasks'])

    if num_subtasks == 1:
        routing_strategy = "single"
    elif num_subtasks <= 3:
        routing_strategy = "multi-agent"
    else:
        routing_strategy = "complex"

    state["routing_strategy"] = routing_strategy
    state["requires_conflict_resolution"] = False

    logger.info(f"Routing strategy: {routing_strategy}")

    return state


def execute_agents_node(state: CoordinationWorkflowState) -> CoordinationWorkflowState:
    """
    Node 2: Execute System 1 Agents

    Routes subtasks to appropriate System 1 agents (Developer/Research)
    and collects their outputs.
    """
    logger.info(f"Execute agents: {len(state['subtasks'])} subtask(s)")

    agent_outputs = []

    for i, subtask in enumerate(state['subtasks'], 1):
        logger.info(f"Executing subtask {i}/{len(state['subtasks'])}: {subtask[:80]}...")

        agent_output = state['coordination_agent']._route_and_execute_system1_agent(
            subtask=subtask,
            context=state['context'],
            verbose=False
        )
        agent_outputs.append(agent_output)

        logger.info(f"Completed: {agent_output.agent_type}")

    state["agent_outputs"] = agent_outputs

    return state


def detect_conflicts_node(state: CoordinationWorkflowState) -> CoordinationWorkflowState:
    """
    Node 3: Detect Conflicts Between Agent Outputs

    Analyzes System 1 agent outputs to identify conflicts that need resolution.
    Only runs for multi-agent scenarios.
    """
    logger.info(f"Detect conflicts: {len(state['agent_outputs'])} agent output(s)")

    if len(state['agent_outputs']) < 2:
        state["conflicts_detected"] = []
        state["requires_conflict_resolution"] = False
        logger.info("No conflicts possible (single agent)")
        return state

    conflicts = state['coordination_agent'].detect_conflicts(
        agent_outputs=state['agent_outputs'],
        context=state['context'],
        verbose=False
    )

    state["conflicts_detected"] = conflicts
    state["requires_conflict_resolution"] = len(conflicts) > 0

    logger.info(f"Conflicts detected: {len(conflicts)}")

    return state


def resolve_conflicts_node(state: CoordinationWorkflowState) -> CoordinationWorkflowState:
    """
    Node 4: Resolve Conflicts (Conditional)

    Resolves detected conflicts using CoordinationAgent's resolution strategies.
    Only runs if conflicts are detected.
    """
    logger.info(f"Resolve conflicts: {len(state['conflicts_detected'])} conflict(s)")

    resolutions = []

    for conflict in state['conflicts_detected']:
        logger.info(f"Resolving conflict: {conflict.conflict_id} ({conflict.severity.value})")

        resolution = state['coordination_agent'].resolve_conflict(
            conflict=conflict,
            agent_outputs=state['agent_outputs'],
            verbose=False
        )
        resolutions.append(resolution)

        logger.info(f"Resolution strategy: {resolution.strategy.value}")

    state["resolutions"] = resolutions

    if state['coordination_agent'].config.enable_consensus_building and len(resolutions) > 0:
        logger.info("Building consensus...")
        consensus_reached = state['coordination_agent'].build_consensus(
            agent_outputs=state['agent_outputs'],
            conflicts=state['conflicts_detected'],
            resolutions=resolutions,
            verbose=False
        )
        state["metadata"]["consensus_reached"] = consensus_reached
        logger.info(f"Consensus reached: {consensus_reached}")

    return state


def verify_results_node(state: CoordinationWorkflowState) -> CoordinationWorkflowState:
    """
    Node 5: Verify System 1 Results Using TierVerification

    Verifies each System 1 agent result using three-way comparison:
    - GOAL: What System 2 asked System 1 to do (subtask)
    - REPORT: What System 1 said it did (agent metadata)
    - ACTUAL: What actually happened (reality check)
    """
    logger.info(f"Verify results: {len(state['agent_outputs'])} agent output(s)")

    tier_verification_results = []

    for i, agent_output in enumerate(state['agent_outputs']):
        logger.info(f"Verifying {i+1}/{len(state['agent_outputs'])}: {agent_output.agent_type}...")

        verification = state['coordination_agent']._verify_system1_result(
            agent_output=agent_output,
            subtask=state['subtasks'][i],
            verbose=False
        )
        tier_verification_results.append(verification)

        logger.info(f"Goal achieved: {verification.goal_achieved}, Report accurate: {verification.report_accurate}")

    state["tier_verification_results"] = tier_verification_results

    coordination_result = CoordinationResult(
        conflicts_detected=state['conflicts_detected'],
        resolutions=state.get('resolutions', []),
        consensus_reached=state['metadata'].get('consensus_reached', True),
        verification_results=[],
        metadata={
            "routing_strategy": state['routing_strategy'],
            "num_subtasks": len(state['subtasks']),
            "num_agents": len(state['agent_outputs']),
            "num_conflicts": len(state['conflicts_detected']),
            "num_resolutions": len(state.get('resolutions', [])),
            "all_verified": all(v.goal_achieved for v in tier_verification_results)
        }
    )

    state["coordination_result"] = coordination_result

    return state


def build_report_node(state: CoordinationWorkflowState) -> CoordinationWorkflowState:
    """
    Node 6: Build Final Coordination Report

    Generates comprehensive report of coordination results including:
    - Agent executions
    - Conflict detection/resolution
    - Tier verification results
    - Overall success status
    """
    logger.info("Build report")

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("COORDINATION WORKFLOW REPORT (SYSTEM 2)")
    report_lines.append("=" * 80)
    report_lines.append("")

    report_lines.append(f"Subtasks Coordinated: {len(state['subtasks'])}")
    report_lines.append(f"Routing Strategy: {state['routing_strategy']}")
    report_lines.append("")

    report_lines.append("=" * 80)
    report_lines.append("SYSTEM 1 AGENT EXECUTION")
    report_lines.append("=" * 80)
    report_lines.append("")

    for i, (subtask, agent_output) in enumerate(zip(state['subtasks'], state['agent_outputs']), 1):
        report_lines.append(f"{i}. Subtask: {subtask}")
        report_lines.append(f"   Agent: {agent_output.agent_type}")
        report_lines.append(f"   Output Type: {agent_output.output_type}")
        report_lines.append("")

    report_lines.append("=" * 80)
    report_lines.append("TIER VERIFICATION RESULTS (S2 → S1)")
    report_lines.append("=" * 80)
    report_lines.append("")

    all_verified = all(v.goal_achieved for v in state['tier_verification_results'])
    all_accurate = all(v.report_accurate for v in state['tier_verification_results'])

    report_lines.append(f"All Goals Achieved: {'✅ YES' if all_verified else '❌ NO'}")
    report_lines.append(f"All Reports Accurate: {'✅ YES' if all_accurate else '❌ NO'}")
    report_lines.append("")

    for i, verification in enumerate(state['tier_verification_results'], 1):
        status = "✅ PASS" if verification.goal_achieved else "❌ FAIL"
        report_lines.append(f"{i}. Agent: {state['agent_outputs'][i-1].agent_type} - {status}")
        report_lines.append(f"   Goal Achieved: {verification.goal_achieved}")
        report_lines.append(f"   Report Accurate: {verification.report_accurate}")
        report_lines.append(f"   Confidence: {verification.confidence:.2f}")

        if verification.discrepancies:
            report_lines.append(f"   Discrepancies: {len(verification.discrepancies)}")
            for j, disc in enumerate(verification.discrepancies[:3], 1):
                report_lines.append(f"      {j}. [{disc.severity}/4] {disc.type.value}: {disc.description[:60]}...")
        report_lines.append("")

    if state['conflicts_detected']:
        report_lines.append("=" * 80)
        report_lines.append("CONFLICT DETECTION & RESOLUTION")
        report_lines.append("=" * 80)
        report_lines.append("")

        report_lines.append(f"Conflicts Detected: {len(state['conflicts_detected'])}")
        report_lines.append(f"Resolutions Applied: {len(state.get('resolutions', []))}")
        report_lines.append(f"Consensus Reached: {state['metadata'].get('consensus_reached', 'N/A')}")
        report_lines.append("")

        for i, conflict in enumerate(state['conflicts_detected'], 1):
            report_lines.append(f"{i}. Conflict ID: {conflict.conflict_id}")
            report_lines.append(f"   Type: {conflict.conflict_type.value}")
            report_lines.append(f"   Severity: {conflict.severity.value}")
            report_lines.append(f"   Description: {conflict.description[:80]}...")

            if i-1 < len(state.get('resolutions', [])):
                resolution = state['resolutions'][i-1]
                report_lines.append(f"   Resolution: {resolution.strategy.value}")
                report_lines.append(f"   Implemented: {'✅ Yes' if resolution.implemented else '❌ No'}")
            report_lines.append("")

    report_lines.append("=" * 80)
    report_lines.append("COORDINATION STATUS")
    report_lines.append("=" * 80)
    report_lines.append("")

    overall_success = all_verified and (len(state['conflicts_detected']) == 0 or len(state.get('resolutions', [])) > 0)
    report_lines.append(f"Overall Status: {'✅ SUCCESS' if overall_success else '⚠️  PARTIAL SUCCESS' if all_verified else '❌ FAILED'}")
    report_lines.append(f"All Agents Verified: {'✅ Yes' if all_verified else '❌ No'}")

    if state['conflicts_detected']:
        report_lines.append(f"All Conflicts Resolved: {'✅ Yes' if len(state.get('resolutions', [])) == len(state['conflicts_detected']) else '❌ No'}")

    report_lines.append("")
    report_lines.append("=" * 80)

    final_report = "\n".join(report_lines)
    state["final_report"] = final_report

    state["metadata"].update({
        "overall_success": overall_success,
        "all_verified": all_verified,
        "all_accurate": all_accurate,
        "num_conflicts": len(state['conflicts_detected']),
        "num_resolutions": len(state.get('resolutions', []))
    })

    return state


def should_detect_conflicts(state: CoordinationWorkflowState) -> Literal["detect_conflicts", "verify_results"]:
    """
    Conditional edge: Determine if conflict detection is needed.

    Returns:
        "detect_conflicts" if multi-agent execution, "verify_results" if single agent
    """
    if len(state['agent_outputs']) > 1:
        return "detect_conflicts"
    else:
        return "verify_results"


def should_resolve_conflicts(state: CoordinationWorkflowState) -> Literal["resolve_conflicts", "verify_results"]:
    """
    Conditional edge: Determine if conflict resolution is needed.

    Returns:
        "resolve_conflicts" if conflicts detected, "verify_results" if no conflicts
    """
    if state['requires_conflict_resolution']:
        return "resolve_conflicts"
    else:
        return "verify_results"


def create_coordination_workflow() -> StateGraph:
    """
    Create the coordination workflow with conditional routing.

    Workflow:
    1. analyze_task: Analyze task and determine routing strategy
    2. execute_agents: Execute System 1 agents
    3. detect_conflicts: (conditional) Detect conflicts if multi-agent
    4. resolve_conflicts: (conditional) Resolve conflicts if detected
    5. verify_results: Verify all System 1 results using TierVerification
    6. build_report: Generate final report

    Returns:
        Compiled LangGraph StateGraph
    """
    workflow = StateGraph(CoordinationWorkflowState)

    workflow.add_node("analyze_task", analyze_task_node)
    workflow.add_node("execute_agents", execute_agents_node)
    workflow.add_node("detect_conflicts", detect_conflicts_node)
    workflow.add_node("resolve_conflicts", resolve_conflicts_node)
    workflow.add_node("verify_results", verify_results_node)
    workflow.add_node("build_report", build_report_node)

    workflow.set_entry_point("analyze_task")
    workflow.add_edge("analyze_task", "execute_agents")

    workflow.add_conditional_edges(
        "execute_agents",
        should_detect_conflicts,
        {
            "detect_conflicts": "detect_conflicts",
            "verify_results": "verify_results"
        }
    )

    workflow.add_conditional_edges(
        "detect_conflicts",
        should_resolve_conflicts,
        {
            "resolve_conflicts": "resolve_conflicts",
            "verify_results": "verify_results"
        }
    )

    workflow.add_edge("resolve_conflicts", "verify_results")
    workflow.add_edge("verify_results", "build_report")
    workflow.add_edge("build_report", END)

    return workflow.compile()


def run_coordination_workflow(
    subtasks: List[str],
    context: Optional[Dict[str, Any]] = None,
    coordination_config: Optional[CoordinationConfig] = None,
    memory: Optional[ShortTermMemory] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run the coordination workflow for a list of subtasks.

    This is the main entry point for System 2 (Coordination) workflows.
    Typically called by System 3 (Control) to orchestrate System 1 agents.

    Args:
        subtasks: List of subtasks to coordinate
        context: Additional context (resources, dependencies, constraints)
        coordination_config: CoordinationAgent configuration (default: balanced)
        memory: ShortTermMemory instance for logging (creates new if None)
        verbose: Print progress information

    Returns:
        Final workflow state with coordination results, verifications, and report
    """
    if verbose:
        print("=" * 80)
        print("COORDINATION WORKFLOW (SYSTEM 2) - LANGGRAPH")
        print("=" * 80)
        print(f"\nSubtasks: {len(subtasks)}")
        for i, subtask in enumerate(subtasks, 1):
            print(f"  {i}. {subtask[:80]}...")
        print("=" * 80)
        print()

    if memory is None:
        memory = ShortTermMemory()
        logger.info(f"Created new ShortTermMemory session: {memory.session_id}")

    task_id = memory.start_task(
        agent_id="system2_coordination_workflow",
        agent_type="coordination",
        task_description=f"Coordinate {len(subtasks)} subtasks",
        inputs={"subtasks": subtasks, "context": context}
    )
    logger.info(f"Started coordination workflow: {task_id}")

    config = coordination_config or CoordinationConfig(tier="balanced")
    coordination_agent = CoordinationAgent(config=config)

    app = create_coordination_workflow()

    initial_state = {
        "subtasks": subtasks,
        "context": context or {},
        "coordination_agent": coordination_agent,
        "agent_outputs": [],
        "conflicts_detected": [],
        "resolutions": [],
        "tier_verification_results": [],
        "coordination_result": None,
        "final_report": "",
        "metadata": {
            "tier": str(config.tier),
            "consensus_reached": True
        },
        "memory": memory,
        "task_id": task_id,
        "routing_strategy": "",
        "requires_conflict_resolution": False
    }

    final_state = app.invoke(initial_state)

    memory.end_task(
        task_id=task_id,
        outputs={
            "final_report": final_state["final_report"],
            "num_agents": len(final_state["agent_outputs"]),
            "num_conflicts": len(final_state["conflicts_detected"]),
            "all_verified": final_state["metadata"]["all_verified"]
        },
        metadata=final_state["metadata"]
    )
    logger.info(f"Completed coordination workflow: {task_id}")

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

    if verbose:
        print()
        print(final_state["final_report"])
        print()
        print(f"Session logged to: {memory.session_file}")
        print("=" * 80)

    return final_state


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("=" * 80)
    print("COORDINATION WORKFLOW - LANGGRAPH DEMO")
    print("=" * 80)
    print()
    print("This demo shows:")
    print("- LangGraph workflow with conditional routing")
    print("- System 2 coordinating multiple System 1 agents")
    print("- Conflict detection and resolution (if conflicts arise)")
    print("- TierVerification of System 1 results")
    print("- Comprehensive coordination reporting")
    print()
    print("=" * 80)
    print()

    subtasks = [
        "Research best practices for Python async/await patterns",
        "Implement an async task queue with priority support"
    ]

    result = run_coordination_workflow(
        subtasks=subtasks,
        context={"language": "python", "complexity": "moderate"},
        verbose=True
    )

    print()
    print("=" * 80)
    print("EXECUTION SUMMARY")
    print("=" * 80)
    print(f"Agents executed: {len(result['agent_outputs'])}")
    print(f"Goals achieved: {sum(1 for v in result['tier_verification_results'] if v.goal_achieved)}/{len(result['tier_verification_results'])}")
    print(f"Conflicts detected: {len(result['conflicts_detected'])}")
    print(f"Resolutions applied: {len(result.get('resolutions', []))}")
    print(f"Overall success: {result['metadata']['overall_success']}")
    print("=" * 80)
