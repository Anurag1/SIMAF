"""
Intelligence Workflow - System 4 (Intelligence Layer)

ENHANCED FOR FRACTAL VSM ARCHITECTURE - MAIN USER INTERFACE

This is now the PRIMARY ENTRY POINT for all user tasks in the fractal agent system.
Users (like Claude Code) interact with System 4, which delegates down the hierarchy.

Two workflows provided:
1. User Task Workflow (NEW): Main interface for executing user tasks
   - User → S4 → S3 → S2 → S1
   - S4 verifies S3, S3 verifies S2, S2 verifies S1
   - Full fractal verification chain

2. Performance Monitoring Workflow: Self-analysis of system performance
   - Conditional: check_trigger → analyze (if triggered) → report
   - Used for continuous improvement and optimization

Author: BMad
Date: 2025-10-19 (Enhanced for VSM compliance)
"""

from typing import TypedDict, Literal, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from ..agents.intelligence_agent import IntelligenceAgent, IntelligenceResult
from ..agents.intelligence_config import IntelligenceConfig, PresetIntelligenceConfigs
from ..memory.short_term import ShortTermMemory
from ..verification import (
    TierVerification,
    TierVerificationResult,
    Goal,
    verify_subordinate_tier
)
from .multi_agent_workflow import run_multi_agent_workflow
import json
import logging

# OBSERVABILITY: Import observability framework
from ..observability import (
    init_context, get_correlation_id, get_tracer, get_logger,
    get_event_store, VSMEvent, set_span_attributes
)

# Use structured logger instead of basic logger
logger = get_logger(__name__)

# Note: Observability auto-initializes when fractal_agent.observability is imported
# See fractal_agent/observability/__init__.py for auto-initialization code


class IntelligenceWorkflowState(TypedDict):
    """
    State shared across Intelligence workflow nodes.

    Attributes:
        session_id: ID of the session being analyzed
        session_logs: JSON-formatted session logs
        performance_metrics: Performance metrics dict
        should_trigger: Whether analysis should be triggered
        trigger_reason: Reason for triggering (or not)
        intelligence_result: Output from Intelligence Agent
        report: Final formatted report
    """
    session_id: str
    session_logs: str
    performance_metrics: dict
    session_size: int
    last_analysis_days_ago: int | None
    should_trigger: bool
    trigger_reason: str
    intelligence_result: IntelligenceResult | None
    report: str | None


def check_trigger_node(state: IntelligenceWorkflowState) -> IntelligenceWorkflowState:
    """
    Node 1: Check Trigger Conditions

    Determines if intelligence analysis should be triggered based on:
    - Failure rate
    - Cost spike
    - Scheduled analysis
    """
    logger.info(f"Check trigger: session {state['session_id']}")

    # Create intelligence agent with default config
    config = IntelligenceConfig()
    agent = IntelligenceAgent(config=config)

    # Check if analysis should be triggered
    should_trigger, reason = agent.should_trigger_analysis(
        performance_metrics=state["performance_metrics"],
        session_size=state["session_size"],
        last_analysis_days_ago=state.get("last_analysis_days_ago")
    )

    # Update state
    state["should_trigger"] = should_trigger
    state["trigger_reason"] = reason

    logger.info(f"Trigger decision: {should_trigger} - {reason}")

    return state


def analyze_node(state: IntelligenceWorkflowState) -> IntelligenceWorkflowState:
    """
    Node 2: Intelligence Analysis

    Runs Intelligence Agent to analyze session performance and
    generate actionable insights.
    """
    logger.info(f"Intelligence analysis: session {state['session_id']}")

    # Create intelligence agent with quick analysis config for faster execution
    config = PresetIntelligenceConfigs.quick_analysis()
    agent = IntelligenceAgent(config=config)

    # Run intelligence reflection
    result = agent(
        session_logs=state["session_logs"],
        performance_metrics=state["performance_metrics"],
        session_id=state["session_id"],
        verbose=True
    )

    # Update state
    state["intelligence_result"] = result

    logger.info("Intelligence analysis complete")

    return state


def report_node(state: IntelligenceWorkflowState) -> IntelligenceWorkflowState:
    """
    Node 3: Generate Report

    Formats the intelligence analysis results into a human-readable report.
    """
    logger.info("Report node: generating intelligence report")

    if not state["should_trigger"]:
        # No analysis triggered - minimal report
        report = f"""
================================================================================
INTELLIGENCE WORKFLOW REPORT
================================================================================

Session: {state['session_id']}
Status: No analysis triggered

Reason: {state['trigger_reason']}

Performance Metrics:
- Accuracy: {state['performance_metrics'].get('accuracy', 'N/A'):.2%}
- Total Cost: ${state['performance_metrics'].get('cost', 0):.2f}
- Avg Latency: {state['performance_metrics'].get('latency', 0):.2f}s
- Cache Hit Rate: {state['performance_metrics'].get('cache_hit_rate', 0):.2%}
- Total Tasks: {state['performance_metrics'].get('total_tasks', 0)}
- Failed Tasks: {len(state['performance_metrics'].get('failed_tasks', []))}

================================================================================
No action required - system is performing well
================================================================================
"""
    else:
        # Analysis triggered - full report with intelligence insights
        result = state["intelligence_result"]
        report = f"""
================================================================================
INTELLIGENCE WORKFLOW REPORT (System 4)
================================================================================

Session: {state['session_id']}
Status: Analysis triggered

Trigger Reason: {state['trigger_reason']}

Performance Metrics:
- Accuracy: {state['performance_metrics'].get('accuracy', 'N/A'):.2%}
- Total Cost: ${state['performance_metrics'].get('cost', 0):.2f}
- Avg Latency: {state['performance_metrics'].get('latency', 0):.2f}s
- Cache Hit Rate: {state['performance_metrics'].get('cache_hit_rate', 0):.2%}
- Total Tasks: {state['performance_metrics'].get('total_tasks', 0)}
- Failed Tasks: {len(state['performance_metrics'].get('failed_tasks', []))}

================================================================================
INTELLIGENCE ANALYSIS
================================================================================

{result.analysis}

================================================================================
DETECTED PATTERNS
================================================================================

{result.patterns}

================================================================================
ACTIONABLE INSIGHTS
================================================================================

{result.insights}

================================================================================
PRIORITIZED ACTION PLAN
================================================================================

{result.action_plan}

================================================================================
ANALYSIS METADATA
================================================================================

Timestamp: {result.metadata.get('timestamp', 'N/A')}
Model Tiers:
- Analysis: {result.metadata.get('tiers', {}).get('analysis', 'N/A')}
- Pattern Detection: {result.metadata.get('tiers', {}).get('pattern', 'N/A')}
- Insight Generation: {result.metadata.get('tiers', {}).get('insight', 'N/A')}
- Prioritization: {result.metadata.get('tiers', {}).get('prioritization', 'N/A')}

================================================================================
WORKFLOW COMPLETE
================================================================================
"""

    state["report"] = report

    return state


def should_analyze(state: IntelligenceWorkflowState) -> Literal["analyze", "report"]:
    """
    Conditional edge: determines if analysis should be run.

    Returns:
        "analyze" if triggers are met, "report" to skip analysis
    """
    return "analyze" if state["should_trigger"] else "report"


def create_intelligence_workflow() -> StateGraph:
    """
    Create the intelligence workflow with conditional analysis.

    Returns:
        Compiled LangGraph StateGraph
    """
    # Create graph
    workflow = StateGraph(IntelligenceWorkflowState)

    # Add nodes
    workflow.add_node("check_trigger", check_trigger_node)
    workflow.add_node("analyze", analyze_node)
    workflow.add_node("report", report_node)

    # Define flow with conditional branching
    workflow.set_entry_point("check_trigger")
    workflow.add_conditional_edges(
        "check_trigger",
        should_analyze,
        {
            "analyze": "analyze",
            "report": "report"
        }
    )
    workflow.add_edge("analyze", "report")
    workflow.add_edge("report", END)

    # Compile
    return workflow.compile()


def run_intelligence_workflow(
    memory: ShortTermMemory,
    config: IntelligenceConfig = None,
    last_analysis_days_ago: int = None
) -> dict:
    """
    Run the intelligence workflow on a session.

    Args:
        memory: ShortTermMemory instance with session data
        config: Optional IntelligenceConfig (uses default if None)
        last_analysis_days_ago: Days since last analysis (for scheduled trigger)

    Returns:
        Final workflow state with report
    """
    # Get performance metrics from memory
    metrics = memory.get_performance_metrics()

    # Format session logs
    session_logs = json.dumps({
        "session_id": memory.session_id,
        "tasks": list(memory.tasks.values())
    }, indent=2)

    # Create workflow
    app = create_intelligence_workflow()

    # Initialize state
    initial_state = {
        "session_id": memory.session_id,
        "session_logs": session_logs,
        "performance_metrics": metrics,
        "session_size": metrics["total_tasks"],
        "last_analysis_days_ago": last_analysis_days_ago,
        "should_trigger": False,
        "trigger_reason": "",
        "intelligence_result": None,
        "report": None
    }

    # Execute workflow
    final_state = app.invoke(initial_state)

    return final_state


# ============================================================================
# NEW: User Task Workflow - Main Interface (System 4 Entry Point)
# ============================================================================

def run_user_task(
    user_task: str,
    verify_control: bool = True,
    verbose: bool = False,
    memory: Optional[ShortTermMemory] = None
) -> Dict[str, Any]:
    """
    NEW: Main entry point for user tasks - System 4 Interface

    This is the PRIMARY INTERFACE for all user tasks in the fractal agent system.
    Users (Claude Code, external APIs, etc.) should call this function, not lower-level workflows.

    VSM HIERARCHY:
    1. System 4 (Intelligence) receives user task
    2. System 4 delegates to System 3 (Control) via multi_agent_workflow
    3. System 3 delegates to System 2 (Coordination)
    4. System 2 orchestrates System 1 agents (Developer, Research)
    5. Verification flows upward: S1 → S2 → S3 → S4

    TIER VERIFICATION:
    - System 4 verifies System 3 using TierVerification
    - Three-way comparison: GOAL (user task) vs REPORT (S3 output) vs ACTUAL (reality check)
    - System 3, 2, 1 perform their own tier verification at each level
    - Full fractal verification chain ensures quality at every tier

    Args:
        user_task: The task requested by the user (e.g., "Create a Calculator class")
        verify_control: Whether to verify System 3's results (default: True)
        verbose: Print detailed progress information
        memory: ShortTermMemory instance for logging (creates new if None)

    Returns:
        Dict with:
            - user_task: Original user task (the GOAL)
            - control_result: Results from System 3 (Control)
            - tier_verification: System 4's verification of System 3 (goal vs report vs actual)
            - goal_achieved: Whether the user's goal was achieved
            - report_accurate: Whether System 3's report was accurate
            - final_report: User-facing report with results and verification status
            - metadata: Execution metadata

    Example:
        >>> from fractal_agent.workflows.intelligence_workflow import run_user_task
        >>>
        >>> # User (Claude Code) calls System 4 with a task
        >>> result = run_user_task(
        ...     user_task="Create a Calculator class with add, subtract methods",
        ...     verbose=True
        ... )
        >>>
        >>> # Check if goal was achieved
        >>> if result["goal_achieved"]:
        ...     print("✅ Task completed successfully!")
        ...     print(result["final_report"])
        ... else:
        ...     print("❌ Task failed verification")
        ...     print(f"Issues: {result['tier_verification'].discrepancies}")
    """
    # OBSERVABILITY: Initialize context and tracing for this user task
    ctx = init_context()  # Creates correlation_id, trace_id, and sets session start time
    correlation_id = ctx['correlation_id']
    trace_id = ctx['trace_id']

    # OBSERVABILITY: Get tracer and event store
    tracer = get_tracer(__name__)
    event_store = get_event_store()

    with tracer.start_as_current_span("system4_user_task") as span:
        # OBSERVABILITY: Set span attributes for filtering and analysis
        set_span_attributes({
            "vsm.tier": "System4_Intelligence",
            "vsm.operation": "user_task",
            "task.description": user_task[:100],  # Truncate for cardinality
            "verification.enabled": verify_control,
            "correlation_id": correlation_id,
            "trace_id": trace_id
        })

        # OBSERVABILITY: Emit task_started event
        event_store.append(VSMEvent(
            tier="System4",
            event_type="task_started",
            data={
                "user_task": user_task,
                "verify_control": verify_control,
                "correlation_id": correlation_id
            }
        ))

        # OBSERVABILITY: Use structured logging
        if verbose:
            print("=" * 80)
            print("SYSTEM 4 (INTELLIGENCE LAYER) - USER TASK INTERFACE")
            print("=" * 80)
            print(f"\nUser Task: {user_task}")
            print(f"\nVSM Architecture: User → S4 → S3 → S2 → S1")
            print(f"Verification enabled: {verify_control}")
            print(f"Correlation ID: {correlation_id}")
            print("=" * 80)
            print()

        logger.info(
            "System 4: Received user task",
            extra={
                "user_task": user_task,
                "verify_control": verify_control,
                "correlation_id": correlation_id,
                "trace_id": trace_id
            }
        )

        # Initialize memory if not provided
        if memory is None:
            memory = ShortTermMemory()
            logger.info(f"Created new ShortTermMemory session: {memory.session_id}")

        # Stage 1: Delegate to System 3 (Control) with memory
        if verbose:
            print("[1/3] Delegating to System 3 (Control)...")
            print(f"     Memory session: {memory.session_id}")
            print()

        # OBSERVABILITY: Emit delegation event
        event_store.append(VSMEvent(
            tier="System4",
            event_type="delegating_to_s3",
            data={"user_task": user_task, "memory_session": memory.session_id}
        ))

        control_result = run_multi_agent_workflow(main_task=user_task, memory=memory)

    if verbose:
        print()
        print("[2/3] System 3 completed. Verifying results...")
        print()

    # Stage 2: Verify System 3's results (System 4 verifies subordinate System 3)
    tier_verification = None
    if verify_control:
        # Create goal from user task
        goal = Goal(
            objective=user_task,
            success_criteria=[
                "Task decomposition was appropriate",
                "All subtasks were completed successfully",
                "Final report addresses the user's request",
                "No errors or failures in execution"
            ],
            required_artifacts=[],  # Will be determined from control_result
            context={"user_task": user_task}
        )

        # Build report from System 3's output
        report = {
            "system": "System3_Control",
            "main_task": control_result["control_result"].main_task,
            "subtasks": control_result["control_result"].subtasks,
            "subtask_results": control_result["control_result"].subtask_results,
            "final_report": control_result["control_result"].final_report,
            "claimed_success": True,  # System 3 completed execution
            "metadata": control_result["control_result"].metadata
        }

        # Verify using TierVerification (System 4 verifies System 3)
        tier_verifier = TierVerification(
            tier_name="System4_Intelligence",
            subordinate_tier="System3_Control"
        )

        tier_verification = tier_verifier.verify_subordinate(
            goal=goal,
            report=report,
            context={"user_task": user_task, "control_result": control_result}
        )

        if verbose:
            if tier_verification.goal_achieved:
                print("  ✅ System 3 verification PASSED")
                print(f"     Goal achieved: Yes")
                print(f"     Report accurate: {tier_verification.report_accurate}")
            else:
                print("  ❌ System 3 verification FAILED")
                print(f"     Goal achieved: No")
                print(f"     Discrepancies: {len(tier_verification.discrepancies)}")
                for disc in tier_verification.discrepancies[:3]:  # Show first 3
                    print(f"        - {disc.type.value}: {disc.description[:80]}")
            print()

    # Stage 3: Build final report
    if verbose:
        print("[3/3] Building final user report...")
        print()

    goal_achieved = tier_verification.goal_achieved if tier_verification else True
    report_accurate = tier_verification.report_accurate if tier_verification else True

    # Build user-facing report
    final_report = f"""
{'=' * 80}
FRACTAL AGENT SYSTEM - TASK EXECUTION REPORT
{'=' * 80}

USER TASK:
{user_task}

VSM ARCHITECTURE:
User → System 4 (Intelligence) → System 3 (Control) → System 2 (Coordination) → System 1 (Operational)

{'=' * 80}
SYSTEM 3 (CONTROL) RESULTS
{'=' * 80}

Task Decomposition:
"""
    for i, subtask in enumerate(control_result["control_result"].subtasks, 1):
        final_report += f"\n  {i}. {subtask}"

    final_report += f"""

Subtasks Executed: {len(control_result['control_result'].subtask_results)}

{'=' * 80}
SYSTEM 3 FINAL REPORT
{'=' * 80}

{control_result['control_result'].final_report}

{'=' * 80}
SYSTEM 4 (INTELLIGENCE) VERIFICATION
{'=' * 80}
"""

    if tier_verification:
        final_report += f"""
Goal Achieved: {'✅ YES' if goal_achieved else '❌ NO'}
Report Accurate: {'✅ YES' if report_accurate else '❌ NO'}
Verification Confidence: {tier_verification.confidence:.2f}

"""
        if tier_verification.discrepancies:
            final_report += f"Discrepancies Detected: {len(tier_verification.discrepancies)}\n"
            for i, disc in enumerate(tier_verification.discrepancies[:5], 1):
                final_report += f"  {i}. [{disc.severity}/4] {disc.type.value}: {disc.description}\n"
                if disc.suggested_action:
                    final_report += f"      → {disc.suggested_action}\n"
        else:
            final_report += "No discrepancies detected. All verification checks passed.\n"
    else:
        final_report += "Verification disabled (verify_control=False)\n"

    final_report += f"""
{'=' * 80}
EXECUTION COMPLETE
{'=' * 80}

Status: {'✅ SUCCESS' if goal_achieved else '❌ INCOMPLETE'}
"""

    if verbose:
        print(final_report)

    # End session (saves to JSON and auto-exports to Obsidian if enabled)
    if memory:
        try:
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
                    print(f"\n📝 Session log saved: {memory.session_file}")
                    print(f"📤 Obsidian export: {export_status['status']}\n")
            elif verbose:
                print(f"\n📝 Session log saved: {memory.session_file}\n")
        except Exception as e:
            logger.error(f"Failed to end session: {e}")

    # Return comprehensive result
    return {
        "user_task": user_task,
        "control_result": control_result,
        "tier_verification": tier_verification,
        "goal_achieved": goal_achieved,
        "report_accurate": report_accurate,
        "final_report": final_report,
        "metadata": {
            "system": "System4_Intelligence",
            "subordinate": "System3_Control",
            "verification_enabled": verify_control,
            "control_metadata": control_result["control_result"].metadata
        }
    }


# ============================================================================
# Demo
# ============================================================================

if __name__ == "__main__":
    import logging
    import tempfile
    logging.basicConfig(level=logging.INFO)

    print("=" * 80)
    print("Intelligence Workflow Demo - Phase 3")
    print("=" * 80)
    print()

    # Create sample session with performance issues
    with tempfile.TemporaryDirectory() as tmpdir:
        memory = ShortTermMemory(log_dir=tmpdir)

        # Task 1: Completed
        task1 = memory.start_task(
            "research_agent_001",
            "operational",
            "Research quantum computing",
            {"topic": "quantum computing"}
        )
        memory.end_task(
            task1,
            outputs={"report": "Quantum computing basics..."},
            metadata={"cost": 0.05, "tokens": 2500, "cache_hit": True}
        )

        # Task 2: Failed (high cost)
        task2 = memory.start_task(
            "research_agent_002",
            "operational",
            "Analyze large dataset",
            {"dataset_size": "1GB"}
        )
        memory.tasks[task2]["status"] = "failed"
        memory.tasks[task2]["duration_seconds"] = 8.1
        memory.tasks[task2]["metadata"] = {
            "cost": 0.02,
            "tokens": 1200,
            "error": "Context limit exceeded"
        }

        # Task 3: Completed (expensive)
        task3 = memory.start_task(
            "control_agent_001",
            "control",
            "Coordinate multi-agent research",
            {"subtasks": 5}
        )
        memory.end_task(
            task3,
            outputs={"result": "Coordination complete"},
            metadata={"cost": 0.12, "tokens": 4200, "cache_hit": False}
        )

        # Run intelligence workflow
        result = run_intelligence_workflow(
            memory=memory,
            last_analysis_days_ago=10  # Force scheduled analysis
        )

        # Print report
        print(result["report"])
