"""
Simple Research Workflow - Phase 0

Linear workflow: research → analyze → report
Uses LangGraph for stateful execution.

Author: BMad
Date: 2025-10-18
"""

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from ..agents.research_agent import ResearchAgent, ResearchResult
from ..agents.research_config import ResearchConfig
import logging

logger = logging.getLogger(__name__)


class WorkflowState(TypedDict):
    """
    State shared across workflow nodes.

    Attributes:
        topic: Research topic
        research_result: Output from research node
        analysis: Output from analyze node
        report: Final report output
    """
    topic: str
    research_result: ResearchResult | None
    analysis: str | None
    report: str | None


def research_node(state: WorkflowState) -> WorkflowState:
    """
    Node 1: Research

    Uses ResearchAgent to gather and synthesize information.
    """
    logger.info(f"Research node: {state['topic']}")

    # Create research agent
    agent = ResearchAgent(
        config=ResearchConfig(),
        max_research_questions=2
    )

    # Execute research
    result = agent(topic=state["topic"], verbose=True)

    # Update state
    state["research_result"] = result

    return state


def analyze_node(state: WorkflowState) -> WorkflowState:
    """
    Node 2: Analyze

    Analyzes the research findings for patterns and insights.
    """
    logger.info("Analyze node: processing research results")

    research_result = state["research_result"]

    # Simple analysis: extract key points from synthesis
    analysis = f"""
# Analysis of Research Findings

## Topic
{research_result.topic}

## Key Insights
{research_result.synthesis}

## Validation
{research_result.validation}

## Metadata
- Total tokens used: {research_result.metadata['total_tokens']}
- Number of research questions: {research_result.metadata['num_questions']}
"""

    state["analysis"] = analysis

    return state


def report_node(state: WorkflowState) -> WorkflowState:
    """
    Node 3: Report

    Generates final report from analysis.
    """
    logger.info("Report node: generating final report")

    analysis = state["analysis"]
    research_result = state["research_result"]

    # Generate final report
    report = f"""
================================================================================
RESEARCH WORKFLOW REPORT
================================================================================

Topic: {research_result.topic}

{analysis}

================================================================================
RESEARCH PLAN
================================================================================

{research_result.research_plan}

================================================================================
DETAILED FINDINGS
================================================================================

"""

    for i, finding in enumerate(research_result.findings, 1):
        report += f"\n{i}. {finding['question']}\n"
        report += f"   {finding['answer']}\n\n"

    report += f"""
================================================================================
WORKFLOW COMPLETE
================================================================================
"""

    state["report"] = report

    return state


def create_research_workflow() -> StateGraph:
    """
    Create the linear research workflow.

    Returns:
        Compiled LangGraph StateGraph
    """
    # Create graph
    workflow = StateGraph(WorkflowState)

    # Add nodes
    workflow.add_node("research", research_node)
    workflow.add_node("analyze", analyze_node)
    workflow.add_node("report", report_node)

    # Define linear flow
    workflow.set_entry_point("research")
    workflow.add_edge("research", "analyze")
    workflow.add_edge("analyze", "report")
    workflow.add_edge("report", END)

    # Compile
    return workflow.compile()


def run_research_workflow(topic: str) -> dict:
    """
    Run the complete research workflow.

    Args:
        topic: Research topic

    Returns:
        Final workflow state with report
    """
    # Create workflow
    app = create_research_workflow()

    # Initialize state
    initial_state = {
        "topic": topic,
        "research_result": None,
        "analysis": None,
        "report": None
    }

    # Execute workflow
    final_state = app.invoke(initial_state)

    return final_state


# Quick test
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("=" * 80)
    print("Simple Research Workflow - Phase 0")
    print("=" * 80)
    print()

    # Run workflow
    result = run_research_workflow(
        topic="What is the Viable System Model?"
    )

    # Print final report
    print(result["report"])
