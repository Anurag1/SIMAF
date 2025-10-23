"""
Control Agent - System 3 (VSM)

Task decomposition and delegation for multi-agent coordination.
Phase 1: Vertical Slice implementation.

Author: BMad
Date: 2025-10-18
"""

import dspy
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from ..utils.dspy_integration import FractalDSpyLM, configure_dspy
from ..utils.model_config import Tier
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# DSPy Signatures for Control Agent
# ============================================================================

class TaskDecomposition(dspy.Signature):
    """
    Decompose a complex task into subtasks for delegation.

    Break down the main task into 3-7 specific, independent subtasks
    that can be executed in parallel by operational agents.
    """
    main_task = dspy.InputField(desc="The complex task to decompose")
    subtasks = dspy.OutputField(desc="List of 3-7 specific subtasks, one per line")


class SynthesisCoordination(dspy.Signature):
    """
    Coordinate synthesis of results from multiple operational agents.

    Review all subtask results and create a coherent final report.
    """
    main_task = dspy.InputField(desc="The original complex task")
    subtask_results = dspy.InputField(desc="Results from all operational agents")
    final_report = dspy.OutputField(desc="Synthesized final report combining all results")


# ============================================================================
# Control Agent Result
# ============================================================================

@dataclass
class ControlResult:
    """
    Result from Control Agent execution.

    Attributes:
        main_task: Original task
        subtasks: List of decomposed subtasks
        subtask_results: Results from operational agents
        final_report: Synthesized final report
        metadata: Execution metadata
    """
    main_task: str
    subtasks: List[str]
    subtask_results: List[Dict[str, Any]]
    final_report: str
    metadata: Dict[str, Any]

    def __str__(self) -> str:
        """Human-readable string representation"""
        output = []
        output.append("=" * 80)
        output.append("CONTROL AGENT REPORT (System 3 - VSM)")
        output.append("=" * 80)
        output.append(f"\nMain Task: {self.main_task}")
        output.append(f"\n{'-' * 80}")
        output.append("\nTASK DECOMPOSITION:")
        for i, subtask in enumerate(self.subtasks, 1):
            output.append(f"{i}. {subtask}")
        output.append(f"\n{'-' * 80}")
        output.append(f"\nSUBTASK RESULTS: {len(self.subtask_results)} operational agents executed")
        output.append(f"\n{'-' * 80}")
        output.append("\nFINAL REPORT:")
        output.append(self.final_report)
        output.append(f"\n{'-' * 80}")
        output.append(f"\nMetadata: {self.metadata}")
        output.append("=" * 80)
        return "\n".join(output)


# ============================================================================
# Control Agent
# ============================================================================

class ControlAgent(dspy.Module):
    """
    System 3 (Control) - Task decomposition and coordination.

    Decomposes complex tasks into subtasks, delegates to operational agents,
    and synthesizes results into final report.

    Usage:
        >>> agent = ControlAgent()
        >>> result = agent(
        ...     main_task="Research the Viable System Model",
        ...     operational_agent_runner=run_operational_agents
        ... )
        >>> print(result.final_report)
    """

    def __init__(
        self,
        tier: Tier = "balanced",
        max_tokens: Optional[int] = None,
        temperature: float = 0.7
    ):
        """
        Initialize Control Agent.

        Args:
            tier: Model tier for control operations
            max_tokens: Optional token limit
            temperature: Sampling temperature
        """
        super().__init__()

        # Create LM for control operations
        lm_kwargs = {}
        if max_tokens is not None:
            lm_kwargs["max_tokens"] = max_tokens

        self.control_lm = FractalDSpyLM(
            tier=tier,
            temperature=temperature,
            **lm_kwargs
        )

        # Configure DSPy
        dspy.configure(lm=self.control_lm)

        # Create DSPy modules
        self.decomposer = dspy.ChainOfThought(TaskDecomposition)
        self.synthesizer = dspy.ChainOfThought(SynthesisCoordination)

        logger.info(f"Initialized ControlAgent with tier={tier}")

    def forward(
        self,
        main_task: str,
        operational_agent_runner: callable,
        verbose: bool = True
    ) -> ControlResult:
        """
        Execute control workflow: decompose → delegate → synthesize.

        Args:
            main_task: Complex task to decompose and execute
            operational_agent_runner: Function that executes operational agents
                Signature: (subtask: str) -> Dict[str, Any]
            verbose: Print progress

        Returns:
            ControlResult with final report
        """
        if verbose:
            print(f"\n{'=' * 80}")
            print(f"CONTROL AGENT (System 3) - Task Decomposition")
            print(f"{'=' * 80}")
            print(f"\nMain Task: {main_task}\n")

        # Step 1: Decompose task
        if verbose:
            print("Step 1: Decomposing task...")

        decomposition = self.decomposer(main_task=main_task)

        # Parse subtasks (one per line)
        subtasks = [
            line.strip().lstrip('0123456789.)-•* ')
            for line in decomposition.subtasks.split('\n')
            if line.strip() and len(line.strip()) > 10
        ]

        if verbose:
            print(f"✓ Decomposed into {len(subtasks)} subtasks:")
            for i, subtask in enumerate(subtasks, 1):
                print(f"  {i}. {subtask}")
            print()

        # Step 2: Execute operational agents
        if verbose:
            print(f"Step 2: Delegating to {len(subtasks)} operational agents...")

        subtask_results = []
        for i, subtask in enumerate(subtasks, 1):
            if verbose:
                print(f"\n  Agent {i}/{len(subtasks)}: {subtask}")

            result = operational_agent_runner(subtask)
            subtask_results.append({
                "subtask": subtask,
                "result": result
            })

            if verbose:
                print(f"  ✓ Agent {i} completed")

        if verbose:
            print(f"\n✓ All {len(subtasks)} agents completed\n")

        # Step 3: Synthesize results
        if verbose:
            print("Step 3: Synthesizing final report...")

        # Format results for synthesis
        results_text = "\n\n".join([
            f"Subtask {i}: {sr['subtask']}\n"
            f"Result: {sr['result']}"
            for i, sr in enumerate(subtask_results, 1)
        ])

        synthesis = self.synthesizer(
            main_task=main_task,
            subtask_results=results_text
        )

        if verbose:
            print("✓ Synthesis complete\n")

        # Collect metadata
        metadata = {
            "num_subtasks": len(subtasks),
            "control_metrics": self.control_lm.get_metrics()
        }

        # Create result
        result = ControlResult(
            main_task=main_task,
            subtasks=subtasks,
            subtask_results=subtask_results,
            final_report=synthesis.final_report,
            metadata=metadata
        )

        if verbose:
            print(f"{'=' * 80}")
            print("CONTROL AGENT: Task Complete")
            print(f"{'=' * 80}\n")

        return result


# Quick test
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("=" * 80)
    print("Control Agent (System 3) Test")
    print("=" * 80)
    print()

    # Simple operational agent runner for testing
    def mock_operational_agent(subtask: str) -> str:
        """Mock operational agent that returns a simple result"""
        return f"Completed research on: {subtask}"

    # Create control agent
    agent = ControlAgent(tier="balanced")

    # Execute
    result = agent(
        main_task="Research the Viable System Model",
        operational_agent_runner=mock_operational_agent,
        verbose=True
    )

    # Print result
    print(result)
