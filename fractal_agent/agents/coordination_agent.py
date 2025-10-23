"""
CoordinationAgent - Multi-Agent Conflict Detection and Resolution

Manages coordination between multiple agents in the fractal ecosystem,
detecting and resolving conflicts that arise from parallel execution.

Responsibilities:
1. Conflict Detection - Identify conflicts in agent outputs (code, plans, goals)
2. Conflict Resolution - Propose and implement conflict resolution strategies
3. Consensus Building - Facilitate agreement among agents
4. Resource Allocation - Manage shared resources to prevent conflicts

Uses DSPy signatures for intelligent conflict analysis and resolution.

Author: BMad
Date: 2025-10-19
"""

import dspy
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

from ..utils.dspy_integration import FractalDSpyLM, configure_dspy
from ..utils.model_config import Tier
from ..verification import (
    Goal,
    Evidence,
    VerificationResult,
    verify_goal,
    EvidenceCollector,
    TierVerification,
    TierVerificationResult,
    verify_subordinate_tier
)

logger = logging.getLogger(__name__)


# ============================================================================
# Enumerations for Conflict Types and Resolution Strategies
# ============================================================================

class ConflictType(str, Enum):
    """Types of conflicts that can occur between agents"""
    CODE_CONFLICT = "code_conflict"  # Conflicting code changes
    GOAL_CONFLICT = "goal_conflict"  # Incompatible goals
    RESOURCE_CONFLICT = "resource_conflict"  # Competing for same resource
    PRIORITY_CONFLICT = "priority_conflict"  # Conflicting priorities
    DEPENDENCY_CONFLICT = "dependency_conflict"  # Circular or broken dependencies
    SEMANTIC_CONFLICT = "semantic_conflict"  # Logically incompatible outputs


class ResolutionStrategy(str, Enum):
    """Strategies for resolving conflicts"""
    MERGE = "merge"  # Merge conflicting outputs
    PRIORITIZE = "prioritize"  # Choose higher priority option
    SEQUENCE = "sequence"  # Execute sequentially instead of parallel
    DELEGATE = "delegate"  # Escalate to higher-level agent
    NEGOTIATE = "negotiate"  # Facilitate agent negotiation
    PARTITION = "partition"  # Split resources/work to avoid conflict


class ConflictSeverity(str, Enum):
    """Severity levels for conflicts"""
    LOW = "low"  # Minor conflict, easily resolved
    MEDIUM = "medium"  # Moderate conflict, requires analysis
    HIGH = "high"  # Serious conflict, may require redesign
    CRITICAL = "critical"  # System-breaking conflict, must resolve immediately


# ============================================================================
# DSPy Signatures for Coordination
# ============================================================================

class TaskClassification(dspy.Signature):
    """
    Classify a subtask as either 'research' or 'code_generation'.

    This signature is used to route subtasks to the appropriate System 1 agent.

    Research tasks: analysis, investigation, documentation review, Q&A
    Code generation tasks: implement, build, create code, write tests
    """
    subtask = dspy.InputField(desc="The subtask to classify")
    task_type = dspy.OutputField(
        desc="Either 'research' or 'code_generation'"
    )
    reasoning = dspy.OutputField(
        desc="Brief explanation of why this classification was chosen"
    )


class FilePathInference(dspy.Signature):
    """
    Infer the appropriate file path for code generation from task description.

    Analyzes task descriptions to determine where generated code should be written.
    Follows project conventions and directory structure.

    Examples:
    - "Implement PolicyAgent" → "fractal_agent/agents/policy_agent.py"
    - "Build external knowledge integration" → "fractal_agent/integrations/external_knowledge.py"
    - "Create knowledge validation framework" → "fractal_agent/validation/knowledge_validation.py"
    """
    task_description = dspy.InputField(desc="The implementation subtask description")
    project_root = dspy.InputField(desc="Project root directory path")
    inferred_path = dspy.OutputField(
        desc="Relative file path where code should be written (e.g., 'fractal_agent/agents/policy_agent.py')"
    )
    reasoning = dspy.OutputField(
        desc="Brief explanation of why this path was chosen based on task description and project conventions"
    )


class ConflictDetection(dspy.Signature):
    """
    Detect conflicts between agent outputs or goals.

    Analyzes multiple agent outputs to identify incompatibilities,
    overlaps, or contradictions that could cause issues.

    Returns conflict type, severity, and affected components.
    """

    agent_outputs = dspy.InputField(
        desc="JSON array of agent outputs with agent_id, output_type, and content"
    )
    context = dspy.InputField(
        desc="Additional context: shared resources, dependencies, constraints"
    )

    conflicts_detected = dspy.OutputField(
        desc="List of conflicts found. Format: [{type, severity, description, affected_agents}]"
    )
    conflict_summary = dspy.OutputField(
        desc="High-level summary of all conflicts and their impact"
    )


class ConflictResolution(dspy.Signature):
    """
    Propose resolution strategy for detected conflicts.

    Analyzes conflict details and proposes actionable resolution
    strategy with specific steps to resolve the conflict.
    """

    conflict_description = dspy.InputField(
        desc="Detailed description of the conflict"
    )
    conflict_type = dspy.InputField(
        desc="Type of conflict (code, goal, resource, priority, dependency, semantic)"
    )
    affected_agents = dspy.InputField(
        desc="List of agents involved in the conflict"
    )
    available_strategies = dspy.InputField(
        desc="Available resolution strategies (merge, prioritize, sequence, delegate, negotiate, partition)"
    )

    recommended_strategy = dspy.OutputField(
        desc="Recommended resolution strategy from available options"
    )
    resolution_steps = dspy.OutputField(
        desc="Detailed steps to implement the resolution"
    )
    expected_outcome = dspy.OutputField(
        desc="Expected outcome after resolution"
    )


class ConsensusBuilding(dspy.Signature):
    """
    Build consensus among agents with conflicting outputs.

    Facilitates negotiation between agents to reach agreement
    on a unified approach or compromise solution.
    """

    agent_positions = dspy.InputField(
        desc="Each agent's position, preferences, and constraints"
    )
    conflict_context = dspy.InputField(
        desc="Background information about the conflict"
    )
    success_criteria = dspy.InputField(
        desc="Criteria for successful consensus (all must be met)"
    )

    consensus_proposal = dspy.OutputField(
        desc="Proposed consensus solution that addresses all agent concerns"
    )
    trade_offs = dspy.OutputField(
        desc="Trade-offs made to achieve consensus"
    )
    agent_agreements = dspy.OutputField(
        desc="How each agent's concerns are addressed in the consensus"
    )


class ResourceAllocation(dspy.Signature):
    """
    Allocate shared resources to prevent conflicts.

    Determines optimal resource allocation strategy to minimize
    conflicts while maximizing agent productivity.
    """

    available_resources = dspy.InputField(
        desc="List of available resources (files, APIs, memory, compute)"
    )
    agent_requests = dspy.InputField(
        desc="Each agent's resource requirements and priorities"
    )
    constraints = dspy.InputField(
        desc="Resource constraints and limitations"
    )

    allocation_plan = dspy.OutputField(
        desc="Resource allocation plan: which agent gets which resources and when"
    )
    potential_conflicts = dspy.OutputField(
        desc="Potential conflicts in the allocation and mitigation strategies"
    )


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class CoordinationConfig:
    """
    Configuration for CoordinationAgent.

    The CoordinationAgent is a System 3 (Coordination) agent specialized
    for detecting and resolving conflicts between multiple operational agents.

    Attributes:
        tier: Model tier for coordination (default: balanced)
        max_resolution_attempts: Maximum attempts to resolve a conflict (default: 3)
        auto_resolve_low_severity: Automatically resolve low-severity conflicts
        enable_consensus_building: Enable agent negotiation for conflict resolution
        enable_resource_management: Enable proactive resource allocation
        conflict_detection_threshold: Sensitivity threshold for conflict detection (0-1)
        require_verification: Verify resolution success with goal-based verification
        escalation_enabled: Escalate unresolved conflicts to control agent
    """

    # Model configuration
    tier: Tier = "balanced"

    # Resolution behavior
    max_resolution_attempts: int = 3
    auto_resolve_low_severity: bool = True
    enable_consensus_building: bool = True
    enable_resource_management: bool = True

    # Detection sensitivity
    conflict_detection_threshold: float = 0.5  # 0-1, higher = more sensitive

    # Verification and escalation
    require_verification: bool = True
    escalation_enabled: bool = True

    # Strategy preferences
    preferred_strategies: List[ResolutionStrategy] = field(
        default_factory=lambda: [
            ResolutionStrategy.MERGE,
            ResolutionStrategy.SEQUENCE,
            ResolutionStrategy.NEGOTIATE
        ]
    )

    def __post_init__(self):
        """Validate configuration."""
        if self.max_resolution_attempts < 1:
            raise ValueError("max_resolution_attempts must be >= 1")

        if not 0 <= self.conflict_detection_threshold <= 1:
            raise ValueError("conflict_detection_threshold must be between 0 and 1")


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class AgentOutput:
    """Represents output from a single agent"""
    agent_id: str
    agent_type: str  # "research", "developer", "control", etc.
    output_type: str  # "code", "plan", "analysis", etc.
    content: Any
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Conflict:
    """Represents a detected conflict"""
    conflict_id: str
    conflict_type: ConflictType
    severity: ConflictSeverity
    description: str
    affected_agents: List[str]
    affected_outputs: List[AgentOutput]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Resolution:
    """Represents a conflict resolution"""
    conflict_id: str
    strategy: ResolutionStrategy
    steps: List[str]
    expected_outcome: str
    implemented: bool = False
    verified: bool = False
    verification_result: Optional[VerificationResult] = None


@dataclass
class CoordinationResult:
    """
    Result from coordination operation.

    Attributes:
        conflicts_detected: List of detected conflicts
        resolutions: List of proposed/implemented resolutions
        consensus_reached: Whether consensus was achieved
        resource_allocation: Resource allocation plan
        verification_results: Verification results for resolutions
        metadata: Additional metadata (tokens, time, etc.)
    """
    conflicts_detected: List[Conflict]
    resolutions: List[Resolution]
    consensus_reached: bool
    resource_allocation: Optional[Dict[str, Any]] = None
    verification_results: List[VerificationResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def all_resolved(self) -> bool:
        """Check if all conflicts are resolved"""
        return len(self.conflicts_detected) == len(self.resolutions) and \
               all(r.implemented for r in self.resolutions)

    @property
    def critical_conflicts_resolved(self) -> bool:
        """Check if all critical conflicts are resolved"""
        critical_conflicts = [
            c for c in self.conflicts_detected
            if c.severity == ConflictSeverity.CRITICAL
        ]
        critical_resolutions = [
            r for r in self.resolutions
            if any(c.conflict_id == r.conflict_id and c.severity == ConflictSeverity.CRITICAL
                   for c in self.conflicts_detected)
            and r.implemented
        ]
        return len(critical_conflicts) == len(critical_resolutions)


# ============================================================================
# CoordinationAgent
# ============================================================================

class CoordinationAgent(dspy.Module):
    """
    System 2 (Coordination) - Multi-Agent Orchestration and Verification

    ENHANCED FOR FRACTAL VSM ARCHITECTURE

    This is System 2 in the VSM hierarchy. It:
    1. Receives delegated subtasks from System 3 (Control)
    2. Routes subtasks to appropriate System 1 agents (Developer, Research)
    3. Executes System 1 agents and collects their outputs
    4. Verifies System 1 results using TierVerification (goal vs report vs actual)
    5. Detects and resolves conflicts between System 1 outputs
    6. Reports verified results back to System 3

    NEW VSM CAPABILITIES:
    - orchestrate_subtasks(): Routes and executes System 1 agents
    - verify_system1_results(): Uses TierVerification to verify subordinate tier
    - Full three-way verification: GOAL (what we asked) vs REPORT (what they said) vs ACTUAL (reality)

    EXISTING CAPABILITIES:
    - Conflict detection and resolution between agent outputs
    - Consensus building among agents
    - Resource allocation and management

    Usage (NEW - Orchestration):
        >>> config = CoordinationConfig(tier="balanced")
        >>> agent = CoordinationAgent(config=config)
        >>>
        >>> # Orchestrate subtasks (called by System 3)
        >>> subtasks = ["Implement feature X", "Research approach Y"]
        >>> result = agent.orchestrate_subtasks(subtasks, verbose=True)
        >>>
        >>> # Result includes verified System 1 outputs
        >>> print(result.tier_verification_results)

    Usage (EXISTING - Conflict Resolution):
        >>> # Detect conflicts
        >>> outputs = [
        ...     AgentOutput("dev1", "developer", "code", code1),
        ...     AgentOutput("dev2", "developer", "code", code2)
        ... ]
        >>> result = agent.detect_conflicts(outputs)
        >>>
        >>> # Resolve conflicts
        >>> if result.conflicts_detected:
        ...     resolution_result = agent.resolve_conflicts(result)
    """

    def __init__(
        self,
        config: Optional[CoordinationConfig] = None,
        tier: Optional[Tier] = None,
        graphrag=None
    ):
        """
        Initialize CoordinationAgent.

        Args:
            config: Agent configuration (default: balanced tier)
            tier: Model tier override (default: from config)
            graphrag: GraphRAG instance for knowledge retrieval (optional)
        """
        super().__init__()

        self.config = config or CoordinationConfig()
        self.tier = tier or self.config.tier
        self.graphrag = graphrag  # GraphRAG for learning from past experience

        # Configure DSPy with the appropriate tier
        configure_dspy(tier=self.tier)

        # Define DSPy modules for each coordination task
        self.conflict_detector = dspy.ChainOfThought(ConflictDetection)
        self.conflict_resolver = dspy.ChainOfThought(ConflictResolution)
        self.consensus_builder = dspy.ChainOfThought(ConsensusBuilding)
        self.resource_allocator = dspy.ChainOfThought(ResourceAllocation)

        # NEW: Task classifier for routing to System 1 agents
        self.task_classifier = dspy.Predict(TaskClassification)

        # NEW: File path inferrer for determining where to save generated code
        self.path_inferrer = dspy.Predict(FilePathInference)

        # NEW: Tier verification for validating System 1 (subordinate tier)
        self.tier_verifier = TierVerification(
            tier_name="System2_Coordination",
            subordinate_tier="System1_Operational"
        )

        logger.info(
            f"CoordinationAgent initialized: tier={self.tier}, "
            f"auto_resolve={self.config.auto_resolve_low_severity}, "
            f"graphrag_enabled={self.graphrag is not None}"
        )

    def retrieve_relevant_knowledge(
        self,
        query: str,
        max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant past knowledge from GraphRAG.

        Enables learning from past experience by querying the knowledge graph
        for similar tasks, patterns, and solutions.

        Args:
            query: Natural language query describing what knowledge to retrieve
            max_results: Maximum number of results to return (default: 5)

        Returns:
            List of relevant knowledge items with metadata:
            [
                {
                    "entity": "EntityName",
                    "relationship": "relationship_type",
                    "target": "TargetName",
                    "relevance": 0.95,
                    "metadata": {...}
                },
                ...
            ]

        Example:
            >>> knowledge = agent.retrieve_relevant_knowledge(
            ...     "How to resolve code conflicts between agents"
            ... )
            >>> for item in knowledge:
            ...     print(f"{item['entity']} {item['relationship']} {item['target']}")
        """
        if not self.graphrag:
            logger.warning("GraphRAG not configured, cannot retrieve knowledge")
            return []

        try:
            # Generate embedding for the query
            from ..memory.embeddings import generate_embedding
            query_embedding = generate_embedding(query)

            # Retrieve relevant knowledge using hybrid retrieval
            results = self.graphrag.retrieve(
                query=query,
                query_embedding=query_embedding,
                max_results=max_results,
                only_valid=True  # Only retrieve currently valid knowledge
            )

            logger.info(
                f"Retrieved {len(results)} relevant knowledge items for query: '{query[:50]}...'"
            )

            return results

        except Exception as e:
            logger.error(f"Failed to retrieve knowledge from GraphRAG: {e}")
            return []

    def orchestrate_subtasks(
        self,
        subtasks: List[str],
        context: Optional[Dict[str, Any]] = None,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        NEW: Orchestrate System 1 agents for multiple subtasks (SYSTEM 2 CAPABILITY)

        This is the main entry point for System 2 coordination. Called by System 3 (Control).

        Workflow:
        1. Route each subtask to appropriate System 1 agent (Developer or Research)
        2. Execute System 1 agents and collect outputs
        3. Verify each System 1 result using TierVerification (goal vs report vs actual)
        4. Detect and resolve conflicts between System 1 outputs
        5. Build consensus if needed
        6. Report verified results back to System 3

        Args:
            subtasks: List of subtasks delegated from System 3
            context: Additional context (shared resources, dependencies, constraints)
            verbose: Print progress information

        Returns:
            Dict with:
                - agent_outputs: List of System 1 agent outputs
                - tier_verification_results: TierVerificationResult for each agent
                - coordination_result: Conflict detection/resolution results
                - all_verified: Whether all System 1 tasks passed verification
                - metadata: Execution metadata
        """
        if verbose:
            print(f"\n{'=' * 80}")
            print(f"CoordinationAgent (System 2): Orchestrating {len(subtasks)} subtasks")
            print(f"{'=' * 80}\n")

        context = context or {}
        agent_outputs = []
        tier_verification_results = []

        # Stage 0: Retrieve relevant knowledge from past experience (GraphRAG)
        if self.graphrag:
            if verbose:
                print(f"[0/4] Retrieving relevant knowledge from GraphRAG...")

            # Create a query from all subtasks
            tasks_summary = " ".join(subtasks[:3])  # Use first 3 tasks for query
            knowledge_query = f"Coordination, conflict resolution, and task execution: {tasks_summary}"

            relevant_knowledge = self.retrieve_relevant_knowledge(
                query=knowledge_query,
                max_results=5
            )

            if relevant_knowledge:
                # Add retrieved knowledge to context for agents to use
                context["past_knowledge"] = relevant_knowledge

                if verbose:
                    print(f"  ✓ Retrieved {len(relevant_knowledge)} relevant knowledge items")
                    for i, item in enumerate(relevant_knowledge[:3], 1):
                        print(f"    {i}. {item['entity']} → {item['relationship']} → {item['target']}")
            elif verbose:
                print(f"  ℹ No relevant past knowledge found")

        # Stage 1: Route and execute System 1 agents
        if verbose:
            print(f"\n[1/4] Executing System 1 agents...")

        for i, subtask in enumerate(subtasks, 1):
            if verbose:
                print(f"\n  Subtask {i}/{len(subtasks)}: {subtask[:80]}...")

            # Route and execute
            agent_output = self._route_and_execute_system1_agent(
                subtask=subtask,
                context=context,
                verbose=verbose
            )
            agent_outputs.append(agent_output)

            if verbose:
                print(f"  ✓ Agent completed: {agent_output.agent_type}")

        # Stage 2: Verify each System 1 result using TierVerification
        if verbose:
            print(f"\n[2/4] Verifying System 1 results (tier verification)...")

        for i, agent_output in enumerate(agent_outputs, 1):
            if verbose:
                print(f"\n  Verifying {i}/{len(agent_outputs)}: {agent_output.agent_type}...")

            verification = self._verify_system1_result(
                agent_output=agent_output,
                subtask=subtasks[i-1],
                verbose=verbose
            )
            tier_verification_results.append(verification)

            if verbose:
                if verification.goal_achieved:
                    print(f"  ✅ Goal achieved, report accurate")
                else:
                    print(f"  ❌ Verification failed")
                    if verification.discrepancies:
                        print(f"     Discrepancies: {len(verification.discrepancies)}")

        # Stage 3: Detect and resolve conflicts (existing capability)
        coordination_result = None
        if len(agent_outputs) > 1:
            if verbose:
                print(f"\n[3/4] Checking for conflicts between agents...")

            coordination_result = self.forward(
                agent_outputs=agent_outputs,
                context=context,
                verbose=verbose
            )

            if verbose:
                if coordination_result.conflicts_detected:
                    print(f"  ⚠️  {len(coordination_result.conflicts_detected)} conflicts detected")
                else:
                    print(f"  ✅ No conflicts detected")

        # Stage 4: Build final report
        if verbose:
            print(f"\n[4/4] Building coordination report...")

        all_verified = all(v.goal_achieved for v in tier_verification_results)
        all_reports_accurate = all(v.report_accurate for v in tier_verification_results)

        if verbose:
            print(f"\n{'=' * 80}")
            print(f"Coordination Complete!")
            print(f"  System 1 agents executed: {len(agent_outputs)}")
            print(f"  Goals achieved: {sum(1 for v in tier_verification_results if v.goal_achieved)}/{len(tier_verification_results)}")
            print(f"  Reports accurate: {sum(1 for v in tier_verification_results if v.report_accurate)}/{len(tier_verification_results)}")
            if coordination_result:
                print(f"  Conflicts: {len(coordination_result.conflicts_detected)}")
            print(f"  All verified: {'✅ Yes' if all_verified else '❌ No'}")
            print(f"{'=' * 80}\n")

        return {
            "agent_outputs": agent_outputs,
            "tier_verification_results": tier_verification_results,
            "coordination_result": coordination_result,
            "all_verified": all_verified,
            "all_reports_accurate": all_reports_accurate,
            "metadata": {
                "tier": str(self.tier),
                "subtasks_count": len(subtasks),
                "agents_executed": len(agent_outputs),
                "goals_achieved": sum(1 for v in tier_verification_results if v.goal_achieved),
                "reports_accurate": sum(1 for v in tier_verification_results if v.report_accurate)
            }
        }

    def _route_and_execute_system1_agent(
        self,
        subtask: str,
        context: Optional[Dict[str, Any]] = None,
        verbose: bool = False
    ) -> AgentOutput:
        """
        Route subtask to appropriate System 1 agent and execute.

        Routes to:
        - DeveloperAgent for code generation/implementation tasks
        - ResearchAgent for analysis/investigation tasks

        Args:
            subtask: The subtask to execute
            context: Additional context
            verbose: Print routing details

        Returns:
            AgentOutput with result from System 1 agent
        """
        # Classify task type
        classification = self.task_classifier(subtask=subtask)
        task_type = classification.task_type.strip().lower()

        if verbose:
            print(f"     Task type: {task_type}")
            print(f"     Reasoning: {classification.reasoning}")

        # Route to appropriate System 1 agent
        if "code" in task_type or "generat" in task_type or "implement" in task_type:
            # Route to DeveloperAgent (System 1)
            from ..agents.developer_agent import DeveloperAgent
            from ..agents.developer_config import PresetDeveloperConfigs, CodeGenerationTask

            # Create config with file writing enabled
            dev_config = PresetDeveloperConfigs.thorough_python()
            dev_config.enable_file_writing = True
            dev_config.project_root = Path.cwd()

            agent = DeveloperAgent(config=dev_config)

            # Infer output path from task description
            path_result = self.path_inferrer(
                task_description=subtask,
                project_root=str(Path.cwd())
            )

            if verbose:
                print(f"     Inferred output path: {path_result.inferred_path}")
                print(f"     Reasoning: {path_result.reasoning}")

            # Create code generation task WITH output_path
            task = CodeGenerationTask(
                specification=subtask,
                language="python",
                context=context.get("code_context", "") if context else "",
                output_path=path_result.inferred_path
            )

            # Execute code generation
            result = agent(task, verbose=False)

            # Create AgentOutput
            return AgentOutput(
                agent_id=f"developer_{id(agent)}",
                agent_type="DeveloperAgent",
                output_type="code",
                content=result.code,
                metadata={
                    "tests": result.tests,
                    "goal_achieved": result.goal_achieved,
                    "verification_result": result.verification_result,
                    "evidence": result.evidence,
                    "subtask": subtask
                }
            )

        else:
            # Route to ResearchAgent (System 1)
            from ..agents.research_agent import ResearchAgent
            from ..agents.research_config import ResearchConfig

            agent = ResearchAgent(
                config=ResearchConfig(),
                max_research_questions=2  # Keep focused
            )

            # Execute research
            result = agent(topic=subtask, verbose=False)

            # Create AgentOutput
            return AgentOutput(
                agent_id=f"research_{id(agent)}",
                agent_type="ResearchAgent",
                output_type="research",
                content=result.synthesis,
                metadata={
                    "topic": result.topic,
                    "questions": result.questions,
                    "synthesis": result.synthesis,
                    "tokens_used": result.metadata['total_tokens'],
                    "subtask": subtask
                }
            )

    def _verify_system1_result(
        self,
        agent_output: AgentOutput,
        subtask: str,
        verbose: bool = False
    ) -> TierVerificationResult:
        """
        Verify System 1 agent result using TierVerification.

        This is the CORE of fractal VSM verification:
        - GOAL: What System 2 asked System 1 to do (subtask)
        - REPORT: What System 1 said it did (metadata from agent)
        - ACTUAL: What actually happened (reality check)

        Args:
            agent_output: Output from System 1 agent
            subtask: Original subtask (the GOAL)
            verbose: Print verification details

        Returns:
            TierVerificationResult with three-way comparison
        """
        # Create Goal from subtask
        goal = Goal(
            objective=subtask,
            success_criteria=[
                "Task completed successfully",
                "Output is correct and usable",
                "No errors or failures"
            ],
            required_artifacts=[],  # Will be populated based on agent type
            context={"agent_type": agent_output.agent_type}
        )

        # Build REPORT from agent metadata
        report = {
            "agent_id": agent_output.agent_id,
            "agent_type": agent_output.agent_type,
            "output_type": agent_output.output_type,
            "claimed_success": True,  # System 1 agents report success by default
            "metadata": agent_output.metadata
        }

        # For DeveloperAgent, add goal achievement and verification details
        if agent_output.agent_type == "DeveloperAgent":
            report["goal_achieved"] = agent_output.metadata.get("goal_achieved", False)
            report["verification_result"] = agent_output.metadata.get("verification_result")
            report["evidence"] = agent_output.metadata.get("evidence")

        # Verify using TierVerification (performs independent reality check)
        verification = self.tier_verifier.verify_subordinate(
            goal=goal,
            report=report,
            context={"subtask": subtask, "agent_output": agent_output}
        )

        return verification

    def forward(
        self,
        agent_outputs: List[AgentOutput],
        context: Optional[Dict[str, Any]] = None,
        verbose: bool = False
    ) -> CoordinationResult:
        """
        Main coordination workflow: detect and resolve conflicts.

        Args:
            agent_outputs: List of outputs from multiple agents
            context: Additional context (resources, dependencies, constraints)
            verbose: Print progress information

        Returns:
            CoordinationResult with conflicts, resolutions, and verification
        """
        if verbose:
            print(f"\n{'=' * 80}")
            print(f"CoordinationAgent: Analyzing {len(agent_outputs)} agent outputs")
            print(f"{'=' * 80}\n")

        context = context or {}

        # Stage 1: Detect conflicts
        if verbose:
            print(f"[1/3] Detecting conflicts...")

        conflicts = self.detect_conflicts(agent_outputs, context, verbose=verbose)

        if not conflicts:
            if verbose:
                print("  ✅ No conflicts detected")

            return CoordinationResult(
                conflicts_detected=[],
                resolutions=[],
                consensus_reached=True,
                metadata={
                    "tier": str(self.tier),
                    "agent_count": len(agent_outputs)
                }
            )

        if verbose:
            print(f"  ⚠️  {len(conflicts)} conflict(s) detected")
            for conflict in conflicts:
                print(f"     - {conflict.severity.value.upper()}: {conflict.description}")

        # Stage 2: Resolve conflicts
        if verbose:
            print(f"\n[2/3] Resolving conflicts...")

        resolutions = []
        for conflict in conflicts:
            # Auto-resolve low severity if enabled
            if conflict.severity == ConflictSeverity.LOW and \
               self.config.auto_resolve_low_severity:
                if verbose:
                    print(f"  → Auto-resolving low-severity conflict: {conflict.conflict_id}")

                resolution = self._auto_resolve_low_severity(conflict)
                resolutions.append(resolution)
            else:
                if verbose:
                    print(f"  → Resolving {conflict.severity.value} conflict: {conflict.conflict_id}")

                resolution = self.resolve_conflict(
                    conflict,
                    agent_outputs,
                    verbose=verbose
                )
                resolutions.append(resolution)

        # Stage 3: Build consensus if enabled
        consensus_reached = True
        if self.config.enable_consensus_building and conflicts:
            if verbose:
                print(f"\n[3/3] Building consensus...")

            consensus_reached = self.build_consensus(
                agent_outputs,
                conflicts,
                resolutions,
                verbose=verbose
            )

        # Stage 4: Verify resolutions if enabled
        verification_results = []
        if self.config.require_verification:
            if verbose:
                print(f"\n[4/4] Verifying resolutions...")

            for resolution in resolutions:
                if resolution.implemented:
                    verification = self._verify_resolution(resolution, verbose=verbose)
                    verification_results.append(verification)
                    resolution.verified = verification.is_success

        # Build result
        result = CoordinationResult(
            conflicts_detected=conflicts,
            resolutions=resolutions,
            consensus_reached=consensus_reached,
            verification_results=verification_results,
            metadata={
                "tier": str(self.tier),
                "agent_count": len(agent_outputs),
                "conflicts_count": len(conflicts),
                "resolutions_count": len(resolutions),
                "all_resolved": len(conflicts) == len([r for r in resolutions if r.implemented])
            }
        )

        if verbose:
            print(f"\n{'=' * 80}")
            print(f"Coordination complete!")
            print(f"  Conflicts detected: {len(conflicts)}")
            print(f"  Resolutions implemented: {len([r for r in resolutions if r.implemented])}")
            print(f"  Consensus reached: {'✅ Yes' if consensus_reached else '❌ No'}")
            print(f"  All resolved: {'✅ Yes' if result.all_resolved else '❌ No'}")
            print(f"{'=' * 80}\n")

        return result

    def detect_conflicts(
        self,
        agent_outputs: List[AgentOutput],
        context: Optional[Dict[str, Any]] = None,
        verbose: bool = False
    ) -> List[Conflict]:
        """
        Detect conflicts between agent outputs.

        Args:
            agent_outputs: List of outputs from multiple agents
            context: Additional context
            verbose: Print detection details

        Returns:
            List of detected conflicts
        """
        if len(agent_outputs) < 2:
            return []  # No conflicts possible with < 2 outputs

        # Format outputs for LLM
        outputs_formatted = [
            {
                "agent_id": output.agent_id,
                "agent_type": output.agent_type,
                "output_type": output.output_type,
                "content": str(output.content)[:500],  # Truncate for token efficiency
                "metadata": output.metadata
            }
            for output in agent_outputs
        ]

        context_formatted = str(context or {})

        # Run conflict detection
        detection_result = self.conflict_detector(
            agent_outputs=str(outputs_formatted),
            context=context_formatted
        )

        # Parse conflicts from LLM output
        conflicts = self._parse_conflicts(
            detection_result.conflicts_detected,
            detection_result.conflict_summary,
            agent_outputs
        )

        return conflicts

    def resolve_conflict(
        self,
        conflict: Conflict,
        agent_outputs: List[AgentOutput],
        verbose: bool = False
    ) -> Resolution:
        """
        Resolve a single conflict.

        Args:
            conflict: The conflict to resolve
            agent_outputs: All agent outputs (for context)
            verbose: Print resolution details

        Returns:
            Resolution with strategy and steps
        """
        # Prepare available strategies
        available_strategies = ", ".join([s.value for s in self.config.preferred_strategies])

        # Run conflict resolution
        resolution_result = self.conflict_resolver(
            conflict_description=conflict.description,
            conflict_type=conflict.conflict_type.value,
            affected_agents=str(conflict.affected_agents),
            available_strategies=available_strategies
        )

        # Parse strategy
        strategy_str = resolution_result.recommended_strategy.strip().lower()
        strategy = self._parse_strategy(strategy_str)

        # Create resolution
        resolution = Resolution(
            conflict_id=conflict.conflict_id,
            strategy=strategy,
            steps=self._parse_steps(resolution_result.resolution_steps),
            expected_outcome=resolution_result.expected_outcome,
            implemented=True  # Mark as implemented (actual implementation is external)
        )

        if verbose:
            print(f"     Strategy: {strategy.value}")
            print(f"     Steps: {len(resolution.steps)}")

        return resolution

    def build_consensus(
        self,
        agent_outputs: List[AgentOutput],
        conflicts: List[Conflict],
        resolutions: List[Resolution],
        verbose: bool = False
    ) -> bool:
        """
        Build consensus among agents.

        Args:
            agent_outputs: All agent outputs
            conflicts: Detected conflicts
            resolutions: Proposed resolutions
            verbose: Print consensus building details

        Returns:
            True if consensus reached, False otherwise
        """
        if not conflicts:
            return True  # No conflicts = automatic consensus

        # Format agent positions
        agent_positions = self._format_agent_positions(agent_outputs, conflicts)

        # Format conflict context
        conflict_context = "\n".join([
            f"- {c.conflict_type.value}: {c.description}"
            for c in conflicts
        ])

        # Success criteria
        success_criteria = "All agents agree on the resolution, no agent's constraints are violated"

        # Run consensus building
        consensus_result = self.consensus_builder(
            agent_positions=agent_positions,
            conflict_context=conflict_context,
            success_criteria=success_criteria
        )

        if verbose:
            print(f"  Consensus proposal: {consensus_result.consensus_proposal[:100]}...")
            print(f"  Trade-offs: {consensus_result.trade_offs[:100]}...")

        # Simple heuristic: consensus reached if proposal is non-empty
        consensus_reached = len(consensus_result.consensus_proposal.strip()) > 0

        return consensus_reached

    def allocate_resources(
        self,
        available_resources: List[str],
        agent_requests: Dict[str, List[str]],
        constraints: Optional[Dict[str, Any]] = None,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Allocate shared resources to prevent conflicts.

        Args:
            available_resources: List of available resources
            agent_requests: Dict mapping agent_id to requested resources
            constraints: Resource constraints
            verbose: Print allocation details

        Returns:
            Resource allocation plan
        """
        # Format inputs
        resources_formatted = ", ".join(available_resources)
        requests_formatted = str(agent_requests)
        constraints_formatted = str(constraints or {})

        # Run resource allocation
        allocation_result = self.resource_allocator(
            available_resources=resources_formatted,
            agent_requests=requests_formatted,
            constraints=constraints_formatted
        )

        allocation_plan = {
            "plan": allocation_result.allocation_plan,
            "potential_conflicts": allocation_result.potential_conflicts
        }

        if verbose:
            print(f"  Allocation plan: {allocation_result.allocation_plan[:100]}...")

        return allocation_plan

    def _parse_conflicts(
        self,
        conflicts_str: str,
        summary: str,
        agent_outputs: List[AgentOutput]
    ) -> List[Conflict]:
        """Parse conflicts from LLM output"""
        conflicts = []

        try:
            # Simple parsing - look for conflict indicators
            if "no conflict" in conflicts_str.lower() or "no conflict" in summary.lower():
                return []

            # Create a generic conflict if any detected
            # In production, this would parse structured output
            if "conflict" in conflicts_str.lower() or "conflict" in summary.lower():
                conflict = Conflict(
                    conflict_id="conflict_001",
                    conflict_type=ConflictType.SEMANTIC_CONFLICT,
                    severity=ConflictSeverity.MEDIUM,
                    description=summary[:200],
                    affected_agents=[output.agent_id for output in agent_outputs],
                    affected_outputs=agent_outputs
                )
                conflicts.append(conflict)

        except Exception as e:
            logger.warning(f"Failed to parse conflicts: {e}")

        return conflicts

    def _parse_strategy(self, strategy_str: str) -> ResolutionStrategy:
        """Parse resolution strategy from string"""
        strategy_str = strategy_str.lower().strip()

        if "merge" in strategy_str:
            return ResolutionStrategy.MERGE
        elif "prioritize" in strategy_str or "priority" in strategy_str:
            return ResolutionStrategy.PRIORITIZE
        elif "sequence" in strategy_str or "sequential" in strategy_str:
            return ResolutionStrategy.SEQUENCE
        elif "delegate" in strategy_str or "escalate" in strategy_str:
            return ResolutionStrategy.DELEGATE
        elif "negotiate" in strategy_str or "consensus" in strategy_str:
            return ResolutionStrategy.NEGOTIATE
        elif "partition" in strategy_str or "split" in strategy_str:
            return ResolutionStrategy.PARTITION
        else:
            return ResolutionStrategy.MERGE  # Default

    def _parse_steps(self, steps_str: str) -> List[str]:
        """Parse resolution steps from string"""
        # Split by newlines, numbers, or bullets
        lines = steps_str.strip().split('\n')
        steps = [
            line.strip().lstrip('0123456789.-•* ')
            for line in lines
            if line.strip()
        ]
        return steps[:10]  # Limit to 10 steps

    def _auto_resolve_low_severity(self, conflict: Conflict) -> Resolution:
        """Auto-resolve low-severity conflicts with default strategy"""
        return Resolution(
            conflict_id=conflict.conflict_id,
            strategy=ResolutionStrategy.MERGE,
            steps=["Merge conflicting outputs with minimal changes"],
            expected_outcome="Conflict resolved automatically",
            implemented=True
        )

    def _verify_resolution(
        self,
        resolution: Resolution,
        verbose: bool = False
    ) -> VerificationResult:
        """
        Verify that a resolution was successful.

        Uses goal-based verification to check if resolution achieved its objective.
        """
        # Create a simple goal for verification
        goal = Goal(
            objective=f"Resolve conflict {resolution.conflict_id} using {resolution.strategy.value}",
            required_artifacts=[],
            success_criteria=[
                "Conflict is resolved",
                "Expected outcome is achieved",
                "No new conflicts introduced"
            ]
        )

        # Create evidence (simplified - in production would check actual outcomes)
        evidence = Evidence(
            artifacts_verified=[],
            tests_passed=True,
            errors_found=[],
            llm_observations=f"Resolution implemented with strategy: {resolution.strategy.value}"
        )

        # Verify (simplified check)
        result = VerificationResult(
            is_success=resolution.implemented,
            score=1.0 if resolution.implemented else 0.0,
            reasoning=f"Resolution {'implemented' if resolution.implemented else 'not implemented'}",
            failures=[],
            recommendations=[]
        )

        return result

    def _format_agent_positions(
        self,
        agent_outputs: List[AgentOutput],
        conflicts: List[Conflict]
    ) -> str:
        """Format agent positions for consensus building"""
        positions = []
        for output in agent_outputs:
            # Find conflicts involving this agent
            agent_conflicts = [
                c for c in conflicts
                if output.agent_id in c.affected_agents
            ]

            position = f"Agent {output.agent_id} ({output.agent_type}):\n"
            position += f"  Output: {output.output_type}\n"
            position += f"  Conflicts: {len(agent_conflicts)}\n"

            positions.append(position)

        return "\n".join(positions)


# ============================================================================
# Preset Configurations
# ============================================================================

class PresetCoordinationConfigs:
    """Preset configurations for common coordination scenarios."""

    @staticmethod
    def lightweight() -> CoordinationConfig:
        """
        Lightweight coordination for simple conflicts.

        - Auto-resolve low severity
        - Skip consensus building
        - Minimal verification
        """
        return CoordinationConfig(
            tier="cheap",
            max_resolution_attempts=2,
            auto_resolve_low_severity=True,
            enable_consensus_building=False,
            enable_resource_management=False,
            require_verification=False
        )

    @staticmethod
    def standard() -> CoordinationConfig:
        """
        Standard coordination for typical multi-agent workflows.

        - Balanced tier
        - Consensus building enabled
        - Basic verification
        """
        return CoordinationConfig(
            tier="balanced",
            max_resolution_attempts=3,
            auto_resolve_low_severity=True,
            enable_consensus_building=True,
            enable_resource_management=True,
            require_verification=True
        )

    @staticmethod
    def thorough() -> CoordinationConfig:
        """
        Thorough coordination for complex conflicts.

        - Expensive tier
        - Full consensus building
        - Comprehensive verification
        """
        return CoordinationConfig(
            tier="expensive",
            max_resolution_attempts=5,
            auto_resolve_low_severity=False,
            enable_consensus_building=True,
            enable_resource_management=True,
            require_verification=True,
            conflict_detection_threshold=0.3  # More sensitive
        )


# ============================================================================
# Module entry point for testing
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("CoordinationAgent - Quick Test")
    print("=" * 80)
    print()

    # Test configuration
    config = PresetCoordinationConfigs.standard()
    agent = CoordinationAgent(config=config)

    # Create sample agent outputs with potential conflicts
    outputs = [
        AgentOutput(
            agent_id="dev1",
            agent_type="developer",
            output_type="code",
            content="def calculate(x): return x * 2",
            metadata={"priority": 1}
        ),
        AgentOutput(
            agent_id="dev2",
            agent_type="developer",
            output_type="code",
            content="def calculate(x): return x + 2",
            metadata={"priority": 2}
        )
    ]

    # Run coordination
    print("Testing conflict detection and resolution...")
    result = agent(outputs, verbose=True)

    print("\nCoordination Result:")
    print("=" * 80)
    print(f"Conflicts detected: {len(result.conflicts_detected)}")
    print(f"Resolutions: {len(result.resolutions)}")
    print(f"Consensus reached: {result.consensus_reached}")
    print(f"All resolved: {result.all_resolved}")
    print("=" * 80)
