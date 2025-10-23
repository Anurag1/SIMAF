"""
Context Preparation Agent - Intelligent Context Assembly System

This agent intelligently prepares optimal context for execution agents by:
1. Analyzing task requirements to determine what knowledge is needed
2. Querying all available knowledge sources (GraphRAG, memory, Obsidian, web)
3. Evaluating if retrieved context is sufficient
4. Iteratively researching to fill gaps
5. Crafting the perfect context package (not too little, not too much)
6. Learning from outcomes to improve over time

CRITICAL PRINCIPLE: NO execution agent ever runs without perfect context.

Author: BMad
Date: 2025-10-22
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import dspy
from datetime import datetime

logger = logging.getLogger(__name__)


# ============================================================================
# DSPy Signatures for Context Preparation
# ============================================================================

class RequirementsAnalysis(dspy.Signature):
    """
    Analyze a task to determine what context and knowledge is needed for successful execution.

    Think carefully about:
    - What domain knowledge is required?
    - What recent examples would help?
    - What current/external information is needed?
    - How confident are we in this analysis?
    """

    task = dspy.InputField(desc="The task that will be executed by an agent")
    agent_type = dspy.InputField(desc="Type of agent (research, developer, coordination)")

    domain_knowledge_needed = dspy.OutputField(
        desc="List of domain/topic areas needed (e.g., ['VSM', 'Python concurrency', 'OAuth2'])"
    )
    recent_context_needed = dspy.OutputField(
        desc="What recent task history would help? (e.g., 'similar research tasks', 'recent code implementations')"
    )
    external_data_needed = dspy.OutputField(
        desc="What current/external info needed? (e.g., 'Python 3.12 docs', 'best practices 2025')"
    )
    confidence = dspy.OutputField(
        desc="Confidence (0.0-1.0) that we identified the right requirements"
    )
    reasoning = dspy.OutputField(
        desc="Explain WHY these requirements were identified"
    )


class ContextEvaluation(dspy.Signature):
    """
    Evaluate if retrieved context is sufficient for task execution.

    Think critically about:
    - Do we have all the information needed?
    - Is the context relevant or is there noise?
    - What critical pieces are missing?
    - Can we execute successfully with what we have?
    """

    task = dspy.InputField(desc="The task to be executed")
    requirements = dspy.InputField(desc="What knowledge we identified as needed")
    retrieved_context = dspy.InputField(desc="Summary of what we found from all sources")

    is_sufficient = dspy.OutputField(
        desc="True if we have enough context to execute well, False if critical gaps exist"
    )
    relevance_scores = dspy.OutputField(
        desc="Dict mapping each context source to relevance score 0.0-1.0"
    )
    missing_topics = dspy.OutputField(
        desc="List of critical knowledge areas still missing"
    )
    confidence = dspy.OutputField(
        desc="Confidence (0.0-1.0) that we can execute successfully with this context"
    )
    reasoning = dspy.OutputField(
        desc="Explain WHY this context is/isn't sufficient"
    )


class ContextCrafting(dspy.Signature):
    """
    Craft the optimal context package from all retrieved information.

    Select ONLY what's relevant and helpful:
    - Not too little: Include everything needed for success
    - Not too much: Exclude noise and irrelevant information
    - Well-formatted: Present context in a way the agent can use

    Think about what will actually help the agent execute well.
    """

    task = dspy.InputField(desc="The task to be executed")
    all_retrieved = dspy.InputField(desc="All context from all sources")
    evaluation = dspy.InputField(desc="Evaluation of what's relevant and what's missing")
    agent_type = dspy.InputField(desc="Type of agent that will receive this context")

    selected_context = dspy.OutputField(
        desc="Carefully selected relevant context pieces organized by type"
    )
    formatting_strategy = dspy.OutputField(
        desc="How to present this context to the agent (structure, emphasis, etc.)"
    )
    confidence = dspy.OutputField(
        desc="Confidence (0.0-1.0) that this context package is optimal"
    )
    reasoning = dspy.OutputField(
        desc="Explain WHY these pieces were selected and others excluded"
    )


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class RequirementsResult:
    """Result of requirements analysis"""
    domain_knowledge: List[str]
    recent_context: str
    external_data: List[str]
    confidence: float
    reasoning: str


@dataclass
class EvaluationResult:
    """Result of context evaluation"""
    is_sufficient: bool
    relevance_scores: Dict[str, float]
    missing_topics: List[str]
    confidence: float
    reasoning: str
    needs_more_research: bool  # Computed from is_sufficient and confidence


# ============================================================================
# Context Preparation Agent
# ============================================================================

class ContextPreparationAgent(dspy.Module):
    """
    Intelligent context preparation agent.

    This agent THINKS about what context is needed and intelligently assembles it
    from multiple sources. It NEVER lets an execution agent run without perfect context.

    Workflow:
    1. Analyze task requirements (what knowledge is needed?)
    2. Query ALL sources in parallel (GraphRAG, memory, Obsidian, web)
    3. Evaluate sufficiency (do we have enough? what's missing?)
    4. Iterate to fill gaps (research missing topics)
    5. Craft optimal context package (select only relevant pieces)
    6. Log for continuous improvement

    Usage:
        >>> prep_agent = ContextPreparationAgent(
        ...     graphrag=graphrag,
        ...     memory=memory,
        ...     obsidian_vault=vault
        ... )
        >>> context = prep_agent.prepare_context(
        ...     user_task="Research VSM coordination patterns",
        ...     agent_type="research"
        ... )
        >>> # context is now ready for execution agent
    """

    def __init__(
        self,
        graphrag: Optional[Any] = None,
        memory: Optional[Any] = None,
        obsidian_vault: Optional[Any] = None,
        web_search: Optional[Any] = None,
        max_iterations: int = 3,
        min_confidence: float = 0.7
    ):
        """
        Initialize context preparation agent.

        Args:
            graphrag: GraphRAG instance for domain knowledge retrieval
            memory: ShortTermMemory instance for recent task history
            obsidian_vault: ObsidianVault instance for human-curated knowledge
            web_search: WebSearch instance for current information
            max_iterations: Maximum research iterations to fill gaps
            min_confidence: Minimum confidence threshold for sufficient context
        """
        super().__init__()

        # Knowledge sources
        self.graphrag = graphrag
        self.memory = memory
        self.obsidian_vault = obsidian_vault
        self.web_search = web_search

        # Configuration
        self.max_iterations = max_iterations
        self.min_confidence = min_confidence

        # DSPy modules for intelligent analysis
        self.requirements_analyzer = dspy.ChainOfThought(RequirementsAnalysis)
        self.context_evaluator = dspy.ChainOfThought(ContextEvaluation)
        self.context_crafter = dspy.ChainOfThought(ContextCrafting)

        # Preparation history for learning
        self.preparation_history: List[Dict[str, Any]] = []

        logger.info(f"Initialized ContextPreparationAgent with sources: "
                   f"GraphRAG={graphrag is not None}, "
                   f"Memory={memory is not None}, "
                   f"Obsidian={obsidian_vault is not None}, "
                   f"Web={web_search is not None}")

    def prepare_context(
        self,
        user_task: str,
        agent_type: str,
        verbose: bool = False
    ) -> 'ContextPackage':
        """
        Main workflow: Intelligently prepare optimal context for task execution.

        This method orchestrates the entire context preparation process:
        1. Analyze what's needed
        2. Query all sources
        3. Evaluate sufficiency
        4. Iterate if needed
        5. Craft final package
        6. Log for improvement

        Args:
            user_task: The task that will be executed
            agent_type: Type of agent (research, developer, coordination)
            verbose: Print detailed progress

        Returns:
            ContextPackage with optimal context for execution
        """
        if verbose:
            print(f"\n{'='*80}")
            print(f"CONTEXT PREPARATION")
            print(f"{'='*80}")
            print(f"Task: {user_task}")
            print(f"Agent: {agent_type}")
            print(f"{'='*80}\n")

        start_time = datetime.now()

        # Step 1: Analyze requirements
        if verbose:
            print("[1/6] Analyzing task requirements...")

        requirements = self.analyze_requirements(user_task, agent_type, verbose)

        if verbose:
            print(f"  Domain knowledge: {requirements.domain_knowledge}")
            print(f"  Recent context: {requirements.recent_context}")
            print(f"  External data: {requirements.external_data}")
            print(f"  Confidence: {requirements.confidence:.2f}")
            print()

        # Step 2: Query all sources in parallel
        if verbose:
            print("[2/6] Querying all knowledge sources...")

        retrieved = self.query_all_sources(requirements, verbose)

        if verbose:
            sources_found = [s for s, data in retrieved.items() if data]
            print(f"  Found context from: {', '.join(sources_found)}")
            print()

        # Step 3: Evaluate sufficiency
        if verbose:
            print("[3/6] Evaluating context sufficiency...")

        evaluation = self.evaluate_retrieved_context(
            user_task, requirements, retrieved, verbose
        )

        if verbose:
            print(f"  Sufficient: {evaluation.is_sufficient}")
            print(f"  Confidence: {evaluation.confidence:.2f}")
            if evaluation.missing_topics:
                print(f"  Missing: {evaluation.missing_topics}")
            print()

        # Step 4: Iterate if needed (research missing topics)
        iteration = 0
        while evaluation.needs_more_research and iteration < self.max_iterations:
            iteration += 1

            if verbose:
                print(f"[4/6] Iteration {iteration}: Researching missing topics...")
                print(f"  Topics: {evaluation.missing_topics}")

            additional = self.research_missing_context(
                evaluation.missing_topics, verbose
            )

            # Merge new findings
            for source, data in additional.items():
                if source in retrieved:
                    # Merge with existing
                    if isinstance(retrieved[source], list):
                        retrieved[source].extend(data)
                    elif isinstance(retrieved[source], dict):
                        retrieved[source].update(data)
                else:
                    retrieved[source] = data

            # Re-evaluate with new context
            evaluation = self.evaluate_retrieved_context(
                user_task, requirements, retrieved, verbose
            )

            if verbose:
                print(f"  Updated confidence: {evaluation.confidence:.2f}")
                print()

        # Step 5: Craft optimal context package
        if verbose:
            print("[5/6] Crafting optimal context package...")

        context_package = self.craft_context_package(
            user_task, retrieved, evaluation, agent_type, verbose
        )

        # Add metadata
        context_package.iterations = iteration
        context_package.preparation_time = (datetime.now() - start_time).total_seconds()

        if verbose:
            print(f"  Confidence: {context_package.confidence:.2f}")
            print(f"  Sources used: {len(context_package.sources_used)}")
            print(f"  Preparation time: {context_package.preparation_time:.2f}s")
            print()

        # Step 6: Log for improvement
        if verbose:
            print("[6/6] Logging for continuous improvement...")

        self.log_preparation(
            user_task, agent_type, requirements, evaluation, context_package
        )

        if verbose:
            print(f"\n{'='*80}")
            print("CONTEXT PREPARATION COMPLETE")
            print(f"{'='*80}\n")

        return context_package

    def analyze_requirements(
        self,
        user_task: str,
        agent_type: str,
        verbose: bool = False
    ) -> RequirementsResult:
        """
        Analyze task to determine what context/knowledge is needed.

        Uses LLM (via DSPy) to intelligently analyze the task and identify:
        - Domain knowledge required
        - Recent task history that would help
        - External/current information needed

        Args:
            user_task: The task to analyze
            agent_type: Type of agent (affects what context is needed)
            verbose: Print analysis details

        Returns:
            RequirementsResult with identified needs
        """
        result = self.requirements_analyzer(
            task=user_task,
            agent_type=agent_type
        )

        return RequirementsResult(
            domain_knowledge=self._parse_list(result.domain_knowledge_needed),
            recent_context=result.recent_context_needed,
            external_data=self._parse_list(result.external_data_needed),
            confidence=self._parse_float(result.confidence),
            reasoning=result.reasoning
        )

    def query_all_sources(
        self,
        requirements: RequirementsResult,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Query ALL knowledge sources in parallel.

        Uses MultiSourceRetriever to query GraphRAG, ShortTermMemory,
        ObsidianVault, and WebSearch concurrently.

        Args:
            requirements: What knowledge is needed
            verbose: Print query progress

        Returns:
            Dict mapping source name to retrieved data
        """
        from ..memory.multi_source_retriever import MultiSourceRetriever

        if verbose:
            logger.info("Querying all knowledge sources in parallel...")

        # Create retriever with available sources
        retriever = MultiSourceRetriever(
            graphrag=self.graphrag,
            short_term_memory=self.memory,
            obsidian_vault=self.obsidian_vault,
            enable_web_search=self.web_search is not None,
            max_workers=4
        )

        # Build comprehensive query from requirements
        query_parts = []

        # Add domain knowledge topics
        if requirements.domain_knowledge:
            query_parts.extend(requirements.domain_knowledge)

        # Add recent context needs
        if requirements.recent_context:
            query_parts.append(requirements.recent_context)

        # Add external data topics
        if requirements.external_data:
            query_parts.extend(requirements.external_data)

        # Combine into query string
        query = " ".join(query_parts) if query_parts else requirements.reasoning

        # Execute parallel retrieval
        result = retriever.retrieve_all(
            query=query,
            max_results_per_source=5,
            verbose=verbose
        )

        if verbose:
            summary = result.get_summary()
            logger.info(
                f"Retrieved {summary['total_results']} total results "
                f"from {summary['sources_succeeded']}/4 sources "
                f"in {result.total_time:.2f}s"
            )

        # Return results organized by source
        return result.get_all_results()

    def evaluate_retrieved_context(
        self,
        user_task: str,
        requirements: RequirementsResult,
        retrieved: Dict[str, Any],
        verbose: bool = False
    ) -> EvaluationResult:
        """
        Evaluate if retrieved context is sufficient for execution.

        Uses LLM to critically analyze:
        - Do we have all needed information?
        - Is context relevant or noisy?
        - What's missing?

        Args:
            user_task: The task to execute
            requirements: What we identified as needed
            retrieved: What we found from all sources
            verbose: Print evaluation details

        Returns:
            EvaluationResult with sufficiency assessment
        """
        # Format retrieved context for evaluation
        context_summary = self._summarize_retrieved(retrieved)
        requirements_summary = self._summarize_requirements(requirements)

        result = self.context_evaluator(
            task=user_task,
            requirements=requirements_summary,
            retrieved_context=context_summary
        )

        confidence = self._parse_float(result.confidence)
        is_sufficient = self._parse_bool(result.is_sufficient)

        # Determine if more research needed
        needs_more = not is_sufficient or confidence < self.min_confidence

        return EvaluationResult(
            is_sufficient=is_sufficient,
            relevance_scores=self._parse_dict(result.relevance_scores),
            missing_topics=self._parse_list(result.missing_topics),
            confidence=confidence,
            reasoning=result.reasoning,
            needs_more_research=needs_more
        )

    def research_missing_context(
        self,
        missing_topics: List[str],
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Research missing topics to fill context gaps.

        Attempts to fill knowledge gaps using available sources:
        1. Web search (if enabled)
        2. ObsidianVault search
        3. Short-term memory

        For full research integration with ResearchAgent, this would trigger
        a research workflow to find and store missing information in GraphRAG.

        Args:
            missing_topics: List of topics to research
            verbose: Print research progress

        Returns:
            Dict of additional context found
        """
        if not missing_topics:
            return {}

        additional_context = {}

        # Try web search if available
        if self.web_search:
            try:
                for topic in missing_topics[:3]:  # Limit to top 3 topics
                    results = self.web_search.search(topic, max_results=2)
                    if results:
                        additional_context[f"web_{topic}"] = results
            except Exception as e:
                logger.debug(f"Web search failed for missing topics: {e}")

        # Try Obsidian vault if available
        if self.obsidian_vault and not additional_context:
            try:
                query = " ".join(missing_topics[:3])
                results = self.obsidian_vault.search_notes(query, limit=5)
                if results:
                    additional_context["obsidian_notes"] = results
            except Exception as e:
                logger.debug(f"Obsidian search failed for missing topics: {e}")

        # Note: For full implementation, would trigger ResearchAgent here
        # to conduct comprehensive research and store in GraphRAG
        if verbose and not additional_context:
            logger.info(f"No additional context found for {len(missing_topics)} missing topics")

        return additional_context

    def craft_context_package(
        self,
        user_task: str,
        retrieved: Dict[str, Any],
        evaluation: EvaluationResult,
        agent_type: str,
        verbose: bool = False
    ) -> 'ContextPackage':
        """
        Craft optimal context package from all retrieved information.

        Uses LLM to carefully select:
        - What to include (relevant and helpful)
        - What to exclude (noise and irrelevant)
        - How to format for target agent

        Args:
            user_task: The task to execute
            retrieved: All available context
            evaluation: Assessment of what's relevant
            agent_type: Type of target agent
            verbose: Print crafting details

        Returns:
            ContextPackage ready for execution agent
        """
        # Format inputs for crafting
        all_retrieved_summary = self._summarize_retrieved(retrieved)
        evaluation_summary = self._summarize_evaluation(evaluation)

        result = self.context_crafter(
            task=user_task,
            all_retrieved=all_retrieved_summary,
            evaluation=evaluation_summary,
            agent_type=agent_type
        )

        # Parse selected context
        selected = self._parse_dict(result.selected_context)

        # Import here to avoid circular dependency
        from ..memory.context_package import ContextPackage

        package = ContextPackage(
            domain_knowledge=selected.get("domain_knowledge", ""),
            recent_examples=selected.get("recent_examples", []),
            constraints=selected.get("constraints", ""),
            current_info=selected.get("current_info", ""),
            confidence=self._parse_float(result.confidence),
            sources_used=list(retrieved.keys()),
            total_tokens=0,  # TODO: Calculate
            preparation_time=0.0,  # Set by caller
            iterations=0,  # Set by caller
            requirements=evaluation.reasoning,
            evaluation=result.reasoning,
            context_pieces=[]  # TODO: Build from retrieved
        )

        return package

    def log_preparation(
        self,
        user_task: str,
        agent_type: str,
        requirements: RequirementsResult,
        evaluation: EvaluationResult,
        context_package: 'ContextPackage'
    ):
        """
        Log context preparation for continuous improvement.

        This data will be used by ContextImprovementAgent to learn
        which strategies work best.

        Args:
            user_task: The task
            agent_type: Agent type
            requirements: What we identified as needed
            evaluation: How we evaluated context
            context_package: Final context package
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "task": user_task,
            "agent_type": agent_type,
            "requirements": {
                "domain_knowledge": requirements.domain_knowledge,
                "recent_context": requirements.recent_context,
                "external_data": requirements.external_data,
                "confidence": requirements.confidence
            },
            "evaluation": {
                "is_sufficient": evaluation.is_sufficient,
                "confidence": evaluation.confidence,
                "missing_topics": evaluation.missing_topics
            },
            "result": {
                "confidence": context_package.confidence,
                "sources_used": context_package.sources_used,
                "iterations": context_package.iterations,
                "preparation_time": context_package.preparation_time
            }
        }

        self.preparation_history.append(log_entry)
        logger.info(f"Logged context preparation for task: {user_task[:50]}...")

    # Helper methods for parsing DSPy outputs

    def _parse_list(self, value: Any) -> List[str]:
        """Parse DSPy output to list"""
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            # Try to parse as list
            if value.startswith("[") and value.endswith("]"):
                import ast
                try:
                    return ast.literal_eval(value)
                except:
                    pass
            # Split by comma
            return [item.strip() for item in value.split(",") if item.strip()]
        return []

    def _parse_float(self, value: Any) -> float:
        """Parse DSPy output to float"""
        try:
            return float(value)
        except:
            return 0.0

    def _parse_bool(self, value: Any) -> bool:
        """Parse DSPy output to bool"""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ["true", "yes", "1"]
        return False

    def _parse_dict(self, value: Any) -> Dict[str, Any]:
        """Parse DSPy output to dict"""
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            import ast
            try:
                return ast.literal_eval(value)
            except:
                return {}
        return {}

    def _summarize_retrieved(self, retrieved: Dict[str, Any]) -> str:
        """Summarize retrieved context for LLM evaluation"""
        parts = []

        for source, data in retrieved.items():
            if data:
                parts.append(f"## {source.upper()}")
                if isinstance(data, list):
                    parts.append(f"Found {len(data)} items")
                elif isinstance(data, dict):
                    parts.append(f"Found {len(data)} entries")
                else:
                    parts.append(str(data)[:200])

        return "\n".join(parts) if parts else "No context retrieved yet"

    def _summarize_requirements(self, requirements: RequirementsResult) -> str:
        """Summarize requirements for LLM evaluation"""
        return f"""
Domain knowledge: {', '.join(requirements.domain_knowledge)}
Recent context: {requirements.recent_context}
External data: {', '.join(requirements.external_data)}
Confidence: {requirements.confidence:.2f}
"""

    def _summarize_evaluation(self, evaluation: EvaluationResult) -> str:
        """Summarize evaluation for context crafting"""
        return f"""
Sufficient: {evaluation.is_sufficient}
Confidence: {evaluation.confidence:.2f}
Relevance scores: {evaluation.relevance_scores}
Missing topics: {evaluation.missing_topics}
Reasoning: {evaluation.reasoning}
"""
