"""
ResearchAgent - Multi-Stage Research with Synthesis

A VSM-inspired research agent that performs:
1. Research Planning (System 4: Intelligence)
2. Information Gathering (System 3: Coordination)
3. Synthesis & Analysis (System 5: Policy)
4. Validation (System 2: Monitoring)

Uses tier-based model selection:
- Cheap tier for planning and validation
- Balanced tier for research
- Expensive tier for synthesis

Author: BMad
Date: 2025-10-18
"""

import dspy
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from ..utils.dspy_integration import FractalDSpyLM, configure_dspy
from ..utils.model_config import Tier
from .research_config import ResearchConfig
from ..observability import (
    get_correlation_id, get_tracer, get_logger,
    get_event_store, VSMEvent, set_span_attributes
)
import logging

# Use observability-aware structured logger
logger = get_logger(__name__)


# ============================================================================
# DSPy Signatures for Research Workflow
# ============================================================================

class ResearchPlanning(dspy.Signature):
    """
    Plan a research strategy for a given topic.

    Break down the topic into specific research questions and
    identify key areas to investigate.
    """
    topic = dspy.InputField(desc="The research topic or question")
    domain_knowledge = dspy.InputField(desc="Relevant domain knowledge from past research and documentation")
    recent_examples = dspy.InputField(desc="Recent similar research tasks and their outcomes")
    constraints = dspy.InputField(desc="Guidelines and best practices for research planning")
    current_info = dspy.InputField(desc="Current information and recent developments related to the topic")
    research_plan = dspy.OutputField(desc="A structured research plan with specific questions")


class InformationGathering(dspy.Signature):
    """
    Gather information on a specific research question.

    Provide detailed, factual information with reasoning.
    """
    research_question = dspy.InputField(desc="The specific question to research")
    context = dspy.InputField(desc="Additional context about the research topic")
    domain_knowledge = dspy.InputField(desc="Relevant domain knowledge for answering this question")
    recent_examples = dspy.InputField(desc="Recent similar research findings")
    current_info = dspy.InputField(desc="Current information relevant to this question")
    findings = dspy.OutputField(desc="Detailed findings and information")


class SynthesisAndAnalysis(dspy.Signature):
    """
    Synthesize multiple research findings into coherent insights.

    Analyze the findings, identify patterns, and draw conclusions.
    """
    topic = dspy.InputField(desc="The original research topic")
    all_findings = dspy.InputField(desc="All research findings to synthesize")
    domain_knowledge = dspy.InputField(desc="Domain knowledge to inform synthesis")
    constraints = dspy.InputField(desc="Guidelines for synthesis quality and format")
    synthesis = dspy.OutputField(desc="Synthesized analysis with key insights and conclusions")


class ValidationCheck(dspy.Signature):
    """
    Validate research synthesis for completeness and accuracy.
    
    Check if the synthesis adequately addresses the original topic.
    """
    topic = dspy.InputField(desc="The original research topic")
    synthesis = dspy.InputField(desc="The synthesized research")
    validation = dspy.OutputField(desc="Validation result: complete/incomplete with reasons")


# ============================================================================
# Research Agent
# ============================================================================

@dataclass
class ResearchResult:
    """
    Result of a research operation.
    
    Attributes:
        topic: The original research topic
        research_plan: The generated research plan
        findings: List of findings for each research question
        synthesis: The synthesized analysis
        validation: Validation result
        metadata: Additional metadata (tokens used, models used, etc.)
    """
    topic: str
    research_plan: str
    findings: List[Dict[str, str]]
    synthesis: str
    validation: str
    metadata: Dict[str, Any]

    def __str__(self) -> str:
        """Human-readable string representation"""
        output = []
        output.append("=" * 80)
        output.append("RESEARCH REPORT")
        output.append("=" * 80)
        output.append(f"\nTopic: {self.topic}")
        output.append(f"\n{'-' * 80}")
        output.append("\nRESEARCH PLAN:")
        output.append(self.research_plan)
        output.append(f"\n{'-' * 80}")
        output.append("\nFINDINGS:")
        for i, finding in enumerate(self.findings, 1):
            output.append(f"\n{i}. Question: {finding['question']}")
            output.append(f"   Answer: {finding['answer'][:200]}...")
        output.append(f"\n{'-' * 80}")
        output.append("\nSYNTHESIS:")
        output.append(self.synthesis)
        output.append(f"\n{'-' * 80}")
        output.append("\nVALIDATION:")
        output.append(self.validation)
        output.append(f"\n{'-' * 80}")
        output.append(f"\nMetadata: {self.metadata}")
        output.append("=" * 80)
        return "\n".join(output)


class ResearchAgent(dspy.Module):
    """
    Multi-stage research agent with VSM-inspired architecture.

    Implements a four-stage research workflow:
    1. Planning (System 4) - Strategy and decomposition
    2. Gathering (System 3) - Coordinated information collection
    3. Synthesis (System 5) - High-level analysis and policy
    4. Validation (System 2) - Quality monitoring

    Uses tier-based model selection. DSPy can optimize prompts and demonstrations.

    Usage:
        >>> agent = ResearchAgent()
        >>> result = agent(topic="What is the Viable System Model?")
        >>> print(result)

        >>> # With custom config
        >>> config = ResearchConfig(planning_tier="premium")
        >>> agent = ResearchAgent(config=config)
        >>> result = agent(topic="Complex research topic")
    """

    def __init__(
        self,
        config: Optional[ResearchConfig] = None,
        max_research_questions: int = 3,
        graphrag=None,
        memory=None,
        obsidian_vault=None,
        enable_context_prep: bool = True
    ):
        """
        Initialize ResearchAgent.

        Args:
            config: ResearchConfig with tier selection (default: use defaults)
            max_research_questions: Maximum number of research questions to generate
            graphrag: GraphRAG instance for knowledge retrieval (optional)
            memory: ShortTermMemory for recent task history (optional)
            obsidian_vault: ObsidianVault for human-curated knowledge (optional)
            enable_context_prep: Whether to use intelligent context preparation (default: True)
        """
        super().__init__()  # Important for dspy.Module

        # Use provided config or create default
        self.config = config if config is not None else ResearchConfig()
        self.max_research_questions = max_research_questions
        self.graphrag = graphrag  # GraphRAG for learning from past research
        self.memory = memory  # ShortTermMemory for recent tasks
        self.obsidian_vault = obsidian_vault  # ObsidianVault for human knowledge
        self.enable_context_prep = enable_context_prep

        # Initialize context preparation agent if enabled
        self.context_prep_agent = None
        if self.enable_context_prep:
            try:
                from .context_preparation_agent import ContextPreparationAgent
                self.context_prep_agent = ContextPreparationAgent(
                    graphrag=self.graphrag,
                    memory=self.memory,
                    obsidian_vault=self.obsidian_vault,
                    web_search=None,  # Web search optional
                    max_iterations=2,  # Research agent gets 2 iterations max
                    min_confidence=0.7
                )
                logger.info("Context preparation agent initialized for ResearchAgent")
            except ImportError:
                logger.warning("ContextPreparationAgent not available, using simple context")
                self.enable_context_prep = False

        # Create LM instances for each stage
        # NO max_tokens by default - use full model context
        # Only apply if specified in config
        lm_kwargs = {}
        if self.config.max_tokens is not None:
            lm_kwargs["max_tokens"] = self.config.max_tokens
        if self.config.temperature is not None:
            lm_kwargs["temperature"] = self.config.temperature

        self.planning_lm = FractalDSpyLM(tier=self.config.planning_tier, **lm_kwargs)
        self.research_lm = FractalDSpyLM(tier=self.config.research_tier, **lm_kwargs)
        self.synthesis_lm = FractalDSpyLM(tier=self.config.synthesis_tier, **lm_kwargs)
        self.validation_lm = FractalDSpyLM(tier=self.config.validation_tier, **lm_kwargs)

        # Configure default LM (planning LM) so predictors can be created
        dspy.configure(lm=self.planning_lm)

        # Create DSPy modules for each stage
        # These use ChainOfThought for better reasoning
        self.planner = dspy.ChainOfThought(ResearchPlanning)
        self.gatherer = dspy.ChainOfThought(InformationGathering)
        self.synthesizer = dspy.ChainOfThought(SynthesisAndAnalysis)
        self.validator = dspy.Predict(ValidationCheck)

        logger.info(
            f"Initialized ResearchAgent with tiers: "
            f"planning={self.config.planning_tier}, research={self.config.research_tier}, "
            f"synthesis={self.config.synthesis_tier}, validation={self.config.validation_tier}, "
            f"graphrag_enabled={self.graphrag is not None}, "
            f"context_prep_enabled={self.enable_context_prep}"
        )

    def retrieve_relevant_knowledge(
        self,
        query: str,
        max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant past research from GraphRAG.

        Enables learning from past research by querying the knowledge graph
        for similar topics, findings, and patterns.

        Args:
            query: Natural language query describing what knowledge to retrieve
            max_results: Maximum number of results to return (default: 5)

        Returns:
            List of relevant knowledge items with metadata

        Example:
            >>> knowledge = agent.retrieve_relevant_knowledge(
            ...     "Past research on VSM System 1 operational capabilities"
            ... )
            >>> for item in knowledge:
            ...     print(f"{item['entity']} {item['relationship']} {item['target']}")
        """
        if not self.graphrag:
            logger.debug("GraphRAG not configured, cannot retrieve knowledge")
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
                f"Retrieved {len(results)} relevant knowledge items for research query"
            )

            return results

        except Exception as e:
            logger.error(f"Failed to retrieve knowledge from GraphRAG: {e}")
            return []

    def forward(
        self,
        topic: str,
        verbose: bool = True
    ) -> ResearchResult:
        """
        Perform multi-stage research on a topic.

        This is the DSPy Module forward() method.
        Call the agent instance directly: agent(topic="...")

        Args:
            topic: The research topic or question
            verbose: Whether to print progress updates

        Returns:
            ResearchResult with complete research findings
        """
        # OBSERVABILITY: Get tracer, event store, and correlation ID
        tracer = get_tracer(__name__)
        event_store = get_event_store()
        correlation_id = get_correlation_id()

        with tracer.start_as_current_span("system1_research") as span:
            # OBSERVABILITY: Set span attributes
            set_span_attributes({
                "vsm.tier": "System1_Research",
                "vsm.operation": "multi_stage_research",
                "task.topic": topic[:100],
                "correlation_id": correlation_id
            })

            # OBSERVABILITY: Emit research started event
            event_store.append(VSMEvent(
                tier="System1_Research",
                event_type="research_started",
                data={
                    "topic": topic,
                    "max_questions": self.max_research_questions,
                    "correlation_id": correlation_id
                }
            ))

            logger.info(
                f"System 1 (Research) started: {topic}",
                extra={"correlation_id": correlation_id, "topic": topic}
            )

            if verbose:
                print(f"\n{'=' * 80}")
                print(f"Starting research on: {topic}")
                print(f"{'=' * 80}\n")

            # Stage 0: Prepare context for planning
            if verbose:
                print(f"Stage 0: Preparing context for research planning...")

            # Use context preparation agent if available
            planning_context = None
            if self.context_prep_agent:
                planning_context = self.context_prep_agent.prepare_context(
                    user_task=f"Plan research strategy for: {topic}",
                    agent_type="research",
                    verbose=verbose
                )

                if verbose:
                    print(f"✓ Context prepared (confidence={planning_context.confidence:.2f})")
                    print(f"  Sources: {', '.join(planning_context.sources_used)}")
                    print(f"  Token budget: ~{planning_context.total_tokens}")
                    print()

            # Stage 1: Planning (System 4 - Intelligence)
            if verbose:
                print(f"Stage 1: Planning... (tier={self.config.planning_tier})")

            dspy.configure(lm=self.planning_lm)

            # Format context for planner
            if planning_context:
                formatted_context = planning_context.format_for_agent("research")
                plan_result = self.planner(
                    topic=topic,
                    domain_knowledge=formatted_context["domain_knowledge"],
                    recent_examples=formatted_context["recent_examples"],
                    constraints=formatted_context["constraints"],
                    current_info=formatted_context["current_info"]
                )
            else:
                # Fallback: no context
                plan_result = self.planner(
                    topic=topic,
                    domain_knowledge="No domain knowledge available",
                    recent_examples="No recent examples available",
                    constraints="Follow research best practices",
                    current_info="No current information available"
                )

            research_plan = plan_result.research_plan

            if verbose:
                print(f"✓ Research plan created")
                print(f"  Plan: {research_plan[:100]}...\n")

            # Extract research questions from plan
            research_questions = self._extract_questions(research_plan)

            # OBSERVABILITY: Emit planning completed event
            event_store.append(VSMEvent(
                tier="System1_Research",
                event_type="planning_completed",
                data={
                    "topic": topic,
                    "num_questions": len(research_questions),
                    "correlation_id": correlation_id
                }
            ))

            # Stage 2: Information Gathering (System 3 - Coordination)
            if verbose:
                print(f"Stage 2: Gathering information... (tier={self.config.research_tier})")

            dspy.configure(lm=self.research_lm)
            findings = []

            for i, question in enumerate(research_questions, 1):
                if verbose:
                    print(f"  {i}/{len(research_questions)}: {question}")

                # Prepare context for this specific question
                gathering_context = None
                if self.context_prep_agent:
                    gathering_context = self.context_prep_agent.prepare_context(
                        user_task=f"Research question: {question} (Topic: {topic})",
                        agent_type="research",
                        verbose=False  # Don't spam output for each question
                    )

                # Gather information with context
                if gathering_context:
                    formatted_context = gathering_context.format_for_agent("research")
                    gather_result = self.gatherer(
                        research_question=question,
                        context=f"Original topic: {topic}",
                        domain_knowledge=formatted_context["domain_knowledge"],
                        recent_examples=formatted_context["recent_examples"],
                        current_info=formatted_context["current_info"]
                    )
                else:
                    # Fallback: no context
                    gather_result = self.gatherer(
                        research_question=question,
                        context=f"Original topic: {topic}",
                        domain_knowledge="No domain knowledge available",
                        recent_examples="No recent examples available",
                        current_info="No current information available"
                    )

                findings.append({
                    "question": question,
                    "answer": gather_result.findings
                })

            if verbose:
                print(f"✓ Gathered {len(findings)} findings\n")

            # OBSERVABILITY: Emit gathering completed event
            event_store.append(VSMEvent(
                tier="System1_Research",
                event_type="gathering_completed",
                data={
                    "topic": topic,
                    "num_findings": len(findings),
                    "correlation_id": correlation_id
                }
            ))

            # Stage 3: Synthesis (System 5 - Policy)
            if verbose:
                print(f"Stage 3: Synthesizing... (tier={self.config.synthesis_tier})")

            dspy.configure(lm=self.synthesis_lm)

            # Format all findings for synthesis
            all_findings_text = "\n\n".join([
                f"Q: {f['question']}\nA: {f['answer']}"
                for f in findings
            ])

            # Prepare context for synthesis
            synthesis_context = None
            if self.context_prep_agent:
                synthesis_context = self.context_prep_agent.prepare_context(
                    user_task=f"Synthesize research findings on: {topic}",
                    agent_type="research",
                    verbose=False
                )

            # Synthesize with context
            if synthesis_context:
                formatted_context = synthesis_context.format_for_agent("research")
                synthesis_result = self.synthesizer(
                    topic=topic,
                    all_findings=all_findings_text,
                    domain_knowledge=formatted_context["domain_knowledge"],
                    constraints=formatted_context["constraints"]
                )
            else:
                # Fallback: no context
                synthesis_result = self.synthesizer(
                    topic=topic,
                    all_findings=all_findings_text,
                    domain_knowledge="No domain knowledge available",
                    constraints="Provide clear, comprehensive synthesis"
                )

            synthesis = synthesis_result.synthesis

            if verbose:
                print(f"✓ Synthesis complete\n")

            # OBSERVABILITY: Emit synthesis completed event
            event_store.append(VSMEvent(
                tier="System1_Research",
                event_type="synthesis_completed",
                data={
                    "topic": topic,
                    "synthesis_length": len(synthesis),
                    "correlation_id": correlation_id
                }
            ))

            # Stage 4: Validation (System 2 - Monitoring)
            if verbose:
                print(f"Stage 4: Validating... (tier={self.config.validation_tier})")

            dspy.configure(lm=self.validation_lm)
            validation_result = self.validator(
                topic=topic,
                synthesis=synthesis
            )
            validation = validation_result.validation

            if verbose:
                print(f"✓ Validation complete\n")

            # OBSERVABILITY: Emit validation completed event
            event_store.append(VSMEvent(
                tier="System1_Research",
                event_type="validation_completed",
                data={
                    "topic": topic,
                    "validation": validation,
                    "correlation_id": correlation_id
                }
            ))

            # Collect metadata
            metadata = {
                "num_questions": len(research_questions),
                "planning_metrics": self.planning_lm.get_metrics(),
                "research_metrics": self.research_lm.get_metrics(),
                "synthesis_metrics": self.synthesis_lm.get_metrics(),
                "validation_metrics": self.validation_lm.get_metrics(),
                "total_tokens": (
                    self.planning_lm.get_metrics()['total_tokens'] +
                    self.research_lm.get_metrics()['total_tokens'] +
                    self.synthesis_lm.get_metrics()['total_tokens'] +
                    self.validation_lm.get_metrics()['total_tokens']
                )
            }

            # Create result
            result = ResearchResult(
                topic=topic,
                research_plan=research_plan,
                findings=findings,
                synthesis=synthesis,
                validation=validation,
                metadata=metadata
            )

            # OBSERVABILITY: Emit research completed event
            event_store.append(VSMEvent(
                tier="System1_Research",
                event_type="research_completed",
                data={
                    "topic": topic,
                    "num_questions": len(research_questions),
                    "num_findings": len(findings),
                    "total_tokens": metadata['total_tokens'],
                    "correlation_id": correlation_id
                }
            ))

            logger.info(
                f"System 1 (Research) completed: {topic}",
                extra={
                    "correlation_id": correlation_id,
                    "topic": topic,
                    "total_tokens": metadata['total_tokens']
                }
            )

            if verbose:
                print(f"{'=' * 80}")
                print(f"Research complete!")
                print(f"Total tokens used: {metadata['total_tokens']}")
                print(f"{'=' * 80}\n")

            return result

    def _extract_questions(self, research_plan: str) -> List[str]:
        """
        Extract research questions from the plan.
        
        Looks for numbered questions or bullet points.
        Falls back to splitting by sentences if no clear structure.
        """
        questions = []

        # Try to find numbered questions
        lines = research_plan.split('\n')
        for line in lines:
            line = line.strip()
            # Look for patterns like "1.", "1)", "Q1:", "Question 1:", etc.
            if any(line.startswith(prefix) for prefix in ['1', '2', '3', '4', '5', '-', '•', '*']):
                # Clean up the question
                question = line.lstrip('0123456789.)-•* :')
                if question and len(question) > 10:  # Reasonable question length
                    questions.append(question)

        # If we found questions, limit to max
        if questions:
            return questions[:self.max_research_questions]

        # Fallback: treat the whole plan as context and generate generic questions
        return [
            f"What are the key concepts related to this topic?",
            f"What are the main applications or use cases?",
            f"What are the benefits and challenges?"
        ][:self.max_research_questions]


# Quick test
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("=" * 80)
    print("ResearchAgent Test")
    print("=" * 80)
    print()

    # Create agent with defaults
    # planning=expensive, research=cheap, synthesis=balanced, validation=balanced
    agent = ResearchAgent(
        max_research_questions=2  # Limit for quick test
    )

    # Test research - call agent directly (DSPy Module pattern)
    result = agent(
        topic="What is the Viable System Model and how does it apply to organizations?",
        verbose=True
    )

    # Print result
    print(result)
