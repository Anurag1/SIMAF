"""
KnowledgeExtractionAgent - Automatic Knowledge Extraction from Task Outputs

A VSM System 1 (Operational) agent that extracts structured knowledge from
agent task outputs for storage in the long-term GraphRAG memory system.

Extracts:
- Entities: Key concepts, agents, techniques, tools
- Relationships: How entities relate to each other
- Confidence: Quality metric for extraction

Uses cheap/operational tier models for cost-effective continuous learning.

Author: BMad
Date: 2025-10-22
"""

import dspy
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pydantic import BaseModel, Field
from ..utils.dspy_integration import FractalDSpyLM, configure_dspy
from ..utils.model_config import Tier
from ..observability import (
    get_correlation_id, get_tracer, get_logger,
    get_event_store, VSMEvent, set_span_attributes
)

# Use observability-aware structured logger
logger = get_logger(__name__)


# ============================================================================
# Pydantic Models for Knowledge Structures
# ============================================================================

class Entity(BaseModel):
    """Extracted entity with type and description"""
    name: str = Field(description="Entity name (e.g., 'ResearchAgent', 'DSPy', 'GraphRAG')")
    type: str = Field(description="Entity type: concept, agent, technique, tool, system, pattern")
    description: str = Field(description="Brief description of the entity")

    def to_dict(self) -> Dict[str, str]:
        return {
            "name": self.name,
            "type": self.type,
            "description": self.description
        }


class Relationship(BaseModel):
    """Extracted relationship between entities"""
    from_entity: str = Field(description="Source entity name")
    to_entity: str = Field(description="Target entity name")
    type: str = Field(description="Relationship type: produces, uses, requires, implements, coordinates, monitors")
    strength: float = Field(ge=0.0, le=1.0, description="Confidence in relationship (0.0-1.0)")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_entity": self.from_entity,
            "to_entity": self.to_entity,
            "type": self.type,
            "strength": self.strength
        }


# ============================================================================
# DSPy Signatures for Knowledge Extraction
# ============================================================================

class KnowledgeExtractionSignature(dspy.Signature):
    """
    Extract structured knowledge from agent task output.

    Identify key entities (concepts, agents, techniques, tools) and
    relationships between them. Focus on actionable, reusable knowledge.
    """
    task_description: str = dspy.InputField(desc="What the agent was asked to do")
    task_output: str = dspy.InputField(desc="What the agent produced (text, code, analysis)")
    context: str = dspy.InputField(desc="Additional context (optional)")

    entities: str = dspy.OutputField(
        desc="JSON list of entities: [{name, type, description}]. Types: concept, agent, technique, tool, system, pattern"
    )
    relationships: str = dspy.OutputField(
        desc="JSON list of relationships: [{from_entity, to_entity, type, strength}]. Types: produces, uses, requires, implements, coordinates, monitors. Strength: 0.0-1.0"
    )
    confidence: float = dspy.OutputField(
        desc="Confidence in extraction quality (0.0-1.0). High=0.9+, Medium=0.7-0.9, Low=<0.7"
    )


# ============================================================================
# Knowledge Extraction Agent
# ============================================================================

@dataclass
class ExtractionResult:
    """
    Result of knowledge extraction.

    Attributes:
        entities: List of extracted entities
        relationships: List of extracted relationships
        confidence: Quality confidence (0.0-1.0)
        metadata: Additional metadata (tokens used, etc.)
    """
    entities: List[Dict[str, str]]
    relationships: List[Dict[str, Any]]
    confidence: float
    metadata: Dict[str, Any]


class KnowledgeExtractionAgent(dspy.Module):
    """
    VSM System 1 (Operational) agent for automatic knowledge extraction.

    Extracts structured knowledge from agent task outputs for storage
    in long-term GraphRAG memory.

    Usage:
        >>> agent = KnowledgeExtractionAgent()
        >>> result = agent(
        ...     task_description="Research VSM System 1",
        ...     task_output="VSM System 1 is the operational tier..."
        ... )
        >>> print(f"Extracted {len(result.entities)} entities")
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize knowledge extraction agent.

        Args:
            config: Optional configuration (tier, model overrides, etc.)
        """
        super().__init__()

        self.config = config or {}

        # Use operational/cheap tier for cost-effective extraction
        tier = self.config.get("tier", "cheap")

        # Configure DSPy with appropriate model
        # configure_dspy creates the LM and sets it globally
        self.lm = configure_dspy(tier=tier)

        # Create extraction module with Chain of Thought
        self.extract = dspy.ChainOfThought(KnowledgeExtractionSignature)

        logger.info(f"Initialized KnowledgeExtractionAgent with tier={tier}")

    def forward(
        self,
        task_description: str,
        task_output: str,
        context: str = ""
    ) -> ExtractionResult:
        """
        Extract knowledge from task output.

        Args:
            task_description: What the agent was asked to do
            task_output: What the agent produced
            context: Additional context (optional)

        Returns:
            ExtractionResult with entities, relationships, and confidence

        Raises:
            ValueError: If extraction fails or produces invalid results
        """
        tracer = get_tracer()
        with tracer.start_as_current_span("knowledge_extraction") as span:
            set_span_attributes({
                "agent.type": "knowledge_extraction",
                "task.description_length": len(task_description),
                "task.output_length": len(task_output)
            })

            try:
                # Run extraction
                result = self.extract(
                    task_description=task_description,
                    task_output=task_output,
                    context=context or ""
                )

                # Parse JSON outputs
                import json

                try:
                    entities = json.loads(result.entities)
                except json.JSONDecodeError:
                    logger.warning("Failed to parse entities JSON, using empty list")
                    entities = []

                try:
                    relationships = json.loads(result.relationships)
                except json.JSONDecodeError:
                    logger.warning("Failed to parse relationships JSON, using empty list")
                    relationships = []

                # Validate extraction quality
                if len(entities) == 0 and len(relationships) == 0:
                    logger.warning("No entities or relationships extracted")
                    confidence = 0.0
                else:
                    confidence = float(result.confidence) if hasattr(result, 'confidence') else 0.5

                # Create extraction result
                extraction_result = ExtractionResult(
                    entities=entities,
                    relationships=relationships,
                    confidence=confidence,
                    metadata={
                        "entity_count": len(entities),
                        "relationship_count": len(relationships),
                        "tier": self.config.get("tier", "cheap"),
                        "correlation_id": get_correlation_id()
                    }
                )

                # Log metrics
                event_store = get_event_store()
                event_store.append(VSMEvent(
                    tier="System1",
                    event_type="knowledge_extraction",
                    data={
                        "agent": "KnowledgeExtractionAgent",
                        "entities": len(entities),
                        "relationships": len(relationships),
                        "confidence": confidence
                    }
                ))

                set_span_attributes({
                    "extraction.entity_count": len(entities),
                    "extraction.relationship_count": len(relationships),
                    "extraction.confidence": confidence
                })

                logger.info(
                    f"Extracted {len(entities)} entities, {len(relationships)} relationships "
                    f"(confidence={confidence:.2f})"
                )

                return extraction_result

            except Exception as e:
                logger.error(f"Knowledge extraction failed: {e}", exc_info=True)
                set_span_attributes({"error": str(e)})
                raise ValueError(f"Knowledge extraction failed: {e}") from e

    def __call__(
        self,
        task_description: str,
        task_output: str,
        context: str = ""
    ) -> Dict[str, Any]:
        """
        Convenience method for extraction.

        Returns dict instead of ExtractionResult for backward compatibility.
        """
        result = self.forward(task_description, task_output, context)
        return {
            "entities": result.entities,
            "relationships": result.relationships,
            "confidence": result.confidence,
            "metadata": result.metadata
        }


# ============================================================================
# Helper Functions
# ============================================================================

def extract_knowledge(
    task_description: str,
    task_output: str,
    context: str = "",
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Convenience function for knowledge extraction.

    Args:
        task_description: What the agent was asked to do
        task_output: What the agent produced
        context: Additional context (optional)
        config: Optional configuration

    Returns:
        Dict with entities, relationships, confidence, and metadata

    Usage:
        >>> result = extract_knowledge(
        ...     "Research GraphRAG",
        ...     "GraphRAG combines graph and vector retrieval..."
        ... )
        >>> print(result["entities"])
    """
    agent = KnowledgeExtractionAgent(config=config)
    return agent(task_description, task_output, context)


# Demo
if __name__ == "__main__":
    print("=" * 80)
    print("Knowledge Extraction Agent Demo")
    print("=" * 80)
    print()

    try:
        # Test extraction
        agent = KnowledgeExtractionAgent()

        result = agent(
            task_description="Research VSM System 1",
            task_output="""
            VSM System 1 is the operational tier that performs primary activities.
            It handles day-to-day operations and produces outputs for higher tiers.
            System 1 uses operational-tier models for cost efficiency.
            """
        )

        print(f"✅ Extracted {len(result['entities'])} entities:")
        for entity in result["entities"]:
            print(f"   - {entity['name']} ({entity['type']}): {entity['description']}")
        print()

        print(f"✅ Extracted {len(result['relationships'])} relationships:")
        for rel in result["relationships"]:
            print(f"   - {rel['from_entity']} --[{rel['type']}]--> {rel['to_entity']} (strength={rel['strength']})")
        print()

        print(f"✅ Confidence: {result['confidence']:.2f}")
        print()

        print("=" * 80)
        print("Knowledge Extraction Demo Complete!")
        print("=" * 80)

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
