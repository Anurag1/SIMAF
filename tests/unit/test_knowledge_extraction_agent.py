"""
Unit tests for KnowledgeExtractionAgent

Tests automatic knowledge extraction from task outputs.

Author: BMad
Date: 2025-10-22
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Direct import to avoid corrupted agent files
import importlib.util
spec = importlib.util.spec_from_file_location(
    "knowledge_extraction_agent",
    Path(__file__).parent.parent.parent / "fractal_agent" / "agents" / "knowledge_extraction_agent.py"
)
knowledge_extraction_agent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(knowledge_extraction_agent)

KnowledgeExtractionAgent = knowledge_extraction_agent.KnowledgeExtractionAgent
extract_knowledge = knowledge_extraction_agent.extract_knowledge
Entity = knowledge_extraction_agent.Entity
Relationship = knowledge_extraction_agent.Relationship


def test_entity_model():
    """Test Entity pydantic model"""
    entity = Entity(
        name="ResearchAgent",
        type="agent",
        description="Multi-stage research with synthesis"
    )

    assert entity.name == "ResearchAgent"
    assert entity.type == "agent"
    assert entity.description == "Multi-stage research with synthesis"

    entity_dict = entity.to_dict()
    assert entity_dict["name"] == "ResearchAgent"


def test_relationship_model():
    """Test Relationship pydantic model"""
    rel = Relationship(
        from_entity="ResearchAgent",
        to_entity="DSPy",
        type="uses",
        strength=0.9
    )

    assert rel.from_entity == "ResearchAgent"
    assert rel.to_entity == "DSPy"
    assert rel.type == "uses"
    assert rel.strength == 0.9

    rel_dict = rel.to_dict()
    assert rel_dict["from_entity"] == "ResearchAgent"


def test_knowledge_extraction_agent_initialization():
    """Test agent initialization"""
    agent = KnowledgeExtractionAgent()

    assert agent is not None
    assert agent.extract is not None
    assert agent.lm is not None


def test_knowledge_extraction_basic():
    """Test basic knowledge extraction"""
    agent = KnowledgeExtractionAgent()

    result = agent(
        task_description="Research VSM System 1",
        task_output="""
        VSM System 1 is the operational tier that performs primary activities.
        It handles day-to-day operations and produces outputs for higher tiers.
        System 1 uses operational-tier models for cost efficiency.
        """
    )

    # Check result structure
    assert "entities" in result
    assert "relationships" in result
    assert "confidence" in result
    assert "metadata" in result

    # Check metadata
    assert "entity_count" in result["metadata"]
    assert "relationship_count" in result["metadata"]

    # Basic validation
    assert isinstance(result["entities"], list)
    assert isinstance(result["relationships"], list)
    assert isinstance(result["confidence"], float)
    assert 0.0 <= result["confidence"] <= 1.0

    print(f"\n✅ Extracted {len(result['entities'])} entities")
    print(f"✅ Extracted {len(result['relationships'])} relationships")
    print(f"✅ Confidence: {result['confidence']:.2f}")


def test_knowledge_extraction_with_context():
    """Test extraction with additional context"""
    agent = KnowledgeExtractionAgent()

    result = agent(
        task_description="Implement GraphRAG storage",
        task_output="""
        GraphRAG combines Neo4j graph database with Qdrant vector store.
        Neo4j stores entities and relationships with temporal validity.
        Qdrant stores embeddings for semantic retrieval.
        """,
        context="Memory system implementation, Phase 3"
    )

    assert result is not None
    assert result["confidence"] > 0.0

    print(f"\n✅ Extraction with context successful")
    print(f"   Entities: {len(result['entities'])}")
    print(f"   Relationships: {len(result['relationships'])}")


def test_extract_knowledge_convenience_function():
    """Test convenience function"""
    result = extract_knowledge(
        task_description="Research DSPy framework",
        task_output="DSPy is a framework for building LLM applications with signatures and modules."
    )

    assert result is not None
    assert "entities" in result
    assert "relationships" in result
    assert "confidence" in result

    print(f"\n✅ Convenience function works")


def test_extraction_result_format():
    """Test that extraction result has correct format for GraphRAG storage"""
    agent = KnowledgeExtractionAgent()

    result = agent(
        task_description="Test task",
        task_output="Test output with concepts and relationships"
    )

    # Verify entities format
    for entity in result["entities"]:
        assert "name" in entity
        assert "type" in entity
        assert "description" in entity

    # Verify relationships format
    for rel in result["relationships"]:
        assert "from_entity" in rel
        assert "to_entity" in rel
        assert "type" in rel
        # strength is optional
        if "strength" in rel:
            assert 0.0 <= rel["strength"] <= 1.0

    print(f"\n✅ Result format valid for GraphRAG storage")


if __name__ == "__main__":
    print("=" * 80)
    print("Knowledge Extraction Agent Unit Tests")
    print("=" * 80)
    print()

    tests = [
        ("Entity Model", test_entity_model),
        ("Relationship Model", test_relationship_model),
        ("Agent Initialization", test_knowledge_extraction_agent_initialization),
        ("Basic Extraction", test_knowledge_extraction_basic),
        ("Extraction with Context", test_knowledge_extraction_with_context),
        ("Convenience Function", test_extract_knowledge_convenience_function),
        ("Result Format", test_extraction_result_format)
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            print(f"Running: {test_name}...")
            test_func()
            print(f"✅ PASSED: {test_name}")
            passed += 1
        except Exception as e:
            print(f"❌ FAILED: {test_name}")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        print()

    print("=" * 80)
    print(f"Tests: {passed} passed, {failed} failed, {passed + failed} total")
    print("=" * 80)

    sys.exit(0 if failed == 0 else 1)
