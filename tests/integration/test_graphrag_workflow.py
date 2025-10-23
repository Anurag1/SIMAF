"""
Integration tests for GraphRAG (Long-Term Memory) workflow

Tests the complete hybrid Graph + Vector search workflow with real databases.
Requires Docker services to be running (Neo4j + Qdrant).

Run: pytest tests/integration/test_graphrag_workflow.py -v
"""

import pytest
from datetime import datetime, timedelta
import random

from fractal_agent.memory.long_term import GraphRAG
from fractal_agent.memory.embeddings import EmbeddingProvider


# Mark all tests as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def graphrag():
    """
    Create GraphRAG instance for integration tests.

    Requires Docker services to be running:
    - Neo4j on ports 7474/7687
    - Qdrant on ports 6333/6334
    """
    try:
        graphrag = GraphRAG(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="fractal_password",
            qdrant_host="localhost",
            qdrant_port=6333,
            collection_name="fractal_test_integration",
            embedding_dim=1536
        )
        yield graphrag
        graphrag.close()
    except Exception as e:
        pytest.skip(f"Docker services not available: {e}")


@pytest.fixture(scope="module")
def embedding_provider():
    """Create embedding provider for generating test embeddings"""
    try:
        # Use sentence-transformers for fast, consistent test embeddings
        provider = EmbeddingProvider(provider="sentence-transformers", dimension=1536)
        return provider
    except Exception:
        # Fallback to mock embeddings if sentence-transformers not available
        return None


def generate_test_embedding(embedding_provider, text):
    """Generate embedding for test (with fallback to mock)"""
    if embedding_provider:
        return embedding_provider.generate(text)
    else:
        # Mock embedding for testing without sentence-transformers
        random.seed(hash(text) % 2**32)
        return [random.random() for _ in range(1536)]


class TestGraphRAGBasicWorkflow:
    """Test basic GraphRAG operations with real databases"""

    def test_store_and_retrieve_single_knowledge(self, graphrag, embedding_provider):
        """Test storing and retrieving a single knowledge triple"""
        # Generate embedding
        text = "ResearchAgent produces high-quality research reports"
        embedding = generate_test_embedding(embedding_provider, text)

        # Store knowledge
        graphrag.store_knowledge(
            entity="ResearchAgent",
            relationship="produces",
            target="research_report",
            embedding=embedding,
            metadata={"quality": "high", "confidence": 0.95}
        )

        # Retrieve knowledge
        query_embedding = generate_test_embedding(embedding_provider, "research report production")
        results = graphrag.retrieve(
            query="What does ResearchAgent produce?",
            query_embedding=query_embedding,
            max_results=5
        )

        # Verify results
        assert len(results) > 0, "Should retrieve at least one result"

        # Find our stored knowledge
        found = False
        for result in results:
            if (result["entity"] == "ResearchAgent" and
                result["relationship"] == "produces" and
                result["target"] == "research_report"):
                found = True
                assert result["metadata"]["quality"] == "high"
                assert result["metadata"]["confidence"] == 0.95
                assert result["t_invalid"] is None  # Should be currently valid
                break

        assert found, "Should find the stored knowledge triple"

    def test_store_multiple_knowledge_triples(self, graphrag, embedding_provider):
        """Test storing multiple related knowledge triples"""
        knowledge_triples = [
            {
                "entity": "IntelligenceAgent",
                "relationship": "analyzes",
                "target": "performance_metrics",
                "text": "IntelligenceAgent analyzes performance metrics",
                "metadata": {"tier": "expensive", "priority": "high"}
            },
            {
                "entity": "ControlAgent",
                "relationship": "coordinates",
                "target": "multi_agent_workflow",
                "text": "ControlAgent coordinates multi-agent workflows",
                "metadata": {"complexity": "high", "criticality": "medium"}
            },
            {
                "entity": "ResearchAgent",
                "relationship": "uses",
                "target": "MIPRO_optimizer",
                "text": "ResearchAgent uses MIPRO optimizer for prompt tuning",
                "metadata": {"version": "2.0", "status": "active"}
            }
        ]

        # Store all triples
        for triple in knowledge_triples:
            embedding = generate_test_embedding(embedding_provider, triple["text"])
            graphrag.store_knowledge(
                entity=triple["entity"],
                relationship=triple["relationship"],
                target=triple["target"],
                embedding=embedding,
                metadata=triple["metadata"]
            )

        # Retrieve and verify
        query_embedding = generate_test_embedding(embedding_provider, "agent capabilities and functions")
        results = graphrag.retrieve(
            query="What do the agents do?",
            query_embedding=query_embedding,
            max_results=10
        )

        assert len(results) >= 3, "Should retrieve at least the 3 stored triples"

    def test_temporal_validity_filtering(self, graphrag, embedding_provider):
        """Test temporal validity tracking and filtering"""
        # Store knowledge that's currently valid
        text_valid = "TestEntity performs valid operation"
        embedding_valid = generate_test_embedding(embedding_provider, text_valid)

        graphrag.store_knowledge(
            entity="TestEntity",
            relationship="performs",
            target="valid_operation",
            embedding=embedding_valid,
            metadata={"status": "valid"}
        )

        # Store knowledge that's already invalid
        text_invalid = "TestEntity performs invalid operation"
        embedding_invalid = generate_test_embedding(embedding_provider, text_invalid)

        past_time = datetime.now() - timedelta(days=1)
        graphrag.store_knowledge(
            entity="TestEntity",
            relationship="performs",
            target="invalid_operation",
            embedding=embedding_invalid,
            t_valid=past_time,
            t_invalid=past_time,  # Already invalid
            metadata={"status": "invalid"}
        )

        # Retrieve with only_valid=True
        query_embedding = generate_test_embedding(embedding_provider, "TestEntity operations")
        valid_results = graphrag.retrieve(
            query="What does TestEntity do?",
            query_embedding=query_embedding,
            max_results=10,
            only_valid=True
        )

        # Should only get valid results
        for result in valid_results:
            if result["entity"] == "TestEntity":
                assert result["t_invalid"] is None, "Only valid knowledge should be returned"

        # Retrieve with only_valid=False
        all_results = graphrag.retrieve(
            query="What does TestEntity do?",
            query_embedding=query_embedding,
            max_results=10,
            only_valid=False
        )

        # Should get both valid and invalid
        assert len(all_results) >= len(valid_results), "All results should include valid results"


class TestGraphRAGTemporalOperations:
    """Test temporal knowledge management"""

    def test_invalidate_knowledge(self, graphrag, embedding_provider):
        """Test marking knowledge as invalid"""
        # Store knowledge
        text = "TemporalEntity has temporary_property"
        embedding = generate_test_embedding(embedding_provider, text)

        graphrag.store_knowledge(
            entity="TemporalEntity",
            relationship="has",
            target="temporary_property",
            embedding=embedding,
            metadata={"temporary": True}
        )

        # Verify it's valid
        query_embedding = generate_test_embedding(embedding_provider, "TemporalEntity properties")
        results_before = graphrag.retrieve(
            query="TemporalEntity properties",
            query_embedding=query_embedding,
            max_results=5,
            only_valid=True
        )

        found_before = any(
            r["entity"] == "TemporalEntity" and r["target"] == "temporary_property"
            for r in results_before
        )
        assert found_before, "Knowledge should be valid before invalidation"

        # Invalidate the knowledge
        graphrag.invalidate_knowledge(
            entity="TemporalEntity",
            relationship="has",
            target="temporary_property"
        )

        # Verify it's no longer returned with only_valid=True
        results_after = graphrag.retrieve(
            query="TemporalEntity properties",
            query_embedding=query_embedding,
            max_results=5,
            only_valid=True
        )

        found_after = any(
            r["entity"] == "TemporalEntity" and r["target"] == "temporary_property"
            for r in results_after
        )
        assert not found_after, "Knowledge should not be valid after invalidation"

    def test_knowledge_evolution(self, graphrag, embedding_provider):
        """Test tracking knowledge evolution over time"""
        entity = "EvolvingAgent"

        # Version 1: Initial capability
        text_v1 = "EvolvingAgent uses simple strategy"
        embedding_v1 = generate_test_embedding(embedding_provider, text_v1)

        t_v1 = datetime.now() - timedelta(days=30)
        graphrag.store_knowledge(
            entity=entity,
            relationship="uses",
            target="simple_strategy",
            embedding=embedding_v1,
            t_valid=t_v1,
            metadata={"version": "1.0"}
        )

        # Invalidate v1
        graphrag.invalidate_knowledge(
            entity=entity,
            relationship="uses",
            target="simple_strategy"
        )

        # Version 2: Updated capability
        text_v2 = "EvolvingAgent uses advanced strategy"
        embedding_v2 = generate_test_embedding(embedding_provider, text_v2)

        graphrag.store_knowledge(
            entity=entity,
            relationship="uses",
            target="advanced_strategy",
            embedding=embedding_v2,
            metadata={"version": "2.0"}
        )

        # Retrieve current knowledge (only valid)
        query_embedding = generate_test_embedding(embedding_provider, "EvolvingAgent strategy")
        current_results = graphrag.retrieve(
            query="What strategy does EvolvingAgent use?",
            query_embedding=query_embedding,
            max_results=5,
            only_valid=True
        )

        # Should only get v2
        for result in current_results:
            if result["entity"] == entity:
                assert result["target"] == "advanced_strategy", "Should only get current version"
                assert result["metadata"]["version"] == "2.0"


class TestGraphRAGHybridSearch:
    """Test hybrid vector + graph search capabilities"""

    def test_semantic_similarity_search(self, graphrag, embedding_provider):
        """Test vector similarity finds semantically related knowledge"""
        # Store knowledge about similar concepts
        similar_knowledge = [
            {
                "entity": "DataProcessor",
                "relationship": "transforms",
                "target": "raw_data",
                "text": "DataProcessor transforms raw data into structured format"
            },
            {
                "entity": "DataCleaner",
                "relationship": "cleanses",
                "target": "noisy_data",
                "text": "DataCleaner cleanses noisy data and removes outliers"
            },
            {
                "entity": "DataValidator",
                "relationship": "validates",
                "target": "input_data",
                "text": "DataValidator validates input data for correctness"
            }
        ]

        for knowledge in similar_knowledge:
            embedding = generate_test_embedding(embedding_provider, knowledge["text"])
            graphrag.store_knowledge(
                entity=knowledge["entity"],
                relationship=knowledge["relationship"],
                target=knowledge["target"],
                embedding=embedding
            )

        # Search for semantically similar content
        query_text = "data processing and cleaning operations"
        query_embedding = generate_test_embedding(embedding_provider, query_text)

        results = graphrag.retrieve(
            query=query_text,
            query_embedding=query_embedding,
            max_results=5
        )

        # Should find data-related entities
        data_entities = {"DataProcessor", "DataCleaner", "DataValidator"}
        found_entities = {r["entity"] for r in results if r["entity"] in data_entities}

        assert len(found_entities) > 0, "Should find semantically related entities"

    def test_graph_traversal_relationships(self, graphrag, embedding_provider):
        """Test graph traversal finds related entities"""
        # Create a knowledge graph structure
        graph_structure = [
            {
                "entity": "SystemCoordinator",
                "relationship": "manages",
                "target": "SubsystemA",
                "text": "SystemCoordinator manages SubsystemA"
            },
            {
                "entity": "SystemCoordinator",
                "relationship": "manages",
                "target": "SubsystemB",
                "text": "SystemCoordinator manages SubsystemB"
            },
            {
                "entity": "SystemCoordinator",
                "relationship": "coordinates",
                "target": "InteractionPattern",
                "text": "SystemCoordinator coordinates interaction patterns"
            }
        ]

        for knowledge in graph_structure:
            embedding = generate_test_embedding(embedding_provider, knowledge["text"])
            graphrag.store_knowledge(
                entity=knowledge["entity"],
                relationship=knowledge["relationship"],
                target=knowledge["target"],
                embedding=embedding
            )

        # Query for SystemCoordinator
        query_embedding = generate_test_embedding(embedding_provider, "SystemCoordinator")
        results = graphrag.retrieve(
            query="What does SystemCoordinator do?",
            query_embedding=query_embedding,
            max_results=10
        )

        # Should find multiple relationships
        coordinator_results = [r for r in results if r["entity"] == "SystemCoordinator"]
        assert len(coordinator_results) >= 3, "Should find all SystemCoordinator relationships"


class TestGraphRAGScalability:
    """Test GraphRAG with larger datasets"""

    def test_batch_storage_and_retrieval(self, graphrag, embedding_provider):
        """Test storing and retrieving larger batches of knowledge"""
        # Generate 20 knowledge triples
        batch_size = 20
        knowledge_batch = []

        for i in range(batch_size):
            knowledge = {
                "entity": f"BatchEntity_{i % 5}",  # 5 different entities
                "relationship": "performs",
                "target": f"operation_{i}",
                "text": f"BatchEntity_{i % 5} performs operation_{i} efficiently",
                "metadata": {"batch": "test", "index": i}
            }
            knowledge_batch.append(knowledge)

        # Store batch
        for knowledge in knowledge_batch:
            embedding = generate_test_embedding(embedding_provider, knowledge["text"])
            graphrag.store_knowledge(
                entity=knowledge["entity"],
                relationship=knowledge["relationship"],
                target=knowledge["target"],
                embedding=embedding,
                metadata=knowledge["metadata"]
            )

        # Retrieve and verify count
        query_embedding = generate_test_embedding(embedding_provider, "BatchEntity operations")
        results = graphrag.retrieve(
            query="What do BatchEntities do?",
            query_embedding=query_embedding,
            max_results=25
        )

        # Should retrieve significant portion of stored knowledge
        batch_results = [r for r in results if r["entity"].startswith("BatchEntity_")]
        assert len(batch_results) >= 10, f"Should retrieve at least half of stored knowledge, got {len(batch_results)}"

    def test_complex_metadata_storage(self, graphrag, embedding_provider):
        """Test storing and retrieving complex nested metadata"""
        complex_metadata = {
            "performance": {
                "accuracy": 0.95,
                "latency_ms": 150,
                "throughput": 1000
            },
            "config": {
                "model": "claude-sonnet-4-5",
                "temperature": 0.7,
                "max_tokens": 4096
            },
            "tags": ["production", "validated", "optimized"],
            "timestamp": datetime.now().isoformat()
        }

        text = "ComplexEntity performs complex_operation with detailed configuration"
        embedding = generate_test_embedding(embedding_provider, text)

        graphrag.store_knowledge(
            entity="ComplexEntity",
            relationship="performs",
            target="complex_operation",
            embedding=embedding,
            metadata=complex_metadata
        )

        # Retrieve and verify metadata
        query_embedding = generate_test_embedding(embedding_provider, "ComplexEntity")
        results = graphrag.retrieve(
            query="ComplexEntity operations",
            query_embedding=query_embedding,
            max_results=5
        )

        # Find and verify our complex metadata
        found = False
        for result in results:
            if result["entity"] == "ComplexEntity":
                found = True
                assert result["metadata"]["performance"]["accuracy"] == 0.95
                assert result["metadata"]["config"]["model"] == "claude-sonnet-4-5"
                assert "production" in result["metadata"]["tags"]
                break

        assert found, "Should retrieve knowledge with complex metadata intact"


# Cleanup fixture
@pytest.fixture(scope="module", autouse=True)
def cleanup_test_data(graphrag):
    """Clean up test data after all tests complete"""
    yield
    # Note: In production, you might want to delete test collection/data
    # For now, using separate test collection keeps test data isolated
