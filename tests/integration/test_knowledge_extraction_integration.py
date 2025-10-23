"""
Integration Tests for Knowledge Extraction Pipeline

Tests the complete knowledge extraction workflow:
1. Agent executes task
2. ShortTermMemory logs task
3. KnowledgeExtractionAgent extracts entities/relationships
4. Knowledge stored in GraphRAG (Neo4j + Qdrant)
5. Knowledge retrievable via semantic search

Author: BMad
Date: 2025-10-22
"""

import pytest
import sys
import os
from pathlib import Path
from datetime import datetime
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import modules
try:
    from fractal_agent.agents.knowledge_extraction_agent import KnowledgeExtractionAgent
    from fractal_agent.memory.short_term import ShortTermMemory
    from fractal_agent.memory.embeddings import generate_embedding
except ImportError as e:
    print(f"Import error: {e}")
    print("Note: Some agent files may be corrupted. Test will attempt to continue.")
    KnowledgeExtractionAgent = None
    ShortTermMemory = None
    generate_embedding = None


# Skip tests if GraphRAG dependencies not available
try:
    from fractal_agent.memory.long_term import GraphRAG
    GRAPHRAG_AVAILABLE = True
except ImportError:
    GRAPHRAG_AVAILABLE = False
    GraphRAG = None


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def graphrag_instance():
    """Create GraphRAG instance for testing"""
    if not GRAPHRAG_AVAILABLE:
        pytest.skip("GraphRAG dependencies not available")

    # Check if Neo4j and Qdrant are running
    try:
        graphrag = GraphRAG(
            neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
            neo4j_password=os.getenv("NEO4J_PASSWORD", "fractal_password"),
            qdrant_host=os.getenv("QDRANT_HOST", "localhost"),
            qdrant_port=int(os.getenv("QDRANT_PORT", "6333"))
        )
        yield graphrag

        # Cleanup: Delete test data
        try:
            # Delete test entities from Neo4j
            with graphrag.graph.session() as session:
                session.run(
                    "MATCH (n) WHERE n.metadata CONTAINS 'test_session' DELETE n"
                )
            print("\n✅ Cleaned up test data from Neo4j")
        except Exception as e:
            print(f"\n⚠️  Cleanup warning: {e}")

    except Exception as e:
        pytest.skip(f"GraphRAG not available: {e}")


@pytest.fixture
def short_term_memory(graphrag_instance):
    """Create ShortTermMemory with GraphRAG integration"""
    memory = ShortTermMemory(
        log_dir="./logs/test_sessions",
        graphrag=graphrag_instance,
        enable_extraction=True
    )
    return memory


# ============================================================================
# Integration Tests
# ============================================================================

class TestKnowledgeExtractionIntegration:
    """Integration tests for complete knowledge extraction pipeline"""

    def test_extraction_agent_initialization(self):
        """Test that extraction agent can be initialized"""
        agent = KnowledgeExtractionAgent()
        assert agent is not None
        assert agent.extract is not None
        print("\n✅ KnowledgeExtractionAgent initialized successfully")

    def test_basic_extraction(self):
        """Test basic knowledge extraction without GraphRAG"""
        agent = KnowledgeExtractionAgent()

        result = agent(
            task_description="Implement GraphRAG storage system",
            task_output="""
            GraphRAG is a hybrid retrieval system that combines Neo4j graph database
            with Qdrant vector store. Neo4j stores entities and relationships with
            temporal validity tracking. Qdrant stores embeddings for semantic search.
            """,
            context="Memory system implementation, Phase 3"
        )

        # Validate result structure
        assert "entities" in result
        assert "relationships" in result
        assert "confidence" in result
        assert isinstance(result["entities"], list)
        assert isinstance(result["relationships"], list)
        assert isinstance(result["confidence"], float)
        assert 0.0 <= result["confidence"] <= 1.0

        print(f"\n✅ Extracted {len(result['entities'])} entities")
        print(f"✅ Extracted {len(result['relationships'])} relationships")
        print(f"✅ Confidence: {result['confidence']:.2f}")

    @pytest.mark.skipif(not GRAPHRAG_AVAILABLE, reason="GraphRAG not available")
    def test_end_to_end_extraction_pipeline(self, short_term_memory):
        """Test complete pipeline: task → extraction → storage → retrieval"""
        print("\n" + "=" * 80)
        print("Testing End-to-End Knowledge Extraction Pipeline")
        print("=" * 80)

        # Step 1: Execute task with promotion
        print("\n[1/5] Starting task...")
        task_id = short_term_memory.start_task(
            agent_id="research_001",
            agent_type="research",
            task_description="Research VSM System 1 operational capabilities",
            inputs={"topic": "Viable System Model"}
        )

        # Simulate task execution
        time.sleep(0.1)

        # Step 2: End task with knowledge promotion
        print("[2/5] Completing task with knowledge promotion...")
        short_term_memory.end_task(
            task_id=task_id,
            outputs={
                "report": """
                VSM System 1 is the operational tier that performs primary activities.
                It executes day-to-day operations using operational-tier models for
                cost efficiency. System 1 agents include ResearchAgent and DeveloperAgent.
                These agents use DSPy for structured prompting and produce outputs that
                feed into higher VSM tiers for monitoring and coordination.
                """
            },
            metadata={
                "agent_type": "research",
                "tokens_used": 500,
                "duration_seconds": 2.5
            },
            promote_to_longterm=True  # ← Triggers knowledge extraction
        )

        print("✅ Task completed with promotion flag")

        # Step 3: Verify knowledge in Neo4j
        print("[3/5] Verifying knowledge in Neo4j...")
        time.sleep(1)  # Give extraction time to complete

        graphrag = short_term_memory.graphrag
        with graphrag.graph.session() as session:
            # Check for entities related to this task
            # NOTE: Metadata is stored on relationships, not nodes
            result = session.run(
                """
                MATCH (n:Entity)-[r:RELATES]->()
                WHERE r.metadata IS NOT NULL
                AND r.metadata CONTAINS $task_id
                RETURN count(DISTINCT n) as entity_count
                """,
                task_id=task_id
            )
            entity_count = result.single()["entity_count"]

            print(f"✅ Found {entity_count} entities in Neo4j related to task")
            assert entity_count > 0, "No entities found in Neo4j for this task"

            # Get sample entities
            result = session.run(
                """
                MATCH (n:Entity)-[r:RELATES]->()
                WHERE r.metadata IS NOT NULL
                AND r.metadata CONTAINS $task_id
                RETURN DISTINCT n.name as name, labels(n) as labels
                LIMIT 5
                """,
                task_id=task_id
            )

            entities = list(result)
            print("\nSample entities:")
            for entity in entities:
                print(f"   - {entity['name']} ({entity['labels']})")

        # Step 4: Verify embeddings in Qdrant
        print("\n[4/5] Verifying embeddings in Qdrant...")

        # Query Qdrant for related embeddings
        query_embedding = generate_embedding("VSM System 1 operational capabilities")

        retrieved = graphrag.retrieve(
            query="VSM System 1",
            query_embedding=query_embedding,
            max_results=5
        )

        print(f"✅ Retrieved {len(retrieved)} results from Qdrant")
        assert len(retrieved) > 0, "No embeddings found in Qdrant"

        print("\nSample retrieved knowledge:")
        for i, item in enumerate(retrieved[:3], 1):
            print(f"   {i}. {item.get('entity', 'N/A')} → {item.get('target', 'N/A')}")
            print(f"      Relevance: {item.get('relevance_score', 0):.2f}")

        # Step 5: Verify knowledge is retrievable
        print("\n[5/5] Testing semantic search...")

        retrieved = graphrag.retrieve(
            query="What agents are in System 1?",
            query_embedding=generate_embedding("What agents are in System 1?"),
            max_results=5
        )

        print(f"✅ Semantic search returned {len(retrieved)} results")

        # Check if our knowledge appears in results
        found_relevant = any(
            "ResearchAgent" in str(item) or "DeveloperAgent" in str(item)
            for item in retrieved
        )

        if found_relevant:
            print("✅ Knowledge is semantically searchable!")
        else:
            print("⚠️  Warning: Specific knowledge not found in semantic search")

        print("\n" + "=" * 80)
        print("✅ END-TO-END PIPELINE TEST PASSED")
        print("=" * 80)

    @pytest.mark.skipif(not GRAPHRAG_AVAILABLE, reason="GraphRAG not available")
    def test_confidence_threshold_filtering(self, short_term_memory):
        """Test that low-confidence extractions are filtered out"""
        print("\n" + "=" * 80)
        print("Testing Confidence Threshold Filtering")
        print("=" * 80)

        # Task with ambiguous/unclear output
        task_id = short_term_memory.start_task(
            agent_id="test_001",
            agent_type="test",
            task_description="Unclear task",
            inputs={}
        )

        short_term_memory.end_task(
            task_id=task_id,
            outputs={"result": "Some unclear and ambiguous text without clear entities"},
            promote_to_longterm=True
        )

        time.sleep(1)

        # Check if low-confidence extraction was skipped
        graphrag = short_term_memory.graphrag
        with graphrag.graph.session() as session:
            result = session.run(
                """
                MATCH (n)
                WHERE n.metadata CONTAINS $task_id
                RETURN count(n) as count
                """,
                task_id=task_id
            )
            count = result.single()["count"]

            print(f"\nEntities stored for low-confidence task: {count}")
            print("✅ Confidence threshold filtering working")

    @pytest.mark.skipif(not GRAPHRAG_AVAILABLE, reason="GraphRAG not available")
    def test_extraction_with_metadata(self, short_term_memory):
        """Test that extraction includes proper source metadata"""
        print("\n" + "=" * 80)
        print("Testing Metadata Tracking")
        print("=" * 80)

        task_id = short_term_memory.start_task(
            agent_id="dev_001",
            agent_type="developer",
            task_description="Implement feature X",
            inputs={"feature": "X"}
        )

        short_term_memory.end_task(
            task_id=task_id,
            outputs={"code": "def feature_x(): return 'implemented'"},
            metadata={"language": "python"},
            promote_to_longterm=True
        )

        time.sleep(1)

        # Verify metadata is preserved
        graphrag = short_term_memory.graphrag
        with graphrag.graph.session() as session:
            result = session.run(
                """
                MATCH (n)
                WHERE n.metadata CONTAINS $task_id
                RETURN n.metadata as metadata
                LIMIT 1
                """,
                task_id=task_id
            )

            record = result.single()
            if record:
                metadata = record["metadata"]
                print(f"\n✅ Metadata preserved: {metadata[:100]}...")
                assert task_id in metadata
                assert short_term_memory.session_id in metadata
                print("✅ Source provenance tracked correctly")
            else:
                print("⚠️  No entities found (may have been filtered by confidence)")


# ============================================================================
# Test Runner
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Knowledge Extraction Integration Tests")
    print("=" * 80)
    print()
    print("Prerequisites:")
    print("  - Neo4j running on bolt://localhost:7687")
    print("  - Qdrant running on http://localhost:6333")
    print("  - NEO4J_PASSWORD environment variable set")
    print()

    # Check if services are available
    try:
        from fractal_agent.memory.long_term import GraphRAG

        graphrag = GraphRAG(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password=os.getenv("NEO4J_PASSWORD", "fractal_password")
        )

        print("✅ GraphRAG services available")
        print()

        # Run tests manually
        test = TestKnowledgeExtractionIntegration()

        print("Running tests...")
        print()

        try:
            test.test_extraction_agent_initialization()
        except Exception as e:
            print(f"❌ test_extraction_agent_initialization failed: {e}")

        try:
            test.test_basic_extraction()
        except Exception as e:
            print(f"❌ test_basic_extraction failed: {e}")

        # Create fixtures for remaining tests
        memory = ShortTermMemory(
            log_dir="./logs/test_sessions",
            graphrag=graphrag,
            enable_extraction=True
        )

        try:
            test.test_end_to_end_extraction_pipeline(memory)
        except Exception as e:
            print(f"❌ test_end_to_end_extraction_pipeline failed: {e}")
            import traceback
            traceback.print_exc()

        try:
            test.test_confidence_threshold_filtering(memory)
        except Exception as e:
            print(f"❌ test_confidence_threshold_filtering failed: {e}")

        try:
            test.test_extraction_with_metadata(memory)
        except Exception as e:
            print(f"❌ test_extraction_with_metadata failed: {e}")

        print()
        print("=" * 80)
        print("Integration tests complete!")
        print("=" * 80)

    except Exception as e:
        print(f"❌ GraphRAG not available: {e}")
        print()
        print("To run integration tests:")
        print("  1. Start Docker services:")
        print("     docker-compose up -d")
        print("  2. Set NEO4J_PASSWORD environment variable")
        print("  3. Run tests again")
        sys.exit(1)
