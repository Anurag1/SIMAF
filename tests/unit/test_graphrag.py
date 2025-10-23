"""
Unit tests for GraphRAG (Long-Term Memory)

Tests the hybrid Graph + Vector search implementation using Neo4j and Qdrant.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
import json

from fractal_agent.memory.long_term import GraphRAG


class TestGraphRAGInitialization:
    """Test GraphRAG initialization and connection"""

    @patch('fractal_agent.memory.long_term.GraphDatabase')
    @patch('fractal_agent.memory.long_term.QdrantClient')
    def test_init_successful_connection(self, mock_qdrant, mock_neo4j):
        """Test successful initialization with both databases"""
        # Setup mocks
        mock_neo4j.driver.return_value.session.return_value.__enter__.return_value.run.return_value = None
        mock_qdrant.return_value.get_collections.return_value = []

        # Initialize
        graphrag = GraphRAG(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="test_password",
            qdrant_host="localhost",
            qdrant_port=6333
        )

        # Verify connections
        mock_neo4j.driver.assert_called_once()
        mock_qdrant.assert_called_once()
        assert graphrag.collection_name == "fractal_knowledge"
        assert graphrag.embedding_dim == 1536

    @patch('fractal_agent.memory.long_term.GraphDatabase')
    @patch('fractal_agent.memory.long_term.QdrantClient')
    def test_init_neo4j_connection_failure(self, mock_qdrant, mock_neo4j):
        """Test initialization fails when Neo4j connection fails"""
        mock_neo4j.driver.side_effect = Exception("Connection refused")

        with pytest.raises(ConnectionError, match="Neo4j connection failed"):
            GraphRAG(
                neo4j_uri="bolt://localhost:7687",
                neo4j_user="neo4j",
                neo4j_password="test_password"
            )

    @patch('fractal_agent.memory.long_term.GraphDatabase')
    @patch('fractal_agent.memory.long_term.QdrantClient')
    def test_init_qdrant_connection_failure(self, mock_qdrant, mock_neo4j):
        """Test initialization fails when Qdrant connection fails"""
        # Neo4j succeeds
        mock_neo4j.driver.return_value.session.return_value.__enter__.return_value.run.return_value = None
        # Qdrant fails
        mock_qdrant.return_value.get_collections.side_effect = Exception("Connection refused")

        with pytest.raises(ConnectionError, match="Qdrant connection failed"):
            GraphRAG(
                neo4j_uri="bolt://localhost:7687",
                neo4j_user="neo4j",
                neo4j_password="test_password",
                qdrant_host="localhost",
                qdrant_port=6333
            )

    @patch('fractal_agent.memory.long_term.GraphDatabase')
    @patch('fractal_agent.memory.long_term.QdrantClient')
    def test_custom_collection_name_and_dimension(self, mock_qdrant, mock_neo4j):
        """Test initialization with custom collection name and embedding dimension"""
        mock_neo4j.driver.return_value.session.return_value.__enter__.return_value.run.return_value = None
        mock_qdrant.return_value.get_collections.return_value = []

        graphrag = GraphRAG(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="test_password",
            collection_name="custom_collection",
            embedding_dim=768
        )

        assert graphrag.collection_name == "custom_collection"
        assert graphrag.embedding_dim == 768


class TestGraphRAGKnowledgeStorage:
    """Test knowledge storage with temporal validity"""

    @pytest.fixture
    def mock_graphrag(self):
        """Create a mock GraphRAG instance"""
        with patch('fractal_agent.memory.long_term.GraphDatabase'), \
             patch('fractal_agent.memory.long_term.QdrantClient'):
            graphrag = GraphRAG(
                neo4j_uri="bolt://localhost:7687",
                neo4j_user="neo4j",
                neo4j_password="test_password"
            )
            graphrag.graph = Mock()
            graphrag.vector_db = Mock()
            return graphrag

    def test_store_knowledge_basic(self, mock_graphrag):
        """Test basic knowledge storage"""
        # Mock session with proper context manager
        mock_session = MagicMock()
        mock_context = MagicMock()
        mock_context.__enter__ = Mock(return_value=mock_session)
        mock_context.__exit__ = Mock(return_value=False)
        mock_graphrag.graph.session = Mock(return_value=mock_context)

        # Store knowledge
        embedding = [0.1] * 1536
        mock_graphrag.store_knowledge(
            entity="ResearchAgent",
            relationship="produces",
            target="research_report",
            embedding=embedding,
            metadata={"quality": "high"}
        )

        # Verify Neo4j was called
        mock_session.run.assert_called_once()
        args = mock_session.run.call_args
        assert "ResearchAgent" in str(args)
        assert "produces" in str(args)
        assert "research_report" in str(args)

        # Verify Qdrant was called
        mock_graphrag.vector_db.upsert.assert_called_once()

    def test_store_knowledge_wrong_dimension(self, mock_graphrag):
        """Test storage fails with wrong embedding dimension"""
        embedding = [0.1] * 512  # Wrong dimension

        with pytest.raises(ValueError, match="Embedding dimension mismatch"):
            mock_graphrag.store_knowledge(
                entity="ResearchAgent",
                relationship="produces",
                target="research_report",
                embedding=embedding
            )

    def test_store_knowledge_with_temporal_validity(self, mock_graphrag):
        """Test storage with custom temporal validity"""
        mock_session = MagicMock()
        mock_context = MagicMock()
        mock_context.__enter__ = Mock(return_value=mock_session)
        mock_context.__exit__ = Mock(return_value=False)
        mock_graphrag.graph.session = Mock(return_value=mock_context)

        t_valid = datetime(2025, 1, 1)
        t_invalid = datetime(2025, 12, 31)
        embedding = [0.1] * 1536

        mock_graphrag.store_knowledge(
            entity="TestEntity",
            relationship="relates_to",
            target="TestTarget",
            embedding=embedding,
            t_valid=t_valid,
            t_invalid=t_invalid
        )

        # Verify temporal data was passed
        call_kwargs = mock_session.run.call_args[1]
        assert call_kwargs['t_valid'] == t_valid.isoformat()
        assert call_kwargs['t_invalid'] == t_invalid.isoformat()

    def test_store_knowledge_metadata_json_encoding(self, mock_graphrag):
        """Test metadata is JSON-encoded for Neo4j"""
        mock_session = MagicMock()
        mock_context = MagicMock()
        mock_context.__enter__ = Mock(return_value=mock_session)
        mock_context.__exit__ = Mock(return_value=False)
        mock_graphrag.graph.session = Mock(return_value=mock_context)

        embedding = [0.1] * 1536
        metadata = {"nested": {"key": "value"}, "count": 42}

        mock_graphrag.store_knowledge(
            entity="TestEntity",
            relationship="test",
            target="TestTarget",
            embedding=embedding,
            metadata=metadata
        )

        # Verify metadata was JSON-encoded
        call_kwargs = mock_session.run.call_args[1]
        metadata_arg = call_kwargs['metadata']
        assert isinstance(metadata_arg, str)
        assert json.loads(metadata_arg) == metadata


class TestGraphRAGRetrieval:
    """Test hybrid retrieval with vector search and graph traversal"""

    @pytest.fixture
    def mock_graphrag(self):
        """Create a mock GraphRAG instance"""
        with patch('fractal_agent.memory.long_term.GraphDatabase'), \
             patch('fractal_agent.memory.long_term.QdrantClient'):
            graphrag = GraphRAG(
                neo4j_uri="bolt://localhost:7687",
                neo4j_user="neo4j",
                neo4j_password="test_password"
            )
            graphrag.graph = Mock()
            graphrag.vector_db = Mock()
            return graphrag

    def test_retrieve_basic(self, mock_graphrag):
        """Test basic retrieval with hybrid search"""
        # Mock vector search results
        mock_result = Mock()
        mock_result.payload = {"entity": "ResearchAgent"}
        mock_graphrag.vector_db.search.return_value = [mock_result]

        # Mock graph traversal results
        mock_record = {
            "entity": "ResearchAgent",
            "relationship": "produces",
            "target": "research_report",
            "t_valid": datetime.now(),
            "t_invalid": None,
            "metadata": json.dumps({"quality": "high"})
        }
        mock_session = MagicMock()
        mock_session.run.return_value = [mock_record]
        mock_context = MagicMock()
        mock_context.__enter__ = Mock(return_value=mock_session)
        mock_context.__exit__ = Mock(return_value=False)
        mock_graphrag.graph.session = Mock(return_value=mock_context)

        # Retrieve
        query_embedding = [0.1] * 1536
        results = mock_graphrag.retrieve(
            query="What does ResearchAgent produce?",
            query_embedding=query_embedding
        )

        # Verify vector search was called
        mock_graphrag.vector_db.search.assert_called_once()

        # Verify results
        assert len(results) > 0
        assert results[0]["entity"] == "ResearchAgent"
        assert results[0]["relationship"] == "produces"
        assert isinstance(results[0]["metadata"], dict)

    def test_retrieve_wrong_dimension(self, mock_graphrag):
        """Test retrieval fails with wrong embedding dimension"""
        query_embedding = [0.1] * 512  # Wrong dimension

        with pytest.raises(ValueError, match="Query embedding dimension mismatch"):
            mock_graphrag.retrieve(
                query="test",
                query_embedding=query_embedding
            )

    def test_retrieve_only_valid_knowledge(self, mock_graphrag):
        """Test retrieval filters by temporal validity"""
        mock_result = Mock()
        mock_result.payload = {"entity": "TestEntity"}
        mock_graphrag.vector_db.search.return_value = [mock_result]

        mock_session = MagicMock()
        mock_session.run.return_value = []
        mock_context = MagicMock()
        mock_context.__enter__ = Mock(return_value=mock_session)
        mock_context.__exit__ = Mock(return_value=False)
        mock_graphrag.graph.session = Mock(return_value=mock_context)

        query_embedding = [0.1] * 1536
        mock_graphrag.retrieve(
            query="test",
            query_embedding=query_embedding,
            only_valid=True
        )

        # Verify only_valid was passed to graph query
        call_kwargs = mock_session.run.call_args[1]
        assert call_kwargs['only_valid'] is True

    def test_retrieve_metadata_decoding(self, mock_graphrag):
        """Test metadata is JSON-decoded on retrieval"""
        mock_result = Mock()
        mock_result.payload = {"entity": "TestEntity"}
        mock_graphrag.vector_db.search.return_value = [mock_result]

        metadata = {"nested": {"key": "value"}}
        mock_record = {
            "entity": "TestEntity",
            "relationship": "test",
            "target": "TestTarget",
            "t_valid": datetime.now(),
            "t_invalid": None,
            "metadata": json.dumps(metadata)
        }
        mock_session = MagicMock()
        mock_session.run.return_value = [mock_record]
        mock_context = MagicMock()
        mock_context.__enter__ = Mock(return_value=mock_session)
        mock_context.__exit__ = Mock(return_value=False)
        mock_graphrag.graph.session = Mock(return_value=mock_context)

        query_embedding = [0.1] * 1536
        results = mock_graphrag.retrieve(
            query="test",
            query_embedding=query_embedding
        )

        # Verify metadata was decoded
        assert results[0]["metadata"] == metadata


class TestGraphRAGInvalidation:
    """Test temporal knowledge invalidation"""

    @pytest.fixture
    def mock_graphrag(self):
        """Create a mock GraphRAG instance"""
        with patch('fractal_agent.memory.long_term.GraphDatabase'), \
             patch('fractal_agent.memory.long_term.QdrantClient'):
            graphrag = GraphRAG(
                neo4j_uri="bolt://localhost:7687",
                neo4j_user="neo4j",
                neo4j_password="test_password"
            )
            graphrag.graph = Mock()
            graphrag.vector_db = Mock()
            return graphrag

    def test_invalidate_knowledge_basic(self, mock_graphrag):
        """Test basic knowledge invalidation"""
        mock_result = Mock()
        mock_result.single.return_value = {"updated_count": 1}
        mock_session = MagicMock()
        mock_session.run.return_value = mock_result
        mock_context = MagicMock()
        mock_context.__enter__ = Mock(return_value=mock_session)
        mock_context.__exit__ = Mock(return_value=False)
        mock_graphrag.graph.session = Mock(return_value=mock_context)

        mock_graphrag.invalidate_knowledge(
            entity="ResearchAgent",
            relationship="produces",
            target="research_report"
        )

        # Verify Neo4j update was called
        mock_session.run.assert_called_once()
        args = mock_session.run.call_args
        assert "ResearchAgent" in str(args)
        assert "produces" in str(args)

    def test_invalidate_knowledge_with_custom_timestamp(self, mock_graphrag):
        """Test invalidation with custom timestamp"""
        mock_result = Mock()
        mock_result.single.return_value = {"updated_count": 1}
        mock_session = MagicMock()
        mock_session.run.return_value = mock_result
        mock_context = MagicMock()
        mock_context.__enter__ = Mock(return_value=mock_session)
        mock_context.__exit__ = Mock(return_value=False)
        mock_graphrag.graph.session = Mock(return_value=mock_context)

        t_invalid = datetime(2025, 12, 31)
        mock_graphrag.invalidate_knowledge(
            entity="TestEntity",
            relationship="test",
            target="TestTarget",
            t_invalid=t_invalid
        )

        # Verify timestamp was passed
        call_kwargs = mock_session.run.call_args[1]
        assert call_kwargs['t_invalid'] == t_invalid.isoformat()


class TestGraphRAGConnectionManagement:
    """Test connection cleanup and resource management"""

    @patch('fractal_agent.memory.long_term.GraphDatabase')
    @patch('fractal_agent.memory.long_term.QdrantClient')
    def test_close_connections(self, mock_qdrant, mock_neo4j):
        """Test graceful connection closure"""
        mock_neo4j.driver.return_value.session.return_value.__enter__.return_value.run.return_value = None
        mock_qdrant.return_value.get_collections.return_value = []

        graphrag = GraphRAG(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="test_password"
        )

        graphrag.close()

        # Verify Neo4j connection was closed
        graphrag.graph.close.assert_called_once()
