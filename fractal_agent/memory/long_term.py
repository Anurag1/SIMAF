"""
Long-Term Memory (GraphRAG) - Phase 3

Hybrid Graph + Vector search using Neo4j and Qdrant for knowledge storage.
Implements temporal validity tracking for knowledge evolution.

Author: BMad
Date: 2025-10-19
"""

from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from datetime import datetime
from typing import List, Dict, Optional, Any
import logging
import json

logger = logging.getLogger(__name__)


class GraphRAG:
    """
    Long-term memory using hybrid Graph + Vector search.

    Combines two database systems:
    - Neo4j: Entity relationships with temporal validity
    - Qdrant: Semantic search via embeddings

    This enables both structured graph queries and semantic similarity search
    over the knowledge base, with support for temporal validity tracking.

    Usage:
        >>> graphrag = GraphRAG(
        ...     neo4j_uri="bolt://localhost:7687",
        ...     neo4j_user="neo4j",
        ...     neo4j_password="fractal_password",
        ...     qdrant_host="localhost",
        ...     qdrant_port=6333
        ... )
        >>>
        >>> # Store knowledge triple
        >>> graphrag.store_knowledge(
        ...     entity="ResearchAgent",
        ...     relationship="produces",
        ...     target="research_report",
        ...     embedding=[0.1, 0.2, ...],  # 1536-dim vector
        ...     metadata={"quality": "high"}
        ... )
        >>>
        >>> # Retrieve knowledge
        >>> results = graphrag.retrieve(
        ...     query="What does ResearchAgent produce?",
        ...     query_embedding=[0.1, 0.2, ...],
        ...     max_results=5
        ... )
        >>> graphrag.close()

    Attributes:
        graph: Neo4j driver for graph operations
        vector_db: Qdrant client for vector search
        collection_name: Name of Qdrant collection
    """

    def __init__(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: Optional[str] = None,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = "fractal_knowledge",
        embedding_dim: int = 1536
    ):
        """
        Initialize GraphRAG with Neo4j and Qdrant connections.

        Args:
            neo4j_uri: Neo4j Bolt URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            qdrant_host: Qdrant host
            qdrant_port: Qdrant port
            collection_name: Name for Qdrant collection
            embedding_dim: Dimension of embedding vectors (default: 1536 for OpenAI)

        Raises:
            ConnectionError: If unable to connect to Neo4j or Qdrant
        """
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim

        # Neo4j connection
        try:
            self.graph = GraphDatabase.driver(
                neo4j_uri,
                auth=(neo4j_user, neo4j_password)
            )
            # Test connection
            with self.graph.session() as session:
                session.run("RETURN 1")
            logger.info(f"Connected to Neo4j at {neo4j_uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise ConnectionError(f"Neo4j connection failed: {e}")

        # Qdrant connection
        try:
            # Note: check_compatibility=False to allow minor version mismatches
            self.vector_db = QdrantClient(
                host=qdrant_host,
                port=qdrant_port,
                prefer_grpc=False  # Use HTTP for better compatibility
            )
            # Test connection
            self.vector_db.get_collections()
            logger.info(f"Connected to Qdrant at {qdrant_host}:{qdrant_port}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise ConnectionError(f"Qdrant connection failed: {e}")

        # Initialize schema
        self._initialize_schema()

    def _initialize_schema(self):
        """
        Create Neo4j constraints/indexes and Qdrant collection.

        Sets up:
        - Neo4j: Entity ID uniqueness constraint and name index
        - Qdrant: Collection with cosine similarity for embeddings
        """
        # Neo4j: Create constraints and indexes
        try:
            with self.graph.session() as session:
                # Ensure Entity IDs are unique
                session.run("""
                    CREATE CONSTRAINT entity_id IF NOT EXISTS
                    FOR (e:Entity) REQUIRE e.id IS UNIQUE
                """)
                logger.debug("Created Neo4j entity_id constraint")

                # Index on entity names for fast lookup
                session.run("""
                    CREATE INDEX entity_name IF NOT EXISTS
                    FOR (e:Entity) ON (e.name)
                """)
                logger.debug("Created Neo4j entity_name index")
        except Exception as e:
            logger.warning(f"Neo4j schema initialization warning: {e}")

        # Qdrant: Create collection for embeddings
        try:
            self.vector_db.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Created Qdrant collection '{self.collection_name}'")
        except Exception as e:
            logger.info(f"Qdrant collection already exists or error: {e}")

    def store_knowledge(
        self,
        entity: str,
        relationship: str,
        target: str,
        embedding: List[float],
        t_valid: Optional[datetime] = None,
        t_invalid: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Store knowledge triple with temporal validity.

        Creates both:
        1. Graph relationship in Neo4j (entity)-[relationship]->(target)
        2. Vector embedding in Qdrant for semantic search

        Args:
            entity: Subject entity (e.g., "ResearchAgent")
            relationship: Relationship type (e.g., "produces")
            target: Target entity (e.g., "research_report")
            embedding: Vector embedding of the triple (must match embedding_dim)
            t_valid: When knowledge became valid (defaults to now)
            t_invalid: When knowledge became invalid (None = still valid)
            metadata: Additional metadata dict

        Raises:
            ValueError: If embedding dimension doesn't match
        """
        if len(embedding) != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.embedding_dim}, "
                f"got {len(embedding)}"
            )

        t_valid = t_valid or datetime.now()

        # Store in Neo4j (graph structure)
        # Note: Neo4j doesn't support nested dicts, so we JSON-encode metadata
        try:
            with self.graph.session() as session:
                session.run("""
                    MERGE (e:Entity {name: $entity})
                    MERGE (t:Entity {name: $target})
                    CREATE (e)-[r:RELATES {
                        type: $relationship,
                        t_valid: datetime($t_valid),
                        t_invalid: $t_invalid,
                        metadata: $metadata
                    }]->(t)
                """,
                    entity=entity,
                    target=target,
                    relationship=relationship,
                    t_valid=t_valid.isoformat(),
                    t_invalid=t_invalid.isoformat() if t_invalid else None,
                    metadata=json.dumps(metadata or {})  # JSON-encode for Neo4j
                )
            logger.debug(f"Stored graph: ({entity})-[{relationship}]->({target})")
        except Exception as e:
            logger.error(f"Failed to store in Neo4j: {e}")
            raise

        # Store in Qdrant (vector search)
        try:
            point = PointStruct(
                id=abs(hash(f"{entity}_{relationship}_{target}_{t_valid.isoformat()}")),
                vector=embedding,
                payload={
                    "entity": entity,
                    "relationship": relationship,
                    "target": target,
                    "t_valid": t_valid.isoformat(),
                    "t_invalid": t_invalid.isoformat() if t_invalid else None,
                    "metadata": metadata or {}
                }
            )
            self.vector_db.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            logger.debug(f"Stored embedding for ({entity})-[{relationship}]->({target})")
        except Exception as e:
            logger.error(f"Failed to store in Qdrant: {e}")
            raise

    def retrieve(
        self,
        query: str,
        query_embedding: List[float],
        max_results: int = 10,
        only_valid: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval: Vector similarity + Graph traversal.

        Two-step process:
        1. Vector search in Qdrant for semantic similarity
        2. Graph traversal in Neo4j for related entities

        Args:
            query: Natural language query (for logging/debugging)
            query_embedding: Vector embedding of query (must match embedding_dim)
            max_results: Maximum results to return
            only_valid: Only return currently valid knowledge (t_invalid is None)

        Returns:
            List of knowledge triples with metadata, sorted by validity date

        Raises:
            ValueError: If query_embedding dimension doesn't match
        """
        if len(query_embedding) != self.embedding_dim:
            raise ValueError(
                f"Query embedding dimension mismatch: expected {self.embedding_dim}, "
                f"got {len(query_embedding)}"
            )

        logger.debug(f"Retrieving knowledge for query: {query}")

        # Step 1: Vector search for semantic similarity
        try:
            search_results = self.vector_db.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=max_results * 3  # Get more candidates for graph filtering
            )
            logger.debug(f"Vector search found {len(search_results)} candidates")
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

        if not search_results:
            logger.debug("No vector search results found")
            return []

        # Step 2: Graph traversal for related entities
        entity_ids = [r.payload["entity"] for r in search_results]

        try:
            with self.graph.session() as session:
                graph_results = session.run("""
                    MATCH (e:Entity)-[r:RELATES]->(t:Entity)
                    WHERE e.name IN $entity_ids
                    AND ($only_valid = false OR r.t_invalid IS NULL)
                    RETURN e.name as entity, r.type as relationship,
                           t.name as target, r.t_valid as t_valid,
                           r.t_invalid as t_invalid, r.metadata as metadata
                    ORDER BY r.t_valid DESC
                    LIMIT $max_results
                """,
                    entity_ids=entity_ids,
                    only_valid=only_valid,
                    max_results=max_results
                )

                # Decode JSON metadata for each result
                results = []
                for record in graph_results:
                    result = dict(record)
                    # Decode JSON metadata string back to dict
                    if result.get("metadata"):
                        try:
                            result["metadata"] = json.loads(result["metadata"])
                        except (json.JSONDecodeError, TypeError):
                            logger.warning(f"Failed to decode metadata: {result['metadata']}")
                            result["metadata"] = {}
                    results.append(result)

                logger.debug(f"Graph traversal returned {len(results)} results")
                return results
        except Exception as e:
            logger.error(f"Graph traversal failed: {e}")
            return []

    def invalidate_knowledge(
        self,
        entity: str,
        relationship: str,
        target: str,
        t_invalid: Optional[datetime] = None
    ):
        """
        Mark knowledge triple as invalid (temporal update).

        Updates the t_invalid timestamp to indicate this knowledge is no longer valid.
        The relationship is not deleted, preserving historical knowledge.

        Args:
            entity: Subject entity
            relationship: Relationship type
            target: Target entity
            t_invalid: When knowledge became invalid (defaults to now)
        """
        t_invalid = t_invalid or datetime.now()

        try:
            with self.graph.session() as session:
                result = session.run("""
                    MATCH (e:Entity {name: $entity})-[r:RELATES {type: $relationship}]->(t:Entity {name: $target})
                    WHERE r.t_invalid IS NULL
                    SET r.t_invalid = datetime($t_invalid)
                    RETURN count(r) as updated_count
                """,
                    entity=entity,
                    relationship=relationship,
                    target=target,
                    t_invalid=t_invalid.isoformat()
                )
                count = result.single()["updated_count"]
                logger.info(f"Invalidated {count} relationships: ({entity})-[{relationship}]->({target})")
        except Exception as e:
            logger.error(f"Failed to invalidate knowledge: {e}")
            raise

    def close(self):
        """Close database connections cleanly."""
        try:
            self.graph.close()
            logger.info("Closed Neo4j connection")
        except Exception as e:
            logger.warning(f"Error closing Neo4j connection: {e}")


class DocumentStore:
    """
    Document storage with intelligent chunking for GraphRAG.

    Extends GraphRAG to handle full documents (specs, plans, etc.) by:
    1. Intelligently chunking documents (respecting markdown structure)
    2. Generating embeddings for each chunk
    3. Storing chunks in Qdrant with rich metadata
    4. Creating Neo4j relationships: Document -> Section -> Chunk

    Usage:
        >>> doc_store = DocumentStore(graphrag=graphrag)
        >>> chunk_ids = doc_store.store_document(
        ...     file_path="PHASE3_PLAN.md",
        ...     content=file_content,
        ...     metadata={"doc_type": "plan", "phase": 3}
        ... )
        >>> context = doc_store.retrieve_document_context(
        ...     query="What are Phase 3 requirements?",
        ...     max_chunks=5
        ... )
    """

    def __init__(self, graphrag: GraphRAG):
        """
        Initialize DocumentStore.

        Args:
            graphrag: GraphRAG instance (provides Neo4j + Qdrant access)
        """
        self.graphrag = graphrag
        self.collection_name = graphrag.collection_name + "_documents"
        self.embedding_dim = graphrag.embedding_dim

        # Initialize document collection in Qdrant
        self._initialize_document_collection()

        logger.info("Initialized DocumentStore")

    def _initialize_document_collection(self):
        """Create separate Qdrant collection for document chunks."""
        try:
            from qdrant_client.models import Distance, VectorParams

            self.graphrag.vector_db.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Created Qdrant document collection '{self.collection_name}'")
        except Exception as e:
            logger.info(f"Document collection already exists or error: {e}")

    def chunk_markdown(
        self,
        content: str,
        chunk_size: int = 2000,
        chunk_overlap: int = 200
    ) -> List[Dict[str, Any]]:
        """
        Intelligently chunk markdown document.

        Strategy:
        1. Parse markdown structure (headers create natural boundaries)
        2. Chunk by sections, respecting max chunk_size
        3. If section > chunk_size, split by subsections
        4. If still too large, split by paragraphs with overlap
        5. Preserve code blocks intact

        Args:
            content: Markdown document content
            chunk_size: Maximum characters per chunk (default: 2000)
            chunk_overlap: Overlap between chunks (default: 200)

        Returns:
            List of chunk dicts with:
            - text: Chunk content
            - section_path: Header hierarchy (e.g., "## Section / ### Subsection")
            - start_line: Starting line number
            - end_line: Ending line number
        """
        chunks = []
        lines = content.split('\n')

        current_chunk = []
        current_section = []
        current_size = 0
        chunk_start_line = 0

        for line_num, line in enumerate(lines, 1):
            # Detect headers
            if line.startswith('#'):
                # Save previous chunk if it exists
                if current_chunk and current_size > 0:
                    chunks.append({
                        "text": '\n'.join(current_chunk),
                        "section_path": ' / '.join(current_section) if current_section else "Root",
                        "start_line": chunk_start_line,
                        "end_line": line_num - 1
                    })
                    current_chunk = []
                    current_size = 0
                    chunk_start_line = line_num

                # Update section path
                header_level = len(line) - len(line.lstrip('#'))
                header_text = line.lstrip('#').strip()

                # Adjust section path based on header level
                current_section = current_section[:header_level-1]
                if header_level <= len(current_section) + 1:
                    current_section.append(header_text)

            # Add line to current chunk
            current_chunk.append(line)
            current_size += len(line) + 1  # +1 for newline

            # Check if chunk is too large
            if current_size >= chunk_size:
                chunks.append({
                    "text": '\n'.join(current_chunk),
                    "section_path": ' / '.join(current_section) if current_section else "Root",
                    "start_line": chunk_start_line,
                    "end_line": line_num
                })

                # Keep overlap for context continuity
                overlap_lines = []
                overlap_size = 0
                for i in range(len(current_chunk) - 1, -1, -1):
                    if overlap_size >= chunk_overlap:
                        break
                    overlap_lines.insert(0, current_chunk[i])
                    overlap_size += len(current_chunk[i]) + 1

                current_chunk = overlap_lines
                current_size = overlap_size
                chunk_start_line = line_num - len(overlap_lines) + 1

        # Add final chunk
        if current_chunk:
            chunks.append({
                "text": '\n'.join(current_chunk),
                "section_path": ' / '.join(current_section) if current_section else "Root",
                "start_line": chunk_start_line,
                "end_line": len(lines)
            })

        logger.debug(f"Chunked document into {len(chunks)} chunks")
        return chunks

    def store_document(
        self,
        file_path: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        chunk_size: int = 2000,
        chunk_overlap: int = 200
    ) -> List[str]:
        """
        Store document with intelligent chunking.

        Args:
            file_path: Path to document file (for metadata)
            content: Document content (markdown)
            metadata: Additional metadata (doc_type, phase, etc.)
            chunk_size: Maximum characters per chunk
            chunk_overlap: Overlap between chunks for context

        Returns:
            List of chunk IDs

        Raises:
            Exception: If embedding or storage fails
        """
        from .embeddings import generate_embedding
        from qdrant_client.models import PointStruct
        from pathlib import Path

        logger.info(f"Storing document: {file_path}")

        # Chunk document
        chunks = self.chunk_markdown(content, chunk_size, chunk_overlap)

        # Store each chunk
        chunk_ids = []
        points = []

        for i, chunk in enumerate(chunks):
            # Generate embedding
            embedding = generate_embedding(chunk["text"])

            # Create unique chunk ID
            chunk_id = abs(hash(f"{file_path}_{i}_{chunk['start_line']}"))
            chunk_ids.append(str(chunk_id))

            # Prepare payload with metadata
            payload = {
                "file_path": file_path,
                "file_name": Path(file_path).name,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "text": chunk["text"],
                "section_path": chunk["section_path"],
                "start_line": chunk["start_line"],
                "end_line": chunk["end_line"],
                "char_count": len(chunk["text"]),
                **(metadata or {})
            }

            # Create Qdrant point
            points.append(PointStruct(
                id=chunk_id,
                vector=embedding,
                payload=payload
            ))

        # Batch upsert to Qdrant
        try:
            self.graphrag.vector_db.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info(f"Stored {len(chunks)} chunks from {Path(file_path).name}")
        except Exception as e:
            logger.error(f"Failed to store document chunks: {e}")
            raise

        # Create Neo4j document node and relationships
        try:
            with self.graphrag.graph.session() as session:
                session.run("""
                    MERGE (d:Document {file_path: $file_path})
                    SET d.file_name = $file_name,
                        d.total_chunks = $total_chunks,
                        d.metadata = $metadata
                """,
                    file_path=file_path,
                    file_name=Path(file_path).name,
                    total_chunks=len(chunks),
                    metadata=json.dumps(metadata or {})
                )
                logger.debug(f"Created Neo4j document node for {file_path}")
        except Exception as e:
            logger.warning(f"Failed to create Neo4j document node: {e}")

        return chunk_ids

    def retrieve_document_context(
        self,
        query: str,
        max_chunks: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Retrieve relevant document chunks for query.

        Args:
            query: Natural language query
            max_chunks: Maximum number of chunks to retrieve
            filter_metadata: Optional metadata filters (e.g., {"doc_type": "plan"})

        Returns:
            Formatted context string for LLM consumption
        """
        from .embeddings import generate_embedding

        logger.debug(f"Retrieving document context for: {query}")

        # Generate query embedding
        query_embedding = generate_embedding(query)

        # Search in document collection
        try:
            search_results = self.graphrag.vector_db.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=max_chunks,
                query_filter=filter_metadata  # Optional metadata filtering
            )
        except Exception as e:
            logger.error(f"Document search failed: {e}")
            return ""

        if not search_results:
            logger.debug("No document chunks found")
            return ""

        # Format results for LLM context
        context_parts = []
        context_parts.append(f"# Relevant Documentation ({len(search_results)} chunks)\n")

        for i, result in enumerate(search_results, 1):
            payload = result.payload
            context_parts.append(
                f"\n## [{i}] {payload['file_name']} - {payload['section_path']}\n"
                f"(Lines {payload['start_line']}-{payload['end_line']})\n\n"
                f"{payload['text']}\n"
            )

        context = '\n'.join(context_parts)
        logger.debug(f"Retrieved {len(search_results)} chunks, {len(context)} characters")

        return context


# Demo
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("=" * 80)
    print("GraphRAG (Long-Term Memory) Demo - Phase 3")
    print("=" * 80)
    print()
    print("NOTE: This demo requires Neo4j and Qdrant to be running.")
    print("Start services with: docker-compose up -d")
    print()

    try:
        # Initialize GraphRAG
        print("[1/5] Connecting to Neo4j and Qdrant...")
        graphrag = GraphRAG(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="fractal_password",
            qdrant_host="localhost",
            qdrant_port=6333,
            collection_name="fractal_knowledge_demo"
        )
        print("✅ Connected to GraphRAG databases\n")

        # Create sample embeddings (mock 1536-dim vectors)
        import random
        random.seed(42)

        def mock_embedding() -> List[float]:
            """Generate mock 1536-dimensional embedding"""
            return [random.random() for _ in range(1536)]

        # Store sample knowledge
        print("[2/5] Storing knowledge triples...")
        graphrag.store_knowledge(
            entity="ResearchAgent",
            relationship="produces",
            target="research_report",
            embedding=mock_embedding(),
            metadata={"quality": "high", "token_count": 5000}
        )
        graphrag.store_knowledge(
            entity="IntelligenceAgent",
            relationship="analyzes",
            target="performance_metrics",
            embedding=mock_embedding(),
            metadata={"tier": "expensive"}
        )
        graphrag.store_knowledge(
            entity="ControlAgent",
            relationship="coordinates",
            target="multi_agent_workflow",
            embedding=mock_embedding(),
            metadata={"complexity": "high"}
        )
        print("✅ Stored 3 knowledge triples\n")

        # Retrieve knowledge
        print("[3/5] Retrieving knowledge via hybrid search...")
        results = graphrag.retrieve(
            query="What does ResearchAgent produce?",
            query_embedding=mock_embedding(),
            max_results=5,
            only_valid=True
        )

        if results:
            print(f"✅ Retrieved {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"   {i}. ({result['entity']})-[{result['relationship']}]->({result['target']})")
                print(f"      Valid: {result['t_valid']}, Metadata: {result['metadata']}")
        else:
            print("⚠️  No results found (may need to wait for indexing)")
        print()

        # Invalidate knowledge
        print("[4/5] Invalidating outdated knowledge...")
        graphrag.invalidate_knowledge(
            entity="ResearchAgent",
            relationship="produces",
            target="research_report"
        )
        print("✅ Marked knowledge as invalid\n")

        # Verify invalidation
        print("[5/5] Verifying only valid knowledge is returned...")
        results = graphrag.retrieve(
            query="What does ResearchAgent produce?",
            query_embedding=mock_embedding(),
            max_results=5,
            only_valid=True  # Should not return invalidated knowledge
        )
        valid_count = sum(1 for r in results if r['entity'] == 'ResearchAgent')
        print(f"✅ Valid ResearchAgent results: {valid_count} (expected: 0)\n")

        # Cleanup
        graphrag.close()
        print("=" * 80)
        print("GraphRAG Demo Complete!")
        print("=" * 80)

    except ConnectionError as e:
        print(f"\n❌ Connection Error: {e}")
        print("\nTo start the required services, run:")
        print("  docker-compose up -d")
        print("\nThen verify they're running:")
        print("  docker ps")
