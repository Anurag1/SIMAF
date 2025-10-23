"""
Graph Partitioning - Phase 4

Provides query-based, temporal, and domain-based graph partitioning for efficient
subgraph extraction from GraphRAG knowledge base.

Features:
- Query-based partitioning: Extract relevant subgraphs via semantic search
- Temporal partitioning: Partition by time periods (recent, historical, etc.)
- Domain-based partitioning: Partition by entity types, relationships, or metadata
- Integration with Neo4j and Qdrant for hybrid graph+vector operations
- Memory-efficient subgraph extraction for context management

Author: BMad
Date: 2025-10-19
"""

import logging
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class PartitionStrategy(Enum):
    """
    Graph partitioning strategies.
    """
    QUERY_BASED = "query_based"
    TEMPORAL = "temporal"
    DOMAIN_BASED = "domain_based"
    HYBRID = "hybrid"


class TemporalPeriod(Enum):
    """
    Temporal periods for time-based partitioning.
    """
    LAST_HOUR = "last_hour"
    LAST_DAY = "last_day"
    LAST_WEEK = "last_week"
    LAST_MONTH = "last_month"
    LAST_YEAR = "last_year"
    CUSTOM = "custom"


@dataclass
class GraphPartition:
    """
    A partition of the knowledge graph.

    Attributes:
        partition_id: Unique identifier for this partition
        entities: Set of entity names in partition
        relationships: List of relationship dicts (entity, type, target)
        metadata: Additional metadata (strategy, filters, etc.)
        size: Number of relationships in partition
        created_at: When partition was created
    """
    partition_id: str
    entities: Set[str] = field(default_factory=set)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    size: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def add_relationship(
        self,
        entity: str,
        rel_type: str,
        target: str,
        rel_data: Dict[str, Any]
    ):
        """
        Add a relationship to the partition.

        Args:
            entity: Source entity
            rel_type: Relationship type
            target: Target entity
            rel_data: Relationship data (t_valid, t_invalid, metadata)
        """
        self.entities.add(entity)
        self.entities.add(target)
        self.relationships.append({
            "entity": entity,
            "relationship": rel_type,
            "target": target,
            **rel_data
        })
        self.size += 1

    def get_entity_count(self) -> int:
        """Get number of unique entities in partition."""
        return len(self.entities)

    def get_relationship_count(self) -> int:
        """Get number of relationships in partition."""
        return self.size

    def to_dict(self) -> Dict[str, Any]:
        """Convert partition to dictionary representation."""
        return {
            "partition_id": self.partition_id,
            "entities": list(self.entities),
            "relationships": self.relationships,
            "metadata": self.metadata,
            "size": self.size,
            "entity_count": self.get_entity_count(),
            "created_at": self.created_at
        }


class GraphPartitioner:
    """
    Graph partitioning engine for knowledge graph subgraph extraction.

    Supports three partitioning strategies:
    1. Query-based: Extract subgraphs relevant to semantic queries
    2. Temporal: Extract subgraphs by time periods
    3. Domain-based: Extract subgraphs by entity/relationship types

    Usage:
        >>> from fractal_agent.memory.long_term import GraphRAG
        >>> graphrag = GraphRAG()
        >>> partitioner = GraphPartitioner(graphrag=graphrag)
        >>>
        >>> # Query-based partitioning
        >>> partition = partitioner.partition_by_query(
        ...     query="agent coordination patterns",
        ...     query_embedding=embedding,
        ...     max_hops=2,
        ...     max_entities=50
        ... )
        >>>
        >>> # Temporal partitioning
        >>> recent_partition = partitioner.partition_by_time(
        ...     period=TemporalPeriod.LAST_WEEK
        ... )
        >>>
        >>> # Domain-based partitioning
        >>> domain_partition = partitioner.partition_by_domain(
        ...     relationship_types=["coordinates", "executes"]
        ... )
    """

    def __init__(self, graphrag):
        """
        Initialize GraphPartitioner.

        Args:
            graphrag: GraphRAG instance (provides Neo4j + Qdrant access)
        """
        self.graphrag = graphrag
        self.partitions: Dict[str, GraphPartition] = {}
        logger.info("Initialized GraphPartitioner")

    def partition_by_query(
        self,
        query: str,
        query_embedding: List[float],
        max_hops: int = 2,
        max_entities: int = 100,
        max_relationships: int = 200,
        only_valid: bool = True
    ) -> GraphPartition:
        """
        Partition graph by semantic query relevance.

        Strategy:
        1. Vector search to find most relevant entities
        2. Graph traversal (N-hop) from those entities
        3. Extract subgraph containing relevant knowledge

        Args:
            query: Natural language query
            query_embedding: Query embedding vector
            max_hops: Maximum graph traversal depth
            max_entities: Maximum entities to include
            max_relationships: Maximum relationships to include
            only_valid: Only include currently valid knowledge

        Returns:
            GraphPartition containing query-relevant subgraph

        Example:
            >>> partition = partitioner.partition_by_query(
            ...     query="agent workflows",
            ...     query_embedding=generate_embedding("agent workflows"),
            ...     max_hops=2
            ... )
            >>> print(f"Entities: {partition.get_entity_count()}")
        """
        logger.info(f"Partitioning by query: '{query}' (max_hops={max_hops})")

        partition_id = f"query_{abs(hash(query))}"
        partition = GraphPartition(
            partition_id=partition_id,
            metadata={
                "strategy": PartitionStrategy.QUERY_BASED.value,
                "query": query,
                "max_hops": max_hops,
                "only_valid": only_valid
            }
        )

        # Step 1: Vector search for seed entities
        try:
            search_results = self.graphrag.vector_db.search(
                collection_name=self.graphrag.collection_name,
                query_vector=query_embedding,
                limit=max_entities // 2
            )
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return partition

        if not search_results:
            logger.warning("No vector search results found")
            return partition

        # Extract seed entities
        seed_entities = set()
        for result in search_results:
            if hasattr(result, 'payload'):
                entity = result.payload.get('entity')
                target = result.payload.get('target')
                if entity:
                    seed_entities.add(entity)
                if target:
                    seed_entities.add(target)

        logger.debug(f"Found {len(seed_entities)} seed entities from vector search")

        # Step 2: Graph traversal (N-hop expansion)
        try:
            with self.graphrag.graph.session() as session:
                query_cypher = f"""
                    MATCH path = (start:Entity)-[r:RELATES*1..{max_hops}]-(end:Entity)
                    WHERE start.name IN $seed_entities
                    AND ($only_valid = false OR all(rel in relationships(path) WHERE rel.t_invalid IS NULL))
                    WITH relationships(path) as rels
                    UNWIND rels as rel
                    MATCH (e:Entity)-[rel]->(t:Entity)
                    RETURN DISTINCT
                        e.name as entity,
                        rel.type as relationship,
                        t.name as target,
                        rel.t_valid as t_valid,
                        rel.t_invalid as t_invalid,
                        rel.metadata as metadata
                    LIMIT $max_relationships
                """

                result = session.run(
                    query_cypher,
                    seed_entities=list(seed_entities),
                    only_valid=only_valid,
                    max_relationships=max_relationships
                )

                for record in result:
                    partition.add_relationship(
                        entity=record["entity"],
                        rel_type=record["relationship"],
                        target=record["target"],
                        rel_data={
                            "t_valid": str(record["t_valid"]) if record["t_valid"] else None,
                            "t_invalid": str(record["t_invalid"]) if record["t_invalid"] else None,
                            "metadata": record["metadata"]
                        }
                    )

        except Exception as e:
            logger.error(f"Graph traversal failed: {e}")
            return partition

        logger.info(
            f"Created query-based partition: {partition.get_entity_count()} entities, "
            f"{partition.get_relationship_count()} relationships"
        )

        self.partitions[partition_id] = partition
        return partition

    def partition_by_time(
        self,
        period: TemporalPeriod = TemporalPeriod.LAST_WEEK,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        max_relationships: int = 1000,
        granularity: str = "day"
    ) -> GraphPartition:
        """
        Partition graph by temporal period.

        Extracts knowledge that became valid within the specified time period.

        Args:
            period: Predefined temporal period
            start_time: Custom start time (for CUSTOM period)
            end_time: Custom end time (for CUSTOM period, defaults to now)
            max_relationships: Maximum relationships to include
            granularity: Temporal grouping ("day", "week", "month")

        Returns:
            GraphPartition containing temporally-filtered subgraph

        Example:
            >>> partition = partitioner.partition_by_time(
            ...     period=TemporalPeriod.LAST_WEEK
            ... )
        """
        logger.info(f"Partitioning by time: {period.value}")

        now = datetime.now()
        end_time = end_time or now

        if period == TemporalPeriod.CUSTOM:
            if not start_time:
                raise ValueError("start_time required for CUSTOM temporal period")
        elif period == TemporalPeriod.LAST_HOUR:
            start_time = now - timedelta(hours=1)
        elif period == TemporalPeriod.LAST_DAY:
            start_time = now - timedelta(days=1)
        elif period == TemporalPeriod.LAST_WEEK:
            start_time = now - timedelta(weeks=1)
        elif period == TemporalPeriod.LAST_MONTH:
            start_time = now - timedelta(days=30)
        elif period == TemporalPeriod.LAST_YEAR:
            start_time = now - timedelta(days=365)

        partition_id = f"temporal_{period.value}_{start_time.strftime('%Y%m%d')}"
        partition = GraphPartition(
            partition_id=partition_id,
            metadata={
                "strategy": PartitionStrategy.TEMPORAL.value,
                "period": period.value,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "granularity": granularity
            }
        )

        try:
            with self.graphrag.graph.session() as session:
                result = session.run("""
                    MATCH (e:Entity)-[r:RELATES]->(t:Entity)
                    WHERE datetime($start_time) <= r.t_valid <= datetime($end_time)
                    RETURN
                        e.name as entity,
                        r.type as relationship,
                        t.name as target,
                        r.t_valid as t_valid,
                        r.t_invalid as t_invalid,
                        r.metadata as metadata
                    ORDER BY r.t_valid DESC
                    LIMIT $max_relationships
                """,
                    start_time=start_time.isoformat(),
                    end_time=end_time.isoformat(),
                    max_relationships=max_relationships
                )

                for record in result:
                    partition.add_relationship(
                        entity=record["entity"],
                        rel_type=record["relationship"],
                        target=record["target"],
                        rel_data={
                            "t_valid": str(record["t_valid"]) if record["t_valid"] else None,
                            "t_invalid": str(record["t_invalid"]) if record["t_invalid"] else None,
                            "metadata": record["metadata"]
                        }
                    )

        except Exception as e:
            logger.error(f"Temporal partitioning failed: {e}")
            return partition

        logger.info(
            f"Created temporal partition: {partition.get_entity_count()} entities, "
            f"{partition.get_relationship_count()} relationships"
        )

        self.partitions[partition_id] = partition
        return partition

    def partition_by_domain(
        self,
        entity_types: Optional[List[str]] = None,
        relationship_types: Optional[List[str]] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        max_relationships: int = 1000,
        only_valid: bool = True
    ) -> GraphPartition:
        """
        Partition graph by domain (entity types, relationship types, metadata).

        Extracts subgraph containing specific types of entities and relationships.

        Args:
            entity_types: Entity types to include (reserved for future use)
            relationship_types: Relationship types to include
            metadata_filters: Metadata key-value filters
            max_relationships: Maximum relationships to include
            only_valid: Only include currently valid knowledge

        Returns:
            GraphPartition containing domain-filtered subgraph

        Example:
            >>> partition = partitioner.partition_by_domain(
            ...     relationship_types=["coordinates", "executes"],
            ...     only_valid=True
            ... )
        """
        logger.info(
            f"Partitioning by domain: entities={entity_types}, "
            f"relationships={relationship_types}"
        )

        partition_id = f"domain_{abs(hash(str(entity_types) + str(relationship_types)))}"
        partition = GraphPartition(
            partition_id=partition_id,
            metadata={
                "strategy": PartitionStrategy.DOMAIN_BASED.value,
                "entity_types": entity_types,
                "relationship_types": relationship_types,
                "metadata_filters": metadata_filters,
                "only_valid": only_valid
            }
        )

        where_clauses = []

        if relationship_types:
            where_clauses.append("r.type IN $relationship_types")

        if only_valid:
            where_clauses.append("r.t_invalid IS NULL")

        where_clause = " AND ".join(where_clauses) if where_clauses else "true"

        try:
            with self.graphrag.graph.session() as session:
                result = session.run(f"""
                    MATCH (e:Entity)-[r:RELATES]->(t:Entity)
                    WHERE {where_clause}
                    RETURN
                        e.name as entity,
                        r.type as relationship,
                        t.name as target,
                        r.t_valid as t_valid,
                        r.t_invalid as t_invalid,
                        r.metadata as metadata
                    LIMIT $max_relationships
                """,
                    relationship_types=relationship_types or [],
                    max_relationships=max_relationships
                )

                for record in result:
                    if metadata_filters:
                        import json
                        try:
                            rel_metadata = json.loads(record["metadata"]) if record["metadata"] else {}
                        except (json.JSONDecodeError, TypeError):
                            rel_metadata = {}

                        if not all(
                            rel_metadata.get(key) == value
                            for key, value in metadata_filters.items()
                        ):
                            continue

                    partition.add_relationship(
                        entity=record["entity"],
                        rel_type=record["relationship"],
                        target=record["target"],
                        rel_data={
                            "t_valid": str(record["t_valid"]) if record["t_valid"] else None,
                            "t_invalid": str(record["t_invalid"]) if record["t_invalid"] else None,
                            "metadata": record["metadata"]
                        }
                    )

        except Exception as e:
            logger.error(f"Domain partitioning failed: {e}")
            return partition

        logger.info(
            f"Created domain partition: {partition.get_entity_count()} entities, "
            f"{partition.get_relationship_count()} relationships"
        )

        self.partitions[partition_id] = partition
        return partition

    def partition_hybrid(
        self,
        query: Optional[str] = None,
        query_embedding: Optional[List[float]] = None,
        temporal_period: Optional[TemporalPeriod] = None,
        relationship_types: Optional[List[str]] = None,
        max_hops: int = 2,
        max_relationships: int = 500
    ) -> GraphPartition:
        """
        Hybrid partitioning: Combine query-based, temporal, and domain filters.

        Extracts subgraph that satisfies multiple criteria simultaneously.

        Args:
            query: Natural language query (optional)
            query_embedding: Query embedding vector (required if query specified)
            temporal_period: Temporal period filter (optional)
            relationship_types: Relationship types to include (optional)
            max_hops: Maximum graph traversal depth for query-based
            max_relationships: Maximum relationships to include

        Returns:
            GraphPartition containing hybrid-filtered subgraph

        Example:
            >>> partition = partitioner.partition_hybrid(
            ...     query="agent workflows",
            ...     query_embedding=embedding,
            ...     temporal_period=TemporalPeriod.LAST_WEEK,
            ...     relationship_types=["executes", "coordinates"]
            ... )
        """
        logger.info("Partitioning with hybrid strategy")

        partition_id = f"hybrid_{abs(hash(str(query) + str(temporal_period) + str(relationship_types)))}"
        partition = GraphPartition(
            partition_id=partition_id,
            metadata={
                "strategy": PartitionStrategy.HYBRID.value,
                "query": query,
                "temporal_period": temporal_period.value if temporal_period else None,
                "relationship_types": relationship_types
            }
        )

        seed_entities = None
        if query and query_embedding:
            try:
                search_results = self.graphrag.vector_db.search(
                    collection_name=self.graphrag.collection_name,
                    query_vector=query_embedding,
                    limit=50
                )
                seed_entities = list({r.payload["entity"] for r in search_results if hasattr(r, 'payload')})
                logger.debug(f"Found {len(seed_entities)} seed entities from query")
            except Exception as e:
                logger.error(f"Vector search failed: {e}")

        time_filter = ""
        time_params = {}
        if temporal_period:
            now = datetime.now()
            if temporal_period == TemporalPeriod.LAST_HOUR:
                start_time = now - timedelta(hours=1)
            elif temporal_period == TemporalPeriod.LAST_DAY:
                start_time = now - timedelta(days=1)
            elif temporal_period == TemporalPeriod.LAST_WEEK:
                start_time = now - timedelta(weeks=1)
            elif temporal_period == TemporalPeriod.LAST_MONTH:
                start_time = now - timedelta(days=30)
            elif temporal_period == TemporalPeriod.LAST_YEAR:
                start_time = now - timedelta(days=365)
            else:
                start_time = None

            if start_time:
                time_filter = "AND datetime($start_time) <= r.t_valid"
                time_params = {"start_time": start_time.isoformat()}

        where_clauses = []

        if seed_entities:
            where_clauses.append("e.name IN $seed_entities")

        if relationship_types:
            where_clauses.append("r.type IN $relationship_types")

        if time_filter:
            where_clauses.append(time_filter.replace("AND ", ""))

        where_clause = " AND ".join(where_clauses) if where_clauses else "true"

        try:
            with self.graphrag.graph.session() as session:
                params = {
                    "seed_entities": seed_entities or [],
                    "relationship_types": relationship_types or [],
                    "max_relationships": max_relationships,
                    **time_params
                }

                result = session.run(f"""
                    MATCH (e:Entity)-[r:RELATES]->(t:Entity)
                    WHERE {where_clause}
                    RETURN
                        e.name as entity,
                        r.type as relationship,
                        t.name as target,
                        r.t_valid as t_valid,
                        r.t_invalid as t_invalid,
                        r.metadata as metadata
                    LIMIT $max_relationships
                """, **params)

                for record in result:
                    partition.add_relationship(
                        entity=record["entity"],
                        rel_type=record["relationship"],
                        target=record["target"],
                        rel_data={
                            "t_valid": str(record["t_valid"]) if record["t_valid"] else None,
                            "t_invalid": str(record["t_invalid"]) if record["t_invalid"] else None,
                            "metadata": record["metadata"]
                        }
                    )

        except Exception as e:
            logger.error(f"Hybrid partitioning failed: {e}")
            return partition

        logger.info(
            f"Created hybrid partition: {partition.get_entity_count()} entities, "
            f"{partition.get_relationship_count()} relationships"
        )

        self.partitions[partition_id] = partition
        return partition

    def get_partition(self, partition_id: str) -> Optional[GraphPartition]:
        """
        Get partition by ID.

        Args:
            partition_id: Partition identifier

        Returns:
            GraphPartition if found, None otherwise
        """
        return self.partitions.get(partition_id)

    def get_all_partitions(self) -> List[GraphPartition]:
        """
        Get all partitions.

        Returns:
            List of all GraphPartition objects
        """
        return list(self.partitions.values())

    def clear_partitions(self):
        """Clear all cached partitions."""
        self.partitions.clear()
        logger.info("Cleared all partitions")

    def export_partition_to_context(self, partition: GraphPartition) -> str:
        """
        Export partition to formatted context string for LLM consumption.

        Args:
            partition: GraphPartition to export

        Returns:
            Formatted context string

        Example:
            >>> context = partitioner.export_partition_to_context(partition)
            >>> print(context)
        """
        lines = [
            f"# Knowledge Graph Partition: {partition.partition_id}",
            "",
            f"Strategy: {partition.metadata.get('strategy', 'unknown')}",
            f"Entities: {partition.get_entity_count()}",
            f"Relationships: {partition.get_relationship_count()}",
            f"Created: {partition.created_at}",
            "",
            "## Relationships",
            ""
        ]

        for rel in partition.relationships:
            lines.append(
                f"- ({rel['entity']})-[{rel['relationship']}]->({rel['target']})"
            )
            if rel.get('metadata'):
                lines.append(f"  Metadata: {rel['metadata']}")

        return "\n".join(lines)

    def merge_partitions(self, partition_ids: List[str]) -> GraphPartition:
        """
        Merge multiple partitions into one.

        Args:
            partition_ids: List of partition IDs to merge

        Returns:
            New GraphPartition containing merged data

        Example:
            >>> merged = partitioner.merge_partitions(["query_123", "temporal_456"])
        """
        logger.info(f"Merging {len(partition_ids)} partitions")

        merged_id = f"merged_{abs(hash('_'.join(partition_ids)))}"
        merged = GraphPartition(
            partition_id=merged_id,
            metadata={
                "strategy": "merged",
                "source_partitions": partition_ids
            }
        )

        for pid in partition_ids:
            partition = self.partitions.get(pid)
            if not partition:
                logger.warning(f"Partition {pid} not found, skipping")
                continue

            for rel in partition.relationships:
                merged.add_relationship(
                    entity=rel["entity"],
                    rel_type=rel["relationship"],
                    target=rel["target"],
                    rel_data={
                        "t_valid": rel.get("t_valid"),
                        "t_invalid": rel.get("t_invalid"),
                        "metadata": rel.get("metadata")
                    }
                )

        logger.info(
            f"Merged partition: {merged.get_entity_count()} entities, "
            f"{merged.get_relationship_count()} relationships"
        )

        self.partitions[merged_id] = merged
        return merged

    def get_partition_stats(self, partition: GraphPartition) -> Dict[str, Any]:
        """
        Compute statistics for a graph partition.

        Args:
            partition: GraphPartition to analyze

        Returns:
            Dict with statistics (node_count, edge_count, avg_degree, density)

        Example:
            >>> stats = partitioner.get_partition_stats(partition)
            >>> print(f"Density: {stats['density']:.3f}")
        """
        nodes = list(partition.entities)
        edges = partition.relationships

        node_count = len(nodes)
        edge_count = len(edges)

        degree_map = defaultdict(int)
        for edge in edges:
            entity = edge.get("entity")
            target = edge.get("target")
            if entity:
                degree_map[entity] += 1
            if target:
                degree_map[target] += 1

        avg_degree = sum(degree_map.values()) / node_count if node_count > 0 else 0

        max_edges = node_count * (node_count - 1) if node_count > 1 else 1
        density = edge_count / max_edges if max_edges > 0 else 0

        stats = {
            "node_count": node_count,
            "edge_count": edge_count,
            "avg_degree": avg_degree,
            "density": density,
            "max_degree": max(degree_map.values()) if degree_map else 0,
            "min_degree": min(degree_map.values()) if degree_map else 0,
            "metadata": partition.metadata
        }

        logger.debug(f"Partition stats: {stats}")
        return stats


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("=" * 80)
    print("GraphPartitioner Demo - Phase 4")
    print("=" * 80)
    print()
    print("NOTE: This demo requires Neo4j and Qdrant with sample data.")
    print("Run the long_term.py demo first to populate sample data.")
    print()

    try:
        from fractal_agent.memory.long_term import GraphRAG
        import random

        print("[1/5] Connecting to GraphRAG...")
        graphrag = GraphRAG(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="fractal_password",
            qdrant_host="localhost",
            qdrant_port=6333,
            collection_name="fractal_knowledge_demo"
        )
        print("✅ Connected to GraphRAG")
        print()

        print("[2/5] Initializing GraphPartitioner...")
        partitioner = GraphPartitioner(graphrag=graphrag)
        print("✅ Initialized GraphPartitioner")
        print()

        print("[3/5] Testing temporal partitioning...")
        temporal_partition = partitioner.partition_by_time(
            period=TemporalPeriod.LAST_DAY,
            max_relationships=100
        )
        print(
            f"✅ Temporal partition: {temporal_partition.get_entity_count()} entities, "
            f"{temporal_partition.get_relationship_count()} relationships"
        )
        print()

        print("[4/5] Testing domain-based partitioning...")
        domain_partition = partitioner.partition_by_domain(
            relationship_types=["produces", "analyzes", "coordinates"],
            max_relationships=100,
            only_valid=True
        )
        print(
            f"✅ Domain partition: {domain_partition.get_entity_count()} entities, "
            f"{domain_partition.get_relationship_count()} relationships"
        )
        print()

        print("[5/5] Exporting partition to context...")
        if domain_partition.get_relationship_count() > 0:
            context = partitioner.export_partition_to_context(domain_partition)
            print(f"✅ Exported context: {len(context)} characters")
            print()
            print("Sample context:")
            print(context[:500] + "..." if len(context) > 500 else context)
        else:
            print("⚠️  No relationships in partition (may need sample data)")
        print()

        graphrag.close()

        print("=" * 80)
        print("GraphPartitioner Demo Complete!")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure to:")
        print("1. Start services: docker-compose up -d")
        print("2. Run long_term.py demo first to populate sample data")
