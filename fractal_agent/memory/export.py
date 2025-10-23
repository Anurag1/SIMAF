"""
Memory Export System - Phase 4

Unified export functionality for all memory systems (ShortTerm, GraphRAG, DocumentStore).
Supports multiple output formats: JSON, Markdown, CSV, YAML.

Features:
- Session log exports with task trees
- Knowledge graph exports with temporal validity
- Document chunk metadata exports
- Performance metrics exports
- Filtered/queried data exports
- Batch export capabilities

Author: BMad
Date: 2025-10-19
"""

import csv
import json
import logging
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class ExportMetadata:
    """Metadata for memory exports."""
    export_timestamp: str
    export_format: str
    source_type: str
    version: str = "1.0"
    total_records: int = 0
    filters_applied: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class MemoryExporter:
    """
    Unified memory export system.

    Handles exports from:
    - ShortTermMemory: JSON session logs
    - GraphRAG: Neo4j knowledge graphs
    - DocumentStore: Qdrant document chunks

    Supports formats:
    - JSON: Structured data with full fidelity
    - Markdown: Human-readable documentation
    - CSV: Tabular data for analysis
    - YAML: Configuration-friendly format

    Usage:
        >>> exporter = MemoryExporter(output_dir="./exports")
        >>>
        >>> # Export session logs
        >>> exporter.export_session(
        ...     memory=short_term_memory,
        ...     format="json",
        ...     filename="session_20251019.json"
        ... )
        >>>
        >>> # Export knowledge graph
        >>> exporter.export_knowledge_graph(
        ...     graphrag=graphrag,
        ...     format="markdown",
        ...     only_valid=True
        ... )
        >>>
        >>> # Export performance metrics
        >>> exporter.export_metrics(
        ...     memory=short_term_memory,
        ...     format="csv"
        ... )

    Attributes:
        output_dir: Directory for exported files
        version: Export format version
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        version: str = "1.0"
    ):
        """
        Initialize memory exporter.

        Args:
            output_dir: Output directory (default: ./exports)
            version: Export format version
        """
        if output_dir is None:
            output_dir = "./exports"

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.version = version

        logger.info(f"Initialized MemoryExporter: {self.output_dir}")

    def export_session(
        self,
        memory: 'ShortTermMemory',
        format: str = "json",
        filename: Optional[str] = None,
        include_metadata: bool = True
    ) -> Path:
        """
        Export session logs from ShortTermMemory.

        Args:
            memory: ShortTermMemory instance
            format: Export format ("json", "markdown", "yaml")
            filename: Custom filename (auto-generated if None)
            include_metadata: Include export metadata

        Returns:
            Path to exported file

        Raises:
            ValueError: If format is not supported
        """
        logger.info(f"Exporting session {memory.session_id} as {format}")

        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ext = self._get_extension(format)
            filename = f"session_{memory.session_id}_{timestamp}.{ext}"

        output_path = self.output_dir / filename

        # Get session data
        session_data = {
            "session_id": memory.session_id,
            "session_summary": memory.get_session_summary(),
            "tasks": list(memory.tasks.values()),
            "performance_metrics": memory.get_performance_metrics()
        }

        # Add metadata
        if include_metadata:
            metadata = ExportMetadata(
                export_timestamp=datetime.now().isoformat(),
                export_format=format,
                source_type="short_term_memory",
                version=self.version,
                total_records=len(memory.tasks)
            )
            session_data["export_metadata"] = metadata.to_dict()

        # Export in requested format
        if format == "json":
            self._write_json(session_data, output_path)
        elif format == "markdown":
            self._write_session_markdown(session_data, output_path)
        elif format == "yaml":
            self._write_yaml(session_data, output_path)
        else:
            raise ValueError(f"Unsupported format for session export: {format}")

        logger.info(f"Session exported to: {output_path}")
        return output_path

    def export_knowledge_graph(
        self,
        graphrag: 'GraphRAG',
        format: str = "json",
        filename: Optional[str] = None,
        only_valid: bool = True,
        entity_filter: Optional[List[str]] = None,
        max_triples: int = 10000,
        include_metadata: bool = True
    ) -> Path:
        """
        Export knowledge graph from GraphRAG.

        Args:
            graphrag: GraphRAG instance
            format: Export format ("json", "markdown", "csv")
            filename: Custom filename (auto-generated if None)
            only_valid: Only export currently valid knowledge
            entity_filter: Filter by entity names
            max_triples: Maximum triples to export
            include_metadata: Include export metadata

        Returns:
            Path to exported file

        Raises:
            ValueError: If format is not supported
        """
        logger.info(f"Exporting knowledge graph as {format}")

        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ext = self._get_extension(format)
            filename = f"knowledge_graph_{timestamp}.{ext}"

        output_path = self.output_dir / filename

        # Query GraphRAG for triples
        triples_data = self._query_graphrag_triples(
            graphrag=graphrag,
            only_valid=only_valid,
            entity_filter=entity_filter,
            max_triples=max_triples
        )

        # Add metadata
        if include_metadata:
            metadata = ExportMetadata(
                export_timestamp=datetime.now().isoformat(),
                export_format=format,
                source_type="graphrag",
                version=self.version,
                total_records=len(triples_data["triples"]),
                filters_applied={
                    "only_valid": only_valid,
                    "entity_filter": entity_filter,
                    "max_triples": max_triples
                }
            )
            triples_data["export_metadata"] = metadata.to_dict()

        # Export in requested format
        if format == "json":
            self._write_json(triples_data, output_path)
        elif format == "markdown":
            self._write_knowledge_markdown(triples_data, output_path)
        elif format == "csv":
            self._write_triples_csv(triples_data["triples"], output_path)
        else:
            raise ValueError(f"Unsupported format for knowledge graph export: {format}")

        logger.info(f"Knowledge graph exported to: {output_path}")
        return output_path

    def export_document_metadata(
        self,
        graphrag: 'GraphRAG',
        format: str = "json",
        filename: Optional[str] = None,
        file_filter: Optional[str] = None,
        include_metadata: bool = True
    ) -> Path:
        """
        Export document chunk metadata from DocumentStore.

        Args:
            graphrag: GraphRAG instance (with DocumentStore collection)
            format: Export format ("json", "csv", "markdown")
            filename: Custom filename (auto-generated if None)
            file_filter: Filter by file path pattern
            include_metadata: Include export metadata

        Returns:
            Path to exported file

        Raises:
            ValueError: If format is not supported
        """
        logger.info(f"Exporting document metadata as {format}")

        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ext = self._get_extension(format)
            filename = f"document_metadata_{timestamp}.{ext}"

        output_path = self.output_dir / filename

        # Query document metadata
        doc_data = self._query_document_metadata(
            graphrag=graphrag,
            file_filter=file_filter
        )

        # Add metadata
        if include_metadata:
            metadata = ExportMetadata(
                export_timestamp=datetime.now().isoformat(),
                export_format=format,
                source_type="document_store",
                version=self.version,
                total_records=len(doc_data["documents"]),
                filters_applied={"file_filter": file_filter}
            )
            doc_data["export_metadata"] = metadata.to_dict()

        # Export in requested format
        if format == "json":
            self._write_json(doc_data, output_path)
        elif format == "csv":
            self._write_documents_csv(doc_data["documents"], output_path)
        elif format == "markdown":
            self._write_documents_markdown(doc_data, output_path)
        else:
            raise ValueError(f"Unsupported format for document metadata export: {format}")

        logger.info(f"Document metadata exported to: {output_path}")
        return output_path

    def export_metrics(
        self,
        memory: 'ShortTermMemory',
        format: str = "json",
        filename: Optional[str] = None,
        include_metadata: bool = True
    ) -> Path:
        """
        Export performance metrics from ShortTermMemory.

        Args:
            memory: ShortTermMemory instance
            format: Export format ("json", "csv", "yaml")
            filename: Custom filename (auto-generated if None)
            include_metadata: Include export metadata

        Returns:
            Path to exported file

        Raises:
            ValueError: If format is not supported
        """
        logger.info(f"Exporting performance metrics as {format}")

        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ext = self._get_extension(format)
            filename = f"metrics_{memory.session_id}_{timestamp}.{ext}"

        output_path = self.output_dir / filename

        # Get metrics
        metrics = memory.get_performance_metrics()
        metrics_data = {
            "session_id": memory.session_id,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics
        }

        # Add metadata
        if include_metadata:
            metadata = ExportMetadata(
                export_timestamp=datetime.now().isoformat(),
                export_format=format,
                source_type="performance_metrics",
                version=self.version,
                total_records=1
            )
            metrics_data["export_metadata"] = metadata.to_dict()

        # Export in requested format
        if format == "json":
            self._write_json(metrics_data, output_path)
        elif format == "csv":
            self._write_metrics_csv(metrics, output_path)
        elif format == "yaml":
            self._write_yaml(metrics_data, output_path)
        else:
            raise ValueError(f"Unsupported format for metrics export: {format}")

        logger.info(f"Metrics exported to: {output_path}")
        return output_path

    def export_batch_sessions(
        self,
        log_dir: str,
        format: str = "json",
        output_filename: Optional[str] = None,
        date_filter: Optional[str] = None
    ) -> Path:
        """
        Export multiple sessions from log directory.

        Args:
            log_dir: Directory containing session JSON files
            format: Export format ("json", "markdown")
            output_filename: Custom filename (auto-generated if None)
            date_filter: Date pattern for filtering (e.g., "20251019")

        Returns:
            Path to exported file
        """
        logger.info(f"Batch exporting sessions from {log_dir}")

        log_path = Path(log_dir)
        sessions = []

        # Load all session files
        for session_file in log_path.glob("session_*.json"):
            # Apply date filter if provided
            if date_filter and date_filter not in session_file.stem:
                continue

            try:
                with open(session_file, 'r') as f:
                    session_data = json.load(f)
                    sessions.append(session_data)
            except Exception as e:
                logger.warning(f"Failed to load {session_file}: {e}")

        # Generate filename
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ext = self._get_extension(format)
            output_filename = f"batch_sessions_{timestamp}.{ext}"

        output_path = self.output_dir / output_filename

        # Prepare batch data
        batch_data = {
            "export_timestamp": datetime.now().isoformat(),
            "total_sessions": len(sessions),
            "date_filter": date_filter,
            "sessions": sessions
        }

        # Export
        if format == "json":
            self._write_json(batch_data, output_path)
        elif format == "markdown":
            self._write_batch_markdown(batch_data, output_path)
        else:
            raise ValueError(f"Unsupported format for batch export: {format}")

        logger.info(f"Batch export complete: {len(sessions)} sessions -> {output_path}")
        return output_path

    # Internal helper methods

    def _query_graphrag_triples(
        self,
        graphrag: 'GraphRAG',
        only_valid: bool,
        entity_filter: Optional[List[str]],
        max_triples: int
    ) -> Dict[str, Any]:
        """Query knowledge triples from GraphRAG."""
        try:
            with graphrag.graph.session() as session:
                cypher_query = """
                    MATCH (e:Entity)-[r:RELATES]->(t:Entity)
                    WHERE ($entity_filter IS NULL OR e.name IN $entity_filter)
                    AND ($only_valid = false OR r.t_invalid IS NULL)
                    RETURN e.name as entity, r.type as relationship,
                           t.name as target, r.t_valid as t_valid,
                           r.t_invalid as t_invalid, r.metadata as metadata
                    ORDER BY r.t_valid DESC
                    LIMIT $max_triples
                """

                results = session.run(
                    cypher_query,
                    entity_filter=entity_filter,
                    only_valid=only_valid,
                    max_triples=max_triples
                )

                triples = []
                entities = set()
                relationships = set()

                for record in results:
                    triple = {
                        "entity": record["entity"],
                        "relationship": record["relationship"],
                        "target": record["target"],
                        "t_valid": record["t_valid"],
                        "t_invalid": record["t_invalid"]
                    }

                    # Decode JSON metadata if present
                    if record.get("metadata"):
                        try:
                            triple["metadata"] = json.loads(record["metadata"])
                        except (json.JSONDecodeError, TypeError):
                            triple["metadata"] = {}
                    else:
                        triple["metadata"] = {}

                    triples.append(triple)
                    entities.add(record["entity"])
                    entities.add(record["target"])
                    relationships.add(record["relationship"])

                return {
                    "triples": triples,
                    "statistics": {
                        "total_triples": len(triples),
                        "unique_entities": len(entities),
                        "unique_relationships": len(relationships),
                        "entities": sorted(list(entities)),
                        "relationships": sorted(list(relationships))
                    }
                }
        except Exception as e:
            logger.error(f"Failed to query GraphRAG: {e}")
            return {"triples": [], "statistics": {}, "error": str(e)}

    def _query_document_metadata(
        self,
        graphrag: 'GraphRAG',
        file_filter: Optional[str]
    ) -> Dict[str, Any]:
        """Query document metadata from Neo4j."""
        try:
            with graphrag.graph.session() as session:
                cypher_query = """
                    MATCH (d:Document)
                    WHERE ($file_filter IS NULL OR d.file_path CONTAINS $file_filter)
                    RETURN d.file_path as file_path,
                           d.file_name as file_name,
                           d.total_chunks as total_chunks,
                           d.metadata as metadata
                    ORDER BY d.file_name
                """

                results = session.run(
                    cypher_query,
                    file_filter=file_filter
                )

                documents = []
                total_chunks = 0

                for record in results:
                    doc = {
                        "file_path": record["file_path"],
                        "file_name": record["file_name"],
                        "total_chunks": record["total_chunks"]
                    }

                    # Decode JSON metadata
                    if record.get("metadata"):
                        try:
                            doc["metadata"] = json.loads(record["metadata"])
                        except (json.JSONDecodeError, TypeError):
                            doc["metadata"] = {}
                    else:
                        doc["metadata"] = {}

                    documents.append(doc)
                    total_chunks += record["total_chunks"]

                return {
                    "documents": documents,
                    "statistics": {
                        "total_documents": len(documents),
                        "total_chunks": total_chunks
                    }
                }
        except Exception as e:
            logger.error(f"Failed to query document metadata: {e}")
            return {"documents": [], "statistics": {}, "error": str(e)}

    def _get_extension(self, format: str) -> str:
        """Get file extension for format."""
        extensions = {
            "json": "json",
            "markdown": "md",
            "csv": "csv",
            "yaml": "yaml"
        }
        return extensions.get(format, "txt")

    def _write_json(self, data: Dict[str, Any], path: Path):
        """Write data as JSON."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)

    def _write_yaml(self, data: Dict[str, Any], path: Path):
        """Write data as YAML."""
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def _write_session_markdown(self, session_data: Dict[str, Any], path: Path):
        """Write session as Markdown."""
        lines = []

        # Title and metadata
        lines.append(f"# Session Export: {session_data['session_id']}")
        lines.append("")
        lines.append(f"**Export Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Summary
        summary = session_data["session_summary"]
        lines.append("## Session Summary")
        lines.append("")
        lines.append(f"- **Total Tasks**: {summary['total_tasks']}")
        lines.append(f"- **Completed**: {summary['completed_tasks']}")
        lines.append(f"- **In Progress**: {summary['in_progress_tasks']}")
        lines.append(f"- **Average Duration**: {summary['avg_duration_seconds']:.2f}s")
        lines.append("")

        # Performance Metrics
        if "performance_metrics" in session_data:
            metrics = session_data["performance_metrics"]
            lines.append("## Performance Metrics")
            lines.append("")
            lines.append(f"- **Accuracy**: {metrics['accuracy']:.2%}")
            lines.append(f"- **Total Cost**: ${metrics['cost']:.4f}")
            lines.append(f"- **Latency**: {metrics['latency']:.2f}s")
            lines.append(f"- **Cache Hit Rate**: {metrics['cache_hit_rate']:.2%}")
            lines.append("")

        # Tasks
        lines.append("## Tasks")
        lines.append("")
        for task in session_data["tasks"]:
            status_emoji = {"completed": "✅", "in_progress": "🔄", "failed": "❌"}.get(
                task.get("status", ""), "⚪"
            )
            lines.append(f"### {status_emoji} {task['task_description']}")
            lines.append("")
            lines.append(f"- **Agent**: `{task['agent_id']}`")
            lines.append(f"- **Type**: {task['agent_type']}")
            lines.append(f"- **Status**: {task['status']}")
            if task.get("duration_seconds"):
                lines.append(f"- **Duration**: {task['duration_seconds']:.2f}s")
            lines.append("")

        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

    def _write_knowledge_markdown(self, triples_data: Dict[str, Any], path: Path):
        """Write knowledge graph as Markdown."""
        lines = []

        # Title
        lines.append("# Knowledge Graph Export")
        lines.append("")
        lines.append(f"**Export Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Statistics
        stats = triples_data.get("statistics", {})
        lines.append("## Statistics")
        lines.append("")
        lines.append(f"- **Total Triples**: {stats.get('total_triples', 0)}")
        lines.append(f"- **Unique Entities**: {stats.get('unique_entities', 0)}")
        lines.append(f"- **Unique Relationships**: {stats.get('unique_relationships', 0)}")
        lines.append("")

        # Group triples by entity
        entity_map = {}
        for triple in triples_data["triples"]:
            entity = triple["entity"]
            if entity not in entity_map:
                entity_map[entity] = []
            entity_map[entity].append(triple)

        # Render grouped triples
        lines.append("## Knowledge Triples")
        lines.append("")
        for entity, triples in sorted(entity_map.items()):
            lines.append(f"### {entity}")
            lines.append("")
            for triple in triples:
                status = "" if triple["t_invalid"] is None else " (invalid)"
                lines.append(f"- **{triple['relationship']}**: {triple['target']}{status}")
            lines.append("")

        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

    def _write_triples_csv(self, triples: List[Dict[str, Any]], path: Path):
        """Write knowledge triples as CSV."""
        if not triples:
            # Write empty CSV with headers
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['entity', 'relationship', 'target', 't_valid', 't_invalid'])
            return

        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=['entity', 'relationship', 'target', 't_valid', 't_invalid']
            )
            writer.writeheader()
            for triple in triples:
                writer.writerow({
                    'entity': triple['entity'],
                    'relationship': triple['relationship'],
                    'target': triple['target'],
                    't_valid': triple.get('t_valid', ''),
                    't_invalid': triple.get('t_invalid', '')
                })

    def _write_documents_csv(self, documents: List[Dict[str, Any]], path: Path):
        """Write document metadata as CSV."""
        if not documents:
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['file_path', 'file_name', 'total_chunks'])
            return

        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=['file_path', 'file_name', 'total_chunks']
            )
            writer.writeheader()
            for doc in documents:
                writer.writerow({
                    'file_path': doc['file_path'],
                    'file_name': doc['file_name'],
                    'total_chunks': doc['total_chunks']
                })

    def _write_documents_markdown(self, doc_data: Dict[str, Any], path: Path):
        """Write document metadata as Markdown."""
        lines = []

        lines.append("# Document Metadata Export")
        lines.append("")
        lines.append(f"**Export Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Statistics
        stats = doc_data.get("statistics", {})
        lines.append("## Statistics")
        lines.append("")
        lines.append(f"- **Total Documents**: {stats.get('total_documents', 0)}")
        lines.append(f"- **Total Chunks**: {stats.get('total_chunks', 0)}")
        lines.append("")

        # Documents
        lines.append("## Documents")
        lines.append("")
        for doc in doc_data["documents"]:
            lines.append(f"### {doc['file_name']}")
            lines.append("")
            lines.append(f"- **Path**: `{doc['file_path']}`")
            lines.append(f"- **Chunks**: {doc['total_chunks']}")
            lines.append("")

        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

    def _write_metrics_csv(self, metrics: Dict[str, Any], path: Path):
        """Write performance metrics as CSV."""
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'value'])
            for key, value in metrics.items():
                if key != 'failed_tasks':  # Skip list field
                    writer.writerow([key, value])

    def _write_batch_markdown(self, batch_data: Dict[str, Any], path: Path):
        """Write batch sessions as Markdown."""
        lines = []

        lines.append("# Batch Session Export")
        lines.append("")
        lines.append(f"**Export Date**: {batch_data['export_timestamp']}")
        lines.append(f"**Total Sessions**: {batch_data['total_sessions']}")
        if batch_data.get('date_filter'):
            lines.append(f"**Date Filter**: {batch_data['date_filter']}")
        lines.append("")

        # Session summaries
        lines.append("## Sessions")
        lines.append("")
        for session in batch_data["sessions"]:
            session_id = session.get("session_id", "unknown")
            num_tasks = session.get("num_tasks", 0)
            lines.append(f"### {session_id}")
            lines.append("")
            lines.append(f"- **Tasks**: {num_tasks}")
            lines.append(f"- **Timestamp**: {session.get('timestamp', 'N/A')}")
            lines.append("")

        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))


def export_memory_snapshot(
    short_term_memory: Optional['ShortTermMemory'] = None,
    graphrag: Optional['GraphRAG'] = None,
    output_dir: str = "./memory_snapshot",
    format: str = "json"
) -> Dict[str, Path]:
    """
    Export complete memory snapshot (convenience function).

    Creates a comprehensive export of all memory systems:
    - Session logs (if short_term_memory provided)
    - Knowledge graph (if graphrag provided)
    - Document metadata (if graphrag provided)
    - Performance metrics (if short_term_memory provided)

    Args:
        short_term_memory: Optional ShortTermMemory instance
        graphrag: Optional GraphRAG instance
        output_dir: Output directory for snapshot
        format: Export format (default: json)

    Returns:
        Dict mapping export types to file paths

    Usage:
        >>> from fractal_agent.memory.export import export_memory_snapshot
        >>> paths = export_memory_snapshot(
        ...     short_term_memory=memory,
        ...     graphrag=graphrag,
        ...     output_dir="./snapshots/20251019",
        ...     format="json"
        ... )
        >>> print(paths['session'])
        >>> print(paths['knowledge_graph'])
    """
    logger.info("Creating memory snapshot")

    exporter = MemoryExporter(output_dir=output_dir)
    export_paths = {}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Export session if available
    if short_term_memory:
        try:
            path = exporter.export_session(
                memory=short_term_memory,
                format=format,
                filename=f"session_{timestamp}.{exporter._get_extension(format)}"
            )
            export_paths['session'] = path
        except Exception as e:
            logger.error(f"Failed to export session: {e}")

    # Export knowledge graph if available
    if graphrag:
        try:
            path = exporter.export_knowledge_graph(
                graphrag=graphrag,
                format=format,
                filename=f"knowledge_graph_{timestamp}.{exporter._get_extension(format)}"
            )
            export_paths['knowledge_graph'] = path
        except Exception as e:
            logger.error(f"Failed to export knowledge graph: {e}")

        # Export document metadata
        try:
            path = exporter.export_document_metadata(
                graphrag=graphrag,
                format=format,
                filename=f"documents_{timestamp}.{exporter._get_extension(format)}"
            )
            export_paths['documents'] = path
        except Exception as e:
            logger.error(f"Failed to export document metadata: {e}")

    # Export metrics if available
    if short_term_memory:
        try:
            path = exporter.export_metrics(
                memory=short_term_memory,
                format=format,
                filename=f"metrics_{timestamp}.{exporter._get_extension(format)}"
            )
            export_paths['metrics'] = path
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")

    logger.info(f"Memory snapshot complete: {len(export_paths)} exports")
    return export_paths


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("=" * 80)
    print("Memory Export System Demo - Phase 4")
    print("=" * 80)
    print()

    # Demo with ShortTermMemory
    print("[1/3] Testing session export...")
    from fractal_agent.memory.short_term import ShortTermMemory

    memory = ShortTermMemory(log_dir="./test_logs")
    task_id = memory.start_task(
        agent_id="demo_001",
        agent_type="research",
        task_description="Demo task for export testing",
        inputs={"test": "data"}
    )
    memory.end_task(
        task_id=task_id,
        outputs={"result": "success"},
        metadata={"tokens": 1000}
    )

    exporter = MemoryExporter(output_dir="./test_exports")

    # Test JSON export
    json_path = exporter.export_session(memory, format="json")
    print(f"✅ JSON export: {json_path}")

    # Test Markdown export
    md_path = exporter.export_session(memory, format="markdown")
    print(f"✅ Markdown export: {md_path}")

    # Test YAML export
    yaml_path = exporter.export_session(memory, format="yaml")
    print(f"✅ YAML export: {yaml_path}")
    print()

    # Test metrics export
    print("[2/3] Testing metrics export...")
    metrics_path = exporter.export_metrics(memory, format="csv")
    print(f"✅ Metrics CSV export: {metrics_path}")
    print()

    # Test snapshot function
    print("[3/3] Testing memory snapshot...")
    snapshot_paths = export_memory_snapshot(
        short_term_memory=memory,
        output_dir="./test_snapshot",
        format="json"
    )
    print(f"✅ Snapshot created: {len(snapshot_paths)} files")
    for export_type, path in snapshot_paths.items():
        print(f"   - {export_type}: {path.name}")
    print()

    print("=" * 80)
    print("Memory Export System Demo Complete!")
    print("=" * 80)
    print()
    print("Exported files:")
    print(f"- Session JSON: {json_path}")
    print(f"- Session Markdown: {md_path}")
    print(f"- Session YAML: {yaml_path}")
    print(f"- Metrics CSV: {metrics_path}")
    print()
    print("Note: GraphRAG exports require Neo4j and Qdrant to be running.")
    print("Start services with: docker-compose up -d")
