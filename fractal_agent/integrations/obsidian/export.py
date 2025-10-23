"""
Obsidian Export with Observability Instrumentation

Exports agent outputs and observability data to Obsidian vault for human review:
- Research results with full metadata
- Intelligence analyses and recommendations
- LLM call metrics and cost tracking
- Event timeline visualization
- Knowledge graph triples

Fully instrumented with observability framework.

Author: BMad
Date: 2025-01-20
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import json

# Observability imports
from ...observability import (
    get_logger, get_tracer, get_event_store, get_correlation_id,
    VSMEvent, set_span_attributes
)
from ...observability.metrics import record_agent_operation

logger = get_logger(__name__)
tracer = get_tracer(__name__)


@dataclass
class ExportMetadata:
    """Metadata for exported content"""
    timestamp: datetime
    correlation_id: str
    agent_type: str
    task_description: str
    model_used: Optional[str] = None
    tokens_used: Optional[int] = None
    cost_usd: Optional[float] = None
    duration_seconds: Optional[float] = None


class ObsidianExporter:
    """
    Exports agent outputs and observability data to Obsidian vault.

    Fully instrumented with:
    - Distributed tracing
    - Event emissions
    - Structured logging
    - Metrics recording
    """

    def __init__(self, vault_path: str = "knowledge_vault"):
        """
        Initialize Obsidian exporter.

        Args:
            vault_path: Path to Obsidian vault directory
        """
        self.vault_path = Path(vault_path)
        self.event_store = get_event_store()

        # Ensure vault structure exists
        self._ensure_vault_structure()

        logger.info(f"ObsidianExporter initialized", extra={
            "vault_path": str(self.vault_path),
            "correlation_id": get_correlation_id()
        })

    def _ensure_vault_structure(self):
        """Create vault directory structure if it doesn't exist"""
        directories = [
            "agents/research",
            "agents/intelligence",
            "agents/developer",
            "observability/metrics",
            "observability/events",
            "observability/traces",
            "knowledge/entities",
            "knowledge/relations"
        ]

        for dir_path in directories:
            full_path = self.vault_path / dir_path
            full_path.mkdir(parents=True, exist_ok=True)

    def export_research_result(
        self,
        research_output: Dict[str, Any],
        metadata: ExportMetadata
    ) -> Path:
        """
        Export research agent results to Obsidian vault.

        Args:
            research_output: Research results (decomposition, questions, findings)
            metadata: Export metadata with observability info

        Returns:
            Path to exported markdown file
        """
        with tracer.start_as_current_span("export_research_result") as span:
            set_span_attributes({
                "export.type": "research",
                "export.vault_path": str(self.vault_path),
                "metadata.agent_type": metadata.agent_type,
                "correlation_id": metadata.correlation_id
            })

            # Emit event
            self.event_store.append(VSMEvent(
                tier="Obsidian_Export",
                event_type="export_started",
                data={
                    "export_type": "research",
                    "task": metadata.task_description,
                    "correlation_id": metadata.correlation_id
                }
            ))

            start_time = datetime.now()

            try:
                # Generate markdown content
                content = self._generate_research_markdown(research_output, metadata)

                # Generate filename
                safe_task = self._sanitize_filename(metadata.task_description)
                timestamp_str = metadata.timestamp.strftime("%Y%m%d_%H%M%S")
                filename = f"{timestamp_str}_{safe_task}.md"
                file_path = self.vault_path / "agents/research" / filename

                # Write file
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)

                duration = (datetime.now() - start_time).total_seconds()

                # Log success
                logger.info(
                    f"Research results exported to Obsidian",
                    extra={
                        "file_path": str(file_path),
                        "correlation_id": metadata.correlation_id,
                        "duration_seconds": duration
                    }
                )

                # Emit completion event
                self.event_store.append(VSMEvent(
                    tier="Obsidian_Export",
                    event_type="export_completed",
                    data={
                        "export_type": "research",
                        "file_path": str(file_path),
                        "duration_seconds": duration,
                        "correlation_id": metadata.correlation_id
                    }
                ))

                # Record metrics
                record_agent_operation(
                    agent_type="ObsidianExporter",
                    operation="export_research",
                    duration_seconds=duration,
                    status="success"
                )

                set_span_attributes({
                    "export.success": True,
                    "export.file_path": str(file_path),
                    "export.duration_seconds": duration
                })

                return file_path

            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()

                logger.error(
                    f"Failed to export research results",
                    extra={
                        "error": str(e),
                        "correlation_id": metadata.correlation_id
                    }
                )

                # Emit failure event
                self.event_store.append(VSMEvent(
                    tier="Obsidian_Export",
                    event_type="export_failed",
                    data={
                        "export_type": "research",
                        "error": str(e),
                        "correlation_id": metadata.correlation_id
                    }
                ))

                # Record failure metrics
                record_agent_operation(
                    agent_type="ObsidianExporter",
                    operation="export_research",
                    duration_seconds=duration,
                    status="error"
                )

                set_span_attributes({
                    "export.success": False,
                    "export.error": str(e)
                })

                raise

    def export_intelligence_analysis(
        self,
        analysis: Dict[str, Any],
        metadata: ExportMetadata
    ) -> Path:
        """
        Export intelligence layer analysis to Obsidian vault.

        Args:
            analysis: Intelligence analysis results
            metadata: Export metadata with observability info

        Returns:
            Path to exported markdown file
        """
        with tracer.start_as_current_span("export_intelligence_analysis") as span:
            set_span_attributes({
                "export.type": "intelligence",
                "correlation_id": metadata.correlation_id
            })

            start_time = datetime.now()

            try:
                # Generate markdown content
                content = self._generate_intelligence_markdown(analysis, metadata)

                # Generate filename
                timestamp_str = metadata.timestamp.strftime("%Y%m%d_%H%M%S")
                filename = f"{timestamp_str}_intelligence_analysis.md"
                file_path = self.vault_path / "agents/intelligence" / filename

                # Write file
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)

                duration = (datetime.now() - start_time).total_seconds()

                logger.info(
                    f"Intelligence analysis exported to Obsidian",
                    extra={
                        "file_path": str(file_path),
                        "correlation_id": metadata.correlation_id,
                        "duration_seconds": duration
                    }
                )

                # Record metrics
                record_agent_operation(
                    agent_type="ObsidianExporter",
                    operation="export_intelligence",
                    duration_seconds=duration,
                    status="success"
                )

                return file_path

            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                logger.error(f"Failed to export intelligence analysis", extra={"error": str(e)})

                record_agent_operation(
                    agent_type="ObsidianExporter",
                    operation="export_intelligence",
                    duration_seconds=duration,
                    status="error"
                )
                raise

    def export_observability_snapshot(
        self,
        correlation_id: str,
        include_events: bool = True,
        include_metrics: bool = True
    ) -> Path:
        """
        Export observability snapshot for a specific request.

        Args:
            correlation_id: Correlation ID to export
            include_events: Include event timeline
            include_metrics: Include metrics summary

        Returns:
            Path to exported markdown file
        """
        with tracer.start_as_current_span("export_observability_snapshot") as span:
            set_span_attributes({
                "export.type": "observability",
                "correlation_id": correlation_id
            })

            start_time = datetime.now()

            try:
                # Build content
                content_parts = [
                    f"# Observability Snapshot\n",
                    f"**Correlation ID**: `{correlation_id}`\n",
                    f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                ]

                # Include events timeline
                if include_events:
                    events = self.event_store.get_by_correlation_id(correlation_id)
                    content_parts.append("## Event Timeline\n\n")
                    for event in events:
                        timestamp = event.timestamp.strftime('%H:%M:%S.%f')[:-3]
                        content_parts.append(f"- **{timestamp}** [{event.tier}] {event.event_type}\n")
                        if event.event_type == "llm_call_completed":
                            data = event.data
                            content_parts.append(f"  - Model: {data.get('provider')}/{data.get('model')}\n")
                            content_parts.append(f"  - Tokens: {data.get('tokens_used')}\n")
                            content_parts.append(f"  - Cost: ${data.get('estimated_cost', 0):.4f}\n")
                            content_parts.append(f"  - Latency: {data.get('latency_ms')}ms\n")
                    content_parts.append("\n")

                # Include metrics summary
                if include_metrics:
                    content_parts.append("## Metrics Summary\n\n")
                    content_parts.append("*Metrics aggregation from Prometheus would go here*\n\n")

                # Write file
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{timestamp_str}_observability_{correlation_id[:8]}.md"
                file_path = self.vault_path / "observability/traces" / filename

                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(''.join(content_parts))

                duration = (datetime.now() - start_time).total_seconds()

                logger.info(
                    f"Observability snapshot exported",
                    extra={
                        "file_path": str(file_path),
                        "correlation_id": correlation_id,
                        "duration_seconds": duration
                    }
                )

                record_agent_operation(
                    agent_type="ObsidianExporter",
                    operation="export_observability",
                    duration_seconds=duration,
                    status="success"
                )

                return file_path

            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                logger.error(f"Failed to export observability snapshot", extra={"error": str(e)})

                record_agent_operation(
                    agent_type="ObsidianExporter",
                    operation="export_observability",
                    duration_seconds=duration,
                    status="error"
                )
                raise

    def _generate_research_markdown(
        self,
        research_output: Dict[str, Any],
        metadata: ExportMetadata
    ) -> str:
        """Generate markdown content for research results"""
        lines = [
            f"# Research: {metadata.task_description}\n",
            f"\n",
            f"## Metadata\n",
            f"- **Timestamp**: {metadata.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n",
            f"- **Correlation ID**: `{metadata.correlation_id}`\n",
            f"- **Agent**: {metadata.agent_type}\n"
        ]

        if metadata.model_used:
            lines.append(f"- **Model**: {metadata.model_used}\n")
        if metadata.tokens_used:
            lines.append(f"- **Tokens Used**: {metadata.tokens_used:,}\n")
        if metadata.cost_usd:
            lines.append(f"- **Cost**: ${metadata.cost_usd:.4f}\n")
        if metadata.duration_seconds:
            lines.append(f"- **Duration**: {metadata.duration_seconds:.2f}s\n")

        lines.append("\n")

        # Add research content
        if "decomposition" in research_output:
            lines.append("## Task Decomposition\n\n")
            lines.append(research_output["decomposition"] + "\n\n")

        if "questions" in research_output:
            lines.append("## Research Questions\n\n")
            for i, question in enumerate(research_output["questions"], 1):
                lines.append(f"{i}. {question}\n")
            lines.append("\n")

        if "findings" in research_output:
            lines.append("## Findings\n\n")
            lines.append(research_output["findings"] + "\n\n")

        # Add tags for Obsidian
        lines.append("---\n")
        lines.append(f"#research #agent-output #correlation/{metadata.correlation_id[:8]}\n")

        return ''.join(lines)

    def _generate_intelligence_markdown(
        self,
        analysis: Dict[str, Any],
        metadata: ExportMetadata
    ) -> str:
        """Generate markdown content for intelligence analysis"""
        lines = [
            f"# Intelligence Analysis\n",
            f"\n",
            f"## Metadata\n",
            f"- **Timestamp**: {metadata.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n",
            f"- **Correlation ID**: `{metadata.correlation_id}`\n",
            f"- **Agent**: {metadata.agent_type}\n\n"
        ]

        # Add analysis content
        if "trigger_reason" in analysis:
            lines.append(f"## Trigger Reason\n\n{analysis['trigger_reason']}\n\n")

        if "observations" in analysis:
            lines.append("## Observations\n\n")
            for obs in analysis["observations"]:
                lines.append(f"- {obs}\n")
            lines.append("\n")

        if "recommendations" in analysis:
            lines.append("## Recommendations\n\n")
            for rec in analysis["recommendations"]:
                lines.append(f"- {rec}\n")
            lines.append("\n")

        # Add tags
        lines.append("---\n")
        lines.append(f"#intelligence #analysis #system4 #correlation/{metadata.correlation_id[:8]}\n")

        return ''.join(lines)

    def _sanitize_filename(self, text: str, max_length: int = 50) -> str:
        """Sanitize text for use in filename"""
        # Remove/replace invalid characters
        safe = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in text)
        # Trim whitespace and limit length
        safe = safe.strip()[:max_length]
        # Replace spaces with underscores
        safe = safe.replace(' ', '_')
        return safe.lower()
