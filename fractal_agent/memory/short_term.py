"""
Short-Term Memory - JSON Session Logs (Phase 1)

Logs full task trees for debugging, auditing, and human review.

Schema: Structured logs with agent_id, task_id, inputs, outputs, metrics
Retention: 30 days (configurable)
Purpose: Debugging, auditing, performance analysis

Author: BMad
Date: 2025-10-18
"""

import json
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import uuid

logger = logging.getLogger(__name__)


@dataclass
class TaskLog:
    """
    Log entry for a single task execution.

    Attributes:
        task_id: Unique task identifier
        agent_id: Agent that executed the task
        agent_type: Type of agent (control, operational, etc.)
        parent_task_id: ID of parent task (for task trees)
        task_description: What the task was
        inputs: Task inputs
        outputs: Task outputs
        metadata: Additional metrics and info
        timestamp_start: When task started
        timestamp_end: When task completed
        duration_seconds: How long it took
    """
    task_id: str
    agent_id: str
    agent_type: str
    parent_task_id: Optional[str]
    task_description: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    metadata: Dict[str, Any]
    timestamp_start: str
    timestamp_end: str
    duration_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


class ShortTermMemory:
    """
    Short-term memory manager using JSON file storage.

    Stores structured logs of all task executions with full task trees.

    Usage:
        >>> memory = ShortTermMemory()
        >>> task_id = memory.start_task(
        ...     agent_id="control_001",
        ...     agent_type="control",
        ...     task_description="Research VSM",
        ...     inputs={"topic": "Viable System Model"}
        ... )
        >>> memory.end_task(
        ...     task_id=task_id,
        ...     outputs={"report": "..."},
        ...     metadata={"tokens": 1000}
        ... )
        >>> memory.save_session()
    """

    def __init__(
        self,
        log_dir: Optional[str] = None,
        retention_days: int = 30,
        graphrag: Optional[Any] = None,
        enable_extraction: bool = True,
        obsidian_vault: Optional[Any] = None,
        enable_auto_export: bool = True
    ):
        """
        Initialize short-term memory.

        Args:
            log_dir: Directory for log files (default: ./logs/sessions)
            retention_days: How long to keep logs
            graphrag: GraphRAG instance for long-term storage (optional)
            enable_extraction: Whether to enable automatic knowledge extraction
            obsidian_vault: ObsidianVault instance for automatic exports (optional)
            enable_auto_export: Whether to enable automatic Obsidian exports
        """
        if log_dir is None:
            log_dir = "./logs/sessions"

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.retention_days = retention_days
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.session_file = self.log_dir / f"{self.session_id}.json"

        # In-memory task log for current session
        self.tasks: Dict[str, Dict[str, Any]] = {}

        # Knowledge extraction setup
        self.graphrag = graphrag
        self.enable_extraction = enable_extraction
        self.extraction_agent = None

        # Lazy-load extraction agent only if needed
        if self.enable_extraction and self.graphrag:
            try:
                from ..agents.knowledge_extraction_agent import KnowledgeExtractionAgent
                self.extraction_agent = KnowledgeExtractionAgent()
                logger.info("Knowledge extraction enabled for session")
            except ImportError:
                logger.warning("KnowledgeExtractionAgent not available, extraction disabled")
                self.enable_extraction = False

        # Obsidian export setup
        self.obsidian_vault = obsidian_vault
        self.enable_auto_export = enable_auto_export
        self.export_status: Optional[str] = None  # None, "pending", "exporting", "completed", "failed"
        self.export_path: Optional[Path] = None

        if self.enable_auto_export and not self.obsidian_vault:
            logger.warning("Auto-export enabled but no ObsidianVault provided - exports disabled")
            self.enable_auto_export = False

        logger.info(f"Initialized ShortTermMemory: {self.session_file}")

    def start_task(
        self,
        agent_id: str,
        agent_type: str,
        task_description: str,
        inputs: Dict[str, Any],
        parent_task_id: Optional[str] = None
    ) -> str:
        """
        Log the start of a task.

        Args:
            agent_id: Unique agent identifier
            agent_type: Type of agent (control, operational, etc.)
            task_description: What the task is
            inputs: Task input parameters
            parent_task_id: ID of parent task (for task trees)

        Returns:
            task_id: Unique identifier for this task
        """
        task_id = f"task_{uuid.uuid4().hex[:8]}"

        self.tasks[task_id] = {
            "task_id": task_id,
            "agent_id": agent_id,
            "agent_type": agent_type,
            "parent_task_id": parent_task_id,
            "task_description": task_description,
            "inputs": inputs,
            "outputs": None,
            "metadata": {},
            "timestamp_start": datetime.now().isoformat(),
            "timestamp_end": None,
            "duration_seconds": None,
            "status": "in_progress"
        }

        logger.debug(f"Started task {task_id}: {task_description}")

        return task_id

    def end_task(
        self,
        task_id: str,
        outputs: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        promote_to_longterm: bool = False
    ):
        """
        Log the completion of a task.

        Args:
            task_id: Task identifier from start_task()
            outputs: Task outputs
            metadata: Additional metadata (tokens, models used, etc.)
            promote_to_longterm: If True, extract knowledge and store in GraphRAG
        """
        if task_id not in self.tasks:
            logger.error(f"Task {task_id} not found in memory")
            return

        task = self.tasks[task_id]

        # Calculate duration
        start_time = datetime.fromisoformat(task["timestamp_start"])
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Update task
        task["timestamp_end"] = end_time.isoformat()
        task["duration_seconds"] = duration
        task["outputs"] = outputs
        task["metadata"] = metadata or {}
        task["status"] = "completed"

        logger.debug(f"Completed task {task_id} in {duration:.2f}s")

        # Knowledge extraction
        if promote_to_longterm and self.enable_extraction and self.extraction_agent and self.graphrag:
            self._extract_and_store_knowledge(task_id, task)

    def _extract_and_store_knowledge(self, task_id: str, task: Dict[str, Any]):
        """
        Extract knowledge from task and store in GraphRAG.

        Args:
            task_id: Task identifier
            task: Task dictionary with all task data
        """
        try:
            from ..memory.embeddings import generate_embedding

            # Extract knowledge using extraction agent
            result = self.extraction_agent(
                task_description=task["task_description"],
                task_output=str(task["outputs"]),
                context=str(task.get("inputs", ""))
            )

            # Only store if confidence threshold met
            confidence_threshold = 0.7
            if result["confidence"] < confidence_threshold:
                logger.info(
                    f"Extraction confidence {result['confidence']:.2f} below threshold "
                    f"{confidence_threshold}, skipping GraphRAG storage"
                )
                return

            # Store each entity
            for entity_data in result["entities"]:
                try:
                    # Generate embedding for entity
                    text = f"{entity_data['name']}: {entity_data['description']}"
                    embedding = generate_embedding(text)

                    # Store in GraphRAG
                    self.graphrag.store_knowledge(
                        entity=entity_data["name"],
                        relationship="is_a",  # Type relationship
                        target=entity_data["type"],
                        embedding=embedding,
                        metadata={
                            "description": entity_data["description"],
                            "source_task": task_id,
                            "source_session": self.session_id,
                            "extraction_confidence": result["confidence"],
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    )
                except Exception as e:
                    logger.error(f"Failed to store entity {entity_data.get('name', 'unknown')}: {e}")

            # Store each relationship
            for rel_data in result["relationships"]:
                try:
                    # Generate embedding for relationship
                    text = f"{rel_data['from_entity']} {rel_data['type']} {rel_data['to_entity']}"
                    embedding = generate_embedding(text)

                    # Store in GraphRAG
                    self.graphrag.store_knowledge(
                        entity=rel_data["from_entity"],
                        relationship=rel_data["type"],
                        target=rel_data["to_entity"],
                        embedding=embedding,
                        metadata={
                            "strength": rel_data.get("strength", 1.0),
                            "source_task": task_id,
                            "source_session": self.session_id,
                            "extraction_confidence": result["confidence"],
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    )
                except Exception as e:
                    logger.error(f"Failed to store relationship {rel_data.get('from_entity', 'unknown')} -> {rel_data.get('to_entity', 'unknown')}: {e}")

            logger.info(
                f"Extracted {len(result['entities'])} entities, {len(result['relationships'])} relationships "
                f"to GraphRAG (confidence={result['confidence']:.2f})"
            )

        except Exception as e:
            logger.error(f"Knowledge extraction failed for task {task_id}: {e}", exc_info=True)
            # Don't fail the task if extraction fails

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task log by ID"""
        return self.tasks.get(task_id)

    def get_recent_tasks(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get most recent tasks from session.

        Args:
            limit: Maximum number of tasks to return

        Returns:
            List of task dictionaries, most recent first
        """
        # Sort tasks by start_time (most recent first)
        sorted_tasks = sorted(
            self.tasks.values(),
            key=lambda t: t.get("start_time", ""),
            reverse=True
        )
        return sorted_tasks[:limit]

    def get_task_tree(self, root_task_id: str) -> List[Dict[str, Any]]:
        """
        Get full task tree starting from root task.

        Returns list of tasks in tree order (root first, then children)
        """
        tree = []

        def add_task_and_children(task_id: str):
            task = self.tasks.get(task_id)
            if task:
                tree.append(task)

                # Find child tasks
                children = [
                    t for t in self.tasks.values()
                    if t.get("parent_task_id") == task_id
                ]

                for child in children:
                    add_task_and_children(child["task_id"])

        add_task_and_children(root_task_id)
        return tree

    def save_session(self):
        """
        Save current session to JSON file.

        Creates structured JSON with:
        - Session metadata
        - All task logs
        - Task trees (relationships preserved)
        """
        session_data = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "num_tasks": len(self.tasks),
            "tasks": list(self.tasks.values())
        }

        with open(self.session_file, 'w') as f:
            json.dump(session_data, f, indent=2)

        logger.info(f"Saved session with {len(self.tasks)} tasks to {self.session_file}")

    def end_session(
        self,
        async_export: bool = True,
        include_approval: bool = True
    ) -> Optional[Path]:
        """
        End the current session and trigger automatic exports.

        This method:
        1. Saves session to JSON log file
        2. Triggers Obsidian export (if enabled)
        3. Can export synchronously or asynchronously

        Args:
            async_export: If True, export runs in background thread (non-blocking)
            include_approval: Include human approval template in Obsidian export

        Returns:
            Path to Obsidian export file (if sync), or None (if async or disabled)
        """
        logger.info(f"Ending session {self.session_id}")

        # Save session to JSON
        self.save_session()

        # Trigger Obsidian export if enabled
        if self.enable_auto_export and self.obsidian_vault:
            if async_export:
                # Background export (non-blocking)
                self.export_status = "pending"
                export_thread = threading.Thread(
                    target=self._async_export_to_obsidian,
                    args=(include_approval,),
                    daemon=True
                )
                export_thread.start()
                logger.info("Started async Obsidian export")
                return None
            else:
                # Synchronous export (blocking)
                return self._export_to_obsidian(include_approval)
        else:
            logger.debug("Obsidian auto-export not enabled")
            return None

    def _export_to_obsidian(self, include_approval: bool = True) -> Optional[Path]:
        """
        Export session to Obsidian (synchronous).

        Args:
            include_approval: Include human approval template

        Returns:
            Path to exported file, or None on failure
        """
        try:
            self.export_status = "exporting"
            logger.info(f"Exporting session {self.session_id} to Obsidian...")

            file_path = self.obsidian_vault.export_session(
                memory=self,
                include_approval=include_approval
            )

            self.export_status = "completed"
            self.export_path = file_path
            logger.info(f"✅ Exported session to Obsidian: {file_path}")
            return file_path

        except Exception as e:
            self.export_status = "failed"
            logger.error(f"❌ Failed to export session to Obsidian: {e}", exc_info=True)
            return None

    def _async_export_to_obsidian(self, include_approval: bool = True):
        """
        Export session to Obsidian asynchronously (background thread).

        Args:
            include_approval: Include human approval template
        """
        self._export_to_obsidian(include_approval)

    def get_export_status(self) -> Dict[str, Any]:
        """
        Get current export status.

        Returns:
            Dict with export status, path, and metadata
        """
        return {
            "status": self.export_status,
            "path": str(self.export_path) if self.export_path else None,
            "enabled": self.enable_auto_export,
            "vault_configured": self.obsidian_vault is not None
        }

    def load_session(self, session_file: str):
        """Load a previous session from JSON file"""
        path = Path(session_file)
        if not path.exists():
            logger.error(f"Session file not found: {session_file}")
            return

        with open(path, 'r') as f:
            session_data = json.load(f)

        # Load tasks into memory
        self.tasks = {
            task["task_id"]: task
            for task in session_data["tasks"]
        }

        logger.info(f"Loaded session with {len(self.tasks)} tasks from {session_file}")

    def cleanup_old_sessions(self):
        """
        Delete session files older than retention_days.
        """
        cutoff = datetime.now() - timedelta(days=self.retention_days)

        for session_file in self.log_dir.glob("session_*.json"):
            # Parse timestamp from filename
            try:
                timestamp_str = session_file.stem.split('_', 1)[1]
                file_time = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")

                if file_time < cutoff:
                    session_file.unlink()
                    logger.info(f"Deleted old session: {session_file}")

            except Exception as e:
                logger.error(f"Error processing {session_file}: {e}")

    def get_session_summary(self) -> Dict[str, Any]:
        """
        Get summary of current session.

        Returns:
            Summary with task counts, duration stats, etc.
        """
        completed_tasks = [t for t in self.tasks.values() if t["status"] == "completed"]
        durations = [t["duration_seconds"] for t in completed_tasks if t["duration_seconds"]]

        return {
            "session_id": self.session_id,
            "total_tasks": len(self.tasks),
            "completed_tasks": len(completed_tasks),
            "in_progress_tasks": len(self.tasks) - len(completed_tasks),
            "avg_duration_seconds": sum(durations) / len(durations) if durations else 0,
            "total_duration_seconds": sum(durations)
        }

    def get_performance_metrics(self, lookback_days: int = 7) -> Dict[str, Any]:
        """
        Calculate performance metrics for Intelligence agent analysis.

        Aggregates metrics from task logs including success rate, cost,
        latency, cache hit rate, and failed task details.

        Args:
            lookback_days: Number of days to look back (not yet implemented - uses current session)

        Returns:
            Dict with:
                - accuracy: Task success rate (0.0 to 1.0)
                - cost: Total token cost (if tracked in metadata)
                - latency: Average task duration in seconds
                - cache_hit_rate: Average cache hit rate (if tracked)
                - failed_tasks: List of failed task IDs
                - avg_cost: Average cost per task
                - total_tasks: Total number of tasks
                - total_tokens: Total tokens used (if tracked)
        """
        if not self.tasks:
            return {
                "accuracy": 1.0,
                "cost": 0.0,
                "latency": 0.0,
                "cache_hit_rate": 0.0,
                "failed_tasks": [],
                "avg_cost": 0.0,
                "total_tasks": 0,
                "total_tokens": 0
            }

        # Separate completed and failed tasks
        completed_tasks = [t for t in self.tasks.values() if t["status"] == "completed"]
        failed_tasks = [t for t in self.tasks.values() if t["status"] == "failed"]
        total_tasks = len(completed_tasks) + len(failed_tasks)

        # Calculate accuracy (success rate)
        accuracy = len(completed_tasks) / total_tasks if total_tasks > 0 else 1.0

        # Calculate latency (average duration)
        all_durations = []
        for task in self.tasks.values():
            if task.get("duration_seconds"):
                all_durations.append(task["duration_seconds"])

        latency = sum(all_durations) / len(all_durations) if all_durations else 0.0

        # Calculate cost (from metadata)
        total_cost = 0.0
        total_tokens = 0
        cache_hits = 0
        cache_misses = 0

        for task in self.tasks.values():
            metadata = task.get("metadata", {})

            # Aggregate cost
            if "cost" in metadata:
                total_cost += float(metadata["cost"])

            # Aggregate tokens
            if "tokens" in metadata:
                total_tokens += int(metadata["tokens"])
            elif "tokens_used" in metadata:
                total_tokens += int(metadata["tokens_used"])

            # Aggregate cache hits
            if "cache_hit" in metadata:
                if metadata["cache_hit"]:
                    cache_hits += 1
                else:
                    cache_misses += 1

        # Calculate averages
        avg_cost = total_cost / total_tasks if total_tasks > 0 else 0.0

        # Calculate cache hit rate
        total_cache_attempts = cache_hits + cache_misses
        cache_hit_rate = cache_hits / total_cache_attempts if total_cache_attempts > 0 else 0.0

        # Get failed task IDs
        failed_task_ids = [t["task_id"] for t in failed_tasks]

        return {
            "accuracy": accuracy,
            "cost": total_cost,
            "latency": latency,
            "cache_hit_rate": cache_hit_rate,
            "failed_tasks": failed_task_ids,
            "avg_cost": avg_cost,
            "total_tasks": total_tasks,
            "total_tokens": total_tokens,
            "num_completed": len(completed_tasks),
            "num_failed": len(failed_tasks)
        }

    def get_tasks(self) -> List[Dict[str, Any]]:
        """
        Get all tasks in current session.

        Returns:
            List of all task dictionaries
        """
        return list(self.tasks.values())

    def semantic_search(
        self,
        query: str,
        max_results: int = 5,
        min_score: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Search for similar tasks using semantic similarity.

        Uses embeddings to find tasks semantically similar to the query.

        Args:
            query: Search query
            max_results: Maximum number of results to return
            min_score: Minimum similarity score (0.0-1.0)

        Returns:
            List of tasks sorted by relevance (most relevant first)

        Example:
            >>> results = memory.semantic_search("implement authentication")
            >>> for task in results:
            ...     print(f"{task['task_description']} (score: {task['score']:.2f})")
        """
        try:
            from ..memory.embeddings import generate_embedding
            import numpy as np

            # Generate query embedding
            query_embedding = generate_embedding(query)

            # Calculate similarity for each task
            scored_tasks = []

            for task in self.tasks.values():
                # Create task text for embedding
                task_text = (
                    f"{task.get('task_description', '')} "
                    f"{str(task.get('inputs', ''))} "
                    f"{str(task.get('outputs', ''))}"
                )

                # Generate task embedding
                task_embedding = generate_embedding(task_text)

                # Calculate cosine similarity
                query_norm = np.linalg.norm(query_embedding)
                task_norm = np.linalg.norm(task_embedding)

                if query_norm > 0 and task_norm > 0:
                    similarity = np.dot(query_embedding, task_embedding) / (query_norm * task_norm)
                else:
                    similarity = 0.0

                # Only include if above minimum score
                if similarity >= min_score:
                    # Create a copy of task with score
                    task_with_score = task.copy()
                    task_with_score['score'] = float(similarity)
                    scored_tasks.append(task_with_score)

            # Sort by score (descending)
            scored_tasks.sort(key=lambda x: x['score'], reverse=True)

            # Return top results
            return scored_tasks[:max_results]

        except Exception as e:
            logger.error(f"Semantic search failed: {e}", exc_info=True)
            # Fallback to simple keyword search
            return self._keyword_search(query, max_results)

    def _keyword_search(
        self,
        query: str,
        max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Fallback keyword-based search if semantic search fails.

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            List of tasks sorted by keyword relevance
        """
        query_lower = query.lower()
        query_terms = query_lower.split()

        scored_tasks = []

        for task in self.tasks.values():
            task_text = (
                task.get("task_description", "") + " " +
                str(task.get("inputs", "")) + " " +
                str(task.get("outputs", ""))
            ).lower()

            # Count matching terms
            match_count = sum(1 for term in query_terms if term in task_text)
            relevance = match_count / len(query_terms) if query_terms else 0.0

            if relevance > 0.2:  # At least 20% of terms match
                task_with_score = task.copy()
                task_with_score['score'] = relevance
                scored_tasks.append(task_with_score)

        # Sort by score
        scored_tasks.sort(key=lambda x: x['score'], reverse=True)

        return scored_tasks[:max_results]


# Quick test
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("=" * 80)
    print("Short-Term Memory Test")
    print("=" * 80)
    print()

    # Create memory
    memory = ShortTermMemory(log_dir="./test_logs")

    # Simulate task tree: control task with 3 operational subtasks
    print("Logging task tree...")

    # Control task
    control_task_id = memory.start_task(
        agent_id="control_001",
        agent_type="control",
        task_description="Research the Viable System Model",
        inputs={"topic": "VSM"}
    )

    # Operational subtasks
    for i in range(3):
        subtask_id = memory.start_task(
            agent_id=f"operational_{i+1:03d}",
            agent_type="operational",
            task_description=f"Research subtask {i+1}",
            inputs={"subtopic": f"VSM aspect {i+1}"},
            parent_task_id=control_task_id
        )

        # Simulate work
        import time
        time.sleep(0.1)

        memory.end_task(
            task_id=subtask_id,
            outputs={"synthesis": f"Findings for aspect {i+1}"},
            metadata={"tokens_used": 1000 + i*100}
        )

    # Complete control task
    memory.end_task(
        task_id=control_task_id,
        outputs={"final_report": "Complete VSM report"},
        metadata={"total_tokens": 3300}
    )

    # Print summary
    print()
    print("Session Summary:")
    summary = memory.get_session_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")

    # Print task tree
    print()
    print("Task Tree:")
    tree = memory.get_task_tree(control_task_id)
    for task in tree:
        indent = "  " if task.get("parent_task_id") else ""
        print(f"{indent}{task['task_id']}: {task['task_description']} ({task['status']})")

    # Save session
    print()
    memory.save_session()
    print(f"✓ Saved to {memory.session_file}")

    print()
    print("=" * 80)
    print("✓ Short-term memory test complete!")
    print("=" * 80)
