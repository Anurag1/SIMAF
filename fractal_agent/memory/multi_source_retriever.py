"""
Multi-Source Retriever - Parallel Knowledge Retrieval

Queries ALL knowledge sources in parallel to gather context for agents:
- GraphRAG: Domain knowledge from past tasks (entities + relationships)
- ShortTermMemory: Recent similar tasks (semantic search)
- ObsidianVault: Human-curated knowledge (tag/text search)
- WebSearch: Current information (DuckDuckGo/SerpAPI)

Author: BMad
Date: 2025-10-22
"""

import logging
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from dataclasses import dataclass, field

# Type hints for dependencies
try:
    from fractal_agent.memory.long_term import GraphRAG
    from fractal_agent.memory.short_term import ShortTermMemory
    from fractal_agent.memory.obsidian_vault import ObsidianVault
except ImportError:
    # Allow imports to fail for testing
    GraphRAG = Any
    ShortTermMemory = Any
    ObsidianVault = Any

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """
    Result from a single source retrieval.

    Tracks what was found, how long it took, and any errors.
    """
    source: str  # "graphrag", "short_term", "obsidian", "web"
    success: bool
    data: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    retrieval_time: float = 0.0
    query_used: str = ""
    result_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "source": self.source,
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "retrieval_time": self.retrieval_time,
            "query_used": self.query_used,
            "result_count": self.result_count
        }


@dataclass
class MultiSourceResult:
    """
    Combined results from all sources.

    Aggregates individual retrieval results and provides summary metrics.
    """
    graphrag: RetrievalResult
    short_term: RetrievalResult
    obsidian: RetrievalResult
    web: RetrievalResult
    total_time: float = 0.0

    def get_all_results(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all results organized by source"""
        return {
            "graphrag": self.graphrag.data if self.graphrag.success else [],
            "short_term": self.short_term.data if self.short_term.success else [],
            "obsidian": self.obsidian.data if self.obsidian.success else [],
            "web": self.web.data if self.web.success else []
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics"""
        return {
            "total_results": (
                self.graphrag.result_count +
                self.short_term.result_count +
                self.obsidian.result_count +
                self.web.result_count
            ),
            "sources_succeeded": sum([
                self.graphrag.success,
                self.short_term.success,
                self.obsidian.success,
                self.web.success
            ]),
            "total_time": self.total_time,
            "by_source": {
                "graphrag": self.graphrag.result_count,
                "short_term": self.short_term.result_count,
                "obsidian": self.obsidian.result_count,
                "web": self.web.result_count
            }
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "graphrag": self.graphrag.to_dict(),
            "short_term": self.short_term.to_dict(),
            "obsidian": self.obsidian.to_dict(),
            "web": self.web.to_dict(),
            "total_time": self.total_time,
            "summary": self.get_summary()
        }


class MultiSourceRetriever:
    """
    Intelligent multi-source knowledge retrieval.

    Queries all available knowledge sources in parallel and aggregates results.
    Each source is queried independently with proper error handling.

    Usage:
        >>> retriever = MultiSourceRetriever(
        ...     graphrag=graphrag_instance,
        ...     short_term_memory=memory_instance,
        ...     obsidian_vault=vault_instance
        ... )
        >>> results = retriever.retrieve_all(
        ...     query="VSM System 2 coordination mechanisms",
        ...     max_results_per_source=5
        ... )
        >>> print(f"Found {results.get_summary()['total_results']} total results")
    """

    def __init__(
        self,
        graphrag: Optional[Any] = None,
        short_term_memory: Optional[Any] = None,
        obsidian_vault: Optional[Any] = None,
        enable_web_search: bool = False,
        max_workers: int = 4
    ):
        """
        Initialize multi-source retriever.

        Args:
            graphrag: GraphRAG instance for domain knowledge
            short_term_memory: ShortTermMemory for recent tasks
            obsidian_vault: ObsidianVault for human knowledge
            enable_web_search: Whether to query web (default False)
            max_workers: Max parallel workers (default 4)
        """
        self.graphrag = graphrag
        self.short_term_memory = short_term_memory
        self.obsidian_vault = obsidian_vault
        self.enable_web_search = enable_web_search
        self.max_workers = max_workers

        logger.info(f"MultiSourceRetriever initialized with {self._count_sources()} sources")

    def _count_sources(self) -> int:
        """Count how many sources are enabled"""
        count = 0
        if self.graphrag:
            count += 1
        if self.short_term_memory:
            count += 1
        if self.obsidian_vault:
            count += 1
        if self.enable_web_search:
            count += 1
        return count

    def retrieve_all(
        self,
        query: str,
        max_results_per_source: int = 5,
        timeout: float = 30.0,
        verbose: bool = False
    ) -> MultiSourceResult:
        """
        Query all sources in parallel.

        Args:
            query: Search query string
            max_results_per_source: Max results per source
            timeout: Timeout for entire operation (seconds)
            verbose: Whether to log detailed progress

        Returns:
            MultiSourceResult with all retrieval results
        """
        start_time = time.time()

        if verbose:
            logger.info(f"Starting parallel retrieval for query: '{query[:50]}...'")
            logger.info(f"Enabled sources: {self._count_sources()}")

        # Prepare retrieval tasks
        tasks = []

        if self.graphrag:
            tasks.append(("graphrag", self._query_graphrag))
        if self.short_term_memory:
            tasks.append(("short_term", self._query_short_term))
        if self.obsidian_vault:
            tasks.append(("obsidian", self._query_obsidian))
        if self.enable_web_search:
            tasks.append(("web", self._query_web))

        # Execute in parallel
        results = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_source = {
                executor.submit(
                    task_func,
                    query,
                    max_results_per_source,
                    verbose
                ): source_name
                for source_name, task_func in tasks
            }

            # Collect results as they complete
            for future in as_completed(future_to_source, timeout=timeout):
                source_name = future_to_source[future]
                try:
                    result = future.result()
                    results[source_name] = result

                    if verbose:
                        logger.info(
                            f"✓ {source_name}: {result.result_count} results "
                            f"in {result.retrieval_time:.2f}s"
                        )
                except Exception as e:
                    logger.error(f"✗ {source_name} failed: {e}")
                    results[source_name] = RetrievalResult(
                        source=source_name,
                        success=False,
                        error=str(e)
                    )

        # Ensure all sources have results (even if they failed)
        for source_name, _ in tasks:
            if source_name not in results:
                results[source_name] = RetrievalResult(
                    source=source_name,
                    success=False,
                    error="Timeout or not executed"
                )

        # Create empty results for disabled sources
        if "graphrag" not in results:
            results["graphrag"] = RetrievalResult(
                source="graphrag",
                success=False,
                error="Source not configured"
            )
        if "short_term" not in results:
            results["short_term"] = RetrievalResult(
                source="short_term",
                success=False,
                error="Source not configured"
            )
        if "obsidian" not in results:
            results["obsidian"] = RetrievalResult(
                source="obsidian",
                success=False,
                error="Source not configured"
            )
        if "web" not in results:
            results["web"] = RetrievalResult(
                source="web",
                success=False,
                error="Source not configured or disabled"
            )

        total_time = time.time() - start_time

        multi_result = MultiSourceResult(
            graphrag=results["graphrag"],
            short_term=results["short_term"],
            obsidian=results["obsidian"],
            web=results["web"],
            total_time=total_time
        )

        if verbose:
            summary = multi_result.get_summary()
            logger.info(
                f"Retrieval complete: {summary['total_results']} results from "
                f"{summary['sources_succeeded']}/{self._count_sources()} sources "
                f"in {total_time:.2f}s"
            )

        return multi_result

    def _query_graphrag(
        self,
        query: str,
        max_results: int,
        verbose: bool
    ) -> RetrievalResult:
        """
        Query GraphRAG for domain knowledge.

        Uses semantic search over entity/relationship graph.
        """
        start_time = time.time()

        try:
            if verbose:
                logger.info(f"Querying GraphRAG: '{query[:50]}...'")

            # Query GraphRAG using retrieve method
            results = self.graphrag.retrieve(
                query=query,
                max_results=max_results
            )

            # Format results
            formatted_results = []
            for item in results:
                formatted_results.append({
                    "type": item.get("type", "entity"),  # "entity" or "relationship"
                    "content": item.get("content", ""),
                    "properties": item.get("properties", {}),
                    "score": item.get("score", 0.0),
                    "source_id": item.get("id", ""),
                    "metadata": item.get("metadata", {})
                })

            retrieval_time = time.time() - start_time

            return RetrievalResult(
                source="graphrag",
                success=True,
                data=formatted_results,
                retrieval_time=retrieval_time,
                query_used=query,
                result_count=len(formatted_results)
            )

        except Exception as e:
            logger.error(f"GraphRAG query failed: {e}")
            return RetrievalResult(
                source="graphrag",
                success=False,
                error=str(e),
                retrieval_time=time.time() - start_time,
                query_used=query
            )

    def _query_short_term(
        self,
        query: str,
        max_results: int,
        verbose: bool
    ) -> RetrievalResult:
        """
        Query ShortTermMemory for recent similar tasks.

        Uses semantic search over task history with embeddings.
        """
        start_time = time.time()

        try:
            if verbose:
                logger.info(f"Querying ShortTermMemory: '{query[:50]}...'")

            # Use semantic search if available
            if hasattr(self.short_term_memory, 'semantic_search'):
                matched_tasks = self.short_term_memory.semantic_search(
                    query=query,
                    max_results=max_results,
                    min_score=0.3  # 30% similarity minimum
                )

                # Reformat to consistent structure
                formatted_tasks = []
                for task in matched_tasks:
                    formatted_tasks.append({
                        "task_id": task.get("task_id", ""),
                        "description": task.get("task_description", ""),
                        "agent_type": task.get("agent_type", ""),
                        "inputs": task.get("inputs", {}),
                        "outputs": task.get("outputs", {}),
                        "score": task.get("score", 0.0),
                        "timestamp": task.get("timestamp_start", "")
                    })
            else:
                # Fallback: simple keyword matching
                all_tasks = self.short_term_memory.get_tasks()
                query_lower = query.lower()
                query_terms = query_lower.split()

                matched_tasks = []
                for task in all_tasks:
                    task_text = (
                        task.get("task_description", "") + " " +
                        str(task.get("inputs", "")) + " " +
                        str(task.get("outputs", ""))
                    ).lower()

                    match_count = sum(1 for term in query_terms if term in task_text)
                    relevance = match_count / len(query_terms) if query_terms else 0.0

                    if relevance > 0.2:
                        matched_tasks.append({
                            "task_id": task.get("task_id", ""),
                            "description": task.get("task_description", ""),
                            "agent_type": task.get("agent_type", ""),
                            "inputs": task.get("inputs", {}),
                            "outputs": task.get("outputs", {}),
                            "score": relevance,
                            "timestamp": task.get("timestamp_start", "")
                        })

                matched_tasks.sort(key=lambda x: x["score"], reverse=True)
                formatted_tasks = matched_tasks[:max_results]

            retrieval_time = time.time() - start_time

            return RetrievalResult(
                source="short_term",
                success=True,
                data=formatted_tasks,
                retrieval_time=retrieval_time,
                query_used=query,
                result_count=len(formatted_tasks)
            )

        except Exception as e:
            logger.error(f"ShortTermMemory query failed: {e}")
            return RetrievalResult(
                source="short_term",
                success=False,
                error=str(e),
                retrieval_time=time.time() - start_time,
                query_used=query
            )

    def _query_obsidian(
        self,
        query: str,
        max_results: int,
        verbose: bool
    ) -> RetrievalResult:
        """
        Query ObsidianVault for human-curated knowledge.

        Uses text search over notes.
        """
        start_time = time.time()

        try:
            if verbose:
                logger.info(f"Querying ObsidianVault: '{query[:50]}...'")

            # Query vault using search_notes method
            if hasattr(self.obsidian_vault, 'search_notes'):
                results = self.obsidian_vault.search_notes(
                    query=query,
                    max_results=max_results
                )

                formatted_results = []
                for note in results:
                    formatted_results.append({
                        "note_name": note.get("name", ""),
                        "content": note.get("content", ""),
                        "tags": note.get("tags", []),
                        "path": note.get("path", ""),
                        "score": note.get("score", 0.0)
                    })
            else:
                # Fallback: return empty results if method doesn't exist yet
                formatted_results = []

            retrieval_time = time.time() - start_time

            return RetrievalResult(
                source="obsidian",
                success=True,
                data=formatted_results,
                retrieval_time=retrieval_time,
                query_used=query,
                result_count=len(formatted_results)
            )

        except Exception as e:
            logger.error(f"ObsidianVault query failed: {e}")
            return RetrievalResult(
                source="obsidian",
                success=False,
                error=str(e),
                retrieval_time=time.time() - start_time,
                query_used=query
            )

    def _query_web(
        self,
        query: str,
        max_results: int,
        verbose: bool
    ) -> RetrievalResult:
        """
        Query web for current information.

        Uses DuckDuckGo search (no API key needed).
        """
        start_time = time.time()

        try:
            if verbose:
                logger.info(f"Querying Web: '{query[:50]}...'")

            # Try to import DuckDuckGo search
            try:
                from duckduckgo_search import DDGS

                with DDGS() as ddgs:
                    results = list(ddgs.text(
                        query,
                        max_results=max_results
                    ))

                formatted_results = []
                for result in results:
                    formatted_results.append({
                        "title": result.get("title", ""),
                        "snippet": result.get("body", ""),
                        "url": result.get("href", ""),
                        "score": 1.0  # DuckDuckGo doesn't provide scores
                    })

            except ImportError:
                logger.warning("duckduckgo-search not installed, web search disabled")
                formatted_results = []

            retrieval_time = time.time() - start_time

            return RetrievalResult(
                source="web",
                success=True,
                data=formatted_results,
                retrieval_time=retrieval_time,
                query_used=query,
                result_count=len(formatted_results)
            )

        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return RetrievalResult(
                source="web",
                success=False,
                error=str(e),
                retrieval_time=time.time() - start_time,
                query_used=query
            )


# ============================================================================
# Helper Functions
# ============================================================================

def format_results_for_context(
    multi_result: MultiSourceResult,
    max_total_items: int = 20
) -> Dict[str, str]:
    """
    Format multi-source results into context sections.

    Args:
        multi_result: MultiSourceResult from retrieve_all
        max_total_items: Max total items to include

    Returns:
        Dict with keys: domain_knowledge, recent_examples, constraints, current_info
    """
    # Domain knowledge from GraphRAG
    domain_knowledge = []
    for item in multi_result.graphrag.data[:max_total_items // 4]:
        domain_knowledge.append(
            f"- {item.get('content', '')} (score: {item.get('score', 0.0):.2f})"
        )

    # Recent examples from ShortTermMemory
    recent_examples = []
    for item in multi_result.short_term.data[:max_total_items // 4]:
        recent_examples.append(
            f"- {item.get('description', '')} → {item.get('outputs', {})} "
            f"(score: {item.get('score', 0.0):.2f})"
        )

    # Constraints from Obsidian
    constraints = []
    for item in multi_result.obsidian.data[:max_total_items // 4]:
        constraints.append(
            f"- {item.get('note_name', '')}: {item.get('content', '')[:100]}..."
        )

    # Current info from web
    current_info = []
    for item in multi_result.web.data[:max_total_items // 4]:
        current_info.append(
            f"- {item.get('title', '')}: {item.get('snippet', '')[:100]}..."
        )

    return {
        "domain_knowledge": "\n".join(domain_knowledge) if domain_knowledge else "No domain knowledge found",
        "recent_examples": "\n".join(recent_examples) if recent_examples else "No recent examples found",
        "constraints": "\n".join(constraints) if constraints else "No constraints found",
        "current_info": "\n".join(current_info) if current_info else "No current information found"
    }


# ============================================================================
# Module Testing
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("MultiSourceRetriever Test")
    print("=" * 80)
    print()

    # This would require actual instances of GraphRAG, ShortTermMemory, etc.
    # For now, just test initialization

    retriever = MultiSourceRetriever(
        graphrag=None,
        short_term_memory=None,
        obsidian_vault=None,
        enable_web_search=False
    )

    print(f"✓ Retriever initialized with {retriever._count_sources()} sources")
    print()

    # Test empty retrieval
    results = retriever.retrieve_all(
        query="test query",
        max_results_per_source=5,
        verbose=True
    )

    print()
    print("Results:")
    print(f"  Total time: {results.total_time:.2f}s")
    print(f"  Summary: {results.get_summary()}")
    print()

    print("=" * 80)
    print("MultiSourceRetriever Test Complete")
    print("=" * 80)
