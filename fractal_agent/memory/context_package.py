"""
Context Package - Structured Context Bundle for Execution Agents

Defines the data structures for packaging context that gets passed to execution agents.
Each package includes not just the context content, but also metadata for attribution
tracking and continuous improvement.

Author: BMad
Date: 2025-10-22
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime
import json


@dataclass
class ContextPiece:
    """
    Individual piece of context with metadata for attribution tracking.

    Each piece can be tracked back to its source, allowing us to measure
    which context is actually used by the LLM and which is ignored.
    """

    content: str  # The actual context text
    source: str  # "graphrag", "recent_tasks", "obsidian", "web"
    source_id: str  # Specific entity ID, task ID, note name, or URL
    relevance_score: float  # 0.0-1.0 how relevant this is to the task
    piece_type: str  # "domain_knowledge", "example", "constraint", "current_info"
    used: bool = False  # Will be set True by attribution analysis
    evidence: Optional[str] = None  # Where in output this was referenced

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    def __str__(self) -> str:
        """String representation"""
        return f"[{self.source}:{self.piece_type}] {self.content[:50]}... (relevance={self.relevance_score:.2f})"


@dataclass
class ContextPackage:
    """
    Perfect context bundle for execution agents.

    This package contains ALL the context an execution agent needs, carefully
    selected and formatted. It also includes metadata for attribution tracking
    and continuous improvement.

    Usage:
        >>> package = ContextPackage(
        ...     domain_knowledge="VSM is a cybernetic model...",
        ...     recent_examples=[{"task": "...", "outcome": "..."}],
        ...     constraints="Follow Python PEP 8 standards",
        ...     current_info="Python 3.12 was released in 2023",
        ...     confidence=0.85,
        ...     sources_used=["graphrag", "recent_tasks", "web"]
        ... )
        >>> formatted = package.format_for_agent("research")
        >>> # Pass formatted to agent's DSPy signature
    """

    # ========================================================================
    # Core Content (what the agent actually gets)
    # ========================================================================

    domain_knowledge: str = ""  # From GraphRAG - domain concepts and facts
    recent_examples: List[Dict[str, Any]] = field(default_factory=list)  # From short-term memory
    constraints: str = ""  # From Obsidian - guidelines and best practices
    current_info: str = ""  # From web search - up-to-date information

    # ========================================================================
    # Metadata (for tracking and improvement)
    # ========================================================================

    confidence: float = 0.0  # 0.0-1.0 confidence this context is optimal
    sources_used: List[str] = field(default_factory=list)  # Which sources contributed
    total_tokens: int = 0  # Approximate size of context
    preparation_time: float = 0.0  # How long context preparation took
    iterations: int = 0  # How many research loops were needed

    # For evaluation and learning
    requirements: str = ""  # What was identified as needed
    evaluation: str = ""  # Why this context was deemed sufficient
    preparation_reasoning: str = ""  # Why specific pieces were included/excluded

    # For attribution tracking (critical for validation)
    context_pieces: List[ContextPiece] = field(default_factory=list)

    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def format_for_agent(self, agent_type: str) -> Dict[str, str]:
        """
        Format context appropriately for target agent's DSPy signatures.

        Different agents have different signature parameters, so we need
        to format the context to match what each agent expects.

        Args:
            agent_type: "research", "developer", "coordination", etc.

        Returns:
            Dict with keys matching agent's signature input fields
        """
        if agent_type == "research":
            return {
                "domain_knowledge": self.domain_knowledge,
                "recent_examples": self._format_examples_for_research(),
                "constraints": self.constraints,
                "current_info": self.current_info
            }
        elif agent_type == "developer":
            return {
                "domain_knowledge": self.domain_knowledge,
                "recent_examples": self._format_examples_for_developer(),
                "constraints": self.constraints,
                "current_info": self.current_info
            }
        elif agent_type == "coordination":
            return {
                "domain_knowledge": self.domain_knowledge,
                "recent_examples": self._format_examples_for_coordination(),
                "constraints": self.constraints,
                "current_info": self.current_info
            }
        else:
            # Default formatting
            return {
                "domain_knowledge": self.domain_knowledge,
                "recent_examples": self._format_examples_generic(),
                "constraints": self.constraints,
                "current_info": self.current_info
            }

    def get_attribution_map(self) -> Dict[str, str]:
        """
        Map each context piece to its source for attribution tracking.

        Returns:
            Dict mapping piece content (first 50 chars) to source info
        """
        return {
            piece.content[:50]: f"{piece.source}:{piece.source_id}"
            for piece in self.context_pieces
        }

    def get_section(self, section: str) -> str:
        """
        Get specific context section.

        Args:
            section: "domain_knowledge", "recent_examples", "constraints", "current_info"

        Returns:
            The requested section content
        """
        return getattr(self, section, "")

    def count_tokens(self) -> int:
        """
        Estimate total tokens in this context package.

        Uses rough approximation: 1 token ≈ 4 characters

        Returns:
            Approximate token count
        """
        total_chars = 0

        # Count domain knowledge
        total_chars += len(self.domain_knowledge)

        # Count examples
        for example in self.recent_examples:
            total_chars += len(str(example))

        # Count constraints
        total_chars += len(self.constraints)

        # Count current info
        total_chars += len(self.current_info)

        # Rough token estimate (1 token ≈ 4 chars)
        tokens = total_chars // 4

        self.total_tokens = tokens
        return tokens

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Returns:
            Dict representation of this context package
        """
        return {
            "domain_knowledge": self.domain_knowledge,
            "recent_examples": self.recent_examples,
            "constraints": self.constraints,
            "current_info": self.current_info,
            "confidence": self.confidence,
            "sources_used": self.sources_used,
            "total_tokens": self.total_tokens,
            "preparation_time": self.preparation_time,
            "iterations": self.iterations,
            "requirements": self.requirements,
            "evaluation": self.evaluation,
            "preparation_reasoning": self.preparation_reasoning,
            "context_pieces": [piece.to_dict() for piece in self.context_pieces],
            "created_at": self.created_at
        }

    def to_json(self) -> str:
        """
        Serialize to JSON string.

        Returns:
            JSON representation
        """
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContextPackage':
        """
        Deserialize from dictionary.

        Args:
            data: Dict representation

        Returns:
            ContextPackage instance
        """
        # Convert context_pieces back to ContextPiece objects
        pieces = data.get("context_pieces", [])
        if pieces and isinstance(pieces[0], dict):
            pieces = [ContextPiece(**p) for p in pieces]

        return cls(
            domain_knowledge=data.get("domain_knowledge", ""),
            recent_examples=data.get("recent_examples", []),
            constraints=data.get("constraints", ""),
            current_info=data.get("current_info", ""),
            confidence=data.get("confidence", 0.0),
            sources_used=data.get("sources_used", []),
            total_tokens=data.get("total_tokens", 0),
            preparation_time=data.get("preparation_time", 0.0),
            iterations=data.get("iterations", 0),
            requirements=data.get("requirements", ""),
            evaluation=data.get("evaluation", ""),
            preparation_reasoning=data.get("preparation_reasoning", ""),
            context_pieces=pieces,
            created_at=data.get("created_at", datetime.now().isoformat())
        )

    @classmethod
    def from_json(cls, json_str: str) -> 'ContextPackage':
        """
        Deserialize from JSON string.

        Args:
            json_str: JSON representation

        Returns:
            ContextPackage instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    def to_summary(self) -> str:
        """
        Create human-readable summary of this context package.

        Returns:
            Multi-line summary string
        """
        lines = []
        lines.append("=" * 80)
        lines.append("CONTEXT PACKAGE SUMMARY")
        lines.append("=" * 80)
        lines.append(f"Confidence: {self.confidence:.2f}")
        lines.append(f"Sources: {', '.join(self.sources_used)}")
        lines.append(f"Tokens: ~{self.total_tokens}")
        lines.append(f"Preparation time: {self.preparation_time:.2f}s")
        lines.append(f"Iterations: {self.iterations}")
        lines.append("")

        if self.domain_knowledge:
            lines.append(f"Domain Knowledge: {len(self.domain_knowledge)} chars")
        if self.recent_examples:
            lines.append(f"Recent Examples: {len(self.recent_examples)} items")
        if self.constraints:
            lines.append(f"Constraints: {len(self.constraints)} chars")
        if self.current_info:
            lines.append(f"Current Info: {len(self.current_info)} chars")

        lines.append("")
        lines.append(f"Context Pieces: {len(self.context_pieces)} total")

        if self.context_pieces:
            by_source = {}
            for piece in self.context_pieces:
                by_source[piece.source] = by_source.get(piece.source, 0) + 1

            for source, count in by_source.items():
                lines.append(f"  {source}: {count} pieces")

        lines.append("=" * 80)

        return "\n".join(lines)

    # ========================================================================
    # Private formatting helpers
    # ========================================================================

    def _format_examples_for_research(self) -> str:
        """Format examples for research agent"""
        if not self.recent_examples:
            return "No recent examples available"

        formatted = []
        for i, example in enumerate(self.recent_examples[:5], 1):  # Max 5 examples
            formatted.append(f"{i}. {example.get('task', 'Unknown task')}")
            formatted.append(f"   Outcome: {example.get('outcome', 'Unknown outcome')}")

        return "\n".join(formatted)

    def _format_examples_for_developer(self) -> str:
        """Format examples for developer agent"""
        if not self.recent_examples:
            return "No recent code examples available"

        formatted = []
        for i, example in enumerate(self.recent_examples[:3], 1):  # Max 3 for code
            formatted.append(f"{i}. {example.get('task', 'Unknown task')}")
            if "code" in example:
                formatted.append(f"```python\n{example['code']}\n```")

        return "\n".join(formatted)

    def _format_examples_for_coordination(self) -> str:
        """Format examples for coordination agent"""
        if not self.recent_examples:
            return "No recent coordination examples available"

        formatted = []
        for i, example in enumerate(self.recent_examples[:5], 1):
            formatted.append(f"{i}. {example.get('task', 'Unknown task')}")
            formatted.append(f"   Agents: {example.get('agents', 'Unknown')}")
            formatted.append(f"   Result: {example.get('result', 'Unknown')}")

        return "\n".join(formatted)

    def _format_examples_generic(self) -> str:
        """Generic example formatting"""
        if not self.recent_examples:
            return "No recent examples available"

        return "\n".join(
            f"{i}. {str(example)[:100]}"
            for i, example in enumerate(self.recent_examples[:5], 1)
        )

    def __str__(self) -> str:
        """String representation"""
        return f"ContextPackage(confidence={self.confidence:.2f}, sources={len(self.sources_used)}, tokens={self.total_tokens})"

    def __repr__(self) -> str:
        """Detailed representation"""
        return self.to_summary()


# ============================================================================
# Helper functions
# ============================================================================

def create_context_piece(
    content: str,
    source: str,
    source_id: str,
    relevance_score: float,
    piece_type: str
) -> ContextPiece:
    """
    Helper to create a ContextPiece.

    Args:
        content: The context text
        source: Source name ("graphrag", "recent_tasks", etc.)
        source_id: Specific ID within source
        relevance_score: 0.0-1.0 relevance
        piece_type: Type of context

    Returns:
        ContextPiece instance
    """
    return ContextPiece(
        content=content,
        source=source,
        source_id=source_id,
        relevance_score=relevance_score,
        piece_type=piece_type
    )


def merge_context_packages(packages: List[ContextPackage]) -> ContextPackage:
    """
    Merge multiple context packages into one.

    Useful when combining context from multiple iterations or sources.

    Args:
        packages: List of ContextPackage instances

    Returns:
        Merged ContextPackage
    """
    if not packages:
        return ContextPackage()

    if len(packages) == 1:
        return packages[0]

    # Merge all content
    merged = ContextPackage()

    for pkg in packages:
        # Concatenate knowledge
        if pkg.domain_knowledge:
            merged.domain_knowledge += "\n\n" + pkg.domain_knowledge

        # Merge examples
        merged.recent_examples.extend(pkg.recent_examples)

        # Concatenate constraints
        if pkg.constraints:
            merged.constraints += "\n\n" + pkg.constraints

        # Concatenate current info
        if pkg.current_info:
            merged.current_info += "\n\n" + pkg.current_info

        # Merge metadata
        merged.sources_used.extend(pkg.sources_used)
        merged.context_pieces.extend(pkg.context_pieces)
        merged.iterations = max(merged.iterations, pkg.iterations)
        merged.preparation_time += pkg.preparation_time

    # Deduplicate sources
    merged.sources_used = list(set(merged.sources_used))

    # Average confidence
    merged.confidence = sum(p.confidence for p in packages) / len(packages)

    # Recalculate tokens
    merged.count_tokens()

    return merged


# ============================================================================
# Module testing
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("ContextPackage Test")
    print("=" * 80)
    print()

    # Create a sample package
    package = ContextPackage(
        domain_knowledge="VSM (Viable System Model) is a cybernetic organizational model...",
        recent_examples=[
            {"task": "Research VSM System 2", "outcome": "Completed successfully"},
            {"task": "Implement coordination agent", "outcome": "Passed all tests"}
        ],
        constraints="Follow Python PEP 8 style guide. Use type hints.",
        current_info="Python 3.12 was released in October 2023",
        confidence=0.85,
        sources_used=["graphrag", "recent_tasks", "obsidian", "web"],
        iterations=2,
        preparation_time=3.5
    )

    # Add some context pieces
    package.context_pieces = [
        create_context_piece(
            "VSM is a model of organizational structure",
            "graphrag",
            "entity_vsm_001",
            0.92,
            "domain_knowledge"
        ),
        create_context_piece(
            "Previous VSM task took 2.3s",
            "recent_tasks",
            "task_123",
            0.75,
            "example"
        )
    ]

    # Calculate tokens
    package.count_tokens()

    # Print summary
    print(package.to_summary())
    print()

    # Test formatting
    print("Formatted for research agent:")
    print("-" * 80)
    formatted = package.format_for_agent("research")
    for key, value in formatted.items():
        print(f"{key}:")
        print(f"  {str(value)[:100]}...")
        print()

    print("=" * 80)
    print("ContextPackage Test Complete")
    print("=" * 80)
