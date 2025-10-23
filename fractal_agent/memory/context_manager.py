"""
Context Manager - Phase 4

Manages context loading and token budget within 200K token limit.
Implements tiered context loading, token tracking, and truncation strategies.

Features:
- Tiered context prioritization (task -> history -> memory -> docs)
- Token budget management with 200K limit
- Multiple truncation strategies (head, tail, middle, sliding window, summarization)
- Integration with GraphRAG, ObsidianVault, and ShortTermMemory
- Automatic overflow handling

Author: BMad
Date: 2025-10-19
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import tiktoken

from .long_term import GraphRAG, DocumentStore
from .obsidian_vault import ObsidianVault
from .short_term import ShortTermMemory

logger = logging.getLogger(__name__)


class ContextTier(Enum):
    """
    Context priority tiers.

    Lower values = higher priority (always loaded first)
    """
    CURRENT_TASK = 1      # Current prompt/task (highest priority)
    RECENT_HISTORY = 2    # Recent conversation/session
    LONG_TERM_MEMORY = 3  # GraphRAG knowledge
    VAULT_DOCS = 4        # ObsidianVault documentation
    SYSTEM_PROMPTS = 5    # System configuration/prompts


class TruncationStrategy(Enum):
    """
    Strategies for truncating context when approaching token limit.
    """
    HEAD = "head"                    # Keep most recent content
    TAIL = "tail"                    # Keep oldest content
    MIDDLE = "middle"                # Keep beginning and end, drop middle
    SLIDING_WINDOW = "sliding_window"  # Keep most relevant chunks
    SUMMARIZE = "summarize"          # Compress older content via LLM


@dataclass
class ContextBlock:
    """
    A block of context with metadata.

    Attributes:
        content: Text content
        tier: Priority tier
        tokens: Token count
        metadata: Additional metadata (source, timestamp, etc.)
    """
    content: str
    tier: ContextTier
    tokens: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TokenBudget:
    """
    Token budget allocation per tier.

    Attributes:
        total_budget: Maximum total tokens (200K default)
        tier_budgets: Budget allocation per tier
        reserved: Reserved tokens for safety margin
    """
    total_budget: int = 200000
    tier_budgets: Dict[ContextTier, int] = field(default_factory=dict)
    reserved: int = 10000  # Safety margin

    def __post_init__(self):
        """Set default tier budgets if not specified."""
        if not self.tier_budgets:
            available = self.total_budget - self.reserved
            self.tier_budgets = {
                ContextTier.CURRENT_TASK: 50000,      # 25%
                ContextTier.RECENT_HISTORY: 70000,    # 35%
                ContextTier.LONG_TERM_MEMORY: 40000,  # 20%
                ContextTier.VAULT_DOCS: 30000,        # 15%
                ContextTier.SYSTEM_PROMPTS: 10000     # 5%
            }

    def get_budget(self, tier: ContextTier) -> int:
        """Get budget for specific tier."""
        return self.tier_budgets.get(tier, 0)

    def set_budget(self, tier: ContextTier, tokens: int):
        """Set budget for specific tier."""
        self.tier_budgets[tier] = tokens


class TokenCounter:
    """
    Accurate token counting using tiktoken.

    Uses cl100k_base encoding (GPT-4, Claude-compatible).
    """

    def __init__(self, encoding_name: str = "cl100k_base"):
        """
        Initialize token counter.

        Args:
            encoding_name: Tiktoken encoding name (default: cl100k_base)
        """
        try:
            self.encoding = tiktoken.get_encoding(encoding_name)
        except Exception as e:
            logger.warning(f"Failed to load tiktoken encoding: {e}, using fallback")
            self.encoding = None

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Text to count

        Returns:
            Token count
        """
        if not text:
            return 0

        if self.encoding:
            return len(self.encoding.encode(text))
        else:
            # Fallback: rough approximation (1 token ≈ 4 chars)
            return len(text) // 4

    def count_tokens_batch(self, texts: List[str]) -> List[int]:
        """
        Count tokens for multiple texts.

        Args:
            texts: List of texts

        Returns:
            List of token counts
        """
        return [self.count_tokens(text) for text in texts]


class ContextManager:
    """
    Manages context loading with tiered prioritization and token budget.

    Features:
    - Tiered context loading (task -> history -> memory -> docs)
    - Token budget enforcement (200K limit)
    - Multiple truncation strategies
    - Automatic overflow handling
    - Integration with memory systems

    Usage:
        >>> manager = ContextManager(
        ...     graphrag=graphrag,
        ...     vault=vault,
        ...     memory=memory
        ... )
        >>>
        >>> # Add context blocks
        >>> manager.add_context(
        ...     content="Current task description",
        ...     tier=ContextTier.CURRENT_TASK
        ... )
        >>>
        >>> # Build final context
        >>> context = manager.build_context(
        ...     strategy=TruncationStrategy.SLIDING_WINDOW
        ... )
    """

    def __init__(
        self,
        graphrag: Optional[GraphRAG] = None,
        vault: Optional[ObsidianVault] = None,
        memory: Optional[ShortTermMemory] = None,
        budget: Optional[TokenBudget] = None,
        token_counter: Optional[TokenCounter] = None
    ):
        """
        Initialize ContextManager.

        Args:
            graphrag: GraphRAG instance for long-term memory
            vault: ObsidianVault instance for documentation
            memory: ShortTermMemory instance for session history
            budget: Token budget configuration
            token_counter: Token counter instance
        """
        self.graphrag = graphrag
        self.vault = vault
        self.memory = memory

        self.budget = budget or TokenBudget()
        self.token_counter = token_counter or TokenCounter()

        # Context blocks organized by tier
        self.context_blocks: Dict[ContextTier, List[ContextBlock]] = {
            tier: [] for tier in ContextTier
        }

        # Token usage tracking
        self.token_usage: Dict[ContextTier, int] = {
            tier: 0 for tier in ContextTier
        }

        logger.info(
            f"Initialized ContextManager: "
            f"budget={self.budget.total_budget} tokens, "
            f"reserved={self.budget.reserved} tokens"
        )

    def add_context(
        self,
        content: str,
        tier: ContextTier,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Add context block to manager.

        Args:
            content: Text content to add
            tier: Priority tier for this content
            metadata: Additional metadata

        Returns:
            Token count for added content
        """
        if not content or not content.strip():
            return 0

        # Count tokens
        tokens = self.token_counter.count_tokens(content)

        # Create context block
        block = ContextBlock(
            content=content,
            tier=tier,
            tokens=tokens,
            metadata=metadata or {}
        )

        # Add to appropriate tier
        self.context_blocks[tier].append(block)
        self.token_usage[tier] += tokens

        logger.debug(
            f"Added context to {tier.name}: {tokens} tokens "
            f"(tier total: {self.token_usage[tier]})"
        )

        return tokens

    def add_task_context(self, task_description: str, inputs: Dict[str, Any]):
        """
        Add current task context (Tier 1).

        Args:
            task_description: Description of current task
            inputs: Task input parameters
        """
        content = f"# Current Task\n\n{task_description}\n\n"

        if inputs:
            content += "## Inputs\n\n"
            for key, value in inputs.items():
                content += f"- {key}: {value}\n"

        self.add_context(
            content=content,
            tier=ContextTier.CURRENT_TASK,
            metadata={"source": "task"}
        )

    def add_history_context(
        self,
        max_tasks: int = 10,
        include_outputs: bool = True
    ):
        """
        Add recent session history from ShortTermMemory (Tier 2).

        Args:
            max_tasks: Maximum number of recent tasks to include
            include_outputs: Include task outputs in context
        """
        if not self.memory:
            logger.warning("No ShortTermMemory instance available")
            return

        # Get recent tasks
        tasks = list(self.memory.tasks.values())
        tasks = sorted(
            tasks,
            key=lambda t: t.get("timestamp_start", ""),
            reverse=True
        )[:max_tasks]

        if not tasks:
            return

        # Build history content
        content_parts = ["# Recent Session History\n"]

        for task in reversed(tasks):  # Chronological order
            content_parts.append(
                f"\n## Task: {task['task_description']}\n"
                f"- Agent: {task['agent_id']} ({task['agent_type']})\n"
                f"- Status: {task['status']}\n"
            )

            if task.get('inputs'):
                content_parts.append(f"- Inputs: {task['inputs']}\n")

            if include_outputs and task.get('outputs'):
                content_parts.append(f"- Outputs: {task['outputs']}\n")

        content = "\n".join(content_parts)

        self.add_context(
            content=content,
            tier=ContextTier.RECENT_HISTORY,
            metadata={"source": "memory", "num_tasks": len(tasks)}
        )

    def add_graphrag_context(
        self,
        query: str,
        query_embedding: List[float],
        max_results: int = 5
    ):
        """
        Add relevant knowledge from GraphRAG (Tier 3).

        Args:
            query: Query for retrieving relevant knowledge
            query_embedding: Embedding vector for query
            max_results: Maximum number of results to retrieve
        """
        if not self.graphrag:
            logger.warning("No GraphRAG instance available")
            return

        # Retrieve relevant knowledge
        results = self.graphrag.retrieve(
            query=query,
            query_embedding=query_embedding,
            max_results=max_results,
            only_valid=True
        )

        if not results:
            logger.debug("No GraphRAG results found")
            return

        # Build knowledge content
        content_parts = [f"# Relevant Knowledge ({len(results)} items)\n"]

        for i, result in enumerate(results, 1):
            content_parts.append(
                f"\n## [{i}] ({result['entity']})-[{result['relationship']}]->({result['target']})\n"
                f"- Valid since: {result['t_valid']}\n"
            )

            if result.get('metadata'):
                content_parts.append(f"- Metadata: {result['metadata']}\n")

        content = "\n".join(content_parts)

        self.add_context(
            content=content,
            tier=ContextTier.LONG_TERM_MEMORY,
            metadata={"source": "graphrag", "num_results": len(results)}
        )

    def add_vault_context(
        self,
        query: str,
        max_chunks: int = 5,
        tag_filter: Optional[List[str]] = None
    ):
        """
        Add relevant documentation from ObsidianVault (Tier 4).

        Args:
            query: Query for retrieving relevant docs
            max_chunks: Maximum number of document chunks
            tag_filter: Optional tag filter for vault search
        """
        if not self.vault:
            logger.warning("No ObsidianVault instance available")
            return

        # Query vault
        vault_context = self.vault.query(
            query_text=query,
            max_results=max_chunks,
            tag_filter=tag_filter
        )

        if not vault_context:
            logger.debug("No vault context found")
            return

        self.add_context(
            content=vault_context,
            tier=ContextTier.VAULT_DOCS,
            metadata={"source": "vault", "max_chunks": max_chunks}
        )

    def add_system_prompt(self, system_prompt: str):
        """
        Add system prompt (Tier 5).

        Args:
            system_prompt: System prompt text
        """
        self.add_context(
            content=system_prompt,
            tier=ContextTier.SYSTEM_PROMPTS,
            metadata={"source": "system"}
        )

    def get_total_tokens(self) -> int:
        """Get total token count across all tiers."""
        return sum(self.token_usage.values())

    def get_tier_tokens(self, tier: ContextTier) -> int:
        """Get token count for specific tier."""
        return self.token_usage[tier]

    def is_over_budget(self) -> bool:
        """Check if total tokens exceed budget."""
        return self.get_total_tokens() > (self.budget.total_budget - self.budget.reserved)

    def truncate_tier(
        self,
        tier: ContextTier,
        target_tokens: int,
        strategy: TruncationStrategy = TruncationStrategy.HEAD
    ):
        """
        Truncate a specific tier to target token count.

        Args:
            tier: Tier to truncate
            target_tokens: Target token count after truncation
            strategy: Truncation strategy to use
        """
        blocks = self.context_blocks[tier]

        if not blocks:
            return

        current_tokens = self.token_usage[tier]

        if current_tokens <= target_tokens:
            logger.debug(f"Tier {tier.name} within budget, no truncation needed")
            return

        logger.info(
            f"Truncating {tier.name}: {current_tokens} -> {target_tokens} tokens "
            f"(strategy: {strategy.name})"
        )

        if strategy == TruncationStrategy.HEAD:
            # Keep most recent content
            self._truncate_head(tier, target_tokens)
        elif strategy == TruncationStrategy.TAIL:
            # Keep oldest content
            self._truncate_tail(tier, target_tokens)
        elif strategy == TruncationStrategy.MIDDLE:
            # Keep beginning and end
            self._truncate_middle(tier, target_tokens)
        elif strategy == TruncationStrategy.SLIDING_WINDOW:
            # Keep most relevant chunks
            self._truncate_sliding_window(tier, target_tokens)
        elif strategy == TruncationStrategy.SUMMARIZE:
            # Compress via summarization (not implemented yet)
            logger.warning("SUMMARIZE strategy not yet implemented, using HEAD")
            self._truncate_head(tier, target_tokens)

    def _truncate_head(self, tier: ContextTier, target_tokens: int):
        """Keep most recent content (truncate from beginning)."""
        blocks = self.context_blocks[tier]

        # Keep blocks from end until we hit target
        kept_blocks = []
        total_tokens = 0

        for block in reversed(blocks):
            if total_tokens + block.tokens <= target_tokens:
                kept_blocks.insert(0, block)
                total_tokens += block.tokens
            else:
                break

        self.context_blocks[tier] = kept_blocks
        self.token_usage[tier] = total_tokens

        logger.debug(f"HEAD truncation: kept {len(kept_blocks)}/{len(blocks)} blocks")

    def _truncate_tail(self, tier: ContextTier, target_tokens: int):
        """Keep oldest content (truncate from end)."""
        blocks = self.context_blocks[tier]

        # Keep blocks from beginning until we hit target
        kept_blocks = []
        total_tokens = 0

        for block in blocks:
            if total_tokens + block.tokens <= target_tokens:
                kept_blocks.append(block)
                total_tokens += block.tokens
            else:
                break

        self.context_blocks[tier] = kept_blocks
        self.token_usage[tier] = total_tokens

        logger.debug(f"TAIL truncation: kept {len(kept_blocks)}/{len(blocks)} blocks")

    def _truncate_middle(self, tier: ContextTier, target_tokens: int):
        """Keep beginning and end, drop middle."""
        blocks = self.context_blocks[tier]

        if len(blocks) <= 2:
            # Not enough blocks to truncate middle
            self._truncate_head(tier, target_tokens)
            return

        # Allocate 50% to beginning, 50% to end
        head_budget = target_tokens // 2
        tail_budget = target_tokens - head_budget

        # Keep from beginning
        head_blocks = []
        head_tokens = 0
        for block in blocks:
            if head_tokens + block.tokens <= head_budget:
                head_blocks.append(block)
                head_tokens += block.tokens
            else:
                break

        # Keep from end
        tail_blocks = []
        tail_tokens = 0
        for block in reversed(blocks):
            if tail_tokens + block.tokens <= tail_budget:
                tail_blocks.insert(0, block)
                tail_tokens += block.tokens
            else:
                break

        # Combine (avoid duplicates)
        kept_blocks = head_blocks
        for block in tail_blocks:
            if block not in kept_blocks:
                kept_blocks.append(block)

        total_tokens = head_tokens + tail_tokens

        self.context_blocks[tier] = kept_blocks
        self.token_usage[tier] = total_tokens

        logger.debug(
            f"MIDDLE truncation: kept {len(kept_blocks)}/{len(blocks)} blocks "
            f"(head={len(head_blocks)}, tail={len(tail_blocks)})"
        )

    def _truncate_sliding_window(self, tier: ContextTier, target_tokens: int):
        """Keep most relevant chunks (priority to recent content)."""
        # For now, use HEAD strategy (can be enhanced with relevance scoring)
        self._truncate_head(tier, target_tokens)

    def load_context(
        self,
        vsm_tier: str,
        task_description: str,
        graphrag_query: Optional[str] = None,
        max_graphrag_results: int = 10
    ) -> str:
        """
        Load context with VSM tier-specific token budgets.

        Implements 4-tier loading system aligned with VSM architecture:
        - System1 (Research): 50K context - Detailed research and analysis
        - System2 (Coordination): 15K context - Focused coordination and routing
        - System3 (Intelligence): 75K context - Maximum context for optimization
        - System4 (Strategic): 60K context - Strategic planning and policy

        Workflow:
        1. Allocate token budget based on VSM tier
        2. Retrieve relevant knowledge from GraphRAG
        3. Load recent session history
        4. Load vault documentation if space permits
        5. Build final context respecting budget

        Args:
            vsm_tier: VSM system tier (System1, System2, System3, or System4)
            task_description: Current task description (always loaded)
            graphrag_query: Query for GraphRAG retrieval (defaults to task_description)
            max_graphrag_results: Maximum GraphRAG results to retrieve

        Returns:
            Assembled context string ready for LLM

        Example:
            >>> context = manager.load_context(
            ...     vsm_tier="System1",
            ...     task_description="Research VSM System 1 patterns",
            ...     graphrag_query="VSM operational tier patterns"
            ... )
            >>> print(f"Context length: {len(context)} chars")
        """
        # Define VSM tier-specific token budgets
        vsm_budgets = {
            "System1": 50000,   # Research - Detailed analysis
            "System2": 15000,   # Coordination - Focused routing
            "System3": 75000,   # Intelligence - Maximum context
            "System4": 60000    # Strategic - Planning
        }

        # Get budget for this tier (default to 50K if unknown tier)
        context_budget = vsm_budgets.get(vsm_tier, 50000)

        logger.info(
            f"Loading context for {vsm_tier} with {context_budget} token budget"
        )

        # Clear existing context
        self.clear()

        # 1. Add current task (highest priority, always loaded)
        task_tokens = self.add_context(
            content=f"# Current Task\n\n{task_description}",
            tier=ContextTier.CURRENT_TASK,
            metadata={"vsm_tier": vsm_tier}
        )

        remaining_budget = context_budget - task_tokens

        # 2. Retrieve and add GraphRAG knowledge if available
        if self.graphrag and remaining_budget > 5000:
            query = graphrag_query or task_description

            try:
                from .embeddings import generate_embedding
                query_embedding = generate_embedding(query)

                results = self.graphrag.retrieve(
                    query=query,
                    query_embedding=query_embedding,
                    max_results=max_graphrag_results,
                    only_valid=True
                )

                if results:
                    # Format GraphRAG results
                    knowledge_content = "# Relevant Past Knowledge\n\n"
                    for i, item in enumerate(results, 1):
                        knowledge_content += f"{i}. {item['entity']} → {item['relationship']} → {item['target']}\n"
                        if 'metadata' in item and 'description' in item['metadata']:
                            knowledge_content += f"   {item['metadata']['description']}\n"
                        knowledge_content += "\n"

                    knowledge_tokens = self.add_context(
                        content=knowledge_content,
                        tier=ContextTier.LONG_TERM_MEMORY,
                        metadata={"source": "graphrag", "num_results": len(results)}
                    )

                    remaining_budget -= knowledge_tokens
                    logger.info(f"Loaded {len(results)} GraphRAG items ({knowledge_tokens} tokens)")

            except Exception as e:
                logger.error(f"Failed to load GraphRAG context: {e}")

        # 3. Load recent session history if available and budget permits
        if self.memory and remaining_budget > 10000:
            try:
                # Get recent tasks from short-term memory
                recent_tasks = self.memory.get_recent_tasks(limit=5)

                if recent_tasks:
                    history_content = "# Recent Session History\n\n"
                    for task in recent_tasks:
                        history_content += f"- Task: {task.get('task_description', 'N/A')}\n"
                        if 'outputs' in task:
                            history_content += f"  Output: {str(task['outputs'])[:200]}...\n"
                        history_content += "\n"

                    history_tokens = self.add_context(
                        content=history_content,
                        tier=ContextTier.RECENT_HISTORY,
                        metadata={"source": "short_term_memory", "num_tasks": len(recent_tasks)}
                    )

                    remaining_budget -= history_tokens
                    logger.info(f"Loaded {len(recent_tasks)} recent tasks ({history_tokens} tokens)")

            except Exception as e:
                logger.error(f"Failed to load session history: {e}")

        # 4. Build final context with truncation if needed
        final_context = self.build_context(
            strategy=TruncationStrategy.SLIDING_WINDOW,
            max_tokens=context_budget
        )

        actual_tokens = self.token_counter.count_tokens(final_context)
        logger.info(
            f"Context loaded for {vsm_tier}: {actual_tokens}/{context_budget} tokens used"
        )

        return final_context

    def build_context(
        self,
        strategy: TruncationStrategy = TruncationStrategy.SLIDING_WINDOW,
        enforce_budget: bool = True
    ) -> str:
        """
        Build final context string with automatic truncation if needed.

        Args:
            strategy: Truncation strategy to use if over budget
            enforce_budget: Whether to enforce token budget

        Returns:
            Final context string ready for LLM consumption
        """
        # Check if over budget
        if enforce_budget and self.is_over_budget():
            logger.warning(
                f"Context over budget: {self.get_total_tokens()} tokens "
                f"(limit: {self.budget.total_budget - self.budget.reserved})"
            )

            # Truncate tiers in reverse priority order
            for tier in reversed(list(ContextTier)):
                if not self.is_over_budget():
                    break

                budget = self.budget.get_budget(tier)
                current = self.get_tier_tokens(tier)

                if current > budget:
                    # Truncate this tier to budget
                    self.truncate_tier(tier, budget, strategy)

            # If still over budget, do aggressive truncation
            if self.is_over_budget():
                logger.warning("Still over budget after tier truncation, applying aggressive reduction")
                self._aggressive_truncation(strategy)

        # Build final context by concatenating tiers in priority order
        context_parts = []

        for tier in ContextTier:
            blocks = self.context_blocks[tier]

            if not blocks:
                continue

            # Add tier separator
            context_parts.append(f"\n{'='*80}\n")
            context_parts.append(f"# Context Tier: {tier.name}\n")
            context_parts.append(f"{'='*80}\n\n")

            # Add all blocks in tier
            for block in blocks:
                context_parts.append(block.content)
                context_parts.append("\n\n")

        context = "".join(context_parts)

        final_tokens = self.token_counter.count_tokens(context)

        logger.info(
            f"Built context: {final_tokens} tokens "
            f"({len(self.context_blocks)} blocks across {len(ContextTier)} tiers)"
        )

        return context

    def _aggressive_truncation(self, strategy: TruncationStrategy):
        """
        Aggressive truncation when normal tier budgets aren't enough.

        Reduces lower-priority tiers more aggressively.
        """
        target = self.budget.total_budget - self.budget.reserved
        current = self.get_total_tokens()
        reduction_needed = current - target

        logger.warning(f"Aggressive truncation: need to reduce by {reduction_needed} tokens")

        # Reduce tiers in reverse priority order
        for tier in reversed(list(ContextTier)):
            if reduction_needed <= 0:
                break

            current_tier_tokens = self.get_tier_tokens(tier)

            if current_tier_tokens == 0:
                continue

            # Reduce tier by 50% or reduction needed, whichever is smaller
            reduction = min(current_tier_tokens // 2, reduction_needed)
            new_target = current_tier_tokens - reduction

            if new_target > 0:
                self.truncate_tier(tier, new_target, strategy)
                reduction_needed -= reduction

    def clear(self):
        """Clear all context blocks and reset token usage."""
        self.context_blocks = {tier: [] for tier in ContextTier}
        self.token_usage = {tier: 0 for tier in ContextTier}
        logger.debug("Cleared all context blocks")

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of current context state.

        Returns:
            Dict with token usage, budget status, and tier breakdown
        """
        total_tokens = self.get_total_tokens()
        budget_limit = self.budget.total_budget - self.budget.reserved

        tier_breakdown = {}
        for tier in ContextTier:
            tier_breakdown[tier.name] = {
                "tokens": self.token_usage[tier],
                "blocks": len(self.context_blocks[tier]),
                "budget": self.budget.get_budget(tier),
                "utilization": (
                    self.token_usage[tier] / self.budget.get_budget(tier)
                    if self.budget.get_budget(tier) > 0 else 0.0
                )
            }

        return {
            "total_tokens": total_tokens,
            "budget_limit": budget_limit,
            "reserved": self.budget.reserved,
            "utilization": total_tokens / budget_limit if budget_limit > 0 else 0.0,
            "over_budget": self.is_over_budget(),
            "tier_breakdown": tier_breakdown
        }


# Demo
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("=" * 80)
    print("ContextManager Demo - Phase 4")
    print("=" * 80)
    print()

    # Test 1: Initialize
    print("[1/5] Initializing ContextManager...")
    manager = ContextManager()
    print(f"✅ Initialized with {manager.budget.total_budget} token budget")
    print()

    # Test 2: Add task context
    print("[2/5] Adding task context...")
    manager.add_task_context(
        task_description="Implement new feature for agent coordination",
        inputs={"feature": "context_manager", "priority": "high"}
    )
    print(f"✅ Added task context: {manager.get_tier_tokens(ContextTier.CURRENT_TASK)} tokens")
    print()

    # Test 3: Add system prompt
    print("[3/5] Adding system prompt...")
    system_prompt = """You are a helpful AI assistant specialized in software development.
    You help implement features, debug code, and provide technical guidance.
    Always follow best practices and write clean, maintainable code."""
    manager.add_system_prompt(system_prompt)
    print(f"✅ Added system prompt: {manager.get_tier_tokens(ContextTier.SYSTEM_PROMPTS)} tokens")
    print()

    # Test 4: Check summary
    print("[4/5] Context summary...")
    summary = manager.get_summary()
    print(f"✅ Total tokens: {summary['total_tokens']}/{summary['budget_limit']}")
    print(f"   Utilization: {summary['utilization']:.1%}")
    print(f"   Over budget: {summary['over_budget']}")
    print()

    # Test 5: Build context
    print("[5/5] Building final context...")
    context = manager.build_context()
    print(f"✅ Built context: {len(context)} characters")
    print()

    print("=" * 80)
    print("ContextManager Demo Complete!")
    print("=" * 80)
