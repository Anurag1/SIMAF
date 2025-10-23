# Phase 4: Coordination & Human Review - Implementation Plan

**Date:** 2025-10-19
**Phase:** Phase 4 (Week 13-16)
**Status:** 🔄 PLANNING
**Dependencies:** Phase 3 Complete ✅

---

## Overview

Phase 4 implements the human-in-the-loop review system via Obsidian integration, advanced coordination capabilities with System 2 agent, and context management for scaling to 10+ parallel agents.

**Phase 4 Requirements:**

- [ ] Obsidian vault integration with Git sync
- [ ] Human review workflow (CLI tool)
- [ ] System 2 Coordination Agent (conflict detection/resolution)
- [ ] Tiered context management for 200K token budget compliance

---

## Previously Completed (Phases 0-3)

### ✅ Phase 0: Foundation (Week 1-2)

- **UnifiedLM**: Multi-provider LLM with Anthropic → Gemini failover
- **Model Registry**: Dynamic model discovery and tier-based selection
- **DSPy Integration**: FractalDSpyLM wrapper
- **ResearchAgent**: First operational agent
- **File:** `fractal_agent/PHASE0_COMPLETE.md`

### ✅ Phase 1: Multi-Agent Coordination (Week 3-4)

- **Control Agent**: Task decomposition
- **LangGraph Workflow**: Hub-and-Spoke topology with dynamic spawning
- **Task Registry**: Duplicate work prevention
- **Token Budget Pool**: Budget compliance
- **SynthesisAgent**: Multi-result aggregation
- **File:** `fractal_agent/PHASE1_COMPLETE.md`

### ✅ Phase 2: Production Hardening (Week 5-8)

- **Security**: PII redaction (Presidio), prompt injection detection
- **Resilience**: Circuit breakers, retry with backoff, fallback chains
- **Testing**: 80%+ test coverage, CI/CD pipeline
- **File:** `PHASE2_COMPLETE.md`

### ✅ Phase 3: Intelligence Layer (Week 9-12)

- **Intelligence Agent**: System 4 performance reflection
- **GraphRAG**: Neo4j + Qdrant hybrid search with temporal validity
- **A/B Testing Framework**: Statistical variant comparison
- **MIPRO Integration**: Automated prompt optimization
- **Files:** `PHASE3_PLAN.md`, `PHASE3_PROGRESS.md`, `MIPRO_VERIFICATION.md`

---

## Implementation Plan

## Task 1: Obsidian Integration (Week 13-14)

**Duration:** 2 weeks
**Priority:** HIGH

### 1.1 Vault Structure Design

**Objective:** Create Obsidian vault structure for human review of agent outputs

**Vault Organization:**

```
obsidian-vault/
├── .obsidian/                    # Obsidian config
│   ├── workspace.json
│   └── plugins/
├── Templates/
│   ├── research-output.md        # Template for ResearchAgent outputs
│   ├── analysis-output.md        # Template for AnalysisAgent outputs
│   ├── synthesis-output.md       # Template for SynthesisAgent outputs
│   ├── intelligence-report.md    # Template for Intelligence Agent insights
│   └── knowledge-triple.md       # Template for knowledge graph entries
├── Inbox/                        # New outputs for review
│   ├── pending/
│   └── reviewed/
├── Knowledge/                    # Approved knowledge
│   ├── entities/
│   ├── relationships/
│   └── insights/
├── Archive/                      # Historical outputs
│   └── YYYY-MM-DD/
└── .git/                         # Version control
```

**File:** `fractal_agent/integrations/obsidian/vault_structure.py`

```python
from pathlib import Path
from typing import Dict, Optional
import yaml

class ObsidianVault:
    """
    Manages Obsidian vault structure for agent output review.

    Features:
    - Creates standardized folder structure
    - Manages file templates
    - Handles Git synchronization
    - Tracks review status
    """

    def __init__(self, vault_path: str):
        self.vault_path = Path(vault_path)
        self.inbox_path = self.vault_path / "Inbox"
        self.knowledge_path = self.vault_path / "Knowledge"
        self.templates_path = self.vault_path / "Templates"
        self.archive_path = self.vault_path / "Archive"

    def initialize_vault(self) -> None:
        """Create vault folder structure"""
        pass

    def create_from_template(
        self,
        template_name: str,
        content: Dict,
        output_dir: str = "Inbox/pending"
    ) -> Path:
        """Create new file from template with agent output"""
        pass

    def move_to_reviewed(self, file_path: Path) -> None:
        """Move file from pending to reviewed"""
        pass

    def promote_to_knowledge(self, file_path: Path) -> None:
        """Promote reviewed file to Knowledge base"""
        pass

    def sync_to_git(self, commit_message: str) -> None:
        """Sync vault to Git repository"""
        pass
```

**Implementation Tasks:**

- [ ] Create `ObsidianVault` class
- [ ] Implement folder structure initialization
- [ ] Create file templates (5 templates)
- [ ] Implement template rendering with Jinja2
- [ ] Add Git integration (via GitPython)
- [ ] Test vault creation and file management

**Success Metrics:**

- ✅ Vault structure created correctly
- ✅ Templates render with agent output data
- ✅ Git sync works (commit + push)

---

### 1.2 CLI Review Tool

**Objective:** Create CLI tool for reviewing agent outputs in terminal

**File:** `fractal_agent/integrations/obsidian/review_cli.py`

```python
import typer
from rich.console import Console
from rich.table import Table
from rich.markdown import Markdown
from pathlib import Path
from typing import List

app = typer.Typer()
console = Console()

@app.command()
def list_pending():
    """List all pending agent outputs awaiting review"""
    pass

@app.command()
def review(file_id: str):
    """Review a specific output file"""
    # Display file content
    # Show approval options
    # Handle approval/rejection/edit
    pass

@app.command()
def approve(file_id: str, promote_to_graphrag: bool = True):
    """Approve an output and optionally promote to GraphRAG"""
    pass

@app.command()
def reject(file_id: str, reason: str):
    """Reject an output with reason"""
    pass

@app.command()
def batch_approve(pattern: str):
    """Approve multiple outputs matching pattern"""
    pass

@app.command()
def sync():
    """Sync vault to Git and optionally to GraphRAG"""
    pass

if __name__ == "__main__":
    app()
```

**CLI Usage Examples:**

```bash
# List pending reviews
fractal-review list-pending

# Review specific output
fractal-review review research-2025-10-19-1234

# Approve and promote to GraphRAG
fractal-review approve research-2025-10-19-1234 --promote-to-graphrag

# Reject with reason
fractal-review reject analysis-2025-10-19-5678 --reason "Outdated information"

# Batch approve all research outputs from today
fractal-review batch-approve "research-2025-10-19-*"

# Sync to Git
fractal-review sync
```

**Implementation Tasks:**

- [ ] Implement CLI with Typer
- [ ] Add Rich formatting for pretty output
- [ ] Implement file listing with metadata
- [ ] Add review workflow (approve/reject/edit)
- [ ] Implement batch operations
- [ ] Add GraphRAG promotion integration
- [ ] Write unit tests for CLI commands

**Success Metrics:**

- ✅ CLI lists pending outputs correctly
- ✅ Review workflow is smooth (< 30 seconds per output)
- ✅ Approved knowledge flows to GraphRAG automatically
- ✅ Git commits track all reviews with metadata

---

### 1.3 Agent Output Export Integration

**Objective:** Integrate Obsidian export into agent workflows

**File:** `fractal_agent/integrations/obsidian/export.py`

```python
from typing import Dict, Any
from pathlib import Path
import dspy
from fractal_agent.integrations.obsidian.vault_structure import ObsidianVault

class ObsidianExporter:
    """
    Export agent outputs to Obsidian vault for human review.

    Integrates with agent workflows to automatically export
    outputs to Obsidian inbox for review.
    """

    def __init__(self, vault: ObsidianVault):
        self.vault = vault

    def export_research_output(
        self,
        query: str,
        answer: str,
        citations: List[str],
        metadata: Dict[str, Any]
    ) -> Path:
        """Export ResearchAgent output to Obsidian"""
        pass

    def export_intelligence_insights(
        self,
        performance_analysis: str,
        recommendations: List[str],
        metrics: Dict[str, float]
    ) -> Path:
        """Export Intelligence Agent insights to Obsidian"""
        pass

    def export_knowledge_triple(
        self,
        subject: str,
        predicate: str,
        object: str,
        confidence: float,
        source: str
    ) -> Path:
        """Export knowledge triple for review before GraphRAG insertion"""
        pass
```

**Integration Example:**

```python
# In ResearchAgent workflow
from fractal_agent.integrations.obsidian import ObsidianExporter

class ResearchWorkflow:
    def __init__(self, obsidian_exporter: ObsidianExporter = None):
        self.exporter = obsidian_exporter

    def run(self, query: str) -> str:
        # Run research
        result = self.agent(query=query)

        # Export to Obsidian for human review
        if self.exporter:
            self.exporter.export_research_output(
                query=query,
                answer=result.answer,
                citations=result.citations,
                metadata={
                    "timestamp": datetime.now(),
                    "model": self.agent.lm.model,
                    "cost": result.cost
                }
            )

        return result.answer
```

**Implementation Tasks:**

- [ ] Create `ObsidianExporter` class
- [ ] Implement export methods for each agent type
- [ ] Add export hooks to agent workflows
- [ ] Create metadata tracking (timestamp, model, cost)
- [ ] Test integration with ResearchAgent
- [ ] Test integration with Intelligence Agent
- [ ] Document export format for each agent

**Success Metrics:**

- ✅ Agent outputs automatically exported to Obsidian
- ✅ Metadata correctly tracked
- ✅ Templates render correctly with agent data
- ✅ No performance impact (export is async)

---

## Task 2: Advanced Coordination (Week 15)

**Duration:** 1 week
**Priority:** MEDIUM

### 2.1 System 2 Coordination Agent

**Objective:** Implement VSM System 2 agent for conflict detection and resolution

**Background:** Phase 1 implemented lightweight coordination (Task Registry, Token Budget Pool, Rate Limiter). Phase 4 adds an intelligent Coordination Agent that can detect and resolve conflicts between parallel agents.

**File:** `fractal_agent/agents/coordination_agent.py`

```python
import dspy
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class Conflict:
    """Represents a conflict between agents"""
    type: str  # "duplicate_work", "contradictory_output", "budget_violation"
    agents: List[str]  # Agent IDs involved
    description: str
    severity: int  # 1-10
    suggested_resolution: str

class ConflictDetector(dspy.Signature):
    """
    Detect conflicts between parallel agent activities.

    Input: Activity logs from multiple agents
    Output: List of detected conflicts
    """
    agent_logs: str = dspy.InputField(desc="Logs from parallel agents")
    active_tasks: str = dspy.InputField(desc="Currently active tasks")
    conflicts: List[Conflict] = dspy.OutputField(desc="Detected conflicts")

class ConflictResolver(dspy.Signature):
    """
    Propose resolutions for detected conflicts.

    Input: Conflict description and context
    Output: Resolution strategy
    """
    conflict: Conflict = dspy.InputField(desc="Conflict to resolve")
    context: str = dspy.InputField(desc="System context and constraints")
    resolution: str = dspy.OutputField(desc="Proposed resolution strategy")
    actions: List[str] = dspy.OutputField(desc="Concrete actions to resolve")

class CoordinationAgent(dspy.Module):
    """
    VSM System 2 - Coordination Agent

    Monitors parallel agent activities and resolves conflicts.
    Integrates with lightweight coordination infrastructure from Phase 1.
    """

    def __init__(self, config: CoordinationConfig = None):
        super().__init__()
        self.config = config or CoordinationConfig()

        # DSPy signatures
        self.detect_conflicts = dspy.ChainOfThought(ConflictDetector)
        self.resolve_conflict = dspy.ChainOfThought(ConflictResolver)

    def forward(self, agent_logs: List[Dict], active_tasks: List[Dict]):
        """
        Monitor agent activities and resolve conflicts.

        Args:
            agent_logs: Recent activity logs from all agents
            active_tasks: Currently executing tasks

        Returns:
            Coordination actions to take
        """
        # Stage 1: Detect conflicts
        conflicts = self.detect_conflicts(
            agent_logs=str(agent_logs),
            active_tasks=str(active_tasks)
        )

        # Stage 2: Resolve high-severity conflicts
        resolutions = []
        for conflict in conflicts.conflicts:
            if conflict.severity >= self.config.min_severity_to_resolve:
                resolution = self.resolve_conflict(
                    conflict=conflict,
                    context=self._get_system_context()
                )
                resolutions.append(resolution)

        return resolutions
```

**Configuration:**

```python
@dataclass
class CoordinationConfig:
    """Configuration for Coordination Agent"""
    min_severity_to_resolve: int = 5  # Only resolve conflicts >= severity 5
    monitor_interval_seconds: int = 30  # Check for conflicts every 30s
    max_parallel_agents: int = 10  # Alert if more than 10 agents active
    budget_alert_threshold: float = 0.8  # Alert at 80% budget usage
```

**Implementation Tasks:**

- [ ] Create `CoordinationAgent` class with DSPy
- [ ] Implement `ConflictDetector` signature
- [ ] Implement `ConflictResolver` signature
- [ ] Add conflict type detection (duplicate work, contradictory outputs, budget violations)
- [ ] Integrate with Task Registry from Phase 1
- [ ] Integrate with Token Budget Pool from Phase 1
- [ ] Create unit tests for conflict detection
- [ ] Create integration tests for conflict resolution

**Success Metrics:**

- ✅ Detects duplicate work (Task Registry integration)
- ✅ Detects contradictory outputs between agents
- ✅ Detects budget violations (Token Budget Pool integration)
- ✅ Proposes valid resolutions
- ✅ Integrates with Phase 1 lightweight infrastructure

---

### 2.2 Coordination Workflow

**Objective:** Integrate Coordination Agent into LangGraph workflow

**File:** `fractal_agent/workflows/coordination_workflow.py`

```python
from langgraph.graph import StateGraph, END
from fractal_agent.agents import CoordinationAgent
from fractal_agent.memory import ShortTermMemory

def create_coordination_workflow(coordination_agent: CoordinationAgent):
    """
    Create LangGraph workflow with coordination monitoring.

    Extends Phase 1 hub-and-spoke workflow with coordination layer.
    """

    workflow = StateGraph(dict)

    # Nodes
    workflow.add_node("control", control_node)
    workflow.add_node("workers", worker_node)
    workflow.add_node("coordination", coordination_node)  # NEW
    workflow.add_node("synthesis", synthesis_node)

    # Edges
    workflow.add_edge("control", "workers")
    workflow.add_edge("workers", "coordination")  # NEW: Check for conflicts
    workflow.add_conditional_edges(
        "coordination",
        route_after_coordination,
        {
            "continue": "workers",  # No conflicts, continue
            "resolve": "coordination",  # Conflicts detected, resolve
            "synthesize": "synthesis"  # All done, synthesize
        }
    )
    workflow.add_edge("synthesis", END)

    workflow.set_entry_point("control")

    return workflow.compile()

def coordination_node(state: dict) -> dict:
    """
    Coordination Agent node.

    Monitors agent activities and resolves conflicts.
    """
    coordination_agent = state["coordination_agent"]
    agent_logs = state["agent_logs"]
    active_tasks = state["active_tasks"]

    # Detect and resolve conflicts
    resolutions = coordination_agent(
        agent_logs=agent_logs,
        active_tasks=active_tasks
    )

    # Apply resolutions
    for resolution in resolutions:
        apply_resolution(state, resolution)

    state["conflicts_resolved"] = len(resolutions)

    return state
```

**Implementation Tasks:**

- [ ] Extend Phase 1 workflow with coordination node
- [ ] Add conditional routing after coordination
- [ ] Implement conflict resolution application
- [ ] Test workflow with simulated conflicts
- [ ] Validate no performance degradation (< 5% overhead)

**Success Metrics:**

- ✅ Coordination node integrates into workflow
- ✅ Conflicts detected and resolved in real-time
- ✅ < 5% performance overhead
- ✅ Workflow scales to 10+ parallel agents

---

## Task 3: Context Management (Week 16)

**Duration:** 1 week
**Priority:** HIGH

### 3.1 Tiered Context Loading

**Objective:** Implement tiered context loading to stay within 200K token budget

**Background:** With GraphRAG, Intelligence Layer, and 10+ parallel agents, context can easily exceed the 200K token limit. Tiered loading prioritizes the most relevant context.

**File:** `fractal_agent/memory/context_manager.py`

```python
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class ContextBudget:
    """Token budget allocation for context tiers"""
    tier1_active: int = 50_000      # Active working memory (highest priority)
    tier2_short_term: int = 75_000  # Recent session memory
    tier3_long_term: int = 60_000   # GraphRAG retrieval
    tier4_meta: int = 15_000        # Intelligence insights
    total_limit: int = 200_000

    def validate(self):
        total = self.tier1_active + self.tier2_short_term + self.tier3_long_term + self.tier4_meta
        assert total <= self.total_limit, f"Budget exceeds limit: {total} > {self.total_limit}"

class ContextManager:
    """
    Manages context loading with tiered priority system.

    Ensures total context stays within 200K token limit while
    prioritizing the most relevant information.
    """

    def __init__(self, budget: ContextBudget = None):
        self.budget = budget or ContextBudget()
        self.budget.validate()

    def load_context(
        self,
        query: str,
        active_memory: Dict,
        short_term_memory: ShortTermMemory,
        long_term_memory: LongTermMemory,
        meta_memory: MetaMemory
    ) -> Dict[str, Any]:
        """
        Load context from all tiers within budget.

        Priority order:
        1. Active working memory (current task state)
        2. Short-term memory (recent session)
        3. Long-term memory (GraphRAG relevant knowledge)
        4. Meta memory (Intelligence insights)

        Returns:
            Context dict with content from each tier
        """
        context = {}
        tokens_used = 0

        # Tier 1: Active Memory (highest priority)
        tier1_content, tier1_tokens = self._load_tier1(active_memory)
        if tier1_tokens <= self.budget.tier1_active:
            context["active"] = tier1_content
            tokens_used += tier1_tokens
        else:
            # Truncate if needed
            context["active"] = self._truncate(tier1_content, self.budget.tier1_active)
            tokens_used += self.budget.tier1_active

        # Tier 2: Short-term Memory
        tier2_content, tier2_tokens = self._load_tier2(short_term_memory, query)
        remaining = self.budget.tier2_short_term
        if tier2_tokens <= remaining:
            context["short_term"] = tier2_content
            tokens_used += tier2_tokens
        else:
            context["short_term"] = self._truncate(tier2_content, remaining)
            tokens_used += remaining

        # Tier 3: Long-term Memory (GraphRAG)
        tier3_content, tier3_tokens = self._load_tier3(long_term_memory, query)
        remaining = self.budget.tier3_long_term
        if tier3_tokens <= remaining:
            context["long_term"] = tier3_content
            tokens_used += tier3_tokens
        else:
            context["long_term"] = self._truncate(tier3_content, remaining)
            tokens_used += remaining

        # Tier 4: Meta Memory (Intelligence insights)
        tier4_content, tier4_tokens = self._load_tier4(meta_memory, query)
        remaining = self.budget.tier4_meta
        if tier4_tokens <= remaining:
            context["meta"] = tier4_content
            tokens_used += tier4_tokens
        else:
            context["meta"] = self._truncate(tier4_content, remaining)
            tokens_used += remaining

        # Track usage
        context["_tokens_used"] = tokens_used
        context["_budget_compliance"] = tokens_used <= self.budget.total_limit

        return context

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation: 1 token ≈ 4 chars)"""
        return len(text) // 4

    def _truncate(self, content: str, max_tokens: int) -> str:
        """Truncate content to fit within token budget"""
        max_chars = max_tokens * 4
        if len(content) <= max_chars:
            return content
        return content[:max_chars] + "... [truncated]"
```

**Implementation Tasks:**

- [ ] Create `ContextManager` class
- [ ] Implement `ContextBudget` configuration
- [ ] Implement tiered loading with priority
- [ ] Add token estimation (use tiktoken for accuracy)
- [ ] Implement truncation strategies
- [ ] Add budget compliance tracking
- [ ] Test with varying context sizes
- [ ] Validate stays within 200K limit

**Success Metrics:**

- ✅ Context never exceeds 200K tokens
- ✅ Most relevant content prioritized
- ✅ Graceful degradation when context is large
- ✅ < 100ms overhead for context loading

---

### 3.2 Graph Partitioning (for Large Graphs)

**Objective:** Implement graph partitioning for GraphRAG queries when knowledge base is large

**File:** `fractal_agent/memory/graph_partitioning.py`

```python
from typing import List, Set
from neo4j import GraphDatabase

class GraphPartitioner:
    """
    Partition large knowledge graphs for efficient retrieval.

    When GraphRAG contains millions of nodes/edges, partition
    into smaller subgraphs based on query relevance.
    """

    def __init__(self, neo4j_driver):
        self.driver = neo4j_driver

    def partition_by_query(
        self,
        query: str,
        max_nodes: int = 1000
    ) -> Set[str]:
        """
        Partition graph to most relevant subgraph for query.

        Args:
            query: User query
            max_nodes: Maximum nodes to include

        Returns:
            Set of node IDs to include in context
        """
        # 1. Find seed nodes (most relevant entities)
        seed_nodes = self._find_seed_nodes(query, limit=10)

        # 2. Expand to neighbors (breadth-first)
        subgraph = self._expand_subgraph(seed_nodes, max_nodes)

        return subgraph

    def partition_by_time(
        self,
        start_date: str,
        end_date: str
    ) -> Set[str]:
        """Partition graph by temporal validity"""
        pass

    def partition_by_domain(
        self,
        domain: str
    ) -> Set[str]:
        """Partition graph by knowledge domain"""
        pass
```

**Implementation Tasks:**

- [ ] Create `GraphPartitioner` class
- [ ] Implement query-based partitioning
- [ ] Implement temporal partitioning
- [ ] Implement domain-based partitioning
- [ ] Add partition caching
- [ ] Test with large graphs (10K+ nodes)
- [ ] Validate retrieval quality maintained

**Success Metrics:**

- ✅ Partitioning reduces query time by 50%+
- ✅ Retrieval quality maintained (vs. full graph)
- ✅ Works with graphs up to 1M nodes

---

## Success Criteria

| Criterion                 | Target     | Measurement                                     |
| ------------------------- | ---------- | ----------------------------------------------- |
| Human review workflow     | Smooth     | < 30 seconds per review                         |
| Knowledge promotion       | Automatic  | Approved → GraphRAG without manual steps        |
| Context budget compliance | 100%       | Never exceed 200K tokens                        |
| System scaling            | 10+ agents | Support 10+ parallel agents without issues      |
| Review adoption           | High       | > 80% of agent outputs reviewed within 24 hours |
| Conflict detection        | Accurate   | 90%+ conflict detection rate                    |
| Obsidian sync             | Reliable   | 100% Git sync success rate                      |

---

## File Structure

```
fractal_agent/
├── integrations/
│   └── obsidian/
│       ├── __init__.py
│       ├── vault_structure.py       # NEW: Vault management
│       ├── export.py                # NEW: Agent output export
│       └── review_cli.py            # NEW: CLI review tool
├── agents/
│   ├── coordination_agent.py       # NEW: System 2 coordination
│   └── coordination_config.py      # NEW: Coordination config
├── memory/
│   ├── context_manager.py          # NEW: Tiered context loading
│   └── graph_partitioning.py       # NEW: Graph partitioning
└── workflows/
    └── coordination_workflow.py     # NEW: Workflow with coordination

obsidian-vault/                     # NEW: Obsidian vault
├── .obsidian/
├── Templates/
├── Inbox/
├── Knowledge/
├── Archive/
└── .git/

tests/
├── unit/
│   ├── test_obsidian_vault.py      # NEW
│   ├── test_coordination_agent.py  # NEW
│   └── test_context_manager.py     # NEW
└── integration/
    ├── test_obsidian_workflow.py   # NEW
    ├── test_coordination_workflow.py # NEW
    └── test_context_scaling.py     # NEW

docs/
└── obsidian_integration.md         # NEW: Obsidian integration guide
```

---

## Implementation Order

### Week 13 ⏳

1. Create Obsidian vault structure
2. Implement file templates
3. Add Git integration
4. Test vault creation and management

### Week 14 ⏳

1. Implement CLI review tool
2. Create export integration for agents
3. Test review workflow end-to-end
4. Document Obsidian integration

### Week 15 ⏳

1. Implement Coordination Agent (System 2)
2. Add conflict detection signatures
3. Integrate with Phase 1 infrastructure
4. Test conflict resolution

### Week 16 ⏳

1. Implement Context Manager with tiered loading
2. Add graph partitioning for large graphs
3. Test context budget compliance
4. Validate 10+ parallel agent scaling

---

## Dependencies

### Python Packages

```bash
pip install typer>=0.9.0           # CLI framework
pip install rich>=13.0.0           # Terminal formatting
pip install GitPython>=3.1.0       # Git integration
pip install Jinja2>=3.1.0          # Template rendering
pip install tiktoken>=0.5.0        # Accurate token counting
```

### External Services

- **Git Repository**: For Obsidian vault version control
- **Obsidian**: Desktop app for human review (optional - vault is plain markdown)
- **Neo4j**: From Phase 3 (for graph partitioning)

---

## Phase 4 Deliverables

### Code

- [ ] `fractal_agent/integrations/obsidian/` - Complete Obsidian integration
- [ ] `fractal_agent/agents/coordination_agent.py` - System 2 coordination
- [ ] `fractal_agent/memory/context_manager.py` - Tiered context loading
- [ ] `fractal_agent/memory/graph_partitioning.py` - Graph partitioning

### Documentation

- [ ] `docs/obsidian_integration.md` - Obsidian integration guide
- [ ] CLI tool documentation (in README)
- [ ] Context management best practices

### Tests

- [ ] Unit tests for Obsidian vault (>80% coverage)
- [ ] Unit tests for Coordination Agent (>80% coverage)
- [ ] Integration tests for review workflow
- [ ] Context budget compliance tests
- [ ] Scaling tests (10+ agents)

### Vault

- [ ] Obsidian vault with templates
- [ ] Git repository initialized
- [ ] Example reviewed outputs

---

## Risk Mitigation

| Risk                          | Likelihood | Impact | Mitigation                                                 |
| ----------------------------- | ---------- | ------ | ---------------------------------------------------------- |
| Low review adoption           | Medium     | High   | Make CLI extremely fast and easy; show value with insights |
| Git conflicts in vault        | Medium     | Medium | Use append-only structure; separate files per output       |
| Context budget exceeded       | Low        | High   | Implement strict tiered loading; alert when near limit     |
| Coordination overhead         | Low        | Medium | Make coordination optional; only activate for 5+ agents    |
| Graph partitioning complexity | Medium     | Medium | Start with simple query-based partitioning; iterate        |

---

## Next Steps After Phase 4

**Phase 5: Policy & Production (Week 17-20)**

- Policy Agent (VSM System 5) for ethical boundaries
- External knowledge integration (web search, documents)
- Production monitoring (Prometheus + Grafana)
- Final production deployment

---

**Created:** 2025-10-19
**Phase:** 4 (Planning)
**Dependencies:** Phase 3 Complete ✅
**Next Phase:** Phase 5 (Policy & Production)
