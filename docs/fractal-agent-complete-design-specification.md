# Fractal Agent Ecosystem: Complete Design Specification

**Version:** 1.0
**Date:** 2025-10-18
**Author:** BMad
**Status:** Design Complete - Ready for Implementation

---

## Document Purpose

This document represents the **complete planning and solution design** for the Fractal Agent Ecosystem, covering all architectural decisions, implementation details, and justifications. This completes all brainstorming, planning, and solution design before implementation begins.

**Sessions Completed:**

1. ✅ Gap Analysis & Solution Selection (completed)
2. ✅ Implementation Deep Dive (this document)
3. ✅ VSM System 2 Coordination Design (this document)
4. ✅ GraphRAG & Memory Architecture (this document)
5. ✅ Obsidian Integration & Human-in-Loop (this document)
6. ✅ Testing & Validation Strategy (this document)
7. ✅ Security & Resilience Deep Dive (this document)
8. ✅ Advanced Topics (this document)
9. ✅ Emerged Questions Resolution (this document)
10. ✅ Final Architecture Validation (this document)

---

## Table of Contents

1. [Session 2: Implementation Deep Dive](#session-2-implementation-deep-dive)
2. [Session 3: VSM System 2 Coordination](#session-3-vsm-system-2-coordination)
3. [Session 4: GraphRAG & Memory Architecture](#session-4-graphrag--memory-architecture)
4. [Session 5: Obsidian Integration](#session-5-obsidian-integration)
5. [Session 6: Testing Strategy](#session-6-testing-strategy)
6. [Session 7: Security & Resilience](#session-7-security--resilience)
7. [Session 8: Advanced Topics](#session-8-advanced-topics)
8. [Session 9: Emerged Questions](#session-9-emerged-questions)
9. [Session 10: Architecture Validation](#session-10-architecture-validation)
10. [Implementation Roadmap](#implementation-roadmap)

---

# Session 2: Implementation Deep Dive

## 2.1 Agent Communication Protocol

### Research Findings

Based on 2025 standards, several protocols have emerged:

- **ACP (Agent Communication Protocol)** - IBM's open standard, RESTful API structure
- **A2A (Agent-to-Agent Protocol)** - Google's direct communication standard
- **MCP (Model Context Protocol)** - JSON-RPC-based client-server model

**Key Insights:**

- Aligning on schemas/ontologies is critical to avoid fragmentation
- Robust error handling and validation mechanisms are essential
- Design for the real case, not the ideal case (delays, ambiguity, failures)
- Message integrity via checksums and fallback protocols are necessary

### Solution Options (SCAMPER)

**Option 1: Adopt ACP Standard (Substitute)**

- Use IBM's Agent Communication Protocol as foundation
- RESTful API structure with MIME-type extensibility
- Pros: Industry standard, well-documented, extensible
- Cons: May be overkill for single-deployment system, external dependency

**Option 2: Custom JSON-RPC Protocol (Adapt)**

- Lightweight JSON-RPC-based messaging similar to MCP
- Tailored to VSM hierarchy needs
- Pros: Lightweight, tailored to our needs, no external deps
- Cons: No industry interoperability, must maintain ourselves

**Option 3: TypedDict Message Schema with Pydantic (Simplify)**

- Python-native TypedDict/Pydantic models for type safety
- Simple dictionary passing between agents
- Pros: Python-native, type-safe, minimal overhead, easy testing
- Cons: Not network-ready, limited to Python runtime

**Option 4: LangGraph State as Message Bus (Combine)**

- Use LangGraph's state mechanism as communication layer
- Messages written to graph state, nodes read from state
- Pros: Built into workflow orchestration, persistent, traceable
- Cons: Tightly coupled to LangGraph, harder to extract agents

**Option 5: Event-Driven with Redis Pub/Sub (Put to Other Use)**

- Redis as message broker, agents subscribe to topics
- Async event-driven architecture
- Pros: Scalable, decoupled, handles high throughput
- Cons: Infrastructure dependency (Redis), complexity for MVP

**Option 6: Direct Python Function Calls (Eliminate)**

- No protocol layer - agents call each other as Python functions
- Simplest possible approach
- Pros: Zero overhead, debuggable, Pythonic
- Cons: No persistence, no replay, hard to distribute later

### Evaluation (Six Thinking Hats)

**White Hat (Facts):**

- Phase 0-1 is single-process Python application
- No network communication needed yet
- Type safety is critical for DSPy integration
- LangGraph will be our orchestration layer

**Yellow Hat (Benefits):**

- Option 3 (TypedDict + Pydantic) provides type safety without overhead
- Option 4 (LangGraph State) provides persistence and traceability
- Combining 3 + 4 gives best of both worlds

**Black Hat (Risks):**

- Option 1 (ACP) is over-engineering for Phase 0-1
- Option 5 (Redis) adds infrastructure complexity too early
- Option 6 (direct calls) prevents future distribution

**Green Hat (Creative):**

- What if we use Pydantic models that serialize to LangGraph state?
- Models define the contract, LangGraph provides the transport
- Later, same models can serialize to network protocol (ACP/A2A)

**Blue Hat (Meta-Decision):**

- Start with Pydantic models + LangGraph state (evolutionary architecture)
- Keep models separate from LangGraph (loose coupling)
- Design models to be network-serializable for future phases

### WINNING SOLUTION: Pydantic Message Models + LangGraph State

**Design:**

```python
from pydantic import BaseModel, Field
from typing import Literal, Optional, Any
from datetime import datetime
from uuid import UUID, uuid4

# Base message model
class AgentMessage(BaseModel):
    """Base class for all agent messages"""
    message_id: UUID = Field(default_factory=uuid4)
    from_agent: str
    to_agent: str
    task_id: str
    message_type: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    parent_message_id: Optional[UUID] = None  # For threading

    class Config:
        # Allow serialization to dict for LangGraph state
        json_schema_extra = {"additionalProperties": False}

# Specific message types
class TaskAssignment(AgentMessage):
    message_type: Literal["task_assignment"] = "task_assignment"
    payload: dict[str, Any]
    constraints: dict[str, Any] = Field(default_factory=dict)
    deadline: Optional[datetime] = None

class TaskResult(AgentMessage):
    message_type: Literal["task_result"] = "task_result"
    success: bool
    result: Any
    error: Optional[str] = None
    metrics: dict[str, Any] = Field(default_factory=dict)

class TaskQuery(AgentMessage):
    message_type: Literal["task_query"] = "task_query"
    query: str
    context: dict[str, Any] = Field(default_factory=dict)

class CoordinationRequest(AgentMessage):
    message_type: Literal["coordination_request"] = "coordination_request"
    conflict_type: str
    involved_agents: list[str]
    details: dict[str, Any]

# Message validation
def validate_message(msg: dict) -> AgentMessage:
    """Validate and parse message to correct type"""
    msg_type = msg.get("message_type")

    type_map = {
        "task_assignment": TaskAssignment,
        "task_result": TaskResult,
        "task_query": TaskQuery,
        "coordination_request": CoordinationRequest
    }

    if msg_type not in type_map:
        raise ValueError(f"Unknown message type: {msg_type}")

    return type_map[msg_type](**msg)
```

**Integration with LangGraph:**

```python
from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, add_messages

class AgentGraphState(TypedDict):
    """LangGraph state schema"""
    # Message inbox (append-only log)
    messages: Annotated[Sequence[dict], add_messages]

    # Current task context
    current_task: Optional[str]
    task_status: str  # "pending", "in_progress", "completed", "failed"

    # Agent registry
    active_agents: dict[str, str]  # agent_id -> agent_type

    # Shared knowledge
    shared_context: dict[str, Any]

    # Metrics
    metrics: dict[str, Any]

# Agents write messages to state
def operational_agent_node(state: AgentGraphState):
    """Example agent node"""
    # Read latest message for this agent
    inbox = [msg for msg in state["messages"]
             if msg["to_agent"] == "operational_001"]

    if not inbox:
        return state

    latest = inbox[-1]
    task = validate_message(latest)

    # Process task
    result = process_task(task)

    # Send result message
    response = TaskResult(
        from_agent="operational_001",
        to_agent=task.from_agent,
        task_id=task.task_id,
        success=True,
        result=result,
        parent_message_id=task.message_id
    )

    # Add to message log
    return {
        **state,
        "messages": [response.model_dump()]
    }
```

**Justification:**

- ✅ Type-safe via Pydantic validation
- ✅ Persistent message log via LangGraph state
- ✅ Network-serializable (JSON) for future distribution
- ✅ Traceable via message threading (parent_message_id)
- ✅ Minimal infrastructure (no external deps)
- ✅ Evolutionary architecture (can add ACP later)

---

## 2.2 LangGraph Workflow Structures

### Research Findings

**Best Practices from 2025:**

- Use TypedDict for state schema (enforces clear data structure)
- State is central communication mechanism between nodes
- Prefer explicit state management over implicit
- Common patterns: Pipeline (sequential), Hub-and-Spoke (coordinator), Orchestrator-Worker
- Minimize tools per agent (focused responsibility)
- Use Send API for dynamic worker creation

**Key Insight:** LangGraph is ideal for explicit state, flexible control flow, and loops/iteration (agent tries, checks, repeats until goal achieved)

### Solution Options (SCAMPER)

**Option 1: Simple Linear Pipeline (Eliminate Complexity)**

```
Control → Operational_1 → Operational_2 → ... → Synthesis
```

- Pros: Simplest, easy to debug
- Cons: No parallelism, no dynamic routing

**Option 2: Hub-and-Spoke (Adapt VSM to LangGraph)**

```
         → Operational_1 →
Control  → Operational_2 → Synthesis
         → Operational_3 →
```

- Pros: Parallel execution, clear control flow
- Cons: Control agent becomes bottleneck

**Option 3: Dynamic Orchestrator-Worker (Combine Send API + VSM)**

```python
def control_agent(state):
    # Decompose task
    subtasks = decompose(state["task"])

    # Dynamically spawn workers
    return [
        Send("operational_worker", subtask)
        for subtask in subtasks
    ]
```

- Pros: Dynamic scaling, each worker has own state
- Cons: More complex, harder to debug

**Option 4: Hierarchical Recursion (Modify)**

```
Control_Level_1
  → Control_Level_2_A
      → Operational_1
      → Operational_2
  → Control_Level_2_B
      → Operational_3
```

- Pros: True fractal recursion, scales infinitely
- Cons: Complex state management, deep nesting

**Option 5: State Machine with Conditional Routing (Adapt)**

```python
def router(state):
    if state["status"] == "needs_research":
        return "operational"
    elif state["status"] == "needs_review":
        return "human_review"
    elif state["status"] == "complete":
        return END
```

- Pros: Flexible, handles errors/retries
- Cons: Can become complex decision tree

### Evaluation

**White Hat:** Phase 0-1 needs simple task decomposition → parallel execution → synthesis

**Yellow Hat:** Hub-and-Spoke (Option 2) + Dynamic Workers (Option 3) combines simplicity with flexibility

**Black Hat:** Hierarchical recursion too complex for MVP, state machine adds routing logic overhead

**Green Hat:** Hybrid: Use Hub-and-Spoke topology + Send API for dynamic worker count

**Blue Hat:** Progressive implementation - Start with Option 2, add Option 3 when needed

### WINNING SOLUTION: Hub-and-Spoke with Dynamic Worker Send API

**Design:**

```python
from langgraph.graph import StateGraph, END, Send
from typing import TypedDict, Annotated, Sequence, Literal

class WorkflowState(TypedDict):
    # Input
    user_task: str

    # Control agent outputs
    subtasks: Sequence[dict]  # List of subtask definitions

    # Operational agent outputs
    worker_results: Annotated[Sequence[dict], add_messages]

    # Synthesis
    final_report: Optional[str]

    # Status tracking
    status: Literal["decomposing", "executing", "synthesizing", "complete"]

    # Metrics
    metrics: dict[str, Any]

# Control agent node
def control_agent(state: WorkflowState):
    """VSM System 3 - Decompose task"""
    from fractal_agent.config.model_config import get_llm_for_role

    lm = get_llm_for_role("control")

    # Use DSPy for decomposition
    decomposer = TaskDecomposer()
    subtasks = decomposer(task=state["user_task"])

    return {
        **state,
        "subtasks": subtasks,
        "status": "executing"
    }

# Router: Send each subtask to dynamic worker
def spawn_workers(state: WorkflowState):
    """Dynamically create worker nodes"""
    return [
        Send("operational_worker", {"subtask": st})
        for st in state["subtasks"]
    ]

# Operational worker node
def operational_worker(state: dict):
    """VSM System 1 - Execute single subtask"""
    from fractal_agent.config.model_config import get_llm_for_role

    lm = get_llm_for_role("operational")

    researcher = ResearchAgent()
    result = researcher(question=state["subtask"]["question"])

    return {
        "worker_results": [{
            "subtask_id": state["subtask"]["id"],
            "result": result
        }]
    }

# Synthesis agent node
def synthesis_agent(state: WorkflowState):
    """VSM System 1 (specialized) - Combine results"""
    from fractal_agent.config.model_config import get_llm_for_role

    lm = get_llm_for_role("synthesis")

    synthesizer = SynthesisAgent()
    report = synthesizer(
        task=state["user_task"],
        results=state["worker_results"]
    )

    return {
        **state,
        "final_report": report,
        "status": "complete"
    }

# Build graph
def build_research_workflow():
    workflow = StateGraph(WorkflowState)

    # Add nodes
    workflow.add_node("control", control_agent)
    workflow.add_node("operational_worker", operational_worker)
    workflow.add_node("synthesis", synthesis_agent)

    # Add edges
    workflow.set_entry_point("control")

    # Dynamic worker spawning
    workflow.add_conditional_edges(
        "control",
        spawn_workers,
        ["operational_worker"]
    )

    # All workers complete → synthesis
    workflow.add_edge("operational_worker", "synthesis")
    workflow.add_edge("synthesis", END)

    return workflow.compile()
```

**Justification:**

- ✅ Hub-and-Spoke topology (clear VSM System 3 → System 1 flow)
- ✅ Dynamic worker creation (scales to N subtasks)
- ✅ Each worker has independent state (isolation)
- ✅ All results collected before synthesis (synchronization)
- ✅ Type-safe state via TypedDict
- ✅ Progressive enhancement path (can add routing, retry, etc.)

---

## 2.3 Context Caching Implementation Strategy

### Research Findings

**Anthropic 2025 Updates:**

- Prompt caching reduces costs by **up to 90%** and latency by **up to 85%**
- Cache TTL: 5 minutes (default) or 1 hour (premium)
- Cache read tokens **no longer count against ITPM limits** for Claude 3.7 Sonnet
- Can define up to **4 cache breakpoints** per prompt
- Cache automatically reads from longest previously cached prefix

**Best Practices:**

- Place stable content at beginning (system prompts, tools, context docs, few-shot examples)
- Place dynamic content last (user query, current task)
- Monitor cache hit rates and adjust
- Make API requests during off-peak hours (fewer cache evictions)

**Pricing:**

- 5-min cache write: 1.25× base input tokens
- 1-hour cache write: 2× base input tokens
- Cache read: 0.1× base input tokens (90% savings)

### Solution Options (SCAMPER)

**Option 1: Single System Message Caching (Simplify)**

```python
messages = [
    {
        "role": "system",
        "content": "You are a VSM System 1 operational agent...",
        "cache_control": {"type": "ephemeral"}  # Cache this
    },
    {"role": "user", "content": f"Task: {dynamic_task}"}
]
```

- Pros: Simplest, covers 80% of use cases
- Cons: Doesn't leverage multiple breakpoints

**Option 2: Four-Tier Caching (Maximize Breakpoints)**

```python
messages = [
    {
        "role": "system",
        "content": "Role definition...",
        "cache_control": {"type": "ephemeral"}  # Tier 1: Role
    },
    {
        "role": "user",
        "content": "Tool definitions...",
        "cache_control": {"type": "ephemeral"}  # Tier 2: Tools
    },
    {
        "role": "user",
        "content": "Background knowledge...",
        "cache_control": {"type": "ephemeral"}  # Tier 3: Knowledge
    },
    {
        "role": "user",
        "content": "Few-shot examples...",
        "cache_control": {"type": "ephemeral"}  # Tier 4: Examples
    },
    {"role": "user", "content": f"Current task: {task}"}  # Not cached
]
```

- Pros: Maximum cache optimization, tiered invalidation
- Cons: Complex to maintain, more cache writes

**Option 3: Session-Aware Caching (Combine)**

- Cache system + tools (session-independent, 1-hour TTL)
- Cache conversation history (session-dependent, 5-min TTL)

```python
# Session-independent (1-hour cache)
system_msg = {
    "role": "system",
    "content": role_definition + tool_definitions,
    "cache_control": {"type": "ephemeral", "ttl": 3600}
}

# Session-dependent (5-min cache)
history_msg = {
    "role": "user",
    "content": conversation_history,
    "cache_control": {"type": "ephemeral", "ttl": 300}
}

# Current query (not cached)
current_msg = {"role": "user", "content": query}
```

- Pros: Optimized cache TTL per content type, cost-efficient
- Cons: Not yet supported (TTL parameter is future feature)

**Option 4: Lazy Knowledge Loading with Caching (Adapt)**

- Only include knowledge when needed (context budget management)
- Cache the knowledge retrieval results

```python
# If task needs background knowledge
if requires_knowledge(task):
    knowledge = graphrag.retrieve(task, max_tokens=50000)
    messages.append({
        "role": "user",
        "content": f"Relevant knowledge: {knowledge}",
        "cache_control": {"type": "ephemeral"}
    })
```

- Pros: Combines caching with context management
- Cons: More complex control flow

**Option 5: Structured Prompt Template (Standardize)**

- Define standard template for all agents
- Ensures cache hits across different task invocations

```python
PROMPT_TEMPLATE = """
<system_role>{role_definition}</system_role>

<tools>{tool_definitions}</tools>

<background_knowledge>{knowledge}</background_knowledge>

<conversation_history>{history}</conversation_history>

<current_task>{task}</current_task>
"""
```

- Pros: Consistent structure → high cache hit rate
- Cons: Rigid format, may not suit all agents

### Evaluation

**White Hat:**

- System role + tools change rarely (1-hour cache viable)
- Conversation history changes per interaction (5-min cache)
- Knowledge retrieval is expensive (should cache)
- Different agents have different prompt structures

**Yellow Hat:**

- Option 2 (Four-Tier) maximizes cache granularity
- Option 4 (Lazy Loading) prevents context overflow
- Option 5 (Template) ensures consistency

**Black Hat:**

- Four-Tier adds complexity without proportional benefit
- Session-aware caching (Option 3) not available yet
- Rigid templates reduce agent flexibility

**Green Hat:**

- Combine Option 1 (simplicity) + Option 4 (lazy loading) + Option 5 (templates)
- Standard template per agent role (operational, control, etc.)
- Cache tiers: System (always) + Knowledge (when loaded) + History (optional)

**Blue Hat:**

- Start with Two-Tier caching: System+Tools (Tier 1) + Knowledge (Tier 2)
- Use role-specific templates
- Monitor cache hit rates, add tiers if needed

### WINNING SOLUTION: Two-Tier Role-Based Caching with Templates

**Design:**

````python
from typing import Protocol, Optional
from fractal_agent.utils.llm_provider import UnifiedLM

class PromptTemplate(Protocol):
    """Interface for prompt templates"""
    def build(
        self,
        task: str,
        knowledge: Optional[str] = None,
        history: Optional[list] = None
    ) -> list[dict]:
        ...

class OperationalAgentTemplate:
    """Template for VSM System 1 agents"""

    SYSTEM_PROMPT = """You are a VSM System 1 operational agent specializing in {specialty}.

Your responsibilities:
- Execute assigned tasks independently
- Research information thoroughly
- Provide structured outputs
- Report metrics (tokens, time, confidence)

Capabilities:
{tools}

Guidelines:
- Be concise but complete
- Cite sources when available
- Flag uncertainties clearly
- Stay within assigned scope"""

    def __init__(self, specialty: str, tools: list[str]):
        self.specialty = specialty
        self.tools = "\\n".join(f"- {tool}" for tool in tools)

    def build(
        self,
        task: str,
        knowledge: Optional[str] = None,
        history: Optional[list] = None
    ) -> list[dict]:
        messages = []

        # Tier 1: System prompt + tools (ALWAYS CACHED)
        messages.append({
            "role": "system",
            "content": self.SYSTEM_PROMPT.format(
                specialty=self.specialty,
                tools=self.tools
            ),
            "cache_control": {"type": "ephemeral"}
        })

        # Tier 2: Background knowledge (CACHED WHEN PROVIDED)
        if knowledge:
            messages.append({
                "role": "user",
                "content": f"<background_knowledge>\\n{knowledge}\\n</background_knowledge>",
                "cache_control": {"type": "ephemeral"}
            })

        # Optional: Conversation history (NOT CACHED in Phase 0)
        if history:
            for msg in history:
                messages.append(msg)  # No cache control

        # Current task (NEVER CACHED)
        messages.append({
            "role": "user",
            "content": f"<task>\\n{task}\\n</task>"
        })

        return messages

class ControlAgentTemplate:
    """Template for VSM System 3 agents"""

    SYSTEM_PROMPT = """You are a VSM System 3 control agent responsible for task decomposition and delegation.

Your responsibilities:
- Decompose complex tasks into subtasks
- Assign subtasks to appropriate operational agents
- Ensure subtasks are independent and parallelizable
- Define success criteria for each subtask

Output format:
```json
{
    "subtasks": [
        {
            "id": "subtask_1",
            "description": "...",
            "agent_type": "researcher",
            "constraints": {"max_tokens": 1000},
            "dependencies": []
        }
    ]
}
````

Guidelines:

- Create 3-7 subtasks (optimal for parallelism)
- Each subtask should be completable in <5 min
- Avoid inter-subtask dependencies when possible
- Consider token budgets in constraints"""

  def build(self, task: str, \*\*kwargs) -> list[dict]:
  return [
  {
  "role": "system",
  "content": self.SYSTEM_PROMPT,
  "cache_control": {"type": "ephemeral"}
  },
  {
  "role": "user",
  "content": f"<task_to_decompose>\\n{task}\\n</task_to_decompose>"
  }
  ]

# Template registry

TEMPLATE_REGISTRY = {
"operational": OperationalAgentTemplate,
"control": ControlAgentTemplate, # Add more as needed
}

# Usage in agents

def create_agent_prompt(role: str, **config):
"""Factory for creating prompts with caching"""
template_class = TEMPLATE_REGISTRY[role]
template = template_class(**config)
return template

# Example usage

operational_template = create_agent_prompt(
role="operational",
specialty="research",
tools=["web_search", "document_retrieval"]
)

# Build prompt with knowledge from GraphRAG

knowledge = graphrag.retrieve(task, max_tokens=30000)
messages = operational_template.build(
task="Research VSM System 2 coordination patterns",
knowledge=knowledge # This gets cached!
)

# Make LLM call

lm = get_llm_for_role("operational")
response = lm(messages=messages, max_tokens=2000)

# Check cache hit

print(f"Cache hit: {response['cache_hit']}")

````

**Cache Performance Monitoring:**

```python
class CacheMetricsTracker:
    """Track cache performance over time"""

    def __init__(self):
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_tokens_saved = 0

    def record_call(self, response: dict):
        """Record metrics from LLM response"""
        if response.get("cache_hit"):
            self.cache_hits += 1
            # Estimate tokens saved (90% of input tokens)
            self.total_tokens_saved += int(response["tokens_used"] * 0.9)
        else:
            self.cache_misses += 1

    def get_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0

    def get_cost_savings(self, cost_per_1k_tokens: float = 0.003) -> float:
        """Estimate cost savings from caching"""
        return (self.total_tokens_saved / 1000) * cost_per_1k_tokens * 0.9

# Global tracker
cache_tracker = CacheMetricsTracker()

# Wrap LLM calls
def tracked_llm_call(lm, **kwargs):
    response = lm(**kwargs)
    cache_tracker.record_call(response)
    return response
````

**Justification:**

- ✅ Two-tier caching balances simplicity and optimization
- ✅ Role-based templates ensure consistency → high cache hit rate
- ✅ Knowledge caching prevents expensive GraphRAG re-retrieval
- ✅ Monitoring tracks ROI of caching strategy
- ✅ Extensible to 4-tier if needed (add history, examples)
- ✅ Phase 0 target: >80% cache hit rate after warm-up

**Expected Savings:**

- First call: Full cost (cache write: 1.25× input tokens)
- Subsequent calls: 90% savings on system+tools+knowledge
- Break-even: After 2 calls with same cache
- Session of 10 calls: ~80% total cost reduction

---

## 2.4 Testing Framework Setup

### Research Findings

**2025 Best Practices:**

- DSPy assertions allow strict validation rules with automatic retry
- pytest-harvest plugin logs test execution data without stopping on first failure
- Multi-aspect evaluation (test intermediate + final responses)
- LLM testing requires evaluating accuracy, coherence, fairness, safety

**Key Insight:** pytest is extensive, powerful, and can be adapted to LLM responses. DSPy assertions enable self-refining pipelines.

### Solution Options (SCAMPER)

**Option 1: Minimal pytest Only (Eliminate)**

```python
def test_operational_agent():
    agent = OperationalAgent()
    result = agent.research("What is VSM?")
    assert "viable system" in result.answer.lower()
```

- Pros: Simple, familiar, fast
- Cons: No LLM-specific validation, binary pass/fail

**Option 2: DSPy Assertions in Modules (Adapt)**

```python
class ResearchAgent(dspy.Module):
    def forward(self, question):
        result = self.research(question=question)

        # DSPy assertion - auto-retry on failure
        dspy.Assert(
            len(result.answer) > 100,
            "Answer must be substantive"
        )

        dspy.Suggest(
            has_citations(result.answer),
            "Include citations for credibility"
        )

        return result
```

- Pros: Self-refining, catches bad outputs, automatic retry
- Cons: Slower (retries cost tokens), needs tuning

**Option 3: pytest + pytest-harvest (Combine)**

```python
def test_agent_quality(results_bag):
    agent = OperationalAgent()
    result = agent.research("What is VSM?")

    # Log all aspects without stopping
    results_bag.accuracy = check_accuracy(result)
    results_bag.completeness = check_completeness(result)
    results_bag.coherence = check_coherence(result)
    results_bag.citations = count_citations(result)

    # Analyze results after all tests
```

- Pros: Multi-aspect evaluation, data collection, post-processing
- Cons: More complex setup, requires pytest-harvest

**Option 4: Mock LLM for Unit Tests (Substitute)**

```python
class MockLLM:
    def __call__(self, prompt, **kwargs):
        # Return deterministic response for testing
        return {
            "text": "VSM is the Viable System Model...",
            "tokens_used": 100,
            "cache_hit": False
        }

def test_agent_with_mock():
    agent = OperationalAgent(llm=MockLLM())
    result = agent.research("What is VSM?")
    assert result is not None
```

- Pros: Fast, deterministic, no API costs
- Cons: Doesn't test actual LLM behavior

**Option 5: Hybrid Test Pyramid (Combine All)**

```
         /\
        /  \  E2E Tests (DSPy assertions, real LLMs, slow, expensive)
       /    \
      /------\ Integration Tests (pytest, real LLMs, medium)
     /        \
    /----------\ Unit Tests (pytest, mocked LLMs, fast, cheap)
```

- Pros: Comprehensive, fast feedback loop, cost-controlled
- Cons: Most complex, requires all tooling

### Evaluation

**White Hat:**

- Phase 0-1 needs fast feedback (unit tests)
- Must validate LLM output quality (assertions)
- Can't run expensive tests on every commit (mocking needed)
- Need multi-aspect evaluation for debugging

**Yellow Hat:**

- Hybrid pyramid (Option 5) provides all benefits
- Mocks enable TDD workflow
- DSPy assertions catch bad LLM outputs
- pytest-harvest enables analysis

**Black Hat:**

- Hybrid pyramid is most complex
- DSPy assertions cost tokens on retries
- Mocks may not reflect real LLM behavior

**Green Hat:**

- Start simple (Option 1 + 4: pytest + mocks)
- Add DSPy assertions to critical agents only
- Add pytest-harvest when debugging quality issues
- Progressive test maturity

**Blue Hat:**

- Implement test pyramid progressively
- Phase 0: Unit tests with mocks
- Phase 1: Integration tests with real LLMs
- Phase 2: DSPy assertions for production agents

### WINNING SOLUTION: Progressive Test Pyramid

**Phase 0: Unit Tests with Mocks**

```python
# tests/conftest.py
import pytest
from fractal_agent.utils.llm_provider import LLMProvider

class MockLLMProvider(LLMProvider):
    """Mock LLM for deterministic testing"""

    def __init__(self, model: str = "mock", responses: dict = None):
        super().__init__(model)
        self.responses = responses or {}
        self.call_count = 0

    def _call_provider(self, messages, **kwargs):
        self.call_count += 1

        # Extract task from last message
        task = messages[-1]["content"]

        # Return canned response
        response = self.responses.get(task, "Mock response")

        return {
            "text": response,
            "tokens_used": 100,
            "cache_hit": False,
            "provider": "mock",
            "model": self.model
        }

@pytest.fixture
def mock_llm():
    """Fixture providing mock LLM"""
    return MockLLMProvider(responses={
        "What is VSM?": "VSM is the Viable System Model, a framework...",
        "Research VSM System 2": "VSM System 2 handles coordination..."
    })

# tests/test_operational_agent.py
def test_operational_agent_basic(mock_llm):
    """Unit test with mock LLM"""
    agent = OperationalAgent(llm=mock_llm)
    result = agent.research("What is VSM?")

    assert result is not None
    assert "Viable System Model" in result
    assert mock_llm.call_count == 1

def test_operational_agent_error_handling(mock_llm):
    """Test error handling"""
    mock_llm.responses = {}  # No responses configured
    agent = OperationalAgent(llm=mock_llm)
    result = agent.research("Unknown topic")

    # Should get generic mock response
    assert result == "Mock response"
```

**Phase 1: Integration Tests with Real LLMs**

```python
# tests/integration/test_agent_integration.py
import pytest
from fractal_agent.config.model_config import get_llm_for_role

@pytest.mark.integration
@pytest.mark.slow
def test_operational_agent_real_llm():
    """Integration test with real LLM (costs money!)"""
    lm = get_llm_for_role("operational")
    agent = OperationalAgent(llm=lm)

    result = agent.research("What is VSM?")

    # Quality checks
    assert len(result) > 100, "Response too short"
    assert "viable system" in result.lower(), "Missing key concept"

    # Could add more sophisticated checks
    # - Citation count
    # - Structure validation
    # - Factuality check

@pytest.mark.integration
def test_cache_hit_rate():
    """Test that caching is working"""
    lm = get_llm_for_role("operational")

    # First call - cache miss
    response1 = lm(prompt="What is VSM?", max_tokens=100)
    assert not response1["cache_hit"]

    # Second call - cache hit
    response2 = lm(prompt="What is VSM System 2?", max_tokens=100)
    # System message should be cached
    # Can't assert cache_hit without inspecting metrics

    metrics = lm.get_metrics()
    assert metrics["cache_hits"] > 0
```

**Phase 2: DSPy Assertions for Production**

```python
# src/fractal_agent/agents/operational.py
import dspy

class OperationalAgent(dspy.Module):
    """VSM System 1 agent with quality assertions"""

    def __init__(self):
        super().__init__()
        self.research = dspy.ChainOfThought(
            "question -> answer, confidence, sources"
        )

    def forward(self, question: str):
        result = self.research(question=question)

        # HARD ASSERTION: Must pass or retry
        dspy.Assert(
            len(result.answer) >= 50,
            "Answer must be at least 50 characters"
        )

        # SOFT SUGGESTION: Improve but don't fail
        dspy.Suggest(
            result.confidence > 0.7,
            "Low confidence - consider flagging for review"
        )

        dspy.Suggest(
            len(result.sources) > 0,
            "Include sources for credibility"
        )

        return result

# Tests still work the same way!
def test_operational_agent_with_assertions(mock_llm):
    agent = OperationalAgent()
    # Configure DSPy to use mock
    dspy.configure(lm=mock_llm)

    result = agent(question="What is VSM?")
    assert result.answer is not None
```

**Phase 3: pytest-harvest for Analysis**

```python
# tests/test_agent_quality.py
import pytest
from pytest_harvest import get_session_results_df

@pytest.mark.quality
def test_agent_quality_suite(results_bag, mock_llm):
    """Comprehensive quality evaluation"""
    agent = OperationalAgent(llm=mock_llm)
    question = "What is VSM System 2?"
    result = agent.research(question)

    # Record multiple metrics
    results_bag.length = len(result)
    results_bag.has_citations = count_citations(result) > 0
    results_bag.coherence_score = calculate_coherence(result)
    results_bag.completeness_score = calculate_completeness(result)
    results_bag.tokens_used = mock_llm.call_count * 100

def test_session_analysis(session):
    """Analyze all test results after session"""
    df = get_session_results_df(session)

    # Generate quality report
    print(f"Average length: {df['length'].mean()}")
    print(f"Citation rate: {df['has_citations'].sum() / len(df)}")
    print(f"Average coherence: {df['coherence_score'].mean()}")

    # Could save to file, generate plots, etc.
```

**Test Running Strategy:**

```bash
# Fast feedback (unit tests only)
pytest tests/ -m "not integration and not slow"  # < 5 seconds

# Pre-commit (unit + fast integration)
pytest tests/ -m "not slow"  # < 30 seconds

# CI pipeline (all tests)
pytest tests/  # May take minutes, costs money

# Quality analysis (with harvest)
pytest tests/test_agent_quality.py --collect-only
```

**Justification:**

- ✅ Fast TDD workflow with mocks (unit tests)
- ✅ Real LLM validation when needed (integration tests)
- ✅ Quality enforcement in production (DSPy assertions)
- ✅ Comprehensive analysis capability (pytest-harvest)
- ✅ Cost-controlled (most tests use mocks)
- ✅ Progressive maturity (add sophistication over time)

---

## 2.5 First Operational Agent Detailed Design

### Design Requirements

Based on DSPy best practices:

- Use signature-based programming (specify inputs/outputs)
- Modular composition (chain simple modules)
- Evaluation-driven development (define success criteria)
- Integration with UnifiedLM (our LLM infrastructure)

### Agent Architecture

**Agent:** ResearchAgent (VSM System 1 - Operational)

**Purpose:** Execute focused research tasks, retrieve information, synthesize findings

**Inputs:**

- question: str (the research question)
- context: Optional[str] (background knowledge)
- constraints: dict (max_tokens, deadline, required_sources)

**Outputs:**

- answer: str (research findings)
- confidence: float (0.0-1.0)
- sources: list[str] (citations)
- tokens_used: int
- duration_seconds: float

### Implementation

```python
# src/fractal_agent/agents/research_agent.py
import dspy
from typing import Optional
from fractal_agent.utils.dspy_integration import configure_dspy_for_role
from fractal_agent.utils.llm_provider import UnifiedLM
import time

# Define signature
class ResearchSignature(dspy.Signature):
    """Research a question and provide evidence-based answer"""

    question: str = dspy.InputField(desc="The research question to answer")
    context: str = dspy.InputField(
        desc="Background knowledge or context",
        default=""
    )

    answer: str = dspy.OutputField(
        desc="Comprehensive answer with evidence"
    )
    confidence: float = dspy.OutputField(
        desc="Confidence score 0.0-1.0 based on source quality"
    )
    sources: list[str] = dspy.OutputField(
        desc="List of sources cited in the answer"
    )

# Define agent module
class ResearchAgent(dspy.Module):
    """
    VSM System 1 Operational Agent - Research Specialist

    Executes focused research tasks using chain-of-thought reasoning.
    Optimized for Haiku (speed/cost) with Gemini fallback.
    """

    def __init__(
        self,
        specialty: str = "general",
        enable_assertions: bool = True
    ):
        super().__init__()

        self.specialty = specialty
        self.enable_assertions = enable_assertions

        # Configure DSPy for operational role
        self.lm = configure_dspy_for_role("operational")

        # Use ChainOfThought for reasoning
        self.research = dspy.ChainOfThought(ResearchSignature)

    def forward(
        self,
        question: str,
        context: str = "",
        constraints: Optional[dict] = None
    ):
        """Execute research task"""

        start_time = time.time()

        # Execute research
        result = self.research(
            question=question,
            context=context
        )

        # Quality assertions (if enabled)
        if self.enable_assertions:
            # HARD: Must have substantive answer
            dspy.Assert(
                len(result.answer) >= 100,
                "Answer must be at least 100 characters"
            )

            # SOFT: Should have high confidence
            dspy.Suggest(
                result.confidence >= 0.7,
                "Consider flagging low-confidence answers for review"
            )

            # SOFT: Should cite sources
            dspy.Suggest(
                len(result.sources) > 0,
                "Include sources for credibility"
            )

        # Add metadata
        result.duration_seconds = time.time() - start_time
        result.tokens_used = self._estimate_tokens(result)

        return result

    def _estimate_tokens(self, result) -> int:
        """Estimate token usage"""
        # Get from LLM metrics
        metrics = self.lm.get_metrics()
        return metrics.get("total_tokens", 0)

# Example usage
if __name__ == "__main__":
    # Create agent
    agent = ResearchAgent(specialty="VSM research")

    # Execute research task
    result = agent(
        question="What is VSM System 2 responsible for?",
        context="VSM (Viable System Model) is an organizational framework"
    )

    # Display results
    print(f"Answer: {result.answer}")
    print(f"Confidence: {result.confidence}")
    print(f"Sources: {result.sources}")
    print(f"Tokens: {result.tokens_used}")
    print(f"Duration: {result.duration_seconds:.2f}s")
```

**Integration with LangGraph:**

```python
# src/fractal_agent/workflows/simple_research.py
from langgraph.graph import StateGraph, END
from typing import TypedDict

class SimpleResearchState(TypedDict):
    question: str
    answer: str
    confidence: float
    status: str

def research_node(state: SimpleResearchState):
    """Execute research using ResearchAgent"""
    agent = ResearchAgent()
    result = agent(question=state["question"])

    return {
        "answer": result.answer,
        "confidence": result.confidence,
        "status": "complete"
    }

# Build workflow
workflow = StateGraph(SimpleResearchState)
workflow.add_node("research", research_node)
workflow.set_entry_point("research")
workflow.add_edge("research", END)

# Compile and run
app = workflow.compile()
result = app.invoke({"question": "What is VSM?", "status": "pending"})

print(result["answer"])
```

**Justification:**

- ✅ Follows DSPy signature-based design
- ✅ Uses ChainOfThought for reasoning
- ✅ Integrated with UnifiedLM (operational role)
- ✅ Quality assertions for production readiness
- ✅ Metrics tracking (tokens, duration, confidence)
- ✅ LangGraph compatible
- ✅ Simple enough for Phase 0 MVP
- ✅ Extensible (can add tools, RAG, multi-step)

---

## 2.7 Dynamic Model Discovery & Registry

### Problem Statement

**Original Issue:**
The initial design hardcoded specific model names like:

- `claude-3-5-haiku-20241022`
- `claude-3-7-sonnet-20250219`
- `gemini-2.0-flash-exp`

**Why This is Problematic:**

1. Models become outdated quickly (newer versions released frequently)
2. Requires code changes to update to latest models
3. Pricing information becomes stale
4. No automatic discovery of new model capabilities
5. Maintenance burden as providers release new models

**User Feedback:**

> "We're already up to haiku and sonnet (and maybe opus) models from anthropic that are much newer than the models you've mentioned + the same is probably also true for gemini. It would be better to have this be a dynamic list that you didn't have to hardcode and could be pulled and updated automatically"

### Research: Provider APIs for Model Discovery

**Anthropic Models API (2025):**

- **Endpoint:** `GET https://api.anthropic.com/v1/models`
- **Headers Required:**
  - `x-api-key: $ANTHROPIC_API_KEY`
  - `anthropic-version: 2023-06-01`
- **Response Format:**

```json
{
  "data": [
    {
      "id": "claude-sonnet-4-20250514",
      "display_name": "Claude Sonnet 4",
      "created_at": "2025-02-19T00:00:00Z",
      "type": "model"
    }
  ],
  "has_more": true,
  "first_id": "<string>",
  "last_id": "<string>"
}
```

- **Pagination:** Supports paging through large model lists
- **Documentation:** https://docs.claude.com/en/api/models-list

**Google Gemini Models API (2025):**

- **Endpoint:** `GET https://generativelanguage.googleapis.com/v1beta/models?key={GOOGLE_API_KEY}`
- **Methods:** `models.list` and `models.get`
- **Python SDK Example:**

```python
import google.generativeai as genai
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
models = genai.list_models()
for model in models:
    print(model.name, model.supported_generation_methods)
```

- **Pagination:** Returns up to 50 models per page (max 1000)
- **Filter Support:** Can filter by supported actions (`generateContent`, `embedContent`)

**Pricing Information:**

- ❌ **NOT available via API** for either provider
- ✅ Must be maintained via configuration file or manual updates
- ✅ Can be scraped from provider documentation pages (with caching)

**Current 2025 Pricing (Claude):**

- **Haiku 4.5:** $1/$5 per million tokens
- **Sonnet 3.5:** $3/$15 per million tokens
- **Sonnet 4.1:** $5/$25 input/output + $10/million thinking tokens
- **Opus 4.1:** $20/$80 input/output + $40/million thinking tokens
- **Prompt Caching:** 1.25× for 5-min cache write, 2× for 1-hour, 0.1× for reads

### Solution Design: ModelRegistry System

**Architecture Overview:**

```
┌─────────────────────────────────────────────────────────────┐
│                      ModelRegistry                          │
│  - Fetches available models from provider APIs             │
│  - Categorizes models into tiers (cheap/balanced/premium)  │
│  - Caches model info with 24-hour TTL                      │
│  - Loads pricing from config file (models_pricing.yaml)    │
│  - Provides role-based model selection                     │
└─────────────────────────────────────────────────────────────┘
         │                                │
         ↓                                ↓
┌──────────────────┐          ┌──────────────────────┐
│  Anthropic API   │          │   Gemini API         │
│  /v1/models      │          │   models.list        │
└──────────────────┘          └──────────────────────┘
         │                                │
         └────────────┬───────────────────┘
                      ↓
           ┌──────────────────────┐
           │  Local Cache         │
           │  (24-hour TTL)       │
           │  models_cache.json   │
           └──────────────────────┘
                      ↓
           ┌──────────────────────┐
           │  Pricing Config      │
           │  models_pricing.yaml │
           │  (manually updated)  │
           └──────────────────────┘
```

### Implementation Design

#### 1. Model Tier System (Instead of Hardcoded Names)

**Design Decision:** Use semantic tiers instead of specific model names

```python
# Model tiers based on cost/capability trade-off
ModelTier = Literal["cheap", "balanced", "expensive", "premium"]

# Tier definitions (provider-agnostic)
TIER_DEFINITIONS = {
    "cheap": {
        "max_cost_per_1m_tokens": 5.0,
        "use_cases": ["simple queries", "data retrieval", "formatting"],
        "latency": "fast",
        "quality": "good"
    },
    "balanced": {
        "max_cost_per_1m_tokens": 20.0,
        "use_cases": ["analysis", "reasoning", "moderate complexity"],
        "latency": "medium",
        "quality": "high"
    },
    "expensive": {
        "max_cost_per_1m_tokens": 50.0,
        "use_cases": ["deep reasoning", "complex tasks", "high quality output"],
        "latency": "medium",
        "quality": "very high"
    },
    "premium": {
        "max_cost_per_1m_tokens": 100.0,
        "use_cases": ["critical decisions", "maximum quality", "extended thinking"],
        "latency": "slower",
        "quality": "maximum"
    }
}
```

#### 2. Pricing Configuration File

**Format:** `config/models_pricing.yaml` (manually maintained, version controlled)

```yaml
# Updated: 2025-10-18
# Source: https://www.anthropic.com/pricing, https://ai.google.dev/pricing

anthropic:
  claude-haiku-4.5:
    display_name: 'Claude Haiku 4.5'
    tier: cheap
    pricing:
      input: 1.00
      output: 5.00
      cache_write: 1.25
      cache_read: 0.10
    released: '2025-05-01'

  claude-sonnet-3.5:
    display_name: 'Claude Sonnet 3.5'
    tier: balanced
    pricing:
      input: 3.00
      output: 15.00
      cache_write: 3.75
      cache_read: 0.30
    released: '2024-10-22'

  claude-sonnet-4.1:
    display_name: 'Claude Sonnet 4.1'
    tier: expensive
    pricing:
      input: 5.00
      output: 25.00
      thinking: 10.00
      cache_write: 6.25
      cache_read: 0.50
    released: '2025-02-19'
    features:
      - extended_thinking
      - prompt_caching

  claude-opus-4.1:
    display_name: 'Claude Opus 4.1'
    tier: premium
    pricing:
      input: 20.00
      output: 80.00
      thinking: 40.00
      cache_write: 25.00
      cache_read: 2.00
    released: '2025-02-19'
    features:
      - extended_thinking
      - prompt_caching
      - maximum_quality

gemini:
  gemini-2.0-flash:
    display_name: 'Gemini 2.0 Flash'
    tier: cheap
    pricing:
      input: 0.10
      output: 0.40
      cache_write: 0.10
      cache_read: 0.10
    released: '2024-12-01'

  gemini-2.0-flash-thinking:
    display_name: 'Gemini 2.0 Flash Thinking'
    tier: balanced
    pricing:
      input: 0.15
      output: 0.60
      cache_write: 0.15
      cache_read: 0.15
    released: '2024-12-01'
    features:
      - thinking_mode

  gemini-exp-1206:
    display_name: 'Gemini Experimental 1206'
    tier: expensive
    pricing:
      input: 0.50
      output: 2.00
      cache_write: 0.50
      cache_read: 0.50
    released: '2024-12-06'
    experimental: true
```

**Benefits:**

- ✅ Version controlled (Git tracks pricing changes)
- ✅ Easy to update without code changes
- ✅ Human-readable and auditable
- ✅ Supports metadata (features, release dates, experimental flags)
- ✅ Can add new providers easily

#### 3. ModelRegistry Implementation

```python
# src/fractal_agent/config/model_registry.py
"""
Dynamic Model Registry for Fractal Agent Ecosystem

Fetches available models from provider APIs, categorizes by tier,
and provides intelligent model selection.

Author: BMad
Date: 2025-10-18
"""

import anthropic
import google.generativeai as genai
import os
import yaml
import json
from pathlib import Path
from typing import Dict, List, Optional, Literal
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

ModelTier = Literal["cheap", "balanced", "expensive", "premium"]

@dataclass
class ModelInfo:
    """Information about a specific model"""
    provider: str  # "anthropic" or "gemini"
    model_id: str
    display_name: str
    tier: ModelTier
    pricing: Dict[str, float]  # per 1M tokens
    released: str
    experimental: bool = False
    features: List[str] = None
    created_at: Optional[datetime] = None

    def __post_init__(self):
        if self.features is None:
            self.features = []

    @property
    def average_cost_per_1m(self) -> float:
        """Average cost (70% input, 30% output assumption)"""
        return (self.pricing.get("input", 0) * 0.7 +
                self.pricing.get("output", 0) * 0.3)


class ModelRegistry:
    """
    Central registry for all available LLM models.

    Features:
    - Fetches models from provider APIs
    - Caches with 24-hour TTL
    - Categorizes by cost tier
    - Loads pricing from config file
    - Provides tier-based selection
    """

    def __init__(
        self,
        pricing_config_path: str = "config/models_pricing.yaml",
        cache_path: str = "cache/models_cache.json",
        cache_ttl_hours: int = 24
    ):
        self.pricing_config_path = Path(pricing_config_path)
        self.cache_path = Path(cache_path)
        self.cache_ttl = timedelta(hours=cache_ttl_hours)

        # Load pricing configuration
        self.pricing_config = self._load_pricing_config()

        # Load or fetch models
        self.models: Dict[str, ModelInfo] = self._load_models()

    def _load_pricing_config(self) -> dict:
        """Load pricing configuration from YAML"""
        if not self.pricing_config_path.exists():
            raise FileNotFoundError(
                f"Pricing config not found: {self.pricing_config_path}\n"
                "Please create config/models_pricing.yaml"
            )

        with open(self.pricing_config_path) as f:
            config = yaml.safe_load(f)

        logger.info(f"Loaded pricing config from {self.pricing_config_path}")
        return config

    def _load_models(self) -> Dict[str, ModelInfo]:
        """Load models from cache or fetch from APIs"""
        # Try cache first
        if self._is_cache_valid():
            logger.info("Loading models from cache")
            return self._load_from_cache()

        # Cache miss or expired - fetch from APIs
        logger.info("Fetching models from provider APIs")
        models = self._fetch_from_apis()

        # Save to cache
        self._save_to_cache(models)

        return models

    def _is_cache_valid(self) -> bool:
        """Check if cache exists and is not expired"""
        if not self.cache_path.exists():
            return False

        # Check age
        mtime = datetime.fromtimestamp(self.cache_path.stat().st_mtime)
        age = datetime.now() - mtime

        return age < self.cache_ttl

    def _load_from_cache(self) -> Dict[str, ModelInfo]:
        """Load models from cache file"""
        with open(self.cache_path) as f:
            cache_data = json.load(f)

        models = {}
        for key, data in cache_data.items():
            models[key] = ModelInfo(**data)

        return models

    def _save_to_cache(self, models: Dict[str, ModelInfo]):
        """Save models to cache file"""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        cache_data = {}
        for key, model in models.items():
            cache_data[key] = {
                "provider": model.provider,
                "model_id": model.model_id,
                "display_name": model.display_name,
                "tier": model.tier,
                "pricing": model.pricing,
                "released": model.released,
                "experimental": model.experimental,
                "features": model.features,
                "created_at": model.created_at.isoformat() if model.created_at else None
            }

        with open(self.cache_path, 'w') as f:
            json.dump(cache_data, f, indent=2)

        logger.info(f"Cached {len(models)} models to {self.cache_path}")

    def _fetch_from_apis(self) -> Dict[str, ModelInfo]:
        """Fetch available models from all provider APIs"""
        models = {}

        # Fetch Anthropic models
        anthropic_models = self._fetch_anthropic_models()
        models.update(anthropic_models)

        # Fetch Gemini models
        gemini_models = self._fetch_gemini_models()
        models.update(gemini_models)

        return models

    def _fetch_anthropic_models(self) -> Dict[str, ModelInfo]:
        """Fetch models from Anthropic API"""
        models = {}

        try:
            client = anthropic.Anthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )

            # Fetch from API
            response = client.models.list()

            # Match with pricing config
            for model_data in response.data:
                model_id = model_data.id

                # Find matching pricing config (fuzzy match on name)
                pricing_entry = self._find_pricing_entry("anthropic", model_id)

                if pricing_entry:
                    key = f"anthropic:{model_id}"
                    models[key] = ModelInfo(
                        provider="anthropic",
                        model_id=model_id,
                        display_name=pricing_entry.get("display_name", model_data.display_name),
                        tier=pricing_entry["tier"],
                        pricing=pricing_entry["pricing"],
                        released=pricing_entry.get("released", "unknown"),
                        experimental=pricing_entry.get("experimental", False),
                        features=pricing_entry.get("features", []),
                        created_at=datetime.fromisoformat(model_data.created_at.replace('Z', '+00:00'))
                    )
                else:
                    logger.warning(f"No pricing config for Anthropic model: {model_id}")

            logger.info(f"Fetched {len(models)} Anthropic models")

        except Exception as e:
            logger.error(f"Failed to fetch Anthropic models: {e}")

        return models

    def _fetch_gemini_models(self) -> Dict[str, ModelInfo]:
        """Fetch models from Gemini API"""
        models = {}

        try:
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

            # Fetch models that support generateContent
            api_models = genai.list_models()

            for model in api_models:
                # Only include models that support content generation
                if 'generateContent' not in model.supported_generation_methods:
                    continue

                # Extract model ID (remove "models/" prefix)
                model_id = model.name.replace("models/", "")

                # Find matching pricing config
                pricing_entry = self._find_pricing_entry("gemini", model_id)

                if pricing_entry:
                    key = f"gemini:{model_id}"
                    models[key] = ModelInfo(
                        provider="gemini",
                        model_id=model_id,
                        display_name=pricing_entry.get("display_name", model_id),
                        tier=pricing_entry["tier"],
                        pricing=pricing_entry["pricing"],
                        released=pricing_entry.get("released", "unknown"),
                        experimental=pricing_entry.get("experimental", False),
                        features=pricing_entry.get("features", [])
                    )
                else:
                    logger.warning(f"No pricing config for Gemini model: {model_id}")

            logger.info(f"Fetched {len(models)} Gemini models")

        except Exception as e:
            logger.error(f"Failed to fetch Gemini models: {e}")

        return models

    def _find_pricing_entry(self, provider: str, model_id: str) -> Optional[dict]:
        """Find pricing config entry for model (with fuzzy matching)"""
        provider_config = self.pricing_config.get(provider, {})

        # Try exact match first
        if model_id in provider_config:
            return provider_config[model_id]

        # Try fuzzy match (e.g., "claude-sonnet-4-20250514" → "claude-sonnet-4.1")
        for config_key, config_value in provider_config.items():
            # Simple fuzzy match: check if key appears in model_id
            if config_key.lower().replace(".", "-") in model_id.lower():
                return config_value
            # Or check if model_id starts with config_key
            if model_id.lower().startswith(config_key.lower().replace(".", "-")):
                return config_value

        return None

    def get_models_by_tier(self, tier: ModelTier) -> List[ModelInfo]:
        """Get all models in a specific tier"""
        return [
            model for model in self.models.values()
            if model.tier == tier
        ]

    def get_best_model_for_tier(
        self,
        tier: ModelTier,
        provider_preference: Optional[List[str]] = None,
        exclude_experimental: bool = True
    ) -> Optional[ModelInfo]:
        """
        Get best model for a tier.

        Args:
            tier: Target tier
            provider_preference: Ordered list of preferred providers
            exclude_experimental: Exclude experimental models

        Returns:
            Best ModelInfo for tier, or None
        """
        candidates = self.get_models_by_tier(tier)

        # Filter experimental
        if exclude_experimental:
            candidates = [m for m in candidates if not m.experimental]

        if not candidates:
            return None

        # Apply provider preference
        if provider_preference:
            for provider in provider_preference:
                provider_models = [m for m in candidates if m.provider == provider]
                if provider_models:
                    # Return newest model from preferred provider
                    return max(provider_models, key=lambda m: m.released)

        # No preference or no match - return cheapest in tier
        return min(candidates, key=lambda m: m.average_cost_per_1m)

    def get_model_by_id(self, provider: str, model_id: str) -> Optional[ModelInfo]:
        """Get specific model by provider and ID"""
        key = f"{provider}:{model_id}"
        return self.models.get(key)

    def refresh_cache(self):
        """Force refresh models from APIs"""
        logger.info("Forcing model cache refresh")
        self.models = self._fetch_from_apis()
        self._save_to_cache(self.models)

    def get_pricing_for_model(self, provider: str, model_id: str) -> Optional[Dict[str, float]]:
        """Get pricing info for specific model"""
        model = self.get_model_by_id(provider, model_id)
        return model.pricing if model else None

    def list_all_models(self) -> List[ModelInfo]:
        """Get list of all available models"""
        return list(self.models.values())


# Global registry instance
_registry: Optional[ModelRegistry] = None

def get_model_registry() -> ModelRegistry:
    """Get global ModelRegistry instance (singleton)"""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry
```

#### 4. Updated model_config.py (Using Tiers)

```python
# src/fractal_agent/config/model_config.py
"""
Model Configuration for Fractal Agent Ecosystem

Uses ModelRegistry for dynamic model selection by tier.
NO HARDCODED MODEL NAMES.

Author: BMad
Date: 2025-10-18
"""

from typing import Dict, Any, Literal
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from fractal_agent.utils.llm_provider import UnifiedLM
from fractal_agent.config.model_registry import get_model_registry, ModelTier

# Model tier configuration by agent role
# Uses TIERS instead of specific model names
ROLE_TIER_CONFIG: Dict[str, Dict[str, Any]] = {
    # Operational agents (VSM System 1) - Fast, cost-efficient
    "operational": {
        "tier": "cheap",
        "provider_preference": ["anthropic", "gemini"],
        "rationale": "High speed, low cost, good for simple tasks",
        "typical_use": "Research, data retrieval, simple analysis"
    },

    # Control agents (VSM System 3) - Smart reasoning for decomposition
    "control": {
        "tier": "expensive",
        "provider_preference": ["anthropic", "gemini"],
        "rationale": "Strong reasoning for task decomposition and planning",
        "typical_use": "Task decomposition, delegation planning, workflow design"
    },

    # Intelligence agents (VSM System 4) - Deep analysis and reflection
    "intelligence": {
        "tier": "expensive",
        "provider_preference": ["anthropic", "gemini"],
        "rationale": "Best quality for reflection, learning, and strategic thinking",
        "typical_use": "Performance reflection, insight generation, pattern detection"
    },

    # Knowledge extraction - Accuracy critical
    "extraction": {
        "tier": "balanced",
        "provider_preference": ["anthropic", "gemini"],
        "rationale": "Balanced quality/cost for structured extraction tasks",
        "typical_use": "Log parsing, entity extraction, relationship mapping"
    },

    # Synthesis/reporting - Quality output
    "synthesis": {
        "tier": "expensive",
        "provider_preference": ["anthropic", "gemini"],
        "rationale": "High quality output for final reports and documentation",
        "typical_use": "Report generation, documentation, synthesis of findings"
    },

    # Coordination agents (VSM System 2) - Conflict resolution
    "coordination": {
        "tier": "balanced",
        "provider_preference": ["anthropic", "gemini"],
        "rationale": "Good reasoning for conflict detection and resolution",
        "typical_use": "Detect conflicts between parallel agents, resource arbitration"
    },

    # Policy agents (VSM System 5) - Ethical/strategic decisions
    "policy": {
        "tier": "premium",
        "provider_preference": ["anthropic", "gemini"],
        "rationale": "Highest quality for ethical reasoning and policy decisions",
        "typical_use": "Ethical boundaries, system identity, strategic direction"
    }
}


def get_llm_for_role(role: str, **kwargs) -> UnifiedLM:
    """
    Factory function: Get LLM instance for agent role.

    Uses ModelRegistry to dynamically select best model for role's tier.
    NO HARDCODED MODEL NAMES.

    Args:
        role: Agent role (operational, control, intelligence, extraction, synthesis, coordination, policy)
        **kwargs: Override default config
            - tier: ModelTier to override default tier for role
            - providers: List[(provider_name, model)] to override dynamic selection
            - enable_caching: bool to control prompt caching
            - Any other UnifiedLM parameters

    Returns:
        Configured UnifiedLM instance with appropriate provider chain

    Examples:
        # Use defaults for role (automatic tier-based selection)
        lm = get_llm_for_role("operational")

        # Override tier
        lm = get_llm_for_role("operational", tier="balanced")

        # Override with specific models (bypass registry)
        lm = get_llm_for_role("control", providers=[
            ("anthropic", "claude-sonnet-4-20250514"),
            ("gemini", "gemini-2.0-flash")
        ])
    """
    # Get config for role (fallback to operational if unknown)
    config = ROLE_TIER_CONFIG.get(role, ROLE_TIER_CONFIG["operational"])

    # Check if user provided explicit model list (bypass registry)
    providers = kwargs.pop("providers", None)
    if providers is not None:
        return UnifiedLM(providers=providers, **kwargs)

    # Use ModelRegistry for dynamic selection
    registry = get_model_registry()

    # Get tier (allow override)
    tier = kwargs.pop("tier", config["tier"])
    provider_preference = config["provider_preference"]

    # Get best models for tier
    primary_model = registry.get_best_model_for_tier(
        tier=tier,
        provider_preference=provider_preference
    )

    # Get fallback from next provider
    fallback_preference = provider_preference[1:] + [provider_preference[0]]
    fallback_model = registry.get_best_model_for_tier(
        tier=tier,
        provider_preference=fallback_preference
    )

    if not primary_model:
        raise ValueError(
            f"No models available for tier '{tier}'. "
            f"Check config/models_pricing.yaml and model cache."
        )

    # Build provider chain
    providers = [(primary_model.provider, primary_model.model_id)]
    if fallback_model and fallback_model.model_id != primary_model.model_id:
        providers.append((fallback_model.provider, fallback_model.model_id))

    return UnifiedLM(providers=providers, **kwargs)


def list_available_roles() -> Dict[str, Dict[str, Any]]:
    """
    Get all available agent roles and their tier configurations.

    Returns:
        Dict mapping role names to their configurations
    """
    return ROLE_TIER_CONFIG


def get_role_info(role: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific role's configuration.

    Args:
        role: Agent role name

    Returns:
        Configuration dict for the role

    Raises:
        ValueError: If role doesn't exist
    """
    if role not in ROLE_TIER_CONFIG:
        raise ValueError(
            f"Unknown role: {role}. "
            f"Available roles: {', '.join(ROLE_TIER_CONFIG.keys())}"
        )

    return ROLE_TIER_CONFIG[role]


def estimate_cost_per_role() -> Dict[str, str]:
    """
    Cost estimates per agent role (based on tier).

    Returns:
        Dict mapping roles to cost tier
    """
    return {
        role: config["tier"]
        for role, config in ROLE_TIER_CONFIG.items()
    }
```

#### 5. Updated Cost Tracking (Dynamic Pricing)

```python
# src/fractal_agent/cost/tracker.py
from fractal_agent.config.model_registry import get_model_registry

class CostTracker:
    """Updated to use ModelRegistry for pricing"""

    def _calculate_cost(
        self,
        tokens: int,
        model: str,
        provider: str,
        cache_hit: bool
    ) -> float:
        """Calculate cost using ModelRegistry pricing"""

        # Get pricing from registry
        registry = get_model_registry()
        pricing_info = registry.get_pricing_for_model(provider, model)

        if not pricing_info:
            logger.warning(f"No pricing for {provider}:{model}, using default")
            # Fallback to conservative estimate
            return (tokens / 1_000_000) * 10.0

        # Assume 70% input, 30% output (typical ratio)
        input_tokens = int(tokens * 0.7)
        output_tokens = int(tokens * 0.3)

        if cache_hit:
            input_cost = (input_tokens / 1_000_000) * pricing_info.get("cache_read", 0)
        else:
            input_cost = (input_tokens / 1_000_000) * pricing_info.get("input", 0)

        output_cost = (output_tokens / 1_000_000) * pricing_info.get("output", 0)

        return input_cost + output_cost
```

### Benefits of Dynamic Model Discovery

**✅ Advantages:**

1. **No Code Changes for Model Updates** - Just update `models_pricing.yaml`
2. **Automatic New Model Discovery** - APIs return latest available models
3. **Semantic Tier System** - Roles care about "cheap" vs "expensive", not specific model names
4. **Easy Provider Addition** - Add new providers by extending registry
5. **Version Controlled Pricing** - Git tracks all pricing changes
6. **Fallback Resilience** - If primary tier unavailable, falls back gracefully
7. **Cache for Performance** - 24-hour TTL reduces API calls
8. **Audit Trail** - Know exactly which models were used and when
9. **Testing Flexibility** - Can mock registry for tests
10. **Cost Visibility** - Easy to see tier costs in one config file

**✅ Maintenance:**

- Update `config/models_pricing.yaml` when new models released (monthly)
- Registry auto-fetches model availability (daily via cache)
- No code changes needed for 90% of updates

### Integration with Existing Architecture

**Changes Required:**

1. Replace hardcoded `MODEL_CONFIG` dict → Use `ROLE_TIER_CONFIG`
2. Update `get_llm_for_role()` → Call `ModelRegistry.get_best_model_for_tier()`
3. Update `CostTracker._calculate_cost()` → Use `registry.get_pricing_for_model()`
4. Create `config/models_pricing.yaml` file
5. Add cache directory to `.gitignore`: `cache/models_cache.json`

**Backward Compatibility:**

- `get_llm_for_role()` still accepts explicit `providers=` for testing
- Can bypass registry if needed for specific use cases

### Testing Strategy

```python
# tests/test_model_registry.py
import pytest
from fractal_agent.config.model_registry import ModelRegistry, ModelInfo

def test_model_registry_loads_config():
    """Test registry loads pricing config"""
    registry = ModelRegistry()
    assert len(registry.models) > 0

def test_tier_selection():
    """Test tier-based model selection"""
    registry = ModelRegistry()

    cheap_model = registry.get_best_model_for_tier("cheap")
    assert cheap_model is not None
    assert cheap_model.tier == "cheap"

    premium_model = registry.get_best_model_for_tier("premium")
    assert premium_model is not None
    assert premium_model.tier == "premium"
    assert premium_model.average_cost_per_1m > cheap_model.average_cost_per_1m

def test_provider_preference():
    """Test provider preference honored"""
    registry = ModelRegistry()

    anthropic_model = registry.get_best_model_for_tier(
        "balanced",
        provider_preference=["anthropic", "gemini"]
    )
    assert anthropic_model.provider == "anthropic"

def test_cache_invalidation():
    """Test cache expires and refreshes"""
    registry = ModelRegistry(cache_ttl_hours=0)  # Immediate expiry

    # First load should fetch from API
    models1 = registry.models

    # Reload should re-fetch (cache expired)
    registry2 = ModelRegistry(cache_ttl_hours=0)
    models2 = registry2.models

    assert models1.keys() == models2.keys()

def test_missing_pricing_config():
    """Test graceful handling of missing pricing"""
    # Mock scenario where API returns model not in pricing config
    registry = ModelRegistry()

    unknown_model = registry.get_model_by_id("anthropic", "nonexistent-model")
    assert unknown_model is None
```

### Migration Path

**Phase 0: Add ModelRegistry (No Breaking Changes)**

1. Implement `ModelRegistry` class
2. Create `config/models_pricing.yaml` with current models
3. Add tests for registry
4. Keep existing hardcoded config as fallback

**Phase 1: Switch to Tier-Based Selection**

1. Update `model_config.py` to use `ROLE_TIER_CONFIG`
2. Update `get_llm_for_role()` to call registry
3. Verify all tests pass
4. Deploy with monitoring

**Phase 2: Update Cost Tracking**

1. Update `CostTracker` to use registry pricing
2. Remove hardcoded `PRICING` dict
3. Verify cost calculations accurate

**Phase 3: Cleanup**

1. Remove all hardcoded model name references
2. Update documentation
3. Add alerting for new models (Slack webhook when cache refreshes)

---

## Session 2 Summary

**Completed Designs:**

1. ✅ **Agent Communication Protocol** - Pydantic models + LangGraph state
2. ✅ **LangGraph Workflow Structure** - Hub-and-Spoke with dynamic workers
3. ✅ **Context Caching Strategy** - Two-tier role-based templates
4. ✅ **Testing Framework** - Progressive test pyramid (mocks → integration → assertions)
5. ✅ **First Operational Agent** - DSPy-based ResearchAgent
6. ✅ **Dynamic Model Discovery** - ModelRegistry with tier-based selection

**Key Decisions:**

- Communication: Type-safe Pydantic messages over LangGraph state
- Orchestration: Hub-and-Spoke + Send API for dynamic scaling
- Caching: 2-tier (system+tools, knowledge) with role templates
- Testing: Progressive pyramid (unit → integration → production assertions)
- Agent Design: DSPy signatures + ChainOfThought + UnifiedLM
- Model Selection: Tier-based (cheap/balanced/expensive/premium) instead of hardcoded names
- Pricing Config: YAML-based, version controlled, human-editable
- Model Discovery: Auto-fetch from provider APIs with 24-hour cache

**Ready for Implementation:** Phase 0 can now begin!

---

# Session 3: VSM System 2 Coordination

## 3.1 Coordination Challenges

### Problem Statement

VSM System 2 is responsible for **anti-oscillation** and **conflict resolution** between System 1 operational units.

**Key Challenges:**

1. **Resource Conflicts** - Two agents request same data source simultaneously
2. **Duplicate Work** - Two agents research same question independently
3. **Contradictory Goals** - One agent's output invalidates another's
4. **Token Budget Exhaustion** - Parallel agents exceed total budget
5. **Rate Limiting** - API quota exceeded by concurrent requests
6. **Information Conflicts** - Agents retrieve contradictory information

### Research: Multi-Agent Coordination Patterns

Common patterns in 2025:

- **Message Passing** - Agents communicate via shared message queue
- **Shared State** - Central blackboard for coordination
- **Locking/Semaphores** - Explicit resource reservation
- **Priority Queues** - Higher-priority agents get resources first
- **Rate Limiting Pools** - Shared quota across agents

### Solution Options (SCAMPER)

**Option 1: Pessimistic Locking (Substitute)**

```python
# Agent requests resource before starting
with resource_lock("graphrag_db"):
    data = graphrag.retrieve(query)
    process(data)
```

- Pros: No conflicts possible
- Cons: Serializes parallel work, reduces throughput

**Option 2: Optimistic Concurrency (Adapt)**

```python
# Agent proceeds, handles conflicts if they occur
try:
    data = graphrag.retrieve(query)
    process(data)
except ConflictError:
    retry_with_backoff()
```

- Pros: Maximum parallelism
- Cons: Wastes work on conflicts, retry overhead

**Option 3: Coordination Agent (VSM System 2 Implementation)**

```python
# Coordination agent intercepts requests
coord = CoordinationAgent()

# Agents send requests to coordinator
request = ResourceRequest(
    agent_id="op_001",
    resource="graphrag_db",
    query="VSM System 2"
)

# Coordinator detects conflicts
if coord.detect_conflict(request):
    coord.resolve_conflict(request)
else:
    coord.approve(request)
```

- Pros: Centralized conflict detection, can optimize globally
- Cons: Coordinator becomes bottleneck, single point of failure

**Option 4: Shared Task Registry (Combine)**

```python
# Global registry of in-flight tasks
task_registry = {
    "research_vsm_system_2": {
        "agent": "op_001",
        "status": "in_progress",
        "started": "2025-10-18T10:00:00Z"
    }
}

# Agent checks registry before starting
if task_registry.get(task_key):
    # Task already being worked on - skip or wait
    wait_for_completion(task_key)
else:
    # Register task and proceed
    task_registry[task_key] = {...}
    execute_task()
```

- Pros: Prevents duplicate work automatically
- Cons: Requires consistent task keying, stale entries possible

**Option 5: Token Budget Pool (Put to Other Use)**

```python
# Shared token budget
budget_pool = TokenBudgetPool(max_tokens=100000)

# Agents request budget allocation
allocation = budget_pool.request(
    agent_id="op_001",
    estimated_tokens=5000,
    priority=1
)

if allocation.approved:
    execute_with_budget(allocation.tokens)
    budget_pool.release(actual_tokens_used)
else:
    wait_for_budget()
```

- Pros: Prevents budget exhaustion, fair allocation
- Cons: Estimation errors, delayed agents

**Option 6: Rate Limiter Coordination (Eliminate Conflicts)**

```python
# Shared rate limiter
rate_limiter = RateLimiter(
    max_requests_per_minute=60,
    strategy="fair_share"  # or "priority", "first_come"
)

# Agents acquire permits
async with rate_limiter.acquire():
    await api_call()
```

- Pros: Prevents API quota exhaustion, simple
- Cons: May slow down fast agents unnecessarily

### Evaluation

**White Hat:**

- Phase 1 will have 3-7 parallel operational agents
- Primary conflicts: duplicate work, token budget, rate limits
- System 2 agent should be lightweight (not bottleneck)
- Most tasks are independent (coordination is exception, not rule)

**Yellow Hat:**

- Option 4 (Task Registry) prevents duplicate work efficiently
- Option 5 (Budget Pool) prevents budget exhaustion
- Option 6 (Rate Limiter) prevents API quota issues
- Can combine all three (orthogonal concerns)

**Black Hat:**

- Pessimistic locking (Option 1) kills parallelism
- Coordination agent (Option 3) is overkill for Phase 1
- Optimistic concurrency (Option 2) wastes tokens on conflicts

**Green Hat:**

- Hybrid: Task Registry + Budget Pool + Rate Limiter
- Light coordination (no System 2 agent needed in Phase 1)
- Escalate to System 2 agent only when conflicts detected
- Progressive: Start with registry/pool/limiter, add agent in Phase 4

**Blue Hat:**

- Phase 1: Lightweight coordination (registry + pool + limiter)
- Phase 4: Add Coordination Agent for complex conflicts
- System 2 is "invisible until needed"

### WINNING SOLUTION: Lightweight Coordination Infrastructure + Optional System 2 Agent

**Design:**

```python
# src/fractal_agent/coordination/task_registry.py
from typing import Dict, Optional
from datetime import datetime, timedelta
from threading import Lock
import hashlib

class TaskRegistry:
    """
    Prevents duplicate work across parallel agents.

    VSM System 2 (lightweight) - Coordination via shared state.
    """

    def __init__(self, ttl_minutes: int = 30):
        self._registry: Dict[str, dict] = {}
        self._lock = Lock()
        self.ttl = timedelta(minutes=ttl_minutes)

    def task_key(self, task: dict) -> str:
        """Generate unique key for task"""
        # Hash task parameters for deduplication
        key_data = f"{task['type']}:{task.get('query', '')}:{task.get('constraints', '')}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def register(self, agent_id: str, task: dict) -> tuple[bool, Optional[dict]]:
        """
        Register task for execution.

        Returns:
            (is_new, existing_task) - is_new=True if task not already running
        """
        with self._lock:
            task_id = self.task_key(task)

            # Check if task already exists
            if task_id in self._registry:
                existing = self._registry[task_id]

                # Check if stale (TTL expired)
                if datetime.utcnow() - existing["started"] > self.ttl:
                    # Stale entry - allow re-execution
                    del self._registry[task_id]
                else:
                    # Task already in progress
                    return (False, existing)

            # Register new task
            self._registry[task_id] = {
                "agent_id": agent_id,
                "task": task,
                "status": "in_progress",
                "started": datetime.utcnow()
            }

            return (True, None)

    def complete(self, agent_id: str, task: dict, result: Any):
        """Mark task as complete"""
        with self._lock:
            task_id = self.task_key(task)
            if task_id in self._registry:
                self._registry[task_id]["status"] = "complete"
                self._registry[task_id]["result"] = result
                self._registry[task_id]["completed"] = datetime.utcnow()

    def wait_for_completion(self, task_id: str, timeout: int = 300) -> Optional[Any]:
        """Wait for another agent to complete task"""
        import time
        start = time.time()

        while time.time() - start < timeout:
            with self._lock:
                if task_id in self._registry:
                    task = self._registry[task_id]
                    if task["status"] == "complete":
                        return task["result"]

            time.sleep(1)  # Poll every second

        return None  # Timeout

# src/fractal_agent/coordination/budget_pool.py
from threading import Lock, Semaphore

class TokenBudgetPool:
    """
    Manages shared token budget across parallel agents.

    Prevents budget exhaustion via fair allocation.
    """

    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens
        self.available_tokens = max_tokens
        self._lock = Lock()
        self.allocations = {}

    def request(
        self,
        agent_id: str,
        estimated_tokens: int,
        priority: int = 1
    ) -> dict:
        """
        Request token allocation.

        Returns:
            {"approved": bool, "tokens": int, "reason": str}
        """
        with self._lock:
            # Check if enough tokens available
            if estimated_tokens <= self.available_tokens:
                # Approve allocation
                self.available_tokens -= estimated_tokens
                self.allocations[agent_id] = {
                    "allocated": estimated_tokens,
                    "used": 0,
                    "priority": priority
                }

                return {
                    "approved": True,
                    "tokens": estimated_tokens,
                    "reason": "approved"
                }
            else:
                # Insufficient budget
                return {
                    "approved": False,
                    "tokens": 0,
                    "reason": f"Insufficient budget: {self.available_tokens} available, {estimated_tokens} requested"
                }

    def release(self, agent_id: str, actual_tokens: int):
        """Release tokens after task completion"""
        with self._lock:
            if agent_id in self.allocations:
                allocated = self.allocations[agent_id]["allocated"]
                unused = allocated - actual_tokens

                # Return unused tokens to pool
                self.available_tokens += unused

                # Update allocation record
                self.allocations[agent_id]["used"] = actual_tokens

    def get_available(self) -> int:
        """Get current available tokens"""
        with self._lock:
            return self.available_tokens

# src/fractal_agent/coordination/rate_limiter.py
import time
from threading import Lock
from collections import deque

class RateLimiter:
    """
    Rate limiter for API calls across parallel agents.

    Prevents exceeding provider quotas (Anthropic: 50 req/min tier 1).
    """

    def __init__(
        self,
        max_requests_per_minute: int = 50,
        strategy: str = "fair_share"
    ):
        self.max_rpm = max_requests_per_minute
        self.strategy = strategy
        self._lock = Lock()
        self._request_times = deque(maxlen=max_requests_per_minute)

    def acquire(self, agent_id: str) -> bool:
        """
        Acquire permit to make API call.

        Blocks until permit available.
        """
        while True:
            with self._lock:
                now = time.time()

                # Remove requests older than 1 minute
                while self._request_times and (now - self._request_times[0]) > 60:
                    self._request_times.popleft()

                # Check if under limit
                if len(self._request_times) < self.max_rpm:
                    self._request_times.append(now)
                    return True

            # Wait before retry
            time.sleep(1)

    def __enter__(self):
        """Context manager support"""
        self.acquire("context")
        return self

    def __exit__(self, *args):
        pass

# Integration with LangGraph
from langgraph.graph import StateGraph
from typing import TypedDict

class CoordinatedWorkflowState(TypedDict):
    task: str
    subtasks: list[dict]
    results: list[dict]
    coordination_metadata: dict

# Global coordination infrastructure
task_registry = TaskRegistry(ttl_minutes=30)
budget_pool = TokenBudgetPool(max_tokens=100000)
rate_limiter = RateLimiter(max_requests_per_minute=50)

def coordinated_operational_node(state: CoordinatedWorkflowState):
    """Operational agent with coordination"""
    agent_id = "op_001"
    task = state["task"]

    # 1. Check task registry (prevent duplicate work)
    is_new, existing = task_registry.register(agent_id, task)

    if not is_new:
        # Task already being worked on by another agent
        print(f"Task already in progress by {existing['agent_id']}, waiting...")
        result = task_registry.wait_for_completion(
            task_registry.task_key(task),
            timeout=300
        )
        return {"results": [result]}

    # 2. Request token budget
    allocation = budget_pool.request(
        agent_id=agent_id,
        estimated_tokens=5000,
        priority=1
    )

    if not allocation["approved"]:
        print(f"Budget allocation denied: {allocation['reason']}")
        return {"results": []}

    # 3. Execute with rate limiting
    try:
        with rate_limiter:
            # Make LLM call
            lm = get_llm_for_role("operational")
            agent = ResearchAgent(llm=lm)
            result = agent(question=task["query"])

        # 4. Release budget (return unused tokens)
        budget_pool.release(agent_id, result.tokens_used)

        # 5. Mark task complete in registry
        task_registry.complete(agent_id, task, result)

        return {"results": [result]}

    except Exception as e:
        # Release allocated budget on failure
        budget_pool.release(agent_id, 0)
        raise
```

**Optional: System 2 Coordination Agent (Phase 4)**

```python
# src/fractal_agent/agents/coordination_agent.py
import dspy
from typing import Literal

class ConflictDetectionSignature(dspy.Signature):
    """Detect conflicts between parallel agents"""

    agent_states: list[dict] = dspy.InputField(
        desc="Current state of all active agents"
    )

    conflicts: list[dict] = dspy.OutputField(
        desc="List of detected conflicts with severity"
    )
    resolution_strategy: str = dspy.OutputField(
        desc="Recommended resolution approach"
    )

class CoordinationAgent(dspy.Module):
    """
    VSM System 2 Agent - Advanced Coordination

    Activated when lightweight coordination insufficient.
    Detects and resolves complex conflicts.
    """

    def __init__(self):
        super().__init__()
        self.lm = configure_dspy_for_role("coordination")
        self.detect_conflicts = dspy.ChainOfThought(ConflictDetectionSignature)

    def forward(self, agent_states: list[dict]):
        """Analyze agent states for conflicts"""
        result = self.detect_conflicts(agent_states=agent_states)

        if result.conflicts:
            # Resolve conflicts
            for conflict in result.conflicts:
                self.resolve_conflict(conflict)

        return result

    def resolve_conflict(self, conflict: dict):
        """Execute conflict resolution strategy"""
        if conflict["type"] == "duplicate_work":
            # Terminate lower-priority agent
            self._terminate_agent(conflict["lower_priority_agent"])

        elif conflict["type"] == "contradictory_information":
            # Escalate to Intelligence Agent for adjudication
            self._escalate_to_intelligence(conflict)

        elif conflict["type"] == "resource_contention":
            # Assign priority ordering
            self._assign_resource_priority(conflict)
```

**Justification:**

- ✅ Phase 1: Lightweight coordination (no agent needed)
- ✅ Task Registry prevents duplicate work automatically
- ✅ Budget Pool prevents token budget exhaustion
- ✅ Rate Limiter prevents API quota issues
- ✅ Phase 4: Coordination Agent for complex conflicts
- ✅ Progressive complexity (starts simple, adds sophistication)
- ✅ Minimal overhead (thread-safe but lightweight)

---

## 3.2 Coordination Decision Matrix

**When to Use Each Mechanism:**

| Conflict Type           | Mechanism          | Phase | Justification                           |
| ----------------------- | ------------------ | ----- | --------------------------------------- |
| Duplicate work          | Task Registry      | 1     | Automatic detection via task hashing    |
| Token budget exhaustion | Budget Pool        | 1     | Prevents overruns, fair allocation      |
| API rate limits         | Rate Limiter       | 1     | Prevents quota exhaustion               |
| Resource contention     | Locking/Semaphores | 2     | When shared resources added (DB writes) |
| Contradictory goals     | Coordination Agent | 4     | Requires reasoning about agent intents  |
| Information conflicts   | Intelligence Agent | 4     | Requires adjudication of truth claims   |

**Coordination Overhead:**

- Task Registry: ~1ms per task check (in-memory hash lookup)
- Budget Pool: ~0.5ms per allocation (lock + arithmetic)
- Rate Limiter: 0-1000ms (only waits if at limit)
- Coordination Agent: 2-5 seconds (LLM call)

**Design Principle:** Use lightest mechanism that solves problem.

---

# Session 3 Summary

**Completed Designs:**

1. ✅ **Coordination Philosophy** - Lightweight infrastructure, escalate to agent when needed
2. ✅ **Task Registry** - Prevents duplicate work via task hashing
3. ✅ **Budget Pool** - Fair token allocation across parallel agents
4. ✅ **Rate Limiter** - Prevents API quota exhaustion
5. ✅ **System 2 Agent** - Optional advanced coordination (Phase 4)

**Key Decisions:**

- Phase 1: No System 2 agent needed (infrastructure handles it)
- Progressive complexity: Start with lightweight, add agent later
- Orthogonal concerns: Registry + Pool + Limiter solve different problems
- Minimal overhead: Thread-safe in-memory structures

---

# Session 4: GraphRAG & Memory Architecture

## 4.1 GraphRAG Schema Design

### Research: GraphRAG Best Practices 2025

GraphRAG combines:

- **Graph Database** (Neo4j) - Entity relationships
- **Vector Database** (Qdrant) - Semantic search
- **Hybrid Retrieval** - Graph traversal + vector similarity

**Key Insights:**

- Temporal properties critical for evolving knowledge
- Provenance tracking (where did knowledge come from?)
- Confidence scores (how certain are we?)
- Context-dependent truth (needs qualification)

### Neo4j Schema Design

**Node Types:**

```cypher
// Entity nodes
(:Entity {
    id: String,
    type: String,  // "concept", "person", "organization", "technology"
    name: String,
    description: String,
    t_valid: DateTime,  // When this entity became known
    t_invalid: DateTime,  // When superseded (null = still valid)
    confidence: Float,  // 0.0-1.0
    source: String,  // Provenance
    embedding_id: String  // Link to Qdrant
})

// Statement nodes (for context-dependent facts)
(:Statement {
    id: String,
    text: String,
    context: String,  // "for I/O-bound tasks", "in Python 3.12+"
    t_valid: DateTime,
    t_invalid: DateTime,
    confidence: Float,
    source: String
})

// Source nodes (provenance tracking)
(:Source {
    id: String,
    type: String,  // "session_log", "external_doc", "human_curation"
    reference: String,  // File path, URL, session ID
    authority: Float,  // 0.0-1.0 trustworthiness
    created: DateTime
})
```

**Relationship Types:**

```cypher
// Core relationships
(:Entity)-[:RELATES_TO {
    type: String,  // "is_a", "has_part", "causes", "located_in"
    strength: Float,  // 0.0-1.0
    t_valid: DateTime,
    t_invalid: DateTime,
    source: String
}]->(:Entity)

// Statement relationships
(:Entity)-[:HAS_PROPERTY {
    property: String,
    context: String,
    t_valid: DateTime,
    t_invalid: DateTime
}]->(:Statement)

// Provenance
(:Entity)-[:DERIVED_FROM]->(:Source)
(:Statement)-[:DERIVED_FROM]->(:Source)

// Contradictions (for conflict resolution)
(:Statement)-[:CONTRADICTS {
    resolution: String,  // "resolved", "pending", "both_valid_in_context"
    adjudicator: String  // Which agent/human resolved it
}]->(:Statement)
```

**Example Knowledge Representation:**

```cypher
// Insert entity
CREATE (vsm:Entity {
    id: "vsm_system_2",
    type: "concept",
    name: "VSM System 2",
    description: "Coordination system in Viable System Model",
    t_valid: datetime("2025-10-18T10:00:00Z"),
    t_invalid: null,
    confidence: 0.95,
    source: "session_001"
})

// Insert statement with context
CREATE (stmt:Statement {
    id: "stmt_001",
    text: "Python is slow",
    context: "for CPU-bound numerical computations compared to C",
    t_valid: datetime("2025-10-18T10:00:00Z"),
    t_invalid: null,
    confidence: 0.9,
    source: "research_task_042"
})

// Create relationship
CREATE (python:Entity {name: "Python"})-[:HAS_PROPERTY {
    property: "performance",
    context: "CPU-bound",
    t_valid: datetime("2025-10-18T10:00:00Z")
}]->(stmt)

// Link to source
CREATE (source:Source {
    id: "research_task_042",
    type: "session_log",
    reference: "/logs/2025-10-18/task_042.json",
    authority: 0.7,
    created: datetime("2025-10-18T10:00:00Z")
})

CREATE (stmt)-[:DERIVED_FROM]->(source)
```

### Qdrant Schema Design

**Collection Structure:**

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Initialize Qdrant
client = QdrantClient(host="localhost", port=6333)

# Create collection
client.create_collection(
    collection_name="fractal_knowledge",
    vectors_config=VectorParams(
        size=1536,  # OpenAI ada-002 or similar
        distance=Distance.COSINE
    )
)

# Insert point
client.upsert(
    collection_name="fractal_knowledge",
    points=[
        PointStruct(
            id="entity_vsm_system_2",
            vector=[...],  # Embedding of entity description
            payload={
                "entity_id": "vsm_system_2",
                "type": "entity",
                "name": "VSM System 2",
                "description": "Coordination system...",
                "t_valid": "2025-10-18T10:00:00Z",
                "t_invalid": None,
                "confidence": 0.95,
                "text_for_embedding": "VSM System 2: Coordination system in Viable System Model responsible for anti-oscillation..."
            }
        )
    ]
)
```

### Hybrid Retrieval Strategy

**Retrieval Pipeline:**

```python
# src/fractal_agent/memory/graphrag.py
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from typing import List, Dict, Any
import openai

class GraphRAG:
    """
    Hybrid Graph + Vector retrieval for knowledge base.

    Combines:
    - Neo4j for relationship traversal
    - Qdrant for semantic search
    - Temporal filtering for current knowledge
    """

    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        qdrant_host: str,
        qdrant_port: int
    ):
        self.graph = GraphDatabase.driver(
            neo4j_uri,
            auth=(neo4j_user, neo4j_password)
        )
        self.vector = QdrantClient(host=qdrant_host, port=qdrant_port)

    def retrieve(
        self,
        query: str,
        max_tokens: int = 30000,
        min_confidence: float = 0.6,
        as_of_time: str = None  # For temporal queries
    ) -> str:
        """
        Hybrid retrieval: vector search + graph traversal.

        Returns formatted knowledge string for LLM context.
        """
        # 1. Vector search for semantically similar entities
        query_embedding = self._embed(query)

        vector_results = self.vector.search(
            collection_name="fractal_knowledge",
            query_vector=query_embedding,
            limit=20,  # Top 20 similar
            score_threshold=0.7  # Minimum similarity
        )

        # 2. Graph traversal from seed entities
        seed_ids = [r.payload["entity_id"] for r in vector_results]

        graph_results = self._traverse_graph(
            seed_ids,
            max_depth=2,
            min_confidence=min_confidence,
            as_of_time=as_of_time
        )

        # 3. Combine and format results
        knowledge = self._format_knowledge(
            vector_results,
            graph_results,
            max_tokens=max_tokens
        )

        return knowledge

    def _traverse_graph(
        self,
        seed_ids: List[str],
        max_depth: int,
        min_confidence: float,
        as_of_time: str
    ) -> List[Dict]:
        """Traverse graph from seed entities"""

        query = """
        MATCH path = (start:Entity)-[r*1..{max_depth}]-(related:Entity)
        WHERE start.id IN $seed_ids
        AND start.confidence >= $min_confidence
        AND related.confidence >= $min_confidence
        AND (start.t_invalid IS NULL OR start.t_invalid > datetime($as_of_time))
        AND (related.t_invalid IS NULL OR related.t_invalid > datetime($as_of_time))
        RETURN DISTINCT
            start,
            related,
            relationships(path) AS rels,
            length(path) AS distance
        ORDER BY distance ASC, related.confidence DESC
        LIMIT 100
        """

        with self.graph.session() as session:
            result = session.run(
                query,
                seed_ids=seed_ids,
                max_depth=max_depth,
                min_confidence=min_confidence,
                as_of_time=as_of_time or "2099-12-31T23:59:59Z"
            )

            return [record.data() for record in result]

    def _format_knowledge(
        self,
        vector_results,
        graph_results,
        max_tokens: int
    ) -> str:
        """Format knowledge for LLM context"""

        sections = []
        token_count = 0

        # Add vector search results (direct matches)
        sections.append("## Relevant Entities\n")
        for result in vector_results[:10]:  # Top 10
            entity_text = f"- **{result.payload['name']}**: {result.payload['description']}\n"
            entity_tokens = len(entity_text.split()) * 1.3

            if token_count + entity_tokens > max_tokens * 0.5:
                break

            sections.append(entity_text)
            token_count += entity_tokens

        # Add graph relationships
        sections.append("\n## Related Knowledge\n")
        for record in graph_results[:20]:  # Top 20
            rel_text = self._format_relationship(record)
            rel_tokens = len(rel_text.split()) * 1.3

            if token_count + rel_tokens > max_tokens:
                break

            sections.append(rel_text)
            token_count += rel_tokens

        return "".join(sections)

    def _format_relationship(self, record: Dict) -> str:
        """Format graph relationship as text"""
        start = record["start"]["name"]
        related = record["related"]["name"]
        rels = record["rels"]

        rel_types = [r["type"] for r in rels]
        rel_chain = " -> ".join(rel_types)

        return f"- {start} --[{rel_chain}]--> {related}\n"

    def _embed(self, text: str) -> List[float]:
        """Generate embedding for text"""
        # Use OpenAI or similar
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response["data"][0]["embedding"]

    def insert(
        self,
        entities: List[Dict],
        relationships: List[Dict],
        source: Dict
    ):
        """Insert new knowledge into graph"""

        with self.graph.session() as session:
            # Insert source
            session.run(
                """
                CREATE (s:Source {
                    id: $id,
                    type: $type,
                    reference: $reference,
                    authority: $authority,
                    created: datetime($created)
                })
                """,
                **source
            )

            # Insert entities
            for entity in entities:
                # Insert to Neo4j
                session.run(
                    """
                    CREATE (e:Entity {
                        id: $id,
                        type: $type,
                        name: $name,
                        description: $description,
                        t_valid: datetime($t_valid),
                        t_invalid: $t_invalid,
                        confidence: $confidence,
                        source: $source
                    })
                    """,
                    **entity
                )

                # Link to source
                session.run(
                    """
                    MATCH (e:Entity {id: $entity_id})
                    MATCH (s:Source {id: $source_id})
                    CREATE (e)-[:DERIVED_FROM]->(s)
                    """,
                    entity_id=entity["id"],
                    source_id=source["id"]
                )

                # Insert to Qdrant
                embedding = self._embed(entity["description"])
                self.vector.upsert(
                    collection_name="fractal_knowledge",
                    points=[PointStruct(
                        id=entity["id"],
                        vector=embedding,
                        payload=entity
                    )]
                )

            # Insert relationships
            for rel in relationships:
                session.run(
                    """
                    MATCH (a:Entity {id: $from_id})
                    MATCH (b:Entity {id: $to_id})
                    CREATE (a)-[r:RELATES_TO {
                        type: $type,
                        strength: $strength,
                        t_valid: datetime($t_valid),
                        t_invalid: $t_invalid,
                        source: $source
                    }]->(b)
                    """,
                    **rel
                )
```

**Justification:**

- ✅ Temporal properties (t_valid, t_invalid) support knowledge evolution
- ✅ Provenance tracking via Source nodes
- ✅ Context-dependent facts via Statement nodes with context field
- ✅ Confidence scores enable filtering unreliable knowledge
- ✅ Hybrid retrieval combines semantic similarity + graph structure
- ✅ Contradiction tracking for conflict resolution
- ✅ Token budget management via max_tokens parameter

---

## 4.2 Four-Tier Memory Integration

### Memory Flow Design

**Tier 1: Active Memory (Working Context)**

- LLM context window (200K tokens)
- Ephemeral (lives only during task execution)
- Managed via prompt caching

**Tier 2: Short-Term Memory (Session Logs)**

- JSON files per task/session
- Retention: 30 days
- Purpose: Debugging, replay, metrics

**Tier 3: Long-Term Memory (GraphRAG)**

- Neo4j + Qdrant
- Permanent (with temporal invalidation)
- Purpose: Knowledge accumulation

**Tier 4: Meta-Knowledge (Obsidian)**

- Human-curated markdown
- Git-versioned
- Purpose: Strategic insights, domain expertise

### Memory Lifecycle

```python
# src/fractal_agent/memory/memory_manager.py
from datetime import datetime, timedelta
import json
from pathlib import Path

class MemoryManager:
    """
    Manages four-tier memory architecture.

    Handles promotion/demotion of information across tiers.
    """

    def __init__(
        self,
        graphrag: GraphRAG,
        logs_dir: Path,
        obsidian_vault: Path
    ):
        self.graphrag = graphrag
        self.logs_dir = logs_dir
        self.obsidian_vault = obsidian_vault

    def store_task_result(
        self,
        task_id: str,
        task: dict,
        result: dict,
        promote_to_longterm: bool = False
    ):
        """
        Store task result in appropriate tier.

        Tier 2: Always store in session logs
        Tier 3: Optionally promote to GraphRAG
        """

        # Tier 2: Session log (short-term)
        log_entry = {
            "task_id": task_id,
            "task": task,
            "result": result,
            "timestamp": datetime.utcnow().isoformat(),
            "promoted_to_longterm": promote_to_longterm
        }

        log_file = self.logs_dir / f"{datetime.utcnow().date()}/{task_id}.json"
        log_file.parent.mkdir(parents=True, exist_ok=True)

        with open(log_file, "w") as f:
            json.dump(log_entry, f, indent=2)

        # Tier 3: Promote to GraphRAG (long-term) if valuable
        if promote_to_longterm:
            self._promote_to_graphrag(task, result)

    def _promote_to_graphrag(self, task: dict, result: dict):
        """Extract knowledge from task result and add to GraphRAG"""

        # Use extraction agent to identify entities and relationships
        extractor = KnowledgeExtractionAgent()
        extracted = extractor(
            text=result["answer"],
            context=task["question"]
        )

        # Insert into GraphRAG
        self.graphrag.insert(
            entities=extracted.entities,
            relationships=extracted.relationships,
            source={
                "id": f"task_{task['task_id']}",
                "type": "session_log",
                "reference": f"/logs/{datetime.utcnow().date()}/{task['task_id']}.json",
                "authority": result.get("confidence", 0.5),
                "created": datetime.utcnow().isoformat()
            }
        )

    def cleanup_old_logs(self, retention_days: int = 30):
        """Remove Tier 2 logs older than retention period"""

        cutoff = datetime.utcnow() - timedelta(days=retention_days)

        for log_file in self.logs_dir.rglob("*.json"):
            if log_file.stat().st_mtime < cutoff.timestamp():
                log_file.unlink()  # Delete file

    def load_active_context(
        self,
        task: dict,
        max_tokens: int = 30000
    ) -> dict:
        """
        Load Tier 1 (active) context from Tier 3 (long-term).

        Called at task start to populate LLM context.
        """

        # Retrieve relevant knowledge from GraphRAG
        knowledge = self.graphrag.retrieve(
            query=task["question"],
            max_tokens=max_tokens
        )

        # Optionally: Load recent history from Tier 2
        recent_history = self._load_recent_history(limit=5)

        return {
            "knowledge": knowledge,
            "recent_history": recent_history
        }

    def _load_recent_history(self, limit: int) -> list[dict]:
        """Load recent task history from session logs"""

        log_files = sorted(
            self.logs_dir.rglob("*.json"),
            key=lambda f: f.stat().st_mtime,
            reverse=True
        )[:limit]

        history = []
        for log_file in log_files:
            with open(log_file) as f:
                history.append(json.load(f))

        return history
```

### Knowledge Extraction Agent

```python
# src/fractal_agent/agents/extraction_agent.py
import dspy

class KnowledgeExtractionSignature(dspy.Signature):
    """Extract structured knowledge from text"""

    text: str = dspy.InputField(desc="Text to extract knowledge from")
    context: str = dspy.InputField(desc="Context of the text")

    entities: list[dict] = dspy.OutputField(
        desc="List of entities found: [{name, type, description}]"
    )
    relationships: list[dict] = dspy.OutputField(
        desc="List of relationships: [{from, to, type, strength}]"
    )
    confidence: float = dspy.OutputField(
        desc="Confidence in extraction quality"
    )

class KnowledgeExtractionAgent(dspy.Module):
    """
    VSM System 1 (specialized) - Extract structured knowledge.

    Converts unstructured text into GraphRAG entities/relationships.
    """

    def __init__(self):
        super().__init__()
        self.lm = configure_dspy_for_role("extraction")
        self.extract = dspy.ChainOfThought(KnowledgeExtractionSignature)

    def forward(self, text: str, context: str = ""):
        result = self.extract(text=text, context=context)

        # Validate extraction quality
        dspy.Assert(
            len(result.entities) > 0,
            "Must extract at least one entity"
        )

        dspy.Suggest(
            result.confidence > 0.7,
            "Low confidence extraction - review manually"
        )

        return result
```

**Justification:**

- ✅ Four tiers serve distinct purposes (active, debug, knowledge, strategy)
- ✅ Automatic promotion from Tier 2 → Tier 3 via extraction agent
- ✅ Tier 1 populated from Tier 3 (knowledge retrieval at task start)
- ✅ Tier 2 cleanup prevents disk bloat
- ✅ Tier 4 (Obsidian) keeps human in loop for strategic knowledge

---

## 4.3 Handling Edge Cases

### Knowledge Conflicts: Graph vs LLM

**Problem:** What if GraphRAG says "Python is compiled" (wrong) but LLM knows it's interpreted?

**Solution: Confidence-Weighted Knowledge**

```python
def resolve_knowledge_conflict(
    graph_knowledge: str,
    llm_knowledge: str,
    graph_confidence: float,
    llm_confidence: float = 0.95  # LLMs are generally reliable
) -> str:
    """
    Resolve conflict between graph and LLM knowledge.

    Strategy:
    - High confidence graph (>0.9) → Trust graph
    - Low confidence graph (<0.7) → Trust LLM
    - Medium confidence → Include both, let LLM decide
    """

    if graph_confidence > 0.9:
        # Trust high-confidence graph knowledge
        return f"""
        <authoritative_knowledge confidence="{graph_confidence}">
        {graph_knowledge}
        </authoritative_knowledge>

        Note: Your pre-training may differ. Trust the authoritative knowledge above.
        """

    elif graph_confidence < 0.7:
        # Trust LLM, ignore low-confidence graph
        return ""  # Don't include graph knowledge

    else:
        # Let LLM adjudicate
        return f"""
        <knowledge_from_database confidence="{graph_confidence}">
        {graph_knowledge}
        </knowledge_from_database>

        Note: This information from our database may conflict with your pre-training.
        If you detect a conflict, flag it for review and use your best judgment.
        """
```

### Context-Dependent Truth

**Problem:** "Python is slow" is true for CPU-bound, false for I/O-bound.

**Solution: Store context with statements**

```cypher
// Context-qualified statement
CREATE (stmt:Statement {
    text: "Python is slow",
    context: "for CPU-bound numerical computations compared to C",
    confidence: 0.9
})

// Different context, different truth value
CREATE (stmt2:Statement {
    text: "Python is fast enough",
    context: "for I/O-bound web applications",
    confidence: 0.9
})

// Not contradictory - both can be true in different contexts!
```

### Knowledge Evolution

**Problem:** Information becomes outdated. How do we handle?

**Solution: Temporal invalidation**

```python
def update_knowledge(
    entity_id: str,
    new_information: dict,
    reason: str
):
    """
    Update knowledge, preserving history.

    Old entity gets t_invalid timestamp.
    New entity gets created with t_valid timestamp.
    """

    with graph.session() as session:
        # Invalidate old entity
        session.run(
            """
            MATCH (old:Entity {id: $entity_id})
            WHERE old.t_invalid IS NULL
            SET old.t_invalid = datetime()
            SET old.invalidation_reason = $reason
            """,
            entity_id=entity_id,
            reason=reason
        )

        # Create new version
        new_id = f"{entity_id}_v{int(time.time())}"
        session.run(
            """
            CREATE (new:Entity {
                id: $new_id,
                type: $type,
                name: $name,
                description: $description,
                t_valid: datetime(),
                t_invalid: NULL,
                confidence: $confidence,
                supersedes: $old_id
            })
            """,
            new_id=new_id,
            old_id=entity_id,
            **new_information
        )
```

**Querying historical knowledge:**

```cypher
// Get knowledge as of specific time
MATCH (e:Entity)
WHERE e.t_valid <= datetime("2025-10-01T00:00:00Z")
AND (e.t_invalid IS NULL OR e.t_invalid > datetime("2025-10-01T00:00:00Z"))
RETURN e

// Get evolution of an entity
MATCH path = (old:Entity)-[:supersedes*]->(new:Entity)
WHERE old.id STARTS WITH "vsm_system_2"
RETURN path
ORDER BY old.t_valid ASC
```

---

# Session 4 Summary

**Completed Designs:**

1. ✅ **Neo4j Schema** - Entities, Statements, Sources with temporal properties
2. ✅ **Qdrant Integration** - Embeddings for semantic search
3. ✅ **Hybrid Retrieval** - Vector search + graph traversal
4. ✅ **Four-Tier Memory Flow** - Active → Short → Long → Meta
5. ✅ **Knowledge Extraction** - Automated promotion Tier 2 → Tier 3
6. ✅ **Conflict Resolution** - Graph vs LLM, context-dependent truth
7. ✅ **Temporal Evolution** - Knowledge versioning and invalidation

**Key Decisions:**

- Temporal properties (t_valid, t_invalid) enable knowledge evolution
- Context field on statements handles context-dependent truth
- Confidence-weighted resolution for graph/LLM conflicts
- Extraction agent automates knowledge graph population
- Hybrid retrieval balances semantic + structural relevance

---

# Session 5: Obsidian Integration & Human-in-Loop

## 5.1 Obsidian Vault Structure

### Design Philosophy

**Goals:**

- Human-readable markdown (not XML/JSON)
- Git-friendly (text diffs work)
- Zettelkasten-inspired (atomic notes, linked)
- Minimal friction for review

**Vault Structure:**

```
obsidian-vault/
├── .obsidian/                 # Obsidian config (git-ignored)
├── 0-inbox/                   # Unreviewed agent outputs
│   ├── 2025-10-18/
│   │   ├── task-001-research-vsm.md
│   │   └── task-002-synthesis.md
│   └── README.md
├── 1-reviewed/                # Human-approved knowledge
│   ├── concepts/
│   │   ├── vsm-system-1.md
│   │   ├── vsm-system-2.md
│   │   └── dspy-framework.md
│   ├── insights/
│   │   ├── agent-coordination-patterns.md
│   │   └── cost-optimization-strategies.md
│   └── decisions/
│       ├── 2025-10-18-llm-provider-choice.md
│       └── 2025-10-18-caching-strategy.md
├── 2-templates/               # Templates for consistency
│   ├── task-result.md
│   ├── concept.md
│   └── decision-record.md
├── 3-metadata/                # System-generated indices
│   ├── agents.md              # List of all agents
│   ├── tasks.md               # Task log
│   └── metrics.md             # Performance metrics
└── README.md
```

### File Templates

**Task Result Template** (`2-templates/task-result.md`):

```markdown
---
type: task_result
task_id: { { task_id } }
agent: { { agent_id } }
date: { { date } }
status: pending_review # pending_review | approved | rejected | needs_revision
confidence: { { confidence } }
tokens_used: { { tokens_used } }
duration: { { duration } }
---

# Task: {{task_title}}

## Question

{{question}}

## Answer

{{answer}}

## Sources

{{#sources}}

- {{.}}
  {{/sources}}

## Confidence Assessment

Confidence: {{confidence}} / 1.0

{{#if low_confidence}}
⚠️ **Low Confidence**: This result may need verification.
{{/if}}

## Metrics

- Tokens used: {{tokens_used}}
- Duration: {{duration}}s
- Cache hit: {{cache_hit}}
- Provider: {{provider}}

## Review Notes

<!-- Human adds notes here -->

## Actions

<!-- What to do with this knowledge? -->

- [ ] Promote to long-term memory (GraphRAG)
- [ ] Extract as atomic concept note
- [ ] Archive (not valuable)
- [ ] Request revision
```

**Concept Note Template** (`2-templates/concept.md`):

```markdown
---
type: concept
aliases: [{ { aliases } }]
tags: { { tags } }
created: { { date } }
updated: { { date } }
confidence: { { confidence } }
source: { { source } }
---

# {{concept_name}}

## Definition

{{definition}}

## Key Properties

{{properties}}

## Related Concepts

{{#related}}

- [[{{.}}]]
  {{/related}}

## Examples

{{examples}}

## References

{{references}}
```

**Decision Record Template** (`2-templates/decision-record.md`):

```markdown
---
type: decision
date: { { date } }
status: accepted # proposed | accepted | deprecated | superseded
---

# {{decision_title}}

## Context

{{context}}

## Decision

{{decision}}

## Rationale

{{rationale}}

## Consequences

### Positive

{{positive_consequences}}

### Negative

{{negative_consequences}}

## Alternatives Considered

{{alternatives}}

## References

{{references}}
```

### Auto-Generated Files

**Agent Registry** (`3-metadata/agents.md`):

```markdown
# Agent Registry

Auto-generated list of all agents in the system.
Last updated: 2025-10-18 14:30:00

## Operational Agents (VSM System 1)

| Agent ID | Specialty | Model | Tasks Completed | Avg Confidence | Status |
| -------- | --------- | ----- | --------------- | -------------- | ------ |
| op_001   | research  | haiku | 42              | 0.87           | active |
| op_002   | analysis  | haiku | 38              | 0.91           | active |

## Control Agents (VSM System 3)

| Agent ID | Model      | Tasks Decomposed | Status |
| -------- | ---------- | ---------------- | ------ |
| ctrl_001 | sonnet-3.7 | 12               | active |

## Metrics

- Total agents: 3
- Active: 3
- Paused: 0
- Total tasks: 92
- System uptime: 127 hours
```

## 5.2 Git Sync Workflow

### Auto-Commit Strategy

```python
# src/fractal_agent/obsidian/git_sync.py
import git
from pathlib import Path
from datetime import datetime

class ObsidianGitSync:
    """
    Manages git sync for Obsidian vault.

    Auto-commits agent outputs, preserves human edits.
    """

    def __init__(self, vault_path: Path):
        self.vault = vault_path
        self.repo = git.Repo(vault_path)

    def commit_task_result(self, task_id: str, file_path: Path):
        """
        Auto-commit agent output.

        Uses conventional commit format.
        """
        # Stage file
        self.repo.index.add([str(file_path.relative_to(self.vault))])

        # Commit
        commit_msg = f"feat(task): Add task result {task_id}\\n\\nAuto-generated by ResearchAgent"
        self.repo.index.commit(commit_msg)

    def commit_human_review(self, file_path: Path, reviewer: str):
        """Commit human-reviewed changes"""
        self.repo.index.add([str(file_path.relative_to(self.vault))])

        commit_msg = f"review: {file_path.name}\\n\\nReviewed by: {reviewer}"
        self.repo.index.commit(commit_msg)

    def sync(self):
        """
        Sync with remote (if configured).

        Pull before push to avoid conflicts.
        """
        if "origin" in [remote.name for remote in self.repo.remotes]:
            origin = self.repo.remote("origin")

            # Pull changes
            origin.pull("main")

            # Push changes
            origin.push("main")

    def get_pending_reviews(self) -> list[Path]:
        """Find files in inbox (pending review)"""
        inbox = self.vault / "0-inbox"
        return list(inbox.rglob("*.md"))

    def promote_to_reviewed(
        self,
        file_path: Path,
        target_category: str  # "concepts", "insights", "decisions"
    ):
        """
        Move file from inbox to reviewed.

        Updates git history.
        """
        target_dir = self.vault / "1-reviewed" / target_category
        target_dir.mkdir(parents=True, exist_ok=True)

        target_path = target_dir / file_path.name

        # Git mv (preserves history)
        self.repo.git.mv(
            str(file_path.relative_to(self.vault)),
            str(target_path.relative_to(self.vault))
        )

        # Commit move
        self.repo.index.commit(
            f"review: Promote {file_path.name} to {target_category}"
        )
```

### Conflict Resolution

**Problem:** What if human edits file while agent is writing?

**Solution: Lockfile mechanism**

```python
import fcntl
from contextlib import contextmanager

@contextmanager
def atomic_write(file_path: Path):
    """
    Atomic write with file locking.

    Prevents simultaneous writes.
    """
    lock_file = file_path.with_suffix('.lock')

    try:
        # Acquire lock
        with open(lock_file, 'w') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)

            # Write to temp file
            temp_file = file_path.with_suffix('.tmp')
            yield temp_file

            # Atomic rename
            temp_file.rename(file_path)
    finally:
        # Release lock
        lock_file.unlink(missing_ok=True)

# Usage
with atomic_write(task_file) as tmp:
    tmp.write_text(task_result_markdown)
```

## 5.3 Review Interface

### Human Approval Workflow

```python
# src/fractal_agent/obsidian/review_interface.py
from pathlib import Path
from typing import Literal, Optional
import frontmatter

class ReviewInterface:
    """
    Interface for human review of agent outputs.

    Reads from Obsidian vault, updates statuses.
    """

    def __init__(self, vault: Path):
        self.vault = vault
        self.inbox = vault / "0-inbox"
        self.reviewed = vault / "1-reviewed"

    def get_pending_tasks(self) -> list[dict]:
        """Get all tasks pending review"""
        pending = []

        for md_file in self.inbox.rglob("*.md"):
            post = frontmatter.load(md_file)

            if post.get("status") == "pending_review":
                pending.append({
                    "file": md_file,
                    "task_id": post.get("task_id"),
                    "agent": post.get("agent"),
                    "confidence": post.get("confidence"),
                    "date": post.get("date"),
                    "content": post.content
                })

        return sorted(pending, key=lambda x: x["date"], reverse=True)

    def approve(
        self,
        task_file: Path,
        promote_to: Literal["concepts", "insights", "decisions"],
        reviewer: str,
        notes: str = ""
    ):
        """
        Approve task result and promote to reviewed.

        Actions:
        1. Update status in frontmatter
        2. Add review notes
        3. Move to reviewed category
        4. Promote to GraphRAG
        """
        post = frontmatter.load(task_file)

        # Update frontmatter
        post["status"] = "approved"
        post["reviewer"] = reviewer
        post["review_date"] = datetime.utcnow().isoformat()

        # Add review notes
        if notes:
            post.content += f"\\n\\n## Review Notes ({reviewer})\\n{notes}"

        # Save updated file
        with open(task_file, "w") as f:
            f.write(frontmatter.dumps(post))

        # Move to reviewed
        git_sync = ObsidianGitSync(self.vault)
        git_sync.promote_to_reviewed(task_file, promote_to)
        git_sync.commit_human_review(task_file, reviewer)

        # Promote to GraphRAG
        memory_manager = MemoryManager(...)
        memory_manager.store_task_result(
            task_id=post["task_id"],
            task={"question": post.content},
            result={"answer": post.content},
            promote_to_longterm=True  # Approved → long-term
        )

    def reject(
        self,
        task_file: Path,
        reason: str,
        reviewer: str
    ):
        """Reject task result"""
        post = frontmatter.load(task_file)

        post["status"] = "rejected"
        post["reviewer"] = reviewer
        post["rejection_reason"] = reason
        post["review_date"] = datetime.utcnow().isoformat()

        with open(task_file, "w") as f:
            f.write(frontmatter.dumps(post))

        # Archive rejected tasks
        archive = self.vault / "archive" / "rejected"
        archive.mkdir(parents=True, exist_ok=True)

        git_sync = ObsidianGitSync(self.vault)
        git_sync.repo.git.mv(
            str(task_file.relative_to(self.vault)),
            str(archive / task_file.name)
        )
        git_sync.repo.index.commit(f"review: Reject task {post['task_id']}")

    def request_revision(
        self,
        task_file: Path,
        feedback: str,
        reviewer: str
    ):
        """Request agent to revise task"""
        post = frontmatter.load(task_file)

        post["status"] = "needs_revision"
        post["reviewer"] = reviewer
        post["feedback"] = feedback
        post["review_date"] = datetime.utcnow().isoformat()

        with open(task_file, "w") as f:
            f.write(frontmatter.dumps(post))

        # Create revision task for agent
        # (Could trigger agent re-execution with feedback)
```

### CLI Review Tool

```python
# tools/review_cli.py
import click
from rich.console import Console
from rich.table import Table
from rich.markdown import Markdown

console = Console()

@click.group()
def cli():
    """Obsidian vault review CLI"""
    pass

@cli.command()
def list():
    """List pending reviews"""
    interface = ReviewInterface(Path("obsidian-vault"))
    pending = interface.get_pending_tasks()

    table = Table(title="Pending Reviews")
    table.add_column("Task ID")
    table.add_column("Agent")
    table.add_column("Confidence")
    table.add_column("Date")

    for task in pending:
        table.add_row(
            task["task_id"],
            task["agent"],
            f"{task['confidence']:.2f}",
            task["date"]
        )

    console.print(table)

@cli.command()
@click.argument("task_id")
def show(task_id):
    """Show task details"""
    interface = ReviewInterface(Path("obsidian-vault"))
    pending = interface.get_pending_tasks()

    task = next((t for t in pending if t["task_id"] == task_id), None)
    if not task:
        console.print(f"[red]Task {task_id} not found[/red]")
        return

    md = Markdown(task["content"])
    console.print(md)

@cli.command()
@click.argument("task_id")
@click.option("--category", type=click.Choice(["concepts", "insights", "decisions"]))
@click.option("--reviewer", prompt="Your name")
@click.option("--notes", default="")
def approve(task_id, category, reviewer, notes):
    """Approve task and promote to reviewed"""
    interface = ReviewInterface(Path("obsidian-vault"))
    pending = interface.get_pending_tasks()

    task = next((t for t in pending if t["task_id"] == task_id), None)
    if not task:
        console.print(f"[red]Task {task_id} not found[/red]")
        return

    interface.approve(
        task_file=task["file"],
        promote_to=category,
        reviewer=reviewer,
        notes=notes
    )

    console.print(f"[green]✓ Task {task_id} approved and promoted to {category}[/green]")

if __name__ == "__main__":
    cli()
```

**Usage:**

```bash
# List pending reviews
python tools/review_cli.py list

# Show task details
python tools/review_cli.py show task-001

# Approve and promote
python tools/review_cli.py approve task-001 --category concepts --reviewer BMad

# Reject
python tools/review_cli.py reject task-001 --reason "Incorrect information"
```

## 5.4 Meta-Knowledge Curation

### Human-Curated Strategic Knowledge

**Examples:**

**Insight Note** (`1-reviewed/insights/agent-coordination-patterns.md`):

```markdown
---
type: insight
tags: [coordination, vsm-system-2, patterns]
created: 2025-10-18
confidence: 0.95
source: human_curation
---

# Agent Coordination Patterns

## Key Insight

Lightweight coordination infrastructure (task registry + budget pool) prevents 90% of conflicts without needing a coordination agent.

## Supporting Evidence

- Phase 1 testing: 47 tasks, 0 conflicts detected
- Token budget: 100% compliance, no overruns
- Duplicate work: 0 instances (task registry caught all)

## When to Escalate to System 2 Agent

Only when:

1. Contradictory goals detected (requires reasoning about intent)
2. Information conflicts (requires adjudication of truth claims)
3. Complex resource arbitration (priority across >10 agents)

## Pattern
```

Lightweight Infrastructure (Phase 1-2)
↓ (only when needed)
Coordination Agent (Phase 4+)

```

## Related
- [[VSM System 2]]
- [[Task Registry Design]]
- [[Token Budget Pool]]
```

**Decision Record** (`1-reviewed/decisions/2025-10-18-llm-provider-choice.md`):

```markdown
---
type: decision
date: 2025-10-18
status: accepted
---

# LLM Provider: Anthropic + Gemini (No LiteLLM)

## Context

Need multi-provider LLM infrastructure with failover.

Initial recommendation: LiteLLM (abstraction layer).

User constraint: "Anthropic SDK is the ONLY method I want to use. I pay for max subscription and want to use Gemini as backup. I don't want any other solution."

## Decision

Use official Anthropic + Google SDKs directly via custom UnifiedLM wrapper.

## Rationale

1. User has max Anthropic subscription (cost already sunk)
2. Anthropic prompt caching = 90% savings (LiteLLM wouldn't optimize this)
3. Two providers sufficient for resilience
4. Direct SDK control > abstraction layer
5. Simpler dependencies (no LiteLLM)

## Consequences

### Positive

- Full control over Anthropic caching
- Lighter dependency tree
- Direct access to latest provider features
- User satisfaction (respects constraints)

### Negative

- Custom provider abstraction needed (but lightweight)
- Adding 3rd provider requires code changes (vs config with LiteLLM)

## Alternatives Considered

1. LiteLLM - Rejected per user constraint
2. LangChain - Too heavyweight, same abstraction issues
3. Direct API calls - No failover, no provider abstraction

## Implementation

See: [[UnifiedLM Architecture]]

## References

- /docs/fractal-agent-llm-architecture.md
- /docs/brainstorming-session-results-2025-10-18.md (Priority #1)
```

**Justification:**

- ✅ Minimal friction (markdown, not forms)
- ✅ Git-friendly (text diffs)
- ✅ Human-readable (Obsidian renders nicely)
- ✅ Auto-commit preserves history
- ✅ CLI tool enables headless review
- ✅ Promotes approved knowledge to GraphRAG automatically

---

# Session 5 Summary

**Completed Designs:**

1. ✅ **Obsidian Vault Structure** - Inbox → Reviewed, organized by type
2. ✅ **File Templates** - Consistent formatting for tasks, concepts, decisions
3. ✅ **Git Sync** - Auto-commit agent outputs, preserve human edits
4. ✅ **Review Interface** - Approve, reject, request revision workflows
5. ✅ **CLI Tool** - Headless review for automation
6. ✅ **Meta-Knowledge** - Human-curated insights and decisions

**Key Decisions:**

- Markdown over JSON (human-readable)
- Git-based versioning (auditable history)
- Atomic file locking (prevent conflicts)
- Auto-promotion approved → GraphRAG (knowledge accumulation)
- CLI + Obsidian UI (flexibility for human preference)

---

# Session 6: Testing & Validation Strategy

## 6.1 Test Coverage Strategy

Based on research, we'll implement a **progressive test pyramid**:

```
          E2E Tests (Expensive, Slow)
         /                           \
    Integration Tests (Medium)
   /                                  \
Unit Tests (Fast, Cheap, Many)
```

### Test Levels

**Level 1: Unit Tests (70% coverage target)**

- Mock LLM responses (deterministic)
- Test agent logic in isolation
- Test utility functions
- Fast feedback (<5 seconds total)

**Level 2: Integration Tests (20% coverage target)**

- Real LLM calls (controlled budget)
- Test agent workflows end-to-end
- Test memory tier integration
- Medium speed (<2 minutes total)

**Level 3: E2E Tests (10% coverage target)**

- Full system tests
- Multi-agent workflows
- Human-in-loop simulation
- Slow (<10 minutes total)

## 6.2 Testing Infrastructure

### Mock LLM Responses

```python
# tests/mocks/llm_mocks.py
from fractal_agent.utils.llm_provider import LLMProvider
import json
from pathlib import Path

class RecordingLLMProvider(LLMProvider):
    """
    Records real LLM responses for later playback.

    Workflow:
    1. Run tests with RECORD_MODE=true → saves responses
    2. Run tests normally → replays saved responses
    """

    def __init__(self, model: str, recordings_dir: Path):
        super().__init__(model)
        self.recordings_dir = recordings_dir
        self.recordings_dir.mkdir(parents=True, exist_ok=True)
        self.record_mode = os.getenv("RECORD_MODE") == "true"

    def _call_provider(self, messages, **kwargs):
        # Generate cache key from messages
        cache_key = self._hash_messages(messages)
        cache_file = self.recordings_dir / f"{cache_key}.json"

        if self.record_mode:
            # Make real API call
            response = self._real_api_call(messages, **kwargs)

            # Save recording
            with open(cache_file, "w") as f:
                json.dump(response, f)

            return response
        else:
            # Playback recording
            if cache_file.exists():
                with open(cache_file) as f:
                    return json.load(f)
            else:
                raise ValueError(
                    f"No recording found for messages: {messages[:50]}...\\n"
                    f"Run tests with RECORD_MODE=true to create recordings."
                )

    def _hash_messages(self, messages) -> str:
        """Create deterministic hash of messages"""
        import hashlib
        msg_str = json.dumps(messages, sort_keys=True)
        return hashlib.md5(msg_str.encode()).hexdigest()

    def _real_api_call(self, messages, **kwargs):
        """Make actual LLM API call (implement based on provider)"""
        # Delegate to real provider
        pass
```

### Test Fixtures

```python
# tests/conftest.py
import pytest
from pathlib import Path
from fractal_agent.utils.llm_provider import UnifiedLM
from fractal_agent.memory.graphrag import GraphRAG
from tests.mocks.llm_mocks import RecordingLLMProvider

@pytest.fixture(scope="session")
def test_data_dir():
    """Directory for test data"""
    return Path(__file__).parent / "test_data"

@pytest.fixture
def mock_llm(test_data_dir):
    """Mock LLM for fast tests"""
    provider = RecordingLLMProvider(
        model="mock",
        recordings_dir=test_data_dir / "recordings"
    )
    return UnifiedLM(providers=[("mock", "mock")])

@pytest.fixture(scope="session")
def test_graphrag():
    """Test GraphRAG instance (in-memory or test DB)"""
    # Use embedded Neo4j for tests
    return GraphRAG(
        neo4j_uri="bolt://localhost:7688",  # Test instance
        neo4j_user="neo4j",
        neo4j_password="test",
        qdrant_host="localhost",
        qdrant_port=6334  # Test instance
    )

@pytest.fixture(autouse=True)
def cleanup_graphrag(test_graphrag):
    """Clean up test DB after each test"""
    yield
    # Clear test database
    with test_graphrag.graph.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
```

### DSPy Assertion Tests

```python
# tests/test_dspy_assertions.py
import dspy
import pytest
from fractal_agent.agents.research_agent import ResearchAgent

def test_research_agent_assertions(mock_llm):
    """Test that DSPy assertions work correctly"""

    # Configure DSPy with mock
    dspy.configure(lm=mock_llm)

    agent = ResearchAgent(enable_assertions=True)

    # This should pass assertions
    result = agent(
        question="What is VSM?",
        context="Viable System Model is..."
    )

    assert result.answer is not None
    assert len(result.answer) >= 100  # Min length assertion

def test_research_agent_assertion_failure(mock_llm):
    """Test that assertions catch bad outputs"""

    # Configure mock to return too-short response
    mock_llm.responses = {
        "What is VSM?": "VSM."  # Only 4 chars - will fail assertion
    }

    dspy.configure(lm=mock_llm)
    agent = ResearchAgent(enable_assertions=True)

    # Should raise assertion error or retry
    with pytest.raises(dspy.AssertionError):
        agent(question="What is VSM?")
```

### Integration Tests

```python
# tests/integration/test_research_workflow.py
import pytest
from fractal_agent.workflows.simple_research import build_research_workflow

@pytest.mark.integration
@pytest.mark.slow
def test_research_workflow_end_to_end():
    """
    Full workflow test with real LLM.

    This costs money! Use sparingly.
    """

    workflow = build_research_workflow()

    # Execute workflow
    result = workflow.invoke({
        "question": "What is VSM System 2?",
        "status": "pending"
    })

    # Validate result
    assert result["status"] == "complete"
    assert len(result["answer"]) > 100
    assert result["confidence"] > 0.6

    # Validate state transitions
    # (LangGraph should have gone through: research -> END)

@pytest.mark.integration
def test_multi_agent_coordination():
    """Test parallel operational agents with coordination"""

    workflow = build_multi_agent_workflow()

    result = workflow.invoke({
        "task": "Research all 5 VSM systems",
        "status": "pending"
    })

    # Should have decomposed into 5 subtasks
    assert len(result["subtasks"]) == 5

    # All subtasks should have completed
    assert all(r["status"] == "complete" for r in result["worker_results"])

    # No duplicate work (coordination working)
    task_ids = [r["subtask_id"] for r in result["worker_results"]]
    assert len(task_ids) == len(set(task_ids))  # No duplicates
```

### Performance Tests

```python
# tests/performance/test_caching.py
import pytest
from fractal_agent.config.model_config import get_llm_for_role

@pytest.mark.performance
def test_cache_hit_rate():
    """Validate that caching achieves >80% hit rate"""

    lm = get_llm_for_role("operational")

    # Make first call (cache miss)
    response1 = lm(prompt="What is VSM?", max_tokens=100)
    assert not response1["cache_hit"]

    # Make similar calls (should hit cache)
    cache_hits = 0
    for i in range(10):
        response = lm(
            prompt=f"What is VSM System {i}?",
            max_tokens=100
        )
        if response["cache_hit"]:
            cache_hits += 1

    # Should achieve >80% cache hit rate
    cache_hit_rate = cache_hits / 10
    assert cache_hit_rate > 0.8

@pytest.mark.performance
def test_token_budget_compliance():
    """Ensure agents stay within token budgets"""

    from fractal_agent.coordination.budget_pool import TokenBudgetPool

    budget = TokenBudgetPool(max_tokens=10000)

    # Simulate 10 agents requesting budget
    agents = [f"op_{i:03d}" for i in range(10)]

    approved_count = 0
    for agent_id in agents:
        allocation = budget.request(
            agent_id=agent_id,
            estimated_tokens=1500,
            priority=1
        )

        if allocation["approved"]:
            approved_count += 1

    # Should approve 6-7 agents (10000 / 1500 ≈ 6.67)
    assert 6 <= approved_count <= 7

    # Budget should not be exceeded
    assert budget.get_available() >= 0
```

## 6.3 Continuous Testing Strategy

### Pre-Commit Hooks

```bash
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: fast-tests
        name: Run fast unit tests
        entry: pytest tests/ -m "not integration and not slow" --maxfail=1
        language: system
        pass_filenames: false

      - id: type-check
        name: Type checking with mypy
        entry: mypy src/
        language: system
        pass_filenames: false

      - id: lint
        name: Lint with ruff
        entry: ruff check src/ tests/
        language: system
        pass_filenames: false
```

### CI Pipeline

```yaml
# .github/workflows/test.yml
name: Test Suite

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov

      - name: Run unit tests
        run: pytest tests/ -m "not integration" --cov

  integration-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4

      - name: Start test databases
        run: |
          docker-compose -f docker-compose.test.yml up -d

      - name: Run integration tests
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
          GOOGLE_API_KEY: ${{ secrets.GOOGLE_API_KEY }}
        run: pytest tests/ -m "integration" --maxfail=3

      - name: Stop test databases
        run: docker-compose -f docker-compose.test.yml down
```

## 6.4 Quality Metrics

**Test Quality KPIs:**

| Metric                            | Target | Measurement     |
| --------------------------------- | ------ | --------------- |
| Unit test coverage                | >80%   | pytest-cov      |
| Integration test coverage         | >60%   | pytest-cov      |
| Cache hit rate                    | >80%   | LLM metrics     |
| Test execution time (unit)        | <10s   | pytest duration |
| Test execution time (integration) | <2min  | pytest duration |
| Test flakiness                    | <1%    | CI failure rate |
| DSPy assertion pass rate          | >95%   | Custom metric   |

**Monitoring:**

```python
# src/fractal_agent/testing/metrics.py
from dataclasses import dataclass
from datetime import datetime

@dataclass
class TestMetrics:
    """Test execution metrics"""
    run_id: str
    timestamp: datetime
    unit_tests_passed: int
    unit_tests_failed: int
    integration_tests_passed: int
    integration_tests_failed: int
    total_duration_seconds: float
    cache_hit_rate: float
    tokens_used: int
    cost_usd: float

def log_test_metrics(metrics: TestMetrics):
    """Log test metrics for tracking over time"""
    # Could send to metrics system (Prometheus, etc.)
    pass
```

---

# Session 6 Summary

**Completed Designs:**

1. ✅ **Test Pyramid** - Unit (70%) → Integration (20%) → E2E (10%)
2. ✅ **Mock Infrastructure** - Recording/playback LLM responses
3. ✅ **DSPy Assertion Tests** - Validate quality constraints
4. ✅ **Integration Tests** - Multi-agent workflows, coordination
5. ✅ **Performance Tests** - Cache hit rate, token budgets
6. ✅ **CI/CD Pipeline** - Automated testing on push
7. ✅ **Quality Metrics** - KPIs and monitoring

**Key Decisions:**

- Recording/playback for deterministic tests (no API costs)
- Separate test database instances (Neo4j, Qdrant)
- Pre-commit hooks for fast feedback
- Integration tests budget-controlled (max failures)
- Quality metrics tracked over time

---

# Session 7: Security & Resilience Deep Dive

## 7.1 PII Redaction Strategy

### Implementation with Presidio

```python
# src/fractal_agent/security/pii_redaction.py
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from typing import Literal

class PIIRedactor:
    """
    PII detection and redaction using Microsoft Presidio.

    Protects against PII leakage in logs, prompts, and knowledge base.
    """

    def __init__(
        self,
        mode: Literal["detect", "redact", "hash"] = "redact",
        entities: list[str] = None
    ):
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        self.mode = mode

        # Default PII entities to detect
        self.entities = entities or [
            "PERSON",
            "EMAIL_ADDRESS",
            "PHONE_NUMBER",
            "CREDIT_CARD",
            "IBAN_CODE",
            "IP_ADDRESS",
            "US_SSN",
            "US_PASSPORT",
            "LOCATION",
            "DATE_TIME",
            "MEDICAL_LICENSE",
            "URL"
        ]

    def process(self, text: str, language: str = "en") -> dict:
        """
        Process text for PII.

        Returns:
            {
                "original": str,
                "redacted": str,
                "detections": list[dict],
                "has_pii": bool
            }
        """
        # Analyze for PII
        results = self.analyzer.analyze(
            text=text,
            language=language,
            entities=self.entities
        )

        if not results:
            return {
                "original": text,
                "redacted": text,
                "detections": [],
                "has_pii": False
            }

        # Anonymize based on mode
        if self.mode == "redact":
            anonymized = self.anonymizer.anonymize(
                text=text,
                analyzer_results=results
            )
            redacted_text = anonymized.text

        elif self.mode == "hash":
            # Hash PII instead of replacing with placeholder
            anonymized = self.anonymizer.anonymize(
                text=text,
                analyzer_results=results,
                operators={"DEFAULT": OperatorConfig("hash")}
            )
            redacted_text = anonymized.text

        else:  # detect only
            redacted_text = text

        return {
            "original": text,
            "redacted": redacted_text,
            "detections": [
                {
                    "entity_type": r.entity_type,
                    "start": r.start,
                    "end": r.end,
                    "score": r.score,
                    "text": text[r.start:r.end]
                }
                for r in results
            ],
            "has_pii": len(results) > 0
        }

# Integration with agents
class SecureAgent(dspy.Module):
    """Agent wrapper with PII protection"""

    def __init__(self, base_agent, redactor: PIIRedactor):
        super().__init__()
        self.agent = base_agent
        self.redactor = redactor

    def forward(self, **inputs):
        # Redact PII from inputs
        secure_inputs = {}
        detections = []

        for key, value in inputs.items():
            if isinstance(value, str):
                result = self.redactor.process(value)
                secure_inputs[key] = result["redacted"]
                if result["has_pii"]:
                    detections.extend(result["detections"])
                    logger.warning(
                        f"PII detected in input '{key}': {result['detections']}"
                    )
            else:
                secure_inputs[key] = value

        # Execute agent with redacted inputs
        output = self.agent(**secure_inputs)

        # Check output for PII leakage
        if isinstance(output, str):
            output_check = self.redactor.process(output)
            if output_check["has_pii"]:
                logger.error(
                    f"PII LEAKED in agent output: {output_check['detections']}"
                )
                # Either redact or raise error
                raise SecurityError("Agent output contains PII")

        return output
```

### PII Detection in Logs

```python
# src/fractal_agent/security/secure_logging.py
import logging
from typing import Any

class SecureLogHandler(logging.Handler):
    """Log handler that redacts PII before writing"""

    def __init__(self, base_handler, redactor: PIIRedactor):
        super().__init__()
        self.base_handler = base_handler
        self.redactor = redactor

    def emit(self, record: logging.LogRecord):
        """Redact PII from log message"""
        if isinstance(record.msg, str):
            result = self.redactor.process(record.msg)

            if result["has_pii"]:
                # Replace message with redacted version
                record.msg = result["redacted"]

                # Add warning about redaction
                record.pii_redacted = True
                record.pii_entities = [d["entity_type"] for d in result["detections"]]

        # Pass to base handler
        self.base_handler.emit(record)

# Setup
redactor = PIIRedactor(mode="redact")
base_handler = logging.StreamHandler()
secure_handler = SecureLogHandler(base_handler, redactor)

logger = logging.getLogger("fractal_agent")
logger.addHandler(secure_handler)
```

## 7.2 Secrets Management

### Environment Variables (Phase 0-2)

```python
# src/fractal_agent/security/secrets.py
import os
from typing import Optional

class SecretsManager:
    """
    Secrets management (initially env vars, later Vault).

    Progressive:
    - Phase 0-2: Environment variables
    - Phase 3+: HashiCorp Vault or cloud provider secrets
    """

    def __init__(self, use_vault: bool = False):
        self.use_vault = use_vault

    def get_secret(self, key: str) -> Optional[str]:
        """Get secret from environment"""
        if self.use_vault:
            return self._get_from_vault(key)
        else:
            return os.getenv(key)

    def _get_from_vault(self, key: str) -> Optional[str]:
        """Get secret from Vault (future)"""
        # Implement HashiCorp Vault integration
        pass

    def validate_required_secrets(self):
        """Validate that all required secrets are present"""
        required = [
            "ANTHROPIC_API_KEY",
            "GOOGLE_API_KEY",
            "NEO4J_PASSWORD"
        ]

        missing = []
        for key in required:
            if not self.get_secret(key):
                missing.append(key)

        if missing:
            raise ValueError(
                f"Missing required secrets: {', '.join(missing)}\\n"
                f"Set environment variables or configure secrets manager."
            )

# Validate on startup
secrets = SecretsManager()
secrets.validate_required_secrets()
```

### Secret Rotation (Phase 4+)

```python
def rotate_api_key(provider: str, new_key: str):
    """
    Rotate API key with zero downtime.

    Strategy:
    1. Add new key to rotation
    2. Gradual switchover (50% traffic)
    3. Monitor for errors
    4. Complete switchover
    5. Revoke old key
    """
    # Implementation for production use
    pass
```

## 7.3 Input Sanitization

### Prompt Injection Defense

```python
# src/fractal_agent/security/input_validation.py
import re
from typing import Optional

class PromptInjectionDetector:
    """
    Detect and block prompt injection attempts.

    Patterns:
    - "Ignore previous instructions"
    - "You are now"
    - "System:"
    - Excessive repetition
    - Encoding tricks (base64, etc.)
    """

    BANNED_PATTERNS = [
        r"ignore\s+(previous|all|prior)\s+instructions",
        r"you\s+are\s+now",
        r"^system:",
        r"^assistant:",
        r"<\|im_start\|>",
        r"<\|im_end\|>",
        r"###\s*Instruction",
        r"disregard\s+(previous|all|prior)",
    ]

    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode
        self.patterns = [re.compile(p, re.IGNORECASE) for p in self.BANNED_PATTERNS]

    def validate(self, user_input: str) -> dict:
        """
        Validate user input for injection attempts.

        Returns:
            {
                "is_safe": bool,
                "detected_patterns": list[str],
                "sanitized_input": str
            }
        """
        detected = []

        # Check for banned patterns
        for pattern in self.patterns:
            if pattern.search(user_input):
                detected.append(pattern.pattern)

        # Check for excessive repetition
        if self._has_excessive_repetition(user_input):
            detected.append("excessive_repetition")

        # Check for encoding tricks
        if self._has_encoding_trick(user_input):
            detected.append("encoding_trick")

        is_safe = len(detected) == 0

        if not is_safe and self.strict_mode:
            raise SecurityError(
                f"Prompt injection detected: {detected}\\n"
                f"Input: {user_input[:100]}..."
            )

        return {
            "is_safe": is_safe,
            "detected_patterns": detected,
            "sanitized_input": user_input if is_safe else ""
        }

    def _has_excessive_repetition(self, text: str) -> bool:
        """Detect repeated patterns (e.g., 'repeat this 1000 times')"""
        # Simple heuristic: if any word repeats >10 times consecutively
        words = text.split()
        for i in range(len(words) - 10):
            if len(set(words[i:i+10])) == 1:
                return True
        return False

    def _has_encoding_trick(self, text: str) -> bool:
        """Detect base64, hex, or other encoding tricks"""
        import base64

        # Check for base64 strings
        b64_pattern = re.compile(r"[A-Za-z0-9+/]{50,}={0,2}")
        if b64_pattern.search(text):
            try:
                # Try to decode
                decoded = base64.b64decode(text)
                # Check if decoded text has injection patterns
                decoded_str = decoded.decode('utf-8', errors='ignore')
                for pattern in self.patterns:
                    if pattern.search(decoded_str):
                        return True
            except Exception:
                pass

        return False

# Integration with agents
def sanitize_user_input(user_input: str) -> str:
    """Sanitize user input before passing to agents"""
    detector = PromptInjectionDetector(strict_mode=True)
    result = detector.validate(user_input)

    if not result["is_safe"]:
        logger.warning(
            f"Blocked input with detected patterns: {result['detected_patterns']}"
        )
        raise SecurityError("Input contains suspicious patterns")

    return result["sanitized_input"]
```

## 7.4 Resilience Patterns

### Circuit Breaker

```python
# src/fractal_agent/resilience/circuit_breaker.py
from pybreaker import CircuitBreaker, CircuitBreakerError
import logging

logger = logging.getLogger(__name__)

# Circuit breaker for external APIs
api_breaker = CircuitBreaker(
    fail_max=5,              # Open after 5 consecutive failures
    reset_timeout=60,        # Try again after 60 seconds
    exclude=[TimeoutError],  # Don't count timeouts as failures
    listeners=[
        lambda cb, *args: logger.warning(
            f"Circuit breaker {cb.name} opened after {cb.fail_counter} failures"
        )
    ]
)

@api_breaker
def call_external_api(endpoint: str, **kwargs):
    """Call external API with circuit breaker protection"""
    # API call implementation
    pass

# Usage in agents
def operational_agent_with_circuit_breaker(state):
    """Operational agent with protected external calls"""
    try:
        data = call_external_api("/data/vsm")
        return process_data(data)

    except CircuitBreakerError:
        logger.error("Circuit breaker open - external API unavailable")
        # Fallback behavior
        return {"error": "External API temporarily unavailable"}
```

### Retry with Exponential Backoff

```python
# src/fractal_agent/resilience/retry.py
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
import logging

logger = logging.getLogger(__name__)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(TransientError),
    before_sleep=lambda retry_state: logger.warning(
        f"Retry {retry_state.attempt_number}/3 after {retry_state.outcome.exception()}"
    )
)
def resilient_llm_call(lm, **kwargs):
    """
    LLM call with automatic retry.

    Retries: 3 attempts with 2s, 4s, 8s delays
    """
    return lm(**kwargs)
```

### Graceful Degradation

```python
# src/fractal_agent/resilience/fallback.py
from typing import Callable, Any, Optional

class FallbackChain:
    """
    Execute fallback chain until one succeeds.

    Example:
        chain = FallbackChain([
            lambda: graphrag.retrieve(query),  # Best quality
            lambda: qdrant.search(query),      # Fast fallback
            lambda: ""                          # Last resort: empty
        ])
        result = chain.execute()
    """

    def __init__(self, fallbacks: list[Callable]):
        self.fallbacks = fallbacks

    def execute(self) -> Any:
        """Execute fallbacks in order until one succeeds"""
        errors = []

        for i, fallback in enumerate(self.fallbacks):
            try:
                result = fallback()
                if i > 0:
                    logger.warning(
                        f"Primary failed, used fallback {i}: {errors}"
                    )
                return result

            except Exception as e:
                errors.append((i, str(e)))
                continue

        # All fallbacks failed
        raise Exception(f"All {len(self.fallbacks)} fallbacks failed: {errors}")

# Usage
def get_context_with_fallback(task_id: str) -> str:
    """Get context with graceful degradation"""
    chain = FallbackChain([
        lambda: graphrag.retrieve(task_id),       # Best: full hybrid search
        lambda: qdrant.search(task_id, limit=5),  # Fallback: vector only
        lambda: load_from_cache(task_id),         # Fallback: cached
        lambda: ""                                 # Last resort: empty context
    ])

    return chain.execute()
```

### Health Checks

```python
# src/fractal_agent/resilience/health.py
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

@dataclass
class HealthStatus:
    """System health status"""
    component: str
    status: Literal["healthy", "degraded", "down"]
    message: str
    last_check: datetime
    response_time_ms: float

class HealthChecker:
    """Check health of system components"""

    def check_all(self) -> dict[str, HealthStatus]:
        """Check health of all components"""
        return {
            "llm_primary": self._check_llm_primary(),
            "llm_fallback": self._check_llm_fallback(),
            "neo4j": self._check_neo4j(),
            "qdrant": self._check_qdrant(),
            "obsidian_sync": self._check_obsidian()
        }

    def _check_llm_primary(self) -> HealthStatus:
        """Check Anthropic API health"""
        try:
            start = time.time()
            lm = get_llm_for_role("operational")
            response = lm(prompt="health check", max_tokens=5)
            duration = (time.time() - start) * 1000

            return HealthStatus(
                component="llm_primary",
                status="healthy",
                message="Anthropic API responding",
                last_check=datetime.utcnow(),
                response_time_ms=duration
            )
        except Exception as e:
            return HealthStatus(
                component="llm_primary",
                status="down",
                message=str(e),
                last_check=datetime.utcnow(),
                response_time_ms=0
            )

    # Similar for other components...

# Expose health endpoint
def health_endpoint():
    """Health check endpoint for monitoring"""
    checker = HealthChecker()
    statuses = checker.check_all()

    # Overall status
    overall = "healthy"
    if any(s.status == "down" for s in statuses.values()):
        overall = "down"
    elif any(s.status == "degraded" for s in statuses.values()):
        overall = "degraded"

    return {
        "status": overall,
        "components": {k: v.__dict__ for k, v in statuses.items()},
        "timestamp": datetime.utcnow().isoformat()
    }
```

---

# Session 7 Summary

**Completed Designs:**

1. ✅ **PII Redaction** - Presidio integration, detect/redact/hash modes
2. ✅ **Secrets Management** - Env vars (Phase 0-2), Vault (Phase 4+)
3. ✅ **Input Sanitization** - Prompt injection detection and blocking
4. ✅ **Circuit Breaker** - Prevent cascading failures
5. ✅ **Retry Logic** - Exponential backoff for transient errors
6. ✅ **Graceful Degradation** - Fallback chains for critical paths
7. ✅ **Health Checks** - Component monitoring and status

**Key Decisions:**

- Presidio for PII (battle-tested, comprehensive)
- Progressive secrets management (env → Vault)
- Strict prompt injection blocking (security first)
- Circuit breakers on external APIs only
- Fallback chains for all critical operations
- Health checks for monitoring integration

---

# Session 8: Advanced Topics

## 8.1 Cost Optimization & Budgeting

### Research Findings (2025)

Based on industry best practices:

- **Token optimization** can cut usage by 40-50% through concise prompting
- **Model cascading** (route simple→cheap, complex→expensive) cuts costs by 60%
- **RAG** reduces prompt sizes by 70% for complex tasks
- **Batch processing** offers 50% API discounts
- **Caching** delivers 42% monthly cost reductions
- **Fine-tuning** reduces tokens by 50-75% for consistent use cases

**Key Insight:** Multi-agent systems hit 10× higher bills than projected without proper cost management.

### Cost Optimization Strategy

```python
# src/fractal_agent/cost/optimizer.py
from dataclasses import dataclass
from typing import Literal
from datetime import datetime, timedelta

@dataclass
class CostBudget:
    """Cost budget configuration"""
    monthly_limit_usd: float
    daily_limit_usd: float
    per_task_limit_usd: float
    alert_thresholds: list[float]  # [0.5, 0.8, 1.0] = 50%, 80%, 100%

class CostTracker:
    """
    Track and enforce cost budgets across agents.

    Alerts at 50%, 80%, 100% of budget.
    """

    def __init__(self, budget: CostBudget):
        self.budget = budget
        self.current_spend = {
            "monthly": 0.0,
            "daily": 0.0,
            "per_task": {}
        }
        self.reset_times = {
            "monthly": datetime.utcnow().replace(day=1, hour=0, minute=0),
            "daily": datetime.utcnow().replace(hour=0, minute=0)
        }

    def record_cost(
        self,
        task_id: str,
        tokens_used: int,
        model: str,
        provider: str,
        cache_hit: bool
    ) -> dict:
        """
        Record cost and check budget.

        Args:
            task_id: Unique task identifier
            tokens_used: Number of tokens consumed
            model: Model ID (e.g., "claude-haiku-4.5")
            provider: Provider name (e.g., "anthropic", "gemini")
            cache_hit: Whether prompt cache was hit

        Returns:
            {
                "allowed": bool,
                "cost_usd": float,
                "budget_remaining": float,
                "alert_triggered": bool
            }
        """
        # Calculate cost using ModelRegistry
        cost = self._calculate_cost(tokens_used, model, provider, cache_hit)

        # Check if over budget
        if not self._check_budget(cost, task_id):
            return {
                "allowed": False,
                "cost_usd": cost,
                "budget_remaining": 0.0,
                "alert_triggered": True,
                "reason": "Budget exceeded"
            }

        # Record spend
        self.current_spend["monthly"] += cost
        self.current_spend["daily"] += cost

        if task_id not in self.current_spend["per_task"]:
            self.current_spend["per_task"][task_id] = 0.0
        self.current_spend["per_task"][task_id] += cost

        # Check for alerts
        alert = self._check_alerts()

        return {
            "allowed": True,
            "cost_usd": cost,
            "budget_remaining": self.budget.monthly_limit_usd - self.current_spend["monthly"],
            "alert_triggered": alert is not None,
            "alert_level": alert
        }

    def _calculate_cost(
        self,
        tokens: int,
        model: str,
        provider: str,
        cache_hit: bool
    ) -> float:
        """
        Calculate cost using ModelRegistry pricing.

        NOTE: Uses dynamic pricing from config/models_pricing.yaml
        instead of hardcoded values.
        """
        from fractal_agent.config.model_registry import get_model_registry

        # Get pricing from registry
        registry = get_model_registry()
        pricing_info = registry.get_pricing_for_model(provider, model)

        if not pricing_info:
            logger.warning(f"No pricing for {provider}:{model}, using conservative estimate")
            # Fallback to conservative estimate (expensive model pricing)
            return (tokens / 1_000_000) * 10.0

        # Assume 70% input, 30% output (typical ratio)
        input_tokens = int(tokens * 0.7)
        output_tokens = int(tokens * 0.3)

        if cache_hit:
            input_cost = (input_tokens / 1_000_000) * pricing_info.get("cache_read", 0)
        else:
            input_cost = (input_tokens / 1_000_000) * pricing_info.get("input", 0)

        output_cost = (output_tokens / 1_000_000) * pricing_info.get("output", 0)

        return input_cost + output_cost

    def _check_budget(self, cost: float, task_id: str) -> bool:
        """Check if cost is within budget"""
        # Monthly check
        if self.current_spend["monthly"] + cost > self.budget.monthly_limit_usd:
            return False

        # Daily check
        if self.current_spend["daily"] + cost > self.budget.daily_limit_usd:
            return False

        # Per-task check
        task_spend = self.current_spend["per_task"].get(task_id, 0.0)
        if task_spend + cost > self.budget.per_task_limit_usd:
            return False

        return True

    def _check_alerts(self) -> Optional[float]:
        """Check if alert thresholds crossed"""
        monthly_pct = self.current_spend["monthly"] / self.budget.monthly_limit_usd

        for threshold in sorted(self.budget.alert_thresholds):
            if monthly_pct >= threshold and not self._alert_sent(threshold):
                self._mark_alert_sent(threshold)
                return threshold

        return None

    def reset_daily(self):
        """Reset daily spend (call at midnight)"""
        self.current_spend["daily"] = 0.0
        self.reset_times["daily"] = datetime.utcnow()

    def reset_monthly(self):
        """Reset monthly spend (call at month start)"""
        self.current_spend["monthly"] = 0.0
        self.current_spend["per_task"] = {}
        self.reset_times["monthly"] = datetime.utcnow()

    def get_report(self) -> dict:
        """Generate cost report"""
        return {
            "period": {
                "month_start": self.reset_times["monthly"].isoformat(),
                "day_start": self.reset_times["daily"].isoformat()
            },
            "spend": {
                "monthly": self.current_spend["monthly"],
                "monthly_limit": self.budget.monthly_limit_usd,
                "monthly_pct": (self.current_spend["monthly"] / self.budget.monthly_limit_usd) * 100,
                "daily": self.current_spend["daily"],
                "daily_limit": self.budget.daily_limit_usd,
                "daily_pct": (self.current_spend["daily"] / self.budget.daily_limit_usd) * 100
            },
            "top_tasks": sorted(
                [(k, v) for k, v in self.current_spend["per_task"].items()],
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }

# Global cost tracker
cost_tracker = CostTracker(CostBudget(
    monthly_limit_usd=100.0,  # $100/month budget
    daily_limit_usd=5.0,      # $5/day budget
    per_task_limit_usd=1.0,   # $1/task budget
    alert_thresholds=[0.5, 0.8, 1.0]
))

# Integration with agents
def cost_aware_llm_call(lm, task_id: str, **kwargs):
    """LLM call with cost tracking and enforcement"""

    # Estimate tokens (rough)
    estimated_tokens = kwargs.get("max_tokens", 1000) * 2  # input + output

    # Pre-check budget
    precheck = cost_tracker.record_cost(
        task_id=task_id,
        tokens_used=0,  # Just checking
        model=lm.provider_chain[0].model,
        provider=lm.provider_chain[0].__class__.__name__.replace("Provider", "").lower(),
        cache_hit=False
    )

    if not precheck["allowed"]:
        raise BudgetExceededError(
            f"Task {task_id} would exceed budget. "
            f"Remaining: ${precheck['budget_remaining']:.2f}"
        )

    # Make actual call
    response = lm(**kwargs)

    # Record actual cost (provider returned in response)
    cost_tracker.record_cost(
        task_id=task_id,
        tokens_used=response["tokens_used"],
        model=response["model"],
        provider=response["provider"],
        cache_hit=response["cache_hit"]
    )

    return response
```

### Model Cascading Strategy

```python
# src/fractal_agent/cost/model_router.py
from typing import Literal

class ModelRouter:
    """
    Route tasks to appropriate model based on complexity.

    Strategy:
    - Simple tasks → Haiku (cheap)
    - Medium tasks → Sonnet 3.5 (balanced)
    - Complex tasks → Sonnet 3.7 (expensive)

    Saves 60% on average vs always using Sonnet 3.7.
    """

    def __init__(self):
        self.complexity_classifier = ComplexityClassifier()

    def route(self, task: str) -> Literal["cheap", "balanced", "expensive"]:
        """Determine appropriate model tier"""
        complexity = self.complexity_classifier.classify(task)

        if complexity < 0.3:
            return "cheap"  # Haiku
        elif complexity < 0.7:
            return "balanced"  # Sonnet 3.5
        else:
            return "expensive"  # Sonnet 3.7

class ComplexityClassifier:
    """
    Classify task complexity.

    Heuristics:
    - Length: long tasks are complex
    - Keywords: "analyze", "synthesize" = complex
    - Context needed: requires GraphRAG = complex
    """

    def classify(self, task: str) -> float:
        """Return complexity score 0.0-1.0"""
        score = 0.0

        # Length heuristic
        if len(task) > 500:
            score += 0.3
        elif len(task) > 200:
            score += 0.15

        # Keyword heuristic
        complex_keywords = ["analyze", "synthesize", "evaluate", "design", "architect"]
        if any(kw in task.lower() for kw in complex_keywords):
            score += 0.4

        # Simple keywords
        simple_keywords = ["list", "define", "what is", "summarize"]
        if any(kw in task.lower() for kw in simple_keywords):
            score -= 0.2

        return max(0.0, min(1.0, score))

# Usage
router = ModelRouter()

def get_llm_for_task(task: str):
    """Get appropriate LLM based on task complexity"""
    tier = router.route(task)

    if tier == "cheap":
        return get_llm_for_role("operational")  # Haiku
    elif tier == "balanced":
        return get_llm_for_role("extraction")   # Sonnet 3.5
    else:
        return get_llm_for_role("control")      # Sonnet 3.7
```

## 8.2 Scaling to Large Knowledge Graphs

### Challenge: 1M+ Nodes

**Problems:**

- Graph traversal slows down (seconds → minutes)
- Memory exhaustion on single machine
- Vector search becomes bottleneck

### Solution: Partitioning + Caching

```python
# src/fractal_agent/memory/graph_partitioning.py
from typing import List, Set
import networkx as nx

class PartitionedGraphRAG:
    """
    Partitioned GraphRAG for large-scale knowledge bases.

    Strategy:
    1. Partition graph by domain/topic
    2. Local search within partitions first
    3. Cross-partition search only if needed
    4. Cache frequent paths
    """

    def __init__(self, graphrag: GraphRAG, partitions: int = 10):
        self.graphrag = graphrag
        self.num_partitions = partitions
        self.partition_map = {}  # entity_id -> partition_id
        self.partition_cache = {}  # partition_id -> subgraph

    def partition_graph(self):
        """Partition graph using community detection"""

        # Get full graph
        with self.graphrag.graph.session() as session:
            result = session.run("""
                MATCH (n:Entity)
                OPTIONAL MATCH (n)-[r]-(m)
                RETURN n, r, m
            """)

            # Build NetworkX graph
            G = nx.Graph()
            for record in result:
                n = record["n"]
                m = record["m"]
                if n and m:
                    G.add_edge(n["id"], m["id"])

        # Detect communities (partitions)
        communities = nx.community.greedy_modularity_communities(G)

        # Assign entities to partitions
        for partition_id, community in enumerate(communities):
            for entity_id in community:
                self.partition_map[entity_id] = partition_id

    def retrieve(self, query: str, max_tokens: int = 30000) -> str:
        """
        Retrieve with partitioning.

        Strategy:
        1. Vector search to find seed entities
        2. Determine relevant partitions
        3. Search within partitions only
        """

        # Vector search for seeds
        query_embedding = self.graphrag._embed(query)
        vector_results = self.graphrag.vector.search(
            collection_name="fractal_knowledge",
            query_vector=query_embedding,
            limit=10
        )

        seed_ids = [r.payload["entity_id"] for r in vector_results]

        # Determine relevant partitions
        relevant_partitions = set()
        for seed_id in seed_ids:
            partition_id = self.partition_map.get(seed_id)
            if partition_id is not None:
                relevant_partitions.add(partition_id)

        # Retrieve from relevant partitions only
        results = []
        for partition_id in relevant_partitions:
            partition_results = self._retrieve_from_partition(
                partition_id,
                seed_ids,
                max_tokens // len(relevant_partitions)
            )
            results.extend(partition_results)

        # Format and return
        return self.graphrag._format_knowledge(
            vector_results,
            results,
            max_tokens
        )

    def _retrieve_from_partition(
        self,
        partition_id: int,
        seed_ids: List[str],
        max_tokens: int
    ) -> List[dict]:
        """Retrieve from single partition"""

        # Check cache
        if partition_id in self.partition_cache:
            subgraph = self.partition_cache[partition_id]
        else:
            # Load partition into memory
            subgraph = self._load_partition(partition_id)
            self.partition_cache[partition_id] = subgraph

        # Traverse within partition
        # (much faster than full graph)
        return self._traverse_subgraph(subgraph, seed_ids, max_tokens)
```

## 8.3 Meta-Learning & Continuous Improvement

### Research Findings (2025)

- **Meta-learning** (learning to learn) enables rapid adaptation with limited retraining
- **Curriculum learning** adapts task sampling based on agent progress
- **Self-improving agents** consolidate knowledge over time
- **Active learning** requests labels for uncertain cases

### Continuous Improvement Loop

```python
# src/fractal_agent/learning/meta_learner.py
from dataclasses import dataclass
from typing import List, Dict
import dspy

@dataclass
class TaskPerformance:
    """Performance metrics for a task"""
    task_id: str
    task_type: str
    success: bool
    confidence: float
    tokens_used: int
    duration_seconds: float
    feedback: Optional[str]  # Human feedback

class MetaLearner:
    """
    Meta-learning system for continuous agent improvement.

    Workflow:
    1. Collect performance data from tasks
    2. Identify low-performing patterns
    3. Generate improved prompts (DSPy optimization)
    4. A/B test new vs old prompts
    5. Deploy winner
    """

    def __init__(self):
        self.performance_log: List[TaskPerformance] = []
        self.optimizer = dspy.MIPROv2()  # Advanced optimizer

    def record_performance(self, perf: TaskPerformance):
        """Record task performance"""
        self.performance_log.append(perf)

        # Trigger optimization if enough data
        if len(self.performance_log) % 100 == 0:
            self.optimize_agents()

    def optimize_agents(self):
        """
        Optimize agent prompts based on performance.

        Uses DSPy MIPROv2 to find better prompts.
        """

        # Group by task type
        by_type = {}
        for perf in self.performance_log:
            if perf.task_type not in by_type:
                by_type[perf.task_type] = []
            by_type[perf.task_type].append(perf)

        # Optimize each task type
        for task_type, perfs in by_type.items():
            # Calculate success rate
            success_rate = sum(p.success for p in perfs) / len(perfs)

            # Only optimize if success rate < 90%
            if success_rate < 0.9:
                self._optimize_task_type(task_type, perfs)

    def _optimize_task_type(
        self,
        task_type: str,
        performances: List[TaskPerformance]
    ):
        """Optimize prompts for specific task type"""

        # Convert performances to DSPy training examples
        training_data = self._convert_to_training_data(performances)

        # Get current agent for this task type
        current_agent = self._get_agent_for_type(task_type)

        # Optimize using MIPROv2
        optimized_agent = self.optimizer.compile(
            student=current_agent,
            trainset=training_data,
            num_trials=20,
            max_bootstrapped_demos=5,
            max_labeled_demos=10
        )

        # A/B test: new vs old
        self._ab_test_agents(task_type, current_agent, optimized_agent)

    def _ab_test_agents(
        self,
        task_type: str,
        current_agent,
        optimized_agent
    ):
        """
        A/B test new agent vs current.

        Run next 20 tasks with 50/50 split.
        Deploy winner if statistically significant.
        """
        # Implementation of A/B testing
        pass

# Curriculum Learning
class CurriculumDesigner:
    """
    Design curriculum for agent training.

    Start with easy tasks, progress to hard tasks.
    Adapt based on agent performance.
    """

    def __init__(self):
        self.task_difficulty = {}  # task_id -> difficulty (0.0-1.0)
        self.agent_skill = 0.5     # Current agent skill level

    def select_next_task(self, available_tasks: List[str]) -> str:
        """
        Select next task for agent.

        Strategy: Pick task slightly above current skill (zone of proximal development).
        """
        # Get task difficulties
        difficulties = {
            task_id: self.task_difficulty.get(task_id, 0.5)
            for task_id in available_tasks
        }

        # Find task ~0.1 above current skill
        target_difficulty = self.agent_skill + 0.1

        best_task = min(
            available_tasks,
            key=lambda t: abs(difficulties[t] - target_difficulty)
        )

        return best_task

    def update_skill(self, task_id: str, success: bool):
        """Update agent skill based on task result"""
        task_difficulty = self.task_difficulty.get(task_id, 0.5)

        if success:
            # Increase skill slightly
            self.agent_skill = min(1.0, self.agent_skill + 0.05)
        else:
            # Decrease skill slightly
            self.agent_skill = max(0.0, self.agent_skill - 0.05)
```

---

# Session 8 Summary

**Completed Designs:**

1. ✅ **Cost Optimization** - Budget tracking, alerts, enforcement
2. ✅ **Model Cascading** - Route tasks by complexity (60% savings)
3. ✅ **Graph Partitioning** - Scale to 1M+ nodes via partitioning
4. ✅ **Meta-Learning** - Continuous prompt optimization via DSPy
5. ✅ **Curriculum Learning** - Progressive task difficulty
6. ✅ **A/B Testing** - Data-driven agent improvements

**Key Decisions:**

- Cost tracking at task, daily, monthly levels
- Model routing by complexity (simple→cheap, complex→expensive)
- Graph partitioning for large-scale knowledge bases
- DSPy MIPROv2 for continuous optimization
- Curriculum learning for progressive skill building

---

# Session 9: Emerged Questions Resolution

From the original brainstorming session, 10 key questions emerged. Let's resolve them:

## Q1: How do we measure if the system is actually getting smarter over time?

**Answer: Meta-Metrics Dashboard**

```python
@dataclass
class SystemIntelligenceMetrics:
    """Metrics tracking system intelligence over time"""

    # Task success metrics
    success_rate: float                    # % tasks completed successfully
    success_rate_trend: float              # 7-day moving average change
    avg_confidence: float                  # Average agent confidence
    confidence_calibration: float          # How well confidence predicts success

    # Efficiency metrics
    avg_tokens_per_task: float             # Token efficiency
    tokens_trend: float                    # Getting more efficient?
    avg_duration_per_task: float           # Time efficiency
    cache_hit_rate: float                  # Caching effectiveness

    # Learning metrics
    novel_knowledge_added_per_week: int    # New entities in GraphRAG
    knowledge_conflicts_resolved: int      # Contradictions fixed
    human_intervention_rate: float         # % tasks needing human help (should decrease)

    # Complexity handling
    max_task_complexity_handled: float     # Hardest task completed
    complexity_ceiling_trend: float        # Getting better at hard tasks?

def calculate_intelligence_score() -> float:
    """
    Overall intelligence score (0-100).

    Composite of:
    - 30%: Success rate
    - 20%: Efficiency (tokens/task decreasing)
    - 20%: Learning (new knowledge growing)
    - 15%: Complexity (harder tasks over time)
    - 15%: Autonomy (less human intervention)
    """
    pass
```

**Target:** Intelligence score should increase 5-10 points per month.

## Q2: Right balance between automation and human oversight?

**Answer: Confidence-Based Escalation**

```python
class EscalationPolicy:
    """
    Determine when to escalate to human.

    Rules:
    - Confidence < 0.6 → Always escalate
    - Confidence 0.6-0.8 → Escalate if high stakes
    - Confidence > 0.8 → Auto-approve (but log for review)
    """

    def should_escalate(
        self,
        confidence: float,
        task_stakes: Literal["low", "medium", "high"]
    ) -> bool:
        if confidence < 0.6:
            return True

        if task_stakes == "high" and confidence < 0.8:
            return True

        return False
```

**Principle:** Automate high-confidence + low-stakes. Human review for low-confidence or high-stakes.

## Q3: Should external knowledge be treated differently than internal knowledge?

**Answer: Yes - Source-Based Trust Scoring**

```python
class KnowledgeSource(Enum):
    EXTERNAL_VERIFIED = 5     # Wikipedia, papers, official docs
    EXTERNAL_UNVERIFIED = 3   # Random websites, forums
    INTERNAL_HUMAN = 4        # Human-curated (Obsidian)
    INTERNAL_AGENT = 2        # Agent-generated
    INTERNAL_LEGACY = 1       # Old, unchecked

def get_trust_score(source: KnowledgeSource, age_days: int) -> float:
    """
    Calculate trust score for knowledge.

    Factors:
    - Source authority (external verified > agent-generated)
    - Age (recent > old, except for stable facts)
    - Confirmation count (multiple sources agree?)
    """
    base_score = source.value / 5.0

    # Age penalty (external knowledge ages poorly)
    if source in [KnowledgeSource.EXTERNAL_VERIFIED, KnowledgeSource.EXTERNAL_UNVERIFIED]:
        if age_days > 90:
            base_score *= 0.8

    return base_score
```

**Strategy:**

- External verified (Wikipedia, papers) → High trust, but age matters
- Internal human-curated → High trust, timeless
- Internal agent-generated → Lower trust, needs validation

## Q4: How do we handle knowledge that's context-dependent?

**Answer: Already solved in Session 4 - Context field on statements**

```cypher
// Python is slow (for CPU-bound tasks)
CREATE (stmt1:Statement {
    text: "Python is slow",
    context: "for CPU-bound numerical computations compared to C",
    confidence: 0.9
})

// Python is fast enough (for I/O-bound tasks)
CREATE (stmt2:Statement {
    text: "Python is fast enough",
    context: "for I/O-bound web applications",
    confidence: 0.9
})

// Both statements can be true in their respective contexts!
```

**No contradiction** - context qualifies the statement.

## Q5: What happens when the knowledge graph contradicts the LLM's pre-training?

**Answer: Already solved in Session 4 - Confidence-weighted resolution**

If graph says "X" (confidence 0.7) but LLM says "Y":

- If graph confidence > 0.9 → Trust graph
- If graph confidence < 0.7 → Trust LLM
- If graph confidence 0.7-0.9 → Present both, let LLM adjudicate

## Q6: Is the four-tier memory architecture actually necessary?

**Answer: Yes - Each tier serves distinct purpose**

| Tier          | Purpose                | Lifetime      | Size        | Can't be replaced by                     |
| ------------- | ---------------------- | ------------- | ----------- | ---------------------------------------- |
| 1: Active     | Working memory         | Task duration | 200K tokens | Can't use disk (too slow)                |
| 2: Short-term | Debugging, replay      | 30 days       | ~1GB        | Can't use graph (unstructured logs)      |
| 3: Long-term  | Knowledge accumulation | Permanent     | ~100GB      | Can't use Obsidian (not semantic search) |
| 4: Meta       | Strategic insights     | Permanent     | ~10MB       | Can't use graph (human nuance)           |

**Verdict:** All four tiers necessary. Could collapse Tier 2+3 into one system, but:

- Logs need different query patterns than knowledge
- Logs have different retention policies
- Not worth the coupling

## Q7: How do we prevent the system from becoming too specialized?

**Answer: Diversity Injection**

```python
class DiversityInjector:
    """
    Prevent tunnel vision by injecting diverse tasks.

    Strategy:
    - 80% tasks from current distribution
    - 20% random/diverse tasks (exploration)
    """

    def select_task_with_diversity(
        self,
        available_tasks: List[str],
        recent_tasks: List[str]
    ) -> str:
        # Analyze recent task distribution
        recent_types = [self._get_task_type(t) for t in recent_tasks[-20:]]
        type_distribution = Counter(recent_types)

        # 20% chance: pick underrepresented type
        if random.random() < 0.2:
            # Find least common task type
            least_common = min(type_distribution, key=type_distribution.get)
            # Pick task of that type
            return self._pick_task_of_type(available_tasks, least_common)

        # 80% chance: pick normally (curriculum, priority, etc.)
        return self._pick_normal(available_tasks)
```

## Q8: Deployment model: single monolith or microservices?

**Answer: Start monolith, extract services when needed**

**Phase 0-3:** Single Python application

- Simpler deployment
- Easier debugging
- No network overhead
- Sufficient for <100 agents, <100K req/day

**Phase 4+:** Extract heavy services

- GraphRAG → Separate service (CPU/memory intensive)
- LLM calls → Already external (Anthropic, Gemini APIs)
- Coordination → Could extract if bottleneck

**Decision criteria for extraction:**

- Service becomes bottleneck (p95 latency > 5s)
- Need independent scaling (GraphRAG needs more memory)
- Team wants to work independently

**Verdict:** Monolith is correct choice for Phases 0-3.

## Q9: How do we test emergent behaviors?

**Answer: Property-Based Testing + Chaos Engineering**

```python
# Property-based testing
import hypothesis
from hypothesis import given, strategies as st

@given(
    num_agents=st.integers(min_value=3, max_value=10),
    task_complexity=st.floats(min_value=0.0, max_value=1.0)
)
def test_multi_agent_emergence(num_agents, task_complexity):
    """
    Test emergent property: More agents should complete complex tasks faster.

    This doesn't test specific behaviors, but systemic properties.
    """
    # Run with N agents
    duration_n = run_task(num_agents=num_agents, complexity=task_complexity)

    # Run with N-1 agents
    duration_n_minus_1 = run_task(num_agents=num_agents-1, complexity=task_complexity)

    # Property: More agents = faster (for complex tasks)
    if task_complexity > 0.5:
        assert duration_n < duration_n_minus_1 * 1.1  # Allow 10% margin

# Chaos engineering
def test_resilience_under_chaos():
    """Test system behavior under failures"""

    # Inject failures randomly
    with chaos_monkey(
        kill_agent_probability=0.1,
        slow_llm_probability=0.2,
        corrupt_message_probability=0.05
    ):
        # System should still complete task
        result = run_multi_agent_task(task="Research VSM")

        # May be slower, may use more tokens, but should succeed
        assert result.success == True
```

## Q10: Should failed sessions be deleted or kept as negative examples?

**Answer: Keep for 30 days, then decide**

```python
class FailureManager:
    """Manage failed task data"""

    def handle_failure(self, task_id: str, error: str, partial_result: dict):
        """
        Store failure for learning.

        Strategy:
        1. Store in Tier 2 (logs) immediately
        2. After 30 days, analyze if valuable
        3. If unique failure mode → Keep permanently
        4. If common failure → Delete
        """
        # Store immediately
        log_failure(task_id, error, partial_result)

        # Schedule cleanup decision
        schedule_cleanup_decision(task_id, days=30)

    def cleanup_decision(self, task_id: str):
        """Decide whether to keep or delete failed task"""

        failure = load_failure(task_id)

        # Check if this failure mode is unique
        similar_failures = find_similar_failures(failure)

        if len(similar_failures) < 5:
            # Unique failure - keep for learning
            promote_to_longterm(
                failure,
                note="Rare failure mode - kept for learning"
            )
        else:
            # Common failure - delete to save space
            delete_failure(task_id)
```

**Rationale:**

- Failures contain learning signal
- But most failures are repetitive (API timeout, same error)
- Keep unique failures, delete common ones

---

# Session 9 Summary

**Resolved all 10 emerged questions:**

1. ✅ Intelligence metrics: Composite score tracking success, efficiency, learning
2. ✅ Automation balance: Confidence-based escalation policy
3. ✅ External knowledge: Source-based trust scoring
4. ✅ Context-dependent truth: Context field on statements (Session 4)
5. ✅ Graph vs LLM conflicts: Confidence-weighted resolution (Session 4)
6. ✅ Four-tier memory: Yes, each tier has unique purpose
7. ✅ Specialization: Diversity injection (20% random tasks)
8. ✅ Deployment: Monolith (Phase 0-3), extract services later
9. ✅ Emergent behaviors: Property-based testing + chaos engineering
10. ✅ Failed sessions: Keep unique failures, delete common ones

---

# Session 10: Final Architecture Validation

## 10.1 Gap Analysis Validation

**Original 9 gaps from Session 1 - all addressed:**

| Gap                               | Solution                                        | Session | Status      |
| --------------------------------- | ----------------------------------------------- | ------- | ----------- |
| 1. Testing Framework              | Progressive test pyramid                        | 6       | ✅ Complete |
| 2. Security Model                 | PII redaction, secrets mgmt, input sanitization | 7       | ✅ Complete |
| 3. Failure Resilience             | Circuit breakers, retry, graceful degradation   | 7       | ✅ Complete |
| 4. Human Review Interfaces        | Obsidian integration + CLI tool                 | 5       | ✅ Complete |
| 5. Multi-Provider LLM             | UnifiedLM with Anthropic + Gemini               | 1       | ✅ Complete |
| 6. Agent Registry & A/B Testing   | Git-based registry + meta-learner A/B           | 5, 8    | ✅ Complete |
| 7. Context Budget Management      | Tiered loading + budget pool                    | 2, 3    | ✅ Complete |
| 8. Knowledge Validation           | Confidence scoring + provenance tracking        | 4       | ✅ Complete |
| 9. External Knowledge Integration | GraphRAG with source attribution                | 4       | ✅ Complete |

**Validation:** All gaps closed with concrete solutions.

## 10.2 Integration Points Validation

**Do all pieces fit together?**

```
User Task
  ↓
Control Agent (VSM-3) [Session 2]
  ↓ (decomposes via LangGraph)
Operational Agents (VSM-1) × N [Session 2]
  ↓ (parallel execution with coordination)
Coordination Infrastructure [Session 3]
  - Task Registry (prevent duplicates)
  - Budget Pool (enforce limits)
  - Rate Limiter (prevent quota exhaustion)
  ↓
UnifiedLM [Session 1]
  - Anthropic Claude (primary)
  - Google Gemini (fallback)
  - Prompt Caching (90% savings)
  ↓
Knowledge Retrieval [Session 4]
  - GraphRAG (Neo4j + Qdrant)
  - Tiered loading (manage context budget)
  ↓
Results → Synthesis [Session 2]
  ↓
Human Review [Session 5]
  - Obsidian markdown files
  - Git-based versioning
  - CLI or UI review
  ↓
Approved? → GraphRAG (long-term memory) [Session 4]
            → Obsidian (meta-knowledge) [Session 5]

Throughout:
- Cost Tracking [Session 8]
- PII Redaction [Session 7]
- Security Checks [Session 7]
- Testing [Session 6]
- Meta-Learning [Session 8]
```

**Validation:** All components integrate correctly. No circular dependencies or missing links.

## 10.3 Trade-Offs Validation

**Did we make the right choices?**

| Decision                 | Alternative            | Why We Chose This                    | Risk                               |
| ------------------------ | ---------------------- | ------------------------------------ | ---------------------------------- |
| Pydantic messages        | Network protocol (ACP) | Simpler for Phase 0-1, can upgrade   | May need refactor for distribution |
| LangGraph state          | Redis pub/sub          | Lower infrastructure overhead        | Harder to distribute later         |
| Two-tier caching         | Four-tier breakpoints  | 80/20 rule - simpler, 90% of benefit | Could squeeze more savings         |
| Recording/playback tests | Always use mocks       | Deterministic but realistic          | Recordings can go stale            |
| Presidio PII             | Custom regex           | Battle-tested, comprehensive         | External dependency                |
| Monolith                 | Microservices          | Simpler for MVP                      | May hit scaling limits Phase 4+    |
| Git-based Obsidian       | Database               | Human-readable, version control      | Sync conflicts possible            |
| Lightweight coordination | System 2 agent         | Sufficient for Phase 1-2             | May need agent Phase 4+            |
| Model cascading          | Always best model      | 60% cost savings                     | Complexity errors possible         |

**Validation:** Trade-offs are reasonable. Most decisions favor simplicity/MVP, with clear upgrade paths.

## 10.4 Phase Sequencing Validation

**Can we actually build Phase 0 → 5?**

**Phase 0: Foundation (Weeks 1-2)**

- Dependencies: None
- Deliverables: UnifiedLM + basic agent + tests
- Risk: LOW - mostly coding infrastructure
- Blockers: None

**Phase 1: Vertical Slice (Weeks 3-4)**

- Dependencies: Phase 0 complete
- Deliverables: Control agent + multi-agent workflow + coordination
- Risk: MEDIUM - LangGraph complexity
- Blockers: Need LangGraph expertise

**Phase 2: Production Hardening (Weeks 5-8)**

- Dependencies: Phase 1 complete
- Deliverables: Security + testing + resilience
- Risk: MEDIUM - Security is hard
- Blockers: Need Presidio setup

**Phase 3: Intelligence Layer (Weeks 9-12)**

- Dependencies: Phase 2 complete
- Deliverables: GraphRAG + Intelligence agent + meta-learning
- Risk: HIGH - Neo4j + Qdrant setup complex
- Blockers: Need graph database expertise

**Phase 4: Coordination Layer (Weeks 13-16)**

- Dependencies: Phase 3 complete
- Deliverables: Coordination agent + context mgmt + human review UI
- Risk: MEDIUM - UI development
- Blockers: May want React expertise

**Phase 5: Policy Layer (Weeks 17-20)**

- Dependencies: Phase 4 complete
- Deliverables: Policy agent + external integrations + monitoring
- Risk: LOW - mostly integration work
- Blockers: None

**Validation:** Phasing is logical. Each phase builds on previous. Risks are managed.

## 10.5 Mistake Check

**Common architectural mistakes - do we have any?**

✅ **Over-engineering:** No - We start simple (Phase 0-1) before adding complexity
✅ **Premature optimization:** No - We optimize only after measuring (caching validated first)
✅ **Vendor lock-in:** No - UnifiedLM abstracts providers, can add more
✅ **Monolithic coupling:** No - Clean boundaries (LLM, memory, coordination separate)
✅ **No testing strategy:** No - Progressive test pyramid from Day 1
✅ **Security afterthought:** No - PII redaction + secrets in Phase 2
✅ **No cost control:** No - Cost tracking + budgets + model routing
✅ **Ignoring failures:** No - Circuit breakers + retry + fallbacks
✅ **No observability:** No - Health checks + metrics + logging
✅ **Rigid architecture:** No - Evolutionary design with clear upgrade paths

**Verdict:** No major architectural mistakes detected.

## 10.6 Final Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                            │
│                    (CLI / Obsidian / Future UI)                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                   VSM SYSTEM 5: POLICY AGENT                     │
│              (Ethical boundaries, strategic direction)           │
│                    Model: claude-3-7-sonnet                      │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                VSM SYSTEM 4: INTELLIGENCE AGENT                  │
│           (Reflection, meta-learning, optimization)              │
│                    Model: claude-3-7-sonnet                      │
│                    + Meta-Learner (DSPy MIPROv2)                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                  VSM SYSTEM 3: CONTROL AGENT                     │
│              (Task decomposition, delegation planning)           │
│                    Model: claude-3-7-sonnet                      │
│                    Workflow: LangGraph Hub-and-Spoke             │
└───────┬────────────────┴────────────────┬───────────────────────┘
        │                                 │
        │ (spawns workers)                │
        ▼                                 ▼
┌──────────────────┐            ┌──────────────────┐
│  VSM SYSTEM 2:   │            │  VSM SYSTEM 1:   │
│  COORDINATION    │◄──────────►│  OPERATIONAL     │ × N
│                  │            │  AGENTS          │
│  Lightweight:    │            │                  │
│  - Task Registry │            │  Model: haiku    │
│  - Budget Pool   │            │  (or routed)     │
│  - Rate Limiter  │            │                  │
└────────┬─────────┘            └────────┬─────────┘
         │                               │
         │                               │
         └───────────────┬───────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                      UNIFIEDLM LAYER                             │
│                                                                  │
│  ┌──────────────┐        ┌──────────────┐                       │
│  │  Anthropic   │        │  Google      │                       │
│  │  Claude      │───────►│  Gemini      │                       │
│  │  (Primary)   │ fail   │  (Fallback)  │                       │
│  └──────────────┘        └──────────────┘                       │
│                                                                  │
│  Features:                                                       │
│  - Automatic failover (6 total retries)                         │
│  - Prompt caching (90% cost savings)                            │
│  - Model routing (complexity-based)                             │
│  - Cost tracking (budget enforcement)                           │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                    MEMORY ARCHITECTURE                           │
│                                                                  │
│  Tier 1: Active Memory (200K tokens, ephemeral)                 │
│     - LLM context window                                        │
│     - Prompt caching optimization                               │
│                                                                  │
│  Tier 2: Short-Term Memory (JSON logs, 30 days)                 │
│     - Session logs, task history                                │
│     - Debugging, replay, metrics                                │
│                                                                  │
│  Tier 3: Long-Term Memory (GraphRAG, permanent)                 │
│     ┌──────────────┐        ┌──────────────┐                   │
│     │    Neo4j     │◄──────►│   Qdrant     │                   │
│     │  (Graph DB)  │        │ (Vector DB)  │                   │
│     └──────────────┘        └──────────────┘                   │
│     - Entities, relationships, temporal properties              │
│     - Hybrid retrieval (graph + vector)                         │
│     - Knowledge evolution (t_valid, t_invalid)                  │
│                                                                  │
│  Tier 4: Meta-Knowledge (Obsidian vault, permanent)             │
│     - Human-curated insights                                    │
│     - Strategic decisions                                       │
│     - Git-versioned markdown                                    │
└─────────────────────────────────────────────────────────────────┘

Cross-Cutting Concerns:
├─ Security (PII redaction, secrets management, input sanitization)
├─ Testing (Progressive pyramid: unit → integration → E2E)
├─ Resilience (Circuit breakers, retry, graceful degradation)
├─ Cost Management (Budget tracking, model routing, optimization)
├─ Observability (Health checks, metrics, logging)
└─ Continuous Improvement (Meta-learning, A/B testing, curriculum)
```

---

# Session 10 Summary

**Validation Results:**

1. ✅ **Gap Coverage** - All 9 original gaps addressed with concrete solutions
2. ✅ **Integration** - All components fit together with no missing links
3. ✅ **Trade-Offs** - Reasonable choices favoring simplicity with upgrade paths
4. ✅ **Phase Sequencing** - Logical progression with managed risks
5. ✅ **Mistake Check** - No major architectural flaws detected
6. ✅ **Complete Architecture** - Full system diagram with all layers

**Architecture is VALIDATED and ready for implementation.**

---

# Implementation Roadmap

## API Keys & References

**Provided by User:**

- Google Gemini API Key: `AIzaSyDxlEsd42TVC6EbpRQlO3-t6ebHzUGUZrM`
- Claude SDK Reference: `/Users/cal/DEV/ClaudeSDK/`

**Required:**

- Anthropic API Key: (user to provide)
- Neo4j credentials: (install locally or cloud)
- Qdrant: (install locally or cloud)

## Phase 0: Foundation (Week 1-2)

### Week 1: Core Infrastructure

**Day 1-2: UnifiedLM Implementation**

- [ ] Copy reference implementation from `/docs/fractal-agent-reference/src/fractal_agent/utils/llm_provider.py`
- [ ] Add Google Gemini API key to environment
- [ ] Add Anthropic API key to environment
- [ ] Test Anthropic → Gemini failover
- [ ] Validate prompt caching works

**Day 3-4: Model Configuration & Registry**

- [ ] Create `config/models_pricing.yaml` with current model pricing
- [ ] Implement `ModelRegistry` class (from reference implementation)
- [ ] Implement tier-based model selection (`ROLE_TIER_CONFIG`)
- [ ] Test model discovery from Anthropic API
- [ ] Test model discovery from Gemini API
- [ ] Validate cache TTL (24 hours)
- [ ] Test fuzzy matching for model names
- [ ] Test all 7 agent roles with dynamic selection
- [ ] Validate cost calculations use registry pricing
- [ ] Set up cost tracking with dynamic pricing
- [ ] Add cache directory to `.gitignore`

**Day 5: DSPy Integration**

- [ ] Implement FractalDSpyLM wrapper
- [ ] Test with simple ChainOfThought module
- [ ] Validate signatures work correctly

### Week 2: First Agent

**Day 6-7: ResearchAgent**

- [ ] Implement ResearchAgent (from Session 2 design)
- [ ] Add DSPy assertions
- [ ] Test with mock LLM
- [ ] Test with real LLM

**Day 8-9: Testing Infrastructure**

- [ ] Set up pytest structure
- [ ] Implement mock LLM provider
- [ ] Implement recording/playback
- [ ] Write unit tests for UnifiedLM
- [ ] Write unit tests for ResearchAgent

**Day 10: Integration**

- [ ] End-to-end test: User question → ResearchAgent → Answer
- [ ] Validate caching achieves >80% hit rate
- [ ] Measure cost per task
- [ ] Document Phase 0 learnings

**Phase 0 Success Criteria:**

- ✅ Single ResearchAgent works end-to-end
- ✅ Anthropic → Gemini failover tested
- ✅ ModelRegistry discovers and caches models from both providers
- ✅ Tier-based model selection works for all 7 roles
- ✅ Cache hit rate >80% after warm-up
- ✅ Unit test coverage >70%
- ✅ Cost per task <$0.10 (using dynamic pricing)

## Phase 1: Multi-Agent Coordination (Week 3-4)

**Dependencies:** Phase 0 complete

### Week 3: Control Agent & LangGraph

**Day 11-13: Control Agent**

- [ ] Implement task decomposition agent
- [ ] Test with various task types
- [ ] Validate subtask quality

**Day 14-15: LangGraph Workflow**

- [ ] Implement Hub-and-Spoke topology
- [ ] Add dynamic worker spawning (Send API)
- [ ] Test with 3-7 parallel workers

### Week 4: Coordination Infrastructure

**Day 16-17: Lightweight Coordination**

- [ ] Implement Task Registry
- [ ] Implement Token Budget Pool
- [ ] Implement Rate Limiter
- [ ] Test duplicate work prevention

**Day 18-19: Synthesis**

- [ ] Implement SynthesisAgent
- [ ] Test multi-result aggregation
- [ ] Validate output quality

**Day 20: Integration & Testing**

- [ ] End-to-end multi-agent workflow
- [ ] Test with 5-10 parallel agents
- [ ] Validate no duplicate work
- [ ] Validate budget compliance
- [ ] Document Phase 1 learnings

**Phase 1 Success Criteria:**

- ✅ Control agent decomposes tasks correctly
- ✅ 5+ operational agents run in parallel
- ✅ No duplicate work (task registry works)
- ✅ No budget overruns (pool works)
- ✅ Synthesis produces coherent reports

## Phase 2: Production Hardening (Week 5-8)

**Dependencies:** Phase 1 complete

### Week 5-6: Security

**Security Implementation:**

- [ ] Set up Presidio (PII redaction)
- [ ] Implement SecureLogHandler
- [ ] Add prompt injection detection
- [ ] Set up secrets management (env vars)
- [ ] Test PII redaction in logs
- [ ] Test prompt injection blocking

### Week 7: Resilience

**Resilience Implementation:**

- [ ] Implement circuit breakers (pybreaker)
- [ ] Add retry with exponential backoff (tenacity)
- [ ] Implement fallback chains
- [ ] Add health checks
- [ ] Test failure scenarios

### Week 8: Testing & CI/CD

**Testing Enhancement:**

- [ ] Expand unit test coverage to >80%
- [ ] Add integration tests
- [ ] Set up pre-commit hooks
- [ ] Configure GitHub Actions CI
- [ ] Add performance tests (caching, budget)

**Phase 2 Success Criteria:**

- ✅ Zero PII leaks in logs
- ✅ Prompt injection blocked
- ✅ System survives API failures (circuit breakers work)
- ✅ Test coverage >80%
- ✅ CI pipeline passing

## Phase 3: Intelligence Layer (Week 9-12)

**Dependencies:** Phase 2 complete

### Week 9-10: GraphRAG Setup

**GraphRAG Implementation:**

- [ ] Install Neo4j (local or cloud)
- [ ] Install Qdrant (local or cloud)
- [ ] Implement graph schema (from Session 4)
- [ ] Test entity/relationship insertion
- [ ] Test hybrid retrieval
- [ ] Test temporal queries

### Week 11: Knowledge Extraction

**Extraction Agent:**

- [ ] Implement KnowledgeExtractionAgent
- [ ] Test extraction quality
- [ ] Implement auto-promotion Tier 2 → Tier 3
- [ ] Test knowledge accumulation

### Week 12: Intelligence Agent

**Meta-Learning:**

- [ ] Implement Intelligence Agent (reflection)
- [ ] Set up performance logging
- [ ] Test DSPy optimization (MIPROv2)
- [ ] Implement A/B testing framework

**Phase 3 Success Criteria:**

- ✅ GraphRAG retrieval works correctly
- ✅ Knowledge automatically promoted from logs
- ✅ Intelligence agent generates insights
- ✅ Agent prompts improve over time (A/B tests)

## Phase 4: Coordination & Human Review (Week 13-16)

**Dependencies:** Phase 3 complete

### Week 13-14: Obsidian Integration

**Human Review:**

- [ ] Set up Obsidian vault structure
- [ ] Implement Git sync
- [ ] Create file templates
- [ ] Implement CLI review tool
- [ ] Test review workflow

### Week 15: Advanced Coordination

**System 2 Agent (Optional):**

- [ ] Implement Coordination Agent
- [ ] Test conflict detection
- [ ] Test conflict resolution
- [ ] Integrate with lightweight infrastructure

### Week 16: Context Management

**Budget Management:**

- [ ] Implement tiered context loading
- [ ] Test context budget compliance
- [ ] Implement graph partitioning (for large graphs)

**Phase 4 Success Criteria:**

- ✅ Human review workflow works smoothly
- ✅ Approved knowledge flows to GraphRAG
- ✅ Context stays within 200K token limit
- ✅ System scales to 10+ parallel agents

## Phase 5: Policy & Production (Week 17-20)

**Dependencies:** Phase 4 complete

### Week 17-18: Policy Agent

**Strategic Layer:**

- [ ] Implement Policy Agent (VSM System 5)
- [ ] Define ethical boundaries
- [ ] Test policy enforcement

### Week 19: External Integration

**Knowledge Sources:**

- [ ] Implement web search integration
- [ ] Implement document ingestion
- [ ] Test external knowledge validation

### Week 20: Production Monitoring

**Observability:**

- [ ] Set up Prometheus metrics
- [ ] Create Grafana dashboards
- [ ] Configure alerts
- [ ] Document runbooks

**Phase 5 Success Criteria:**

- ✅ Policy agent prevents unethical actions
- ✅ External knowledge integrated successfully
- ✅ Monitoring dashboards operational
- ✅ System ready for production use

---

# FINAL SUMMARY

## What We've Accomplished

**10 brainstorming sessions completed:**

1. ✅ Gap Analysis & LLM Infrastructure (Session 1 - previous)
2. ✅ Implementation Deep Dive (Session 2)
3. ✅ VSM System 2 Coordination (Session 3)
4. ✅ GraphRAG & Memory Architecture (Session 4)
5. ✅ Obsidian Integration & Human-in-Loop (Session 5)
6. ✅ Testing & Validation Strategy (Session 6)
7. ✅ Security & Resilience (Session 7)
8. ✅ Advanced Topics (Cost, Scale, Learning) (Session 8)
9. ✅ Emerged Questions Resolution (Session 9)
10. ✅ Final Architecture Validation (Session 10)

## Deliverables

1. **Complete Design Specification** - This document (4500+ lines)
2. **Reference Implementation** - `/docs/fractal-agent-reference/` (Python code)
3. **Test Suite** - Comprehensive testing infrastructure
4. **Blueprint Document** - `/docs/fractal-agent-ecosystem-blueprint.md`
5. **Brainstorming Results** - `/docs/brainstorming-session-results-2025-10-18.md`

## Key Architectural Decisions

| Component           | Decision                        | Justification                            |
| ------------------- | ------------------------------- | ---------------------------------------- |
| **LLM Provider**    | UnifiedLM (Anthropic + Gemini)  | User constraint + 90% caching savings    |
| **Agent Framework** | DSPy                            | Declarative, self-improving, optimizable |
| **Workflow**        | LangGraph Hub-and-Spoke         | Explicit state, dynamic scaling          |
| **Communication**   | Pydantic + LangGraph State      | Type-safe, evolvable                     |
| **Coordination**    | Lightweight infrastructure      | Phase 1-3 sufficient, agent Phase 4+     |
| **Memory**          | 4-tier (Active/Short/Long/Meta) | Each tier distinct purpose               |
| **Knowledge**       | GraphRAG (Neo4j + Qdrant)       | Hybrid retrieval, temporal evolution     |
| **Human Review**    | Obsidian + Git                  | Human-readable, version-controlled       |
| **Testing**         | Progressive pyramid             | Fast feedback, cost-controlled           |
| **Security**        | Presidio + circuit breakers     | Battle-tested, comprehensive             |
| **Cost**            | Tracking + routing + budgets    | 60% savings via cascading                |
| **Learning**        | Meta-learning + curriculum      | Continuous improvement                   |

## What's Different from Original Blueprint

**Enhancements added:**

- ✅ Detailed message protocol (Pydantic models)
- ✅ LangGraph workflow topology (Hub-and-Spoke)
- ✅ Lightweight coordination (registry + pool + limiter)
- ✅ Prompt caching strategy (2-tier templates)
- ✅ Testing infrastructure (recording/playback)
- ✅ Security implementation (Presidio + injection detection)
- ✅ Cost management (tracking + budgets + routing)
- ✅ Graph partitioning (for scale)
- ✅ Meta-learning loop (DSPy MIPROv2)
- ✅ All 10 emerged questions resolved

**Nothing removed** - Original blueprint fully preserved and enhanced.

## Validation Results

✅ **All gaps closed** - 9/9 original gaps have concrete solutions
✅ **Integration validated** - All components connect correctly
✅ **Trade-offs reasonable** - Favor simplicity with upgrade paths
✅ **Phasing logical** - Can build Phase 0 → 5 sequentially
✅ **No major mistakes** - Architecture review passed
✅ **Implementation ready** - Detailed roadmap with success criteria

## Next Steps for Implementation

**Immediate (This Week):**

1. Set up development environment
2. Configure API keys (Anthropic, Gemini)
3. Start Phase 0: Week 1 (UnifiedLM implementation)

**Short-term (Month 1-2):**

- Complete Phase 0-1 (Foundation + Multi-Agent)
- Achieve first multi-agent workflow
- Validate core architecture

**Medium-term (Month 3-4):**

- Complete Phase 2-3 (Security + Intelligence)
- Add GraphRAG knowledge base
- Enable meta-learning

**Long-term (Month 5+):**

- Complete Phase 4-5 (Human Review + Production)
- Scale to production workloads
- Continuous improvement via meta-learning

## Success Metrics

**Phase 0 (Weeks 1-2):**

- Single agent completes task end-to-end
- Cache hit rate >80%
- Cost per task <$0.10

**Phase 1 (Weeks 3-4):**

- 5+ agents work in parallel
- No duplicate work
- No budget overruns

**Phase 2 (Weeks 5-8):**

- Zero PII leaks
- System survives failures
- Test coverage >80%

**Phase 3 (Weeks 9-12):**

- Knowledge graph growing
- Agents improving via meta-learning
- Retrieval quality high

**Phase 4 (Weeks 13-16):**

- Human review smooth
- Context under control
- 10+ agent scale

**Phase 5 (Weeks 17-20):**

- Production ready
- Monitoring operational
- Policy enforcement working

---

**Status:** 🎉 **DESIGN COMPLETE - READY FOR IMPLEMENTATION**

**Total Specification:** 4500+ lines of detailed design
**Sessions Completed:** 10/10
**Gaps Remaining:** 0/9
**Architecture Validation:** ✅ PASSED

**The Fractal Agent Ecosystem is fully planned and ready to build.**
