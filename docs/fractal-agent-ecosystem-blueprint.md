# The Fractal Agent Ecosystem: Unified Implementation Blueprint

**Version:** 2.0
**Date:** 2025-10-18
**Author:** BMad
**Status:** Implementation Ready

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Part I: Vision & Principles](#part-i-vision--principles)
3. [Part II: System Architecture](#part-ii-system-architecture)
4. [Part III: LLM Infrastructure](#part-iii-llm-infrastructure)
5. [Part IV: Implementation Strategy](#part-iv-implementation-strategy)
6. [Part V: Production Readiness](#part-v-production-readiness)
7. [Appendices](#appendices)

---

## Executive Summary

The Fractal Agent Ecosystem is a self-organizing, multi-agent system built on three foundational principles:

1. **Viable System Model (VSM)** - Hierarchical organizational structure with 5 systems
2. **DSPy Framework** - Declarative, self-optimizing agent modules
3. **Four-Tier Memory Architecture** - Active, short-term, long-term, and meta-knowledge

This blueprint provides a complete implementation guide from MVP to production, with emphasis on:

- **Unified LLM Infrastructure** - Single interface for all AI calls with automatic failover
- **Cost Optimization** - 90% savings via prompt caching and tiered model selection
- **Extensibility** - Provider chain architecture for future expansion
- **Production Readiness** - Testing, security, resilience, and monitoring

**Implementation Reference:** Complete Python implementation available in `/docs/fractal-agent-reference/`

---

## Part I: Vision & Principles

### 1.1 The Fractal Hypothesis

Just as complex systems exhibit self-similarity across scales (coastlines, ferns, organizations), intelligent agent systems should demonstrate recursive patterns:

- **Fractal Structure**: Each agent contains sub-agents with the same organizational principles
- **Self-Organization**: Agents emerge, evolve, and adapt based on environmental needs
- **Scale Invariance**: Patterns that work at micro-level (single agent) work at macro-level (agent swarms)

### 1.2 Core Architectural Principles

#### Principle 1: Unified LLM Interface

**ALL LLM calls in the system route through a single interface.**

- Single point of configuration (change models globally)
- Automatic failover between providers (resilience)
- Built-in cost optimization (caching, tiered selection)
- Extensible for future providers (registry pattern)

**Implementation:** `UnifiedLM` class with provider chain architecture

#### Principle 2: Declarative Agent Design (DSPy)

Agents are **specifications**, not implementations.

- Define what agents should do (signatures), not how
- Automatic optimization via DSPy compilers
- Version control for prompts and behaviors
- A/B testing built into the framework

**Implementation:** `FractalDSpyLM` wrapper integrating UnifiedLM with DSPy

#### Principle 3: Hierarchical Organization (VSM)

Five-system model from Viable System Model theory:

1. **System 1 (Operations)** - Task execution agents (researchers, analysts)
2. **System 2 (Coordination)** - Conflict resolution and resource arbitration
3. **System 3 (Control)** - Task decomposition and delegation planning
4. **System 4 (Intelligence)** - Reflection, learning, pattern detection
5. **System 5 (Policy)** - Ethical boundaries and strategic direction

**Implementation:** Role-based model selection via `get_llm_for_role()`

#### Principle 4: Four-Tier Memory

Information flows through four memory tiers with increasing permanence:

1. **Active Memory** - Agent working memory (conversation context)
2. **Short-Term** - Session logs, task history (JSON files)
3. **Long-Term** - GraphRAG knowledge base (Neo4j + Qdrant)
4. **Meta-Knowledge** - Human-curated wisdom (Obsidian vault)

**Implementation:** Temporal graph properties (t_valid, t_invalid) for knowledge evolution

---

## Part II: System Architecture

### 2.1 Agent Hierarchy (VSM Systems)

```
┌─────────────────────────────────────────────┐
│         System 5: Policy Agent              │
│   (Ethical boundaries, strategic direction) │
│         Model: claude-3-7-sonnet            │
└─────────────────────────────────────────────┘
                     │
┌─────────────────────────────────────────────┐
│       System 4: Intelligence Agent          │
│  (Reflection, learning, pattern detection)  │
│         Model: claude-3-7-sonnet            │
└─────────────────────────────────────────────┘
                     │
┌─────────────────────────────────────────────┐
│         System 3: Control Agent             │
│   (Task decomposition, delegation planning) │
│         Model: claude-3-7-sonnet            │
└─────────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
┌───────────────────┐   ┌───────────────────┐
│  System 2: Coord  │   │  System 1: Ops    │
│  (Arbitration)    │   │  (Execution)      │
│  Sonnet 3.5       │   │  Haiku            │
└───────────────────┘   └───────────────────┘
```

### 2.2 Agent Workflow (LangGraph)

Agents operate as stateful graphs with:

- **Nodes**: Agent actions (research, analyze, synthesize)
- **Edges**: State transitions with conditional routing
- **State**: Shared context across agent invocations
- **Checkpoints**: Resumable execution for long-running tasks

**Key Pattern**: Human-in-the-loop approval gates at critical transitions

### 2.3 Memory Architecture

#### Active Memory (Context Window)

- **Size**: 200K tokens (Claude 3.5 Sonnet)
- **Optimization**: Prompt caching for system messages (90% cost reduction)
- **Structure**: System → Tools → History → Current Task

#### Short-Term Memory (Logs)

- **Storage**: JSON files per session
- **Schema**: Structured logs with agent_id, task_id, inputs, outputs, metrics
- **Retention**: 30 days (configurable)
- **Purpose**: Debugging, auditing, performance analysis

#### Long-Term Memory (GraphRAG)

- **Graph DB**: Neo4j for entity relationships
- **Vector DB**: Qdrant for semantic search
- **Temporal Properties**:
  - `t_valid`: Timestamp when knowledge became valid
  - `t_invalid`: Timestamp when superseded (null = still valid)
- **Query Pattern**: Hybrid graph traversal + vector similarity

#### Meta-Knowledge (Obsidian)

- **Format**: Markdown with YAML frontmatter
- **Structure**: Zettelkasten-style atomic notes
- **Integration**: Git-based versioning, human-editable
- **Purpose**: Strategic insights, domain expertise, ethical guidelines

### 2.4 Communication Protocols

Agents communicate via structured messages:

```python
{
    "from": "control_agent_001",
    "to": "operational_agent_042",
    "task_id": "T-2025-10-18-001",
    "message_type": "task_assignment",
    "payload": {
        "task": "Research VSM System 2 coordination patterns",
        "constraints": {"max_tokens": 1000, "deadline": "2025-10-18T15:00:00Z"},
        "context": {...}
    },
    "timestamp": "2025-10-18T14:30:00Z"
}
```

---

## Part III: LLM Infrastructure

### 3.1 Architecture Overview

**Core Principle:** ALL LLM calls in the Fractal Agent Ecosystem route through a single unified interface.

**Benefits:**

- **Consistency**: Same behavior across all agents
- **Resilience**: Automatic failover between providers
- **Cost Optimization**: Built-in caching and tiered model selection
- **Flexibility**: Change models system-wide via configuration
- **Extensibility**: Add new providers without changing agent code

### 3.2 UnifiedLM Core

The `UnifiedLM` class is the central LLM interface:

```python
from fractal_agent.utils.llm_provider import UnifiedLM

# Initialize with provider chain (priority order)
lm = UnifiedLM(
    providers=[
        ("anthropic", "claude-3-5-haiku-20241022"),
        ("gemini", "gemini-2.0-flash-exp")
    ],
    enable_caching=True
)

# Make a call (tries Anthropic, falls back to Gemini on failure)
response = lm(
    prompt="What is the Viable System Model?",
    max_tokens=1000
)

# Access result
print(response["text"])           # Generated text
print(response["provider"])       # "anthropic" or "gemini"
print(response["cache_hit"])      # True if cache was used
print(response["tokens_used"])    # Total tokens consumed
```

### 3.3 Provider Chain Architecture

UnifiedLM tries each provider in priority order until one succeeds:

1. **Primary Provider** (e.g., Anthropic Claude)
   - 3 retry attempts with exponential backoff
   - If all retries fail → Continue to next provider

2. **Fallback Provider** (e.g., Google Gemini)
   - 3 retry attempts with exponential backoff
   - If all retries fail → Raise exception

**Total Resilience:** 6 attempts before failure (3 per provider × 2 providers)

### 3.4 Provider Interface

New providers can be added by implementing the `LLMProvider` abstract base class:

```python
from fractal_agent.utils.llm_provider import LLMProvider, PROVIDER_REGISTRY

class CustomProvider(LLMProvider):
    """Custom LLM provider implementation"""

    def __init__(self, model: str, **config):
        super().__init__(model, **config)
        # Provider-specific initialization

    def _call_provider(self, messages: List[Dict], **kwargs) -> Dict:
        """
        Implement provider-specific API call.

        Must return: {
            "text": str,
            "tokens_used": int,
            "cache_hit": bool,
            "provider": str,
            "model": str
        }
        """
        # Implementation here
        pass

# Register provider for use in UnifiedLM
PROVIDER_REGISTRY["custom"] = CustomProvider
```

### 3.5 Model Configuration by Role

Different agent roles use different models optimized for their tasks:

```python
from fractal_agent.config.model_config import get_llm_for_role

# Operational agents (VSM System 1) - Fast & cheap
lm_ops = get_llm_for_role("operational")
# → claude-3-5-haiku → gemini-2.0-flash

# Control agents (VSM System 3) - Smart reasoning
lm_control = get_llm_for_role("control")
# → claude-3-7-sonnet → gemini-2.0-flash-thinking

# Intelligence agents (VSM System 4) - Deep analysis
lm_intel = get_llm_for_role("intelligence")
# → claude-3-7-sonnet → gemini-exp-1206
```

**Complete Role Matrix:**

| Role             | Primary Model     | Fallback Model            | Cost Tier | Use Case                         |
| ---------------- | ----------------- | ------------------------- | --------- | -------------------------------- |
| **operational**  | claude-3-5-haiku  | gemini-2.0-flash          | Low       | Simple tasks, data retrieval     |
| **control**      | claude-3-7-sonnet | gemini-2.0-flash-thinking | High      | Task decomposition, planning     |
| **intelligence** | claude-3-7-sonnet | gemini-exp-1206           | High      | Reflection, strategic thinking   |
| **extraction**   | claude-3-5-sonnet | gemini-2.0-flash          | Medium    | Log parsing, entity extraction   |
| **synthesis**    | claude-3-7-sonnet | gemini-exp-1206           | High      | Report generation, documentation |
| **coordination** | claude-3-5-sonnet | gemini-2.0-flash-thinking | Medium    | Conflict resolution, arbitration |
| **policy**       | claude-3-7-sonnet | gemini-exp-1206           | High      | Ethical decisions, boundaries    |

**Configuration is centralized** - Change models globally in `model_config.py`:

```python
MODEL_CONFIG = {
    "operational": {
        "primary": "claude-3-5-haiku-20241022",
        "fallback": "gemini-2.0-flash-exp",
        "rationale": "High speed, low cost, good for simple tasks",
        "typical_use": "Research, data retrieval, simple analysis"
    },
    # ... other roles
}
```

### 3.6 DSPy Integration

All DSPy agents use `FractalDSpyLM` wrapper for UnifiedLM compatibility:

```python
from fractal_agent.utils.dspy_integration import configure_dspy_for_role
import dspy

# Configure DSPy for operational role
lm = configure_dspy_for_role("operational")

# Define DSPy module
class ResearchAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.research = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        return self.research(question=question)

# Use agent (automatically uses configured UnifiedLM)
agent = ResearchAgent()
result = agent("What is VSM System 2?")

# Access metrics
metrics = lm.get_metrics()
print(f"Calls: {metrics['total_calls']}")
print(f"Cache hits: {metrics['cache_hits']}")
print(f"Tokens: {metrics['total_tokens']}")
```

### 3.7 Prompt Caching (Anthropic)

**Game-Changing Feature:** 90% cost reduction for repeated system messages.

**How It Works:**

- Anthropic caches the beginning of prompts (system messages, tools, etc.)
- Subsequent calls with same cached content → 10% of input token cost
- Cache TTL: 5 minutes (refreshed on each use)

**Best Practices:**

1. **Put stable content first** - System messages, tool definitions, knowledge
2. **Put variable content last** - User query, current task, dynamic context
3. **Structure for reuse** - Same system message across all calls in a session

**Example Structure:**

```python
messages = [
    {
        "role": "system",
        "content": "You are a VSM System 1 operational agent...",
        "cache_control": {"type": "ephemeral"}  # Cache this
    },
    {
        "role": "user",
        "content": f"Research: {dynamic_query}"  # Not cached (changes each call)
    }
]
```

**Cost Impact:**

- **Without caching**: 100,000 input tokens × $3/MTok = $0.30
- **With caching**: 100,000 cached tokens × $0.30/MTok = $0.03
- **Savings**: 90% reduction

### 3.8 Metrics & Monitoring

UnifiedLM tracks comprehensive metrics:

```python
metrics = lm.get_metrics()

# Output:
{
    "total_calls": 42,
    "total_failures": 0,
    "failure_rate": 0.0,
    "total_tokens": 125000,
    "cache_hits": 38,
    "cache_misses": 4,
    "cache_hit_rate": 0.90,
    "provider_chain": [
        {"provider": "anthropic", "model": "claude-3-5-haiku-20241022"},
        {"provider": "gemini", "model": "gemini-2.0-flash-exp"}
    ],
    "per_provider": [
        {
            "provider": "anthropic",
            "model": "claude-3-5-haiku-20241022",
            "calls": 42,
            "tokens_used": 125000,
            "cache_hits": 38,
            "cache_misses": 4
        },
        {
            "provider": "gemini",
            "model": "gemini-2.0-flash-exp",
            "calls": 0,
            "tokens_used": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
    ]
}
```

**Global Metrics Aggregation:**

```python
from fractal_agent.utils.llm_provider import LLMMetricsAggregator

# Register UnifiedLM instances for global tracking
LLMMetricsAggregator.register(lm1)
LLMMetricsAggregator.register(lm2)

# Get system-wide metrics
global_metrics = LLMMetricsAggregator.get_global_metrics()
```

### 3.9 Implementation Files

Complete reference implementation:

```
fractal-agent-reference/
├── src/
│   └── fractal_agent/
│       ├── utils/
│       │   ├── llm_provider.py          # UnifiedLM core (487 lines)
│       │   └── dspy_integration.py      # DSPy wrapper (180 lines)
│       └── config/
│           └── model_config.py          # Role-based config (185 lines)
└── tests/
    └── test_llm_provider.py             # Comprehensive tests (550+ lines)
```

**Key Files:**

- **llm_provider.py** - `UnifiedLM`, `LLMProvider` base class, `AnthropicProvider`, `GeminiProvider`, `PROVIDER_REGISTRY`
- **model_config.py** - `MODEL_CONFIG` dict, `get_llm_for_role()` factory
- **dspy_integration.py** - `FractalDSpyLM` wrapper, `configure_dspy_for_role()`
- **test_llm_provider.py** - 20+ test cases covering all scenarios

### 3.10 Usage Examples

#### Example 1: Simple Operational Agent

```python
from fractal_agent.config.model_config import get_llm_for_role

# Get LLM for operational role (uses Haiku for cost efficiency)
lm = get_llm_for_role("operational")

# Make a research call
response = lm(
    prompt="List the 5 key principles of VSM",
    max_tokens=500
)

print(response["text"])
# Automatically used claude-3-5-haiku-20241022
# Fell back to gemini-2.0-flash-exp if Anthropic failed
```

#### Example 2: Control Agent with Custom Models

```python
# Override default models for control agent
lm = get_llm_for_role(
    "control",
    providers=[
        ("anthropic", "claude-3-5-haiku-20241022"),  # Cheaper for testing
        ("gemini", "gemini-2.0-flash-exp")
    ]
)

response = lm(
    messages=[
        {"role": "system", "content": "You are a task decomposition agent"},
        {"role": "user", "content": "Break down: Build a VSM system"}
    ],
    max_tokens=2000
)
```

#### Example 3: DSPy Module with Intelligence Role

```python
from fractal_agent.utils.dspy_integration import configure_dspy_for_role
import dspy

# Configure DSPy for intelligence role (uses Sonnet 3.7 for quality)
lm = configure_dspy_for_role("intelligence")

# Define reflection module
class ReflectionAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.reflect = dspy.ChainOfThought(
            "session_logs, performance_metrics -> insights, improvements"
        )

    def forward(self, session_logs, performance_metrics):
        return self.reflect(
            session_logs=session_logs,
            performance_metrics=performance_metrics
        )

# Use agent
agent = ReflectionAgent()
result = agent(
    session_logs="...",
    performance_metrics={"accuracy": 0.85, "cost": 12.50}
)

print(result.insights)
print(result.improvements)

# Check what actually ran
history = lm.inspect_history(n=1)
print(f"Used: {history[0]['model']}")
print(f"Cache hit: {history[0]['cache_hit']}")
```

#### Example 4: Multi-Role System

```python
from fractal_agent.utils.dspy_integration import create_multi_role_config

# Create LLM instances for all roles
lms = create_multi_role_config()

# Use different LLMs for different agents
operational_lm = lms["operational"]   # Haiku - cheap
control_lm = lms["control"]           # Sonnet 3.7 - smart
intelligence_lm = lms["intelligence"] # Sonnet 3.7 - deep

# Each agent uses appropriate model for its role
```

---

## Part IV: Implementation Strategy

### 4.1 Phased Rollout

#### Phase 0: Foundation (Week 1-2)

**Goal:** Establish core infrastructure

- [x] UnifiedLM implementation (`llm_provider.py`)
- [x] Model configuration system (`model_config.py`)
- [x] DSPy integration (`dspy_integration.py`)
- [x] Comprehensive test suite (`test_llm_provider.py`)
- [ ] Basic operational agent (DSPy module)
- [ ] Simple LangGraph workflow (linear: research → analyze → report)

**Success Criteria:**

- Single operational agent completes simple task end-to-end
- Anthropic → Gemini failover works
- Prompt caching achieves >80% hit rate after warm-up

#### Phase 1: Vertical Slice (Week 3-4)

**Goal:** Complete VSM hierarchy with minimal features

- [ ] Control agent (task decomposition)
- [ ] Multiple operational agents (parallel execution)
- [ ] Short-term memory (JSON logs)
- [ ] Human review interface (Obsidian integration)

**Use Case:** Control agent decomposes "Research VSM" → spawns 5 operational agents → writes synthesis report

**Success Criteria:**

- Multi-agent coordination works
- Logs capture full task tree
- Human can review and approve in Obsidian

#### Phase 2: Production Hardening (Week 5-8)

**Goal:** Make system reliable and secure

**From Brainstorming Session - Priority #1-3:**

1. **Multi-Provider LLM** ✅ COMPLETE
   - UnifiedLM with Anthropic + Gemini
   - Automatic failover
   - Prompt caching (90% savings)

2. **Testing Framework**
   - Minimal: pytest + DSPy assertions
   - Integration tests for agent workflows
   - Mocks for LLM calls (cost control)

3. **Security Model**
   - PII redaction (Presidio library)
   - Secrets management (environment variables)
   - Input sanitization for prompts

**Success Criteria:**

- 95% uptime (measured over 1 week)
- Zero PII leaks in logs
- Test coverage >80%

#### Phase 3: Intelligence Layer (Week 9-12)

**Goal:** Add learning and optimization

- [ ] Intelligence agent (System 4) - Performance reflection
- [ ] Long-term memory (GraphRAG with Neo4j + Qdrant)
- [ ] A/B testing framework for agent variants
- [ ] Automated prompt optimization (DSPy compilers)

**Success Criteria:**

- Intelligence agent identifies performance improvement opportunities
- GraphRAG retrieval improves task accuracy by 15%
- Optimized prompts reduce token usage by 20%

#### Phase 4: Coordination Layer (Week 13-16)

**Goal:** Enable large-scale agent swarms

- [ ] Coordination agent (System 2) - Conflict resolution
- [ ] Resource arbitration (token budgets, rate limits)
- [ ] Context budget management (tiered loading)
- [ ] Advanced human review (React Admin dashboard)

**Success Criteria:**

- 10+ agents work without conflicts
- Context stays within 200K token limit
- Human can inspect any agent's state via dashboard

#### Phase 5: Policy Layer (Week 17-20)

**Goal:** Strategic direction and ethics

- [ ] Policy agent (System 5) - Ethical boundaries
- [ ] External knowledge integration (API connectors)
- [ ] Knowledge validation framework
- [ ] Production monitoring (Prometheus + Grafana)

**Success Criteria:**

- Policy agent prevents ethical violations
- System learns from external sources (papers, docs)
- False knowledge detected and flagged

### 4.2 Development Workflow

```
1. Design (DSPy Signature)
   ↓
2. Implement (Python module)
   ↓
3. Test (pytest + DSPy assertions)
   ↓
4. Deploy (Git versioning)
   ↓
5. Monitor (Metrics collection)
   ↓
6. Optimize (DSPy compilers) → Back to step 1
```

### 4.3 Critical Success Factors

1. **Start with Real Tasks**: Build agents for actual work, not toy problems
2. **Measure Everything**: Token usage, cache hit rate, task success rate, cost
3. **Incremental Complexity**: Add one VSM system at a time
4. **Human-in-Loop Early**: Catch issues before they compound
5. **Cost Consciousness**: Use caching and tiered models from day 1

---

## Part V: Production Readiness

### 5.1 Testing Framework

**Philosophy:** Progressive testing - start minimal, expand as system matures

#### Level 1: Unit Tests (pytest)

```python
def test_operational_agent():
    agent = OperationalAgent()
    result = agent.research(question="What is VSM?")
    assert len(result.answer) > 100
    assert "viable system" in result.answer.lower()
```

#### Level 2: DSPy Assertions

```python
class ResearchAgent(dspy.Module):
    def forward(self, question):
        result = self.research(question=question)

        # DSPy assertion - fails if not met
        dspy.Assert(
            len(result.answer) > 50,
            "Answer must be substantive (>50 chars)"
        )

        return result
```

#### Level 3: Integration Tests

```python
def test_control_to_operational_workflow():
    # Spawn control agent
    control = ControlAgent()

    # Decompose task
    subtasks = control.decompose("Research VSM Systems 1-5")

    # Execute via operational agents
    results = [OperationalAgent().execute(st) for st in subtasks]

    # Verify all succeeded
    assert all(r.success for r in results)
```

#### Level 4: End-to-End Tests

```python
def test_full_vsm_research_pipeline():
    # Control agent decomposes
    # Operational agents execute in parallel
    # Synthesis agent generates report
    # Human reviews in Obsidian
    # Intelligence agent reflects on performance

    pipeline = ResearchPipeline()
    report = pipeline.run("Analyze VSM applicability to AI systems")

    assert report.sections == ["intro", "analysis", "conclusion"]
    assert pipeline.metrics.cache_hit_rate > 0.8
    assert pipeline.metrics.total_cost < 5.00  # Budget constraint
```

### 5.2 Security Model

**Threat Model:**

- PII leakage into logs/prompts
- Secrets exposure (API keys)
- Prompt injection attacks
- Unauthorized access to knowledge base

**Mitigations:**

#### PII Redaction (Presidio)

```python
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

# Before logging or sending to LLM
text = "John Doe's email is john@example.com"
results = analyzer.analyze(text=text, language='en')
anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
# Output: "<PERSON>'s email is <EMAIL_ADDRESS>"
```

#### Secrets Management

```python
# NEVER hardcode API keys
# Use environment variables
import os

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# For production: Use secrets manager (AWS Secrets Manager, HashiCorp Vault)
```

#### Input Sanitization

```python
def sanitize_prompt(user_input: str) -> str:
    # Remove prompt injection attempts
    banned_patterns = [
        "ignore previous instructions",
        "you are now",
        "system:",
        "assistant:"
    ]

    for pattern in banned_patterns:
        if pattern in user_input.lower():
            raise ValueError(f"Suspected prompt injection: {pattern}")

    return user_input
```

### 5.3 Failure Resilience

**Resilience Patterns:**

#### Circuit Breaker (pybreaker)

```python
from pybreaker import CircuitBreaker

# Prevent cascading failures
breaker = CircuitBreaker(
    fail_max=5,        # Open circuit after 5 failures
    reset_timeout=60   # Try again after 60 seconds
)

@breaker
def call_external_api():
    # API call that might fail
    pass
```

#### Retry with Exponential Backoff (tenacity)

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def unreliable_operation():
    # Automatically retries with: 2s, 4s, 8s delays
    pass
```

#### Graceful Degradation

```python
def get_context_for_task(task_id):
    try:
        # Try GraphRAG (best quality)
        return graphrag.retrieve(task_id)
    except Exception:
        try:
            # Fallback to vector search (fast)
            return qdrant.search(task_id)
        except Exception:
            # Last resort: empty context
            return {}
```

### 5.4 Context Budget Management

**Problem:** Claude 3.5 Sonnet has 200K token limit. Complex tasks can exceed this.

**Solution:** Tiered context loading

```python
def load_context_tiered(task, budget=200000):
    context = {}
    remaining = budget

    # Tier 1: Always load (critical context)
    system_msg = load_system_message()  # 2K tokens
    remaining -= 2000
    context["system"] = system_msg

    # Tier 2: Recent history (if space available)
    if remaining > 10000:
        history = load_recent_history(max_tokens=min(10000, remaining))
        remaining -= len(history)
        context["history"] = history

    # Tier 3: GraphRAG retrieval (remaining space)
    if remaining > 5000:
        knowledge = graphrag.retrieve(task, max_tokens=remaining - 1000)
        context["knowledge"] = knowledge

    return context
```

### 5.5 Knowledge Validation

**Challenge:** LLMs can generate false information ("hallucinations")

**Solution:** Multi-stage validation

```python
class KnowledgeValidator:
    def validate(self, claim: str) -> ValidationResult:
        # Stage 1: Source attribution
        sources = self.find_sources(claim)
        if not sources:
            return ValidationResult(valid=False, reason="No sources found")

        # Stage 2: Consistency check
        contradictions = self.check_contradictions(claim, sources)
        if contradictions:
            return ValidationResult(valid=False, reason=f"Contradicts: {contradictions}")

        # Stage 3: Temporal validity
        if self.is_time_sensitive(claim):
            age = self.get_source_age(sources)
            if age > timedelta(days=90):
                return ValidationResult(valid=False, reason="Outdated information")

        # Stage 4: Confidence scoring
        confidence = self.calculate_confidence(claim, sources)

        return ValidationResult(
            valid=True,
            confidence=confidence,
            sources=sources
        )
```

### 5.6 Monitoring & Observability

**Key Metrics:**

```python
# Cost metrics
- total_cost_usd
- cost_per_task
- cache_hit_rate
- tokens_per_task

# Performance metrics
- task_success_rate
- average_task_duration
- agent_utilization
- context_window_usage

# Reliability metrics
- provider_failure_rate
- retry_rate
- circuit_breaker_trips
- error_rate_by_type
```

**Implementation:**

```python
# Prometheus metrics
from prometheus_client import Counter, Histogram

llm_calls = Counter('llm_calls_total', 'Total LLM calls', ['provider', 'role'])
llm_duration = Histogram('llm_duration_seconds', 'LLM call duration')
llm_cost = Counter('llm_cost_usd', 'LLM cost in USD', ['provider', 'model'])

# Grafana dashboard
# - Time series: Cost over time
# - Gauge: Current cache hit rate
# - Bar chart: Calls by provider
# - Alert: Cost > $100/day
```

---

## Appendices

### Appendix A: Quick Start Guide

```bash
# 1. Clone repository
git clone https://github.com/yourusername/fractal-agent-ecosystem
cd fractal-agent-ecosystem

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set API keys
export ANTHROPIC_API_KEY=your_key_here
export GOOGLE_API_KEY=your_key_here

# 4. Run tests
pytest tests/

# 5. Run example agent
python examples/simple_research_agent.py
```

### Appendix B: Dependencies

```
# Core
anthropic>=0.18.0
google-generativeai>=0.3.0
dspy-ai>=2.0.0
langgraph>=0.0.20

# Resilience
tenacity>=8.2.0
pybreaker>=1.0.0

# Security
presidio-analyzer>=2.2.0
presidio-anonymizer>=2.2.0

# Memory
neo4j>=5.14.0
qdrant-client>=1.7.0

# Testing
pytest>=7.4.0
pytest-asyncio>=0.21.0

# Monitoring
prometheus-client>=0.19.0
```

### Appendix C: File Structure

```
fractal-agent-ecosystem/
├── src/
│   └── fractal_agent/
│       ├── agents/
│       │   ├── operational.py      # VSM System 1 agents
│       │   ├── coordination.py     # VSM System 2 agents
│       │   ├── control.py          # VSM System 3 agents
│       │   ├── intelligence.py     # VSM System 4 agents
│       │   └── policy.py           # VSM System 5 agents
│       ├── workflows/
│       │   ├── research.py         # Research pipeline
│       │   └── synthesis.py        # Synthesis pipeline
│       ├── memory/
│       │   ├── graphrag.py         # GraphRAG integration
│       │   └── obsidian.py         # Meta-knowledge interface
│       ├── utils/
│       │   ├── llm_provider.py     # UnifiedLM core
│       │   └── dspy_integration.py # DSPy wrapper
│       └── config/
│           └── model_config.py     # Role-based model config
├── tests/
│   ├── test_llm_provider.py
│   ├── test_agents.py
│   └── test_workflows.py
├── docs/
│   ├── fractal-agent-ecosystem-blueprint.md  # This document
│   ├── fractal-agent-llm-architecture.md     # Detailed LLM docs
│   └── brainstorming-session-results.md      # Gap analysis
├── examples/
│   ├── simple_research_agent.py
│   └── multi_agent_workflow.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

### Appendix D: Glossary

- **VSM**: Viable System Model - Organizational cybernetics framework with 5 systems
- **DSPy**: Declarative Self-improving Python - Framework for LLM agents
- **LangGraph**: Graph-based workflow orchestration for agents
- **GraphRAG**: Graph + Retrieval Augmented Generation
- **Prompt Caching**: Anthropic feature to cache prompt prefixes (90% cost savings)
- **Provider Chain**: Priority-ordered list of LLM providers for failover
- **UnifiedLM**: Single interface for all LLM calls in Fractal ecosystem
- **FractalDSpyLM**: DSPy-compatible wrapper around UnifiedLM
- **Operational Agent**: VSM System 1 - Task execution
- **Control Agent**: VSM System 3 - Task decomposition and delegation
- **Intelligence Agent**: VSM System 4 - Reflection and learning
- **Coordination Agent**: VSM System 2 - Conflict resolution
- **Policy Agent**: VSM System 5 - Ethical boundaries

### Appendix E: Key Design Decisions

#### Decision 1: Why Anthropic + Gemini (not OpenAI)?

**Rationale:**

- User has max Anthropic subscription (cost efficiency)
- Anthropic prompt caching = 90% savings
- Gemini as backup (different infrastructure for resilience)
- Official SDKs only (no LiteLLM wrapper)

#### Decision 2: Why UnifiedLM (not direct API calls)?

**Rationale:**

- Change models globally via configuration
- Automatic failover between providers
- Built-in metrics tracking
- Future extensibility (add providers without changing agents)

#### Decision 3: Why DSPy (not LangChain)?

**Rationale:**

- Declarative signatures (specify what, not how)
- Automatic optimization via compilers
- Better suited for self-improving agents
- LangGraph still used for workflows (complementary)

#### Decision 4: Why GraphRAG (not vector DB only)?

**Rationale:**

- Relationships matter (graph traversal)
- Temporal validity tracking (t_valid, t_invalid)
- Hybrid search (graph + vector)
- Knowledge evolution over time

#### Decision 5: Why Obsidian for meta-knowledge?

**Rationale:**

- Human-readable markdown
- Git versioning
- Extensible via plugins
- Zettelkasten methodology (atomic notes)

---

## Conclusion

The Fractal Agent Ecosystem blueprint provides a complete path from concept to production:

✅ **Unified LLM Infrastructure** - Single interface, automatic failover, 90% cost savings
✅ **VSM Architecture** - 5-system hierarchy for organizational intelligence
✅ **DSPy Framework** - Declarative, self-optimizing agents
✅ **Four-Tier Memory** - From working memory to meta-knowledge
✅ **Production Ready** - Testing, security, resilience, monitoring
✅ **Reference Implementation** - Complete Python codebase

**Next Steps:**

1. Run Phase 0 implementation (Weeks 1-2)
2. Build first operational agent
3. Validate Anthropic → Gemini failover
4. Achieve >80% cache hit rate
5. Begin Phase 1 (multi-agent coordination)

**Questions?** See documentation in `/docs/fractal-agent-reference/` or review brainstorming session results for detailed solution analysis.

---

**Document Status:** ✅ Complete with LLM Architecture Integration
**Last Updated:** 2025-10-18
**Version:** 2.0 (includes unified LLM infrastructure)
