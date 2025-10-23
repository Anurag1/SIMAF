# FractalAI

**Self-improving agent framework based on the Viable System Model (VSM)**

FractalAI is an advanced multi-agent AI system that uses the Viable System Model's cybernetic principles to create autonomous, self-improving agents with built-in observability, ethical governance, and knowledge management.

## 🌟 Features

### Core Capabilities
- **Multi-Agent Coordination**: System 1-5 agents with hierarchical task decomposition
- **Self-Improving Intelligence**: MIPRO optimization and A/B testing for continuous improvement
- **GraphRAG Knowledge System**: Neo4j + Qdrant vector search for intelligent context retrieval
- **Production-Grade Observability**: Prometheus metrics, OpenTelemetry tracing, structured logging
- **Ethical Governance**: Policy agent (System 5) for ethical boundary enforcement

### Technical Highlights
- **Unified LLM Provider**: Automatic failover (Claude → Gemini) with tier-based model selection
- **DSPy Integration**: Declarative self-improving prompts
- **Human-in-the-Loop**: Obsidian vault integration for review workflows
- **Enterprise Security**: PII redaction, input sanitization, comprehensive test suite

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Docker & docker-compose (for infrastructure)
- Claude Code authentication (via `claude-agent-sdk`)

### Installation

```bash
# Clone the repository
git clone https://github.com/PMI-CAL/FractalAI.git
cd FractalAI

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start infrastructure services
docker-compose up -d

# Verify installation
python3 test_runtime_integration.py
```

### First Agent Execution

```python
from fractal_agent.agents.research_agent import ResearchAgent
from fractal_agent.utils.model_config import configure_lm

# Initialize LLM
lm = configure_lm(tier="balanced")

# Create research agent
agent = ResearchAgent()

# Execute research task
result = agent.forward(
    topic="Viable System Model",
    depth="comprehensive"
)

print(f"Confidence: {result.confidence}")
print(f"Report: {result.report}")
```

## 📁 Project Structure

```
fractalAI/
├── fractal_agent/          # Core agent framework
│   ├── agents/             # System 1-5 agents
│   ├── memory/             # Short-term, long-term, GraphRAG
│   ├── observability/      # Metrics, tracing, events
│   ├── utils/              # LLM provider, DSPy integration
│   ├── validation/         # Learning tracker, context validation
│   └── workflows/          # Multi-agent coordination
├── tests/                  # Test suites
│   ├── unit/               # Unit tests
│   └── integration/        # Integration tests
├── config/                 # Model configs and pricing
├── observability/          # Prometheus, Grafana, OpenTelemetry
├── docs/                   # Architecture documentation
└── test_*.py               # Runtime and phase verification tests
```

## 🧪 Testing

### Runtime Integration Tests
Tests actual code execution (not just imports):

```bash
# All runtime tests
python3 test_runtime_integration.py

# Results: 5/5 tests passing
# - Observability (correlation_id, metrics, tracing)
# - Database writes (PostgreSQL event store)
# - LLM calls (Claude Haiku 4.5)
# - Context preparation (research_missing_context)
# - Embeddings (1536-dim consistency)
```

### Phase Comprehensive Tests
Tests by development phase:

```bash
python3 test_phase0_comprehensive.py  # Foundation (6 tests)
python3 test_phase1_comprehensive.py  # Multi-Agent (6 tests)
python3 test_phase2_comprehensive.py  # Production (6 tests + 278 unit tests)
python3 test_phase3_comprehensive.py  # Intelligence (6 tests)
python3 test_phase4_comprehensive.py  # Coordination (6 tests)
python3 test_phase5_comprehensive.py  # Policy & Knowledge (6 tests)
python3 test_phase6_comprehensive.py  # Context Prep (6 tests)
```

## 🏗️ Architecture

### Viable System Model (VSM) Mapping

| VSM System | FractalAI Component | Function |
|------------|---------------------|----------|
| System 1 | Research Agent, Developer Agent | Operational units executing tasks |
| System 2 | Coordination Agent | Resource coordination & conflict resolution |
| System 3 | Intelligence Agent | Internal optimization & efficiency |
| System 4 | Context Preparation Agent | Environmental scanning & adaptation |
| System 5 | Policy Agent | Strategic governance & ethical boundaries |

### LLM Provider Hierarchy

```
configure_lm(tier="cheap")           # Tier-based helper
    ↓
UnifiedLM(providers=[...])           # Low-level provider chain
    ↓
AnthropicProvider / GeminiProvider   # Individual providers
    ↓
claude-agent-sdk / genai             # SDKs
```

**Tiers:**
- `cheap`: Fast, high-volume tasks (Haiku models)
- `balanced`: Most production workloads (Sonnet 3.5)
- `expensive`: Complex reasoning (Sonnet 4.5)
- `premium`: Maximum capability (Opus models)

### Memory System

```
Short-Term Memory (SQLite)
    ↓
Knowledge Extraction Agent
    ↓
Long-Term Memory (Neo4j + Qdrant)
    ↓
GraphRAG Retrieval
```

## 📊 Observability

### Prometheus Metrics
- LLM call latency & token usage
- Agent execution times
- Memory system performance
- Cost tracking per tier

### OpenTelemetry Tracing
- Distributed request tracing
- Correlation IDs across agents
- Span hierarchy for debugging

### Grafana Dashboards
- VSM System Overview
- Agent Performance
- Cost Tracking
- System Health

**Access:** http://localhost:3000 (after `docker-compose up`)

## 🔧 Configuration

### Model Tiers
Edit `config/models_pricing.yaml` to configure:
- Model selection per tier
- Pricing per token
- Provider priorities
- Capability flags (vision, caching)

### Observability
Edit `docker-compose.yml` to configure:
- Prometheus scrape intervals
- Grafana data sources
- OpenTelemetry endpoints
- PostgreSQL event store

## 🧠 Key Concepts

### DSPy Integration
Declarative self-improving prompts:

```python
from fractal_agent.utils.dspy_integration import configure_dspy

# Configure DSPy with FractalAI
lm = configure_dspy(tier="balanced")

# Define signature
class TaskDecomposition(dspy.Signature):
    """Decompose complex task into subtasks"""
    task = dspy.InputField(desc="The complex task to decompose")
    subtasks = dspy.OutputField(desc="List of subtasks")

# Use with auto-optimization
decomposer = dspy.Predict(TaskDecomposition)
result = decomposer(task="Build distributed system")
```

### Knowledge Extraction
Automatic GraphRAG integration:

```python
from fractal_agent.agents.knowledge_extraction_agent import KnowledgeExtractionAgent

agent = KnowledgeExtractionAgent()
knowledge = agent.extract(
    text=task_output,
    confidence_threshold=0.7
)

# Automatically stored in Neo4j + Qdrant
# Retrieved via semantic search
```

### Policy Enforcement
Ethical governance:

```python
from fractal_agent.agents.policy_agent import PolicyAgent
from fractal_agent.agents.policy_config import PolicyMode

policy = PolicyAgent(mode=PolicyMode.STRICT)
evaluation = policy.evaluate(
    action="access_user_data",
    context={"purpose": "analytics"}
)

if not evaluation.approved:
    raise PolicyViolation(evaluation.reason)
```

## 📚 Documentation

- **[Runtime Verification](RUNTIME_VERIFICATION_COMPLETE.md)**: Complete test report
- **[Architecture Overview](docs/fractal-agent-complete-design-specification.md)**: System design
- **[LLM Integration](docs/fractal-agent-llm-architecture.md)**: Provider architecture
- **[Phase Reports](PHASE*.md)**: Development phase summaries

## 🤝 Contributing

FractalAI was developed using the BMAD development framework. Contributions welcome!

### Development Setup
```bash
# Install dev dependencies
pip install -r requirements.txt

# Run full test suite
pytest tests/

# Run runtime verification
python3 test_runtime_integration.py

# Check code coverage
pytest --cov=fractal_agent tests/
```

## 📄 License

[Add your license here]

## 🙏 Acknowledgments

- **Stafford Beer**: Creator of the Viable System Model
- **BMAD Framework**: Development tool used to build FractalAI
- **Anthropic**: Claude LLM API
- **DSPy**: Self-improving prompting framework

## 📧 Contact

[Add your contact information]

---

**Status:** ✅ Production-Ready
**Test Pass Rate:** 100% (5/5 runtime tests + all phase tests)
**Last Verified:** 2025-10-23
