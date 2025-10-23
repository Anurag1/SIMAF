# Fractal VSM Observability - Quick Start Guide

## Access URLs

- **Grafana:** http://localhost:3002 (admin / admin)
- **Prometheus:** http://localhost:9090
- **Jaeger:** http://localhost:16686

## Working Dashboard

**OTEL Collector Monitoring** (Shows real data right now!)
http://localhost:3002/d/otel-collector-monitoring

This dashboard displays:

- OTEL Collector uptime and health
- Memory and CPU usage
- Metrics processing rates
- Service status for all components

## To Integrate Into Your Application

### 1. Initialize Observability at Startup

```python
from fractal_agent.observability import init_tracing, configure_logging
from fractal_agent.observability.metrics_server import start_metrics_server
import threading

# Initialize tracing
init_tracing(
    service_name="fractal-agent",
    otlp_endpoint="http://localhost:4317"
)

# Configure logging
configure_logging()

# Start metrics server in background thread
metrics_thread = threading.Thread(
    target=start_metrics_server,
    args=(8000, '0.0.0.0'),
    daemon=True
)
metrics_thread.start()
```

### 2. Add LLM Instrumentation

```python
from fractal_agent.observability import get_tracer, record_llm_call

tracer = get_tracer(__name__)

def call_llm(prompt, tier="System1_Research"):
    with tracer.start_as_current_span("llm_call") as span:
        start = time.time()

        try:
            response = llm.generate(prompt)
            latency_ms = (time.time() - start) * 1000

            # Record metrics
            record_llm_call(
                tier=tier,
                provider="anthropic",
                model="claude-sonnet-4.5",
                tokens=response.usage.total_tokens,
                latency_ms=latency_ms,
                cost=response.usage.total_tokens * 0.000003,
                cache_hit=response.cache_hit,
                success=True
            )

            return response

        except Exception as e:
            record_llm_call(
                tier=tier,
                provider="anthropic",
                model="claude-sonnet-4.5",
                tokens=0,
                latency_ms=(time.time() - start) * 1000,
                cost=0,
                cache_hit=False,
                success=False,
                error_type=type(e).__name__
            )
            raise
```

### 3. Add Task Tracking

```python
from fractal_agent.observability import record_task_completion

def execute_task(task_type, tier):
    start = time.time()
    try:
        # ... task execution ...
        duration = time.time() - start
        record_task_completion(tier, task_type, duration, success=True)
    except Exception as e:
        duration = time.time() - start
        record_task_completion(tier, task_type, duration, success=False)
        raise
```

## Verify It's Working

```bash
# Check metrics are being generated
curl http://localhost:8000/metrics | grep fractal_llm_calls_total

# Check Prometheus is scraping them
curl 'http://localhost:9090/api/v1/query?query=fractal_llm_calls_total'

# Check traces in Jaeger
open http://localhost:16686/search?service=fractal-agent
```

## Dashboards Will Auto-Populate

Once your application generates metrics, these dashboards will show data:

1. **Fractal VSM - System Overview**
   - http://localhost:3002/d/fractal-vsm-overview
   - LLM costs, tokens, latency, cache hits

2. **Fractal VSM - Agent Performance**
   - http://localhost:3002/d/fractal-vsm-agents
   - Agent operations, research stages, event store

## For More Details

See `/Users/cal/DEV/bmad/BMAD-METHOD/observability/OBSERVABILITY_STATUS.md` for the full report.
