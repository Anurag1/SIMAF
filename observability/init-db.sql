-- PostgreSQL Schema for Fractal VSM Observability
-- Creates tables for event sourcing and LangGraph checkpointing
--
-- Author: BMad
-- Date: 2025-01-20

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ==============================================================================
-- EVENT SOURCING TABLES
-- ==============================================================================

-- VSM Events table - immutable audit trail
CREATE TABLE IF NOT EXISTS vsm_events (
    id BIGSERIAL PRIMARY KEY,
    event_id UUID NOT NULL DEFAULT uuid_generate_v4() UNIQUE,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Tracing context
    trace_id UUID,
    span_id TEXT,
    parent_span_id TEXT,

    -- Event classification
    tier TEXT NOT NULL,  -- System1, System2, System3, System4
    event_type TEXT NOT NULL,  -- task_started, llm_called, verification_completed, etc

    -- Event data (JSONB for queryability)
    data JSONB NOT NULL DEFAULT '{}',

    -- Metadata
    correlation_id UUID,
    user_id TEXT,
    session_id TEXT,

    -- Indexes
    CONSTRAINT vsm_events_tier_check CHECK (tier IN ('System1', 'System2', 'System3', 'System4'))
);

-- Indexes for common queries
CREATE INDEX idx_vsm_events_timestamp ON vsm_events(timestamp DESC);
CREATE INDEX idx_vsm_events_trace_id ON vsm_events(trace_id) WHERE trace_id IS NOT NULL;
CREATE INDEX idx_vsm_events_tier ON vsm_events(tier);
CREATE INDEX idx_vsm_events_event_type ON vsm_events(event_type);
CREATE INDEX idx_vsm_events_correlation_id ON vsm_events(correlation_id) WHERE correlation_id IS NOT NULL;
CREATE INDEX idx_vsm_events_session_id ON vsm_events(session_id) WHERE session_id IS NOT NULL;

-- GIN index for JSONB queries
CREATE INDEX idx_vsm_events_data_gin ON vsm_events USING GIN (data);

-- ==============================================================================
-- LANGGRAPH CHECKPOINTING TABLES
-- ==============================================================================

-- LangGraph checkpoints table
CREATE TABLE IF NOT EXISTS langgraph_checkpoints (
    checkpoint_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Checkpoint data
    checkpoint JSONB NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}',
    parent_checkpoint_id UUID REFERENCES langgraph_checkpoints(checkpoint_id),

    -- Status
    is_latest BOOLEAN NOT NULL DEFAULT TRUE,

    UNIQUE(thread_id, checkpoint_ns, checkpoint_at)
);

-- Indexes for checkpoint queries
CREATE INDEX idx_langgraph_thread_id ON langgraph_checkpoints(thread_id);
CREATE INDEX idx_langgraph_latest ON langgraph_checkpoints(thread_id, checkpoint_ns, checkpoint_at DESC)
    WHERE is_latest = TRUE;

-- ==============================================================================
-- METRICS AGGREGATION TABLES (for analytics)
-- ==============================================================================

-- LLM call metrics (aggregated from events)
CREATE TABLE IF NOT EXISTS llm_call_metrics (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- LLM details
    tier TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,

    -- Metrics
    duration_ms INTEGER NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    total_tokens INTEGER NOT NULL,
    cost_usd DECIMAL(10, 6) NOT NULL,

    -- Result
    success BOOLEAN NOT NULL,
    error_message TEXT,

    -- Tracing
    trace_id UUID,
    span_id TEXT
);

CREATE INDEX idx_llm_metrics_timestamp ON llm_call_metrics(timestamp DESC);
CREATE INDEX idx_llm_metrics_tier ON llm_call_metrics(tier);
CREATE INDEX idx_llm_metrics_provider ON llm_call_metrics(provider);
CREATE INDEX idx_llm_metrics_trace_id ON llm_call_metrics(trace_id) WHERE trace_id IS NOT NULL;

-- ==============================================================================
-- WORKFLOW EXECUTION TRACKING
-- ==============================================================================

-- Workflow executions (derived from events)
CREATE TABLE IF NOT EXISTS workflow_executions (
    execution_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    trace_id UUID UNIQUE NOT NULL,

    -- Timing
    started_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ,
    duration_seconds INTEGER,

    -- Task
    user_task TEXT NOT NULL,

    -- Results
    goal_achieved BOOLEAN,
    verification_passed BOOLEAN,

    -- Metrics
    total_llm_calls INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    total_cost_usd DECIMAL(10, 4) DEFAULT 0,

    -- Status
    status TEXT NOT NULL DEFAULT 'running',
    error_message TEXT,

    CONSTRAINT workflow_status_check CHECK (status IN ('running', 'completed', 'failed', 'cancelled'))
);

CREATE INDEX idx_workflow_started_at ON workflow_executions(started_at DESC);
CREATE INDEX idx_workflow_trace_id ON workflow_executions(trace_id);
CREATE INDEX idx_workflow_status ON workflow_executions(status);

-- ==============================================================================
-- HELPER FUNCTIONS
-- ==============================================================================

-- Function to get latest checkpoint for a thread
CREATE OR REPLACE FUNCTION get_latest_checkpoint(p_thread_id TEXT, p_checkpoint_ns TEXT DEFAULT '')
RETURNS JSONB AS $$
    SELECT checkpoint
    FROM langgraph_checkpoints
    WHERE thread_id = p_thread_id
      AND checkpoint_ns = p_checkpoint_ns
      AND is_latest = TRUE
    ORDER BY checkpoint_at DESC
    LIMIT 1;
$$ LANGUAGE SQL STABLE;

-- Function to query events by trace
CREATE OR REPLACE FUNCTION get_events_by_trace(p_trace_id UUID)
RETURNS TABLE (
    event_id UUID,
    timestamp TIMESTAMPTZ,
    tier TEXT,
    event_type TEXT,
    data JSONB
) AS $$
    SELECT event_id, timestamp, tier, event_type, data
    FROM vsm_events
    WHERE trace_id = p_trace_id
    ORDER BY timestamp ASC;
$$ LANGUAGE SQL STABLE;

-- ==============================================================================
-- INITIAL DATA / EXAMPLES
-- ==============================================================================

-- Insert a test event to verify setup
INSERT INTO vsm_events (tier, event_type, data) VALUES
    ('System4', 'system_initialized', '{"version": "1.0.0", "environment": "development"}');

-- Grant permissions (adjust as needed for production)
-- GRANT SELECT, INSERT ON ALL TABLES IN SCHEMA public TO fractal_app;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO fractal_app;

COMMENT ON TABLE vsm_events IS 'Immutable event log for Fractal VSM operations (event sourcing)';
COMMENT ON TABLE langgraph_checkpoints IS 'LangGraph workflow checkpoints for crash recovery';
COMMENT ON TABLE llm_call_metrics IS 'Aggregated metrics for LLM API calls';
COMMENT ON TABLE workflow_executions IS 'High-level workflow execution tracking';
