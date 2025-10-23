"""
Real-Time Observability Dashboard Backend

FastAPI + WebSocket server that streams live observability data:
- LLM call metrics (tokens, cost, latency)
- VSM tier activity
- Agent operations
- Event stream

Author: BMad
Date: 2025-01-20
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from collections import defaultdict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import psycopg2
from psycopg2.extras import RealDictCursor

# Observability imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from fractal_agent.observability import get_event_store, get_correlation_id
from fractal_agent.observability.metrics import registry

# FastAPI app
app = FastAPI(
    title="Fractal VSM Observability Dashboard",
    description="Real-time observability dashboard for Fractal VSM multi-agent system",
    version="1.0.0"
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)

        # Remove disconnected clients
        for conn in disconnected:
            self.active_connections.remove(conn)

manager = ConnectionManager()

# ============================================================================
# Pydantic Models
# ============================================================================

class MetricSnapshot(BaseModel):
    """Snapshot of current metrics"""
    timestamp: datetime
    llm_calls_total: int
    llm_tokens_total: int
    llm_cost_total: float
    tier_tasks_total: Dict[str, int]
    cache_hit_rate: float
    avg_latency_ms: float

class RecentEvent(BaseModel):
    """Recent event from event store"""
    id: int
    timestamp: datetime
    correlation_id: str
    tier: str
    event_type: str
    data: dict

class AgentActivity(BaseModel):
    """Agent activity summary"""
    agent_type: str
    operations_count: int
    avg_duration_seconds: float
    success_rate: float

# ============================================================================
# Helper Functions
# ============================================================================

def get_db_connection():
    """Get PostgreSQL connection"""
    try:
        conn = psycopg2.connect(
            host="postgres-observability",
            port=5432,
            database="fractal_observability",
            user="fractal",
            password="fractal_dev_password"
        )
        return conn
    except Exception as e:
        print(f"Failed to connect to PostgreSQL: {e}")
        return None

def collect_prometheus_metrics() -> Dict[str, Any]:
    """Collect current metrics from Prometheus registry"""
    metrics = {}

    try:
        # Collect all metrics from registry
        for collector in registry._collector_to_names.keys():
            for metric in collector.collect():
                # Extract metric values
                for sample in metric.samples:
                    metric_name = sample.name
                    labels = sample.labels
                    value = sample.value

                    # Store in structured format
                    if metric_name not in metrics:
                        metrics[metric_name] = []

                    metrics[metric_name].append({
                        "labels": labels,
                        "value": value
                    })

    except Exception as e:
        print(f"Error collecting Prometheus metrics: {e}")

    return metrics

def aggregate_llm_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Aggregate LLM metrics for dashboard"""
    aggregated = {
        "total_calls": 0,
        "total_tokens": 0,
        "total_cost": 0.0,
        "cache_hits": 0,
        "total_cache_calls": 0,
        "by_tier": defaultdict(lambda: {"calls": 0, "tokens": 0, "cost": 0.0}),
        "by_provider": defaultdict(lambda: {"calls": 0, "tokens": 0}),
    }

    # LLM calls
    if "fractal_llm_calls_total" in metrics:
        for sample in metrics["fractal_llm_calls_total"]:
            tier = sample["labels"].get("tier", "unknown")
            provider = sample["labels"].get("provider", "unknown")
            cache_hit = sample["labels"].get("cache_hit", "false")
            status = sample["labels"].get("status", "unknown")
            value = sample["value"]

            if status == "success":
                aggregated["total_calls"] += value
                aggregated["by_tier"][tier]["calls"] += value
                aggregated["by_provider"][provider]["calls"] += value

                if cache_hit == "true":
                    aggregated["cache_hits"] += value

                aggregated["total_cache_calls"] += value

    # LLM tokens
    if "fractal_llm_tokens_total" in metrics:
        for sample in metrics["fractal_llm_tokens_total"]:
            tier = sample["labels"].get("tier", "unknown")
            provider = sample["labels"].get("provider", "unknown")
            value = sample["value"]

            aggregated["total_tokens"] += value
            aggregated["by_tier"][tier]["tokens"] += value
            aggregated["by_provider"][provider]["tokens"] += value

    # LLM cost
    if "fractal_llm_cost_usd_total" in metrics:
        for sample in metrics["fractal_llm_cost_usd_total"]:
            tier = sample["labels"].get("tier", "unknown")
            value = sample["value"]

            aggregated["total_cost"] += value
            aggregated["by_tier"][tier]["cost"] += value

    # Calculate cache hit rate
    if aggregated["total_cache_calls"] > 0:
        aggregated["cache_hit_rate"] = (
            aggregated["cache_hits"] / aggregated["total_cache_calls"]
        )
    else:
        aggregated["cache_hit_rate"] = 0.0

    # Convert defaultdicts to regular dicts
    aggregated["by_tier"] = dict(aggregated["by_tier"])
    aggregated["by_provider"] = dict(aggregated["by_provider"])

    return aggregated

async def stream_metrics():
    """Background task to stream metrics to connected clients"""
    while True:
        try:
            # Collect metrics
            raw_metrics = collect_prometheus_metrics()
            llm_metrics = aggregate_llm_metrics(raw_metrics)

            # Get recent events
            conn = get_db_connection()
            recent_events = []
            if conn:
                try:
                    cursor = conn.cursor(cursor_factory=RealDictCursor)
                    cursor.execute("""
                        SELECT id, timestamp, correlation_id, tier, event_type, data
                        FROM events
                        ORDER BY timestamp DESC
                        LIMIT 10
                    """)
                    recent_events = [dict(row) for row in cursor.fetchall()]
                    cursor.close()
                    conn.close()
                except Exception as e:
                    print(f"Error fetching events: {e}")

            # Broadcast to all connected clients
            await manager.broadcast({
                "type": "metrics_update",
                "timestamp": datetime.now().isoformat(),
                "llm_metrics": llm_metrics,
                "recent_events": recent_events
            })

        except Exception as e:
            print(f"Error in stream_metrics: {e}")

        # Wait 2 seconds before next update
        await asyncio.sleep(2)

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Fractal VSM Observability Dashboard",
        "version": "1.0.0"
    }

@app.get("/api/metrics/summary")
async def get_metrics_summary():
    """Get current metrics summary"""
    raw_metrics = collect_prometheus_metrics()
    llm_metrics = aggregate_llm_metrics(raw_metrics)

    return {
        "timestamp": datetime.now().isoformat(),
        "llm_metrics": llm_metrics
    }

@app.get("/api/events/recent")
async def get_recent_events(limit: int = 50):
    """Get recent events from event store"""
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=503, detail="Event store unavailable")

    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            SELECT id, timestamp, correlation_id, tier, event_type, data
            FROM events
            ORDER BY timestamp DESC
            LIMIT %s
        """, (limit,))

        events = [dict(row) for row in cursor.fetchall()]
        cursor.close()
        conn.close()

        return {"events": events, "count": len(events)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/events/by_correlation/{correlation_id}")
async def get_events_by_correlation(correlation_id: str):
    """Get all events for a specific correlation ID"""
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=503, detail="Event store unavailable")

    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            SELECT id, timestamp, correlation_id, tier, event_type, data
            FROM events
            WHERE correlation_id = %s
            ORDER BY timestamp ASC
        """, (correlation_id,))

        events = [dict(row) for row in cursor.fetchall()]
        cursor.close()
        conn.close()

        return {"correlation_id": correlation_id, "events": events, "count": len(events)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/metrics/llm/cost_by_tier")
async def get_cost_by_tier(hours: int = 24):
    """Get LLM cost breakdown by tier"""
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=503, detail="Event store unavailable")

    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            SELECT
                tier,
                COUNT(*) as call_count,
                SUM((data->>'tokens_used')::int) as total_tokens,
                SUM((data->>'estimated_cost')::float) as total_cost
            FROM events
            WHERE event_type = 'llm_call_completed'
              AND timestamp > NOW() - INTERVAL '%s hours'
            GROUP BY tier
            ORDER BY total_cost DESC
        """, (hours,))

        results = [dict(row) for row in cursor.fetchall()]
        cursor.close()
        conn.close()

        return {"period_hours": hours, "by_tier": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/metrics/llm/timeline")
async def get_llm_timeline(hours: int = 1, bucket_minutes: int = 5):
    """Get LLM call timeline (calls per time bucket)"""
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=503, detail="Event store unavailable")

    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            SELECT
                date_trunc('minute', timestamp) AS minute,
                COUNT(*) as call_count,
                SUM((data->>'tokens_used')::int) as tokens,
                SUM((data->>'estimated_cost')::float) as cost
            FROM events
            WHERE event_type = 'llm_call_completed'
              AND timestamp > NOW() - INTERVAL '%s hours'
            GROUP BY minute
            ORDER BY minute ASC
        """, (hours,))

        results = [dict(row) for row in cursor.fetchall()]
        cursor.close()
        conn.close()

        return {"period_hours": hours, "bucket_minutes": bucket_minutes, "timeline": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# WebSocket Endpoint
# ============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time metrics streaming"""
    await manager.connect(websocket)

    try:
        # Send initial data
        raw_metrics = collect_prometheus_metrics()
        llm_metrics = aggregate_llm_metrics(raw_metrics)

        await websocket.send_json({
            "type": "connected",
            "message": "Connected to Fractal VSM Observability Dashboard",
            "llm_metrics": llm_metrics
        })

        # Keep connection alive and listen for client messages
        while True:
            try:
                data = await websocket.receive_text()
                # Echo back (or handle commands)
                await websocket.send_json({
                    "type": "echo",
                    "message": f"Received: {data}"
                })
            except WebSocketDisconnect:
                break

    except Exception as e:
        print(f"WebSocket error: {e}")

    finally:
        manager.disconnect(websocket)

# ============================================================================
# Startup Event
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Start background tasks on startup"""
    asyncio.create_task(stream_metrics())
    print("Real-time observability dashboard backend started")
    print("WebSocket endpoint: ws://localhost:8000/ws")
    print("API docs: http://localhost:8000/docs")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
