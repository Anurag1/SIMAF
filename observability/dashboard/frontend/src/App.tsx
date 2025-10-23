/**
 * Real-Time Observability Dashboard - Main App Component
 *
 * Connects to FastAPI WebSocket backend and displays:
 * - Real-time LLM metrics (calls, tokens, cost, latency)
 * - VSM tier activity
 * - Recent events stream
 * - Cost trends
 *
 * Author: BMad
 * Date: 2025-01-20
 */

import React, { useState, useEffect, useRef } from 'react';
import {
  LineChart, Line, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';
import './App.css';

// ============================================================================
// TypeScript Interfaces
// ============================================================================

interface LLMMetrics {
  total_calls: number;
  total_tokens: number;
  total_cost: number;
  cache_hit_rate: number;
  by_tier: Record<string, { calls: number; tokens: number; cost: number }>;
  by_provider: Record<string, { calls: number; tokens: number }>;
}

interface Event {
  id: number;
  timestamp: string;
  correlation_id: string;
  tier: string;
  event_type: string;
  data: Record<string, any>;
}

interface MetricsUpdate {
  type: string;
  timestamp: string;
  llm_metrics: LLMMetrics;
  recent_events: Event[];
}

// ============================================================================
// Main App Component
// ============================================================================

function App() {
  const [connected, setConnected] = useState(false);
  const [metrics, setMetrics] = useState<LLMMetrics | null>(null);
  const [events, setEvents] = useState<Event[]>([]);
  const [metricsHistory, setMetricsHistory] = useState<any[]>([]);
  const ws = useRef<WebSocket | null>(null);

  // Colors for charts
  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];

  // ============================================================================
  // WebSocket Connection
  // ============================================================================

  useEffect(() => {
    // Connect to WebSocket
    const connectWebSocket = () => {
      const websocket = new WebSocket('ws://localhost:8000/ws');

      websocket.onopen = () => {
        console.log('WebSocket connected');
        setConnected(true);
      };

      websocket.onmessage = (event) => {
        const data: MetricsUpdate = JSON.parse(event.data);
        console.log('Received:', data);

        if (data.type === 'metrics_update' || data.type === 'connected') {
          setMetrics(data.llm_metrics);

          // Add to metrics history for timeline
          setMetricsHistory((prev) => {
            const newHistory = [
              ...prev,
              {
                timestamp: new Date(data.timestamp).toLocaleTimeString(),
                calls: data.llm_metrics.total_calls,
                tokens: data.llm_metrics.total_tokens,
                cost: data.llm_metrics.total_cost,
              }
            ];
            // Keep only last 30 data points
            return newHistory.slice(-30);
          });

          if (data.recent_events) {
            setEvents(data.recent_events);
          }
        }
      };

      websocket.onclose = () => {
        console.log('WebSocket disconnected');
        setConnected(false);
        // Attempt to reconnect after 5 seconds
        setTimeout(connectWebSocket, 5000);
      };

      websocket.onerror = (error) => {
        console.error('WebSocket error:', error);
      };

      ws.current = websocket;
    };

    connectWebSocket();

    // Cleanup on unmount
    return () => {
      if (ws.current) {
        ws.current.close();
      }
    };
  }, []);

  // ============================================================================
  // Render Functions
  // ============================================================================

  const renderConnectionStatus = () => (
    <div className={`connection-status ${connected ? 'connected' : 'disconnected'}`}>
      <span className="status-indicator"></span>
      {connected ? 'Connected' : 'Disconnected'}
    </div>
  );

  const renderMetricCard = (title: string, value: string | number, subtitle?: string) => (
    <div className="metric-card">
      <h3>{title}</h3>
      <div className="metric-value">{value}</div>
      {subtitle && <div className="metric-subtitle">{subtitle}</div>}
    </div>
  );

  const renderTierBreakdown = () => {
    if (!metrics || !metrics.by_tier) return null;

    const tierData = Object.entries(metrics.by_tier).map(([tier, data]) => ({
      name: tier,
      calls: data.calls,
      tokens: data.tokens,
      cost: data.cost,
    }));

    return (
      <div className="chart-container">
        <h2>LLM Calls by Tier</h2>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={tierData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" angle={-45} textAnchor="end" height={100} />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="calls" fill="#8884d8" name="Calls" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    );
  };

  const renderProviderBreakdown = () => {
    if (!metrics || !metrics.by_provider) return null;

    const providerData = Object.entries(metrics.by_provider).map(([provider, data]) => ({
      name: provider,
      calls: data.calls,
    }));

    return (
      <div className="chart-container">
        <h2>Calls by Provider</h2>
        <ResponsiveContainer width="100%" height={300}>
          <PieChart>
            <Pie
              data={providerData}
              dataKey="calls"
              nameKey="name"
              cx="50%"
              cy="50%"
              outerRadius={100}
              label
            >
              {providerData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
            <Legend />
          </PieChart>
        </ResponsiveContainer>
      </div>
    );
  };

  const renderMetricsTimeline = () => {
    if (metricsHistory.length === 0) return null;

    return (
      <div className="chart-container">
        <h2>Metrics Over Time</h2>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={metricsHistory}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="timestamp" />
            <YAxis yAxisId="left" />
            <YAxis yAxisId="right" orientation="right" />
            <Tooltip />
            <Legend />
            <Line
              yAxisId="left"
              type="monotone"
              dataKey="calls"
              stroke="#8884d8"
              name="Total Calls"
            />
            <Line
              yAxisId="right"
              type="monotone"
              dataKey="cost"
              stroke="#82ca9d"
              name="Total Cost ($)"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    );
  };

  const renderEventsStream = () => {
    if (events.length === 0) return null;

    return (
      <div className="events-container">
        <h2>Recent Events</h2>
        <div className="events-list">
          {events.map((event) => (
            <div key={event.id} className="event-item">
              <div className="event-header">
                <span className="event-tier">{event.tier}</span>
                <span className="event-type">{event.event_type}</span>
                <span className="event-time">
                  {new Date(event.timestamp).toLocaleTimeString()}
                </span>
              </div>
              <div className="event-correlation">
                Correlation ID: {event.correlation_id.substring(0, 8)}...
              </div>
              {event.event_type === 'llm_call_completed' && (
                <div className="event-data">
                  {event.data.provider}/{event.data.model} - {event.data.tokens_used} tokens
                  - ${event.data.estimated_cost?.toFixed(4)} - {event.data.latency_ms}ms
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    );
  };

  // ============================================================================
  // Main Render
  // ============================================================================

  return (
    <div className="App">
      <header className="App-header">
        <h1>Fractal VSM Observability Dashboard</h1>
        {renderConnectionStatus()}
      </header>

      {!connected && (
        <div className="loading-message">
          <p>Connecting to observability backend...</p>
          <p>Make sure FastAPI backend is running on port 8000</p>
        </div>
      )}

      {connected && metrics && (
        <div className="dashboard-content">
          {/* Top Metrics Cards */}
          <div className="metrics-grid">
            {renderMetricCard(
              'Total LLM Calls',
              metrics.total_calls.toLocaleString()
            )}
            {renderMetricCard(
              'Total Tokens',
              metrics.total_tokens.toLocaleString()
            )}
            {renderMetricCard(
              'Total Cost',
              `$${metrics.total_cost.toFixed(2)}`
            )}
            {renderMetricCard(
              'Cache Hit Rate',
              `${(metrics.cache_hit_rate * 100).toFixed(1)}%`
            )}
          </div>

          {/* Charts Row 1 */}
          <div className="charts-row">
            {renderTierBreakdown()}
            {renderProviderBreakdown()}
          </div>

          {/* Charts Row 2 */}
          <div className="charts-row">
            {renderMetricsTimeline()}
          </div>

          {/* Events Stream */}
          <div className="events-row">
            {renderEventsStream()}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
