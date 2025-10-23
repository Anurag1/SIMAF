"""
Enhanced Prometheus Metrics Server for Production Monitoring

HTTP server that exposes Prometheus metrics with production monitoring integration.
Serves metrics at /metrics endpoint and provides additional endpoints for health
checking and cost monitoring.

Features:
- Standard Prometheus /metrics endpoint
- /health endpoint for container health checks
- /cost endpoint for real-time cost status (JSON)
- /budget endpoint for budget status (JSON)
- Auto-starts as background daemon
- Graceful shutdown handling

Usage:
    from fractal_agent.observability.metrics_server_enhanced import start_metrics_server

    # Start server (runs in background)
    server = start_metrics_server(port=9100)

    # Server runs until process exits
    # Access at: http://localhost:9100/metrics

Author: BMad
Date: 2025-01-20
"""

import time
import json
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread
from typing import Optional
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from .production_monitoring import get_production_monitor

logger = logging.getLogger(__name__)


class MetricsHandler(BaseHTTPRequestHandler):
    """HTTP request handler for metrics and monitoring endpoints"""

    def log_message(self, format, *args):
        """Override to use Python logging instead of stderr"""
        logger.debug(f"{self.address_string()} - {format % args}")

    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/metrics':
            self._serve_metrics()
        elif self.path == '/health':
            self._serve_health()
        elif self.path == '/cost':
            self._serve_cost_status()
        elif self.path == '/budget':
            self._serve_budget_status()
        elif self.path == '/':
            self._serve_index()
        else:
            self._serve_404()

    def _serve_metrics(self):
        """Serve Prometheus metrics"""
        try:
            monitor = get_production_monitor()
            metrics_output = monitor.export_metrics()

            self.send_response(200)
            self.send_header('Content-Type', CONTENT_TYPE_LATEST)
            self.end_headers()
            self.wfile.write(metrics_output.encode('utf-8'))

        except Exception as e:
            logger.error(f"Error serving metrics: {e}")
            self.send_response(500)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(f"Error: {str(e)}".encode('utf-8'))

    def _serve_health(self):
        """Serve health check endpoint"""
        try:
            monitor = get_production_monitor()

            health = {
                'status': 'healthy',
                'timestamp': time.time(),
                'uptime_seconds': time.time() - getattr(monitor, '_start_time', time.time()),
                'metrics_available': True
            }

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(health, indent=2).encode('utf-8'))

        except Exception as e:
            logger.error(f"Error serving health: {e}")
            health = {
                'status': 'unhealthy',
                'error': str(e)
            }
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(health, indent=2).encode('utf-8'))

    def _serve_cost_status(self):
        """Serve real-time cost status"""
        try:
            monitor = get_production_monitor()
            cost_breakdown = monitor.get_cost_breakdown()

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(cost_breakdown, indent=2).encode('utf-8'))

        except Exception as e:
            logger.error(f"Error serving cost status: {e}")
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}, indent=2).encode('utf-8'))

    def _serve_budget_status(self):
        """Serve budget status"""
        try:
            monitor = get_production_monitor()
            budget_status = monitor.get_budget_status()

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(budget_status, indent=2).encode('utf-8'))

        except Exception as e:
            logger.error(f"Error serving budget status: {e}")
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}, indent=2).encode('utf-8'))

    def _serve_index(self):
        """Serve index page with links to endpoints"""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Fractal Agent Metrics</title>
            <style>
                body { font-family: monospace; margin: 40px; }
                h1 { color: #333; }
                ul { list-style: none; padding: 0; }
                li { margin: 10px 0; }
                a { color: #0066cc; text-decoration: none; }
                a:hover { text-decoration: underline; }
                .endpoint { background: #f4f4f4; padding: 5px; border-radius: 3px; }
            </style>
        </head>
        <body>
            <h1>Fractal Agent Metrics Server</h1>
            <p>Production monitoring endpoints:</p>
            <ul>
                <li><span class="endpoint">GET <a href="/metrics">/metrics</a></span> - Prometheus metrics (text format)</li>
                <li><span class="endpoint">GET <a href="/health">/health</a></span> - Health check (JSON)</li>
                <li><span class="endpoint">GET <a href="/cost">/cost</a></span> - Real-time cost status (JSON)</li>
                <li><span class="endpoint">GET <a href="/budget">/budget</a></span> - Budget status (JSON)</li>
            </ul>
            <p><small>Metrics server for Fractal VSM Observability Stack</small></p>
        </body>
        </html>
        """
        self.send_response(200)
        self.send_header('Content-Type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))

    def _serve_404(self):
        """Serve 404 not found"""
        self.send_response(404)
        self.send_header('Content-Type', 'text/plain')
        self.end_headers()
        self.wfile.write(b'404 Not Found')


class MetricsServer:
    """
    Prometheus metrics HTTP server with production monitoring integration.

    Runs as a background daemon thread serving metrics at /metrics endpoint.
    """

    def __init__(self, port: int = 9100, host: str = '0.0.0.0'):
        """
        Initialize metrics server.

        Args:
            port: Port to listen on (default 9100)
            host: Host to bind to (default 0.0.0.0 for all interfaces)
        """
        self.port = port
        self.host = host
        self.httpd: Optional[HTTPServer] = None
        self.thread: Optional[Thread] = None
        self._running = False

    def start(self):
        """Start the metrics server in a background thread"""
        if self._running:
            logger.warning("Metrics server already running")
            return

        try:
            self.httpd = HTTPServer((self.host, self.port), MetricsHandler)
            self._running = True

            # Start server thread
            self.thread = Thread(target=self._run_server, daemon=True)
            self.thread.start()

            logger.info(f"Metrics server started on http://{self.host}:{self.port}")
            logger.info(f"  - Prometheus metrics: http://{self.host}:{self.port}/metrics")
            logger.info(f"  - Health check: http://{self.host}:{self.port}/health")
            logger.info(f"  - Cost status: http://{self.host}:{self.port}/cost")
            logger.info(f"  - Budget status: http://{self.host}:{self.port}/budget")

        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
            self._running = False
            raise

    def _run_server(self):
        """Run the HTTP server (runs in background thread)"""
        try:
            logger.info("Metrics server thread started")
            self.httpd.serve_forever()
        except Exception as e:
            logger.error(f"Metrics server error: {e}")
        finally:
            self._running = False
            logger.info("Metrics server thread stopped")

    def stop(self):
        """Stop the metrics server"""
        if not self._running:
            logger.warning("Metrics server not running")
            return

        logger.info("Stopping metrics server...")
        self._running = False

        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()

        if self.thread:
            self.thread.join(timeout=5)

        logger.info("Metrics server stopped")

    def is_running(self) -> bool:
        """Check if server is running"""
        return self._running


# Global server instance
_global_server: Optional[MetricsServer] = None


def start_metrics_server(port: int = 9100, host: str = '0.0.0.0') -> MetricsServer:
    """
    Start the global metrics server.

    Args:
        port: Port to listen on (default 9100)
        host: Host to bind to (default 0.0.0.0)

    Returns:
        MetricsServer instance
    """
    global _global_server

    if _global_server is not None and _global_server.is_running():
        logger.info("Metrics server already running")
        return _global_server

    _global_server = MetricsServer(port=port, host=host)
    _global_server.start()

    return _global_server


def stop_metrics_server():
    """Stop the global metrics server"""
    global _global_server

    if _global_server is not None:
        _global_server.stop()
        _global_server = None


def get_metrics_server() -> Optional[MetricsServer]:
    """Get the global metrics server instance"""
    return _global_server
