"""
Prometheus Metrics HTTP Server for Fractal VSM

Exposes metrics on http://localhost:8000/metrics for scraping by Prometheus.

Usage:
    python -m fractal_agent.observability.metrics_server

Author: BMad
Date: 2025-01-20
"""

import logging
import time
from prometheus_client import start_http_server, REGISTRY
from .metrics import registry

logger = logging.getLogger(__name__)


def start_metrics_server(port: int = 9100, addr: str = '127.0.0.1'):
    """
    Start the Prometheus metrics HTTP server in a background thread.

    The prometheus_client.start_http_server() automatically starts a background
    daemon thread, so this function returns immediately and the metrics server
    continues running in the background.

    Args:
        port: Port to listen on (default: 9100)
        addr: Address to bind to (default: 127.0.0.1)
    """
    # Use our custom registry instead of the default one
    # This starts a background daemon thread and returns immediately
    start_http_server(port, addr=addr, registry=registry)
    logger.info(f"Metrics server started on {addr}:{port} (background thread)")
    # No blocking loop needed - server runs in background daemon thread


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    start_metrics_server()
