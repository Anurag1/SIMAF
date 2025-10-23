"""
Structured Logging for Fractal VSM

Replaces print() statements with proper structured logging that:
- Writes to both file and console
- Includes correlation IDs automatically
- Formats consistently with timestamps
- Integrates with OpenTelemetry traces

Usage:
    from fractal_agent.observability.logging import get_logger

    logger = get_logger(__name__)
    logger.info("Task started", extra={"task_id": "123", "tier": "System3"})

Author: BMad
Date: 2025-01-20
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from .context import get_correlation_id, get_trace_id

# Global logging configuration state
_configured = False


def configure_logging(
    log_level: str = "INFO",
    log_dir: Optional[str] = "logs",
    log_file: str = "fractal_agent.log",
    enable_console: bool = True
):
    """
    Configure structured logging for Fractal VSM.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (None to disable file logging)
        log_file: Log file name
        enable_console: If True, also log to console

    Note:
        Safe to call multiple times - will only configure once.
    """
    global _configured

    if _configured:
        return

    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Create formatter with correlation IDs
    formatter = CorrelationIDFormatter(
        fmt='%(asctime)s [%(correlation_id)s] [%(name)s:%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File handler
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path / log_file)
        file_handler.setLevel(logging.DEBUG)  # File gets more detail than console
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    _configured = True
    logging.info(f"Structured logging configured: {log_level} -> {log_dir}/{log_file}")


class CorrelationIDFormatter(logging.Formatter):
    """
    Custom formatter that automatically adds correlation ID and trace ID to log records.
    """

    def format(self, record):
        # Add correlation ID if available
        record.correlation_id = get_correlation_id() or "no-correlation-id"
        record.trace_id = get_trace_id() or "no-trace-id"

        return super().format(record)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with automatic configuration.

    Args:
        name: Logger name (typically __name__ of module)

    Returns:
        Configured logger instance

    Note:
        First call auto-configures logging if not already configured.
    """
    global _configured

    if not _configured:
        configure_logging()

    return logging.getLogger(name)


# Convenience function for quick logging without creating logger
def log_info(message: str, **extra):
    """Quick log info message with extra fields"""
    logger = get_logger("fractal_agent")
    logger.info(message, extra=extra)


def log_warning(message: str, **extra):
    """Quick log warning message with extra fields"""
    logger = get_logger("fractal_agent")
    logger.warning(message, extra=extra)


def log_error(message: str, exc_info=None, **extra):
    """Quick log error message with extra fields"""
    logger = get_logger("fractal_agent")
    logger.error(message, exc_info=exc_info, extra=extra)


# Auto-configure on import if not already configured
# This ensures logging works even if user forgets to call configure_logging()
if not _configured:
    configure_logging()
