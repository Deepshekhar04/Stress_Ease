"""
Centralized logging configuration for StressEase backend.

This module provides structured logging with different levels:
- DEBUG: Detailed debug information (only shown when DEBUG=True)
- INFO: Important events (session creation, API calls)
- WARNING: Unexpected but handled situations
- ERROR: Error events that need attention
- CRITICAL: Critical failures

Usage:
    from stressease.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("User session created", extra={"user_id": user_id})
"""

import logging
import sys
from typing import Optional


# Global flag to track if logging has been configured
_logging_configured = False


def setup_logging(debug: bool = False) -> None:
    """
    Configure application-wide logging.

    Args:
        debug (bool): If True, set log level to DEBUG. Otherwise INFO.
    """
    global _logging_configured

    if _logging_configured:
        return

    # Set log level based on debug flag
    log_level = logging.DEBUG if debug else logging.INFO

    # Create formatter with timestamp, level, name, and message
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Add console handler (cloud platforms capture stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Reduce noise from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("google").setLevel(logging.WARNING)
    logging.getLogger("firebase_admin").setLevel(logging.WARNING)

    _logging_configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.

    Args:
        name (str): Module name (use __name__)

    Returns:
        logging.Logger: Configured logger instance
    """
    return logging.getLogger(name)


def log_with_context(
    logger: logging.Logger,
    level: int,
    message: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    endpoint: Optional[str] = None,
    **kwargs,
) -> None:
    """
    Log a message with contextual information.

    Args:
        logger: Logger instance
        level: Log level (logging.DEBUG, logging.INFO, etc.)
        message: Log message
        user_id: Optional user ID
        session_id: Optional session ID
        endpoint: Optional endpoint name
        **kwargs: Additional context to include
    """
    context = {}

    if user_id:
        context["user_id"] = user_id
    if session_id:
        context["session_id"] = session_id
    if endpoint:
        context["endpoint"] = endpoint

    context.update(kwargs)

    # Format context as key=value pairs
    context_str = " ".join(f"{k}={v}" for k, v in context.items())

    if context_str:
        full_message = f"{message} | {context_str}"
    else:
        full_message = message

    logger.log(level, full_message)
