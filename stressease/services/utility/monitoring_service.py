"""
Monitoring and Tracing Service.

This module handles the initialization of LangSmith for tracing and monitoring
LLM performance, latency, and token usage.
"""

import os
from config import Config
import logging

logger = logging.getLogger(__name__)


def init_monitoring():
    """
    Initialize LangSmith tracing and monitoring.

    This function sets the necessary environment variables for LangChain to
    automatically trace all LLM calls to LangSmith.
    """
    if Config.LANGCHAIN_TRACING_V2 and Config.LANGCHAIN_API_KEY:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = Config.LANGCHAIN_API_KEY
        os.environ["LANGCHAIN_PROJECT"] = Config.LANGCHAIN_PROJECT

        logger.info(
            f"LangSmith monitoring initialized successfully - Project: {Config.LANGCHAIN_PROJECT}"
        )
    else:
        logger.info("LangSmith monitoring skipped (missing API key or disabled)")


def log_error(error_type, error_message, context=None):
    """
    Log an error to the monitoring system.

    Args:
        error_type (str): Category of error (e.g., 'llm_error', 'validation_error')
        error_message (str): Description of the error
        context (dict, optional): Additional context data
    """
    # Log with structured format
    if context:
        logger.error(f"[{error_type.upper()}] {error_message} | Context: {context}")
    else:
        logger.error(f"[{error_type.upper()}] {error_message}")
