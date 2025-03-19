"""
Logging configuration for the Algotrader project.
This module provides standard logging configuration across the application.
"""

import os
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(
    level=logging.INFO,
    log_dir="logs",
    console=True,
    file_logging=True,
    max_size=10 * 1024 * 1024,  # 10 MB
    backup_count=5
):
    """
    Configure logging for the application.
    
    Args:
        level: Logging level (default: INFO)
        log_dir: Directory to store log files
        console: Whether to log to console
        file_logging: Whether to log to file
        max_size: Maximum size of log file before rotation
        backup_count: Number of backup log files to keep
    """
    # Create logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Add console handler if requested
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # Add file handler if requested
    if file_logging:
        # Create log directory if it doesn't exist
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True, parents=True)
        
        # Add rotating file handler
        file_handler = RotatingFileHandler(
            log_path / "algotrader.log",
            maxBytes=max_size,
            backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Log initial message
    root_logger.info("Logging initialized")
    return root_logger