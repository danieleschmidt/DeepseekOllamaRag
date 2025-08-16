"""Comprehensive logging configuration for DeepseekOllamaRag application."""

import os
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
from typing import Optional
from config import config


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support for console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset_color = self.COLORS['RESET']
        
        # Add color to levelname
        record.levelname = f"{log_color}{record.levelname}{reset_color}"
        
        return super().format(record)


class StructuredLogger:
    """Structured logging with multiple handlers and formatting."""
    
    def __init__(self, name: str = "deepseek_rag"):
        self.name = name
        self.logger = logging.getLogger(name)
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        
        # Clear existing handlers
        self.logger.handlers = []
        
        # Set log level
        level = logging.DEBUG if config.app.debug else logging.INFO
        self.logger.setLevel(level)
        
        self._setup_console_handler()
        self._setup_file_handler()
        self._setup_error_handler()
        
        # Prevent duplicate logs
        self.logger.propagate = False
    
    def _setup_console_handler(self):
        """Setup colored console handler."""
        console_handler = logging.StreamHandler()
        console_formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.INFO)
        self.logger.addHandler(console_handler)
    
    def _setup_file_handler(self):
        """Setup rotating file handler for general logs."""
        log_file = self.log_dir / "app.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
    
    def _setup_error_handler(self):
        """Setup separate handler for errors."""
        error_file = self.log_dir / "errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=3,
            encoding='utf-8'
        )
        
        error_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d\n'
            'Message: %(message)s\n'
            'Exception: %(exc_info)s\n'
            '---\n',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        error_handler.setFormatter(error_formatter)
        error_handler.setLevel(logging.ERROR)
        self.logger.addHandler(error_handler)
    
    def get_logger(self) -> logging.Logger:
        """Get the configured logger."""
        return self.logger


# Utility functions for specific logging scenarios
def log_document_processing(file_path: str, chunks_count: int, processing_time: float):
    """Log document processing metrics."""
    logger = StructuredLogger().get_logger()
    logger.info(
        f"Document processed: {file_path} | "
        f"Chunks: {chunks_count} | "
        f"Time: {processing_time:.2f}s"
    )


def log_query_processing(question: str, response_time: float, success: bool):
    """Log query processing metrics."""
    logger = StructuredLogger().get_logger()
    status = "SUCCESS" if success else "FAILED"
    logger.info(
        f"Query processed: {status} | "
        f"Question: {question[:50]}... | "
        f"Time: {response_time:.2f}s"
    )


def log_error_with_context(error: Exception, context: dict):
    """Log error with additional context."""
    logger = StructuredLogger().get_logger()
    context_str = " | ".join([f"{k}: {v}" for k, v in context.items()])
    logger.error(f"Error: {str(error)} | Context: {context_str}", exc_info=True)


def log_system_health(metrics: dict):
    """Log system health metrics."""
    logger = StructuredLogger().get_logger()
    metrics_str = " | ".join([f"{k}: {v}" for k, v in metrics.items()])
    logger.info(f"System Health: {metrics_str}")


# Performance monitoring decorators
def log_execution_time(func):
    """Decorator to log function execution time."""
    import functools
    import time
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = StructuredLogger().get_logger()
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"{func.__name__} executed in {execution_time:.3f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.3f}s: {str(e)}")
            raise
    
    return wrapper


# Global logger instance
global_logger = StructuredLogger().get_logger()