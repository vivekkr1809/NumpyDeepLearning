"""Structured logging utilities for production debuggability.

This module provides a centralized logging infrastructure with:
- Structured logging with context
- Proper error handling with stacktraces
- Configurable log levels and handlers
- Domain-specific loggers
- Performance tracking

Following Clean Architecture principles:
- Separation of concerns: Logging infrastructure separate from business logic
- Dependency Inversion: Components depend on logging abstraction
- Open/Closed: Easy to extend with custom handlers
"""

import logging
import sys
import traceback
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime
import json


class StructuredLogger:
    """
    Structured logger with context and error handling.

    Provides enhanced logging capabilities including:
    - Automatic context injection (timestamp, logger name, level)
    - Structured data logging (JSON-compatible)
    - Exception logging with full stacktraces
    - Performance tracking
    - Configurable output formats

    Args:
        name: Logger name (typically module name)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to write logs
        console: Whether to output to console
        structured: Whether to use structured (JSON) format

    Example:
        >>> logger = StructuredLogger(__name__)
        >>> logger.info("Training started", epoch=1, lr=0.001)
        >>> logger.error("Training failed", error=str(e), exc_info=True)
    """

    def __init__(
        self,
        name: str,
        level: int = logging.INFO,
        log_file: Optional[Path] = None,
        console: bool = True,
        structured: bool = False
    ):
        self.name = name
        self.structured = structured
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False

        # Remove existing handlers to avoid duplicates
        self.logger.handlers = []

        # Create formatter
        if structured:
            formatter = StructuredFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )

        # Console handler
        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # File handler
        if log_file:
            log_file = Path(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def _format_message(self, message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Format message with optional context."""
        if context and not self.structured:
            context_str = " | ".join(f"{k}={v}" for k, v in context.items())
            return f"{message} | {context_str}"
        return message

    def debug(self, message: str, exc_info: bool = False, **context):
        """Log debug message with optional context."""
        msg = self._format_message(message, context)
        extra = {'context': context} if self.structured else {}
        self.logger.debug(msg, exc_info=exc_info, extra=extra)

    def info(self, message: str, exc_info: bool = False, **context):
        """Log info message with optional context."""
        msg = self._format_message(message, context)
        extra = {'context': context} if self.structured else {}
        self.logger.info(msg, exc_info=exc_info, extra=extra)

    def warning(self, message: str, exc_info: bool = False, **context):
        """Log warning message with optional context."""
        msg = self._format_message(message, context)
        extra = {'context': context} if self.structured else {}
        self.logger.warning(msg, exc_info=exc_info, extra=extra)

    def error(self, message: str, exc_info: bool = True, **context):
        """
        Log error message with stacktrace.

        Args:
            message: Error message
            exc_info: Whether to include exception info (default: True)
            **context: Additional context to log
        """
        msg = self._format_message(message, context)
        extra = {'context': context} if self.structured else {}
        self.logger.error(msg, exc_info=exc_info, extra=extra)

    def critical(self, message: str, exc_info: bool = True, **context):
        """Log critical message with stacktrace."""
        msg = self._format_message(message, context)
        extra = {'context': context} if self.structured else {}
        self.logger.critical(msg, exc_info=exc_info, extra=extra)

    def exception(self, message: str, **context):
        """
        Log exception with full stacktrace and context.

        Should be called from an exception handler.

        Args:
            message: Error description
            **context: Additional context about the error

        Example:
            >>> try:
            ...     risky_operation()
            ... except Exception as e:
            ...     logger.exception("Operation failed", operation="training", epoch=5)
        """
        msg = self._format_message(message, context)
        extra = {'context': context} if self.structured else {}
        self.logger.exception(msg, extra=extra)

    def log_performance(self, operation: str, duration: float, **context):
        """
        Log performance metrics.

        Args:
            operation: Name of the operation
            duration: Duration in seconds
            **context: Additional metrics
        """
        context['duration_seconds'] = duration
        self.info(f"Performance: {operation}", **context)


class StructuredFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.

    Outputs log records as JSON objects for easier parsing and analysis.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }

        # Add context if available
        if hasattr(record, 'context') and record.context:
            log_data['context'] = record.context

        # Add exception info if available
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'stacktrace': self.formatException(record.exc_info)
            }

        return json.dumps(log_data)


class LoggerFactory:
    """
    Factory for creating domain-specific loggers.

    Provides consistent logger configuration across the application.
    Follows the Factory pattern for object creation.

    Example:
        >>> factory = LoggerFactory(level=logging.DEBUG, log_dir='./logs')
        >>> trainer_logger = factory.get_logger('training')
        >>> optimizer_logger = factory.get_logger('optimizer')
    """

    def __init__(
        self,
        level: int = logging.INFO,
        log_dir: Optional[Path] = None,
        console: bool = True,
        structured: bool = False
    ):
        """
        Initialize logger factory.

        Args:
            level: Default logging level
            log_dir: Directory for log files (None for no file logging)
            console: Whether to output to console
            structured: Whether to use structured (JSON) format
        """
        self.level = level
        self.log_dir = Path(log_dir) if log_dir else None
        self.console = console
        self.structured = structured
        self._loggers: Dict[str, StructuredLogger] = {}

    def get_logger(self, name: str) -> StructuredLogger:
        """
        Get or create a logger for a specific domain.

        Args:
            name: Logger name (e.g., 'training', 'optimizer', 'data')

        Returns:
            StructuredLogger instance
        """
        if name in self._loggers:
            return self._loggers[name]

        log_file = None
        if self.log_dir:
            log_file = self.log_dir / f"{name}.log"

        logger = StructuredLogger(
            name=f"numpy_dl.{name}",
            level=self.level,
            log_file=log_file,
            console=self.console,
            structured=self.structured
        )

        self._loggers[name] = logger
        return logger

    def set_level(self, level: int):
        """Set logging level for all loggers."""
        self.level = level
        for logger in self._loggers.values():
            logger.logger.setLevel(level)


# Global logger factory instance
_global_factory: Optional[LoggerFactory] = None


def configure_logging(
    level: int = logging.INFO,
    log_dir: Optional[str] = None,
    console: bool = True,
    structured: bool = False
):
    """
    Configure global logging settings.

    This should be called once at application startup.

    Args:
        level: Logging level (logging.DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (None for no file logging)
        console: Whether to output to console
        structured: Whether to use structured (JSON) format

    Example:
        >>> import logging
        >>> configure_logging(level=logging.DEBUG, log_dir='./logs')
    """
    global _global_factory
    _global_factory = LoggerFactory(
        level=level,
        log_dir=log_dir,
        console=console,
        structured=structured
    )


def get_logger(name: str) -> StructuredLogger:
    """
    Get a logger for a specific domain.

    If logging hasn't been configured, uses default settings.

    Args:
        name: Logger name (e.g., 'training', 'optimizer', 'data')

    Returns:
        StructuredLogger instance

    Example:
        >>> logger = get_logger('training')
        >>> logger.info("Training started", epoch=1, batch_size=32)
    """
    global _global_factory
    if _global_factory is None:
        _global_factory = LoggerFactory()
    return _global_factory.get_logger(name)


class ContextLogger:
    """
    Context manager for logging with automatic timing and error handling.

    Automatically logs operation start, duration, and any exceptions.

    Example:
        >>> logger = get_logger('training')
        >>> with ContextLogger(logger, "epoch_training", epoch=1):
        ...     train_model()
        # Automatically logs start, duration, and any exceptions
    """

    def __init__(self, logger: StructuredLogger, operation: str, **context):
        """
        Initialize context logger.

        Args:
            logger: StructuredLogger instance
            operation: Name of the operation
            **context: Additional context to log
        """
        self.logger = logger
        self.operation = operation
        self.context = context
        self.start_time = None

    def __enter__(self):
        """Enter context: log operation start."""
        self.start_time = datetime.now()
        self.logger.debug(f"Starting {self.operation}", **self.context)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context: log duration and any errors."""
        duration = (datetime.now() - self.start_time).total_seconds()

        if exc_type is not None:
            # Log exception with context
            self.logger.exception(
                f"Failed {self.operation}",
                duration_seconds=duration,
                error_type=exc_type.__name__,
                **self.context
            )
            return False  # Re-raise exception
        else:
            # Log successful completion
            self.logger.debug(
                f"Completed {self.operation}",
                duration_seconds=duration,
                **self.context
            )
            return True


def log_function_call(logger: StructuredLogger):
    """
    Decorator for automatic function call logging.

    Logs function entry, exit, duration, and exceptions.

    Args:
        logger: StructuredLogger instance

    Example:
        >>> logger = get_logger('training')
        >>> @log_function_call(logger)
        ... def train_epoch(epoch, lr):
        ...     # Training code
        ...     pass
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            start_time = datetime.now()

            logger.debug(f"Calling {func_name}", args=str(args)[:100], kwargs=str(kwargs)[:100])

            try:
                result = func(*args, **kwargs)
                duration = (datetime.now() - start_time).total_seconds()
                logger.debug(f"Completed {func_name}", duration_seconds=duration)
                return result
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                logger.exception(
                    f"Exception in {func_name}",
                    duration_seconds=duration,
                    error=str(e)
                )
                raise

        return wrapper
    return decorator
