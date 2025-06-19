from core.error_handler import ErrorContext
#!/usr/bin/env python3
"""
Centralized Error Handler - Schwabot Fault Tolerance System
==========================================================

Provides centralized error handling with consistent patterns,
fallback mechanisms, and comprehensive logging for all Schwabot modules.

Based on systematic elimination of 257+ flake8 issues.
"""

import logging
import sys
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type
from functools import wraps

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels for consistent handling"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorContext:
    """Context information for error handling"""
    function_name: str
    module_name: str
    line_number: int
    timestamp: datetime = field(default_factory=datetime.now)
    additional_context: Dict[str, Any] = field(default_factory=dict)
    severity: ErrorSeverity = ErrorSeverity.MEDIUM


class ErrorHandler:
    """Centralized error handling with consistent patterns"""

    def __init__(self) -> None:
        self._error_registry: Dict[Type[Exception], Callable] = {}
        self._error_history: List[ErrorContext] = []
        self._register_default_handlers()

    def _register_default_handlers(self) -> None:
        """Register default error handlers for common exceptions"""
        self._error_registry.update({
            ImportError: self._handle_import_error,
            ValueError: self._handle_value_error,
            TypeError: self._handle_type_error,
            KeyError: self._handle_key_error,
            IndexError: self._handle_index_error,
            AttributeError: self._handle_attribute_error,
            FileNotFoundError: self._handle_file_not_found,
            PermissionError: self._handle_permission_error,
            ConnectionError: self._handle_connection_error,
            TimeoutError: self._handle_timeout_error,
        })

    def safe_execute(self, func: Callable, *args,
                    error_context: Optional[ErrorContext] = None,
                    default_return: Any = None,
                    reraise: bool = False,
                    **kwargs) -> Any:
        """
        Safely execute a function with comprehensive error handling

        Args:
            func: Function to execute
            error_context: Context information for error handling
            default_return: Value to return if function fails
            reraise: Whether to re-raise the exception after handling
            *args, **kwargs: Arguments to pass to the function

        Returns:
            Function result or default_return if error occurs
        """
        if error_context is None:
            error_context = ErrorContext(
                function_name=func.__name__,
                module_name=func.__module__,
                line_number=getattr(func, '__code__', None).co_firstlineno if hasattr(func, '__code__') else 0
            )

        try:
            return func(*args, **kwargs)

        except Exception as e:
            self._handle_exception(e, error_context)

            if reraise:
                raise

            return default_return

    def _handle_exception(self, exception: Exception, context: ErrorContext) -> None:
        """Handle an exception with the appropriate handler"""
        exception_type = type(exception)

        # Get the appropriate handler
        handler = self._error_registry.get(exception_type, self._handle_generic_error)

        # Execute the handler
        handler(exception, context)

        # Record the error
        self._error_history.append(context)

    def _handle_import_error(self, exception: ImportError, context: ErrorContext) -> None:
        """Handle ImportError with fallback suggestions"""
        logger.warning(
            f"Import error in {context.module_name}.{context.function_name}: "
            f"Module '{exception.name}' not available. Using fallback."
        )
        context.severity = ErrorSeverity.LOW

    def _handle_value_error(self, exception: ValueError, context: ErrorContext) -> None:
        """Handle ValueError with parameter validation context"""
        logger.error(
            f"Value error in {context.module_name}.{context.function_name}: "
            f"Invalid value provided: {exception}"
        )
        context.severity = ErrorSeverity.MEDIUM

    def _handle_type_error(self, exception: TypeError, context: ErrorContext) -> None:
        """Handle TypeError with type checking context"""
        logger.error(
            f"Type error in {context.module_name}.{context.function_name}: "
            f"Type mismatch: {exception}"
        )
        context.severity = ErrorSeverity.MEDIUM

    def _handle_key_error(self, exception: KeyError, context: ErrorContext) -> None:
        """Handle KeyError with dictionary access context"""
        logger.error(
            f"Key error in {context.module_name}.{context.function_name}: "
            f"Missing key: {exception}"
        )
        context.severity = ErrorSeverity.MEDIUM

    def _handle_index_error(self, exception: IndexError, context: ErrorContext) -> None:
        """Handle IndexError with list/array access context"""
        logger.error(
            f"Index error in {context.module_name}.{context.function_name}: "
            f"Invalid index: {exception}"
        )
        context.severity = ErrorSeverity.MEDIUM

    def _handle_attribute_error(self, exception: AttributeError, context: ErrorContext) -> None:
        """Handle AttributeError with object attribute access context"""
        logger.error(
            f"Attribute error in {context.module_name}.{context.function_name}: "
            f"Missing attribute: {exception}"
        )
        context.severity = ErrorSeverity.MEDIUM

    def _handle_file_not_found(self, exception: FileNotFoundError, context: ErrorContext) -> None:
        """Handle FileNotFoundError with file path context"""
        logger.error(
            f"File not found in {context.module_name}.{context.function_name}: "
            f"File: {exception.filename}"
        )
        context.severity = ErrorSeverity.HIGH

    def _handle_permission_error(self, exception: PermissionError, context: ErrorContext) -> None:
        """Handle PermissionError with file system context"""
        logger.error(
            f"Permission error in {context.module_name}.{context.function_name}: "
            f"Access denied: {exception.filename}"
        )
        context.severity = ErrorSeverity.HIGH

    def _handle_connection_error(self, exception: ConnectionError, context: ErrorContext) -> None:
        """Handle ConnectionError with network context"""
        logger.error(
            f"Connection error in {context.module_name}.{context.function_name}: "
            f"Network issue: {exception}"
        )
        context.severity = ErrorSeverity.HIGH

    def _handle_timeout_error(self, exception: TimeoutError, context: ErrorContext) -> None:
        """Handle TimeoutError with timing context"""
        logger.error(
            f"Timeout error in {context.module_name}.{context.function_name}: "
            f"Operation timed out: {exception}"
        )
        context.severity = ErrorSeverity.MEDIUM

    def _handle_generic_error(self, exception: Exception, context: ErrorContext) -> None:
        """Handle any unregistered exception type"""
        logger.error(
            f"Unexpected error in {context.module_name}.{context.function_name}: "
            f"{type(exception).__name__}: {exception}"
        )
        context.severity = ErrorSeverity.CRITICAL

    def register_handler(self, exception_type: Type[Exception],
                        handler: Callable[[Exception, ErrorContext], None]) -> None:
        """Register a custom error handler for a specific exception type"""
        self._error_registry[exception_type] = handler

    def get_error_summary(self) -> Dict[str, int]:
        """Get a summary of errors by severity"""
        summary = {severity.value: 0 for severity in ErrorSeverity}
        for context in self._error_history:
            summary[context.severity.value] += 1
        return summary

    def clear_history(self) -> None:
        """Clear the error history"""
        self._error_history.clear()


# Global error handler instance
error_handler = ErrorHandler()


def safe_execute(func: Callable, *args,
                error_context: Optional[ErrorContext] = None,
                default_return: Any = None,
                reraise: bool = False,
                **kwargs) -> Any:
    """Convenience function for safe execution"""
    return error_handler.safe_execute(func, *args,
                                     error_context=error_context,
                                     default_return=default_return,
                                     reraise=reraise, **kwargs)


def error_handler_decorator(default_return: Any = None, reraise: bool = False) -> Callable:
    """Decorator for automatic error handling"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            return safe_execute(func, *args, default_return=default_return, 
                              reraise=reraise, **kwargs)
        return wrapper
    return decorator


def safe_import_decorator(module_name: str, class_names: List[str]) -> Callable:
    """Decorator for safe import handling"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except ImportError as e:
                logger.warning(f"Import error in {func.__name__}: {e}")
                return None
        return wrapper
    return decorator