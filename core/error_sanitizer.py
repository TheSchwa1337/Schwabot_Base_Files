# -*- coding: utf-8 -*-\nfrom core.unified_math_system import unified_math
import numpy as np
import math
# #!/usr/bin/env python3
"""Error Sanitizer - Comprehensive Exception Sanitization and Recovery.

This module provides comprehensive error sanitization for the mathematical
trading system, building on the existing ErrorHandler infrastructure to
provide mathematical-specific error recovery and sanitization.

Architecture:
- Integrates with existing ErrorHandler for consistency
- Provides mathematical computation error recovery
- Sanitizes all exceptions with formatted tracebacks
- Maintains error history for pattern analysis
"""

import logging
import traceback
import functools
import time
import sys
from typing import Any, Callable, Dict, List, Optional, Type, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from core.error_handler import ErrorHandler, ErrorContext, ErrorSeverity

logger = logging.getLogger(__name__)


class SanitizationLevel(Enum):


    """Error sanitization levels."""

BASIC = "basic"           # Basic exception catching
DETAILED = "detailed"     # Detailed traceback logging
RECOVERY = "recovery"     # Attempt error recovery
MATHEMATICAL = "mathematical"  # Mathematical computation recovery


@dataclass
class SanitizedError:


    """Represents a sanitized error with recovery information."""

original_exception: Exception
sanitized_message: str
traceback_formatted: str
recovery_attempted: bool
recovery_successful: bool
fallback_value: Any
timestamp: datetime = field(default_factory=datetime.now)
    function_name: str = ""
module_name: str = ""
sanitization_level: SanitizationLevel = SanitizationLevel.BASIC


class ErrorSanitizer:


    """Comprehensive error sanitization with mathematical trading focus."""

def __init__(self, sanitization_level: SanitizationLevel = SanitizationLevel.DETAILED):


    pass
    pass
        """Initialize the error sanitizer."""
self.sanitization_level = sanitization_level
self.error_handler = ErrorHandler()
        self.sanitized_errors: List[SanitizedError] = []
self.max_error_history = 1000

        # Mathematical recovery defaults
self.mathematical_defaults = {
'float': 0.0,
'int': 0,
'list': [],
'dict': {},
'bool': False,
'str': "",
'numpy_array': None,  # Will be handled specially
'dataframe': None     # Will be handled specially
}

        # Register mathematical error handlers
self._register_mathematical_handlers()

logger.info(f"ErrorSanitizer initialized with level: {sanitization_level.value}")

def _register_mathematical_handlers(self) -> None:


    pass
    pass
        """Register mathematical-specific error handlers."""
        # Register handlers for mathematical computation errors
self.error_handler.register_handler(ZeroDivisionError, self._handle_zero_division)
        self.error_handler.register_handler(OverflowError, self._handle_overflow)
        self.error_handler.register_handler(FloatingPointError, self._handle_floating_point)

        # Register handlers for numpy/pandas errors if available
        try:
    pass
    pass
#             from core.unified_math_system import unified_math  # F811: duplicate import
self.error_handler.register_handler(np.linalg.LinAlgError, self._handle_linalg_error)
        except ImportError:
    pass
    pass
            pass

        try:
    pass
    pass
import pandas as pd
            # Register pandas-specific handlers if needed
        except ImportError:
    pass
    pass
            pass

def catch(self,


              func: Callable,
*args,
fallback_value: Any = None,
recovery_strategy: str = "default",
reraise: bool = False,
**kwargs) -> Any:
"""
Catch and sanitize exceptions from function execution.

Args:
func: Function to execute safely
*args: Arguments for the function
fallback_value: Value to return if function fails
recovery_strategy: Strategy for error recovery
reraise: Whether to re-raise after sanitization
**kwargs: Keyword arguments for the function

Returns:
Function result or fallback value
"""
start_time = time.time()

        try:
    pass
    pass
result = func(*args, **kwargs)
            return result

        except Exception as e:
            # Create sanitized error record
sanitized_error = self._create_sanitized_error(
                e, func, fallback_value, recovery_strategy


            # Store in history
self._store_sanitized_error(sanitized_error)

            # Log sanitized error
self._log_sanitized_error(sanitized_error)

            # Attempt recovery if enabled
            if self.sanitization_level in [SanitizationLevel.RECOVERY, SanitizationLevel.MATHEMATICAL]:
recovery_result = self._attempt_recovery(sanitized_error, func, args, kwargs)
                if recovery_result is not None:
sanitized_error.recovery_successful = True
sanitized_error.fallback_value = recovery_result
                    return recovery_result

            # Re-raise if requested
            if reraise:
raise

            # Return fallback value
            return fallback_value if fallback_value is not None else sanitized_error.fallback_value

def _create_sanitized_error(self,


                               exception: Exception,
func: Callable,
fallback_value: Any,
recovery_strategy: str) -> SanitizedError:
"""Create a sanitized error record."""
        # Get function information
func_name = getattr(func, '__name__', 'unknown')
        module_name = getattr(func, '__module__', 'unknown')

        # Format traceback
traceback_formatted = traceback.format_exc()

        # Create sanitized message
sanitized_message = self._sanitize_error_message(exception, func_name)

        # Determine recovery attempt
recovery_attempted = self.sanitization_level in [
SanitizationLevel.RECOVERY,
SanitizationLevel.MATHEMATICAL
]

        return SanitizedError(
            original_exception=exception,
sanitized_message=sanitized_message,
traceback_formatted=traceback_formatted,
recovery_attempted=recovery_attempted,
recovery_successful=False,
fallback_value=fallback_value,
function_name=func_name,
module_name=module_name,
sanitization_level=self.sanitization_level


def _sanitize_error_message(self, exception: Exception, func_name: str) -> str:


    pass
    pass
        """Sanitize error message for safe logging."""
error_type = type(exception).__name__
        error_msg = str(exception)

        # Remove potentially sensitive information
sanitized_msg = error_msg.replace(sys.path[0], '[PROJECT_ROOT]')

        # Format for mathematical context
        if any(math_term in error_msg.lower() for math_term in)
               ['division', 'overflow', 'underflow', 'nan', 'in']):
            return f"[MATH ERROR] {error_type} in {func_name}: {sanitized_msg}"

        return f"[SANITIZED ERROR] {error_type} in {func_name}: {sanitized_msg}"

def _attempt_recovery(self,


                         sanitized_error: SanitizedError,
func: Callable,
args: tuple,
kwargs: dict) -> Any:
"""Attempt error recovery based on error type and context."""
exception = sanitized_error.original_exception

        # Mathematical recovery strategies
        if self.sanitization_level == SanitizationLevel.MATHEMATICAL:
            return self._mathematical_recovery(exception, func, args, kwargs)

        # Generic recovery strategies
        return self._generic_recovery(exception, func, args, kwargs)

def _mathematical_recovery(self,


                              exception: Exception,
func: Callable,
args: tuple,
kwargs: dict) -> Any:
"""Mathematical-specific error recovery."""
        if isinstance(exception, ZeroDivisionError):
            # Return infinity or a large number for division by zero
            return float('in')

        elif isinstance(exception, OverflowError):
            # Return maximum float value
            return sys.float_info.max

        elif isinstance(exception, (ValueError, TypeError)):
            # Attempt to infer return type from function name or args
func_name = getattr(func, '__name__', '').lower()

            if 'calculate' in func_name or 'compute' in func_name:
                return 0.0  # Mathematical calculation default
            elif 'validate' in func_name or 'check' in func_name:
                return False  # Validation default
            elif 'get' in func_name or 'fetch' in func_name:
                return None  # Getter default

        return None

def _generic_recovery(self,


                         exception: Exception,
func: Callable,
args: tuple,
kwargs: dict) -> Any:
"""Generic error recovery strategies."""
        # Try to infer appropriate default based on exception type
        if isinstance(exception, (KeyError, AttributeError)):
            return None
        elif isinstance(exception, (IndexError, ValueError)):
            return self.mathematical_defaults.get('list', [])
        elif isinstance(exception, TypeError):
            return self.mathematical_defaults.get('dict', {})

        return None

def _handle_zero_division(self, exception: ZeroDivisionError, context: ErrorContext) -> None:


    pass
    pass
        """Handle zero division errors specifically."""
logger.warning(
            f"Zero division in {context.module_name}.{context.function_name}: "
"Mathematical operation attempted division by zero"

context.severity = ErrorSeverity.MEDIUM

def _handle_overflow(self, exception: OverflowError, context: ErrorContext) -> None:


    pass
    pass
        """Handle overflow errors specifically."""
logger.warning(
            f"Overflow in {context.module_name}.{context.function_name}: "
"Mathematical computation exceeded limits"

context.severity = ErrorSeverity.HIGH

def _handle_floating_point(self, exception: FloatingPointError, context: ErrorContext) -> None:


    pass
    pass
        """Handle floating point errors specifically."""
logger.warning(
            f"Floating point error in {context.module_name}.{context.function_name}: "
"Numerical precision issue detected"

context.severity = ErrorSeverity.MEDIUM

def _handle_linalg_error(self, exception: Exception, context: ErrorContext) -> None:


    pass
    pass
        """Handle linear algebra errors specifically."""
logger.warning(
            f"Linear algebra error in {context.module_name}.{context.function_name}: "
"Matrix operation failed"

context.severity = ErrorSeverity.HIGH

def _store_sanitized_error(self, sanitized_error: SanitizedError) -> None:


    pass
    pass
        """Store sanitized error in history."""
self.sanitized_errors.append(sanitized_error)

        # Maintain history size
        if len(self.sanitized_errors) > self.max_error_history:
            self.sanitized_errors = self.sanitized_errors[-self.max_error_history:]

def _log_sanitized_error(self, sanitized_error: SanitizedError) -> None:


    pass
    pass
        """Log sanitized error with appropriate level."""
        if self.sanitization_level == SanitizationLevel.BASIC:
logger.error(sanitized_error.sanitized_message)
        else:
logger.error(
                f"{sanitized_error.sanitized_message}\n"
f"Recovery attempted: {sanitized_error.recovery_attempted}\n"
f"Traceback:\n{sanitized_error.traceback_formatted}"


def get_error_statistics(self) -> Dict[str, Any]:


    pass
    pass
        """Get error statistics for monitoring."""
        if not self.sanitized_errors:
            return {'total_errors': 0}

total_errors = len(self.sanitized_errors)
        recovery_attempts = sum(1 for e in self.sanitized_errors if e.recovery_attempted)
        recovery_successes = sum(1 for e in self.sanitized_errors if e.recovery_successful)

        # Group by error type
error_types = {}
        for error in self.sanitized_errors:
error_type = type(error.original_exception).__name__
            error_types[error_type] = error_types.get(error_type, 0) + 1

        return {
'total_errors': total_errors,
'recovery_attempts': recovery_attempts,
'recovery_successes': recovery_successes,
'recovery_rate': recovery_successes / recovery_attempts if recovery_attempts > 0 else 0.0,
'error_types': error_types,
'most_common_error': unified_math.max(error_types.items(), key=lambda x: x[1])[0] if error_types else None
        }

def get_recent_errors(self, hours: int = 1) -> List[SanitizedError]:


    pass
    pass
        """Get recent sanitized errors."""
cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
error for error in self.sanitized_errors
            if error.timestamp > cutoff_time
]

def clear_error_history(self) -> None:


    pass
    pass
        """Clear error history."""
self.sanitized_errors.clear()
        logger.info("Error sanitizer history cleared")


def sanitize_errors(sanitization_level: SanitizationLevel = SanitizationLevel.DETAILED):


    pass
    pass
    """Decorator for automatic error sanitization."""
def decorator(func: Callable) -> Callable:


    pass
    pass
        @functools.wraps(func)
def wrapper(*args, **kwargs):


    pass
    pass
            sanitizer = ErrorSanitizer(sanitization_level)
            return sanitizer.catch(func, *args, **kwargs)
        return wrapper
    return decorator


def create_error_sanitizer(level: SanitizationLevel = SanitizationLevel.DETAILED) -> ErrorSanitizer:


    pass
    pass
    """Create and return a new ErrorSanitizer instance."""
    return ErrorSanitizer(level)


# Convenience functions for common use cases
def sanitize_mathematical_computation(func: Callable) -> Callable:


    pass
    pass
    """Decorator specifically for mathematical computations."""
    return sanitize_errors(SanitizationLevel.MATHEMATICAL)(func)


def safe_execute_with_recovery(func: Callable, *args, **kwargs) -> Any:


    pass
    pass
    """Execute function with mathematical error recovery."""
sanitizer = ErrorSanitizer(SanitizationLevel.MATHEMATICAL)
    return sanitizer.catch(func, *args, **kwargs)
