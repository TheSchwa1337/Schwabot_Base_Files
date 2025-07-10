from typing import Any, Dict, Optional

#!/usr/bin/env python3
"""
Hash Recollection Exceptions
============================

Custom exceptions for the hash_recollection trading system.
All exceptions are designed to be informative and actionable.
"""


class HashRecollectionError(Exception):
    """Base exception for all hash_recollection errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Initialize the exception with message and optional details."""
        super().__init__(message)
        self.message = message
        self.details = details or {}


class EntropyCalculationError(HashRecollectionError):
    """Raised when entropy calculation fails."""

    def __init__(self, message: str, price_data_length: Optional[int] = None):
        """Initialize with error message and optional data length."""
        details = {"price_data_length": price_data_length} if price_data_length else {}
        super().__init__(f"Entropy calculation failed: {message}", details)


class BitOperationError(HashRecollectionError):
    """Raised when bit operations fail."""

    def __init__(self, message: str, operation: Optional[str] = None):
        """Initialize with error message and operation type."""
        details = {"operation": operation} if operation else {}
        super().__init__(f"Bit operation failed: {message}", details)


class PatternDetectionError(HashRecollectionError):
    """Raised when pattern detection fails."""

    def __init__(self, message: str, pattern_type: Optional[str] = None):
        """Initialize with error message and pattern type."""
        details = {"pattern_type": pattern_type} if pattern_type else {}
        super().__init__(f"Pattern detection failed: {message}", details)


class APIError(HashRecollectionError):
    """Raised when API operations fail."""

    def __init__(
        self,
        message: str,
        endpoint: Optional[str] = None,
        status_code: Optional[int] = None,
    ):
        """Initialize with error message, endpoint, and status code."""
        details = {"endpoint": endpoint, "status_code": status_code} if endpoint or status_code else {}
        super().__init__(f"API error: {message}", details)


class ConfigurationError(HashRecollectionError):
    """Raised when configuration is invalid or missing."""

    def __init__(self, message: str, config_key: Optional[str] = None):
        """Initialize with error message and config key."""
        details = {"config_key": config_key} if config_key else {}
        super().__init__(f"Configuration error: {message}", details)


class DataValidationError(HashRecollectionError):
    """Raised when data validation fails."""

    def __init__(
        self,
        message: str,
        data_type: Optional[str] = None,
        data_length: Optional[int] = None,
    ):
        """Initialize with error message, data type, and length."""
        details = {"data_type": data_type, "data_length": data_length} if data_type or data_length else {}
        super().__init__(f"Data validation failed: {message}", details)


class SignalGenerationError(HashRecollectionError):
    """Raised when signal generation fails."""

    def __init__(
        self,
        message: str,
        signal_type: Optional[str] = None,
        confidence: Optional[float] = None,
    ):
        """Initialize with error message, signal type, and confidence."""
        details = {"signal_type": signal_type, "confidence": confidence} if signal_type or confidence else {}
        super().__init__(f"Signal generation failed: {message}", details)


class MathSystemError(HashRecollectionError):
    """Raised when unified math system operations fail."""

    def __init__(self, message: str, operation: Optional[str] = None):
        """Initialize with error message and operation type."""
        details = {"operation": operation} if operation else {}
        super().__init__(f"Math system error: {message}", details)


class MemoryError(HashRecollectionError):
    """Raised when memory operations fail."""

    def __init__(self, message: str, memory_type: Optional[str] = None):
        """Initialize with error message and memory type."""
        details = {"memory_type": memory_type} if memory_type else {}
        super().__init__(f"Memory error: {message}", details)


class IntegrationError(HashRecollectionError):
    """Raised when module integration fails."""

    def __init__(self, message: str, module: Optional[str] = None):
        """Initialize with error message and module name."""
        details = {"module": module} if module else {}
        super().__init__(f"Integration error: {message}", details)


# Utility functions for exception handling
def handle_entropy_error(func):
    """Decorator to handle entropy calculation errors."""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            raise EntropyCalculationError(str(e))

    return wrapper


def handle_bit_operation_error(func):
    """Decorator to handle bit operation errors."""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            raise BitOperationError(str(e))

    return wrapper


def handle_pattern_error(func):
    """Decorator to handle pattern detection errors."""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            raise PatternDetectionError(str(e))

    return wrapper


def handle_api_error(func):
    """Decorator to handle API errors."""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            raise APIError(str(e))

    return wrapper
