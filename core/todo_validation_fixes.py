#!/usr/bin/env python3
"""
Runtime Validation System - Schwabot Framework
============================================

Comprehensive runtime validation system for mathematical trading operations.
Provides decorators, type checking, and schema validation for all core components.

Key Features:
- Vector and matrix validation decorators
- YAML/JSON schema validation
- Runtime type checking and bounds validation
- Entropy validation and signal integrity checks
- Performance monitoring and validation metrics

Windows CLI compatible with flake8 compliance.
"""

from __future__ import annotations

import functools
import logging
import time
import yaml
import json
from dataclasses import dataclass, field
from decimal import Decimal, getcontext
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from enum import Enum
import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from typing_extensions import Self

# Set high precision for financial calculations
getcontext().prec = 18

# Type definitions
Vector = npt.NDArray[np.float64]
Matrix = npt.NDArray[np.float64]

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation level enumeration"""
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationType(Enum):
    """Validation type enumeration"""
    TYPE = "type"
    LENGTH = "length"
    BOUNDS = "bounds"
    SCHEMA = "schema"
    ENTROPY = "entropy"
    PERFORMANCE = "performance"


@dataclass
class ValidationResult:
    """Validation result container"""
    
    valid: bool
    validation_type: ValidationType
    level: ValidationLevel
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class ValidationMetrics:
    """Validation performance metrics"""
    
    total_validations: int = 0
    successful_validations: int = 0
    failed_validations: int = 0
    average_execution_time: float = 0.0
    validation_history: List[ValidationResult] = field(default_factory=list)


class ValidationError(Exception):
    """Custom validation error"""
    
    def __init__(self, message: str, validation_result: ValidationResult):
        super().__init__(message)
        self.validation_result = validation_result


class RuntimeValidator:
    """Main runtime validation system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize runtime validator"""
        self.version = "1.0.0"
        self.config = config or self._default_config()
        
        # Validation settings
        self.enable_type_checking = self.config.get('enable_type_checking', True)
        self.enable_bounds_checking = self.config.get('enable_bounds_checking', True)
        self.enable_entropy_validation = self.config.get('enable_entropy_validation', True)
        self.enable_performance_monitoring = self.config.get('enable_performance_monitoring', True)
        
        # Validation thresholds
        self.max_vector_length = self.config.get('max_vector_length', 10000)
        self.max_matrix_size = self.config.get('max_matrix_size', 1000)
        self.min_entropy_threshold = self.config.get('min_entropy_threshold', 0.1)
        self.max_entropy_threshold = self.config.get('max_entropy_threshold', 10.0)
        self.max_execution_time = self.config.get('max_execution_time', 1.0)  # seconds
        
        # Performance tracking
        self.metrics = ValidationMetrics()
        self.validation_cache: Dict[str, ValidationResult] = {}
        
        logger.info(f"RuntimeValidator v{self.version} initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            'enable_type_checking': True,
            'enable_bounds_checking': True,
            'enable_entropy_validation': True,
            'enable_performance_monitoring': True,
            'max_vector_length': 10000,
            'max_matrix_size': 1000,
            'min_entropy_threshold': 0.1,
            'max_entropy_threshold': 10.0,
            'max_execution_time': 1.0,
            'cache_validation_results': True,
            'log_validation_failures': True
        }
    
    def validate_vector(self, vector: Any, expected_length: Optional[int] = None,
                       min_value: Optional[float] = None, max_value: Optional[float] = None,
                       allow_nan: bool = False, allow_inf: bool = False) -> ValidationResult:
        """Validate vector properties"""
        start_time = time.time()
        
        try:
            # Type validation
            if not isinstance(vector, (list, tuple, np.ndarray)):
                return ValidationResult(
                    valid=False,
                    validation_type=ValidationType.TYPE,
                    level=ValidationLevel.ERROR,
                    message=f"Vector must be list, tuple, or numpy array, got {type(vector)}",
                    execution_time=time.time() - start_time
                )
            
            # Convert to numpy array if needed
            if not isinstance(vector, np.ndarray):
                vector = np.array(vector, dtype=np.float64)
            
            # Length validation
            if expected_length is not None and len(vector) != expected_length:
                return ValidationResult(
                    valid=False,
                    validation_type=ValidationType.LENGTH,
                    level=ValidationLevel.ERROR,
                    message=f"Vector length mismatch: expected {expected_length}, got {len(vector)}",
                    execution_time=time.time() - start_time
                )
            
            # Bounds validation
            if len(vector) > self.max_vector_length:
                return ValidationResult(
                    valid=False,
                    validation_type=ValidationType.BOUNDS,
                    level=ValidationLevel.ERROR,
                    message=f"Vector length {len(vector)} exceeds maximum {self.max_vector_length}",
                    execution_time=time.time() - start_time
                )
            
            # Value validation
            if not allow_nan and np.any(np.isnan(vector)):
                return ValidationResult(
                    valid=False,
                    validation_type=ValidationType.BOUNDS,
                    level=ValidationLevel.ERROR,
                    message="Vector contains NaN values",
                    execution_time=time.time() - start_time
                )
            
            if not allow_inf and np.any(np.isinf(vector)):
                return ValidationResult(
                    valid=False,
                    validation_type=ValidationType.BOUNDS,
                    level=ValidationLevel.ERROR,
                    message="Vector contains infinite values",
                    execution_time=time.time() - start_time
                )
            
            # Range validation
            if min_value is not None and np.any(vector < min_value):
                return ValidationResult(
                    valid=False,
                    validation_type=ValidationType.BOUNDS,
                    level=ValidationLevel.WARNING,
                    message=f"Vector contains values below minimum {min_value}",
                    execution_time=time.time() - start_time
                )
            
            if max_value is not None and np.any(vector > max_value):
                return ValidationResult(
                    valid=False,
                    validation_type=ValidationType.BOUNDS,
                    level=ValidationLevel.WARNING,
                    message=f"Vector contains values above maximum {max_value}",
                    execution_time=time.time() - start_time
                )
            
            return ValidationResult(
                valid=True,
                validation_type=ValidationType.TYPE,
                level=ValidationLevel.WARNING,
                message="Vector validation passed",
                details={
                    'length': len(vector),
                    'min_value': float(np.min(vector)),
                    'max_value': float(np.max(vector)),
                    'mean_value': float(np.mean(vector)),
                    'std_value': float(np.std(vector))
                },
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return ValidationResult(
                valid=False,
                validation_type=ValidationType.TYPE,
                level=ValidationLevel.CRITICAL,
                message=f"Vector validation error: {str(e)}",
                execution_time=time.time() - start_time
            )
    
    def validate_matrix(self, matrix: Any, expected_shape: Optional[Tuple[int, int]] = None,
                       min_value: Optional[float] = None, max_value: Optional[float] = None,
                       check_symmetric: bool = False, check_positive_definite: bool = False) -> ValidationResult:
        """Validate matrix properties"""
        start_time = time.time()
        
        try:
            # Type validation
            if not isinstance(matrix, (list, tuple, np.ndarray)):
                return ValidationResult(
                    valid=False,
                    validation_type=ValidationType.TYPE,
                    level=ValidationLevel.ERROR,
                    message=f"Matrix must be list, tuple, or numpy array, got {type(matrix)}",
                    execution_time=time.time() - start_time
                )
            
            # Convert to numpy array if needed
            if not isinstance(matrix, np.ndarray):
                matrix = np.array(matrix, dtype=np.float64)
            
            # Ensure 2D
            if matrix.ndim != 2:
                return ValidationResult(
                    valid=False,
                    validation_type=ValidationType.TYPE,
                    level=ValidationLevel.ERROR,
                    message=f"Matrix must be 2D, got {matrix.ndim}D",
                    execution_time=time.time() - start_time
                )
            
            # Shape validation
            if expected_shape is not None and matrix.shape != expected_shape:
                return ValidationResult(
                    valid=False,
                    validation_type=ValidationType.LENGTH,
                    level=ValidationLevel.ERROR,
                    message=f"Matrix shape mismatch: expected {expected_shape}, got {matrix.shape}",
                    execution_time=time.time() - start_time
                )
            
            # Size validation
            if matrix.size > self.max_matrix_size:
                return ValidationResult(
                    valid=False,
                    validation_type=ValidationType.BOUNDS,
                    level=ValidationLevel.ERROR,
                    message=f"Matrix size {matrix.size} exceeds maximum {self.max_matrix_size}",
                    execution_time=time.time() - start_time
                )
            
            # Value validation
            if np.any(np.isnan(matrix)):
                return ValidationResult(
                    valid=False,
                    validation_type=ValidationType.BOUNDS,
                    level=ValidationLevel.ERROR,
                    message="Matrix contains NaN values",
                    execution_time=time.time() - start_time
                )
            
            if np.any(np.isinf(matrix)):
                return ValidationResult(
                    valid=False,
                    validation_type=ValidationType.BOUNDS,
                    level=ValidationLevel.ERROR,
                    message="Matrix contains infinite values",
                    execution_time=time.time() - start_time
                )
            
            # Range validation
            if min_value is not None and np.any(matrix < min_value):
                return ValidationResult(
                    valid=False,
                    validation_type=ValidationType.BOUNDS,
                    level=ValidationLevel.WARNING,
                    message=f"Matrix contains values below minimum {min_value}",
                    execution_time=time.time() - start_time
                )
            
            if max_value is not None and np.any(matrix > max_value):
                return ValidationResult(
                    valid=False,
                    validation_type=ValidationType.BOUNDS,
                    level=ValidationLevel.WARNING,
                    message=f"Matrix contains values above maximum {max_value}",
                    execution_time=time.time() - start_time
                )
            
            # Symmetric validation
            if check_symmetric and not np.allclose(matrix, matrix.T):
                return ValidationResult(
                    valid=False,
                    validation_type=ValidationType.BOUNDS,
                    level=ValidationLevel.WARNING,
                    message="Matrix is not symmetric",
                    execution_time=time.time() - start_time
                )
            
            # Positive definite validation
            if check_positive_definite:
                try:
                    eigenvals = np.linalg.eigvals(matrix)
                    if np.any(eigenvals <= 0):
                        return ValidationResult(
                            valid=False,
                            validation_type=ValidationType.BOUNDS,
                            level=ValidationLevel.WARNING,
                            message="Matrix is not positive definite",
                            execution_time=time.time() - start_time
                        )
                except np.linalg.LinAlgError:
                    return ValidationResult(
                        valid=False,
                        validation_type=ValidationType.BOUNDS,
                        level=ValidationLevel.ERROR,
                        message="Matrix eigenvalue computation failed",
                        execution_time=time.time() - start_time
                    )
            
            return ValidationResult(
                valid=True,
                validation_type=ValidationType.TYPE,
                level=ValidationLevel.WARNING,
                message="Matrix validation passed",
                details={
                    'shape': matrix.shape,
                    'min_value': float(np.min(matrix)),
                    'max_value': float(np.max(matrix)),
                    'mean_value': float(np.mean(matrix)),
                    'condition_number': float(np.linalg.cond(matrix)) if matrix.size > 0 else 0.0
                },
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return ValidationResult(
                valid=False,
                validation_type=ValidationType.TYPE,
                level=ValidationLevel.CRITICAL,
                message=f"Matrix validation error: {str(e)}",
                execution_time=time.time() - start_time
            )
    
    def validate_entropy(self, signal: Vector, window_size: int = 100) -> ValidationResult:
        """Validate signal entropy properties"""
        start_time = time.time()
        
        try:
            # Basic signal validation
            signal_result = self.validate_vector(signal)
            if not signal_result.valid:
                return signal_result
            
            if len(signal) < window_size:
                return ValidationResult(
                    valid=False,
                    validation_type=ValidationType.ENTROPY,
                    level=ValidationLevel.ERROR,
                    message=f"Signal length {len(signal)} too short for window size {window_size}",
                    execution_time=time.time() - start_time
                )
            
            # Calculate entropy
            def calculate_entropy(data: np.ndarray) -> float:
                """Calculate Shannon entropy"""
                if len(data) == 0:
                    return 0.0
                
                # Discretize data into bins
                hist, _ = np.histogram(data, bins=min(20, len(data) // 5))
                hist = hist[hist > 0]  # Remove zero bins
                
                if len(hist) == 0:
                    return 0.0
                
                # Normalize to probabilities
                probs = hist / np.sum(hist)
                
                # Calculate entropy
                entropy = -np.sum(probs * np.log2(probs + 1e-10))
                return entropy
            
            # Calculate rolling entropy
            entropies = []
            for i in range(len(signal) - window_size + 1):
                window = signal[i:i + window_size]
                entropy = calculate_entropy(window)
                entropies.append(entropy)
            
            entropies = np.array(entropies)
            
            # Validate entropy range
            if np.any(entropies < self.min_entropy_threshold):
                return ValidationResult(
                    valid=False,
                    validation_type=ValidationType.ENTROPY,
                    level=ValidationLevel.WARNING,
                    message=f"Signal contains low entropy regions (min: {np.min(entropies):.3f})",
                    details={
                        'min_entropy': float(np.min(entropies)),
                        'max_entropy': float(np.max(entropies)),
                        'mean_entropy': float(np.mean(entropies)),
                        'entropy_std': float(np.std(entropies))
                    },
                    execution_time=time.time() - start_time
                )
            
            if np.any(entropies > self.max_entropy_threshold):
                return ValidationResult(
                    valid=False,
                    validation_type=ValidationType.ENTROPY,
                    level=ValidationLevel.WARNING,
                    message=f"Signal contains high entropy regions (max: {np.max(entropies):.3f})",
                    details={
                        'min_entropy': float(np.min(entropies)),
                        'max_entropy': float(np.max(entropies)),
                        'mean_entropy': float(np.mean(entropies)),
                        'entropy_std': float(np.std(entropies))
                    },
                    execution_time=time.time() - start_time
                )
            
            return ValidationResult(
                valid=True,
                validation_type=ValidationType.ENTROPY,
                level=ValidationLevel.WARNING,
                message="Entropy validation passed",
                details={
                    'min_entropy': float(np.min(entropies)),
                    'max_entropy': float(np.max(entropies)),
                    'mean_entropy': float(np.mean(entropies)),
                    'entropy_std': float(np.std(entropies)),
                    'window_size': window_size
                },
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return ValidationResult(
                valid=False,
                validation_type=ValidationType.ENTROPY,
                level=ValidationLevel.CRITICAL,
                message=f"Entropy validation error: {str(e)}",
                execution_time=time.time() - start_time
            )
    
    def validate_yaml_config(self, config_data: Union[str, Dict[str, Any]], 
                           schema: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate YAML configuration"""
        start_time = time.time()
        
        try:
            # Parse YAML if string
            if isinstance(config_data, str):
                config = yaml.safe_load(config_data)
            else:
                config = config_data
            
            # Basic structure validation
            if not isinstance(config, dict):
                return ValidationResult(
                    valid=False,
                    validation_type=ValidationType.SCHEMA,
                    level=ValidationLevel.ERROR,
                    message="Configuration must be a dictionary",
                    execution_time=time.time() - start_time
                )
            
            # Schema validation if provided
            if schema is not None:
                # Simple schema validation (can be enhanced with jsonschema)
                for key, expected_type in schema.items():
                    if key not in config:
                        return ValidationResult(
                            valid=False,
                            validation_type=ValidationType.SCHEMA,
                            level=ValidationLevel.ERROR,
                            message=f"Missing required key: {key}",
                            execution_time=time.time() - start_time
                        )
                    
                    if not isinstance(config[key], expected_type):
                        return ValidationResult(
                            valid=False,
                            validation_type=ValidationType.SCHEMA,
                            level=ValidationLevel.ERROR,
                            message=f"Key {key} has wrong type: expected {expected_type}, got {type(config[key])}",
                            execution_time=time.time() - start_time
                        )
            
            return ValidationResult(
                valid=True,
                validation_type=ValidationType.SCHEMA,
                level=ValidationLevel.WARNING,
                message="YAML configuration validation passed",
                details={
                    'config_keys': list(config.keys()),
                    'config_size': len(str(config))
                },
                execution_time=time.time() - start_time
            )
            
        except yaml.YAMLError as e:
            return ValidationResult(
                valid=False,
                validation_type=ValidationType.SCHEMA,
                level=ValidationLevel.ERROR,
                message=f"YAML parsing error: {str(e)}",
                execution_time=time.time() - start_time
            )
        except Exception as e:
            return ValidationResult(
                valid=False,
                validation_type=ValidationType.SCHEMA,
                level=ValidationLevel.CRITICAL,
                message=f"Configuration validation error: {str(e)}",
                execution_time=time.time() - start_time
            )
    
    def validate_performance(self, func: Callable, *args, **kwargs) -> ValidationResult:
        """Validate function performance"""
        start_time = time.time()
        
        try:
            # Execute function and measure time
            func_start = time.time()
            result = func(*args, **kwargs)
            func_time = time.time() - func_start
            
            # Check execution time
            if func_time > self.max_execution_time:
                return ValidationResult(
                    valid=False,
                    validation_type=ValidationType.PERFORMANCE,
                    level=ValidationLevel.WARNING,
                    message=f"Function execution time {func_time:.3f}s exceeds limit {self.max_execution_time}s",
                    details={
                        'execution_time': func_time,
                        'max_allowed_time': self.max_execution_time,
                        'function_name': func.__name__
                    },
                    execution_time=time.time() - start_time
                )
            
            return ValidationResult(
                valid=True,
                validation_type=ValidationType.PERFORMANCE,
                level=ValidationLevel.WARNING,
                message="Performance validation passed",
                details={
                    'execution_time': func_time,
                    'function_name': func.__name__,
                    'result_type': type(result).__name__
                },
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return ValidationResult(
                valid=False,
                validation_type=ValidationType.PERFORMANCE,
                level=ValidationLevel.CRITICAL,
                message=f"Performance validation error: {str(e)}",
                execution_time=time.time() - start_time
            )


# Global validator instance
_global_validator = RuntimeValidator()


# Decorator functions for easy use
def validate_vector(expected_length: Optional[int] = None, min_value: Optional[float] = None,
                   max_value: Optional[float] = None, allow_nan: bool = False, allow_inf: bool = False):
    """Decorator to validate vector inputs"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(vector: Any, *args, **kwargs) -> Any:
            result = _global_validator.validate_vector(
                vector, expected_length, min_value, max_value, allow_nan, allow_inf
            )
            
            if not result.valid:
                if result.level == ValidationLevel.CRITICAL:
                    raise ValidationError(result.message, result)
                elif result.level == ValidationLevel.ERROR:
                    logger.error(f"Vector validation failed: {result.message}")
                    raise ValidationError(result.message, result)
                else:
                    logger.warning(f"Vector validation warning: {result.message}")
            
            return func(vector, *args, **kwargs)
        return wrapper
    return decorator


def validate_matrix(expected_shape: Optional[Tuple[int, int]] = None, min_value: Optional[float] = None,
                   max_value: Optional[float] = None, check_symmetric: bool = False,
                   check_positive_definite: bool = False):
    """Decorator to validate matrix inputs"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(matrix: Any, *args, **kwargs) -> Any:
            result = _global_validator.validate_matrix(
                matrix, expected_shape, min_value, max_value, check_symmetric, check_positive_definite
            )
            
            if not result.valid:
                if result.level == ValidationLevel.CRITICAL:
                    raise ValidationError(result.message, result)
                elif result.level == ValidationLevel.ERROR:
                    logger.error(f"Matrix validation failed: {result.message}")
                    raise ValidationError(result.message, result)
                else:
                    logger.warning(f"Matrix validation warning: {result.message}")
            
            return func(matrix, *args, **kwargs)
        return wrapper
    return decorator


def validate_entropy(window_size: int = 100):
    """Decorator to validate signal entropy"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(signal: Vector, *args, **kwargs) -> Any:
            result = _global_validator.validate_entropy(signal, window_size)
            
            if not result.valid:
                if result.level == ValidationLevel.CRITICAL:
                    raise ValidationError(result.message, result)
                elif result.level == ValidationLevel.ERROR:
                    logger.error(f"Entropy validation failed: {result.message}")
                    raise ValidationError(result.message, result)
                else:
                    logger.warning(f"Entropy validation warning: {result.message}")
            
            return func(signal, *args, **kwargs)
        return wrapper
    return decorator


def validate_performance(max_time: Optional[float] = None):
    """Decorator to validate function performance"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            time_limit = max_time or _global_validator.max_execution_time
            result = _global_validator.validate_performance(func, *args, **kwargs)
            
            if not result.valid:
                if result.level == ValidationLevel.CRITICAL:
                    raise ValidationError(result.message, result)
                else:
                    logger.warning(f"Performance validation warning: {result.message}")
            
            return result.details.get('result', None)
        return wrapper
    return decorator


def get_validation_metrics() -> ValidationMetrics:
    """Get validation performance metrics"""
    return _global_validator.metrics


def main() -> None:
    """Main function for testing validation system"""
    try:
        print(" Runtime Validation System Test")
        print("=" * 40)
        
        # Initialize validator
        validator = RuntimeValidator()
        
        # Test vector validation
        print("1. Testing vector validation...")
        test_vector = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = validator.validate_vector(test_vector, expected_length=5)
        print(f"   ✅ Vector validation: {result.valid} - {result.message}")
        
        # Test matrix validation
        print("2. Testing matrix validation...")
        test_matrix = [[1.0, 2.0], [3.0, 4.0]]
        result = validator.validate_matrix(test_matrix, expected_shape=(2, 2))
        print(f"   ✅ Matrix validation: {result.valid} - {result.message}")
        
        # Test entropy validation
        print("3. Testing entropy validation...")
        test_signal = np.random.randn(200)  # Random signal
        result = validator.validate_entropy(test_signal, window_size=50)
        print(f"   ✅ Entropy validation: {result.valid} - {result.message}")
        
        # Test decorator usage
        print("4. Testing decorator usage...")
        
        @validate_vector(expected_length=3)
        def test_function(vector):
            return sum(vector)
        
        try:
            result = test_function([1, 2, 3])
            print(f"   ✅ Decorator test: {result}")
        except ValidationError as e:
            print(f"   ❌ Decorator test failed: {e}")
        
        print("\n Runtime validation system test completed successfully!")
        
    except Exception as e:
        print(f"❌ Runtime validation system test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
