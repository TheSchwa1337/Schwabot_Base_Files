# -*- coding: utf-8 -*-
"""
Error Handling Mathematical Foundations
======================================

This module provides mathematical foundations for error handling, including
error propagation models, fault correlation matrices, error recovery probability
calculations, and system resilience mathematical modeling.

Mathematical Foundations:
- Error propagation: E_propagated = E_initial * propagation_matrix
- Fault correlation: C_ij = Σ(E_i * E_j) / √(ΣE_i² * ΣE_j²)
- Recovery probability: P_recovery = 1 - exp(-λ * t) * resilience_factor
- System resilience: R = Σ(w_i * component_reliability_i) / Σw_i
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum

import numpy as np

# Import unified math system
try:
    from core.unified_math_system import unified_math
except ImportError:
    import math as unified_math

# Import Windows CLI compatibility
try:
    from core.utils.windows_cli_compatibility import (
        safe_print, info, warn, error, success, debug
    )
    CLI_COMPATIBILITY_AVAILABLE = True
except ImportError:
    CLI_COMPATIBILITY_AVAILABLE = False
    # Fallback functions
    def safe_print(message): print(message)
    def info(message): print(f"[INFO] {message}")
    def warn(message): print(f"[WARN] {message}")
    def error(message): print(f"[ERROR] {message}")
    def success(message): print(f"[SUCCESS] {message}")
    def debug(message): print(f"[DEBUG] {message}")

# Configure logging
logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels for mathematical modeling."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorType(Enum):
    """Error types for mathematical modeling."""
    NUMERICAL_OVERFLOW = "numerical_overflow"
    CONVERGENCE_FAILURE = "convergence_failure"
    THERMAL_ERROR = "thermal_error"
    MEMORY_ERROR = "memory_error"
    NETWORK_ERROR = "network_error"
    TIMEOUT_ERROR = "timeout_error"
    VALIDATION_ERROR = "validation_error"
    INTEGRATION_ERROR = "integration_error"


@dataclass
class ErrorMathematicalConfig:
    """Configuration for error mathematical foundations."""
    propagation_decay_rate: float = 0.1
    correlation_threshold: float = 0.7
    recovery_rate: float = 0.05
    resilience_weight_base: float = 1.0
    error_memory_size: int = 1000
    enable_error_propagation: bool = True
    enable_fault_correlation: bool = True
    enable_recovery_modeling: bool = True
    enable_resilience_calculation: bool = True


@dataclass
class ErrorPropagationResult:
    """Result of error propagation calculation."""
    propagated_errors: np.ndarray
    propagation_matrix: np.ndarray
    decay_factor: float
    affected_components: List[str]
    propagation_strength: float
    metadata: Dict[str, Any]


@dataclass
class FaultCorrelationResult:
    """Result of fault correlation calculation."""
    correlation_matrix: np.ndarray
    correlation_strength: float
    correlated_components: List[Tuple[str, str]]
    fault_clusters: List[List[str]]
    correlation_confidence: float
    metadata: Dict[str, Any]


@dataclass
class RecoveryProbabilityResult:
    """Result of recovery probability calculation."""
    recovery_probability: float
    recovery_time: float
    resilience_factor: float
    component_reliability: Dict[str, float]
    recovery_confidence: float
    metadata: Dict[str, Any]


@dataclass
class SystemResilienceResult:
    """Result of system resilience calculation."""
    overall_resilience: float
    component_resilience: Dict[str, float]
    resilience_weights: Dict[str, float]
    critical_paths: List[List[str]]
    resilience_confidence: float
    metadata: Dict[str, Any]


class ErrorMathematicalFoundations:
    """Mathematical foundations for error handling and system resilience."""

    def __init__(self, config: Optional[ErrorMathematicalConfig] = None):
        """Initialize error mathematical foundations."""
        self.config = config or ErrorMathematicalConfig()
        self.error_history: List[Dict[str, Any]] = []
        self.component_reliability: Dict[str, float] = {}
        self.error_propagation_matrix: Optional[np.ndarray] = None
        self.fault_correlation_matrix: Optional[np.ndarray] = None
        self.operation_count = 0

        # Initialize default component reliability
        self._initialize_component_reliability()

        logger.info("Error Mathematical Foundations initialized")

    def _initialize_component_reliability(self) -> None:
        """Initialize default component reliability scores."""
        default_components = [
            "tensor_algebra", "thermal_manager", "profit_router",
            "entropy_compensator", "hash_encoder", "bit_phase_processor",
            "memory_vault", "fault_bus", "config_manager", "validation_engine"
        ]

        for component in default_components:
            # 95% default reliability
            self.component_reliability[component] = 0.95

    def calculate_error_propagation(self,
                                    initial_errors: Dict[str,
                                                         float],
                                    component_network: Dict[str,
                                                            List[str]]) -> ErrorPropagationResult:
        """
        Calculate error propagation through component network.

        Mathematical: E_propagated = E_initial * propagation_matrix
        """
        try:
            self.operation_count += 1

            # Create component list
            components = list(component_network.keys())
            n_components = len(components)

            # Create initial error vector
            initial_error_vector = np.zeros(n_components)
            for i, component in enumerate(components):
                initial_error_vector[i] = initial_errors.get(component, 0.0)

            # Create propagation matrix
            propagation_matrix = np.zeros((n_components, n_components))

            for i, component in enumerate(components):
                connected_components = component_network[component]
                for connected in connected_components:
                    if connected in components:
                        j = components.index(connected)
                        # Propagation strength based on component reliability
                        reliability = self.component_reliability.get(
                            component, 0.95)
                        propagation_strength = (
                            1.0 - reliability) * self.config.propagation_decay_rate
                        propagation_matrix[i, j] = propagation_strength

            # Apply propagation
            propagated_errors = np.dot(
                propagation_matrix, initial_error_vector)

            # Calculate decay factor
            decay_factor = np.exp(-self.config.propagation_decay_rate *
                                  np.sum(initial_error_vector))

            # Find affected components
            affected_components = [
                components[i] for i in range(n_components)
                if propagated_errors[i] > 0.01
            ]

            # Calculate propagation strength
            propagation_strength = np.sum(propagated_errors) / np.sum(
                initial_error_vector) if np.sum(initial_error_vector) > 0 else 0.0

            return ErrorPropagationResult(
                propagated_errors=propagated_errors,
                propagation_matrix=propagation_matrix,
                decay_factor=decay_factor,
                affected_components=affected_components,
                propagation_strength=propagation_strength,
                metadata={
                    "components": components,
                    "initial_errors": initial_errors,
                    "operation_count": self.operation_count
                }
            )

        except Exception as e:
            logger.error(f"Error propagation calculation failed: {e}")
            return self._create_fallback_propagation_result()

    def calculate_fault_correlation(
            self, error_history: List[Dict[str, Any]]) -> FaultCorrelationResult:
        """
        Calculate fault correlation matrix from error history.

        Mathematical: C_ij = Σ(E_i * E_j) / √(ΣE_i² * ΣE_j²)
        """
        try:
            self.operation_count += 1

            if not error_history:
                return self._create_fallback_correlation_result()

            # Extract error components and their occurrences
            component_errors = {}
            for error in error_history:
                component = error.get('component', 'unknown')
                severity = error.get('severity', 'medium')
                severity_value = self._severity_to_value(severity)

                if component not in component_errors:
                    component_errors[component] = []
                component_errors[component].append(severity_value)

            # Create component list
            components = list(component_errors.keys())
            n_components = len(components)

            if n_components == 0:
                return self._create_fallback_correlation_result()

            # Create correlation matrix
            correlation_matrix = np.zeros((n_components, n_components))

            for i in range(n_components):
                for j in range(n_components):
                    if i == j:
                        correlation_matrix[i, j] = 1.0  # Self-correlation
                    else:
                        # Calculate correlation between components i and j
                        errors_i = component_errors[components[i]]
                        errors_j = component_errors[components[j]]

                        # Pad shorter list with zeros
                        max_len = max(len(errors_i), len(errors_j))
                        errors_i_padded = errors_i + \
                            [0.0] * (max_len - len(errors_i))
                        errors_j_padded = errors_j + \
                            [0.0] * (max_len - len(errors_j))

                        # Calculate correlation
                        correlation = self._calculate_correlation(
                            errors_i_padded, errors_j_padded)
                        correlation_matrix[i, j] = correlation

            # Find correlated components
            correlated_components = []
            for i in range(n_components):
                for j in range(i + 1, n_components):
                    if correlation_matrix[i,
                                          j] > self.config.correlation_threshold:
                        correlated_components.append(
                            (components[i], components[j]))

            # Find fault clusters
            fault_clusters = self._find_fault_clusters(
                correlation_matrix, components)

            # Calculate correlation strength
            correlation_strength = np.mean(
                correlation_matrix[np.triu_indices(n_components, k=1)])

            # Calculate correlation confidence
            correlation_confidence = min(1.0, len(error_history) / 100.0)

            return FaultCorrelationResult(
                correlation_matrix=correlation_matrix,
                correlation_strength=correlation_strength,
                correlated_components=correlated_components,
                fault_clusters=fault_clusters,
                correlation_confidence=correlation_confidence,
                metadata={
                    "components": components,
                    "error_history_size": len(error_history),
                    "operation_count": self.operation_count
                }
            )

        except Exception as e:
            logger.error(f"Fault correlation calculation failed: {e}")
            return self._create_fallback_correlation_result()

    def calculate_recovery_probability(
            self,
            error_type: ErrorType,
            component: str,
            time_elapsed: float) -> RecoveryProbabilityResult:
        """
        Calculate recovery probability for a given error.

        Mathematical: P_recovery = 1 - exp(-λ * t) * resilience_factor
        """
        try:
            self.operation_count += 1

            # Get component reliability
            component_reliability = self.component_reliability.get(
                component, 0.95)

            # Calculate recovery rate based on error type and component
            base_recovery_rate = self.config.recovery_rate
            error_type_multiplier = self._get_error_type_multiplier(error_type)
            component_multiplier = component_reliability

            recovery_rate = base_recovery_rate * error_type_multiplier * component_multiplier

            # Calculate recovery probability
            recovery_probability = 1.0 - np.exp(-recovery_rate * time_elapsed)

            # Calculate resilience factor
            resilience_factor = component_reliability * \
                (1.0 - np.exp(-time_elapsed / 60.0))

            # Calculate recovery time (time to 95% recovery probability)
            recovery_time = - \
                np.log(0.05) / recovery_rate if recovery_rate > 0 else float('inf')

            # Calculate recovery confidence
            # 5 minutes for full confidence
            recovery_confidence = min(1.0, time_elapsed / 300.0)

            return RecoveryProbabilityResult(
                recovery_probability=recovery_probability,
                recovery_time=recovery_time,
                resilience_factor=resilience_factor,
                component_reliability={component: component_reliability},
                recovery_confidence=recovery_confidence,
                metadata={
                    "error_type": error_type.value,
                    "component": component,
                    "time_elapsed": time_elapsed,
                    "operation_count": self.operation_count
                }
            )

        except Exception as e:
            logger.error(f"Recovery probability calculation failed: {e}")
            return self._create_fallback_recovery_result()

    def calculate_system_resilience(self,
                                    component_network: Dict[str,
                                                            List[str]]) -> SystemResilienceResult:
        """
        Calculate overall system resilience.

        Mathematical: R = Σ(w_i * component_reliability_i) / Σw_i
        """
        try:
            self.operation_count += 1

            components = list(component_network.keys())

            if not components:
                return self._create_fallback_resilience_result()

            # Calculate component resilience weights
            resilience_weights = {}
            component_resilience = {}

            for component in components:
                # Weight based on number of connections (more connections =
                # higher weight)
                connections = len(component_network[component])
                weight = self.config.resilience_weight_base * \
                    (1.0 + connections * 0.1)
                resilience_weights[component] = weight

                # Component resilience based on reliability
                reliability = self.component_reliability.get(component, 0.95)
                component_resilience[component] = reliability

            # Calculate overall resilience
            total_weight = sum(resilience_weights.values())
            weighted_resilience = sum(
                resilience_weights[comp] * component_resilience[comp]
                for comp in components
            )

            overall_resilience = weighted_resilience / \
                total_weight if total_weight > 0 else 0.0

            # Find critical paths (components with high weight and low
            # reliability)
            critical_paths = []
            for component in components:
                if (
                        resilience_weights[component] > self.config.resilience_weight_base *
                        1.5 and component_resilience[component] < 0.8):
                    critical_paths.append([component])

            # Calculate resilience confidence
            resilience_confidence = min(1.0, len(components) / 20.0)

            return SystemResilienceResult(
                overall_resilience=overall_resilience,
                component_resilience=component_resilience,
                resilience_weights=resilience_weights,
                critical_paths=critical_paths,
                resilience_confidence=resilience_confidence,
                metadata={
                    "components": components,
                    "total_components": len(components),
                    "operation_count": self.operation_count
                }
            )

        except Exception as e:
            logger.error(f"System resilience calculation failed: {e}")
            return self._create_fallback_resilience_result()

    def _severity_to_value(self, severity: str) -> float:
        """Convert severity string to numerical value."""
        severity_map = {
            "low": 0.25,
            "medium": 0.5,
            "high": 0.75,
            "critical": 1.0
        }
        return severity_map.get(severity.lower(), 0.5)

    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate correlation coefficient between two lists."""
        try:
            if not x or not y or len(x) != len(y):
                return 0.0

            x_array = np.array(x)
            y_array = np.array(y)

            # Calculate correlation
            correlation = np.corrcoef(x_array, y_array)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0

        except Exception:
            return 0.0

    def _find_fault_clusters(self,
                             correlation_matrix: np.ndarray,
                             components: List[str]) -> List[List[str]]:
        """Find fault clusters based on correlation matrix."""
        try:
            clusters = []
            n_components = len(components)
            visited = [False] * n_components

            for i in range(n_components):
                if not visited[i]:
                    cluster = [components[i]]
                    visited[i] = True

                    # Find all components correlated with this one
                    for j in range(n_components):
                        if not visited[j] and correlation_matrix[i,
                                                                 j] > self.config.correlation_threshold:
                            cluster.append(components[j])
                            visited[j] = True

                    if len(
                            cluster) > 1:  # Only include clusters with multiple components
                        clusters.append(cluster)

            return clusters

        except Exception:
            return []

    def _get_error_type_multiplier(self, error_type: ErrorType) -> float:
        """Get recovery rate multiplier for error type."""
        multipliers = {
            ErrorType.NUMERICAL_OVERFLOW: 1.5,
            ErrorType.CONVERGENCE_FAILURE: 1.2,
            ErrorType.THERMAL_ERROR: 0.8,
            ErrorType.MEMORY_ERROR: 0.6,
            ErrorType.NETWORK_ERROR: 1.0,
            ErrorType.TIMEOUT_ERROR: 1.1,
            ErrorType.VALIDATION_ERROR: 1.3,
            ErrorType.INTEGRATION_ERROR: 0.9
        }
        return multipliers.get(error_type, 1.0)

    def _create_fallback_propagation_result(self) -> ErrorPropagationResult:
        """Create fallback error propagation result."""
        return ErrorPropagationResult(
            propagated_errors=np.array([0.0]),
            propagation_matrix=np.array([[0.0]]),
            decay_factor=1.0,
            affected_components=[],
            propagation_strength=0.0,
            metadata={"fallback": True}
        )

    def _create_fallback_correlation_result(self) -> FaultCorrelationResult:
        """Create fallback fault correlation result."""
        return FaultCorrelationResult(
            correlation_matrix=np.array([[1.0]]),
            correlation_strength=0.0,
            correlated_components=[],
            fault_clusters=[],
            correlation_confidence=0.0,
            metadata={"fallback": True}
        )

    def _create_fallback_recovery_result(self) -> RecoveryProbabilityResult:
        """Create fallback recovery probability result."""
        return RecoveryProbabilityResult(
            recovery_probability=0.5,
            recovery_time=60.0,
            resilience_factor=0.5,
            component_reliability={},
            recovery_confidence=0.0,
            metadata={"fallback": True}
        )

    def _create_fallback_resilience_result(self) -> SystemResilienceResult:
        """Create fallback system resilience result."""
        return SystemResilienceResult(
            overall_resilience=0.5,
            component_resilience={},
            resilience_weights={},
            critical_paths=[],
            resilience_confidence=0.0,
            metadata={"fallback": True}
        )

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error mathematical statistics."""
        return {
            "total_operations": self.operation_count,
            "component_reliability": self.component_reliability,
            "error_history_size": len(
                self.error_history),
            "propagation_matrix_available": self.error_propagation_matrix is not None,
            "correlation_matrix_available": self.fault_correlation_matrix is not None}

    def reset_error_statistics(self) -> None:
        """Reset error mathematical statistics."""
        self.operation_count = 0
        self.error_history.clear()
        logger.info("Error mathematical statistics reset")


# Global error mathematical foundations instance
_error_math_instance: Optional[ErrorMathematicalFoundations] = None


def get_error_mathematical_foundations(
        config: Optional[ErrorMathematicalConfig] = None) -> ErrorMathematicalFoundations:
    """Get global error mathematical foundations instance."""
    global _error_math_instance
    if _error_math_instance is None:
        _error_math_instance = ErrorMathematicalFoundations(config)
    return _error_math_instance


def main():
    """Main function for testing error mathematical foundations."""
    try:
        # Create error mathematical foundations
        error_math = get_error_mathematical_foundations()

        # Test error propagation
        initial_errors = {"tensor_algebra": 0.3, "thermal_manager": 0.2}
        component_network = {
            "tensor_algebra": ["thermal_manager", "profit_router"],
            "thermal_manager": ["tensor_algebra", "entropy_compensator"],
            "profit_router": ["tensor_algebra"],
            "entropy_compensator": ["thermal_manager"]
        }
        prop_result = error_math.calculate_error_propagation(
            initial_errors, component_network)
        print(
            f"Error propagation: strength={
                prop_result.propagation_strength:.3f}, affected={
                len(
                    prop_result.affected_components)}")

        # Test fault correlation
        error_history = [
            {"component": "tensor_algebra", "severity": "medium"},
            {"component": "thermal_manager", "severity": "high"},
            {"component": "tensor_algebra", "severity": "low"}
        ]
        corr_result = error_math.calculate_fault_correlation(error_history)
        print(
            f"Fault correlation: strength={
                corr_result.correlation_strength:.3f}, clusters={
                len(
                    corr_result.fault_clusters)}")

        # Test recovery probability
        recovery_result = error_math.calculate_recovery_probability(
            ErrorType.THERMAL_ERROR, "thermal_manager", 30.0
        )
        print(
            f"Recovery probability: {
                recovery_result.recovery_probability:.3f}, time={
                recovery_result.recovery_time:.1f}s")

        # Test system resilience
        resilience_result = error_math.calculate_system_resilience(
            component_network)
        print(
            f"System resilience: {
                resilience_result.overall_resilience:.3f}, critical_paths={
                len(
                    resilience_result.critical_paths)}")

        # Get statistics
        stats = error_math.get_error_statistics()
        print(f"Error statistics: {stats}")

    except Exception as e:
        logger.error(f"Error mathematical foundations test failed: {e}")


if __name__ == "__main__":
    main()
