# -*- coding: utf-8 -*-
"""
Thermal-Adaptive Mathematical Integration
========================================

This module provides thermal-aware mathematical operations that dynamically
adjust precision, performance, and resource allocation based on thermal state.

Mathematical Foundations:
- Thermal-adaptive tensor operations: T_thermal = T_base * f(thermal_state)
- Heat dissipation modeling: Q = k * A * ΔT / d
- Temperature-aware profit calculations: P_thermal = P_base * efficiency_factor
- Thermal state transition models: dT/dt = α * (T_target - T_current)
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

# Import tensor algebra
try:
    from core.math.tensor_algebra import UnifiedTensorAlgebra, BitPhaseResult
    TENSOR_ALGEBRA_AVAILABLE = True
except ImportError:
    TENSOR_ALGEBRA_AVAILABLE = False

# Import thermal boundary manager
try:
    from core.thermal_boundary_manager import ThermalBoundaryManager, ThermalState
    THERMAL_MANAGER_AVAILABLE = True
except ImportError:
    THERMAL_MANAGER_AVAILABLE = False
    # Fallback ThermalState enum
    class ThermalState(Enum):
        """Fallback thermal state enum."""
        NORMAL = "normal"
        WARNING = "warning"
        CRITICAL = "critical"
        EMERGENCY = "emergency"

# Import Windows CLI compatibility
try:
    from core.utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug
    CLI_COMPATIBILITY_AVAILABLE = True
except ImportError:
    CLI_COMPATIBILITY_AVAILABLE = False
    # Fallback functions
    def safe_print(message): 
        print(message)
    
    def info(message): 
        print(f"[INFO] {message}")
    
    def warn(message): 
        print(f"[WARN] {message}")
    
    def error(message): 
        print(f"[ERROR] {message}")
    
    def success(message): 
        print(f"[SUCCESS] {message}")
    
    def debug(message): 
        print(f"[DEBUG] {message}")

# Configure logging
logger = logging.getLogger(__name__)


class ThermalAdaptationMode(Enum):
    """Thermal adaptation modes for mathematical operations."""
    CONSERVATIVE = "conservative"  # Reduce precision and performance
    BALANCED = "balanced"         # Moderate adjustments
    AGGRESSIVE = "aggressive"     # Maintain performance with thermal monitoring
    EMERGENCY = "emergency"       # Minimal operations only


@dataclass
class ThermalMathematicalConfig:
    """Configuration for thermal-adaptive mathematical operations."""
    base_precision: np.dtype = np.float64
    thermal_scaling_factor: float = 0.8
    heat_dissipation_coefficient: float = 0.1
    thermal_transition_rate: float = 0.05
    max_thermal_threshold: float = 85.0
    min_thermal_threshold: float = 30.0
    adaptation_mode: ThermalAdaptationMode = ThermalAdaptationMode.BALANCED
    enable_thermal_monitoring: bool = True
    enable_adaptive_precision: bool = True
    enable_heat_modeling: bool = True


@dataclass
class ThermalMathematicalResult:
    """Result of thermal-adaptive mathematical operation."""
    result: Any
    thermal_state: str
    adaptation_factor: float
    precision_used: np.dtype
    heat_generated: float
    execution_time: float
    thermal_efficiency: float
    metadata: Dict[str, Any]


class ThermalMathematicalIntegration:
    """Thermal-adaptive mathematical integration system."""
    
    def __init__(self, config: Optional[ThermalMathematicalConfig] = None):
        """Initialize thermal mathematical integration."""
        self.config = config or ThermalMathematicalConfig()
        self.thermal_manager = None
        self.tensor_algebra = None
        self.current_thermal_state = ThermalState.NORMAL
        self.thermal_history: List[Dict[str, Any]] = []
        self.operation_count = 0
        self.heat_accumulation = 0.0
        
        # Initialize components
        self._initialize_components()
        
        logger.info("Thermal Mathematical Integration initialized")
    
    def _initialize_components(self) -> None:
        """Initialize thermal and mathematical components."""
        try:
            # Initialize thermal manager
            if THERMAL_MANAGER_AVAILABLE:
                self.thermal_manager = ThermalBoundaryManager()
                logger.info("Thermal manager integrated")
            
            # Initialize tensor algebra
            if TENSOR_ALGEBRA_AVAILABLE:
                self.tensor_algebra = UnifiedTensorAlgebra()
                logger.info("Tensor algebra integrated")
                
        except Exception as e:
            logger.warning(f"Component initialization failed: {e}")
    
    def get_thermal_adaptation_factor(self) -> float:
        """
        Calculate thermal adaptation factor based on current thermal state.
        
        Mathematical: α = 1 - (T_current - T_optimal) / (T_max - T_optimal)
        """
        try:
            if not self.thermal_manager:
                return 1.0
            
            # Get current thermal state
            thermal_data = self.thermal_manager.get_thermal_state()
            current_temp = thermal_data.get('cpu_temperature', 50.0)
            
            # Calculate adaptation factor
            optimal_temp = 50.0
            max_temp = self.config.max_thermal_threshold
            
            if current_temp <= optimal_temp:
                adaptation_factor = 1.0
            elif current_temp >= max_temp:
                adaptation_factor = 0.1  # Emergency mode
            else:
                adaptation_factor = 1.0 - (current_temp - optimal_temp) / (max_temp - optimal_temp)
                adaptation_factor = max(0.1, adaptation_factor)
            
            return adaptation_factor
            
        except Exception as e:
            logger.error(f"Thermal adaptation factor calculation failed: {e}")
            return 0.5  # Safe fallback
    
    def calculate_heat_dissipation(self, operation_complexity: float, 
                                 execution_time: float) -> float:
        """
        Calculate heat dissipation for mathematical operations.
        
        Mathematical: Q = k * A * ΔT / d * complexity * time
        """
        try:
            # Heat dissipation formula
            k = self.config.heat_dissipation_coefficient
            complexity_factor = operation_complexity
            time_factor = execution_time
            
            # Base heat generation
            base_heat = k * complexity_factor * time_factor
            
            # Apply thermal state adjustment
            thermal_factor = self.get_thermal_adaptation_factor()
            heat_generated = base_heat * (1.0 - thermal_factor * 0.5)
            
            # Update heat accumulation
            self.heat_accumulation += heat_generated
            
            return heat_generated
            
        except Exception as e:
            logger.error(f"Heat dissipation calculation failed: {e}")
            return 0.0
    
    def get_adaptive_precision(self) -> np.dtype:
        """Get adaptive precision based on thermal state."""
        try:
            adaptation_factor = self.get_thermal_adaptation_factor()
            
            if adaptation_factor > 0.8:
                return np.float64  # Full precision
            elif adaptation_factor > 0.5:
                return np.float32  # Reduced precision
            else:
                return np.float16  # Minimal precision
                
        except Exception as e:
            logger.error(f"Adaptive precision calculation failed: {e}")
            return np.float32  # Safe fallback
    
    def thermal_tensor_contraction(self, A: np.ndarray, B: np.ndarray,
                                 axes: Union[int, List[int]] = 1) -> ThermalMathematicalResult:
        """
        Perform thermal-adaptive tensor contraction.
        
        Mathematical: T_thermal = T_base * α * f(thermal_state)
        """
        start_time = time.time()
        
        try:
            # Get thermal adaptation factor
            adaptation_factor = self.get_thermal_adaptation_factor()
            
            # Get adaptive precision
            precision = self.get_adaptive_precision()
            
            # Perform base tensor contraction
            if self.tensor_algebra:
                base_result = self.tensor_algebra.tensor_contraction(A, B, axes)
            else:
                base_result = np.tensordot(A, B, axes=axes)
            
            # Apply thermal adaptation
            thermal_result = base_result * adaptation_factor
            
            # Convert to adaptive precision
            thermal_result = thermal_result.astype(precision)
            
            # Calculate execution time and heat
            execution_time = time.time() - start_time
            operation_complexity = A.size * B.size / 1000000  # Normalized complexity
            heat_generated = self.calculate_heat_dissipation(operation_complexity, execution_time)
            
            # Calculate thermal efficiency
            thermal_efficiency = adaptation_factor * (1.0 - heat_generated / 100.0)
            thermal_efficiency = max(0.0, min(1.0, thermal_efficiency))
            
            # Update operation count
            self.operation_count += 1
            
            return ThermalMathematicalResult(
                result=thermal_result,
                thermal_state=self.current_thermal_state.value,
                adaptation_factor=adaptation_factor,
                precision_used=precision,
                heat_generated=heat_generated,
                execution_time=execution_time,
                thermal_efficiency=thermal_efficiency,
                metadata={
                    "operation_type": "tensor_contraction",
                    "input_shapes": [A.shape, B.shape],
                    "output_shape": thermal_result.shape,
                    "operation_count": self.operation_count
                }
            )
            
        except Exception as e:
            logger.error(f"Thermal tensor contraction failed: {e}")
            return self._create_fallback_result("tensor_contraction", e)
    
    def thermal_profit_calculation(self, profit_data: np.ndarray,
                                 routing_weights: np.ndarray) -> ThermalMathematicalResult:
        """
        Perform thermal-adaptive profit calculation.
        
        Mathematical: P_thermal = P_base * efficiency_factor * thermal_adaptation
        """
        start_time = time.time()
        
        try:
            # Get thermal adaptation factor
            adaptation_factor = self.get_thermal_adaptation_factor()
            
            # Get adaptive precision
            precision = self.get_adaptive_precision()
            
            # Perform base profit calculation
            if self.tensor_algebra:
                base_result = self.tensor_algebra.profit_routing_tensor(profit_data, routing_weights)
            else:
                # Fallback calculation
                if profit_data.ndim == 1:
                    profit_data = profit_data.reshape(-1, 1)
                if routing_weights.ndim == 1:
                    routing_weights = routing_weights.reshape(1, -1)
                base_result = np.dot(routing_weights, profit_data)
            
            # Apply thermal adaptation
            thermal_result = base_result * adaptation_factor
            
            # Convert to adaptive precision
            thermal_result = thermal_result.astype(precision)
            
            # Calculate execution time and heat
            execution_time = time.time() - start_time
            operation_complexity = profit_data.size * routing_weights.size / 1000000
            heat_generated = self.calculate_heat_dissipation(operation_complexity, execution_time)
            
            # Calculate thermal efficiency
            thermal_efficiency = adaptation_factor * (1.0 - heat_generated / 50.0)
            thermal_efficiency = max(0.0, min(1.0, thermal_efficiency))
            
            # Update operation count
            self.operation_count += 1
            
            return ThermalMathematicalResult(
                result=thermal_result,
                thermal_state=self.current_thermal_state.value,
                adaptation_factor=adaptation_factor,
                precision_used=precision,
                heat_generated=heat_generated,
                execution_time=execution_time,
                thermal_efficiency=thermal_efficiency,
                metadata={
                    "operation_type": "profit_calculation",
                    "profit_data_shape": profit_data.shape,
                    "weights_shape": routing_weights.shape,
                    "operation_count": self.operation_count
                }
            )
            
        except Exception as e:
            logger.error(f"Thermal profit calculation failed: {e}")
            return self._create_fallback_result("profit_calculation", e)
    
    def thermal_entropy_compensation(self, data: np.ndarray,
                                   compensation_factor: float = 1.0) -> ThermalMathematicalResult:
        """
        Perform thermal-adaptive entropy compensation.
        
        Mathematical: E_thermal = E_base * thermal_efficiency + heat_compensation
        """
        start_time = time.time()
        
        try:
            # Get thermal adaptation factor
            adaptation_factor = self.get_thermal_adaptation_factor()
            
            # Get adaptive precision
            precision = self.get_adaptive_precision()
            
            # Perform base entropy compensation
            if self.tensor_algebra:
                base_result = self.tensor_algebra.entropy_compensation(data, compensation_factor)
            else:
                # Fallback calculation
                if data.size == 0:
                    base_result = data
                else:
                    data_norm = data / (np.max(np.abs(data)) + 1e-12)
                    gradient = np.gradient(data_norm)
                    gradient_magnitude = np.sqrt(np.sum(gradient**2, axis=0))
                    compensation = compensation_factor * np.log(1 + gradient_magnitude)
                    base_result = data_norm + compensation
            
            # Apply thermal adaptation
            thermal_result = base_result * adaptation_factor
            
            # Convert to adaptive precision
            thermal_result = thermal_result.astype(precision)
            
            # Calculate execution time and heat
            execution_time = time.time() - start_time
            operation_complexity = data.size / 1000000
            heat_generated = self.calculate_heat_dissipation(operation_complexity, execution_time)
            
            # Calculate thermal efficiency
            thermal_efficiency = adaptation_factor * (1.0 - heat_generated / 30.0)
            thermal_efficiency = max(0.0, min(1.0, thermal_efficiency))
            
            # Update operation count
            self.operation_count += 1
            
            return ThermalMathematicalResult(
                result=thermal_result,
                thermal_state=self.current_thermal_state.value,
                adaptation_factor=adaptation_factor,
                precision_used=precision,
                heat_generated=heat_generated,
                execution_time=execution_time,
                thermal_efficiency=thermal_efficiency,
                metadata={
                    "operation_type": "entropy_compensation",
                    "data_shape": data.shape,
                    "compensation_factor": compensation_factor,
                    "operation_count": self.operation_count
                }
            )
            
        except Exception as e:
            logger.error(f"Thermal entropy compensation failed: {e}")
            return self._create_fallback_result("entropy_compensation", e)
    
    def _create_fallback_result(self, operation_type: str, error: Exception) -> ThermalMathematicalResult:
        """Create fallback result for error cases."""
        return ThermalMathematicalResult(
            result=np.array([0.0]),
            thermal_state="unknown",
            adaptation_factor=0.5,
            precision_used=np.float32,
            heat_generated=0.0,
            execution_time=0.0,
            thermal_efficiency=0.0,
            metadata={
                "operation_type": operation_type,
                "error": str(error),
                "fallback": True
            }
        )
    
    def get_thermal_statistics(self) -> Dict[str, Any]:
        """Get thermal mathematical statistics."""
        return {
            "total_operations": self.operation_count,
            "current_thermal_state": self.current_thermal_state.value,
            "heat_accumulation": self.heat_accumulation,
            "thermal_history_size": len(self.thermal_history),
            "adaptation_factor": self.get_thermal_adaptation_factor(),
            "adaptive_precision": str(self.get_adaptive_precision()),
            "thermal_manager_available": THERMAL_MANAGER_AVAILABLE,
            "tensor_algebra_available": TENSOR_ALGEBRA_AVAILABLE
        }
    
    def reset_thermal_statistics(self) -> None:
        """Reset thermal mathematical statistics."""
        self.operation_count = 0
        self.heat_accumulation = 0.0
        self.thermal_history.clear()
        logger.info("Thermal mathematical statistics reset")


# Global thermal mathematical integration instance
_thermal_math_instance: Optional[ThermalMathematicalIntegration] = None


def get_thermal_mathematical_integration(config: Optional[ThermalMathematicalConfig] = None) -> ThermalMathematicalIntegration:
    """Get global thermal mathematical integration instance."""
    global _thermal_math_instance
    if _thermal_math_instance is None:
        _thermal_math_instance = ThermalMathematicalIntegration(config)
    return _thermal_math_instance


def main():
    """Main function for testing thermal mathematical integration."""
    try:
        # Create thermal mathematical integration
        thermal_math = get_thermal_mathematical_integration()
        
        # Test thermal tensor contraction
        A = np.random.rand(3, 4).astype(np.float64)
        B = np.random.rand(4, 2).astype(np.float64)
        result1 = thermal_math.thermal_tensor_contraction(A, B)
        print(f"Thermal tensor contraction: shape={result1.result.shape}, efficiency={result1.thermal_efficiency:.3f}")
        
        # Test thermal profit calculation
        profit_data = np.random.rand(5, 3).astype(np.float64)
        weights = np.random.rand(1, 5).astype(np.float64)
        result2 = thermal_math.thermal_profit_calculation(profit_data, weights)
        print(f"Thermal profit calculation: shape={result2.result.shape}, efficiency={result2.thermal_efficiency:.3f}")
        
        # Test thermal entropy compensation
        data = np.random.rand(100).astype(np.float64)
        result3 = thermal_math.thermal_entropy_compensation(data)
        print(f"Thermal entropy compensation: shape={result3.result.shape}, efficiency={result3.thermal_efficiency:.3f}")
        
        # Get statistics
        stats = thermal_math.get_thermal_statistics()
        print(f"Thermal statistics: {stats}")
        
    except Exception as e:
        logger.error(f"Thermal mathematical integration test failed: {e}")


if __name__ == "__main__":
    main() 