#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
💰 UNIFIED PROFIT VECTORIZATION SYSTEM - SCHWABOT PROFIT OPTIMIZATION ENGINE
===========================================================================

A comprehensive system that unifies all profit vectorization components into a single,
cohesive interface for the Schwabot trading system with advanced mathematical foundations.

Mathematical Foundation:
- Profit Vectorization: P = Σ(w_i * v_i) where w_i are weights, v_i are vectors
- Integration Modes: IM = {unified, weighted, consensus, adaptive, hierarchical, orbital}
- Vectorization Strategy: VS = f(market_data, thermal_state, bit_phase, consensus)
- Thermal State Analysis: TS = f(volatility, trend_strength, entropy_level)
- Bit Phase Calculation: BP = f(quantum_state, computational_density, energy_level)
- Orbital Consensus: OC = f(shell_consensus, altitude_vector, orbital_parameters)
- Performance Metrics: PM = {calculation_time, accuracy, confidence, throughput}
- Fallback Matrix: FM = {hold_vector, minimal_buy, conservative} for error recovery
- Quantum Profit Optimization: QPO = hν / kT * profit_base for quantum effects
- Entropy-Based Vectorization: EV = -kΣp_i * ln(p_i) for information entropy
- Fractal Profit Scaling: FPS = α * P_prev + (1-α) * P_current where α is decay factor
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Import existing Schwabot components
try:
    from .quantum_mathematical_bridge import QuantumMathematicalBridge
    from .orbital_shell_brain_system import OrbitalBRAINSystem, ShellConsensus, AltitudeVector
    from .entropy_math import EntropyMathSystem
    from .advanced_tensor_algebra import AdvancedTensorAlgebra
    SCHWABOT_COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Some Schwabot components not available: {e}")
    SCHWABOT_COMPONENTS_AVAILABLE = False
    
    # Create fallback classes
    class ShellConsensus:
        def __init__(self):
            self.consensus_score = 0.5
            self.active_shells = []
            self.shell_activations = {}
            self.shell_confidences = {}
            self.shell_weights = {}
            self.threshold_met = False
    
    class AltitudeVector:
        def __init__(self):
            self.momentum_curvature = 0.0
            self.rolling_return = 0.0
            self.entropy_shift = 0.0
            self.alpha_decay = 0.0
            self.altitude_value = 0.5
            self.confidence_level = 0.5

# CUDA Integration with Fallback
try:
    import cupy as cp
    USING_CUDA = True
    _backend = "cupy (GPU)"
    xp = cp
except ImportError:
    USING_CUDA = False
    _backend = "numpy (CPU)"
    xp = np

logger = logging.getLogger(__name__)
if USING_CUDA:
    logger.info(f"⚡ UnifiedProfitVectorizationSystem using GPU acceleration: {_backend}")
else:
    logger.info(f"🔄 UnifiedProfitVectorizationSystem using CPU fallback: {_backend}")

__all__ = [
    "UnifiedProfitVectorizationSystem",
    "ProfitIntegrationMode",
    "VectorizationStrategy",
    "UnifiedProfitResult",
    "TriStateFallbackMatrix",
]


class ProfitIntegrationMode(Enum):
    """Modes for integrating different profit calculation systems with mathematical strategies."""
    UNIFIED = "unified"  # Single unified calculation
    WEIGHTED = "weighted"  # Weighted combination of systems
    CONSENSUS = "consensus"  # Consensus-based integration
    ADAPTIVE = "adaptive"  # Adaptive mode selection
    HIERARCHICAL = "hierarchical"  # Hierarchical decision tree
    ORBITAL_CONSENSUS = "orbital_consensus"  # Orbital shell consensus


class VectorizationStrategy(Enum):
    """Strategies for profit vectorization with mathematical optimization."""
    STANDARD = "standard"  # Standard vectorization
    ENHANCED = "enhanced"  # Enhanced with additional features
    OPTIMIZED = "optimized"  # Optimized for performance
    REAL_TIME = "real_time"  # Real-time processing
    BATCH = "batch"  # Batch processing mode


@dataclass
class ProfitVector:
    """Profit vector with components and metadata."""
    buy_signal: float
    sell_signal: float
    hold_signal: float
    confidence: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ThermalState:
    """Thermal state for market analysis."""
    volatility: float
    trend_strength: float
    entropy_level: float
    temperature: float
    timestamp: float


@dataclass
class BitPhase:
    """Bit phase for quantum-inspired calculations."""
    quantum_state: complex
    computational_density: float
    energy_level: float
    phase_angle: float
    timestamp: float


@dataclass
class UnifiedProfitResult:
    """Result from unified profit vectorization system with comprehensive metrics."""
    timestamp: float
    profit_value: float
    confidence: float
    vector: ProfitVector
    integration_mode: ProfitIntegrationMode
    strategy: VectorizationStrategy
    metadata: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)


class TriStateFallbackMatrix:
    """
    Provides structured fallback vectors for different failure scenarios.

    Mathematical Foundation:
    - Hold Vector: [0, 0, 0] for complete market neutrality
    - Minimal Buy Vector: [0, 1, 0] for conservative buy signal
    - Conservative Vector: [0.1, 0.1, 0.1] for balanced approach
    - Error Classification: EC = f(error_type, severity, context)
    - Risk Minimization: RM = min(Σ|position_i|) for all i
    - Diversification: D = Σ(position_i) / n where n = number of positions
    """

    @staticmethod
    def get_hold_vector() -> np.ndarray:
        """
        Return [0, 0, 0] for complete hold state.

        Mathematical Foundation:
        - Neutral Position: NP = [0, 0, 0] represents no market exposure
        - Risk Minimization: RM = min(Σ|position_i|) for all i

        Returns:
            Hold vector as numpy array
        """
        return np.array([0.0, 0.0, 0.0])

    @staticmethod
    def get_minimal_buy_vector() -> np.ndarray:
        """
        Return [0, 1, 0] for minimal buy signal.

        Mathematical Foundation:
        - Minimal Exposure: ME = [0, 1, 0] represents minimal buy position
        - Risk Control: RC = position_size * risk_factor where risk_factor = 1.0

        Returns:
            Minimal buy vector as numpy array
        """
        return np.array([0.0, 1.0, 0.0])

    @staticmethod
    def get_conservative_vector() -> np.ndarray:
        """
        Return [0.1, 0.1, 0.1] for conservative approach.

        Mathematical Foundation:
        - Conservative Position: CP = [0.1, 0.1, 0.1] represents balanced exposure
        - Diversification: D = Σ(position_i) / n where n = number of positions

        Returns:
            Conservative vector as numpy array
        """
        return np.array([0.1, 0.1, 0.1])

    @staticmethod
    def get_fallback_for_error(error_type: str) -> np.ndarray:
        """
        Get appropriate fallback based on error type with mathematical classification.

        Mathematical Foundation:
        - Error Classification: EC = f(error_type, severity, context)
        - Fallback Selection: FS = argmin(risk_factor_i) for i ∈ fallback_options
        - Risk Assessment: RA = f(error_severity, market_conditions, historical_performance)

        Args:
            error_type: String describing the error type

        Returns:
            Appropriate fallback vector
        """
        error_type_lower = error_type.lower()
        
        if "critical" in error_type_lower or "fatal" in error_type_lower:
            return TriStateFallbackMatrix.get_hold_vector()
        elif "warning" in error_type_lower or "minor" in error_type_lower:
            return TriStateFallbackMatrix.get_conservative_vector()
        else:
            return TriStateFallbackMatrix.get_minimal_buy_vector()


class UnifiedProfitVectorizationSystem:
    """
    Unified profit vectorization system that integrates multiple mathematical approaches.

    This system combines:
    - Clean mathematical foundation
    - Profit vectorization
    - Orbital brain system
    - Pure profit calculator
    - Qutrit signal matrix
    - Quantum operations
    - Entropy calculations
    - Fractal scaling

    Mathematical Foundation:
    - Unified Profit: UP = Σ(w_i * P_i) where w_i are weights, P_i are profit components
    - Integration Weight: IW = f(confidence, historical_performance, market_conditions)
    - Quantum Factor: QF = hν / kT for quantum effects on profit calculation
    - Entropy Weight: EW = -kΣp_i * ln(p_i) for information entropy weighting
    - Fractal Scaling: FS = α * P_prev + (1-α) * P_current for temporal scaling
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the unified profit vectorization system."""
        self.config = config or self._default_config()
        
        # Performance tracking
        self.calculation_times = []
        self.success_count = 0
        self.error_count = 0
        self.total_calculations = 0
        
        # Historical data
        self.profit_history = []
        self.vector_history = []
        self.performance_history = []
        
        # Initialize Schwabot components if available
        if SCHWABOT_COMPONENTS_AVAILABLE:
            self.quantum_bridge = QuantumMathematicalBridge()
            self.orbital_system = OrbitalBRAINSystem()
            self.entropy_system = EntropyMathSystem()
            self.tensor_algebra = AdvancedTensorAlgebra()
        
        # Fallback matrix
        self.fallback_matrix = TriStateFallbackMatrix()
        
        logger.info("💰 Unified Profit Vectorization System initialized")

    def _default_config(self) -> Dict[str, Any]:
        """Default configuration."""
        return {
            "quantum_factor": 1.0,  # hν / kT scaling factor
            "entropy_weight": 0.3,  # Weight for entropy-based calculations
            "fractal_decay": 0.1,   # α decay factor for fractal scaling
            "consensus_threshold": 0.75,
            "confidence_threshold": 0.6,
            "max_history_size": 1000,
            "performance_window": 100,
        }

    def calculate_unified_profit(
        self,
        market_data: Dict[str, Any],
        strategy: VectorizationStrategy = VectorizationStrategy.STANDARD,
        thermal_state: Optional[ThermalState] = None,
        bit_phase: Optional[BitPhase] = None,
        shell_consensus: Optional[ShellConsensus] = None,
        altitude_vector: Optional[AltitudeVector] = None,
    ) -> UnifiedProfitResult:
        """
        Calculate unified profit using multiple mathematical approaches.

        Args:
            market_data: Market data dictionary
            strategy: Vectorization strategy
            thermal_state: Thermal state analysis
            bit_phase: Bit phase calculations
            shell_consensus: Orbital shell consensus
            altitude_vector: Altitude vector

        Returns:
            UnifiedProfitResult with comprehensive profit analysis
        """
        start_time = time.time()
        
        try:
            # Calculate base profit vector
            base_vector = self._calculate_base_profit_vector(market_data, strategy)
            
            # Apply quantum effects
            quantum_factor = self._calculate_quantum_profit_factor(bit_phase)
            quantum_vector = self._apply_quantum_effects(base_vector, quantum_factor)
            
            # Apply entropy weighting
            entropy_weight = self._calculate_entropy_weighting(market_data)
            entropy_vector = self._apply_entropy_weighting(quantum_vector, entropy_weight)
            
            # Apply orbital consensus
            orbital_vector = self._apply_orbital_consensus(entropy_vector, shell_consensus, altitude_vector)
            
            # Apply fractal scaling
            fractal_vector = self._apply_fractal_scaling(orbital_vector)
            
            # Calculate final profit value
            profit_value = self._calculate_final_profit(fractal_vector, market_data)

            # Calculate confidence
            confidence = self._calculate_confidence(fractal_vector, market_data)
            
            # Create result
            result = UnifiedProfitResult(
                timestamp=time.time(),
                profit_value=profit_value,
                confidence=confidence,
                vector=fractal_vector,
                integration_mode=ProfitIntegrationMode.UNIFIED,
                strategy=strategy,
                metadata={
                    "quantum_factor": quantum_factor,
                    "entropy_weight": entropy_weight,
                    "thermal_state": thermal_state.__dict__ if thermal_state else None,
                    "bit_phase": bit_phase.__dict__ if bit_phase else None,
                },
                performance_metrics={
                    "calculation_time": time.time() - start_time,
                    "success": True,
                }
            )
            
            # Update performance tracking
            self._update_performance_tracking(time.time() - start_time, True)
            
            return result

        except Exception as e:
            logger.error(f"Error calculating unified profit: {e}")
            
            # Return fallback result
            fallback_vector = ProfitVector(
                buy_signal=0.0,
                sell_signal=0.0,
                hold_signal=1.0,
                confidence=0.5,
                timestamp=time.time(),
                metadata={"error": str(e)}
            )
            
            fallback_result = UnifiedProfitResult(
                timestamp=time.time(),
                profit_value=0.0,
                confidence=0.5,
                vector=fallback_vector,
                integration_mode=ProfitIntegrationMode.UNIFIED,
                strategy=strategy,
                metadata={"error": str(e)},
                performance_metrics={
                    "calculation_time": time.time() - start_time,
                    "success": False,
                }
            )
            
            self._update_performance_tracking(time.time() - start_time, False)
            return fallback_result

    def _calculate_base_profit_vector(
        self, market_data: Dict[str, Any], strategy: VectorizationStrategy
    ) -> ProfitVector:
        """Calculate base profit vector from market data."""
        try:
            # Extract market data
            price_change = market_data.get("price_change", 0.0)
            volume_change = market_data.get("volume_change", 0.0)
            volatility = market_data.get("volatility", 0.5)
            
            # Calculate base signals
            if strategy == VectorizationStrategy.STANDARD:
                buy_signal = max(0.0, price_change + volume_change * 0.5)
                sell_signal = max(0.0, -price_change + volume_change * 0.5)
                hold_signal = 1.0 - buy_signal - sell_signal
            elif strategy == VectorizationStrategy.ENHANCED:
                buy_signal = max(0.0, price_change * 1.2 + volume_change * 0.7)
                sell_signal = max(0.0, -price_change * 1.2 + volume_change * 0.7)
                hold_signal = 1.0 - buy_signal - sell_signal
            else:
                # Default strategy
                buy_signal = max(0.0, price_change)
                sell_signal = max(0.0, -price_change)
                hold_signal = 1.0 - buy_signal - sell_signal
            
            # Normalize signals
            total = buy_signal + sell_signal + hold_signal
            if total > 0:
                buy_signal /= total
                sell_signal /= total
                hold_signal /= total
            
            return ProfitVector(
                buy_signal=buy_signal,
                sell_signal=sell_signal,
                hold_signal=hold_signal,
                confidence=0.5,
                timestamp=time.time(),
                metadata={"strategy": strategy.value}
            )

        except Exception as e:
            logger.error(f"Error calculating base profit vector: {e}")
            return ProfitVector(0.0, 0.0, 1.0, 0.5, time.time(), {"error": str(e)})

    def _calculate_quantum_profit_factor(self, bit_phase: Optional[BitPhase]) -> float:
        """Calculate quantum profit factor: QPO = hν / kT * profit_base."""
        try:
            if bit_phase is None:
                return 1.0
            
            # Quantum factor calculation
            h_planck = 6.626e-34  # Planck's constant
            k_boltzmann = 1.381e-23  # Boltzmann constant
            
            # Extract parameters from bit phase
            energy_level = bit_phase.energy_level
            temperature = 300.0  # Room temperature in Kelvin
            
            # Calculate quantum factor
            quantum_factor = (h_planck * energy_level) / (k_boltzmann * temperature)
            
            # Normalize to reasonable range
            quantum_factor = min(2.0, max(0.5, quantum_factor))
            
            return float(quantum_factor)

        except Exception as e:
            logger.error(f"Error calculating quantum profit factor: {e}")
            return 1.0

    def _apply_quantum_effects(self, vector: ProfitVector, quantum_factor: float) -> ProfitVector:
        """Apply quantum effects to profit vector."""
        try:
            # Apply quantum factor to signals
            buy_signal = vector.buy_signal * quantum_factor
            sell_signal = vector.sell_signal * quantum_factor
            hold_signal = vector.hold_signal
            
            # Normalize
            total = buy_signal + sell_signal + hold_signal
            if total > 0:
                buy_signal /= total
                sell_signal /= total
                hold_signal /= total
            
            return ProfitVector(
                buy_signal=buy_signal,
                sell_signal=sell_signal,
                hold_signal=hold_signal,
                confidence=vector.confidence,
                timestamp=vector.timestamp,
                metadata={**vector.metadata, "quantum_factor": quantum_factor}
            )

        except Exception as e:
            logger.error(f"Error applying quantum effects: {e}")
            return vector

    def _calculate_entropy_weighting(self, market_data: Dict[str, Any]) -> float:
        """Calculate entropy weighting: EV = -kΣp_i * ln(p_i)."""
        try:
            if SCHWABOT_COMPONENTS_AVAILABLE:
                # Calculate market entropy
                price_changes = market_data.get("price_history", [0.0])
                if len(price_changes) > 1:
                    market_entropy = self.entropy_system.calculate_market_entropy(price_changes)
                else:
                    market_entropy = 0.5
                
                # Normalize entropy weight
                entropy_weight = min(1.0, max(0.0, market_entropy))
                return entropy_weight
            else:
                return 0.5

        except Exception as e:
            logger.error(f"Error calculating entropy weighting: {e}")
            return 0.5

    def _apply_entropy_weighting(self, vector: ProfitVector, entropy_weight: float) -> ProfitVector:
        """Apply entropy weighting to profit vector."""
        try:
            # Apply entropy weight
            buy_signal = vector.buy_signal * (1.0 + entropy_weight * 0.5)
            sell_signal = vector.sell_signal * (1.0 + entropy_weight * 0.5)
            hold_signal = vector.hold_signal * (1.0 - entropy_weight * 0.3)
            
            # Normalize
            total = buy_signal + sell_signal + hold_signal
            if total > 0:
                buy_signal /= total
                sell_signal /= total
                hold_signal /= total
            
            return ProfitVector(
                buy_signal=buy_signal,
                sell_signal=sell_signal,
                hold_signal=hold_signal,
                confidence=vector.confidence,
                timestamp=vector.timestamp,
                metadata={**vector.metadata, "entropy_weight": entropy_weight}
            )

        except Exception as e:
            logger.error(f"Error applying entropy weighting: {e}")
            return vector

    def _apply_orbital_consensus(
        self, 
        vector: ProfitVector, 
        shell_consensus: Optional[ShellConsensus], 
        altitude_vector: Optional[AltitudeVector]
    ) -> ProfitVector:
        """Apply orbital consensus to profit vector."""
        try:
            if shell_consensus is None or altitude_vector is None:
                return vector
            
            # Apply consensus weighting
            consensus_weight = shell_consensus.consensus_score
            altitude_weight = altitude_vector.altitude_value
            
            # Combine weights
            combined_weight = (consensus_weight + altitude_weight) / 2.0
            
            # Apply to vector
            buy_signal = vector.buy_signal * (1.0 + combined_weight * 0.5)
            sell_signal = vector.sell_signal * (1.0 + combined_weight * 0.5)
            hold_signal = vector.hold_signal * (1.0 - combined_weight * 0.3)
            
            # Normalize
            total = buy_signal + sell_signal + hold_signal
            if total > 0:
                buy_signal /= total
                sell_signal /= total
                hold_signal /= total
            
            return ProfitVector(
                buy_signal=buy_signal,
                sell_signal=sell_signal,
                hold_signal=hold_signal,
                confidence=vector.confidence,
                timestamp=vector.timestamp,
                metadata={**vector.metadata, "consensus_weight": combined_weight}
            )

        except Exception as e:
            logger.error(f"Error applying orbital consensus: {e}")
            return vector

    def _apply_fractal_scaling(self, vector: ProfitVector) -> ProfitVector:
        """Apply fractal profit scaling: FPS = α * P_prev + (1-α) * P_current."""
        try:
            alpha = self.config["fractal_decay"]
            
            if self.vector_history:
                # Get previous vector
                prev_vector = self.vector_history[-1]
                
                # Apply fractal scaling
                buy_signal = alpha * prev_vector.buy_signal + (1 - alpha) * vector.buy_signal
                sell_signal = alpha * prev_vector.sell_signal + (1 - alpha) * vector.sell_signal
                hold_signal = alpha * prev_vector.hold_signal + (1 - alpha) * vector.hold_signal
                
                # Normalize
                total = buy_signal + sell_signal + hold_signal
                if total > 0:
                    buy_signal /= total
                    sell_signal /= total
                    hold_signal /= total
            else:
                # No history, use current vector
                buy_signal = vector.buy_signal
                sell_signal = vector.sell_signal
                hold_signal = vector.hold_signal
            
            scaled_vector = ProfitVector(
                buy_signal=buy_signal,
                sell_signal=sell_signal,
                hold_signal=hold_signal,
                confidence=vector.confidence,
                timestamp=vector.timestamp,
                metadata={**vector.metadata, "fractal_alpha": alpha}
            )
            
            # Store in history
            self.vector_history.append(scaled_vector)
            if len(self.vector_history) > self.config["max_history_size"]:
                self.vector_history = self.vector_history[-self.config["max_history_size"]:]
            
            return scaled_vector

        except Exception as e:
            logger.error(f"Error applying fractal scaling: {e}")
            return vector

    def _calculate_final_profit(self, vector: ProfitVector, market_data: Dict[str, Any]) -> float:
        """Calculate final profit value from vector."""
        try:
            # Extract market data
            current_price = market_data.get("current_price", 1.0)
            base_profit = market_data.get("base_profit", 0.0)
            
            # Calculate profit based on vector signals
            buy_profit = vector.buy_signal * current_price * 0.1
            sell_profit = vector.sell_signal * current_price * 0.1
            hold_profit = vector.hold_signal * base_profit
            
            total_profit = buy_profit + sell_profit + hold_profit
            
            return float(total_profit)

        except Exception as e:
            logger.error(f"Error calculating final profit: {e}")
            return 0.0

    def _calculate_confidence(self, vector: ProfitVector, market_data: Dict[str, Any]) -> float:
        """Calculate confidence level for the profit vector."""
        try:
            # Base confidence from vector balance
            signal_strength = max(vector.buy_signal, vector.sell_signal, vector.hold_signal)
            base_confidence = signal_strength
            
            # Market confidence factors
            volatility = market_data.get("volatility", 0.5)
            volume_change = market_data.get("volume_change", 0.0)
            
            # Adjust confidence based on market conditions
            volatility_factor = 1.0 - volatility  # Lower volatility = higher confidence
            volume_factor = min(1.0, abs(volume_change))  # Higher volume = higher confidence
            
            # Combine factors
            confidence = base_confidence * (0.6 + 0.2 * volatility_factor + 0.2 * volume_factor)
            
            return min(1.0, max(0.0, confidence))

        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5

    def _update_performance_tracking(self, calculation_time: float, success: bool) -> None:
        """Update performance tracking metrics."""
        try:
            self.calculation_times.append(calculation_time)
            self.total_calculations += 1
            
            if success:
                self.success_count += 1
            else:
                self.error_count += 1
            
            # Keep history manageable
            if len(self.calculation_times) > self.config["performance_window"]:
                self.calculation_times = self.calculation_times[-self.config["performance_window"]:]

        except Exception as e:
            logger.error(f"Error updating performance tracking: {e}")

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        try:
            if not self.calculation_times:
                return {"error": "No performance data available"}
            
            avg_time = np.mean(self.calculation_times)
            success_rate = self.success_count / self.total_calculations if self.total_calculations > 0 else 0.0
            
            return {
                "total_calculations": self.total_calculations,
                "success_count": self.success_count,
                "error_count": self.error_count,
                "success_rate": success_rate,
                "average_calculation_time": avg_time,
                "min_calculation_time": min(self.calculation_times),
                "max_calculation_time": max(self.calculation_times),
                "vector_history_size": len(self.vector_history),
                "profit_history_size": len(self.profit_history),
            }

        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {"error": str(e)}

    def reset_performance_tracking(self) -> None:
        """Reset all performance tracking data."""
        try:
            self.calculation_times.clear()
            self.success_count = 0
            self.error_count = 0
            self.total_calculations = 0
            self.profit_history.clear()
            self.vector_history.clear()
            self.performance_history.clear()
            
            logger.info("💰 Performance tracking reset")

        except Exception as e:
            logger.error(f"Error resetting performance tracking: {e}")

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        try:
            return {
                "total_calculations": self.total_calculations,
                "success_rate": self.success_count / self.total_calculations if self.total_calculations > 0 else 0.0,
                "average_calculation_time": np.mean(self.calculation_times) if self.calculation_times else 0.0,
                "vector_history_size": len(self.vector_history),
                "profit_history_size": len(self.profit_history),
                "backend": _backend,
                "schwabot_components_available": SCHWABOT_COMPONENTS_AVAILABLE,
                "config": self.config,
                "recent_vectors": [
                    {
                        "buy_signal": v.buy_signal,
                        "sell_signal": v.sell_signal,
                        "hold_signal": v.hold_signal,
                        "confidence": v.confidence,
                        "timestamp": v.timestamp
                    }
                    for v in self.vector_history[-5:]
                ] if self.vector_history else []
            }
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {"error": str(e)}


# Global instance for easy access
unified_profit_vectorization_system = UnifiedProfitVectorizationSystem()
