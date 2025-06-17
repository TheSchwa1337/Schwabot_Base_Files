"""
Mathematical Library v3.0: Sustainment-Unified Framework
========================================================

Core mathematical library integrating the 8 principles of sustainment as fundamental
mathematical operations. This extends mathlib and mathlib_v2 with deep mathematical
integration of sustainment theory.

Mathematical Formulation:
SI(t) = Σᵢ wᵢ Pᵢ(t) where Pᵢ are the 8 principles
Each principle Pᵢ has specific mathematical models:

1. Anticipation: A(t) = τ · ∂/∂t[E[ψ(x,t)]] + K·∇²Φ
2. Integration: I(t) = ∑ᵢ softmax(αᵢ·hᵢ) · sᵢ  
3. Responsiveness: R(t) = e^(-ℓ/λ) · σ(Δt)
4. Simplicity: S(t) = 1 - K(ops)/K_max + entropy_penalty
5. Economy: E(t) = ΔProfit/(ΔCPU + ΔGPU + ΔMem)
6. Survivability: Sv(t) = ∫ ∂²U/∂ψ² dψ (positive curvature)
7. Continuity: C(t) = (1/T)∫[t-T,t] ψ(τ)dτ · coherence_factor
8. Improvisation: Im(t) = lim[n→∞] ||Φⁿ⁺¹ - Φⁿ|| < δ
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from abc import ABC, abstractmethod
import logging
from enum import Enum

# GPU support
try:
    import cupy as cp
    GPU_ENABLED = True
except ImportError:
    cp = np
    GPU_ENABLED = False

# Import existing mathematical foundations
try:
    from .mathlib import CoreMathLib, GradedProfitVector
    from .mathlib_v2 import CoreMathLibV2, SmartStop
except ImportError:
    from mathlib import CoreMathLib, GradedProfitVector
    from mathlib_v2 import CoreMathLibV2, SmartStop

logger = logging.getLogger(__name__)

class SustainmentPrinciple(Enum):
    """8 Principle Sustainment Framework"""
    ANTICIPATION = 0
    INTEGRATION = 1  
    RESPONSIVENESS = 2
    SIMPLICITY = 3
    ECONOMY = 4
    SURVIVABILITY = 5
    CONTINUITY = 6
    IMPROVISATION = 7

@dataclass
class SustainmentVector:
    """Mathematical representation of complete sustainment state"""
    principles: np.ndarray = field(default_factory=lambda: np.ones(8) * 0.5)
    confidence: np.ndarray = field(default_factory=lambda: np.ones(8) * 0.5)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def sustainment_index(self, weights: Optional[np.ndarray] = None) -> float:
        """Calculate weighted sustainment index SI(t)"""
        if weights is None:
            weights = np.array([0.15, 0.15, 0.10, 0.10, 0.15, 0.15, 0.10, 0.10])
        return float(np.dot(self.principles * self.confidence, weights))
    
    def is_sustainable(self, threshold: float = 0.65) -> bool:
        """Check if system meets sustainment threshold"""
        return self.sustainment_index() >= threshold
    
    def failing_principles(self, threshold: float = 0.5) -> List[SustainmentPrinciple]:
        """Get principles below threshold"""
        failing = []
        for i, value in enumerate(self.principles):
            if value < threshold:
                failing.append(SustainmentPrinciple(i))
        return failing

@dataclass
class MathematicalContext:
    """Context for mathematical operations with sustainment integration"""
    current_state: Dict[str, Any] = field(default_factory=dict)
    historical_data: List[Dict[str, Any]] = field(default_factory=list)
    system_metrics: Dict[str, float] = field(default_factory=dict)
    thermal_state: Dict[str, float] = field(default_factory=dict)
    gpu_state: Dict[str, float] = field(default_factory=dict)
    
class SustainmentMathLib(CoreMathLibV2):
    """
    Advanced mathematical library with deep sustainment principle integration.
    All mathematical operations are evaluated through the sustainment framework.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Sustainment mathematical parameters
        self.principle_weights = np.array([0.15, 0.15, 0.10, 0.10, 0.15, 0.15, 0.10, 0.10])
        self.sustainment_threshold = kwargs.get('sustainment_threshold', 0.65)
        self.adaptation_rate = kwargs.get('adaptation_rate', 0.02)
        self.correction_gain = kwargs.get('correction_gain', 0.1)
        
        # Mathematical constants for each principle
        self.anticipation_tau = kwargs.get('anticipation_tau', 0.1)
        self.kalman_gain = kwargs.get('kalman_gain', 0.3)
        self.response_lambda = kwargs.get('response_lambda', 100.0)
        self.complexity_max = kwargs.get('complexity_max', 1000.0)
        self.curvature_window = kwargs.get('curvature_window', 10)
        self.continuity_window = kwargs.get('continuity_window', 50)
        self.convergence_threshold = kwargs.get('convergence_threshold', 0.01)
        
        # State tracking for sustainment calculations
        self.prediction_buffer = deque(maxlen=20)
        self.latency_buffer = deque(maxlen=20)
        self.complexity_buffer = deque(maxlen=20)
        self.efficiency_buffer = deque(maxlen=20)
        self.shock_buffer = deque(maxlen=10)
        self.continuity_buffer = deque(maxlen=self.continuity_window)
        self.iteration_buffer = deque(maxlen=20)
        
        # Integration weights for subsystems
        self.subsystem_weights = {}
        
        logger.info("SustainmentMathLib initialized with 8-principle integration")
    
    # === PRINCIPLE 1: ANTICIPATION ===
    # Mathematical Model: A(t) = τ · ∂/∂t[E[ψ(x,t)]] + K·∇²Φ
    
    def calculate_anticipation_principle(self, context: MathematicalContext) -> Tuple[float, float]:
        """
        Calculate anticipation using predictive derivatives and Kalman filtering.
        
        Returns:
            Tuple of (anticipation_value, confidence)
        """
        try:
            current_state = context.current_state
            price = current_state.get('price', 0.0)
            entropy = current_state.get('entropy', 0.0)
            volume = current_state.get('volume', 0.0)
            
            # Kalman-style prediction
            if len(self.prediction_buffer) > 1:
                last_prediction = self.prediction_buffer[-1]
                prediction_error = price - last_prediction['predicted_price']
                
                # Update prediction with Kalman gain
                next_price = price + self.kalman_gain * prediction_error
                entropy_derivative = (entropy - last_prediction.get('entropy', entropy)) / self.anticipation_tau
            else:
                next_price = price
                entropy_derivative = 0.0
            
            # Store prediction
            prediction_data = {
                'predicted_price': next_price,
                'actual_price': price,
                'entropy': entropy,
                'timestamp': time.time()
            }
            self.prediction_buffer.append(prediction_data)
            
            # Calculate anticipation metric
            if len(self.prediction_buffer) >= 10:
                # Prediction accuracy over recent history
                recent_predictions = list(self.prediction_buffer)[-10:]
                errors = [abs(p['predicted_price'] - p['actual_price']) for p in recent_predictions[1:]]
                avg_error = np.mean(errors) if errors else 0.0
                
                # Normalize anticipation (lower error = higher anticipation)
                max_error = max(errors) if errors else 1.0
                anticipation_value = 1.0 - (avg_error / (max_error + 1e-8))
                
                # Add entropy derivative component
                entropy_component = np.tanh(abs(entropy_derivative)) * 0.3
                anticipation_value = 0.7 * anticipation_value + entropy_component
            else:
                anticipation_value = 0.5  # Default when insufficient history
            
            # Confidence based on prediction history length and consistency
            confidence = min(1.0, len(self.prediction_buffer) / 20.0)
            if len(self.prediction_buffer) > 5:
                prediction_variance = np.var([p['predicted_price'] for p in self.prediction_buffer])
                confidence *= np.exp(-prediction_variance * 0.1)  # Lower variance = higher confidence
            
            return max(0.0, min(1.0, anticipation_value)), max(0.0, min(1.0, confidence))
            
        except Exception as e:
            logger.error(f"Anticipation calculation error: {e}")
            return 0.0, 0.0
    
    # === PRINCIPLE 2: INTEGRATION ===
    # Mathematical Model: I(t) = ∑ᵢ softmax(αᵢ·hᵢ) · sᵢ
    
    def calculate_integration_principle(self, context: MathematicalContext) -> Tuple[float, float]:
        """
        Calculate integration using constraint projection and weighted aggregation.
        
        Returns:
            Tuple of (integration_value, confidence)
        """
        try:
            subsystem_scores = context.system_metrics
            if not subsystem_scores:
                return 0.0, 0.1
            
            # Apply softmax normalization
            scores = np.array(list(subsystem_scores.values()))
            exp_scores = np.exp(scores)
            weights = exp_scores / np.sum(exp_scores)
            
            # Calculate integration metric
            # Perfect integration = balanced weights + high individual scores
            weight_balance = 1.0 - np.std(weights)  # Lower std = better balance
            avg_score = np.mean(scores)
            
            integration_value = weight_balance * avg_score
            
            # Store weights for other modules
            self.subsystem_weights = dict(zip(subsystem_scores.keys(), weights))
            
            # Confidence based on number of subsystems and score quality
            confidence = min(1.0, len(subsystem_scores) / 5.0)
            confidence *= (1.0 - np.std(scores) * 0.5)  # Penalize high variance
            
            return max(0.0, min(1.0, integration_value)), max(0.0, min(1.0, confidence))
            
        except Exception as e:
            logger.error(f"Integration calculation error: {e}")
            return 0.0, 0.0
    
    # === PRINCIPLE 3: RESPONSIVENESS ===
    # Mathematical Model: R(t) = e^(-ℓ/λ) · σ(Δt)
    
    def calculate_responsiveness_principle(self, context: MathematicalContext) -> Tuple[float, float]:
        """
        Calculate responsiveness using exponential latency decay.
        
        Returns:
            Tuple of (responsiveness_value, confidence)
        """
        try:
            # Get current latency from context
            current_latency = context.system_metrics.get('latency_ms', 50.0)
            self.latency_buffer.append(current_latency)
            
            # Calculate responsiveness using exponential decay
            responsiveness_value = np.exp(-current_latency / self.response_lambda)
            
            if len(self.latency_buffer) > 5:
                # Add consistency component
                latency_std = np.std(list(self.latency_buffer))
                consistency_factor = np.exp(-latency_std * 0.01)
                responsiveness_value *= consistency_factor
            
            # Confidence based on measurement history
            confidence = min(1.0, len(self.latency_buffer) / 20.0)
            
            return max(0.0, min(1.0, responsiveness_value)), max(0.0, min(1.0, confidence))
            
        except Exception as e:
            logger.error(f"Responsiveness calculation error: {e}")
            return 0.0, 0.0
    
    # === PRINCIPLE 4: SIMPLICITY ===
    # Mathematical Model: S(t) = 1 - K(ops)/K_max + entropy_penalty
    
    def calculate_simplicity_principle(self, context: MathematicalContext) -> Tuple[float, float]:
        """
        Calculate simplicity using complexity normalization.
        
        Returns:
            Tuple of (simplicity_value, confidence)
        """
        try:
            operations_count = context.system_metrics.get('operations_count', 0)
            strategy_count = context.system_metrics.get('active_strategies', 1)
            
            self.complexity_buffer.append(operations_count)
            
            # Calculate simplicity metric
            complexity_ratio = operations_count / self.complexity_max
            simplicity_value = 1.0 - min(1.0, complexity_ratio)
            
            # Penalize high strategy count
            strategy_penalty = min(0.3, strategy_count * 0.02)
            simplicity_value -= strategy_penalty
            
            # Add trend penalty for increasing complexity
            if len(self.complexity_buffer) > 10:
                recent_trend = np.mean(list(self.complexity_buffer)[-5:]) - np.mean(list(self.complexity_buffer)[-10:-5])
                if recent_trend > 0:
                    trend_penalty = min(0.2, recent_trend / self.complexity_max)
                    simplicity_value -= trend_penalty
            
            # Confidence based on measurement stability
            confidence = 0.8  # High confidence for operational metrics
            if len(self.complexity_buffer) > 5:
                complexity_variance = np.var(list(self.complexity_buffer))
                confidence *= np.exp(-complexity_variance * 0.001)
            
            return max(0.0, min(1.0, simplicity_value)), max(0.0, min(1.0, confidence))
            
        except Exception as e:
            logger.error(f"Simplicity calculation error: {e}")
            return 0.0, 0.0
    
    # === PRINCIPLE 5: ECONOMY ===
    # Mathematical Model: E(t) = ΔProfit/(ΔCPU + ΔGPU + ΔMem)
    
    def calculate_economy_principle(self, context: MathematicalContext) -> Tuple[float, float]:
        """
        Calculate economy using profit-per-resource ratios.
        
        Returns:
            Tuple of (economy_value, confidence)
        """
        try:
            profit_delta = context.system_metrics.get('profit_delta', 0.0)
            cpu_cost = context.system_metrics.get('cpu_cost', 1.0)
            gpu_cost = context.system_metrics.get('gpu_cost', 2.0)
            memory_cost = context.system_metrics.get('memory_cost', 0.5)
            
            # Calculate total resource cost
            total_cost = cpu_cost + gpu_cost + memory_cost
            
            # Calculate efficiency ratio
            if total_cost > 0:
                efficiency = profit_delta / total_cost
            else:
                efficiency = 0.0
            
            self.efficiency_buffer.append(efficiency)
            
            # Normalize using sigmoid for better bounds
            economy_value = 2.0 / (1.0 + np.exp(-efficiency)) - 1.0  # Sigmoid mapping to [-1, 1]
            economy_value = (economy_value + 1.0) / 2.0  # Map to [0, 1]
            
            # Add consistency component
            if len(self.efficiency_buffer) > 5:
                efficiency_variance = np.var(list(self.efficiency_buffer))
                consistency_bonus = np.exp(-efficiency_variance * 0.1) * 0.1
                economy_value += consistency_bonus
            
            # Confidence based on profit magnitude and consistency
            confidence = 0.7
            if abs(profit_delta) > 1e-6:  # Non-zero profit increases confidence
                confidence += 0.2
            if len(self.efficiency_buffer) > 10:
                confidence += 0.1
            
            return max(0.0, min(1.0, economy_value)), max(0.0, min(1.0, confidence))
            
        except Exception as e:
            logger.error(f"Economy calculation error: {e}")
            return 0.0, 0.0
    
    # === PRINCIPLE 6: SURVIVABILITY ===
    # Mathematical Model: Sv(t) = ∫ ∂²U/∂ψ² dψ (positive curvature)
    
    def calculate_survivability_principle(self, context: MathematicalContext) -> Tuple[float, float]:
        """
        Calculate survivability using utility curvature and shock response.
        
        Returns:
            Tuple of (survivability_value, confidence)
        """
        try:
            shock_magnitude = context.system_metrics.get('shock_magnitude', 0.0)
            utility_values = context.system_metrics.get('utility_history', [])
            
            self.shock_buffer.append(shock_magnitude)
            
            survivability_value = 0.5  # Default
            
            # Calculate utility curvature if we have enough history
            if len(utility_values) >= 3:
                # Calculate second derivative (curvature)
                second_derivatives = []
                for i in range(1, len(utility_values) - 1):
                    second_deriv = utility_values[i+1] - 2*utility_values[i] + utility_values[i-1]
                    second_derivatives.append(second_deriv)
                
                if second_derivatives:
                    avg_curvature = np.mean(second_derivatives)
                    # Positive curvature indicates better survivability
                    curvature_component = np.tanh(avg_curvature) * 0.5 + 0.5
                    survivability_value = 0.7 * curvature_component
            
            # Add shock response component
            if len(self.shock_buffer) > 3:
                recent_shocks = list(self.shock_buffer)[-3:]
                shock_response = 1.0 - min(1.0, np.mean(recent_shocks))
                survivability_value += 0.3 * shock_response
            
            # Confidence based on data availability
            confidence = 0.6
            if len(utility_values) >= 10:
                confidence += 0.2
            if len(self.shock_buffer) >= 5:
                confidence += 0.2
            
            return max(0.0, min(1.0, survivability_value)), max(0.0, min(1.0, confidence))
            
        except Exception as e:
            logger.error(f"Survivability calculation error: {e}")
            return 0.0, 0.0
    
    # === PRINCIPLE 7: CONTINUITY ===
    # Mathematical Model: C(t) = (1/T)∫[t-T,t] ψ(τ)dτ · coherence_factor
    
    def calculate_continuity_principle(self, context: MathematicalContext) -> Tuple[float, float]:
        """
        Calculate continuity using integral memory and coherence.
        
        Returns:
            Tuple of (continuity_value, confidence)
        """
        try:
            system_state = context.system_metrics.get('system_state', 0.5)
            uptime_ratio = context.system_metrics.get('uptime_ratio', 1.0)
            
            self.continuity_buffer.append(system_state)
            
            # Calculate integral memory
            if len(self.continuity_buffer) >= 10:
                buffer_array = np.array(list(self.continuity_buffer))
                integral_memory = np.mean(buffer_array)
                
                # Calculate stability (low fluctuation = high continuity)
                fluctuation_penalty = np.std(buffer_array) * 0.3
                stability_component = max(0.0, 1.0 - fluctuation_penalty)
                
                continuity_value = 0.6 * integral_memory + 0.2 * stability_component + 0.2 * uptime_ratio
            else:
                # Insufficient history - use uptime and current state
                continuity_value = 0.7 * uptime_ratio + 0.3 * system_state
            
            # Confidence increases with buffer fullness
            confidence = min(1.0, len(self.continuity_buffer) / self.continuity_window)
            
            return max(0.0, min(1.0, continuity_value)), max(0.0, min(1.0, confidence))
            
        except Exception as e:
            logger.error(f"Continuity calculation error: {e}")
            return 0.0, 0.0
    
    # === PRINCIPLE 8: IMPROVISATION (TRANSCENDENCE) ===
    # Mathematical Model: Im(t) = lim[n→∞] ||Φⁿ⁺¹ - Φⁿ|| < δ
    
    def calculate_improvisation_principle(self, context: MathematicalContext) -> Tuple[float, float]:
        """
        Calculate improvisation using convergence analysis.
        
        Returns:
            Tuple of (improvisation_value, confidence)
        """
        try:
            current_iteration = context.system_metrics.get('optimization_state', 0.5)
            adaptation_rate = context.system_metrics.get('adaptation_rate', 0.0)
            
            self.iteration_buffer.append(current_iteration)
            
            improvisation_value = 0.5  # Default
            
            if len(self.iteration_buffer) >= 5:
                iterations = np.array(list(self.iteration_buffer))
                
                # Calculate convergence rate
                if len(iterations) > 1:
                    differences = np.abs(np.diff(iterations))
                    
                    # Check if converging (differences getting smaller)
                    if len(differences) > 2:
                        convergence_trend = np.mean(differences[-3:]) - np.mean(differences[:-3])
                        if convergence_trend < 0:  # Decreasing differences = converging
                            convergence_component = min(1.0, abs(convergence_trend) * 10.0)
                        else:
                            convergence_component = 0.3  # Still searching
                    else:
                        convergence_component = 0.5
                    
                    # Check proximity to convergence threshold
                    latest_difference = differences[-1] if len(differences) > 0 else 1.0
                    proximity_component = 1.0 - min(1.0, latest_difference / self.convergence_threshold)
                    
                    # Combine components
                    improvisation_value = 0.4 * convergence_component + 0.4 * proximity_component + 0.2 * adaptation_rate
            
            # Confidence based on iteration history
            confidence = min(1.0, len(self.iteration_buffer) / 20.0)
            
            return max(0.0, min(1.0, improvisation_value)), max(0.0, min(1.0, confidence))
            
        except Exception as e:
            logger.error(f"Improvisation calculation error: {e}")
            return 0.0, 0.0
    
    # === UNIFIED SUSTAINMENT CALCULATION ===
    
    def calculate_sustainment_vector(self, context: MathematicalContext) -> SustainmentVector:
        """
        Calculate complete sustainment vector using all 8 principles.
        
        Args:
            context: Current mathematical context
            
        Returns:
            Complete sustainment vector with confidence scores
        """
        try:
            # Calculate all principles
            anticipation, anticipation_conf = self.calculate_anticipation_principle(context)
            integration, integration_conf = self.calculate_integration_principle(context)
            responsiveness, responsiveness_conf = self.calculate_responsiveness_principle(context)
            simplicity, simplicity_conf = self.calculate_simplicity_principle(context)
            economy, economy_conf = self.calculate_economy_principle(context)
            survivability, survivability_conf = self.calculate_survivability_principle(context)
            continuity, continuity_conf = self.calculate_continuity_principle(context)
            improvisation, improvisation_conf = self.calculate_improvisation_principle(context)
            
            # Assemble vectors
            principles = np.array([
                anticipation, integration, responsiveness, simplicity,
                economy, survivability, continuity, improvisation
            ])
            
            confidence = np.array([
                anticipation_conf, integration_conf, responsiveness_conf, simplicity_conf,
                economy_conf, survivability_conf, continuity_conf, improvisation_conf
            ])
            
            # Create sustainment vector
            sustainment_vector = SustainmentVector(
                principles=principles,
                confidence=confidence,
                timestamp=datetime.now(),
                metadata={
                    'context_size': len(context.current_state),
                    'historical_depth': len(context.historical_data),
                    'subsystem_weights': self.subsystem_weights.copy()
                }
            )
            
            return sustainment_vector
            
        except Exception as e:
            logger.error(f"Sustainment vector calculation error: {e}")
            return SustainmentVector()  # Return default vector
    
    # === MATHEMATICAL CORRECTION OPERATIONS ===
    
    def calculate_sustainment_correction(self, current_vector: SustainmentVector, 
                                       target_threshold: float = None) -> Dict[str, float]:
        """
        Calculate mathematical corrections needed to achieve sustainment.
        
        Args:
            current_vector: Current sustainment state
            target_threshold: Target sustainment index (default: self.sustainment_threshold)
            
        Returns:
            Dictionary of correction magnitudes for each principle
        """
        if target_threshold is None:
            target_threshold = self.sustainment_threshold
        
        current_si = current_vector.sustainment_index(self.principle_weights)
        
        corrections = {}
        
        if current_si < target_threshold:
            deficit = target_threshold - current_si
            
            # Calculate corrections proportional to principle weights and current deficits
            for i, principle in enumerate(SustainmentPrinciple):
                principle_deficit = max(0.0, target_threshold - current_vector.principles[i])
                correction_magnitude = self.correction_gain * principle_deficit * self.principle_weights[i]
                corrections[principle.value] = float(correction_magnitude)
        
        return corrections
    
    # === ADVANCED SUSTAINMENT OPERATIONS ===
    
    def sustainment_aware_trading_decision(self, prices: np.ndarray, volumes: np.ndarray,
                                         context: MathematicalContext) -> Dict[str, Any]:
        """
        Make trading decisions that consider sustainment principles.
        
        Args:
            prices: Price series
            volumes: Volume series  
            context: Current mathematical context
            
        Returns:
            Trading decision enhanced with sustainment analysis
        """
        # Get base trading analysis
        base_results = super().apply_advanced_strategies_v2(prices, volumes)
        
        # Calculate sustainment vector
        sustainment_vector = self.calculate_sustainment_vector(context)
        
        # Modify trading signals based on sustainment
        si = sustainment_vector.sustainment_index(self.principle_weights)
        
        # Reduce position size if sustainment is poor
        position_multiplier = min(1.0, si / self.sustainment_threshold) if si < self.sustainment_threshold else 1.0
        
        # Add sustainment-based signals
        sustainment_results = {
            'sustainment_index': si,
            'sustainment_vector': sustainment_vector,
            'position_multiplier': position_multiplier,
            'failing_principles': sustainment_vector.failing_principles(),
            'corrections_needed': self.calculate_sustainment_correction(sustainment_vector),
            'is_sustainable': sustainment_vector.is_sustainable()
        }
        
        # Merge with base results
        base_results.update(sustainment_results)
        
        return base_results
    
    def get_sustainment_metrics(self) -> Dict[str, Any]:
        """Get current sustainment system metrics"""
        return {
            'principle_weights': self.principle_weights.tolist(),
            'sustainment_threshold': self.sustainment_threshold,
            'buffer_states': {
                'prediction_buffer_size': len(self.prediction_buffer),
                'latency_buffer_size': len(self.latency_buffer),
                'complexity_buffer_size': len(self.complexity_buffer),
                'efficiency_buffer_size': len(self.efficiency_buffer),
                'shock_buffer_size': len(self.shock_buffer),
                'continuity_buffer_size': len(self.continuity_buffer),
                'iteration_buffer_size': len(self.iteration_buffer)
            },
            'subsystem_weights': self.subsystem_weights.copy()
        }

# === GPU-ACCELERATED SUSTAINMENT OPERATIONS ===

def gpu_sustainment_vector_operations(vectors: List[SustainmentVector], 
                                    weights: np.ndarray) -> Dict[str, float]:
    """GPU-accelerated sustainment vector operations"""
    if not GPU_ENABLED or not vectors:
        return {}
    
    try:
        # Convert to GPU arrays
        principle_matrix = cp.array([v.principles for v in vectors])
        confidence_matrix = cp.array([v.confidence for v in vectors])
        weights_gpu = cp.array(weights)
        
        # Batch calculate sustainment indices
        weighted_principles = principle_matrix * confidence_matrix
        sustainment_indices = cp.dot(weighted_principles, weights_gpu)
        
        # Calculate statistics
        results = {
            'mean_si': float(cp.mean(sustainment_indices)),
            'std_si': float(cp.std(sustainment_indices)),
            'min_si': float(cp.min(sustainment_indices)),
            'max_si': float(cp.max(sustainment_indices)),
            'sustainable_ratio': float(cp.mean(sustainment_indices >= 0.65))
        }
        
        return results
        
    except Exception as e:
        logger.error(f"GPU sustainment operations error: {e}")
        return {}

# === TESTING UTILITIES ===

def create_test_context() -> MathematicalContext:
    """Create test mathematical context"""
    return MathematicalContext(
        current_state={
            'price': 100.0,
            'entropy': 0.5,
            'volume': 1000.0
        },
        system_metrics={
            'latency_ms': 25.0,
            'operations_count': 500,
            'active_strategies': 3,
            'profit_delta': 10.0,
            'cpu_cost': 5.0,
            'gpu_cost': 10.0,
            'memory_cost': 2.0,
            'shock_magnitude': 0.1,
            'system_state': 0.8,
            'uptime_ratio': 0.95,
            'optimization_state': 0.7,
            'adaptation_rate': 0.1
        }
    )

if __name__ == "__main__":
    # Basic testing
    math_lib = SustainmentMathLib()
    context = create_test_context()
    
    sustainment_vector = math_lib.calculate_sustainment_vector(context)
    print(f"Sustainment Index: {sustainment_vector.sustainment_index():.3f}")
    print(f"Is Sustainable: {sustainment_vector.is_sustainable()}")
    print(f"Failing Principles: {sustainment_vector.failing_principles()}") 