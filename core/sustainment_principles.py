"""
Sustainment Principles Mathematical Framework
==========================================

Core mathematical implementation of the 8 principles of sustainment for Schwabot.
Each principle is modeled with concrete equations and real-time calculations.

Principles:
1. Integration - Constraint projection and weighted aggregation
2. Anticipation - Predictive derivatives and Kalman filtering
3. Responsiveness - Time-constant response and latency modeling
4. Simplicity - Kolmogorov complexity proxy and optimization
5. Economy - Profit-per-compute ratios and efficiency metrics
6. Survivability - Positive curvature and shock absorption
7. Continuity - Integral memory and sliding-window coherence
8. Transcendence - Recursive convergence and fixed-point iteration
"""

import numpy as np
import time
from typing import Dict, Any, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from collections import deque
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

@dataclass
class PrincipleMetrics:
    """Metrics for a single sustainment principle"""
    value: float = 0.0
    confidence: float = 0.0
    threshold: float = 0.5
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_healthy(self) -> bool:
        """Check if principle meets threshold with confidence"""
        return self.value >= self.threshold and self.confidence >= 0.7

@dataclass
class SustainmentState:
    """Complete sustainment state across all principles"""
    integration: PrincipleMetrics = field(default_factory=PrincipleMetrics)
    anticipation: PrincipleMetrics = field(default_factory=PrincipleMetrics)
    responsiveness: PrincipleMetrics = field(default_factory=PrincipleMetrics)
    simplicity: PrincipleMetrics = field(default_factory=PrincipleMetrics)
    economy: PrincipleMetrics = field(default_factory=PrincipleMetrics)
    survivability: PrincipleMetrics = field(default_factory=PrincipleMetrics)
    continuity: PrincipleMetrics = field(default_factory=PrincipleMetrics)
    transcendence: PrincipleMetrics = field(default_factory=PrincipleMetrics)
    
    def composite_score(self) -> float:
        """Calculate weighted composite sustainment score"""
        weights = {
            'integration': 0.15,
            'anticipation': 0.15,
            'responsiveness': 0.12,
            'simplicity': 0.10,
            'economy': 0.15,
            'survivability': 0.13,
            'continuity': 0.10,
            'transcendence': 0.10
        }
        
        total = 0.0
        for principle, weight in weights.items():
            metric = getattr(self, principle)
            total += weight * metric.value * metric.confidence
            
        return min(1.0, max(0.0, total))
    
    def failing_principles(self) -> List[str]:
        """Get list of principles below threshold"""
        failing = []
        for name in ['integration', 'anticipation', 'responsiveness', 'simplicity',
                    'economy', 'survivability', 'continuity', 'transcendence']:
            metric = getattr(self, name)
            if not metric.is_healthy():
                failing.append(name)
        return failing

class BasePrinciple(ABC):
    """Base class for sustainment principle calculators"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.history = deque(maxlen=config.get('history_size', 100))
        
    @abstractmethod
    def calculate(self, context: Dict[str, Any]) -> PrincipleMetrics:
        """Calculate principle metric from current context"""
        pass
    
    def update_history(self, metric: PrincipleMetrics) -> None:
        """Update principle calculation history"""
        self.history.append(metric)

class IntegrationPrinciple(BasePrinciple):
    """
    Principle 1: Integration
    Mathematical Model: Constraint projection with softmax normalization
    
    ∑ᵢ wᵢ(x) = 1, wᵢ ≥ 0
    wᵢ(x) = exp(α·hᵢ(x)) / ∑ⱼ exp(α·hⱼ(x))
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.softmax_alpha = config.get('integration_softmax_alpha', 1.0)
        self.subsystem_weights = {}
        
    def calculate(self, context: Dict[str, Any]) -> PrincipleMetrics:
        """Calculate integration using softmax weight normalization"""
        try:
            # Get subsystem scores from context
            subsystem_scores = context.get('subsystem_scores', {})
            if not subsystem_scores:
                return PrincipleMetrics(value=0.0, confidence=0.1, 
                                      metadata={'error': 'No subsystem scores'})
            
            # Apply softmax normalization
            scores = np.array(list(subsystem_scores.values()))
            exp_scores = np.exp(self.softmax_alpha * scores)
            weights = exp_scores / np.sum(exp_scores)
            
            # Calculate integration metric
            # Perfect integration = all weights balanced, high individual scores
            weight_balance = 1.0 - np.std(weights)  # Lower std = better balance
            avg_score = np.mean(scores)
            
            integration_value = weight_balance * avg_score
            
            # Calculate confidence based on data quality
            confidence = min(1.0, len(subsystem_scores) / 5.0)  # Need at least 5 subsystems for full confidence
            
            # Store weights for other modules to use
            self.subsystem_weights = dict(zip(subsystem_scores.keys(), weights))
            
            metric = PrincipleMetrics(
                value=integration_value,
                confidence=confidence,
                threshold=self.config.get('integration_threshold', 0.6),
                metadata={
                    'weights': self.subsystem_weights,
                    'weight_balance': weight_balance,
                    'avg_score': avg_score,
                    'subsystem_count': len(subsystem_scores)
                }
            )
            
            self.update_history(metric)
            return metric
            
        except Exception as e:
            logger.error(f"Integration calculation error: {e}")
            return PrincipleMetrics(value=0.0, confidence=0.0, 
                                  metadata={'error': str(e)})

class AnticipationPrinciple(BasePrinciple):
    """
    Principle 2: Anticipation  
    Mathematical Model: Predictive derivative with Kalman filtering
    
    A(x,t) = τ · ∂/∂t[E[ψ(x,t)]]
    p̂ₜ₊₁ = pₜ + Kₜ(pₜ - p̂ₜ⁻)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.tau = config.get('anticipation_tau', 0.1)
        self.kalman_gain = config.get('kalman_gain', 0.3)
        self.prediction_buffer = deque(maxlen=20)
        self.prediction_errors = deque(maxlen=10)
        
    def calculate(self, context: Dict[str, Any]) -> PrincipleMetrics:
        """Calculate anticipation using Kalman-style prediction"""
        try:
            current_state = context.get('current_state', {})
            if not current_state:
                return PrincipleMetrics(value=0.0, confidence=0.1,
                                      metadata={'error': 'No current state'})
            
            # Get key metrics for prediction
            price = current_state.get('price', 0.0)
            entropy = current_state.get('entropy', 0.0)
            volume = current_state.get('volume', 0.0)
            
            # Simple Kalman-style prediction
            if len(self.prediction_buffer) > 1:
                last_prediction = self.prediction_buffer[-1]
                prediction_error = price - last_prediction['predicted_price']
                self.prediction_errors.append(abs(prediction_error))
                
                # Update prediction with Kalman gain
                next_price = price + self.kalman_gain * prediction_error
            else:
                next_price = price  # No history, predict current price
            
            # Calculate entropy derivative (rate of change)
            if len(self.history) > 0:
                last_entropy = self.history[-1].metadata.get('entropy', entropy)
                entropy_derivative = (entropy - last_entropy) * self.tau
            else:
                entropy_derivative = 0.0
            
            # Anticipation value based on prediction accuracy and derivative
            if self.prediction_errors:
                avg_error = np.mean(list(self.prediction_errors))
                prediction_accuracy = max(0.0, 1.0 - avg_error / (abs(price) + 1e-6))
            else:
                prediction_accuracy = 0.5  # No history
            
            # Weight anticipation by how well we're tracking changes
            anticipation_value = prediction_accuracy * (1.0 + abs(entropy_derivative))
            anticipation_value = min(1.0, anticipation_value)
            
            # Store prediction for next iteration
            self.prediction_buffer.append({
                'timestamp': time.time(),
                'predicted_price': next_price,
                'actual_price': price,
                'entropy_derivative': entropy_derivative
            })
            
            confidence = min(1.0, len(self.prediction_buffer) / 10.0)
            
            metric = PrincipleMetrics(
                value=anticipation_value,
                confidence=confidence,
                threshold=self.config.get('anticipation_threshold', 0.5),
                metadata={
                    'predicted_price': next_price,
                    'entropy_derivative': entropy_derivative,
                    'prediction_accuracy': prediction_accuracy,
                    'avg_prediction_error': np.mean(list(self.prediction_errors)) if self.prediction_errors else 0.0
                }
            )
            
            self.update_history(metric)
            return metric
            
        except Exception as e:
            logger.error(f"Anticipation calculation error: {e}")
            return PrincipleMetrics(value=0.0, confidence=0.0, 
                                  metadata={'error': str(e)})

class ResponsivenessPrinciple(BasePrinciple):
    """
    Principle 3: Responsiveness
    Mathematical Model: Time-constant response
    
    R = e^(-ℓ/λ) where λ = desired max latency
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.lambda_max = config.get('max_latency_ms', 100.0)
        self.latency_buffer = deque(maxlen=20)
        
    def calculate(self, context: Dict[str, Any]) -> PrincipleMetrics:
        """Calculate responsiveness based on system latency"""
        try:
            # Get current latency metrics
            latency_ms = context.get('system_latency_ms', 50.0)
            event_response_ms = context.get('event_response_ms', 20.0)
            
            self.latency_buffer.append(latency_ms)
            
            # Calculate responsiveness using exponential decay
            responsiveness = np.exp(-latency_ms / self.lambda_max)
            
            # Also factor in consistency - less variance = more responsive
            if len(self.latency_buffer) > 5:
                latency_std = np.std(list(self.latency_buffer))
                consistency_factor = np.exp(-latency_std / self.lambda_max)
                responsiveness *= consistency_factor
            
            # Factor in event response time
            event_responsiveness = np.exp(-event_response_ms / (self.lambda_max * 0.5))
            responsiveness = 0.7 * responsiveness + 0.3 * event_responsiveness
            
            confidence = min(1.0, len(self.latency_buffer) / 10.0)
            
            metric = PrincipleMetrics(
                value=responsiveness,
                confidence=confidence,
                threshold=self.config.get('responsiveness_threshold', 0.7),
                metadata={
                    'latency_ms': latency_ms,
                    'event_response_ms': event_response_ms,
                    'avg_latency': np.mean(list(self.latency_buffer)) if self.latency_buffer else 0.0,
                    'latency_std': np.std(list(self.latency_buffer)) if len(self.latency_buffer) > 1 else 0.0
                }
            )
            
            self.update_history(metric)
            return metric
            
        except Exception as e:
            logger.error(f"Responsiveness calculation error: {e}")
            return PrincipleMetrics(value=0.0, confidence=0.0, 
                                  metadata={'error': str(e)})

class SimplicityPrinciple(BasePrinciple):
    """
    Principle 4: Simplicity
    Mathematical Model: Kolmogorov complexity proxy
    
    S = 1 - ops/ops_max
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.max_ops = config.get('max_operations', 1000)
        self.complexity_buffer = deque(maxlen=20)
        
    def calculate(self, context: Dict[str, Any]) -> PrincipleMetrics:
        """Calculate simplicity based on operational complexity"""
        try:
            # Get complexity metrics
            operation_count = context.get('operation_count', 0)
            ncco_complexity = context.get('ncco_complexity', 0)
            active_strategies = context.get('active_strategies', 0)
            
            # Calculate normalized complexity
            total_ops = operation_count + ncco_complexity + active_strategies * 10
            normalized_complexity = min(1.0, total_ops / self.max_ops)
            
            # Simplicity is inverse of complexity
            simplicity = 1.0 - normalized_complexity
            
            self.complexity_buffer.append(total_ops)
            
            # Factor in complexity trend - increasing complexity reduces simplicity
            if len(self.complexity_buffer) > 5:
                recent_trend = np.polyfit(range(len(self.complexity_buffer)), 
                                        list(self.complexity_buffer), 1)[0]
                trend_penalty = max(0.0, recent_trend / self.max_ops)
                simplicity -= trend_penalty
                simplicity = max(0.0, simplicity)
            
            confidence = 0.9  # High confidence in complexity measurement
            
            metric = PrincipleMetrics(
                value=simplicity,
                confidence=confidence,
                threshold=self.config.get('simplicity_threshold', 0.6),
                metadata={
                    'operation_count': operation_count,
                    'ncco_complexity': ncco_complexity,
                    'active_strategies': active_strategies,
                    'total_ops': total_ops,
                    'normalized_complexity': normalized_complexity,
                    'complexity_trend': np.polyfit(range(len(self.complexity_buffer)), 
                                                 list(self.complexity_buffer), 1)[0] if len(self.complexity_buffer) > 5 else 0.0
                }
            )
            
            self.update_history(metric)
            return metric
            
        except Exception as e:
            logger.error(f"Simplicity calculation error: {e}")
            return PrincipleMetrics(value=0.0, confidence=0.0, 
                                  metadata={'error': str(e)})

class EconomyPrinciple(BasePrinciple):
    """
    Principle 5: Economy
    Mathematical Model: Profit-per-compute ratio
    
    E = ΔProfit / (ΔCPU_cycles + ΔGPU_cycles)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.min_efficiency = config.get('min_efficiency', 0.001)
        self.efficiency_buffer = deque(maxlen=20)
        
    def calculate(self, context: Dict[str, Any]) -> PrincipleMetrics:
        """Calculate economy based on profit-per-compute efficiency"""
        try:
            # Get efficiency metrics
            profit_delta = context.get('profit_delta', 0.0)
            cpu_cycles = context.get('cpu_cycles', 1.0)  # Avoid division by zero
            gpu_cycles = context.get('gpu_cycles', 0.0)
            memory_usage = context.get('memory_usage_mb', 0.0)
            
            # Calculate total compute cost (normalized)
            total_compute = cpu_cycles + gpu_cycles * 2.0 + memory_usage * 0.001
            
            # Calculate efficiency
            if total_compute > 0:
                efficiency = profit_delta / total_compute
            else:
                efficiency = 0.0
            
            self.efficiency_buffer.append(efficiency)
            
            # Normalize efficiency to [0, 1] using sigmoid
            normalized_efficiency = 1.0 / (1.0 + np.exp(-efficiency / self.min_efficiency))
            
            # Factor in consistency of efficiency
            if len(self.efficiency_buffer) > 5:
                efficiency_mean = np.mean(list(self.efficiency_buffer))
                efficiency_std = np.std(list(self.efficiency_buffer))
                consistency_factor = np.exp(-efficiency_std / max(abs(efficiency_mean), 1e-6))
                normalized_efficiency *= consistency_factor
            
            confidence = min(1.0, len(self.efficiency_buffer) / 10.0)
            
            metric = PrincipleMetrics(
                value=normalized_efficiency,
                confidence=confidence,
                threshold=self.config.get('economy_threshold', 0.5),
                metadata={
                    'profit_delta': profit_delta,
                    'cpu_cycles': cpu_cycles,
                    'gpu_cycles': gpu_cycles,
                    'memory_usage_mb': memory_usage,
                    'total_compute': total_compute,
                    'efficiency': efficiency,
                    'avg_efficiency': np.mean(list(self.efficiency_buffer)) if self.efficiency_buffer else 0.0
                }
            )
            
            self.update_history(metric)
            return metric
            
        except Exception as e:
            logger.error(f"Economy calculation error: {e}")
            return PrincipleMetrics(value=0.0, confidence=0.0, 
                                  metadata={'error': str(e)})

class SurvivabilityPrinciple(BasePrinciple):
    """
    Principle 6: Survivability
    Mathematical Model: Positive curvature requirement
    
    ∂²U/∂ψ² > 0 (utility increases with entropy shocks)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.shock_memory = deque(maxlen=10)
        self.utility_memory = deque(maxlen=10)
        
    def calculate(self, context: Dict[str, Any]) -> PrincipleMetrics:
        """Calculate survivability based on shock response"""
        try:
            # Get survivability metrics
            current_utility = context.get('current_utility', 0.0)
            entropy_level = context.get('entropy_level', 0.0)
            shock_magnitude = context.get('shock_magnitude', 0.0)
            recovery_rate = context.get('recovery_rate', 1.0)
            
            self.shock_memory.append(shock_magnitude)
            self.utility_memory.append(current_utility)
            
            # Calculate curvature if we have enough history
            survivability = 0.5  # Default
            
            if len(self.utility_memory) >= 3:
                # Estimate second derivative of utility w.r.t. entropy
                utilities = list(self.utility_memory)
                
                # Simple second derivative approximation
                if len(utilities) >= 3:
                    u_t_minus_1 = utilities[-3]
                    u_t = utilities[-2]
                    u_t_plus_1 = utilities[-1]
                    
                    second_derivative = u_t_plus_1 - 2*u_t + u_t_minus_1
                    
                    # Positive curvature indicates good survivability
                    survivability = 1.0 / (1.0 + np.exp(-second_derivative))
            
            # Factor in shock recovery
            if self.shock_memory:
                recent_shocks = list(self.shock_memory)[-5:]
                avg_shock = np.mean(recent_shocks) if recent_shocks else 0.0
                
                # Better survivability if we handle shocks well
                shock_response = recovery_rate / (1.0 + avg_shock)
                survivability = 0.7 * survivability + 0.3 * min(1.0, shock_response)
            
            confidence = min(1.0, len(self.utility_memory) / 5.0)
            
            metric = PrincipleMetrics(
                value=survivability,
                confidence=confidence,
                threshold=self.config.get('survivability_threshold', 0.6),
                metadata={
                    'current_utility': current_utility,
                    'entropy_level': entropy_level,
                    'shock_magnitude': shock_magnitude,
                    'recovery_rate': recovery_rate,
                    'avg_shock': np.mean(list(self.shock_memory)) if self.shock_memory else 0.0,
                    'utility_trend': np.polyfit(range(len(self.utility_memory)), 
                                               list(self.utility_memory), 1)[0] if len(self.utility_memory) > 2 else 0.0
                }
            )
            
            self.update_history(metric)
            return metric
            
        except Exception as e:
            logger.error(f"Survivability calculation error: {e}")
            return PrincipleMetrics(value=0.0, confidence=0.0, 
                                  metadata={'error': str(e)})

class ContinuityPrinciple(BasePrinciple):
    """
    Principle 7: Continuity
    Mathematical Model: Integral memory
    
    C = (1/T) ∫[t-T to t] ψ(τ) dτ
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.window_size = config.get('continuity_window', 50)
        self.coherence_buffer = deque(maxlen=self.window_size)
        
    def calculate(self, context: Dict[str, Any]) -> PrincipleMetrics:
        """Calculate continuity using sliding-window coherence"""
        try:
            # Get continuity metrics
            coherence = context.get('coherence', 0.0)
            stability = context.get('stability', 0.0)
            uptime_ratio = context.get('uptime_ratio', 1.0)
            
            self.coherence_buffer.append(coherence)
            
            # Calculate integral memory (sliding window average)
            if self.coherence_buffer:
                integral_memory = np.mean(list(self.coherence_buffer))
            else:
                integral_memory = 0.0
            
            # Factor in stability and uptime
            continuity = 0.6 * integral_memory + 0.2 * stability + 0.2 * uptime_ratio
            
            # Penalize large fluctuations in coherence
            if len(self.coherence_buffer) > 10:
                coherence_std = np.std(list(self.coherence_buffer))
                fluctuation_penalty = min(0.3, coherence_std)
                continuity -= fluctuation_penalty
                continuity = max(0.0, continuity)
            
            confidence = min(1.0, len(self.coherence_buffer) / (self.window_size * 0.5))
            
            metric = PrincipleMetrics(
                value=continuity,
                confidence=confidence,
                threshold=self.config.get('continuity_threshold', 0.6),
                metadata={
                    'coherence': coherence,
                    'stability': stability,
                    'uptime_ratio': uptime_ratio,
                    'integral_memory': integral_memory,
                    'coherence_std': np.std(list(self.coherence_buffer)) if len(self.coherence_buffer) > 1 else 0.0,
                    'buffer_fill': len(self.coherence_buffer) / self.window_size
                }
            )
            
            self.update_history(metric)
            return metric
            
        except Exception as e:
            logger.error(f"Continuity calculation error: {e}")
            return PrincipleMetrics(value=0.0, confidence=0.0, 
                                  metadata={'error': str(e)})

class TranscendencePrinciple(BasePrinciple):
    """
    Principle 8: Transcendence
    Mathematical Model: Recursive convergence
    
    lim[n→∞] Φ^(n)(ψ₀) = ψ*
    ||Φ^(n+1) - Φ^(n)|| < δ
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.convergence_threshold = config.get('convergence_threshold', 0.01)
        self.iteration_history = deque(maxlen=20)
        self.fixed_point_target = config.get('fixed_point_target', 0.8)
        
    def calculate(self, context: Dict[str, Any]) -> PrincipleMetrics:
        """Calculate transcendence using fixed-point convergence"""
        try:
            # Get transcendence metrics
            current_state = context.get('optimization_state', 0.0)
            learning_rate = context.get('learning_rate', 0.1)
            improvement_rate = context.get('improvement_rate', 0.0)
            
            # Simple fixed-point iteration: Φ(x) = x + α(target - x)
            if self.iteration_history:
                last_state = self.iteration_history[-1]
                new_state = last_state + learning_rate * (self.fixed_point_target - last_state)
            else:
                new_state = current_state
            
            self.iteration_history.append(new_state)
            
            # Check convergence
            if len(self.iteration_history) >= 2:
                convergence_delta = abs(self.iteration_history[-1] - self.iteration_history[-2])
                is_converging = convergence_delta < self.convergence_threshold
                
                # Calculate convergence rate
                if len(self.iteration_history) >= 5:
                    recent_deltas = [abs(self.iteration_history[i] - self.iteration_history[i-1]) 
                                   for i in range(-4, 0)]
                    avg_delta = np.mean(recent_deltas)
                    convergence_rate = max(0.0, 1.0 - avg_delta / self.convergence_threshold)
                else:
                    convergence_rate = 0.5
            else:
                convergence_rate = 0.0
                is_converging = False
            
            # Factor in improvement rate and proximity to fixed point
            distance_to_target = abs(new_state - self.fixed_point_target)
            proximity_factor = max(0.0, 1.0 - distance_to_target)
            
            transcendence = 0.4 * convergence_rate + 0.4 * proximity_factor + 0.2 * min(1.0, improvement_rate)
            
            confidence = min(1.0, len(self.iteration_history) / 10.0)
            
            metric = PrincipleMetrics(
                value=transcendence,
                confidence=confidence,
                threshold=self.config.get('transcendence_threshold', 0.7),
                metadata={
                    'current_state': current_state,
                    'new_state': new_state,
                    'convergence_rate': convergence_rate,
                    'is_converging': is_converging,
                    'distance_to_target': distance_to_target,
                    'improvement_rate': improvement_rate,
                    'iterations': len(self.iteration_history)
                }
            )
            
            self.update_history(metric)
            return metric
            
        except Exception as e:
            logger.error(f"Transcendence calculation error: {e}")
            return PrincipleMetrics(value=0.0, confidence=0.0, 
                                  metadata={'error': str(e)})

class SustainmentCalculator:
    """Main calculator for all 8 sustainment principles"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize principle calculators
        self.principles = {
            'integration': IntegrationPrinciple(config.get('integration', {})),
            'anticipation': AnticipationPrinciple(config.get('anticipation', {})),
            'responsiveness': ResponsivenessPrinciple(config.get('responsiveness', {})),
            'simplicity': SimplicityPrinciple(config.get('simplicity', {})),
            'economy': EconomyPrinciple(config.get('economy', {})),
            'survivability': SurvivabilityPrinciple(config.get('survivability', {})),
            'continuity': ContinuityPrinciple(config.get('continuity', {})),
            'transcendence': TranscendencePrinciple(config.get('transcendence', {}))
        }
        
        self.calculation_history = deque(maxlen=100)
        
    def calculate_all(self, context: Dict[str, Any]) -> SustainmentState:
        """Calculate all 8 principles and return complete state"""
        state = SustainmentState()
        
        for name, calculator in self.principles.items():
            try:
                metric = calculator.calculate(context)
                setattr(state, name, metric)
            except Exception as e:
                logger.error(f"Error calculating {name}: {e}")
                setattr(state, name, PrincipleMetrics(value=0.0, confidence=0.0, 
                                                    metadata={'error': str(e)}))
        
        # Store calculation for history
        self.calculation_history.append({
            'timestamp': time.time(),
            'state': state,
            'composite_score': state.composite_score()
        })
        
        return state
    
    def get_integration_weights(self) -> Dict[str, float]:
        """Get current integration weights for subsystem allocation"""
        integration_calc = self.principles['integration']
        return getattr(integration_calc, 'subsystem_weights', {})
    
    def get_anticipation_prediction(self) -> Dict[str, Any]:
        """Get current anticipation prediction"""
        anticipation_calc = self.principles['anticipation']
        if anticipation_calc.prediction_buffer:
            return anticipation_calc.prediction_buffer[-1]
        return {}
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report"""
        if not self.calculation_history:
            return {'status': 'no_data'}
        
        latest = self.calculation_history[-1]
        state = latest['state']
        
        return {
            'timestamp': latest['timestamp'],
            'composite_score': latest['composite_score'],
            'healthy_principles': sum(1 for name in ['integration', 'anticipation', 'responsiveness', 
                                                   'simplicity', 'economy', 'survivability', 
                                                   'continuity', 'transcendence']
                                    if getattr(state, name).is_healthy()),
            'failing_principles': state.failing_principles(),
            'principle_values': {name: getattr(state, name).value 
                               for name in ['integration', 'anticipation', 'responsiveness', 
                                          'simplicity', 'economy', 'survivability', 
                                          'continuity', 'transcendence']},
            'confidence_levels': {name: getattr(state, name).confidence 
                                for name in ['integration', 'anticipation', 'responsiveness', 
                                           'simplicity', 'economy', 'survivability', 
                                           'continuity', 'transcendence']},
            'overall_health': 'healthy' if latest['composite_score'] > 0.7 else 
                            'degraded' if latest['composite_score'] > 0.4 else 'critical'
        }
    
    def get_performance_trends(self, window: int = 20) -> Dict[str, Any]:
        """Get performance trends over recent history"""
        if len(self.calculation_history) < 2:
            return {'status': 'insufficient_data'}
        
        recent_history = list(self.calculation_history)[-window:]
        composite_scores = [h['composite_score'] for h in recent_history]
        
        # Calculate trends
        if len(composite_scores) > 2:
            trend_slope = np.polyfit(range(len(composite_scores)), composite_scores, 1)[0]
            trend_direction = 'improving' if trend_slope > 0.01 else 'declining' if trend_slope < -0.01 else 'stable'
        else:
            trend_slope = 0.0
            trend_direction = 'unknown'
        
        return {
            'trend_slope': trend_slope,
            'trend_direction': trend_direction,
            'current_score': composite_scores[-1],
            'avg_score': np.mean(composite_scores),
            'score_volatility': np.std(composite_scores),
            'samples': len(composite_scores)
        } 