#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠⚛️ TENSOR WEIGHT MEMORY SYSTEM
================================

Neural Memory Tensor Weight Evaluation System for Schwabot

This module provides:
1. Real-time updates to shell activation weights (Shell_1 → Shell_8)
2. Dynamic feedback based on trade success/failure + hash entropy vectors
3. Integration with consensus_altitude() function for trade pipeline control
4. Memory feedback loops for Φ_tensors and ψ states

Mathematical Framework:
- Wₜ₊₁ = Wₜ + α·∇L(Wₜ) + β·Hₑ·Sₜ
- Φₜ = Σ(ψₙₜ · Wₙₜ) for n=1 to 8
- ℳₜ = [Wₜ, Hₜ, Sₜ, Φₜ] memory tensor
- 𝒞ₐ = consensus_altitude(Φₜ, ℳₜ, threshold)
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

# Import existing Schwabot components
try:
    from .neural_processing_engine import NeuralProcessingEngine
    from .orbital_shell_brain_system import OrbitalShell
    from .quantum_mathematical_bridge import QuantumMathematicalBridge

    SCHWABOT_COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Some Schwabot components not available: {e}")
    SCHWABOT_COMPONENTS_AVAILABLE = False

from core.backend_math import backend_info, get_backend

xp = get_backend()

logger = logging.getLogger(__name__)

# Log backend status
backend_status = backend_info()
if backend_status["accelerated"]:
    logger.info("🧠 TensorWeightMemory using GPU acceleration: CuPy (GPU)")
else:
    logger.info("🧠 TensorWeightMemory using CPU fallback: NumPy (CPU)")


class MemoryUpdateMode(Enum):
    """Modes for memory tensor updates"""

    REINFORCEMENT = "reinforcement"  # Positive feedback
    DECAY = "decay"  # Negative feedback
    NEUTRAL = "neutral"  # No change
    ADAPTIVE = "adaptive"  # Dynamic adjustment


class WeightUpdateStrategy(Enum):
    """Strategies for weight updates"""

    GRADIENT_DESCENT = "gradient_descent"
    MOMENTUM = "momentum"
    ADAM = "adam"
    ENTROPY_DRIVEN = "entropy_driven"
    ORBITAL_ADAPTIVE = "orbital_adaptive"


@dataclass
class MemoryTensor:
    """Memory tensor ℳₜ for storing shell weights and history"""

    timestamp: float
    shell_weights: xp.ndarray  # Wₜ [8] - weights for each shell
    hash_entropy: xp.ndarray  # Hₜ [64] - hash entropy vector
    success_scores: xp.ndarray  # Sₜ [8] - success/failure scores
    phi_tensor: xp.ndarray  # Φₜ [8] - combined tensor state
    confidence: float
    update_mode: MemoryUpdateMode
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WeightUpdateResult:
    """Result of weight update operation"""

    new_weights: xp.ndarray
    weight_delta: xp.ndarray
    entropy_contribution: float
    success_contribution: float
    confidence_change: float
    update_mode: MemoryUpdateMode
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsensusAltitudeResult:
    """Result of consensus altitude calculation"""

    altitude_value: float
    consensus_met: bool
    active_shells: List[int]
    confidence_level: float
    trade_allowed: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


class TensorWeightMemory:
    """
    🧠⚛️ Neural Memory Tensor Weight Evaluation System

    Provides real-time updates to shell activation weights based on:
    - Trade success/failure history
    - Hash entropy vector analysis
    - Dynamic feedback loops
    - Integration with consensus_altitude() function
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()

        # Initialize weight tensors
        self.shell_weights = xp.ones(8) * 0.5  # Wₜ - initial weights
        self.weight_history: List[xp.ndarray] = []
        self.memory_tensors: List[MemoryTensor] = []

        # Neural processing components
        if SCHWABOT_COMPONENTS_AVAILABLE:
            self.quantum_bridge = QuantumMathematicalBridge(quantum_dimension=16)
            self.neural_engine = NeuralProcessingEngine()

        # Performance tracking
        self.update_count = 0
        self.last_update_time = time.time()
        self.performance_metrics = {
            "total_updates": 0,
            "reinforcement_updates": 0,
            "decay_updates": 0,
            "average_confidence": 0.0,
            "weight_stability": 0.0,
        }

        # Threading
        self.memory_lock = threading.Lock()
        self.active = False

        logger.info("🧠⚛️ TensorWeightMemory initialized")

    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            "learning_rate": 0.01,  # α - learning rate
            "entropy_weight": 0.3,  # β - entropy contribution weight
            "momentum": 0.9,  # μ - momentum coefficient
            "decay_rate": 0.001,  # λ - weight decay rate
            "consensus_threshold": 0.75,
            "altitude_threshold": 0.6,
            "max_memory_size": 1000,
            "update_interval": 1.0,  # seconds
            "weight_bounds": (0.1, 0.9),  # min/max weight values
        }

    def update_shell_weights(
        self,
        trade_result: Dict[str, Any],
        hash_entropy: xp.ndarray,
        current_shell: OrbitalShell,
        strategy_id: str,
    ) -> WeightUpdateResult:
        """
        Update shell weights based on trade result and entropy

        Args:
            trade_result: Trade outcome with profit/loss data
            hash_entropy: Current hash entropy vector Hₜ
            current_shell: Shell that executed the trade
            strategy_id: Strategy identifier for tracking

        Returns:
            WeightUpdateResult with updated weights and metadata
        """
        with self.memory_lock:
            try:
                # Calculate success score
                success_score = self._calculate_success_score(trade_result)

                # Determine update mode
                update_mode = self._determine_update_mode(success_score, trade_result)

                # Calculate weight gradients
                weight_gradient = self._calculate_weight_gradient(success_score, hash_entropy, current_shell)

                # Apply weight update
                new_weights = self._apply_weight_update(weight_gradient, update_mode)

                # Create memory tensor
                memory_tensor = self._create_memory_tensor(new_weights, hash_entropy, success_score, update_mode)

                # Update system state
                self._update_system_state(memory_tensor, new_weights)

                # Calculate result
                weight_delta = new_weights - self.shell_weights
                self.shell_weights = new_weights

                return WeightUpdateResult(
                    new_weights=new_weights,
                    weight_delta=weight_delta,
                    entropy_contribution=xp.mean(hash_entropy),
                    success_contribution=success_score,
                    confidence_change=memory_tensor.confidence - self._get_previous_confidence(),
                    update_mode=update_mode,
                    metadata={
                        "strategy_id": strategy_id,
                        "shell": current_shell.name,
                        "trade_result": trade_result,
                    },
                )

            except Exception as e:
                logger.error(f"Error updating shell weights: {e}")
                return self._get_fallback_result()

    def consensus_altitude(
        self, phi_tensor: xp.ndarray, memory_tensor: MemoryTensor, threshold: Optional[float] = None
    ) -> ConsensusAltitudeResult:
        """
        Calculate consensus altitude for trade pipeline control

        Args:
            phi_tensor: Current Φₜ tensor
            memory_tensor: Current memory tensor ℳₜ
            threshold: Optional custom threshold

        Returns:
            ConsensusAltitudeResult with trade decision
        """
        try:
            threshold = threshold or self.config["consensus_threshold"]

            # Calculate altitude components
            momentum_curvature = self._calculate_momentum_curvature(phi_tensor)
            rolling_return = self._calculate_rolling_return(memory_tensor)
            entropy_shift = self._calculate_entropy_shift(memory_tensor.hash_entropy)
            alpha_decay = self._calculate_alpha_decay(memory_tensor)

            # Calculate altitude vector ℵₐ(t)
            altitude_value = momentum_curvature * 0.3 + rolling_return * 0.3 + entropy_shift * 0.2 + alpha_decay * 0.2

            # Determine consensus
            consensus_met = altitude_value >= threshold

            # Identify active shells
            active_shells = [i for i, weight in enumerate(memory_tensor.shell_weights) if weight > 0.6]

            # Calculate confidence
            confidence_level = xp.mean(memory_tensor.shell_weights)

            # Determine if trade is allowed
            trade_allowed = (
                consensus_met and confidence_level >= self.config["altitude_threshold"] and len(active_shells) > 0
            )

            return ConsensusAltitudeResult(
                altitude_value=altitude_value,
                consensus_met=consensus_met,
                active_shells=active_shells,
                confidence_level=confidence_level,
                trade_allowed=trade_allowed,
                metadata={
                    "momentum_curvature": momentum_curvature,
                    "rolling_return": rolling_return,
                    "entropy_shift": entropy_shift,
                    "alpha_decay": alpha_decay,
                    "threshold": threshold,
                },
            )

        except Exception as e:
            logger.error(f"Error calculating consensus altitude: {e}")
            return ConsensusAltitudeResult(
                altitude_value=0.0,
                consensus_met=False,
                active_shells=[],
                confidence_level=0.0,
                trade_allowed=False,
                metadata={"error": str(e)},
            )

    def _calculate_success_score(self, trade_result: Dict[str, Any]) -> float:
        """Calculate success score from trade result"""
        try:
            profit = trade_result.get("profit", 0.0)
            duration = trade_result.get("duration", 1.0)
            risk = trade_result.get("risk", 1.0)

            # Normalize profit by duration and risk
            normalized_profit = profit / (duration * risk + 1e-8)

            # Apply sigmoid to get score between 0 and 1
            success_score = 1.0 / (1.0 + xp.exp(-normalized_profit))

            return float(success_score)

        except Exception as e:
            logger.error(f"Error calculating success score: {e}")
            return 0.5  # Neutral score

    def _determine_update_mode(self, success_score: float, trade_result: Dict[str, Any]) -> MemoryUpdateMode:
        """Determine update mode based on success score and trade result"""
        if success_score > 0.7:
            return MemoryUpdateMode.REINFORCEMENT
        elif success_score < 0.3:
            return MemoryUpdateMode.DECAY
        else:
            return MemoryUpdateMode.NEUTRAL

    def _calculate_weight_gradient(
        self, success_score: float, hash_entropy: xp.ndarray, current_shell: OrbitalShell
    ) -> xp.ndarray:
        """Calculate weight gradient ∇L(Wₜ)"""
        try:
            # Base gradient from success score
            base_gradient = xp.zeros(8)
            base_gradient[current_shell.value] = success_score - 0.5

            # Entropy contribution
            entropy_contribution = xp.mean(hash_entropy) * xp.ones(8)

            # Combine gradients
            gradient = (
                self.config["learning_rate"] * base_gradient + self.config["entropy_weight"] * entropy_contribution
            )

            return gradient

        except Exception as e:
            logger.error(f"Error calculating weight gradient: {e}")
            return xp.zeros(8)

    def _apply_weight_update(self, gradient: xp.ndarray, update_mode: MemoryUpdateMode) -> xp.ndarray:
        """Apply weight update Wₜ₊₁ = Wₜ + α·∇L(Wₜ) + β·Hₑ·Sₜ"""
        try:
            # Apply gradient update
            new_weights = self.shell_weights + gradient

            # Apply mode-specific adjustments
            if update_mode == MemoryUpdateMode.REINFORCEMENT:
                new_weights += 0.01 * xp.ones(8)
            elif update_mode == MemoryUpdateMode.DECAY:
                new_weights -= 0.01 * xp.ones(8)

            # Apply weight decay
            new_weights *= 1.0 - self.config["decay_rate"]

            # Clamp weights to bounds
            min_weight, max_weight = self.config["weight_bounds"]
            new_weights = xp.clip(new_weights, min_weight, max_weight)

            return new_weights

        except Exception as e:
            logger.error(f"Error applying weight update: {e}")
            return self.shell_weights

    def _create_memory_tensor(
        self,
        weights: xp.ndarray,
        hash_entropy: xp.ndarray,
        success_score: float,
        update_mode: MemoryUpdateMode,
    ) -> MemoryTensor:
        """Create memory tensor ℳₜ"""
        try:
            # Calculate phi tensor Φₜ = Σ(ψₙₜ · Wₙₜ)
            phi_tensor = weights * success_score

            # Calculate confidence
            confidence = float(xp.mean(weights))

            memory_tensor = MemoryTensor(
                timestamp=time.time(),
                shell_weights=weights.copy(),
                hash_entropy=hash_entropy.copy(),
                success_scores=xp.full(8, success_score),
                phi_tensor=phi_tensor,
                confidence=confidence,
                update_mode=update_mode,
                metadata={
                    "update_count": self.update_count,
                    "performance_metrics": self.performance_metrics.copy(),
                },
            )

            return memory_tensor

        except Exception as e:
            logger.error(f"Error creating memory tensor: {e}")
            return self._get_empty_memory_tensor()

    def _update_system_state(self, memory_tensor: MemoryTensor, new_weights: xp.ndarray):
        """Update system state with new memory tensor"""
        try:
            # Add to history
            self.memory_tensors.append(memory_tensor)
            self.weight_history.append(new_weights.copy())

            # Maintain history size
            if len(self.memory_tensors) > self.config["max_memory_size"]:
                self.memory_tensors.pop(0)
                self.weight_history.pop(0)

            # Update metrics
            self.update_count += 1
            self.last_update_time = time.time()

            # Update performance metrics
            self._update_performance_metrics(memory_tensor)

        except Exception as e:
            logger.error(f"Error updating system state: {e}")

    def _update_performance_metrics(self, memory_tensor: MemoryTensor):
        """Update performance metrics"""
        try:
            self.performance_metrics["total_updates"] += 1

            if memory_tensor.update_mode == MemoryUpdateMode.REINFORCEMENT:
                self.performance_metrics["reinforcement_updates"] += 1
            elif memory_tensor.update_mode == MemoryUpdateMode.DECAY:
                self.performance_metrics["decay_updates"] += 1

            # Update average confidence
            total_updates = self.performance_metrics["total_updates"]
            current_avg = self.performance_metrics["average_confidence"]
            new_avg = (current_avg * (total_updates - 1) + memory_tensor.confidence) / total_updates
            self.performance_metrics["average_confidence"] = new_avg

            # Calculate weight stability
            if len(self.weight_history) > 1:
                weight_variance = xp.var(self.weight_history[-1] - self.weight_history[-2])
                self.performance_metrics["weight_stability"] = float(1.0 / (1.0 + weight_variance))

        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")

    def _calculate_momentum_curvature(self, phi_tensor: xp.ndarray) -> float:
        """Calculate momentum curvature ∇ψₜ"""
        try:
            if len(self.memory_tensors) < 2:
                return 0.0

            current_phi = phi_tensor
            previous_phi = self.memory_tensors[-2].phi_tensor

            curvature = float(xp.mean(current_phi - previous_phi))
            return curvature

        except Exception as e:
            logger.error(f"Error calculating momentum curvature: {e}")
            return 0.0

    def _calculate_rolling_return(self, memory_tensor: MemoryTensor) -> float:
        """Calculate rolling return ρ(t)"""
        try:
            if len(self.memory_tensors) < 5:
                return 0.0

            # Calculate rolling average of success scores
            recent_scores = [tensor.success_scores[0] for tensor in self.memory_tensors[-5:]]
            rolling_return = float(xp.mean(recent_scores))

            return rolling_return

        except Exception as e:
            logger.error(f"Error calculating rolling return: {e}")
            return 0.0

    def _calculate_entropy_shift(self, hash_entropy: xp.ndarray) -> float:
        """Calculate entropy shift εₜ"""
        try:
            if len(self.memory_tensors) < 2:
                return 0.0

            current_entropy = xp.mean(hash_entropy)
            previous_entropy = xp.mean(self.memory_tensors[-2].hash_entropy)

            entropy_shift = float(current_entropy - previous_entropy)
            return entropy_shift

        except Exception as e:
            logger.error(f"Error calculating entropy shift: {e}")
            return 0.0

    def _calculate_alpha_decay(self, memory_tensor: MemoryTensor) -> float:
        """Calculate alpha decay ∂Φ/∂t"""
        try:
            if len(self.memory_tensors) < 2:
                return 0.0

            current_phi = xp.mean(memory_tensor.phi_tensor)
            previous_phi = xp.mean(self.memory_tensors[-2].phi_tensor)

            alpha_decay = float(current_phi - previous_phi)
            return alpha_decay

        except Exception as e:
            logger.error(f"Error calculating alpha decay: {e}")
            return 0.0

    def _get_previous_confidence(self) -> float:
        """Get confidence from previous memory tensor"""
        try:
            if len(self.memory_tensors) > 0:
                return self.memory_tensors[-1].confidence
            return 0.5
        except Exception:
            return 0.5

    def _get_fallback_result(self) -> WeightUpdateResult:
        """Get fallback result when update fails"""
        return WeightUpdateResult(
            new_weights=self.shell_weights.copy(),
            weight_delta=xp.zeros(8),
            entropy_contribution=0.0,
            success_contribution=0.5,
            confidence_change=0.0,
            update_mode=MemoryUpdateMode.NEUTRAL,
            metadata={"error": "fallback_result"},
        )

    def _get_empty_memory_tensor(self) -> MemoryTensor:
        """Get empty memory tensor for fallback"""
        return MemoryTensor(
            timestamp=time.time(),
            shell_weights=xp.ones(8) * 0.5,
            hash_entropy=xp.zeros(64),
            success_scores=xp.ones(8) * 0.5,
            phi_tensor=xp.ones(8) * 0.25,
            confidence=0.5,
            update_mode=MemoryUpdateMode.NEUTRAL,
            metadata={"error": "empty_tensor"},
        )

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            return {
                "active": self.active,
                "update_count": self.update_count,
                "last_update_time": self.last_update_time,
                "current_weights": self.shell_weights.tolist(),
                "memory_size": len(self.memory_tensors),
                "performance_metrics": self.performance_metrics,
                "backend": backend_status["backend"],
                "accelerated": backend_status["accelerated"],
            }
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {"error": str(e)}

    def start_memory_system(self):
        """Start the memory system"""
        self.active = True
        logger.info("🧠⚛️ TensorWeightMemory system started")

    def stop_memory_system(self):
        """Stop the memory system"""
        self.active = False
        logger.info("🧠⚛️ TensorWeightMemory system stopped")
