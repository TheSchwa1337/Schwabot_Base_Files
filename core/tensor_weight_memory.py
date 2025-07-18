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
import numpy as np

# Import existing Schwabot components
try:
    from .neural_processing_engine import NeuralProcessingEngine
    from .orbital_shell_brain_system import OrbitalShell
    from .quantum_mathematical_bridge import QuantumMathematicalBridge
    SCHWABOT_COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Some Schwabot components not available: {e}")
    SCHWABOT_COMPONENTS_AVAILABLE = False

try:
    from core.backend_math import backend_info, get_backend
    xp = get_backend()
except ImportError:
    xp = np
    print("⚠️ Using NumPy fallback for tensor operations")

logger = logging.getLogger(__name__)

# Log backend status
try:
    backend_status = backend_info()
    if backend_status["accelerated"]:
        logger.info("🧠 TensorWeightMemory using GPU acceleration: CuPy (GPU)")
    else:
        logger.info("🧠 TensorWeightMemory using CPU fallback: NumPy (CPU)")
except:
    logger.info("🧠 TensorWeightMemory using NumPy fallback")

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
    shell_weights: np.ndarray  # Wₜ [8] - weights for each shell
    hash_entropy: np.ndarray  # Hₜ [64] - hash entropy vector
    success_scores: np.ndarray  # Sₜ [8] - success/failure scores
    phi_tensor: np.ndarray  # Φₜ [8] - combined tensor state
    confidence: float
    update_mode: MemoryUpdateMode
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WeightUpdateResult:
    """Result of weight update operation"""
    new_weights: np.ndarray
    weight_delta: np.ndarray
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
    
    def __init__(self, config: Dict[str, Any] = None) -> None:
        self.config = config or self._default_config()
        
        # Initialize weight tensors
        self.shell_weights = xp.ones(8) * 0.5  # Wₜ - initial weights
        self.weight_history: List[np.ndarray] = []
        self.memory_tensors: List[MemoryTensor] = []
        
        # Neural processing components
        if SCHWABOT_COMPONENTS_AVAILABLE:
            self.quantum_bridge = QuantumMathematicalBridge(quantum_dimension=16)
            self.neural_engine = NeuralProcessingEngine()
        else:
            self.quantum_bridge = None
            self.neural_engine = None
        
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
        hash_entropy: np.ndarray,
        current_shell: Any,  # OrbitalShell or similar
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
        try:
            with self.memory_lock:
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
                        "shell": getattr(current_shell, 'name', 'unknown'),
                        "trade_result": trade_result,
                    },
                )
        except Exception as e:
            logger.error(f"❌ Weight update failed: {e}")
            return self._get_fallback_result()
    
    def _calculate_success_score(self, trade_result: Dict[str, Any]) -> float:
        """Calculate success score from trade result"""
        try:
            profit = trade_result.get('profit', 0.0)
            duration = trade_result.get('duration', 1.0)
            risk_adjustment = trade_result.get('risk_adjustment', 1.0)
            
            # Normalize profit by duration and risk
            normalized_profit = profit / (duration * risk_adjustment)
            
            # Apply sigmoid activation for bounded output
            success_score = 1.0 / (1.0 + xp.exp(-normalized_profit))
            
            return float(success_score)
        except Exception as e:
            logger.error(f"❌ Success score calculation failed: {e}")
            return 0.5
    
    def _determine_update_mode(self, success_score: float, trade_result: Dict[str, Any]) -> MemoryUpdateMode:
        """Determine update mode based on success score and trade result"""
        if success_score > 0.7:
            return MemoryUpdateMode.REINFORCEMENT
        elif success_score < 0.3:
            return MemoryUpdateMode.DECAY
        else:
            return MemoryUpdateMode.NEUTRAL
    
    def _calculate_weight_gradient(
        self, success_score: float, hash_entropy: np.ndarray, current_shell: Any
    ) -> np.ndarray:
        """Calculate weight gradient using mathematical framework"""
        try:
            # Extract shell index
            shell_index = getattr(current_shell, 'index', 0)
            
            # Calculate gradient components
            success_gradient = (success_score - 0.5) * xp.ones(8)
            entropy_gradient = xp.mean(hash_entropy) * xp.ones(8)
            
            # Combine gradients with weights
            combined_gradient = (
                self.config["learning_rate"] * success_gradient +
                self.config["entropy_weight"] * entropy_gradient
            )
            
            # Apply shell-specific weighting
            combined_gradient[shell_index] *= 2.0  # Higher weight for active shell
            
            return combined_gradient
        except Exception as e:
            logger.error(f"❌ Gradient calculation failed: {e}")
            return xp.zeros(8)
    
    def _apply_weight_update(self, gradient: np.ndarray, update_mode: MemoryUpdateMode) -> np.ndarray:
        """Apply weight update with bounds and momentum"""
        try:
            # Calculate new weights
            new_weights = self.shell_weights + gradient
            
            # Apply momentum if available
            if len(self.weight_history) > 0:
                momentum = self.config["momentum"] * (self.shell_weights - self.weight_history[-1])
                new_weights += momentum
            
            # Apply weight decay
            new_weights *= (1.0 - self.config["decay_rate"])
            
            # Apply bounds
            min_weight, max_weight = self.config["weight_bounds"]
            new_weights = xp.clip(new_weights, min_weight, max_weight)
            
            return new_weights
        except Exception as e:
            logger.error(f"❌ Weight update application failed: {e}")
            return self.shell_weights
    
    def _create_memory_tensor(
        self,
        weights: np.ndarray,
        hash_entropy: np.ndarray,
        success_score: float,
        update_mode: MemoryUpdateMode,
    ) -> MemoryTensor:
        """Create memory tensor ℳₜ"""
        try:
            # Calculate Φₜ tensor
            phi_tensor = xp.zeros(8)
            for i in range(8):
                psi_state = self._calculate_psi_state(i, hash_entropy)
                phi_tensor[i] = psi_state * weights[i]
            
            # Calculate confidence
            confidence = xp.mean(phi_tensor) * success_score
            
            # Create success scores array
            success_scores = xp.ones(8) * success_score
            
            return MemoryTensor(
                timestamp=time.time(),
                shell_weights=weights,
                hash_entropy=hash_entropy,
                success_scores=success_scores,
                phi_tensor=phi_tensor,
                confidence=float(confidence),
                update_mode=update_mode,
                metadata={"update_count": self.update_count}
            )
        except Exception as e:
            logger.error(f"❌ Memory tensor creation failed: {e}")
            return self._get_empty_memory_tensor()
    
    def _calculate_psi_state(self, shell_index: int, hash_entropy: np.ndarray) -> float:
        """Calculate ψ state for given shell"""
        try:
            # Extract relevant entropy components
            entropy_slice = hash_entropy[shell_index * 8:(shell_index + 1) * 8]
            
            # Calculate ψ state using quantum bridge if available
            if self.quantum_bridge:
                psi_state = self.quantum_bridge.calculate_psi_state(entropy_slice, shell_index)
            else:
                # Fallback calculation
                psi_state = xp.mean(entropy_slice) * xp.sin(shell_index * xp.pi / 4)
            
            return float(psi_state)
        except Exception as e:
            logger.error(f"❌ Psi state calculation failed: {e}")
            return 0.5
    
    def _update_system_state(self, memory_tensor: MemoryTensor, new_weights: np.ndarray) -> None:
        """Update system state with new memory tensor"""
        try:
            # Store memory tensor
            self.memory_tensors.append(memory_tensor)
            
            # Store weight history
            self.weight_history.append(new_weights.copy())
            
            # Limit memory size
            max_size = self.config["max_memory_size"]
            if len(self.memory_tensors) > max_size:
                self.memory_tensors = self.memory_tensors[-max_size:]
                self.weight_history = self.weight_history[-max_size:]
            
            # Update performance metrics
            self._update_performance_metrics(memory_tensor)
            
            # Update counters
            self.update_count += 1
            self.last_update_time = time.time()
            
        except Exception as e:
            logger.error(f"❌ System state update failed: {e}")
    
    def _update_performance_metrics(self, memory_tensor: MemoryTensor) -> None:
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
            self.performance_metrics["average_confidence"] = (
                (current_avg * (total_updates - 1) + memory_tensor.confidence) / total_updates
            )
            
            # Calculate weight stability
            if len(self.weight_history) > 1:
                weight_variance = xp.var(self.weight_history[-1] - self.weight_history[-2])
                self.performance_metrics["weight_stability"] = 1.0 / (1.0 + weight_variance)
            
        except Exception as e:
            logger.error(f"❌ Performance metrics update failed: {e}")
    
    def calculate_consensus_altitude(self, phi_tensor: np.ndarray, memory_tensor: MemoryTensor) -> ConsensusAltitudeResult:
        """
        Calculate consensus altitude for trade pipeline control
        
        Args:
            phi_tensor: Current Φₜ tensor
            memory_tensor: Current memory tensor ℳₜ
            
        Returns:
            ConsensusAltitudeResult with altitude value and trade decision
        """
        try:
            # Calculate cosine similarity between current and memory tensors
            current_norm = xp.linalg.norm(phi_tensor)
            memory_norm = xp.linalg.norm(memory_tensor.phi_tensor)
            
            if current_norm == 0 or memory_norm == 0:
                altitude_value = 0.0
            else:
                similarity = xp.dot(phi_tensor, memory_tensor.phi_tensor) / (current_norm * memory_norm)
                altitude_value = float(similarity)
            
            # Determine consensus
            consensus_met = altitude_value >= self.config["altitude_threshold"]
            
            # Find active shells (weights above threshold)
            active_shells = [i for i, w in enumerate(memory_tensor.shell_weights) if w > 0.6]
            
            # Calculate confidence level
            confidence_level = memory_tensor.confidence * altitude_value
            
            # Determine if trade is allowed
            trade_allowed = consensus_met and confidence_level >= self.config["consensus_threshold"]
            
            return ConsensusAltitudeResult(
                altitude_value=altitude_value,
                consensus_met=consensus_met,
                active_shells=active_shells,
                confidence_level=confidence_level,
                trade_allowed=trade_allowed,
                metadata={
                    "phi_tensor_norm": float(current_norm),
                    "memory_tensor_norm": float(memory_norm),
                    "similarity": altitude_value
                }
            )
            
        except Exception as e:
            logger.error(f"❌ Consensus altitude calculation failed: {e}")
            return ConsensusAltitudeResult(
                altitude_value=0.0,
                consensus_met=False,
                active_shells=[],
                confidence_level=0.0,
                trade_allowed=False,
                metadata={"error": str(e)}
            )
    
    def _get_previous_confidence(self) -> float:
        """Get confidence from previous memory tensor"""
        try:
            if self.memory_tensors:
                return self.memory_tensors[-1].confidence
            return 0.5
        except Exception as e:
            logger.error(f"❌ Previous confidence retrieval failed: {e}")
            return 0.5
    
    def _get_fallback_result(self) -> WeightUpdateResult:
        """Get fallback result when update fails"""
        return WeightUpdateResult(
            new_weights=self.shell_weights,
            weight_delta=xp.zeros(8),
            entropy_contribution=0.0,
            success_contribution=0.5,
            confidence_change=0.0,
            update_mode=MemoryUpdateMode.NEUTRAL,
            metadata={"error": "fallback_result"}
        )
    
    def _get_empty_memory_tensor(self) -> MemoryTensor:
        """Get empty memory tensor for fallback"""
        return MemoryTensor(
            timestamp=time.time(),
            shell_weights=self.shell_weights,
            hash_entropy=xp.zeros(64),
            success_scores=xp.ones(8) * 0.5,
            phi_tensor=xp.zeros(8),
            confidence=0.5,
            update_mode=MemoryUpdateMode.NEUTRAL,
            metadata={"error": "empty_tensor"}
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
                "backend": "numpy",  # Simplified for compatibility
                "accelerated": False,  # Simplified for compatibility
            }
        except Exception as e:
            logger.error(f"❌ System status retrieval failed: {e}")
            return {"error": str(e)}
    
    def start_memory_system(self) -> None:
        """Start the memory system"""
        self.active = True
        logger.info("🧠⚛️ TensorWeightMemory system started")
    
    def stop_memory_system(self) -> None:
        """Stop the memory system"""
        self.active = False
        logger.info("🧠⚛️ TensorWeightMemory system stopped")
    
    # Compatibility methods for KoboldCPP integration
    def store_tensor_score(self, symbol: str, score: float, metadata: Optional[Dict[str, Any]] = None):
        """Store a tensor score for a symbol (compatibility method)"""
        try:
            # Convert to internal format
            trade_result = {
                'profit': score * 100,  # Scale score to profit-like value
                'duration': 1.0,
                'risk_adjustment': 1.0
            }
            
            # Create dummy hash entropy
            hash_entropy = xp.random.random(64)
            
            # Create dummy shell
            class DummyShell:
                def __init__(self, name):
                    self.name = name
                    self.index = 0
            
            dummy_shell = DummyShell(symbol)
            
            # Update weights
            result = self.update_shell_weights(trade_result, hash_entropy, dummy_shell, f"kobold_{symbol}")
            
            logger.info(f"🧠 Stored tensor score for {symbol}: {score}")
            
        except Exception as e:
            logger.error(f"❌ Error storing tensor score: {e}")
    
    def get_tensor_score(self, symbol: str) -> Optional[float]:
        """Get the latest tensor score for a symbol (compatibility method)"""
        try:
            if self.memory_tensors:
                # Return the average of the latest phi tensor
                latest_phi = self.memory_tensors[-1].phi_tensor
                return float(xp.mean(latest_phi))
            return None
        except Exception as e:
            logger.error(f"❌ Error retrieving tensor score: {e}")
            return None
    
    def get_average_tensor_score(self, symbol: str, window: int = 10) -> Optional[float]:
        """Get average tensor score over a window (compatibility method)"""
        try:
            if len(self.memory_tensors) >= window:
                recent_tensors = self.memory_tensors[-window:]
                scores = [float(xp.mean(tensor.phi_tensor)) for tensor in recent_tensors]
                return sum(scores) / len(scores)
            return None
        except Exception as e:
            logger.error(f"❌ Error calculating average tensor score: {e}")
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics (compatibility method)"""
        try:
            status = self.get_system_status()
            return {
                'total_stores': status.get('update_count', 0),
                'total_retrievals': status.get('update_count', 0),
                'symbols_tracked': 1,  # Simplified
                'total_scores': len(self.memory_tensors),
                'enabled': status.get('active', False)
            }
        except Exception as e:
            logger.error(f"❌ Error getting statistics: {e}")
            return {'error': str(e)}