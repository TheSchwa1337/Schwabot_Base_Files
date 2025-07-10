#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧮 MATH PACKAGE - Unified Mathematical Framework & Decision Engine
=================================================================

This package provides comprehensive mathematical operations for trading:
- Unified Tensor Algebra for advanced tensor operations
- Mathematical decision engine for market analysis
- Tensor-based market entry/exit/hold decisions
- Mathematical consensus and signal aggregation

Core Components:
- UnifiedTensorAlgebra: Advanced tensor operations with GPU support
- MathematicalDecisionEngine: Market decisions based on tensor analysis
- TensorSignalProcessor: Processing of tensor-based signals
- MathematicalConsensus: Consensus building across mathematical modules
"""

import logging
import os
import yaml
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

import numpy as np

# Import tensor algebra subpackage
from .tensor_algebra import UnifiedTensorAlgebra

# Import other math modules
try:
    from ..advanced_tensor_algebra import AdvancedTensorAlgebra
    from ..clean_unified_math import CleanUnifiedMathSystem
    ADVANCED_MATH_AVAILABLE = True
except ImportError:
    ADVANCED_MATH_AVAILABLE = False
    AdvancedTensorAlgebra = None
    CleanUnifiedMathSystem = None

logger = logging.getLogger(__name__)


class MathDecision(Enum):
    """Mathematical-based market decision types."""
    ENTER_TENSOR_ALIGNMENT = "enter_tensor_alignment"      # Enter on tensor alignment
    ENTER_EIGENVALUE_SIGNAL = "enter_eigenvalue_signal"    # Enter on eigenvalue signal
    EXIT_TENSOR_DECOMPOSITION = "exit_tensor_decomposition"  # Exit on tensor decomposition
    HOLD_TENSOR_STABILITY = "hold_tensor_stability"        # Hold on tensor stability
    WAIT_TENSOR_CONVERGENCE = "wait_tensor_convergence"    # Wait for tensor convergence
    EMERGENCY_TENSOR_COLLAPSE = "emergency_tensor_collapse"  # Emergency exit on tensor collapse


class TensorState(Enum):
    """Tensor state classifications."""
    STABLE_TENSOR = "stable_tensor"           # Stable tensor state
    OSCILLATING_TENSOR = "oscillating_tensor" # Oscillating tensor state
    DECOMPOSING_TENSOR = "decomposing_tensor" # Decomposing tensor state
    COLLAPSING_TENSOR = "collapsing_tensor"   # Collapsing tensor state
    ALIGNING_TENSOR = "aligning_tensor"       # Aligning tensor state
    CONVERGING_TENSOR = "converging_tensor"   # Converging tensor state


@dataclass
class MathSignal:
    """Mathematical-based market signal."""
    timestamp: float
    price: float
    volume: float
    tensor_state: TensorState
    decision: MathDecision
    confidence: float
    risk_level: float
    eigenvalue_score: float
    tensor_norm: float
    cosine_similarity: float
    collapse_function: float
    fourier_transform_magnitude: float
    metadata: Dict[str, Any]


@dataclass
class MathSystemConfig:
    """Configuration for mathematical system operations."""
    # UnifiedTensorAlgebra parameters
    max_rank: int = 3
    collapse_threshold: float = 0.1
    fourier_resolution: int = 64
    gamma_shift: float = 0.1
    eigenvalue_threshold: float = 1e-6
    norm_threshold: float = 1e-8
    
    # Mathematical decision parameters
    tensor_alignment_threshold: float = 0.7
    eigenvalue_signal_threshold: float = 0.6
    tensor_decomposition_threshold: float = 0.8
    tensor_stability_threshold: float = 0.5
    tensor_convergence_threshold: float = 0.4
    
    # Consensus parameters
    consensus_threshold: float = 0.6
    min_agreement_count: int = 3
    signal_aggregation_weight: float = 0.5
    
    # Risk management
    max_risk_level: float = 0.8
    min_confidence: float = 0.3
    emergency_collapse_threshold: float = 0.95


class MathematicalDecisionEngine:
    """
    Mathematical decision engine for market analysis.
    
    Uses UnifiedTensorAlgebra and other mathematical modules to analyze
    market data and make entry/exit/hold decisions based on tensor analysis.
    """
    
    def __init__(self, config: Optional[MathSystemConfig] = None):
        """Initialize the mathematical decision engine."""
        self.config = config or MathSystemConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize UnifiedTensorAlgebra
        tensor_config = {
            'max_rank': self.config.max_rank,
            'collapse_threshold': self.config.collapse_threshold,
            'fourier_resolution': self.config.fourier_resolution,
            'gamma_shift': self.config.gamma_shift,
            'eigenvalue_threshold': self.config.eigenvalue_threshold,
            'norm_threshold': self.config.norm_threshold
        }
        self.tensor_algebra = UnifiedTensorAlgebra(tensor_config)
        
        # Initialize advanced math modules if available
        self.advanced_tensor = None
        self.clean_math = None
        
        if ADVANCED_MATH_AVAILABLE:
            try:
                self.advanced_tensor = AdvancedTensorAlgebra()
                self.clean_math = CleanUnifiedMathSystem()
                self.logger.info("Advanced math modules loaded")
            except Exception as e:
                self.logger.warning(f"Could not load advanced math modules: {e}")
        
        # State tracking
        self.signal_history: List[MathSignal] = []
        self.tensor_history: List[TensorState] = []
        self.decision_history: List[MathDecision] = []
        
        self.logger.info("Mathematical decision engine initialized")
    
    def analyze_market_mathematics(self, price_data: np.ndarray, volume_data: np.ndarray,
                                 current_price: float, current_volume: float) -> MathSignal:
        """
        Analyze market using mathematical operations.
        
        Args:
            price_data: Historical price data
            volume_data: Historical volume data
            current_price: Current market price
            current_volume: Current market volume
            
        Returns:
            MathSignal with decision and analysis
        """
        try:
            # Create tensors from market data
            price_tensor = self._create_price_tensor(price_data)
            volume_tensor = self._create_volume_tensor(volume_data)
            
            # Perform tensor operations
            tensor_result = self.tensor_algebra.perform_tensor_operation(
                'contraction', [price_tensor, volume_tensor]
            )
            
            # Compute eigenvalues and eigenvectors
            eigenvalues, eigenvectors = self.tensor_algebra.eigenvalue_decomposition(price_tensor)
            
            # Compute tensor norm
            tensor_norm = self.tensor_algebra.tensor_norm(price_tensor)
            
            # Compute cosine similarity
            cosine_similarity = self.tensor_algebra.compute_cosine_similarity(price_tensor, volume_tensor)
            
            # Compute collapse function
            collapse_function = self._compute_collapse_function(price_tensor, volume_tensor)
            
            # Compute Fourier transform
            fourier_transform = self.tensor_algebra.compute_fourier_tensor_dual_transform(price_tensor)
            fourier_magnitude = np.mean(np.abs(fourier_transform))
            
            # Determine tensor state
            tensor_state = self._classify_tensor_state(
                eigenvalues, tensor_norm, cosine_similarity, collapse_function
            )
            
            # Make mathematical decision
            decision = self._make_math_decision(
                tensor_state, eigenvalues, tensor_norm, cosine_similarity, collapse_function
            )
            
            # Calculate confidence and risk
            confidence = self._calculate_math_confidence(
                eigenvalues, tensor_norm, cosine_similarity, collapse_function
            )
            risk_level = self._calculate_math_risk(tensor_state, eigenvalues, collapse_function)
            
            # Calculate eigenvalue score
            eigenvalue_score = np.mean(np.abs(eigenvalues))
            
            # Create math signal
            signal = MathSignal(
                timestamp=0.0,  # Will be set by caller
                price=current_price,
                volume=current_volume,
                tensor_state=tensor_state,
                decision=decision,
                confidence=confidence,
                risk_level=risk_level,
                eigenvalue_score=eigenvalue_score,
                tensor_norm=tensor_norm,
                cosine_similarity=cosine_similarity,
                collapse_function=collapse_function,
                fourier_transform_magnitude=fourier_magnitude,
                metadata={
                    'eigenvalues': eigenvalues.tolist(),
                    'tensor_result_shape': tensor_result.output_shape,
                    'fourier_transform_shape': fourier_transform.shape
                }
            )
            
            # Update history
            self.signal_history.append(signal)
            self.tensor_history.append(tensor_state)
            self.decision_history.append(decision)
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error analyzing market mathematics: {e}")
            # Return default signal
            return MathSignal(
                timestamp=0.0,
                price=current_price,
                volume=current_volume,
                tensor_state=TensorState.STABLE_TENSOR,
                decision=MathDecision.WAIT_TENSOR_CONVERGENCE,
                confidence=0.0,
                risk_level=1.0,
                eigenvalue_score=0.0,
                tensor_norm=0.0,
                cosine_similarity=0.0,
                collapse_function=0.0,
                fourier_transform_magnitude=0.0,
                metadata={'error': str(e)}
            )
    
    def _create_price_tensor(self, price_data: np.ndarray) -> np.ndarray:
        """Create tensor from price data."""
        # Reshape price data into a 2D tensor
        if len(price_data) < 10:
            # Pad with zeros if insufficient data
            padded_data = np.pad(price_data, (0, 10 - len(price_data)), 'constant')
        else:
            padded_data = price_data[-10:]  # Use last 10 data points
        
        # Create 2D tensor (price x time)
        tensor = np.array(padded_data).reshape(-1, 1)
        return tensor
    
    def _create_volume_tensor(self, volume_data: np.ndarray) -> np.ndarray:
        """Create tensor from volume data."""
        # Similar to price tensor
        if len(volume_data) < 10:
            padded_data = np.pad(volume_data, (0, 10 - len(volume_data)), 'constant')
        else:
            padded_data = volume_data[-10:]
        
        tensor = np.array(padded_data).reshape(-1, 1)
        return tensor
    
    def _compute_collapse_function(self, price_tensor: np.ndarray, 
                                 volume_tensor: np.ndarray) -> float:
        """Compute collapse function from tensors."""
        try:
            # Use tensor contraction as collapse function
            contraction = self.tensor_algebra.tensor_contraction(price_tensor, volume_tensor)
            return float(np.mean(contraction))
        except Exception as e:
            self.logger.warning(f"Error computing collapse function: {e}")
            return 0.0
    
    def _classify_tensor_state(self, eigenvalues: np.ndarray, tensor_norm: float,
                             cosine_similarity: float, collapse_function: float) -> TensorState:
        """Classify tensor state based on mathematical properties."""
        eigenvalue_magnitude = np.mean(np.abs(eigenvalues))
        
        # Check for tensor collapse
        if collapse_function > self.config.emergency_collapse_threshold:
            return TensorState.COLLAPSING_TENSOR
        
        # Check for tensor decomposition
        if eigenvalue_magnitude > self.config.tensor_decomposition_threshold:
            return TensorState.DECOMPOSING_TENSOR
        
        # Check for tensor alignment
        if cosine_similarity > self.config.tensor_alignment_threshold:
            return TensorState.ALIGNING_TENSOR
        
        # Check for tensor convergence
        if tensor_norm < self.config.tensor_convergence_threshold:
            return TensorState.CONVERGING_TENSOR
        
        # Check for tensor oscillation
        if 0.3 < eigenvalue_magnitude < 0.7:
            return TensorState.OSCILLATING_TENSOR
        
        # Default to stable tensor
        return TensorState.STABLE_TENSOR
    
    def _make_math_decision(self, tensor_state: TensorState, eigenvalues: np.ndarray,
                          tensor_norm: float, cosine_similarity: float, 
                          collapse_function: float) -> MathDecision:
        """Make mathematical-based market decision."""
        eigenvalue_magnitude = np.mean(np.abs(eigenvalues))
        
        # Emergency exit on tensor collapse
        if tensor_state == TensorState.COLLAPSING_TENSOR:
            return MathDecision.EMERGENCY_TENSOR_COLLAPSE
        
        # Exit on tensor decomposition
        if tensor_state == TensorState.DECOMPOSING_TENSOR:
            return MathDecision.EXIT_TENSOR_DECOMPOSITION
        
        # Enter on tensor alignment
        if tensor_state == TensorState.ALIGNING_TENSOR:
            return MathDecision.ENTER_TENSOR_ALIGNMENT
        
        # Enter on eigenvalue signal
        if (eigenvalue_magnitude > self.config.eigenvalue_signal_threshold and
            cosine_similarity > 0.5):
            return MathDecision.ENTER_EIGENVALUE_SIGNAL
        
        # Hold on tensor stability
        if tensor_state == TensorState.STABLE_TENSOR:
            return MathDecision.HOLD_TENSOR_STABILITY
        
        # Wait for tensor convergence
        if tensor_state == TensorState.CONVERGING_TENSOR:
            return MathDecision.WAIT_TENSOR_CONVERGENCE
        
        # Default to waiting
        return MathDecision.WAIT_TENSOR_CONVERGENCE
    
    def _calculate_math_confidence(self, eigenvalues: np.ndarray, tensor_norm: float,
                                 cosine_similarity: float, collapse_function: float) -> float:
        """Calculate confidence based on mathematical properties."""
        eigenvalue_magnitude = np.mean(np.abs(eigenvalues))
        
        # Base confidence from tensor norm
        confidence = min(1.0, tensor_norm)
        
        # Adjust based on cosine similarity
        confidence *= (0.5 + 0.5 * cosine_similarity)
        
        # Adjust based on eigenvalue magnitude
        if eigenvalue_magnitude > 0.5:
            confidence *= 1.2
        elif eigenvalue_magnitude < 0.2:
            confidence *= 0.8
        
        # Penalize high collapse function
        if collapse_function > 0.5:
            confidence *= (1.0 - collapse_function)
        
        return max(0.0, min(1.0, confidence))
    
    def _calculate_math_risk(self, tensor_state: TensorState, eigenvalues: np.ndarray,
                           collapse_function: float) -> float:
        """Calculate risk level based on mathematical properties."""
        eigenvalue_magnitude = np.mean(np.abs(eigenvalues))
        
        # Base risk from eigenvalue magnitude
        risk = eigenvalue_magnitude
        
        # Adjust based on tensor state
        if tensor_state == TensorState.COLLAPSING_TENSOR:
            risk *= 2.0
        elif tensor_state == TensorState.DECOMPOSING_TENSOR:
            risk *= 1.5
        elif tensor_state == TensorState.OSCILLATING_TENSOR:
            risk *= 1.2
        elif tensor_state == TensorState.STABLE_TENSOR:
            risk *= 0.8
        
        # Adjust based on collapse function
        risk *= (1.0 + collapse_function)
        
        return max(0.0, min(1.0, risk))
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get mathematical system status."""
        return {
            'tensor_algebra_status': self.tensor_algebra.get_algebra_summary(),
            'signal_count': len(self.signal_history),
            'recent_decisions': self.decision_history[-10:] if self.decision_history else [],
            'tensor_states': [state.value for state in self.tensor_history[-10:]] if self.tensor_history else [],
            'advanced_math_available': ADVANCED_MATH_AVAILABLE,
            'config': {
                'max_rank': self.config.max_rank,
                'collapse_threshold': self.config.collapse_threshold,
                'tensor_alignment_threshold': self.config.tensor_alignment_threshold,
                'eigenvalue_signal_threshold': self.config.eigenvalue_signal_threshold
            }
        }


class MathSystemFactory:
    """Factory for creating mathematical system instances."""
    
    @staticmethod
    def create_from_config(config_path: Optional[str] = None) -> MathematicalDecisionEngine:
        """Create mathematical system from configuration file."""
        config = MathSystemFactory._load_config(config_path)
        return MathematicalDecisionEngine(config)
    
    @staticmethod
    def create_with_params(**kwargs) -> MathematicalDecisionEngine:
        """Create mathematical system with custom parameters."""
        config = MathSystemConfig(**kwargs)
        return MathematicalDecisionEngine(config)
    
    @staticmethod
    def _load_config(config_path: Optional[str] = None) -> MathSystemConfig:
        """Load configuration from file."""
        if config_path is None:
            # Try to find default config
            default_paths = [
                "config/math_system_config.yaml",
                "config/schwabot_config.yaml"
            ]
            
            for path in default_paths:
                if os.path.exists(path):
                    config_path = path
                    break
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                # Extract math system config
                math_config = config_data.get('math_system', {})
                return MathSystemConfig(**math_config)
                
            except Exception as e:
                logger.warning(f"Could not load math system config from {config_path}: {e}")
        
        # Return default config
        return MathSystemConfig()


# Auto-load mathematical functions registry if available
MATH_FUNCTIONS_REGISTRY = {}

try:
    registry_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'mathematical_functions_registry.yaml')
    if os.path.exists(registry_path):
        with open(registry_path, 'r') as f:
            registry_data = yaml.safe_load(f)
            math_functions = registry_data.get('mathematical_functions', {})
            MATH_FUNCTIONS_REGISTRY.update(math_functions)
except Exception as e:
    logger.warning(f"Could not load math functions registry: {e}")


# Export main classes and functions
__all__ = [
    "UnifiedTensorAlgebra",
    "MathematicalDecisionEngine",
    "MathSystemConfig",
    "MathSystemFactory",
    "MathDecision",
    "TensorState",
    "MathSignal",
    "MATH_FUNCTIONS_REGISTRY",
    "ADVANCED_MATH_AVAILABLE"
]

# Add advanced math classes if available
if ADVANCED_MATH_AVAILABLE:
    __all__.extend(["AdvancedTensorAlgebra", "CleanUnifiedMathSystem"])

# Convenience functions for quick access
def create_tensor_algebra(*args, **kwargs) -> UnifiedTensorAlgebra:
    """Factory for UnifiedTensorAlgebra."""
    return UnifiedTensorAlgebra(*args, **kwargs)

def create_math_system(*args, **kwargs) -> MathematicalDecisionEngine:
    """Factory for MathematicalDecisionEngine."""
    return MathematicalDecisionEngine(*args, **kwargs)

def analyze_market_mathematics(price_data: np.ndarray, volume_data: np.ndarray,
                             current_price: float, current_volume: float,
                             config: Optional[MathSystemConfig] = None) -> MathSignal:
    """Quick function to analyze market mathematics."""
    math_system = MathematicalDecisionEngine(config)
    return math_system.analyze_market_mathematics(price_data, volume_data, current_price, current_volume) 