#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔢 TENSOR ALGEBRA SUBPACKAGE - Advanced Tensor Operations & Decision Engine
==========================================================================

This subpackage provides advanced tensor algebra operations for trading:
- UnifiedTensorAlgebra for rank-2 and rank-3 tensor operations
- Tensor-based market analysis and decision making
- Fourier-Tensor dual transforms
- Canonical collapse tensor computations

Core Components:
- UnifiedTensorAlgebra: Main tensor algebra engine
- TensorDecisionEngine: Market decisions based on tensor analysis
- TensorSignalProcessor: Processing of tensor-based signals
"""

import logging
import os
import yaml
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

import numpy as np

# Import the main UnifiedTensorAlgebra class
from .unified_tensor_algebra import (
    UnifiedTensorAlgebra,
    TensorRank,
    TensorOperation,
    TensorResult,
    CollapseResult
)

logger = logging.getLogger(__name__)


class TensorDecision(Enum):
    """Tensor-based market decision types."""
    ENTER_TENSOR_CONTRACTION = "enter_tensor_contraction"      # Enter on tensor contraction
    ENTER_FOURIER_SIGNAL = "enter_fourier_signal"              # Enter on Fourier signal
    EXIT_TENSOR_DECOMPOSITION = "exit_tensor_decomposition"    # Exit on tensor decomposition
    HOLD_TENSOR_STABILITY = "hold_tensor_stability"            # Hold on tensor stability
    WAIT_TENSOR_CONVERGENCE = "wait_tensor_convergence"        # Wait for tensor convergence
    EMERGENCY_TENSOR_COLLAPSE = "emergency_tensor_collapse"    # Emergency exit on tensor collapse


class TensorAnalysisState(Enum):
    """Tensor analysis state classifications."""
    STABLE_TENSOR = "stable_tensor"           # Stable tensor state
    CONTRACTING_TENSOR = "contracting_tensor" # Contracting tensor state
    DECOMPOSING_TENSOR = "decomposing_tensor" # Decomposing tensor state
    COLLAPSING_TENSOR = "collapsing_tensor"   # Collapsing tensor state
    FOURIER_ACTIVE = "fourier_active"         # Active Fourier transform
    CONVERGING_TENSOR = "converging_tensor"   # Converging tensor state


@dataclass
class TensorSignal:
    """Tensor-based market signal."""
    timestamp: float
    price: float
    volume: float
    tensor_state: TensorAnalysisState
    decision: TensorDecision
    confidence: float
    risk_level: float
    tensor_norm: float
    eigenvalue_magnitude: float
    fourier_magnitude: float
    collapse_function: float
    cosine_similarity: float
    metadata: Dict[str, Any]


@dataclass
class TensorAlgebraConfig:
    """Configuration for tensor algebra operations."""
    # UnifiedTensorAlgebra parameters
    max_rank: int = 3
    collapse_threshold: float = 0.1
    fourier_resolution: int = 64
    gamma_shift: float = 0.1
    eigenvalue_threshold: float = 1e-6
    norm_threshold: float = 1e-8
    contraction_axes: Optional[Tuple[int, ...]] = None
    
    # Tensor decision parameters
    tensor_contraction_threshold: float = 0.7
    fourier_signal_threshold: float = 0.6
    tensor_decomposition_threshold: float = 0.8
    tensor_stability_threshold: float = 0.5
    tensor_convergence_threshold: float = 0.4
    
    # Analysis parameters
    min_tensor_size: int = 2
    max_tensor_size: int = 10
    eigenvalue_analysis_depth: int = 5
    
    # Risk management
    max_risk_level: float = 0.8
    min_confidence: float = 0.3
    emergency_collapse_threshold: float = 0.95


class TensorDecisionEngine:
    """
    Tensor-based decision engine for market analysis.
    
    Uses UnifiedTensorAlgebra to analyze market data and make
    entry/exit/hold decisions based on tensor operations.
    """
    
    def __init__(self, config: Optional[TensorAlgebraConfig] = None):
        """Initialize the tensor decision engine."""
        self.config = config or TensorAlgebraConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize UnifiedTensorAlgebra
        tensor_config = {
            'max_rank': self.config.max_rank,
            'collapse_threshold': self.config.collapse_threshold,
            'fourier_resolution': self.config.fourier_resolution,
            'gamma_shift': self.config.gamma_shift,
            'eigenvalue_threshold': self.config.eigenvalue_threshold,
            'norm_threshold': self.config.norm_threshold,
            'contraction_axes': self.config.contraction_axes
        }
        self.tensor_algebra = UnifiedTensorAlgebra(tensor_config)
        
        # State tracking
        self.signal_history: List[TensorSignal] = []
        self.tensor_history: List[TensorAnalysisState] = []
        self.decision_history: List[TensorDecision] = []
        
        self.logger.info("Tensor decision engine initialized")
    
    def analyze_market_tensors(self, price_data: np.ndarray, volume_data: np.ndarray,
                             current_price: float, current_volume: float) -> TensorSignal:
        """
        Analyze market using tensor operations.
        
        Args:
            price_data: Historical price data
            volume_data: Historical volume data
            current_price: Current market price
            current_volume: Current market volume
            
        Returns:
            TensorSignal with decision and analysis
        """
        try:
            # Create tensors from market data
            price_tensor = self._create_market_tensor(price_data, "price")
            volume_tensor = self._create_market_tensor(volume_data, "volume")
            
            # Perform tensor operations
            tensor_contraction = self.tensor_algebra.tensor_contraction(price_tensor, volume_tensor)
            tensor_product = self.tensor_algebra.tensor_product(price_tensor, volume_tensor)
            
            # Compute eigenvalues and eigenvectors
            eigenvalues, eigenvectors = self.tensor_algebra.eigenvalue_decomposition(price_tensor)
            
            # Compute tensor norm
            tensor_norm = self.tensor_algebra.tensor_norm(price_tensor)
            
            # Compute Fourier transform
            fourier_transform = self.tensor_algebra.compute_fourier_tensor_dual_transform(price_tensor)
            fourier_magnitude = np.mean(np.abs(fourier_transform))
            
            # Compute cosine similarity
            cosine_similarity = self.tensor_algebra.compute_cosine_similarity(price_tensor, volume_tensor)
            
            # Compute collapse function
            collapse_function = self._compute_tensor_collapse(price_tensor, volume_tensor)
            
            # Determine tensor state
            tensor_state = self._classify_tensor_state(
                eigenvalues, tensor_norm, fourier_magnitude, collapse_function, tensor_contraction
            )
            
            # Make tensor-based decision
            decision = self._make_tensor_decision(
                tensor_state, eigenvalues, tensor_norm, fourier_magnitude, 
                collapse_function, tensor_contraction
            )
            
            # Calculate confidence and risk
            confidence = self._calculate_tensor_confidence(
                eigenvalues, tensor_norm, fourier_magnitude, cosine_similarity
            )
            risk_level = self._calculate_tensor_risk(tensor_state, eigenvalues, collapse_function)
            
            # Calculate eigenvalue magnitude
            eigenvalue_magnitude = np.mean(np.abs(eigenvalues))
            
            # Create tensor signal
            signal = TensorSignal(
                timestamp=0.0,  # Will be set by caller
                price=current_price,
                volume=current_volume,
                tensor_state=tensor_state,
                decision=decision,
                confidence=confidence,
                risk_level=risk_level,
                tensor_norm=tensor_norm,
                eigenvalue_magnitude=eigenvalue_magnitude,
                fourier_magnitude=fourier_magnitude,
                collapse_function=collapse_function,
                cosine_similarity=cosine_similarity,
                metadata={
                    'eigenvalues': eigenvalues.tolist(),
                    'tensor_contraction_shape': tensor_contraction.shape,
                    'tensor_product_shape': tensor_product.shape,
                    'fourier_transform_shape': fourier_transform.shape
                }
            )
            
            # Update history
            self.signal_history.append(signal)
            self.tensor_history.append(tensor_state)
            self.decision_history.append(decision)
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error analyzing market tensors: {e}")
            # Return default signal
            return TensorSignal(
                timestamp=0.0,
                price=current_price,
                volume=current_volume,
                tensor_state=TensorAnalysisState.STABLE_TENSOR,
                decision=TensorDecision.WAIT_TENSOR_CONVERGENCE,
                confidence=0.0,
                risk_level=1.0,
                tensor_norm=0.0,
                eigenvalue_magnitude=0.0,
                fourier_magnitude=0.0,
                collapse_function=0.0,
                cosine_similarity=0.0,
                metadata={'error': str(e)}
            )
    
    def _create_market_tensor(self, data: np.ndarray, data_type: str) -> np.ndarray:
        """Create tensor from market data."""
        # Ensure minimum size
        if len(data) < self.config.min_tensor_size:
            # Pad with zeros if insufficient data
            padded_data = np.pad(data, (0, self.config.min_tensor_size - len(data)), 'constant')
        elif len(data) > self.config.max_tensor_size:
            # Truncate if too large
            padded_data = data[-self.config.max_tensor_size:]
        else:
            padded_data = data
        
        # Create 2D tensor (data x time)
        tensor = np.array(padded_data).reshape(-1, 1)
        
        # Normalize based on data type
        if data_type == "price":
            # Normalize price data
            if np.max(tensor) > 0:
                tensor = tensor / np.max(tensor)
        elif data_type == "volume":
            # Normalize volume data
            if np.max(tensor) > 0:
                tensor = tensor / np.max(tensor)
        
        return tensor
    
    def _compute_tensor_collapse(self, price_tensor: np.ndarray, 
                               volume_tensor: np.ndarray) -> float:
        """Compute tensor collapse function."""
        try:
            # Use tensor contraction as collapse function
            contraction = self.tensor_algebra.tensor_contraction(price_tensor, volume_tensor)
            return float(np.mean(contraction))
        except Exception as e:
            self.logger.warning(f"Error computing tensor collapse: {e}")
            return 0.0
    
    def _classify_tensor_state(self, eigenvalues: np.ndarray, tensor_norm: float,
                             fourier_magnitude: float, collapse_function: float,
                             tensor_contraction: np.ndarray) -> TensorAnalysisState:
        """Classify tensor state based on tensor properties."""
        eigenvalue_magnitude = np.mean(np.abs(eigenvalues))
        contraction_magnitude = np.mean(np.abs(tensor_contraction))
        
        # Check for tensor collapse
        if collapse_function > self.config.emergency_collapse_threshold:
            return TensorAnalysisState.COLLAPSING_TENSOR
        
        # Check for tensor decomposition
        if eigenvalue_magnitude > self.config.tensor_decomposition_threshold:
            return TensorAnalysisState.DECOMPOSING_TENSOR
        
        # Check for tensor contraction
        if contraction_magnitude > self.config.tensor_contraction_threshold:
            return TensorAnalysisState.CONTRACTING_TENSOR
        
        # Check for Fourier activity
        if fourier_magnitude > self.config.fourier_signal_threshold:
            return TensorAnalysisState.FOURIER_ACTIVE
        
        # Check for tensor convergence
        if tensor_norm < self.config.tensor_convergence_threshold:
            return TensorAnalysisState.CONVERGING_TENSOR
        
        # Default to stable tensor
        return TensorAnalysisState.STABLE_TENSOR
    
    def _make_tensor_decision(self, tensor_state: TensorAnalysisState, eigenvalues: np.ndarray,
                            tensor_norm: float, fourier_magnitude: float,
                            collapse_function: float, tensor_contraction: np.ndarray) -> TensorDecision:
        """Make tensor-based market decision."""
        eigenvalue_magnitude = np.mean(np.abs(eigenvalues))
        contraction_magnitude = np.mean(np.abs(tensor_contraction))
        
        # Emergency exit on tensor collapse
        if tensor_state == TensorAnalysisState.COLLAPSING_TENSOR:
            return TensorDecision.EMERGENCY_TENSOR_COLLAPSE
        
        # Exit on tensor decomposition
        if tensor_state == TensorAnalysisState.DECOMPOSING_TENSOR:
            return TensorDecision.EXIT_TENSOR_DECOMPOSITION
        
        # Enter on tensor contraction
        if tensor_state == TensorAnalysisState.CONTRACTING_TENSOR:
            return TensorDecision.ENTER_TENSOR_CONTRACTION
        
        # Enter on Fourier signal
        if tensor_state == TensorAnalysisState.FOURIER_ACTIVE:
            return TensorDecision.ENTER_FOURIER_SIGNAL
        
        # Hold on tensor stability
        if tensor_state == TensorAnalysisState.STABLE_TENSOR:
            return TensorDecision.HOLD_TENSOR_STABILITY
        
        # Wait for tensor convergence
        if tensor_state == TensorAnalysisState.CONVERGING_TENSOR:
            return TensorDecision.WAIT_TENSOR_CONVERGENCE
        
        # Default to waiting
        return TensorDecision.WAIT_TENSOR_CONVERGENCE
    
    def _calculate_tensor_confidence(self, eigenvalues: np.ndarray, tensor_norm: float,
                                   fourier_magnitude: float, cosine_similarity: float) -> float:
        """Calculate confidence based on tensor properties."""
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
        
        # Adjust based on Fourier magnitude
        if fourier_magnitude > 0.5:
            confidence *= 1.1
        
        return max(0.0, min(1.0, confidence))
    
    def _calculate_tensor_risk(self, tensor_state: TensorAnalysisState, eigenvalues: np.ndarray,
                             collapse_function: float) -> float:
        """Calculate risk level based on tensor properties."""
        eigenvalue_magnitude = np.mean(np.abs(eigenvalues))
        
        # Base risk from eigenvalue magnitude
        risk = eigenvalue_magnitude
        
        # Adjust based on tensor state
        if tensor_state == TensorAnalysisState.COLLAPSING_TENSOR:
            risk *= 2.0
        elif tensor_state == TensorAnalysisState.DECOMPOSING_TENSOR:
            risk *= 1.5
        elif tensor_state == TensorAnalysisState.CONTRACTING_TENSOR:
            risk *= 1.1
        elif tensor_state == TensorAnalysisState.STABLE_TENSOR:
            risk *= 0.8
        
        # Adjust based on collapse function
        risk *= (1.0 + collapse_function)
        
        return max(0.0, min(1.0, risk))
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get tensor algebra system status."""
        return {
            'tensor_algebra_status': self.tensor_algebra.get_algebra_summary(),
            'signal_count': len(self.signal_history),
            'recent_decisions': self.decision_history[-10:] if self.decision_history else [],
            'tensor_states': [state.value for state in self.tensor_history[-10:]] if self.tensor_history else [],
            'config': {
                'max_rank': self.config.max_rank,
                'collapse_threshold': self.config.collapse_threshold,
                'tensor_contraction_threshold': self.config.tensor_contraction_threshold,
                'fourier_signal_threshold': self.config.fourier_signal_threshold
            }
        }


class TensorAlgebraFactory:
    """Factory for creating tensor algebra instances."""
    
    @staticmethod
    def create_from_config(config_path: Optional[str] = None) -> TensorDecisionEngine:
        """Create tensor algebra system from configuration file."""
        config = TensorAlgebraFactory._load_config(config_path)
        return TensorDecisionEngine(config)
    
    @staticmethod
    def create_with_params(**kwargs) -> TensorDecisionEngine:
        """Create tensor algebra system with custom parameters."""
        config = TensorAlgebraConfig(**kwargs)
        return TensorDecisionEngine(config)
    
    @staticmethod
    def _load_config(config_path: Optional[str] = None) -> TensorAlgebraConfig:
        """Load configuration from file."""
        if config_path is None:
            # Try to find default config
            default_paths = [
                "config/tensor_algebra_config.yaml",
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
                
                # Extract tensor algebra config
                tensor_config = config_data.get('tensor_algebra', {})
                return TensorAlgebraConfig(**tensor_config)
                
            except Exception as e:
                logger.warning(f"Could not load tensor algebra config from {config_path}: {e}")
        
        # Return default config
        return TensorAlgebraConfig()


# Auto-load mathematical functions registry if available
TENSOR_ALGEBRA_FUNCTIONS_REGISTRY = {}

try:
    registry_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'config', 'mathematical_functions_registry.yaml')
    if os.path.exists(registry_path):
        with open(registry_path, 'r') as f:
            registry_data = yaml.safe_load(f)
            tensor_functions = registry_data.get('mathematical_functions', {}).get('tensor_algebra', {})
            TENSOR_ALGEBRA_FUNCTIONS_REGISTRY.update(tensor_functions)
except Exception as e:
    logger.warning(f"Could not load tensor algebra functions registry: {e}")


# Export main classes and functions
__all__ = [
    "UnifiedTensorAlgebra",
    "TensorRank",
    "TensorOperation", 
    "TensorResult",
    "CollapseResult",
    "TensorDecisionEngine",
    "TensorAlgebraConfig",
    "TensorAlgebraFactory",
    "TensorDecision",
    "TensorAnalysisState",
    "TensorSignal",
    "TENSOR_ALGEBRA_FUNCTIONS_REGISTRY"
]

# Convenience functions for quick access
def create_unified_tensor_algebra(*args, **kwargs) -> UnifiedTensorAlgebra:
    """Factory for UnifiedTensorAlgebra."""
    return UnifiedTensorAlgebra(*args, **kwargs)

def create_tensor_decision_engine(*args, **kwargs) -> TensorDecisionEngine:
    """Factory for TensorDecisionEngine."""
    return TensorDecisionEngine(*args, **kwargs)

def analyze_market_tensors(price_data: np.ndarray, volume_data: np.ndarray,
                          current_price: float, current_volume: float,
                          config: Optional[TensorAlgebraConfig] = None) -> TensorSignal:
    """Quick function to analyze market tensors."""
    tensor_engine = TensorDecisionEngine(config)
    return tensor_engine.analyze_market_tensors(price_data, volume_data, current_price, current_volume) 