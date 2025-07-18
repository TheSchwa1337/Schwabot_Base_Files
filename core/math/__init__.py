#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core Mathematical Framework for Schwabot Trading System
======================================================

This module provides the core mathematical framework for advanced trading analysis.
It integrates tensor algebra, mathematical decision making, and signal processing
to provide sophisticated market analysis capabilities.

Core Components:
- UnifiedTensorAlgebra: Advanced tensor operations with GPU support
- MathematicalDecisionEngine: Market decisions based on tensor analysis
- TensorSignalProcessor: Processing of tensor-based signals
- MathematicalConsensus: Consensus building across mathematical modules
"""

import logging
import os
import time
import yaml
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional
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

    ENTER_TENSOR_ALIGNMENT = "enter_tensor_alignment"  # Enter on tensor alignment
    ENTER_EIGENVALUE_SIGNAL = "enter_eigenvalue_signal"  # Enter on eigenvalue signal
    EXIT_TENSOR_DECOMPOSITION = (
        "exit_tensor_decomposition"  # Exit on tensor decomposition
    )
    HOLD_TENSOR_STABILITY = "hold_tensor_stability"  # Hold on tensor stability
    WAIT_TENSOR_CONVERGENCE = "wait_tensor_convergence"  # Wait for tensor convergence
    EMERGENCY_TENSOR_COLLAPSE = (
        "emergency_tensor_collapse"  # Emergency exit on tensor collapse
    )


class TensorState(Enum):
    """Tensor state classifications."""

    STABLE_TENSOR = "stable_tensor"  # Stable tensor state
    OSCILLATING_TENSOR = "oscillating_tensor"  # Oscillating tensor state
    DECOMPOSING_TENSOR = "decomposing_tensor"  # Decomposing tensor state
    COLLAPSING_TENSOR = "collapsing_tensor"  # Collapsing tensor state
    ALIGNING_TENSOR = "aligning_tensor"  # Aligning tensor state
    CONVERGING_TENSOR = "converging_tensor"  # Converging tensor state


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

    def __init__(self, config: Optional[MathSystemConfig] = None) -> None:
        """Initialize the mathematical decision engine."""
        self.config = config or MathSystemConfig()
        self.logger = logging.getLogger(__name__)
        # Initialize UnifiedTensorAlgebra
        tensor_config = {
            "max_rank": self.config.max_rank,
            "collapse_threshold": self.config.collapse_threshold,
            "fourier_resolution": self.config.fourier_resolution,
            "gamma_shift": self.config.gamma_shift,
            "eigenvalue_threshold": self.config.eigenvalue_threshold,
            "norm_threshold": self.config.norm_threshold,
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

    def analyze_market_mathematics(
        self,
        price_data: np.ndarray,
        volume_data: np.ndarray,
        current_price: float,
        current_volume: float,
    ) -> MathSignal:
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
                "contraction", [price_tensor, volume_tensor]
            )
            # Compute eigenvalues and eigenvectors
            eigenvalues, eigenvectors = self.tensor_algebra.eigenvalue_decomposition(
                price_tensor
            )
            # Compute tensor norm
            tensor_norm = self.tensor_algebra.tensor_norm(price_tensor)
            # Compute cosine similarity
            cosine_similarity = self.tensor_algebra.compute_cosine_similarity(
                price_tensor, volume_tensor
            )
            # Compute collapse function
            collapse_function = self._compute_collapse_function(
                price_tensor, volume_tensor
            )
            # Compute Fourier transform
            fourier_transform = (
                self.tensor_algebra.compute_fourier_tensor_dual_transform(price_tensor)
            )
            fourier_magnitude = np.mean(np.abs(fourier_transform))
            # Determine tensor state
            tensor_state = self._classify_tensor_state(
                eigenvalues, tensor_norm, cosine_similarity, collapse_function
            )
            # Make mathematical decision
            decision = self._make_math_decision(
                tensor_state,
                eigenvalues,
                tensor_norm,
                cosine_similarity,
                collapse_function,
            )
            # Calculate confidence and risk
            confidence = self._calculate_math_confidence(
                eigenvalues, tensor_norm, cosine_similarity, collapse_function
            )
            risk_level = self._calculate_math_risk(
                eigenvalues, tensor_norm, cosine_similarity, collapse_function
            )
            # Create math signal
            math_signal = MathSignal(
                timestamp=time.time(),
                price=current_price,
                volume=current_volume,
                tensor_state=tensor_state,
                decision=decision,
                confidence=confidence,
                risk_level=risk_level,
                eigenvalue_score=np.mean(np.abs(eigenvalues)),
                tensor_norm=tensor_norm,
                cosine_similarity=cosine_similarity,
                collapse_function=collapse_function,
                fourier_transform_magnitude=fourier_magnitude,
                metadata={
                    "eigenvalues": eigenvalues.tolist(),
                    "tensor_result": tensor_result.tolist()
                    if hasattr(tensor_result, "tolist")
                    else str(tensor_result),
                },
            )
            # Update history
            self.signal_history.append(math_signal)
            self.tensor_history.append(tensor_state)
            self.decision_history.append(decision)
            return math_signal
        except Exception as e:
            self.logger.error(f"Error in mathematical analysis: {e}")
            # Return a default signal
            return MathSignal(
                timestamp=time.time(),
                price=current_price,
                volume=current_volume,
                tensor_state=TensorState.STABLE_TENSOR,
                decision=MathDecision.HOLD_TENSOR_STABILITY,
                confidence=0.0,
                risk_level=1.0,
                eigenvalue_score=0.0,
                tensor_norm=0.0,
                cosine_similarity=0.0,
                collapse_function=0.0,
                fourier_transform_magnitude=0.0,
                metadata={"error": str(e)},
            )

    def _create_price_tensor(self, price_data: np.ndarray) -> np.ndarray:
        """Create price tensor from price data."""
        return np.array(price_data).reshape(-1, 1)

    def _create_volume_tensor(self, volume_data: np.ndarray) -> np.ndarray:
        """Create volume tensor from volume data."""
        return np.array(volume_data).reshape(-1, 1)

    def _compute_collapse_function(
        self, price_tensor: np.ndarray, volume_tensor: np.ndarray
    ) -> float:
        """Compute collapse function between price and volume tensors."""
        try:
            # Simple collapse function based on tensor correlation
            correlation = np.corrcoef(price_tensor.flatten(), volume_tensor.flatten())[0, 1]
            return abs(correlation) if not np.isnan(correlation) else 0.0
        except Exception:
            return 0.0

    def _classify_tensor_state(
        self,
        eigenvalues: np.ndarray,
        tensor_norm: float,
        cosine_similarity: float,
        collapse_function: float,
    ) -> TensorState:
        """Classify the current tensor state."""
        # Simple classification logic
        if collapse_function > self.config.emergency_collapse_threshold:
            return TensorState.COLLAPSING_TENSOR
        elif tensor_norm < self.config.norm_threshold:
            return TensorState.DECOMPOSING_TENSOR
        elif cosine_similarity > self.config.tensor_alignment_threshold:
            return TensorState.ALIGNING_TENSOR
        elif abs(np.mean(eigenvalues)) < self.config.eigenvalue_threshold:
            return TensorState.CONVERGING_TENSOR
        else:
            return TensorState.STABLE_TENSOR

    def _make_math_decision(
        self,
        tensor_state: TensorState,
        eigenvalues: np.ndarray,
        tensor_norm: float,
        cosine_similarity: float,
        collapse_function: float,
    ) -> MathDecision:
        """Make mathematical decision based on tensor analysis."""
        if tensor_state == TensorState.COLLAPSING_TENSOR:
            return MathDecision.EMERGENCY_TENSOR_COLLAPSE
        elif tensor_state == TensorState.ALIGNING_TENSOR:
            return MathDecision.ENTER_TENSOR_ALIGNMENT
        elif tensor_state == TensorState.DECOMPOSING_TENSOR:
            return MathDecision.EXIT_TENSOR_DECOMPOSITION
        elif tensor_state == TensorState.CONVERGING_TENSOR:
            return MathDecision.WAIT_TENSOR_CONVERGENCE
        else:
            return MathDecision.HOLD_TENSOR_STABILITY

    def _calculate_math_confidence(
        self,
        eigenvalues: np.ndarray,
        tensor_norm: float,
        cosine_similarity: float,
        collapse_function: float,
    ) -> float:
        """Calculate confidence level based on mathematical analysis."""
        # Simple confidence calculation
        confidence = (
            min(tensor_norm, 1.0) * 0.3 +
            min(cosine_similarity, 1.0) * 0.3 +
            min(1.0 - collapse_function, 1.0) * 0.4
        )
        return max(0.0, min(1.0, confidence))

    def _calculate_math_risk(
        self,
        eigenvalues: np.ndarray,
        tensor_norm: float,
        cosine_similarity: float,
        collapse_function: float,
    ) -> float:
        """Calculate risk level based on mathematical analysis."""
        # Simple risk calculation
        risk = (
            collapse_function * 0.4 +
            (1.0 - tensor_norm) * 0.3 +
            (1.0 - cosine_similarity) * 0.3
        )
        return max(0.0, min(1.0, risk))

    def get_signal_history(self, limit: Optional[int] = None) -> List[MathSignal]:
        """Get signal history."""
        if limit is None:
            return self.signal_history
        return self.signal_history[-limit:]

    def get_tensor_history(self, limit: Optional[int] = None) -> List[TensorState]:
        """Get tensor state history."""
        if limit is None:
            return self.tensor_history
        return self.tensor_history[-limit:]

    def get_decision_history(self, limit: Optional[int] = None) -> List[MathDecision]:
        """Get decision history."""
        if limit is None:
            return self.decision_history
        return self.decision_history[-limit:]

    def reset_history(self):
        """Reset all history."""
        self.signal_history.clear()
        self.tensor_history.clear()
        self.decision_history.clear()
