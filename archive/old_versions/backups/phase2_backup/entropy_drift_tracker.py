"""Module for Schwabot trading system."""

import logging
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np

# !/usr/bin/env python3
"""
🌌⚙️ ENTROPY DRIFT TRACKER
==========================

Tracks entropy variance between trade vectors over time.
Measures the instability of strategy vectors to predict optimal execution windows.

Core Concept: ΔE = ||Vₙ - Vₙ₋₁||
Where Vₙ is current vector and Vₙ₋₁ is previous vector.

    CUDA Integration:
    - GPU-accelerated entropy drift tracking with automatic CPU fallback
    - Performance monitoring and optimization
    - Cross-platform compatibility (Windows, macOS, Linux)
    - Comprehensive error handling for hybrid states
    """

    # CUDA Integration with Fallback
        try:
        import cupy as cp

        USING_CUDA = True
        _backend = 'cupy (GPU)'
        xp = cp
            except ImportError:
            import numpy as cp  # fallback to numpy

            USING_CUDA = False
            _backend = 'numpy (CPU)'
            xp = cp

            # Log backend status
            logger = logging.getLogger(__name__)
                if USING_CUDA:
                logger.info("⚡ Entropy Drift Tracker using GPU acceleration: {0}".format(_backend))
                    else:
                    logger.info("🔄 Entropy Drift Tracker using CPU fallback: {0}".format(_backend))


                        class DriftState(Enum):
    """Class for Schwabot trading functionality."""
                        """Class for Schwabot trading functionality."""
                        """Enumeration for drift states."""

                        STABLE = "stable"
                        INCREASING = "increasing"
                        DECREASING = "decreasing"
                        VOLATILE = "volatile"
                        HYBRID = "hybrid"
                        ERROR = "error"


                            class HybridMode(Enum):
    """Class for Schwabot trading functionality."""
                            """Class for Schwabot trading functionality."""
                            """Enumeration for hybrid processing modes."""

                            GPU_PREFERRED = "gpu_preferred"
                            CPU_PREFERRED = "cpu_preferred"
                            ADAPTIVE = "adaptive"
                            FALLBACK = "fallback"


                            @dataclass
                                class DriftSnapshot:
    """Class for Schwabot trading functionality."""
                                """Class for Schwabot trading functionality."""
                                """Snapshot of vector drift at a specific time"""

                                timestamp: float
                                vector: xp.ndarray
                                drift_value: float
                                entropy_score: float
                                drift_state: DriftState = DriftState.STABLE
                                hybrid_mode: HybridMode = HybridMode.ADAPTIVE
                                metadata: Dict[str, Any] = None

                                    def __post_init__(self) -> None:
                                        if self.metadata is None:
                                        self.metadata = {}


                                        @dataclass
                                            class DriftError:
    """Class for Schwabot trading functionality."""
                                            """Class for Schwabot trading functionality."""
                                            """Error information for drift tracking."""

                                            error_type: str
                                            error_message: str
                                            timestamp: float
                                            strategy_id: str
                                            fallback_used: bool = False
                                            hybrid_mode: HybridMode = HybridMode.FALLBACK


                                                class EntropyDriftTracker:
    """Class for Schwabot trading functionality."""
                                                """Class for Schwabot trading functionality."""
                                                """
                                                Entropy Drift Tracker with GPU/CPU Hybrid Support

                                                Tracks vector instability over time and computes drift-based warp windows
                                                for optimal trade execution timing with comprehensive error handling.
                                                """

                                                def __init__(
                                                self,
                                                max_history: int = 100,
                                                warp_threshold: float = 0.15,
                                                hybrid_mode: HybridMode = HybridMode.ADAPTIVE,
                                                    ):
                                                    """
                                                    Initialize entropy drift tracker

                                                        Args:
                                                        max_history: Maximum number of vector snapshots to store
                                                        warp_threshold: Threshold for warp window activation
                                                        hybrid_mode: Hybrid processing mode preference
                                                        """
                                                        self.history: Dict[str, deque] = {}
                                                        self.max_history = max_history
                                                        self.warp_threshold = warp_threshold
                                                        self.hybrid_mode = hybrid_mode
                                                        self.drift_stats: Dict[str, Dict[str, float]] = {}
                                                        self.error_log: List[DriftError] = []
                                                        self.performance_metrics = {
                                                        'gpu_operations': 0,
                                                        'cpu_operations': 0,
                                                        'fallback_operations': 0,
                                                        'total_operations': 0,
                                                        'avg_operation_time': 0.0,
                                                        }

                                                        logger.info(
                                                        "Entropy Drift Tracker initialized (max_history: {0}, threshold: {1}, "
                                                        "hybrid_mode: {2})".format(max_history, warp_threshold, hybrid_mode.value)
                                                        )

                                                            def record_vector(self, strategy_id: str, vector: Union[xp.ndarray, np.ndarray], force_cpu: bool = False) -> float:
                                                            """
                                                            Record a new vector and compute its drift with hybrid processing

                                                                Args:
                                                                strategy_id: Strategy identifier
                                                                vector: Current profit vector
                                                                force_cpu: Force CPU processing (for error, recovery)

                                                                    Returns:
                                                                    Computed drift value
                                                                    """
                                                                    start_time = time.time()
                                                                    self.performance_metrics['total_operations'] += 1

                                                                        try:
                                                                        current_time = time.time()

                                                                        # Convert vector to appropriate backend
                                                                            if force_cpu or self.hybrid_mode == HybridMode.CPU_PREFERRED:
                                                                                if USING_CUDA and isinstance(vector, cp.ndarray):
                                                                                vector = cp.asnumpy(vector)
                                                                                processing_mode = HybridMode.CPU_PREFERRED
                                                                                self.performance_metrics['cpu_operations'] += 1
                                                                                    elif self.hybrid_mode == HybridMode.GPU_PREFERRED and USING_CUDA:
                                                                                        if isinstance(vector, np.ndarray):
                                                                                        vector = cp.asarray(vector)
                                                                                        processing_mode = HybridMode.GPU_PREFERRED
                                                                                        self.performance_metrics['gpu_operations'] += 1
                                                                                            else:
                                                                                            # Adaptive mode - use current backend
                                                                                            processing_mode = HybridMode.ADAPTIVE
                                                                                                if USING_CUDA:
                                                                                                self.performance_metrics['gpu_operations'] += 1
                                                                                                    else:
                                                                                                    self.performance_metrics['cpu_operations'] += 1

                                                                                                    # Initialize history for new strategy
                                                                                                        if strategy_id not in self.history:
                                                                                                        self.history[strategy_id] = deque(maxlen=self.max_history)
                                                                                                        self.drift_stats[strategy_id] = {
                                                                                                        'avg_drift': 0.0,
                                                                                                        'max_drift': 0.0,
                                                                                                        'min_drift': float('inf'),
                                                                                                        'drift_count': 0,
                                                                                                        'error_count': 0,
                                                                                                        'hybrid_switches': 0,
                                                                                                        }

                                                                                                        # Compute drift from previous vector
                                                                                                        drift_value = 0.0
                                                                                                            if len(self.history[strategy_id]) > 0:
                                                                                                            last_snapshot = self.history[strategy_id][-1]
                                                                                                                try:
                                                                                                                # Ensure both vectors are on same backend
                                                                                                                    if USING_CUDA and isinstance(last_snapshot.vector, np.ndarray):
                                                                                                                    last_vector = cp.asarray(last_snapshot.vector)
                                                                                                                        elif not USING_CUDA and isinstance(last_snapshot.vector, cp.ndarray):
                                                                                                                        last_vector = cp.asnumpy(last_snapshot.vector)
                                                                                                                            else:
                                                                                                                            last_vector = last_snapshot.vector

                                                                                                                            drift_value = float(xp.linalg.norm(vector - last_vector))
                                                                                                                                except Exception as e:
                                                                                                                                logger.warning("Drift calculation failed, using fallback: {0}".format(e))
                                                                                                                                drift_value = self._calculate_fallback_drift(vector, last_snapshot.vector)
                                                                                                                                self.performance_metrics['fallback_operations'] += 1

                                                                                                                                # Compute entropy score (vector, variance)
                                                                                                                                    try:
                                                                                                                                    entropy_score = float(xp.std(vector)) if len(vector) > 1 else 0.0
                                                                                                                                        except Exception as e:
                                                                                                                                        logger.warning("Entropy calculation failed, using fallback: {0}".format(e))
                                                                                                                                        entropy_score = self._calculate_fallback_entropy(vector)
                                                                                                                                        self.performance_metrics['fallback_operations'] += 1

                                                                                                                                        # Determine drift state
                                                                                                                                        drift_state = self._determine_drift_state(drift_value, strategy_id)

                                                                                                                                        # Create drift snapshot
                                                                                                                                        snapshot = DriftSnapshot(
                                                                                                                                        timestamp=current_time,
                                                                                                                                        vector=vector.copy(),
                                                                                                                                        drift_value=drift_value,
                                                                                                                                        entropy_score=entropy_score,
                                                                                                                                        drift_state=drift_state,
                                                                                                                                        hybrid_mode=processing_mode,
                                                                                                                                        metadata={
                                                                                                                                        'processing_backend': _backend,
                                                                                                                                        'vector_shape': vector.shape,
                                                                                                                                        'vector_dtype': str(vector.dtype),
                                                                                                                                        },
                                                                                                                                        )

                                                                                                                                        # Add to history
                                                                                                                                        self.history[strategy_id].append(snapshot)

                                                                                                                                        # Update drift statistics
                                                                                                                                        stats = self.drift_stats[strategy_id]
                                                                                                                                        stats['drift_count'] += 1
                                                                                                                                        stats['avg_drift'] = (stats['avg_drift'] * (stats['drift_count'] - 1) + drift_value) / stats['drift_count']
                                                                                                                                        stats['max_drift'] = max(stats['max_drift'], drift_value)
                                                                                                                                        stats['min_drift'] = min(stats['min_drift'], drift_value)

                                                                                                                                        # Update performance metrics
                                                                                                                                        operation_time = time.time() - start_time
                                                                                                                                        self.performance_metrics['avg_operation_time'] = (
                                                                                                                                        self.performance_metrics['avg_operation_time'] * (self.performance_metrics['total_operations'] - 1)
                                                                                                                                        + operation_time
                                                                                                                                        ) / self.performance_metrics['total_operations']

                                                                                                                                        logger.debug(
                                                                                                                                        "Recorded vector for {0}: drift={1}, entropy={2}, state={3}, mode={4}".format(
                                                                                                                                        strategy_id,
                                                                                                                                        drift_value,
                                                                                                                                        entropy_score,
                                                                                                                                        drift_state.value,
                                                                                                                                        processing_mode.value,
                                                                                                                                        )
                                                                                                                                        )
                                                                                                                                    return drift_value

                                                                                                                                        except Exception as e:
                                                                                                                                        error = DriftError(
                                                                                                                                        error_type=type(e).__name__,
                                                                                                                                        error_message=str(e),
                                                                                                                                        timestamp=time.time(),
                                                                                                                                        strategy_id=strategy_id,
                                                                                                                                        fallback_used=True,
                                                                                                                                        hybrid_mode=HybridMode.FALLBACK,
                                                                                                                                        )
                                                                                                                                        self.error_log.append(error)
                                                                                                                                        self.drift_stats[strategy_id]['error_count'] += 1
                                                                                                                                        logger.error("Error recording vector for {0}: {1}".format(strategy_id, e))
                                                                                                                                    return self._safe_fallback_drift()

                                                                                                                                    def _calculate_fallback_drift(
                                                                                                                                    self, vector1: Union[xp.ndarray, np.ndarray], vector2: Union[xp.ndarray, np.ndarray]
                                                                                                                                        ) -> float:
                                                                                                                                        """Calculate drift using fallback method."""
                                                                                                                                            try:
                                                                                                                                            # Convert to numpy for fallback calculation
                                                                                                                                                if USING_CUDA and isinstance(vector1, cp.ndarray):
                                                                                                                                                v1 = cp.asnumpy(vector1)
                                                                                                                                                    else:
                                                                                                                                                    v1 = vector1

                                                                                                                                                        if USING_CUDA and isinstance(vector2, cp.ndarray):
                                                                                                                                                        v2 = cp.asnumpy(vector2)
                                                                                                                                                            else:
                                                                                                                                                            v2 = vector2

                                                                                                                                                            # Simple Euclidean distance as fallback
                                                                                                                                                        return float(np.linalg.norm(v1 - v2))
                                                                                                                                                            except Exception:
                                                                                                                                                        return 0.0

                                                                                                                                                            def _calculate_fallback_entropy(self, vector: Union[xp.ndarray, np.ndarray]) -> float:
                                                                                                                                                            """Calculate entropy using fallback method."""
                                                                                                                                                                try:
                                                                                                                                                                # Convert to numpy for fallback calculation
                                                                                                                                                                    if USING_CUDA and isinstance(vector, cp.ndarray):
                                                                                                                                                                    v = cp.asnumpy(vector)
                                                                                                                                                                        else:
                                                                                                                                                                        v = vector

                                                                                                                                                                    return float(np.std(v)) if len(v) > 1 else 0.0
                                                                                                                                                                        except Exception:
                                                                                                                                                                    return 0.0

                                                                                                                                                                        def _determine_drift_state(self, drift_value: float, strategy_id: str) -> DriftState:
                                                                                                                                                                        """Determine drift state based on current and historical drift."""
                                                                                                                                                                            try:
                                                                                                                                                                                if strategy_id not in self.drift_stats:
                                                                                                                                                                            return DriftState.STABLE

                                                                                                                                                                            stats = self.drift_stats[strategy_id]
                                                                                                                                                                                if stats['drift_count'] < 3:
                                                                                                                                                                            return DriftState.STABLE

                                                                                                                                                                            # Compare current drift to historical average
                                                                                                                                                                            avg_drift = stats['avg_drift']
                                                                                                                                                                                if avg_drift == 0:
                                                                                                                                                                            return DriftState.STABLE

                                                                                                                                                                            ratio = drift_value / avg_drift

                                                                                                                                                                                if ratio > 2.0:
                                                                                                                                                                            return DriftState.VOLATILE
                                                                                                                                                                                elif ratio > 1.5:
                                                                                                                                                                            return DriftState.INCREASING
                                                                                                                                                                                elif ratio < 0.5:
                                                                                                                                                                            return DriftState.DECREASING
                                                                                                                                                                                else:
                                                                                                                                                                            return DriftState.STABLE

                                                                                                                                                                                except Exception:
                                                                                                                                                                            return DriftState.ERROR

                                                                                                                                                                                def _safe_fallback_drift(self) -> float:
                                                                                                                                                                                """Safe fallback drift value."""
                                                                                                                                                                            return 0.0

                                                                                                                                                                                def compute_drift(self, strategy_id: str, window_size: Optional[int] = None, use_hybrid: bool = True) -> float:
                                                                                                                                                                                """
                                                                                                                                                                                Compute average drift over recent vectors with hybrid processing

                                                                                                                                                                                    Args:
                                                                                                                                                                                    strategy_id: Strategy identifier
                                                                                                                                                                                    window_size: Number of recent vectors to consider (None = all)
                                                                                                                                                                                    use_hybrid: Use hybrid processing mode

                                                                                                                                                                                        Returns:
                                                                                                                                                                                        Average drift value
                                                                                                                                                                                        """
                                                                                                                                                                                            try:
                                                                                                                                                                                                if strategy_id not in self.history or len(self.history[strategy_id]) < 2:
                                                                                                                                                                                            return 0.0

                                                                                                                                                                                            vectors = list(self.history[strategy_id])

                                                                                                                                                                                            # Use specified window size or all available
                                                                                                                                                                                                if window_size is not None:
                                                                                                                                                                                                vectors = vectors[-window_size:]

                                                                                                                                                                                                    if len(vectors) < 2:
                                                                                                                                                                                                return 0.0

                                                                                                                                                                                                # Compute drift between consecutive vectors
                                                                                                                                                                                                drifts = []
                                                                                                                                                                                                    for i in range(1, len(vectors)):
                                                                                                                                                                                                        try:
                                                                                                                                                                                                        v1 = vectors[i].vector
                                                                                                                                                                                                        v2 = vectors[i - 1].vector

                                                                                                                                                                                                        # Ensure both vectors are on same backend
                                                                                                                                                                                                            if USING_CUDA and isinstance(v1, np.ndarray):
                                                                                                                                                                                                            v1 = cp.asarray(v1)
                                                                                                                                                                                                                elif not USING_CUDA and isinstance(v1, cp.ndarray):
                                                                                                                                                                                                                v1 = cp.asnumpy(v1)

                                                                                                                                                                                                                    if USING_CUDA and isinstance(v2, np.ndarray):
                                                                                                                                                                                                                    v2 = cp.asarray(v2)
                                                                                                                                                                                                                        elif not USING_CUDA and isinstance(v2, cp.ndarray):
                                                                                                                                                                                                                        v2 = cp.asnumpy(v2)

                                                                                                                                                                                                                        drift = float(xp.linalg.norm(v1 - v2))
                                                                                                                                                                                                                        drifts.append(drift)
                                                                                                                                                                                                                            except Exception as e:
                                                                                                                                                                                                                            logger.warning("Drift computation failed for pair {0}: {1}".format(i, e))
                                                                                                                                                                                                                            # Use fallback calculation
                                                                                                                                                                                                                            fallback_drift = self._calculate_fallback_drift(vectors[i].vector, vectors[i - 1].vector)
                                                                                                                                                                                                                            drifts.append(fallback_drift)

                                                                                                                                                                                                                            avg_drift = xp.mean(drifts) if drifts else 0.0
                                                                                                                                                                                                                        return float(avg_drift)

                                                                                                                                                                                                                            except Exception as e:
                                                                                                                                                                                                                            logger.error("Error computing drift for {0}: {1}".format(strategy_id, e))
                                                                                                                                                                                                                        return 0.0

                                                                                                                                                                                                                            def is_warp_window(self, strategy_id: str, threshold: Optional[float] = None) -> bool:
                                                                                                                                                                                                                            """
                                                                                                                                                                                                                            Check if current state is within a warp window

                                                                                                                                                                                                                                Args:
                                                                                                                                                                                                                                strategy_id: Strategy identifier
                                                                                                                                                                                                                                threshold: Custom threshold (uses default if, None)

                                                                                                                                                                                                                                    Returns:
                                                                                                                                                                                                                                    True if in warp window
                                                                                                                                                                                                                                    """
                                                                                                                                                                                                                                        try:
                                                                                                                                                                                                                                        drift = self.compute_drift(strategy_id)
                                                                                                                                                                                                                                        thresh = threshold or self.warp_threshold
                                                                                                                                                                                                                                    return drift > thresh

                                                                                                                                                                                                                                        except Exception as e:
                                                                                                                                                                                                                                        logger.error("Error checking warp window for {0}: {1}".format(strategy_id, e))
                                                                                                                                                                                                                                    return False

                                                                                                                                                                                                                                        def get_drift_trend(self, strategy_id: str, window_size: int = 10) -> str:
                                                                                                                                                                                                                                        """
                                                                                                                                                                                                                                        Get drift trend direction

                                                                                                                                                                                                                                            Args:
                                                                                                                                                                                                                                            strategy_id: Strategy identifier
                                                                                                                                                                                                                                            window_size: Number of recent vectors to analyze

                                                                                                                                                                                                                                                Returns:
                                                                                                                                                                                                                                                Trend direction: "increasing", "decreasing", "stable"
                                                                                                                                                                                                                                                """
                                                                                                                                                                                                                                                    try:
                                                                                                                                                                                                                                                        if strategy_id not in self.history or len(self.history[strategy_id]) < window_size:
                                                                                                                                                                                                                                                    return "stable"

                                                                                                                                                                                                                                                    vectors = list(self.history[strategy_id])[-window_size:]
                                                                                                                                                                                                                                                        if len(vectors) < 3:
                                                                                                                                                                                                                                                    return "stable"

                                                                                                                                                                                                                                                    # Compute drift values
                                                                                                                                                                                                                                                    drifts = []
                                                                                                                                                                                                                                                        for i in range(1, len(vectors)):
                                                                                                                                                                                                                                                            try:
                                                                                                                                                                                                                                                            drift = float(xp.linalg.norm(vectors[i].vector - vectors[i - 1].vector))
                                                                                                                                                                                                                                                            drifts.append(drift)
                                                                                                                                                                                                                                                                except Exception:
                                                                                                                                                                                                                                                                # Use fallback
                                                                                                                                                                                                                                                                fallback_drift = self._calculate_fallback_drift(vectors[i].vector, vectors[i - 1].vector)
                                                                                                                                                                                                                                                                drifts.append(fallback_drift)

                                                                                                                                                                                                                                                                    if len(drifts) < 2:
                                                                                                                                                                                                                                                                return "stable"

                                                                                                                                                                                                                                                                # Compute trend
                                                                                                                                                                                                                                                                first_half = xp.mean(drifts[: len(drifts) // 2])
                                                                                                                                                                                                                                                                second_half = xp.mean(drifts[len(drifts) // 2 :])

                                                                                                                                                                                                                                                                    if second_half > first_half * 1.2:
                                                                                                                                                                                                                                                                return "increasing"
                                                                                                                                                                                                                                                                    elif second_half < first_half * 0.8:
                                                                                                                                                                                                                                                                return "decreasing"
                                                                                                                                                                                                                                                                    else:
                                                                                                                                                                                                                                                                return "stable"

                                                                                                                                                                                                                                                                    except Exception as e:
                                                                                                                                                                                                                                                                    logger.error("Error computing drift trend for {0}: {1}".format(strategy_id, e))
                                                                                                                                                                                                                                                                return "stable"

                                                                                                                                                                                                                                                                    def get_entropy_score(self, strategy_id: str) -> float:
                                                                                                                                                                                                                                                                    """
                                                                                                                                                                                                                                                                    Get current entropy score for a strategy

                                                                                                                                                                                                                                                                        Args:
                                                                                                                                                                                                                                                                        strategy_id: Strategy identifier

                                                                                                                                                                                                                                                                            Returns:
                                                                                                                                                                                                                                                                            Current entropy score
                                                                                                                                                                                                                                                                            """
                                                                                                                                                                                                                                                                                try:
                                                                                                                                                                                                                                                                                    if strategy_id not in self.history or len(self.history[strategy_id]) == 0:
                                                                                                                                                                                                                                                                                return 0.0

                                                                                                                                                                                                                                                                                latest_snapshot = self.history[strategy_id][-1]
                                                                                                                                                                                                                                                                            return latest_snapshot.entropy_score

                                                                                                                                                                                                                                                                                except Exception as e:
                                                                                                                                                                                                                                                                                logger.error("Error getting entropy score for {0}: {1}".format(strategy_id, e))
                                                                                                                                                                                                                                                                            return 0.0

                                                                                                                                                                                                                                                                                def predict_warp_delay(self, strategy_id: str, alpha: float = 100.0) -> float:
                                                                                                                                                                                                                                                                                """
                                                                                                                                                                                                                                                                                Predict optimal delay before next warp window

                                                                                                                                                                                                                                                                                    Args:
                                                                                                                                                                                                                                                                                    strategy_id: Strategy identifier
                                                                                                                                                                                                                                                                                    alpha: Sensitivity parameter

                                                                                                                                                                                                                                                                                        Returns:
                                                                                                                                                                                                                                                                                        Predicted delay in seconds
                                                                                                                                                                                                                                                                                        """
                                                                                                                                                                                                                                                                                            try:
                                                                                                                                                                                                                                                                                            drift = self.compute_drift(strategy_id)
                                                                                                                                                                                                                                                                                                if drift == 0:
                                                                                                                                                                                                                                                                                            return 0.0

                                                                                                                                                                                                                                                                                            # Inverse relationship: higher drift = shorter delay
                                                                                                                                                                                                                                                                                            delay = alpha / (1.0 + drift)
                                                                                                                                                                                                                                                                                        return max(0.1, min(60.0, delay))  # Bound between 0.1 and 60 seconds

                                                                                                                                                                                                                                                                                            except Exception as e:
                                                                                                                                                                                                                                                                                            logger.error("Error predicting warp delay for {0}: {1}".format(strategy_id, e))
                                                                                                                                                                                                                                                                                        return 1.0  # Default 1 second delay

                                                                                                                                                                                                                                                                                            def get_drift_statistics(self, strategy_id: str) -> Dict[str, float]:
                                                                                                                                                                                                                                                                                            """
                                                                                                                                                                                                                                                                                            Get comprehensive drift statistics

                                                                                                                                                                                                                                                                                                Args:
                                                                                                                                                                                                                                                                                                strategy_id: Strategy identifier

                                                                                                                                                                                                                                                                                                    Returns:
                                                                                                                                                                                                                                                                                                    Dictionary of drift statistics
                                                                                                                                                                                                                                                                                                    """
                                                                                                                                                                                                                                                                                                        try:
                                                                                                                                                                                                                                                                                                            if strategy_id not in self.drift_stats:
                                                                                                                                                                                                                                                                                                        return {}

                                                                                                                                                                                                                                                                                                        stats = self.drift_stats[strategy_id].copy()

                                                                                                                                                                                                                                                                                                        # Add performance metrics
                                                                                                                                                                                                                                                                                                        stats.update(
                                                                                                                                                                                                                                                                                                        {
                                                                                                                                                                                                                                                                                                        'gpu_operations': self.performance_metrics['gpu_operations'],
                                                                                                                                                                                                                                                                                                        'cpu_operations': self.performance_metrics['cpu_operations'],
                                                                                                                                                                                                                                                                                                        'fallback_operations': self.performance_metrics['fallback_operations'],
                                                                                                                                                                                                                                                                                                        'avg_operation_time': self.performance_metrics['avg_operation_time'],
                                                                                                                                                                                                                                                                                                        'backend': _backend,
                                                                                                                                                                                                                                                                                                        'hybrid_mode': self.hybrid_mode.value,
                                                                                                                                                                                                                                                                                                        }
                                                                                                                                                                                                                                                                                                        )

                                                                                                                                                                                                                                                                                                    return stats

                                                                                                                                                                                                                                                                                                        except Exception as e:
                                                                                                                                                                                                                                                                                                        logger.error("Error getting drift statistics for {0}: {1}".format(strategy_id, e))
                                                                                                                                                                                                                                                                                                    return {}

                                                                                                                                                                                                                                                                                                        def cleanup_old_data(self, max_age_hours: float = 24.0) -> int:
                                                                                                                                                                                                                                                                                                        """
                                                                                                                                                                                                                                                                                                        Clean up old drift data

                                                                                                                                                                                                                                                                                                            Args:
                                                                                                                                                                                                                                                                                                            max_age_hours: Maximum age in hours

                                                                                                                                                                                                                                                                                                                Returns:
                                                                                                                                                                                                                                                                                                                Number of cleaned entries
                                                                                                                                                                                                                                                                                                                """
                                                                                                                                                                                                                                                                                                                    try:
                                                                                                                                                                                                                                                                                                                    current_time = time.time()
                                                                                                                                                                                                                                                                                                                    max_age_seconds = max_age_hours * 3600
                                                                                                                                                                                                                                                                                                                    cleaned_count = 0

                                                                                                                                                                                                                                                                                                                        for strategy_id in list(self.history.keys()):
                                                                                                                                                                                                                                                                                                                        original_count = len(self.history[strategy_id])

                                                                                                                                                                                                                                                                                                                        # Remove old snapshots
                                                                                                                                                                                                                                                                                                                        self.history[strategy_id] = deque(
                                                                                                                                                                                                                                                                                                                        snapshot
                                                                                                                                                                                                                                                                                                                        for snapshot in self.history[strategy_id]
                                                                                                                                                                                                                                                                                                                        if current_time - snapshot.timestamp < max_age_seconds
                                                                                                                                                                                                                                                                                                                        )

                                                                                                                                                                                                                                                                                                                        cleaned_count += original_count - len(self.history[strategy_id])

                                                                                                                                                                                                                                                                                                                        # Remove strategy if no data left
                                                                                                                                                                                                                                                                                                                            if len(self.history[strategy_id]) == 0:
                                                                                                                                                                                                                                                                                                                            del self.history[strategy_id]
                                                                                                                                                                                                                                                                                                                                if strategy_id in self.drift_stats:
                                                                                                                                                                                                                                                                                                                                del self.drift_stats[strategy_id]

                                                                                                                                                                                                                                                                                                                                logger.info("Cleaned up {0} old drift entries".format(cleaned_count))
                                                                                                                                                                                                                                                                                                                            return cleaned_count

                                                                                                                                                                                                                                                                                                                                except Exception as e:
                                                                                                                                                                                                                                                                                                                                logger.error("Error cleaning up old data: {0}".format(e))
                                                                                                                                                                                                                                                                                                                            return 0

                                                                                                                                                                                                                                                                                                                                def get_error_summary(self) -> Dict[str, Any]:
                                                                                                                                                                                                                                                                                                                                """Get summary of errors encountered."""
                                                                                                                                                                                                                                                                                                                                error_counts = {}
                                                                                                                                                                                                                                                                                                                                    for error in self.error_log:
                                                                                                                                                                                                                                                                                                                                    error_type = error.error_type
                                                                                                                                                                                                                                                                                                                                    error_counts[error_type] = error_counts.get(error_type, 0) + 1

                                                                                                                                                                                                                                                                                                                                return {
                                                                                                                                                                                                                                                                                                                                'total_errors': len(self.error_log),
                                                                                                                                                                                                                                                                                                                                'error_types': error_counts,
                                                                                                                                                                                                                                                                                                                                'fallback_usage': sum(1 for e in self.error_log if e.fallback_used),
                                                                                                                                                                                                                                                                                                                                'recent_errors': [e for e in self.error_log[-10:]],  # Last 10 errors
                                                                                                                                                                                                                                                                                                                                }


                                                                                                                                                                                                                                                                                                                                    def create_entropy_drift_tracker() -> EntropyDriftTracker:
                                                                                                                                                                                                                                                                                                                                    """Create a new entropy drift tracker instance."""
                                                                                                                                                                                                                                                                                                                                return EntropyDriftTracker(max_history=100, warp_threshold=0.15, hybrid_mode=HybridMode.ADAPTIVE)


                                                                                                                                                                                                                                                                                                                                    def test_entropy_drift_tracker():
                                                                                                                                                                                                                                                                                                                                    """Test the entropy drift tracker functionality."""
                                                                                                                                                                                                                                                                                                                                    print("=== Testing Entropy Drift Tracker ===")

                                                                                                                                                                                                                                                                                                                                    tracker = create_entropy_drift_tracker()

                                                                                                                                                                                                                                                                                                                                    # Test with sample vectors
                                                                                                                                                                                                                                                                                                                                    strategy_id = "test_strategy"
                                                                                                                                                                                                                                                                                                                                    test_vectors = [
                                                                                                                                                                                                                                                                                                                                    xp.array([1.0, 2.0, 3.0]),
                                                                                                                                                                                                                                                                                                                                    xp.array([1.1, 2.1, 3.1]),
                                                                                                                                                                                                                                                                                                                                    xp.array([0.9, 1.9, 2.9]),
                                                                                                                                                                                                                                                                                                                                    xp.array([1.2, 2.2, 3.2]),
                                                                                                                                                                                                                                                                                                                                    ]

                                                                                                                                                                                                                                                                                                                                        for i, vector in enumerate(test_vectors):
                                                                                                                                                                                                                                                                                                                                        drift = tracker.record_vector(strategy_id, vector)
                                                                                                                                                                                                                                                                                                                                        print("Vector {0}: drift = {1}".format(i + 1, drift))

                                                                                                                                                                                                                                                                                                                                        # Test statistics
                                                                                                                                                                                                                                                                                                                                        stats = tracker.get_drift_statistics(strategy_id)
                                                                                                                                                                                                                                                                                                                                        print("Statistics: {0}".format(stats))

                                                                                                                                                                                                                                                                                                                                        # Test error summary
                                                                                                                                                                                                                                                                                                                                        error_summary = tracker.get_error_summary()
                                                                                                                                                                                                                                                                                                                                        print("Error summary: {0}".format(error_summary))


                                                                                                                                                                                                                                                                                                                                            if __name__ == "__main__":
                                                                                                                                                                                                                                                                                                                                            test_entropy_drift_tracker()
