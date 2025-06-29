# -*- coding: utf-8 -*-
"""
Balance Loader System
====================

Advanced balance loading system for ALIF/ALEPH coordination with GPU/CPU
optimization, float decay monitoring, and cross-pipeline load balancing.

Features:
- Dynamic load balancing between ALIF and ALEPH
- GPU/CPU entropy distribution
- Float decay monitoring and correction
- Cross-pipeline optimization
- Real-time load adjustment
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class LoadMode(Enum):
    """Load balancing modes."""

    BALANCED = "balanced"
    ALIF_HEAVY = "alif_heavy"
    ALEPH_HEAVY = "aleph_heavy"
    COMPRESSED = "compressed"
    OVERLOAD = "overload"


@dataclass
class LoadMetrics:
    """Load metrics for balance monitoring."""

    alif_load: float = 0.0
    aleph_load: float = 0.0
    gpu_entropy: float = 0.0
    cpu_entropy: float = 0.0
    float_decay: float = 0.0
    drift_threshold: float = 0.023
    balance_needed: bool = False
    compression_ratio: float = 1.0


@dataclass
class BalanceAction:
    """Action to take for load balancing."""

    action_type: str
    target_component: str
    adjustment_value: float
    priority: int
    timestamp: float = field(default_factory=time.time)


class BalanceLoader:
    """
    Advanced balance loader for ALIF/ALEPH coordination.

    Manages load balancing, float decay monitoring, and cross-pipeline
    optimization to ensure optimal performance.
    """

    def __init__(self):
        self.load_metrics = LoadMetrics()
        self.current_mode = LoadMode.BALANCED
        self.balance_threshold = 5.0
        self.compression_threshold = 0.8
        self.overload_threshold = 0.9

        # Performance tracking
        self.total_adjustments = 0
        self.successful_adjustments = 0
        self.failed_adjustments = 0

        # Callbacks for load changes
        self.load_callbacks: List[Callable[[LoadMetrics], None]] = []
        self.balance_callbacks: List[Callable[[BalanceAction], None]] = []

        # Historical data
        self.load_history: List[LoadMetrics] = []
        self.max_history_size = 1000

        logger.info("⚖️ Balance Loader initialized")

    def register_load_callback(self, callback: Callable[[LoadMetrics], None]):
        """Register callback for load metric updates."""
        self.load_callbacks.append(callback)
        logger.debug(f"Registered load callback: {callback.__name__}")

    def register_balance_callback(self, callback: Callable[[BalanceAction], None]):
        """Register callback for balance actions."""
        self.balance_callbacks.append(callback)
        logger.debug(f"Registered balance callback: {callback.__name__}")

    def update_load_metrics(
        self, alif_load: float, aleph_load: float, gpu_entropy: float, cpu_entropy: float, float_decay: float = 0.0
    ) -> LoadMetrics:
        """Update load metrics and determine if balance is needed."""
        # Update metrics
        self.load_metrics.alif_load = alif_load
        self.load_metrics.aleph_load = aleph_load
        self.load_metrics.gpu_entropy = gpu_entropy
        self.load_metrics.cpu_entropy = cpu_entropy
        self.load_metrics.float_decay = float_decay

        # Calculate load difference
        load_diff = abs(alif_load - aleph_load)
        self.load_metrics.balance_needed = load_diff > self.balance_threshold

        # Calculate compression ratio
        total_load = alif_load + aleph_load
        if total_load > 0:
            self.load_metrics.compression_ratio = min(1.0, total_load / 20.0)
        else:
            self.load_metrics.compression_ratio = 0.0

        # Determine load mode
        self._determine_load_mode()

        # Store in history
        self._store_load_history()

        # Execute callbacks
        self._execute_load_callbacks()

        # Check if balance action is needed
        if self.load_metrics.balance_needed:
            self._create_balance_action()

        return self.load_metrics

    def _determine_load_mode(self):
        """Determine the current load mode based on metrics."""
        if self.load_metrics.compression_ratio > self.overload_threshold:
            self.current_mode = LoadMode.OVERLOAD
        elif self.load_metrics.compression_ratio > self.compression_threshold:
            self.current_mode = LoadMode.COMPRESSED
        elif self.load_metrics.alif_load > self.load_metrics.aleph_load * 1.5:
            self.current_mode = LoadMode.ALIF_HEAVY
        elif self.load_metrics.aleph_load > self.load_metrics.alif_load * 1.5:
            self.current_mode = LoadMode.ALEPH_HEAVY
        else:
            self.current_mode = LoadMode.BALANCED

    def _store_load_history(self):
        """Store current load metrics in history."""
        # Create a copy of current metrics
        metrics_copy = LoadMetrics(
            alif_load=self.load_metrics.alif_load,
            aleph_load=self.load_metrics.aleph_load,
            gpu_entropy=self.load_metrics.gpu_entropy,
            cpu_entropy=self.load_metrics.cpu_entropy,
            float_decay=self.load_metrics.float_decay,
            drift_threshold=self.load_metrics.drift_threshold,
            balance_needed=self.load_metrics.balance_needed,
            compression_ratio=self.load_metrics.compression_ratio,
        )

        self.load_history.append(metrics_copy)

        # Limit history size
        if len(self.load_history) > self.max_history_size:
            self.load_history = self.load_history[-self.max_history_size // 2 :]

    def _execute_load_callbacks(self):
        """Execute all registered load callbacks."""
        for callback in self.load_callbacks:
            try:
                callback(self.load_metrics)
            except Exception as e:
                logger.error(f"Load callback error: {e}")

    def _create_balance_action(self):
        """Create and execute balance action."""
        self.total_adjustments += 1

        # Determine action type based on load mode
        if self.current_mode == LoadMode.ALIF_HEAVY:
            action = BalanceAction(
                action_type="reduce_alif_float",
                target_component="ALIF",
                adjustment_value=self.load_metrics.alif_load - self.load_metrics.aleph_load,
                priority=1,
            )
        elif self.current_mode == LoadMode.ALEPH_HEAVY:
            action = BalanceAction(
                action_type="reduce_aleph_trigger",
                target_component="ALEPH",
                adjustment_value=self.load_metrics.aleph_load - self.load_metrics.alif_load,
                priority=1,
            )
        elif self.current_mode == LoadMode.COMPRESSED:
            action = BalanceAction(
                action_type="compress_entropy",
                target_component="BOTH",
                adjustment_value=self.load_metrics.compression_ratio,
                priority=2,
            )
        elif self.current_mode == LoadMode.OVERLOAD:
            action = BalanceAction(
                action_type="emergency_throttle", target_component="BOTH", adjustment_value=1.0, priority=3
            )
        else:
            return

        # Execute balance callbacks
        self._execute_balance_callbacks(action)

        # Log the action
        logger.info(
            f"Balance action: {
                action.action_type} for {
                action.target_component} "
            f"(adjustment: {
                action.adjustment_value:.2f})"
        )

    def _execute_balance_callbacks(self, action: BalanceAction):
        """Execute all registered balance callbacks."""
        for callback in self.balance_callbacks:
            try:
                callback(action)
            except Exception as e:
                logger.error(f"Balance callback error: {e}")
                self.failed_adjustments += 1
        else:
            self.successful_adjustments += 1

    def get_optimal_route(self, alif_load: float, aleph_load: float) -> str:
        """Determine optimal routing based on current loads."""
        if alif_load > aleph_load * 1.5:
            return "ALEPH"  # Route to ALEPH to reduce ALIF load
        elif aleph_load > alif_load * 1.5:
            return "ALIF"  # Route to ALIF to reduce ALEPH load
        else:
            return "SHARED"  # Route can go to either

    def monitor_float_decay(self, predicted_tick: float, actual_tick: float) -> bool:
        """Monitor float decay and flag if threshold exceeded."""
        decay = abs(predicted_tick - actual_tick)
        if decay > self.load_metrics.drift_threshold:
            logger.warning(
                f"Float decay detected: {
                    decay:.3f}s (threshold: {
                    self.load_metrics.drift_threshold:.3f}s)"
            )
            return True
        return False

    def get_compression_suggestions(self) -> Dict[str, Any]:
        """Get suggestions for compression based on current loads."""
        suggestions = {
            "compression_needed": self.current_mode in [LoadMode.COMPRESSED, LoadMode.OVERLOAD],
            "suggested_mode": self.current_mode.value,
            "alif_reduction": 0.0,
            "aleph_reduction": 0.0,
            "entropy_compression": 0.0,
        }

        if self.current_mode == LoadMode.ALIF_HEAVY:
            suggestions["alif_reduction"] = (self.load_metrics.alif_load - self.load_metrics.aleph_load) * 0.3
        elif self.current_mode == LoadMode.ALEPH_HEAVY:
            suggestions["aleph_reduction"] = (self.load_metrics.aleph_load - self.load_metrics.alif_load) * 0.3
        elif self.current_mode in [LoadMode.COMPRESSED, LoadMode.OVERLOAD]:
            suggestions["entropy_compression"] = self.load_metrics.compression_ratio * 0.5

        return suggestions

    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        return {
            "current_mode": self.current_mode.value,
            "total_adjustments": self.total_adjustments,
            "successful_adjustments": self.successful_adjustments,
            "failed_adjustments": self.failed_adjustments,
            "success_rate": self.successful_adjustments / max(1, self.total_adjustments),
            "load_history_size": len(self.load_history),
            "balance_needed": self.load_metrics.balance_needed,
            "compression_ratio": self.load_metrics.compression_ratio,
            "float_decay": self.load_metrics.float_decay,
            "alif_load": self.load_metrics.alif_load,
            "aleph_load": self.load_metrics.aleph_load,
            "gpu_entropy": self.load_metrics.gpu_entropy,
            "cpu_entropy": self.load_metrics.cpu_entropy,
        }

    def get_load_trends(self, window_size: int = 10) -> Dict[str, List[float]]:
        """Get load trends over the specified window."""
        if len(self.load_history) < window_size:
            window_size = len(self.load_history)

        recent_metrics = self.load_history[-window_size:]

        return {
            "alif_loads": [m.alif_load for m in recent_metrics],
            "aleph_loads": [m.aleph_load for m in recent_metrics],
            "gpu_entropies": [m.gpu_entropy for m in recent_metrics],
            "cpu_entropies": [m.cpu_entropy for m in recent_metrics],
            "compression_ratios": [m.compression_ratio for m in recent_metrics],
        }


# Global balance loader instance
balance_loader = BalanceLoader()

# Integration functions for external use


def get_balance_loader() -> BalanceLoader:
    """Get the global balance loader instance."""
    return balance_loader


def update_load_metrics(
    alif_load: float, aleph_load: float, gpu_entropy: float, cpu_entropy: float, float_decay: float = 0.0
) -> LoadMetrics:
    """Update load metrics and get balance recommendations."""
    return balance_loader.update_load_metrics(alif_load, aleph_load, gpu_entropy, cpu_entropy, float_decay)


def get_optimal_route(alif_load: float, aleph_load: float) -> str:
    """Get optimal routing recommendation."""
    return balance_loader.get_optimal_route(alif_load, aleph_load)


def monitor_float_decay(predicted_tick: float, actual_tick: float) -> bool:
    """Monitor float decay and return True if threshold exceeded."""
    return balance_loader.monitor_float_decay(predicted_tick, actual_tick)


def get_balance_statistics() -> Dict[str, Any]:
    """Get current balance statistics."""
    return balance_loader.get_system_statistics()


def register_load_callback(callback: Callable[[LoadMetrics], None]):
    """Register a callback for load metric updates."""
    balance_loader.register_load_callback(callback)


def register_balance_callback(callback: Callable[[BalanceAction], None]):
    """Register a callback for balance actions."""
    balance_loader.register_balance_callback(callback)
