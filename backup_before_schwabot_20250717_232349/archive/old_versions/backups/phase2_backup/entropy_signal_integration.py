"""Module for Schwabot trading system."""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entropy Signal Integration Module for Schwabot Trading System.

This module implements the entropy signal flow through the trading pipeline,
managing timing cycles, signal processing, and performance monitoring based on
the entropy_signal_integration.yaml configuration.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml

# Import core modules
    try:
    from .dual_state_router import DualStateRouter, get_dual_state_router
    from .neural_processing_engine import NeuralProcessingEngine
    from .order_book_analyzer import OrderBookAnalyzer
        except ImportError:
        # Fallback imports for testing
        OrderBookAnalyzer = None
        DualStateRouter = None
        NeuralProcessingEngine = None

        logger = logging.getLogger(__name__)


            class EntropyState(Enum):
    """Class for Schwabot trading functionality."""
            """Class for Schwabot trading functionality."""
            """Entropy-based system states."""

            INERT = "INERT"
            NEUTRAL = "NEUTRAL"
            AGGRESSIVE = "AGGRESSIVE"
            PASSIVE = "PASSIVE"
            ENTROPIC_INVERSION_ACTIVATED = "ENTROPIC_INVERSION_ACTIVATED"
            ENTROPIC_SURGE = "ENTROPIC_SURGE"
            ENTROPIC_CALM = "ENTROPIC_CALM"


            @dataclass
                class EntropySignal:
    """Class for Schwabot trading functionality."""
                """Class for Schwabot trading functionality."""
                """Represents an entropy signal with metadata."""

                timestamp: float
                entropy_value: float
                routing_state: str
                quantum_state: str
                confidence: float
                metadata: Dict[str, Any] = field(default_factory=dict)


                @dataclass
                    class TimingCycle:
    """Class for Schwabot trading functionality."""
                    """Class for Schwabot trading functionality."""
                    """Represents a timing cycle with entropy adaptation."""

                    cycle_type: str
                    base_interval_ms: int
                    current_interval_ms: int
                    entropy_multiplier: float
                    last_execution: float
                    next_execution: float
                    enabled: bool = True


                    @dataclass
                        class PerformanceMetrics:
    """Class for Schwabot trading functionality."""
                        """Class for Schwabot trading functionality."""
                        """Performance metrics for entropy signal processing."""

                        entropy_detection_rate: float
                        signal_latency_ms: float
                        routing_accuracy: float
                        quantum_state_activation_rate: float
                        timestamp: float


                            class EntropySignalIntegrator:
    """Class for Schwabot trading functionality."""
                            """Class for Schwabot trading functionality."""
                            """
                            Main integrator for entropy signals in the trading pipeline.

                                Manages the flow of entropy signals through:
                                - Order book analysis
                                - Dual state router
                                - Neural processing engine
                                - Timing cycle adaptation
                                - Performance monitoring
                                """

                                    def __init__(self, config_path: str = "config/entropy_signal_integration.yaml") -> None:
                                    """Initialize the entropy signal integrator."""
                                    self.config = self._load_config(config_path)
                                    self.current_state = EntropyState.NEUTRAL

                                    # Initialize core components
                                    self.order_book_analyzer = None
                                    self.dual_state_router = None
                                    self.neural_engine = None

                                    # Signal processing
                                    self.entropy_buffer = []
                                    self.signal_history = []
                                    self.performance_metrics = []

                                    # Timing cycles
                                    self.tick_cycle = None
                                    self.routing_cycle = None
                                    self.cycles = {}

                                    # Performance tracking
                                    self.metrics_start_time = time.time()
                                    self.total_signals_processed = 0
                                    self.successful_detections = 0

                                    # Initialize components
                                    self._initialize_components()
                                    self._setup_timing_cycles()

                                    logger.info("Entropy Signal Integrator initialized")

                                        def _load_config(self, config_path: str) -> Dict[str, Any]:
                                        """Load configuration from YAML file."""
                                            try:
                                            config_file = Path(config_path)
                                                if not config_file.exists():
                                                logger.warning(f"Config file {config_path} not found, using defaults")
                                            return self._get_default_config()

                                                with open(config_file, 'r') as f:
                                                config = yaml.safe_load(f)

                                                logger.info(f"Loaded entropy signal configuration from {config_path}")
                                            return config

                                                except Exception as e:
                                                logger.error(f"Error loading config: {e}")
                                            return self._get_default_config()

                                                def _get_default_config(self) -> Dict[str, Any]:
                                                """Get default configuration if YAML file is not available."""
                                            return {
                                            "entropy_signal_flow": {
                                            "order_book_analysis": {
                                            "enabled": True,
                                            "scan_interval_ms": 100,
                                            "entropy_calculation": {
                                            "method": "spread_volatility",
                                            "lookback_periods": 5,
                                            "threshold_high": 0.022,
                                            "threshold_medium": 0.015,
                                            "threshold_low": 0.008,
                                            },
                                            },
                                            "dual_state_router": {
                                            "enabled": True,
                                            "entropy_routing": {
                                            "aggressive_threshold": 0.018,
                                            "passive_threshold": 0.012,
                                            "buffer_size": 100,
                                            "decision_window": 3,
                                            },
                                            },
                                            "neural_processing": {
                                            "enabled": True,
                                            "phase_entropy": {
                                            "accumulation_rate": 0.2,
                                            "decay_rate": 0.8,
                                            "activation_threshold": 0.019,
                                            },
                                            },
                                            },
                                            "timing_cycles": {
                                            "tick_cycle": {
                                            "base_interval_ms": 50,
                                            "entropy_adaptive": True,
                                            "tick_rate_adjustment": {
                                            "high_entropy_multiplier": 0.5,
                                            "low_entropy_multiplier": 2.0,
                                            "max_tick_rate_ms": 10,
                                            "min_tick_rate_ms": 200,
                                            },
                                            },
                                            "routing_cycle": {
                                            "base_interval_ms": 500,
                                            "entropy_adaptive": True,
                                            "routing_frequency": {
                                            "aggressive_mode_interval_ms": 200,
                                            "passive_mode_interval_ms": 1000,
                                            "neutral_mode_interval_ms": 500,
                                            },
                                            },
                                            },
                                            }

                                                def _initialize_components(self) -> None:
                                                """Initialize core trading components."""
                                                    try:
                                                    # Initialize order book analyzer
                                                        if OrderBookAnalyzer:
                                                        self.order_book_analyzer = OrderBookAnalyzer()
                                                        logger.info("Order book analyzer initialized")

                                                        # Initialize dual state router
                                                            if DualStateRouter:
                                                            self.dual_state_router = get_dual_state_router()
                                                            logger.info("Dual state router initialized")

                                                            # Initialize neural processing engine
                                                                if NeuralProcessingEngine:
                                                                self.neural_engine = NeuralProcessingEngine()
                                                                logger.info("Neural processing engine initialized")

                                                                    except Exception as e:
                                                                    logger.error(f"Error initializing components: {e}")

                                                                        def _setup_timing_cycles(self) -> None:
                                                                        """Setup timing cycles based on configuration."""
                                                                            try:
                                                                            timing_config = self.config.get("timing_cycles", {})

                                                                            # Setup tick cycle
                                                                            tick_config = timing_config.get("tick_cycle", {})
                                                                            self.tick_cycle = TimingCycle(
                                                                            cycle_type="tick",
                                                                            base_interval_ms=tick_config.get("base_interval_ms", 50),
                                                                            current_interval_ms=tick_config.get("base_interval_ms", 50),
                                                                            entropy_multiplier=1.0,
                                                                            last_execution=time.time(),
                                                                            next_execution=time.time() + (tick_config.get("base_interval_ms", 50) / 1000.0),
                                                                            enabled=tick_config.get("entropy_adaptive", True),
                                                                            )

                                                                            # Setup routing cycle
                                                                            routing_config = timing_config.get("routing_cycle", {})
                                                                            self.routing_cycle = TimingCycle(
                                                                            cycle_type="routing",
                                                                            base_interval_ms=routing_config.get("base_interval_ms", 500),
                                                                            current_interval_ms=routing_config.get("base_interval_ms", 500),
                                                                            entropy_multiplier=1.0,
                                                                            last_execution=time.time(),
                                                                            next_execution=time.time() + (routing_config.get("base_interval_ms", 500) / 1000.0),
                                                                            enabled=routing_config.get("entropy_adaptive", True),
                                                                            )

                                                                            # Store cycles
                                                                            self.cycles = {"tick": self.tick_cycle, "routing": self.routing_cycle}

                                                                            logger.info("Timing cycles initialized")

                                                                                except Exception as e:
                                                                                logger.error(f"Error setting up timing cycles: {e}")

                                                                                    def process_entropy_signal(self, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]]) -> EntropySignal:
                                                                                    """
                                                                                    Process entropy signal through the complete pipeline.

                                                                                        Args:
                                                                                        bids: Order book bids
                                                                                        asks: Order book asks

                                                                                            Returns:
                                                                                            Processed entropy signal with routing and quantum states
                                                                                            """
                                                                                            start_time = time.time()

                                                                                                try:
                                                                                                # Step 1: Order book analysis and entropy calculation
                                                                                                entropy_result = self._calculate_entropy(bids, asks)
                                                                                                entropy_value = entropy_result.get("entropy", 0.0)

                                                                                                # Step 2: Route entropy through dual state router
                                                                                                routing_state = self._route_entropy(entropy_value)

                                                                                                # Step 3: Inject phase entropy into neural engine
                                                                                                quantum_state = self._inject_phase_entropy(entropy_value)

                                                                                                # Step 4: Calculate confidence
                                                                                                confidence = self._calculate_signal_confidence(entropy_value, routing_state, quantum_state)

                                                                                                # Create entropy signal
                                                                                                signal = EntropySignal(
                                                                                                timestamp=time.time(),
                                                                                                entropy_value=entropy_value,
                                                                                                routing_state=routing_state,
                                                                                                quantum_state=quantum_state,
                                                                                                confidence=confidence,
                                                                                                metadata={
                                                                                                "processing_time_ms": (time.time() - start_time) * 1000,
                                                                                                "signal_quality": self._assess_signal_quality(entropy_value),
                                                                                                },
                                                                                                )

                                                                                                # Update buffers and history
                                                                                                self._update_signal_buffers(signal)

                                                                                                # Update timing cycles
                                                                                                self._adapt_timing_cycles(signal)

                                                                                                # Update performance metrics
                                                                                                self._update_performance_metrics(signal)

                                                                                                self.total_signals_processed += 1
                                                                                                    if signal.confidence > 0.5:
                                                                                                    self.successful_detections += 1

                                                                                                    logger.debug(
                                                                                                    f"Processed entropy signal: {entropy_value:.6f}, " f"routing: {routing_state}, quantum: {quantum_state}"
                                                                                                    )

                                                                                                return signal

                                                                                                    except Exception as e:
                                                                                                    logger.error(f"Error processing entropy signal: {e}")
                                                                                                    # Return fallback signal
                                                                                                return EntropySignal(
                                                                                                timestamp=time.time(),
                                                                                                entropy_value=0.0,
                                                                                                routing_state="NEUTRAL",
                                                                                                quantum_state="INERT",
                                                                                                confidence=0.0,
                                                                                                metadata={"error": str(e)},
                                                                                                )

                                                                                                    def _calculate_entropy(self, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]]) -> Dict[str, Any]:
                                                                                                    """Calculate entropy from order book data."""
                                                                                                        try:
                                                                                                            if self.order_book_analyzer:
                                                                                                            # Use the scan_entropy method from order book analyzer
                                                                                                        return self.order_book_analyzer.scan_entropy(bids, asks)
                                                                                                            else:
                                                                                                            # Fallback entropy calculation
                                                                                                                if len(bids) >= 5 and len(asks) >= 5:
                                                                                                                bid_prices = [price for price, _ in bids[:5]]
                                                                                                                ask_prices = [price for price, _ in asks[:5]]
                                                                                                                spreads = [ask - bid for bid, ask in zip(bid_prices, ask_prices)]
                                                                                                                spread_changes = np.diff(spreads)
                                                                                                                entropy_sigma = np.std(spread_changes)
                                                                                                            return {"signal": entropy_sigma > 0.022, "entropy": entropy_sigma}
                                                                                                                else:
                                                                                                            return {"signal": False, "entropy": 0.0}

                                                                                                                except Exception as e:
                                                                                                                logger.error(f"Error calculating entropy: {e}")
                                                                                                            return {"signal": False, "entropy": 0.0}

                                                                                                                def _route_entropy(self, entropy_value: float) -> str:
                                                                                                                """Route entropy through dual state router."""
                                                                                                                    try:
                                                                                                                        if self.dual_state_router:
                                                                                                                    return self.dual_state_router.route_entropy(entropy_value)
                                                                                                                        else:
                                                                                                                        # Fallback routing logic
                                                                                                                            if entropy_value > 0.018:
                                                                                                                        return "ROUTE_ACTIVE"
                                                                                                                            else:
                                                                                                                        return "ROUTE_PASSIVE"

                                                                                                                            except Exception as e:
                                                                                                                            logger.error(f"Error routing entropy: {e}")
                                                                                                                        return "ROUTE_PASSIVE"

                                                                                                                            def _inject_phase_entropy(self, entropy_value: float) -> str:
                                                                                                                            """Inject phase entropy into neural processing engine."""
                                                                                                                                try:
                                                                                                                                    if self.neural_engine:
                                                                                                                                return self.neural_engine.inject_phase_entropy(entropy_value)
                                                                                                                                    else:
                                                                                                                                    # Fallback quantum state logic
                                                                                                                                        if entropy_value > 0.019:
                                                                                                                                    return "ENTROPIC_INVERSION_ACTIVATED"
                                                                                                                                        else:
                                                                                                                                    return "INERT"

                                                                                                                                        except Exception as e:
                                                                                                                                        logger.error(f"Error injecting phase entropy: {e}")
                                                                                                                                    return "INERT"

                                                                                                                                        def _calculate_signal_confidence(self, entropy_value: float, routing_state: str, quantum_state: str) -> float:
                                                                                                                                        """Calculate confidence in the entropy signal."""
                                                                                                                                            try:
                                                                                                                                            # Base confidence from entropy value
                                                                                                                                            base_confidence = min(entropy_value * 10, 1.0)

                                                                                                                                            # Adjust based on routing state
                                                                                                                                            routing_confidence = {"ROUTE_ACTIVE": 0.8, "ROUTE_PASSIVE": 0.6, "NEUTRAL": 0.5}.get(routing_state, 0.5)

                                                                                                                                            # Adjust based on quantum state
                                                                                                                                            quantum_confidence = {
                                                                                                                                            "ENTROPIC_INVERSION_ACTIVATED": 0.9,
                                                                                                                                            "INERT": 0.3,
                                                                                                                                            "ENTROPIC_SURGE": 0.8,
                                                                                                                                            "ENTROPIC_CALM": 0.4,
                                                                                                                                            }.get(quantum_state, 0.5)

                                                                                                                                            # Weighted average
                                                                                                                                            confidence = base_confidence * 0.4 + routing_confidence * 0.3 + quantum_confidence * 0.3

                                                                                                                                        return min(max(confidence, 0.0), 1.0)

                                                                                                                                            except Exception as e:
                                                                                                                                            logger.error(f"Error calculating signal confidence: {e}")
                                                                                                                                        return 0.5

                                                                                                                                            def _assess_signal_quality(self, entropy_value: float) -> str:
                                                                                                                                            """Assess the quality of the entropy signal."""
                                                                                                                                                try:
                                                                                                                                                thresholds = (
                                                                                                                                                self.config.get("entropy_signal_flow", {}).get("order_book_analysis", {}).get("entropy_calculation", {})
                                                                                                                                                )

                                                                                                                                                high_threshold = thresholds.get("threshold_high", 0.022)
                                                                                                                                                medium_threshold = thresholds.get("threshold_medium", 0.015)
                                                                                                                                                low_threshold = thresholds.get("threshold_low", 0.008)

                                                                                                                                                    if entropy_value >= high_threshold:
                                                                                                                                                return "HIGH"
                                                                                                                                                    elif entropy_value >= medium_threshold:
                                                                                                                                                return "MEDIUM"
                                                                                                                                                    elif entropy_value >= low_threshold:
                                                                                                                                                return "LOW"
                                                                                                                                                    else:
                                                                                                                                                return "VERY_LOW"

                                                                                                                                                    except Exception as e:
                                                                                                                                                    logger.error(f"Error assessing signal quality: {e}")
                                                                                                                                                return "UNKNOWN"

                                                                                                                                                    def _update_signal_buffers(self, signal: EntropySignal) -> None:
                                                                                                                                                    """Update signal buffers and history."""
                                                                                                                                                        try:
                                                                                                                                                        # Update entropy buffer
                                                                                                                                                        self.entropy_buffer.append(signal.entropy_value)

                                                                                                                                                        # Maintain buffer size
                                                                                                                                                        max_buffer_size = (
                                                                                                                                                        self.config.get("entropy_signal_flow", {})
                                                                                                                                                        .get("dual_state_router", {})
                                                                                                                                                        .get("entropy_routing", {})
                                                                                                                                                        .get("buffer_size", 100)
                                                                                                                                                        )
                                                                                                                                                            if len(self.entropy_buffer) > max_buffer_size:
                                                                                                                                                            self.entropy_buffer.pop(0)

                                                                                                                                                            # Update signal history
                                                                                                                                                            self.signal_history.append(signal)

                                                                                                                                                            # Maintain history size
                                                                                                                                                            max_history = 1000
                                                                                                                                                                if len(self.signal_history) > max_history:
                                                                                                                                                                self.signal_history.pop(0)

                                                                                                                                                                    except Exception as e:
                                                                                                                                                                    logger.error(f"Error updating signal buffers: {e}")

                                                                                                                                                                        def _adapt_timing_cycles(self, signal: EntropySignal) -> None:
                                                                                                                                                                        """Adapt timing cycles based on entropy signal."""
                                                                                                                                                                            try:
                                                                                                                                                                            # Get timing configuration
                                                                                                                                                                            timing_config = self.config.get("timing_cycles", {})

                                                                                                                                                                            # Adapt tick cycle
                                                                                                                                                                                if self.tick_cycle and self.tick_cycle.enabled:
                                                                                                                                                                                tick_config = timing_config.get("tick_cycle", {})
                                                                                                                                                                                adjustment_config = tick_config.get("tick_rate_adjustment", {})

                                                                                                                                                                                # Calculate entropy multiplier
                                                                                                                                                                                if signal.entropy_value > 0.018:  # High entropy
                                                                                                                                                                                multiplier = adjustment_config.get("high_entropy_multiplier", 0.5)
                                                                                                                                                                                elif signal.entropy_value < 0.008:  # Low entropy
                                                                                                                                                                                multiplier = adjustment_config.get("low_entropy_multiplier", 2.0)
                                                                                                                                                                                else:  # Medium entropy
                                                                                                                                                                                multiplier = 1.0

                                                                                                                                                                                # Update tick cycle
                                                                                                                                                                                self.tick_cycle.entropy_multiplier = multiplier
                                                                                                                                                                                new_interval = int(self.tick_cycle.base_interval_ms * multiplier)

                                                                                                                                                                                # Apply limits
                                                                                                                                                                                max_rate = adjustment_config.get("max_tick_rate_ms", 10)
                                                                                                                                                                                min_rate = adjustment_config.get("min_tick_rate_ms", 200)
                                                                                                                                                                                new_interval = max(min_rate, min(max_rate, new_interval))

                                                                                                                                                                                self.tick_cycle.current_interval_ms = new_interval

                                                                                                                                                                                # Adapt routing cycle
                                                                                                                                                                                    if self.routing_cycle and self.routing_cycle.enabled:
                                                                                                                                                                                    routing_config = timing_config.get("routing_cycle", {})
                                                                                                                                                                                    frequency_config = routing_config.get("routing_frequency", {})

                                                                                                                                                                                    # Determine routing interval based on state
                                                                                                                                                                                        if signal.routing_state == "ROUTE_ACTIVE":
                                                                                                                                                                                        interval = frequency_config.get("aggressive_mode_interval_ms", 200)
                                                                                                                                                                                            elif signal.routing_state == "ROUTE_PASSIVE":
                                                                                                                                                                                            interval = frequency_config.get("passive_mode_interval_ms", 1000)
                                                                                                                                                                                                else:
                                                                                                                                                                                                interval = frequency_config.get("neutral_mode_interval_ms", 500)

                                                                                                                                                                                                self.routing_cycle.current_interval_ms = interval

                                                                                                                                                                                                    except Exception as e:
                                                                                                                                                                                                    logger.error(f"Error adapting timing cycles: {e}")

                                                                                                                                                                                                        def _update_performance_metrics(self, signal: EntropySignal) -> None:
                                                                                                                                                                                                        """Update performance metrics."""
                                                                                                                                                                                                            try:
                                                                                                                                                                                                            # Calculate current metrics
                                                                                                                                                                                                            detection_rate = self.successful_detections / max(self.total_signals_processed, 1)

                                                                                                                                                                                                            # Calculate average latency
                                                                                                                                                                                                                if self.signal_history:
                                                                                                                                                                                                                latencies = [s.metadata.get("processing_time_ms", 0) for s in self.signal_history[-100:]]
                                                                                                                                                                                                                avg_latency = np.mean(latencies)
                                                                                                                                                                                                                    else:
                                                                                                                                                                                                                    avg_latency = 0.0

                                                                                                                                                                                                                    # Calculate routing accuracy (simplified)
                                                                                                                                                                                                                    routing_accuracy = 0.85  # Placeholder - would need historical data

                                                                                                                                                                                                                    # Calculate quantum state activation rate
                                                                                                                                                                                                                        if self.signal_history:
                                                                                                                                                                                                                        activations = sum(1 for s in self.signal_history if s.quantum_state == "ENTROPIC_INVERSION_ACTIVATED")
                                                                                                                                                                                                                        activation_rate = activations / len(self.signal_history)
                                                                                                                                                                                                                            else:
                                                                                                                                                                                                                            activation_rate = 0.0

                                                                                                                                                                                                                            # Create metrics object
                                                                                                                                                                                                                            metrics = PerformanceMetrics(
                                                                                                                                                                                                                            entropy_detection_rate=detection_rate,
                                                                                                                                                                                                                            signal_latency_ms=avg_latency,
                                                                                                                                                                                                                            routing_accuracy=routing_accuracy,
                                                                                                                                                                                                                            quantum_state_activation_rate=activation_rate,
                                                                                                                                                                                                                            timestamp=time.time(),
                                                                                                                                                                                                                            )

                                                                                                                                                                                                                            self.performance_metrics.append(metrics)

                                                                                                                                                                                                                            # Maintain metrics history
                                                                                                                                                                                                                            max_metrics = 1000
                                                                                                                                                                                                                                if len(self.performance_metrics) > max_metrics:
                                                                                                                                                                                                                                self.performance_metrics.pop(0)

                                                                                                                                                                                                                                    except Exception as e:
                                                                                                                                                                                                                                    logger.error(f"Error updating performance metrics: {e}")

                                                                                                                                                                                                                                        def should_execute_cycle(self, cycle_type: str) -> bool:
                                                                                                                                                                                                                                        """Check if a timing cycle should execute."""
                                                                                                                                                                                                                                            try:
                                                                                                                                                                                                                                            cycle = self.cycles.get(cycle_type)
                                                                                                                                                                                                                                                if not cycle or not cycle.enabled:
                                                                                                                                                                                                                                            return False

                                                                                                                                                                                                                                            current_time = time.time()
                                                                                                                                                                                                                                                if current_time >= cycle.next_execution:
                                                                                                                                                                                                                                                # Update execution times
                                                                                                                                                                                                                                                cycle.last_execution = current_time
                                                                                                                                                                                                                                                cycle.next_execution = current_time + (cycle.current_interval_ms / 1000.0)
                                                                                                                                                                                                                                            return True

                                                                                                                                                                                                                                        return False

                                                                                                                                                                                                                                            except Exception as e:
                                                                                                                                                                                                                                            logger.error(f"Error checking cycle execution: {e}")
                                                                                                                                                                                                                                        return False

                                                                                                                                                                                                                                            def get_current_state(self) -> Dict[str, Any]:
                                                                                                                                                                                                                                            """Get current system state."""
                                                                                                                                                                                                                                                try:
                                                                                                                                                                                                                                            return {
                                                                                                                                                                                                                                            "current_entropy_state": self.current_state.value,
                                                                                                                                                                                                                                            "tick_cycle": {
                                                                                                                                                                                                                                            "enabled": self.tick_cycle.enabled if self.tick_cycle else False,
                                                                                                                                                                                                                                            "current_interval_ms": (self.tick_cycle.current_interval_ms if self.tick_cycle else 0),
                                                                                                                                                                                                                                            "entropy_multiplier": (self.tick_cycle.entropy_multiplier if self.tick_cycle else 1.0),
                                                                                                                                                                                                                                            },
                                                                                                                                                                                                                                            "routing_cycle": {
                                                                                                                                                                                                                                            "enabled": self.routing_cycle.enabled if self.routing_cycle else False,
                                                                                                                                                                                                                                            "current_interval_ms": (self.routing_cycle.current_interval_ms if self.routing_cycle else 0),
                                                                                                                                                                                                                                            "entropy_multiplier": (self.routing_cycle.entropy_multiplier if self.routing_cycle else 1.0),
                                                                                                                                                                                                                                            },
                                                                                                                                                                                                                                            "signal_stats": {
                                                                                                                                                                                                                                            "total_processed": self.total_signals_processed,
                                                                                                                                                                                                                                            "successful_detections": self.successful_detections,
                                                                                                                                                                                                                                            "detection_rate": self.successful_detections / max(self.total_signals_processed, 1),
                                                                                                                                                                                                                                            "buffer_size": len(self.entropy_buffer),
                                                                                                                                                                                                                                            "history_size": len(self.signal_history),
                                                                                                                                                                                                                                            },
                                                                                                                                                                                                                                            }

                                                                                                                                                                                                                                                except Exception as e:
                                                                                                                                                                                                                                                logger.error(f"Error getting current state: {e}")
                                                                                                                                                                                                                                            return {}

                                                                                                                                                                                                                                                def get_performance_summary(self) -> Dict[str, Any]:
                                                                                                                                                                                                                                                """Get performance summary."""
                                                                                                                                                                                                                                                    try:
                                                                                                                                                                                                                                                        if not self.performance_metrics:
                                                                                                                                                                                                                                                    return {}

                                                                                                                                                                                                                                                    recent_metrics = self.performance_metrics[-100:]  # Last 100 metrics

                                                                                                                                                                                                                                                return {
                                                                                                                                                                                                                                                "average_detection_rate": np.mean([m.entropy_detection_rate for m in recent_metrics]),
                                                                                                                                                                                                                                                "average_latency_ms": np.mean([m.signal_latency_ms for m in recent_metrics]),
                                                                                                                                                                                                                                                "average_routing_accuracy": np.mean([m.routing_accuracy for m in recent_metrics]),
                                                                                                                                                                                                                                                "average_activation_rate": np.mean([m.quantum_state_activation_rate for m in recent_metrics]),
                                                                                                                                                                                                                                                "total_signals_processed": self.total_signals_processed,
                                                                                                                                                                                                                                                "uptime_seconds": time.time() - self.metrics_start_time,
                                                                                                                                                                                                                                                }

                                                                                                                                                                                                                                                    except Exception as e:
                                                                                                                                                                                                                                                    logger.error(f"Error getting performance summary: {e}")
                                                                                                                                                                                                                                                return {}


                                                                                                                                                                                                                                                # Global integrator instance
                                                                                                                                                                                                                                                _integrator = None


                                                                                                                                                                                                                                                def get_entropy_integrator(
                                                                                                                                                                                                                                                config_path: str = "config/entropy_signal_integration.yaml",
                                                                                                                                                                                                                                                    ) -> EntropySignalIntegrator:
                                                                                                                                                                                                                                                    """Get global entropy signal integrator instance."""
                                                                                                                                                                                                                                                    global _integrator
                                                                                                                                                                                                                                                        if _integrator is None:
                                                                                                                                                                                                                                                        _integrator = EntropySignalIntegrator(config_path)
                                                                                                                                                                                                                                                    return _integrator


                                                                                                                                                                                                                                                        def process_entropy_signal(bids: List[Tuple[float, float]], asks: List[Tuple[float, float]]) -> EntropySignal:
                                                                                                                                                                                                                                                        """Convenience function to process entropy signal."""
                                                                                                                                                                                                                                                        integrator = get_entropy_integrator()
                                                                                                                                                                                                                                                    return integrator.process_entropy_signal(bids, asks)


                                                                                                                                                                                                                                                        def should_execute_tick() -> bool:
                                                                                                                                                                                                                                                        """Check if tick cycle should execute."""
                                                                                                                                                                                                                                                        integrator = get_entropy_integrator()
                                                                                                                                                                                                                                                    return integrator.should_execute_cycle("tick")


                                                                                                                                                                                                                                                        def should_execute_routing() -> bool:
                                                                                                                                                                                                                                                        """Check if routing cycle should execute."""
                                                                                                                                                                                                                                                        integrator = get_entropy_integrator()
                                                                                                                                                                                                                                                    return integrator.should_execute_cycle("routing")
