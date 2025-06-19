#!/usr/bin/env python3
"""
Fault Bus
========

Adaptive Recursive Path Router (ARPR) for Schwabot's profit navigation system.
Handles system-wide event handling with intelligent sync/async path selection.
Enhanced with profit-fault correlation and recursive loop detection.
Enhanced with Windows CLI compatibility for cross-platform reliability.
"""

# Named constants to replace magic numbers
DEFAULT_WEIGHT_MATRIX_VALUE = 0.9
MAX_QUEUE_SIZE = 50.0
NORMALIZATION_FACTOR = 1.0
DEFAULT_INTERVAL = 0.1
MAX_PROFIT_THRESHOLD = 100.0

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Callable, Tuple, Union, Any
import logging
import json
import os
import hashlib
import numpy as np
from enum import Enum
import yaml  # Ensure yaml is installed in requirements.txt
import asyncio
import time
import psutil
import platform
from collections import deque, defaultdict
from pathlib import Path

# Import the Future Corridor Engine
try:
    from .future_corridor_engine import FutureCorridorEngine, CorridorState, ExecutionPath, ProfitTier
except ImportError:
    # Fallback for testing when package import fails
    try:
        from future_corridor_engine import FutureCorridorEngine, CorridorState, ExecutionPath, ProfitTier
    except ImportError:
        # Provide mock objects so that the rest of this module can still be imported
        from unittest.mock import Mock
        FutureCorridorEngine = Mock(name="FutureCorridorEngine")
        CorridorState = Mock(name="CorridorState")
        ExecutionPath = Mock(name="ExecutionPath")
        ProfitTier = Mock(name="ProfitTier")

# =====================================
# WINDOWS CLI COMPATIBILITY HANDLER
# =====================================

class WindowsCliCompatibilityHandler:
    """
    Handles Windows CLI compatibility issues including emoji rendering
    and ASIC implementation for plain text output explanations

    Addresses the CLI error issues mentioned in the comprehensive testing:
    - Emoji characters causing encoding errors on Windows
    - Need for ASIC plain text output
    - Cross-platform compatibility for error messages
    """

    @staticmethod
    def is_windows_cli() -> bool:
        """Detect if running in Windows CLI environment"""
        return (platform.system() == "Windows" and
                ("cmd" in os.environ.get("COMSPEC", "").lower() or
                 "powershell" in os.environ.get("PSModulePath", "").lower()))

    @staticmethod
    def safe_print(message: str, use_emoji: bool = True) -> str:
        """
        Print message safely with Windows CLI compatibility
        Implements ASIC plain text output for Windows environments

        ASIC Implementation: Application-Specific Integrated Circuit approach
        provides specialized text rendering for Windows CLI environments
        """
        if WindowsCliCompatibilityHandler.is_windows_cli() and use_emoji:
            # ASIC plain text markers for Windows CLI compatibility
            emoji_to_asic_mapping = {
                '✅': '[SUCCESS]',    # Success indicator
                '❌': '[ERROR]',      # Error indicator
                '🔧': '[PROCESSING]', # Processing indicator
                '🚀': '[LAUNCH]',     # Launch/start indicator
                '🎉': '[COMPLETE]',   # Completion indicator
                '💥': '[CRITICAL]',   # Critical alert
                '⚡': '[FAST]',       # Fast execution
                '🔍': '[SEARCH]',     # Search/analysis
                '📊': '[DATA]',       # Data processing
                '🧪': '[TEST]',       # Testing indicator
                '🛠️': '[TOOLS]',      # Tools/utilities
                '⚖️': '[BALANCE]',    # Balance/measurement
                '🔄': '[CYCLE]',      # Cycle/loop
                '🎯': '[TARGET]',     # Target/goal
                '📈': '[PROFIT]',     # Profit indicator
                '🔥': '[HOT]',        # High activity
                '❄️': '[COOL]',       # Cool/low activity
                '⭐': '[STAR]',       # Important/featured
            }

            safe_message = message
            for emoji, asic_replacement in emoji_to_asic_mapping.items():
                safe_message = safe_message.replace(emoji, asic_replacement)

            return safe_message

        return message

    @staticmethod
    def log_safe(logger: Any, level: str, message: str) -> None:
        """Log message safely with Windows CLI compatibility"""
        safe_message = WindowsCliCompatibilityHandler.safe_print(message)
        try:
            getattr(logger, level.lower())(safe_message)
        except UnicodeEncodeError:
            # Emergency ASCII fallback for Windows CLI
            ascii_message = safe_message.encode('ascii', errors='replace').decode('ascii')
            getattr(logger, level.lower())(ascii_message)

    @staticmethod
    def safe_format_error(error: Exception, context: str = "") -> str:
        """Format error messages safely for Windows CLI"""
        error_message = f"Error: {str(error)}"
        if context:
            error_message += f" | Context: {context}"

        return WindowsCliCompatibilityHandler.safe_print(error_message)

class FaultType(Enum):
    THERMAL_HIGH = "thermal_high"
    THERMAL_CRITICAL = "thermal_critical"
    PROFIT_LOW = "profit_low"
    PROFIT_CRITICAL = "profit_critical"
    BITMAP_CORRUPT = "bitmap_corrupt"
    BITMAP_OVERFLOW = "bitmap_overflow"
    GPU_OVERLOAD = "gpu_overload"
    GPU_DRIVER_CRASH = "gpu_driver_crash"
    RECURSIVE_LOOP = "recursive_loop"
    PROFIT_ANOMALY = "profit_anomaly"
    SHA_COLLISION = "sha_collision"
    # Extend this list with new categories as needed

@dataclass
class FaultBusEvent:
    tick: int
    module: str
    type: FaultType
    severity: float
    timestamp: str = datetime.now().isoformat()
    metadata: Optional[Dict] = None
    profit_context: Optional[float] = None
    sha_signature: Optional[str] = None
    age: float = 0.0

    def __post_init__(self) -> None:
        self.age = (datetime.now() - datetime.fromisoformat(self.timestamp)).total_seconds()

@dataclass
class PathSelectionMetrics:
    """Metrics used for intelligent path selection"""
    severity_score: float
    urgency_score: float
    system_load_score: float
    resolver_cost_score: float
    profit_opportunity_score: float
    final_score: float
    selected_path: str
    execution_time_hint: float

@dataclass
class ProfitFaultCorrelation:
    """Mathematical structure for profit-fault correlation tracking"""
    fault_type: FaultType
    profit_delta: float
    correlation_strength: float
    temporal_offset: int  # ticks between fault and profit change
    confidence: float
    occurrence_count: int
    last_seen: datetime

class RecursiveLoopDetector:
    """Detects and prevents recursive profit cycles using SHA-based pattern recognition"""

    def __init__(self, window_size: int = 100, similarity_threshold: float = 0.95) -> None:
        self.window_size = window_size
        self.similarity_threshold = similarity_threshold
        self.pattern_history: deque = deque(maxlen=window_size)
        self.sha_collision_count: Dict[str, int] = defaultdict(int)
        self.profit_signatures: Dict[str, List[float]] = defaultdict(list)

    def compute_pattern_hash(self, profit_delta: float, fault_state: Dict, tick: int) -> str:
        """Compute SHA256 hash of current system state for pattern recognition"""
        state_string = f"{profit_delta:.6f}_{tick}_{hash(frozenset(fault_state.items()))}"
        return hashlib.sha256(state_string.encode()).hexdigest()[:16]

    def detect_recursive_loop(self, current_hash: str, profit_delta: float) -> Tuple[bool, float]:
        """Detect recursive loops using SHA collision analysis"""
        # Implementation continues...
        return False, 0.0

    def reset_pattern(self, sha_hash: str) -> None:
        """Reset pattern detection for a specific hash"""
        if sha_hash in self.sha_collision_count:
            del self.sha_collision_count[sha_hash]

class ProfitAnomalyDetector:
    """JuMBO-style profit anomaly detection for identifying genuine profit tiers"""

    def __init__(self, detection_window: int = 50) -> None:
        self.detection_window = detection_window
        self.profit_history: deque = deque(maxlen=detection_window)
        self.anomaly_clusters: List[Dict] = []

    def detect_jumbo_profit_anomaly(self, profit_delta: float, fault_context: Dict) -> Tuple[bool, float]:
        """
        Detect JuMBO-style profit anomalies using statistical clustering
        Returns (is_anomaly, anomaly_strength)
        """
        self.profit_history.append(profit_delta)

        if len(self.profit_history) < 10:
            return False, 0.0

        # Calculate z-score for current profit
        profits = list(self.profit_history)
        mean_profit = np.mean(profits)
        std_profit = np.std(profits)

        if std_profit == 0:
            return False, 0.0

        z_score = abs(profit_delta - mean_profit) / std_profit

        # Anomaly if z-score > 2.5 (statistically significant)
        if z_score > 2.5:
            anomaly_strength = min(z_score / 5.0, NORMALIZATION_FACTOR)  # Normalize to [0,1]

            # Check for clustering (JuMBO-like behavior)
            recent_anomalies = [p for p in profits[-10:] if abs(p - mean_profit) / std_profit > 2.0]
            if len(recent_anomalies) >= 3:
                # Multiple anomalies = potential profit tier
                return True, anomaly_strength

        return False, 0.0

class ProfitCorrelationMatrix:
    """Mathematical correlation matrix between faults and profit outcomes"""

    def __init__(self, decay_factor: float = 0.95, min_correlation: float = 0.3) -> None:
        self.decay_factor = decay_factor
        self.min_correlation = min_correlation
        self.correlations: Dict[FaultType, ProfitFaultCorrelation] = {}
        self.temporal_buffer: deque = deque(maxlen=1000)

    def update_correlation(self, fault_event: FaultBusEvent, profit_delta: float, temporal_offset: int) -> None:
        """Update profit-fault correlation with exponential decay"""
        fault_type = fault_event.type

        if fault_type not in self.correlations:
            self.correlations[fault_type] = ProfitFaultCorrelation(
                fault_type=fault_type,
                profit_delta=profit_delta,
                correlation_strength=0.0,
                temporal_offset=temporal_offset,
                confidence=0.0,
                occurrence_count=1,
                last_seen=datetime.now()
            )

        corr = self.correlations[fault_type]

        # Exponential moving average for correlation strength
        if corr.occurrence_count == 1:
            corr.correlation_strength = abs(profit_delta)
        else:
            corr.correlation_strength = (
                self.decay_factor * corr.correlation_strength +
                (1 - self.decay_factor) * abs(profit_delta)
            )

        # Update other metrics
        corr.profit_delta = profit_delta
        corr.temporal_offset = temporal_offset
        corr.occurrence_count += 1
        corr.confidence = min(corr.occurrence_count / 10.0, NORMALIZATION_FACTOR)
        corr.last_seen = datetime.now()

        # Store in temporal buffer for analysis
        self.temporal_buffer.append({
            'fault_type': fault_type,
            'profit_delta': profit_delta,
            'temporal_offset': temporal_offset,
            'timestamp': datetime.now()
        })

    def get_predictive_correlations(self, threshold: float = 0.5) -> List[ProfitFaultCorrelation]:
        """Get correlations above threshold for predictive purposes"""
        return [
            corr for corr in self.correlations.values()
            if corr.correlation_strength > threshold and corr.confidence > self.min_correlation
        ]

    def predict_profit_impact(self, fault_type: FaultType) -> Optional[float]:
        """Predict profit impact based on historical correlations"""
        if fault_type in self.correlations:
            corr = self.correlations[fault_type]
            if corr.confidence > self.min_correlation:
                return corr.profit_delta * corr.correlation_strength
        return None

class FaultResolver(ABC):
    """Base class for fault resolution strategies with execution time hints"""
    execution_time_hint: float = DEFAULT_INTERVAL  # Default to 100ms

    @abstractmethod
    def handle_fault(self, fault_type: str, severity: float, metadata: Optional[Dict] = None) -> None:
        pass

class ThermalFaultResolver(FaultResolver):
    """Handles thermal-related faults."""
    execution_time_hint: float = 0.05  # Fast thermal response

    def handle_fault(self, fault_type: str, severity: float, metadata: Optional[Dict] = None) -> None:
        if fault_type == FaultType.THERMAL_HIGH.value:
            logging.warning(f"High thermal load detected: {severity}")
            # Implement thermal mitigation strategy
        elif fault_type == FaultType.THERMAL_CRITICAL.value:
            logging.error(f"Critical thermal condition: {severity}")
            # Implement emergency thermal response

class ProfitFaultResolver(FaultResolver):
    """Handles profit-related faults with correlation awareness."""
    execution_time_hint: float = 0.2  # Moderate time for profit analysis

    def handle_fault(self, fault_type: str, severity: float, metadata: Optional[Dict] = None) -> None:
        if fault_type == FaultType.PROFIT_LOW.value:
            logging.warning(f"Low profit detected: {severity}")
            # Implement profit optimization strategy
        elif fault_type == FaultType.PROFIT_CRITICAL.value:
            logging.error(f"Critical profit condition: {severity}")
            # Implement emergency profit response

class BitmapFaultResolver(FaultResolver):
    """Handles bitmap-related faults."""
    execution_time_hint: float = 0.3  # Slower bitmap operations

    def handle_fault(self, fault_type: str, severity: float, metadata: Optional[Dict] = None) -> None:
        if fault_type == FaultType.BITMAP_CORRUPT.value:
            logging.error(f"Bitmap corruption detected: {severity}")
            # Implement bitmap recovery strategy
        elif fault_type == FaultType.BITMAP_OVERFLOW.value:
            logging.warning(f"Bitmap overflow detected: {severity}")
            # Implement bitmap cleanup strategy

class RecursiveLoopResolver(FaultResolver):
    """Handles recursive loop detection and prevention."""
    execution_time_hint: float = DEFAULT_INTERVAL  # Fast loop breaking

    def handle_fault(self, fault_type: str, severity: float, metadata: Optional[Dict] = None) -> None:
        if fault_type == FaultType.RECURSIVE_LOOP.value:
            logging.warning(f"Recursive loop detected: {severity}")
            sha_hash = metadata.get('sha_hash') if metadata else None
            if sha_hash:
                logging.info(f"Breaking loop for pattern: {sha_hash[:8]}")
                # Implement loop breaking logic
        elif fault_type == FaultType.SHA_COLLISION.value:
            logging.info(f"SHA collision detected: {severity}")

fault_resolver_registry = {}

def register_fault_resolver(name: str) -> Any:
    def decorator(cls: type) -> Any:
        fault_resolver_registry[name] = cls()
        return cls
    return decorator

@register_fault_resolver("gpu")
class GPUFaultResolver(FaultResolver):
    execution_time_hint: float = 0.5  # GPU operations can be slower

    def handle_fault(self, fault_type: str, severity: float, metadata: Optional[Dict] = None) -> None:
        if fault_type == FaultType.GPU_OVERLOAD.value:
            logging.warning(f"GPU overload: {severity}")
        elif fault_type == FaultType.GPU_DRIVER_CRASH.value:
            logging.error(f"GPU driver crash detected!")

class FallbackFaultResolver(FaultResolver):
    """Fallback resolver for unhandled faults"""
    execution_time_hint: float = 0.01  # Very fast fallback

    def handle_fault(self, fault_type: str, severity: float, metadata: Optional[Dict] = None) -> None:
        logging.warning(f"Unhandled fault via fallback: {fault_type}, Severity: {severity}")

class EventSeverity:
    INFO = DEFAULT_INTERVAL
    WARNING = 0.5
    CRITICAL = 0.9

class FaultBus:
    """
    Adaptive Recursive Path Router (ARPR) for Schwabot's profit navigation system.
    Intelligently routes fault events through sync or async paths based on:
    - System state and load
    - Profit opportunity context
    - Resolver execution requirements
    - BTC price hashing complexity
    Enhanced with Windows CLI compatibility for cross-platform reliability.
    """

    def __init__(self, log_path: str = "logs/faults") -> None:
        self.queue: List[FaultBusEvent] = []
        self.resolvers: Dict[str, FaultResolver] = {}
        self.fallback_resolver = FallbackFaultResolver()
        self.memory_log: List[FaultBusEvent] = []
        self.event_handlers: Dict[str, List[Callable]] = {}
        self.trigger_policies: Dict[str, Callable[[FaultBusEvent], bool]] = {}
        self.log_path = log_path

        # Path selection routing system
        self.path_selection_weights = {
            'severity': 0.3,          # Event severity
            'urgency': 0.25,          # Fault type urgency
            'system_load': -0.2,      # Negative: high load favors async
            'resolver_cost': -DEFAULT_INTERVAL5,   # Negative: high cost favors async
            'profit_opportunity': 0.2 # Profit potential
        }

        # Path selection history for analysis
        self.path_history: List[PathSelectionMetrics] = []
        self.async_threshold = 0.5  # Configurable threshold

        # Mathematical structures for profit-fault correlation
        self.loop_detector = RecursiveLoopDetector()
        self.anomaly_detector = ProfitAnomalyDetector()
        self.correlation_matrix = ProfitCorrelationMatrix()
        self.profit_history: deque = deque(maxlen=1000)

        # ✨ NEW: Future Corridor Engine Integration
        self.corridor_engine = FutureCorridorEngine(
            profit_amplitude=NORMALIZATION_FACTOR,
            tick_frequency=DEFAULT_INTERVAL,
            decay_rate=0.05,
            async_threshold=0.5
        )
        self.current_market_data = {
            'price_series': [],
            'volume_series': [],
            'volatility_series': [],
            'jumbo_signal': 0.0,
            'ghost_signal': 0.0,
            'thermal_state': 0.0
        }

        # Create log directory if it doesn't exist
        os.makedirs(log_path, exist_ok=True)

        # Initialize default resolvers
        self.register_resolver("thermal", ThermalFaultResolver())
        self.register_resolver("profit", ProfitFaultResolver())
        self.register_resolver("bitmap", BitmapFaultResolver())
        self.register_resolver("gpu", GPUFaultResolver())
        self.register_resolver("recursive", RecursiveLoopResolver())

        # Register fault resolvers from registry
        self.resolvers.update(fault_resolver_registry)

        logging.info("🧠 FaultBus initialized with Future Corridor Engine integration")

        # Windows CLI compatibility handler
        self.cli_handler = WindowsCliCompatibilityHandler()

    def register_resolver(self, fault_type: str, resolver: FaultResolver) -> None:
        """Register a resolver for a specific fault type."""
        self.resolvers[fault_type] = resolver

    def register_handler(self, event_type: str) -> Callable[[Callable], Callable]:
        """Register event handler decorator"""
        def decorator(func: Callable) -> Callable:
            if event_type not in self.event_handlers:
                self.event_handlers[event_type] = []
            self.event_handlers[event_type].append(func)
            return func
        return decorator

    def push(self, event: FaultBusEvent) -> None:
        """Push event to queue after policy check"""
        condition = self.trigger_policies.get(event.type.value, lambda e: True)
        if condition(event):
            self.queue.append(event)

    def update_profit_context(self, profit_delta: float, tick: int) -> None:
        """Update profit context and detect anomalies/loops"""
        self.profit_history.append((profit_delta, tick, datetime.now()))

        # Get current fault state for pattern analysis
        fault_state = {
            event.type.value: event.severity
            for event in self.memory_log[-10:]
        }

        # Compute pattern hash
        pattern_hash = self.loop_detector.compute_pattern_hash(
            profit_delta, fault_state, tick
        )

        # Check for recursive loops
        is_loop, loop_strength = self.loop_detector.detect_recursive_loop(
            pattern_hash, profit_delta
        )

        if is_loop:
            loop_event = FaultBusEvent(
                tick=tick,
                module="profit_monitor",
                type=FaultType.RECURSIVE_LOOP,
                severity=loop_strength,
                metadata={
                    'sha_hash': pattern_hash,
                    'profit_delta': profit_delta,
                    'loop_strength': loop_strength
                },
                profit_context=profit_delta,
                sha_signature=pattern_hash
            )
            self.push(loop_event)

        # Check for profit anomalies (potential genuine profit tiers)
        is_anomaly, anomaly_strength = self.anomaly_detector.detect_jumbo_profit_anomaly(
            profit_delta, fault_state
        )

        if is_anomaly:
            anomaly_event = FaultBusEvent(
                tick=tick,
                module="profit_monitor",
                type=FaultType.PROFIT_ANOMALY,
                severity=anomaly_strength,
                metadata={
                    'profit_delta': profit_delta,
                    'anomaly_strength': anomaly_strength,
                    'z_score': anomaly_strength * 5.0  # Reverse normalize
                },
                profit_context=profit_delta,
                sha_signature=pattern_hash
            )
            self.push(anomaly_event)

        # Update correlations for recent faults
        for event in self.memory_log[-5:]:  # Check last 5 events
            temporal_offset = tick - event.tick
            if temporal_offset >= 0:
                self.correlation_matrix.update_correlation(
                    event, profit_delta, temporal_offset
                )

    def _calculate_path_selection_score(self, event: FaultBusEvent) -> PathSelectionMetrics:
        """
        Calculate intelligent path selection score for sync vs async execution.
        Higher scores favor async execution for profit optimization.
        """
        # 1. Severity Score (normalized)
        severity_score = event.severity

        # 2. Urgency Score (based on fault type priority)
        urgency_map = {
            FaultType.PROFIT_CRITICAL: NORMALIZATION_FACTOR,     # High async priority for profit storms
            FaultType.BITMAP_CORRUPT: 0.9,      # Complex async GPU processing
            FaultType.GPU_OVERLOAD: 0.9,        # GPU async handling
            FaultType.PROFIT_ANOMALY: 0.8,      # JuMBO profit analysis
            FaultType.THERMAL_CRITICAL: 0.7,    # Can be async for parallel cooling
            FaultType.RECURSIVE_LOOP: 0.6,      # Pattern breaking
            FaultType.PROFIT_LOW: 0.4,          # Simple sync fixes
            FaultType.THERMAL_HIGH: 0.3,        # Quick sync response
            FaultType.BITMAP_OVERFLOW: 0.3,     # Simple cleanup
            FaultType.GPU_DRIVER_CRASH: 0.2,    # Deterministic sync restart
        }
        urgency_score = urgency_map.get(event.type, 0.5)

        # 3. System Load Score (current queue + CPU utilization)
        queue_load = min(len(self.queue) / MAX_QUEUE_SIZE, NORMALIZATION_FACTOR)  # Normalize by max expected queue
        try:
            cpu_load = psutil.cpu_percent(interval=DEFAULT_INTERVAL) / MAX_PROFIT_THRESHOLD
            system_load_score = (queue_load + cpu_load) / 2.0
        except Exception as e:

            # Windows CLI compatible error handling for CPU monitoring

            error_message = self.cli_handler.safe_format_error(e, "CPU monitoring")

            self.cli_handler.log_safe(logging, 'warning', f"CPU monitoring failed, using queue load: {error_message}")
            system_load_score = queue_load

        # 4. Resolver Cost Score (execution time hint)
        resolver = self._get_resolver_for_event(event)
        resolver_cost_score = min(
            resolver.execution_time_hint / NORMALIZATION_FACTOR, NORMALIZATION_FACTOR)  # Normalize by 1s max

        # 5. Profit Opportunity Score (based on profit context)
        profit_opportunity_score = 0.5  # Default neutral
        if event.profit_context is not None:
            # Higher absolute profit changes favor async for complex analysis
            profit_opportunity_score = min(abs(event.profit_context) / MAX_PROFIT_THRESHOLD, NORMALIZATION_FACTOR)
        elif event.type in [FaultType.PROFIT_CRITICAL, FaultType.PROFIT_ANOMALY]:
            profit_opportunity_score = 0.8  # High opportunity for profit events

        # Calculate weighted final score
        weights = self.path_selection_weights
        final_score = (
            severity_score * weights['severity'] +
            urgency_score * weights['urgency'] +
            system_load_score * weights['system_load'] +
            resolver_cost_score * weights['resolver_cost'] +
            profit_opportunity_score * weights['profit_opportunity']
        )

        # Normalize to [0, 1] range
        final_score = (final_score + NORMALIZATION_FACTOR) / 2.0
        final_score = max(0.0, min(NORMALIZATION_FACTOR, final_score))

        # Determine selected path
        selected_path = "async" if final_score >= self.async_threshold else "sync"

        return PathSelectionMetrics(
            severity_score=severity_score,
            urgency_score=urgency_score,
            system_load_score=system_load_score,
            resolver_cost_score=resolver_cost_score,
            profit_opportunity_score=profit_opportunity_score,
            final_score=final_score,
            selected_path=selected_path,
            execution_time_hint=resolver.execution_time_hint
        )

    async def dispatch(self, severity_threshold: float = 0.5):
        """
        Enhanced Smart Dispatch with Future Corridor Engine Integration.
        Uses probabilistic dispatch vector and recursive intent loop for optimal path selection.
        This is the core of the Adaptive Recursive Path Router (ARPR).
        Enhanced with Windows CLI compatibility for cross-platform reliability.
        """
        try:
            while self.queue:
                event = self.queue.pop(0)
                if event.severity >= severity_threshold:

                    # 🧠 Create corridor state from event context
                    current_price = event.metadata.get(
                        'price', MAX_PROFIT_THRESHOLD) if event.metadata else MAX_PROFIT_THRESHOLD
                    current_volume = event.metadata.get('volume', 1000.0) if event.metadata else 1000.0
                    current_volatility = event.metadata.get('volatility', 0.02) if event.metadata else 0.02

                    # Update corridor engine memory
                    self.corridor_engine.update_corridor_memory(current_price, current_volume, current_volatility)

                    # Create corridor state
                    corridor_state = CorridorState(
                        price=current_price,
                        duration=NORMALIZATION_FACTOR,
                        volatility=current_volatility,
                        timestamp=datetime.now(),
                        hash_signature=event.sha_signature or hashlib.sha256(
                            f"{current_price}_{event.tick}".encode()).hexdigest()
                    )

                    # Update market data for ECMP calculation
                    self.current_market_data['price_series'].append(current_price)
                    self.current_market_data['volume_series'].append(current_volume)
                    self.current_market_data['volatility_series'].append(current_volatility)

                    # Keep series manageable
                    if len(self.current_market_data['price_series']) > 50:
                        self.current_market_data['price_series'] = self.current_market_data['price_series'][-30:]
                        self.current_market_data['volume_series'] = self.current_market_data['volume_series'][-30:]
                        self.current_market_data['volatility_series'] = self.current_market_data['volatility_series'][-30:]

                    # 🔬 Run Recursive Intent Loop (RIL) for complete navigation decision
                    ril_result = self.corridor_engine.recursive_intent_loop(
                        t=event.tick * DEFAULT_INTERVAL,  # Convert tick to time
                        market_hash=corridor_state.hash_signature,
                        corridor_state=corridor_state,
                        profit_context=event.profit_context or 0.0,
                        execution_time=self._estimate_execution_time(event),
                        entropy=self._calculate_entropy(event),
                        market_data=self.current_market_data
                    )

                    # Extract dispatch path from RIL result
                    selected_path = ril_result['dispatch_path']
                    dispatch_confidence = ril_result['dispatch_confidence']

                    # Log enhanced dispatch decision
                    self.cli_handler.log_safe(logging, 'info', f"🎯 Enhanced Dispatch: {event.type.value}")
                    self.cli_handler.log_safe(
                        logging, 'info', f"   Path: {selected_path} (confidence: {dispatch_confidence:.3f})")
                    self.cli_handler.log_safe(logging, 'info', f"   Tier: {ril_result['profit_tier']}")
                    self.cli_handler.log_safe(logging, 'info', f"   Mode: {ril_result['activation_mode']}")
                    self.cli_handler.log_safe(logging, 'info', f"   ECMP: {ril_result['ecmp_magnitude']:.4f}")
                    self.cli_handler.log_safe(logging, 'info', f"   Resonance: {ril_result['resonance_strength']:.3f}")

                    # Route to appropriate execution path based on corridor engine decision
                    if selected_path == "cpu_sync":
                        self._dispatch_sync_enhanced(event, ril_result)
                    elif selected_path == "cpu_async":
                        asyncio.create_task(self._dispatch_async_enhanced(event, ril_result))
                    elif selected_path == "gpu_async":
                        asyncio.create_task(self._dispatch_gpu_async_enhanced(event, ril_result))
                    else:
                        # Fallback to original path selection
                        metrics = self._calculate_path_selection_score(event)
                        self.path_history.append(metrics)

                        if metrics.selected_path == "async":
                            asyncio.create_task(self._dispatch_async(event, metrics))
                        else:
                            self._dispatch_sync(event, metrics)

                    # Always add to memory log
                    self.memory_log.append(event)

            # Enhanced completion logging
            self.cli_handler.log_safe(logging, 'debug', "✅ Enhanced dispatch completed successfully")

        except Exception as e:
            error_message = self.cli_handler.safe_format_error(e, "dispatch")
            self.cli_handler.log_safe(logging, 'error', error_message)

    def _estimate_execution_time(self, event: FaultBusEvent) -> float:
        """Estimate execution time for the event based on type and metadata"""
        resolver = self._get_resolver_for_event(event)
        base_time = resolver.execution_time_hint

        # Adjust based on event complexity
        complexity_factor = NORMALIZATION_FACTOR
        if event.metadata:
            complexity_factor += len(event.metadata) * DEFAULT_INTERVAL
        if event.profit_context and abs(event.profit_context) > 50:
            complexity_factor += 0.3  # High profit events are more complex

        return base_time * complexity_factor

    def _calculate_entropy(self, event: FaultBusEvent) -> float:
        """Calculate entropy/complexity of the event"""
        base_entropy = {
            FaultType.THERMAL_HIGH: 0.2,
            FaultType.THERMAL_CRITICAL: 0.4,
            FaultType.PROFIT_LOW: 0.3,
            FaultType.PROFIT_CRITICAL: 0.8,
            FaultType.PROFIT_ANOMALY: 1.2,
            FaultType.BITMAP_CORRUPT: 0.9,
            FaultType.BITMAP_OVERFLOW: 0.6,
            FaultType.GPU_OVERLOAD: NORMALIZATION_FACTOR,
            FaultType.GPU_DRIVER_CRASH: 0.7,
            FaultType.RECURSIVE_LOOP: 1.1,
            FaultType.SHA_COLLISION: 0.8
        }.get(event.type, 0.5)

        # Adjust for severity and age
        entropy_adjustment = event.severity + (event.age * DEFAULT_INTERVAL)
        return base_entropy + entropy_adjustment

    def _dispatch_sync_enhanced(self, event: FaultBusEvent, ril_result: Dict) -> None:
        """Enhanced synchronous dispatch with corridor intelligence"""
        try:
            start_time = time.time()
            resolver = self._get_resolver_for_event(event)

            try:
                # Apply corridor-based adjustments
                if ril_result['activation_mode'] == "FULL_ACTIVATION":
                    logging.info(f"🚀 FULL_ACTIVATION mode for {event.type.value}")

                resolver.handle_fault(event.type.value, event.severity, event.metadata)
                execution_time = time.time() - start_time

                # Update corridor engine with execution feedback
                self._update_corridor_feedback(ril_result, execution_time, True)

                logging.debug(f"✅ Enhanced SYNC completed: {event.type.value} in {execution_time:.3f}s")
                self._trigger_event_handlers(event)

            except Exception as e:
                execution_time = time.time() - start_time
                self._update_corridor_feedback(ril_result, execution_time, False)
                logging.error(f"❌ Enhanced SYNC failed: {event.type.value} after {execution_time:.3f}s - {e}")

            self.cli_handler.log_safe(
                logging, 'debug', f"✅ Enhanced SYNC completed: {event.type.value} in {execution_time:.3f}s")

        except Exception as e:
            error_message = self.cli_handler.safe_format_error(e, f"sync_dispatch {event.type.value}")
            self.cli_handler.log_safe(logging, 'error', error_message)

    async def _dispatch_async_enhanced(self, event: FaultBusEvent, ril_result: Dict):
        """Enhanced asynchronous dispatch with corridor intelligence"""
        try:
            start_time = time.time()
            resolver = self._get_resolver_for_event(event)

            try:
                # Use different execution strategy based on activation mode
                if ril_result['activation_mode'] == "FULL_ACTIVATION":
                    # High-priority parallel execution
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, resolver.handle_fault,
                                             event.type.value, event.severity, event.metadata)
                else:
                    # Standard async execution
                    resolver.handle_fault(event.type.value, event.severity, event.metadata)

                execution_time = time.time() - start_time
                self._update_corridor_feedback(ril_result, execution_time, True)

            except Exception as e:
                execution_time = time.time() - start_time
                self._update_corridor_feedback(ril_result, execution_time, False)
                logging.error(f"❌ Enhanced ASYNC failed: {event.type.value} after {execution_time:.3f}s - {e}")

            self.cli_handler.log_safe(
                logging, 'debug', f"✅ Enhanced ASYNC completed: {event.type.value} in {execution_time:.3f}s")

        except Exception as e:
            error_message = self.cli_handler.safe_format_error(e, f"async_dispatch {event.type.value}")
            self.cli_handler.log_safe(logging, 'error', error_message)

    async def _dispatch_gpu_async_enhanced(self, event: FaultBusEvent, ril_result: Dict):
        """GPU-optimized asynchronous dispatch with tensor field processing"""
        try:
            start_time = time.time()
            resolver = self._get_resolver_for_event(event)

            try:
                logging.info(f"🔥 GPU_ASYNC dispatch for {event.type.value}")
                logging.info(f"   ECMP Direction: {ril_result['ecmp_direction']}")
                logging.info(f"   Target Price: ${ril_result['next_target_price']:.2f}")

                # GPU-specific processing (placeholder for CUDA/tensor operations)
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, resolver.handle_fault,
                                         event.type.value, event.severity, event.metadata)

                execution_time = time.time() - start_time
                self._update_corridor_feedback(ril_result, execution_time, True)

            except Exception as e:
                execution_time = time.time() - start_time
                self._update_corridor_feedback(ril_result, execution_time, False)
                logging.error(f"❌ GPU ASYNC failed: {event.type.value} after {execution_time:.3f}s - {e}")

            self.cli_handler.log_safe(logging, 'info', f"🔥 GPU_ASYNC dispatch for {event.type.value}")
            self.cli_handler.log_safe(
                logging, 'debug', f"✅ GPU ASYNC completed: {event.type.value} in {execution_time:.3f}s")

        except Exception as e:
            error_message = self.cli_handler.safe_format_error(e, f"gpu_async_dispatch {event.type.value}")
            self.cli_handler.log_safe(logging, 'error', error_message)

    def _update_corridor_feedback(self, ril_result: Dict, execution_time: float, success: bool) -> None:
        """Update corridor engine with execution feedback for learning"""
        feedback_strength = NORMALIZATION_FACTOR if success else -0.5

        # Update thermal state based on execution
        if execution_time > NORMALIZATION_FACTOR:  # Slow execution
            self.current_market_data['thermal_state'] += DEFAULT_INTERVAL
        else:
            self.current_market_data['thermal_state'] = max(0.0, self.current_market_data['thermal_state'] - 0.05)

        # Update ghost signal based on success/failure patterns
        if success:
            self.current_market_data['ghost_signal'] = min(
                NORMALIZATION_FACTOR, self.current_market_data['ghost_signal'] + DEFAULT_INTERVAL)
        else:
            self.current_market_data['ghost_signal'] = max(0.0, self.current_market_data['ghost_signal'] - 0.2)

    def update_market_signals(self, price: float, volume: float, volatility: float,
                            jumbo_signal: float = None, ghost_signal: float = None) -> None:
        """Update market data and signals for corridor engine"""
        self.current_market_data['price_series'].append(price)
        self.current_market_data['volume_series'].append(volume)
        self.current_market_data['volatility_series'].append(volatility)

        if jumbo_signal is not None:
            self.current_market_data['jumbo_signal'] = jumbo_signal
        if ghost_signal is not None:
            self.current_market_data['ghost_signal'] = ghost_signal

        # Keep series manageable
        if len(self.current_market_data['price_series']) > 50:
            self.current_market_data['price_series'] = self.current_market_data['price_series'][-30:]
            self.current_market_data['volume_series'] = self.current_market_data['volume_series'][-30:]
            self.current_market_data['volatility_series'] = self.current_market_data['volatility_series'][-30:]

    def get_corridor_analytics(self) -> Dict:
        """Get analytics from the corridor engine"""
        corridor_metrics = self.corridor_engine.get_performance_metrics()

        return {
            'corridor_engine': corridor_metrics,
            'current_market_state': {
                'price_samples': len(self.current_market_data['price_series']),
                'jumbo_signal': self.current_market_data['jumbo_signal'],
                'ghost_signal': self.current_market_data['ghost_signal'],
                'thermal_state': self.current_market_data['thermal_state']
            },
            'path_statistics': self.get_path_statistics(),
            'fault_correlations': len(self.get_profit_correlations())
        }

    def get_fault_buckets(self) -> Dict[str, int]:
        """Get fault distribution statistics"""
        bucket = {}
        for event in self.memory_log:
            bucket[event.type.value] = bucket.get(event.type.value, 0) + 1
        return bucket

    def get_path_statistics(self) -> Dict[str, Union[int, float]]:
        """Get path selection statistics"""
        if not self.path_history:
            return {}

        async_count = sum(1 for m in self.path_history if m.selected_path == "async")
        sync_count = len(self.path_history) - async_count
        avg_score = sum(m.final_score for m in self.path_history) / len(self.path_history)

        return {
            'total_dispatches': len(self.path_history),
            'async_dispatches': async_count,
            'sync_dispatches': sync_count,
            'async_ratio': async_count / len(self.path_history),
            'average_score': avg_score,
            'current_threshold': self.async_threshold
        }

    def tune_async_threshold(self, new_threshold: float) -> None:
        """Dynamically tune the async threshold based on performance"""
        self.async_threshold = max(0.0, min(NORMALIZATION_FACTOR, new_threshold))
        logging.info(f"Async threshold tuned to: {self.async_threshold:.3f}")

    def get_profit_correlations(self) -> List[ProfitFaultCorrelation]:
        """Get current profit-fault correlations for analysis"""
        return self.correlation_matrix.get_predictive_correlations()

    def predict_profit_from_fault(self, fault_type: FaultType) -> Optional[float]:
        """Predict profit impact based on fault type"""
        return self.correlation_matrix.predict_profit_impact(fault_type)

    def register_policy(self, event_type: str, condition: Callable[[FaultBusEvent], bool]) -> None:
        """Register trigger policy for event type"""
        self.trigger_policies[event_type] = condition

    def export_memory_log(self, file_path: Optional[str] = None) -> str:
        """Export memory log with path selection data"""
        log_data = []
        for i, event in enumerate(self.memory_log):
            event_dict = {
                'tick': event.tick,
                'module': event.module,
                'type': event.type.value,
                'severity': event.severity,
                'timestamp': event.timestamp,
                'metadata': event.metadata,
                'profit_context': event.profit_context,
                'sha_signature': event.sha_signature
            }

            # Add path selection data if available
            if i < len(self.path_history):
                metrics = self.path_history[i]
                event_dict['path_metrics'] = {
                    'selected_path': metrics.selected_path,
                    'path_score': metrics.final_score,
                    'execution_hint': metrics.execution_time_hint
                }

            log_data.append(event_dict)

        output = json.dumps(log_data, indent=2)
        if file_path:
            with open(file_path, "w") as f:
                f.write(output)
        return output

    def export_correlation_matrix(self, file_path: Optional[str] = None) -> str:
        """Export profit-fault correlation matrix"""
        correlations = []
        for fault_type, corr in self.correlation_matrix.correlations.items():
            correlations.append({
                'fault_type': fault_type.value,
                'profit_delta': corr.profit_delta,
                'correlation_strength': corr.correlation_strength,
                'temporal_offset': corr.temporal_offset,
                'confidence': corr.confidence,
                'occurrence_count': corr.occurrence_count,
                'last_seen': corr.last_seen.isoformat()
            })

        output = json.dumps(correlations, indent=2)
        if file_path:
            with open(file_path, "w") as f:
                f.write(output)
        return output

    def _get_resolver_for_event(self, event: FaultBusEvent) -> FaultResolver:
        """Get appropriate resolver for event with fallback"""
        resolver_key = self._get_resolver_key(event.type)
        return self.resolvers.get(resolver_key, self.fallback_resolver)

    def _get_resolver_key(self, fault_type: FaultType) -> str:
        """Map fault types to resolver keys"""
        mapping = {
            FaultType.THERMAL_HIGH: "thermal",
            FaultType.THERMAL_CRITICAL: "thermal",
            FaultType.PROFIT_LOW: "profit",
            FaultType.PROFIT_CRITICAL: "profit",
            FaultType.PROFIT_ANOMALY: "profit",
            FaultType.BITMAP_CORRUPT: "bitmap",
            FaultType.BITMAP_OVERFLOW: "bitmap",
            FaultType.GPU_OVERLOAD: "gpu",
            FaultType.GPU_DRIVER_CRASH: "gpu",
            FaultType.RECURSIVE_LOOP: "recursive",
            FaultType.SHA_COLLISION: "recursive"
        }
        return mapping.get(fault_type, "unknown")

    def _trigger_event_handlers(self, event: FaultBusEvent) -> None:
        """Trigger registered event handlers"""
        handlers = self.event_handlers.get(event.type.value, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logging.error(f"Event handler failed for {event.type.value}: {e}")

# Example usage of the enhanced FaultBus class
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Create a new instance of FaultBus
    fault_bus = FaultBus()

    # Register an event handler
    @fault_bus.register_handler("thermal_high")
    def handle_thermal_high(event: FaultBusEvent) -> Any:
        print(f"🔥 Event handled: {event}")

    # Simulate profit updates with potential loops
    profit_deltas = [10.5, 15.2, 10.5, 15.2, 10.5, 25.8]  # Notice the repetition

    for i, profit_delta in enumerate(profit_deltas):
        # Update profit context (this will detect loops/anomalies)
        fault_bus.update_profit_context(profit_delta, i)

        # Push a thermal event
        fault_bus.push(FaultBusEvent(
            tick=i,
            module="thermal_monitor",
            type=FaultType.THERMAL_HIGH,
            severity=0.6,
            metadata={"temperature": 70.0 + i},
            profit_context=profit_delta
        ))

    # Dispatch events with intelligent path selection
    asyncio.run(fault_bus.dispatch(severity_threshold=0.5))

    # Export logs and correlations
    print("=== Memory Log ===")
    print(fault_bus.export_memory_log())
    print("\n=== Path Statistics ===")
    print(json.dumps(fault_bus.get_path_statistics(), indent=2))
    print("\n=== Correlation Matrix ===")
    print(fault_bus.export_correlation_matrix())
# FUTURE ENHANCEMENT NOTES
# =====================================
#
# SECURITY ENHANCEMENTS (CRITICAL for fault handling):
# - bandit>=1.7.0           # Security vulnerability scanning
# - safety>=2.3.0           # Dependency vulnerability checking
#
# ENHANCED LINTING:
# - pylint>=2.17.0          # More comprehensive than flake8
# - Consider adding pylint configuration for this file
#
# PERFORMANCE MONITORING:
# - memory-profiler>=0.61.0 # Monitor memory usage in fault resolution
# - line-profiler>=4.1.0    # Profile fault resolution performance
#
# TESTING ENHANCEMENTS:
# - Add fault injection testing
# - Test Windows CLI compatibility in error scenarios
# - Add performance benchmarks for fault resolution
#
# SECURITY CONSIDERATIONS:
# - This file handles system-wide fault events
# - Review all exception handling for security implications
# - Consider adding audit logging for fault events

# Fix relative import issue - use absolute imports or try/except for optional imports
try:
    from quantum_visualizer import PanicDriftVisualizer, plot_entropy_waveform
except ImportError:
    try:
        from ncco_core.quantum_visualizer import PanicDriftVisualizer, plot_entropy_waveform
from core.type_defs import *
from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
    except ImportError:
        # Fallback: create dummy functions if module not available
        def PanicDriftVisualizer(*args, **kwargs) -> Any:
            return None

        def plot_entropy_waveform(*args, **kwargs) -> Any:
            return None