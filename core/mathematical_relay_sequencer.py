# -*- coding: utf-8 -*-
"""
Mathematical Relay Sequencer
===========================

Comprehensive sequencing system for mathematical relay integration across all math libraries
and unified math states. Provides exact timing, sequencing, and time log management for
BTC price hashing, bit-depth tensor switching, dual-channel switching, and profit optimization.

Features:
- Precise time log management with microsecond precision
- Cross-library mathematical relay sequencing
- BTC price hash synchronization with timestamp correlation
- Bit-depth tensor switching with phase tracking
- Dual-channel switching with handoff timing
- Profit optimization with basket-tier navigation
- Legacy system compatibility and state continuity
- Real-time sequencing validation and error recovery
"""

import hashlib
import json
import logging
import os
import queue
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Import core modules with safe fallbacks
try:
    from core.basket_relay_engine import BasketRelayEngine
    from core.ferris_rde_engine import FerrisRDEEngine
    from core.ghost_file_system import GhostFileSystem
    from core.mathematical_backlog_manager import MathematicalBacklogManager
    from core.mathematical_relay_integration import MathematicalRelayIntegration
    from core.mathematical_relay_navigator import MathematicalRelayNavigator
    from core.mathematical_utilities import (
        MathematicalConstants,
        calculate_drift_differential,
        calculate_profit_vectorization_score,
    )
    from core.price_hash_event_sequencer import PriceHashEventSequencer
    from core.quicktime_event_manager import QuickTimeEventManager
    from core.trend_state_manager import TrendStateManager
except ImportError as e:
    logging.warning(f"Some core modules not available: {e}")
    MathematicalRelayNavigator = None
    MathematicalRelayIntegration = None
    TrendStateManager = None
    BasketRelayEngine = None
    PriceHashEventSequencer = None
    GhostFileSystem = None
    MathematicalBacklogManager = None
    QuickTimeEventManager = None
    FerrisRDEEngine = None
    MathematicalConstants = None
    calculate_drift_differential = None
    calculate_profit_vectorization_score = None

logger = logging.getLogger(__name__)


class SequenceType(Enum):
    """Types of mathematical relay sequences."""

    BTC_PRICE_HASH = "btc_price_hash"
    BIT_DEPTH_SWITCH = "bit_depth_switch"
    CHANNEL_SWITCH = "channel_switch"
    PROFIT_OPTIMIZATION = "profit_optimization"
    BASKET_NAVIGATION = "basket_navigation"
    GHOST_LOGIC = "ghost_logic"
    LEGACY_HANDOFF = "legacy_handoff"
    STATE_CONTINUITY = "state_continuity"
    MATHEMATICAL_VALIDATION = "mathematical_validation"
    SYSTEM_SYNCHRONIZATION = "system_synchronization"
    FERRIS_RDE_EXECUTION = "ferris_rde_execution"


class SequenceStatus(Enum):
    """Status of mathematical relay sequences."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class TimeLogLevel(Enum):
    """Levels of time logging detail."""

    MICROSECOND = "microsecond"
    MILLISECOND = "millisecond"
    SECOND = "second"
    MINUTE = "minute"


@dataclass
class TimeLogEntry:
    """Individual time log entry with precise timing."""

    entry_id: str
    sequence_id: str
    sequence_type: SequenceType
    timestamp: datetime
    microsecond_precision: int
    operation: str
    duration_microseconds: int
    status: SequenceStatus
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize with precise timestamp."""
        if not hasattr(self, "timestamp") or self.timestamp is None:
            self.timestamp = datetime.now()
        if not hasattr(self, "microsecond_precision"):
            self.microsecond_precision = self.timestamp.microsecond


@dataclass
class MathematicalSequence:
    """Complete mathematical relay sequence."""

    sequence_id: str
    sequence_type: SequenceType
    start_time: datetime
    end_time: Optional[datetime] = None
    status: SequenceStatus = SequenceStatus.PENDING
    time_logs: List[TimeLogEntry] = field(default_factory=list)
    btc_price: Optional[float] = None
    btc_hash: Optional[str] = None
    bit_depth: Optional[int] = None
    channel: Optional[str] = None
    profit_target: Optional[float] = None
    basket_tier: Optional[str] = None
    ghost_logic: Optional[Dict[str, Any]] = None
    legacy_state: Optional[Dict[str, Any]] = None
    mathematical_validation: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    total_duration_microseconds: int = 0
    ferris_rde_result: Optional[Dict[str, Any]] = None

    def add_time_log(self, entry: TimeLogEntry) -> None:
        """Add time log entry to sequence."""
        self.time_logs.append(entry)
        if entry.status == SequenceStatus.COMPLETED:
            self.total_duration_microseconds += entry.duration_microseconds

    def complete(self, end_time: Optional[datetime] = None) -> None:
        """Mark sequence as completed."""
        self.end_time = end_time or datetime.now()
        self.status = SequenceStatus.COMPLETED
        if self.start_time and self.end_time:
            duration = (self.end_time - self.start_time).total_seconds() * 1_000_000
            self.total_duration_microseconds = int(duration)


class MathematicalRelaySequencer:
    """
    Comprehensive mathematical relay sequencer with precise time log management.
    """

    def __init__(
        self,
        mode: str = "demo",
        log_level: str = "INFO",
        time_log_level: TimeLogLevel = TimeLogLevel.MICROSECOND,
        gpu_enabled: bool = False,
    ):
        self.mode = mode
        self.log_level = log_level
        self.time_log_level = time_log_level
        self.start_time = datetime.now()

        # Core sequencing state
        self.active_sequences: Dict[str, MathematicalSequence] = {}
        self.completed_sequences: List[MathematicalSequence] = []
        self.sequence_counter = 0

        # Time log management
        self.time_logs: List[TimeLogEntry] = []
        self.time_log_queue = queue.Queue()
        self.time_log_lock = threading.RLock()

        # Core system integrations
        self.relay_navigator = MathematicalRelayNavigator(mode, log_level) if MathematicalRelayNavigator else None
        self.relay_integration = MathematicalRelayIntegration(mode, log_level) if MathematicalRelayIntegration else None
        self.trend_manager = TrendStateManager() if TrendStateManager else None
        self.basket_engine = BasketRelayEngine() if BasketRelayEngine else None
        self.price_sequencer = PriceHashEventSequencer() if PriceHashEventSequencer else None
        self.ghost_system = GhostFileSystem() if GhostFileSystem else None
        self.backlog_manager = MathematicalBacklogManager() if MathematicalBacklogManager else None
        self.quicktime_manager = None  # Will be initialized after other managers
        self.ferris_rde_engine = FerrisRDEEngine(gpu_enabled=gpu_enabled) if FerrisRDEEngine else None

        # Initialize QuickTime manager with callback
        if all([self.trend_manager, self.basket_engine, self.backlog_manager]):
            self.quicktime_manager = QuickTimeEventManager(
                trend_manager=self.trend_manager,
                basket_engine=self.basket_engine,
                backlog_manager=self.backlog_manager,
                event_callback=self._handle_quicktime_event,
            )

        # Threading and synchronization
        self.sequencing_lock = threading.RLock()
        self.time_log_thread = threading.Thread(target=self._time_log_loop, daemon=True)
        self.sequence_validation_thread = threading.Thread(target=self._sequence_validation_loop, daemon=True)

        # Start background threads
        self.time_log_thread.start()
        self.sequence_validation_thread.start()

        # Initialize logging
        self._setup_logging()

        logger.info(
            f"MathematicalRelaySequencer initialized in {mode} mode with {time_log_level.value} precision, GPU enabled: {gpu_enabled}"
        )

    def _setup_logging(self) -> None:
        """Setup logging system."""
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)

        # File handler
        file_handler = logging.FileHandler(f"logs/mathematical_relay_sequencer_{self.mode}.log")
        file_handler.setLevel(getattr(logging, self.log_level.upper()))
        file_handler.setFormatter(logging.Formatter(log_format))

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, self.log_level.upper()))
        console_handler.setFormatter(logging.Formatter(log_format))

        # Configure logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.setLevel(getattr(logging, self.log_level.upper()))

    def _generate_sequence_id(self, sequence_type: SequenceType) -> str:
        """Generate unique sequence ID."""
        self.sequence_counter += 1
        timestamp = int(time.time() * 1_000_000)  # Microsecond precision
        return f"seq_{sequence_type.value}_{timestamp}_{self.sequence_counter}"

    def _generate_time_log_id(self, sequence_id: str, operation: str) -> str:
        """Generate unique time log entry ID."""
        timestamp = int(time.time() * 1_000_000)  # Microsecond precision
        return f"log_{sequence_id}_{operation}_{timestamp}"

    def start_sequence(
        self,
        sequence_type: SequenceType,
        btc_price: Optional[float] = None,
        btc_hash: Optional[str] = None,
        bit_depth: Optional[int] = None,
        channel: Optional[str] = None,
        profit_target: Optional[float] = None,
        basket_tier: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Start a new mathematical relay sequence."""
        try:
            sequence_id = self._generate_sequence_id(sequence_type)
            start_time = datetime.now()

            # Create sequence
            sequence = MathematicalSequence(
                sequence_id=sequence_id,
                sequence_type=sequence_type,
                start_time=start_time,
                btc_price=btc_price,
                btc_hash=btc_hash,
                bit_depth=bit_depth,
                channel=channel,
                profit_target=profit_target,
                basket_tier=basket_tier,
            )

            # Add initial time log
            initial_log = TimeLogEntry(
                entry_id=self._generate_time_log_id(sequence_id, "start"),
                sequence_id=sequence_id,
                sequence_type=sequence_type,
                timestamp=start_time,
                microsecond_precision=start_time.microsecond,
                operation="sequence_start",
                duration_microseconds=0,
                status=SequenceStatus.IN_PROGRESS,
                metadata=metadata or {},
            )
            sequence.add_time_log(initial_log)

            # Store sequence
            with self.sequencing_lock:
                self.active_sequences[sequence_id] = sequence

            # Add to time log queue
            self.time_log_queue.put(initial_log)

            logger.info(f"Started sequence {sequence_id}: {sequence_type.value}")
            return sequence_id

        except Exception as e:
            logger.error(f"Error starting sequence: {e}")
            raise

    def log_sequence_operation(
        self,
        sequence_id: str,
        operation: str,
        status: SequenceStatus = SequenceStatus.IN_PROGRESS,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Log an operation within a sequence."""
        try:
            with self.sequencing_lock:
                if sequence_id not in self.active_sequences:
                    raise ValueError(f"Sequence {sequence_id} not found")

                sequence = self.active_sequences[sequence_id]

            # Calculate duration from last operation
            current_time = datetime.now()
            duration_microseconds = 0

            if sequence.time_logs:
                last_log = sequence.time_logs[-1]
                duration = (current_time - last_log.timestamp).total_seconds() * 1_000_000
                duration_microseconds = int(duration)

            # Create time log entry
            log_entry = TimeLogEntry(
                entry_id=self._generate_time_log_id(sequence_id, operation),
                sequence_id=sequence_id,
                sequence_type=sequence.sequence_type,
                timestamp=current_time,
                microsecond_precision=current_time.microsecond,
                operation=operation,
                duration_microseconds=duration_microseconds,
                status=status,
                metadata=metadata or {},
            )

            # Add to sequence
            sequence.add_time_log(log_entry)

            # Add to time log queue
            self.time_log_queue.put(log_entry)

            logger.debug(f"Logged operation {operation} for sequence {sequence_id}")
            return log_entry.entry_id

        except Exception as e:
            logger.error(f"Error logging sequence operation: {e}")
            raise

    def complete_sequence(
        self,
        sequence_id: str,
        success: bool = True,
        error_message: Optional[str] = None,
        final_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Complete a mathematical relay sequence."""
        try:
            with self.sequencing_lock:
                if sequence_id not in self.active_sequences:
                    raise ValueError(f"Sequence {sequence_id} not found")

                sequence = self.active_sequences[sequence_id]
                end_time = datetime.now()

            # Update sequence status
            if success:
                sequence.status = SequenceStatus.COMPLETED
            else:
                sequence.status = SequenceStatus.FAILED
                sequence.error_message = error_message

            # Complete sequence
            sequence.complete(end_time)

            # Log final operation
            final_log = TimeLogEntry(
                entry_id=self._generate_time_log_id(sequence_id, "complete"),
                sequence_id=sequence_id,
                sequence_type=sequence.sequence_type,
                timestamp=end_time,
                microsecond_precision=end_time.microsecond,
                operation="sequence_complete",
                duration_microseconds=sequence.total_duration_microseconds,
                status=sequence.status,
                metadata=final_metadata or {},
            )
            sequence.add_time_log(final_log)

            # Move to completed sequences
            with self.sequencing_lock:
                del self.active_sequences[sequence_id]
                self.completed_sequences.append(sequence)

                # Keep only last 1000 completed sequences
                if len(self.completed_sequences) > 1000:
                    self.completed_sequences = self.completed_sequences[-1000:]

            # Add to time log queue
            self.time_log_queue.put(final_log)

            result = {
                "sequence_id": sequence_id,
                "success": success,
                "total_duration_microseconds": sequence.total_duration_microseconds,
                "total_duration_seconds": sequence.total_duration_microseconds / 1_000_000,
                "time_logs_count": len(sequence.time_logs),
                "end_time": end_time.isoformat(),
            }

            logger.info(
                f"Completed sequence {sequence_id}: {sequence.sequence_type.value} "
                f"in {result['total_duration_seconds']:.6f}s"
            )

            # Log the completed or failed sequence to the backlog
            if self.backlog_manager:
                self.backlog_manager.log_event("sequence_logs", asdict(sequence))

            return result

        except Exception as e:
            logger.error(f"Error completing sequence: {e}")
            raise

    def sequence_btc_price_hash(
        self, btc_price: float, btc_volume: float, phase: int = 32, additional_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Sequence BTC price hash operation with precise timing using Ferris RDE."""
        try:
            # Start sequence
            sequence_id = self.start_sequence(
                sequence_type=SequenceType.BTC_PRICE_HASH, btc_price=btc_price, metadata=additional_data
            )

            # Log BTC price processing
            self.log_sequence_operation(sequence_id, "btc_price_processing")

            # Perform RDE execution
            rde_result = {}
            if self.ferris_rde_engine:
                self.log_sequence_operation(sequence_id, "ferris_rde_execution_start")
                # Placeholder for historical_average_price and time_since_last_pull
                # These would typically come from a market data manager or historical data service
                historical_average_price = btc_price * 0.99  # Example
                time_since_last_pull = 1.0  # Example: 1 second/tick
                profit_delta = btc_price * 0.001  # Example
                risk_factor = 0.005  # Example

                rde_result = self.ferris_rde_engine.execute_ferris_rde(
                    current_price=btc_price,
                    historical_average_price=historical_average_price,
                    time_since_last_pull=time_since_last_pull,
                    profit_delta=profit_delta,
                    risk_factor=risk_factor,
                    bit_depth=phase,  # Use provided phase as bit_depth for RDE
                    market_condition_data=additional_data,
                )
                self.log_sequence_operation(sequence_id, "ferris_rde_execution_complete", metadata=rde_result)
                # Store RDE result in the sequence
                with self.sequencing_lock:
                    self.active_sequences[sequence_id].ferris_rde_result = rde_result
            else:
                rde_result = {"success": False, "error": "Ferris RDE Engine not available"}

            # Extract BTC hash from RDE result if available, otherwise generate fallback
            btc_hash = rde_result.get("time_memory_key", self._generate_btc_hash_fallback(btc_price, btc_volume, phase))

            # Process through relay navigator if available
            if self.relay_navigator:
                self.log_sequence_operation(sequence_id, "relay_navigator_processing")
                nav_result = self.relay_navigator.process_btc_price_update(
                    btc_price=btc_price, btc_volume=btc_volume, phase=phase, additional_data=additional_data
                )
                self.log_sequence_operation(sequence_id, "relay_navigation_complete")
            else:
                nav_result = {"success": False, "error": "Relay navigator not available"}

            # Process through relay integration if available
            if self.relay_integration:
                self.log_sequence_operation(sequence_id, "relay_integration_processing")
                integration_result = self.relay_integration.process_btc_price_update(
                    btc_price=btc_price, btc_volume=btc_volume, phase=phase, additional_data=additional_data
                )
                self.log_sequence_operation(sequence_id, "relay_integration_complete")
            else:
                integration_result = {"success": False, "error": "Relay integration not available"}

            # Complete sequence
            success = (
                nav_result.get("success", False)
                and integration_result.get("success", False)
                and rde_result.get("success", True)
            )  # Assuming RDE success if no error
            final_metadata = {
                "btc_hash": btc_hash,
                "nav_result": nav_result,
                "integration_result": integration_result,
                "ferris_rde_result": rde_result,
                "phase": phase,
            }

            result = self.complete_sequence(sequence_id=sequence_id, success=success, final_metadata=final_metadata)

            result.update(
                {
                    "btc_hash": btc_hash,
                    "nav_result": nav_result,
                    "integration_result": integration_result,
                    "ferris_rde_result": rde_result,
                }
            )

            return result

        except Exception as e:
            logger.error(f"Error in BTC price hash sequencing: {e}")
            return {"success": False, "error": str(e)}

    def sequence_bit_depth_switch(
        self,
        from_bit_depth: int,
        to_bit_depth: int,
        channel: str = "primary",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Sequence bit depth switching operation."""
        try:
            # Start sequence
            sequence_id = self.start_sequence(
                sequence_type=SequenceType.BIT_DEPTH_SWITCH, bit_depth=to_bit_depth, channel=channel, metadata=metadata
            )

            # Log bit depth switch initiation
            self.log_sequence_operation(sequence_id, "bit_depth_switch_initiation")

            # Switch bit depth in relay navigator if available
            if self.relay_navigator:
                self.log_sequence_operation(sequence_id, "relay_navigator_bit_depth_switch")
                # Note: This would call the actual bit depth switching method
                self.relay_navigator.switch_bit_depth(to_bit_depth)  # Assuming method exists and takes int
                self.log_sequence_operation(sequence_id, "bit_depth_switch_complete")

            # Perform a conceptual RDE operation related to bit depth change
            rde_result = {}
            if self.ferris_rde_engine:
                self.log_sequence_operation(sequence_id, "ferris_rde_bit_depth_impact_assessment")
                # Example: use some dummy values or retrieve real ones for RDE assessment
                current_price_dummy = 1.0  # Placeholder
                historical_average_dummy = 1.0  # Placeholder
                time_since_last_pull_dummy = 0.1  # Placeholder
                profit_delta_dummy = 0.0  # Placeholder
                risk_factor_dummy = 0.0  # Placeholder

                rde_result = self.ferris_rde_engine.execute_ferris_rde(
                    current_price=current_price_dummy,
                    historical_average_price=historical_average_dummy,
                    time_since_last_pull=time_since_last_pull_dummy,
                    profit_delta=profit_delta_dummy,
                    risk_factor=risk_factor_dummy,
                    bit_depth=to_bit_depth,
                    market_condition_data=metadata,
                )
                self.log_sequence_operation(sequence_id, "ferris_rde_bit_depth_impact_complete", metadata=rde_result)
                with self.sequencing_lock:
                    self.active_sequences[sequence_id].ferris_rde_result = rde_result

            # Complete sequence
            result = self.complete_sequence(
                sequence_id=sequence_id,
                success=True,
                final_metadata={
                    "from_bit_depth": from_bit_depth,
                    "to_bit_depth": to_bit_depth,
                    "rde_impact": rde_result,
                },
            )

            return result

        except Exception as e:
            logger.error(f"Error in bit depth switch sequencing: {e}")
            return {"success": False, "error": str(e)}

    def sequence_profit_optimization(
        self, profit_target: float, basket_tier: str, btc_price: float, metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Sequence profit optimization operation."""
        try:
            # Start sequence
            sequence_id = self.start_sequence(
                sequence_type=SequenceType.PROFIT_OPTIMIZATION,
                profit_target=profit_target,
                basket_tier=basket_tier,
                btc_price=btc_price,
                metadata=metadata,
            )

            # Log profit optimization initiation
            self.log_sequence_operation(sequence_id, "profit_optimization_initiation")

            # Process through basket engine if available
            if self.basket_engine:
                self.log_sequence_operation(sequence_id, "basket_engine_processing")
                # Note: This would call the actual basket processing method
                self.log_sequence_operation(sequence_id, "basket_processing_complete")

            # Perform a conceptual RDE operation related to profit optimization
            rde_result = {}
            if self.ferris_rde_engine:
                self.log_sequence_operation(sequence_id, "ferris_rde_profit_optimization_assessment")
                # Example: use actual profit target and current price for RDE assessment
                current_price_for_rde = btc_price
                historical_average_price_for_rde = btc_price * 0.995  # Example
                time_since_last_pull_for_rde = 1.0  # Example
                profit_delta_for_rde = profit_target * btc_price  # Example profit in absolute terms
                risk_factor_for_rde = 0.01  # Example

                rde_result = self.ferris_rde_engine.execute_ferris_rde(
                    current_price=current_price_for_rde,
                    historical_average_price=historical_average_price_for_rde,
                    time_since_last_pull=time_since_last_pull_for_rde,
                    profit_delta=profit_delta_for_rde,
                    risk_factor=risk_factor_for_rde,
                    bit_depth=32,  # Default or derive from basket tier
                    market_condition_data=metadata,
                )
                self.log_sequence_operation(sequence_id, "ferris_rde_profit_optimization_complete", metadata=rde_result)
                with self.sequencing_lock:
                    self.active_sequences[sequence_id].ferris_rde_result = rde_result

            # Complete sequence
            result = self.complete_sequence(
                sequence_id=sequence_id,
                success=True,
                final_metadata={
                    "profit_target": profit_target,
                    "basket_tier": basket_tier,
                    "rde_assessment": rde_result,
                },
            )

            return result

        except Exception as e:
            logger.error(f"Error in profit optimization sequencing: {e}")
            return {"success": False, "error": str(e)}

    def _generate_btc_hash_fallback(self, btc_price: float, btc_volume: float, phase: int) -> str:
        """Generate BTC hash fallback when RDE engine is not used for primary hashing."""
        # Create hash input
        hash_input = f"{btc_price:.8f}_{btc_volume:.8f}_{phase}_{int(time.time() * 1_000_000)}"

        # Generate hash
        btc_hash = hashlib.sha256(hash_input.encode()).hexdigest()

        return btc_hash

    def _handle_quicktime_event(self, event: Dict[str, Any]) -> None:
        """Handle QuickTime events with sequencing."""
        try:
            event_type = event.get("event_type", "unknown")
            context = event.get("context", {})

            # Start sequence for QuickTime event
            sequence_id = self.start_sequence(
                sequence_type=SequenceType.GHOST_LOGIC, metadata={"event_type": event_type, "context": context}
            )

            # Log event processing
            self.log_sequence_operation(sequence_id, f"quicktime_event_{event_type}")

            # Optionally, run RDE for QuickTime event impact assessment
            rde_result = {}
            if self.ferris_rde_engine and "btc_price" in context and "volume" in context:
                self.log_sequence_operation(sequence_id, "ferris_rde_quicktime_impact_assessment")
                current_price_for_rde = context["btc_price"]
                historical_average_price_for_rde = current_price_for_rde * 0.999  # Example
                time_since_last_pull_for_rde = 0.05  # Fast event
                profit_delta_for_rde = 0.0  # No direct profit yet, just event impact
                risk_factor_for_rde = 0.0  # Assess risk separately

                rde_result = self.ferris_rde_engine.execute_ferris_rde(
                    current_price=current_price_for_rde,
                    historical_average_price=historical_average_price_for_rde,
                    time_since_last_pull=time_since_last_pull_for_rde,
                    profit_delta=profit_delta_for_rde,
                    risk_factor=risk_factor_for_rde,
                    bit_depth=context.get("bit_depth", 32),
                    market_condition_data=context,
                )
                self.log_sequence_operation(sequence_id, "ferris_rde_quicktime_impact_complete", metadata=rde_result)
                with self.sequencing_lock:
                    self.active_sequences[sequence_id].ferris_rde_result = rde_result

            # Complete sequence
            self.complete_sequence(sequence_id, success=True, final_metadata={"rde_assessment": rde_result})

        except Exception as e:
            logger.error(f"Error handling QuickTime event: {e}")

    def _time_log_loop(self) -> None:
        """Background thread for processing and storing time logs."""
        while True:
            try:
                entry: TimeLogEntry = self.time_log_queue.get(timeout=1)
                with self.time_log_lock:
                    self.time_logs.append(entry)
                    # Log time log entry to backlog manager
                    if self.backlog_manager:
                        self.backlog_manager.log_event("time_logs", asdict(entry))

            except queue.Empty:
                pass
            except Exception as e:
                logger.error(f"Error in time log loop: {e}")
            finally:
                self.time_log_queue.task_done()

    def _sequence_validation_loop(self) -> None:
        """Background sequence validation loop."""
        while True:
            try:
                time.sleep(5)  # Check every 5 seconds

                # Validate active sequences
                with self.sequencing_lock:
                    current_time = datetime.now()
                    sequences_to_timeout = []

                    for sequence_id, sequence in self.active_sequences.items():
                        # Check for timeout (5 minutes)
                        if (current_time - sequence.start_time).total_seconds() > 300:
                            sequences_to_timeout.append(sequence_id)

                    # Timeout sequences
                    for sequence_id in sequences_to_timeout:
                        logger.warning(f"Sequence {sequence_id} timed out")
                        self.complete_sequence(sequence_id=sequence_id, success=False, error_message="Sequence timeout")

            except Exception as e:
                logger.error(f"Error in sequence validation loop: {e}")

    def get_sequence_status(self, sequence_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific sequence."""
        try:
            with self.sequencing_lock:
                # Check active sequences
                if sequence_id in self.active_sequences:
                    sequence = self.active_sequences[sequence_id]
                    return {
                        "sequence_id": sequence_id,
                        "status": sequence.status.value,
                        "type": sequence.sequence_type.value,
                        "start_time": sequence.start_time.isoformat(),
                        "time_logs_count": len(sequence.time_logs),
                        "total_duration_microseconds": sequence.total_duration_microseconds,
                        "ferris_rde_result": sequence.ferris_rde_result,
                    }

                # Check completed sequences
                for sequence in self.completed_sequences:
                    if sequence.sequence_id == sequence_id:
                        return {
                            "sequence_id": sequence_id,
                            "status": sequence.status.value,
                            "type": sequence.sequence_type.value,
                            "start_time": sequence.start_time.isoformat(),
                            "end_time": sequence.end_time.isoformat() if sequence.end_time else None,
                            "time_logs_count": len(sequence.time_logs),
                            "total_duration_microseconds": sequence.total_duration_microseconds,
                            "error_message": sequence.error_message,
                            "ferris_rde_result": sequence.ferris_rde_result,
                        }

                return None

        except Exception as e:
            logger.error(f"Error getting sequence status: {e}")
            return None

    def get_time_logs(
        self,
        sequence_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Get time logs with filtering options."""
        try:
            with self.time_log_lock:
                logs = self.time_logs.copy()

            # Filter by sequence ID
            if sequence_id:
                logs = [log for log in logs if log.sequence_id == sequence_id]

            # Filter by time range
            if start_time:
                logs = [log for log in logs if log.timestamp >= start_time]
            if end_time:
                logs = [log for log in logs if log.timestamp <= end_time]

            # Sort by timestamp
            logs.sort(key=lambda x: x.timestamp)

            # Limit results
            logs = logs[-limit:]

            # Convert to dictionaries
            return [asdict(log) for log in logs]

        except Exception as e:
            logger.error(f"Error getting time logs: {e}")
            return []

    def get_sequencing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive sequencing statistics."""
        try:
            with self.sequencing_lock:
                active_count = len(self.active_sequences)
                completed_count = len(self.completed_sequences)

            with self.time_log_lock:
                time_log_count = len(self.time_logs)

            # Calculate sequence type distribution
            type_distribution = {}
            status_distribution = {}

            with self.sequencing_lock:
                # Active sequences
                for sequence in self.active_sequences.values():
                    seq_type = sequence.sequence_type.value
                    seq_status = sequence.status.value
                    type_distribution[seq_type] = type_distribution.get(seq_type, 0) + 1
                    status_distribution[seq_status] = status_distribution.get(seq_status, 0) + 1

                # Completed sequences
                for sequence in self.completed_sequences:
                    seq_type = sequence.sequence_type.value
                    seq_status = sequence.status.value
                    type_distribution[seq_type] = type_distribution.get(seq_type, 0) + 1
                    status_distribution[seq_status] = status_distribution.get(seq_status, 0) + 1

            # Calculate average duration
            total_duration = 0
            completed_with_duration = 0

            with self.sequencing_lock:
                for sequence in self.completed_sequences:
                    if sequence.total_duration_microseconds > 0:
                        total_duration += sequence.total_duration_microseconds
                        completed_with_duration += 1

            avg_duration = total_duration / max(completed_with_duration, 1)

            statistics = {
                "active_sequences": active_count,
                "completed_sequences": completed_count,
                "total_sequences": active_count + completed_count,
                "time_logs_count": time_log_count,
                "sequence_type_distribution": type_distribution,
                "sequence_status_distribution": status_distribution,
                "average_duration_microseconds": avg_duration,
                "average_duration_seconds": avg_duration / 1_000_000,
                "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
                "timestamp": datetime.now().isoformat(),
            }

            return statistics

        except Exception as e:
            logger.error(f"Error getting sequencing statistics: {e}")
            return {"error": str(e)}

    def export_sequencing_data(self, filename: Optional[str] = None) -> str:
        """Export complete sequencing data to file."""
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"mathematical_relay_sequencing_{self.mode}_{timestamp}.json"

            # Get comprehensive data
            export_data = {
                "sequencer_info": {
                    "mode": self.mode,
                    "time_log_level": self.time_log_level.value,
                    "start_time": self.start_time.isoformat(),
                    "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
                },
                "statistics": self.get_sequencing_statistics(),
                "active_sequences": [
                    {
                        "sequence_id": seq.sequence_id,
                        "sequence_type": seq.sequence_type.value,
                        "start_time": seq.start_time.isoformat(),
                        "status": seq.status.value,
                        "time_logs_count": len(seq.time_logs),
                        "ferris_rde_result": seq.ferris_rde_result,
                    }
                    for seq in self.active_sequences.values()
                ],
                "recent_completed_sequences": [
                    {
                        "sequence_id": seq.sequence_id,
                        "sequence_type": seq.sequence_type.value,
                        "start_time": seq.start_time.isoformat(),
                        "end_time": seq.end_time.isoformat() if seq.end_time else None,
                        "status": seq.status.value,
                        "total_duration_microseconds": seq.total_duration_microseconds,
                        "error_message": seq.error_message,
                        "ferris_rde_result": seq.ferris_rde_result,
                    }
                    for seq in self.completed_sequences[-100:]  # Last 100 completed
                ],
                "recent_time_logs": self.get_time_logs(limit=1000),
                "export_timestamp": datetime.now().isoformat(),
            }

            # Write to file
            with open(filename, "w") as f:
                json.dump(export_data, f, indent=2, default=str)

            logger.info(f"Sequencing data exported to: {filename}")
            return filename

        except Exception as e:
            logger.error(f"Error exporting sequencing data: {e}")
            raise
