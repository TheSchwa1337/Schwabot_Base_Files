# Import safe print for Windows compatibility
try:
    from .utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug
import numpy as np
import math
except ImportError:
    try:
#         from core.utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug  # F811: duplicate import
    except ImportError:
def safe_print(message):
    print(message)
def info(message):
    print(f"[INFO] {message}")
def warn(message):
    print(f"[WARN] {message}")
def error(message):
    print(f"[ERROR] {message}")
def success(message):
    print(f"[SUCCESS] {message}")
def debug(message):
    print(f"[DEBUG] {message}")
from core.unified_math_system import unified_math
# #!/usr/bin/env python3
"""
Demo Pipeline Runner - Schwabot UROS v1.0
========================================

Executes the complete Schwabot pipeline:
DLT waveform + tick input → hash phase → strategy execution → profit output

Features:
- Complete pipeline execution simulation
- Real-time tick processing
- Strategy decision making
- Portfolio management
- Performance tracking
- Demo/live mode switching
"""

import time
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
# from core.unified_math_system import unified_math  # F811: duplicate import
import hashlib
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import queue

logger = logging.getLogger(__name__)

class PipelineMode(Enum):
    """Pipeline execution modes."""
    DEMO = "demo"
    LIVE = "live"
    BACKTEST = "backtest"
    SIMULATION = "simulation"

class PipelineStatus(Enum):
    """Pipeline execution status."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"

@dataclass
class TickEvent:
    """Tick event data."""
    timestamp: datetime
    asset: str
    price: float
    volume: float
    market_data: Dict[str, Any]
    hash_value: str
    bit_phases: Dict[str, int]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StrategyDecision:
    """Strategy decision result."""
    timestamp: datetime
    asset: str
    decision: str  # "buy", "sell", "hold", "rebalance"
    confidence: float
    tensor_score: float
    bit_phase: int
    basket_id: str
    quantity: float
    price: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PipelineResult:
    """Pipeline execution result."""
    execution_id: str
    start_time: datetime
    end_time: datetime
    status: PipelineStatus
    total_ticks: int
    total_decisions: int
    total_trades: int
    performance_metrics: Dict[str, Any]
    error_log: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class DemoPipelineRunner:
    """
    Demo pipeline runner for complete Schwabot execution.

    Mathematical Foundation:
    - Tick Processing: T(t) = f(price, volume, market_data)
    - Hash Generation: H(t) = hash(tick_data + timestamp)
    - Bit Phase Resolution: P(t) = resolve_bit_phase(H(t), mode)
    - Strategy Decision: S(t) = f(tensor_score, bit_phase, market_conditions)
    - Portfolio Update: P(t+1) = P(t) + Σ(trades * impacts)
    """

    def __init__(self, config_path: str = "./config/demo_runner_config.json"):
        self.config_path = config_path

        # Pipeline state
        self.mode: PipelineMode = PipelineMode.DEMO
        self.status: PipelineStatus = PipelineStatus.IDLE
        self.is_running: bool = False

        # Execution tracking
        self.execution_id: str = ""
        self.start_time: datetime = None
        self.end_time: datetime = None
        self.tick_count: int = 0
        self.decision_count: int = 0
        self.trade_count: int = 0

        # Data storage
        self.tick_history: List[TickEvent] = []
        self.decision_history: List[StrategyDecision] = []
        self.trade_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, Any] = {}

        # Initialize core components with real implementations
        self._initialize_core_components()

        # Execution queues
        self.tick_queue = queue.Queue()
        self.decision_queue = queue.Queue()
        self.trade_queue = queue.Queue()

        # Threading
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.stop_event = threading.Event()

        # Load configuration
        self._load_configuration()
        logger.info("Demo Pipeline Runner initialized with real core components")

    def _initialize_core_components(self) -> None:
        """Initialize all core components with real implementations."""
        try:
            # Import and initialize real core components
            from .dlt_waveform_engine import DLTWaveformEngine
            from .matrix_mapper import MatrixMapper
            from .profit_cycle_allocator import ProfitCycleAllocator
            from .real_trading_integration import get_real_trading_integration
            from .ferris_rde_core import get_ferris_rde_core
            from .tick_hash_processor import TickHashProcessor
            from .unified_mathematics_config import get_unified_math
            from .integrated_alif_aleph_system import IntegratedAlifAlephSystem

            # Initialize core components
            self.dlt_engine = DLTWaveformEngine()
            self.tensor_matcher = MatrixMapper()  # MatrixMapper handles tensor matching
            self.bit_phase_engine = MatrixMapper()  # MatrixMapper handles bit phase resolution
            self.matrix_mapper = MatrixMapper()
            self.profit_allocator = ProfitCycleAllocator()
            self.trade_simulator = get_real_trading_integration()
            self.demo_injector = IntegratedAlifAlephSystem()
            self.vector_exporter = get_unified_math()

            # Additional core components
            self.ferris_rde = get_ferris_rde_core()
            self.tick_processor = TickHashProcessor()
            self.unified_math = get_unified_math()

            logger.info("✅ All core components initialized successfully")

        except ImportError as e:
            logger.error(f"❌ Failed to import core component: {e}")
            raise RuntimeError(f"Critical core component missing: {e}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize core components: {e}")
            raise RuntimeError(f"Core component initialization failed: {e}")

    def _load_configuration(self) -> None:
        """Load demo runner configuration."""
        try:
            # Default configuration
            config = {
                "pipeline_settings": {
                    "default_mode": "demo",
                    "tick_interval_ms": 1000,
                    "max_execution_time_hours": 24,
                    "auto_save_interval_minutes": 5
                },
                "assets": ["BTC", "ETH", "USDC", "XRP", "SOL"],
                "market_conditions": {
                    "normal": {"volatility": 0.02, "trend": 0.0},
                    "volatile": {"volatility": 0.05, "trend": 0.0},
                    "bull": {"volatility": 0.03, "trend": 0.01},
                    "bear": {"volatility": 0.04, "trend": -0.008}
                },
                "strategy_configs": {
                    "conservative": {"risk_tolerance": 0.1, "position_size": 0.2},
                    "balanced": {"risk_tolerance": 0.3, "position_size": 0.3},
                    "aggressive": {"risk_tolerance": 0.5, "position_size": 0.4}
                }
            }

            logger.info("Demo runner configuration loaded")

        except Exception as e:
            logger.error(f"Error loading configuration: {e}")

    def set_mode(self, mode: PipelineMode) -> None:
        """Set pipeline execution mode."""
        self.mode = mode
        logger.info(f"Pipeline mode set to: {mode.value}")

    def start_pipeline(self, duration_minutes: int = 60) -> bool:
        """
        Start the demo pipeline execution.

        Parameters:
        -----------
        duration_minutes : int
            Duration to run the pipeline in minutes

        Returns:
        --------
        bool
            True if pipeline started successfully
        """
        try:
            if self.is_running:
                logger.warning("Pipeline is already running")
                return False

            # Initialize execution
            self.execution_id = f"pipeline_{int(time.time())}"
            self.start_time = datetime.now()
            self.end_time = self.start_time + timedelta(minutes=duration_minutes)
            self.status = PipelineStatus.RUNNING
            self.is_running = True
            self.stop_event.clear()

            # Reset counters
            self.tick_count = 0
            self.decision_count = 0
            self.trade_count = 0

            # Clear history
            self.tick_history.clear()
            self.decision_history.clear()
            self.trade_history.clear()

            # Start execution threads
            self._start_execution_threads()

            logger.info(f"Pipeline started: {self.execution_id} (duration: {duration_minutes} minutes)")
            return True

        except Exception as e:
            logger.error(f"Error starting pipeline: {e}")
            self.status = PipelineStatus.ERROR
            return False

    def stop_pipeline(self) -> bool:
        """
        Stop the demo pipeline execution.

        Returns:
        --------
        bool
            True if pipeline stopped successfully
        """
        try:
            if not self.is_running:
                logger.warning("Pipeline is not running")
                return False

            # Signal stop
            self.stop_event.set()
            self.is_running = False
            self.status = PipelineStatus.STOPPED
            self.end_time = datetime.now()

            # Wait for threads to finish
            self.executor.shutdown(wait=True)

            # Calculate final metrics
            self._calculate_final_metrics()

            # Export results
            self._export_pipeline_results()

            logger.info(f"Pipeline stopped: {self.execution_id}")
            return True

        except Exception as e:
            logger.error(f"Error stopping pipeline: {e}")
            return False

    def _start_execution_threads(self) -> None:
        """Start pipeline execution threads."""
        try:
            # Start tick generation thread
            self.executor.submit(self._tick_generation_loop)

            # Start decision processing thread
            self.executor.submit(self._decision_processing_loop)

            # Start trade execution thread
            self.executor.submit(self._trade_execution_loop)

            # Start monitoring thread
            self.executor.submit(self._monitoring_loop)

            logger.info("Pipeline execution threads started")

        except Exception as e:
            logger.error(f"Error starting execution threads: {e}")

    def _tick_generation_loop(self) -> None:
        """Generate real ticks using BTC price hashing and 16-bit mapping."""
        logger.info("🔄 Starting real tick generation loop")

        while not self.stop_event.is_set():
            try:
                # Generate real BTC price data
                btc_price = self._generate_real_btc_price()

                # Process through Ferris RDE for 16-bit mapping
                price_mapping = self.ferris_rde.map_btc_price_16bit(btc_price)

                # Generate real tick hash
                tick_hash = self.tick_processor.generate_tick_hash(
                    price=btc_price,
                    volume=np.random.uniform(500000, 2000000),
                    timestamp=time.time()
                )

                # Create real tick event
                tick_event = TickEvent(
                    timestamp=datetime.now(),
                    asset="BTC/USDC",
                    price=btc_price,
                    volume=np.random.uniform(500000, 2000000),
                    market_data={
                        "mapped_16bit": price_mapping.mapped_price,
                        "ferris_phase": self.ferris_rde.current_phase.value,
                        "volatility": np.random.uniform(0.01, 0.05),
                        "entropy_level": np.random.uniform(1.0, 8.0)
                    },
                    hash_value=tick_hash,
                    bit_phases={"BTC": price_mapping.mapped_price % 16}
                )

                # Add to queue
                self.tick_queue.put(tick_event)
                self.tick_count += 1

                # Sleep based on configuration
                time.sleep(self.config.get("pipeline_settings", {}).get("tick_interval_ms", 1000) / 1000.0)

            except Exception as e:
                logger.error(f"❌ Error in tick generation: {e}")
                time.sleep(1.0)  # Brief pause on error

    def _generate_real_btc_price(self) -> float:
        """Generate realistic BTC price using mathematical models."""
        try:
            # Use unified mathematics for price generation
            base_price = 50000.0

            # Get market conditions from configuration
            market_conditions = self.config.get("market_conditions", {}).get("normal", {})
            volatility = market_conditions.get("volatility", 0.02)
            trend = market_conditions.get("trend", 0.0)

            # Calculate price change using mathematical models
            price_change = np.random.normal(trend, volatility) * base_price

            # Apply DLT waveform adjustments if available
            if self.dlt_engine:
                dlt_adjustment = self.dlt_engine.calculate_waveform_adjustment(price_change)
                price_change *= dlt_adjustment

            # Calculate new price
            new_price = base_price + price_change

            # Ensure price stays within reasonable bounds
            new_price = unified_math.max(new_price, base_price * 0.5)  # Minimum 50% of base
            new_price = unified_math.min(new_price, base_price * 2.0)  # Maximum 200% of base

            return new_price

        except Exception as e:
            logger.error(f"Error generating BTC price: {e}")
            return 50000.0  # Fallback to base price

    def _decision_processing_loop(self) -> None:
        """Process tick events and make strategy decisions."""
        try:
            while self.is_running and not self.stop_event.is_set():
                try:
                    # Get tick from queue (non-blocking)
                    tick_event = self.tick_queue.get(timeout=1.0)

                    # Process tick through pipeline
                    decision = self._process_tick(tick_event)

                    if decision:
                        # Add to decision queue
                        self.decision_queue.put(decision)
                        self.decision_count += 1

                    # Store tick in history
                    self.tick_history.append(tick_event)

                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Error processing tick: {e}")

        except Exception as e:
            logger.error(f"Error in decision processing loop: {e}")
            self.status = PipelineStatus.ERROR

    def _process_tick(self, tick_event: TickEvent) -> Optional[StrategyDecision]:
        """Process tick using real mathematical logic and DLT integration."""
        try:
            # Calculate tensor score using real matrix mapping
            tensor_score = self.matrix_mapper.calculate_tensor_score(
                price=tick_event.price,
                volume=tick_event.volume,
                market_data=tick_event.market_data
            )

            # Determine bit phase using real bit phase engine
            bit_phase = self.bit_phase_engine.resolve_bit_phase(
                tick_event.hash_value,
                tick_event.market_data.get("mapped_16bit", 0)
            )

            # Make strategy decision using real mathematical logic
            decision = self._make_strategy_decision(tick_event, tensor_score, bit_phase)

            return decision

        except Exception as e:
            logger.error(f"❌ Error processing tick: {e}")
            return None

    def _make_strategy_decision(self, tick_event: TickEvent, tensor_score: float, bit_phase: int) -> StrategyDecision:
        """Make strategy decision using real mathematical logic."""
        try:
            # Use DLT engine for decision making
            dlt_decision = self.dlt_engine.analyze_tick_for_decision(
                price=tick_event.price,
                volume=tick_event.volume,
                tensor_score=tensor_score,
                bit_phase=bit_phase
            )

            # Calculate confidence using unified mathematics
            confidence = self.unified_math.execute_with_monitoring(
                "decision_confidence",
                self._calculate_decision_confidence,
                tensor_score, bit_phase, dlt_decision
            )

            # Determine action based on mathematical analysis
            if confidence > 0.7 and tensor_score > 0.6:
                decision_type = "buy"
                quantity = self._calculate_position_size(confidence, tensor_score)
            elif confidence < 0.3 or tensor_score < 0.4:
                decision_type = "sell"
                quantity = self._calculate_position_size(confidence, tensor_score)
            else:
                decision_type = "hold"
                quantity = 0.0

            # Generate basket ID using real matrix mapping
            basket_id = self.matrix_mapper.generate_basket_id(
                tick_event.hash_value,
                bit_phase,
                tensor_score
            )

            return StrategyDecision(
                timestamp=tick_event.timestamp,
                asset=tick_event.asset,
                decision=decision_type,
                confidence=confidence,
                tensor_score=tensor_score,
                bit_phase=bit_phase,
                basket_id=basket_id,
                quantity=quantity,
                price=tick_event.price,
                metadata={
                    "dlt_decision": dlt_decision,
                    "hash_value": tick_event.hash_value,
                    "mapped_16bit": tick_event.market_data.get("mapped_16bit", 0)
                }
            )

        except Exception as e:
            logger.error(f"❌ Error making strategy decision: {e}")
            # Return safe hold decision
            return StrategyDecision(
                timestamp=tick_event.timestamp,
                asset=tick_event.asset,
                decision="hold",
                confidence=0.5,
                tensor_score=0.5,
                bit_phase=0,
                basket_id="safe_hold",
                quantity=0.0,
                price=tick_event.price
            )

    def _calculate_decision_confidence(self, tensor_score: float, bit_phase: int, dlt_decision: float) -> float:
        """Calculate decision confidence using mathematical models."""
        try:
            # Base confidence from tensor score
            base_confidence = tensor_score

            # Bit phase adjustment
            bit_phase_adjustment = unified_math.min(bit_phase / 16.0, 1.0)

            # DLT decision adjustment
            dlt_adjustment = dlt_decision if dlt_decision > 0 else 0.5

            # Combine using weighted average
            confidence = (
                base_confidence * 0.4 +
                bit_phase_adjustment * 0.3 +
                dlt_adjustment * 0.3
            )

            return unified_math.max(0.0, unified_math.min(1.0, confidence))

        except Exception as e:
            logger.error(f"Error calculating decision confidence: {e}")
            return 0.5

    def _calculate_position_size(self, confidence: float, tensor_score: float) -> float:
        """Calculate position size using mathematical models."""
        try:
            # Base position size from confidence
            base_size = confidence * 0.1  # 10% of portfolio max

            # Tensor score adjustment
            tensor_adjustment = tensor_score * 0.05  # Additional 5% based on tensor

            # Apply risk management
            max_position = 0.15  # Maximum 15% of portfolio

            position_size = unified_math.min(base_size + tensor_adjustment, max_position)

            return unified_math.max(0.0, position_size)

        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0

    def _trade_execution_loop(self) -> None:
        """Execute trades based on strategy decisions."""
        try:
            while self.is_running and not self.stop_event.is_set():
                try:
                    # Get decision from queue (non-blocking)
                    decision = self.decision_queue.get(timeout=1.0)

                    # Execute trade
                    if decision.decision in ["buy", "sell"]:
                        trade_result = self._execute_trade(decision)

                        if trade_result:
                            # Add to trade queue
                            self.trade_queue.put(trade_result)
                            self.trade_count += 1

                    # Store decision in history
                    self.decision_history.append(decision)

                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Error executing trade: {e}")

        except Exception as e:
            logger.error(f"Error in trade execution loop: {e}")
            self.status = PipelineStatus.ERROR

    def _execute_trade(self, decision: StrategyDecision) -> Optional[Dict[str, Any]]:
        """Execute a trade based on strategy decision."""
        try:
            if self.trade_simulator:
                # Create strategy bucket
                strategy_bucket = {
                    'asset': decision.asset,
                    'strategy_id': 'demo_strategy',
                    'tensor_score': decision.tensor_score,
                    'bit_phase': decision.bit_phase,
                    'basket_id': decision.basket_id,
                    'current_price': decision.price,
                    'market_data': decision.metadata.get('market_data', {})
                }

                # Simulate trade
                trade_result = self.trade_simulator.simulate_trade(strategy_bucket, self.mode.value.upper())

                if trade_result and trade_result.status.value == "executed":
                    return {
                        'timestamp': decision.timestamp,
                        'asset': decision.asset,
                        'trade_type': decision.decision,
                        'quantity': decision.quantity,
                        'price': decision.price,
                        'tensor_score': decision.tensor_score,
                        'bit_phase': decision.bit_phase,
                        'basket_id': decision.basket_id,
                        'execution_id': trade_result.trade_id
                    }

            return None

        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return None

    def _monitoring_loop(self) -> None:
        """Monitor pipeline execution and performance."""
        try:
            last_save_time = datetime.now()

            while self.is_running and not self.stop_event.is_set():
                current_time = datetime.now()

                # Check if execution time exceeded
                if current_time >= self.end_time:
                    logger.info("Pipeline execution time exceeded")
                    break

                # Auto-save every 5 minutes
                if (current_time - last_save_time).total_seconds() > 300:  # 5 minutes
                    self._save_pipeline_state()
                    last_save_time = current_time

                # Update performance metrics
                self._update_performance_metrics()

                # Wait before next check
                time.sleep(10.0)  # Check every 10 seconds

        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
            self.status = PipelineStatus.ERROR

    def _update_performance_metrics(self) -> None:
        """Update real-time performance metrics."""
        try:
            current_time = datetime.now()
            execution_time = (current_time - self.start_time).total_seconds() if self.start_time else 0

            self.performance_metrics = {
                'execution_time_seconds': execution_time,
                'tick_count': self.tick_count,
                'decision_count': self.decision_count,
                'trade_count': self.trade_count,
                'ticks_per_second': self.tick_count / execution_time if execution_time > 0 else 0,
                'decisions_per_second': self.decision_count / execution_time if execution_time > 0 else 0,
                'trades_per_second': self.trade_count / execution_time if execution_time > 0 else 0,
                'decision_rate': self.decision_count / self.tick_count if self.tick_count > 0 else 0,
                'trade_rate': self.trade_count / self.decision_count if self.decision_count > 0 else 0
            }

        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")

    def _calculate_final_metrics(self) -> None:
        """Calculate final pipeline metrics."""
        try:
            if not self.start_time or not self.end_time:
                return

            total_time = (self.end_time - self.start_time).total_seconds()

            # Calculate final metrics
            final_metrics = {
                'total_execution_time_seconds': total_time,
                'total_ticks_processed': self.tick_count,
                'total_decisions_made': self.decision_count,
                'total_trades_executed': self.trade_count,
                'average_ticks_per_second': self.tick_count / total_time if total_time > 0 else 0,
                'average_decisions_per_second': self.decision_count / total_time if total_time > 0 else 0,
                'average_trades_per_second': self.trade_count / total_time if total_time > 0 else 0,
                'decision_efficiency': self.decision_count / self.tick_count if self.tick_count > 0 else 0,
                'trade_efficiency': self.trade_count / self.decision_count if self.decision_count > 0 else 0,
                'pipeline_status': self.status.value,
                'execution_mode': self.mode.value
            }

            self.performance_metrics.update(final_metrics)

        except Exception as e:
            logger.error(f"Error calculating final metrics: {e}")

    def _save_pipeline_state(self) -> None:
        """Save current pipeline state."""
        try:
            state_data = {
                'execution_id': self.execution_id,
                'timestamp': datetime.now().isoformat(),
                'mode': self.mode.value,
                'status': self.status.value,
                'tick_count': self.tick_count,
                'decision_count': self.decision_count,
                'trade_count': self.trade_count,
                'performance_metrics': self.performance_metrics
            }

            # Save to file
            filename = f"pipeline_state_{self.execution_id}.json"
            with open(filename, 'w') as f:
                json.dump(state_data, f, indent=2, default=str)

            logger.info(f"Pipeline state saved: {filename}")

        except Exception as e:
            logger.error(f"Error saving pipeline state: {e}")

    def _export_pipeline_results(self) -> None:
        """Export final pipeline results."""
        try:
            # Create pipeline result
            result = PipelineResult(
                execution_id=self.execution_id,
                start_time=self.start_time,
                end_time=self.end_time,
                status=self.status,
                total_ticks=self.tick_count,
                total_decisions=self.decision_count,
                total_trades=self.trade_count,
                performance_metrics=self.performance_metrics,
                metadata={
                    'mode': self.mode.value,
                    'tick_history_count': len(self.tick_history),
                    'decision_history_count': len(self.decision_history),
                    'trade_history_count': len(self.trade_history)
                }
            )

            # Export using vector exporter
            if self.vector_exporter:
                export_data = {
                    'execution_id': result.execution_id,
                    'start_time': result.start_time.isoformat(),
                    'end_time': result.end_time.isoformat(),
                    'status': result.status.value,
                    'total_ticks': result.total_ticks,
                    'total_decisions': result.total_decisions,
                    'total_trades': result.total_trades,
                    'performance_metrics': result.performance_metrics,
                    'metadata': result.metadata
                }

                self.vector_exporter.export_vector_snapshot(
                    snapshot_type=self.vector_exporter.SnapshotType.COMPLETE_STATE,
                    data=export_data,
                    export_format=self.vector_exporter.ExportFormat.JSON,
                    compress=True
                )

            logger.info(f"Pipeline results exported for execution: {self.execution_id}")

        except Exception as e:
            logger.error(f"Error exporting pipeline results: {e}")

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        return {
            'execution_id': self.execution_id,
            'mode': self.mode.value,
            'status': self.status.value,
            'is_running': self.is_running,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'tick_count': self.tick_count,
            'decision_count': self.decision_count,
            'trade_count': self.trade_count,
            'performance_metrics': self.performance_metrics
        }

    def set_dlt_engine(self, dlt_engine) -> None:
        """Set DLT engine for integration."""
        self.dlt_engine = dlt_engine
        logger.info("DLT engine integrated with demo runner")

    def set_tensor_matcher(self, tensor_matcher) -> None:
        """Set tensor matcher for integration."""
        self.tensor_matcher = tensor_matcher
        logger.info("Tensor matcher integrated with demo runner")

    def set_bit_phase_engine(self, bit_engine) -> None:
        """Set bit phase engine for integration."""
        self.bit_phase_engine = bit_engine
        logger.info("Bit phase engine integrated with demo runner")

    def set_matrix_mapper(self, matrix_mapper) -> None:
        """Set matrix mapper for integration."""
        self.matrix_mapper = matrix_mapper
        logger.info("Matrix mapper integrated with demo runner")

    def set_profit_allocator(self, profit_allocator) -> None:
        """Set profit allocator for integration."""
        self.profit_allocator = profit_allocator
        logger.info("Profit allocator integrated with demo runner")

    def set_trade_simulator(self, trade_simulator) -> None:
        """Set trade simulator for integration."""
        self.trade_simulator = trade_simulator
        logger.info("Trade simulator integrated with demo runner")

    def set_demo_injector(self, demo_injector) -> None:
        """Set demo injector for integration."""
        self.demo_injector = demo_injector
        logger.info("Demo injector integrated with demo runner")

    def set_vector_exporter(self, vector_exporter) -> None:
        """Set vector exporter for integration."""
        self.vector_exporter = vector_exporter
        logger.info("Vector exporter integrated with demo runner")

if __name__ == "__main__":
    # Test demo pipeline runner
    runner = DemoPipelineRunner()

    # Set to demo mode
    runner.set_mode(PipelineMode.DEMO)

    # Start pipeline for 2 minutes
    safe_print("🚀 Starting demo pipeline...")
    success = runner.start_pipeline(duration_minutes=2)

    if success:
        safe_print("✅ Pipeline started successfully")

        # Monitor for 10 seconds
        for i in range(10):
            time.sleep(1)
            status = runner.get_pipeline_status()
            safe_print(f"📊 Status: {status['status']} | Ticks: {status['tick_count']} | Decisions: {status['decision_count']} | Trades: {status['trade_count']}")

        # Stop pipeline
        safe_print("⏹️ Stopping pipeline...")
        runner.stop_pipeline()

        # Final status
        final_status = runner.get_pipeline_status()
        safe_print(f"🏁 Final Status: {final_status['status']}")
        safe_print(f"📈 Performance: {final_status['performance_metrics']}")
    else:
        safe_print("❌ Failed to start pipeline")
