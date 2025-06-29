# -*- coding: utf-8 -*-
"""
Ferris RDE Daemon - 24/7 Trading Pipeline with Integrated Components.

This daemon implements a comprehensive 24/7 trading system that integrates:
- Ferris RDE (Recursive Dualistic Engine) for cyclical trading patterns
- Multi-bit state management with CPU/GPU/distributed processing
- Advanced mathematical frameworks (Ferris Wheel, Quantum Thermal, Void Well)
- Unified connectivity management for API, trading, and visualization
- Real-time order book vectorization and strategy bit mapping
- Entry/exit logic with risk management
- Ghost overlay and phase transition monitoring
- Comprehensive health monitoring and performance metrics

The daemon runs continuously, processing market data through the complete pipeline
and making trading decisions based on integrated mathematical analysis.
"""

import asyncio
import logging
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

# Core imports
from core.trading_pipeline_integration import (
    TradingPipelineIntegration,
    TradeTickPipeline,
    TradingSignal,
    PortfolioState,
)
from core.unified_connectivity_manager import (
    UnifiedConnectivityManager,
    get_unified_connectivity_manager,
)
from core.advanced_mathematical_core import (
    calculate_ferris_wheel_state,
    calculate_quantum_thermal_state,
    calculate_void_well_metrics,
    calculate_profit_state,
    calculate_kelly_metrics,
    calculate_recursive_time_lock_sync,
)
from core.api_bridge import APIBridge
from core.order_book_vectorizer import OrderBookVectorizer
from core.strategy_bit_mapper import StrategyBitMapper
from core.entry_exit_logic import EntryExitLogic
from core.multi_bit_state_manager import MultiBitStateManager, ProcessingMode
from core.dualistic_thought_engines import DualisticThoughtEngines
from core.settings_manager import get_settings_manager

# Schwabot core imports
try:
    from schwabot.core.ferris_rde import FerrisRDE, FerrisPhase, FerrisState
    from schwabot.core.ghost_field_stabilizer import GhostFieldStabilizer
    from schwabot.core.phase.phase_transition_monitor import PhaseTransitionMonitor
    from schwabot.core.dlt_waveform_engine import DLTWaveformEngine
    from schwabot.core.wallet_tracker import WalletTracker
except ImportError as e:
    logging.warning(f"Some schwabot core modules not available: {e}")
    FerrisRDE = None
    GhostFieldStabilizer = None
    PhaseTransitionMonitor = None
    DLTWaveformEngine = None
    WalletTracker = None

# Utils
from utils.safe_print import debug, error, info, safe_print, success, warn

logger = logging.getLogger(__name__)


@dataclass
class DaemonConfig:
    """Configuration for the Ferris RDE Daemon."""
    
    # Core settings
    enabled: bool = True
    daemon_name: str = "FerrisRDE"
    log_level: str = "INFO"
    
    # Trading settings
    trading_enabled: bool = True
    paper_trading: bool = True  # Set to False for live trading
    max_concurrent_trades: int = 10
    risk_management_enabled: bool = True
    
    # Processing settings
    enable_gpu: bool = True
    enable_distributed: bool = False
    bit_depth_range: Tuple[int, int] = (2, 42)
    
    # Timing settings
    tick_interval_seconds: float = 1.0
    health_check_interval_seconds: float = 30.0
    performance_report_interval_seconds: float = 300.0  # 5 minutes
    
    # Asset settings
    primary_assets: List[str] = field(default_factory=lambda: ["BTC/USD", "ETH/USD"])
    secondary_assets: List[str] = field(default_factory=lambda: ["XRP/USD", "ADA/USD"])
    
    # Ferris RDE settings
    ferris_cycle_duration_minutes: int = 60
    ferris_phase_transitions: Dict[str, float] = field(default_factory=lambda: {
        "tick_to_pivot": 0.8,
        "pivot_to_ascent": 0.7,
        "ascent_to_descent": 0.6,
        "descent_to_tick": 0.9
    })
    
    # Mathematical settings
    mathematical_update_interval_seconds: float = 5.0
    enable_quantum_thermal: bool = True
    enable_void_well_metrics: bool = True
    enable_kelly_criterion: bool = True
    
    # Monitoring settings
    enable_health_monitoring: bool = True
    enable_performance_tracking: bool = True
    enable_error_logging: bool = True
    max_error_count: int = 1000
    
    # Visualization settings
    enable_visualization: bool = True
    dashboard_port: int = 8080
    websocket_port: int = 8081
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.tick_interval_seconds < 0.1:
            raise ValueError("Tick interval must be at least 0.1 seconds")
        if self.max_concurrent_trades < 1:
            raise ValueError("Max concurrent trades must be at least 1")
        if self.bit_depth_range[0] < 1 or self.bit_depth_range[1] > 64:
            raise ValueError("Bit depth range must be between 1 and 64")


@dataclass
class DaemonMetrics:
    """Metrics tracking for the daemon."""
    
    start_time: datetime = field(default_factory=datetime.now)
    total_ticks_processed: int = 0
    total_signals_generated: int = 0
    total_trades_executed: int = 0
    total_errors: int = 0
    total_warnings: int = 0
    
    # Performance metrics
    avg_tick_processing_time_ms: float = 0.0
    avg_signal_generation_time_ms: float = 0.0
    avg_trade_execution_time_ms: float = 0.0
    
    # System metrics
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    active_connections: int = 0
    
    # Trading metrics
    total_profit: float = 0.0
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    
    # Mathematical metrics
    ferris_cycles_completed: int = 0
    mathematical_states_updated: int = 0
    phase_transitions: int = 0
    
    def update_processing_time(self, processing_time_ms: float) -> None:
        """Update average processing time."""
        if self.total_ticks_processed == 0:
            self.avg_tick_processing_time_ms = processing_time_ms
        else:
            self.avg_tick_processing_time_ms = (
                (self.avg_tick_processing_time_ms * self.total_ticks_processed + processing_time_ms) /
                (self.total_ticks_processed + 1)
            )
    
    def get_uptime_seconds(self) -> float:
        """Get daemon uptime in seconds."""
        return (datetime.now() - self.start_time).total_seconds()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        return {
            "uptime_seconds": self.get_uptime_seconds(),
            "total_ticks_processed": self.total_ticks_processed,
            "total_signals_generated": self.total_signals_generated,
            "total_trades_executed": self.total_trades_executed,
            "total_errors": self.total_errors,
            "total_warnings": self.total_warnings,
            "avg_tick_processing_time_ms": self.avg_tick_processing_time_ms,
            "avg_signal_generation_time_ms": self.avg_signal_generation_time_ms,
            "avg_trade_execution_time_ms": self.avg_trade_execution_time_ms,
            "cpu_usage_percent": self.cpu_usage_percent,
            "memory_usage_mb": self.memory_usage_mb,
            "active_connections": self.active_connections,
            "total_profit": self.total_profit,
            "win_rate": self.win_rate,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "ferris_cycles_completed": self.ferris_cycles_completed,
            "mathematical_states_updated": self.mathematical_states_updated,
            "phase_transitions": self.phase_transitions,
        }


class FerrisRDEDaemon:
    """
    Ferris RDE Daemon - 24/7 Trading Pipeline with Integrated Components.
    
    This daemon runs continuously, processing market data through the complete
    pipeline and making trading decisions based on integrated mathematical analysis.
    """
    
    def __init__(self, config: Optional[DaemonConfig] = None):
        """
        Initialize the Ferris RDE Daemon.
        
        Args:
            config: Daemon configuration
        """
        self.config = config or DaemonConfig()
        self.metrics = DaemonMetrics()
        
        # Initialize core components
        self._initialize_core_components()
        
        # Initialize trading components
        self._initialize_trading_components()
        
        # Initialize mathematical components
        self._initialize_mathematical_components()
        
        # Initialize monitoring components
        self._initialize_monitoring_components()
        
        # State management
        self.running = False
        self.shutdown_requested = False
        self._shutdown_event = threading.Event()
        self._main_loop_task: Optional[asyncio.Task] = None
        
        # Threading
        self._executor = ThreadPoolExecutor(max_workers=self.config.max_concurrent_trades)
        self._lock = threading.RLock()
        
        logger.info(f"🎡 Ferris RDE Daemon initialized: {self.config.daemon_name}")
    
    def _initialize_core_components(self) -> None:
        """Initialize core system components."""
        try:
            # Settings manager
            self.settings_manager = get_settings_manager()
            
            # Unified connectivity manager
            self.connectivity_manager = get_unified_connectivity_manager()
            
            # API bridge
            self.api_bridge = APIBridge()
            
            # Multi-bit state manager
            self.multi_bit_manager = MultiBitStateManager(
                enable_gpu=self.config.enable_gpu,
                enable_distributed=self.config.enable_distributed,
            )
            
            # Dualistic thought engines
            self.dualistic_engines = DualisticThoughtEngines()
            
            logger.info("✅ Core components initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize core components: {e}")
            raise
    
    def _initialize_trading_components(self) -> None:
        """Initialize trading-specific components."""
        try:
            # Trading pipeline integration
            self.trading_pipeline = TradingPipelineIntegration(
                enable_gpu=self.config.enable_gpu,
                enable_distributed=self.config.enable_distributed,
                max_concurrent_trades=self.config.max_concurrent_trades,
                risk_management_enabled=self.config.risk_management_enabled,
            )
            
            # Order book vectorizer
            self.order_book_vectorizer = OrderBookVectorizer()
            
            # Strategy bit mapper
            self.strategy_bit_mapper = StrategyBitMapper()
            
            # Entry/exit logic
            self.entry_exit_logic = EntryExitLogic()
            
            # Trade tick pipelines for each asset
            self.trade_tick_pipelines: Dict[str, TradeTickPipeline] = {}
            for asset in self.config.primary_assets + self.config.secondary_assets:
                pipeline = TradeTickPipeline(bit_depth=16)
                self.trade_tick_pipelines[asset] = pipeline
                self.connectivity_manager.register_trade_tick_pipeline(asset, pipeline)
            
            logger.info("✅ Trading components initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize trading components: {e}")
            raise
    
    def _initialize_mathematical_components(self) -> None:
        """Initialize mathematical components."""
        try:
            # Ferris RDE
            if FerrisRDE:
                self.ferris_rde = FerrisRDE()
                logger.info("✅ Ferris RDE initialized")
            else:
                self.ferris_rde = None
                logger.warning("⚠️ Ferris RDE not available")
            
            # Ghost field stabilizer
            if GhostFieldStabilizer:
                self.ghost_stabilizer = GhostFieldStabilizer()
                logger.info("✅ Ghost field stabilizer initialized")
            else:
                self.ghost_stabilizer = None
                logger.warning("⚠️ Ghost field stabilizer not available")
            
            # Phase transition monitor
            if PhaseTransitionMonitor:
                self.phase_monitor = PhaseTransitionMonitor()
                logger.info("✅ Phase transition monitor initialized")
            else:
                self.phase_monitor = None
                logger.warning("⚠️ Phase transition monitor not available")
            
            # DLT waveform engine
            if DLTWaveformEngine:
                self.dlt_engine = DLTWaveformEngine()
                logger.info("✅ DLT waveform engine initialized")
            else:
                self.dlt_engine = None
                logger.warning("⚠️ DLT waveform engine not available")
            
            # Wallet tracker
            if WalletTracker:
                self.wallet_tracker = WalletTracker()
                logger.info("✅ Wallet tracker initialized")
            else:
                self.wallet_tracker = None
                logger.warning("⚠️ Wallet tracker not available")
            
            logger.info("✅ Mathematical components initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize mathematical components: {e}")
            raise
    
    def _initialize_monitoring_components(self) -> None:
        """Initialize monitoring and health components."""
        try:
            # Health monitoring
            self.health_check_interval = self.config.health_check_interval_seconds
            self.last_health_check = time.time()
            
            # Performance tracking
            self.performance_report_interval = self.config.performance_report_interval_seconds
            self.last_performance_report = time.time()
            
            # Error tracking
            self.error_count = 0
            self.warning_count = 0
            self.error_log: List[Dict[str, Any]] = []
            
            logger.info("✅ Monitoring components initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize monitoring components: {e}")
            raise
    
    async def start(self) -> bool:
        """
        Start the Ferris RDE Daemon.
        
        Returns:
            True if started successfully
        """
        try:
            with self._lock:
                if self.running:
                    logger.warning("Daemon already running")
                    return True
                
                logger.info("🚀 Starting Ferris RDE Daemon...")
                
                # Start connectivity manager
                await self.connectivity_manager.start()
                
                # Start Ferris RDE cycle
                if self.ferris_rde:
                    self.ferris_rde.start_cycle()
                
                # Set running state
                self.running = True
                self.shutdown_requested = False
                self._shutdown_event.clear()
                
                # Start main loop
                self._main_loop_task = asyncio.create_task(self._main_loop())
                
                logger.info("✅ Ferris RDE Daemon started successfully")
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to start daemon: {e}")
            return False
    
    async def stop(self) -> bool:
        """
        Stop the Ferris RDE Daemon.
        
        Returns:
            True if stopped successfully
        """
        try:
            with self._lock:
                if not self.running:
                    logger.warning("Daemon not running")
                    return True
                
                logger.info("🛑 Stopping Ferris RDE Daemon...")
                
                # Request shutdown
                self.shutdown_requested = True
                self._shutdown_event.set()
                
                # Cancel main loop
                if self._main_loop_task:
                    self._main_loop_task.cancel()
                    try:
                        await self._main_loop_task
                    except asyncio.CancelledError:
                        pass
                
                # Stop connectivity manager
                await self.connectivity_manager.stop()
                
                # End Ferris RDE cycle
                if self.ferris_rde:
                    self.ferris_rde.end_cycle()
                
                # Cleanup
                self._cleanup()
                
                # Set stopped state
                self.running = False
                
                logger.info("✅ Ferris RDE Daemon stopped successfully")
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to stop daemon: {e}")
            return False
    
    def _cleanup(self) -> None:
        """Cleanup daemon resources."""
        try:
            # Shutdown executor
            if self._executor:
                self._executor.shutdown(wait=True)
            
            # Cleanup trading pipeline
            if self.trading_pipeline:
                self.trading_pipeline.cleanup()
            
            # Cleanup multi-bit manager
            if self.multi_bit_manager:
                self.multi_bit_manager.cleanup()
            
            logger.info("✅ Daemon cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")
    
    async def _main_loop(self) -> None:
        """Main daemon loop."""
        logger.info("🔄 Starting main daemon loop")
        
        try:
            while not self.shutdown_requested:
                loop_start_time = time.time()
                
                # Process market data for all assets
                await self._process_all_assets()
                
                # Update mathematical states
                await self._update_mathematical_states()
                
                # Perform health checks
                await self._perform_health_checks()
                
                # Generate performance reports
                await self._generate_performance_reports()
                
                # Calculate sleep time
                processing_time = time.time() - loop_start_time
                sleep_time = max(0, self.config.tick_interval_seconds - processing_time)
                
                # Sleep or wait for shutdown
                if sleep_time > 0:
                    try:
                        await asyncio.wait_for(
                            asyncio.sleep(sleep_time),
                            timeout=sleep_time
                        )
                    except asyncio.TimeoutError:
                        pass
                
        except asyncio.CancelledError:
            logger.info("🔄 Main loop cancelled")
        except Exception as e:
            logger.error(f"❌ Error in main loop: {e}")
            self._log_error("main_loop", str(e))
        finally:
            logger.info("🔄 Main loop ended")
    
    async def _process_all_assets(self) -> None:
        """Process market data for all configured assets."""
        try:
            # Process primary assets first
            for asset in self.config.primary_assets:
                await self._process_asset(asset, priority="high")
            
            # Process secondary assets
            for asset in self.config.secondary_assets:
                await self._process_asset(asset, priority="normal")
                
        except Exception as e:
            logger.error(f"❌ Error processing assets: {e}")
            self._log_error("asset_processing", str(e))
    
    async def _process_asset(self, asset: str, priority: str = "normal") -> None:
        """
        Process market data for a specific asset.
        
        Args:
            asset: Asset symbol (e.g., "BTC/USD")
            priority: Processing priority ("high", "normal", "low")
        """
        start_time = time.time()
        
        try:
            # Fetch market data
            market_data = await self._fetch_market_data(asset)
            if not market_data:
                return
            
            # Process through trading pipeline
            trading_signal = await self.trading_pipeline.process_market_data(
                market_data=market_data,
                asset=asset,
                thermal_state="warm"
            )
            
            # Process through Ferris RDE if available
            if self.ferris_rde:
                ferris_signal = self.ferris_rde.generate_signal(market_data)
                if ferris_signal:
                    logger.debug(f"Ferris signal for {asset}: {ferris_signal.signal_type}")
            
            # Process through trade tick pipeline
            if asset in self.trade_tick_pipelines:
                pipeline = self.trade_tick_pipelines[asset]
                
                # Extract order book from market data
                order_book = market_data.get("order_book", {})
                if order_book:
                    # Calculate Ferris phase
                    ferris_phase = self._calculate_ferris_phase(market_data)
                    
                    # Calculate ghost signal
                    ghost_signal = self._calculate_ghost_signal(market_data)
                    
                    # Process tick
                    tick = pipeline.process_tick(
                        order_book=order_book,
                        symbol=asset,
                        ferris_phase=ferris_phase,
                        ghost_signal=ghost_signal
                    )
                    
                    # Update metrics
                    self.metrics.total_ticks_processed += 1
            
            # Execute trading signal if confidence is high enough
            if trading_signal.confidence > 0.7:
                await self._execute_trading_signal(trading_signal, market_data)
            
            # Update processing metrics
            processing_time = (time.time() - start_time) * 1000
            self.metrics.update_processing_time(processing_time)
            
        except Exception as e:
            logger.error(f"❌ Error processing asset {asset}: {e}")
            self._log_error("asset_processing", f"{asset}: {str(e)}")
    
    async def _fetch_market_data(self, asset: str) -> Optional[Dict[str, Any]]:
        """
        Fetch market data for an asset.
        
        Args:
            asset: Asset symbol
            
        Returns:
            Market data dictionary or None if failed
        """
        try:
            # Fetch price data
            price_data = await self.api_bridge.fetch_price_data(asset)
            
            # Fetch order book
            order_book = await self.api_bridge.fetch_order_book(asset, limit=20)
            
            # Fetch additional market data
            market_data = {
                "asset": asset,
                "timestamp": time.time(),
                "price_data": price_data,
                "order_book": order_book,
                "current_price": price_data.get("price", 0.0) if price_data else 0.0,
                "price_change": price_data.get("change_24h", 0.0) if price_data else 0.0,
                "volume": price_data.get("volume_24h", 0.0) if price_data else 0.0,
                "volatility": self._calculate_volatility(price_data),
                "temperature": 300.0,  # Default temperature
            }
            
            return market_data
            
        except Exception as e:
            logger.error(f"❌ Error fetching market data for {asset}: {e}")
            return None
    
    def _calculate_volatility(self, price_data: Dict[str, Any]) -> float:
        """Calculate volatility from price data."""
        try:
            if not price_data or "price" not in price_data:
                return 0.0
            
            # Simple volatility calculation
            price = price_data["price"]
            change_24h = price_data.get("change_24h", 0.0)
            
            # Volatility as percentage change
            volatility = abs(change_24h) / 100.0 if price > 0 else 0.0
            
            return min(volatility, 1.0)  # Cap at 100%
            
        except Exception:
            return 0.0
    
    def _calculate_ferris_phase(self, market_data: Dict[str, Any]) -> float:
        """Calculate Ferris phase from market data."""
        try:
            # Simple phase calculation based on timestamp
            timestamp = market_data.get("timestamp", time.time())
            phase_period = self.config.ferris_cycle_duration_minutes * 60  # Convert to seconds
            
            # Calculate phase as fraction of cycle
            phase = (timestamp % phase_period) / phase_period
            
            return phase
            
        except Exception:
            return 0.0
    
    def _calculate_ghost_signal(self, market_data: Dict[str, Any]) -> float:
        """Calculate ghost signal from market data."""
        try:
            # Simple ghost signal based on price movement
            price_change = market_data.get("price_change", 0.0)
            volatility = market_data.get("volatility", 0.0)
            
            # Ghost signal as normalized price change
            ghost_signal = np.tanh(price_change / 100.0) * (1.0 + volatility)
            
            return float(ghost_signal)
            
        except Exception:
            return 0.0
    
    async def _execute_trading_signal(self, signal: TradingSignal, market_data: Dict[str, Any]) -> None:
        """
        Execute a trading signal.
        
        Args:
            signal: Trading signal to execute
            market_data: Associated market data
        """
        start_time = time.time()
        
        try:
            if self.config.paper_trading:
                # Paper trading - simulate execution
                logger.info(f"📄 Paper trade: {signal.signal_type} {signal.asset} "
                          f"(confidence: {signal.confidence:.3f})")
                
                # Simulate trade execution
                await asyncio.sleep(0.1)  # Simulate execution time
                
                # Update metrics
                self.metrics.total_trades_executed += 1
                
            else:
                # Live trading - implement actual execution
                logger.warning("⚠️ Live trading not implemented - use paper trading mode")
                return
            
            # Update execution metrics
            execution_time = (time.time() - start_time) * 1000
            if self.metrics.total_trades_executed == 1:
                self.metrics.avg_trade_execution_time_ms = execution_time
            else:
                self.metrics.avg_trade_execution_time_ms = (
                    (self.metrics.avg_trade_execution_time_ms * (self.metrics.total_trades_executed - 1) + execution_time) /
                    self.metrics.total_trades_executed
                )
            
        except Exception as e:
            logger.error(f"❌ Error executing trading signal: {e}")
            self._log_error("trade_execution", str(e))
    
    async def _update_mathematical_states(self) -> None:
        """Update mathematical states with current data."""
        try:
            current_time = time.time()
            
            # Only update at specified intervals
            if (current_time - getattr(self, '_last_math_update', 0)) < self.config.mathematical_update_interval_seconds:
                return
            
            self._last_math_update = current_time
            
            # Get sample data for mathematical updates
            sample_price_data = [50000, 50100, 50200, 50300, 50400]
            sample_volume_data = [100, 120, 110, 130, 125]
            sample_time_series = list(range(len(sample_price_data)))
            
            # Update mathematical states through connectivity manager
            mathematical_states = await self.connectivity_manager.update_mathematical_states(
                price_data=sample_price_data,
                volume_data=sample_volume_data,
                time_series=sample_time_series,
                metadata={"source": "daemon", "timestamp": current_time}
            )
            
            # Update metrics
            self.metrics.mathematical_states_updated += 1
            
            logger.debug(f"Updated {len(mathematical_states)} mathematical states")
            
        except Exception as e:
            logger.error(f"❌ Error updating mathematical states: {e}")
            self._log_error("math_update", str(e))
    
    async def _perform_health_checks(self) -> None:
        """Perform health checks on all components."""
        try:
            current_time = time.time()
            
            # Only perform health checks at specified intervals
            if (current_time - self.last_health_check) < self.health_check_interval:
                return
            
            self.last_health_check = current_time
            
            # Get system status from connectivity manager
            system_status = await self.connectivity_manager.get_system_status()
            
            # Check overall system health
            api_status = system_status.get("api_status", {}).get("overall_status", "unknown")
            trading_status = system_status.get("trading_status", {}).get("overall_status", "unknown")
            visualization_status = system_status.get("visualization_status", {}).get("overall_status", "unknown")
            
            # Log health status
            logger.info(f"🏥 Health Check - API: {api_status}, Trading: {trading_status}, "
                       f"Visualization: {visualization_status}")
            
            # Check for critical issues
            if api_status == "error" or trading_status == "error":
                logger.error("🚨 Critical system health issues detected")
                self._log_error("health_check", "Critical system health issues")
            
        except Exception as e:
            logger.error(f"❌ Error performing health checks: {e}")
            self._log_error("health_check", str(e))
    
    async def _generate_performance_reports(self) -> None:
        """Generate and log performance reports."""
        try:
            current_time = time.time()
            
            # Only generate reports at specified intervals
            if (current_time - self.last_performance_report) < self.performance_report_interval:
                return
            
            self.last_performance_report = current_time
            
            # Get performance summary
            performance_summary = self.metrics.get_summary()
            
            # Log performance report
            logger.info("📊 Performance Report:")
            logger.info(f"  Uptime: {performance_summary['uptime_seconds']:.1f}s")
            logger.info(f"  Ticks Processed: {performance_summary['total_ticks_processed']}")
            logger.info(f"  Signals Generated: {performance_summary['total_signals_generated']}")
            logger.info(f"  Trades Executed: {performance_summary['total_trades_executed']}")
            logger.info(f"  Avg Processing Time: {performance_summary['avg_tick_processing_time_ms']:.2f}ms")
            logger.info(f"  Errors: {performance_summary['total_errors']}")
            logger.info(f"  Warnings: {performance_summary['total_warnings']}")
            
            # Log Ferris RDE status if available
            if self.ferris_rde:
                ferris_summary = self.ferris_rde.get_ferris_summary()
                logger.info(f"  Ferris Cycles: {ferris_summary.get('total_cycles', 0)}")
                logger.info(f"  Current Phase: {ferris_summary.get('current_phase', 'unknown')}")
            
        except Exception as e:
            logger.error(f"❌ Error generating performance report: {e}")
            self._log_error("performance_report", str(e))
    
    def _log_error(self, error_type: str, error_message: str) -> None:
        """Log error for monitoring."""
        try:
            self.metrics.total_errors += 1
            self.error_count += 1
            
            error_entry = {
                "timestamp": datetime.now().isoformat(),
                "type": error_type,
                "message": error_message,
                "daemon_uptime": self.metrics.get_uptime_seconds(),
            }
            
            self.error_log.append(error_entry)
            
            # Keep only recent errors
            if len(self.error_log) > self.config.max_error_count:
                self.error_log = self.error_log[-self.config.max_error_count:]
            
        except Exception as e:
            logger.error(f"❌ Error logging error: {e}")
    
    def get_daemon_status(self) -> Dict[str, Any]:
        """
        Get comprehensive daemon status.
        
        Returns:
            Dictionary with daemon status information
        """
        try:
            return {
                "daemon_info": {
                    "name": self.config.daemon_name,
                    "running": self.running,
                    "shutdown_requested": self.shutdown_requested,
                    "start_time": self.metrics.start_time.isoformat(),
                    "uptime_seconds": self.metrics.get_uptime_seconds(),
                },
                "configuration": {
                    "trading_enabled": self.config.trading_enabled,
                    "paper_trading": self.config.paper_trading,
                    "enable_gpu": self.config.enable_gpu,
                    "enable_distributed": self.config.enable_distributed,
                    "primary_assets": self.config.primary_assets,
                    "secondary_assets": self.config.secondary_assets,
                },
                "metrics": self.metrics.get_summary(),
                "components": {
                    "ferris_rde": self.ferris_rde is not None,
                    "ghost_stabilizer": self.ghost_stabilizer is not None,
                    "phase_monitor": self.phase_monitor is not None,
                    "dlt_engine": self.dlt_engine is not None,
                    "wallet_tracker": self.wallet_tracker is not None,
                },
                "error_summary": {
                    "total_errors": self.metrics.total_errors,
                    "recent_errors": self.error_log[-5:] if self.error_log else [],
                },
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting daemon status: {e}")
            return {"error": str(e)}


# Global daemon instance
_daemon_instance: Optional[FerrisRDEDaemon] = None


def get_daemon_instance() -> FerrisRDEDaemon:
    """
    Get the global daemon instance.
    
    Returns:
        Daemon instance
    """
    global _daemon_instance
    if _daemon_instance is None:
        _daemon_instance = FerrisRDEDaemon()
    return _daemon_instance


def signal_handler(signum: int, frame) -> None:
    """Handle shutdown signals."""
    logger.info(f"📡 Received signal {signum}, initiating shutdown...")
    
    daemon = get_daemon_instance()
    if daemon.running:
        asyncio.create_task(daemon.stop())


async def main() -> None:
    """Main function for running the daemon."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("ferris_rde_daemon.log"),
        ]
    )
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Get daemon instance
    daemon = get_daemon_instance()
    
    try:
        # Start the daemon
        success = await daemon.start()
        if not success:
            logger.error("❌ Failed to start daemon")
            return
        
        # Keep the daemon running
        while daemon.running and not daemon.shutdown_requested:
            await asyncio.sleep(1)
        
        logger.info("👋 Daemon main loop ended")
        
    except KeyboardInterrupt:
        logger.info("⌨️ Keyboard interrupt received")
    except Exception as e:
        logger.error(f"❌ Error in main: {e}")
    finally:
        # Ensure daemon is stopped
        if daemon.running:
            await daemon.stop()


if __name__ == "__main__":
    # Run the daemon
    asyncio.run(main()) 