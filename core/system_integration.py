# -*- coding: utf-8 -*-
"""
System Integration Module
========================

Coordinates all mathematical relay system components to ensure full connectivity
and proper data flow throughout the entire filebase.

Components Integrated:
- MathematicalBacklogManager (persistent logging)
- CCXTTradingExecutor (trading execution)
- MathematicalRelaySequencer (sequence management)
- MathematicalVisualizationAPI (real-time visualization)
- HistoricalDataManager (backtesting data)
- SimpleBacktester (strategy testing)

Features:
- Centralized system initialization
- Component lifecycle management
- Data flow coordination
- Error handling and recovery
- Cross-platform compatibility
"""

import asyncio
import logging
import threading
import time
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from backtesting.historical_data_manager import HistoricalDataManager
from backtesting.simple_backtester import SimpleBacktester
from core.ccxt_trading_executor import CCXTTradingExecutor, TradingPair

# Import core components
from core.mathematical_backlog_manager import MathematicalBacklogManager
from core.mathematical_relay_sequencer import MathematicalRelaySequencer
from core.visualization_api import MathematicalVisualizationAPI, start_visualization_server

logger = logging.getLogger(__name__)


class SystemIntegrationManager:
    """Manages integration and coordination of all system components."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        enable_visualization: bool = True,
        enable_backtesting: bool = True,
        enable_live_trading: bool = False,
    ):

        self.config = config or {}
        self.enable_visualization = enable_visualization
        self.enable_backtesting = enable_backtesting
        self.enable_live_trading = enable_live_trading

        # System state
        self.is_initialized = False
        self.is_running = False
        self.start_time = None

        # Core components
        self.backlog_manager: Optional[MathematicalBacklogManager] = None
        self.trading_executor: Optional[CCXTTradingExecutor] = None
        self.sequencer: Optional[MathematicalRelaySequencer] = None
        self.visualization_api: Optional[MathematicalVisualizationAPI] = None
        self.historical_data_manager: Optional[HistoricalDataManager] = None
        self.backtester: Optional[SimpleBacktester] = None

        # Threading
        self.system_thread = None
        self.coordination_lock = threading.RLock()

        # Performance tracking
        self.performance_metrics = {
            "total_sequences": 0,
            "total_trades": 0,
            "total_market_data_points": 0,
            "system_uptime": 0,
            "last_update": None,
        }

        logger.info("SystemIntegrationManager initialized")

    async def initialize_system(self) -> bool:
        """Initialize all system components in the correct order."""
        try:
            with self.coordination_lock:
                logger.info("Starting system initialization...")

                # Step 1: Initialize MathematicalBacklogManager (foundation)
                logger.info("Initializing MathematicalBacklogManager...")
                self.backlog_manager = MathematicalBacklogManager()
                self.backlog_manager.log_event(
                    "system_events",
                    {
                        "event": "system_initialization_started",
                        "timestamp": datetime.now().isoformat(),
                        "components": ["backlog_manager"],
                    },
                )

                # Step 2: Initialize MathematicalRelaySequencer
                logger.info("Initializing MathematicalRelaySequencer...")
                self.sequencer = MathematicalRelaySequencer(
                    mode=self.config.get("sequencer_mode", "demo"),
                    log_level=self.config.get("log_level", "INFO"),
                    gpu_enabled=self.config.get("gpu_enabled", False),
                )

                # Step 3: Initialize CCXTTradingExecutor
                logger.info("Initializing CCXTTradingExecutor...")
                trading_config = self.config.get("trading_config", {})
                self.trading_executor = CCXTTradingExecutor(trading_config)

                # Step 4: Initialize HistoricalDataManager (if backtesting enabled)
                if self.enable_backtesting:
                    logger.info("Initializing HistoricalDataManager...")
                    self.historical_data_manager = HistoricalDataManager(
                        start_date=datetime(2023, 1, 1), end_date=datetime(2023, 1, 31), interval_minutes=1440
                    )

                    # Initialize SimpleBacktester
                    self.backtester = SimpleBacktester(
                        initial_capital=Decimal("10000"),
                        start_date=datetime(2023, 1, 1),
                        end_date=datetime(2023, 1, 31),
                        trading_pair=TradingPair.BTC_USDC,
                    )

                # Step 5: Initialize Visualization API (if enabled)
                if self.enable_visualization:
                    logger.info("Initializing MathematicalVisualizationAPI...")
                    viz_config = self.config.get("visualization_config", {})
                    self.visualization_api = MathematicalVisualizationAPI(
                        host=viz_config.get("host", "0.0.0.0"),
                        port=viz_config.get("port", 8000),
                        static_dir=viz_config.get("static_dir", "static"),
                    )

                # Step 6: Log successful initialization
                self.backlog_manager.log_event(
                    "system_events",
                    {
                        "event": "system_initialization_completed",
                        "timestamp": datetime.now().isoformat(),
                        "components": [
                            "backlog_manager",
                            "sequencer",
                            "trading_executor",
                            "historical_data_manager" if self.enable_backtesting else None,
                            "backtester" if self.enable_backtesting else None,
                            "visualization_api" if self.enable_visualization else None,
                        ],
                    },
                )

                self.is_initialized = True
                self.start_time = datetime.now()

                logger.info("System initialization completed successfully")
                return True

        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            self.backlog_manager.log_event(
                "system_events",
                {"event": "system_initialization_failed", "timestamp": datetime.now().isoformat(), "error": str(e)},
            )
            return False

    async def start_system(self) -> bool:
        """Start all system components."""
        if not self.is_initialized:
            logger.error("System must be initialized before starting")
            return False

        try:
            with self.coordination_lock:
                logger.info("Starting system components...")

                # Start trading executor price monitoring
                if self.trading_executor:
                    logger.info("Starting trading executor price monitoring...")
                    self.trading_executor.start_price_monitoring()

                # Start visualization server in background thread
                if self.enable_visualization and self.visualization_api:
                    logger.info("Starting visualization server...")
                    self.system_thread = threading.Thread(target=self._start_visualization_server, daemon=True)
                    self.system_thread.start()

                # Start system coordination loop
                self.is_running = True
                asyncio.create_task(self._system_coordination_loop())

                self.backlog_manager.log_event(
                    "system_events", {"event": "system_started", "timestamp": datetime.now().isoformat()}
                )

                logger.info("System started successfully")
                return True

        except Exception as e:
            logger.error(f"System startup failed: {e}")
            self.backlog_manager.log_event(
                "system_events",
                {"event": "system_startup_failed", "timestamp": datetime.now().isoformat(), "error": str(e)},
            )
            return False

    def _start_visualization_server(self):
        """Start visualization server in background thread."""
        try:
            if self.visualization_api:
                self.visualization_api.start()
        except Exception as e:
            logger.error(f"Visualization server startup failed: {e}")

    async def _system_coordination_loop(self):
        """Main system coordination loop."""
        while self.is_running:
            try:
                # Update performance metrics
                await self._update_performance_metrics()

                # Generate sample data for demonstration
                await self._generate_sample_data()

                # Coordinate between components
                await self._coordinate_components()

                # Sleep for coordination interval
                await asyncio.sleep(5)  # Coordinate every 5 seconds

            except Exception as e:
                logger.error(f"Error in system coordination loop: {e}")
                await asyncio.sleep(10)  # Wait longer on error

    async def _update_performance_metrics(self):
        """Update system performance metrics."""
        try:
            if self.sequencer:
                stats = self.sequencer.get_sequencing_statistics()
                self.performance_metrics["total_sequences"] = stats.get("total_sequences", 0)

            if self.trading_executor:
                self.performance_metrics["total_trades"] = self.trading_executor.total_trades

            if self.backlog_manager:
                market_data = self.backlog_manager.retrieve_events("market_data", limit=1)
                self.performance_metrics["total_market_data_points"] = len(market_data)

            if self.start_time:
                self.performance_metrics["system_uptime"] = (datetime.now() - self.start_time).total_seconds()

            self.performance_metrics["last_update"] = datetime.now().isoformat()

        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")

    async def _generate_sample_data(self):
        """Generate sample data for demonstration purposes."""
        try:
            if not self.is_running:
                return

            # Generate sample BTC price hash sequence
            if self.sequencer:
                btc_price = 45000.0 + (time.time() % 1000)  # Simulate price variation
                result = self.sequencer.sequence_btc_price_hash(btc_price=btc_price, btc_volume=1000.0, phase=32)
                logger.debug(f"Generated BTC price hash sequence: {result.get('sequence_id')}")

            # Generate sample market data
            if self.trading_executor and self.backlog_manager:
                for pair in [TradingPair.BTC_USDC, TradingPair.ETH_USDC]:
                    price = 45000.0 + (time.time() % 1000) if "BTC" in pair.value else 3000.0 + (time.time() % 100)
                    self.trading_executor.price_data[pair] = Decimal(str(price))

                # Log market data
                for pair, price in self.trading_executor.price_data.items():
                    self.backlog_manager.log_event(
                        "market_data",
                        {"pair": pair.value, "price": str(price), "timestamp": datetime.now().isoformat()},
                    )

        except Exception as e:
            logger.error(f"Error generating sample data: {e}")

    async def _coordinate_components(self):
        """Coordinate data flow between components."""
        try:
            # Ensure backlog manager is receiving data from all components
            if self.backlog_manager and self.sequencer:
                # The sequencer already logs to backlog manager
                pass

            if self.backlog_manager and self.trading_executor:
                # The trading executor already logs to backlog manager
                pass

            # Log system health
            self.backlog_manager.log_event(
                "system_events",
                {
                    "event": "system_health_check",
                    "timestamp": datetime.now().isoformat(),
                    "performance_metrics": self.performance_metrics,
                    "components_status": {
                        "backlog_manager": "active" if self.backlog_manager else "inactive",
                        "sequencer": "active" if self.sequencer else "inactive",
                        "trading_executor": "active" if self.trading_executor else "inactive",
                        "visualization_api": "active" if self.visualization_api else "inactive",
                    },
                },
            )

        except Exception as e:
            logger.error(f"Error coordinating components: {e}")

    async def run_backtest(
        self,
        initial_capital: Decimal = Decimal("10000"),
        start_date: datetime = datetime(2023, 1, 1),
        end_date: datetime = datetime(2023, 1, 31),
        trading_pair: TradingPair = TradingPair.BTC_USDC,
    ) -> Dict[str, Any]:
        """Run a backtest using the integrated system."""
        if not self.is_initialized or not self.backtester:
            raise RuntimeError("System not initialized or backtesting not enabled")

        try:
            logger.info("Starting integrated backtest...")

            # Update backtester configuration
            self.backtester.initial_capital = initial_capital
            self.backtester.start_date = start_date
            self.backtester.end_date = end_date
            self.backtester.trading_pair = trading_pair

            # Run backtest
            await self.backtester.run_backtest(data_source="mock")

            # Log backtest completion
            self.backlog_manager.log_event(
                "system_events",
                {
                    "event": "backtest_completed",
                    "timestamp": datetime.now().isoformat(),
                    "initial_capital": str(initial_capital),
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "trading_pair": trading_pair.value,
                },
            )

            return {
                "success": True,
                "message": "Backtest completed successfully",
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            self.backlog_manager.log_event(
                "system_events", {"event": "backtest_failed", "timestamp": datetime.now().isoformat(), "error": str(e)}
            )
            return {"success": False, "error": str(e), "timestamp": datetime.now().isoformat()}

    async def stop_system(self) -> bool:
        """Stop all system components gracefully."""
        try:
            with self.coordination_lock:
                logger.info("Stopping system components...")

                self.is_running = False

                # Stop trading executor
                if self.trading_executor:
                    self.trading_executor.stop_price_monitoring()

                # Log system shutdown
                if self.backlog_manager:
                    self.backlog_manager.log_event(
                        "system_events",
                        {
                            "event": "system_stopped",
                            "timestamp": datetime.now().isoformat(),
                            "final_performance_metrics": self.performance_metrics,
                        },
                    )

                logger.info("System stopped successfully")
                return True

        except Exception as e:
            logger.error(f"System shutdown failed: {e}")
            return False

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "is_initialized": self.is_initialized,
            "is_running": self.is_running,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "performance_metrics": self.performance_metrics,
            "components": {
                "backlog_manager": "active" if self.backlog_manager else "inactive",
                "sequencer": "active" if self.sequencer else "inactive",
                "trading_executor": "active" if self.trading_executor else "inactive",
                "visualization_api": "active" if self.visualization_api else "inactive",
                "historical_data_manager": "active" if self.historical_data_manager else "inactive",
                "backtester": "active" if self.backtester else "inactive",
            },
            "configuration": {
                "enable_visualization": self.enable_visualization,
                "enable_backtesting": self.enable_backtesting,
                "enable_live_trading": self.enable_live_trading,
            },
        }

    def export_system_data(self, format_type: str = "json") -> Dict[str, Any]:
        """Export all system data for analysis."""
        try:
            export_data = {
                "system_status": self.get_system_status(),
                "performance_metrics": self.performance_metrics,
                "export_timestamp": datetime.now().isoformat(),
            }

            if self.backlog_manager:
                export_data["backlog_data"] = self.backlog_manager.export_all_data()

            if self.sequencer:
                export_data["sequencer_data"] = self.sequencer.export_sequencing_data()

            return export_data

        except Exception as e:
            logger.error(f"Error exporting system data: {e}")
            return {"error": str(e)}


# Global system integration manager
system_manager = None


async def initialize_and_start_system(config: Optional[Dict[str, Any]] = None) -> SystemIntegrationManager:
    """Initialize and start the complete mathematical relay system."""
    global system_manager

    try:
        # Create system manager
        system_manager = SystemIntegrationManager(config=config)

        # Initialize system
        if not await system_manager.initialize_system():
            raise RuntimeError("System initialization failed")

        # Start system
        if not await system_manager.start_system():
            raise RuntimeError("System startup failed")

        logger.info("Mathematical relay system initialized and started successfully")
        return system_manager

    except Exception as e:
        logger.error(f"Failed to initialize and start system: {e}")
        raise


async def run_demo_system():
    """Run a complete demo of the mathematical relay system."""
    try:
        # Configuration for demo
        config = {
            "sequencer_mode": "demo",
            "log_level": "INFO",
            "gpu_enabled": False,
            "visualization_config": {"host": "0.0.0.0", "port": 8000, "static_dir": "static"},
            "trading_config": {"unified_api_config": {"ccxt_config": {"timeout": 30000, "enableRateLimit": True}}},
        }

        # Initialize and start system
        manager = await initialize_and_start_system(config)

        # Run a quick backtest
        logger.info("Running demo backtest...")
        backtest_result = await manager.run_backtest(
            initial_capital=Decimal("10000"),
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 7),  # Short demo period
            trading_pair=TradingPair.BTC_USDC,
        )

        logger.info(f"Demo backtest result: {backtest_result}")

        # Keep system running for visualization
        logger.info("System running. Access dashboard at http://localhost:8000")
        logger.info("Press Ctrl+C to stop...")

        # Keep running
        while True:
            await asyncio.sleep(60)  # Check every minute

    except KeyboardInterrupt:
        logger.info("Received shutdown signal...")
        if system_manager:
            await system_manager.stop_system()
        logger.info("System shutdown complete")
    except Exception as e:
        logger.error(f"Demo system error: {e}")
        if system_manager:
            await system_manager.stop_system()


if __name__ == "__main__":
    # Run demo system
    asyncio.run(run_demo_system())
