#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schwabot Auto Trading System - Complete Integration.

Main auto trading system that integrates all components:
- Real-time market data streaming
- Advanced order book analysis
- Quantum mathematical analysis
- Smart order execution
- Advanced risk management
- Real-time strategy execution
"""

import asyncio
import json
import logging
import os
import signal
import sys
import time
from typing import Any, Dict, List, Optional

# Import configuration
from config.schwabot_config import load_config
from core.advanced_risk_manager import AdvancedRiskManager
from core.advanced_tensor_algebra import AdvancedTensorAlgebra
from core.clean_trading_pipeline import CleanTradingPipeline
from core.order_book_analyzer import OrderBookAnalyzer

# Import all core components
from core.real_time_execution_engine import RealTimeExecutionEngine, start_real_time_execution_engine
from core.real_time_market_data import RealTimeMarketDataStream
from core.smart_order_executor import SmartOrderExecutor
from core.zpe_zbe_core import create_zpe_zbe_core

logger = logging.getLogger(__name__)


class SchwabotAutoTradingSystem:
    """
    Complete Schwabot Auto Trading System with full integration.
    """

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the complete auto trading system."""
        self.config_path = config_path or "config/schwabot_config.json"
        self.config = self._load_system_config()

        # Core components
        self.execution_engine: Optional[RealTimeExecutionEngine] = None
        self.market_data_stream: Optional[RealTimeMarketDataStream] = None
        self.order_executor: Optional[SmartOrderExecutor] = None
        self.risk_manager: Optional[AdvancedRiskManager] = None
        self.order_book_analyzer: Optional[OrderBookAnalyzer] = None
        self.trading_pipeline: Optional[CleanTradingPipeline] = None

        # Quantum components
        self.zpe_zbe_core = create_zpe_zbe_core()
        self.tensor_algebra = AdvancedTensorAlgebra()

        # System state
        self.running = False
        self.initialized = False

        # Performance tracking
        self.start_time = time.time()
        self.system_metrics = {
            "uptime": 0.0,
            "total_signals": 0,
            "total_trades": 0,
            "total_pnl": 0.0,
            "system_health": "healthy",
        }

        # Setup logging
        self._setup_logging()

        logger.info("Schwabot Auto Trading System initialized")

    def _load_system_config(self) -> Dict[str, Any]:
        """Load system configuration."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
            else:
                config = self._get_default_config()

            logger.info("Loaded system configuration from %s", self.config_path)
            return config

        except Exception as e:
            logger.error("Failed to load configuration: %s", e)
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default system configuration."""
        return {
            "system": {
                "name": "Schwabot Auto Trading System",
                "version": "1.0.0",
                "mode": "paper_trading",  # paper_trading, live_trading
                "log_level": "INFO",
                "data_directory": "data/",
                "backup_directory": "backups/",
            },
            "exchanges": {
                "primary": ["binance", "coinbase"],
                "secondary": ["kraken", "kucoin"],
                "paper_trading": True,
            },
            "trading": {
                "symbols": ["BTC/USDT", "ETH/USDT", "ADA/USDT"],
                "base_capital": 10000.0,
                "max_positions": 5,
                "position_sizing": "kelly",
                "risk_management": {
                    "max_daily_loss": 0.05,
                    "max_drawdown": 0.15,
                    "max_position_size": 0.1,
                },
            },
            "analysis": {
                "enable_quantum": True,
                "enable_tensor": True,
                "enable_zpe_zbe": True,
                "enable_order_book": True,
                "enable_technical": True,
            },
            "execution": {
                "strategy": "balanced",
                "max_slippage": 0.001,
                "execution_timeout": 30.0,
                "enable_smart_routing": True,
            },
            "monitoring": {
                "update_interval": 1.0,
                "performance_logging": True,
                "health_checks": True,
                "alerting": True,
            },
        }

    def _setup_logging(self):
        """Setup system logging."""
        try:
            log_level = getattr(logging, self.config["system"]["log_level"])

            # Create logs directory if it doesn't exist
            os.makedirs("logs", exist_ok=True)

            # Configure logging
            logging.basicConfig(
                level=log_level,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(f"logs/schwabot_{int(time.time())}.log"),
                    logging.StreamHandler(sys.stdout),
                ],
            )

            logger.info("Logging configured successfully")

        except Exception as e:
            logger.error("Failed to setup logging: %s", e)

    async def initialize(self):
        """Initialize all system components."""
        try:
            logger.info("Initializing Schwabot Auto Trading System...")

            # Create data directories
            self._create_directories()

            # Initialize execution engine
            self.execution_engine = await start_real_time_execution_engine(self.config)

            # Initialize additional components
            self.market_data_stream = self.execution_engine.market_data_stream
            self.order_executor = self.execution_engine.order_executor
            self.risk_manager = self.execution_engine.risk_manager
            self.order_book_analyzer = self.execution_engine.order_book_analyzer
            self.trading_pipeline = self.execution_engine.trading_pipeline

            # Register callbacks
            self._register_callbacks()

            # Initialize quantum components
            await self._initialize_quantum_components()

            # Perform system health check
            await self._perform_health_check()

            self.initialized = True
            self.running = True

            logger.info("Schwabot Auto Trading System initialized successfully")

        except Exception as e:
            logger.error("Failed to initialize system: %s", e)
            raise

    def _create_directories(self):
        """Create necessary directories."""
        try:
            directories = [
                self.config["system"]["data_directory"],
                self.config["system"]["backup_directory"],
                "logs",
                "data/market_data",
                "data/trades",
                "data/performance",
            ]

            for directory in directories:
                os.makedirs(directory, exist_ok=True)

            logger.info("Created system directories")

        except Exception as e:
            logger.error("Failed to create directories: %s", e)

    def _register_callbacks(self):
        """Register system callbacks."""
        try:
            # Register execution callbacks
            self.execution_engine.register_execution_callback(self._on_trade_executed)
            self.execution_engine.register_signal_callback(self._on_signal_generated)

            # Register market data callbacks
            self.market_data_stream.register_callback(
                self.market_data_stream.DataType.TICKER, self._on_market_data_update
            )

            logger.info("Registered system callbacks")

        except Exception as e:
            logger.error("Failed to register callbacks: %s", e)

    async def _initialize_quantum_components(self):
        """Initialize quantum mathematical components."""
        try:
            # Initialize ZPE-ZBE core
            await self.zpe_zbe_core.initialize()

            # Initialize tensor algebra
            await self.tensor_algebra.initialize()

            logger.info("Quantum components initialized")

        except Exception as e:
            logger.error("Failed to initialize quantum components: %s", e)

    async def _perform_health_check(self):
        """Perform system health check."""
        try:
            health_status = {
                "execution_engine": self.execution_engine.running,
                "market_data_stream": self.market_data_stream.running,
                "order_executor": self.order_executor is not None,
                "risk_manager": self.risk_manager is not None,
                "quantum_components": True,  # Simplified check
            }

            all_healthy = all(health_status.values())

            if all_healthy:
                logger.info("System health check passed")
                self.system_metrics["system_health"] = "healthy"
            else:
                logger.warning("System health check failed: %s", health_status)
                self.system_metrics["system_health"] = "degraded"

        except Exception as e:
            logger.error("Health check failed: %s", e)
            self.system_metrics["system_health"] = "unhealthy"

    async def start(self):
        """Start the auto trading system."""
        try:
            if not self.initialized:
                await self.initialize()

            logger.info("Starting Schwabot Auto Trading System...")

            # Start monitoring tasks
            monitoring_tasks = [
                asyncio.create_task(self._system_monitoring_task()),
                asyncio.create_task(self._performance_tracking_task()),
                asyncio.create_task(self._health_monitoring_task()),
                asyncio.create_task(self._backup_task()),
            ]

            # Keep system running
            try:
                await asyncio.gather(*monitoring_tasks)
            except KeyboardInterrupt:
                logger.info("Received shutdown signal")
            finally:
                await self.stop()

        except Exception as e:
            logger.error("Failed to start system: %s", e)
            raise

    async def _system_monitoring_task(self):
        """Main system monitoring task."""
        while self.running:
            try:
                # Update system metrics
                self.system_metrics["uptime"] = time.time() - self.start_time

                # Get performance metrics
                if self.execution_engine:
                    performance = self.execution_engine.get_performance_summary()
                    self.system_metrics.update(
                        {
                            "total_signals": performance["total_signals"],
                            "total_trades": performance["successful_signals"],
                            "total_pnl": performance["total_pnl"],
                        }
                    )

                # Log system status
                if self.system_metrics["total_signals"] % 10 == 0:  # Every 10 signals
                    logger.info("System Status: %s", self._get_system_status())

                await asyncio.sleep(30)  # Update every 30 seconds

            except Exception as e:
                logger.error("System monitoring error: %s", e)
                await asyncio.sleep(30)

    async def _performance_tracking_task(self):
        """Track and log performance metrics."""
        while self.running:
            try:
                if self.execution_engine:
                    # Get detailed performance metrics
                    performance = self.execution_engine.get_performance_summary()

                    # Save performance data
                    await self._save_performance_data(performance)

                    # Log performance summary
                    logger.info("Performance: %s", self._format_performance_summary(performance))

                await asyncio.sleep(60)  # Update every minute

            except Exception as e:
                logger.error("Performance tracking error: %s", e)
                await asyncio.sleep(60)

    async def _health_monitoring_task(self):
        """Monitor system health."""
        while self.running:
            try:
                # Perform health check
                await self._perform_health_check()

                # Check for critical issues
                if self.system_metrics["system_health"] == "unhealthy":
                    logger.error("Critical system health issue detected")
                    await self._handle_critical_issue()

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error("Health monitoring error: %s", e)
                await asyncio.sleep(60)

    async def _backup_task(self):
        """Perform system backups."""
        while self.running:
            try:
                # Create backup of important data
                await self._create_system_backup()

                await asyncio.sleep(3600)  # Backup every hour

            except Exception as e:
                logger.error("Backup error: %s", e)
                await asyncio.sleep(3600)

    async def _on_trade_executed(self, execution_result):
        """Callback for trade execution."""
        try:
            logger.info(
                "Trade executed: %s %s %s (P&L: %.2f)",
                execution_result.signal.signal_type.value,
                execution_result.signal.quantity,
                execution_result.signal.symbol,
                execution_result.pnl,
            )

            # Update metrics
            self.system_metrics["total_trades"] += 1
            self.system_metrics["total_pnl"] += execution_result.pnl

            # Save trade data
            await self._save_trade_data(execution_result)

        except Exception as e:
            logger.error("Trade execution callback error: %s", e)

    async def _on_signal_generated(self, signal):
        """Callback for signal generation."""
        try:
            logger.debug(
                "Signal generated: %s %s (confidence: %.2f)",
                signal.signal_type.value,
                signal.symbol,
                signal.confidence,
            )

            # Update metrics
            self.system_metrics["total_signals"] += 1

        except Exception as e:
            logger.error("Signal generation callback error: %s", e)

    async def _on_market_data_update(self, market_event):
        """Callback for market data updates."""
        try:
            # Process market data updates
            symbol = market_event.symbol
            price = market_event.processed_data.get("price", 0.0)

            logger.debug("Market data update: %s @ %.2f", symbol, price)

        except Exception as e:
            logger.error("Market data callback error: %s", e)

    async def _save_performance_data(self, performance: Dict[str, Any]):
        """Save performance data to file."""
        try:
            timestamp = int(time.time())
            filename = f"data/performance/performance_{timestamp}.json"

            with open(filename, 'w') as f:
                json.dump(performance, f, indent=2)

        except Exception as e:
            logger.error("Failed to save performance data: %s", e)

    async def _save_trade_data(self, execution_result):
        """Save trade data to file."""
        try:
            timestamp = int(time.time())
            filename = f"data/trades/trade_{timestamp}.json"

            trade_data = {
                "timestamp": timestamp,
                "symbol": execution_result.signal.symbol,
                "side": execution_result.signal.signal_type.value,
                "quantity": execution_result.signal.quantity,
                "price": execution_result.signal.price,
                "pnl": execution_result.pnl,
                "success": execution_result.success,
            }

            with open(filename, 'w') as f:
                json.dump(trade_data, f, indent=2)

        except Exception as e:
            logger.error("Failed to save trade data: %s", e)

    async def _create_system_backup(self):
        """Create system backup."""
        try:
            timestamp = int(time.time())
            backup_dir = f"{self.config['system']['backup_directory']}backup_{timestamp}/"

            os.makedirs(backup_dir, exist_ok=True)

            # Backup configuration
            with open(f"{backup_dir}config.json", 'w') as f:
                json.dump(self.config, f, indent=2)

            # Backup performance data
            if self.execution_engine:
                performance = self.execution_engine.get_performance_summary()
                with open(f"{backup_dir}performance.json", 'w') as f:
                    json.dump(performance, f, indent=2)

            logger.info("System backup created: %s", backup_dir)

        except Exception as e:
            logger.error("Failed to create backup: %s", e)

    async def _handle_critical_issue(self):
        """Handle critical system issues."""
        try:
            logger.error("Handling critical system issue")

            # Stop trading
            if self.execution_engine:
                await self.execution_engine.stop()

            # Send alerts (would implement alerting system)
            logger.error("CRITICAL: System stopped due to health issues")

        except Exception as e:
            logger.error("Failed to handle critical issue: %s", e)

    def _get_system_status(self) -> str:
        """Get formatted system status."""
        uptime_hours = self.system_metrics["uptime"] / 3600
        return (
            f"Uptime: {uptime_hours:.1f}h, "
            f"Signals: {self.system_metrics['total_signals']}, "
            f"Trades: {self.system_metrics['total_trades']}, "
            f"P&L: ${self.system_metrics['total_pnl']:.2f}, "
            f"Health: {self.system_metrics['system_health']}"
        )

    def _format_performance_summary(self, performance: Dict[str, Any]) -> str:
        """Format performance summary."""
        return (
            f"Win Rate: {performance['win_rate']:.2%}, "
            f"Avg P&L: ${performance['average_pnl']:.2f}, "
            f"Max DD: {performance['max_drawdown']:.2%}, "
            f"Sharpe: {performance['sharpe_ratio']:.2f}"
        )

    async def stop(self):
        """Stop the auto trading system."""
        try:
            logger.info("Stopping Schwabot Auto Trading System...")

            self.running = False

            # Stop execution engine
            if self.execution_engine:
                await self.execution_engine.stop()

            # Create final backup
            await self._create_system_backup()

            # Log final statistics
            logger.info("Final Statistics: %s", self._get_system_status())

            logger.info("Schwabot Auto Trading System stopped")

        except Exception as e:
            logger.error("Failed to stop system: %s", e)

    def get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            "name": self.config["system"]["name"],
            "version": self.config["system"]["version"],
            "mode": self.config["system"]["mode"],
            "uptime": self.system_metrics["uptime"],
            "health": self.system_metrics["system_health"],
            "performance": self.system_metrics,
        }


async def main():
    """Main entry point for the Schwabot Auto Trading System."""
    try:
        # Create and start the system
        system = SchwabotAutoTradingSystem()

        # Setup signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            logger.info("Received shutdown signal")
            asyncio.create_task(system.stop())

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Start the system
        await system.start()

    except KeyboardInterrupt:
        logger.info("System interrupted by user")
    except Exception as e:
        logger.error("System error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    # Run the auto trading system
    asyncio.run(main())
