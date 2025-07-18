#!/usr/bin/env python3
"""
Schwabot Main Integrated Entry Point
===================================

Complete integration of all Schwabot components including:
- Phantom Math system
- Algorithmic portfolio balancing
- BTC/USDC trading integration
- Real-time market data processing
- Web dashboard
- API server
"""

import argparse
import asyncio
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict

# Add core to path
sys.path.append(str(Path(__file__).parent / "core"))

    create_clean_trading_system,
    get_system_status,
    PORTFOLIO_BALANCER_AVAILABLE,
    BTC_USDC_INTEGRATION_AVAILABLE,
)
from core.phantom_band_navigator import PhantomBandNavigator
from core.phantom_detector import PhantomDetector
from core.phantom_logger import PhantomLogger
from core.phantom_registry import PhantomRegistry
from utils.logging_setup import setup_logging
from utils.safe_print import error, info, safe_print, success, warn

# Global variables for graceful shutdown
shutdown_event = asyncio.Event()
components = {}


class SchwabotIntegratedSystem:
    """Complete integrated Schwabot trading system."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.system = None
        self.phantom_detector = None
        self.phantom_registry = None
        self.phantom_logger = None
        self.phantom_navigator = None
        self.portfolio_balancer = None
        self.btc_usdc_integration = None

        # Trading state
        self.is_running = False
        self.market_data_task = None
        self.trading_task = None
        self.rebalancing_task = None

        # Performance tracking
        self.start_time = None
        self.performance_metrics = {}

    async def initialize(self) -> bool:
        """Initialize the complete Schwabot system."""
        try:
            info("🚀 Initializing Schwabot Integrated System...")

            # Check system status
            system_status = get_system_status()
            if not system_status["system_operational"]:
                error("❌ System components not available")
                return False

            # Create clean trading system
            initial_capital = self.config.get("initial_capital", 100000.0)
            self.system = create_clean_trading_system(initial_capital)

            # Initialize Phantom Math components
            self.phantom_detector = PhantomDetector()
            self.phantom_registry = PhantomRegistry()
            self.phantom_logger = PhantomLogger()
            self.phantom_navigator = PhantomBandNavigator()

            # Initialize portfolio balancer
            if PORTFOLIO_BALANCER_AVAILABLE:
                self.portfolio_balancer = self.system.get("portfolio_balancer")

            # Initialize BTC/USDC integration
            if BTC_USDC_INTEGRATION_AVAILABLE:
                self.btc_usdc_integration = self.system.get("btc_usdc_integration")
                await self.btc_usdc_integration.initialize()

            success("✅ Schwabot Integrated System initialized successfully")
            return True

        except Exception as e:
            error(f"❌ Error initializing Schwabot system: {e}")
            return False

    async def start(self) -> None:
        """Start the integrated trading system."""
        try:
            if not self.is_running:
                self.is_running = True
                self.start_time = time.time()

                info("🚀 Starting Schwabot Integrated Trading System...")

                # Start market data processing
                self.market_data_task = asyncio.create_task()
                    self._market_data_loop()
                )

                # Start trading loop
                self.trading_task = asyncio.create_task()
                    self._trading_loop()
                )

                # Start portfolio rebalancing loop
                if self.portfolio_balancer:
                    self.rebalancing_task = asyncio.create_task()
                        self._rebalancing_loop()
                    )

                success("✅ Schwabot Integrated System started successfully")

        except Exception as e:
            error(f"❌ Error starting Schwabot system: {e}")

    async def stop(self) -> None:
        """Stop the integrated trading system."""
        try:
            if self.is_running:
                self.is_running = False

                info("🛑 Stopping Schwabot Integrated System...")

                # Cancel tasks
                if self.market_data_task:
                    self.market_data_task.cancel()
                if self.trading_task:
                    self.trading_task.cancel()
                if self.rebalancing_task:
                    self.rebalancing_task.cancel()

                # Wait for tasks to complete
                await asyncio.gather()
                    self.market_data_task, 
                    self.trading_task, 
                    self.rebalancing_task,
                    return_exceptions=True
                )

                success("✅ Schwabot Integrated System stopped successfully")

        except Exception as e:
            error(f"❌ Error stopping Schwabot system: {e}")

    async def _market_data_loop(self) -> None:
        """Market data processing loop."""
        try:
            while self.is_running and not shutdown_event.is_set():
                # Simulate market data updates
                market_data = await self._get_market_data()

                # Process Phantom Math detection
                if self.phantom_detector:
                    phantom_zones = await self.phantom_detector.detect_phantom_zones()
                        market_data
                    )

                    # Log Phantom Zones
                    for zone in phantom_zones:
                        await self.phantom_logger.log_phantom_zone(zone)
                        await self.phantom_registry.register_phantom_zone(zone)

                # Update portfolio state
                if self.portfolio_balancer:
                    await self.portfolio_balancer.update_portfolio_state(market_data)

                await asyncio.sleep(1.0)  # 1 second interval

        except asyncio.CancelledError:
            info("Market data loop cancelled")
        except Exception as e:
            error(f"Error in market data loop: {e}")

    async def _trading_loop(self) -> None:
        """Main trading loop."""
        try:
            while self.is_running and not shutdown_event.is_set():
                # Get current market data
                market_data = await self._get_market_data()

                # Process BTC/USDC trading
                if self.btc_usdc_integration:
                    decision = await self.btc_usdc_integration.process_market_data(market_data)

                    if decision:
                        success = await self.btc_usdc_integration.execute_trade(decision)
                        if success:
                            info(f"Trade executed: {decision.symbol} {decision.action.value}")

                await asyncio.sleep(5.0)  # 5 second interval

        except asyncio.CancelledError:
            info("Trading loop cancelled")
        except Exception as e:
            error(f"Error in trading loop: {e}")

    async def _rebalancing_loop(self) -> None:
        """Portfolio rebalancing loop."""
        try:
            while self.is_running and not shutdown_event.is_set():
                if self.portfolio_balancer:
                    # Check if rebalancing is needed
                    needs_rebalancing = await self.portfolio_balancer.check_rebalancing_needs()

                    if needs_rebalancing:
                        info("🔄 Portfolio rebalancing needed")

                        # Get market data
                        market_data = await self._get_market_data()

                        # Generate rebalancing decisions
                        decisions = await self.portfolio_balancer.generate_rebalancing_decisions()
                            market_data
                        )

                        if decisions:
                            # Execute rebalancing
                            success = await self.portfolio_balancer.execute_rebalancing(decisions)
                            if success:
                                info(f"✅ Portfolio rebalancing completed: {len(decisions)} trades")
                            else:
                                warn("⚠️ Portfolio rebalancing failed")

                await asyncio.sleep(60.0)  # Check every minute

        except asyncio.CancelledError:
            info("Rebalancing loop cancelled")
        except Exception as e:
            error(f"Error in rebalancing loop: {e}")

    async def _get_market_data(self) -> Dict[str, Any]:
        """Get current market data."""
        try:
            # Simulate market data for BTC, ETH, SOL
            current_time = time.time()
            market_data = {}
                "BTC": {}
                    "price": 50000.0 + (current_time % 1000) * 0.1,  # Simulate price movement
                    "volume": 2000000,
                    "timestamp": current_time
                },
                "ETH": {}
                    "price": 3000.0 + (current_time % 500) * 0.1,
                    "volume": 1500000,
                    "timestamp": current_time
                },
                "SOL": {}
                    "price": 100.0 + (current_time % 200) * 0.01,
                    "volume": 800000,
                    "timestamp": current_time
                },
                "USDC": {}
                    "price": 1.0,
                    "volume": 5000000,
                    "timestamp": current_time
                }
            }
            return market_data

        except Exception as e:
            error(f"Error getting market data: {e}")
            return {}

    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics."""
        try:
            metrics = {}
                "system_status": {}
                    "running": self.is_running,
                    "uptime": time.time() - self.start_time if self.start_time else 0,
                    "components_available": get_system_status()
                },
                "trading_metrics": {},
                "portfolio_metrics": {},
                "phantom_metrics": {}
            }

            # Get trading metrics
            if self.btc_usdc_integration:
                metrics["trading_metrics"] = await self.btc_usdc_integration.get_performance_metrics()

            # Get portfolio metrics
            if self.portfolio_balancer:
                metrics["portfolio_metrics"] = await self.portfolio_balancer.get_portfolio_metrics()

            # Get Phantom Math metrics
            if self.phantom_registry:
                metrics["phantom_metrics"] = {}
                    "total_zones": await self.phantom_registry.get_total_zones(),
                    "recent_zones": await self.phantom_registry.get_recent_zones("BTC/USDC", hours=1)
                }

            return metrics

        except Exception as e:
            error(f"Error getting system metrics: {e}")
            return {}


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Schwabot Integrated Trading System")
    parser.add_argument("--config", type=str, default="config/trading_config.yaml",)
                        help="Configuration file path")
    parser.add_argument("--initial-capital", type=float, default=100000.0,)
                        help="Initial trading capital")
    parser.add_argument("--demo", action="store_true",)
                        help="Run in demo mode")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],)
                        default="INFO", help="Logging level")

    args = parser.parse_args()

    # Setup logging
    setup_logging(level=getattr(logging, args.log_level))

    # Configuration
    config = {}
        "initial_capital": args.initial_capital,
        "demo_mode": args.demo,
        "portfolio_config": {}
            "rebalancing_strategy": "phantom_adaptive",
            "rebalance_threshold": 0.5,
            "max_rebalance_frequency": 3600,
        },
        "btc_usdc_config": {}
            "symbol": "BTC/USDC",
            "base_order_size": 0.01,
            "max_order_size": 0.1,
            "enable_portfolio_balancing": True,
        },
        "exchange_config": {}
            "exchange": "binance",
            "sandbox": args.demo,
        }
    }

    # Create and initialize system
    schwabot = SchwabotIntegratedSystem(config)

    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        info(f"Received signal {signum}, shutting down gracefully...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Initialize system
        if not await schwabot.initialize():
            error("Failed to initialize Schwabot system")
            return 1

        # Start system
        await schwabot.start()

        # Main loop
        while not shutdown_event.is_set():
            await asyncio.sleep(1.0)

            # Print periodic status
            if int(time.time()) % 60 == 0:  # Every minute
                metrics = await schwabot.get_system_metrics()
                info(f"System Status: Running={metrics['system_status']['running']}, ")
                     f"Uptime={metrics['system_status']['uptime']:.0f}s")

    except KeyboardInterrupt:
        info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        error(f"Unexpected error: {e}")
        return 1
    finally:
        # Stop system
        await schwabot.stop()
        info("Schwabot system shutdown complete")

    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except Exception as e:
        error(f"Fatal error: {e}")
        sys.exit(1) 