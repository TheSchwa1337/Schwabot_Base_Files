from core.mathematical_backlog_manager import MathematicalBacklogManager
from core.mathematical_relay_sequencer import MathematicalRelaySequencer
from backtesting.simple_backtester import SimpleBacktester
from core.ccxt_trading_executor import TradingPair
from core.system_integration import SystemIntegrationManager, initialize_and_start_system
from datetime import datetime
from decimal import Decimal
import argparse
import asyncio
import logging
import sys

# -*- coding: utf-8 -*-
"""
Mathematical Relay System - Main Entry Point
===========================================

Main entry point for the complete mathematical relay trading system.
Demonstrates full connectivity across all components including:
- Real-time visualization dashboard
- Backtesting with historical data
- Mathematical sequence processing
- Trading execution simulation
- Persistent data logging

Usage:
    python main.py                    # Run demo system
    python main.py --backtest         # Run backtest only
    python main.py --visualization    # Run visualization only
    python main.py --full             # Run complete system
"""

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/mathematical_relay_system.log"),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)

# Import system components
try:
    from core.system_integration import (
        SystemIntegrationManager,
        initialize_and_start_system,
    )
    SYSTEM_AVAILABLE = True
except ImportError as e:
    logger.error(f"System components not available: {e}")
    SYSTEM_AVAILABLE = False


async def run_backtest_demo():
    """Run a comprehensive backtest demonstration."""
    if not SYSTEM_AVAILABLE:
        logger.error("System components not available")
        return

    try:
        logger.info("=== Starting Backtest Demo ===")

        # Initialize system with backtesting only
        config = {
            "sequencer_mode": "demo",
            "log_level": "INFO",
            "gpu_enabled": False,
            "enable_visualization": False,
            "enable_backtesting": True,
            "enable_live_trading": False,
        }
        manager = SystemIntegrationManager(config=config)
        await manager.initialize_system()

        # Run multiple backtests with different parameters
        backtest_scenarios = [
            {
                "name": "BTC/USDC - 1 Week",
                "initial_capital": Decimal("10000"),
                "start_date": datetime(2023, 1, 1),
                "end_date": datetime(2023, 1, 7),
                "trading_pair": TradingPair.BTC_USDC,
            },
            {
                "name": "ETH/USDC - 2 Weeks",
                "initial_capital": Decimal("15000"),
                "start_date": datetime(2023, 1, 1),
                "end_date": datetime(2023, 1, 14),
                "trading_pair": TradingPair.ETH_USDC,
            },
            {
                "name": "XRP/USDC - 1 Month",
                "initial_capital": Decimal("5000"),
                "start_date": datetime(2023, 1, 1),
                "end_date": datetime(2023, 1, 31),
                "trading_pair": TradingPair.XRP_USDC,
            },
        ]
        for scenario in backtest_scenarios:
            logger.info(f"Running backtest: {scenario['name']}")
            result = await manager.run_backtest(
                initial_capital=scenario["initial_capital"],
                start_date=scenario["start_date"],
                end_date=scenario["end_date"],
                trading_pair=scenario["trading_pair"],
            )
            logger.info(f"Backtest result: {result}")

        # Export system data
        manager.export_system_data()
        logger.info("System data exported successfully")

        logger.info("=== Backtest Demo Completed ===")

    except Exception as e:
        logger.error(f"Backtest demo failed: {e}")


async def run_visualization_demo():
    """Run visualization dashboard demonstration."""
    if not SYSTEM_AVAILABLE:
        logger.error("System components not available")
        return

    try:
        logger.info("=== Starting Visualization Demo ===")

        # Initialize system with visualization only
        config = {
            "sequencer_mode": "demo",
            "log_level": "INFO",
            "gpu_enabled": False,
            "visualization_config": {
                "host": "0.0.0.0",
                "port": 8000,
                "static_dir": "static",
            },
            "enable_visualization": True,
            "enable_backtesting": False,
            "enable_live_trading": False,
        }
        manager = await initialize_and_start_system(config)

        logger.info("Visualization dashboard started at http://localhost:8000")
        logger.info("Press Ctrl+C to stop...")

        # Keep running for visualization
        while True:
            await asyncio.sleep(60)

    except KeyboardInterrupt:
        logger.info("Received shutdown signal...")
        if manager:
            await manager.stop_system()
        logger.info("Visualization demo stopped")
    except Exception as e:
        logger.error(f"Visualization demo failed: {e}")


async def run_full_system_demo():
    """Run complete system demonstration with all components."""
    if not SYSTEM_AVAILABLE:
        logger.error("System components not available")
        return

    try:
        logger.info("=== Starting Full System Demo ===")

        # Configuration for full system
        config = {
            "sequencer_mode": "demo",
            "log_level": "INFO",
            "gpu_enabled": False,
            "visualization_config": {
                "host": "0.0.0.0",
                "port": 8000,
                "static_dir": "static",
            },
            "trading_config": {
                "unified_api_config": {
                    "ccxt_config": {"timeout": 30000, "enableRateLimit": True}
                }
            },
            "enable_visualization": True,
            "enable_backtesting": True,
            "enable_live_trading": False,
        }
        # Initialize and start complete system
        manager = await initialize_and_start_system(config)

        # Run a quick backtest to generate data
        logger.info("Running initial backtest to generate data...")
        backtest_result = await manager.run_backtest(
            initial_capital=Decimal("10000"),
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 7),
            trading_pair=TradingPair.BTC_USDC,
        )
        logger.info(f"Initial backtest completed: {backtest_result}")

        # Display system status
        status = manager.get_system_status()
        logger.info("=== System Status ===")
        logger.info(f"Initialized: {status['is_initialized']}")
        logger.info(f"Running: {status['is_running']}")
        logger.info(f"Components: {status['components']}")
        logger.info(f"Performance: {status['performance_metrics']}")

        logger.info("=== Full System Demo Running ===")
        logger.info("Access dashboard at http://localhost:8000")
        logger.info("Press Ctrl+C to stop...")

        # Keep system running
        while True:
            await asyncio.sleep(60)

            # Log periodic status
            current_status = manager.get_system_status()
            logger.info(f"System uptime: {current_status['performance_metrics']['system_uptime']:.0f}s")

    except KeyboardInterrupt:
        logger.info("Received shutdown signal...")
        if manager:
            await manager.stop_system()
        logger.info("Full system demo stopped")
    except Exception as e:
        logger.error(f"Full system demo failed: {e}")


async def run_component_test():
    """Test individual components for connectivity."""
    if not SYSTEM_AVAILABLE:
        logger.error("System components not available")
        return

    try:
        logger.info("=== Testing Individual Components ===")

        # Test MathematicalBacklogManager
        logger.info("Testing MathematicalBacklogManager...")
        backlog_manager = MathematicalBacklogManager()
        backlog_manager.log_event(
            "test_events",
            {"test": "backlog_manager", "timestamp": datetime.now().isoformat()},
        )
        events = backlog_manager.retrieve_events("test_events", limit=10)
        logger.info(f"Backlog manager test: {len(events)} events retrieved")

        # Test MathematicalRelaySequencer
        logger.info("Testing MathematicalRelaySequencer...")
        sequencer = MathematicalRelaySequencer(mode="demo", log_level="INFO")
        result = sequencer.sequence_btc_price_hash(
            btc_price=45000.0, btc_volume=1000.0, phase=32
        )
        logger.info(f"Sequencer test: {result.get('sequence_id')}")

        # Test SimpleBacktester
        logger.info("Testing SimpleBacktester...")
        SimpleBacktester(
            initial_capital=Decimal("1000"),
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 3),
            trading_pair=TradingPair.BTC_USDC,
        )
        logger.info("Backtester initialized successfully")

        logger.info("=== Component Tests Completed ===")

    except Exception as e:
        logger.error(f"Component test failed: {e}")


def main():
    """Main entry point with command line argument parsing."""
    parser = argparse.ArgumentParser(description="Mathematical Relay Trading System")
    parser.add_argument(
        "--mode",
        choices=["backtest", "visualization", "full", "test"],
        default="full",
        help="System mode to run",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Visualization server port"
    )

    args = parser.parse_args()

    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    logger.info("=== Mathematical Relay Trading System ===")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Log Level: {args.log_level}")
    logger.info(f"Port: {args.port}")

    try:
        if args.mode == "backtest":
            asyncio.run(run_backtest_demo())
        elif args.mode == "visualization":
            asyncio.run(run_visualization_demo())
        elif args.mode == "full":
            asyncio.run(run_full_system_demo())
        elif args.mode == "test":
            asyncio.run(run_component_test())
        else:
            logger.error(f"Unknown mode: {args.mode}")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("System interrupted by user")
    except Exception as e:
        logger.error(f"System error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
