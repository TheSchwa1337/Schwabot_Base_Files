#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schwabot Trading System - Main Entry Point
==========================================

This is the main entry point for the Schwabot trading system.
It provides a command-line interface for running the system in different modes.
"""

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path
from typing import Optional

# Core system imports
from core.schwabot_core_system import SchwabotCoreSystem, run_system, get_system_instance
from utils.logging_setup import setup_logging

# Setup logging
logger = logging.getLogger(__name__)


def create_directories():
    """Create necessary directories if they don't exist."""
    directories = ["logs", "data", "config", "static", "backups"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)


async def run_demo_mode():
    """Run the system in demo mode with all components."""
    logger.info("=== Starting Schwabot Demo Mode ===")
    
    try:
        # Create and initialize the core system
        system = SchwabotCoreSystem("config/schwabot_config.yaml")
        
        # Initialize the system
        if not await system.initialize():
            logger.error("Failed to initialize system")
            return False
        
        # Get system status
        status = system.get_system_status()
        logger.info(f"System Status: {status}")
        
        # Start the system
        if not await system.start():
            logger.error("Failed to start system")
            return False
        
        logger.info("System started successfully in demo mode")
        logger.info("Press Ctrl+C to stop the system")
        
        # Run for a limited time in demo mode
        try:
            await asyncio.sleep(30)  # Run for 30 seconds in demo mode
        except asyncio.CancelledError:
            pass
        
        # Stop the system
        await system.stop()
        
        logger.info("=== Demo Mode Completed ===")
        return True
        
    except Exception as e:
        logger.error(f"Demo mode failed: {e}")
        return False


async def run_live_trading_mode():
    """Run the system in live trading mode."""
    logger.info("=== Starting Schwabot Live Trading Mode ===")
    
    try:
        # Create and initialize the core system
        system = SchwabotCoreSystem("config/schwabot_config.yaml")
        
        # Initialize the system
        if not await system.initialize():
            logger.error("Failed to initialize system")
            return False
        
        # Start the system
        if not await system.start():
            logger.error("Failed to start system")
            return False
        
        logger.info("System started successfully in live trading mode")
        logger.info("Press Ctrl+C to stop the system")
        
        # Run the main trading loop
        await system.run_trading_loop()
        
        return True
        
    except Exception as e:
        logger.error(f"Live trading mode failed: {e}")
        return False


async def run_backtest_mode():
    """Run the system in backtest mode."""
    logger.info("=== Starting Schwabot Backtest Mode ===")
    
    try:
        # Import backtesting components
        from test.simple_trading_test import SimpleBacktester
        from core.type_defs import TradingPair
        from datetime import datetime
        from decimal import Decimal
        
        # Initialize backtester
        backtester = SimpleBacktester(
            initial_capital=Decimal("10000"),
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 7),
            trading_pair=TradingPair.BTC_USDC,
        )
        
        # Run backtest
        results = await backtester.run_backtest()
        
        logger.info("Backtest completed successfully")
        logger.info(f"Results: {results}")
        
        return True
        
    except Exception as e:
        logger.error(f"Backtest mode failed: {e}")
        return False


async def run_test_mode():
    """Run the system in test mode."""
    logger.info("=== Starting Schwabot Test Mode ===")
    
    try:
        # Create and initialize the core system
        system = SchwabotCoreSystem("config/schwabot_config.yaml")
        
        # Initialize the system
        if not await system.initialize():
            logger.error("Failed to initialize system")
            return False
        
        # Test individual components
        logger.info("Testing individual components...")
        
        # Test trading engine
        if system.trading_engine:
            logger.info("✅ Trading engine initialized")
        else:
            logger.error("❌ Trading engine not initialized")
        
        # Test risk manager
        if system.risk_manager:
            logger.info("✅ Risk manager initialized")
        else:
            logger.error("❌ Risk manager not initialized")
        
        # Test exchange manager
        if system.exchange_manager:
            logger.info("✅ Exchange manager initialized")
        else:
            logger.error("❌ Exchange manager not initialized")
        
        # Test pipeline manager
        if system.pipeline_manager:
            logger.info("✅ Pipeline manager initialized")
        else:
            logger.error("❌ Pipeline manager not initialized")
        
        # Test math core
        if system.math_core:
            logger.info("✅ Math core initialized")
        else:
            logger.error("❌ Math core not initialized")
        
        # Test market data
        if system.market_data:
            logger.info("✅ Market data initialized")
        else:
            logger.error("❌ Market data not initialized")
        
        # Test execution engine
        if system.execution_engine:
            logger.info("✅ Execution engine initialized")
        else:
            logger.error("❌ Execution engine not initialized")
        
        # Test portfolio tracker
        if system.portfolio_tracker:
            logger.info("✅ Portfolio tracker initialized")
        else:
            logger.error("❌ Portfolio tracker not initialized")
        
        # Test strategy components
        if system.strategy_loader:
            logger.info("✅ Strategy loader initialized")
        else:
            logger.error("❌ Strategy loader not initialized")
        
        if system.strategy_executor:
            logger.info("✅ Strategy executor initialized")
        else:
            logger.error("❌ Strategy executor not initialized")
        
        # Get system status
        status = system.get_system_status()
        logger.info(f"System Status: {status}")
        
        logger.info("=== Test Mode Completed ===")
        return True
        
    except Exception as e:
        logger.error(f"Test mode failed: {e}")
        return False


async def run_api_mode():
    """Run the system in API mode."""
    logger.info("=== Starting Schwabot API Mode ===")
    
    try:
        # Import API components
        from core.api.integration_manager import IntegrationManager
        
        # Create and initialize the core system
        system = SchwabotCoreSystem("config/schwabot_config.yaml")
        
        # Initialize the system
        if not await system.initialize():
            logger.error("Failed to initialize system")
            return False
        
        # Initialize API manager
        api_manager = IntegrationManager()
        await api_manager.initialize()
        
        # Start the system
        if not await system.start():
            logger.error("Failed to start system")
            return False
        
        logger.info("System started successfully in API mode")
        logger.info("API server running on http://localhost:5000")
        logger.info("Press Ctrl+C to stop the system")
        
        # Start API server
        await api_manager.start_server()
        
        return True
        
    except Exception as e:
        logger.error(f"API mode failed: {e}")
        return False


def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def main():
    """Main entry point with command line argument parsing."""
    parser = argparse.ArgumentParser(description="Schwabot Trading System")
    parser.add_argument(
        "--mode",
        choices=["demo", "live", "backtest", "test", "api"],
        default="demo",
        help="System mode to run",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/schwabot_config.yaml",
        help="Configuration file path"
    )

    args = parser.parse_args()

    # Create necessary directories
    create_directories()

    # Setup logging
    setup_logging(
        level=args.log_level,
        log_file=f"logs/schwabot_{args.mode}.log"
    )

    # Setup signal handlers
    setup_signal_handlers()

    logger.info("=== Schwabot Trading System ===")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Log Level: {args.log_level}")
    logger.info(f"Config: {args.config}")

    try:
        # Run the appropriate mode
        if args.mode == "demo":
            success = asyncio.run(run_demo_mode())
        elif args.mode == "live":
            success = asyncio.run(run_live_trading_mode())
        elif args.mode == "backtest":
            success = asyncio.run(run_backtest_mode())
        elif args.mode == "test":
            success = asyncio.run(run_test_mode())
        elif args.mode == "api":
            success = asyncio.run(run_api_mode())
        else:
            logger.error(f"Unknown mode: {args.mode}")
            sys.exit(1)

        if success:
            logger.info("System completed successfully")
            sys.exit(0)
        else:
            logger.error("System failed")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("System interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"System error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
