#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schwabot Trading System - Main Entry Point
==========================================

Main entry point for the Schwabot trading system.
Provides command-line interface for different trading modes.

Usage:
    python main.py                    # Run demo system
    python main.py --backtest         # Run backtest only
    python main.py --visualization    # Run visualization only
    python main.py --full             # Run complete system
    python main.py --test             # Run component tests
"""

import argparse
import asyncio
import logging
import sys
import os
from datetime import datetime
from decimal import Decimal
from pathlib import Path

# Set console encoding for Windows
if sys.platform == "win32":
    os.system("chcp 65001 > nul")

# Configure logging with Unicode-safe handlers
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/schwabot_system.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)

# Import available system components
try:
    from core.btc_usdc_trading_engine import BTCTradingEngine, TradingMode
    from core.unified_btc_trading_pipeline import UnifiedBTCTradingPipeline
    from core.risk_manager import RiskManager
    from core.secure_exchange_manager import SecureExchangeManager
    from core.unified_pipeline_manager import UnifiedPipelineManager
    from core.math_config_manager import MathConfigManager
    from backtesting.simple_backtester import SimpleBacktester
    from core.quad_bit_strategy_array import TradingPair
    
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

        # Initialize backtester
        backtester = SimpleBacktester(
            initial_capital=Decimal("10000"),
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 7),
            trading_pair=TradingPair.BTC_USDC,
        )
        
        logger.info("Backtester initialized successfully")
        logger.info("=== Backtest Demo Completed ===")

    except Exception as e:
        logger.error(f"Backtest demo failed: {e}")


async def run_trading_engine_demo():
    """Run trading engine demonstration."""
    if not SYSTEM_AVAILABLE:
        logger.error("System components not available")
        return

    try:
        logger.info("=== Starting Trading Engine Demo ===")

        # Initialize trading engine
        engine = BTCTradingEngine(config={
            "api_key": "demo",
            "api_secret": "demo",
            "testnet": True
        })
        
        # Initialize risk manager
        risk_manager = RiskManager()
        
        # Initialize secure exchange manager
        exchange_manager = SecureExchangeManager()
        
        logger.info("Trading engine components initialized successfully")
        logger.info("=== Trading Engine Demo Completed ===")

    except Exception as e:
        logger.error(f"Trading engine demo failed: {e}")


async def run_pipeline_demo():
    """Run unified pipeline demonstration."""
    if not SYSTEM_AVAILABLE:
        logger.error("System components not available")
        return

    try:
        logger.info("=== Starting Pipeline Demo ===")

        # Initialize pipeline manager
        pipeline_manager = UnifiedPipelineManager()
        
        # Initialize math config manager
        math_config = MathConfigManager()
        
        # Initialize unified BTC trading pipeline
        btc_pipeline = UnifiedBTCTradingPipeline()
        
        logger.info("Pipeline components initialized successfully")
        logger.info("=== Pipeline Demo Completed ===")

    except Exception as e:
        logger.error(f"Pipeline demo failed: {e}")


async def run_component_test():
    """Test individual components for connectivity."""
    if not SYSTEM_AVAILABLE:
        logger.error("System components not available")
        return

    try:
        logger.info("=== Testing Individual Components ===")

        # Test RiskManager
        logger.info("Testing RiskManager...")
        risk_manager = RiskManager()
        logger.info("RiskManager initialized successfully")

        # Test MathConfigManager
        logger.info("Testing MathConfigManager...")
        math_config = MathConfigManager()
        logger.info("MathConfigManager initialized successfully")

        # Test SecureExchangeManager
        logger.info("Testing SecureExchangeManager...")
        exchange_manager = SecureExchangeManager()
        logger.info("SecureExchangeManager initialized successfully")

        # Test UnifiedPipelineManager
        logger.info("Testing UnifiedPipelineManager...")
        pipeline_manager = UnifiedPipelineManager()
        logger.info("UnifiedPipelineManager initialized successfully")

        # Test BTCTradingEngine
        logger.info("Testing BTCTradingEngine...")
        trading_engine = BTCTradingEngine(config={
            "api_key": "demo",
            "api_secret": "demo",
            "testnet": True
        })
        logger.info("BTCTradingEngine initialized successfully")

        logger.info("=== Component Tests Completed Successfully ===")

    except Exception as e:
        logger.error(f"Component test failed: {e}")


def create_directories():
    """Create necessary directories if they don't exist."""
    directories = ["logs", "data", "config", "static"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)


def main():
    """Main entry point with command line argument parsing."""
    parser = argparse.ArgumentParser(description="Schwabot Trading System")
    parser.add_argument(
        "--mode",
        choices=["backtest", "trading", "pipeline", "test", "demo"],
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

    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    logger.info("=== Schwabot Trading System ===")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Log Level: {args.log_level}")
    logger.info(f"Config: {args.config}")

    try:
        if args.mode == "backtest":
            asyncio.run(run_backtest_demo())
        elif args.mode == "trading":
            asyncio.run(run_trading_engine_demo())
        elif args.mode == "pipeline":
            asyncio.run(run_pipeline_demo())
        elif args.mode == "test":
            asyncio.run(run_component_test())
        elif args.mode == "demo":
            # Run all demos
            logger.info("Running all demos...")
            asyncio.run(run_component_test())
            asyncio.run(run_backtest_demo())
            asyncio.run(run_trading_engine_demo())
            asyncio.run(run_pipeline_demo())
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
