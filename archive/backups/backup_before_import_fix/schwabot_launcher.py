#!/usr/bin/env python3
"""
Schwabot Launcher
=================

The master launch script for the Schwabot Trading System.
This script initializes and runs the entire trading pipeline.

Usage:
  - For paper trading:
    python schwabot_launcher.py --mode paper --pair BTC/USDT

  - For live trading (use with extreme, caution):
    python schwabot_launcher.py --mode live --pair BTC/USDT

  - For backtesting (feature to be, implemented):
    python schwabot_launcher.py --mode backtest --pair BTC/USDT --days 30
"""

import argparse
import logging
import sys
import time
from typing import Any, Dict

import yaml

from core import CleanTradingSystem, create_clean_trading_system
from utils.logging_setup import setup_logging
from utils.secure_config_manager import SecureConfigManager

logger = logging.getLogger(__name__)

def load_config(config_path: str = 'config.yml') -> Dict[str, Any]:
    """Loads the YAML configuration file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"✅ Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"❌ Configuration file not found at {config_path}.")
        logger.error("Please copy 'config.yml.template' to 'config.yml' and fill in your details.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Error loading configuration: {e}")
        sys.exit(1)

def run_trading_loop(system: CleanTradingSystem, pair: str, mode: str):
    """
    Main trading loop.
    This is where the bot will continuously run, fetch data, and execute trades.
    """
    logger.info(f"🚀 Starting Schwabot trading loop for {pair} in {mode.upper()} mode.")
    logger.info("Press Ctrl+C to stop the bot.")

    try:
        while True:
            # 1. Fetch market data
            logger.debug("Fetching market data...")
            market_data = system.market_data_pipeline.get_latest_market_data(pair)

            # 2. Process data and generate signals
            logger.debug("Processing data and generating signals...")
            # This is where your strategy logic would be called.
            # For now, we'll simulate a placeholder for the main cycle.'
            system.run_main_cycle()

            # 3. Print status or perform actions
            logger.info(f"Heartbeat: System running for {pair}. Last price: {market_data.get('last', 'N/A')}")

            # Sleep for a defined interval (e.g., 5 seconds) before the next cycle
            time.sleep(5)

    except KeyboardInterrupt:
        logger.info("🛑 User interrupted the trading loop. Shutting down gracefully.")
    except Exception as e:
        logger.error(f"❌ An unexpected error occurred in the trading loop: {e}", exc_info=True)
    finally:
        logger.info("🧹 Cleaning up resources...")
        system.cleanup()
        logger.info("✅ System shut down complete.")

def main():
    """Main function to launch the Schwabot."""
    parser = argparse.ArgumentParser(description="Schwabot Trading System Launcher")

    # Load config to get default values
    config = load_config()

    parser.add_argument('--mode', type=str, default=config['trading']['default_mode'],)
                        choices=['live', 'paper', 'backtest'],
                        help="Trading mode: 'live', 'paper', or 'backtest'.")
    parser.add_argument('--pair', type=str, default=config['trading']['default_pair'],)
                        help="The trading pair to use, e.g., 'BTC/USDT'.")
    parser.add_argument('--config', type=str, default='config.yml',)
                        help="Path to the configuration file.")

    args = parser.parse_args()

    # Setup logging based on config
    setup_logging(level=config['system']['log_level'], log_file=config['system']['log_file'])

    logger.info("--- Schwabot Trading System Initializing ---")

    # Initialize the trading system
    logger.info("Initializing CleanTradingSystem...")
    trading_system = create_clean_trading_system()
        config=config,
        enable_gpu=config['system']['enable_gpu']
    )

    if not trading_system:
        logger.error("❌ Failed to initialize the trading system. Exiting.")
        sys.exit(1)

    logger.info("✅ Trading system initialized successfully.")

    # Run the appropriate loop based on mode
    if args.mode in ['live', 'paper']:
        run_trading_loop(trading_system, args.pair, args.mode)
    elif args.mode == 'backtest':
        logger.warning("Backtesting mode is not yet fully implemented.")
        # Here you would call your backtesting function
        # backtester.run(trading_system, args.pair, ...)

    logger.info("--- Schwabot has shut down ---")

if __name__ == "__main__":
    main() 