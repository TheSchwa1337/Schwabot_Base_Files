#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Trading System Entry Point

This script initializes and runs the complete entropy-enhanced trading system.
It orchestrates all components including entropy signal processing, strategy
bit mapping, profit calculation, risk management, and order execution.

Usage:
    python main_trading_system.py --config config/entropy_trading_system_config.yaml
    python main_trading_system.py --demo  # Run in demo mode
    python main_trading_system.py --backtest  # Run backtesting
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from core.clean_trading_pipeline import CleanTradingPipeline

# Core imports
from core.entropy_enhanced_trading_executor import EntropyEnhancedTradingExecutor, create_trading_executor
from core.entropy_signal_integration import EntropySignalIntegration
from core.pure_profit_calculator import PureProfitCalculator
from core.real_time_execution_engine import RealTimeExecutionEngine
from core.strategy_bit_mapper import StrategyBitMapper

# Utility imports
from utils.logging_setup import setup_logging
from utils.safe_print import safe_print

logger = logging.getLogger(__name__)


class TradingSystemManager:
    """
    Main trading system manager that orchestrates all components.
    """

    def __init__(self, config_path: str):
        """Initialize the trading system manager."""
        self.config_path = config_path
        self.config = self._load_config()
        self.executor = None
        self.is_running = False

        # Setup logging
        self._setup_logging()

        logger.info("🔄 Trading System Manager initialized")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)

            # Replace environment variables
            config = self._replace_env_vars(config)

            logger.info(f"✅ Configuration loaded from {self.config_path}")
            return config

        except Exception as e:
            logger.error(f"❌ Failed to load configuration: {e}")
            raise

    def _replace_env_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Replace environment variables in configuration."""
        import re

        def replace_in_value(value):
            if isinstance(value, str):
                # Replace ${VAR_NAME} with environment variable
                pattern = r'\$\{([^}]+)\}'
                matches = re.findall(pattern, value)

                for match in matches:
                    env_value = os.getenv(match)
                    if env_value is not None:
                        value = value.replace(f'${{{match}}}', env_value)
                    else:
                        logger.warning(f"⚠️ Environment variable {match} not found")

            return value

        def replace_in_dict(d):
            for key, value in d.items():
                if isinstance(value, dict):
                    d[key] = replace_in_dict(value)
                elif isinstance(value, list):
                    d[key] = [replace_in_value(v) for v in value]
                else:
                    d[key] = replace_in_value(value)
            return d

        return replace_in_dict(config)

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_config = self.config.get('logging', {})

        setup_logging(
            level=log_config.get('level', 'INFO'),
            file_logging=log_config.get('file_logging', True),
            log_file=log_config.get('log_file', './logs/trading_system.log'),
            max_file_size=log_config.get('max_file_size', 100),
            backup_count=log_config.get('backup_count', 5),
        )

    def _create_executor(self) -> EntropyEnhancedTradingExecutor:
        """Create the trading executor with configuration."""
        try:
            # Extract configuration sections
            exchange_config = self.config.get('exchange', {})
            strategy_config = self.config.get('strategy', {})
            entropy_config = self.config.get('entropy', {})
            risk_config = self.config.get('risk', {})

            # Create executor
            executor = create_trading_executor(
                exchange_config=exchange_config,
                strategy_config=strategy_config,
                entropy_config=entropy_config,
                risk_config=risk_config,
            )

            logger.info("✅ Trading executor created successfully")
            return executor

        except Exception as e:
            logger.error(f"❌ Failed to create trading executor: {e}")
            raise

    async def start_trading_system(self) -> None:
        """Start the complete trading system."""
        try:
            logger.info("🚀 Starting Entropy-Enhanced Trading System")

            # Create executor
            self.executor = self._create_executor()

            # Check if trading is enabled
            execution_config = self.config.get('execution', {})
            trading_loop_config = execution_config.get('trading_loop', {})

            if not trading_loop_config.get('enabled', True):
                logger.warning("⚠️ Trading loop is disabled in configuration")
                return

            # Get trading interval
            interval = trading_loop_config.get('interval', 60)

            # Start trading loop
            self.is_running = True
            await self.executor.run_trading_loop(interval_seconds=interval)

        except KeyboardInterrupt:
            logger.info("🛑 Trading system stopped by user")
        except Exception as e:
            logger.error(f"❌ Trading system error: {e}")
            raise
        finally:
            self.is_running = False

    async def run_demo_mode(self) -> None:
        """Run the trading system in demo mode."""
        try:
            logger.info("🎮 Starting Demo Mode")

            # Create executor
            self.executor = self._create_executor()

            # Run a few demo cycles
            for i in range(5):
                logger.info(f"🔄 Demo cycle {i + 1}/5")
                result = await self.executor.execute_trading_cycle()

                # Show results
                safe_print(f"Demo cycle {i + 1} result: {result.action.value}")
                if result.success:
                    safe_print(f"  Executed: {result.executed_quantity:.6f} BTC @ ${result.executed_price:,.2f}")
                else:
                    safe_print(f"  Status: {result.metadata.get('reason', 'Unknown')}")

                # Wait between cycles
                await asyncio.sleep(10)

            # Show final performance
            performance = self.executor.get_performance_summary()
            safe_print(f"\n📊 Final Performance: {performance}")

        except Exception as e:
            logger.error(f"❌ Demo mode error: {e}")
            raise

    async def run_backtest_mode(self) -> None:
        """Run the trading system in backtest mode."""
        try:
            logger.info("📊 Starting Backtest Mode")

            # Create executor
            self.executor = self._create_executor()

            # TODO: Implement backtesting functionality
            logger.info("🔄 Backtesting functionality not yet implemented")

        except Exception as e:
            logger.error(f"❌ Backtest mode error: {e}")
            raise

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        status = {
            'is_running': self.is_running,
            'config_loaded': self.config is not None,
            'executor_created': self.executor is not None,
        }

        if self.executor:
            status.update(self.executor.get_performance_summary())

        return status

    def stop_system(self) -> None:
        """Stop the trading system."""
        logger.info("🛑 Stopping trading system")
        self.is_running = False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Entropy-Enhanced Trading System')
    parser.add_argument(
        '--config',
        default='config/entropy_trading_system_config.yaml',
        help='Path to configuration file',
    )
    parser.add_argument('--demo', action='store_true', help='Run in demo mode')
    parser.add_argument('--backtest', action='store_true', help='Run in backtest mode')
    parser.add_argument('--status', action='store_true', help='Show system status')

    args = parser.parse_args()

    try:
        # Create system manager
        manager = TradingSystemManager(args.config)

        if args.status:
            # Show system status
            status = manager.get_system_status()
            safe_print("📊 System Status:")
            for key, value in status.items():
                safe_print(f"  {key}: {value}")
            return

        if args.demo:
            # Run demo mode
            asyncio.run(manager.run_demo_mode())
        elif args.backtest:
            # Run backtest mode
            asyncio.run(manager.run_backtest_mode())
        else:
            # Run normal trading system
            asyncio.run(manager.start_trading_system())

    except KeyboardInterrupt:
        safe_print("\n🛑 Trading system stopped by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        safe_print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
