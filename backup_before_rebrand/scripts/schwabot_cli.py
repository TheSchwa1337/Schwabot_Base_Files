#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schwabot Command Line Interface.

Simple CLI to control the Schwabot trading system.
"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

from core.live_api_backtesting import LiveAPIBacktesting, LiveAPIConfig, create_live_api_backtesting

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('logs/schwabot_cli.log'), logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


class SchwabotCLI:
    """Command Line Interface for Schwabot trading system."""

    def __init__(self):
        """Initialize the CLI."""
        self.backtesting: Optional[LiveAPIBacktesting] = None
        self.config = self._load_config()

    def _load_config(self) -> LiveAPIConfig:
        """Load configuration from file or create default."""
        config_path = Path("config/schwabot_cli_config.json")

        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config_data = json.load(f)

                return LiveAPIConfig(
                    exchange=config_data.get("exchange", "binance"),
                    api_key=config_data.get("api_key", ""),
                    api_secret=config_data.get("api_secret", ""),
                    sandbox=config_data.get("sandbox", True),
                    symbols=config_data.get("symbols", ["BTC/USDC", "ETH/USDC"]),
                    update_interval=config_data.get("update_interval", 1.0),
                    enable_trading=config_data.get("enable_trading", False),
                    max_position_size=config_data.get("max_position_size", 0.1),
                    risk_management=config_data.get("risk_management", True),
                )
            except Exception as e:
                logger.error(f"Failed to load config: {e}")

        # Default configuration
        return LiveAPIConfig(
            exchange="binance",
            api_key="",
            api_secret="",
            sandbox=True,
            symbols=["BTC/USDC", "ETH/USDC"],
            update_interval=1.0,
            enable_trading=False,
            max_position_size=0.1,
            risk_management=True,
        )

    def _save_config(self):
        """Save current configuration."""
        config_path = Path("config/schwabot_cli_config.json")
        config_path.parent.mkdir(exist_ok=True)

        config_data = {
            "exchange": self.config.exchange,
            "api_key": self.config.api_key,
            "api_secret": self.config.api_secret,
            "sandbox": self.config.sandbox,
            "symbols": self.config.symbols,
            "update_interval": self.config.update_interval,
            "enable_trading": self.config.enable_trading,
            "max_position_size": self.config.max_position_size,
            "risk_management": self.config.risk_management,
        }

        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)

    async def start_system(self):
        """Start the Schwabot system."""
        if self.backtesting and self.backtesting.is_running:
            print("⚠️  System is already running!")
            return

        print("🚀 Starting Schwabot Trading System...")
        print(f"Exchange: {self.config.exchange}")
        print(f"Symbols: {', '.join(self.config.symbols)}")
        print(f"Trading enabled: {self.config.enable_trading}")
        print(f"Sandbox mode: {self.config.sandbox}")

        try:
            self.backtesting = create_live_api_backtesting(self.config)
            await self.backtesting.start()
        except Exception as e:
            logger.error(f"Failed to start system: {e}")
            print(f"❌ Failed to start system: {e}")

    async def stop_system(self):
        """Stop the Schwabot system."""
        if not self.backtesting or not self.backtesting.is_running:
            print("⚠️  System is not running!")
            return

        print("🛑 Stopping Schwabot Trading System...")

        try:
            await self.backtesting.stop()
            print("✅ System stopped successfully")
        except Exception as e:
            logger.error(f"Failed to stop system: {e}")
            print(f"❌ Failed to stop system: {e}")

    def enable_trading(self):
        """Enable live trading."""
        if not self.backtesting:
            print("⚠️  System is not running! Start the system first.")
            return

        self.backtesting.enable_trading()
        self.config.enable_trading = True
        self._save_config()
        print("🟢 LIVE TRADING ENABLED")

    def disable_trading(self):
        """Disable live trading."""
        if not self.backtesting:
            print("⚠️  System is not running! Start the system first.")
            return

        self.backtesting.disable_trading()
        self.config.enable_trading = False
        self._save_config()
        print("🔴 LIVE TRADING DISABLED")

    def show_status(self):
        """Show current system status."""
        if not self.backtesting:
            print("📊 System Status: NOT RUNNING")
            return

        status = self.backtesting.get_status()
        portfolio = self.backtesting.get_portfolio_summary()

        print("\n" + "=" * 50)
        print("📊 SCHWABOT SYSTEM STATUS")
        print("=" * 50)
        print(f"Running: {'🟢 YES' if status['is_running'] else '🔴 NO'}")
        print(f"Trading Enabled: {'🟢 YES' if status['is_trading_enabled'] else '🔴 NO'}")
        print(f"Exchange: {status['exchange']}")
        print(f"Symbols: {', '.join(status['symbols'])}")
        print(f"Uptime: {status['uptime']:.1f} seconds")
        print(f"Total Trades: {status['total_trades']}")
        print(f"Successful Trades: {status['successful_trades']}")
        print(f"Success Rate: {status['success_rate']:.1f}%")
        print(f"Total PnL: ${status['total_pnl']:.2f}")
        print(f"Registry Entries: {status['registry_entries']}")

        if portfolio:
            print(f"\n💰 PORTFOLIO SUMMARY")
            print(f"Total Value: ${portfolio.get('total_value', 0):.2f}")
            print(f"Total PnL: ${portfolio.get('total_pnl', 0):.2f}")
            print(f"Daily PnL: ${portfolio.get('daily_pnl', 0):.2f}")
            print(f"Win Rate: {portfolio.get('win_rate', 0):.1f}%")
            print(f"Max Drawdown: {portfolio.get('max_drawdown', 0):.1f}%")
            print(f"Risk Level: {portfolio.get('risk_level', 'unknown')}")
            print(f"Active Positions: {portfolio.get('active_positions', 0)}")

        print("=" * 50)

    def show_help(self):
        """Show help information."""
        print("\n" + "=" * 50)
        print("🤖 SCHWABOT CLI HELP")
        print("=" * 50)
        print("Commands:")
        print("  start           - Start the Schwabot system")
        print("  stop            - Stop the Schwabot system")
        print("  status          - Show current system status")
        print("  enable-trading  - Enable live trading")
        print("  disable-trading - Disable live trading")
        print("  config          - Show current configuration")
        print("  help            - Show this help message")
        print("  quit            - Exit the CLI")
        print("\nExamples:")
        print("  > start")
        print("  > enable-trading")
        print("  > status")
        print("  > disable-trading")
        print("  > stop")
        print("=" * 50)

    def show_config(self):
        """Show current configuration."""
        print("\n" + "=" * 50)
        print("⚙️  CURRENT CONFIGURATION")
        print("=" * 50)
        print(f"Exchange: {self.config.exchange}")
        print(f"API Key: {'*' * len(self.config.api_key) if self.config.api_key else 'NOT SET'}")
        print(f"API Secret: {'*' * len(self.config.api_secret) if self.config.api_secret else 'NOT SET'}")
        print(f"Sandbox Mode: {self.config.sandbox}")
        print(f"Symbols: {', '.join(self.config.symbols)}")
        print(f"Update Interval: {self.config.update_interval} seconds")
        print(f"Trading Enabled: {self.config.enable_trading}")
        print(f"Max Position Size: {self.config.max_position_size * 100}%")
        print(f"Risk Management: {self.config.risk_management}")
        print("=" * 50)

    async def run_interactive(self):
        """Run the interactive CLI."""
        print("\n" + "=" * 50)
        print("🤖 SCHWABOT TRADING SYSTEM CLI")
        print("=" * 50)
        print("Type 'help' for available commands")
        print("Type 'quit' to exit")
        print("=" * 50)

        while True:
            try:
                command = input("\nschwabot> ").strip().lower()

                if command == "quit" or command == "exit":
                    if self.backtesting and self.backtesting.is_running:
                        await self.stop_system()
                    print("👋 Goodbye!")
                    break

                elif command == "start":
                    await self.start_system()

                elif command == "stop":
                    await self.stop_system()

                elif command == "status":
                    self.show_status()

                elif command == "enable-trading":
                    self.enable_trading()

                elif command == "disable-trading":
                    self.disable_trading()

                elif command == "config":
                    self.show_config()

                elif command == "help":
                    self.show_help()

                elif command == "":
                    continue

                else:
                    print(f"❌ Unknown command: {command}")
                    print("Type 'help' for available commands")

            except KeyboardInterrupt:
                print("\n⚠️  Interrupted by user")
                if self.backtesting and self.backtesting.is_running:
                    await self.stop_system()
                break
            except Exception as e:
                logger.error(f"CLI error: {e}")
                print(f"❌ Error: {e}")


async def main():
    """Main entry point."""
    cli = SchwabotCLI()
    await cli.run_interactive()


if __name__ == "__main__":
    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)

    # Run the CLI
    asyncio.run(main())
