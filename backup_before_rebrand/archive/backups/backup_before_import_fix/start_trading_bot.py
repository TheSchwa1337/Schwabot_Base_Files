#!/usr/bin/env python3
"""
Trading Bot Startup Script

This script starts the Schwabot live trading system with real API connections.
Configure your API keys in trading_bot_config.json before running.

Usage:
    # Execute single trade
    python start_trading_bot.py --mode trade --symbol BTCUSDT

    # Start automated trading
    python start_trading_bot.py --mode start-bot --interval 60

    # Check registry data
    python start_trading_bot.py --mode best-phase --asset BTC
"""

import os
import subprocess
import sys


def main():
    # Add current directory to path
    sys.path.insert(0, os.getcwd())

    # Run the CLI with configuration
    config_file = "trading_bot_config.json"

    if not os.path.exists(config_file):
        print(f"❌ Configuration file not found: {config_file}")
        print("Please create trading_bot_config.json with your API keys")
        return 1

    # Pass all arguments to the CLI
    cmd = [sys.executable, "-m", "core.cli_live_entry", "--config", config_file] + sys.argv[1:]

    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        return e.returncode
    except KeyboardInterrupt:
        print("\n🛑 Trading bot stopped")
        return 0

if __name__ == "__main__":
    exit(main()) 