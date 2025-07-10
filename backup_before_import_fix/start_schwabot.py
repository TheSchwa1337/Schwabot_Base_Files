#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Start Script for Schwabot Auto Trading System.

Easy way to start the complete auto trading system with configuration options.
"""

import argparse
import asyncio
import logging
import os
import sys
from typing import Optional

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from schwabot_auto_trading_system import SchwabotAutoTradingSystem

logger = logging.getLogger(__name__)


def setup_argument_parser() -> argparse.ArgumentParser:
    """Setup command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Schwabot Auto Trading System - Quick Start",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start with default configuration (paper trading)
  python start_schwabot.py

  # Start with custom config file
  python start_schwabot.py --config config/custom_config.json

  # Start in live trading mode
  python start_schwabot.py --mode live

  # Start with specific symbols
  python start_schwabot.py --symbols BTC/USDT ETH/USDT

  # Start with debug logging
  python start_schwabot.py --log-level DEBUG

  # Start with specific capital
  python start_schwabot.py --capital 50000
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/schwabot_config.json",
        help="Path to configuration file (default: config/schwabot_config.json)"
    )
    
    parser.add_argument(
        "--mode",
        choices=["paper", "live"],
        default="paper",
        help="Trading mode: paper or live (default: paper)"
    )
    
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTC/USDT", "ETH/USDT"],
        help="Trading symbols (default: BTC/USDT ETH/USDT)"
    )
    
    parser.add_argument(
        "--capital",
        type=float,
        default=10000.0,
        help="Initial capital (default: 10000.0)"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--exchanges",
        nargs="+",
        default=["binance", "coinbase"],
        help="Trading exchanges (default: binance coinbase)"
    )
    
    parser.add_argument(
        "--max-positions",
        type=int,
        default=5,
        help="Maximum concurrent positions (default: 5)"
    )
    
    parser.add_argument(
        "--risk-level",
        choices=["conservative", "moderate", "aggressive"],
        default="moderate",
        help="Risk level (default: moderate)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in dry-run mode (no actual trades)"
    )
    
    return parser


def create_config_from_args(args) -> dict:
    """Create configuration dictionary from command line arguments."""
    config = {
        "system": {
            "name": "Schwabot Auto Trading System",
            "version": "1.0.0",
            "mode": "live_trading" if args.mode == "live" else "paper_trading",
            "log_level": args.log_level,
            "data_directory": "data/",
            "backup_directory": "backups/",
        },
        "exchanges": {
            "primary": args.exchanges,
            "secondary": ["kraken", "kucoin"],
            "paper_trading": args.mode == "paper",
        },
        "trading": {
            "symbols": args.symbols,
            "base_capital": args.capital,
            "max_positions": args.max_positions,
            "position_sizing": "kelly",
            "risk_management": get_risk_config(args.risk_level),
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
            "dry_run": args.dry_run,
        },
        "monitoring": {
            "update_interval": 1.0,
            "performance_logging": True,
            "health_checks": True,
            "alerting": True,
        },
    }
    
    return config


def get_risk_config(risk_level: str) -> dict:
    """Get risk configuration based on risk level."""
    risk_configs = {
        "conservative": {
            "max_daily_loss": 0.02,  # 2%
            "max_drawdown": 0.10,    # 10%
            "max_position_size": 0.05,  # 5%
        },
        "moderate": {
            "max_daily_loss": 0.05,  # 5%
            "max_drawdown": 0.15,    # 15%
            "max_position_size": 0.10,  # 10%
        },
        "aggressive": {
            "max_daily_loss": 0.10,  # 10%
            "max_drawdown": 0.25,    # 25%
            "max_position_size": 0.20,  # 20%
        },
    }
    
    return risk_configs.get(risk_level, risk_configs["moderate"])


def setup_logging(log_level: str):
    """Setup logging configuration."""
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"logs/schwabot_startup_{int(time.time())}.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )


def print_startup_banner(config: dict):
    """Print startup banner with system information."""
    print("\n" + "="*80)
    print("🚀 SCHWABOT AUTO TRADING SYSTEM")
    print("="*80)
    print(f"Version: {config['system']['version']}")
    print(f"Mode: {config['system']['mode']}")
    print(f"Capital: ${config['trading']['base_capital']:,.2f}")
    print(f"Symbols: {', '.join(config['trading']['symbols'])}")
    print(f"Exchanges: {', '.join(config['exchanges']['primary'])}")
    print(f"Max Positions: {config['trading']['max_positions']}")
    print(f"Risk Level: {config['trading']['risk_management']['max_daily_loss']*100:.1f}% daily loss limit")
    print("="*80)
    print("Starting system...\n")


def print_safety_warnings(mode: str):
    """Print safety warnings for live trading."""
    if mode == "live_trading":
        print("\n" + "!"*80)
        print("⚠️  LIVE TRADING MODE - REAL MONEY AT RISK ⚠️")
        print("!"*80)
        print("• You are about to start live trading with real funds")
        print("• Ensure you have tested thoroughly in paper trading mode")
        print("• Verify all risk management settings are appropriate")
        print("• Monitor the system closely during initial operation")
        print("• Press Ctrl+C to stop the system at any time")
        print("!"*80)
        
        # Ask for confirmation
        response = input("\nDo you want to continue with live trading? (yes/no): ")
        if response.lower() not in ["yes", "y"]:
            print("Live trading cancelled. Exiting...")
            sys.exit(0)


async def main():
    """Main entry point for the quick start script."""
    try:
        # Parse command line arguments
        parser = setup_argument_parser()
        args = parser.parse_args()
        
        # Setup logging
        setup_logging(args.log_level)
        
        # Create configuration
        config = create_config_from_args(args)
        
        # Print startup banner
        print_startup_banner(config)
        
        # Print safety warnings for live trading
        print_safety_warnings(config["system"]["mode"])
        
        # Create and start the system
        system = SchwabotAutoTradingSystem()
        system.config = config  # Override with command line config
        
        # Start the system
        await system.start()
        
    except KeyboardInterrupt:
        logger.info("System interrupted by user")
        print("\n🛑 System stopped by user")
    except Exception as e:
        logger.error("System error: %s", e)
        print(f"\n❌ System error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    import time
    asyncio.run(main()) 