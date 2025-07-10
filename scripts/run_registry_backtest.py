#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Registry Backtest CLI for Schwabot
==================================

Command-line interface for running registry-driven backtests.
Provides comprehensive options for testing hash-based strategies
with Schwabot's mathematical framework.

Usage:
    python scripts/run_registry_backtest.py --hashes registry/hash_list.json --pair BTC_USDC --days 30
    python scripts/run_registry_backtest.py --all-strategies --pair BTC_USDC --start-date 2024-01-01 --end-date 2024-01-31
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import Schwabot components
try:
    from core.profit_feedback_engine import ProfitFeedbackEngine
    from core.quad_bit_strategy_array import TradingPair
    from core.registry_backtester import BacktestConfig, RegistryBacktester
    
    SCHWABOT_AVAILABLE = True
except ImportError as e:
    SCHWABOT_AVAILABLE = False
    print(f"❌ Schwabot components not available: {e}")
    sys.exit(1)


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('backtest.log')
        ]
    )


def load_hash_list(hash_file: str) -> List[str]:
    """Load hash list from file."""
    try:
        with open(hash_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and 'hashes' in data:
            return data['hashes']
        else:
            raise ValueError("Invalid hash file format")
            
    except Exception as e:
        print(f"❌ Error loading hash file {hash_file}: {e}")
        sys.exit(1)


def load_all_strategies(registry_path: str) -> List[str]:
    """Load all strategy hashes from registry."""
    try:
        with open(registry_path, 'r', encoding='utf-8') as f:
            registry = json.load(f)
        
        return list(registry.keys())
        
    except Exception as e:
        print(f"❌ Error loading registry {registry_path}: {e}")
        sys.exit(1)


def parse_date(date_str: str) -> datetime:
    """Parse date string to datetime."""
    try:
        return datetime.fromisoformat(date_str)
    except ValueError:
        print(f"❌ Invalid date format: {date_str}. Use YYYY-MM-DD format.")
        sys.exit(1)


def create_backtest_config(args: argparse.Namespace) -> BacktestConfig:
    """Create backtest configuration from command line arguments."""
    try:
        # Determine hash list
        if args.all_strategies:
            registry_hashes = load_all_strategies(args.registry_path)
        elif args.hashes:
            registry_hashes = load_hash_list(args.hashes)
        else:
            print("❌ Must specify either --hashes or --all-strategies")
            sys.exit(1)
        
        # Determine date range
        if args.start_date and args.end_date:
            start_date = parse_date(args.start_date)
            end_date = parse_date(args.end_date)
        elif args.days:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=args.days)
        else:
            # Default to last 30 days
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
        
        # Parse trading pair
        try:
            trading_pair = TradingPair[args.pair]
        except KeyError:
            print(f"❌ Invalid trading pair: {args.pair}")
            print(f"Available pairs: {[p.name for p in TradingPair]}")
            sys.exit(1)
        
        # Parse initial capital
        try:
            initial_capital = Decimal(str(args.initial_capital))
        except ValueError:
            print(f"❌ Invalid initial capital: {args.initial_capital}")
            sys.exit(1)
        
        return BacktestConfig(
            registry_hashes=registry_hashes,
            start_date=start_date,
            end_date=end_date,
            trading_pair=trading_pair,
            initial_capital=initial_capital,
            tick_interval_minutes=args.tick_interval,
            ferris_wheel_cycles=args.ferris_wheel_cycles,
            enable_feedback=not args.disable_feedback,
            enable_registry_updates=not args.disable_registry_updates,
            output_path=args.output_path,
            memory_path=args.memory_path,
            registry_path=args.registry_path
        )
        
    except Exception as e:
        print(f"❌ Error creating backtest config: {e}")
        sys.exit(1)


def print_config_summary(config: BacktestConfig) -> None:
    """Print backtest configuration summary."""
    print("\n" + "="*60)
    print("🎯 REGISTRY BACKTEST CONFIGURATION")
    print("="*60)
    print(f"📊 Strategies: {len(config.registry_hashes)}")
    print(f"📅 Period: {config.start_date.date()} to {config.end_date.date()}")
    print(f"💱 Trading Pair: {config.trading_pair.value}")
    print(f"💰 Initial Capital: ${config.initial_capital:,.2f}")
    print(f"⏱️  Tick Interval: {config.tick_interval_minutes} minutes")
    print(f"🎡 Ferris Wheel Cycles: {config.ferris_wheel_cycles}")
    print(f"🔄 Feedback Enabled: {config.enable_feedback}")
    print(f"📝 Registry Updates: {config.enable_registry_updates}")
    print(f"📁 Output Path: {config.output_path}")
    print(f"🧠 Memory Path: {config.memory_path}")
    print("="*60)
    
    # Print strategy hashes
    print("\n📋 STRATEGY HASHES:")
    for i, hash_id in enumerate(config.registry_hashes, 1):
        print(f"  {i:2d}. {hash_id[:16]}...")
    print("="*60)


async def run_backtest(config: BacktestConfig) -> None:
    """Run the registry backtest."""
    try:
        print("🚀 Initializing Registry Backtester...")
        
        # Create backtester
        backtester = RegistryBacktester(config)
        
        # Run backtest
        print("🔄 Running backtest...")
        result = await backtester.run_backtest()
        
        # Print results
        backtester.print_summary(result)
        
        print(f"✅ Backtest completed successfully!")
        print(f"📁 Results saved to: {config.output_path}")
        
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        logging.error(f"Backtest error: {e}", exc_info=True)
        sys.exit(1)


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Registry Backtest CLI for Schwabot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test specific strategies
  python scripts/run_registry_backtest.py --hashes registry/hash_list.json --pair BTC_USDC --days 30
  
  # Test all strategies in registry
  python scripts/run_registry_backtest.py --all-strategies --pair BTC_USDC --start-date 2024-01-01 --end-date 2024-01-31
  
  # Quick test with custom parameters
  python scripts/run_registry_backtest.py --all-strategies --pair BTC_USDC --days 7 --initial-capital 10000 --tick-interval 5
        """
    )
    
    # Strategy selection
    strategy_group = parser.add_mutually_exclusive_group(required=True)
    strategy_group.add_argument(
        '--hashes',
        type=str,
        help='Path to JSON file containing list of strategy hashes'
    )
    strategy_group.add_argument(
        '--all-strategies',
        action='store_true',
        help='Test all strategies in the registry'
    )
    
    # Date range
    date_group = parser.add_mutually_exclusive_group()
    date_group.add_argument(
        '--start-date',
        type=str,
        help='Start date in YYYY-MM-DD format'
    )
    date_group.add_argument(
        '--days',
        type=int,
        help='Number of days to test (from today backwards)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date in YYYY-MM-DD format (default: today)'
    )
    
    # Trading parameters
    parser.add_argument(
        '--pair',
        type=str,
        default='BTC_USDC',
        help='Trading pair to test (default: BTC_USDC)'
    )
    parser.add_argument(
        '--initial-capital',
        type=float,
        default=10000.0,
        help='Initial capital in USD (default: 10000.0)'
    )
    parser.add_argument(
        '--tick-interval',
        type=int,
        default=1,
        help='Tick interval in minutes (default: 1)'
    )
    parser.add_argument(
        '--ferris-wheel-cycles',
        type=int,
        default=16,
        help='Number of ticks per Ferris Wheel cycle (default: 16)'
    )
    
    # Feedback and registry options
    parser.add_argument(
        '--disable-feedback',
        action='store_true',
        help='Disable profit feedback engine'
    )
    parser.add_argument(
        '--disable-registry-updates',
        action='store_true',
        help='Disable registry updates during backtest'
    )
    
    # File paths
    parser.add_argument(
        '--registry-path',
        type=str,
        default='registry/hashed_strategies.json',
        help='Path to registry file (default: registry/hashed_strategies.json)'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        default='backtest_results',
        help='Output directory for results (default: backtest_results)'
    )
    parser.add_argument(
        '--memory-path',
        type=str,
        default='memory/cycle_feedback.json',
        help='Path for feedback memory (default: memory/cycle_feedback.json)'
    )
    
    # Other options
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show configuration without running backtest'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Check if Schwabot is available
    if not SCHWABOT_AVAILABLE:
        print("❌ Schwabot components not available")
        sys.exit(1)
    
    try:
        # Create configuration
        config = create_backtest_config(args)
        
        # Print configuration summary
        print_config_summary(config)
        
        # Exit if dry run
        if args.dry_run:
            print("✅ Configuration validated. Use --dry-run to run the actual backtest.")
            return
        
        # Run backtest
        asyncio.run(run_backtest(config))
        
    except KeyboardInterrupt:
        print("\n⚠️ Backtest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        logging.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main() 