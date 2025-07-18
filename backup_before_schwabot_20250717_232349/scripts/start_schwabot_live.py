#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Schwabot Live Trading Startup Script
======================================

Production startup script for Schwabot trading system with:
- 2-gram pattern detection integration
- Real-time market data processing
- Multi-exchange trading execution
- Strategy routing and portfolio management
- Performance monitoring and logging

Usage:
    python start_schwabot_live.py --mode demo
    python start_schwabot_live.py --mode live --config config/schwabot_live_trading_config.yaml
"""

import argparse
import asyncio
import logging
import os
import signal
import sys
import time
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import Schwabot components
from core.cli_live_entry import SchwabotCLI
from utils.logging_setup import setup_logging
from utils.secure_config_manager import SecureConfigManager

# Global variables for signal handling
cli_instance = None
shutdown_requested = False


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global shutdown_requested, cli_instance
    print(f"\n🛑 Received signal {signum}, initiating graceful shutdown...")
    shutdown_requested = True
    if cli_instance:
        asyncio.create_task(cli_instance.stop_trading())


async def check_system_requirements():
    """Check if all system requirements are met."""
    print("🔍 Checking system requirements...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    
    # Check required directories
    required_dirs = ["logs", "data", "backups", "config"]
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"📁 Created directory: {dir_name}")
    
    # Check configuration files
    config_files = [
        "config/schwabot_live_trading_config.yaml",
        "config/schwabot_config.yaml"
    ]
    
    for config_file in config_files:
        if not Path(config_file).exists():
            print(f"⚠️ Configuration file not found: {config_file}")
            if config_file == "config/schwabot_live_trading_config.yaml":
                print("   Using default configuration...")
            else:
                print("❌ Required configuration file missing")
                return False
    
    print("✅ System requirements check completed")
    return True


async def validate_exchange_configuration(config):
    """Validate exchange configuration for live trading."""
    print("🔐 Validating exchange configuration...")
    
    if config.get("system_mode") == "live":
        exchanges = config.get("exchanges", [])
        required_env_vars = []
        
        for exchange in exchanges:
            if exchange.get("enabled", False):
                api_key_env = exchange.get("api_key_env")
                secret_env = exchange.get("secret_env")
                
                if api_key_env and not os.getenv(api_key_env):
                    required_env_vars.append(api_key_env)
                if secret_env and not os.getenv(secret_env):
                    required_env_vars.append(secret_env)
        
        if required_env_vars:
            print("❌ Missing required environment variables:")
            for var in required_env_vars:
                print(f"   - {var}")
            print("\nPlease set the required environment variables before running in live mode.")
            return False
    
    print("✅ Exchange configuration validation completed")
    return True


async def initialize_logging(config):
    """Initialize logging system."""
    log_config = config.get("logging", {})
    
    setup_logging(
        level=getattr(logging, log_config.get("level", "INFO").upper()),
        file_logging=log_config.get("file_logging", True),
        console_logging=log_config.get("console_logging", True),
        log_file=log_config.get("log_file", "logs/schwabot_live.log"),
        max_file_size=log_config.get("max_file_size", "100MB"),
        backup_count=log_config.get("backup_count", 5)
    )


async def display_system_banner():
    """Display Schwabot system banner."""
    banner = """
🧠 SCHWABOT TRADING SYSTEM v2.0
================================

Advanced cryptocurrency trading system with:
• 2-Gram Pattern Detection & Strategy Routing
• Real-Time Market Data Processing
• Multi-Exchange Trading Execution
• Entropy-Enhanced Decision Making
• Fractal Memory & Phantom Detection
• Portfolio Balancing & Risk Management

Production-Ready Trading Intelligence
"""
    print(banner)


async def display_configuration_summary(config):
    """Display configuration summary."""
    print("📋 Configuration Summary:")
    print(f"   Mode: {config.get('system_mode', 'demo')}")
    print(f"   Symbol: {config.get('default_symbol', 'BTC/USDC')}")
    
    # Portfolio information will be displayed after discovery
    print(f"   Safe Mode: {config.get('safe_mode', True)}")
    
    # 2-gram configuration
    two_gram_config = config.get("2gram_config", {})
    print(f"   2-Gram Detector:")
    print(f"     Window Size: {two_gram_config.get('window_size', 100)}")
    print(f"     Burst Threshold: {two_gram_config.get('burst_threshold', 2.0)}")
    print(f"     Fractal Memory: {two_gram_config.get('enable_fractal_memory', True)}")
    
    # Exchange configuration
    exchanges = config.get("exchanges", [])
    enabled_exchanges = [ex for ex in exchanges if ex.get("enabled", False)]
    print(f"   Enabled Exchanges: {len(enabled_exchanges)}")
    for exchange in enabled_exchanges:
        print(f"     - {exchange.get('name', 'Unknown')}")
    
    print()


async def display_portfolio_summary(cli):
    """Display discovered portfolio summary."""
    print("📊 Portfolio Discovery Summary:")
    
    status = cli.get_system_status()
    portfolio = status.get("portfolio", {})
    
    print(f"   Total Portfolio Value: ${portfolio.get('total_value', 0):,.2f}")
    print(f"   Total Assets: {len(portfolio.get('held_assets', {}))}")
    print(f"   Available Trading Pairs: {portfolio.get('available_pairs', 0)}")
    print(f"   Connected Exchanges: {len(portfolio.get('exchanges', []))}")
    
    # Display held assets
    held_assets = portfolio.get("held_assets", {})
    if held_assets:
        print("   Held Assets:")
        for asset, amount in held_assets.items():
            if amount > 0:
                print(f"     {asset}: {amount:,.8f}")
    
    # Display connected exchanges
    exchanges = portfolio.get("exchanges", [])
    if exchanges:
        print("   Connected Exchanges:")
        for exchange in exchanges:
            print(f"     - {exchange}")
    
    print()


async def run_system_health_check(cli):
    """Run initial system health check."""
    print("🏥 Running system health check...")
    
    try:
        # Check 2-gram detector health
        if cli.two_gram_detector:
            health_check = await cli.two_gram_detector.health_check()
            print(f"   2-Gram Detector: {health_check.get('overall_status', 'unknown')}")
        
        # Check strategy router status
        if cli.strategy_router:
            print(f"   Strategy Router: Active")
        
        # Check trading pipeline status
        if cli.trading_pipeline:
            pipeline_summary = cli.trading_pipeline.get_pipeline_summary()
            print(f"   Trading Pipeline: {pipeline_summary.get('status', 'unknown')}")
        
        # Check execution engine status
        if cli.execution_engine:
            print(f"   Execution Engine: Active")
        
        # Check portfolio discovery
        status = cli.get_system_status()
        portfolio = status.get("portfolio", {})
        if portfolio.get("total_value", 0) > 0:
            print(f"   Portfolio Discovery: Successful (${portfolio['total_value']:,.2f})")
        else:
            print(f"   Portfolio Discovery: Using demo values")
        
        print("✅ System health check completed")
        return True
        
    except Exception as e:
        print(f"❌ System health check failed: {e}")
        return False


async def main():
    """Main startup function."""
    global cli_instance
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Schwabot Live Trading System")
    parser.add_argument("--mode", choices=["demo", "live", "backtest"], default="demo",
                       help="Trading mode (default: demo)")
    parser.add_argument("--config", default="config/schwabot_live_trading_config.yaml",
                       help="Configuration file path")
    parser.add_argument("--log-level", default="INFO",
                       help="Logging level")
    parser.add_argument("--no-banner", action="store_true",
                       help="Skip banner display")
    
    args = parser.parse_args()
    
    # Display banner
    if not args.no_banner:
        await display_system_banner()
    
    # Check system requirements
    if not await check_system_requirements():
        sys.exit(1)
    
    # Load configuration
    try:
        from config.schwabot_config import load_config
        config = load_config(args.config)
        
        # Override mode if specified
        if args.mode != "demo":
            config["system_mode"] = args.mode
            config["safe_mode"] = (args.mode == "demo")
        
        # Validate exchange configuration
        if not await validate_exchange_configuration(config):
            sys.exit(1)
        
        # Initialize logging
        await initialize_logging(config)
        
        # Display configuration summary
        await display_configuration_summary(config)
        
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        sys.exit(1)
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create CLI instance
    cli_instance = SchwabotCLI(args.config)
    
    try:
        # Initialize system
        print("🔧 Initializing Schwabot trading system...")
        if not await cli_instance.initialize_system(args.mode):
            print("❌ System initialization failed")
            sys.exit(1)
        
        # Display portfolio summary
        await display_portfolio_summary(cli_instance)
        
        # Run health check
        if not await run_system_health_check(cli_instance):
            print("❌ System health check failed")
            sys.exit(1)
        
        # Start live trading
        print("🚀 Starting live trading operations...")
        if not await cli_instance.start_live_trading():
            print("❌ Failed to start live trading")
            sys.exit(1)
        
        print("✅ Schwabot trading system is now running!")
        print("Press Ctrl+C to stop the system gracefully.")
        
        # Main loop
        while cli_instance.running and not shutdown_requested:
            await asyncio.sleep(1.0)
            
            # Display periodic status updates
            if int(time.time()) % 60 == 0:  # Every minute
                status = cli_instance.get_system_status()
                print(f"📊 Status: {status['total_trades']} trades, ${status['total_profit']:.2f} profit")
        
        # Graceful shutdown
        print("🛑 Initiating graceful shutdown...")
        await cli_instance.stop_trading()
        
        # Final status report
        final_status = cli_instance.get_system_status()
        print("\n📊 Final System Status:")
        print(f"   Total Trades: {final_status['total_trades']}")
        print(f"   Total Profit: ${final_status['total_profit']:.2f}")
        print(f"   Uptime: {final_status['uptime_seconds']:.1f} seconds")
        print("✅ Schwabot trading system shutdown completed")
        
    except KeyboardInterrupt:
        print("\n🛑 Received keyboard interrupt")
        await cli_instance.stop_trading()
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        if cli_instance:
            await cli_instance.stop_trading()
        sys.exit(1)


if __name__ == "__main__":
    # Run the main function
    asyncio.run(main()) 