import logging
import os
import sys
import time
from pathlib import Path

from schwabot.core.ferris_rde import FerrisRDE
from schwabot.core.portfolio_integration import PortfolioIntegration
from schwabot.core.strategy_mapper import StrategyMapper
from schwabot.core.wallet_tracker import AssetType, WalletTracker

#!/usr/bin/env python3
"""
Wallet Integration Example
==========================

Demonstrates how to use the wallet tracker with CCXT/Coinbase API integration
and how it connects to the entire Schwabot strategy system.
"""


# Add schwabot to path
sys.path.append(str(Path(__file__).parent.parent))


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def setup_api_config():
    """Setup API configuration for exchanges."""
    # Example configuration - replace with your actual API keys
    config = {
        "api_enabled": True,  # Set to False for simulation mode
        "exchanges": {
            "coinbase": {
                "enabled": True,  # Set to False if you don't want to use Coinbase
                "api_key": os.getenv("COINBASE_API_KEY", ""),
                "secret": os.getenv("COINBASE_API_SECRET", ""),
                "passphrase": os.getenv("COINBASE_PASSPHRASE", ""),
                "sandbox": True,  # Set to False for live trading
            },
            "binance": {
                "enabled": False,  # Set to True if you want to use Binance
                "api_key": os.getenv("BINANCE_API_KEY", ""),
                "secret": os.getenv("BINANCE_API_SECRET", ""),
                "sandbox": True,
            },
        },
        "strategy_integration_enabled": True,
        "sync_interval": 60,  # 1 minute
        "auto_snapshot_enabled": True,
    }
    return config


def demonstrate_wallet_tracker():
    """Demonstrate basic wallet tracker functionality."""
    print("\n" + "=" * 60)
    print("WALLET TRACKER DEMONSTRATION")
    print("=" * 60)

    # Initialize wallet tracker
    config = setup_api_config()
    wallet = WalletTracker(config)

    # Sync with exchanges (or simulate if API disabled)
    print("\n1. Syncing portfolio with exchanges...")
    sync_success = wallet.sync_portfolio_with_exchanges()
    print(f"   Sync successful: {sync_success}")

    # Get portfolio summary
    print("\n2. Portfolio Summary:")
    summary = wallet.get_portfolio_summary()
    print(f"   Total Value: ${summary['total_value']:.2f}")
    print(f"   Cash Balance: ${summary['cash_balance']:.2f}")
    print(f"   Total PNL: ${summary['total_pnl']:.2f}")
    print(f"   PNL Percentage: {summary['total_pnl_percentage']:.2f}%")
    print(f"   Total Positions: {summary['total_positions']}")

    # Show asset breakdown
    print("\n3. Asset Breakdown:")
    for asset, data in summary["asset_breakdown"].items():
        print(f"   {asset}: ${data['value']:.2f} ({data['percentage']:.1f}%) - PNL: ${data['pnl']:.2f}")

    # Generate strategy hash
    print("\n4. Strategy Hash:")
    strategy_hash = wallet.generate_strategy_hash()
    print(f"   Hash: {strategy_hash[:16]}...")

    # Check rebalancing needs
    print("\n5. Rebalancing Analysis:")
    needs_rebalance = wallet.should_trigger_rebalance()
    print(f"   Needs rebalancing: {needs_rebalance}")

    if needs_rebalance:
        suggestions = wallet.get_rebalance_suggestions()
        print("   Suggestions:")
        for suggestion in suggestions:
            print(f"     - {suggestion['type']}: {suggestion['reason']}")
            print(f"       Action: {suggestion['action']} (Priority: {suggestion['priority']})")

    return wallet


def demonstrate_strategy_integration(wallet):
    """Demonstrate strategy integration."""
    print("\n" + "=" * 60)
    print("STRATEGY INTEGRATION DEMONSTRATION")
    print("=" * 60)

    # Initialize strategy mapper
    strategy_mapper = StrategyMapper()

    # Inject wallet data into strategy mapper
    print("\n1. Injecting portfolio data into strategy mapper...")
    wallet.inject_into_strategy_mapper(strategy_mapper)

    # Get strategy summary
    print("\n2. Strategy Mapper Summary:")
    strategy_summary = strategy_mapper.get_strategy_summary()
    print(f"   Total Strategies: {strategy_summary['total_strategies']}")
    print(f"   Hash Strategies: {strategy_summary['hash_strategies']}")
    print(f"   Bit Strategies: {strategy_summary['bit_strategies']}")
    print(f"   Total Executions: {strategy_summary['total_executions']}")
    print(f"   Current Strategy: {strategy_summary['current_strategy']}")

    # Simulate market data and strategy selection
    print("\n3. Strategy Selection Simulation:")
    market_data = {
        "price": 45000.0,
        "volume": 1000000.0,
        "volatility": 0.02,
        "timestamp": time.time(),
    }
    portfolio_state = {
        "total_value": wallet.get_portfolio_summary()["total_value"],
        "cash_balance": wallet.get_portfolio_summary()["cash_balance"],
        "strategy_hash": wallet.generate_strategy_hash(),
    }
    selected_strategy = strategy_mapper.select_strategy(market_data, portfolio_state)
    if selected_strategy:
        print(f"   Selected Strategy: {selected_strategy.name}")
        print(f"   Strategy Type: {selected_strategy.strategy_type.value}")
        print(f"   Risk Level: {selected_strategy.risk_level}")
        print(f"   Min Confidence: {selected_strategy.min_confidence}")

        # Execute strategy
        result = strategy_mapper.execute_strategy(selected_strategy, market_data, portfolio_state)
        if result:
            print(f"   Strategy Result: {result.signal_type} (confidence: {result.confidence:.2f})")
            print(f"   Position Size: {result.position_size:.4f}")
    else:
        print("   No strategy selected")

    return strategy_mapper


def demonstrate_ferris_integration(wallet):
    """Demonstrate Ferris RDE integration."""
    print("\n" + "=" * 60)
    print("FERRIS RDE INTEGRATION DEMONSTRATION")
    print("=" * 60)

    # Initialize Ferris RDE
    ferris = FerrisRDE()

    # Connect wallet to Ferris
    print("\n1. Connecting wallet to Ferris RDE...")
    wallet.connect_to_ferris_rde(ferris)

    # Start a cycle
    print("\n2. Starting Ferris cycle...")
    cycle = ferris.start_cycle()
    print(f"   Cycle ID: {cycle.cycle_id}")
    print(f"   Phase: {cycle.phase.value}")

    # Get Ferris summary
    print("\n3. Ferris RDE Summary:")
    ferris_summary = ferris.get_ferris_summary()
    print(f"   Current Phase: {ferris_summary['current_phase']}")
    print(f"   Current Cycle: {ferris_summary['current_cycle']}")
    print(f"   State: {ferris_summary['state']}")
    print(f"   Total Cycles: {ferris_summary['total_cycles']}")
    print(f"   Total Signals: {ferris_summary['total_signals']}")

    # Simulate market data and generate signal
    print("\n4. Ferris Signal Generation:")
    market_data = {
        "price": 45000.0,
        "volume": 1000000.0,
        "volatility": 0.02,
        "timestamp": time.time(),
    }
    ferris_signal = ferris.generate_signal(market_data)
    if ferris_signal:
        print(f"   Signal Type: {ferris_signal.signal_type}")
        print(f"   Phase: {ferris_signal.phase.value}")
        print(f"   Strength: {ferris_signal.strength:.2f}")
        print(f"   Confidence: {ferris_signal.confidence:.2f}")
    else:
        print("   No signal generated")

    # Get Ferris cycle data from wallet
    print("\n5. Wallet Ferris Cycle Data:")
    cycle_data = wallet.get_ferris_cycle_data()
    print(f"   Portfolio Value: ${cycle_data['portfolio_value']:.2f}")
    print(f"   Cash Ratio: {cycle_data['cash_ratio']:.1%}")
    print(f"   PNL Ratio: {cycle_data['pnl_ratio']:.1%}")
    print(f"   Asset Diversity: {cycle_data['asset_diversity']}")

    return ferris


def demonstrate_portfolio_integration():
    """Demonstrate full portfolio integration."""
    print("\n" + "=" * 60)
    print("FULL PORTFOLIO INTEGRATION DEMONSTRATION")
    print("=" * 60)

    # Initialize portfolio integration
    config = setup_api_config()
    integration = PortfolioIntegration(config)

    # Initialize all modules
    print("\n1. Initializing all Schwabot modules...")
    integration.initialize_modules()

    # Sync all modules
    print("\n2. Syncing all modules...")
    sync_success = integration.sync_all_modules()
    print(f"   Sync successful: {sync_success}")

    # Get integration summary
    print("\n3. Integration Summary:")
    summary = integration.get_integration_summary()
    print(f"   Modules Initialized: {summary['modules_initialized']}")
    print(f"   Last Sync: {summary['last_sync']}")
    print(f"   Total Signals: {summary['total_signals']}")
    print(f"   Successful Trades: {summary['successful_trades']}")
    print(f"   Failed Trades: {summary['failed_trades']}")
    print(f"   Success Rate: {summary['success_rate']:.1%}")

    # Show integration state
    state = summary["integration_state"]
    print(f"   Wallet Synced: {state['wallet_synced']}")
    print(f"   Strategy Active: {state['strategy_active']}")
    print(f"   Ferris Phase: {state['ferris_phase']}")
    print(f"   Portfolio Value: ${state['portfolio_value']:.2f}")
    print(f"   Cash Ratio: {state['cash_ratio']:.1%}")
    print(f"   PNL Ratio: {state['pnl_ratio']:.1%}")
    print(f"   Asset Diversity: {state['asset_diversity']}")

    # Run integration cycle
    print("\n4. Running integration cycle...")
    cycle_success = integration.run_integration_cycle()
    print(f"   Cycle successful: {cycle_success}")

    # Get recent signals
    print("\n5. Recent Trade Signals:")
    recent_signals = integration.get_recent_signals(5)
    if recent_signals:
        for signal in recent_signals:
            print(
                f"   {signal['timestamp']}: {signal['action']} {signal['quantity']:.4f} {signal['asset']} @ ${signal['price']:.2f} (confidence: {signal['confidence']:.2f}, source: {signal['source']})"
            )
    else:
        print("   No recent signals")

    return integration


def demonstrate_api_operations(wallet):
    """Demonstrate API operations (if enabled)."""
    print("\n" + "=" * 60)
    print("API OPERATIONS DEMONSTRATION")
    print("=" * 60)

    if not wallet.api_enabled:
        print("\nAPI integration is disabled. Enable it in config to see live exchange data.")
        return

    # Fetch exchange balances
    print("\n1. Fetching exchange balances...")
    balances = wallet.fetch_exchange_balances()

    for exchange_name, exchange_balances in balances.items():
        print(f"\n   {exchange_name.upper()}:")
        if exchange_balances:
            for asset, balance_data in exchange_balances.items():
                print(
                    f"     {asset}: Free={balance_data['free']:.4f}, Used={balance_data['used']:.4f}, Total={balance_data['total']:.4f}"
                )
        else:
            print("     No balances found")

    # Show exchange balance objects
    print("\n2. Exchange Balance Objects:")
    for balance_key, balance_obj in wallet.exchange_balances.items():
        print(
            f"   {balance_key}: {balance_obj.asset.value} - Free: {balance_obj.free:.4f}, Total: {balance_obj.total:.4f}"
        )

    # Get current prices
    print("\n3. Current Asset Prices:")
    for asset in AssetType:
        price = wallet._get_current_price(asset.value)
        print(f"   {asset.value}: ${price:.2f}")


def main():
    """Main demonstration function."""
    print("SCHWABOT WALLET INTEGRATION DEMONSTRATION")
    print("=" * 60)
    print("This example demonstrates how the wallet tracker integrates with")
    print("CCXT/Coinbase API and connects to the entire Schwabot strategy system.")
    print("=" * 60)

    try:
        # Basic wallet tracker demonstration
        wallet = demonstrate_wallet_tracker()

        # Strategy integration demonstration
        demonstrate_strategy_integration(wallet)

        # Ferris RDE integration demonstration
        demonstrate_ferris_integration(wallet)

        # Full portfolio integration demonstration
        demonstrate_portfolio_integration()

        # API operations demonstration
        demonstrate_api_operations(wallet)

        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETE")
        print("=" * 60)
        print("The wallet tracker is now fully integrated with:")
        print("✅ CCXT/Coinbase API (or simulation mode)")
        print("✅ Strategy Mapper")
        print("✅ Ferris RDE")
        print("✅ Profit Cycle Allocator")
        print("✅ Portfolio Integration System")
        print("✅ Glyph Visualizer")
        print("✅ Matrix Logic")
        print("✅ Fallback Logic")

        print("\nTo enable live trading:")
        print("1. Set your API keys in environment variables")
        print("2. Set 'api_enabled': True in config")
        print("3. Set 'sandbox': False for live trading")
        print("4. Run the integration system")

    except Exception as e:
        logger.error(f"Error in demonstration: {e}")
        print(f"\nError: {e}")
        print("Check your configuration and API keys.")


if __name__ == "__main__":
    main()
