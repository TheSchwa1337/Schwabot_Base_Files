# Wallet Tracker Integration Guide
================================

This guide explains how the Schwabot wallet tracker integrates with CCXT/Coinbase API and connects to the entire Schwabot strategy system.

## Overview

The wallet tracker is a core component of Schwabot v0.05 that:

- **Tracks portfolio positions** for BTC, ETH, XRP, USDC, SOL
- **Integrates with exchange APIs** via CCXT (Coinbase, Binance, etc.)
- **Connects to strategy system** for automated decision making
- **Provides real-time PNL tracking** and portfolio management
- **Generates strategy hashes** for recursive decision making
- **Triggers rebalancing** when portfolio conditions change

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Exchange APIs │    │   Wallet Tracker│    │  Strategy System│
│   (CCXT)        │◄──►│                 │◄──►│                 │
│                 │    │                 │    │                 │
│ • Coinbase      │    │ • Position Mgmt │    │ • Strategy Mapper│
│ • Binance       │    │ • PNL Tracking  │    │ • Ferris RDE    │
│ • Kraken        │    │ • Rebalancing   │    │ • Profit Alloc  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │ Portfolio Integ │
                       │                 │
                       │ • Trade Signals │
                       │ • Risk Mgmt     │
                       │ • Visual Output │
                       └─────────────────┘
```

## Core Components

### 1. Wallet Tracker (`wallet_tracker.py`)

**Purpose**: Manages portfolio positions and exchange integration

**Key Features**:
- Position tracking (BTC, ETH, XRP, USDC, SOL)
- PNL calculation and monitoring
- Exchange balance synchronization
- Transaction history
- Portfolio snapshots
- Strategy hash generation

**API Integration**:
```python
# Initialize with API config
config = {
    'api_enabled': True,
    'exchanges': {
        'coinbase': {
            'enabled': True,
            'api_key': 'your_api_key',
            'secret': 'your_secret',
            'passphrase': 'your_passphrase',
            'sandbox': True  # Set to False for live
        }
    }
}
wallet = WalletTracker(config)
```

### 2. Portfolio Integration (`portfolio_integration.py`)

**Purpose**: Connects wallet tracker to entire Schwabot system

**Key Features**:
- Module synchronization
- Trade signal generation
- Strategy evaluation
- Risk management
- Continuous integration loop

**Integration Flow**:
```python
# Initialize integration
integration = PortfolioIntegration(config)
integration.initialize_modules()

# Run integration cycle
integration.run_integration_cycle()

# Get trade signals
signals = integration.get_recent_signals()
```

## API Configuration

### Environment Variables

Set these environment variables for API access:

```bash
# Coinbase API
export COINBASE_API_KEY="your_api_key"
export COINBASE_API_SECRET="your_secret"
export COINBASE_PASSPHRASE="your_passphrase"

# Binance API (optional)
export BINANCE_API_KEY="your_api_key"
export BINANCE_API_SECRET="your_secret"
```

### Configuration Structure

```python
config = {
    'api_enabled': True,  # Enable/disable API integration
    'exchanges': {
        'coinbase': {
            'enabled': True,
            'api_key': os.getenv('COINBASE_API_KEY'),
            'secret': os.getenv('COINBASE_API_SECRET'),
            'passphrase': os.getenv('COINBASE_PASSPHRASE'),
            'sandbox': True  # Use sandbox for testing
        },
        'binance': {
            'enabled': False,  # Set to True to enable
            'api_key': os.getenv('BINANCE_API_KEY'),
            'secret': os.getenv('BINANCE_API_SECRET'),
            'sandbox': True
        }
    },
    'strategy_integration_enabled': True,
    'sync_interval': 60,  # Sync every 60 seconds
    'auto_snapshot_enabled': True
}
```

## Strategy System Integration

### 1. Strategy Mapper Connection

The wallet tracker injects portfolio data into the strategy mapper:

```python
# Inject portfolio state
wallet.inject_into_strategy_mapper(strategy_mapper)

# Generate strategy hash
strategy_hash = wallet.generate_strategy_hash()
```

### 2. Ferris RDE Connection

Connects portfolio state to Ferris wheel cycles:

```python
# Connect to Ferris RDE
wallet.connect_to_ferris_rde(ferris_rde)

# Get cycle data
cycle_data = wallet.get_ferris_cycle_data()
```

### 3. Portfolio State

The wallet provides portfolio state for strategy decisions:

```python
portfolio_state = {
    'total_value': 50000.0,
    'cash_balance': 10000.0,
    'total_pnl': 2500.0,
    'asset_breakdown': {
        'BTC': {'value': 20000.0, 'percentage': 40.0, 'pnl': 1000.0},
        'ETH': {'value': 15000.0, 'percentage': 30.0, 'pnl': 800.0},
        'USDC': {'value': 10000.0, 'percentage': 20.0, 'pnl': 0.0}
    },
    'strategy_hash': 'abc123...'
}
```

## Rebalancing Logic

### Automatic Rebalancing

The wallet tracker automatically detects when rebalancing is needed:

```python
# Check if rebalancing is needed
needs_rebalance = wallet.should_trigger_rebalance()

if needs_rebalance:
    suggestions = wallet.get_rebalance_suggestions()
    for suggestion in suggestions:
        print(f"Type: {suggestion['type']}")
        print(f"Reason: {suggestion['reason']}")
        print(f"Action: {suggestion['action']}")
        print(f"Priority: {suggestion['priority']}")
```

### Rebalancing Triggers

- **High cash ratio** (>80%): Deploy cash into assets
- **Significant losses** (<-10% PNL): Risk management
- **Low diversification** (<2 assets): Increase diversity
- **Asset concentration** (>70% in one asset): Reduce concentration

## Trade Signal Generation

### Signal Sources

The portfolio integration generates trade signals from multiple sources:

1. **Rebalancing Signals**: Portfolio rebalancing needs
2. **Strategy Signals**: Strategy mapper decisions
3. **Ferris Signals**: Ferris RDE phase-based signals
4. **Cash Deployment**: High cash balance deployment

### Signal Structure

```python
TradeSignal(
    signal_id="strategy_1234567890",
    timestamp=1640995200.0,
    asset=AssetType.BTC,
    action="buy",  # "buy", "sell", "hold", "rebalance"
    quantity=0.001,
    price=45000.0,
    confidence=0.85,
    source="strategy",  # "strategy", "ferris", "rebalance"
    metadata={'strategy_id': 'conservative_hash'}
)
```

## Usage Examples

### Basic Wallet Tracker

```python
from schwabot.core.wallet_tracker import WalletTracker

# Initialize wallet tracker
config = setup_api_config()
wallet = WalletTracker(config)

# Sync with exchanges
wallet.sync_portfolio_with_exchanges()

# Get portfolio summary
summary = wallet.get_portfolio_summary()
print(f"Total Value: ${summary['total_value']:.2f}")
print(f"Total PNL: ${summary['total_pnl']:.2f}")

# Check rebalancing
if wallet.should_trigger_rebalance():
    suggestions = wallet.get_rebalance_suggestions()
    print("Rebalancing suggestions:", suggestions)
```

### Full Integration

```python
from schwabot.core.portfolio_integration import PortfolioIntegration

# Initialize full integration
integration = PortfolioIntegration(config)
integration.initialize_modules()

# Run continuous integration
integration.start_continuous_integration(interval=60)

# Or run single cycle
success = integration.run_integration_cycle()
if success:
    signals = integration.get_recent_signals()
    print("Recent signals:", signals)
```

### API Operations

```python
# Fetch exchange balances
balances = wallet.fetch_exchange_balances()
for exchange, exchange_balances in balances.items():
    print(f"{exchange}: {exchange_balances}")

# Get current prices
for asset in AssetType:
    price = wallet._get_current_price(asset.value)
    print(f"{asset.value}: ${price:.2f}")
```

## Error Handling

### API Connection Errors

```python
try:
    wallet.sync_portfolio_with_exchanges()
except Exception as e:
    logger.error(f"API sync failed: {e}")
    # Fall back to simulated data
    wallet._simulate_portfolio_sync()
```

### Strategy Integration Errors

```python
try:
    wallet.inject_into_strategy_mapper(strategy_mapper)
except Exception as e:
    logger.error(f"Strategy injection failed: {e}")
    # Continue with existing strategy state
```

## Monitoring and Logging

### Portfolio Monitoring

```python
# Get integration summary
summary = integration.get_integration_summary()
print(f"Success Rate: {summary['success_rate']:.1%}")
print(f"Total Signals: {summary['total_signals']}")
print(f"Modules Initialized: {summary['modules_initialized']}")
```

### Performance Tracking

```python
# Get portfolio performance
summary = wallet.get_portfolio_summary()
print(f"PNL Percentage: {summary['total_pnl_percentage']:.2f}%")
print(f"Asset Diversity: {len(summary['asset_breakdown'])}")

# Get PNL history
pnl_history = wallet.get_pnl_history(days=30)
for entry in pnl_history:
    print(f"Date: {entry['timestamp']}, PNL: ${entry['total_pnl']:.2f}")
```

## Security Considerations

### API Key Management

1. **Use environment variables** for API keys
2. **Enable sandbox mode** for testing
3. **Use read-only permissions** initially
4. **Rotate keys regularly**
5. **Monitor API usage**

### Risk Management

1. **Set position limits** in configuration
2. **Use stop-loss orders** for live trading
3. **Monitor portfolio concentration**
4. **Enable fallback logic** for system failures
5. **Test thoroughly** in sandbox mode

## Troubleshooting

### Common Issues

1. **API Connection Failed**
   - Check API keys and permissions
   - Verify network connectivity
   - Check exchange status

2. **Strategy Integration Issues**
   - Ensure all modules are initialized
   - Check configuration settings
   - Verify data synchronization

3. **Rebalancing Not Triggering**
   - Check threshold settings
   - Verify portfolio state
   - Review rebalancing logic

### Debug Mode

Enable debug logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with debug output
wallet = WalletTracker(config)
wallet.sync_portfolio_with_exchanges()
```

## Best Practices

1. **Start with simulation mode** before live trading
2. **Monitor portfolio regularly** for rebalancing needs
3. **Use appropriate risk management** settings
4. **Test integration thoroughly** with small amounts
5. **Keep API keys secure** and rotate regularly
6. **Monitor system performance** and logs
7. **Have fallback strategies** for system failures

## Next Steps

1. **Set up API keys** for your preferred exchanges
2. **Configure risk parameters** for your trading style
3. **Test integration** in sandbox mode
4. **Monitor performance** and adjust settings
5. **Scale up gradually** as confidence grows

For more information, see the main Schwabot documentation and examples. 