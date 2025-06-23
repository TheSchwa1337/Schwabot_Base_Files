# Schwabot - Advanced High-Frequency Trading System

## Overview

Schwabot is a sophisticated high-frequency trading system that combines advanced mathematical frameworks, recursive self-validation, and cyclical systems for optimal trading performance. Built with enterprise-grade architecture, it provides comprehensive risk management, regulatory compliance, and observability.

## Quick Start

### Prerequisites

- Python 3.8+
- Linux environment (recommended for production)
- CCXT library for exchange integration
- Required Python packages (see `requirements.txt`)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd schwabot

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp config/schwabot.example.yaml config/schwabot.yaml
# Edit config/schwabot.yaml with your settings
```

### Basic Usage

```python
from core.schwabot_main import SchwabotMain

# Initialize Schwabot
schwabot = SchwabotMain()

# Start trading (paper trading mode)
await schwabot.start_paper_trading()

# Check system status
status = schwabot.get_system_status()
print(f"System Status: {status}")
```

## Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Schwabot Architecture                     │
├─────────────────────────────────────────────────────────────┤
│  User Interface & CLI                                       │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │   Web UI    │ │   CLI       │ │   API       │           │
│  └─────────────┘ └─────────────┘ └─────────────┘           │
├─────────────────────────────────────────────────────────────┤
│  Core Trading Engine                                        │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │   ZPE Core  │ │   VECU Core │ │ Ferris RDE  │           │
│  └─────────────┘ └─────────────┘ └─────────────┘           │
├─────────────────────────────────────────────────────────────┤
│  Risk & Capital Management                                  │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │ Risk Guard  │ │Capital Ctrl │ │Enhanced Risk│           │
│  └─────────────┘ └─────────────┘ └─────────────┘           │
├─────────────────────────────────────────────────────────────┤
│  Exchange & Data Layer                                      │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │Exchange Plbg│ │Secure API   │ │Persistent   │           │
│  │             │ │Manager      │ │State        │           │
│  └─────────────┘ └─────────────┘ └─────────────┘           │
├─────────────────────────────────────────────────────────────┤
│  Observability & Compliance                                 │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │Ops & Obs    │ │Regulatory   │ │Environment  │           │
│  │             │ │Compliance   │ │Manager      │           │
│  └─────────────┘ └─────────────┘ └─────────────┘           │
├─────────────────────────────────────────────────────────────┤
│  Simulation & Testing                                       │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │Long Horizon │ │Precision &  │ │Monte Carlo  │           │
│  │Simulation   │ │Performance  │ │Simulation   │           │
│  └─────────────┘ └─────────────┘ └─────────────┘           │
└─────────────────────────────────────────────────────────────┘
```

### Mathematical Framework

Schwabot implements the "Saw Blade Theory of Recursive Profit Allocation" with the following key components:

1. **ZPE (Zero Point Energy) Core**: Discrete log transform waveform analysis
2. **VECU (Vectorized Electronic Control Unit)**: Profit timing synchronization and PWM modulation
3. **Ferris RDE (Recursive Dynamic Engine)**: Cyclical measurement and 16-bit BTC price mapping
4. **Unified Math Library**: Centralized mathematical operations and constants

## Configuration

### Main Configuration File

```yaml
# config/schwabot.yaml
environment:
  type: "production"  # production, staging, development
  canary_enabled: false
  exchange_testnet: true

trading:
  mode: "paper_trading"  # paper_trading, live_trading, backtesting
  symbols: ["BTC/USD", "ETH/USD"]
  position_size: 0.01  # 1% of capital per trade
  
risk:
  max_drawdown: 0.05  # 5% maximum drawdown
  daily_loss_limit: 0.02  # 2% daily loss limit
  position_limit: 0.1  # 10% maximum position size

exchange:
  name: "binance"
  testnet: true
  api_key: "${BINANCE_API_KEY}"
  api_secret: "${BINANCE_SECRET}"

observability:
  log_level: "INFO"
  metrics_enabled: true
  slack_alerts: true
  slack_webhook: "${SLACK_WEBHOOK_URL}"
```

### Environment Variables

```bash
# Required for live trading
export BINANCE_API_KEY="your_api_key"
export BINANCE_SECRET="your_secret_key"
export SLACK_WEBHOOK_URL="your_slack_webhook"

# Optional
export SCHWABOT_CONFIG_PATH="/path/to/config.yaml"
export SCHWABOT_LOG_LEVEL="DEBUG"
```

## API Reference

### Core Trading Engine

#### ZPE Core

```python
from core.zpe_core import get_zpe_core

zpe = get_zpe_core()

# Calculate resonance for price
resonance = zpe.calculate_resonance(btc_price=50000.0)

# Get profit allocation
allocation = zpe.calculate_profit_allocation(
    base_amount=1000.0,
    resonance_factor=0.75
)
```

#### VECU Core

```python
from core.vecu_core import get_vecu_core

vecu = get_vecu_core()

# Calculate timing phase
phase = vecu.calculate_timing_phase(timestamp=datetime.now())

# Get PWM profit burst
burst = vecu.calculate_pwm_burst(
    base_frequency=1000,
    duty_cycle=0.6
)
```

#### Ferris RDE

```python
from core.ferris_rde_core import get_ferris_rde

ferris = get_ferris_rde()

# Calculate wheel position
position = ferris.calculate_wheel_position(volume=5000.0)

# Get hash sequence
hash_seq = ferris.generate_hash_sequence(length=16)
```

### Risk Management

#### Risk Guard

```python
from core.risk_guard import get_risk_guard

risk_guard = get_risk_guard()

# Check if trade is allowed
allowed = risk_guard.check_trade_allowed(
    symbol="BTC/USD",
    side="buy",
    amount=0.1,
    price=50000.0
)

# Get risk status
status = risk_guard.get_risk_status()
```

#### Capital Controls

```python
from core.capital_controls import get_capital_controls

capital = get_capital_controls()

# Calculate position size
size = capital.calculate_position_size(
    capital=10000.0,
    risk_per_trade=0.02,
    volatility=0.05
)

# Get portfolio state
portfolio = capital.get_portfolio_state()
```

### Exchange Integration

```python
from core.exchange_plumbing import get_exchange_plumbing

exchange = get_exchange_plumbing()

# Place order
order = await exchange.place_order(
    symbol="BTC/USD",
    side="buy",
    order_type="market",
    amount=0.01
)

# Get balance
balance = await exchange.get_balance(currency="USD")
```

### Observability

```python
from core.ops_observability import log_operation, LogLevel

# Log operation
log_operation(
    operation="order_placed",
    component="exchange_plumbing",
    level=LogLevel.INFO,
    success=True,
    order_id="12345",
    symbol="BTC/USD"
)

# Get metrics
from core.ops_observability import get_metrics
metrics = get_metrics()
```

## Trading Strategies

### Basic Strategy

```python
from core.strategy_mapper import get_strategy_mapper

strategy = get_strategy_mapper()

# Define strategy
@strategy.register("basic_momentum")
def basic_momentum_strategy(market_data):
    """Basic momentum strategy."""
    price = market_data['price']
    volume = market_data['volume']
    
    # Simple momentum calculation
    if price > market_data.get('prev_price', price):
        return {'action': 'buy', 'confidence': 0.7}
    else:
        return {'action': 'sell', 'confidence': 0.7}
```

### Advanced ZPE Strategy

```python
@strategy.register("zpe_resonance")
def zpe_resonance_strategy(market_data):
    """ZPE resonance-based strategy."""
    from core.zpe_core import get_zpe_core
    
    zpe = get_zpe_core()
    resonance = zpe.calculate_resonance(market_data['price'])
    
    if resonance > 0.8:
        return {'action': 'buy', 'confidence': resonance}
    elif resonance < 0.2:
        return {'action': 'sell', 'confidence': 1 - resonance}
    else:
        return {'action': 'hold', 'confidence': 0.5}
```

## Backtesting

### Run Backtest

```python
from core.long_horizon_simulation import run_monte_carlo_simulation

# Run Monte Carlo simulation
results = await run_monte_carlo_simulation(
    num_scenarios=100,
    duration_days=7
)

# Analyze results
total_pnl = sum(r.total_pnl for r in results)
avg_sharpe = sum(r.sharpe_ratio for r in results) / len(results)

print(f"Total PnL: ${total_pnl:,.2f}")
print(f"Average Sharpe: {avg_sharpe:.2f}")
```

### Chaos Testing

```python
from core.long_horizon_simulation import run_chaos_monkey_test

# Run chaos monkey test
events = await run_chaos_monkey_test(duration_hours=24)

# Analyze resilience
recovery_rate = sum(1 for e in events if e.recovery_successful) / len(events)
print(f"Recovery Rate: {recovery_rate:.1%}")
```

## Monitoring & Alerts

### Health Check

```python
from core.ops_observability import get_health_status

health = get_health_status()
print(f"System Health: {health['status']}")
print(f"Uptime: {health['uptime']}")
print(f"Active Components: {health['active_components']}")
```

### Slack Alerts

Configure Slack alerts in your config file:

```yaml
observability:
  slack_alerts: true
  slack_webhook: "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
  alert_levels:
    - "ERROR"
    - "WARNING"
    - "CRITICAL"
```

## Security

### API Key Management

```python
from core.secure_api_manager import get_secure_api_manager

api_manager = get_secure_api_manager()

# Store encrypted API key
api_manager.store_api_key(
    exchange="binance",
    api_key="your_api_key",
    api_secret="your_secret"
)

# Retrieve API key
credentials = api_manager.get_api_credentials("binance")
```

### Compliance

```python
from core.regulatory_compliance import log_order_routing

# Log order for compliance
log_order_routing(
    order_request=order_request,
    order_response=order_response,
    routing_type="smart",
    destination="binance",
    execution_venue="binance_spot"
)
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **Configuration Errors**: Check your config file syntax
   ```bash
   python -c "import yaml; yaml.safe_load(open('config/schwabot.yaml'))"
   ```

3. **Exchange Connection Issues**: Verify API credentials and network connectivity

4. **Memory Issues**: Monitor memory usage and adjust allocation settings

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable debug mode in config
environment:
  debug: true
  log_level: "DEBUG"
```

### Performance Profiling

```python
from core.precision_performance import get_precision_performance_manager

perf = get_precision_performance_manager()

# Start profiling
perf.start_profiling("trading_loop")

# Your trading code here
# ...

# Stop profiling and get results
results = perf.stop_profiling("trading_loop")
print(f"Execution time: {results['execution_time']:.4f}s")
```

## Deployment

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "-m", "core.schwabot_main"]
```

### Production Checklist

- [ ] Configure production environment
- [ ] Set up monitoring and alerting
- [ ] Test with paper trading
- [ ] Verify risk controls
- [ ] Set up backup and recovery
- [ ] Configure compliance reporting
- [ ] Test chaos monkey scenarios
- [ ] Validate performance metrics

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:

- Create an issue on GitHub
- Check the documentation
- Review the troubleshooting guide
- Contact the development team

---

**Warning**: This is a sophisticated trading system. Always test thoroughly in paper trading mode before using real funds. Trading cryptocurrencies involves significant risk. 