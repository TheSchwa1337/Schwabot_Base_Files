# 🚀 Schwabot Production Trading Guide

## Overview

The Schwabot Production Trading System is a complete, production-ready cryptocurrency trading platform that integrates:

- **Real CCXT Exchange Connections** with API key management
- **Portfolio Tracking & Position Management** with real-time balance synchronization
- **Risk Management & Circuit Breakers** with configurable thresholds
- **Live Market Data Processing** with entropy-enhanced decision making
- **Order Execution & Performance Monitoring** with comprehensive reporting
- **Error Handling & Recovery** with automatic failover mechanisms

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Production    │    │   Portfolio     │    │   Risk          │
│   Trading       │◄──►│   Tracker       │◄──►│   Manager       │
│   Pipeline      │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CCXT Trading  │    │   Market Data   │    │   Circuit       │
│   Engine        │    │   Feed          │    │   Breakers      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Exchange      │    │   Performance   │    │   Error         │
│   APIs          │    │   Monitoring    │    │   Recovery      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### 1. Environment Setup

Set up your environment variables for API access:

```bash
# Exchange Configuration
export SCHWABOT_EXCHANGE="coinbase"
export SCHWABOT_API_KEY="your_api_key_here"
export SCHWABOT_SECRET="your_secret_here"
export SCHWABOT_SANDBOX="true"  # Set to "false" for live trading

# Trading Configuration
export SCHWABOT_SYMBOLS="BTC/USDC,ETH/USDC"
export SCHWABOT_RISK_TOLERANCE="0.2"
export SCHWABOT_MAX_POSITION_SIZE="0.1"
export SCHWABOT_MAX_DAILY_LOSS="0.05"
```

### 2. Configuration File (Alternative)

Create a configuration file `config/production_config.yaml`:

```yaml
exchange_name: "coinbase"
api_key: "your_api_key_here"
secret: "your_secret_here"
sandbox: true
symbols: ["BTC/USDC", "ETH/USDC"]
risk_tolerance: 0.2
max_position_size: 0.1
max_daily_loss: 0.05
```

### 3. Start Production Trading

```bash
# Using environment variables
python main.py --production

# Using configuration file
python main.py --production --production-config config/production_config.yaml
```

## 📊 Portfolio Management

### Real-Time Portfolio Tracking

The system automatically tracks:

- **Balances**: Real-time synchronization with exchange balances
- **Positions**: Open and closed positions with PnL calculation
- **Transactions**: Complete transaction history
- **Performance**: Realized and unrealized PnL

### Portfolio Commands

```bash
# Get portfolio status
python main.py --production-status

# Sync portfolio with exchange
python main.py --sync-portfolio

# Export comprehensive report
python main.py --export-report
```

### Portfolio Features

- **Decimal Precision**: 18-decimal precision for accurate calculations
- **Multi-Currency Support**: Track multiple cryptocurrencies
- **Real-Time Updates**: Automatic price updates every 5 seconds
- **Balance Synchronization**: Sync with exchange every 30 seconds
- **Transaction Logging**: Complete audit trail of all trades

## 🛡️ Risk Management

### Circuit Breakers

The system includes multiple circuit breakers:

- **Daily Loss Limit**: Automatic stop when daily loss exceeds threshold
- **Position Size Limits**: Maximum position size enforcement
- **Error Thresholds**: Stop trading after too many errors
- **Drawdown Protection**: Emergency stop on significant drawdown

### Risk Configuration

```yaml
risk_tolerance: 0.2        # 20% risk tolerance
max_position_size: 0.1     # 10% max position size
max_daily_loss: 0.05       # 5% max daily loss
enable_circuit_breakers: true

# Circuit breaker settings
circuit_breakers:
  max_drawdown: 0.1
  max_consecutive_losses: 5
  max_daily_errors: 10
```

### Risk Commands

```bash
# Reset circuit breakers
python main.py --reset-circuit-breakers

# Get error log
python main.py --error-log --error-log-limit 50
```

## 📈 Trading Execution

### Trading Cycle

Each trading cycle includes:

1. **Portfolio Synchronization**: Sync balances with exchange
2. **Market Data Collection**: Fetch current prices and order books
3. **Position Price Updates**: Update unrealized PnL for open positions
4. **Entropy Signal Processing**: Generate trading signals
5. **Risk Assessment**: Evaluate trade against risk parameters
6. **Order Execution**: Execute trades if conditions are met
7. **Portfolio Updates**: Update positions and balances
8. **Performance Tracking**: Record metrics and performance

### Supported Exchanges

- **Coinbase** (recommended for beginners)
- **Binance** (high volume, many pairs)
- **Kraken** (good for European users)
- **Any CCXT-supported exchange**

### Order Types

- **Market Orders**: Immediate execution at current price
- **Limit Orders**: Execution at specified price (future enhancement)
- **Stop Loss**: Automatic position closure (future enhancement)

## 🔧 Configuration Options

### Exchange Configuration

```yaml
exchange_name: "coinbase"  # Exchange name
api_key: "your_api_key"    # API key
secret: "your_secret"      # Secret key
sandbox: true              # Sandbox mode (recommended for testing)
```

### Trading Configuration

```yaml
symbols: ["BTC/USDC", "ETH/USDC"]  # Trading pairs
base_currency: "USDC"              # Base currency
risk_tolerance: 0.2                # Risk tolerance (0.0-1.0)
max_position_size: 0.1             # Max position size (0.0-1.0)
max_daily_loss: 0.05               # Max daily loss (0.0-1.0)
```

### Performance Configuration

```yaml
portfolio_sync_interval: 30  # Portfolio sync interval (seconds)
price_update_interval: 5     # Price update interval (seconds)
performance_history_size: 1000  # Performance history records
error_log_size: 100            # Error log records
```

## 📊 Monitoring & Reporting

### Real-Time Monitoring

The system provides real-time monitoring of:

- **Portfolio Value**: Total portfolio value including unrealized PnL
- **Trade Performance**: Win rate, total trades, total PnL
- **Risk Metrics**: Current risk level, circuit breaker status
- **System Health**: Error counts, connection status

### Performance Reports

```bash
# Export comprehensive report
python main.py --export-report
```

Reports include:

- **System Status**: Current trading status and configuration
- **Performance History**: Historical performance metrics
- **Error Log**: Recent errors and their resolution
- **Portfolio History**: Transaction history and position changes

### Report Format

```json
{
  "system_status": {
    "status": {
      "is_running": true,
      "total_trades": 25,
      "successful_trades": 18,
      "total_pnl": 125.50
    },
    "portfolio": {
      "total_value": 1125.50,
      "realized_pnl": 125.50,
      "unrealized_pnl": 0.00,
      "open_positions_count": 0
    },
    "performance": {
      "win_rate": 0.72,
      "total_trades": 25,
      "total_pnl": 125.50
    }
  },
  "performance_history": [...],
  "error_log": [...],
  "portfolio_history": [...]
}
```

## 🛠️ Troubleshooting

### Common Issues

#### 1. API Key Errors

```
❌ Failed to initialize exchange: Invalid API key
```

**Solution**: Verify your API keys are correct and have trading permissions.

#### 2. Insufficient Balance

```
❌ Trade execution failed: Insufficient balance
```

**Solution**: Ensure you have sufficient balance in your exchange account.

#### 3. Rate Limiting

```
❌ Exchange rate limit exceeded
```

**Solution**: The system automatically handles rate limiting, but you may need to reduce trading frequency.

#### 4. Network Issues

```
❌ Network connection failed
```

**Solution**: Check your internet connection and exchange API status.

### Error Recovery

The system includes automatic error recovery:

- **Automatic Reconnection**: Reconnect to exchange on connection loss
- **Error Logging**: Log all errors for analysis
- **Circuit Breakers**: Stop trading on repeated errors
- **Safe Mode**: Reduce trading activity on high error rates

### Debug Mode

Enable debug mode for detailed logging:

```yaml
development:
  debug_mode: true
```

## 🔒 Security Best Practices

### API Key Security

1. **Use Sandbox Mode**: Always test with sandbox mode first
2. **Limit Permissions**: Use API keys with trading permissions only
3. **Environment Variables**: Store API keys in environment variables
4. **Regular Rotation**: Rotate API keys regularly
5. **IP Whitelisting**: Whitelist your IP address on the exchange

### System Security

1. **Firewall**: Configure firewall to allow only necessary connections
2. **VPN**: Use VPN for additional security
3. **Monitoring**: Monitor system access and trading activity
4. **Backups**: Regular backups of configuration and data
5. **Updates**: Keep system and dependencies updated

## 📚 Advanced Features

### Custom Strategies

The system supports custom trading strategies:

```python
# Custom strategy integration
from core.strategy import CustomStrategy

class MyStrategy(CustomStrategy):
    def generate_signals(self, market_data):
        # Your custom logic here
        return signals
```

### WebSocket Integration

For real-time data (future enhancement):

```yaml
websocket:
  enabled: true
  reconnect_attempts: 5
  reconnect_delay: 5
```

### Notifications

Configure notifications for important events:

```yaml
notifications:
  email_enabled: true
  email_address: "your@email.com"
  slack_enabled: true
  slack_webhook: "your_slack_webhook"
```

## 🚀 Deployment

### Production Deployment

1. **Server Setup**: Use a reliable VPS or cloud server
2. **Environment**: Python 3.8+ with required dependencies
3. **Process Management**: Use systemd or supervisor
4. **Logging**: Configure proper log rotation
5. **Monitoring**: Set up monitoring and alerting

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "main.py", "--production"]
```

### Systemd Service

```ini
[Unit]
Description=Schwabot Production Trading
After=network.target

[Service]
Type=simple
User=schwabot
WorkingDirectory=/opt/schwabot
Environment=SCHWABOT_EXCHANGE=coinbase
Environment=SCHWABOT_API_KEY=your_api_key
Environment=SCHWABOT_SECRET=your_secret
ExecStart=/usr/bin/python main.py --production
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

## 📞 Support

For support and questions:

1. **Documentation**: Check this guide and other documentation
2. **Logs**: Review system logs for error details
3. **Testing**: Use sandbox mode for testing
4. **Community**: Join the Schwabot community

## ⚠️ Disclaimer

This trading system is for educational and research purposes. Trading cryptocurrencies involves significant risk. Always:

- Test thoroughly in sandbox mode
- Start with small amounts
- Monitor system performance
- Understand the risks involved
- Never invest more than you can afford to lose

The developers are not responsible for any financial losses incurred through the use of this system. 