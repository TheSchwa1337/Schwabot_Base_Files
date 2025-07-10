# Schwabot Trading System - Fully Operational Trading Platform

A comprehensive, production-ready trading system that provides both CLI and API access for cryptocurrency trading with advanced mathematical models, real-time market data, and extensive integration capabilities.

## 🚀 Features

### Core Trading System
- **Real-time trading** with multiple exchanges (Binance, Coinbase Pro, Kraken, etc.)
- **Advanced mathematical models** including entropy-based algorithms
- **Portfolio management** with real-time tracking and P&L analysis
- **Risk management** with configurable limits and alerts
- **Hot reloading** of subsystems for live updates

### API & Integration
- **REST API server** with FastAPI for programmatic access
- **WebSocket support** for real-time data streaming
- **Multiple integrations**: ngrok, Glassnode, Whale Watcher, Telegram, Discord
- **API key management** with secure storage and rotation
- **Comprehensive monitoring** with health checks and metrics

### Mathematical Framework
- **Tensor algebra** for advanced calculations
- **Quantum mathematical bridge** for quantum-inspired algorithms
- **Entropy-based trading** with drift detection
- **Symbolic mathematics** with SymPy integration
- **Machine learning** models with TensorFlow and PyTorch

## 📋 Requirements

- Python 3.8+
- 8GB+ RAM (16GB+ recommended)
- Stable internet connection
- API keys for desired exchanges

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd AOI_Base_Files_Schwabot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Setup configuration**
   ```bash
   cp config/production.env.template config/production.env
   # Edit config/production.env with your settings
   ```

4. **Create necessary directories**
   ```bash
   python main.py --help
   # This will create required directories automatically
   ```

## 🚀 Quick Start

### 1. Basic System Startup
```bash
# Start the system
python main.py start

# Start with API server
python main.py start --api

# Check system status
python main.py status
```

### 2. API Key Management
```bash
# Add API key for Binance
python main.py api-keys add --exchange binance --key YOUR_API_KEY --secret YOUR_SECRET

# List configured exchanges
python main.py api-keys list

# Remove API key
python main.py api-keys remove --exchange binance
```

### 3. Trading Operations
```bash
# Place a market order
python main.py order --symbol BTC/USDT --side buy --type market --quantity 0.01

# Place a limit order
python main.py order --symbol BTC/USDT --side sell --type limit --quantity 0.01 --price 50000

# Check portfolio
python main.py portfolio
```

### 4. System Management
```bash
# Hot-reload subsystems
python main.py reload

# List all subsystems
python main.py subsystems

# Stop the system
python main.py stop
```

## 🌐 API Usage

### Starting the API Server
```bash
python main.py start --api --api-host 0.0.0.0 --api-port 8000
```

### API Endpoints

#### System Status
```bash
curl http://localhost:8000/status
```

#### Place Order
```bash
curl -X POST http://localhost:8000/order \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTC/USDT",
    "side": "buy",
    "order_type": "market",
    "quantity": 0.01
  }'
```

#### Portfolio Summary
```bash
curl http://localhost:8000/portfolio
```

#### Subsystems List
```bash
curl http://localhost:8000/subsystems
```

### Interactive API Documentation
Once the API server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🔧 Configuration

### Main Configuration (`config/schwabot_config.yaml`)
```yaml
# Trading configuration
trading:
  default_exchange: "binance"
  trading_pairs: ["BTC/USDT", "ETH/USDT"]
  max_order_size: 1000
  risk_limit: 0.02

# Mathematical framework
math:
  entropy_threshold: 0.7
  tensor_precision: "float64"
  quantum_enabled: true

# API server
api:
  host: "0.0.0.0"
  port: 8000
  cors_enabled: true
```

### Integration Configuration (`config/integrations.yaml`)
```yaml
# Ngrok tunnel
ngrok:
  enabled: true
  authtoken: "your_ngrok_token"
  port: 8000

# Glassnode analytics
glassnode:
  enabled: true
  api_key: "your_glassnode_key"

# Whale Watcher
whale_watcher:
  enabled: true
  api_key: "your_whale_watcher_key"
```

## 🔌 Integrations

### Available Integrations

1. **Ngrok** - Expose API server publicly
2. **Glassnode** - On-chain analytics and metrics
3. **Whale Watcher** - Large transaction monitoring
4. **Telegram** - Notifications and alerts
5. **Discord** - Webhook notifications
6. **Email** - SMTP notifications
7. **TradingView** - Alert webhooks
8. **CoinGecko** - Market data (free)
9. **Multiple Exchanges** - Binance, Coinbase Pro, Kraken, FTX

### Setting Up Integrations
```bash
# Setup all integrations
python main.py integrations setup

# Check integration status
python main.py integrations status
```

## 📊 Monitoring & Logging

### Log Files
- `logs/schwabot_system.log` - Main system log
- `logs/schwabot_cli.log` - CLI operations log
- `logs/trading.log` - Trading operations log

### Health Monitoring
```bash
# Check system health
curl http://localhost:8000/health

# Monitor in real-time
tail -f logs/schwabot_system.log
```

### Metrics
The system exposes Prometheus metrics at `/metrics` when monitoring is enabled.

## 🔒 Security

### API Key Security
- API keys are encrypted and stored securely
- Automatic key rotation support
- Sandbox mode for testing
- IP whitelisting capabilities

### Best Practices
1. Use environment variables for sensitive data
2. Enable 2FA where possible
3. Use sandbox/testnet for development
4. Regularly rotate API keys
5. Monitor for suspicious activity

## 🧪 Testing

### Quick Test
Run the comprehensive test suite to validate system functionality:

```bash
python test_ci_functionality.py
```

This test suite validates:
- Core module imports and functionality
- System initialization and configuration
- Registry system and trading pipeline
- CLI functionality and help system
- Directory structure and file integrity

### Unit Tests
```bash
pytest test/unit/
```

### Integration Tests
```bash
pytest test/integration/
```

### Legacy Tests
```bash
python test_unified_trading_pipeline.py
```

## 🤖 Continuous Integration

The repository includes comprehensive CI/CD testing that validates:

- **Dependencies**: Cross-platform package installation
- **Code Quality**: Flake8 syntax and style checks
- **Imports**: Core module availability and loading
- **Functionality**: System initialization and basic operations
- **CLI Interface**: Command-line help and basic commands
- **Configuration**: YAML/JSON config file loading
- **Structure**: Required directories and files

CI runs on every push and pull request, ensuring code quality and functionality.

## 📈 Performance

### Optimization Tips
1. Use SSD storage for better I/O performance
2. Enable GPU acceleration for mathematical calculations
3. Configure appropriate cache settings
4. Monitor memory usage and adjust limits
5. Use connection pooling for database operations

### Resource Requirements
- **Minimum**: 4GB RAM, 2 CPU cores
- **Recommended**: 16GB RAM, 8 CPU cores
- **Production**: 32GB+ RAM, 16+ CPU cores

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install -r requirements.txt --upgrade
   ```

2. **API Connection Issues**
   ```bash
   # Check API keys
   python main.py api-keys list
   
   # Test connection
   python main.py status
   ```

3. **Memory Issues**
   ```bash
   # Check memory usage
   python -c "import psutil; print(psutil.virtual_memory())"
   ```

4. **Port Conflicts**
   ```bash
   # Use different port
   python main.py start --api --api-port 8001
   ```

### Debug Mode
```bash
python main.py start --log-level DEBUG
```

## 📚 Documentation

- [System Architecture](docs/architecture.md)
- [API Reference](docs/api_reference.md)
- [Trading Strategies](docs/trading_strategies.md)
- [Mathematical Framework](docs/mathematical_framework.md)
- [Deployment Guide](docs/PRODUCTION_DEPLOYMENT_GUIDE.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes. Trading cryptocurrencies involves substantial risk of loss. Use at your own risk. The authors are not responsible for any financial losses.

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **Documentation**: [Wiki](https://github.com/your-repo/wiki)

---

**Schwabot Trading System** - Advanced cryptocurrency trading with mathematical precision.

# Schwabot_Base_Files

## Automated Testing

The main test for this repository is now:

```bash
python test_ci_functionality.py
```

This comprehensive test script validates all core functionality including:

- **Dependencies**: Verifies all required packages are available
- **Core Imports**: Tests that all core modules can be imported successfully
- **System Initialization**: Validates the trading system can be initialized
- **Registry System**: Tests trade registry and coordination functionality  
- **Pipeline Integration**: Validates the unified trading pipeline
- **Configuration**: Tests YAML/JSON configuration loading
- **CLI Interface**: Verifies command-line help and basic functionality
- **Directory Structure**: Ensures all required files and folders exist
- **Syntax Validation**: Checks Python syntax across core files

### Legacy Tests

The following tests are also available but are supplementary:

```bash
python test_unified_trading_pipeline.py  # Legacy trading pipeline tests
```

## Running Tests Locally

To run the full test suite:

```bash
python test_ci_functionality.py
```

## Continuous Integration

GitHub Actions CI is configured to run comprehensive tests on every push and pull request:

1. **Main CI Workflow** (`.github/workflows/python-ci.yml`)
   - Runs on Python 3.10 and 3.11
   - Installs dependencies with cross-platform compatibility
   - Executes the comprehensive test suite
   - Validates code quality with Flake8 and Black
   - Tests CLI functionality

2. **Functional Test Workflow** (`.github/workflows/ci.yml`)
   - Quick functional validation
   - Supplementary legacy test execution

The CI system handles cross-platform dependency issues (like Windows-only packages) and provides clear validation that the system works correctly.

You can view the results in the Actions tab of the GitHub repository.
