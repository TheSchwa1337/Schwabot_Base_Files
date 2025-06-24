# 🧠 Schwabot Trading System

## Overview

Schwabot is a **hardware-scale-aware economic kernel** capable of federating diverse devices (Chromebooks, Raspberry Pis, gaming laptops, servers) coordinated via a Flask server and secured with hardware trust modules. It represents the culmination of advanced mathematical trading algorithms with real-time monitoring and distributed architecture.

## 🚀 Features

### Mathematical Foundation
- **Phantom Lag Model** - Quantifies opportunity cost from missed trading signals
- **Meta-Layer Ghost Bridge** - Manages recursive hash echo memory across ghost layers
- **Enhanced Fallback Logic Router** - Mathematical integration with context-aware routing
- **Hash Registry Manager** - Signal memory management with dual-pathway support
- **Tensor Harness Matrix** - Phase-drift-safe routing with mathematical consistency
- **Voltage Lane Mapper** - Bit-depth to voltage mapping for hardware optimization

### Real-time Trading
- **Multi-exchange Support** - Binance, Coinbase, Kraken integration
- **Arbitrage Detection** - Cross-exchange opportunity identification
- **Risk Management** - Comprehensive position sizing and drawdown protection
- **Smart Order Routing** - Intelligent execution across multiple venues

### User Interface
- **Web Dashboard** - Real-time monitoring with live charts
- **RESTful API** - Programmatic access to all system functions
- **Socket.IO Integration** - Real-time updates and notifications
- **Responsive Design** - Works on desktop, tablet, and mobile

### Distributed Architecture
- **Hardware Federation** - Support for multiple device types
- **Scalable Deployment** - From single laptop to server clusters
- **Load Balancing** - Automatic distribution across available resources
- **Fault Tolerance** - Graceful degradation and recovery

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Dashboard │    │   API Server    │    │  Real-time WS   │
│   (Flask/UI)    │    │   (REST/JSON)   │    │  (Socket.IO)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │  Settings Mgr   │
                    │  (YAML/Env)     │
                    └─────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Phantom Lag     │    │ Meta-Layer      │    │ Fallback Logic  │
│ Model           │    │ Ghost Bridge    │    │ Router          │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │ System Integ.   │
                    │ Orchestrator    │
                    └─────────────────┘
                                 │
                    ┌─────────────────┐
                    │ Exchange APIs   │
                    │ (Binance/etc)   │
                    └─────────────────┘
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd schwabot

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Set up your environment variables:

```bash
# Required: Exchange API credentials
export BINANCE_API_KEY="your_binance_api_key"
export BINANCE_API_SECRET="your_binance_api_secret"
export COINBASE_API_KEY="your_coinbase_api_key"
export COINBASE_API_SECRET="your_coinbase_api_secret"
export KRAKEN_API_KEY="your_kraken_api_key"
export KRAKEN_API_SECRET="your_kraken_api_secret"

# Optional: Notifications
export SLACK_WEBHOOK_URL="your_slack_webhook"
export TELEGRAM_BOT_TOKEN="your_telegram_token"
```

### 3. Run Schwabot

```bash
# Start the complete system
python run_schwabot.py
```

### 4. Access Dashboard

Open your web browser and navigate to:
- **Dashboard**: http://localhost:8080
- **API Documentation**: http://localhost:8081/docs

## 📊 Dashboard Features

### Real-time Monitoring
- **System Status** - Live component health monitoring
- **Performance Metrics** - Portfolio value, profit/loss, success rate
- **Trading Activity** - Real-time trade execution and status
- **Mathematical Components** - Phantom Lag Model, Ghost Bridge status

### Configuration Management
- **Settings Interface** - Modify system parameters in real-time
- **Exchange Configuration** - Enable/disable exchanges and set limits
- **Risk Management** - Adjust position sizing and drawdown limits
- **Performance Tuning** - Optimize mathematical component parameters

### Analytics & Reporting
- **Performance Charts** - Historical portfolio performance
- **Component Statistics** - Mathematical model effectiveness
- **Error Monitoring** - System alerts and fallback statistics
- **Resource Usage** - CPU, memory, and network utilization

## 🔧 Configuration

### Main Configuration File
Edit `config/schwabot_config.yaml` to customize:

```yaml
system:
  name: "Schwabot Trading System"
  environment: "production"
  debug_mode: false
  log_level: "INFO"

trading:
  default_symbol: "BTC/USD"
  supported_symbols: ["BTC/USD", "ETH/USD", "ADA/USD"]
  position_sizing:
    max_position_size_usd: 10000
    risk_per_trade_pct: 2.0

exchanges:
  binance:
    enabled: true
    sandbox_mode: true
  coinbase:
    enabled: true
    sandbox_mode: true

user_interface:
  web_dashboard:
    enabled: true
    port: 8080
  api_server:
    enabled: true
    port: 8081
```

### Mathematical Component Settings

#### Phantom Lag Model
```yaml
mathematical_components:
  phantom_lag_model:
    enabled: true
    max_history_size: 1000
    decay_lambda: 0.01
    min_penalty_threshold: 0.1
    enable_adaptive_learning: true
```

#### Meta-Layer Ghost Bridge
```yaml
mathematical_components:
  meta_layer_ghost_bridge:
    enabled: true
    decay_lambda: 0.1
    sync_threshold: 0.002
    enable_arbitrage_detection: true
    min_profit_threshold: 0.001
```

## 📈 Mathematical Foundation

### Phantom Lag Model
The Phantom Lag Model quantifies the opportunity cost from missed trading signals using exponential decay and entropy compensation:

```
P(Δp, λ, P₀) = 1 - exp(-λ * |Δp| / P₀)
```

Where:
- `P` = Phantom lag penalty (0 to 1)
- `Δp` = Price difference from missed opportunity
- `λ` = Decay parameter
- `P₀` = Reference price level

### Meta-Layer Ghost Bridge
The Meta-Layer Ghost Bridge manages recursive hash echo memory across ghost layers for arbitrage detection:

```
G(t) = Σᵢ αᵢ * exp(-βᵢ * t) * Hᵢ(t)
```

Where:
- `G(t)` = Ghost price at time t
- `αᵢ` = Weight for layer i
- `βᵢ` = Decay rate for layer i
- `Hᵢ(t)` = Hash echo at layer i

## 🔒 Security

### Environment-based Configuration
- All API keys stored in environment variables
- No hardcoded secrets in source code
- Secure credential management

### Network Security
- HTTPS support for production deployment
- API authentication and rate limiting
- CORS configuration for web dashboard
- Request logging and monitoring

### Mathematical Validation
- All calculations maintain mathematical consistency
- Input validation on all mathematical operations
- Error handling with graceful degradation

## 🧪 Testing

### Unit Tests
```bash
# Run mathematical component tests
python -m pytest tests/test_mathematical_components.py -v

# Run integration tests
python -m pytest tests/test_integration.py -v
```

### Validation Scripts
```bash
# Validate all components
python validate_components.py

# Run comprehensive system validation
python system_validation.py
```

### Code Quality
```bash
# Run linting
flake8 core/ ui/

# Run type checking
mypy core/ --config-file=mypy.ini
```

## 📚 Documentation

- **[Mathematical Integration Summary](MATHEMATICAL_INTEGRATION_SUMMARY.md)** - Detailed mathematical foundation
- **[Distributed System Summary](DISTRIBUTED_SYSTEM_SUMMARY.md)** - Hardware federation architecture
- **[Production Readiness Checklist](PRODUCTION_READINESS_CHECKLIST.md)** - Deployment guide

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes. Trading cryptocurrencies involves substantial risk of loss. Use at your own risk.

## 🆘 Support

For support and questions:
- Check the documentation
- Review the mathematical foundation papers
- Open an issue on GitHub

---

**🧠 Schwabot - Where Mathematics Meets Trading Intelligence** 