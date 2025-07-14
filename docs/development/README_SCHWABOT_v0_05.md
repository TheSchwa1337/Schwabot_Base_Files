# Schwabot v0.05 - Advanced Trading System

## 🚀 Overview

Schwabot v0.05 is a comprehensive, modular trading system designed for cryptocurrency markets with advanced features including fractal pattern recognition, matrix-based decision making, and intelligent fallback mechanisms.

**"When it's together, it's together."** - This is the minimum unified build threshold for Schwabot v0.05.

## 🧩 Core Functional Modules

### ✅ Strategy Mapper (`strategy_mapper.py`)
- **Compliant + layered hash/bit strategy routing**
- Intelligent strategy selection based on market conditions
- Dynamic strategy activation and deactivation
- Strategy performance tracking and optimization

### ✅ Ferris RDE (`ferris_rde.py`)
- **Modular Ferris wheel engine with tick, pivot, ascent, descent logic**
- Cyclical trading pattern management
- Market phase detection and signal generation
- Adaptive cycle timing based on market volatility

### ✅ Profit Cycle Allocator (`profit_cycle_allocator.py`)
- **Multi-stage, recursive profit allocation**
- Intelligent profit distribution strategies
- Risk-adjusted allocation algorithms
- Portfolio rebalancing and reinvestment logic

### ✅ Wallet Tracker (`wallet_tracker.py`)
- **Tracks BTC, ETH, XRP, USDC, SOL positions with PNL + long-hold ledger**
- Comprehensive portfolio management
- Real-time PNL calculation and tracking
- Transaction history and portfolio snapshots

### ✅ Fallback Logic (`fallback_logic.py`)
- **Re-entry logic when stalled**
- Intelligent fallback mechanisms and recovery strategies
- Stall detection and automatic recovery
- System health monitoring and error handling

### ✅ Matrix Map Logic (`matrix_map_logic.py`)
- **Logic hash selection based on matrix similarity**
- Intelligent matrix-based decision making
- Pattern similarity metrics and correlation analysis
- Advanced matrix operations and transformations

### ✅ Glyph VM (`glyph_vm.py`)
- **Glyph drift visualizer/debug terminal output**
- Real-time system state visualization
- Pattern recognition and drift detection
- Terminal-based monitoring interface

### ✅ Fractal Core (`fractal_core.py`)
- **Fractal matching and error correction**
- Fractal pattern recognition and analysis
- Data correction and smoothing algorithms
- Trend, cyclic, and oscillatory pattern detection

### ✅ Matrix Fault Resolver (`matrix_fault_resolver.py`)
- **Matrix fault resolution and error correction**
- Numerical stability analysis and improvement
- Fault detection and automatic resolution
- Matrix health monitoring and optimization

## 🔐 Security & Environment

- **AES or local-hashed .env key handling**
- **Git-safe deploy structure (recursive build-safe)**
- **Flask bridge with R1, Claude, GPT-4o hash interchange**
- **Public/private API failover**
- **vault_logic.py for ColdBase & 72-hour drip tracking**

## 📊 Market Data & API Integration

- **✅ CCXT integration (Coinbase, Binance, Kraken optional)**
- **✅ Chainlink price API fallback**
- **✅ Smart money replay system + GAN anomaly check**
- **⏳ Orderbook delta velocity logic (still experimental)**
- **⏳ Liquidity vacuum detection (WIP)**

## 🔁 Trade Loop Architecture

```
init/ folder is root of operations
Ferris wheel cycles command through:
├── Strategy trigger layer
├── Logic matrix selection
├── AI command hash ping
├── Real trade validation
└── Profit echo registered → recursive feedback
```

## 🧠 External AI Logic Loop

- **GPT-4o**: hourly/daily/week cycle validation & hash approval
- **Claude**: macro mid-term bag logic
- **R1**: fast reactive decisions (GPU-local, low-latency)

## 🔍 Internal Intelligence Requirements

- **Recursive echo stack**
- **SHA-256-to-strategy basket mapping**
- **Pattern similarity metrics**
- **Glyph state tracking**
- **Matrix health monitoring**
- **Fractal pattern recognition**

## 📦 Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional, for GPU acceleration)
- Git

### Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd schwabot-v0.05

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests
python test_schwabot_v0_05.py
```

### Configuration

1. Copy the example configuration:
```bash
cp config/config.example.yaml config/config.yaml
```

2. Edit `config/config.yaml` with your settings:
```yaml
# Exchange Configuration
exchange:
  name: "coinbase"
  api_key: "${COINBASE_API_KEY}"
  secret: "${COINBASE_SECRET}"
  sandbox: true

# Trading Parameters
trading:
  risk_level: 0.02
  max_position_size: 0.1
  stop_loss: 0.05
  take_profit: 0.15

# System Configuration
system:
  log_level: "INFO"
  data_dir: "./data"
  backup_interval: 3600
```

3. Set up environment variables:
```bash
# Create .env file
cp .env.example .env

# Edit .env with your API keys
COINBASE_API_KEY=your_api_key_here
COINBASE_SECRET=your_secret_here
```

## 🚀 Usage

### Basic Usage

```python
from schwabot.core.strategy_mapper import StrategyMapper
from schwabot.core.ferris_rde import FerrisRDE
from schwabot.core.wallet_tracker import WalletTracker

# Initialize components
strategy_mapper = StrategyMapper()
ferris_rde = FerrisRDE()
wallet_tracker = WalletTracker()

# Start trading cycle
cycle = ferris_rde.start_cycle("main_cycle")

# Process market data
market_data = {
    "price": 50000,
    "volume": 1000000,
    "volatility": 0.02
}

# Update Ferris wheel phase
phase = ferris_rde.update_phase(market_data)

# Generate trading signal
signal = ferris_rde.generate_signal(market_data)

if signal and signal.signal_type == "buy":
    # Execute trade
    position = wallet_tracker.add_position(
        AssetType.BTC, PositionType.LONG, 0.1, market_data["price"]
    )
```

### Advanced Usage

```python
from schwabot.core.matrix_map_logic import MatrixMapLogic
from schwabot.core.fractal_core import FractalCore
from schwabot.core.glyph_vm import GlyphVM

# Initialize advanced components
matrix_logic = MatrixMapLogic()
fractal_core = FractalCore()
glyph_vm = GlyphVM()

# Create market feature matrix
market_matrix = np.random.rand(10, 10)
matrix_logic.add_matrix("market_features", MatrixType.FEATURE, market_matrix)

# Analyze fractal patterns
fractal_data = np.random.rand(100)
fractal_core.add_fractal("price_fractal", FractalType.PRICE, FractalState.ACTIVE, fractal_data)
patterns = fractal_core.detect_patterns()

# Update glyph visualization
glyph_vm.update_glyph("trading_performance", 0.7)
glyph_vm.update_glyph("strategy_confidence", 0.8)

# Start glyph display loop
glyph_vm.start_display_loop(interval=2.0)
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
python test_schwabot_v0_05.py
```

The test suite covers:
- ✅ Strategy Mapper functionality
- ✅ Ferris RDE cycle management
- ✅ Profit Cycle Allocator
- ✅ Wallet Tracker operations
- ✅ Fallback Logic mechanisms
- ✅ Glyph VM visualization
- ✅ Matrix Map Logic
- ✅ Fractal Core pattern recognition
- ✅ Matrix Fault Resolver
- ✅ Integration tests

## 📊 Monitoring

### Glyph VM Terminal Display

The Glyph VM provides real-time terminal visualization:

```
================================================================================
🔮 GLYPH VM - SCHWABOT v0.05
Timestamp: 2024-01-15 14:30:25
================================================================================
📊 GLYPH STATUS:
----------------------------------------
⚙️ system_health           [████████████████████] 1.000 (active)
💰 trading_performance     [████████████████░░░░] 0.700 (stable)
🎯 strategy_confidence     [████████████████████] 0.800 (active)
📊 profit_margin          [░░░░░░░░░░░░░░░░░░░░] 0.000 (stable)
❌ error_rate             [░░░░░░░░░░░░░░░░░░░░] 0.000 (inactive)
⚠️ warning_level          [░░░░░░░░░░░░░░░░░░░░] 0.000 (inactive)

🎯 PATTERN SUMMARY:
----------------------------------------
• drift (confidence: 0.75)
• state_transition (confidence: 0.60)

📈 SYSTEM METRICS:
----------------------------------------
Total Glyphs: 6
Total Patterns: 2
Drift Detections: 3
History Size: 50
Active Glyphs: 3
================================================================================
```

### Health Monitoring

Monitor system health through the Matrix Fault Resolver:

```python
# Analyze matrix health
health = fault_resolver.analyze_matrix_health("market_matrix", matrix_data)
print(f"Health Score: {health.health_score:.3f}")
print(f"Condition Number: {health.condition_number:.2e}")
print(f"Rank: {health.rank}")
```

## 🔧 Configuration

### Strategy Configuration

```yaml
strategies:
  hash_based:
    enabled: true
    parameters:
      hash_threshold: 0.7
      confidence_min: 0.6
      max_positions: 5
  
  bit_based:
    enabled: true
    parameters:
      bit_threshold: 0.8
      volatility_weight: 0.3
      momentum_weight: 0.7
```

### Ferris RDE Configuration

```yaml
ferris_rde:
  cycle_duration: 3600  # 1 hour
  phase_thresholds:
    tick_duration: 300
    pivot_threshold: 0.1
    ascent_threshold: 0.05
    descent_threshold: 0.05
  
  signal_generation:
    min_confidence: 0.7
    max_signals_per_cycle: 10
```

### Fallback Configuration

```yaml
fallback_logic:
  stall_detection:
    trading_stall_threshold: 300
    api_stall_threshold: 60
    strategy_stall_threshold: 600
  
  recovery_strategies:
    re_entry:
      enabled: true
      max_attempts: 3
      backoff_factor: 2.0
    
    strategy_switch:
      enabled: true
      fallback_strategies: ["conservative", "scalping", "swing"]
```

## 📈 Performance Metrics

### Key Performance Indicators

- **Strategy Success Rate**: Percentage of profitable strategies
- **Ferris Cycle Efficiency**: Average profit per cycle
- **Fallback Recovery Rate**: Success rate of automatic recovery
- **Matrix Health Score**: Average matrix condition across system
- **Fractal Pattern Accuracy**: Pattern recognition precision
- **Glyph Drift Rate**: System stability metrics

### Monitoring Dashboard

Access real-time metrics through the Glyph VM or export data for external monitoring:

```python
# Export system data
wallet_tracker.export_portfolio_data("portfolio_export.json")
glyph_vm.export_glyph_data("glyph_export.json")
matrix_logic.export_matrix_data("matrix_export.json")
fractal_core.export_fractal_data("fractal_export.json")
fault_resolver.export_fault_data("fault_export.json")
```

## 🔒 Security Features

### API Key Management

- Encrypted storage of API keys
- Environment variable overrides
- Automatic key rotation
- Failover between testnet and live environments

### Risk Management

- Position size limits
- Stop-loss and take-profit automation
- Maximum drawdown protection
- Portfolio diversification rules

### System Safety

- Automatic fallback mechanisms
- Error recovery and system restart
- Data backup and recovery
- Audit logging and monitoring

## 🚀 Deployment

### Production Deployment

1. **Environment Setup**:
```bash
# Set production environment
export SCHWABOT_ENV=production
export SCHWABOT_LOG_LEVEL=WARNING
```

2. **Service Configuration**:
```bash
# Create systemd service (Linux)
sudo cp schwabot.service /etc/systemd/system/
sudo systemctl enable schwabot
sudo systemctl start schwabot
```

3. **Monitoring Setup**:
```bash
# Start monitoring
python -m schwabot.monitoring.start_monitoring
```

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "main.py"]
```

```bash
# Build and run
docker build -t schwabot-v0.05 .
docker run -d --name schwabot schwabot-v0.05
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run linting
flake8 schwabot/
black schwabot/
mypy schwabot/

# Run tests
pytest tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. Trading cryptocurrencies involves substantial risk of loss. Use at your own risk.

## 🆘 Support

- **Documentation**: [Wiki](link-to-wiki)
- **Issues**: [GitHub Issues](link-to-issues)
- **Discussions**: [GitHub Discussions](link-to-discussions)
- **Email**: support@schwabot.com

## 🎯 Roadmap

### v0.06 (Next Release)
- [ ] Advanced AI integration (GPT-4o, Claude, R1)
- [ ] Enhanced pattern recognition
- [ ] Multi-exchange arbitrage
- [ ] Advanced risk management
- [ ] Web-based dashboard

### v0.07 (Future)
- [ ] Machine learning model training
- [ ] Real-time market prediction
- [ ] Advanced portfolio optimization
- [ ] Social trading features
- [ ] Mobile application

---

**"When it's together, it's together."** - Schwabot v0.05 is the minimum unified build threshold for a comprehensive, intelligent trading system. 