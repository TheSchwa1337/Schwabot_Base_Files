# 🧠 Schwabot Live Trading System

## Overview

Schwabot is an advanced cryptocurrency trading system that uses **2-gram pattern detection** as its foundational vectorization layer for real-time strategy routing and decision making. The system integrates multiple mathematical frameworks including entropy signals, fractal memory, and quantum mathematical bridges for comprehensive trading intelligence.

## 🎯 Core Features

### 2-Gram Pattern Detection
- **Micro-pattern recognition** for strategy routing
- **Burst scoring** for signal intensity measurement
- **Fractal memory integration** for historical pattern matching
- **T-cell health monitoring** for system protection
- **Real-time sequence analysis** for market direction detection

### Live Trading Integration
- **Multi-exchange support** (Coinbase, Binance, Kraken)
- **Real-time market data processing**
- **Strategy trigger routing** with dual-state optimization
- **Portfolio balancing** and risk management
- **Performance monitoring** and logging

### Mathematical Frameworks
- **Entropy signal processing** for timing optimization
- **Fractal memory systems** for pattern resonance
- **Quantum mathematical bridges** for advanced calculations
- **Tensor algebra** for vectorized operations
- **Phantom detection** for market anomaly identification

## 🚀 Quick Start

### 1. System Requirements

```bash
# Python 3.8 or higher
python --version

# Required directories (created automatically)
logs/
data/
backups/
config/
```

### 2. Configuration Setup

```bash
# Copy the live trading configuration
cp config/schwabot_live_trading_config.yaml config/my_config.yaml

# Edit configuration for your needs
nano config/my_config.yaml
```

### 3. Environment Variables (Live Mode)

```bash
# For live trading, set your API keys
export COINBASE_API_KEY="your_coinbase_api_key"
export COINBASE_SECRET="your_coinbase_secret"
export BINANCE_API_KEY="your_binance_api_key"
export BINANCE_SECRET="your_binance_secret"
export KRAKEN_API_KEY="your_kraken_api_key"
export KRAKEN_SECRET="your_kraken_secret"
```

### 4. Start the System

```bash
# Demo mode (no real trades)
python start_schwabot_live.py --mode demo

# Live mode (real trades)
python start_schwabot_live.py --mode live --config config/my_config.yaml

# Backtest mode
python start_schwabot_live.py --mode backtest
```

## 📊 System Architecture

### 2-Gram Detection Layer
```
Market Data → Sequence Conversion → 2-Gram Analysis → Strategy Triggers
     ↓              ↓                    ↓                ↓
Price/Volume → Character Sequence → Pattern Detection → Strategy Routing
```

### Component Integration
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  2-Gram Detector│───▶│Strategy Router  │───▶│Trading Pipeline │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│Fractal Memory   │    │Portfolio Balancer│    │Execution Engine │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│Entropy Signals  │    │Risk Management  │    │CCXT Integration │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## ⚙️ Configuration

### 2-Gram Detector Settings

```yaml
2gram_config:
  window_size: 100              # Rolling window for pattern analysis
  burst_threshold: 2.0          # Threshold for burst detection
  similarity_threshold: 0.85    # Cosine similarity threshold
  t_cell_sensitivity: 0.7       # T-cell immune response sensitivity
  enable_fractal_memory: true   # Enable fractal memory integration
  pattern_memory_size: 1000     # Maximum patterns to remember
  health_check_interval: 30.0   # Health check frequency (seconds)
```

### Trading Parameters

```yaml
default_symbol: "BTC/USDC"      # Default trading pair
initial_capital: 10000.0        # Starting capital
max_position_size: 0.1          # Maximum position size (10%)
stop_loss_pct: 0.02             # Stop loss percentage (2%)
take_profit_pct: 0.04           # Take profit percentage (4%)
```

### Risk Management

```yaml
risk_management:
  global_risk_limits:
    max_total_exposure: 0.8     # Maximum total exposure (80%)
    max_single_position: 0.2    # Maximum single position (20%)
    max_correlation: 0.7        # Maximum correlation between positions
    min_diversification: 3      # Minimum number of positions
```

## 🧬 2-Gram Pattern Types

### Volatility Patterns
- **UD**: Up-Down volatility reversal
- **DU**: Down-Up reversal momentum
- **UU**: Sustained uptrend momentum
- **DD**: Sustained downtrend

### Asset Swap Patterns
- **BE**: BTC-ETH swap arbitrage
- **EB**: ETH-BTC swap reversal
- **BU**: BTC surge pattern
- **EU**: ETH surge pattern

### Anomaly Patterns
- **AA**: Flatline anomaly (caution mode)
- **ZZ**: Dead market pattern
- **XX**: Unknown pattern warning
- **EE**: Entropy spike event

### System Health Patterns
- **OK**: System healthy
- **ER**: Error condition
- **WN**: Warning condition
- **CC**: Consolidation pattern

## 📈 Strategy Routing

### Pattern → Strategy Mapping

```python
strategy_mappings = {
    "UD": "volatility_reversal_entry",
    "DU": "reversal_momentum_entry", 
    "BE": "swap_arbitrage_trigger",
    "EB": "swap_reversal_trigger",
    "UU": "trend_momentum_entry",
    "DD": "downtrend_reversal_watch",
    "AA": "flatline_caution_mode",
    "EE": "entropy_spike_response"
}
```

### Execution Priority System

1. **Priority 10**: System protection (T-cell activation)
2. **Priority 9**: High-risk anomaly responses
3. **Priority 8**: Trend momentum strategies
4. **Priority 7**: Volatility reversal strategies
5. **Priority 6**: Reversal momentum strategies
6. **Priority 5**: Arbitrage triggers
7. **Priority 4**: Portfolio rebalancing

## 🔍 Monitoring and Logging

### Real-Time Monitoring

```bash
# View live logs
tail -f logs/schwabot_live.log

# Monitor system status
grep "Status:" logs/schwabot_live.log

# Check 2-gram detections
grep "2-gram" logs/schwabot_live.log
```

### Performance Metrics

- **Total Trades**: Number of executed trades
- **Total Profit**: Cumulative profit/loss
- **Win Rate**: Percentage of profitable trades
- **Max Drawdown**: Maximum portfolio decline
- **Sharpe Ratio**: Risk-adjusted returns
- **Active Positions**: Current open positions

### Health Monitoring

- **2-Gram Detector Health**: Pattern detection system status
- **T-Cell Activation**: System protection status
- **Fractal Memory**: Historical pattern matching health
- **Entropy Signal Quality**: Signal processing health

## 🛡️ Safety Features

### Demo Mode
- **No real trades executed**
- **Simulated market data**
- **Full system testing**
- **Strategy validation**

### Risk Controls
- **Position size limits**
- **Daily loss limits**
- **Correlation limits**
- **Volatility thresholds**
- **T-cell protection system**

### Emergency Shutdown
```bash
# Graceful shutdown with Ctrl+C
# All positions closed automatically
# System state saved
```

## 🔧 Advanced Configuration

### Entropy Signal Integration

```yaml
entropy_executor_config:
  entropy_signal_enabled: true
  entropy_confidence_threshold: 0.7
  entropy_timing_cycles: [5, 15, 30, 60]  # seconds
  entropy_integration_weight: 0.3
  fractal_memory_enabled: true
  phantom_detection_enabled: true
```

### Fractal Memory Settings

```yaml
advanced:
  two_gram_advanced:
    pattern_weight_decay: 0.95
    burst_score_smoothing: 0.8
    fractal_memory_retention: 1000
    t_cell_activation_threshold: 0.8
    health_score_decay: 0.99
```

## 📊 Example Output

### System Startup
```
🧠 SCHWABOT TRADING SYSTEM v2.0
================================

Advanced cryptocurrency trading system with:
• 2-Gram Pattern Detection & Strategy Routing
• Real-Time Market Data Processing
• Multi-Exchange Trading Execution
• Entropy-Enhanced Decision Making
• Fractal Memory & Phantom Detection
• Portfolio Balancing & Risk Management

🔍 Checking system requirements...
✅ System requirements check completed
🔐 Validating exchange configuration...
✅ Exchange configuration validation completed

📋 Configuration Summary:
   Mode: demo
   Symbol: BTC/USDC
   Initial Capital: $10,000.00
   Safe Mode: True
   2-Gram Detector:
     Window Size: 100
     Burst Threshold: 2.0
     Fractal Memory: True
   Enabled Exchanges: 3
     - coinbase
     - binance
     - kraken

🔧 Initializing Schwabot trading system...
🧬 Initializing 2-gram detector...
🎯 Initializing strategy trigger router...
🔄 Initializing trading pipeline...
⚡ Initializing real-time execution engine...
🧠 Initializing entropy-enhanced trading executor...
⚖️ Initializing portfolio balancer...
💰 Initializing BTC/USDC integration...
🎛️ Initializing master profit coordination system...
🔗 Component injection completed
✅ Schwabot system initialization completed

🏥 Running system health check...
   2-Gram Detector: healthy
   Strategy Router: Active
   Trading Pipeline: active
   Execution Engine: Active
✅ System health check completed

🚀 Starting live trading operations...
🎮 Running in DEMO mode - no real trades will be executed
✅ Live trading started successfully
✅ Schwabot trading system is now running!
Press Ctrl+C to stop the system gracefully.

🧬 Detected 3 significant 2-gram patterns
  Pattern: UD, Burst: 2.3, Priority: 7
  Pattern: BE, Burst: 1.9, Priority: 5
  Pattern: UU, Burst: 3.1, Priority: 8

🎯 Strategy triggered: volatility_reversal_entry from pattern ⚡UD
🎯 Strategy triggered: swap_arbitrage_trigger from pattern 🔁BE
🎯 Strategy triggered: trend_momentum_entry from pattern 📈UU

📊 Status: 5 trades, $127.50 profit
```

## 🚨 Troubleshooting

### Common Issues

1. **Configuration Errors**
   ```bash
   # Check configuration syntax
   python -c "import yaml; yaml.safe_load(open('config/my_config.yaml'))"
   ```

2. **Missing Dependencies**
   ```bash
   # Install required packages
   pip install -r requirements.txt
   ```

3. **API Key Issues**
   ```bash
   # Verify environment variables
   echo $COINBASE_API_KEY
   echo $BINANCE_API_KEY
   ```

4. **Permission Issues**
   ```bash
   # Ensure write permissions
   chmod 755 logs/ data/ backups/
   ```

### Debug Mode

```bash
# Enable debug logging
python start_schwabot_live.py --mode demo --log-level DEBUG
```

## 📚 API Documentation

### 2-Gram Detector API

```python
from core.two_gram_detector import create_two_gram_detector

# Create detector
detector = create_two_gram_detector({
    "window_size": 100,
    "burst_threshold": 2.0
})

# Analyze sequence
signals = await detector.analyze_sequence("UUDDUDUDBEEBBEAAAZZXREEUUDDBEUUDE")

# Get statistics
stats = await detector.get_pattern_statistics()

# Health check
health = await detector.health_check()
```

### Strategy Router API

```python
from core.strategy_trigger_router import create_strategy_trigger_router

# Create router
router = create_strategy_trigger_router({
    "execution_mode": "demo"
})

# Process market data
triggers = await router.process_market_data(market_data)

# Execute trigger
result = await router.execute_trigger(trigger)
```

## ⚠️ Risk Disclaimer

**This software is for educational and research purposes only. Cryptocurrency trading involves substantial risk of loss and is not suitable for all investors. Past performance does not guarantee future results. Always test thoroughly in demo mode before using real funds.**

## 📞 Support

For technical support or questions about the 2-gram detection system:

1. Check the logs for detailed error messages
2. Review the configuration settings
3. Test in demo mode first
4. Monitor system health metrics

## 🔄 Updates

The system is continuously updated with:
- Enhanced 2-gram pattern detection
- Improved strategy routing algorithms
- New exchange integrations
- Advanced risk management features
- Performance optimizations

---

**🧠 Schwabot Trading System - Advanced 2-Gram Pattern Detection for Cryptocurrency Trading** 