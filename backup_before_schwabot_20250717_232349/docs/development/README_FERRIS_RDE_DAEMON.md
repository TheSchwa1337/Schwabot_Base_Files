# Ferris RDE Daemon - 24/7 Trading Pipeline

## Overview

The Ferris RDE Daemon is a comprehensive 24/7 trading system that integrates all components of the Schwabot pipeline for continuous market analysis and trading execution. It implements the Ferris Wheel RDE (Recursive Dualistic Engine) with advanced mathematical frameworks, multi-bit state management, and unified connectivity.

## 🎡 Key Features

### Core Functionality
- **24/7 Continuous Operation**: Runs continuously with automatic health monitoring and recovery
- **Ferris RDE Integration**: Implements cyclical trading patterns with phase transitions
- **Multi-Bit State Management**: Dynamic bit-depth processing (2-42 bits) with CPU/GPU support
- **Advanced Mathematical Frameworks**: Ferris Wheel, Quantum Thermal, Void Well metrics
- **Unified Connectivity**: API, trading, and visualization system integration

### Trading Capabilities
- **Real-time Market Data Processing**: Continuous order book vectorization and analysis
- **Entry/Exit Logic**: Intelligent signal generation with confidence scoring
- **Risk Management**: Kelly criterion, position sizing, stop-loss/take-profit
- **Paper Trading Mode**: Safe testing environment with simulated execution
- **Multi-Asset Support**: Primary (BTC/USD, ETH/USD) and secondary assets

### Mathematical Integration
- **Ferris Wheel State**: Time-phase rotational harmonic cycles
- **Quantum Thermal State**: Decohered quantum-thermal hybrid analysis
- **Void Well Metrics**: Price-volume fractal geometry analysis
- **Kelly Metrics**: Optimal probabilistic position sizing
- **Recursive Time Lock**: Multi-scale temporal synchronization

### Monitoring & Health
- **Comprehensive Health Checks**: API, trading, and visualization system monitoring
- **Performance Metrics**: Real-time processing time, throughput, and error tracking
- **Error Logging**: Detailed error tracking with automatic recovery
- **System Status API**: RESTful endpoints for monitoring and control

## 🚀 Quick Start

### Prerequisites
```bash
# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p logs data backups profiles
```

### Basic Usage

#### 1. Start the Daemon
```bash
# Start with default configuration
python start_ferris_daemon.py

# Start in test mode (faster intervals)
python start_ferris_daemon.py --test

# Start with custom configuration
python start_ferris_daemon.py --config config/custom_config.yaml

# Start with verbose logging
python start_ferris_daemon.py --verbose
```

#### 2. Run Tests
```bash
# Run comprehensive test suite
python test_ferris_rde_daemon.py

# Run quick demo
python test_ferris_rde_daemon.py demo
```

#### 3. Monitor Status
```bash
# Check daemon status (via API)
curl http://localhost:8081/api/status

# View logs
tail -f logs/ferris_rde_daemon.log
```

## 📁 File Structure

```
├── core/
│   ├── ferris_rde_daemon.py          # Main daemon implementation
│   ├── trading_pipeline_integration.py # Trading pipeline integration
│   ├── unified_connectivity_manager.py # Unified connectivity management
│   └── ...
├── config/
│   └── ferris_rde_daemon_config.yaml # Daemon configuration
├── start_ferris_daemon.py            # Startup script
├── test_ferris_rde_daemon.py         # Test suite
├── logs/                             # Log files
├── data/                             # Data storage
└── backups/                          # Backup files
```

## ⚙️ Configuration

### Configuration File Structure

The daemon uses YAML configuration files with the following structure:

```yaml
# Core Daemon Settings
daemon:
  name: "FerrisRDE"
  version: "1.0.0"
  enabled: true
  log_level: "INFO"

# Trading Configuration
trading:
  enabled: true
  paper_trading: true  # Set to false for live trading
  max_concurrent_trades: 10
  risk_management_enabled: true

# Processing Configuration
processing:
  enable_gpu: true
  enable_distributed: false
  bit_depth_range: [2, 42]

# Asset Configuration
assets:
  primary:
    - "BTC/USD"
    - "ETH/USD"
  secondary:
    - "XRP/USD"
    - "ADA/USD"

# Ferris RDE Configuration
ferris_rde:
  enabled: true
  cycle_duration_minutes: 60
  phase_transitions:
    tick_to_pivot: 0.8
    pivot_to_ascent: 0.7
    ascent_to_descent: 0.6
    descent_to_tick: 0.9
```

### Key Configuration Options

| Section | Option | Description | Default |
|---------|--------|-------------|---------|
| `daemon` | `name` | Daemon name | "FerrisRDE" |
| `daemon` | `log_level` | Logging level | "INFO" |
| `trading` | `paper_trading` | Enable paper trading | true |
| `trading` | `max_concurrent_trades` | Max concurrent trades | 10 |
| `processing` | `enable_gpu` | Enable GPU processing | true |
| `timing` | `tick_interval_seconds` | Tick processing interval | 1.0 |
| `assets` | `primary` | Primary trading assets | ["BTC/USD", "ETH/USD"] |

## 🔧 Advanced Usage

### Custom Configuration

Create a custom configuration file:

```yaml
# custom_config.yaml
daemon:
  name: "MyFerrisRDE"
  log_level: "DEBUG"

trading:
  paper_trading: false  # Live trading
  max_concurrent_trades: 5

assets:
  primary:
    - "BTC/USD"
  secondary:
    - "ETH/USD"
    - "ADA/USD"

ferris_rde:
  cycle_duration_minutes: 30  # Faster cycles
```

Start with custom configuration:
```bash
python start_ferris_daemon.py --config custom_config.yaml
```

### Programmatic Usage

```python
import asyncio
from core.ferris_rde_daemon import FerrisRDEDaemon, DaemonConfig

async def main():
    # Create custom configuration
    config = DaemonConfig(
        daemon_name="CustomFerrisRDE",
        tick_interval_seconds=2.0,
        primary_assets=["BTC/USD"],
        paper_trading=True
    )
    
    # Initialize and start daemon
    daemon = FerrisRDEDaemon(config)
    await daemon.start()
    
    # Run for 60 seconds
    await asyncio.sleep(60)
    
    # Stop daemon
    await daemon.stop()

# Run the daemon
asyncio.run(main())
```

### API Integration

The daemon exposes RESTful APIs for monitoring and control:

```python
import requests

# Get system status
response = requests.get("http://localhost:8081/api/status")
status = response.json()

# Get daemon metrics
response = requests.get("http://localhost:8081/api/metrics")
metrics = response.json()

# Get mathematical states
response = requests.get("http://localhost:8081/api/mathematical")
math_states = response.json()
```

## 📊 Monitoring & Metrics

### Health Monitoring

The daemon continuously monitors:
- **API Connectivity**: CoinGecko, CoinMarketCap, CCXT
- **Trading System**: Order book vectorizer, strategy mapper, entry/exit logic
- **Visualization**: Web dashboard, API server, real-time updates
- **Mathematical Components**: Ferris RDE, quantum thermal, void well metrics

### Performance Metrics

Key metrics tracked:
- **Processing Performance**: Tick processing time, signal generation time
- **Trading Performance**: Trades executed, win rate, profit/loss
- **System Performance**: CPU usage, memory usage, active connections
- **Error Tracking**: Error count, error types, recovery success rate

### Logging

The daemon provides comprehensive logging:
- **Main Log**: `logs/ferris_rde_daemon.log`
- **Error Log**: `logs/ferris_rde_errors.log`
- **Performance Log**: `logs/ferris_rde_performance.log`

## 🔄 Pipeline Integration

### Entry/Exit Logic

The daemon integrates with the complete trading pipeline:

1. **Market Data Fetching**: Real-time price and order book data
2. **Order Book Vectorization**: Multi-bit depth vectorization (2-42 bits)
3. **Strategy Bit Mapping**: Dynamic strategy expansion and mapping
4. **Mathematical Analysis**: Ferris wheel, quantum thermal, void well metrics
5. **Signal Generation**: Entry/exit signals with confidence scoring
6. **Risk Management**: Kelly criterion, position sizing, stop-loss
7. **Trade Execution**: Paper trading or live execution

### Mathematical Framework Integration

```python
# Ferris Wheel State
ferris_state = calculate_ferris_wheel_state(
    time_series=price_data,
    periods=[144, 288, 576],  # 1h, 2h, 4h
    current_time=time.time()
)

# Quantum Thermal State
quantum_state = calculate_quantum_thermal_state(
    quantum_state=initial_state,
    temperature=300.0
)

# Void Well Metrics
void_well = calculate_void_well_metrics(
    volume_data=volume_data,
    price_data=price_data
)

# Kelly Metrics
kelly = calculate_kelly_metrics(
    win_probability=0.6,
    expected_return=0.02,
    volatility=0.15
)
```

## 🛡️ Risk Management

### Built-in Risk Controls

1. **Position Sizing**: Kelly criterion with safety factor
2. **Stop Loss**: Automatic stop-loss placement (2% default)
3. **Take Profit**: Automatic take-profit placement (4% default)
4. **Daily Loss Limits**: Maximum daily loss protection
5. **Drawdown Protection**: Maximum drawdown limits
6. **Volatility Adjustment**: Dynamic position sizing based on volatility

### Risk Configuration

```yaml
trading:
  risk_management_enabled: true
  max_position_size: 0.25  # 25% max position
  max_daily_loss: 0.02    # 2% daily loss limit
  max_drawdown: 0.10      # 10% max drawdown
  stop_loss_percentage: 0.02
  take_profit_percentage: 0.04
```

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure all dependencies are installed
   pip install -r requirements.txt
   
   # Check Python path
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

2. **Configuration Errors**
   ```bash
   # Validate configuration
   python -c "import yaml; yaml.safe_load(open('config/ferris_rde_daemon_config.yaml'))"
   ```

3. **Permission Errors**
   ```bash
   # Create necessary directories
   mkdir -p logs data backups
   chmod 755 logs data backups
   ```

4. **API Connection Issues**
   ```bash
   # Check API connectivity
   curl https://api.coingecko.com/api/v3/ping
   ```

### Debug Mode

Enable debug logging for troubleshooting:

```bash
python start_ferris_daemon.py --verbose --test
```

### Log Analysis

```bash
# View recent errors
tail -f logs/ferris_rde_errors.log

# Search for specific errors
grep "ERROR" logs/ferris_rde_daemon.log

# Monitor performance
tail -f logs/ferris_rde_performance.log
```

## 🚀 Production Deployment

### System Requirements

- **CPU**: 4+ cores recommended
- **Memory**: 8GB+ RAM
- **Storage**: 10GB+ free space
- **Network**: Stable internet connection
- **OS**: Linux, macOS, or Windows

### Production Configuration

1. **Disable Paper Trading**
   ```yaml
   trading:
     paper_trading: false
   ```

2. **Enable Database Storage**
   ```yaml
   database:
     enabled: true
     type: "postgresql"
   ```

3. **Configure SSL/TLS**
   ```yaml
   security:
     ssl:
       enabled: true
       cert_file: "/path/to/cert.pem"
       key_file: "/path/to/key.pem"
   ```

4. **Set Up Monitoring**
   ```yaml
   monitoring:
     enable_health_monitoring: true
     enable_performance_tracking: true
   ```

### Process Management

Use systemd for Linux:

```ini
# /etc/systemd/system/ferris-rde.service
[Unit]
Description=Ferris RDE Daemon
After=network.target

[Service]
Type=simple
User=ferris
WorkingDirectory=/opt/ferris-rde
ExecStart=/usr/bin/python3 start_ferris_daemon.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable ferris-rde
sudo systemctl start ferris-rde
sudo systemctl status ferris-rde
```

## 📈 Performance Optimization

### GPU Acceleration

Enable GPU processing for better performance:

```yaml
processing:
  enable_gpu: true
  gpu_memory_limit_mb: 512
  gpu_batch_size: 64
```

### Memory Optimization

```yaml
advanced:
  memory:
    gc_threshold: 0.8
    max_memory_mb: 2048
    memory_check_interval_seconds: 60
```

### Async Optimization

```yaml
advanced:
  async:
    max_concurrent_tasks: 50
    task_timeout_seconds: 30
    retry_attempts: 3
```

## 🔗 Integration Examples

### Custom Strategy Integration

```python
from core.ferris_rde_daemon import FerrisRDEDaemon

class CustomStrategy:
    def __init__(self):
        self.daemon = FerrisRDEDaemon()
    
    async def run(self):
        await self.daemon.start()
        
        # Custom logic here
        while self.daemon.running:
            # Process custom signals
            await self.process_custom_signals()
            await asyncio.sleep(1)
    
    async def process_custom_signals(self):
        # Implement custom signal processing
        pass
```

### External System Integration

```python
import asyncio
import websockets

async def external_monitor():
    uri = "ws://localhost:8081/ws"
    async with websockets.connect(uri) as websocket:
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            
            # Process external data
            if data['type'] == 'trading_signal':
                await process_external_signal(data)
```

## 📚 API Reference

### Daemon Methods

- `start()`: Start the daemon
- `stop()`: Stop the daemon
- `get_daemon_status()`: Get comprehensive status
- `get_metrics()`: Get performance metrics

### Configuration Classes

- `DaemonConfig`: Main configuration class
- `DaemonMetrics`: Metrics tracking class

### Data Structures

- `TradingSignal`: Trading signal with mathematical properties
- `PortfolioState`: Portfolio state with mathematical tracking
- `TradeTick`: Individual trade tick data

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the logs for error details

---

**🎡 Ferris RDE Daemon** - Advanced 24/7 Trading Pipeline with Integrated Mathematical Frameworks 