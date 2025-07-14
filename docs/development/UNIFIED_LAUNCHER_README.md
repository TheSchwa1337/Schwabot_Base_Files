# Schwabot Unified Visual Command Platform v0.5

A comprehensive unified launcher that respects and integrates Schwabot's unique architecture into a coherent visual platform.

## 🎯 Core Philosophy

Unlike traditional trading bots, Schwabot has unique components that don't map to standard paradigms. This unified launcher recognizes and enhances these unique features:

- **Settings as Plugins**: Configuration files act as modular plugin systems
- **Tests as Benchmarks**: Test files function as performance validation systems  
- **Flask/Ngrok as Devices**: Servers and tunnels provide device-like connectivity
- **BTC Processor as Mining Dashboard**: Built-in cryptocurrency processing capabilities
- **Tick Mapping as Task Management**: Hash-driven tick resolution replaces traditional task management
- **Internal Hash Actions**: Real-time monitoring of Schwabot's unique mathematical processes

## 🚀 Features

### Visual Command Center
- **Ferris Wheel Interface**: Central hub with rotating quadrants for each major system
- **Tabbed Organization**: Clean separation of different component types
- **Resource Monitoring**: Real-time CPU/GPU/Memory tracking
- **Status Dashboard**: Live system health and component status

### Component Integration
- **25+ Schwabot Components**: Full integration with existing systems
- **Real-time Health Monitoring**: Component status tracking and error handling
- **Process Management**: Start/stop/restart individual components
- **Configuration Management**: Dynamic loading of YAML/JSON/Python configs

### Unique Schwabot Mappings
- **Plugin System**: Mathematical framework, high-frequency trading, immune system
- **Benchmark System**: Precision profit tests, T-cell validation, ferris wheel backtests
- **Device System**: Flask API servers, Ngrok tunnels, price feed integration
- **Processor System**: BTC mining pools, quantum static core, Galileo tensor operations
- **Manager System**: Live execution mapping, speed lattice trading, hash recollection

## 📁 Directory Structure

```
schwabot/
├── schwabot_unified_launcher.py      # Main launcher GUI
├── core/unified_component_bridge.py  # Component integration bridge
├── launch_schwabot_unified.py        # Quick start script
├── requirements_unified.txt          # Enhanced dependencies
├── data/                             # Historical data integration
│   ├── historical/                   # CSV data storage
│   │   ├── btc_usdc/
│   │   ├── eth_usdc/
│   │   └── xrp_usdc/
│   ├── preprocessed/                 # Parquet files
│   ├── live/                         # Real-time data
│   └── cache/                        # Temporary files
├── core/                             # Existing Schwabot core
├── hash_recollection/                # Memory and pattern systems
├── config/                           # Configuration files
└── logs/                             # System logs
```

## 🛠️ Installation & Setup

### 1. Install Dependencies

```bash
# Install enhanced requirements for unified launcher
pip install -r requirements_unified.txt

# Core dependencies include:
# - tkinter (GUI framework)
# - pandas, numpy (data processing)
# - flask (web server)
# - psutil (system monitoring)
# - yaml, json (configuration)
# - matplotlib, plotly (visualization)
```

### 2. Initialize Data Directories

```bash
# Run the launcher once to auto-create directories
python launch_schwabot_unified.py --skip-checks

# Or manually create:
mkdir -p data/historical/{btc_usdc,eth_usdc,xrp_usdc}
mkdir -p data/{preprocessed,live,cache}
mkdir -p {logs,config}
```

### 3. Add Historical Data (Optional)

```bash
# Place CSV files in appropriate directories:
data/historical/btc_usdc/btc_2023_data.csv
data/historical/eth_usdc/eth_2023_data.csv
data/historical/xrp_usdc/xrp_2023_data.csv

# CSV format:
# timestamp,open,high,low,close,volume
# 2023-01-01 00:00:00,16500,16600,16400,16550,125.4
```

## 🚀 Usage

### Quick Start

```bash
# Launch with all features
python launch_schwabot_unified.py

# Simulation mode (default)
python launch_schwabot_unified.py --simulation-mode

# Live trading mode (requires API keys)
python launch_schwabot_unified.py --live-mode

# Debug mode with verbose logging
python launch_schwabot_unified.py --log-level DEBUG --debug-ui

# Skip pre-flight checks for faster startup
python launch_schwabot_unified.py --skip-checks
```

### Interface Navigation

#### 🎯 Command Hub Tab
- **Ferris Wheel Center**: Start/pause/restart Schwabot
- **Quadrant Navigation**: Click quadrants to jump to specific tabs
- **System Status**: Live overview of component health
- **Resource Monitor**: CPU/GPU/Memory usage

#### 🔌 Plugins Tab (Settings as Plugins)
- **Mathematical Framework**: Tensor integration, quantum static core
- **High-Frequency Trading**: Speed lattice, tick resolution
- **System Interlinking**: Ghost layer, Ferris RDE
- **Immune System**: T-cell signaling, biological protection
- **Profit Vectorization**: Multi-decimal, SHA256 hashing

#### 📊 Benchmarks Tab (Tests as Benchmarks)
- **Precision Profit Integration**: BTC simulation testing
- **Enhanced T-Cell System**: Immune system validation
- **Ferris Wheel Backtest**: Historical performance testing
- **Mathematical Integration**: Tensor algebra validation
- **Complete System Integration**: End-to-end testing

#### 📱 Devices Tab (Flask/Ngrok as Devices)
- **Flask API Server**: Web interface and API endpoints
- **Ngrok Tunnel**: Public access and remote control
- **Price Feed Integration**: Real-time market data
- **Wallet Integration**: Blockchain connectivity

#### ⚙️ Processors Tab (Including BTC Mining)
- **BTC Mining Pool Dashboard**: Pool management and statistics
- **Enhanced Master Cycle**: Profit optimization engine
- **Quantum Static Core**: QSC gates and validation
- **Galileo Tensor Bridge**: Mathematical modeling

#### 📋 Task Managers Tab (Tick Mapping as Tasks)
- **Live Execution Mapper**: Trade execution and portfolio tracking
- **Speed Lattice Trading**: Tick resolution and hash mapping
- **Hash Recollection System**: Pattern tracking and entropy analysis
- **Temporal Intelligence**: Historical analysis and prediction
- **Risk Manager**: Portfolio protection and position sizing

#### 🖥️ System Monitor Tab
- **Resource Usage**: Detailed CPU/GPU/Memory metrics
- **System Logs**: Real-time log monitoring
- **Performance Metrics**: Component-specific statistics
- **Health Dashboard**: Overall system health overview

## 🎨 Ferris Wheel Interface

The central Ferris wheel represents Schwabot's core philosophy:

### Quadrants
- **Lower Left**: Recursive Memory Log (hash recollection, pattern storage)
- **Upper Left**: Strategy AI Echo (decision making, signal processing)  
- **Upper Right**: Profit Delta Visualization (real-time profit tracking)
- **Lower Right**: System Internals (Ferris wheel cycle, phase management)

### Center Controls
- **Play/Pause**: Start or pause the entire Schwabot system
- **Settings Gear**: Access global configuration
- **Lock Icon**: Security and safe mode controls

### Interactive Features
- **Click Quadrants**: Navigate directly to specific system tabs
- **Rotation Animation**: Visual indicator of system activity
- **Color Coding**: Health status indication (green=healthy, yellow=warning, red=error)

## 🔧 Configuration

### Plugin Configuration
```yaml
# config/high_frequency_btc_trading_config.yaml
speed_lattice:
  enabled: true
  tick_resolution: 0.25
  hash_tracking: true

profit_optimization:
  multi_decimal: true
  precision_levels: [2, 6, 8]
  sha256_hashing: true
```

### Benchmark Configuration
```python
# Test files automatically discovered and treated as benchmarks
# Located in root directory: test_*.py
# Metrics extracted from stdout/stderr
```

### Device Configuration
```json
{
  "flask_server": {
    "port": 5000,
    "host": "localhost",
    "debug": false
  },
  "ngrok": {
    "auth_token": "your_token_here",
    "region": "us"
  }
}
```

## 🔍 Monitoring & Debugging

### System Health
- **Component Status**: Real-time health of all 25+ components
- **Process Monitoring**: Active process tracking and management
- **Resource Usage**: CPU/GPU/Memory utilization
- **Error Tracking**: Centralized error logging and reporting

### Debug Features
```bash
# Enable debug mode for verbose logging
python launch_schwabot_unified.py --debug-ui --log-level DEBUG

# Check logs
tail -f schwabot_unified.log

# Component health check
python -c "
from core.unified_component_bridge import get_component_bridge
bridge = get_component_bridge()
print(bridge.get_system_overview())
"
```

## 🤝 Integration with Existing Schwabot

This unified launcher is designed to work seamlessly with your existing Schwabot v0.048b installation:

### Preserved Functionality
- All existing mathematical frameworks remain intact
- Profit vectorization systems continue operating
- Hash recollection and pattern matching unchanged
- Biological immune system fully functional
- Quantum static core operations maintained

### Enhanced Features
- Visual monitoring of internal processes
- Centralized component management
- Historical data integration for backtesting
- Resource monitoring and optimization
- Streamlined development and debugging

### Migration Path
1. Install unified launcher alongside existing system
2. Test in simulation mode first
3. Gradually migrate components to unified management
4. Maintain fallback to original CLI interfaces

## 🚨 Troubleshooting

### Common Issues

#### GUI Not Starting
```bash
# Check tkinter installation
python -c "import tkinter; print('tkinter OK')"

# Install if missing (Ubuntu/Debian)
sudo apt-get install python3-tk

# Install if missing (macOS)
brew install python-tk
```

#### Component Import Errors
```bash
# Check Python path
python -c "import sys; print(sys.path)"

# Add current directory
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Or use absolute imports
python launch_schwabot_unified.py --skip-checks
```

#### Memory/Performance Issues
```bash
# Monitor resource usage
python -c "
import psutil
print(f'CPU: {psutil.cpu_percent()}%')
print(f'Memory: {psutil.virtual_memory().percent}%')
"

# Reduce component load
# Disable unused plugins/processors in GUI
```

### Support

For issues specific to the unified launcher:
1. Check `schwabot_unified.log` for error details
2. Run with `--debug-ui --log-level DEBUG` for verbose output
3. Test components individually using the component bridge
4. Verify all dependencies in `requirements_unified.txt`

## 🛡️ Security Considerations

- **Simulation Mode**: Default mode prevents live trading
- **API Key Protection**: Store sensitive keys in environment variables
- **Process Isolation**: Components run in separate processes when possible
- **Error Containment**: Component failures don't crash entire system
- **Audit Logging**: All actions logged for accountability

## 🎯 Future Enhancements

- **Web Dashboard**: Browser-based interface option
- **Mobile App**: Companion mobile monitoring app
- **Advanced Analytics**: Enhanced performance metrics
- **AI Integration**: Machine learning for component optimization
- **Cloud Deployment**: Distributed component architecture

---

**Schwabot Unified Launcher v0.5** - Respecting the unique, enhancing the possible. 