# Simplified Schwabot API

A clean, simplified API system that addresses key user concerns and provides an easy-to-use interface for the Schwabot trading system.

## 🎯 Key Improvements

### ✅ **Addresses User Concerns**
- **JSON Configuration**: Simple JSON instead of complex YAML files
- **Demo Mode**: Safe testing without live trading
- **Visual Interface**: Clean, modern web dashboard
- **Auto-Defaults**: Works out of the box with sensible defaults
- **Error Prevention**: Robust error handling and fallback mechanisms

### ✅ **User Experience Focus** 
- **Single Command Launch**: `python simplified_schwabot_launcher.py demo`
- **Built-in Dependency Checking**: Automatic verification of requirements
- **Live Data Integration**: Connects with existing 16-bit tick aggregator
- **Performance Optimized**: Handles 10,000 ticks/hour target

## 🚀 Quick Start

### 1. Launch Demo Mode (Recommended)
```bash
python simplified_schwabot_launcher.py demo
```
- ✅ Safe synthetic data
- ✅ No API keys required  
- ✅ Full functionality testing
- ✅ Web dashboard at http://localhost:8000

### 2. Configure System
```bash
python simplified_schwabot_launcher.py config
```
Interactive configuration with sensible defaults.

### 3. Check Status
```bash
python simplified_schwabot_launcher.py status
```
View system health and configuration.

## 📊 Web Dashboard Features

### Real-Time Dashboard
- **Connection Status**: Live WebSocket connection indicator
- **Trading Metrics**: Price, confidence, sustainment index
- **Demo Controls**: Start/stop different market scenarios
- **Trading Controls**: Safe demo trading controls
- **System Log**: Real-time error and status logging

### Demo Scenarios
- **📈 Trending Market**: Gradual upward price movement
- **🌊 Volatile Market**: High volatility testing
- **📉 Crash Test**: Market crash simulation

## 🔧 API Architecture

### Simplified API (`core/simplified_api.py`)
- **Single FastAPI Application**: Unified API endpoints
- **JSON Configuration**: Simple configuration management
- **WebSocket Streaming**: Real-time data updates
- **Demo Mode Built-in**: Synthetic data generation
- **Error Recovery**: Automatic fallback mechanisms

### BTC Integration (`core/simplified_btc_integration.py`)
- **Live Tick Processing**: High-frequency data pipeline
- **Batch Processing**: Optimized for 10,000 ticks/hour
- **CPU Optimization**: Multi-threaded processing
- **Rate Limiting**: Prevents system overload
- **Unified Math Integration**: Uses `schwabot_unified_math_v2.py`

### Simple Frontend (`frontend/simplified_client.html`)
- **Single HTML File**: No complex build process
- **Modern UI**: Clean, responsive design
- **Real-time Updates**: WebSocket integration
- **Mobile Friendly**: Responsive design

## 📁 File Structure

```
├── core/
│   ├── simplified_api.py              # Main API server
│   └── simplified_btc_integration.py  # BTC data integration
├── frontend/
│   └── simplified_client.html         # Web dashboard
├── simplified_schwabot_launcher.py    # Easy launcher
└── SIMPLIFIED_API_README.md          # This file
```

## 🔌 API Endpoints

### Core Endpoints
- `GET /` - API information and status
- `GET /config` - Get current configuration
- `POST /config` - Update configuration
- `GET /status` - System status and metrics
- `WebSocket /ws` - Real-time data stream

### Trading Controls
- `POST /trading` - Start/stop/pause trading
- `POST /demo` - Start demo scenarios

### Configuration Format
```json
{
  "demo_mode": true,
  "live_trading_enabled": false,
  "position_size_limit": 0.1,
  "api_port": 8000,
  "websocket_update_interval": 0.5,
  "max_drawdown": 0.05,
  "stop_loss": 0.02,
  "demo_speed_multiplier": 1.0,
  "synthetic_data_enabled": true,
  "sustainment_threshold": 0.65,
  "confidence_threshold": 0.70
}
```

## 📈 Live Data Integration

### High-Frequency Processing
```python
# Example: Ingest live tick data
btc_integration.ingest_live_tick(
    price=50000.0,
    volume=1500.0,
    bid=49999.0,
    ask=50001.0
)
```

### Performance Metrics
- **Ticks/Second**: Real-time processing rate
- **Latency**: Average processing latency in ms
- **CPU Usage**: System resource monitoring
- **Error Rate**: Error tracking and recovery

## 🛡️ Safety Features

### Demo Mode Safety
- **No Live Trading**: Demo mode prevents real trades
- **Synthetic Data**: Safe testing environment
- **Rate Limiting**: Prevents system overload
- **Error Recovery**: Automatic fallback to safe states

### Risk Management
- **Position Limits**: Configurable position size limits
- **Stop Loss**: Automatic stop loss triggers
- **Drawdown Limits**: Maximum drawdown protection
- **Sustainment Monitoring**: Real-time sustainment index

## 🔧 Integration with Existing System

### Unified Math Integration
```python
from schwabot_unified_math_v2 import (
    UnifiedQuantumTradingController,
    calculate_btc_processor_metrics
)

# Seamless integration with rigorous mathematics
controller = UnifiedQuantumTradingController()
result = controller.evaluate_trade_opportunity(price, volume, market_state)
```

### BTC Processor Bridge
- **Seamless Integration**: Works with existing `btc_data_processor.py`
- **Quantum Core**: Integrates with `quantum_btc_intelligence_core.py`
- **Mathematical Foundation**: Uses `schwabot_unified_math_v2.py`

## 🚨 Error Prevention

### Robust Error Handling
- **Graceful Degradation**: Falls back to mock mode when core systems unavailable
- **Input Validation**: Validates all input data
- **Rate Limiting**: Prevents resource exhaustion
- **Automatic Recovery**: Self-healing error recovery

### Logging and Monitoring
- **Real-time Logs**: Web dashboard shows live system logs
- **Error Tracking**: Comprehensive error tracking
- **Performance Monitoring**: Real-time performance metrics
- **System Health**: Continuous health monitoring

## 📝 Usage Examples

### Basic Demo Usage
```bash
# Start demo mode
python simplified_schwabot_launcher.py demo

# Open web browser to http://localhost:8000
# Click "Start Demo" and select scenario
# Watch real-time trading metrics
```

### Live Integration
```bash
# Start with live BTC integration
python simplified_schwabot_launcher.py live

# System processes live tick data
# Unified math system evaluates trades
# Real-time sustainment monitoring
```

### Configuration
```bash
# Interactive configuration
python simplified_schwabot_launcher.py config

# Adjust risk settings
# Change API port
# Toggle demo mode
# Reset to defaults
```

## 🔍 Monitoring and Debugging

### Real-Time Monitoring
- **WebSocket Dashboard**: Live system monitoring
- **Processing Metrics**: Ticks/second, latency, errors
- **Trading Metrics**: Confidence, sustainment, execution signals
- **System Status**: Component health and connectivity

### Debug Mode
```bash
python simplified_schwabot_launcher.py demo --debug
```
Enables detailed logging for troubleshooting.

## 🎯 Performance Targets

### Tick Processing
- **Target**: 10,000 ticks/hour (2.78 ticks/second)
- **Batch Processing**: 100-tick batches for efficiency
- **Queue Management**: 1000-tick buffer with overflow protection
- **Latency Target**: <50ms average processing time

### API Performance
- **WebSocket Updates**: 2 FPS (500ms intervals)
- **Concurrent Clients**: Support for multiple dashboard connections
- **Memory Management**: Efficient queue and buffer management
- **CPU Optimization**: Multi-threaded processing pipeline

## 🔐 Security Considerations

### Safe Defaults
- **Demo Mode Default**: System starts in safe demo mode
- **Live Trading Disabled**: Requires explicit configuration
- **Input Validation**: All inputs validated before processing
- **Rate Limiting**: Prevents abuse and resource exhaustion

### Configuration Security
- **Local Storage**: Configuration stored in user home directory
- **No Sensitive Data**: No API keys or credentials in configuration
- **Sandboxed Operation**: Demo mode runs in isolated environment

## 🎉 Getting Started Guide

### Step 1: Check Dependencies
```bash
python simplified_schwabot_launcher.py status
```

### Step 2: Start Demo
```bash
python simplified_schwabot_launcher.py demo
```

### Step 3: Open Dashboard
Navigate to http://localhost:8000

### Step 4: Test Demo Scenarios
- Select "Trending Market" scenario
- Set speed multiplier to 2.0 for faster simulation
- Duration: 5 minutes
- Click "Start Demo"

### Step 5: Monitor Metrics
Watch real-time updates of:
- Current Price
- Confidence Level
- Sustainment Index
- Trading Signals

### Step 6: Explore Configuration
```bash
python simplified_schwabot_launcher.py config
```

## 🤝 Integration Points

### Existing System Integration
- **BTC Data Processor**: `btc_data_processor.py`
- **Quantum Intelligence**: `quantum_btc_intelligence_core.py` 
- **Unified Math**: `schwabot_unified_math_v2.py`
- **Configuration**: Replaces complex YAML with simple JSON

### Migration Path
1. **Start with Demo**: Test new system in demo mode
2. **Verify Integration**: Ensure compatibility with existing components
3. **Gradual Migration**: Move from complex to simplified API
4. **Production Deployment**: Enable live mode when ready

---

## 📞 Support

For questions or issues with the Simplified API system, check:
- System logs in the web dashboard
- Debug mode output: `--debug` flag
- Configuration issues: `simplified_schwabot_launcher.py config`
- Status check: `simplified_schwabot_launcher.py status`

**The Simplified API addresses all user concerns while maintaining the mathematical rigor and functionality of the complete Schwabot system.** 