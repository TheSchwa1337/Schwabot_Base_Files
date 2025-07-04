# Schwabot Unified CLI Documentation

## Overview

The Schwabot Unified CLI provides comprehensive command-line access to all Schwabot trading system components across Windows, macOS, and Linux platforms. This CLI system enables full control over CPU/GPU orchestration, tensor state management, system monitoring, and live trading operations.

## Quick Start

### 1. System Information
```bash
python schwabot_unified_cli_simple.py --system
```

### 2. Available Tools
```bash
python schwabot_unified_cli_simple.py --tools
```

### 3. Quick Start Guide
```bash
python schwabot_unified_cli_simple.py --quickstart
```

## Available CLI Tools

### 1. Dual State Router CLI (`dual-state`)
**Purpose**: Control profit-tiered dualistic compute orchestration (CPU/GPU routing)

**Key Features**:
- Monitor CPU/GPU routing decisions
- View profit registry and strategy metadata
- Control routing parameters
- Real-time performance monitoring
- Strategy tier management

**Usage**:
```bash
# Initialize the system
python schwabot_unified_cli_simple.py dual-state --init

# Show system status
python schwabot_unified_cli_simple.py dual-state --status

# Show profit registry
python schwabot_unified_cli_simple.py dual-state --registry

# Test task routing
python schwabot_unified_cli_simple.py dual-state --test market_analysis short 0.75

# Process BTC price data
python schwabot_unified_cli_simple.py dual-state --btc 50000 1000 0.15

# Show performance metrics
python schwabot_unified_cli_simple.py dual-state --performance

# Reset system
python schwabot_unified_cli_simple.py dual-state --reset

# Interactive mode
python schwabot_unified_cli_simple.py dual-state --interactive
```

### 2. Tensor State Manager CLI (`tensor`)
**Purpose**: Manage tensor states and BTC price processing

**Key Features**:
- Inspect tensor states and matrices
- Process BTC price data through tensor pipeline
- Manage tensor memory and cache
- Monitor tensor performance metrics
- Control tensor operations and calculations

**Usage**:
```bash
# Initialize the system
python schwabot_unified_cli_simple.py tensor --init

# Show system status
python schwabot_unified_cli_simple.py tensor --status

# Show tensor cache
python schwabot_unified_cli_simple.py tensor --cache

# Clear tensor cache
python schwabot_unified_cli_simple.py tensor --clear-cache

# Show performance metrics
python schwabot_unified_cli_simple.py tensor --performance

# Inspect specific tensor
python schwabot_unified_cli_simple.py tensor --inspect price_tensor

# Process BTC through tensor pipeline
python schwabot_unified_cli_simple.py tensor --btc 50000 1000 0.15

# Export tensor data
python schwabot_unified_cli_simple.py tensor --export price_tensor data.json

# Interactive mode
python schwabot_unified_cli_simple.py tensor --interactive
```

### 3. System Monitor CLI (`monitor`)
**Purpose**: Monitor system performance and health

**Key Features**:
- Real-time system performance monitoring
- CPU/GPU utilization tracking
- Memory and disk usage monitoring
- Trading performance metrics
- Network and API status
- System health diagnostics

**Usage**:
```bash
# Initialize the system
python schwabot_unified_cli_simple.py monitor --init

# Show system status
python schwabot_unified_cli_simple.py monitor --status

# Show performance metrics
python schwabot_unified_cli_simple.py monitor --performance

# Show trading metrics
python schwabot_unified_cli_simple.py monitor --trading

# Show health diagnostics
python schwabot_unified_cli_simple.py monitor --health

# Start real-time monitoring (5-second intervals)
python schwabot_unified_cli_simple.py monitor --monitor 5

# Export system report
python schwabot_unified_cli_simple.py monitor --export report.json

# Interactive mode
python schwabot_unified_cli_simple.py monitor --interactive
```

### 4. Live Trading CLI (`live`)
**Purpose**: Execute live trades through API connections

**Key Features**:
- Real-time trading execution
- API connection management
- Trade logging and registry
- Safe mode operation
- Market data integration

**Usage**:
```bash
# Execute single trade
python schwabot_unified_cli_simple.py live --mode execute-single --config config.json

# Start automated trading
python schwabot_unified_cli_simple.py live --mode start-bot --config config.json --interval 60

# Stop automated trading
python schwabot_unified_cli_simple.py live --mode stop-bot --config config.json

# Log trigger
python schwabot_unified_cli_simple.py live --mode log-trigger --registry-file registry.json

# Show best phase
python schwabot_unified_cli_simple.py live --mode best-phase --registry-file registry.json

# Safe mode operation
python schwabot_unified_cli_simple.py live --mode trade --config config.json --safe-mode
```

### 5. Strategy Management CLI (`strategy`)
**Purpose**: Manage trading strategies and bit mapping

**Key Features**:
- Schwafit pattern recognition
- Strategy selection and matching
- Matrix operations
- Live handler management
- Ghost trading simulation

**Usage**:
```bash
# Run Schwafit on price series
python schwabot_unified_cli_simple.py strategy fit --prices prices.csv --window 64

# Run mock test
python schwabot_unified_cli_simple.py strategy test

# Show status
python schwabot_unified_cli_simple.py strategy status

# Select strategy
python schwabot_unified_cli_simple.py strategy select-strategy --hash "0.1,0.2,0.3" --matrix-dir data/matrices

# Match matrix
python schwabot_unified_cli_simple.py strategy match-matrix --hash "0.1,0.2,0.3" --matrix-dir data/matrices

# Show live status
python schwabot_unified_cli_simple.py strategy live-status --matrix-dir data/matrices

# Spin Ferris wheel
python schwabot_unified_cli_simple.py strategy ferris-spin --matrix-dir data/matrices --ticks 24

# Simulate live tick
python schwabot_unified_cli_simple.py strategy live-tick --matrix-dir data/matrices --price 50000

# Calculate entry/exit
python schwabot_unified_cli_simple.py strategy entry-exit --matrix-dir data/matrices --hash "0.1,0.2,0.3"

# Simulate ghost trade
python schwabot_unified_cli_simple.py strategy ghost-trade --matrix-dir data/matrices --hash "0.1,0.2,0.3" --price 50000
```

### 6. Cross-Platform Validation CLI (`validate`)
**Purpose**: Validate CLI functionality across platforms

**Key Features**:
- Platform detection and compatibility
- CLI functionality testing
- Network connectivity validation
- Mathematical operations testing
- Cross-platform compatibility reports

**Usage**:
```bash
# Run all validation tests
python schwabot_unified_cli_simple.py validate --all

# Test platform detection
python schwabot_unified_cli_simple.py validate --platform

# Test CLI functionality
python schwabot_unified_cli_simple.py validate --cli

# Test network functionality
python schwabot_unified_cli_simple.py validate --network

# Test mathematical operations
python schwabot_unified_cli_simple.py validate --math

# Generate validation report
python schwabot_unified_cli_simple.py validate --report
```

## Interactive Mode

All CLI tools support interactive mode for enhanced user experience:

```bash
# Dual State Router interactive mode
python schwabot_unified_cli_simple.py dual-state --interactive

# Tensor State Manager interactive mode
python schwabot_unified_cli_simple.py tensor --interactive

# System Monitor interactive mode
python schwabot_unified_cli_simple.py monitor --interactive
```

### Interactive Commands

**Dual State Router Interactive Commands**:
- `status` - Show system status
- `registry` - Show profit registry
- `test <type> <tier> <density>` - Test task routing
- `btc <price> <volume> <vol>` - Process BTC price data
- `performance [strategy]` - Show performance metrics
- `reset` - Reset system
- `quit/exit` - Exit CLI

**Tensor State Manager Interactive Commands**:
- `status` - Show system status
- `cache` - Show tensor cache
- `clear-cache` - Clear tensor cache
- `performance` - Show performance metrics
- `inspect <tensor_name>` - Inspect specific tensor
- `btc <price> <volume> <vol>` - Process BTC through tensor pipeline
- `export <tensor> <file>` - Export tensor data
- `quit/exit` - Exit CLI

**System Monitor Interactive Commands**:
- `status` - Show system status
- `performance` - Show performance metrics
- `trading` - Show trading metrics
- `health` - Show health diagnostics
- `monitor [interval]` - Start real-time monitoring
- `export <file_path>` - Export system report
- `quit/exit` - Exit CLI

## Configuration Files

### Trading Configuration (`trading_bot_config.json`)
```json
{
  "symbol": "BTCUSDT",
  "initial_capital": 10000.0,
  "safe_mode": false,
  "market_data_config": {
    "update_interval": 60,
    "data_sources": ["binance", "coinbase"]
  },
  "exchange_config": {
    "api_key": "your_api_key",
    "api_secret": "your_api_secret",
    "exchange": "binance"
  },
  "registry_file": "data/trade_registry.json"
}
```

### Real Trading Configuration (`real_trading_config.json`)
```json
{
  "symbol": "BTCUSDT",
  "initial_capital": 1000.0,
  "safe_mode": true,
  "risk_management": {
    "max_position_size": 0.1,
    "stop_loss": 0.02,
    "take_profit": 0.05
  },
  "exchange_config": {
    "api_key": "your_api_key",
    "api_secret": "your_api_secret",
    "exchange": "binance",
    "testnet": true
  }
}
```

## File Structure

```
schwabot/
├── core/
│   ├── cli_dual_state_router.py      # CPU/GPU orchestration CLI
│   ├── cli_tensor_state_manager.py   # Tensor state management CLI
│   ├── cli_system_monitor.py         # System monitoring CLI
│   ├── cli_live_entry.py             # Live trading CLI
│   └── system/
│       └── dual_state_router.py      # Dual state router core
├── utils/
│   └── safe_print.py                 # Safe printing utilities
├── data/
│   ├── matrices/                     # Strategy matrices
│   └── cache/                        # System cache
├── schwabot_unified_cli_simple.py    # Main CLI launcher
├── schwabot_cli.py                   # Strategy management CLI
├── cross_platform_cli_validator.py   # Validation CLI
└── CLI_DOCUMENTATION.md              # This documentation
```

## Cross-Platform Compatibility

### Windows
- PowerShell and Command Prompt support
- Windows-specific path handling
- Process monitoring via psutil

### macOS
- Terminal and iTerm2 support
- Unix-style path handling
- System monitoring via psutil

### Linux
- Bash and other shell support
- Unix-style path handling
- System monitoring via psutil

## Error Handling

The CLI system includes comprehensive error handling:

1. **Import Errors**: Graceful fallback when modules are unavailable
2. **Network Errors**: Automatic retry with exponential backoff
3. **File System Errors**: Safe file operations with validation
4. **API Errors**: Proper error reporting and logging
5. **System Errors**: Platform-specific error handling

## Performance Monitoring

### Real-time Metrics
- CPU utilization percentage
- Memory usage and availability
- Disk space and I/O operations
- Network traffic and connectivity
- Trading system performance

### Historical Data
- Performance trend analysis
- System health over time
- Trading success rates
- Resource utilization patterns

## Security Features

1. **Safe Mode**: Prevents actual trading operations
2. **API Key Protection**: Secure storage and handling
3. **Input Validation**: Comprehensive parameter validation
4. **Error Logging**: Secure error reporting
5. **Access Control**: User permission management

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Check Python path
   python -c "import sys; print(sys.path)"
   
   # Install missing dependencies
   pip install -r requirements.txt
   ```

2. **Permission Errors**
   ```bash
   # Check file permissions
   ls -la schwabot_unified_cli_simple.py
   
   # Run with appropriate permissions
   sudo python schwabot_unified_cli_simple.py --system
   ```

3. **Network Connectivity**
   ```bash
   # Test network connectivity
   python schwabot_unified_cli_simple.py validate --network
   
   # Check firewall settings
   ```

4. **Memory Issues**
   ```bash
   # Clear system cache
   python schwabot_unified_cli_simple.py tensor --clear-cache
   
   # Monitor memory usage
   python schwabot_unified_cli_simple.py monitor --status
   ```

### Debug Mode

Enable debug mode for detailed logging:

```bash
# Set debug environment variable
export SCHWABOT_DEBUG=1

# Run CLI with debug output
python schwabot_unified_cli_simple.py --system
```

## Future Enhancements

1. **Web Interface**: Browser-based GUI for CLI tools
2. **Mobile App**: Mobile CLI interface
3. **Plugin System**: Extensible CLI plugin architecture
4. **Cloud Integration**: Cloud-based CLI deployment
5. **Advanced Analytics**: Enhanced performance analytics
6. **Multi-language Support**: Internationalization support

## Support and Maintenance

For support and maintenance:

1. **Documentation**: Refer to this documentation
2. **Logs**: Check system logs for error details
3. **Validation**: Run validation tests for system health
4. **Updates**: Keep CLI tools updated
5. **Community**: Join Schwabot community for support

---

**Version**: 1.0.0  
**Last Updated**: 2024  
**Compatibility**: Windows, macOS, Linux  
**Python Version**: 3.8+ 