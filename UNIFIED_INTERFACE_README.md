# 🌀 Schwabot Unified Trading Interface

## Overview

The Schwabot Unified Trading Interface is a comprehensive web-based trading terminal that provides real-time access to all Schwabot trading capabilities. This interface unifies strategy execution, profile management, visualization, and AI command processing into a single, intuitive dashboard.

## 🚀 Features

### Core Trading Features
- **Real-time Strategy Execution**: Execute trading strategies with confidence and signal strength controls
- **Multi-Profile Management**: Manage multiple Coinbase API profiles with independent trading logic
- **Live Trading Dashboard**: Real-time portfolio tracking, profit monitoring, and performance metrics
- **Session Mode Control**: Switch between demo, live, and backtest modes seamlessly

### Advanced Capabilities
- **AI Command Interface**: Process commands from Claude, GPT-4o, and R1 AI systems
- **GPU/CPU Runtime Switching**: Toggle between GPU acceleration and CPU processing
- **Real-time Visualization**: Generate and display trading charts, technical indicators, and pattern recognition
- **Hardware Auto-Detection**: Automatic detection and optimization for your hardware configuration

### System Integration
- **Socket.IO Real-time Communication**: Live updates and bidirectional communication
- **Strategy Status Monitoring**: Track active strategies and their execution status
- **System Logging**: Comprehensive logging with different severity levels
- **API Health Monitoring**: Real-time system status and health checks

## 📋 Prerequisites

### Required Python Packages
```bash
pip install flask flask-cors flask-socketio numpy matplotlib
```

### Optional Dependencies
```bash
pip install eventlet  # For better Socket.IO performance
pip install gunicorn  # For production deployment
```

### System Requirements
- Python 3.8+
- Modern web browser with JavaScript enabled
- Network access for real-time data feeds
- GPU support (optional, for acceleration)

## 🛠️ Installation & Setup

### 1. Clone and Navigate
```bash
cd AOI_Base_Files_Schwabot
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Profiles
Ensure your `config/coinbase_profiles.yaml` is properly configured with your API credentials.

### 4. Launch the Interface
```bash
python launch_unified_interface.py
```

### 5. Access the Dashboard
Open your browser and navigate to: `http://localhost:8080`

## 🎯 Usage Guide

### Dashboard Overview

The unified interface is organized into several key panels:

#### 1. **Metrics Dashboard**
- **Portfolio Value**: Current total portfolio value
- **Total Profit**: Cumulative profit/loss
- **Win Rate**: Percentage of successful trades
- **Active Strategies**: Number of currently executing strategies
- **Current Price**: Live price of selected asset
- **API Status**: Connection status to trading APIs

#### 2. **Strategy Execution Panel**
- **Strategy ID**: Unique identifier for the strategy
- **Profile Selection**: Choose which API profile to use
- **Asset Selection**: Select trading pair (BTC/USDC, ETH/USDC, etc.)
- **Confidence Slider**: Set strategy confidence level (0.1 - 1.0)
- **Signal Strength Slider**: Set signal strength (0.1 - 1.0)
- **Execute Button**: Start strategy execution
- **Stop All Button**: Halt all active strategies

#### 3. **Profile Management Panel**
- **Profile Cards**: Display all available trading profiles
- **Status Indicators**: Show active/inactive status
- **Trading Pairs**: List supported trading pairs per profile
- **Position Limits**: Maximum open positions per profile
- **Activate Button**: Enable specific profiles

#### 4. **Session Control Panel**
- **Trading Mode**: Switch between demo/live/backtest modes
- **GPU Toggle**: Enable/disable GPU acceleration
- **Hardware Info**: Display detected hardware capabilities
- **Refresh Status**: Update system status

#### 5. **Visualization Panel**
- **Chart Type**: Select visualization type (price chart, volume analysis, etc.)
- **Symbol Selection**: Choose asset for visualization
- **Generate Chart**: Create new visualization
- **Clear Chart**: Reset visualization area

#### 6. **AI Command Interface**
- **Command Hash Input**: Enter AI-generated command hashes
- **AI Source Selection**: Specify AI system (Claude, GPT-4o, R1)
- **Process Command**: Execute AI command

#### 7. **System Log Panel**
- **Real-time Logging**: Live system events and status updates
- **Color-coded Entries**: Different colors for info, warning, and error messages
- **Auto-scroll**: Automatically scrolls to latest entries

### Strategy Execution Workflow

1. **Select Strategy**: Enter a unique strategy ID
2. **Choose Profile**: Select the API profile to execute on
3. **Set Asset**: Choose the trading pair
4. **Adjust Parameters**: Set confidence and signal strength
5. **Execute**: Click "Execute Strategy" to start
6. **Monitor**: Watch the strategy status in real-time
7. **Review**: Check results in the system log

### Profile Management

1. **View Profiles**: All available profiles are displayed as cards
2. **Check Status**: Active profiles are highlighted
3. **Activate Profile**: Click "Activate" to enable a profile
4. **Monitor Performance**: Track profile-specific metrics
5. **Manage Positions**: View current positions and limits

### AI Command Processing

1. **Receive Command**: Get command hash from AI system
2. **Select Source**: Choose the AI system that generated the command
3. **Process**: Click "Process AI Command" to execute
4. **Monitor**: Watch for execution status updates
5. **Review**: Check results in the system log

## 🔧 Configuration

### Environment Variables
```bash
export SCHWABOT_DEBUG=true          # Enable debug mode
export SCHWABOT_HOST=0.0.0.0        # Bind host
export SCHWABOT_PORT=8080           # Bind port
export SCHWABOT_LOG_LEVEL=INFO      # Logging level
```

### Configuration Files

#### `config/unified_settings.yaml`
Main system configuration including:
- Core system settings
- Mathematical framework parameters
- Performance monitoring thresholds
- Integration settings

#### `config/coinbase_profiles.yaml`
API profile configuration:
- API keys and secrets
- Trading parameters
- Position limits
- Risk management settings

## 🚀 Advanced Usage

### Command Line Options
```bash
# Basic launch
python launch_unified_interface.py

# Custom host and port
python launch_unified_interface.py --host 127.0.0.1 --port 9000

# Debug mode
python launch_unified_interface.py --debug

# Skip startup banner
python launch_unified_interface.py --no-banner
```

### Production Deployment
```bash
# Using Gunicorn
gunicorn -w 4 -k eventlet -b 0.0.0.0:8080 gui.unified_schwabot_interface:app

# Using systemd service
sudo systemctl start schwabot-interface
sudo systemctl enable schwabot-interface
```

### API Endpoints

#### System Status
- `GET /api/system/status` - Get system status and metrics

#### Strategy Management
- `POST /api/strategy/execute` - Execute a trading strategy
- `GET /api/strategy/list` - List active strategies

#### Profile Management
- `GET /api/profile/list` - List available profiles
- `POST /api/profile/activate` - Activate a profile

#### Visualization
- `POST /api/visualization/generate` - Generate trading visualization

#### Hardware Control
- `POST /api/hardware/gpu_toggle` - Toggle GPU acceleration

#### AI Commands
- `POST /api/ai/command` - Process AI command

#### Session Control
- `POST /api/session/switch_mode` - Switch trading mode

## 🔍 Troubleshooting

### Common Issues

#### 1. Import Errors
**Problem**: Module import failures
**Solution**: Ensure all dependencies are installed and Python path is correct

#### 2. Configuration Errors
**Problem**: Missing or invalid configuration files
**Solution**: Check that `config/` directory contains required YAML files

#### 3. API Connection Issues
**Problem**: Cannot connect to trading APIs
**Solution**: Verify API credentials and network connectivity

#### 4. GPU Detection Issues
**Problem**: GPU not detected or acceleration not working
**Solution**: Check CUDA installation and hardware compatibility

#### 5. Socket.IO Connection Issues
**Problem**: Real-time updates not working
**Solution**: Check firewall settings and ensure eventlet is installed

### Debug Mode
Enable debug mode for detailed logging:
```bash
python launch_unified_interface.py --debug
```

### Log Files
Check log files for detailed error information:
- `logs/unified_interface.log` - Main application log
- `logs/schwabot_cli.log` - System-wide log

## 🔒 Security Considerations

### API Key Management
- Store API keys in environment variables or secure configuration files
- Never commit API keys to version control
- Use separate API keys for different profiles
- Regularly rotate API keys

### Network Security
- Use HTTPS in production environments
- Implement proper firewall rules
- Restrict access to trusted IP addresses
- Use VPN for remote access

### Data Protection
- Encrypt sensitive configuration data
- Implement proper session management
- Log security events
- Regular security audits

## 📊 Performance Optimization

### Hardware Optimization
- Enable GPU acceleration for mathematical computations
- Use SSD storage for faster data access
- Ensure sufficient RAM for real-time processing
- Optimize network connectivity

### Software Optimization
- Use eventlet for better Socket.IO performance
- Implement connection pooling
- Optimize database queries
- Use caching for frequently accessed data

## 🔄 Updates and Maintenance

### Regular Maintenance
- Update dependencies regularly
- Monitor system performance
- Review and rotate API keys
- Backup configuration files

### Version Updates
- Check for new Schwabot releases
- Review changelog for breaking changes
- Test updates in development environment
- Plan maintenance windows for updates

## 📞 Support

### Documentation
- This README file
- Inline code documentation
- API endpoint documentation
- Configuration file examples

### Logging
- Check log files for detailed error information
- Enable debug mode for troubleshooting
- Monitor system metrics

### Community
- GitHub issues for bug reports
- Feature requests through GitHub
- Community discussions and support

## 📄 License

This project is part of the Schwabot trading system. Please refer to the main project license for terms and conditions.

---

**🌀 Schwabot Unified Trading Interface v0.5**  
*Advanced AI-Powered Multi-Profile Trading System* 