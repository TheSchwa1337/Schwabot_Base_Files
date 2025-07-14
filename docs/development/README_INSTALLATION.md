# Schwabot Trading System - Installation Guide

## 🚀 Quick Start

### Prerequisites
- **Python 3.8+** (3.9+ recommended)
- **Git** (for cloning the repository)
- **Internet connection** (for API access and dependencies)

### Supported Platforms
- ✅ **Windows 10/11** (64-bit)
- ✅ **macOS 10.15+** (Intel/Apple Silicon)
- ✅ **Linux** (Ubuntu 20.04+, CentOS 8+, Debian 11+)

---

## 📦 Installation Methods

### Method 1: Automated Installation (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-username/schwabot-trading-system.git
cd schwabot-trading-system

# Run automated installation
python install.py --auto
```

### Method 2: Manual Installation

```bash
# Clone the repository
git clone https://github.com/your-username/schwabot-trading-system.git
cd schwabot-trading-system

# Create virtual environment
python -m venv schwabot_env

# Activate virtual environment
# Windows:
schwabot_env\Scripts\activate
# macOS/Linux:
source schwabot_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Setup configuration
python install.py --configure
```

### Method 3: Development Installation

```bash
# Clone the repository
git clone https://github.com/your-username/schwabot-trading-system.git
cd schwabot-trading-system

# Install in development mode
pip install -e .

# Install development dependencies
pip install -r requirements.txt
pip install pytest flake8 black mypy
```

---

## 🔧 Platform-Specific Instructions

### Windows Installation

#### Option A: Using Windows Subsystem for Linux (WSL) - Recommended
```bash
# Install WSL2
wsl --install

# Install Python in WSL
sudo apt update
sudo apt install python3.9 python3.9-pip python3.9-venv

# Follow Linux installation steps
```

#### Option B: Native Windows Installation
```powershell
# Install Python from python.org
# Download and install Python 3.9+ from https://python.org

# Verify installation
python --version
pip --version

# Run installation
python install.py --auto
```

### macOS Installation

#### Using Homebrew (Recommended)
```bash
# Install Homebrew if not installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.9

# Install Git
brew install git

# Run installation
python3 install.py --auto
```

#### Using Python.org
```bash
# Download Python from https://python.org
# Install and follow the automated installation steps
```

### Linux Installation

#### Ubuntu/Debian
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install python3.9 python3.9-pip python3.9-venv git curl

# Install additional system dependencies
sudo apt install python3-tk python3-dev build-essential

# Run installation
python3 install.py --auto
```

#### CentOS/RHEL/Fedora
```bash
# Install Python and dependencies
sudo dnf install python3.9 python3.9-pip python3.9-devel git

# Install additional system dependencies
sudo dnf install tkinter python3-tkinter

# Run installation
python3 install.py --auto
```

---

## 🔑 API Configuration

### 1. Coinbase API Setup

1. **Create Coinbase Account**
   - Visit [Coinbase Pro](https://pro.coinbase.com)
   - Create an account and complete verification

2. **Generate API Keys**
   - Go to Settings → API
   - Create new API key with permissions:
     - ✅ View permissions
     - ✅ Trading permissions (for live trading)
   - Save the API key, secret, and passphrase

3. **Configure Environment**
   ```bash
   # Copy template
   cp .env.template .env
   
   # Edit .env file
   nano .env
   ```

   Add your Coinbase credentials:
   ```env
   COINBASE_API_KEY=your_api_key_here
   COINBASE_API_SECRET=your_api_secret_here
   COINBASE_PASSPHRASE=your_passphrase_here
   ```

### 2. CoinMarketCap API Setup

1. **Get API Key**
   - Visit [CoinMarketCap](https://coinmarketcap.com/api/)
   - Sign up for free API access
   - Copy your API key

2. **Configure Environment**
   ```env
   COINMARKETCAP_API_KEY=your_api_key_here
   ```

### 3. Optional: CoinGecko API

CoinGecko provides free API access (no key required):
```env
COINGECKO_API_KEY=  # Leave empty for free tier
```

---

## 🚀 Running the System

### Quick Start
```bash
# Windows
start_schwabot.bat

# macOS/Linux
./start_schwabot.sh

# Or manually
python main.py
```

### GUI Mode (Recommended)
```bash
python main.py --gui
```

### Command Line Mode
```bash
python main.py --cli
```

### Demo Mode (No API Keys Required)
```bash
python main.py --demo
```

### Backtesting Mode
```bash
python main.py --backtest
```

---

## 🧪 Testing the Installation

### Run System Tests
```bash
# Check system requirements
python install.py --check

# Run comprehensive tests
python -m pytest tests/ -v

# Run specific test suites
python -m pytest tests/test_api_integration.py -v
python -m pytest tests/test_gui_system.py -v
```

### Verify API Connections
```bash
# Test API connectivity
python -c "
import asyncio
from core.api_integration_manager import APIIntegrationManager, APIConfig

async def test():
    manager = APIIntegrationManager()
    results = await manager.test_connections()
    print('API Connection Results:', results)

asyncio.run(test())
"
```

---

## 📊 System Features

### Core Components
- ✅ **API Integration**: Coinbase, CoinMarketCap, CoinGecko
- ✅ **GUI System**: Cross-platform tkinter interface
- ✅ **Mathematical Engine**: Advanced trading algorithms
- ✅ **Backtesting**: Historical data analysis
- ✅ **Real-time Monitoring**: Live market data
- ✅ **Risk Management**: Position sizing and stop-loss

### Trading Modes
- 🎯 **Demo Mode**: Paper trading with simulated data
- 🔴 **Live Mode**: Real trading with actual funds
- 📈 **Backtest Mode**: Historical performance analysis

### Visualization Features
- 📊 **Real-time Charts**: Price and volume visualization
- 📈 **Analytics Dashboard**: Mathematical metrics display
- 🔍 **System Monitoring**: Performance and health indicators
- 📋 **Log Management**: Comprehensive logging system

---

## 🔧 Configuration

### Environment Variables
```env
# Trading Configuration
TRADING_MODE=demo                    # demo, live, backtest
SANDBOX_MODE=true                   # true for testing, false for live
MAX_TRADE_AMOUNT=100.0              # Maximum trade size in USD
RISK_PER_TRADE=0.02                 # Risk per trade (2%)

# System Configuration
LOG_LEVEL=INFO                      # DEBUG, INFO, WARNING, ERROR
ENABLE_VISUALIZATION=true           # Enable GUI
ENABLE_BACKTESTING=true             # Enable backtesting
```

### Configuration Files
- `config/basic_config.json`: Basic system configuration
- `config/logging_config.json`: Logging configuration
- `.env`: Environment variables (create from .env.template)

---

## 🛠️ Troubleshooting

### Common Issues

#### 1. Python Version Issues
```bash
# Check Python version
python --version

# If version < 3.8, upgrade Python
# Windows: Download from python.org
# macOS: brew install python@3.9
# Linux: sudo apt install python3.9
```

#### 2. Dependency Installation Issues
```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install with verbose output
pip install -r requirements.txt -v

# Install specific package
pip install numpy pandas matplotlib
```

#### 3. GUI Issues (Windows)
```bash
# Install tkinter
pip install tk

# Or use WSL for better compatibility
wsl --install
```

#### 4. API Connection Issues
```bash
# Check API keys
cat .env

# Test individual APIs
python -c "
import asyncio
from core.api_integration_manager import APIIntegrationManager
manager = APIIntegrationManager()
asyncio.run(manager.test_connections())
"
```

#### 5. Permission Issues (Linux/macOS)
```bash
# Fix permissions
chmod +x start_schwabot.sh
chmod +x install.py

# Run with sudo if needed
sudo python3 install.py --auto
```

### Getting Help

1. **Check Logs**
   ```bash
   tail -f logs/schwabot.log
   ```

2. **Run Diagnostics**
   ```bash
   python install.py --check
   ```

3. **Community Support**
   - GitHub Issues: [Create Issue](https://github.com/your-username/schwabot-trading-system/issues)
   - Documentation: [Wiki](https://github.com/your-username/schwabot-trading-system/wiki)

---

## 🔒 Security Considerations

### API Key Security
- ✅ Never commit API keys to version control
- ✅ Use environment variables for sensitive data
- ✅ Enable API key restrictions (IP whitelist)
- ✅ Use sandbox mode for testing
- ✅ Regularly rotate API keys

### System Security
- ✅ Run in virtual environment
- ✅ Keep dependencies updated
- ✅ Monitor system logs
- ✅ Use firewall rules
- ✅ Enable two-factor authentication

---

## 📈 Performance Optimization

### System Requirements
- **Minimum**: 4GB RAM, 2 CPU cores, 10GB storage
- **Recommended**: 8GB RAM, 4 CPU cores, 50GB storage
- **Optimal**: 16GB RAM, 8 CPU cores, 100GB storage

### Optimization Tips
```bash
# Enable GPU acceleration (if available)
pip install cupy-cuda11x  # For CUDA 11.x
pip install cupy-cuda12x  # For CUDA 12.x

# Optimize Python performance
export PYTHONOPTIMIZE=1
export PYTHONUNBUFFERED=1

# Use SSD storage for better I/O performance
```

---

## 🔄 Updates and Maintenance

### Updating the System
```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade

# Run installation check
python install.py --check
```

### Regular Maintenance
```bash
# Clean old logs
find logs/ -name "*.log" -mtime +30 -delete

# Update system packages
# Windows: Check for updates
# macOS: brew update && brew upgrade
# Linux: sudo apt update && sudo apt upgrade
```

---

## 📚 Additional Resources

### Documentation
- [API Reference](docs/API_REFERENCE.md)
- [Trading Strategies](docs/TRADING_STRATEGIES.md)
- [Mathematical Framework](docs/MATHEMATICAL_FRAMEWORK.md)
- [Troubleshooting Guide](docs/TROUBLESHOOTING.md)

### Examples
- [Basic Usage](examples/basic_usage.py)
- [Advanced Trading](examples/advanced_trading.py)
- [Custom Strategies](examples/custom_strategies.py)

### Community
- [Discord Server](https://discord.gg/schwabot)
- [Reddit Community](https://reddit.com/r/schwabot)
- [YouTube Channel](https://youtube.com/schwabot)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

---

## 📞 Support

- **Email**: support@schwabot.com
- **Discord**: [Join our server](https://discord.gg/schwabot)
- **Documentation**: [Full documentation](https://docs.schwabot.com)

---

**Happy Trading! 🚀📈** 