# Schwabot Trading System

Advanced cryptocurrency trading bot with unified mathematics framework, AI-powered decision making, and comprehensive data integration.

## 🚀 Quick Start

### Automated Setup (Recommended)
```bash
python setup_environment.py
```

### Manual Setup

1. **Check Python Version**
   ```bash
   python --version  # Requires Python 3.8+
   ```

2. **Install Dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Install Development Dependencies (Optional)**
   ```bash
   pip install -r requirements-dev.txt
   ```

4. **Set up Pre-commit Hooks (Optional)**
   ```bash
   pre-commit install
   ```

5. **Test Installation**
   ```bash
   python test_schwabot_integration.py
   ```

6. **Run Schwabot**
   ```bash
   python schwabot_enhanced_launcher.py
   ```

## 📋 System Requirements

- **Python**: 3.8 or higher
- **Operating System**: Windows, macOS, Linux
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 1GB free space

## 🏗️ Architecture

### Core Components

- **Unified Mathematics Framework**: Advanced mathematical operations and calculations
- **Advanced Settings Engine**: Configurable bias coefficients and weighted confidence frameworks
- **API Handlers**: Integration with multiple data sources (WhaleAlert, Glassnode, CoinGecko)
- **Cache Sync Service**: Efficient data caching and synchronization
- **Trading Engine**: Execution and risk management
- **Signal Processing**: Echo-based recursive logic and spatial momentum calculations

### Data Sources

- **Fear & Greed Index**: Market sentiment analysis
- **WhaleAlert**: Large transaction monitoring
- **Glassnode**: On-chain metrics and analytics
- **CoinGecko**: Market data and prices
- **Custom APIs**: Extensible handler system

## ⚙️ Configuration

### Environment Setup

1. Create required directories:
   ```
   flask/feeds/
   flask/feeds/sentiment/
   flask/feeds/whale_data/
   flask/feeds/onchain_data/
   flask/feeds/market_data/
   settings/
   logs/
   ```

2. Configure API keys (optional):
   - WhaleAlert API key
   - Glassnode API key
   - Other custom API keys

### Advanced Settings

The system uses an advanced settings engine that controls behavior through bias coefficients rather than disabling core functionality:

- **Echo Delay Sensitivity**: Controls lag window for signal detection
- **AI Consensus Weight**: Weight given to AI decision making
- **Buy/Sell Wall Aggression**: Trade execution intensity
- **Strategy Memory Decay**: Rate of strategy adaptation

## 🧪 Testing

### Integration Tests
```bash
python test_schwabot_integration.py
```

### Unit Tests
```bash
pytest tests/
```

### Code Quality
```bash
# Format code
black .

# Check style
flake8 .

# Type checking
mypy .

# Run all pre-commit hooks
pre-commit run --all-files
```

## 📊 Usage

### Basic Trading
```python
from schwabot_enhanced_launcher import SchawbotEnhancedLauncher

launcher = SchawbotEnhancedLauncher()
await launcher.start()
```

### Mathematical Operations
```python
from schwabot_unified_math import UnifiedMathematicsFramework

framework = UnifiedMathematicsFramework()
drift_field = framework.compute_unified_drift_field(1.0, 2.0, 0.5, 1.0)
```

### Settings Configuration
```python
from core.advanced_settings_engine import AdvancedSettingsEngine

engine = AdvancedSettingsEngine()
engine.set_setting_value("echo_delay_sensitivity", 1.2)
```

## 🔧 Development

### Code Style
- **Formatter**: Black (88 character line length)
- **Import Sorting**: isort with black profile
- **Linting**: flake8 with custom configuration
- **Type Checking**: mypy with strict settings

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make changes following code style guidelines
4. Run tests and quality checks
5. Submit a pull request

### Project Structure
```
schwabot/
├── core/                    # Core system components
│   ├── api/                # API handlers and data integration
│   ├── advanced_settings_engine.py
│   └── type_defs.py
├── utils/                  # Utility functions
├── schwabot_unified_math.py # Mathematical framework
├── schwabot_enhanced_launcher.py # Main launcher
├── test_schwabot_integration.py # Integration tests
├── requirements.txt        # Production dependencies
├── requirements-dev.txt    # Development dependencies
├── pyproject.toml         # Modern Python configuration
├── mypy.ini              # Type checking configuration
├── .flake8               # Linting configuration
└── .pre-commit-config.yaml # Code quality automation
```

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install --upgrade -r requirements.txt
   ```

2. **Version Conflicts**
   ```bash
   pip install --upgrade pip
   pip install --force-reinstall -r requirements.txt
   ```

3. **Permission Issues**
   ```bash
   pip install --user -r requirements.txt
   ```

4. **API Rate Limits**
   - Configure API keys for higher rate limits
   - Use demo mode for testing

### Performance Optimization

- **Memory**: Monitor with `psutil` integration
- **Caching**: Adjust cache refresh intervals
- **Concurrency**: Configure async operations

## 📈 Features

### Advanced Mathematics
- Unified drift field calculations
- Entropy-stabilized feedback systems
- Recursive identity functions
- Quantum-enhanced processing
- Spatial momentum calculations

### Trading Intelligence
- Multi-source signal integration
- Echo-based decision making
- Adaptive strategy learning
- Risk-adjusted optimization
- Real-time market analysis

### System Monitoring
- Performance metrics tracking
- Resource usage monitoring
- Error detection and handling
- Automated quality checks

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Support

- **Issues**: GitHub Issues
- **Documentation**: In-code documentation
- **Community**: GitHub Discussions

---

**⚠️ Disclaimer**: This is experimental trading software. Use at your own risk. Past performance does not guarantee future results.
