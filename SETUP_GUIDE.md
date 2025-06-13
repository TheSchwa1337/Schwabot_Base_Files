# Schwabot Enhanced Setup Guide

## 🚀 Quick Start

### 1. Automated Setup (Recommended)

Run the automated setup script to configure everything:

```bash
python setup_schwabot.py
```

This script will:
- ✅ Check Python version compatibility (3.8+)
- ✅ Install all required dependencies
- ✅ Create necessary directories and `__init__.py` files
- ✅ Generate default configuration files
- ✅ Validate the configuration system
- ✅ Run integration tests
- ✅ Generate a setup report

### 2. Manual Setup

If you prefer manual setup:

```bash
# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir -p config/schemas tests logs data/matrix_logs

# Run tests
python -m pytest tests/test_config_loading.py -v
```

## 📁 Enhanced Directory Structure

```
schwabot/
├── config/                          # Configuration management
│   ├── __init__.py                  # Config module initialization
│   ├── config_utils.py              # Enhanced config utilities
│   ├── io_utils.py                  # Legacy I/O utilities
│   ├── schemas/                     # Configuration schemas
│   │   ├── __init__.py
│   │   └── quantization.py         # Pydantic schemas
│   ├── tesseract_enhanced.yaml     # Enhanced Tesseract config
│   ├── fractal_core.yaml           # Fractal processing config
│   ├── matrix_response_paths.yaml  # Matrix response config
│   └── risk_config.yaml            # Risk management config
├── core/                            # Core processing modules
│   ├── __init__.py
│   ├── enhanced_tesseract_processor.py  # Enhanced processor
│   └── [other core modules...]
├── tests/                           # Test suite
│   ├── __init__.py
│   ├── test_config_loading.py       # Configuration tests
│   └── [other test files...]
├── dashboard_integration.py         # Dashboard bridge
├── setup_schwabot.py               # Automated setup script
├── requirements.txt                 # Enhanced dependencies
└── SETUP_GUIDE.md                  # This guide
```

## 🔧 Configuration System

### Enhanced Features

1. **Standardized YAML Path Resolution**
   - All config paths are resolved relative to the repository root
   - No more missing file issues when running from different directories

2. **Automatic Default Config Generation**
   - Missing configuration files are automatically created with sensible defaults
   - Supports different config types (tesseract, fractal, matrix, risk)

3. **Schema Validation**
   - Optional Pydantic schema validation for configuration files
   - Ensures configuration integrity and type safety

4. **Robust Error Handling**
   - Clear error messages for configuration issues
   - Graceful fallbacks for missing or invalid configurations

### Configuration Loading Examples

```python
from config.config_utils import load_yaml_config, load_tesseract_config

# Load any YAML config with automatic defaults
config = load_yaml_config("tesseract_enhanced.yaml")

# Load specific config types with validation
tesseract_config = load_tesseract_config()
fractal_config = load_fractal_config()
matrix_config = load_matrix_config()

# Load with schema validation
from config.schemas.quantization import QuantizationSchema
validated_config = load_yaml_config("fractal_core.yaml", schema=QuantizationSchema)
```

## 🎛️ Enhanced Tesseract Processor

### New Features

1. **Robust Configuration Loading**
   - Automatic path standardization
   - Default config creation if missing
   - Enhanced error handling

2. **Pattern History Management**
   - Configurable history size limits
   - Automatic memory management
   - Overflow protection

3. **Enhanced Error Handling**
   - Safe attribute access for risk metrics
   - Graceful degradation on errors
   - Comprehensive logging

4. **Configurable Strategy Switching**
   - Strategy triggers defined in configuration
   - No more hardcoded magic constants
   - Easy to add new strategies

5. **Weighted Profit Vector Blending**
   - Sophisticated profit signal calculation
   - Configurable blending parameters
   - Better signal quality

6. **Test Mode and Debugging**
   - Enhanced test mode with verbose logging
   - Status monitoring and reporting
   - State reset functionality

### Usage Examples

```python
from core.enhanced_tesseract_processor import EnhancedTesseractProcessor

# Initialize with automatic config loading
processor = EnhancedTesseractProcessor()

# Enable test mode for debugging
processor.enable_test_mode(verbose=True)

# Process market data
market_data = {
    'price': 50000.0,
    'volume': 1000.0,
    'volatility': 0.02,
    # ... other market data
}

signals = await processor.process_market_tick(market_data, basket_id="BTC")

# Get processor status
status = processor.get_status()
print(f"Processed {status['tick_counter']} ticks")

# Reset state for testing
processor.reset_state()
```

## 📊 Dashboard Integration

### Features

1. **Rich Console Output**
   - Beautiful formatted displays using Rich library
   - Fallback to simple text if Rich not available

2. **Real-time Status Monitoring**
   - System metrics display
   - Strategy status tracking
   - Debug mode indicators

3. **Configuration Management**
   - Profile display and management
   - Configuration comparison tools
   - Export functionality

4. **Multi-format Support**
   - JSON export for external systems
   - YAML configuration loading
   - Cross-platform compatibility

### Usage Examples

```python
from dashboard_integration import DashboardBridge, quick_status_display

# Create dashboard bridge
dashboard = DashboardBridge()

# Display current profile
dashboard.display_profile()

# Update and display status
status_data = {
    'tick_counter': 1234,
    'active_strategy': 'momentum_cascade',
    'vault_locked': False,
    'test_mode': True
}
dashboard.update_status(status_data)
dashboard.display_status()

# Export dashboard data
export_path = dashboard.export_dashboard_json()

# Quick status display
quick_status_display(status_data)
```

## 🧪 Testing

### Test Suite

The enhanced system includes comprehensive tests:

```bash
# Run all configuration tests
python -m pytest tests/test_config_loading.py -v

# Run specific test categories
python -m pytest tests/test_config_loading.py::TestConfigUtils -v
python -m pytest tests/test_config_loading.py::TestEnhancedTesseractProcessor -v
python -m pytest tests/test_config_loading.py::TestDashboardIntegration -v

# Run with coverage
python -m pytest tests/test_config_loading.py --cov=config --cov=core --cov-report=html
```

### Test Categories

1. **Configuration Utilities Tests**
   - Path standardization
   - YAML loading and saving
   - Default config generation
   - Config merging and validation

2. **Enhanced Tesseract Processor Tests**
   - Initialization with various configs
   - Error handling scenarios
   - Test mode functionality

3. **Dashboard Integration Tests**
   - Bridge initialization
   - Status updates
   - Export functionality

4. **Integration Tests**
   - Full configuration loading chain
   - Cross-module compatibility
   - Real-world scenarios

## 🔧 Configuration Reference

### Tesseract Enhanced Configuration

```yaml
processing:
  baseline_reset_flip_frequency: 100
  max_pattern_history: 1000
  max_shell_history: 500
  profit_blend_alpha: 0.7

dimensions:
  labels: ['price', 'volume', 'volatility', 'momentum', 'rsi', 'macd', 'bb_upper', 'bb_lower']

monitoring:
  alerts:
    var_threshold: 0.05
    var_indexed_threshold: 1.5
    coherence_threshold: 0.5
    coherence_indexed_threshold: 0.8

strategies:
  inversion_burst_rebound:
    trigger_prefix: 'e1a7'
  momentum_cascade:
    trigger_prefix: 'f2b8'
  volatility_breakout:
    trigger_prefix: 'a3c9'

debug:
  test_mode: false
  verbose_logging: false

alert_bus:
  enabled: true
  channels: ['log', 'console']
  severity_levels:
    HIGH: 3
    MEDIUM: 2
    LOW: 1
    INFO: 0
```

### Fractal Core Configuration

```yaml
profile:
  name: 'default'
  type: 'quantization'
  parameters:
    decay_power: 1.5
    terms: 12
    dimension: 8
    epsilon_q: 0.003
    precision: 0.001

processing:
  fft_harmonics: 8
  volatility_window: 100
  alignment_threshold: 0.8
```

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure all dependencies are installed
   pip install -r requirements.txt
   
   # Check Python path
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

2. **Configuration File Not Found**
   ```python
   # The system will automatically create defaults
   # Or manually create using:
   from config.config_utils import create_default_config
   create_default_config(Path("config/missing_config.yaml"))
   ```

3. **Rich Library Not Available**
   ```bash
   # Install Rich for enhanced dashboard display
   pip install rich>=12.0.0
   
   # System will fallback to simple text display if not available
   ```

4. **Test Failures**
   ```bash
   # Run setup script to diagnose issues
   python setup_schwabot.py
   
   # Check setup log for details
   cat setup_schwabot.log
   ```

### Validation Commands

```bash
# Validate Python version
python --version  # Should be 3.8+

# Validate dependencies
pip check

# Validate configuration system
python -c "from config.config_utils import load_yaml_config; print('Config system OK')"

# Validate enhanced processor
python -c "from core.enhanced_tesseract_processor import EnhancedTesseractProcessor; print('Processor OK')"

# Validate dashboard integration
python -c "from dashboard_integration import DashboardBridge; print('Dashboard OK')"
```

## 📈 Performance Considerations

### Memory Management

- Pattern history is automatically trimmed to prevent memory overflow
- Configurable history size limits
- Efficient data structures for large datasets

### Configuration Caching

- Configurations are loaded once and cached
- Reload functionality for dynamic updates
- Minimal file I/O overhead

### Error Recovery

- Graceful degradation on component failures
- Safe fallbacks for missing dependencies
- Comprehensive error logging

## 🔄 Migration from Legacy System

### Backward Compatibility

The enhanced system maintains backward compatibility:

```python
# Legacy imports still work
from config.io_utils import load_config, create_default_config

# Enhanced imports provide additional features
from config.config_utils import load_yaml_config, standardize_config_path
```

### Migration Steps

1. **Update Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Setup Script**
   ```bash
   python setup_schwabot.py
   ```

3. **Update Import Statements** (Optional)
   ```python
   # Old
   from config.io_utils import load_config
   
   # New (enhanced)
   from config.config_utils import load_yaml_config
   ```

4. **Validate System**
   ```bash
   python -m pytest tests/test_config_loading.py -v
   ```

## 🎯 Next Steps

After successful setup:

1. **Explore Configuration Options**
   - Review generated configuration files
   - Customize settings for your use case
   - Add new strategy triggers

2. **Run Integration Tests**
   - Test with real market data
   - Validate alert systems
   - Monitor performance metrics

3. **Set Up Monitoring**
   - Configure dashboard displays
   - Set up alert channels
   - Monitor system health

4. **Customize and Extend**
   - Add new configuration schemas
   - Implement custom strategies
   - Extend dashboard functionality

## 📞 Support

If you encounter issues:

1. Check the setup log: `setup_schwabot.log`
2. Review the setup report: `setup_report.json`
3. Run diagnostic tests: `python -m pytest tests/test_config_loading.py -v`
4. Check configuration files in the `config/` directory

The enhanced system is designed to be robust and self-healing, with comprehensive error handling and automatic recovery mechanisms. 