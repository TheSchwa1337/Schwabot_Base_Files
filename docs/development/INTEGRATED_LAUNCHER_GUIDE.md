# Schwabot Integrated Launcher System

## Overview

The Schwabot Integrated Launcher is a comprehensive control center that unifies all components of the Schwabot trading system into a single, secure, and user-friendly interface. This system provides complete API integration, data pipeline management, ChronoResonance Weather Mapping (CRWM), and profit-driven trading strategies.

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+** is required
2. **Optional Dependencies** (for full functionality):
   ```bash
   pip install numpy scipy psutil cryptography requests aiohttp tkinter
   ```

### Launch the System

```bash
# Simple demo launch
python launch_schwabot_integrated.py

# Or run the full integrated launcher directly
python -m core.schwabot_integrated_launcher
```

## 🏗️ System Architecture

### Core Components

1. **Schwabot Integrated Launcher** (`core/schwabot_integrated_launcher.py`)
   - Main control center with tabbed interface
   - System status monitoring
   - Configuration management
   - Real-time metrics display

2. **Secure API Coordinator** (`core/secure_api_coordinator.py`)
   - Encrypted API key storage
   - Rate limiting and throttling
   - Multi-provider API management
   - Security audit logging

3. **Data Pipeline Visualizer** (`core/data_pipeline_visualizer.py`)
   - Real-time data flow visualization
   - Memory tier management (RAM/Mid/Long-term)
   - Compression monitoring
   - Performance analytics

4. **ChronoResonance Weather Mapper** (`core/chrono_resonance_weather_mapper.py`)
   - Weather-price correlation analysis
   - Atmospheric gradient calculations
   - Resonance frequency analysis
   - Predictive weather modeling

## 🔐 API Integration & Security

### Supported API Providers

| Provider | Purpose | Security Level | Rate Limits |
|----------|---------|----------------|-------------|
| **CoinMarketCap** | BTC Price Data | API Key | 100/min |
| **OpenWeather** | Weather Data (CRWM) | API Key | 60/min |
| **NewsAPI** | Market Sentiment | API Key | 500/min |
| **Twitter** | Social Sentiment | OAuth | 300/min |
| **Binance** | Trading Execution | Exchange | 1200/min |
| **Coinbase** | Trading Execution | Exchange | 10/min |
| **Kraken** | Trading Execution | Exchange | 20/min |

### Security Features

1. **Encrypted Storage**
   - API keys encrypted using Fernet (AES 128)
   - Secure key derivation
   - Platform-specific secure storage paths

2. **Access Control**
   - Security levels: PUBLIC, API_KEY, OAUTH, EXCHANGE
   - Rate limiting per provider
   - Request/response validation

3. **Audit Logging**
   - All API requests logged
   - Performance metrics tracked
   - Error monitoring and alerting

### API Configuration

#### Adding API Keys

1. **Via UI (Recommended)**:
   - Open Launcher → API Management tab
   - Select API provider
   - Enter API key/secret
   - Test connection
   - Save securely

2. **Via Code**:
   ```python
   from core.secure_api_coordinator import SecureAPICoordinator, APIProvider, SecurityLevel
   
   coordinator = SecureAPICoordinator()
   coordinator.store_credentials(
       APIProvider.COINMARKETCAP,
       "your-api-key-here",
       security_level=SecurityLevel.API_KEY
   )
   ```

#### Example API Usage

```python
# Get BTC price
coordinator = SecureAPICoordinator()
price = coordinator.get_btc_price()
print(f"Current BTC Price: ${price:,.2f}")

# Get weather data for CRWM
weather_data = coordinator.get_weather_data("New York")
print(f"Temperature: {weather_data['temperature']}°C")

# Get news sentiment
articles = coordinator.get_news_sentiment("bitcoin", max_articles=5)
print(f"Found {len(articles)} articles")
```

## 💾 Data Pipeline Management

### Data Tiers

1. **RAM Cache** (Short-term)
   - Default limit: 500MB
   - Retention: 30 minutes
   - Use: Real-time trading data

2. **Mid-term Storage** (Local disk)
   - Default limit: 2GB
   - Retention: 24 hours
   - Use: Analysis and backtesting

3. **Long-term Storage** (Persistent)
   - Default limit: 10GB
   - Retention: 7 days
   - Use: Historical analysis

4. **Archive** (Cold storage)
   - Default limit: 50GB
   - Retention: 30 days
   - Use: Long-term backtesting

### Data Categories

- **BTC Hashing**: Bitcoin hash analysis data
- **Trading Signals**: Generated trade signals
- **Market Data**: Price, volume, volatility data
- **Risk Metrics**: Risk assessment results
- **Portfolio State**: Current portfolio status
- **Analysis Results**: Mathematical analysis outputs
- **System Logs**: Application logs
- **API Responses**: Cached API data

### Pipeline Operations

```python
from core.data_pipeline_visualizer import DataPipelineVisualizer, DataCategory, DataTier

pipeline = DataPipelineVisualizer()

# Add data
unit_id = pipeline.add_data_unit(
    DataCategory.TRADING_SIGNALS,
    data_size=1024,
    tier=DataTier.RAM_CACHE,
    priority=1
)

# Move data between tiers
pipeline.move_data_unit(unit_id, DataTier.MID_TERM)

# Get pipeline status
status = pipeline.get_pipeline_status()
print(f"Total units: {status['total_units']}")
```

## 🌤️ ChronoResonance Weather Mapping (CRWM)

### Mathematical Foundation

CRWM uses advanced mathematical models to correlate weather patterns with Bitcoin price movements:

- **Resonance Frequency**: `f_r = 1/(2π√(LC))`
- **Price Gradient**: `∇P = α·∇T + β·∇H + γ·∇W`
- **Chrono-correlation**: `C(τ) = ∫ W(t)·P(t+τ) dt`
- **Atmospheric Momentum**: `M = ρ·v²`

### Weather Parameters

1. **Temperature** (°C)
2. **Atmospheric Pressure** (hPa)
3. **Humidity** (%)
4. **Wind Speed** (m/s)
5. **Wind Direction** (degrees)
6. **Visibility** (km)

### Usage Example

```python
from core.chrono_resonance_weather_mapper import ChronoResonanceWeatherMapper, WeatherDataPoint
from datetime import datetime

crwm = ChronoResonanceWeatherMapper()

# Add weather data
weather_point = WeatherDataPoint(
    timestamp=datetime.now(),
    location="Trading Center",
    temperature=22.5,
    pressure=1013.25,
    humidity=65.0,
    wind_speed=3.2,
    wind_direction=180.0,
    weather_type="partly_cloudy"
)

crwm.add_weather_data(weather_point)
crwm.add_price_data(datetime.now(), 45000.0)

# Get weather signature for trading
signature = crwm.get_weather_signature("4h")
trading_signal = signature['trading_signals']
print(f"Signal: {trading_signal['direction']} ({trading_signal['confidence']:.2f} confidence)")

# Get price prediction
prediction = crwm.predict_weather_price_movement(6)  # 6 hours
print(f"Predicted change: {prediction['predicted_change_percent']:.2f}%")
```

## 💰 Profit-Driven Trading Strategy

### Mathematical Components

The system uses five weighted mathematical components for decision making:

1. **ALEPH Hash Similarity** (25%): Market state fingerprinting
2. **Phase Alignment** (20%): Momentum consistency analysis
3. **NCCO Entropy Score** (20%): Market predictability assessment
4. **Drift Weight** (20%): Temporal/spatial drift compensation
5. **Pattern Confidence** (15%): Price-volume correlation analysis

### Trading Decision Formula

```
Confidence Score: C(t) = α·H_sim + β·φ_align + γ·E_ent + δ·D_drift + ε·P_conf
Trade Decision: T(t) = 1 if C(t) > θ_threshold ∧ P(t) > P_min
```

### Entry Criteria

- **Confidence Threshold**: ≥75%
- **Minimum Profit**: ≥0.5%
- **Maximum Risk**: ≤30%
- **Kelly Criterion**: Position sizing optimization

## 🎛️ Launcher Interface Guide

### Dashboard Tab

- **System Status**: Real-time component status
- **Quick Actions**: Start/pause trading, generate reports
- **Performance Metrics**: Key system metrics

### API Management Tab

- **Provider Configuration**: Setup API credentials
- **Connection Testing**: Verify API connectivity
- **CRWM Integration**: Weather mapping controls
- **Rate Limit Monitoring**: Track API usage

### Data Pipeline Tab

- **Pipeline Visualization**: Real-time data flow
- **Memory Usage**: Tier utilization charts
- **Data Controls**: Cleanup and management tools
- **Performance Statistics**: Throughput and efficiency metrics

### Settings Tab

- **Installation Paths**: Configure file locations
- **Performance Tuning**: RAM limits, update intervals
- **Advanced Features**: CPU dedication, hardware detection
- **Security Options**: Encryption settings

### Monitoring Tab

- **Real-time Metrics**: System performance data
- **Recent Actions**: Activity log
- **Error Tracking**: System health monitoring
- **Export Tools**: Data export capabilities

## ⚙️ Configuration

### Default Settings

```json
{
  "security": {
    "encryption_enabled": true,
    "api_timeout_seconds": 30,
    "max_retry_attempts": 3
  },
  "data_pipeline": {
    "short_term_retention_hours": 24,
    "mid_term_retention_days": 7,
    "long_term_retention_days": 30,
    "max_ram_usage_mb": 500,
    "compression_enabled": true
  },
  "visualization": {
    "real_time_updates": true,
    "update_interval_ms": 1000,
    "max_chart_points": 1000
  },
  "chrono_weather": {
    "enabled": true,
    "update_interval_minutes": 5,
    "data_retention_hours": 48
  }
}
```

### Environment Variables

```bash
# Installation path
export SCHWABOT_INSTALL_PATH="/path/to/schwabot"

# API configuration
export COINMARKETCAP_API_KEY="your-key-here"
export OPENWEATHER_API_KEY="your-key-here"
export NEWSAPI_KEY="your-key-here"

# Security settings
export SCHWABOT_ENCRYPTION_KEY="auto-generated"
```

## 🔧 Advanced Configuration

### Custom Installation Path

```python
# Set custom installation path
import os
os.environ['SCHWABOT_INSTALL_PATH'] = '/custom/path/schwabot'

# Or configure via launcher
launcher = SchwabotIntegratedLauncher()
launcher.installation_path = Path('/custom/path/schwabot')
```

### Performance Tuning

```python
# Configure data pipeline limits
config = {
    'ram_cache_limit_mb': 1000,      # Increase RAM cache
    'mid_term_limit_mb': 5000,       # Increase mid-term storage
    'animation_fps': 60,             # Increase visualization FPS
    'particle_count': 100            # More visualization particles
}

pipeline = DataPipelineVisualizer(config)
```

### Security Hardening

```python
# Enhanced security configuration
api_config = {
    'enable_rate_limiting': True,
    'enable_request_logging': True,
    'auto_key_rotation': True,
    'key_rotation_days': 30
}

coordinator = SecureAPICoordinator(api_config)
```

## 🚨 Troubleshooting

### Common Issues

1. **Dependencies Missing**
   ```bash
   # Install all optional dependencies
   pip install numpy scipy psutil cryptography requests aiohttp
   ```

2. **API Keys Not Working**
   - Verify API key validity
   - Check rate limits
   - Test connection via UI
   - Check firewall/proxy settings

3. **UI Not Loading**
   - Ensure tkinter is installed
   - Try console mode instead
   - Check display settings (Linux)

4. **High Memory Usage**
   - Reduce pipeline limits in settings
   - Enable compression
   - Run cleanup operations

5. **Performance Issues**
   - Lower update intervals
   - Reduce particle count
   - Disable real-time features

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with verbose output
python launch_schwabot_integrated.py --debug
```

### Log Files

- **System Log**: `schwabot_launcher.log`
- **API Log**: `~/.schwabot/secure/api_requests.log`
- **CRWM Log**: `~/.schwabot/crwm/crwm_analysis.log`
- **Pipeline Log**: `~/.schwabot/pipeline/data_flow.log`

## 📊 Performance Monitoring

### Key Metrics

1. **System Performance**
   - CPU usage
   - Memory utilization
   - Disk I/O

2. **API Performance**
   - Request rate
   - Response time
   - Success rate
   - Rate limit status

3. **Pipeline Performance**
   - Data throughput
   - Compression ratio
   - Tier utilization
   - Processing latency

4. **Trading Performance**
   - Signal generation rate
   - Confidence levels
   - Profit calculations
   - Risk assessments

### Monitoring Dashboard

The integrated launcher provides real-time monitoring through:

- **Live metrics display**: CPU, memory, network
- **API status indicators**: Connection health, rate limits
- **Data flow visualization**: Pipeline activity, data movement
- **Trading signals**: Real-time signal generation and analysis

## 🔒 Security Best Practices

### API Key Management

1. **Never hardcode API keys** in source code
2. **Use environment variables** or secure storage
3. **Rotate keys regularly** (30-90 days)
4. **Monitor API usage** for suspicious activity
5. **Use minimum required permissions**

### System Security

1. **Run with minimal privileges**
2. **Keep system updated**
3. **Use secure network connections**
4. **Enable audit logging**
5. **Regular security reviews**

### Data Protection

1. **Encrypt sensitive data** at rest
2. **Use secure communication** (HTTPS/TLS)
3. **Implement data retention policies**
4. **Regular backups** of critical data
5. **Access control** for configuration files

## 📈 Optimization Tips

### Performance Optimization

1. **Tune pipeline limits** based on available memory
2. **Adjust update intervals** for your use case
3. **Enable compression** for large datasets
4. **Use SSD storage** for better I/O performance
5. **Monitor resource usage** and adjust accordingly

### Trading Optimization

1. **Fine-tune confidence thresholds** based on backtesting
2. **Adjust weather correlation parameters** for your market
3. **Optimize risk management settings**
4. **Regular strategy validation** against market conditions
5. **Monitor and adjust position sizing**

## 🤝 Integration Examples

### Custom API Integration

```python
from core.secure_api_coordinator import APIProvider, SecurityLevel

# Add custom API provider
coordinator.store_credentials(
    APIProvider.CUSTOM,
    "custom-api-key",
    api_secret="custom-secret",
    security_level=SecurityLevel.OAUTH
)

# Custom API request
response = coordinator.make_request(
    APIProvider.CUSTOM,
    "/custom/endpoint",
    method="POST",
    data={"param": "value"}
)
```

### External System Integration

```python
# Export data for external analysis
pipeline.export_statistics("pipeline_data.json")
api_coordinator.export_api_data("api_metrics.json")
crwm.export_crwm_data("weather_analysis.json")

# Import external signals
def process_external_signal(signal_data):
    # Process external trading signal
    return trading_decision

# Integration with external risk management
def integrate_risk_system(portfolio_state):
    # Custom risk management logic
    return risk_assessment
```

## 📚 Additional Resources

### Documentation Files

- `PROFIT_DRIVEN_SYSTEM_SUMMARY.md`: Complete profit system documentation
- `ENHANCED_PROFIT_SYSTEM_DOCUMENTATION.md`: Technical implementation details
- Component-specific documentation in each module

### Example Scripts

- `launch_schwabot_integrated.py`: Main launcher demo
- `demo_enhanced_profit_system.py`: Profit system demonstration
- `test_profit_strategy_simple.py`: Simple strategy testing

### Support

For technical support or questions:
1. Check the troubleshooting section above
2. Review log files for error details
3. Consult the comprehensive documentation
4. Test individual components separately

---

## 🎉 Conclusion

The Schwabot Integrated Launcher provides a complete, secure, and powerful trading system with:

- **Secure API Integration**: Multi-provider API management with encryption
- **Advanced Data Pipeline**: Real-time data processing and visualization
- **Weather Correlation**: ChronoResonance Weather Mapping for enhanced signals
- **Profit Optimization**: Mathematical profit-driven trading strategies
- **Comprehensive Monitoring**: Real-time system performance tracking

This system is designed to be both powerful for advanced users and accessible for those getting started with algorithmic trading.

**Happy Trading! 🚀📈** 