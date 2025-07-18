# Entropy-Enhanced Trading System

A complete, production-ready cryptocurrency trading system that integrates entropy signal processing with advanced mathematical frameworks for BTC/USDC trading.

## 🚀 Overview

This system implements a sophisticated trading pipeline that combines:

- **Entropy Signal Processing**: Real-time market entropy analysis for optimal timing
- **Strategy Bit Mapping**: Dynamic strategy selection using 4-bit and 8-bit mapping
- **Pure Profit Calculation**: Mathematically rigorous profit calculation with GPU acceleration
- **Risk Management**: Comprehensive risk assessment and position sizing
- **Order Execution**: CCXT-based order execution with multiple exchange support
- **Portfolio Management**: Real-time portfolio tracking and rebalancing

## 🧠 Core Components

### 1. Entropy Signal Integration (`core/entropy_signal_integration.py`)
- Processes order book data to extract entropy signals
- Provides timing cycles and confidence adjustments
- Integrates with all trading components for enhanced decision making

### 2. Strategy Bit Mapper (`core/strategy_bit_mapper.py`)
- Implements 4-bit and 8-bit strategy mapping
- Supports multiple expansion modes (entropy_adaptive, tensor_weighted, orbital_adaptive)
- Integrates with Ferris_RDE for strategy selection

### 3. Pure Profit Calculator (`core/pure_profit_calculator.py`)
- GPU-accelerated profit calculations with CPU fallback
- Implements mathematical framework: Π = F(M(t), H(t), S)
- Entropy-enhanced confidence and timing adjustments

### 4. Trading Pipeline (`core/clean_trading_pipeline.py`)
- Main trading pipeline with entropy signal integration
- Real-time market data processing
- Trade decision logic with entropy enhancement

### 5. Real-Time Execution Engine (`core/real_time_execution_engine.py`)
- High-frequency signal generation
- Entropy-driven execution timing
- Performance monitoring and optimization

### 6. Entropy-Enhanced Trading Executor (`core/entropy_enhanced_trading_executor.py`)
- Complete trading execution system
- CCXT integration for order execution
- Portfolio management and performance tracking

## 🛠️ Installation

### Prerequisites
```bash
# Python 3.8+
python --version

# Required packages
pip install ccxt numpy cupy-cuda11x pyyaml asyncio
```

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd AOI_Base_Files_Schwabot

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p logs data/historical matrices
```

### Environment Variables
Create a `.env` file with your exchange API credentials:
```bash
COINBASE_API_KEY=your_api_key_here
COINBASE_SECRET=your_secret_here
```

## 🚀 Usage

### Quick Start
```bash
# Run in demo mode (recommended for first use)
python main_trading_system.py --demo

# Run with custom configuration
python main_trading_system.py --config config/entropy_trading_system_config.yaml

# Check system status
python main_trading_system.py --status
```

### Configuration
The system uses a comprehensive YAML configuration file (`config/entropy_trading_system_config.yaml`) that includes:

- **Exchange Settings**: API credentials, trading pairs, rate limits
- **Entropy Configuration**: Timing cycles, confidence thresholds, order book analysis
- **Strategy Parameters**: Position sizing, bit mapping, profit calculation modes
- **Risk Management**: Risk tolerance, stop losses, position limits
- **Execution Settings**: Trading intervals, order types, slippage tolerance

### Trading Modes

#### 1. Demo Mode
- Paper trading with simulated orders
- Perfect for testing and development
- No real money involved

#### 2. Production Mode
- Real trading with actual orders
- Requires valid API credentials
- Comprehensive risk management

#### 3. Backtest Mode
- Historical data testing
- Performance analysis
- Strategy optimization

## 📊 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Trading System Manager                   │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│              Entropy-Enhanced Trading Executor              │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Market    │  │   Entropy   │  │  Strategy   │         │
│  │   Data      │  │   Signals   │  │   Bit Map   │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│         │               │               │                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Profit    │  │    Risk     │  │   Order     │         │
│  │ Calculator  │  │ Management  │  │ Execution   │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                    CCXT Exchange Layer                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Coinbase  │  │   Binance   │  │   Kraken    │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

## 🔧 Key Features

### Entropy Signal Processing
- **Timing Cycles**: 1, 5, 15, 30, 60-minute cycles for optimal execution
- **Confidence Adjustments**: Dynamic confidence scoring based on market entropy
- **Order Book Analysis**: Real-time spread, volume, and imbalance analysis
- **Market Context**: Comprehensive market state analysis

### Strategy Bit Mapping
- **4-bit Mapping**: Quick strategy selection for high-frequency trading
- **8-bit Mapping**: Detailed strategy expansion for complex scenarios
- **Entropy Adaptive**: Strategy selection based on entropy signals
- **Ferris_RDE Integration**: Advanced strategy routing and execution

### Profit Calculation
- **GPU Acceleration**: CuPy-based calculations with automatic CPU fallback
- **Mathematical Rigor**: Pure mathematical framework without ZPE/ZBE dependencies
- **Entropy Enhancement**: Profit scores adjusted by entropy timing and confidence
- **Risk Adjustment**: Comprehensive risk assessment and adjustment

### Risk Management
- **Position Sizing**: Kelly criterion with volatility and correlation adjustment
- **Stop Losses**: Trailing stop losses with configurable percentages
- **Take Profits**: Trailing take profits for optimal exit timing
- **Portfolio Limits**: Maximum position values and daily loss limits

### Order Execution
- **Multiple Exchanges**: Support for Coinbase, Binance, Kraken, and more
- **Order Types**: Market, limit, stop loss, and take profit orders
- **Slippage Control**: Configurable slippage tolerance and monitoring
- **Fee Optimization**: Maker/taker fee optimization

## 📈 Performance Monitoring

The system provides comprehensive performance monitoring:

- **Real-time Metrics**: Trade success rate, profit/loss, Sharpe ratio
- **Entropy Analytics**: Entropy adjustment effectiveness and timing accuracy
- **Risk Metrics**: Maximum drawdown, VaR, position correlation
- **System Health**: API response times, error rates, system uptime

### Performance Dashboard
```bash
# View real-time performance
python main_trading_system.py --status

# Monitor logs
tail -f logs/trading_system.log
tail -f logs/performance.log
tail -f logs/trades.log
```

## 🔒 Security Features

- **API Key Encryption**: Encrypted storage of exchange API credentials
- **Access Control**: IP whitelisting and user authentication
- **Data Protection**: Encryption at rest and in transit
- **Audit Logging**: Comprehensive audit trail for all trading activities

## 🧪 Testing and Development

### Unit Tests
```bash
# Run unit tests
python -m pytest test/

# Run specific test modules
python -m pytest test/test_entropy_integration.py
python -m pytest test/test_profit_calculator.py
```

### Integration Tests
```bash
# Run integration tests
python -m pytest test/integration/

# Test complete trading cycle
python test/integrated_trading_test_suite.py
```

### Development Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with development configuration
python main_trading_system.py --config config/development.yaml
```

## 📚 API Documentation

### Core Classes

#### EntropySignalIntegration
```python
from core.entropy_signal_integration import EntropySignalIntegration

# Initialize
entropy = EntropySignalIntegration()

# Process signals
result = entropy.process_entropy_signals(
    order_book_data=order_book,
    market_context=context
)
```

#### StrategyBitMapper
```python
from core.strategy_bit_mapper import StrategyBitMapper

# Initialize
mapper = StrategyBitMapper(matrix_dir="./matrices")

# Expand strategy bits
strategy_id = mapper.expand_strategy_bits(
    strategy_id=1234,
    target_bits=8,
    mode="entropy_adaptive",
    market_data=market_data
)
```

#### PureProfitCalculator
```python
from core.pure_profit_calculator import PureProfitCalculator, MarketData, HistoryState

# Initialize
calculator = PureProfitCalculator(strategy_params)

# Calculate profit
result = calculator.calculate_profit(
    market_data=market_data,
    history_state=history_state
)
```

#### EntropyEnhancedTradingExecutor
```python
from core.entropy_enhanced_trading_executor import create_trading_executor

# Create executor
executor = create_trading_executor(
    exchange_config=exchange_config,
    strategy_config=strategy_config,
    entropy_config=entropy_config,
    risk_config=risk_config
)

# Execute trading cycle
result = await executor.execute_trading_cycle()
```

## 🚨 Risk Disclaimer

**IMPORTANT**: This trading system is for educational and research purposes. Cryptocurrency trading involves substantial risk of loss. Always:

- Start with small amounts
- Test thoroughly in demo mode
- Monitor system performance closely
- Never invest more than you can afford to lose
- Consider consulting with financial advisors

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:

- Check the documentation in the `docs/` directory
- Review the configuration examples
- Run the demo mode to understand the system
- Check the logs for error messages

## 🔄 Changelog

### Version 1.0.0
- Initial release with complete entropy-enhanced trading system
- Integration of all core components
- Production-ready configuration and deployment
- Comprehensive documentation and testing suite

---

**Built with ❤️ for the crypto trading community** 