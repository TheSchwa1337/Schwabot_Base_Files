# Core Mathematical Library for Quantitative Trading

This project implements a comprehensive mathematical library for quantitative trading systems, providing a wide range of tools and strategies for market analysis and trading decisions.

## Features

### Core Features (v0.1)
- Price and volume analysis
- Technical indicators calculation
- Risk management tools
- Hash-based decision making
- Advanced trading strategies
- Visualization tools

### Extended Features (v0.2x)
- Volume-Weighted Average Price (VWAP)
- True Range and Average True Range (ATR)
- Relative Strength Index (RSI)
- Kelly Criterion position sizing
- Multi-asset correlation analysis
- Risk-parity portfolio weights
- Pair trading Z-scores
- Ornstein-Uhlenbeck mean reversion
- Keltner Channels
- Exponential memory kernel

## Installation

1. Clone the repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage (v0.1)
```python
from mathlib import CoreMathLib

# Initialize the library
math_lib = CoreMathLib(
    base_volume=1.0,
    tick_freq=1.0,
    profit_coef=0.8,
    threshold=0.5
)

# Generate or load your price and volume data
prices = [...]  # Your price data
volumes = [...]  # Your volume data

# Apply advanced strategies
results = math_lib.apply_advanced_strategies(prices, volumes)
```

### Advanced Usage (v0.2x)
```python
from mathlib_v2 import CoreMathLibV2

# Initialize the v0.2x library
math_lib = CoreMathLibV2(
    base_volume=1.0,
    tick_freq=1.0,
    profit_coef=0.8,
    threshold=0.5
)

# Generate or load your OHLCV data
prices = [...]  # Close prices
high = [...]    # High prices
low = [...]     # Low prices
volumes = [...] # Volume data

# Apply extended v0.2x strategies
results = math_lib.apply_advanced_strategies_v2(prices, volumes, high, low)

# Access new metrics
vwap = results['vwap']
rsi = results['rsi']
kelly_fraction = results['kelly_fraction']
pair_zscore = results['pair_zscore']
```

## Key Components

### Core Mathematical Functions (v0.1)
- Price delta and returns calculation
- Volume allocation strategies
- Hash-based decision making
- Drift detection
- EMA and volatility calculations

### Technical Indicators (v0.1)
- Bollinger Bands
- Z-scores
- Momentum signals
- Mean reversion indicators
- Volume-Price Trend (VPT)

### Risk Management (v0.1)
- Risk-adjusted returns
- Sharpe ratio calculation
- Stop-loss boundaries
- Adaptive thresholds

### Advanced Features (v0.1)
- Recursive hash sequences
- Weighted sum calculations
- Cumulative drift metrics
- Price entropy analysis

### Extended Features (v0.2x)
- VWAP for execution price reference
- ATR for dynamic volatility bands
- RSI for momentum/mean-reversion signals
- Kelly Criterion for optimal position sizing
- Multi-asset correlation analysis
- Risk-parity portfolio construction
- Pair trading spread analysis
- Ornstein-Uhlenbeck process simulation
- Keltner Channels for trend following
- Memory-weighted signal decay

## Testing

Run the test files to see the library in action:

```bash
# Test v0.1 features
python test_mathlib.py

# Test v0.2x features
python test_mathlib_v2.py
```

This will generate sample data, apply various strategies, and display visualizations of the results.

## Dependencies

- numpy
- pandas
- matplotlib
- scipy
- pywt (PyWavelets)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 