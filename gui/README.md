# 🚀 Schwabot Trading GUI

Advanced algorithmic trading interface with demo, live trading, and CCXT integration capabilities.

## 🎯 Features

### 📊 Trading Dashboard
- **Real-time Portfolio Tracking**: Monitor portfolio value, profit/loss, win rate
- **Interactive Charts**: Price and profit visualization with Chart.js
- **Trade History**: Complete session trade log with execution details
- **Live Updates**: Auto-refresh every 5 seconds

### 🎮 Trading Modes
- **Demo Mode**: Risk-free testing with simulated trades
- **Live Trading**: Real CCXT exchange integration
- **Backtest Mode**: Historical strategy validation

### 💻 CLI Command Portal
Execute trading commands directly:
```bash
trade BTC 60000 BUY 0.1    # Execute trade
mode live                  # Switch to live mode
portfolio                  # View portfolio status
```

### 🔗 CCXT Integration
- **Multi-Exchange Support**: Coinbase, Binance, Kraken
- **API Management**: Secure credential handling
- **Sandbox Testing**: Safe environment for testing

### 🧮 Mathematical Intelligence
- **Real Math Integration**: Uses `clean_unified_math` for calculations
- **Score Calculation**: Confidence-based mathematical scoring
- **Backlog System**: Save trading data for analysis

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r gui/requirements.txt
```

### 2. Run Development Server
```bash
python gui/flask_app.py
```

### 3. Access Dashboard
Open browser to: http://localhost:5000

## 🔨 Building Executable

### Create Standalone EXE
```bash
python gui/exe_launcher.py --build
```

### Run Executable
```bash
./dist/Schwabot.exe
```

## 📁 File Structure

```
gui/
├── flask_app.py              # Main Flask application
├── exe_launcher.py           # EXE bundling and launcher
├── requirements.txt          # Python dependencies
├── README.md                # This file
└── templates/
    └── dashboard.html       # Main dashboard interface
```

## 🔧 API Endpoints

### Trading Operations
- `POST /api/execute_signal` - Execute trading signal
- `POST /api/switch_mode` - Switch trading mode
- `GET /api/get_portfolio` - Get portfolio status
- `GET /api/get_trade_history` - Get trade history

### Mathematical Operations
- `POST /api/calculate_math_score` - Calculate mathematical score
- `POST /api/save_backlog` - Save data to backlog
- `GET /api/visualize_trades` - Get chart data

### Exchange Integration
- `POST /api/connect_ccxt` - Connect to CCXT exchange

### CLI Commands
- `POST /api/cli_command` - Execute CLI-style commands

## 🎯 Usage Examples

### Execute a Trade Signal
```javascript
// Via JavaScript
axios.post('/api/execute_signal', {
    asset: 'BTC/USDC',
    price: 60000.0,
    quantity: 0.1,
    mode: 'demo'
});
```

### Calculate Math Score
```javascript
axios.post('/api/calculate_math_score', {
    price: 60000.0,
    volume: 1000.0,
    confidence: 0.8
});
```

### Connect to Exchange
```javascript
axios.post('/api/connect_ccxt', {
    exchange: 'coinbase',
    api_key: 'your_api_key',
    secret: 'your_secret'
});
```

## 🛡️ Security Features

- **Session Management**: Unique session IDs for each trading session
- **Hash Generation**: SHA-256 hashing for trade signals
- **API Security**: Secure credential handling for exchanges
- **Demo Mode**: Risk-free testing environment

## 📈 Trading Logic

### Signal Execution Flow
1. **Input Validation**: Validate asset, price, quantity
2. **Math Calculation**: Use `clean_unified_math` for scoring
3. **Visual Execution**: Create `VisualExecutionNode`
4. **Trade Routing**: Route through `UnifiedTradeRouter`
5. **Portfolio Update**: Update positions and profit/loss
6. **Backlog Storage**: Save trade data for analysis

### Portfolio Management
- **Position Tracking**: Real-time position monitoring
- **Profit Calculation**: Automatic P&L calculation
- **Risk Management**: Position sizing and risk controls
- **Performance Metrics**: Win rate and performance tracking

## 🔮 Advanced Features

### Backlog System
- **Data Persistence**: Save trading decisions and outcomes
- **Analysis Support**: Historical data for strategy improvement
- **Hash Tracking**: Unique identifiers for each data point

### Visualization
- **Real-time Charts**: Live price and profit charts
- **Trade Markers**: Visual indicators for buy/sell points
- **Performance Metrics**: Portfolio performance visualization

### CLI Integration
- **Command Parsing**: Natural language command processing
- **Batch Operations**: Execute multiple commands
- **Help System**: Built-in command documentation

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install -r gui/requirements.txt
   ```

2. **Port Already in Use**
   ```bash
   # Change port in flask_app.py
   app.run(port=5001)
   ```

3. **CCXT Connection Issues**
   - Verify API credentials
   - Check exchange status
   - Use sandbox mode for testing

### Debug Mode
```bash
# Enable debug logging
export FLASK_DEBUG=1
python gui/flask_app.py
```

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review API documentation
3. Test with demo mode first
4. Verify all dependencies are installed

## 🔄 Updates

The GUI system integrates with the core Schwabot components:
- `core/visual_execution_node.py` - Visual trade execution
- `core/unified_trade_router.py` - Trade routing logic
- `core/clean_unified_math.py` - Mathematical calculations
- `core/ccxt_integration.py` - Exchange connectivity

All updates to core components are automatically reflected in the GUI. 