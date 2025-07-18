# Schwabot Live Trading System Setup

## Overview
This is a **functional trading bot** that connects to real APIs and executes actual trades. No examples or demos - this is the production trading system.

## Prerequisites

### 1. API Keys Required
- **Binance API** (or your preferred exchange)
- **CoinGecko API** (for market data)
- **Glassnode API** (for on-chain metrics)
- **Whale Alert API** (for whale tracking)

### 2. Configuration
Edit `trading_bot_config.json`:
```json
{
  "exchange_config": {
    "apiKey": "YOUR_ACTUAL_BINANCE_API_KEY",
    "secret": "YOUR_ACTUAL_BINANCE_SECRET_KEY",
    "sandbox": false  // Set to false for live trading
  },
  "market_data_config": {
    "coingecko_api_key": "YOUR_COINGECKO_API_KEY",
    "glassnode_api_key": "YOUR_GLASSNODE_API_KEY",
    "whale_alert_api_key": "YOUR_WHALE_ALERT_API_KEY"
  }
}
```

## Trading Operations

### Execute Single Trade
```bash
python start_trading_bot.py --mode trade --symbol BTCUSDT --force-refresh
```
**Result**: Analyzes current market conditions and executes one trade if signals are positive.

### Start Automated Trading
```bash
python start_trading_bot.py --mode start-bot --interval 60
```
**Result**: Starts continuous trading loop, executing trades every 60 seconds based on your mathematical algorithms.

### Check Best Trading Opportunities
```bash
python start_trading_bot.py --mode best-phase --asset BTC
```
**Result**: Returns the most profitable phase/drift combinations from your registry data.

### Analyze Profit Vectors
```bash
python start_trading_bot.py --mode profit-vector --asset BTC --phase 0.75 --drift 0.3
```
**Result**: Shows profit analysis for specific phase/drift parameters.

## Data Flow

### Real Market Data Input
1. **CoinGecko**: Price, volume, market cap, technical indicators
2. **Glassnode**: MVRV, NVT, SOPR, on-chain metrics  
3. **Fear & Greed**: Market sentiment indicators
4. **Whale Alert**: Large transaction monitoring

### Mathematical Processing
1. **RSI, MACD, Bollinger Bands** calculated from live price data
2. **Profit vectorization** using your advanced algorithms
3. **ZPE-ZBE quantum analysis** for market synchronization
4. **CRLF temporal analysis** for strategy alignment

### Trade Execution
1. **Signal generation** based on mathematical confluence
2. **Risk assessment** using dynamic position sizing
3. **CCXT order placement** to real exchange
4. **Registry logging** for performance tracking

## Risk Management

### Automatic Protections
- Maximum 10% position size per trade
- 2% stop loss on all positions  
- 5% daily loss circuit breaker
- Dynamic position sizing based on volatility

### Manual Controls
```bash
# Stop automated trading
Ctrl+C during bot operation

# Check current positions
python start_trading_bot.py --mode profit-vector --asset CURRENT_SYMBOL
```

## Registry Tracking

### Performance Analysis
```bash
# Best performing assets
python start_trading_bot.py --mode cross-asset-best

# Recent trade history
python start_trading_bot.py --mode last-triggers --asset BTC --limit 20
```

### Data Storage
- All trades logged to `data/soulprint_registry.db`
- Market conditions recorded with each trade
- Mathematical parameters stored for analysis

## Troubleshooting

### Connection Issues
1. Verify API keys in configuration
2. Check network connectivity
3. Confirm exchange API permissions (spot trading enabled)

### Trading Issues
1. Ensure sufficient balance in exchange account
2. Verify symbol exists on exchange (BTCUSDT, ETHUSDT, etc.)
3. Check if trading is enabled for your account region

### Data Quality Issues
1. Monitor API latencies in logs
2. Verify API key rate limits not exceeded
3. Check data completeness scores

## Security Notes

- **Never commit API keys to version control**
- Use sandbox mode for testing: `"sandbox": true`
- Monitor positions regularly during live trading
- Set appropriate risk limits for your capital

## Performance Optimization

### High-Frequency Trading
- Reduce interval to 15-30 seconds for scalping
- Enable all data sources for maximum signal quality
- Use dedicated server with low latency to exchange

### Conservative Trading  
- Increase interval to 300+ seconds
- Raise confidence thresholds in configuration
- Enable additional circuit breakers

This is your **functional trading bot**. Configure it properly and it will execute real trades using your mathematical algorithms. 