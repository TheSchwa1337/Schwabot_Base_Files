# Schwabot Trading System

A functional trading bot that executes real trades using advanced mathematical algorithms.

## What This Does

1. **Connects to real exchanges** via CCXT (Binance, etc.)
2. **Pulls live market data** from CoinGecko, Glassnode, Fear & Greed
3. **Processes data** through your mathematical algorithms (profit vectorization, ZPE-ZBE, CRLF)
4. **Generates trading signals** based on RSI, MACD, sentiment, on-chain metrics
5. **Executes real trades** with automatic stop losses and position sizing
6. **Logs everything** to registry for performance tracking

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API keys** in `trading_bot_config.json`:
   ```json
   {
     "exchange_config": {
       "apiKey": "YOUR_BINANCE_API_KEY",
       "secret": "YOUR_BINANCE_SECRET_KEY",
       "sandbox": true
     }
   }
   ```

3. **Execute a trade:**
   ```bash
   python start_trading_bot.py --mode trade --symbol BTCUSDT
   ```

4. **Start automated trading:**
   ```bash
   python start_trading_bot.py --mode start-bot --interval 60
   ```

## System Components

- **`core/cli_live_entry.py`** - Main trading bot interface
- **`core/clean_trading_pipeline.py`** - Trading logic and signal generation
- **`core/unified_market_data_pipeline.py`** - Live market data aggregation
- **`core/ccxt_trading_executor.py`** - Real exchange order execution
- **`core/soulprint_registry.py`** - Trade logging and performance tracking

## No Examples, No Demos

This is a **production trading system**. Every command executes real trades or analyzes real market data. No simulations, no examples, no demos.

## Risk Warning

This system executes real trades with real money. Use sandbox mode for testing. Monitor positions regularly. Set appropriate risk limits.
