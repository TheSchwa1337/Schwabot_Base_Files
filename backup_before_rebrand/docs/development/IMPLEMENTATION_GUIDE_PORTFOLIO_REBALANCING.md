# Portfolio Rebalancing & API Integration Implementation Guide
## Schwabot Trading System - Complete Setup & Configuration

### 📋 Table of Contents
1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Installation & Setup](#installation--setup)
4. [Configuration](#configuration)
5. [API Key Management](#api-key-management)
6. [Usage Examples](#usage-examples)
7. [Advanced Configuration](#advanced-configuration)
8. [Monitoring & Alerts](#monitoring--alerts)
9. [Security Considerations](#security-considerations)
10. [Troubleshooting](#troubleshooting)
11. [Best Practices](#best-practices)
12. [Performance Optimization](#performance-optimization)

---

## 🎯 Overview

This implementation provides a complete solution for portfolio rebalancing, real-time price integration, and API connectivity in the Schwabot trading system. The system includes:

- **Real-time Market Data**: WebSocket connections to major exchanges (Binance, Coinbase, Kraken)
- **Enhanced Portfolio Tracker**: Automatic price updates and position tracking
- **Intelligent Rebalancing**: Threshold-based and time-based portfolio rebalancing
- **Multi-Exchange Support**: Unified interface for multiple exchanges via CCXT
- **Comprehensive Testing**: Full integration test suite

### Key Features
- ✅ Real-time price updates via WebSocket
- ✅ Automatic portfolio rebalancing
- ✅ Multi-exchange support
- ✅ Comprehensive error handling
- ✅ Performance monitoring
- ✅ Security best practices

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Schwabot Trading System                  │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   Portfolio     │  │   Real-Time     │  │   Exchange   │ │
│  │   Tracker       │◄─┤   Market Data   │◄─┤   Manager    │ │
│  │                 │  │   Integration   │  │              │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
│           │                       │                │        │
│           ▼                       ▼                ▼        │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │  Rebalancing    │  │  WebSocket      │  │  CCXT        │ │
│  │  Engine         │  │  Connections    │  │  Integration │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Component Overview

1. **Enhanced Portfolio Tracker** (`core/enhanced_portfolio_tracker.py`)
   - Real-time position tracking
   - Automatic price updates
   - Portfolio rebalancing logic

2. **Real-Time Market Data Integration** (`core/real_time_market_data_integration.py`)
   - WebSocket connections to exchanges
   - Price caching and validation
   - Callback system for price updates

3. **Exchange Connection Manager** (`core/api/exchange_connection.py`)
   - CCXT-based exchange connections
   - Order placement and management
   - Balance synchronization

---

## 🚀 Installation & Setup

### Prerequisites

```bash
# Python 3.8+ required
python --version

# Install required packages
pip install -r requirements.txt
```

### Required Dependencies

```txt
ccxt>=4.0.0
websockets>=11.0.0
aiohttp>=3.8.0
asyncio
logging
json
time
dataclasses
typing
```

### Quick Setup

```bash
# 1. Clone the repository
git clone <repository-url>
cd schwabot-trading-system

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API keys
cp config/api_keys.json.example config/api_keys.json
# Edit config/api_keys.json with your API keys

# 4. Run the test suite
python test_portfolio_rebalancing_integration.py
```

---

## ⚙️ Configuration

### Basic Configuration

Create a configuration file `config/portfolio_config.yaml`:

```yaml
# Exchange Configuration
exchanges:
  binance:
    enabled: true
    websocket_enabled: true
    symbols: ['btcusdt', 'ethusdt', 'solusdt']
    sandbox: true
    rate_limit_delay: 1
  
  coinbase:
    enabled: true
    websocket_enabled: true
    symbols: ['BTC-USD', 'ETH-USD', 'SOL-USD']
    sandbox: true
    rate_limit_delay: 1
  
  kraken:
    enabled: true
    websocket_enabled: true
    symbols: ['XBT/USD', 'ETH/USD', 'SOL/USD']
    sandbox: true
    rate_limit_delay: 1

# Portfolio Configuration
tracked_symbols: ['BTC/USD', 'ETH/USD', 'SOL/USD']
price_update_interval: 5  # seconds

# Rebalancing Configuration
rebalancing:
  enabled: true
  threshold: 0.05  # 5% deviation threshold
  interval: 3600   # Check every hour
  max_rebalancing_cost: 0.01  # 1% max cost
  target_allocation:
    BTC: 0.6
    ETH: 0.3
    SOL: 0.1
  rebalancing_strategy: 'threshold_based'  # 'threshold_based', 'time_based', 'risk_adjusted'

# Market Data Configuration
reconnect_delay: 5
max_reconnect_attempts: 10
price_cache_ttl: 60  # seconds
```

### Environment Variables

Set up environment variables for API keys:

```bash
# Binance
export BINANCE_API_KEY="your_binance_api_key"
export BINANCE_SECRET="your_binance_secret"

# Coinbase
export COINBASE_API_KEY="your_coinbase_api_key"
export COINBASE_SECRET="your_coinbase_secret"
export COINBASE_PASSPHRASE="your_coinbase_passphrase"

# Kraken
export KRAKEN_API_KEY="your_kraken_api_key"
export KRAKEN_SECRET="your_kraken_secret"
```

---

## 🔑 API Key Management

### Secure API Key Storage

1. **Environment Variables** (Recommended)
```bash
# Add to your shell profile (.bashrc, .zshrc, etc.)
export BINANCE_API_KEY="your_key"
export BINANCE_SECRET="your_secret"
```

2. **Configuration File** (Development only)
```json
{
  "binance": {
    "api_key": "your_api_key",
    "secret": "your_secret",
    "sandbox": true
  }
}
```

### API Key Permissions

For each exchange, ensure your API keys have the following permissions:

- **Binance**: Spot Trading, Read Info
- **Coinbase**: View, Transfer, Trade
- **Kraken**: Query Funds, Query Open Orders, Add/Edit Orders

### Security Best Practices

```python
# Never log API secrets
logger.info("Connecting to exchange...")  # ✅ Good
logger.info(f"API Key: {api_key}")       # ❌ Bad

# Use environment variables
import os
api_key = os.getenv('BINANCE_API_KEY')   # ✅ Good
api_key = "hardcoded_key"                # ❌ Bad

# Validate API keys
if not api_key or not secret:
    raise ValueError("API credentials not configured")
```

---

## 💡 Usage Examples

### Basic Portfolio Tracker

```python
import asyncio
from core.enhanced_portfolio_tracker import EnhancedPortfolioTracker

async def basic_example():
    # Configuration
    config = {
        'exchanges': {
            'binance': {
                'enabled': True,
                'websocket_enabled': True,
                'symbols': ['btcusdt', 'ethusdt']
            }
        },
        'tracked_symbols': ['BTC/USD', 'ETH/USD'],
        'price_update_interval': 5,
        'rebalancing': {
            'enabled': True,
            'threshold': 0.05,
            'interval': 3600,
            'target_allocation': {
                'BTC': 0.6,
                'ETH': 0.4
            }
        }
    }
    
    # Initialize tracker
    tracker = EnhancedPortfolioTracker(config)
    
    # Add callbacks
    def price_callback(price_update):
        print(f"Price update: {price_update.symbol} = ${price_update.price}")
    
    def rebalancing_callback(action, result):
        print(f"Rebalancing: {action.symbol} {action.action} ${action.amount}")
    
    tracker.add_price_update_callback(price_callback)
    tracker.add_rebalancing_callback(rebalancing_callback)
    
    # Start tracker
    await tracker.start()
    
    try:
        # Run for 60 seconds
        await asyncio.sleep(60)
        
        # Get summary
        summary = tracker.get_enhanced_summary()
        print(f"Portfolio value: ${summary['total_value']:.2f}")
        
    finally:
        await tracker.stop()

# Run the example
asyncio.run(basic_example())
```

### Advanced Portfolio Management

```python
async def advanced_example():
    config = {
        'exchanges': {
            'binance': {'enabled': True, 'websocket_enabled': True, 'symbols': ['btcusdt', 'ethusdt']},
            'coinbase': {'enabled': True, 'websocket_enabled': True, 'symbols': ['BTC-USD', 'ETH-USD']}
        },
        'tracked_symbols': ['BTC/USD', 'ETH/USD'],
        'price_update_interval': 5,
        'rebalancing': {
            'enabled': True,
            'threshold': 0.05,
            'interval': 1800,  # 30 minutes
            'target_allocation': {
                'BTC': 0.7,
                'ETH': 0.3
            },
            'rebalancing_strategy': 'risk_adjusted'
        }
    }
    
    tracker = EnhancedPortfolioTracker(config)
    
    # Add initial positions
    tracker.open_position('BTC/USD', 0.1, 50000, 'buy')
    tracker.open_position('ETH/USD', 1.0, 3000, 'buy')
    
    await tracker.start()
    
    try:
        # Monitor for 2 hours
        for _ in range(24):  # 24 * 5 minutes = 2 hours
            await asyncio.sleep(300)  # 5 minutes
            
            # Get performance metrics
            metrics = await tracker.get_performance_metrics()
            print(f"Total PnL: ${metrics['total_pnl']:.2f}")
            print(f"Rebalancing efficiency: {metrics['rebalancing_efficiency']:.2f}")
            
            # Check if rebalancing is needed
            rebalancing_check = await tracker.check_rebalancing_needs()
            if rebalancing_check['needs_rebalancing']:
                print("🔄 Rebalancing needed!")
                
    finally:
        await tracker.stop()
```

### Custom Rebalancing Strategy

```python
class CustomRebalancingStrategy:
    def __init__(self, tracker):
        self.tracker = tracker
    
    async def custom_rebalancing_check(self):
        """Custom rebalancing logic based on market conditions."""
        summary = self.tracker.get_portfolio_summary()
        
        # Get market volatility
        volatility = await self.calculate_market_volatility()
        
        # Adjust threshold based on volatility
        if volatility > 0.1:  # High volatility
            threshold = 0.03  # Tighter rebalancing
        else:
            threshold = 0.05  # Normal rebalancing
        
        # Check rebalancing needs with custom threshold
# Portfolio Rebalancing & API Integration Implementation Guide
## Schwabot Trading System - Complete Setup & Usage

### 🎯 **OVERVIEW**

This guide provides step-by-step instructions for implementing and using the enhanced portfolio rebalancing and API integration features in the Schwabot trading system. The new implementation includes:

- **Real-time market data** from multiple exchanges (Binance, Coinbase, Kraken)
- **Automatic portfolio rebalancing** with configurable thresholds
- **Multi-exchange support** with CCXT integration
- **Secure API key management** with environment variables
- **Comprehensive error handling** and reconnection logic

---

## 🚀 **QUICK START**

### **1. Install Dependencies**

```bash
# Install required packages
pip install ccxt websockets aiohttp asyncio

# Verify installation
python -c "import ccxt; print(f'CCXT version: {ccxt.__version__}')"
```

### **2. Set Up Environment Variables**

```bash
# Create .env file from template
cp config/production.env.template .env

# Edit .env with your API keys
nano .env
```

**Required Environment Variables:**
```bash
# Binance API (for testnet)
BINANCE_API_KEY=your_binance_testnet_api_key
BINANCE_API_SECRET=your_binance_testnet_secret

# Coinbase API (for sandbox)
COINBASE_API_KEY=your_coinbase_sandbox_api_key
COINBASE_API_SECRET=your_coinbase_sandbox_secret
COINBASE_PASSPHRASE=your_coinbase_passphrase

# Trading mode
SCHWABOT_TRADING_MODE=sandbox
SCHWABOT_LIVE_TRADING_ENABLED=false
```

### **3. Basic Configuration**

Create a configuration file `config/portfolio_rebalancing_config.yaml`:

```yaml
# Exchange Configuration
exchanges:
  binance:
    enabled: true
    websocket_enabled: true
    symbols: ['btcusdt', 'ethusdt', 'solusdt']
    sandbox: true
    api_key: "${BINANCE_API_KEY}"
    secret: "${BINANCE_API_SECRET}"
    
  coinbase:
    enabled: true
    websocket_enabled: true
    symbols: ['BTC-USD', 'ETH-USD', 'SOL-USD']
    sandbox: true
    api_key: "${COINBASE_API_KEY}"
    secret: "${COINBASE_API_SECRET}"
    passphrase: "${COINBASE_PASSPHRASE}"

# Rebalancing Configuration
rebalancing:
  enabled: true
  rebalance_threshold: 0.05  # 5% deviation threshold
  rebalance_interval: 3600.0  # 1 hour between rebalances
  max_rebalance_frequency: 300.0  # 5 minutes minimum between checks
  target_allocation:
    BTC: 0.4  # 40% Bitcoin
    ETH: 0.3  # 30% Ethereum
    USDC: 0.3  # 30% Stablecoin
  min_order_size: 10.0  # Minimum order size in USD
  slippage_tolerance: 0.001  # 0.1% slippage tolerance

# System Configuration
price_update_interval: 5  # Price updates every 5 seconds
tracked_symbols: ['BTC/USD', 'ETH/USD', 'SOL/USD']
reconnect_delay: 5
max_reconnect_attempts: 5
price_cache_ttl: 60
```

### **4. Run the Test Suite**

```bash
# Test the complete integration
python test_portfolio_rebalancing_integration.py
```

---

## 📊 **USAGE EXAMPLES**

### **Example 1: Basic Portfolio Tracker**

```python
import asyncio
from core.enhanced_portfolio_tracker import EnhancedPortfolioTracker

async def basic_portfolio_tracker():
    """Basic portfolio tracker example."""
    
    # Configuration
    config = {
        'exchanges': {
            'binance': {
                'enabled': True,
                'websocket_enabled': True,
                'symbols': ['btcusdt', 'ethusdt'],
                'sandbox': True
            }
        },
        'rebalancing': {
            'enabled': True,
            'target_allocation': {'BTC': 0.5, 'ETH': 0.5}
        }
    }
    
    # Create tracker
    tracker = EnhancedPortfolioTracker(config)
    
    # Add callbacks
    def price_callback(price_update):
        print(f"Price: {price_update.symbol} = ${price_update.price}")
    
    def rebalancing_callback(action, result):
        print(f"Rebalancing: {action.symbol} {action.action} {action.amount}")
    
    tracker.add_price_update_callback(price_callback)
    tracker.add_rebalancing_callback(rebalancing_callback)
    
    # Start tracker
    await tracker.start()
    
    # Run for 60 seconds
    await asyncio.sleep(60)
    
    # Get summary
    summary = tracker.get_enhanced_portfolio_summary()
    print(f"Portfolio Summary: {summary}")
    
    # Stop tracker
    await tracker.stop()

# Run the example
asyncio.run(basic_portfolio_tracker())
```

### **Example 2: Custom Rebalancing Strategy**

```python
import asyncio
from core.enhanced_portfolio_tracker import EnhancedPortfolioTracker

async def custom_rebalancing_strategy():
    """Custom rebalancing strategy example."""
    
    # Aggressive rebalancing configuration
    config = {
        'exchanges': {
            'binance': {'enabled': True, 'websocket_enabled': True, 'sandbox': True},
            'coinbase': {'enabled': True, 'websocket_enabled': True, 'sandbox': True}
        },
        'rebalancing': {
            'enabled': True,
            'rebalance_threshold': 0.02,  # 2% threshold (more aggressive)
            'rebalance_interval': 1800.0,  # 30 minutes
            'target_allocation': {
                'BTC': 0.6,   # 60% Bitcoin
                'ETH': 0.25,  # 25% Ethereum
                'SOL': 0.1,   # 10% Solana
                'USDC': 0.05  # 5% Stablecoin
            },
            'min_order_size': 5.0,  # Smaller minimum orders
            'slippage_tolerance': 0.002  # 0.2% slippage tolerance
        }
    }
    
    tracker = EnhancedPortfolioTracker(config)
    
    # Start tracker
    await tracker.start()
    
    # Monitor for 2 hours
    await asyncio.sleep(7200)
    
    # Get statistics
    stats = tracker.get_rebalancing_statistics()
    print(f"Rebalancing Statistics: {stats}")
    
    await tracker.stop()

# Run the example
asyncio.run(custom_rebalancing_strategy())
```

### **Example 3: Real-time Market Data Only**

```python
import asyncio
from core.real_time_market_data_integration import RealTimeMarketDataIntegration

async def market_data_only():
    """Real-time market data example."""
    
    config = {
        'exchanges': {
            'binance': {
                'websocket_enabled': True,
                'symbols': ['btcusdt', 'ethusdt', 'solusdt']
            },
            'coinbase': {
                'websocket_enabled': True,
                'symbols': ['BTC-USD', 'ETH-USD', 'SOL-USD']
            }
        }
    }
    
    market_data = RealTimeMarketDataIntegration(config)
    
    # Add callback for price updates
    def price_callback(price_update):
        print(f"{price_update.exchange}: {price_update.symbol} = ${price_update.price}")
    
    market_data.add_price_callback(price_callback)
    
    # Start market data
    await market_data.start()
    
    # Run for 30 seconds
    await asyncio.sleep(30)
    
    # Get statistics
    stats = market_data.get_statistics()
    print(f"Market Data Statistics: {stats}")
    
    # Get current prices
    prices = market_data.get_all_prices()
    print(f"Current Prices: {prices}")
    
    await market_data.stop()

# Run the example
asyncio.run(market_data_only())
```

---

## 🔧 **ADVANCED CONFIGURATION**

### **Multi-Exchange Setup**

```yaml
# config/advanced_exchanges.yaml
exchanges:
  binance:
    enabled: true
    websocket_enabled: true
    symbols: ['btcusdt', 'ethusdt', 'solusdt', 'adausdt']
    sandbox: true
    api_key: "${BINANCE_API_KEY}"
    secret: "${BINANCE_API_SECRET}"
    rate_limit: 1200  # requests per minute
    
  coinbase:
    enabled: true
    websocket_enabled: true
    symbols: ['BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD']
    sandbox: true
    api_key: "${COINBASE_API_KEY}"
    secret: "${COINBASE_API_SECRET}"
    passphrase: "${COINBASE_PASSPHRASE}"
    rate_limit: 100  # requests per minute
    
  kraken:
    enabled: true
    websocket_enabled: true
    symbols: ['XBT/USD', 'ETH/USD', 'SOL/USD', 'ADA/USD']
    sandbox: false  # Kraken doesn't have sandbox
    api_key: "${KRAKEN_API_KEY}"
    secret: "${KRAKEN_API_SECRET}"
    rate_limit: 15  # requests per 15 seconds
```

### **Dynamic Rebalancing Strategy**

```python
class DynamicRebalancingStrategy:
    """Dynamic rebalancing based on market conditions."""
    
    def __init__(self, tracker: EnhancedPortfolioTracker):
        self.tracker = tracker
        self.market_conditions = {}
    
    async def update_market_conditions(self):
        """Update market conditions and adjust rebalancing strategy."""
        # Get market volatility
        volatility = self.calculate_market_volatility()
        
        # Adjust rebalancing threshold based on volatility
        if volatility > 0.8:  # High volatility
            self.tracker.rebalancing_config.rebalance_threshold = 0.03  # 3%
        elif volatility > 0.5:  # Medium volatility
            self.tracker.rebalancing_config.rebalance_threshold = 0.05  # 5%
        else:  # Low volatility
            self.tracker.rebalancing_config.rebalance_threshold = 0.08  # 8%
    
    def calculate_market_volatility(self):
        """Calculate market volatility from price data."""
        # Implementation here
        return 0.5  # Placeholder
```

### **Risk Management Integration**

```python
class RiskManagedRebalancing:
    """Risk-managed rebalancing with position limits."""
    
    def __init__(self, tracker: EnhancedPortfolioTracker):
        self.tracker = tracker
        self.max_position_size = 0.2  # 20% max per position
        self.max_daily_trades = 10
        self.daily_trades = 0
    
    async def execute_risk_managed_rebalancing(self, actions):
        """Execute rebalancing with risk management."""
        for action in actions:
            # Check position size limits
            if action.amount > self.tracker.get_portfolio_summary()['total_value'] * self.max_position_size:
                logger.warning(f"Position size too large: {action.symbol}")
                continue
            
            # Check daily trade limits
            if self.daily_trades >= self.max_daily_trades:
                logger.warning("Daily trade limit reached")
                break
            
            # Execute trade
            await self.tracker.execute_rebalancing([action])
            self.daily_trades += 1
```

---

## 📈 **MONITORING & ALERTS**

### **Portfolio Monitoring Dashboard**

```python
import asyncio
import time
from core.enhanced_portfolio_tracker import EnhancedPortfolioTracker

class PortfolioMonitor:
    """Real-time portfolio monitoring dashboard."""
    
    def __init__(self, tracker: EnhancedPortfolioTracker):
        self.tracker = tracker
        self.alerts = []
    
    async def start_monitoring(self):
        """Start real-time monitoring."""
        while True:
            try:
                # Get portfolio summary
                summary = self.tracker.get_enhanced_portfolio_summary()
                
                # Check for alerts
                await self.check_alerts(summary)
                
                # Print dashboard
                self.print_dashboard(summary)
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def check_alerts(self, summary):
        """Check for alert conditions."""
        # Check for large deviations
        for asset, allocation in summary['rebalancing']['target_allocation'].items():
            current_value = summary['balances'].get(asset, 0)
            total_value = summary['total_value']
            current_allocation = current_value / total_value if total_value > 0 else 0
            
            deviation = abs(current_allocation - allocation)
            if deviation > 0.1:  # 10% deviation
                alert = f"Large deviation in {asset}: {deviation:.1%}"
                self.alerts.append(alert)
                logger.warning(alert)
    
    def print_dashboard(self, summary):
        """Print monitoring dashboard."""
        print("\n" + "="*60)
        print("PORTFOLIO MONITORING DASHBOARD")
        print("="*60)
        print(f"Total Value: ${summary['total_value']:,.2f}")
        print(f"Realized PnL: ${summary['realized_pnl']:,.2f}")
        print(f"Unrealized PnL: ${summary['unrealized_pnl']:,.2f}")
        print("\nAllocations:")
        for asset, value in summary['balances'].items():
            allocation = value / summary['total_value'] * 100
            print(f"  {asset}: {allocation:.1f}% (${value:,.2f})")
        print("\nAlerts:")
        for alert in self.alerts[-5:]:  # Last 5 alerts
            print(f"  ⚠️ {alert}")
        print("="*60)
```

### **Email Alerts**

```python
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class EmailAlertSystem:
    """Email alert system for portfolio events."""
    
    def __init__(self, smtp_config):
        self.smtp_config = smtp_config
    
    async def send_rebalancing_alert(self, action, result):
        """Send email alert for rebalancing events."""
        subject = f"Portfolio Rebalancing: {action.symbol} {action.action}"
        
        body = f"""
        Portfolio Rebalancing Event
        
        Symbol: {action.symbol}
        Action: {action.action}
        Amount: ${action.amount:,.2f}
        Current Allocation: {action.current_allocation:.1%}
        Target Allocation: {action.target_allocation:.1%}
        Deviation: {action.deviation:.1%}
        
        Order Result: {result.get('success', False)}
        Order ID: {result.get('order', {}).get('id', 'N/A')}
        
        Time: {time.strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        await self.send_email(subject, body)
    
    async def send_email(self, subject, body):
        """Send email using SMTP."""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.smtp_config['username']
            msg['To'] = self.smtp_config['recipients']
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.smtp_config['server'], self.smtp_config['port'])
            server.starttls()
            server.login(self.smtp_config['username'], self.smtp_config['password'])
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email alert sent: {subject}")
            
        except Exception as e:
            logger.error(f"Email alert failed: {e}")
```

---

## 🔒 **SECURITY CONSIDERATIONS**

### **API Key Security**

```python
import os
from cryptography.fernet import Fernet

class SecureAPIKeyManager:
    """Secure API key management with encryption."""
    
    def __init__(self, encryption_key=None):
        self.encryption_key = encryption_key or Fernet.generate_key()
        self.cipher = Fernet(self.encryption_key)
    
    def encrypt_api_key(self, api_key: str) -> bytes:
        """Encrypt API key."""
        return self.cipher.encrypt(api_key.encode())
    
    def decrypt_api_key(self, encrypted_key: bytes) -> str:
        """Decrypt API key."""
        return self.cipher.decrypt(encrypted_key).decode()
    
    def store_encrypted_keys(self, keys: dict):
        """Store encrypted API keys."""
        encrypted_keys = {}
        for exchange, key_data in keys.items():
            encrypted_keys[exchange] = {
                'api_key': self.encrypt_api_key(key_data['api_key']),
                'secret': self.encrypt_api_key(key_data['secret'])
            }
        
        # Store encrypted keys securely
        with open('encrypted_keys.bin', 'wb') as f:
            import pickle
            pickle.dump(encrypted_keys, f)
    
    def load_encrypted_keys(self) -> dict:
        """Load and decrypt API keys."""
        with open('encrypted_keys.bin', 'rb') as f:
            import pickle
            encrypted_keys = pickle.load(f)
        
        decrypted_keys = {}
        for exchange, key_data in encrypted_keys.items():
            decrypted_keys[exchange] = {
                'api_key': self.decrypt_api_key(key_data['api_key']),
                'secret': self.decrypt_api_key(key_data['secret'])
            }
        
        return decrypted_keys
```

### **Network Security**

```python
class NetworkSecurityManager:
    """Network security and rate limiting."""
    
    def __init__(self):
        self.rate_limits = {}
        self.ip_whitelist = []
        self.request_history = []
    
    def check_rate_limit(self, exchange: str, request_type: str) -> bool:
        """Check if request is within rate limits."""
        current_time = time.time()
        key = f"{exchange}_{request_type}"
        
        if key not in self.rate_limits:
            self.rate_limits[key] = []
        
        # Remove old requests
        self.rate_limits[key] = [
            req_time for req_time in self.rate_limits[key]
            if current_time - req_time < 60  # 1 minute window
        ]
        
        # Check limit
        limit = self.get_rate_limit(exchange, request_type)
        if len(self.rate_limits[key]) >= limit:
            return False
        
        # Add current request
        self.rate_limits[key].append(current_time)
        return True
    
    def get_rate_limit(self, exchange: str, request_type: str) -> int:
        """Get rate limit for exchange and request type."""
        limits = {
            'binance': {'public': 1200, 'private': 1200},
            'coinbase': {'public': 100, 'private': 100},
            'kraken': {'public': 15, 'private': 15}
        }
        return limits.get(exchange, {}).get(request_type, 100)
```

---

## 🚨 **TROUBLESHOOTING**

### **Common Issues and Solutions**

#### **1. Exchange Connection Failures**

**Problem**: Cannot connect to exchanges
```python
# Check exchange status
status = exchange_manager.get_all_status()
print(f"Exchange Status: {status}")

# Test individual connections
for name, connection in exchange_manager.connections.items():
    health = await connection.health_check()
    print(f"{name}: {'✅' if health else '❌'}")
```

**Solutions**:
- Verify API keys are correct
- Check network connectivity
- Ensure sandbox mode is enabled for testing
- Check exchange API status pages

#### **2. WebSocket Connection Issues**

**Problem**: No real-time price updates
```python
# Check WebSocket status
stats = market_data.get_statistics()
print(f"WebSocket Stats: {stats}")

# Check connection status
status = market_data.get_connection_status()
print(f"Connection Status: {status}")
```

**Solutions**:
- Check firewall settings
- Verify WebSocket URLs are accessible
- Restart WebSocket connections
- Check exchange WebSocket status

#### **3. Rebalancing Not Triggering**

**Problem**: Portfolio not rebalancing automatically
```python
# Check rebalancing configuration
config = tracker.rebalancing_config
print(f"Rebalancing Config: {config}")

# Force rebalancing check
await tracker.force_rebalancing()

# Check portfolio allocation
summary = tracker.get_enhanced_portfolio_summary()
print(f"Portfolio Allocation: {summary['balances']}")
```

**Solutions**:
- Verify rebalancing is enabled
- Check threshold settings
- Ensure minimum order size is met
- Check time intervals

#### **4. Performance Issues**

**Problem**: High CPU/memory usage
```python
# Monitor system resources
import psutil

def monitor_resources():
    cpu_percent = psutil.cpu_percent()
    memory_percent = psutil.memory_percent()
    print(f"CPU: {cpu_percent}%, Memory: {memory_percent}%")
```

**Solutions**:
- Increase price update intervals
- Reduce number of tracked symbols
- Optimize callback functions
- Use connection pooling

---

## 📚 **API REFERENCE**

### **EnhancedPortfolioTracker**

```python
class EnhancedPortfolioTracker:
    """Enhanced portfolio tracker with real-time price integration."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize enhanced portfolio tracker."""
    
    async def start(self):
        """Start the enhanced portfolio tracker."""
    
    async def stop(self):
        """Stop the enhanced portfolio tracker."""
    
    async def check_rebalancing_needs(self) -> bool:
        """Check if portfolio needs rebalancing."""
    
    async def execute_rebalancing(self, actions: List[RebalancingAction]):
        """Execute rebalancing actions."""
    
    def add_price_update_callback(self, callback: Callable):
        """Add callback for price updates."""
    
    def add_rebalancing_callback(self, callback: Callable):
        """Add callback for rebalancing events."""
    
    def get_enhanced_portfolio_summary(self) -> Dict[str, Any]:
        """Get enhanced portfolio summary."""
    
    def get_rebalancing_statistics(self) -> Dict[str, Any]:
        """Get rebalancing statistics."""
```

### **RealTimeMarketDataIntegration**

```python
class RealTimeMarketDataIntegration:
    """Real-time market data integration."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize market data integration."""
    
    async def start(self):
        """Start WebSocket connections."""
    
    async def stop(self):
        """Stop WebSocket connections."""
    
    def add_price_callback(self, callback: Callable):
        """Add price update callback."""
    
    def get_price(self, symbol: str, exchange: str = None) -> Optional[Dict]:
        """Get current price for symbol."""
    
    def get_all_prices(self) -> Dict[str, Dict]:
        """Get all current prices."""
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get connection statistics."""
```

### **ExchangeManager**

```python
class ExchangeManager:
    """Multi-exchange connection manager."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize exchange manager."""
    
    async def connect_all(self):
        """Connect to all exchanges."""
    
    async def disconnect_all(self):
        """Disconnect from all exchanges."""
    
    def get_connection(self, exchange_name: str) -> Optional[ExchangeConnection]:
        """Get specific exchange connection."""
    
    def get_all_status(self) -> Dict[str, Any]:
        """Get status of all connections."""
    
    async def health_check_all(self) -> Dict[str, bool]:
        """Health check all connections."""
```

---

## 🎯 **BEST PRACTICES**

### **1. Configuration Management**

- Use environment variables for sensitive data
- Keep configuration files in version control (without secrets)
- Use different configurations for development and production
- Validate configuration on startup

### **2. Error Handling**

- Implement comprehensive error handling
- Use exponential backoff for retries
- Log all errors with context
- Implement circuit breakers for failing services

### **3. Monitoring**

- Monitor all exchange connections
- Track rebalancing performance
- Alert on large deviations
- Monitor system resources

### **4. Security**

- Never log API secrets
- Use encrypted storage for sensitive data
- Implement rate limiting
- Use IP whitelisting when possible

### **5. Performance**

- Use connection pooling
- Implement caching for price data
- Optimize callback functions
- Monitor memory usage

---

## 📞 **SUPPORT**

For issues and questions:

1. **Check the troubleshooting section** above
2. **Run the test suite** to validate your setup
3. **Review the logs** for detailed error information
4. **Check exchange API status** pages
5. **Verify your configuration** matches the examples

**Log Files**:
- `logs/portfolio_tracker.log` - Portfolio tracker logs
- `logs/market_data.log` - Market data integration logs
- `logs/exchange_connections.log` - Exchange connection logs

---

*This implementation guide provides everything needed to set up and use the enhanced portfolio rebalancing and API integration features in the Schwabot trading system.* 