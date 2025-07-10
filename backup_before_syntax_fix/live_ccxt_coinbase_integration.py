#!/usr/bin/env python3
"""
🚀 LIVE CCXT COINBASE INTEGRATION
=================================

High-throughput CCXT-based Coinbase Pro integration for Schwabot
Enables live BTC/USDC trading with our complete mathematical framework

Features:
- Real-time market data streaming
- High-frequency trade execution
- Complete integration with Schwabot's mathematical cores'
- Risk management and position sizing
- Live P&L tracking and reporting
"""

import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import ccxt.pro as ccxt
import numpy as np

# Add core to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import Schwabot core systems
    try:
    from core.clean_unified_math import clean_unified_math
    from core.trading_engine_integration import TradeExecution, TradeSignal, generate_trade_signal
    from core.unified_trade_router import UnifiedTradeRouter
    from schwabot_startup_orchestrator import SchwabotStartupOrchestrator
    CORE_AVAILABLE = True
    except ImportError as e:
    print(f"⚠️  Core import warning: {e}")
    CORE_AVAILABLE = False


@dataclass
    class LiveMarketData:
    """Live market data structure"""
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    last: float
    volume: float
    high: float
    low: float
    change: float
    change_percent: float


@dataclass
    class LivePosition:
    """Live trading position"""
    symbol: str
    side: str  # 'long' or 'short'
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    timestamp: datetime


@dataclass
    class TradingConfig:
    """Trading configuration"""
    symbol: str = "BTC/USDC"
    base_position_size: float = 0.01  # 0.01 BTC
    max_position_size: float = 0.1   # 0.1 BTC max
    risk_per_trade: float = 0.2      # 2% risk per trade
    stop_loss_pct: float = 0.15      # 1.5% stop loss
    take_profit_pct: float = 0.3     # 3% take profit
    max_daily_trades: int = 50
    trading_enabled: bool = False      # Start with paper trading

    # Coinbase Pro specific
    sandbox: bool = True  # Use sandbox for testing
    api_key: str = ""
    secret: str = ""
    passphrase: str = ""


class LiveCCXTCoinbaseIntegration:
    """
    Complete CCXT Coinbase integration for Schwabot live trading
    """

    def __init__(self, config: TradingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize CCXT exchange
        self.exchange = None
        self.initialize_exchange()

        # Trading state
        self.positions: Dict[str, LivePosition] = {}
        self.market_data: Dict[str, LiveMarketData] = {}
        self.trade_history: List[Dict] = []
        self.daily_trade_count = 0
        self.last_reset_date = datetime.now().date()

        # Schwabot integration
        self.trade_router = None
        self.math_core = None
        self.initialize_schwabot_cores()

        # Performance tracking
        self.performance_metrics = {}
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_pnl": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "start_time": datetime.now()
        }

        # Market data buffers for analysis
        self.price_buffer = []
        self.volume_buffer = []
        self.max_buffer_size = 1000

    def initialize_exchange(self) -> None:
        """Initialize CCXT Coinbase Pro exchange"""
        try:
            exchange_config = {}
                'apiKey': self.config.api_key,
                'secret': self.config.secret,
                'passphrase': self.config.passphrase,
                'sandbox': self.config.sandbox,
                'enableRateLimit': True,
                'rateLimit': 100,  # 100ms between requests
            }

            self.exchange = ccxt.coinbasepro(exchange_config)

            # Test connection
            if self.config.api_key:  # Only test if credentials provided
                markets = self.exchange.load_markets()
                self.logger.info()
                    f"✅ Connected to Coinbase Pro ({")}
                        'Sandbox' if self.config.sandbox else 'Live'})")"
                self.logger.info(f"📊 Available markets: {len(markets)}")
            else:
                self.logger.info("⚠️  No API credentials - running in data-only mode")

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize exchange: {e}")
            self.exchange = None

    def initialize_schwabot_cores(self) -> None:
        """Initialize Schwabot mathematical cores"""
        if CORE_AVAILABLE:
            try:
                self.trade_router = UnifiedTradeRouter()
                self.math_core = clean_unified_math
                self.logger.info("✅ Schwabot cores initialized")
            except Exception as e:
                self.logger.error(f"⚠️  Schwabot core initialization error: {e}")
        else:
            self.logger.warning("⚠️  Schwabot cores not available - using simplified mode")

    async def start_live_trading(self) -> None:
        """Start the complete live trading system"""
        self.logger.info("🚀 STARTING LIVE CCXT COINBASE INTEGRATION")
        self.logger.info("=" * 60)

        # Reset daily counters if new day
        self._reset_daily_counters()

        try:
            # Start concurrent tasks
            tasks = []
                self.stream_market_data(),
                self.trading_logic_loop(),
                self.risk_management_loop(),
                self.performance_monitoring_loop()
            ]

            await asyncio.gather(*tasks)

        except Exception as e:
            self.logger.error(f"❌ Live trading error: {e}")
        finally:
            await self.cleanup()

    async def stream_market_data(self) -> None:
        """Stream live market data from Coinbase Pro"""
        if not self.exchange:
            self.logger.error("❌ Exchange not initialized")
            return

        self.logger.info(f"📊 Starting market data stream for {self.config.symbol}")

        while True:
            try:
                # Fetch ticker data
                ticker = await self.exchange.fetch_ticker(self.config.symbol)

                # Create market data object
                market_data = LiveMarketData()
                    symbol=self.config.symbol,
                    timestamp=datetime.fromtimestamp(ticker['timestamp'] / 1000),
                    bid=ticker['bid'] or 0,
                    ask=ticker['ask'] or 0,
                    last=ticker['last'],
                    volume=ticker['baseVolume'] or 0,
                    high=ticker['high'] or ticker['last'],
                    low=ticker['low'] or ticker['last'],
                    change=ticker['change'] or 0,
                    change_percent=ticker['percentage'] or 0
                )

                # Update market data
                self.market_data[self.config.symbol] = market_data

                # Update price buffer for analysis
                self.price_buffer.append(market_data.last)
                self.volume_buffer.append(market_data.volume)

                # Keep buffers at max size
                if len(self.price_buffer) > self.max_buffer_size:
                    self.price_buffer = self.price_buffer[-self.max_buffer_size:]
                if len(self.volume_buffer) > self.max_buffer_size:
                    self.volume_buffer = self.volume_buffer[-self.max_buffer_size:]

                # Log market update
                if len(self.price_buffer) % 10 == 0:  # Log every 10th update
                    self.logger.info()
                        f"📈 {self.config.symbol}: ${market_data.last:.2f} "
                        f"({market_data.change_percent:+.2f}%) "
                        f"Vol: {market_data.volume:.2f}"
                    )

                await asyncio.sleep(1)  # 1 second updates

            except Exception as e:
                self.logger.error(f"❌ Market data stream error: {e}")
                await asyncio.sleep(5)  # Wait before retry

    async def trading_logic_loop(self) -> None:
        """Main trading logic loop using Schwabot intelligence"""
        self.logger.info("🧠 Starting Schwabot trading logic loop")

        while True:
            try:
                # Wait for sufficient market data
                if len(self.price_buffer) < 20:
                    await asyncio.sleep(5)
                    continue

                # Check daily trade limit
                if self.daily_trade_count >= self.config.max_daily_trades:
                    self.logger.info("⏸️  Daily trade limit reached")
                    await asyncio.sleep(60)  # Check every minute
                    continue

                # Generate Schwabot trading signal
                signal = await self.generate_schwabot_signal()

                # Execute trade if signal is strong enough
                if signal and signal.confidence > 0.7:
                    await self.execute_signal(signal)

                await asyncio.sleep(5)  # 5 second trading loop

            except Exception as e:
                self.logger.error(f"❌ Trading logic error: {e}")
                await asyncio.sleep(10)

    async def generate_schwabot_signal(self) -> Optional[TradeSignal]:
        """Generate trading signal using Schwabot's mathematical framework"""'
        try:
            if not self.trade_router or len(self.price_buffer) < 20:
                return None

            current_price = self.price_buffer[-1]
            price_history = self.price_buffer[-20:]  # Last 20 data points

            # Create market data for Schwabot
            market_data = {}
                "symbol": self.config.symbol,
                "price": current_price,
                "volume": self.volume_buffer[-1],
                "price_history": price_history,
                "timestamp": datetime.now()
            }

            # Generate signal through unified trade router
            signal = self.trade_router.route_trade_signal(market_data)

            # Enhance with additional analysis
            if signal:
                # Add volatility analysis
                price_changes = np.diff(price_history)
                volatility = np.std(price_changes) / np.mean(price_history)

                # Adjust confidence based on volatility
                if volatility > 0.5:  # High volatility
                    signal.confidence *= 0.8  # Reduce confidence
                elif volatility < 0.1:  # Low volatility
                    signal.confidence *= 1.2  # Increase confidence

                signal.confidence = min(signal.confidence, 1.0)  # Cap at 1.0

                self.logger.info()
                    f"🎯 Schwabot Signal: {signal.action} "
                    f"(Confidence: {signal.confidence:.2f}) "
                    f"Price: ${current_price:.2f}"
                )

            return signal

        except Exception as e:
            self.logger.error(f"❌ Signal generation error: {e}")
            return None

    async def execute_signal(self, signal: TradeSignal) -> bool:
        """Execute trading signal"""
        if not self.config.trading_enabled:
            self.logger.info()
                f"📝 PAPER TRADE: {"}
                    signal.action} {
                    self.config.symbol} (Confidence: {)
                    signal.confidence:.2f})")"
            return True

        try:
            current_price = self.price_buffer[-1]
            position_size = self.calculate_position_size(signal.confidence)

            if signal.action == "BUY":
                await self.place_buy_order(position_size, current_price)
            elif signal.action == "SELL":
                await self.place_sell_order(position_size, current_price)

            # Update counters
            self.daily_trade_count += 1
            self.performance_metrics["total_trades"] += 1

            return True

        except Exception as e:
            self.logger.error(f"❌ Trade execution error: {e}")
            return False

    def calculate_position_size(self, confidence: float) -> float:
        """Calculate position size based on confidence and risk management"""
        # Base size adjusted by confidence
        base_size = self.config.base_position_size * confidence

        # Ensure within limits
        position_size = min(base_size, self.config.max_position_size)

        return position_size

    async def place_buy_order(self, size: float, price: float) -> Optional[Dict]:
        """Place buy order"""
        try:
            if not self.exchange:
                return None

            order = await self.exchange.create_market_buy_order()
                self.config.symbol, size
            )

            self.logger.info(f"✅ BUY ORDER: {size} {self.config.symbol} @ ${price:.2f}")
            self.trade_history.append({)}
                "type": "BUY",
                "size": size,
                "price": price,
                "timestamp": datetime.now(),
                "order_id": order.get('id')
            })

            return order

        except Exception as e:
            self.logger.error(f"❌ Buy order error: {e}")
            return None

    async def place_sell_order(self, size: float, price: float) -> Optional[Dict]:
        """Place sell order"""
        try:
            if not self.exchange:
                return None

            order = await self.exchange.create_market_sell_order()
                self.config.symbol, size
            )

            self.logger.info(f"✅ SELL ORDER: {size} {self.config.symbol} @ ${price:.2f}")
            self.trade_history.append({)}
                "type": "SELL",
                "size": size,
                "price": price,
                "timestamp": datetime.now(),
                "order_id": order.get('id')
            })

            return order

        except Exception as e:
            self.logger.error(f"❌ Sell order error: {e}")
            return None

    async def risk_management_loop(self) -> None:
        """Risk management monitoring loop"""
        self.logger.info("🛡️  Starting risk management loop")

        while True:
            try:
                # Update positions
                await self.update_positions()

                # Check stop losses and take profits
                await self.check_exit_conditions()

                # Monitor drawdown
                self.monitor_drawdown()

                await asyncio.sleep(10)  # Check every 10 seconds

            except Exception as e:
                self.logger.error(f"❌ Risk management error: {e}")
                await asyncio.sleep(30)

    async def update_positions(self) -> None:
        """Update current positions"""
        try:
            if not self.exchange:
                return

            # Fetch current balance
            balance = await self.exchange.fetch_balance()

            # Update positions based on balance
            for currency in ['BTC', 'USDC']:
                if currency in balance and balance[currency]['total'] > 0:
                    # Position logic here
                    pass

        except Exception as e:
            self.logger.error(f"❌ Position update error: {e}")

    async def check_exit_conditions(self) -> None:
        """Check stop loss and take profit conditions"""
        # Implementation for exit condition monitoring
        pass

    def monitor_drawdown(self) -> None:
        """Monitor maximum drawdown"""
        if len(self.trade_history) > 0:
            # Calculate current drawdown
            # Implementation for drawdown calculation
            pass

    async def performance_monitoring_loop(self) -> None:
        """Performance monitoring and reporting loop"""
        self.logger.info("📊 Starting performance monitoring loop")

        while True:
            try:
                # Update performance metrics
                self.update_performance_metrics()

                # Log performance every 5 minutes
                self.log_performance_summary()

                await asyncio.sleep(300)  # 5 minutes

            except Exception as e:
                self.logger.error(f"❌ Performance monitoring error: {e}")
                await asyncio.sleep(60)

    def update_performance_metrics(self) -> None:
        """Update performance metrics"""
        if len(self.trade_history) == 0:
            return

        # Calculate win rate
        winning_trades = len([t for t in self.trade_history if t.get('pnl', 0) > 0])
        total_trades = len(self.trade_history)

        self.performance_metrics.update({)}
            "winning_trades": winning_trades,
            "total_trades": total_trades,
            "win_rate": winning_trades / total_trades if total_trades > 0 else 0,
            # Add more metrics as needed
        })

    def log_performance_summary(self) -> None:
        """Log performance summary"""
        metrics = self.performance_metrics
        self.logger.info("📊 PERFORMANCE SUMMARY")
        self.logger.info(f"   Total Trades: {metrics['total_trades']}")
        self.logger.info(f"   Win Rate: {metrics['win_rate']:.1%}")
        self.logger.info(f"   Total P&L: ${metrics['total_pnl']:.2f}")
        self.logger.info()
            f"   Daily Trades: {self.daily_trade_count}/{self.config.max_daily_trades}")

    def _reset_daily_counters(self) -> None:
        """Reset daily counters if new day"""
        current_date = datetime.now().date()
        if current_date > self.last_reset_date:
            self.daily_trade_count = 0
            self.last_reset_date = current_date
            self.logger.info("🔄 Daily counters reset")

    async def cleanup(self) -> None:
        """Cleanup resources"""
        if self.exchange:
            await self.exchange.close()
        self.logger.info("🛑 Live trading system stopped")

    def get_live_status(self) -> Dict[str, Any]:
        """Get current live trading status"""
        return {}
            "trading_enabled": self.config.trading_enabled,
            "symbol": self.config.symbol,
            "daily_trades": f"{self.daily_trade_count}/{self.config.max_daily_trades}",
            "current_price": self.price_buffer[-1] if self.price_buffer else 0,
            "market_data_points": len(self.price_buffer),
            "performance_metrics": self.performance_metrics,
            "schwabot_cores": "OPERATIONAL" if self.trade_router else "SIMPLIFIED",
            "exchange_status": "CONNECTED" if self.exchange else "DISCONNECTED"
        }


async def main():
    """Main entry point for live CCXT Coinbase integration"""
    print("🚀 SCHWABOT LIVE CCXT COINBASE INTEGRATION")
    print("=" * 60)

    # Configure logging
    logging.basicConfig()
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create trading configuration
    config = TradingConfig()
        symbol="BTC/USDC",
        trading_enabled=False,  # Start with paper trading
        sandbox=True,  # Use sandbox for testing
        base_position_size=0.01,
        max_daily_trades=50
    )

    # Initialize integration
    integration = LiveCCXTCoinbaseIntegration(config)

    # Check if Schwabot cores are available
    if CORE_AVAILABLE:
        print("✅ Schwabot mathematical cores loaded")
        # Initialize startup orchestrator for full system validation
        orchestrator = SchwabotStartupOrchestrator()
        print("🎯 Schwabot startup validation in progress...")
    else:
        print("⚠️  Running in simplified mode without Schwabot cores")

    print(f"📊 Trading pair: {config.symbol}")
    print(f"🎮 Mode: {'LIVE' if config.trading_enabled else 'PAPER'} TRADING")
    print(f"🏦 Exchange: Coinbase Pro ({'Sandbox' if config.sandbox else 'Live'})")
    print("")

    try:
        # Start live trading
        await integration.start_live_trading()

    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
    finally:
        await integration.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
