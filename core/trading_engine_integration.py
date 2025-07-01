#!/usr/bin/env python3
"""
Schwabot Trading Engine Integration
==================================

Integrates with Lantern Core and mathematical framework to provide:
- Live trading via CCXT/Coinbase
- Demo/simulation trading
- Historical data integration via CSV
- Advanced entry/exit logic
- Risk management
- Portfolio tracking
"""

import asyncio
import csv
import hashlib
import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum

import ccxt
import pandas as pd
import numpy as np
from cryptography.fernet import Fernet

# Import Schwabot's core systems
try:
    from utils.secure_config_manager import get_secure_api_key
    from utils.price_bridge import get_secure_price, PriceData
    from core.secure_api_coordinator import SecureAPICoordinator, APIProvider
    from core.unified_math_system import UnifiedMathSystem
    from core.enhanced_tcell_system import EnhancedTCellSystem
    from core.strategy_logic import StrategyLogic
    from core.risk_manager import RiskManager
    from core.portfolio_tracker import PortfolioTracker
except ImportError as e:
    logging.warning(f"Some Schwabot core modules unavailable: {e}")

logger = logging.getLogger(__name__)


class TradingMode(Enum):
    """Trading modes."""

    LIVE = "live"
    DEMO = "demo"
    SIMULATION = "simulation"


class OrderType(Enum):
    """Order types."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order sides."""

    BUY = "buy"
    SELL = "sell"


@dataclass
class TradeSignal:
    """Trading signal with mathematical framework integration."""

    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: int = field(default_factory=lambda: int(time.time()))

    # Schwabot mathematical framework fields
    signal_strength: float = 0.0
    confidence_level: float = 0.0
    mathematical_hash: Optional[str] = None
    drift_field_value: Optional[float] = None
    entropy_level: Optional[float] = None
    quantum_state: Optional[str] = None

    # Risk management
    risk_score: float = 0.0
    max_loss: Optional[float] = None
    target_profit: Optional[float] = None

    def __post_init__(self):
        """Generate mathematical hash after initialization."""
        if not self.mathematical_hash:
            self.mathematical_hash = self._generate_signal_hash()

    def _generate_signal_hash(self) -> str:
        """Generate SHA-256 hash of trading signal."""
        signal_data = f"{self.symbol}:{self.side.value}:{self.quantity}:{self.timestamp}:{self.signal_strength}"
        return hashlib.sha256(signal_data.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "quantity": self.quantity,
            "price": self.price,
            "stop_price": self.stop_price,
            "timestamp": self.timestamp,
            "signal_strength": self.signal_strength,
            "confidence_level": self.confidence_level,
            "mathematical_hash": self.mathematical_hash,
            "drift_field_value": self.drift_field_value,
            "entropy_level": self.entropy_level,
            "quantum_state": self.quantum_state,
            "risk_score": self.risk_score,
            "max_loss": self.max_loss,
            "target_profit": self.target_profit,
        }


@dataclass
class TradeExecution:
    """Trade execution result."""

    signal: TradeSignal
    order_id: Optional[str] = None
    executed_price: Optional[float] = None
    executed_quantity: Optional[float] = None
    execution_time: Optional[int] = None
    status: str = "pending"
    error_message: Optional[str] = None
    fees: Optional[float] = None

    # Schwabot tracking
    execution_hash: Optional[str] = None
    performance_metrics: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Generate execution hash."""
        if not self.execution_hash:
            self.execution_hash = self._generate_execution_hash()

    def _generate_execution_hash(self) -> str:
        """Generate execution hash."""
        execution_data = (
            f"{self.signal.mathematical_hash}:{self.order_id}:{self.execution_time}"
        )
        return hashlib.sha256(execution_data.encode("utf-8")).hexdigest()


class SchwabotTradingEngine:
    """
    Advanced trading engine integrating with Schwabot's mathematical framework.

    Features:
    - Live trading via CCXT/Coinbase
    - Demo/simulation trading
    - Historical data integration
    - Advanced entry/exit logic
    - Risk management
    - Portfolio tracking
    - Mathematical framework integration
    """

    def __init__(self, mode: TradingMode = TradingMode.DEMO):
        """Initialize the trading engine."""
        self.mode = mode
        self.is_running = False

        # Initialize Schwabot core systems
        self._initialize_core_systems()

        # Trading state
        self.active_orders: Dict[str, TradeExecution] = {}
        self.trade_history: List[TradeExecution] = []
        self.portfolio_value = 0.0
        self.available_balance = 0.0

        # Exchange connections
        self.exchanges: Dict[str, Any] = {}
        self._initialize_exchanges()

        # Historical data
        self.historical_data: Dict[str, pd.DataFrame] = {}
        self._load_historical_data()

        # Performance tracking
        self.performance_metrics = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_profit": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
        }

        logger.info(f"🚀 Schwabot Trading Engine initialized in {mode.value} mode")

    def _initialize_core_systems(self):
        """Initialize Schwabot's core mathematical systems."""
        try:
            # Initialize mathematical framework
            self.math_system = UnifiedMathSystem()
            logger.info("✅ Unified Math System initialized")

            # Initialize T-Cell system for immune response
            self.tcell_system = EnhancedTCellSystem()
            logger.info("✅ Enhanced T-Cell System initialized")

            # Initialize strategy logic
            self.strategy_logic = StrategyLogic()
            logger.info("✅ Strategy Logic initialized")

            # Initialize risk manager
            self.risk_manager = RiskManager()
            logger.info("✅ Risk Manager initialized")

            # Initialize portfolio tracker
            self.portfolio_tracker = PortfolioTracker()
            logger.info("✅ Portfolio Tracker initialized")

        except Exception as e:
            logger.warning(f"⚠️  Some core systems unavailable: {e}")
            # Create minimal fallback systems
            self.math_system = None
            self.tcell_system = None
            self.strategy_logic = None
            self.risk_manager = None
            self.portfolio_tracker = None

    def _initialize_exchanges(self):
        """Initialize exchange connections."""
        if self.mode == TradingMode.LIVE:
            # Initialize live trading exchanges
            self._setup_ccxt_exchange()
            self._setup_coinbase_exchange()
        else:
            # Demo mode - use simulated exchanges
            self._setup_demo_exchanges()

    def _setup_ccxt_exchange(self):
        """Setup CCXT exchange connection."""
        try:
            api_key = get_secure_api_key("CCXT_API")
            if api_key:
                # Parse API key (assuming format: exchange:key:secret)
                parts = api_key.split(":")
                if len(parts) >= 3:
                    exchange_name = parts[0]
                    key = parts[1]
                    secret = parts[2]

                    exchange_class = getattr(ccxt, exchange_name)
                    self.exchanges["ccxt"] = exchange_class(
                        {
                            "apiKey": key,
                            "secret": secret,
                            "sandbox": False,
                            "enableRateLimit": True,
                        }
                    )
                    logger.info(f"✅ CCXT {exchange_name} exchange initialized")
                else:
                    logger.warning("⚠️  Invalid CCXT API key format")
        except Exception as e:
            logger.error(f"❌ Failed to setup CCXT exchange: {e}")

    def _setup_coinbase_exchange(self):
        """Setup Coinbase exchange connection."""
        try:
            api_key = get_secure_api_key("COINBASE_API")
            if api_key:
                # Parse Coinbase API key
                parts = api_key.split(":")
                if len(parts) >= 2:
                    key = parts[0]
                    secret = parts[1]

                    self.exchanges["coinbase"] = ccxt.coinbase(
                        {
                            "apiKey": key,
                            "secret": secret,
                            "sandbox": False,
                            "enableRateLimit": True,
                        }
                    )
                    logger.info("✅ Coinbase exchange initialized")
                else:
                    logger.warning("⚠️  Invalid Coinbase API key format")
        except Exception as e:
            logger.error(f"❌ Failed to setup Coinbase exchange: {e}")

    def _setup_demo_exchanges(self):
        """Setup demo exchanges for simulation."""

        class DemoExchange:
            def __init__(self, name):
                self.name = name
                self.balance = {"BTC": 1.0, "USDC": 100000.0}
                self.orders = {}

            async def create_order(self, symbol, order_type, side, amount, price=None):
                order_id = f"demo_{int(time.time())}"
                executed_price = price or await self._get_current_price(symbol)

                # Update balance
                if side == "buy":
                    cost = amount * executed_price
                    if self.balance["USDC"] >= cost:
                        self.balance["USDC"] -= cost
                        self.balance["BTC"] += amount
                    else:
                        raise Exception("Insufficient USDC balance")
                else:
                    if self.balance["BTC"] >= amount:
                        self.balance["BTC"] -= amount
                        self.balance["USDC"] += amount * executed_price
                    else:
                        raise Exception("Insufficient BTC balance")

                return {
                    "id": order_id,
                    "symbol": symbol,
                    "type": order_type,
                    "side": side,
                    "amount": amount,
                    "price": executed_price,
                    "status": "closed",
                }

            async def _get_current_price(self, symbol):
                # Get price from price bridge
                price_data = await get_secure_price("BTC")
                return price_data.price if price_data else 50000.0

        self.exchanges["demo"] = DemoExchange("demo")
        logger.info("✅ Demo exchange initialized")

    def _load_historical_data(self):
        """Load historical data from CSV files."""
        data_dir = Path("data/historical")
        data_dir.mkdir(parents=True, exist_ok=True)

        # Look for CSV files
        csv_files = list(data_dir.glob("*.csv"))

        for csv_file in csv_files:
            try:
                symbol = csv_file.stem.upper()
                df = pd.read_csv(csv_file)

                # Ensure required columns
                required_columns = [
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                ]
                if all(col in df.columns for col in required_columns):
                    self.historical_data[symbol] = df
                    logger.info(
                        f"✅ Loaded historical data for {symbol}: {len(df)} records"
                    )
                else:
                    logger.warning(f"⚠️  CSV file {csv_file} missing required columns")

            except Exception as e:
                logger.error(f"❌ Failed to load {csv_file}: {e}")

    async def analyze_market(self, symbol: str = "BTC") -> Dict[str, Any]:
        """Analyze market using Schwabot's mathematical framework."""
        try:
            # Get current price data
            price_data = await get_secure_price(symbol)
            if not price_data:
                return {"error": "Unable to get price data"}

            # Get historical data
            historical_df = self.historical_data.get(symbol)

            analysis = {
                "symbol": symbol,
                "current_price": price_data.price,
                "timestamp": price_data.timestamp,
                "price_hash": price_data.price_hash,
                "market_state_hash": price_data.market_state_hash,
                "mathematical_indicators": {},
                "trading_signals": {},
                "risk_assessment": {},
            }

            # Apply mathematical framework analysis
            if self.math_system:
                analysis["mathematical_indicators"] = (
                    await self._apply_mathematical_analysis(price_data, historical_df)
                )

            # Apply T-Cell immune system analysis
            if self.tcell_system:
                analysis["immune_response"] = await self._apply_immune_analysis(
                    price_data
                )

            # Generate trading signals
            if self.strategy_logic:
                analysis["trading_signals"] = await self._generate_trading_signals(
                    price_data, historical_df
                )

            # Risk assessment
            if self.risk_manager:
                analysis["risk_assessment"] = await self._assess_risk(
                    price_data, historical_df
                )

            return analysis

        except Exception as e:
            logger.error(f"❌ Market analysis error: {e}")
            return {"error": str(e)}

    async def _apply_mathematical_analysis(
        self, price_data: PriceData, historical_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Apply mathematical framework analysis."""
        try:
            # Calculate drift field value
            drift_field = self.math_system.calculate_drift_field(price_data.price)

            # Calculate entropy level
            entropy = self.math_system.calculate_entropy(price_data.price)

            # Calculate quantum state
            quantum_state = self.math_system.calculate_quantum_state(price_data.price)

            return {
                "drift_field_value": drift_field,
                "entropy_level": entropy,
                "quantum_state": quantum_state,
                "price_momentum": self.math_system.calculate_momentum(price_data.price),
                "volatility_index": self.math_system.calculate_volatility(
                    price_data.price
                ),
            }
        except Exception as e:
            logger.error(f"❌ Mathematical analysis error: {e}")
            return {}

    async def _apply_immune_analysis(self, price_data: PriceData) -> Dict[str, Any]:
        """Apply T-Cell immune system analysis."""
        try:
            # Analyze market health
            market_health = self.tcell_system.analyze_market_health(price_data.price)

            # Detect anomalies
            anomalies = self.tcell_system.detect_anomalies(price_data.price)

            # Generate immune response
            immune_response = self.tcell_system.generate_response(price_data.price)

            return {
                "market_health": market_health,
                "anomalies_detected": anomalies,
                "immune_response": immune_response,
                "risk_level": self.tcell_system.calculate_risk_level(price_data.price),
            }
        except Exception as e:
            logger.error(f"❌ Immune analysis error: {e}")
            return {}

    async def _generate_trading_signals(
        self, price_data: PriceData, historical_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Generate trading signals using strategy logic."""
        try:
            # Generate entry signals
            entry_signals = self.strategy_logic.generate_entry_signals(price_data.price)

            # Generate exit signals
            exit_signals = self.strategy_logic.generate_exit_signals(price_data.price)

            # Calculate signal strength
            signal_strength = self.strategy_logic.calculate_signal_strength(
                price_data.price
            )

            return {
                "entry_signals": entry_signals,
                "exit_signals": exit_signals,
                "signal_strength": signal_strength,
                "confidence_level": self.strategy_logic.calculate_confidence(
                    price_data.price
                ),
            }
        except Exception as e:
            logger.error(f"❌ Signal generation error: {e}")
            return {}

    async def _assess_risk(
        self, price_data: PriceData, historical_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Assess trading risk."""
        try:
            # Calculate position size
            position_size = self.risk_manager.calculate_position_size(price_data.price)

            # Calculate stop loss
            stop_loss = self.risk_manager.calculate_stop_loss(price_data.price)

            # Calculate risk/reward ratio
            risk_reward = self.risk_manager.calculate_risk_reward(price_data.price)

            return {
                "position_size": position_size,
                "stop_loss": stop_loss,
                "risk_reward_ratio": risk_reward,
                "max_drawdown": self.risk_manager.calculate_max_drawdown(
                    price_data.price
                ),
                "var_95": self.risk_manager.calculate_var(price_data.price, 0.95),
            }
        except Exception as e:
            logger.error(f"❌ Risk assessment error: {e}")
            return {}

    async def execute_trade(self, signal: TradeSignal) -> TradeExecution:
        """Execute a trade based on the signal."""
        try:
            # Validate signal
            if not self._validate_signal(signal):
                raise ValueError("Invalid trading signal")

            # Check risk limits
            if not self._check_risk_limits(signal):
                raise ValueError("Signal exceeds risk limits")

            # Execute order
            if self.mode == TradingMode.LIVE:
                execution = await self._execute_live_trade(signal)
            else:
                execution = await self._execute_demo_trade(signal)

            # Update portfolio
            await self._update_portfolio(execution)

            # Log execution
            self.trade_history.append(execution)
            self._update_performance_metrics(execution)

            logger.info(
                f"✅ Trade executed: {signal.side.value} {signal.quantity} {signal.symbol}"
            )
            return execution

        except Exception as e:
            logger.error(f"❌ Trade execution error: {e}")
            return TradeExecution(signal=signal, status="failed", error_message=str(e))

    def _validate_signal(self, signal: TradeSignal) -> bool:
        """Validate trading signal."""
        if not signal.symbol or not signal.quantity or signal.quantity <= 0:
            return False

        if signal.side not in [OrderSide.BUY, OrderSide.SELL]:
            return False

        if signal.order_type not in [OrderType.MARKET, OrderType.LIMIT, OrderType.STOP]:
            return False

        return True

    def _check_risk_limits(self, signal: TradeSignal) -> bool:
        """Check if signal exceeds risk limits."""
        if self.risk_manager:
            return self.risk_manager.check_risk_limits(signal)
        return True  # Allow all trades if no risk manager

    async def _execute_live_trade(self, signal: TradeSignal) -> TradeExecution:
        """Execute live trade via exchange."""
        execution = TradeExecution(signal=signal)

        try:
            # Choose exchange (prioritize CCXT, fallback to Coinbase)
            exchange = self.exchanges.get("ccxt") or self.exchanges.get("coinbase")
            if not exchange:
                raise Exception("No live exchange available")

            # Create order
            order = await exchange.create_order(
                symbol=f"{signal.symbol}/USDC",
                type=signal.order_type.value,
                side=signal.side.value,
                amount=signal.quantity,
                price=signal.price,
            )

            # Update execution
            execution.order_id = order["id"]
            execution.executed_price = order["price"]
            execution.executed_quantity = order["amount"]
            execution.execution_time = int(time.time())
            execution.status = order["status"]
            execution.fees = order.get("fee", {}).get("cost", 0.0)

        except Exception as e:
            execution.status = "failed"
            execution.error_message = str(e)

        return execution

    async def _execute_demo_trade(self, signal: TradeSignal) -> TradeExecution:
        """Execute demo trade."""
        execution = TradeExecution(signal=signal)

        try:
            # Use demo exchange
            exchange = self.exchanges.get("demo")
            if not exchange:
                raise Exception("Demo exchange not available")

            # Create order
            order = await exchange.create_order(
                symbol=f"{signal.symbol}/USDC",
                type=signal.order_type.value,
                side=signal.side.value,
                amount=signal.quantity,
                price=signal.price,
            )

            # Update execution
            execution.order_id = order["id"]
            execution.executed_price = order["price"]
            execution.executed_quantity = order["amount"]
            execution.execution_time = int(time.time())
            execution.status = order["status"]

        except Exception as e:
            execution.status = "failed"
            execution.error_message = str(e)

        return execution

    async def _update_portfolio(self, execution: TradeExecution):
        """Update portfolio after trade execution."""
        if self.portfolio_tracker:
            self.portfolio_tracker.update_portfolio(execution)

    def _update_performance_metrics(self, execution: TradeExecution):
        """Update performance metrics."""
        self.performance_metrics["total_trades"] += 1

        if execution.status == "closed":
            # Calculate profit/loss
            if execution.signal.side == OrderSide.BUY:
                # This is a buy, track for future sell
                pass
            else:
                # This is a sell, calculate P&L
                # Implementation depends on portfolio tracking
                pass

    async def get_portfolio_status(self) -> Dict[str, Any]:
        """Get current portfolio status."""
        try:
            if self.portfolio_tracker:
                return self.portfolio_tracker.get_portfolio_status()
            else:
                # Fallback portfolio status
                return {
                    "total_value": self.portfolio_value,
                    "available_balance": self.available_balance,
                    "positions": {},
                    "performance": self.performance_metrics,
                }
        except Exception as e:
            logger.error(f"❌ Portfolio status error: {e}")
            return {"error": str(e)}

    async def start_trading(self):
        """Start the trading engine."""
        self.is_running = True
        logger.info(f"🚀 Trading engine started in {self.mode.value} mode")

        while self.is_running:
            try:
                # Analyze market
                analysis = await self.analyze_market("BTC")

                # Generate and execute signals
                if analysis.get("trading_signals"):
                    signals = await self._process_trading_signals(analysis)
                    for signal in signals:
                        await self.execute_trade(signal)

                # Update portfolio
                await self.get_portfolio_status()

                # Wait before next iteration
                await asyncio.sleep(30)  # 30 second intervals

            except Exception as e:
                logger.error(f"❌ Trading loop error: {e}")
                await asyncio.sleep(60)  # Wait longer on error

    async def stop_trading(self):
        """Stop the trading engine."""
        self.is_running = False
        logger.info("🛑 Trading engine stopped")

    async def _process_trading_signals(
        self, analysis: Dict[str, Any]
    ) -> List[TradeSignal]:
        """Process trading signals from analysis."""
        signals = []

        try:
            trading_signals = analysis.get("trading_signals", {})

            # Process entry signals
            entry_signals = trading_signals.get("entry_signals", [])
            for entry in entry_signals:
                signal = TradeSignal(
                    symbol="BTC",
                    side=OrderSide.BUY if entry["type"] == "buy" else OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    quantity=entry.get("quantity", 0.001),
                    signal_strength=trading_signals.get("signal_strength", 0.0),
                    confidence_level=trading_signals.get("confidence_level", 0.0),
                )
                signals.append(signal)

            # Process exit signals
            exit_signals = trading_signals.get("exit_signals", [])
            for exit_signal in exit_signals:
                signal = TradeSignal(
                    symbol="BTC",
                    side=(
                        OrderSide.SELL
                        if exit_signal["type"] == "sell"
                        else OrderSide.BUY
                    ),
                    order_type=OrderType.MARKET,
                    quantity=exit_signal.get("quantity", 0.001),
                    signal_strength=trading_signals.get("signal_strength", 0.0),
                    confidence_level=trading_signals.get("confidence_level", 0.0),
                )
                signals.append(signal)

        except Exception as e:
            logger.error(f"❌ Signal processing error: {e}")

        return signals


# Global trading engine instance
trading_engine = SchwabotTradingEngine()


async def start_trading_engine(mode: TradingMode = TradingMode.DEMO):
    """Start the trading engine."""
    global trading_engine
    trading_engine = SchwabotTradingEngine(mode)
    await trading_engine.start_trading()


async def stop_trading_engine():
    """Stop the trading engine."""
    global trading_engine
    await trading_engine.stop_trading()


if __name__ == "__main__":
    """Test the trading engine."""

    async def test_trading_engine():
        print("🚀 Testing Schwabot Trading Engine")
        print("=" * 50)

        # Initialize engine
        engine = SchwabotTradingEngine(TradingMode.DEMO)

        # Test market analysis
        print("\n📊 Testing market analysis:")
        analysis = await engine.analyze_market("BTC")
        print(f"Analysis result: {json.dumps(analysis, indent=2)}")

        # Test portfolio status
        print("\n💰 Testing portfolio status:")
        portfolio = await engine.get_portfolio_status()
        print(f"Portfolio: {json.dumps(portfolio, indent=2)}")

        # Test trade execution
        print("\n📈 Testing trade execution:")
        signal = TradeSignal(
            symbol="BTC",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.001,
            signal_strength=0.8,
            confidence_level=0.7,
        )
        execution = await engine.execute_trade(signal)
        print(f"Trade execution: {execution.status}")

    # Run the test
    asyncio.run(test_trading_engine())
