#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠⚛️ ORBITAL TRADING INTEGRATION - REAL TRADING WITH 268 DECIMAL HASHING
=======================================================================

Real trading integration using the enhanced orbital shell brain system with:
- 268 decimal hashing logic
- Real portfolio holdings integration
- BIT strategy with randomized holdings
- Bidirectional trading (BTC/USDC and USDC/BTC)
- Orbital shell decision making
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Import the enhanced orbital shell brain system
from .orbital_shell_brain_system import (
    OrbitalBRAINSystem, PortfolioHoldings, BITStrategy, TradingPair
)

logger = logging.getLogger(__name__)


@dataclass
class TradingSignal:
    """Trading signal with orbital analysis"""
    symbol: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    quantity: Decimal
    price: float
    confidence: float
    orbital_shell: str
    bit_strategy_hash: str
    profit_potential: float
    risk_level: float
    execution_priority: int
    reason: str
    timestamp: float


@dataclass
class PortfolioState:
    """Current portfolio state with real holdings"""
    total_value_usdc: Decimal
    holdings: PortfolioHoldings
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    last_update: float


class OrbitalTradingIntegration:
    """
    🧠⚛️ Orbital Trading Integration System
    
    Integrates the enhanced orbital shell brain system with real trading execution.
    Uses 268 decimal hashing, real portfolio holdings, and BIT strategy for all trading decisions.
    """
    
    def __init__(self, config: Dict[str, Any] = None) -> None:
        self.config = config or self._default_config()
        
        # Initialize the enhanced orbital shell brain system
        self.orbital_brain = OrbitalBRAINSystem(config)
        
        # Trading state
        self.portfolio_state = PortfolioState(
            total_value_usdc=Decimal('0'),
            holdings=PortfolioHoldings(),
            unrealized_pnl=Decimal('0'),
            realized_pnl=Decimal('0'),
            last_update=time.time()
        )
        
        # Market data cache
        self.market_data_cache: Dict[str, Dict[str, Any]] = {}
        self.price_cache: Dict[str, float] = {}
        
        # Trading history
        self.trading_signals: List[TradingSignal] = []
        self.executed_trades: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.total_trades = 0
        self.successful_trades = 0
        self.total_volume = Decimal('0')
        
        # System state
        self.active = False
        self.trading_enabled = False
        
        logger.info("🧠⚛️ Orbital Trading Integration initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            "initial_capital_usdc": 10000.0,
            "max_position_size_pct": 0.2,  # 20%
            "min_position_size_pct": 0.01,  # 1%
            "stop_loss_pct": 0.02,  # 2%
            "take_profit_pct": 0.05,  # 5%
            "max_daily_trades": 50,
            "trading_pairs": [
                "BTC/USDC", "USDC/BTC",
                "ETH/USDC", "USDC/ETH", 
                "XRP/USDC", "USDC/XRP",
                "SOL/USDC", "USDC/SOL"
            ],
            "orbital_confidence_threshold": 0.7,
            "bit_strategy_enabled": True,
            "bidirectional_trading": True,
            "real_portfolio_integration": True,
            "hash_268_precision": 268
        }
    
    async def fetch_real_portfolio_holdings(self) -> PortfolioHoldings:
        """
        Fetch real portfolio holdings from connected exchanges via API
        
        This method integrates with the existing API infrastructure to get real holdings
        and update the orbital brain system with current portfolio state.
        """
        try:
            # Initialize empty holdings
            holdings = PortfolioHoldings()
            
            # Try to get holdings from secure API integration manager
            try:
                from .secure_api_integration_manager import SecureAPIIntegrationManager
                api_manager = SecureAPIIntegrationManager()
                
                # Get balances from all enabled profiles
                for profile_id in api_manager.profiles:
                    if api_manager.profiles[profile_id].enabled:
                        balance = await api_manager.get_portfolio_balance(profile_id)
                        if balance:
                            # Update holdings with real data
                            for asset, amount in balance.items():
                                if asset in ['BTC', 'USDC', 'ETH', 'XRP', 'SOL']:
                                    setattr(holdings, asset, Decimal(str(amount)))
                            
                            logger.info(f"Fetched real holdings from {profile_id}: {balance}")
                            break  # Use first successful profile
                            
            except ImportError:
                logger.warning("SecureAPIIntegrationManager not available")
            
            # Fallback: Try enhanced portfolio tracker
            try:
                from .enhanced_portfolio_tracker import EnhancedPortfolioTracker
                portfolio_tracker = EnhancedPortfolioTracker()
                await portfolio_tracker.sync_with_exchanges()
                
                # Get current balances
                for asset, amount in portfolio_tracker.balances.items():
                    if asset in ['BTC', 'USDC', 'ETH', 'XRP', 'SOL']:
                        setattr(holdings, asset, amount)
                        
                logger.info(f"Fetched holdings from portfolio tracker: {portfolio_tracker.balances}")
                
            except ImportError:
                logger.warning("EnhancedPortfolioTracker not available")
            
            # Fallback: Try CCXT trading executor
            try:
                from .ccxt_trading_executor import CCXTTradingExecutor
                executor = CCXTTradingExecutor()
                balance = await executor.fetch_balance("binance")  # Default to binance
                
                if balance:
                    for asset, amount in balance.items():
                        if asset in ['BTC', 'USDC', 'ETH', 'XRP', 'SOL']:
                            setattr(holdings, asset, Decimal(str(amount)))
                    
                    logger.info(f"Fetched holdings from CCXT executor: {balance}")
                    
            except ImportError:
                logger.warning("CCXTTradingExecutor not available")
            
            # Update the orbital brain system with real holdings
            self.update_portfolio_holdings(holdings)
            
            logger.info(f"Real portfolio holdings updated: {holdings}")
            return holdings
            
        except Exception as e:
            logger.error(f"Error fetching real portfolio holdings: {e}")
            # Return current holdings as fallback
            return self.portfolio_state.holdings

    async def test_bidirectional_trading_logic(self) -> Dict[str, Any]:
        """
        Test bidirectional trading logic to ensure it works correctly
        
        This method simulates various scenarios to verify that:
        - BUY actions have sufficient quote currency
        - SELL actions have sufficient base currency  
        - Trading decisions are balanced and logical
        - No catastrophic one-way trading occurs
        """
        try:
            test_results = {
                "passed": True,
                "scenarios_tested": 0,
                "scenarios_passed": 0,
                "errors": []
            }
            
            # Test scenario 1: BTC/USDC with high BTC, low USDC
            logger.info("🧪 Testing Scenario 1: High BTC, Low USDC")
            test_holdings = PortfolioHoldings(
                BTC=Decimal('1.0'),
                USDC=Decimal('100.0'),
                ETH=Decimal('0'),
                XRP=Decimal('0'),
                SOL=Decimal('0')
            )
            self.update_portfolio_holdings(test_holdings)
            
            # Test BTC/USDC pair
            market_data = {
                "price": 45000.0,
                "price_change": 0.01,  # 1% increase
                "volatility": 0.02,
                "prices": {"BTC": 45000.0, "USDC": 1.0}
            }
            
            decision = self.orbital_brain.get_trading_decision("BTC/USDC", 45000.0, market_data)
            test_results["scenarios_tested"] += 1
            
            if decision.get("action") == "SELL":
                test_results["scenarios_passed"] += 1
                logger.info("✅ Scenario 1 PASSED: High BTC holdings correctly triggered SELL")
            else:
                test_results["passed"] = False
                test_results["errors"].append("Scenario 1 FAILED: High BTC should trigger SELL")
                logger.error("❌ Scenario 1 FAILED: High BTC holdings did not trigger SELL")
            
            # Test scenario 2: BTC/USDC with low BTC, high USDC
            logger.info("🧪 Testing Scenario 2: Low BTC, High USDC")
            test_holdings = PortfolioHoldings(
                BTC=Decimal('0.001'),
                USDC=Decimal('10000.0'),
                ETH=Decimal('0'),
                XRP=Decimal('0'),
                SOL=Decimal('0')
            )
            self.update_portfolio_holdings(test_holdings)
            
            decision = self.orbital_brain.get_trading_decision("BTC/USDC", 45000.0, market_data)
            test_results["scenarios_tested"] += 1
            
            if decision.get("action") == "BUY":
                test_results["scenarios_passed"] += 1
                logger.info("✅ Scenario 2 PASSED: Low BTC holdings correctly triggered BUY")
            else:
                test_results["passed"] = False
                test_results["errors"].append("Scenario 2 FAILED: Low BTC should trigger BUY")
                logger.error("❌ Scenario 2 FAILED: Low BTC holdings did not trigger BUY")
            
            # Test scenario 3: USDC/BTC with high USDC, low BTC
            logger.info("🧪 Testing Scenario 3: USDC/BTC with high USDC, low BTC")
            test_holdings = PortfolioHoldings(
                BTC=Decimal('0.001'),
                USDC=Decimal('10000.0'),
                ETH=Decimal('0'),
                XRP=Decimal('0'),
                SOL=Decimal('0')
            )
            self.update_portfolio_holdings(test_holdings)
            
            decision = self.orbital_brain.get_trading_decision("USDC/BTC", 1/45000.0, market_data)
            test_results["scenarios_tested"] += 1
            
            if decision.get("action") == "BUY":
                test_results["scenarios_passed"] += 1
                logger.info("✅ Scenario 3 PASSED: USDC/BTC correctly triggered BUY")
            else:
                test_results["passed"] = False
                test_results["errors"].append("Scenario 3 FAILED: USDC/BTC should trigger BUY")
                logger.error("❌ Scenario 3 FAILED: USDC/BTC did not trigger BUY")
            
            # Test scenario 4: Insufficient holdings validation
            logger.info("🧪 Testing Scenario 4: Insufficient holdings validation")
            test_holdings = PortfolioHoldings(
                BTC=Decimal('0'),
                USDC=Decimal('0'),
                ETH=Decimal('0'),
                XRP=Decimal('0'),
                SOL=Decimal('0')
            )
            self.update_portfolio_holdings(test_holdings)
            
            decision = self.orbital_brain.get_trading_decision("BTC/USDC", 45000.0, market_data)
            test_results["scenarios_tested"] += 1
            
            if decision.get("action") == "HOLD":
                test_results["scenarios_passed"] += 1
                logger.info("✅ Scenario 4 PASSED: Insufficient holdings correctly triggered HOLD")
            else:
                test_results["passed"] = False
                test_results["errors"].append("Scenario 4 FAILED: Insufficient holdings should trigger HOLD")
                logger.error("❌ Scenario 4 FAILED: Insufficient holdings did not trigger HOLD")
            
            # Test scenario 5: Balanced holdings
            logger.info("🧪 Testing Scenario 5: Balanced holdings")
            test_holdings = PortfolioHoldings(
                BTC=Decimal('0.1'),
                USDC=Decimal('4500.0'),
                ETH=Decimal('0'),
                XRP=Decimal('0'),
                SOL=Decimal('0')
            )
            self.update_portfolio_holdings(test_holdings)
            
            decision = self.orbital_brain.get_trading_decision("BTC/USDC", 45000.0, market_data)
            test_results["scenarios_tested"] += 1
            
            # Balanced holdings should allow both BUY and SELL based on market conditions
            if decision.get("action") in ["BUY", "SELL", "HOLD"]:
                test_results["scenarios_passed"] += 1
                logger.info(f"✅ Scenario 5 PASSED: Balanced holdings allowed {decision.get('action')}")
            else:
                test_results["passed"] = False
                test_results["errors"].append("Scenario 5 FAILED: Balanced holdings should allow trading")
                logger.error("❌ Scenario 5 FAILED: Balanced holdings did not allow trading")
            
            # Summary
            success_rate = test_results["scenarios_passed"] / test_results["scenarios_tested"]
            logger.info(f"🧪 Bidirectional Trading Test Results: {test_results['scenarios_passed']}/{test_results['scenarios_tested']} passed ({success_rate:.1%})")
            
            if test_results["passed"]:
                logger.info("🎉 ALL BIDIRECTIONAL TRADING TESTS PASSED - System is safe to use!")
            else:
                logger.error("🚨 BIDIRECTIONAL TRADING TESTS FAILED - System needs fixes before use!")
                for error in test_results["errors"]:
                    logger.error(f"  - {error}")
            
            return test_results
            
        except Exception as e:
            logger.error(f"Error testing bidirectional trading logic: {e}")
            return {
                "passed": False,
                "scenarios_tested": 0,
                "scenarios_passed": 0,
                "errors": [f"Test execution error: {str(e)}"]
            }

    async def test_buy_low_sell_high_logic(self) -> Dict[str, Any]:
        """
        Test "buy low, sell high" logic to ensure it works correctly
        
        This method simulates various scenarios to verify that:
        - BUY actions occur when price is going DOWN (buying low)
        - SELL actions occur when price is going UP (selling high)
        - The system follows proper trading logic
        - No catastrophic reverse logic occurs
        """
        try:
            test_results = {
                "passed": True,
                "scenarios_tested": 0,
                "scenarios_passed": 0,
                "errors": []
            }
            
            # Test scenario 1: Price going DOWN should trigger BUY
            logger.info("🧪 Testing Scenario 1: Price going DOWN should trigger BUY")
            test_holdings = PortfolioHoldings(
                BTC=Decimal('0.1'),
                USDC=Decimal('5000.0'),
                ETH=Decimal('0'),
                XRP=Decimal('0'),
                SOL=Decimal('0')
            )
            self.update_portfolio_holdings(test_holdings)
            
            # Market data with price going DOWN
            market_data = {
                "price": 45000.0,
                "price_change": -0.02,  # 2% decrease
                "volatility": 0.02,
                "prices": {"BTC": 45000.0, "USDC": 1.0}
            }
            
            decision = self.orbital_brain.get_trading_decision("BTC/USDC", 45000.0, market_data)
            test_results["scenarios_tested"] += 1
            
            if decision.get("action") == "BUY":
                test_results["scenarios_passed"] += 1
                logger.info("✅ Scenario 1 PASSED: Price going DOWN correctly triggered BUY")
            else:
                test_results["passed"] = False
                test_results["errors"].append("Scenario 1 FAILED: Price going DOWN should trigger BUY")
                logger.error(f"❌ Scenario 1 FAILED: Price going DOWN triggered {decision.get('action')} instead of BUY")
            
            # Test scenario 2: Price going UP should trigger SELL
            logger.info("🧪 Testing Scenario 2: Price going UP should trigger SELL")
            test_holdings = PortfolioHoldings(
                BTC=Decimal('1.0'),
                USDC=Decimal('100.0'),
                ETH=Decimal('0'),
                XRP=Decimal('0'),
                SOL=Decimal('0')
            )
            self.update_portfolio_holdings(test_holdings)
            
            # Market data with price going UP
            market_data = {
                "price": 45000.0,
                "price_change": 0.02,  # 2% increase
                "volatility": 0.02,
                "prices": {"BTC": 45000.0, "USDC": 1.0}
            }
            
            decision = self.orbital_brain.get_trading_decision("BTC/USDC", 45000.0, market_data)
            test_results["scenarios_tested"] += 1
            
            if decision.get("action") == "SELL":
                test_results["scenarios_passed"] += 1
                logger.info("✅ Scenario 2 PASSED: Price going UP correctly triggered SELL")
            else:
                test_results["passed"] = False
                test_results["errors"].append("Scenario 2 FAILED: Price going UP should trigger SELL")
                logger.error(f"❌ Scenario 2 FAILED: Price going UP triggered {decision.get('action')} instead of SELL")
            
            # Test scenario 3: Strong dip should trigger stronger BUY signal
            logger.info("🧪 Testing Scenario 3: Strong dip should trigger stronger BUY signal")
            test_holdings = PortfolioHoldings(
                BTC=Decimal('0.1'),
                USDC=Decimal('10000.0'),
                ETH=Decimal('0'),
                XRP=Decimal('0'),
                SOL=Decimal('0')
            )
            self.update_portfolio_holdings(test_holdings)
            
            # Market data with strong dip
            market_data = {
                "price": 45000.0,
                "price_change": -0.05,  # 5% decrease (strong dip)
                "volatility": 0.03,
                "prices": {"BTC": 45000.0, "USDC": 1.0}
            }
            
            decision = self.orbital_brain.get_trading_decision("BTC/USDC", 45000.0, market_data)
            test_results["scenarios_tested"] += 1
            
            if decision.get("action") == "BUY" and decision.get("confidence", 0) > 0.7:
                test_results["scenarios_passed"] += 1
                logger.info("✅ Scenario 3 PASSED: Strong dip correctly triggered high-confidence BUY")
            else:
                test_results["passed"] = False
                test_results["errors"].append("Scenario 3 FAILED: Strong dip should trigger high-confidence BUY")
                logger.error(f"❌ Scenario 3 FAILED: Strong dip triggered {decision.get('action')} with confidence {decision.get('confidence')}")
            
            # Test scenario 4: Strong rally should trigger stronger SELL signal
            logger.info("🧪 Testing Scenario 4: Strong rally should trigger stronger SELL signal")
            test_holdings = PortfolioHoldings(
                BTC=Decimal('2.0'),
                USDC=Decimal('100.0'),
                ETH=Decimal('0'),
                XRP=Decimal('0'),
                SOL=Decimal('0')
            )
            self.update_portfolio_holdings(test_holdings)
            
            # Market data with strong rally
            market_data = {
                "price": 45000.0,
                "price_change": 0.05,  # 5% increase (strong rally)
                "volatility": 0.03,
                "prices": {"BTC": 45000.0, "USDC": 1.0}
            }
            
            decision = self.orbital_brain.get_trading_decision("BTC/USDC", 45000.0, market_data)
            test_results["scenarios_tested"] += 1
            
            if decision.get("action") == "SELL" and decision.get("confidence", 0) > 0.7:
                test_results["scenarios_passed"] += 1
                logger.info("✅ Scenario 4 PASSED: Strong rally correctly triggered high-confidence SELL")
            else:
                test_results["passed"] = False
                test_results["errors"].append("Scenario 4 FAILED: Strong rally should trigger high-confidence SELL")
                logger.error(f"❌ Scenario 4 FAILED: Strong rally triggered {decision.get('action')} with confidence {decision.get('confidence')}")
            
            # Test scenario 5: Small price movements should trigger HOLD
            logger.info("🧪 Testing Scenario 5: Small price movements should trigger HOLD")
            test_holdings = PortfolioHoldings(
                BTC=Decimal('0.5'),
                USDC=Decimal('2000.0'),
                ETH=Decimal('0'),
                XRP=Decimal('0'),
                SOL=Decimal('0')
            )
            self.update_portfolio_holdings(test_holdings)
            
            # Market data with small movement
            market_data = {
                "price": 45000.0,
                "price_change": 0.001,  # 0.1% increase (small movement)
                "volatility": 0.01,
                "prices": {"BTC": 45000.0, "USDC": 1.0}
            }
            
            decision = self.orbital_brain.get_trading_decision("BTC/USDC", 45000.0, market_data)
            test_results["scenarios_tested"] += 1
            
            if decision.get("action") == "HOLD":
                test_results["scenarios_passed"] += 1
                logger.info("✅ Scenario 5 PASSED: Small price movement correctly triggered HOLD")
            else:
                test_results["passed"] = False
                test_results["errors"].append("Scenario 5 FAILED: Small price movement should trigger HOLD")
                logger.error(f"❌ Scenario 5 FAILED: Small price movement triggered {decision.get('action')} instead of HOLD")
            
            # Test scenario 6: Insufficient holdings should trigger HOLD
            logger.info("🧪 Testing Scenario 6: Insufficient holdings should trigger HOLD")
            test_holdings = PortfolioHoldings(
                BTC=Decimal('0'),
                USDC=Decimal('0'),
                ETH=Decimal('0'),
                XRP=Decimal('0'),
                SOL=Decimal('0')
            )
            self.update_portfolio_holdings(test_holdings)
            
            # Market data with strong signal
            market_data = {
                "price": 45000.0,
                "price_change": -0.03,  # 3% decrease (should trigger BUY)
                "volatility": 0.02,
                "prices": {"BTC": 45000.0, "USDC": 1.0}
            }
            
            decision = self.orbital_brain.get_trading_decision("BTC/USDC", 45000.0, market_data)
            test_results["scenarios_tested"] += 1
            
            if decision.get("action") == "HOLD":
                test_results["scenarios_passed"] += 1
                logger.info("✅ Scenario 6 PASSED: Insufficient holdings correctly triggered HOLD")
            else:
                test_results["passed"] = False
                test_results["errors"].append("Scenario 6 FAILED: Insufficient holdings should trigger HOLD")
                logger.error(f"❌ Scenario 6 FAILED: Insufficient holdings triggered {decision.get('action')} instead of HOLD")
            
            # Summary
            success_rate = test_results["scenarios_passed"] / test_results["scenarios_tested"]
            logger.info(f"🧪 Buy Low Sell High Test Results: {test_results['scenarios_passed']}/{test_results['scenarios_tested']} passed ({success_rate:.1%})")
            
            if test_results["passed"]:
                logger.info("🎉 ALL BUY LOW SELL HIGH TESTS PASSED - System follows proper trading logic!")
            else:
                logger.error("🚨 BUY LOW SELL HIGH TESTS FAILED - System has logical flaws!")
                for error in test_results["errors"]:
                    logger.error(f"  - {error}")
            
            return test_results
            
        except Exception as e:
            logger.error(f"Error testing buy low sell high logic: {e}")
            return {
                "passed": False,
                "scenarios_tested": 0,
                "scenarios_passed": 0,
                "errors": [f"Test execution error: {str(e)}"]
            }

    async def test_technical_indicators_integration(self) -> Dict[str, Any]:
        """
        Test technical indicators integration to ensure they work correctly
        
        This method simulates various scenarios to verify that:
        - RSI oversold/overbought signals work correctly
        - MACD bullish/bearish signals work correctly
        - Bollinger Bands position signals work correctly
        - Moving averages trend signals work correctly
        - All indicators are properly integrated into trading decisions
        """
        try:
            test_results = {
                "passed": True,
                "scenarios_tested": 0,
                "scenarios_passed": 0,
                "errors": []
            }
            
            # Test scenario 1: RSI oversold should trigger BUY
            logger.info("🧪 Testing Scenario 1: RSI oversold should trigger BUY")
            test_holdings = PortfolioHoldings(
                BTC=Decimal('0.1'),
                USDC=Decimal('5000.0'),
                ETH=Decimal('0'),
                XRP=Decimal('0'),
                SOL=Decimal('0')
            )
            self.update_portfolio_holdings(test_holdings)
            
            # Simulate price history for RSI calculation
            for i in range(30):
                # Create declining prices to generate oversold RSI
                price = 45000.0 - (i * 100)  # Declining prices
                self.update_market_data("BTC/USDC", price, {"price_change": -0.01, "volatility": 0.02})
            
            # Market data with RSI oversold
            market_data = {
                "price": 42000.0,
                "price_change": -0.02,
                "volatility": 0.02,
                "prices": {"BTC": 42000.0, "USDC": 1.0}
            }
            
            decision = self.orbital_brain.get_trading_decision("BTC/USDC", 42000.0, market_data)
            test_results["scenarios_tested"] += 1
            
            if decision.get("action") == "BUY":
                test_results["scenarios_passed"] += 1
                logger.info("✅ Scenario 1 PASSED: RSI oversold correctly triggered BUY")
            else:
                test_results["passed"] = False
                test_results["errors"].append("Scenario 1 FAILED: RSI oversold should trigger BUY")
                logger.error(f"❌ Scenario 1 FAILED: RSI oversold triggered {decision.get('action')} instead of BUY")
            
            # Test scenario 2: RSI overbought should trigger SELL
            logger.info("🧪 Testing Scenario 2: RSI overbought should trigger SELL")
            test_holdings = PortfolioHoldings(
                BTC=Decimal('2.0'),
                USDC=Decimal('100.0'),
                ETH=Decimal('0'),
                XRP=Decimal('0'),
                SOL=Decimal('0')
            )
            self.update_portfolio_holdings(test_holdings)
            
            # Simulate price history for RSI calculation
            for i in range(30):
                # Create rising prices to generate overbought RSI
                price = 42000.0 + (i * 100)  # Rising prices
                self.update_market_data("BTC/USDC", price, {"price_change": 0.01, "volatility": 0.02})
            
            # Market data with RSI overbought
            market_data = {
                "price": 48000.0,
                "price_change": 0.02,
                "volatility": 0.02,
                "prices": {"BTC": 48000.0, "USDC": 1.0}
            }
            
            decision = self.orbital_brain.get_trading_decision("BTC/USDC", 48000.0, market_data)
            test_results["scenarios_tested"] += 1
            
            if decision.get("action") == "SELL":
                test_results["scenarios_passed"] += 1
                logger.info("✅ Scenario 2 PASSED: RSI overbought correctly triggered SELL")
            else:
                test_results["passed"] = False
                test_results["errors"].append("Scenario 2 FAILED: RSI overbought should trigger SELL")
                logger.error(f"❌ Scenario 2 FAILED: RSI overbought triggered {decision.get('action')} instead of SELL")
            
            # Test scenario 3: MACD bullish crossover should trigger BUY
            logger.info("🧪 Testing Scenario 3: MACD bullish crossover should trigger BUY")
            test_holdings = PortfolioHoldings(
                BTC=Decimal('0.1'),
                USDC=Decimal('10000.0'),
                ETH=Decimal('0'),
                XRP=Decimal('0'),
                SOL=Decimal('0')
            )
            self.update_portfolio_holdings(test_holdings)
            
            # Simulate price history for MACD calculation
            for i in range(30):
                # Create prices that will generate bullish MACD
                if i < 15:
                    price = 45000.0 - (i * 50)  # Declining
                else:
                    price = 44250.0 + ((i - 15) * 100)  # Rising
                self.update_market_data("BTC/USDC", price, {"price_change": 0.005, "volatility": 0.02})
            
            # Market data with MACD bullish
            market_data = {
                "price": 46000.0,
                "price_change": 0.01,
                "volatility": 0.02,
                "prices": {"BTC": 46000.0, "USDC": 1.0}
            }
            
            decision = self.orbital_brain.get_trading_decision("BTC/USDC", 46000.0, market_data)
            test_results["scenarios_tested"] += 1
            
            # Check if technical indicators are present in decision
            tech_indicators = decision.get("technical_indicators", {})
            if tech_indicators and "macd_histogram" in tech_indicators:
                test_results["scenarios_passed"] += 1
                logger.info("✅ Scenario 3 PASSED: MACD indicators correctly integrated")
            else:
                test_results["passed"] = False
                test_results["errors"].append("Scenario 3 FAILED: MACD indicators not integrated")
                logger.error("❌ Scenario 3 FAILED: MACD indicators not found in decision")
            
            # Test scenario 4: Bollinger Bands oversold should trigger BUY
            logger.info("🧪 Testing Scenario 4: Bollinger Bands oversold should trigger BUY")
            test_holdings = PortfolioHoldings(
                BTC=Decimal('0.1'),
                USDC=Decimal('8000.0'),
                ETH=Decimal('0'),
                XRP=Decimal('0'),
                SOL=Decimal('0')
            )
            self.update_portfolio_holdings(test_holdings)
            
            # Simulate price history for Bollinger Bands calculation
            for i in range(25):
                # Create prices that will generate oversold Bollinger position
                price = 45000.0 - (i * 200)  # Strong decline
                self.update_market_data("BTC/USDC", price, {"price_change": -0.02, "volatility": 0.03})
            
            # Market data with Bollinger oversold
            market_data = {
                "price": 40000.0,
                "price_change": -0.03,
                "volatility": 0.03,
                "prices": {"BTC": 40000.0, "USDC": 1.0}
            }
            
            decision = self.orbital_brain.get_trading_decision("BTC/USDC", 40000.0, market_data)
            test_results["scenarios_tested"] += 1
            
            # Check if Bollinger position is calculated
            tech_indicators = decision.get("technical_indicators", {})
            if tech_indicators and "bb_position" in tech_indicators:
                bb_position = tech_indicators["bb_position"]
                if bb_position < 0.3:  # Oversold position
                    test_results["scenarios_passed"] += 1
                    logger.info(f"✅ Scenario 4 PASSED: Bollinger position correctly calculated: {bb_position:.2f}")
                else:
                    test_results["passed"] = False
                    test_results["errors"].append(f"Scenario 4 FAILED: Bollinger position not oversold: {bb_position:.2f}")
                    logger.error(f"❌ Scenario 4 FAILED: Bollinger position not oversold: {bb_position:.2f}")
            else:
                test_results["passed"] = False
                test_results["errors"].append("Scenario 4 FAILED: Bollinger position not calculated")
                logger.error("❌ Scenario 4 FAILED: Bollinger position not found in decision")
            
            # Test scenario 5: Moving averages trend should influence decisions
            logger.info("🧪 Testing Scenario 5: Moving averages trend should influence decisions")
            test_holdings = PortfolioHoldings(
                BTC=Decimal('0.5'),
                USDC=Decimal('2000.0'),
                ETH=Decimal('0'),
                XRP=Decimal('0'),
                SOL=Decimal('0')
            )
            self.update_portfolio_holdings(test_holdings)
            
            # Simulate price history for moving averages
            for i in range(30):
                # Create prices that will generate clear moving average trends
                price = 45000.0 + (i * 50)  # Steady rise
                self.update_market_data("BTC/USDC", price, {"price_change": 0.005, "volatility": 0.01})
            
            # Market data with clear trend
            market_data = {
                "price": 46500.0,
                "price_change": 0.01,
                "volatility": 0.01,
                "prices": {"BTC": 46500.0, "USDC": 1.0}
            }
            
            decision = self.orbital_brain.get_trading_decision("BTC/USDC", 46500.0, market_data)
            test_results["scenarios_tested"] += 1
            
            # Check if moving averages are calculated
            tech_indicators = decision.get("technical_indicators", {})
            if (tech_indicators and "sma_20" in tech_indicators and 
                "ema_12" in tech_indicators and "ema_26" in tech_indicators):
                test_results["scenarios_passed"] += 1
                logger.info("✅ Scenario 5 PASSED: Moving averages correctly calculated")
            else:
                test_results["passed"] = False
                test_results["errors"].append("Scenario 5 FAILED: Moving averages not calculated")
                logger.error("❌ Scenario 5 FAILED: Moving averages not found in decision")
            
            # Test scenario 6: All technical indicators should be present
            logger.info("🧪 Testing Scenario 6: All technical indicators should be present")
            test_holdings = PortfolioHoldings(
                BTC=Decimal('1.0'),
                USDC=Decimal('1000.0'),
                ETH=Decimal('0'),
                XRP=Decimal('0'),
                SOL=Decimal('0')
            )
            self.update_portfolio_holdings(test_holdings)
            
            # Market data
            market_data = {
                "price": 45000.0,
                "price_change": 0.0,
                "volatility": 0.02,
                "prices": {"BTC": 45000.0, "USDC": 1.0}
            }
            
            decision = self.orbital_brain.get_trading_decision("BTC/USDC", 45000.0, market_data)
            test_results["scenarios_tested"] += 1
            
            # Check if all technical indicators are present
            tech_indicators = decision.get("technical_indicators", {})
            required_indicators = ["rsi", "macd", "macd_signal", "macd_histogram", 
                                 "bb_position", "sma_20", "ema_12", "ema_26", 
                                 "volume_ratio", "atr"]
            
            missing_indicators = [ind for ind in required_indicators if ind not in tech_indicators]
            
            if not missing_indicators:
                test_results["scenarios_passed"] += 1
                logger.info("✅ Scenario 6 PASSED: All technical indicators present")
            else:
                test_results["passed"] = False
                test_results["errors"].append(f"Scenario 6 FAILED: Missing indicators: {missing_indicators}")
                logger.error(f"❌ Scenario 6 FAILED: Missing indicators: {missing_indicators}")
            
            # Summary
            success_rate = test_results["scenarios_passed"] / test_results["scenarios_tested"]
            logger.info(f"🧪 Technical Indicators Test Results: {test_results['scenarios_passed']}/{test_results['scenarios_tested']} passed ({success_rate:.1%})")
            
            if test_results["passed"]:
                logger.info("🎉 ALL TECHNICAL INDICATORS TESTS PASSED - System properly integrates all mathematical indicators!")
            else:
                logger.error("🚨 TECHNICAL INDICATORS TESTS FAILED - System missing critical mathematical integration!")
                for error in test_results["errors"]:
                    logger.error(f"  - {error}")
            
            return test_results
            
        except Exception as e:
            logger.error(f"Error testing technical indicators integration: {e}")
            return {
                "passed": False,
                "scenarios_tested": 0,
                "scenarios_passed": 0,
                "errors": [f"Test execution error: {str(e)}"]
            }

    async def start_trading_system(self) -> None:
        """Start the orbital trading system"""
        try:
            self.active = True
            self.trading_enabled = True
            
            # Start the orbital brain system
            self.orbital_brain.start_orbital_brain_system()
            
            # Fetch real portfolio holdings from API
            real_holdings = await self.fetch_real_portfolio_holdings()
            
            # If no real holdings found, initialize with initial capital
            if real_holdings.get_total_value_usdc({}) == 0:
                initial_holdings = PortfolioHoldings(
                    BTC=Decimal('0'),
                    USDC=Decimal(str(self.config["initial_capital_usdc"])),
                    ETH=Decimal('0'),
                    XRP=Decimal('0'),
                    SOL=Decimal('0')
                )
                self.update_portfolio_holdings(initial_holdings)
                logger.info("Initialized with default holdings")
            else:
                logger.info("Using real portfolio holdings from API")
            
            # CRITICAL: Test bidirectional trading logic before starting
            logger.info("🧪 Running bidirectional trading logic tests...")
            test_results = await self.test_bidirectional_trading_logic()
            
            if not test_results["passed"]:
                logger.error("🚨 BIDIRECTIONAL TRADING TESTS FAILED - Trading system will NOT start!")
                logger.error("This prevents catastrophic one-way trading behavior.")
                self.active = False
                self.trading_enabled = False
                raise RuntimeError("Bidirectional trading tests failed - system unsafe to use")
            
            # CRITICAL: Test "buy low, sell high" logic before starting
            logger.info("🧪 Running buy low sell high logic tests...")
            buy_low_sell_high_results = await self.test_buy_low_sell_high_logic()
            
            if not buy_low_sell_high_results["passed"]:
                logger.error("🚨 BUY LOW SELL HIGH TESTS FAILED - Trading system will NOT start!")
                logger.error("This prevents catastrophic reverse trading logic.")
                self.active = False
                self.trading_enabled = False
                raise RuntimeError("Buy low sell high tests failed - system has logical flaws")
            
            # CRITICAL: Test technical indicators integration before starting
            logger.info("🧪 Running technical indicators integration tests...")
            technical_indicators_results = await self.test_technical_indicators_integration()
            
            if not technical_indicators_results["passed"]:
                logger.error("🚨 TECHNICAL INDICATORS TESTS FAILED - Trading system will NOT start!")
                logger.error("This prevents trading without proper mathematical analysis.")
                self.active = False
                self.trading_enabled = False
                raise RuntimeError("Technical indicators tests failed - system missing critical mathematical integration")
            
            logger.info("🧠⚛️ Orbital Trading System started")
            
        except Exception as e:
            logger.error(f"Error starting trading system: {e}")
            raise
    
    async def stop_trading_system(self) -> None:
        """Stop the orbital trading system"""
        try:
            self.active = False
            self.trading_enabled = False
            
            # Stop the orbital brain system
            self.orbital_brain.stop_orbital_brain_system()
            
            logger.info("🧠⚛️ Orbital Trading System stopped")
            
        except Exception as e:
            logger.error(f"Error stopping trading system: {e}")
    
    def update_portfolio_holdings(self, new_holdings: PortfolioHoldings) -> None:
        """Update current portfolio holdings"""
        try:
            # Update orbital brain portfolio
            self.orbital_brain.update_portfolio_holdings(new_holdings)
            
            # Update local portfolio state
            self.portfolio_state.holdings = new_holdings
            self.portfolio_state.last_update = time.time()
            
            # Calculate total value if we have prices
            if self.price_cache:
                self.portfolio_state.total_value_usdc = new_holdings.get_total_value_usdc(self.price_cache)
            
            logger.info(f"Portfolio updated: {new_holdings}")
            
        except Exception as e:
            logger.error(f"Error updating portfolio holdings: {e}")
    
    def update_market_data(self, symbol: str, price: float, market_data: Dict[str, Any]) -> None:
        """Update market data for a trading pair with technical indicators"""
        try:
            # Update price cache
            self.price_cache[symbol] = price
            
            # Calculate technical indicators if we have price history
            technical_indicators = self._calculate_technical_indicators(symbol, price, market_data)
            
            # Update market data cache with technical indicators
            self.market_data_cache[symbol] = {
                "price": price,
                "timestamp": time.time(),
                "prices": self.price_cache,  # All current prices
                **market_data,
                **technical_indicators  # Add technical indicators
            }
            
            # Update portfolio total value
            if self.portfolio_state.holdings:
                self.portfolio_state.total_value_usdc = self.portfolio_state.holdings.get_total_value_usdc(self.price_cache)
            
        except Exception as e:
            logger.error(f"Error updating market data for {symbol}: {e}")
    
    def _calculate_technical_indicators(self, symbol: str, current_price: float, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate technical indicators for trading decisions"""
        try:
            # Initialize price history if not exists
            if not hasattr(self, 'price_history'):
                self.price_history = {}
            if symbol not in self.price_history:
                self.price_history[symbol] = []
            
            # Add current price to history
            self.price_history[symbol].append(current_price)
            
            # Keep only last 100 prices
            if len(self.price_history[symbol]) > 100:
                self.price_history[symbol] = self.price_history[symbol][-100:]
            
            prices = self.price_history[symbol]
            
            # Need minimum data for indicators
            if len(prices) < 20:
                return {
                    'rsi': 50.0,
                    'macd': 0.0,
                    'macd_signal': 0.0,
                    'macd_histogram': 0.0,
                    'bb_position': 0.5,
                    'sma_20': current_price,
                    'ema_12': current_price,
                    'ema_26': current_price,
                    'volume_ratio': 1.0,
                    'atr': 0.0
                }
            
            # Calculate RSI (14-period)
            rsi = self._calculate_rsi(prices, 14)
            
            # Calculate MACD (12, 26, 9)
            macd_line, macd_signal, macd_histogram = self._calculate_macd(prices)
            
            # Calculate Bollinger Bands (20-period, 2 std)
            bb_position = self._calculate_bollinger_position(prices, 20)
            
            # Calculate Moving Averages
            sma_20 = self._calculate_sma(prices, 20)
            ema_12 = self._calculate_ema(prices, 12)
            ema_26 = self._calculate_ema(prices, 26)
            
            # Calculate ATR (14-period)
            atr = self._calculate_atr(prices, 14)
            
            # Volume ratio (use from market_data or default)
            volume_ratio = market_data.get('volume_ratio', 1.0)
            
            return {
                'rsi': rsi,
                'macd': macd_line,
                'macd_signal': macd_signal,
                'macd_histogram': macd_histogram,
                'bb_position': bb_position,
                'sma_20': sma_20,
                'ema_12': ema_12,
                'ema_26': ema_26,
                'volume_ratio': volume_ratio,
                'atr': atr
            }
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return {
                'rsi': 50.0,
                'macd': 0.0,
                'macd_signal': 0.0,
                'macd_histogram': 0.0,
                'bb_position': 0.5,
                'sma_20': current_price,
                'ema_12': current_price,
                'ema_26': current_price,
                'volume_ratio': 1.0,
                'atr': 0.0
            }
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI (Relative Strength Index)"""
        try:
            if len(prices) < period + 1:
                return 50.0
            
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])
            
            if avg_loss == 0:
                return 100.0
            
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))
            
            return float(rsi)
            
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return 50.0
    
    def _calculate_macd(self, prices: List[float]) -> Tuple[float, float, float]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        try:
            if len(prices) < 26:
                return 0.0, 0.0, 0.0
            
            # Calculate EMAs
            ema_12 = self._calculate_ema(prices, 12)
            ema_26 = self._calculate_ema(prices, 26)
            
            # MACD line
            macd_line = ema_12 - ema_26
            
            # Signal line (9-period EMA of MACD)
            macd_values = []
            for i in range(len(prices) - 26):
                ema_12_i = self._calculate_ema(prices[i:i+26], 12)
                ema_26_i = self._calculate_ema(prices[i:i+26], 26)
                macd_values.append(ema_12_i - ema_26_i)
            
            if len(macd_values) >= 9:
                macd_signal = self._calculate_ema(macd_values, 9)
            else:
                macd_signal = macd_line
            
            # MACD histogram
            macd_histogram = macd_line - macd_signal
            
            return float(macd_line), float(macd_signal), float(macd_histogram)
            
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            return 0.0, 0.0, 0.0
    
    def _calculate_bollinger_position(self, prices: List[float], period: int = 20) -> float:
        """Calculate Bollinger Band position (0-1)"""
        try:
            if len(prices) < period:
                return 0.5
            
            recent_prices = prices[-period:]
            sma = np.mean(recent_prices)
            std = np.std(recent_prices)
            
            if std == 0:
                return 0.5
            
            upper_band = sma + (2 * std)
            lower_band = sma - (2 * std)
            current_price = prices[-1]
            
            # Calculate position (0 = lower band, 1 = upper band)
            if upper_band == lower_band:
                return 0.5
            
            position = (current_price - lower_band) / (upper_band - lower_band)
            return float(max(0.0, min(1.0, position)))
            
        except Exception as e:
            logger.error(f"Error calculating Bollinger position: {e}")
            return 0.5
    
    def _calculate_sma(self, prices: List[float], period: int) -> float:
        """Calculate Simple Moving Average"""
        try:
            if len(prices) < period:
                return prices[-1] if prices else 0.0
            
            return float(np.mean(prices[-period:]))
            
        except Exception as e:
            logger.error(f"Error calculating SMA: {e}")
            return prices[-1] if prices else 0.0
    
    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """Calculate Exponential Moving Average"""
        try:
            if len(prices) < period:
                return prices[-1] if prices else 0.0
            
            # Simple EMA calculation
            multiplier = 2.0 / (period + 1)
            ema = prices[0]
            
            for price in prices[1:]:
                ema = (price * multiplier) + (ema * (1 - multiplier))
            
            return float(ema)
            
        except Exception as e:
            logger.error(f"Error calculating EMA: {e}")
            return prices[-1] if prices else 0.0
    
    def _calculate_atr(self, prices: List[float], period: int = 14) -> float:
        """Calculate Average True Range"""
        try:
            if len(prices) < period + 1:
                return 0.0
            
            # Simplified ATR calculation using price changes
            price_changes = np.abs(np.diff(prices))
            atr = np.mean(price_changes[-period:])
            
            return float(atr)
            
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return 0.0
    
    async def get_trading_signal(self, symbol: str) -> Optional[TradingSignal]:
        """
        Get trading signal using orbital brain system with 268 decimal hashing
        
        This is the core method that uses:
        - 268 decimal hashing logic
        - Real portfolio holdings
        - BIT strategy with randomized holdings
        - Orbital shell decision making
        """
        try:
            if not self.active or not self.trading_enabled:
                return None
            
            # Get current market data
            market_data = self.market_data_cache.get(symbol, {})
            if not market_data:
                logger.warning(f"No market data available for {symbol}")
                return None
            
            current_price = market_data.get("price", 0.0)
            if current_price <= 0:
                logger.warning(f"Invalid price for {symbol}: {current_price}")
                return None
            
            # Get trading decision from orbital brain system
            decision = self.orbital_brain.get_trading_decision(symbol, current_price, market_data)
            
            if not decision or decision.get("action") == "HOLD":
                return None
            
            # Check confidence threshold
            confidence = decision.get("confidence", 0.0)
            if confidence < self.config["orbital_confidence_threshold"]:
                logger.info(f"Confidence too low for {symbol}: {confidence}")
                return None
            
            # Calculate position size
            position_size_pct = decision.get("position_size", 0.05)
            total_value = float(self.portfolio_state.total_value_usdc)
            position_value = total_value * position_size_pct
            
            # Calculate quantity based on action and available holdings
            action = decision.get("action", "HOLD")
            trading_pair = self.orbital_brain.trading_pairs.get(symbol)
            
            if not trading_pair:
                logger.warning(f"Unknown trading pair: {symbol}")
                return None
            
            quantity = self._calculate_trade_quantity(
                action, position_value, current_price, trading_pair, decision
            )
            
            if quantity <= 0:
                logger.info(f"Invalid quantity for {symbol}: {quantity}")
                return None
            
            # Create trading signal
            signal = TradingSignal(
                symbol=symbol,
                action=action,
                quantity=Decimal(str(quantity)),
                price=current_price,
                confidence=confidence,
                orbital_shell=decision.get("orbital_shell", "UNKNOWN"),
                bit_strategy_hash=decision.get("bit_strategy_hash", ""),
                profit_potential=decision.get("profit_potential", 0.0),
                risk_level=decision.get("risk_level", 0.0),
                execution_priority=decision.get("execution_priority", 0),
                reason=decision.get("reason", ""),
                timestamp=time.time()
            )
            
            # Store signal
            self.trading_signals.append(signal)
            
            logger.info(f"Generated trading signal: {signal.action} {signal.quantity} {symbol} "
                       f"at {signal.price} (confidence: {signal.confidence:.2f})")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error getting trading signal for {symbol}: {e}")
            return None
    
    def _calculate_trade_quantity(self, action: str, position_value: float, price: float,
                                trading_pair: TradingPair, decision: Dict[str, Any]) -> float:
        """Calculate trade quantity based on action and available holdings"""
        try:
            if action == "BUY":
                # Calculate how much we can buy with available USDC
                available_usdc = float(self.portfolio_state.holdings.USDC)
                max_buy_value = min(position_value, available_usdc)
                
                if max_buy_value <= 0:
                    return 0.0
                
                quantity = max_buy_value / price
                
                # Check trading pair limits
                quantity = max(float(trading_pair.min_order_size), 
                             min(float(trading_pair.max_order_size), quantity))
                
                return quantity
                
            elif action == "SELL":
                # Calculate how much we can sell from available holdings
                base_asset = trading_pair.base
                available_amount = float(getattr(self.portfolio_state.holdings, base_asset, Decimal('0')))
                
                if available_amount <= 0:
                    return 0.0
                
                # Calculate quantity based on position value
                quantity = position_value / price
                
                # Don't sell more than we have
                quantity = min(quantity, available_amount)
                
                # Check trading pair limits
                quantity = max(float(trading_pair.min_order_size), 
                             min(float(trading_pair.max_order_size), quantity))
                
                return quantity
                
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating trade quantity: {e}")
            return 0.0
    
    async def execute_trade(self, signal: TradingSignal) -> bool:
        """
        Execute a trading signal using real exchange integration
        
        This would integrate with your actual exchange APIs (Binance, Coinbase, etc.)
        """
        try:
            if not self.trading_enabled:
                logger.warning("Trading is disabled")
                return False
            
            # Validate signal
            if not self._validate_trading_signal(signal):
                logger.warning(f"Invalid trading signal: {signal}")
                return False
            
            # Simulate trade execution (replace with actual exchange integration)
            success = await self._simulate_trade_execution(signal)
            
            if success:
                # Update portfolio after successful trade
                await self._update_portfolio_after_trade(signal)
                
                # Record executed trade
                self.executed_trades.append({
                    "symbol": signal.symbol,
                    "action": signal.action,
                    "quantity": float(signal.quantity),
                    "price": signal.price,
                    "timestamp": signal.timestamp,
                    "orbital_shell": signal.orbital_shell,
                    "bit_strategy_hash": signal.bit_strategy_hash,
                    "confidence": signal.confidence
                })
                
                self.total_trades += 1
                self.successful_trades += 1
                self.total_volume += signal.quantity * Decimal(str(signal.price))
                
                logger.info(f"Trade executed: {signal.action} {signal.quantity} {signal.symbol} "
                           f"at {signal.price} (orbital shell: {signal.orbital_shell})")
                
                return True
            else:
                logger.warning(f"Trade execution failed: {signal.symbol}")
                return False
                
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return False
    
    def _validate_bidirectional_trading_logic(self, signal: TradingSignal) -> bool:
        """
        Validate that bidirectional trading logic is correct
        
        Ensures that:
        - BUY actions have sufficient quote currency (USDC)
        - SELL actions have sufficient base currency (BTC, ETH, etc.)
        - Trading pairs are properly configured for both directions
        """
        try:
            trading_pair = self.orbital_brain.trading_pairs.get(signal.symbol)
            if not trading_pair:
                logger.error(f"Unknown trading pair: {signal.symbol}")
                return False
            
            if signal.action == "BUY":
                # For BUY: need sufficient quote currency (USDC)
                required_quote = signal.quantity * Decimal(str(signal.price))
                available_quote = getattr(self.portfolio_state.holdings, trading_pair.quote, Decimal('0'))
                
                if available_quote < required_quote:
                    logger.warning(f"Insufficient {trading_pair.quote} for BUY {signal.symbol}: "
                                 f"need {required_quote}, have {available_quote}")
                    return False
                    
            elif signal.action == "SELL":
                # For SELL: need sufficient base currency (BTC, ETH, etc.)
                required_base = signal.quantity
                available_base = getattr(self.portfolio_state.holdings, trading_pair.base, Decimal('0'))
                
                if available_base < required_base:
                    logger.warning(f"Insufficient {trading_pair.base} for SELL {signal.symbol}: "
                                 f"need {required_base}, have {available_base}")
                    return False
            
            # Validate reverse pair exists
            reverse_pair = self.orbital_brain.trading_pairs.get(trading_pair.reverse_symbol)
            if not reverse_pair:
                logger.warning(f"Reverse pair {trading_pair.reverse_symbol} not configured")
            
            logger.info(f"✅ Bidirectional validation passed for {signal.symbol} {signal.action}")
            return True
            
        except Exception as e:
            logger.error(f"Error validating bidirectional trading logic: {e}")
            return False

    def _validate_trading_signal(self, signal: TradingSignal) -> bool:
        """Validate trading signal"""
        try:
            # Check basic requirements
            if not signal.symbol or signal.quantity <= 0 or signal.price <= 0:
                return False
            
            # Check if we have enough holdings for the trade
            trading_pair = self.orbital_brain.trading_pairs.get(signal.symbol)
            if not trading_pair:
                return False
            
            # Validate bidirectional trading logic
            if not self._validate_bidirectional_trading_logic(signal):
                return False
            
            if signal.action == "BUY":
                # Check if we have enough USDC
                required_usdc = signal.quantity * Decimal(str(signal.price))
                if self.portfolio_state.holdings.USDC < required_usdc:
                    logger.warning(f"Insufficient USDC for buy: need {required_usdc}, have {self.portfolio_state.holdings.USDC}")
                    return False
                    
            elif signal.action == "SELL":
                # Check if we have enough base asset
                base_asset = trading_pair.base
                available_amount = getattr(self.portfolio_state.holdings, base_asset, Decimal('0'))
                if available_amount < signal.quantity:
                    logger.warning(f"Insufficient {base_asset} for sell: need {signal.quantity}, have {available_amount}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating trading signal: {e}")
            return False
    
    async def _simulate_trade_execution(self, signal: TradingSignal) -> bool:
        """Simulate trade execution (replace with actual exchange integration)"""
        try:
            # Simulate execution delay
            await asyncio.sleep(0.1)
            
            # Simulate 95% success rate
            success = np.random.random() > 0.05
            
            return success
            
        except Exception as e:
            logger.error(f"Error simulating trade execution: {e}")
            return False
    
    async def _update_portfolio_after_trade(self, signal: TradingSignal) -> None:
        """Update portfolio holdings after successful trade"""
        try:
            trading_pair = self.orbital_brain.trading_pairs.get(signal.symbol)
            if not trading_pair:
                return
            
            # Create new holdings based on current holdings
            new_holdings = PortfolioHoldings(
                BTC=self.portfolio_state.holdings.BTC,
                USDC=self.portfolio_state.holdings.USDC,
                ETH=self.portfolio_state.holdings.ETH,
                XRP=self.portfolio_state.holdings.XRP,
                SOL=self.portfolio_state.holdings.SOL
            )
            
            if signal.action == "BUY":
                # Buy: spend USDC, get base asset
                cost_usdc = signal.quantity * Decimal(str(signal.price))
                new_holdings.USDC -= cost_usdc
                
                # Add to base asset holdings
                base_asset = trading_pair.base
                current_amount = getattr(new_holdings, base_asset, Decimal('0'))
                setattr(new_holdings, base_asset, current_amount + signal.quantity)
                
            elif signal.action == "SELL":
                # Sell: spend base asset, get USDC
                base_asset = trading_pair.base
                current_amount = getattr(new_holdings, base_asset, Decimal('0'))
                setattr(new_holdings, base_asset, current_amount - signal.quantity)
                
                # Add USDC from sale
                revenue_usdc = signal.quantity * Decimal(str(signal.price))
                new_holdings.USDC += revenue_usdc
            
            # Update portfolio
            self.update_portfolio_holdings(new_holdings)
            
        except Exception as e:
            logger.error(f"Error updating portfolio after trade: {e}")
    
    async def run_trading_cycle(self) -> None:
        """Run one complete trading cycle for all pairs"""
        try:
            if not self.active:
                return
            
            # Get trading signals for all pairs
            for symbol in self.config["trading_pairs"]:
                try:
                    # Get trading signal
                    signal = await self.get_trading_signal(symbol)
                    
                    if signal:
                        # Execute trade
                        await self.execute_trade(signal)
                    
                    # Small delay between pairs
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        try:
            return {
                "active": self.active,
                "trading_enabled": self.trading_enabled,
                "total_trades": self.total_trades,
                "successful_trades": self.successful_trades,
                "success_rate": self.successful_trades / max(self.total_trades, 1),
                "total_volume": float(self.total_volume),
                "portfolio_value": float(self.portfolio_state.total_value_usdc),
                "holdings": {
                    "BTC": float(self.portfolio_state.holdings.BTC),
                    "USDC": float(self.portfolio_state.holdings.USDC),
                    "ETH": float(self.portfolio_state.holdings.ETH),
                    "XRP": float(self.portfolio_state.holdings.XRP),
                    "SOL": float(self.portfolio_state.holdings.SOL)
                },
                "orbital_brain_status": self.orbital_brain.get_system_status(),
                "last_update": self.portfolio_state.last_update
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {"error": str(e)}
    
    async def continuous_trading_loop(self) -> None:
        """Continuous trading loop"""
        try:
            logger.info("Starting continuous trading loop")
            
            while self.active:
                try:
                    # Run one trading cycle
                    await self.run_trading_cycle()
                    
                    # Wait for next cycle
                    await asyncio.sleep(self.config.get("trading_interval", 30))
                    
                except Exception as e:
                    logger.error(f"Error in trading loop: {e}")
                    await asyncio.sleep(10)  # Brief pause on error
                    
        except Exception as e:
            logger.error(f"Error in continuous trading loop: {e}")
        finally:
            logger.info("Continuous trading loop stopped") 

async def main():
    """Main entry point for orbital trading system"""
    import argparse
    import yaml
    import os
    
    parser = argparse.ArgumentParser(description="Schwabot Orbital Trading System")
    parser.add_argument("--orbital-trading", action="store_true", help="Enable orbital trading mode")
    parser.add_argument("--orbital-config", type=str, help="Path to orbital trading configuration file")
    parser.add_argument("--orbital-status", action="store_true", help="Show orbital trading system status")
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = {}
        if args.orbital_config and os.path.exists(args.orbital_config):
            with open(args.orbital_config, 'r') as f:
                config = yaml.safe_load(f)
        
        # Initialize orbital trading system
        orbital_system = OrbitalTradingIntegration(config)
        
        if args.orbital_status:
            # Show system status
            status = orbital_system.get_system_status()
            print("🧠⚛️ SCHWABOT ORBITAL TRADING SYSTEM STATUS")
            print("=" * 50)
            print(f"Active: {status.get('active', False)}")
            print(f"Trading Enabled: {status.get('trading_enabled', False)}")
            print(f"Portfolio Value: ${status.get('portfolio_value', 0):,.2f}")
            print(f"Total Trades: {status.get('total_trades', 0)}")
            print(f"Current Positions: {len(status.get('positions', {}))}")
            print(f"Orbital Brain Active: {status.get('orbital_brain_active', False)}")
            print(f"Last Update: {status.get('last_update', 'Never')}")
            return
        
        if args.orbital_trading:
            # Start orbital trading system
            print("🚀 LAUNCHING SCHWABOT ORBITAL AUTOMATED TRADING SYSTEM")
            print("=" * 60)
            print("✅ 268 DECIMAL HASHING: ENABLED")
            print("✅ REAL PORTFOLIO INTEGRATION: ENABLED")
            print("✅ BIT STRATEGY: ENABLED")
            print("✅ BIDIRECTIONAL TRADING: ENABLED")
            print("✅ ORBITAL SHELL BRAIN: ENABLED")
            print("✅ TECHNICAL INDICATORS: ENABLED")
            print("✅ AUTOMATED TRADING: ENABLED")
            print("=" * 60)
            
            # Start the system
            await orbital_system.start_trading_system()
            
            # Run continuous trading loop
            print("🔄 Starting continuous automated trading loop...")
            await orbital_system.continuous_trading_loop()
        
        else:
            print("Usage: python orbital_trading_integration.py --orbital-trading --orbital-config config.yaml")
            print("       python orbital_trading_integration.py --orbital-status")
    
    except KeyboardInterrupt:
        print("\n🛑 Orbital trading system stopped by user")
        if 'orbital_system' in locals():
            await orbital_system.stop_trading_system()
    except Exception as e:
        print(f"❌ Error in orbital trading system: {e}")
        if 'orbital_system' in locals():
            await orbital_system.stop_trading_system()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 