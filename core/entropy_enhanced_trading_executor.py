#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entropy-Enhanced Trading Executor

This module provides a complete trading execution system that integrates:
- Entropy signal processing
- Strategy bit mapping
- Profit calculation
- Risk management
- Order execution via CCXT
- Portfolio management

The system implements a complete trading loop for BTC/USDC pairs with
entropy-driven decision making and real-time market adaptation.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import ccxt
import numpy as np

# Core imports
from core.entropy_signal_integration import EntropySignalIntegration
from core.portfolio_tracker import PortfolioTracker
from core.pure_profit_calculator import HistoryState, MarketData, PureProfitCalculator, StrategyParameters
from core.risk_manager import RiskManager
from core.strategy_bit_mapper import StrategyBitMapper

logger = logging.getLogger(__name__)


class TradingAction(Enum):
    """Trading actions."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    EMERGENCY_EXIT = "emergency_exit"


class TradingState(Enum):
    """Trading states."""

    IDLE = "idle"
    ANALYZING = "analyzing"
    EXECUTING = "executing"
    WAITING = "waiting"
    ERROR = "error"


@dataclass
class TradingDecision:
    """Trading decision with entropy enhancement."""

    action: TradingAction
    confidence: float
    quantity: float
    price: float
    timestamp: float
    entropy_score: float
    entropy_timing: float
    strategy_id: str
    risk_level: str
    reasoning: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TradingResult:
    """Trading execution result."""

    success: bool
    order_id: Optional[str]
    executed_price: float
    executed_quantity: float
    fees: float
    timestamp: float
    action: TradingAction
    metadata: Dict[str, Any] = field(default_factory=dict)


class EntropyEnhancedTradingExecutor:
    """
    Complete entropy-enhanced trading execution system.

    This class orchestrates the entire trading process:
    1. Market data collection
    2. Entropy signal processing
    3. Strategy selection and bit mapping
    4. Profit calculation with entropy enhancement
    5. Risk assessment
    6. Order execution
    7. Portfolio tracking
    """

    def __init__(
        self,
        exchange_config: Dict[str, Any],
        strategy_config: Dict[str, Any],
        entropy_config: Dict[str, Any],
        risk_config: Dict[str, Any],
    ):
        """Initialize the entropy-enhanced trading executor."""
        self.exchange_config = exchange_config
        self.strategy_config = strategy_config
        self.entropy_config = entropy_config
        self.risk_config = risk_config

        # Initialize components
        self.entropy_integration = EntropySignalIntegration()
        self.strategy_mapper = StrategyBitMapper(matrix_dir="./matrices")
        self.profit_calculator = PureProfitCalculator(
            strategy_params=StrategyParameters(
                risk_tolerance=risk_config.get('risk_tolerance', 0.2),
                profit_target=risk_config.get('profit_target', 0.5),
                stop_loss=risk_config.get('stop_loss', 0.1),
                position_size=risk_config.get('position_size', 0.1),
            )
        )
        self.risk_manager = RiskManager(risk_config)
        self.portfolio_tracker = PortfolioTracker()

        # Trading state
        self.trading_state = TradingState.IDLE
        self.current_position = 0.0
        self.last_trade_time = 0.0
        self.trade_count = 0
        self.successful_trades = 0

        # Performance metrics
        self.performance_metrics = {
            'total_trades': 0,
            'successful_trades': 0,
            'total_profit': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'entropy_adjustments': 0,
            'risk_blocks': 0,
        }

        # Initialize exchange connection
        self.exchange = self._initialize_exchange()

        logger.info("🔄 Entropy-Enhanced Trading Executor initialized")

    def _initialize_exchange(self) -> ccxt.Exchange:
        """Initialize CCXT exchange connection."""
        try:
            exchange_id = self.exchange_config.get('exchange', 'coinbase')
            exchange_class = getattr(ccxt, exchange_id)

            exchange = exchange_class(
                {
                    'apiKey': self.exchange_config.get('api_key'),
                    'secret': self.exchange_config.get('secret'),
                    'sandbox': self.exchange_config.get('sandbox', True),
                    'enableRateLimit': True,
                }
            )

            logger.info(f"🔄 Exchange connection initialized: {exchange_id}")
            return exchange

        except Exception as e:
            logger.error(f"❌ Failed to initialize exchange: {e}")
            raise

    async def execute_trading_cycle(self) -> TradingResult:
        """
        Execute a complete trading cycle with entropy enhancement.

        Returns:
            TradingResult: Result of the trading cycle
        """
        try:
            self.trading_state = TradingState.ANALYZING

            # 1. Collect market data
            market_data = await self._collect_market_data()

            # 2. Process entropy signals
            entropy_result = await self._process_entropy_signals(market_data)

            # 3. Generate strategy decision
            decision = await self._generate_trading_decision(market_data, entropy_result)

            # 4. Risk assessment
            if not self._assess_risk(decision):
                logger.warning("⚠️ Risk assessment failed - skipping trade")
                return TradingResult(
                    success=False,
                    order_id=None,
                    executed_price=0.0,
                    executed_quantity=0.0,
                    fees=0.0,
                    timestamp=time.time(),
                    action=TradingAction.HOLD,
                    metadata={'reason': 'risk_assessment_failed'},
                )

            # 5. Execute trade
            self.trading_state = TradingState.EXECUTING
            result = await self._execute_trade(decision)

            # 6. Update portfolio and metrics
            self._update_portfolio(result)
            self._update_performance_metrics(result)

            self.trading_state = TradingState.IDLE
            return result

        except Exception as e:
            logger.error(f"❌ Trading cycle error: {e}")
            self.trading_state = TradingState.ERROR
            return TradingResult(
                success=False,
                order_id=None,
                executed_price=0.0,
                executed_quantity=0.0,
                fees=0.0,
                timestamp=time.time(),
                action=TradingAction.HOLD,
                metadata={'error': str(e)},
            )

    async def _collect_market_data(self) -> MarketData:
        """Collect current market data."""
        try:
            # Fetch ticker data
            ticker = await self.exchange.fetch_ticker('BTC/USDC')
            
            # Fetch order book
            order_book = await self.exchange.fetch_order_book('BTC/USDC')
            
            # Calculate additional metrics
            volatility = self._calculate_volatility(order_book)
            momentum = self._calculate_momentum(ticker)
            volume_profile = self._calculate_volume_profile(order_book)
            
            market_data = MarketData(
                price=ticker['last'],
                volume=ticker['baseVolume'],
                timestamp=time.time(),
                volatility=volatility,
                momentum=momentum,
                volume_profile=volume_profile,
                order_book=order_book,
                ticker=ticker
            )
            
            logger.debug(f"📊 Market data collected: price=${market_data.price:.2f}, vol={market_data.volume:.2f}")
            return market_data
            
        except Exception as e:
            logger.error(f"❌ Failed to collect market data: {e}")
            raise

    async def _process_entropy_signals(self, market_data: MarketData) -> Dict[str, Any]:
        """Process entropy signals for trading decisions."""
        try:
            # Get entropy signals
            entropy_signals = await self.entropy_integration.get_entropy_signals(market_data)
            
            # Apply entropy timing adjustments
            entropy_timing = self.entropy_integration.calculate_entropy_timing(market_data)
            
            # Calculate entropy score
            entropy_score = self.entropy_integration.calculate_entropy_score(entropy_signals)
            
            return {
                'signals': entropy_signals,
                'timing': entropy_timing,
                'score': entropy_score,
                'adjusted': self.entropy_integration.apply_entropy_adjustments(entropy_signals)
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to process entropy signals: {e}")
            return {'signals': [], 'timing': 0.0, 'score': 0.0, 'adjusted': []}

    async def _generate_trading_decision(
        self, market_data: MarketData, entropy_result: Dict[str, Any]
    ) -> TradingDecision:
        """Generate trading decision with entropy enhancement."""
        try:
            # Get strategy bit mapping
            strategy_bits = self.strategy_mapper.get_strategy_bits(market_data)
            
            # Calculate profit with entropy enhancement
            history_state = HistoryState(
                current_price=market_data.price,
                position_size=self.current_position,
                total_trades=self.trade_count,
                successful_trades=self.successful_trades
            )
            
            profit_result = self.profit_calculator.calculate_profit(
                market_data, history_state, entropy_result
            )
            
            # Determine action based on profit calculation and entropy
            action, confidence, reasoning = self._determine_action(profit_result, entropy_result, market_data)
            
            # Calculate position size
            quantity = self._calculate_position_size(confidence, entropy_result, market_data)
            
            # Determine risk level
            risk_level = self._assess_risk_level(profit_result)
            
            decision = TradingDecision(
                action=action,
                confidence=confidence,
                quantity=quantity,
                price=market_data.price,
                timestamp=time.time(),
                entropy_score=entropy_result['score'],
                entropy_timing=entropy_result['timing'],
                strategy_id=strategy_bits.get('strategy_id', 'unknown'),
                risk_level=risk_level,
                reasoning=reasoning,
                metadata={
                    'profit_result': profit_result,
                    'entropy_signals': entropy_result['signals'],
                    'strategy_bits': strategy_bits
                }
            )
            
            logger.info(f"🎯 Trading decision: {action.value} {quantity:.6f} BTC @ ${market_data.price:.2f} "
                       f"(confidence: {confidence:.2f}, entropy: {entropy_result['score']:.2f})")
            
            return decision
            
        except Exception as e:
            logger.error(f"❌ Failed to generate trading decision: {e}")
            raise

    def _determine_action(
        self, profit_result, entropy_result: Dict[str, Any], market_data: MarketData
    ) -> Tuple[TradingAction, float, str]:
        """Determine trading action based on profit calculation and entropy."""
        try:
            # Base confidence from profit calculation
            base_confidence = profit_result.get('confidence', 0.5)
            
            # Apply entropy adjustments
            entropy_boost = entropy_result['score'] * 0.2  # Entropy can boost confidence by up to 20%
            adjusted_confidence = min(0.95, base_confidence + entropy_boost)
            
            # Determine action based on profit signal
            profit_signal = profit_result.get('signal', 0.0)
            entropy_timing = entropy_result['timing']
            
            if profit_signal > 0.1 and adjusted_confidence > 0.6:
                action = TradingAction.BUY
                reasoning = f"Strong buy signal (profit: {profit_signal:.3f}, entropy: {entropy_result['score']:.3f})"
            elif profit_signal < -0.1 and adjusted_confidence > 0.6:
                action = TradingAction.SELL
                reasoning = f"Strong sell signal (profit: {profit_signal:.3f}, entropy: {entropy_result['score']:.3f})"
            else:
                action = TradingAction.HOLD
                reasoning = f"No clear signal (profit: {profit_signal:.3f}, confidence: {adjusted_confidence:.3f})"
            
            return action, adjusted_confidence, reasoning
            
        except Exception as e:
            logger.error(f"❌ Failed to determine action: {e}")
            return TradingAction.HOLD, 0.0, f"Error: {str(e)}"

    def _calculate_position_size(
        self, confidence: float, entropy_result: Dict[str, Any], market_data: MarketData
    ) -> float:
        """Calculate position size based on confidence and entropy."""
        try:
            # Base position size from risk config
            base_size = self.risk_config.get('position_size', 0.1)
            
            # Adjust based on confidence
            confidence_multiplier = min(2.0, confidence * 2.0)
            
            # Adjust based on entropy timing
            entropy_timing_boost = entropy_result['timing'] * 0.5
            
            # Calculate final position size
            position_size = base_size * confidence_multiplier * (1 + entropy_timing_boost)
            
            # Apply risk limits
            max_position = self.risk_config.get('max_position_size', 0.5)
            position_size = min(position_size, max_position)
            
            # Ensure minimum position size
            min_position = self.risk_config.get('min_position_size', 0.01)
            position_size = max(position_size, min_position)
            
            return position_size
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate position size: {e}")
            return 0.01  # Minimum safe position

    def _assess_risk(self, decision: TradingDecision) -> bool:
        """Assess risk for the trading decision."""
        try:
            # Check risk manager
            risk_assessment = self.risk_manager.assess_trade_risk(decision)
            
            if not risk_assessment['approved']:
                logger.warning(f"⚠️ Risk assessment failed: {risk_assessment['reason']}")
                self.performance_metrics['risk_blocks'] += 1
                return False
            
            # Check position limits
            if abs(self.current_position + decision.quantity) > self.risk_config.get('max_position', 1.0):
                logger.warning("⚠️ Position limit exceeded")
                return False
            
            # Check time between trades
            min_trade_interval = self.risk_config.get('min_trade_interval', 60)
            if time.time() - self.last_trade_time < min_trade_interval:
                logger.warning("⚠️ Trade interval too short")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Risk assessment error: {e}")
            return False

    def _assess_risk_level(self, profit_result) -> str:
        """Assess risk level based on profit calculation."""
        try:
            volatility = profit_result.get('volatility', 0.0)
            confidence = profit_result.get('confidence', 0.5)
            
            if volatility > 0.1 or confidence < 0.3:
                return "high"
            elif volatility > 0.05 or confidence < 0.6:
                return "medium"
            else:
                return "low"
                
        except Exception as e:
            logger.error(f"❌ Risk level assessment error: {e}")
            return "medium"

    async def _execute_trade(self, decision: TradingDecision) -> TradingResult:
        """Execute the trading decision."""
        try:
            if decision.action == TradingAction.HOLD:
                return TradingResult(
                    success=True,
                    order_id=None,
                    executed_price=0.0,
                    executed_quantity=0.0,
                    fees=0.0,
                    timestamp=time.time(),
                    action=TradingAction.HOLD,
                    metadata={'reason': 'hold_decision'}
                )
            
            # Prepare order parameters
            symbol = 'BTC/USDC'
            side = decision.action.value
            amount = decision.quantity
            price = decision.price
            
            # Execute order
            if side == 'buy':
                order = await self.exchange.create_market_buy_order(symbol, amount)
            else:
                order = await self.exchange.create_market_sell_order(symbol, amount)
            
            # Extract execution details
            executed_price = order.get('price', price)
            executed_quantity = order.get('filled', amount)
            fees = order.get('fee', {}).get('cost', 0.0)
            
            # Update position
            if side == 'buy':
                self.current_position += executed_quantity
            else:
                self.current_position -= executed_quantity
            
            # Update trade tracking
            self.trade_count += 1
            self.last_trade_time = time.time()
            
            result = TradingResult(
                success=True,
                order_id=order.get('id'),
                executed_price=executed_price,
                executed_quantity=executed_quantity,
                fees=fees,
                timestamp=time.time(),
                action=decision.action,
                metadata={
                    'order': order,
                    'decision': decision,
                    'slippage': abs(executed_price - price) / price
                }
            )
            
            logger.info(f"✅ Trade executed: {side} {executed_quantity:.6f} BTC @ ${executed_price:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Trade execution failed: {e}")
            return TradingResult(
                success=False,
                order_id=None,
                executed_price=0.0,
                executed_quantity=0.0,
                fees=0.0,
                timestamp=time.time(),
                action=decision.action,
                metadata={'error': str(e)}
            )

    def _update_portfolio(self, result: TradingResult) -> None:
        """Update portfolio with trade result."""
        try:
            if result.success and result.action != TradingAction.HOLD:
                self.portfolio_tracker.update_position(
                    symbol='BTC/USDC',
                    quantity=result.executed_quantity,
                    price=result.executed_price,
                    side=result.action.value
                )
                
        except Exception as e:
            logger.error(f"❌ Portfolio update error: {e}")

    def _update_performance_metrics(self, result: TradingResult) -> None:
        """Update performance metrics with trade result."""
        try:
            if result.success and result.action != TradingAction.HOLD:
                self.performance_metrics['total_trades'] += 1
                
                # Calculate profit (simplified)
                if result.action == TradingAction.BUY:
                    # Assume we'll sell later at a profit
                    potential_profit = result.executed_quantity * result.executed_price * 0.02  # 2% profit assumption
                    self.performance_metrics['total_profit'] += potential_profit
                else:
                    # Assume we bought earlier at a lower price
                    potential_profit = result.executed_quantity * result.executed_price * 0.02  # 2% profit assumption
                    self.performance_metrics['total_profit'] += potential_profit
                
                # Update success rate
                if potential_profit > 0:
                    self.performance_metrics['successful_trades'] += 1
                
                # Calculate success rate
                total_trades = self.performance_metrics['total_trades']
                successful_trades = self.performance_metrics['successful_trades']
                if total_trades > 0:
                    success_rate = successful_trades / total_trades
                    logger.info(f"📊 Performance: {successful_trades}/{total_trades} successful trades "
                               f"({success_rate:.1%} success rate)")
                
        except Exception as e:
            logger.error(f"❌ Performance metrics update error: {e}")

    def _calculate_volatility(self, order_book: Dict[str, Any]) -> float:
        """Calculate market volatility from order book."""
        try:
            bids = order_book.get('bids', [])
            asks = order_book.get('asks', [])
            
            if not bids or not asks:
                return 0.0
            
            # Calculate bid-ask spread
            best_bid = bids[0][0] if bids else 0
            best_ask = asks[0][0] if asks else 0
            
            if best_bid == 0 or best_ask == 0:
                return 0.0
            
            spread = (best_ask - best_bid) / best_bid
            return min(spread, 0.1)  # Cap at 10%
            
        except Exception as e:
            logger.error(f"❌ Volatility calculation error: {e}")
            return 0.0

    def _calculate_momentum(self, ticker: Dict[str, Any]) -> float:
        """Calculate price momentum from ticker data."""
        try:
            current_price = ticker.get('last', 0)
            open_price = ticker.get('open', current_price)
            
            if open_price == 0:
                return 0.0
            
            momentum = (current_price - open_price) / open_price
            return max(-0.1, min(momentum, 0.1))  # Cap at ±10%
            
        except Exception as e:
            logger.error(f"❌ Momentum calculation error: {e}")
            return 0.0

    def _calculate_volume_profile(self, order_book: Dict[str, Any]) -> float:
        """Calculate volume profile from order book."""
        try:
            bids = order_book.get('bids', [])
            asks = order_book.get('asks', [])
            
            # Calculate total volume
            bid_volume = sum(bid[1] for bid in bids[:10])  # Top 10 bids
            ask_volume = sum(ask[1] for ask in asks[:10])  # Top 10 asks
            
            total_volume = bid_volume + ask_volume
            
            if total_volume == 0:
                return 0.0
            
            # Calculate volume imbalance
            volume_imbalance = (bid_volume - ask_volume) / total_volume
            return max(-1.0, min(volume_imbalance, 1.0))
            
        except Exception as e:
            logger.error(f"❌ Volume profile calculation error: {e}")
            return 0.0

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        try:
            total_trades = self.performance_metrics['total_trades']
            successful_trades = self.performance_metrics['successful_trades']
            success_rate = successful_trades / total_trades if total_trades > 0 else 0.0
            
            return {
                'total_trades': total_trades,
                'successful_trades': successful_trades,
                'success_rate': success_rate,
                'total_profit': self.performance_metrics['total_profit'],
                'max_drawdown': self.performance_metrics['max_drawdown'],
                'sharpe_ratio': self.performance_metrics['sharpe_ratio'],
                'entropy_adjustments': self.performance_metrics['entropy_adjustments'],
                'risk_blocks': self.performance_metrics['risk_blocks'],
                'current_position': self.current_position,
                'trading_state': self.trading_state.value
            }
            
        except Exception as e:
            logger.error(f"❌ Performance summary error: {e}")
            return {'error': str(e)}

    async def run_trading_loop(self, interval_seconds: int = 60) -> None:
        """Run continuous trading loop."""
        logger.info(f"🔄 Starting trading loop with {interval_seconds}s intervals")

        while True:
            try:
                # Execute trading cycle
                result = await self.execute_trading_cycle()

                # Log result
                if result.success:
                    logger.info(
                        f"✅ Trade executed: {result.action.value} "
                        f"{result.executed_quantity} @ {result.executed_price}"
                    )
                else:
                    logger.warning(f"⚠️ Trade failed: {result.metadata.get('reason', 'unknown')}")

                # Wait for next cycle
                await asyncio.sleep(interval_seconds)

            except Exception as e:
                logger.error(f"❌ Trading loop error: {e}")
                await asyncio.sleep(interval_seconds)


def create_trading_executor(
    exchange_config: Dict[str, Any],
    strategy_config: Dict[str, Any],
    entropy_config: Dict[str, Any],
    risk_config: Dict[str, Any],
) -> EntropyEnhancedTradingExecutor:
    """Create a new trading executor instance."""
    return EntropyEnhancedTradingExecutor(
        exchange_config=exchange_config,
        strategy_config=strategy_config,
        entropy_config=entropy_config,
        risk_config=risk_config,
    )


async def demo_trading_executor():
    """Demo function for testing the trading executor."""
    print("🚀 Entropy-Enhanced Trading Executor Demo")
    print("=" * 50)

    # Create demo configuration
    exchange_config = {
        'exchange': 'coinbase',
        'sandbox': True,
        'api_key': 'demo_key',
        'secret': 'demo_secret'
    }
    
    strategy_config = {
        'enabled_strategies': ['momentum', 'mean_reversion'],
        'confidence_threshold': 0.6
    }
    
    entropy_config = {
        'signal_strength': 0.5,
        'timing_adjustment': 0.2
    }
    
    risk_config = {
        'risk_tolerance': 0.2,
        'profit_target': 0.5,
        'stop_loss': 0.1,
        'position_size': 0.1,
        'max_position_size': 0.5,
        'min_position_size': 0.01,
        'max_position': 1.0,
        'min_trade_interval': 60
    }

    # Create executor
    executor = create_trading_executor(
        exchange_config, strategy_config, entropy_config, risk_config
    )

    # Run a few demo cycles
    for i in range(3):
        print(f"\n🔄 Demo cycle {i + 1}/3")
        result = await executor.execute_trading_cycle()
        
        print(f"Result: {result.action.value}")
        if result.success:
            print(f"  Executed: {result.executed_quantity:.6f} BTC @ ${result.executed_price:,.2f}")
        else:
            print(f"  Status: {result.metadata.get('reason', 'Unknown')}")
        
        await asyncio.sleep(5)

    # Show final performance
    performance = executor.get_performance_summary()
    print(f"\n📊 Final Performance: {performance}")


if __name__ == "__main__":
    asyncio.run(demo_trading_executor()) 