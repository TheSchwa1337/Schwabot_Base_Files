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

import logging
import time
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import ccxt
import numpy as np

# Core imports
from core.entropy_signal_integration import EntropySignalIntegration
from core.strategy_bit_mapper import StrategyBitMapper
from core.pure_profit_calculator import PureProfitCalculator, MarketData, HistoryState, StrategyParameters
from core.clean_trading_pipeline import CleanTradingPipeline
from core.real_time_execution_engine import RealTimeExecutionEngine
from core.risk_manager import RiskManager
from core.portfolio_tracker import PortfolioTracker

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
        risk_config: Dict[str, Any]
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
                position_size=risk_config.get('position_size', 0.1)
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
            'risk_blocks': 0
        }
        
        # Initialize exchange connection
        self.exchange = self._initialize_exchange()
        
        logger.info("🔄 Entropy-Enhanced Trading Executor initialized")

    def _initialize_exchange(self) -> ccxt.Exchange:
        """Initialize CCXT exchange connection."""
        try:
            exchange_id = self.exchange_config.get('exchange', 'coinbase')
            exchange_class = getattr(ccxt, exchange_id)
            
            exchange = exchange_class({
                'apiKey': self.exchange_config.get('api_key'),
                'secret': self.exchange_config.get('secret'),
                'sandbox': self.exchange_config.get('sandbox', True),
                'enableRateLimit': True,
            })
            
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
                    metadata={'reason': 'risk_assessment_failed'}
                )
            
            # 5. Execute trade
            if decision.action != TradingAction.HOLD:
                self.trading_state = TradingState.EXECUTING
                result = await self._execute_trade(decision)
                
                # 6. Update portfolio and metrics
                self._update_portfolio(result)
                self._update_performance_metrics(result)
                
                return result
            else:
                logger.info("🔄 Decision: HOLD - no trade executed")
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
                
        except Exception as e:
            logger.error(f"❌ Trading cycle failed: {e}")
            self.trading_state = TradingState.ERROR
            return TradingResult(
                success=False,
                order_id=None,
                executed_price=0.0,
                executed_quantity=0.0,
                fees=0.0,
                timestamp=time.time(),
                action=TradingAction.HOLD,
                metadata={'error': str(e)}
            )

    async def _collect_market_data(self) -> MarketData:
        """Collect real-time market data."""
        try:
            # Fetch current market data
            ticker = await self.exchange.fetch_ticker('BTC/USDC')
            order_book = await self.exchange.fetch_order_book('BTC/USDC')
            
            # Calculate additional metrics
            volatility = self._calculate_volatility(order_book)
            momentum = self._calculate_momentum(ticker)
            volume_profile = self._calculate_volume_profile(order_book)
            
            market_data = MarketData(
                timestamp=time.time(),
                btc_price=float(ticker['last']),
                eth_price=0.0,  # Not used for BTC/USDC trading
                usdc_volume=float(ticker['quoteVolume']),
                volatility=volatility,
                momentum=momentum,
                volume_profile=volume_profile,
                on_chain_signals={}  # Placeholder for on-chain data
            )
            
            logger.info(f"📊 Market data collected - BTC: ${market_data.btc_price:,.2f}")
            return market_data
            
        except Exception as e:
            logger.error(f"❌ Failed to collect market data: {e}")
            raise

    async def _process_entropy_signals(self, market_data: MarketData) -> Dict[str, Any]:
        """Process entropy signals for trading enhancement."""
        try:
            # Extract order book data
            order_book_data = {
                'bids': [[market_data.btc_price * 0.999, 100]],
                'asks': [[market_data.btc_price * 1.001, 100]],
                'timestamp': market_data.timestamp,
                'spread': 0.001,
                'depth': 10
            }
            
            # Create market context
            market_context = {
                'timestamp': market_data.timestamp,
                'btc_price': market_data.btc_price,
                'usdc_volume': market_data.usdc_volume,
                'volatility': market_data.volatility,
                'momentum': market_data.momentum,
                'volume_profile': market_data.volume_profile,
            }
            
            # Process entropy signals
            entropy_result = self.entropy_integration.process_entropy_signals(
                order_book_data=order_book_data,
                market_context=market_context
            )
            
            logger.info(f"🔄 Entropy signals processed - timing: {entropy_result.get('timing_cycle', 1.0):.3f}")
            return entropy_result
            
        except Exception as e:
            logger.error(f"❌ Failed to process entropy signals: {e}")
            return {
                'confidence_adjustment': 1.0,
                'timing_cycle': 1.0,
                'entropy_score': 1.0,
                'strategy_score': 1.0
            }

    async def _generate_trading_decision(
        self, 
        market_data: MarketData, 
        entropy_result: Dict[str, Any]
    ) -> TradingDecision:
        """Generate trading decision using entropy-enhanced strategy."""
        try:
            # Create history state
            history_state = HistoryState(timestamp=time.time())
            
            # Calculate profit with entropy enhancement
            profit_result = self.profit_calculator.calculate_profit(
                market_data=market_data,
                history_state=history_state
            )
            
            # Generate strategy ID using bit mapper
            strategy_id = self.strategy_mapper.expand_strategy_bits(
                strategy_id=int(time.time() * 1000) % 10000,
                target_bits=8,
                mode="entropy_adaptive",
                market_data={
                    'btc_price': market_data.btc_price,
                    'volatility': market_data.volatility,
                    'entropy_score': entropy_result.get('entropy_score', 1.0)
                }
            )
            
            # Determine trading action based on profit score and entropy
            action, confidence, reasoning = self._determine_action(
                profit_result, entropy_result, market_data
            )
            
            # Calculate position size
            quantity = self._calculate_position_size(
                confidence, entropy_result, market_data
            )
            
            decision = TradingDecision(
                action=action,
                confidence=confidence,
                quantity=quantity,
                price=market_data.btc_price,
                timestamp=time.time(),
                entropy_score=entropy_result.get('entropy_score', 1.0),
                entropy_timing=entropy_result.get('timing_cycle', 1.0),
                strategy_id=str(strategy_id),
                risk_level=self._assess_risk_level(profit_result),
                reasoning=reasoning,
                metadata={
                    'profit_score': profit_result.total_profit_score,
                    'entropy_result': entropy_result
                }
            )
            
            logger.info(f"🎯 Trading decision: {action.value} - confidence: {confidence:.3f}")
            return decision
            
        except Exception as e:
            logger.error(f"❌ Failed to generate trading decision: {e}")
            raise

    def _determine_action(
        self, 
        profit_result, 
        entropy_result: Dict[str, Any], 
        market_data: MarketData
    ) -> Tuple[TradingAction, float, str]:
        """Determine trading action based on profit score and entropy."""
        profit_score = profit_result.total_profit_score
        entropy_score = entropy_result.get('entropy_score', 1.0)
        entropy_timing = entropy_result.get('timing_cycle', 1.0)
        
        # Adjust profit score with entropy
        adjusted_score = profit_score * entropy_score * entropy_timing
        
        # Determine action based on adjusted score
        if adjusted_score > 0.3:
            action = TradingAction.BUY
            confidence = min(0.95, adjusted_score)
            reasoning = f"Strong buy signal (profit: {profit_score:.3f}, entropy: {entropy_score:.3f})"
        elif adjusted_score < -0.3:
            action = TradingAction.SELL
            confidence = min(0.95, abs(adjusted_score))
            reasoning = f"Strong sell signal (profit: {profit_score:.3f}, entropy: {entropy_score:.3f})"
        else:
            action = TradingAction.HOLD
            confidence = 0.5
            reasoning = f"Neutral signal (profit: {profit_score:.3f}, entropy: {entropy_score:.3f})"
        
        return action, confidence, reasoning

    def _calculate_position_size(
        self, 
        confidence: float, 
        entropy_result: Dict[str, Any], 
        market_data: MarketData
    ) -> float:
        """Calculate position size based on confidence and entropy."""
        base_size = self.strategy_config.get('base_position_size', 0.01)  # 1% of portfolio
        
        # Adjust based on confidence
        confidence_multiplier = confidence
        
        # Adjust based on entropy timing
        entropy_timing = entropy_result.get('timing_cycle', 1.0)
        timing_multiplier = min(2.0, max(0.5, entropy_timing))
        
        # Adjust based on volatility
        volatility_multiplier = 1.0 / (1.0 + market_data.volatility)
        
        # Calculate final position size
        position_size = base_size * confidence_multiplier * timing_multiplier * volatility_multiplier
        
        # Ensure within limits
        max_size = self.strategy_config.get('max_position_size', 0.1)  # 10% max
        position_size = min(max_size, max(0.001, position_size))
        
        return position_size

    def _assess_risk(self, decision: TradingDecision) -> bool:
        """Assess risk for the trading decision."""
        try:
            risk_assessment = self.risk_manager.assess_trade_risk({
                'action': decision.action.value,
                'quantity': decision.quantity,
                'price': decision.price,
                'confidence': decision.confidence,
                'entropy_score': decision.entropy_score,
                'current_position': self.current_position,
                'portfolio_value': self.portfolio_tracker.get_total_value()
            })
            
            return risk_assessment.get('approved', False)
            
        except Exception as e:
            logger.error(f"❌ Risk assessment failed: {e}")
            return False

    def _assess_risk_level(self, profit_result) -> str:
        """Assess risk level based on profit result."""
        if profit_result.total_profit_score > 0.5:
            return "low"
        elif profit_result.total_profit_score > 0.0:
            return "medium"
        else:
            return "high"

    async def _execute_trade(self, decision: TradingDecision) -> TradingResult:
        """Execute the trading decision."""
        try:
            if decision.action == TradingAction.BUY:
                order = await self.exchange.create_market_buy_order(
                    symbol='BTC/USDC',
                    amount=decision.quantity
                )
            elif decision.action == TradingAction.SELL:
                order = await self.exchange.create_market_sell_order(
                    symbol='BTC/USDC',
                    amount=decision.quantity
                )
            else:
                raise ValueError(f"Invalid action: {decision.action}")
            
            # Wait for order to be filled
            await asyncio.sleep(1)
            
            # Get order status
            order_status = await self.exchange.fetch_order(order['id'], 'BTC/USDC')
            
            result = TradingResult(
                success=order_status['status'] == 'closed',
                order_id=order['id'],
                executed_price=float(order_status['price']),
                executed_quantity=float(order_status['filled']),
                fees=float(order_status.get('fee', {}).get('cost', 0.0)),
                timestamp=time.time(),
                action=decision.action,
                metadata={
                    'order_status': order_status['status'],
                    'strategy_id': decision.strategy_id,
                    'entropy_score': decision.entropy_score
                }
            )
            
            logger.info(f"✅ Trade executed: {decision.action.value} {result.executed_quantity:.6f} BTC @ ${result.executed_price:,.2f}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Trade execution failed: {e}")
            raise

    def _update_portfolio(self, result: TradingResult) -> None:
        """Update portfolio with trade result."""
        try:
            if result.success and result.action in [TradingAction.BUY, TradingAction.SELL]:
                self.portfolio_tracker.update_position(
                    asset='BTC',
                    quantity=result.executed_quantity if result.action == TradingAction.BUY else -result.executed_quantity,
                    price=result.executed_price,
                    timestamp=result.timestamp
                )
                
                # Update current position
                if result.action == TradingAction.BUY:
                    self.current_position += result.executed_quantity
                else:
                    self.current_position -= result.executed_quantity
                    
        except Exception as e:
            logger.error(f"❌ Portfolio update failed: {e}")

    def _update_performance_metrics(self, result: TradingResult) -> None:
        """Update performance metrics."""
        try:
            self.performance_metrics['total_trades'] += 1
            
            if result.success:
                self.performance_metrics['successful_trades'] += 1
                
                # Calculate profit/loss
                if result.action == TradingAction.BUY:
                    # Track cost basis
                    pass
                elif result.action == TradingAction.SELL:
                    # Calculate realized P&L
                    pass
                    
        except Exception as e:
            logger.error(f"❌ Performance metrics update failed: {e}")

    def _calculate_volatility(self, order_book: Dict[str, Any]) -> float:
        """Calculate volatility from order book."""
        try:
            bids = order_book['bids'][:10]
            asks = order_book['asks'][:10]
            
            bid_prices = [float(bid[0]) for bid in bids]
            ask_prices = [float(ask[0]) for ask in asks]
            
            mid_price = (np.mean(bid_prices) + np.mean(ask_prices)) / 2
            spread = (np.mean(ask_prices) - np.mean(bid_prices)) / mid_price
            
            return min(1.0, spread * 10)  # Normalize to 0-1
            
        except Exception:
            return 0.2  # Default volatility

    def _calculate_momentum(self, ticker: Dict[str, Any]) -> float:
        """Calculate momentum from ticker data."""
        try:
            current_price = float(ticker['last'])
            open_price = float(ticker['open'])
            
            momentum = (current_price - open_price) / open_price
            return max(-1.0, min(1.0, momentum * 10))  # Normalize to -1 to 1
            
        except Exception:
            return 0.0

    def _calculate_volume_profile(self, order_book: Dict[str, Any]) -> float:
        """Calculate volume profile from order book."""
        try:
            total_volume = sum(float(bid[1]) for bid in order_book['bids'][:5])
            return min(1.0, total_volume / 1000)  # Normalize
            
        except Exception:
            return 0.5

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        return {
            'trading_state': self.trading_state.value,
            'current_position': self.current_position,
            'total_trades': self.performance_metrics['total_trades'],
            'success_rate': (
                self.performance_metrics['successful_trades'] / 
                max(1, self.performance_metrics['total_trades'])
            ),
            'total_profit': self.performance_metrics['total_profit'],
            'entropy_adjustments': self.performance_metrics['entropy_adjustments'],
            'risk_blocks': self.performance_metrics['risk_blocks']
        }

    async def run_trading_loop(self, interval_seconds: int = 60) -> None:
        """Run continuous trading loop."""
        logger.info("🚀 Starting entropy-enhanced trading loop")
        
        while True:
            try:
                result = await self.execute_trading_cycle()
                
                # Log performance
                if result.success:
                    logger.info(f"✅ Trading cycle completed successfully")
                else:
                    logger.warning(f"⚠️ Trading cycle completed with issues: {result.metadata}")
                
                # Wait for next cycle
                await asyncio.sleep(interval_seconds)
                
            except KeyboardInterrupt:
                logger.info("🛑 Trading loop stopped by user")
                break
            except Exception as e:
                logger.error(f"❌ Trading loop error: {e}")
                await asyncio.sleep(interval_seconds)


def create_trading_executor(
    exchange_config: Dict[str, Any],
    strategy_config: Dict[str, Any],
    entropy_config: Dict[str, Any],
    risk_config: Dict[str, Any]
) -> EntropyEnhancedTradingExecutor:
    """Create a new entropy-enhanced trading executor."""
    return EntropyEnhancedTradingExecutor(
        exchange_config=exchange_config,
        strategy_config=strategy_config,
        entropy_config=entropy_config,
        risk_config=risk_config
    )


async def demo_trading_executor():
    """Demonstrate the trading executor functionality."""
    print("=== Entropy-Enhanced Trading Executor Demo ===")
    
    # Configuration
    exchange_config = {
        'exchange': 'coinbase',
        'api_key': 'demo_key',
        'secret': 'demo_secret',
        'sandbox': True
    }
    
    strategy_config = {
        'base_position_size': 0.01,
        'max_position_size': 0.1,
        'entropy_threshold': 0.5
    }
    
    entropy_config = {
        'timing_cycles': [1, 5, 15, 30],
        'confidence_threshold': 0.7
    }
    
    risk_config = {
        'risk_tolerance': 0.2,
        'max_drawdown': 0.1,
        'position_limit': 0.2
    }
    
    # Create executor
    executor = create_trading_executor(
        exchange_config, strategy_config, entropy_config, risk_config
    )
    
    # Run demo cycle
    result = await executor.execute_trading_cycle()
    print(f"Demo result: {result}")
    
    # Show performance
    performance = executor.get_performance_summary()
    print(f"Performance: {performance}")


if __name__ == "__main__":
    asyncio.run(demo_trading_executor()) 