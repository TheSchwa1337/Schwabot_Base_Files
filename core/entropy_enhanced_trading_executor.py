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

# Core imports with fallbacks
try:
    from core.entropy_signal_integration import EntropySignalIntegrator
    ENTROPY_AVAILABLE = True
except ImportError:
    ENTROPY_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("⚠️ EntropySignalIntegrator not available - using fallback")

try:
    from core.portfolio_tracker import PortfolioTracker
    PORTFOLIO_AVAILABLE = True
except ImportError:
    PORTFOLIO_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("⚠️ PortfolioTracker not available - using fallback")

try:
    from core.pure_profit_calculator import HistoryState, MarketData, PureProfitCalculator, StrategyParameters
    PROFIT_CALC_AVAILABLE = True
except ImportError:
    PROFIT_CALC_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("⚠️ PureProfitCalculator not available - using fallback")

try:
    from core.risk_manager import RiskManager
    RISK_MANAGER_AVAILABLE = True
except ImportError:
    RISK_MANAGER_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("⚠️ RiskManager not available - using fallback")

try:
    from core.strategy_bit_mapper import StrategyBitMapper
    STRATEGY_MAPPER_AVAILABLE = True
except ImportError:
    STRATEGY_MAPPER_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("⚠️ StrategyBitMapper not available - using fallback")

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


# Fallback classes for missing components
class FallbackEntropyIntegration:
    """Fallback entropy integration when core module is unavailable."""
    
    def __init__(self):
        self.entropy_score = 0.5
        self.entropy_timing = 1.0
    
    async def process_signals(self, market_data):
        """Process entropy signals with fallback implementation."""
        # Simple entropy calculation based on price volatility
        if hasattr(market_data, 'volatility'):
            volatility = market_data.volatility
        else:
            volatility = 0.02
        
        # Entropy increases with volatility
        self.entropy_score = min(1.0, volatility * 10)
        self.entropy_timing = 1.0 - (volatility * 5)
        
        return {
            'entropy_score': self.entropy_score,
            'entropy_timing': self.entropy_timing,
            'signal_strength': self.entropy_score * self.entropy_timing
        }


class FallbackPortfolioTracker:
    """Fallback portfolio tracker when core module is unavailable."""
    
    def __init__(self):
        self.positions = {}
        self.total_value = 0.0
    
    def update_position(self, symbol: str, quantity: float, price: float):
        """Update portfolio position."""
        self.positions[symbol] = {
            'quantity': quantity,
            'avg_price': price,
            'current_value': quantity * price
        }
        self._calculate_total_value()
    
    def _calculate_total_value(self):
        """Calculate total portfolio value."""
        self.total_value = sum(pos['current_value'] for pos in self.positions.values())


class FallbackRiskManager:
    """Fallback risk manager when core module is unavailable."""
    
    def __init__(self, config: Dict[str, Any]):
        self.max_position_size = config.get('max_position_size', 0.1)
        self.max_daily_loss = config.get('max_daily_loss', 0.05)
        self.max_drawdown = config.get('max_drawdown', 0.1)
    
    def assess_risk(self, decision: TradingDecision) -> bool:
        """Assess trading risk."""
        # Basic risk assessment
        if decision.quantity > self.max_position_size:
            return False
        if decision.confidence < 0.3:
            return False
        return True


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

        # Initialize components with fallbacks
        if ENTROPY_AVAILABLE:
            self.entropy_integration = EntropySignalIntegrator()
        else:
            self.entropy_integration = FallbackEntropyIntegration()
        
        if STRATEGY_MAPPER_AVAILABLE:
            self.strategy_mapper = StrategyBitMapper(matrix_dir="./matrices")
        else:
            self.strategy_mapper = None
        
        if PROFIT_CALC_AVAILABLE:
            self.profit_calculator = PureProfitCalculator(
                strategy_params=StrategyParameters(
                    risk_tolerance=risk_config.get('risk_tolerance', 0.2),
                    profit_target=risk_config.get('profit_target', 0.5),
                    stop_loss=risk_config.get('stop_loss', 0.1),
                    position_size=risk_config.get('position_size', 0.1),
                )
            )
        else:
            self.profit_calculator = None
        
        if RISK_MANAGER_AVAILABLE:
            self.risk_manager = RiskManager(risk_config)
        else:
            self.risk_manager = FallbackRiskManager(risk_config)
        
        if PORTFOLIO_AVAILABLE:
            self.portfolio_tracker = PortfolioTracker()
        else:
            self.portfolio_tracker = FallbackPortfolioTracker()

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
        """Execute a complete trading cycle with portfolio integration."""
        try:
            self.trading_state = TradingState.ANALYZING
            
            # Step 1: Sync portfolio with exchange (if available)
            await self.sync_portfolio_with_exchange()
            
            # Step 2: Collect market data
            market_data = await self._collect_market_data()
            
            # Step 3: Update position prices with current market data
            price_data = {'BTC/USDC': market_data.current_price}
            await self.update_position_prices(price_data)
            
            # Step 4: Process entropy signals
            entropy_result = await self._process_entropy_signals(market_data)
            
            # Step 5: Generate trading decision
            decision = await self._generate_trading_decision(market_data, entropy_result)
            
            # Step 6: Assess risk
            if not self._assess_risk(decision):
                logger.info(f"🚫 Risk assessment failed for {decision.action.value}")
                return TradingResult(
                    success=False,
                    order_id=None,
                    executed_price=0.0,
                    executed_quantity=0.0,
                    fees=0.0,
                    timestamp=time.time(),
                    action=decision.action,
                    metadata={'reason': 'risk_assessment_failed'}
                )
            
            # Step 7: Execute trade
            self.trading_state = TradingState.EXECUTING
            result = await self._execute_trade(decision)
            
            # Step 8: Update portfolio and performance metrics
            self._update_portfolio(result)
            self._update_performance_metrics(result)
            
            # Step 9: Update trading state
            self.trading_state = TradingState.IDLE
            self.last_trade_time = time.time()
            
            return result
            
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
        """Collect current market data."""
        try:
            # Get ticker data
            ticker = await self.exchange.fetch_ticker('BTC/USDC')
            
            # Get order book
            order_book = await self.exchange.fetch_order_book('BTC/USDC')
            
            # Calculate market metrics
            volatility = self._calculate_volatility(order_book)
            momentum = self._calculate_momentum(ticker)
            volume_profile = self._calculate_volume_profile(order_book)
            
            # Create market data object
            if PROFIT_CALC_AVAILABLE:
                market_data = MarketData(
                    timestamp=time.time(),
                    btc_price=ticker['last'],
                    eth_price=ticker['last'] * 0.06,  # Approximate ETH/BTC ratio
                    usdc_volume=ticker['quoteVolume'],
                    volatility=volatility,
                    momentum=momentum,
                    volume_profile=volume_profile,
                    on_chain_signals={'whale_activity': 0.3, 'network_health': 0.9}
                )
            else:
                # Fallback market data structure
                market_data = type('MarketData', (), {
                    'timestamp': time.time(),
                    'btc_price': ticker['last'],
                    'volatility': volatility,
                    'momentum': momentum,
                    'volume_profile': volume_profile
                })()
            
            return market_data

        except Exception as e:
            logger.error(f"❌ Failed to collect market data: {e}")
            raise

    async def _process_entropy_signals(self, market_data: MarketData) -> Dict[str, Any]:
        """Process entropy signals for trading decisions."""
        try:
            if ENTROPY_AVAILABLE:
                entropy_result = await self.entropy_integration.process_signals(market_data)
            else:
                entropy_result = await self.entropy_integration.process_signals(market_data)
            
            self.performance_metrics['entropy_adjustments'] += 1
            return entropy_result

        except Exception as e:
            logger.error(f"❌ Failed to process entropy signals: {e}")
            return {
                'entropy_score': 0.5,
                'entropy_timing': 1.0,
                'signal_strength': 0.5
            }

    async def _generate_trading_decision(
        self, market_data: MarketData, entropy_result: Dict[str, Any]
    ) -> TradingDecision:
        """Generate trading decision with entropy enhancement."""
        try:
            # Calculate profit with entropy enhancement
            if PROFIT_CALC_AVAILABLE and self.profit_calculator:
                history_state = HistoryState(timestamp=time.time())
                profit_result = self.profit_calculator.calculate_profit(market_data, history_state)
                
                # Apply entropy adjustment
                adjusted_profit = profit_result.total_profit_score * entropy_result['entropy_score']
                confidence = profit_result.confidence_score * entropy_result['signal_strength']
            else:
                # Fallback profit calculation
                base_profit = market_data.momentum * 0.3 + market_data.volatility * 0.2
                adjusted_profit = base_profit * entropy_result['entropy_score']
                confidence = 0.5 * entropy_result['signal_strength']

            # Determine action
            action, action_confidence, reasoning = self._determine_action(
                adjusted_profit, entropy_result, market_data
            )

            # Calculate position size
            quantity = self._calculate_position_size(confidence, entropy_result, market_data)

            # Determine risk level
            risk_level = self._assess_risk_level(adjusted_profit)

            return TradingDecision(
                action=action,
                confidence=confidence,
                quantity=quantity,
                price=market_data.btc_price,
                timestamp=time.time(),
                entropy_score=entropy_result['entropy_score'],
                entropy_timing=entropy_result['entropy_timing'],
                strategy_id="entropy_enhanced",
                risk_level=risk_level,
                reasoning=reasoning,
                metadata={
                    'profit_score': adjusted_profit,
                    'entropy_adjustment': entropy_result['entropy_score']
                }
            )

        except Exception as e:
            logger.error(f"❌ Failed to generate trading decision: {e}")
            return TradingDecision(
                action=TradingAction.HOLD,
                confidence=0.0,
                quantity=0.0,
                price=market_data.btc_price,
                timestamp=time.time(),
                entropy_score=0.5,
                entropy_timing=1.0,
                strategy_id="fallback",
                risk_level="high",
                reasoning=f"Error in decision generation: {e}",
                metadata={'error': str(e)}
            )

    def _determine_action(
        self, profit_result, entropy_result: Dict[str, Any], market_data: MarketData
    ) -> Tuple[TradingAction, float, str]:
        """Determine trading action based on profit and entropy."""
        try:
            # Define thresholds
            buy_threshold = 0.1
            sell_threshold = -0.1
            confidence_threshold = 0.3

            if profit_result > buy_threshold and entropy_result['signal_strength'] > confidence_threshold:
                return TradingAction.BUY, 0.8, "Strong buy signal with entropy confirmation"
            elif profit_result < sell_threshold and entropy_result['signal_strength'] > confidence_threshold:
                return TradingAction.SELL, 0.8, "Strong sell signal with entropy confirmation"
            elif abs(profit_result) < 0.05:
                return TradingAction.HOLD, 0.6, "Weak signal - holding position"
            else:
                return TradingAction.HOLD, 0.5, "Insufficient signal strength"

        except Exception as e:
            logger.error(f"❌ Error determining action: {e}")
            return TradingAction.HOLD, 0.0, f"Error in action determination: {e}"

    def _calculate_position_size(
        self, confidence: float, entropy_result: Dict[str, Any], market_data: MarketData
    ) -> float:
        """Calculate position size based on confidence and entropy."""
        try:
            # Base position size from risk config
            base_size = self.risk_config.get('position_size', 0.1)
            
            # Adjust based on confidence
            confidence_multiplier = min(1.0, confidence * 2)
            
            # Adjust based on entropy timing
            timing_multiplier = entropy_result.get('entropy_timing', 1.0)
            
            # Calculate final position size
            position_size = base_size * confidence_multiplier * timing_multiplier
            
            # Apply maximum position limit
            max_position = self.risk_config.get('max_position_size', 0.2)
            position_size = min(position_size, max_position)
            
            return position_size

        except Exception as e:
            logger.error(f"❌ Error calculating position size: {e}")
            return 0.01  # Minimal position size as fallback

    def _assess_risk(self, decision: TradingDecision) -> bool:
        """Assess trading risk."""
        try:
            if RISK_MANAGER_AVAILABLE:
                return self.risk_manager.assess_risk(decision)
            else:
                return self.risk_manager.assess_risk(decision)

        except Exception as e:
            logger.error(f"❌ Error in risk assessment: {e}")
            return False

    def _assess_risk_level(self, profit_result) -> str:
        """Assess risk level based on profit result."""
        try:
            if profit_result > 0.2:
                return "low"
            elif profit_result > 0.0:
                return "medium"
            else:
                return "high"

        except Exception as e:
            logger.error(f"❌ Error assessing risk level: {e}")
            return "high"

    async def _execute_trade(self, decision: TradingDecision) -> TradingResult:
        """Execute trade on exchange."""
        try:
            if decision.action == TradingAction.HOLD:
                return TradingResult(
                    success=True,
                    order_id=None,
                    executed_price=0.0,
                    executed_quantity=0.0,
                    fees=0.0,
                    timestamp=time.time(),
                    action=decision.action,
                    metadata={'reason': 'hold_decision'}
                )

            # Prepare order parameters
            symbol = 'BTC/USDC'
            side = decision.action.value
            amount = decision.quantity
            price = decision.price

            # Execute order
            if decision.action in [TradingAction.BUY, TradingAction.SELL]:
                order = await self.exchange.create_market_order(
                    symbol=symbol,
                    side=side,
                    amount=amount
                )

                return TradingResult(
                    success=True,
                    order_id=order.get('id'),
                    executed_price=order.get('price', price),
                    executed_quantity=order.get('amount', amount),
                    fees=order.get('fee', {}).get('cost', 0.0),
                    timestamp=time.time(),
                    action=decision.action,
                    metadata={'order': order}
                )
            else:
                return TradingResult(
                    success=False,
                    order_id=None,
                    executed_price=0.0,
                    executed_quantity=0.0,
                    fees=0.0,
                    timestamp=time.time(),
                    action=decision.action,
                    metadata={'reason': 'invalid_action'}
                )

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
        """Update portfolio with trade result using production-ready portfolio tracker."""
        try:
            if not result.success or result.executed_quantity <= 0:
                return

            symbol = 'BTC/USDC'
            
            if PORTFOLIO_AVAILABLE and hasattr(self, 'portfolio_tracker'):
                # Use production-ready portfolio tracker
                if result.action == TradingAction.BUY:
                    # Opening a long position
                    pos_id = self.portfolio_tracker.open_position(
                        symbol=symbol,
                        quantity=result.executed_quantity,
                        price=result.executed_price,
                        side='buy',
                        metadata={
                            'order_id': result.order_id,
                            'fees': result.fees,
                            'timestamp': result.timestamp,
                            'entropy_score': getattr(result, 'entropy_score', 0.0),
                            'strategy_id': getattr(result, 'strategy_id', 'unknown')
                        }
                    )
                    logger.info(f"📈 Opened long position: {pos_id} - {result.executed_quantity} BTC at ${result.executed_price}")
                    
                elif result.action == TradingAction.SELL:
                    # Check if we have open positions to close
                    open_positions = [pos_id for pos_id, pos in self.portfolio_tracker.positions.items() 
                                    if pos.symbol == symbol and pos.side == 'buy' and not pos.closed]
                    
                    if open_positions:
                        # Close existing long positions
                        for pos_id in open_positions:
                            closed_pos = self.portfolio_tracker.close_position(pos_id, result.executed_price)
                            if closed_pos:
                                logger.info(f"📉 Closed long position: {pos_id} - {closed_pos.quantity} BTC at ${result.executed_price}")
                                logger.info(f"💰 Realized PnL: ${float(closed_pos.realized_pnl):.2f}")
                    else:
                        # Opening a short position (if supported)
                        pos_id = self.portfolio_tracker.open_position(
                            symbol=symbol,
                            quantity=result.executed_quantity,
                            price=result.executed_price,
                            side='sell',
                            metadata={
                                'order_id': result.order_id,
                                'fees': result.fees,
                                'timestamp': result.timestamp,
                                'entropy_score': getattr(result, 'entropy_score', 0.0),
                                'strategy_id': getattr(result, 'strategy_id', 'unknown')
                            }
                        )
                        logger.info(f"📉 Opened short position: {pos_id} - {result.executed_quantity} BTC at ${result.executed_price}")
                
                # Record transaction
                self.portfolio_tracker.record_transaction({
                    'timestamp': result.timestamp,
                    'action': result.action.value,
                    'symbol': symbol,
                    'quantity': result.executed_quantity,
                    'price': result.executed_price,
                    'fees': result.fees,
                    'order_id': result.order_id,
                    'success': result.success
                })
                
                # Update portfolio summary
                summary = self.portfolio_tracker.get_portfolio_summary()
                logger.info(f"💼 Portfolio Summary - Total Value: ${summary['total_value']:.2f}, "
                          f"Realized PnL: ${summary['realized_pnl']:.2f}, "
                          f"Unrealized PnL: ${summary['unrealized_pnl']:.2f}")
                
            else:
                # Use fallback portfolio tracker
                if result.success and result.executed_quantity > 0:
                    symbol = 'BTC/USDC'
                    self.portfolio_tracker.update_position(
                        symbol, result.executed_quantity, result.executed_price
                    )

        except Exception as e:
            logger.error(f"❌ Failed to update portfolio: {e}")
            # Log detailed error for debugging
            import traceback
            logger.error(f"Portfolio update error details: {traceback.format_exc()}")

    async def sync_portfolio_with_exchange(self) -> None:
        """Synchronize portfolio balances with exchange."""
        try:
            if not hasattr(self, 'exchange') or not self.exchange:
                logger.warning("⚠️ No exchange connection available for portfolio sync")
                return
                
            if PORTFOLIO_AVAILABLE and hasattr(self, 'portfolio_tracker'):
                # Fetch current balances from exchange
                balance = await self.exchange.fetch_balance()
                
                # Update portfolio tracker with real exchange balances
                self.portfolio_tracker.sync_balances(balance)
                
                logger.info(f"🔄 Portfolio synchronized with exchange")
                
                # Log current balances
                summary = self.portfolio_tracker.get_portfolio_summary()
                logger.info(f"💰 Exchange Balances: {summary['balances']}")
                
        except Exception as e:
            logger.error(f"❌ Failed to sync portfolio with exchange: {e}")

    async def update_position_prices(self, price_data: Dict[str, float]) -> None:
        """Update position prices with current market data."""
        try:
            if PORTFOLIO_AVAILABLE and hasattr(self, 'portfolio_tracker'):
                # Update unrealized PnL for all open positions
                self.portfolio_tracker.update_prices(price_data)
                
                # Log position updates
                for pos_id, position in self.portfolio_tracker.positions.items():
                    if not position.closed:
                        logger.debug(f"📊 Position {pos_id}: {position.symbol} - "
                                   f"Unrealized PnL: ${float(position.unrealized_pnl):.2f}")
                
        except Exception as e:
            logger.error(f"❌ Failed to update position prices: {e}")

    def get_portfolio_status(self) -> Dict[str, Any]:
        """Get comprehensive portfolio status."""
        try:
            if PORTFOLIO_AVAILABLE and hasattr(self, 'portfolio_tracker'):
                summary = self.portfolio_tracker.get_portfolio_summary()
                
                # Add additional portfolio metrics
                open_positions_count = len([p for p in self.portfolio_tracker.positions.values() if not p.closed])
                closed_positions_count = len(self.portfolio_tracker.closed_positions)
                
                return {
                    **summary,
                    'open_positions_count': open_positions_count,
                    'closed_positions_count': closed_positions_count,
                    'total_transactions': len(self.portfolio_tracker.transaction_history),
                    'last_sync': self.portfolio_tracker.last_update
                }
            else:
                return {
                    'error': 'Portfolio tracker not available',
                    'fallback_total_value': getattr(self.portfolio_tracker, 'total_value', 0.0)
                }
                
        except Exception as e:
            logger.error(f"❌ Failed to get portfolio status: {e}")
            return {'error': str(e)}

    def _update_performance_metrics(self, result: TradingResult) -> None:
        """Update performance metrics."""
        try:
            self.trade_count += 1
            self.performance_metrics['total_trades'] += 1

            if result.success:
                self.successful_trades += 1
                self.performance_metrics['successful_trades'] += 1

                # Calculate profit using portfolio tracker if available
                if PORTFOLIO_AVAILABLE and hasattr(self, 'portfolio_tracker'):
                    # Use portfolio tracker for accurate profit calculation
                    summary = self.portfolio_tracker.get_portfolio_summary()
                    self.performance_metrics['total_profit'] = summary['realized_pnl']
                else:
                    # Fallback profit calculation
                    if result.action == TradingAction.BUY:
                        profit = result.executed_quantity * (result.executed_price - self.current_position)
                    elif result.action == TradingAction.SELL:
                        profit = result.executed_quantity * (self.current_position - result.executed_price)
                    else:
                        profit = 0.0
                    self.performance_metrics['total_profit'] += profit

            # Update current position
            if result.action == TradingAction.BUY:
                self.current_position += result.executed_quantity
            elif result.action == TradingAction.SELL:
                self.current_position -= result.executed_quantity

        except Exception as e:
            logger.error(f"❌ Failed to update performance metrics: {e}")

    def _calculate_volatility(self, order_book: Dict[str, Any]) -> float:
        """Calculate market volatility from order book."""
        try:
            bids = order_book.get('bids', [])
            asks = order_book.get('asks', [])
            
            if not bids or not asks:
                return 0.02  # Default volatility
            
            # Calculate spread
            best_bid = bids[0][0] if bids else 0
            best_ask = asks[0][0] if asks else 0
            
            if best_bid > 0 and best_ask > 0:
                spread = (best_ask - best_bid) / best_bid
                return min(0.1, spread * 10)  # Normalize to 0-0.1 range
            
            return 0.02

        except Exception as e:
            logger.error(f"❌ Error calculating volatility: {e}")
            return 0.02

    def _calculate_momentum(self, ticker: Dict[str, Any]) -> float:
        """Calculate price momentum from ticker."""
        try:
            current_price = ticker.get('last', 0)
            open_price = ticker.get('open', current_price)
            
            if open_price > 0:
                momentum = (current_price - open_price) / open_price
                return np.clip(momentum, -0.1, 0.1)  # Clip to reasonable range
            
            return 0.0

        except Exception as e:
            logger.error(f"❌ Error calculating momentum: {e}")
            return 0.0

    def _calculate_volume_profile(self, order_book: Dict[str, Any]) -> float:
        """Calculate volume profile from order book."""
        try:
            bids = order_book.get('bids', [])
            asks = order_book.get('asks', [])
            
            if not bids or not asks:
                return 1.0  # Default volume profile
            
            # Calculate total volume on each side
            bid_volume = sum(bid[1] for bid in bids[:10])  # Top 10 bids
            ask_volume = sum(ask[1] for ask in asks[:10])  # Top 10 asks
            
            total_volume = bid_volume + ask_volume
            
            if total_volume > 0:
                # Volume profile as ratio of bid volume to total volume
                volume_profile = bid_volume / total_volume
                return np.clip(volume_profile, 0.1, 2.0)  # Reasonable range
            
            return 1.0

        except Exception as e:
            logger.error(f"❌ Error calculating volume profile: {e}")
            return 1.0

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        try:
            win_rate = (
                self.performance_metrics['successful_trades'] / 
                self.performance_metrics['total_trades']
                if self.performance_metrics['total_trades'] > 0 else 0.0
            )

            return {
                'total_trades': self.performance_metrics['total_trades'],
                'successful_trades': self.performance_metrics['successful_trades'],
                'win_rate': win_rate,
                'total_profit': self.performance_metrics['total_profit'],
                'max_drawdown': self.performance_metrics['max_drawdown'],
                'sharpe_ratio': self.performance_metrics['sharpe_ratio'],
                'entropy_adjustments': self.performance_metrics['entropy_adjustments'],
                'risk_blocks': self.performance_metrics['risk_blocks'],
                'current_position': self.current_position,
                'trading_state': self.trading_state.value,
                'last_trade_time': self.last_trade_time
            }

        except Exception as e:
            logger.error(f"❌ Error getting performance summary: {e}")
            return {'error': str(e)}

    async def run_trading_loop(self, interval_seconds: int = 60) -> None:
        """Run continuous trading loop."""
        logger.info(f"🔄 Starting trading loop with {interval_seconds}s intervals")
        
        try:
            while True:
                try:
                    result = await self.execute_trading_cycle()
                    
                    if result.success:
                        logger.info(f"✅ Trade executed: {result.action.value} {result.executed_quantity} BTC at ${result.executed_price}")
                    else:
                        logger.info(f"ℹ️ No trade executed: {result.metadata.get('reason', 'unknown')}")
                    
                    # Wait for next cycle
                    await asyncio.sleep(interval_seconds)
                    
                except Exception as e:
                    logger.error(f"❌ Error in trading loop: {e}")
                    await asyncio.sleep(interval_seconds)
                    
        except KeyboardInterrupt:
            logger.info("🛑 Trading loop stopped by user")
        except Exception as e:
            logger.error(f"❌ Fatal error in trading loop: {e}")


def create_trading_executor(
    exchange_config: Dict[str, Any],
    strategy_config: Dict[str, Any],
    entropy_config: Dict[str, Any],
    risk_config: Dict[str, Any],
) -> EntropyEnhancedTradingExecutor:
    """Create and configure trading executor."""
    return EntropyEnhancedTradingExecutor(
        exchange_config=exchange_config,
        strategy_config=strategy_config,
        entropy_config=entropy_config,
        risk_config=risk_config
    )


async def demo_trading_executor():
    """Demonstrate trading executor functionality."""
    logger.info("🎯 DEMO: Entropy-Enhanced Trading Executor")
    
    # Sample configuration
    exchange_config = {
        'exchange': 'coinbase',
        'api_key': 'demo_key',
        'secret': 'demo_secret',
        'sandbox': True
    }
    
    strategy_config = {
        'strategy_type': 'entropy_enhanced',
        'timeframe': '1m'
    }
    
    entropy_config = {
        'entropy_threshold': 0.7,
        'signal_strength_min': 0.3
    }
    
    risk_config = {
        'risk_tolerance': 0.2,
        'profit_target': 0.5,
        'stop_loss': 0.1,
        'position_size': 0.1,
        'max_position_size': 0.2
    }
    
    # Create executor
    executor = create_trading_executor(
        exchange_config, strategy_config, entropy_config, risk_config
    )
    
    # Run single trading cycle
    result = await executor.execute_trading_cycle()
    
    # Show results
    logger.info(f"Demo result: {result}")
    
    # Show performance
    performance = executor.get_performance_summary()
    logger.info(f"Performance: {performance}")


if __name__ == "__main__":
    asyncio.run(demo_trading_executor()) 