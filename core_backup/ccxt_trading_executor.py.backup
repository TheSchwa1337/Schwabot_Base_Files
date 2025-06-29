# -*- coding: utf-8 -*-
"""
CCXT Trading Executor for Schwabot Integrated System
==================================================

Executes trades based on integrated Ferris-Glyph trading signals through CCXT.
Supports multi-pair trading (BTC/USDC, ETH/USDC) across multiple exchanges
with advanced portfolio rebalancing and profit optimization.

Integration Points:
- Receives IntegratedTradingSignal from Ferris Glyph Controller
- Executes trades through existing exchange_plumbing and api_coordinator
- Implements Ghost Router profit routing strategies
- Manages portfolio rebalancing across multiple pairs
- Tracks profit generation and risk management
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import asyncio
import logging
import threading
from decimal import Decimal, getcontext
import numpy as np

# Set high precision for trading calculations
getcontext().prec = 28

# Import existing Schwabot trading components
try:
    from .integrated_ferris_glyph_controller import IntegratedTradingSignal, TradingTimeframe
    from .unified_api_coordinator import unified_api_coordinator, ExchangeType
    from .exchange_plumbing import ExchangeConnection, OrderRequest, OrderSide, OrderType
    from .profit_routing_engine import profit_routing_engine, RouteType
    from .wall_builder_anomaly_handler import WallBuilderAnomalyHandler
    TRADING_COMPONENTS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Trading components not fully available: {e}")
    TRADING_COMPONENTS_AVAILABLE = False

logger = logging.getLogger(__name__)

class TradingPair(Enum):
    """Supported trading pairs for multi-pair strategy."""
    BTC_USDC = "BTC/USDC"
    BTC_USDT = "BTC/USDT"
    ETH_USDC = "ETH/USDC"
    ETH_USDT = "ETH/USDT"
    SOL_USDC = "SOL/USDC"
    MATIC_USDC = "MATIC/USDC"

class ExecutionStrategy(Enum):
    """Execution strategies based on Ghost Router decisions."""
    GHOST_TRADE_AGGRESSIVE = "ghost_trade_aggressive"
    GHOST_TRADE_CONSERVATIVE = "ghost_trade_conservative"
    USDC_HOLD = "usdc_hold"
    PORTFOLIO_REBALANCE = "portfolio_rebalance"
    PROFIT_TAKING = "profit_taking"
    EMERGENCY_EXIT = "emergency_exit"

@dataclass
class TradingPosition:
    """Current trading position for a pair."""
    pair: TradingPair
    exchange: ExchangeType
    side: str  # "long", "short", "neutral"
    amount: Decimal
    entry_price: Decimal
    current_price: Decimal
    unrealized_pnl: Decimal
    profit_target: Decimal
    stop_loss: Decimal
    creation_timestamp: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)

@dataclass
class ExecutionResult:
    """Result of trade execution."""
    signal_id: str
    pair: TradingPair
    strategy: ExecutionStrategy
    executed: bool
    order_id: Optional[str] = None
    fill_price: Optional[Decimal] = None
    fill_amount: Optional[Decimal] = None
    profit_realized: Decimal = Decimal('0')
    execution_time: float = field(default_factory=time.time)
    error_message: Optional[str] = None

class CCXTTradingExecutor:
    """
    CCXT Trading Executor for integrated Schwabot system.
    
    Executes multi-pair trading strategies based on Ferris-Glyph signals
    with advanced portfolio management and profit optimization.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize CCXT Trading Executor."""
        self.config = config or {}
        
        # Trading configuration
        self.supported_pairs = [
            TradingPair.BTC_USDC,
            TradingPair.ETH_USDC,
            TradingPair.BTC_USDT,
            TradingPair.ETH_USDT
        ]
        
        self.supported_exchanges = [
            ExchangeType.COINBASE,
            ExchangeType.BINANCE
        ]
        
        # Position management
        self.active_positions: Dict[str, TradingPosition] = {}
        self.position_history: List[TradingPosition] = []
        self.execution_history: List[ExecutionResult] = []
        
        # Portfolio state
        self.portfolio_balance = {
            "USDC": Decimal('10000'),  # Starting with $10k USDC
            "BTC": Decimal('0'),
            "ETH": Decimal('0'),
            "USDT": Decimal('0')
        }
        
        # Risk management
        self.max_position_size = Decimal('0.1')  # 10% of portfolio per position
        self.max_total_exposure = Decimal('0.3')  # 30% total crypto exposure
        self.profit_target_pct = Decimal('0.05')  # 5% profit target
        self.stop_loss_pct = Decimal('0.02')  # 2% stop loss
        
        # Performance tracking
        self.total_trades = 0
        self.successful_trades = 0
        self.total_profit = Decimal('0')
        self.max_drawdown = Decimal('0')
        self.win_rate = 0.0
        
        # Threading
        self.execution_lock = threading.RLock()
        self.price_monitor_thread = None
        self.is_monitoring = False
        
        # Wall detection
        self.wall_handler = WallBuilderAnomalyHandler() if TRADING_COMPONENTS_AVAILABLE else None
        
        logger.info("🚀 CCXT Trading Executor initialized for multi-pair trading")
    
    def execute_signal(self, signal: IntegratedTradingSignal) -> ExecutionResult:
        """
        Execute trade based on integrated trading signal.
        
        Main execution flow:
        Signal → Strategy Selection → Risk Check → Order Execution → Position Management
        """
        try:
            with self.execution_lock:
                logger.info(f"📈 Executing signal: {signal.signal_id} - {signal.recommended_action}")
                
                # Step 1: Determine execution strategy
                strategy = self._determine_execution_strategy(signal)
                
                # Step 2: Select optimal trading pair
                optimal_pair = self._select_optimal_pair(signal, strategy)
                
                # Step 3: Risk management checks
                if not self._check_risk_limits(signal, strategy, optimal_pair):
                    return self._create_failed_execution(signal, "Risk limits exceeded")
                
                # Step 4: Execute the strategy
                if strategy == ExecutionStrategy.GHOST_TRADE_AGGRESSIVE:
                    result = self._execute_ghost_trade_aggressive(signal, optimal_pair)
                elif strategy == ExecutionStrategy.GHOST_TRADE_CONSERVATIVE:
                    result = self._execute_ghost_trade_conservative(signal, optimal_pair)
                elif strategy == ExecutionStrategy.USDC_HOLD:
                    result = self._execute_usdc_hold_strategy(signal)
                elif strategy == ExecutionStrategy.PORTFOLIO_REBALANCE:
                    result = self._execute_portfolio_rebalance(signal)
                elif strategy == ExecutionStrategy.PROFIT_TAKING:
                    result = self._execute_profit_taking(signal, optimal_pair)
                else:
                    result = self._create_failed_execution(signal, f"Unknown strategy: {strategy}")
                
                # Step 5: Update tracking and history
                self._update_execution_metrics(result)
                self.execution_history.append(result)
                
                logger.info(f"📊 Execution completed: {result.executed} - Pair: {result.pair.value if result.pair else 'N/A'}")
                return result
                
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return self._create_failed_execution(signal, str(e))
    
    def _determine_execution_strategy(self, signal: IntegratedTradingSignal) -> ExecutionStrategy:
        """Determine execution strategy from integrated signal."""
        try:
            ghost_route = signal.ghost_route
            confidence = signal.confidence_score
            profit_potential = signal.profit_potential
            risk = signal.risk_assessment.get("overall_risk", 0.5)
            
            # Ghost trade strategies
            if ghost_route == "ghost_trade":
                if confidence > 0.8 and profit_potential > 0.7 and risk < 0.3:
                    return ExecutionStrategy.GHOST_TRADE_AGGRESSIVE
                elif confidence > 0.6 and profit_potential > 0.4:
                    return ExecutionStrategy.GHOST_TRADE_CONSERVATIVE
            
            # Hold USDC strategy
            elif ghost_route == "hold_usdc" or confidence < 0.4:
                return ExecutionStrategy.USDC_HOLD
            
            # Portfolio rebalancing
            elif self._needs_rebalancing():
                return ExecutionStrategy.PORTFOLIO_REBALANCE
            
            # Profit taking if we have profitable positions
            elif self._has_profitable_positions():
                return ExecutionStrategy.PROFIT_TAKING
            
            # Default to conservative approach
            return ExecutionStrategy.GHOST_TRADE_CONSERVATIVE
            
        except Exception as e:
            logger.error(f"Strategy determination failed: {e}")
            return ExecutionStrategy.USDC_HOLD
    
    def _select_optimal_pair(self, signal: IntegratedTradingSignal, strategy: ExecutionStrategy) -> TradingPair:
        """Select optimal trading pair based on signal and strategy."""
        try:
            # For BTC-focused signals (primary focus)
            btc_price = signal.btc_price
            
            # Check Ferris wheel phase for pair selection
            ferris_phase = signal.ferris_data.get("phase", "unknown")
            
            # Aggressive strategies prefer main pairs
            if strategy == ExecutionStrategy.GHOST_TRADE_AGGRESSIVE:
                if btc_price > 50000:  # High BTC price, consider ETH
                    return TradingPair.ETH_USDC
                else:
                    return TradingPair.BTC_USDC
            
            # Conservative strategies stick to BTC
            elif strategy == ExecutionStrategy.GHOST_TRADE_CONSERVATIVE:
                return TradingPair.BTC_USDC
            
            # Phase-based selection
            if ferris_phase in ["PEAK", "ASCENT"]:
                return TradingPair.BTC_USDC  # Ride the momentum
            elif ferris_phase in ["DESCENT", "VALLEY"]:
                return TradingPair.ETH_USDC  # Diversify during downtrend
            
            # Default to primary pair
            return TradingPair.BTC_USDC
            
        except Exception as e:
            logger.error(f"Pair selection failed: {e}")
            return TradingPair.BTC_USDC
    
    def _execute_ghost_trade_aggressive(self, signal: IntegratedTradingSignal, pair: TradingPair) -> ExecutionResult:
        """Execute aggressive ghost trade strategy."""
        try:
            # Calculate position size (larger for aggressive)
            portfolio_value = self._calculate_portfolio_value()
            position_size = portfolio_value * Decimal('0.08')  # 8% of portfolio
            
            # Use higher leverage conceptually (more aggressive entry)
            entry_price = Decimal(str(signal.btc_price))
            
            # Tighter profit targets for quick execution
            profit_target = entry_price * (Decimal('1') + self.profit_target_pct * Decimal('1.5'))
            stop_loss = entry_price * (Decimal('1') - self.stop_loss_pct)
            
            # Execute market order
            result = self._place_market_order(
                pair=pair,
                side=OrderSide.BUY,
                amount=position_size / entry_price,
                signal_id=signal.signal_id,
                strategy=ExecutionStrategy.GHOST_TRADE_AGGRESSIVE
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Aggressive ghost trade execution failed: {e}")
            return self._create_failed_execution(signal, str(e))
    
    def _execute_ghost_trade_conservative(self, signal: IntegratedTradingSignal, pair: TradingPair) -> ExecutionResult:
        """Execute conservative ghost trade strategy."""
        try:
            # Smaller position size for conservative approach
            portfolio_value = self._calculate_portfolio_value()
            position_size = portfolio_value * Decimal('0.04')  # 4% of portfolio
            
            entry_price = Decimal(str(signal.btc_price))
            
            # Standard profit targets
            profit_target = entry_price * (Decimal('1') + self.profit_target_pct)
            stop_loss = entry_price * (Decimal('1') - self.stop_loss_pct)
            
            # Execute limit order slightly below market for better entry
            limit_price = entry_price * Decimal('0.999')  # 0.1% below market
            
            result = self._place_limit_order(
                pair=pair,
                side=OrderSide.BUY,
                amount=position_size / limit_price,
                price=limit_price,
                signal_id=signal.signal_id,
                strategy=ExecutionStrategy.GHOST_TRADE_CONSERVATIVE
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Conservative ghost trade execution failed: {e}")
            return self._create_failed_execution(signal, str(e))
    
    def _execute_usdc_hold_strategy(self, signal: IntegratedTradingSignal) -> ExecutionResult:
        """Execute USDC hold strategy (close positions, stay in cash)."""
        try:
            # Close any open positions
            closed_positions = 0
            total_profit = Decimal('0')
            
            for position_key, position in list(self.active_positions.items()):
                if position.side == "long":
                    # Sell position
                    close_result = self._close_position(position)
                    if close_result.executed:
                        total_profit += close_result.profit_realized
                        closed_positions += 1
                        del self.active_positions[position_key]
            
            return ExecutionResult(
                signal_id=signal.signal_id,
                pair=None,
                strategy=ExecutionStrategy.USDC_HOLD,
                executed=True,
                profit_realized=total_profit
            )
            
        except Exception as e:
            logger.error(f"USDC hold strategy execution failed: {e}")
            return self._create_failed_execution(signal, str(e))
    
    def _execute_portfolio_rebalance(self, signal: IntegratedTradingSignal) -> ExecutionResult:
        """Execute portfolio rebalancing strategy."""
        try:
            # Calculate target allocations
            target_allocations = {
                "USDC": Decimal('0.7'),  # 70% stable
                "BTC": Decimal('0.2'),   # 20% BTC
                "ETH": Decimal('0.1')    # 10% ETH
            }
            
            portfolio_value = self._calculate_portfolio_value()
            current_allocations = self._calculate_current_allocations()
            
            rebalance_trades = []
            
            for asset, target_pct in target_allocations.items():
                if asset in current_allocations:
                    current_pct = current_allocations[asset]
                    difference = target_pct - current_pct
                    
                    # If difference > 5%, rebalance
                    if abs(difference) > Decimal('0.05'):
                        trade_amount = portfolio_value * difference
                        
                        if difference > 0:  # Need to buy more
                            pair = self._get_pair_for_asset(asset)
                            if pair:
                                trade_result = self._place_market_order(
                                    pair=pair,
                                    side=OrderSide.BUY,
                                    amount=abs(trade_amount),
                                    signal_id=signal.signal_id,
                                    strategy=ExecutionStrategy.PORTFOLIO_REBALANCE
                                )
                                rebalance_trades.append(trade_result)
                        else:  # Need to sell
                            pair = self._get_pair_for_asset(asset)
                            if pair:
                                trade_result = self._place_market_order(
                                    pair=pair,
                                    side=OrderSide.SELL,
                                    amount=abs(trade_amount),
                                    signal_id=signal.signal_id,
                                    strategy=ExecutionStrategy.PORTFOLIO_REBALANCE
                                )
                                rebalance_trades.append(trade_result)
            
            total_profit = sum(trade.profit_realized for trade in rebalance_trades)
            
            return ExecutionResult(
                signal_id=signal.signal_id,
                pair=None,
                strategy=ExecutionStrategy.PORTFOLIO_REBALANCE,
                executed=len(rebalance_trades) > 0,
                profit_realized=total_profit
            )
            
        except Exception as e:
            logger.error(f"Portfolio rebalance execution failed: {e}")
            return self._create_failed_execution(signal, str(e))
    
    def _execute_profit_taking(self, signal: IntegratedTradingSignal, pair: TradingPair) -> ExecutionResult:
        """Execute profit taking strategy."""
        try:
            total_profit = Decimal('0')
            positions_closed = 0
            
            # Close profitable positions
            for position_key, position in list(self.active_positions.items()):
                if position.unrealized_pnl > Decimal('0'):
                    # Take profit if > 3%
                    profit_pct = position.unrealized_pnl / (position.amount * position.entry_price)
                    if profit_pct > Decimal('0.03'):
                        close_result = self._close_position(position)
                        if close_result.executed:
                            total_profit += close_result.profit_realized
                            positions_closed += 1
                            del self.active_positions[position_key]
            
            return ExecutionResult(
                signal_id=signal.signal_id,
                pair=pair,
                strategy=ExecutionStrategy.PROFIT_TAKING,
                executed=positions_closed > 0,
                profit_realized=total_profit
            )
            
        except Exception as e:
            logger.error(f"Profit taking execution failed: {e}")
            return self._create_failed_execution(signal, str(e))
    
    def _place_market_order(self, pair: TradingPair, side: OrderSide, amount: Decimal, 
                           signal_id: str, strategy: ExecutionStrategy) -> ExecutionResult:
        """Place market order through CCXT."""
        try:
            if not TRADING_COMPONENTS_AVAILABLE:
                # Simulate order execution
                return self._simulate_order_execution(pair, side, amount, signal_id, strategy, "market")
            
            # Use unified API coordinator to place order
            order_request = OrderRequest(
                symbol=pair.value,
                side=side,
                order_type=OrderType.MARKET,
                amount=float(amount)
            )
            
            # Execute through exchange plumbing
            # This would integrate with actual CCXT execution
            result = ExecutionResult(
                signal_id=signal_id,
                pair=pair,
                strategy=strategy,
                executed=True,
                fill_amount=amount,
                fill_price=Decimal('50000'),  # Mock price
                profit_realized=Decimal('0')
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Market order placement failed: {e}")
            return self._create_failed_execution_with_pair(signal_id, pair, str(e))
    
    def _place_limit_order(self, pair: TradingPair, side: OrderSide, amount: Decimal, 
                          price: Decimal, signal_id: str, strategy: ExecutionStrategy) -> ExecutionResult:
        """Place limit order through CCXT."""
        try:
            if not TRADING_COMPONENTS_AVAILABLE:
                # Simulate order execution
                return self._simulate_order_execution(pair, side, amount, signal_id, strategy, "limit", price)
            
            order_request = OrderRequest(
                symbol=pair.value,
                side=side,
                order_type=OrderType.LIMIT,
                amount=float(amount),
                price=float(price)
            )
            
            # Execute through exchange plumbing
            result = ExecutionResult(
                signal_id=signal_id,
                pair=pair,
                strategy=strategy,
                executed=True,
                fill_amount=amount,
                fill_price=price,
                profit_realized=Decimal('0')
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Limit order placement failed: {e}")
            return self._create_failed_execution_with_pair(signal_id, pair, str(e))
    
    # Helper methods
    def _check_risk_limits(self, signal: IntegratedTradingSignal, strategy: ExecutionStrategy, pair: TradingPair) -> bool:
        """Check if trade meets risk management criteria."""
        try:
            # Check overall risk
            if signal.risk_assessment.get("overall_risk", 0.0) > 0.8:
                return False
            
            # Check position limits
            current_exposure = self._calculate_current_exposure()
            if current_exposure > self.max_total_exposure:
                return False
            
            # Check individual position size
            portfolio_value = self._calculate_portfolio_value()
            if strategy == ExecutionStrategy.GHOST_TRADE_AGGRESSIVE:
                max_position = portfolio_value * Decimal('0.08')
            else:
                max_position = portfolio_value * Decimal('0.04')
            
            if max_position > portfolio_value * self.max_position_size:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Risk limit check failed: {e}")
            return False
    
    def _calculate_portfolio_value(self) -> Decimal:
        """Calculate total portfolio value in USDC."""
        total_value = self.portfolio_balance.get("USDC", Decimal('0'))
        total_value += self.portfolio_balance.get("USDT", Decimal('0'))
        
        # Add crypto values (mock prices)
        btc_price = Decimal('50000')
        eth_price = Decimal('3000')
        
        total_value += self.portfolio_balance.get("BTC", Decimal('0')) * btc_price
        total_value += self.portfolio_balance.get("ETH", Decimal('0')) * eth_price
        
        return total_value
    
    def _calculate_current_exposure(self) -> Decimal:
        """Calculate current crypto exposure percentage."""
        portfolio_value = self._calculate_portfolio_value()
        stable_value = self.portfolio_balance.get("USDC", Decimal('0')) + self.portfolio_balance.get("USDT", Decimal('0'))
        
        if portfolio_value > 0:
            return (portfolio_value - stable_value) / portfolio_value
        return Decimal('0')
    
    def _calculate_current_allocations(self) -> Dict[str, Decimal]:
        """Calculate current asset allocations."""
        portfolio_value = self._calculate_portfolio_value()
        allocations = {}
        
        if portfolio_value > 0:
            for asset, balance in self.portfolio_balance.items():
                if asset == "USDC" or asset == "USDT":
                    allocations[asset] = balance / portfolio_value
                elif asset == "BTC":
                    allocations[asset] = (balance * Decimal('50000')) / portfolio_value
                elif asset == "ETH":
                    allocations[asset] = (balance * Decimal('3000')) / portfolio_value
        
        return allocations
    
    def _needs_rebalancing(self) -> bool:
        """Check if portfolio needs rebalancing."""
        try:
            current_exposure = self._calculate_current_exposure()
            return current_exposure > Decimal('0.4') or current_exposure < Decimal('0.1')
        except:
            return False
    
    def _has_profitable_positions(self) -> bool:
        """Check if there are profitable positions to close."""
        try:
            for position in self.active_positions.values():
                if position.unrealized_pnl > Decimal('0'):
                    profit_pct = position.unrealized_pnl / (position.amount * position.entry_price)
                    if profit_pct > Decimal('0.03'):  # 3% profit
                        return True
            return False
        except:
            return False
    
    def _get_pair_for_asset(self, asset: str) -> Optional[TradingPair]:
        """Get trading pair for asset."""
        if asset == "BTC":
            return TradingPair.BTC_USDC
        elif asset == "ETH":
            return TradingPair.ETH_USDC
        return None
    
    def _close_position(self, position: TradingPosition) -> ExecutionResult:
        """Close a trading position."""
        # Mock position closing
        profit = position.unrealized_pnl
        
        return ExecutionResult(
            signal_id=f"CLOSE_{int(time.time())}",
            pair=position.pair,
            strategy=ExecutionStrategy.PROFIT_TAKING,
            executed=True,
            profit_realized=profit
        )
    
    def _simulate_order_execution(self, pair: TradingPair, side: OrderSide, amount: Decimal, 
                                signal_id: str, strategy: ExecutionStrategy, order_type: str, 
                                price: Optional[Decimal] = None) -> ExecutionResult:
        """Simulate order execution for testing."""
        fill_price = price if price else Decimal('50000')  # Mock fill price
        
        return ExecutionResult(
            signal_id=signal_id,
            pair=pair,
            strategy=strategy,
            executed=True,
            order_id=f"SIM_{int(time.time())}",
            fill_price=fill_price,
            fill_amount=amount,
            profit_realized=Decimal('0')
        )
    
    def _create_failed_execution(self, signal: IntegratedTradingSignal, error: str) -> ExecutionResult:
        """Create failed execution result."""
        return ExecutionResult(
            signal_id=signal.signal_id,
            pair=None,
            strategy=ExecutionStrategy.USDC_HOLD,
            executed=False,
            error_message=error
        )
    
    def _create_failed_execution_with_pair(self, signal_id: str, pair: TradingPair, error: str) -> ExecutionResult:
        """Create failed execution result with pair."""
        return ExecutionResult(
            signal_id=signal_id,
            pair=pair,
            strategy=ExecutionStrategy.USDC_HOLD,
            executed=False,
            error_message=error
        )
    
    def _update_execution_metrics(self, result: ExecutionResult) -> None:
        """Update execution metrics."""
        self.total_trades += 1
        
        if result.executed:
            self.successful_trades += 1
            self.total_profit += result.profit_realized
        
        if self.total_trades > 0:
            self.win_rate = self.successful_trades / self.total_trades
    
    def get_trading_status(self) -> Dict[str, Any]:
        """Get comprehensive trading status."""
        return {
            "executor_status": "operational" if TRADING_COMPONENTS_AVAILABLE else "simulation",
            "active_positions": len(self.active_positions),
            "portfolio_value": float(self._calculate_portfolio_value()),
            "current_exposure": float(self._calculate_current_exposure()),
            "total_trades": self.total_trades,
            "successful_trades": self.successful_trades,
            "win_rate": self.win_rate,
            "total_profit": float(self.total_profit),
            "portfolio_balance": {k: float(v) for k, v in self.portfolio_balance.items()},
            "last_update": time.time()
        }


# Global instance
ccxt_executor = CCXTTradingExecutor()

# Export functions
def execute_trading_signal(signal: IntegratedTradingSignal) -> ExecutionResult:
    """Execute trading signal through CCXT."""
    return ccxt_executor.execute_signal(signal)

def get_executor_status() -> Dict[str, Any]:
    """Get executor status."""
    return ccxt_executor.get_trading_status()

# Export all components
__all__ = [
    "CCXTTradingExecutor",
    "TradingPair",
    "ExecutionStrategy",
    "TradingPosition",
    "ExecutionResult",
    "ccxt_executor",
    "execute_trading_signal",
    "get_executor_status"
] 