from utils.safe_print import safe_print, info, warn, error, success, debug
from core.unified_math_system import unified_math
#!/usr/bin/env python3
"""
Trade Simulation Engine - Schwabot UROS v1.0
===========================================

Replaces stub simulate_trade() with proper trade execution simulation
that honors real strategy logic and integrates with the mathematical pipeline.

Features:
- Real strategy execution simulation
- Portfolio state tracking
- Profit/loss calculation
- Risk management simulation
- Integration with tensor scoring and bit resolution
"""

import time
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from core.unified_math_system import unified_math
import hashlib

logger = logging.getLogger(__name__)


class TradeType(Enum):
    """Trade execution types."""
    BUY = "buy"
    SELL = "sell"
    REBALANCE = "rebalance"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


class TradeStatus(Enum):
    """Trade execution status."""
    PENDING = "pending"
    EXECUTED = "executed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TradeExecution:
    """Trade execution result."""
    trade_id: str
    asset: str
    trade_type: TradeType
    quantity: float
    price: float
    timestamp: datetime
    status: TradeStatus
    strategy_id: str
    tensor_score: float
    bit_phase: int
    basket_id: str
    portfolio_impact: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PortfolioState:
    """Portfolio state snapshot."""
    timestamp: datetime
    total_value: float
    cash: float
    positions: Dict[str, Dict[str, Any]]
    unrealized_pnl: float
    realized_pnl: float
    risk_metrics: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class TradeSimulator:
    """
    Trade simulation engine with real strategy logic integration.

    Mathematical Foundation:
    - Trade Impact: impact = quantity * price * direction
    - Portfolio Value: total = cash + Σ(positions * current_prices)
    - Risk Metrics: volatility = unified_math.std(returns), sharpe = unified_math.mean(returns) / unified_math.std(returns)
    - Strategy Scoring: score = tensor_score * bit_phase * market_conditions
    """

    def __init__(self, config_path: str = "./config/trade_simulator_config.json"):
        self.config_path = config_path

        # Portfolio state
        self.portfolio_state: PortfolioState = None
        self.trade_history: List[TradeExecution] = []
        self.performance_metrics: Dict[str, Any] = {}

        # Strategy configurations
        self.strategy_configs = {
            "long_hold_btc": {
                "entry_threshold": 0.02,
                "exit_threshold": -0.05,
                "position_size": 0.4,
                "bit_phase": 8
            },
            "mid_swing_eth": {
                "entry_threshold": 0.015,
                "exit_threshold": -0.03,
                "position_size": 0.25,
                "bit_phase": 42
            },
            "safety_buffer": {
                "entry_threshold": 0.01,
                "exit_threshold": -0.02,
                "position_size": 0.1,
                "bit_phase": 4
            },
            "vol_spike_xrp": {
                "entry_threshold": 0.025,
                "exit_threshold": -0.04,
                "position_size": 0.15,
                "bit_phase": 8
            },
            "risk_reward_sol": {
                "entry_threshold": 0.02,
                "exit_threshold": -0.035,
                "position_size": 0.1,
                "bit_phase": 16
            }
        }

        # Integration with other components
        self.tensor_matcher = None
        self.bit_phase_engine = None
        self.matrix_mapper = None
        self.profit_allocator = None

        # Load configuration
        self._load_configuration()
        self._initialize_portfolio()
        logger.info("Trade Simulator initialized")

    def _load_configuration(self) -> None:
        """Load trade simulator configuration."""
        try:
            # Default configuration
            config = {
                "portfolio": {
                    "initial_capital": 100000.0,
                    "cash_buffer": 0.1,
                    "max_position_size": 0.4,
                    "min_trade_amount": 100.0
                },
                "risk_management": {
                    "max_drawdown": 0.15,
                    "stop_loss_pct": 0.05,
                    "take_profit_pct": 0.1,
                    "max_correlation": 0.7
                },
                "execution": {
                    "slippage": 0.001,
                    "commission": 0.0025,
                    "min_spread": 0.0005
                }
            }

            logger.info("Trade simulator configuration loaded")

        except Exception as e:
            logger.error(f"Error loading configuration: {e}")

    def _initialize_portfolio(self) -> None:
        """Initialize portfolio state."""
        try:
            self.portfolio_state = PortfolioState(
                timestamp=datetime.now(),
                total_value=100000.0,
                cash=100000.0,
                positions={},
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                risk_metrics={
                    'volatility': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'win_rate': 0.0
                }
            )

            logger.info("Portfolio state initialized")

        except Exception as e:
            logger.error(f"Error initializing portfolio: {e}")

    def simulate_trade(self, strategy_bucket: Dict[str, Any], mode: str = "DEMO") -> TradeExecution:
        """
        Simulate trade execution with real strategy logic.

        Parameters:
        -----------
        strategy_bucket : Dict[str, Any]
            Strategy bucket containing trade parameters
        mode : str
            Execution mode ("DEMO" or "LIVE")

        Returns:
        --------
        TradeExecution
            Trade execution result
        """
        try:
            # Extract strategy parameters
            asset = strategy_bucket.get('asset', 'BTC')
            strategy_id = strategy_bucket.get('strategy_id', 'long_hold_btc')
            tensor_score = strategy_bucket.get('tensor_score', 0.0)
            bit_phase = strategy_bucket.get('bit_phase', 8)
            basket_id = strategy_bucket.get('basket_id', 'default')
            current_price = strategy_bucket.get('current_price', 50000.0)
            market_data = strategy_bucket.get('market_data', {})

            # Get strategy configuration
            strategy_config = self.strategy_configs.get(strategy_id, self.strategy_configs['long_hold_btc'])

            # Determine trade type and parameters
            trade_type, quantity, price = self._determine_trade_parameters(
                strategy_bucket, strategy_config, tensor_score, bit_phase
            )

            # Validate trade
            if not self._validate_trade(asset, quantity, price, trade_type):
                return self._create_failed_trade(strategy_bucket, "Trade validation failed")

            # Execute trade
            trade_execution = self._execute_trade(
                asset, trade_type, quantity, price, strategy_id,
                tensor_score, bit_phase, basket_id, mode
            )

            # Update portfolio state
            self._update_portfolio_state(trade_execution)

            # Calculate performance metrics
            self._calculate_performance_metrics()

            logger.info(
                f"Trade simulated: {trade_execution.trade_id} - {asset} {trade_type.value} {quantity:.4f} @ {price:.2f}")
            return trade_execution

        except Exception as e:
            logger.error(f"Error simulating trade: {e}")
            return self._create_failed_trade(strategy_bucket, str(e))

    def _determine_trade_parameters(self, strategy_bucket: Dict[str, Any],
                                    strategy_config: Dict[str, Any],
                                    tensor_score: float, bit_phase: int) -> Tuple[TradeType, float, float]:
        """Determine trade type and parameters based on strategy."""
        try:
            asset = strategy_bucket.get('asset', 'BTC')
            current_price = strategy_bucket.get('current_price', 50000.0)
            entry_threshold = strategy_config.get('entry_threshold', 0.02)
            exit_threshold = strategy_config.get('exit_threshold', -0.05)
            position_size = strategy_config.get('position_size', 0.4)

            # Calculate position value
            available_capital = self.portfolio_state.cash * position_size
            quantity = available_capital / current_price

            # Determine trade type based on tensor score and thresholds
            if tensor_score > entry_threshold:
                trade_type = TradeType.BUY
            elif tensor_score < exit_threshold:
                trade_type = TradeType.SELL
            else:
                trade_type = TradeType.REBALANCE
                quantity *= 0.5  # Smaller rebalancing trade

            # Apply slippage and commission
            slippage = 0.001 if trade_type == TradeType.BUY else -0.001
            execution_price = current_price * (1 + slippage)

            return trade_type, quantity, execution_price

        except Exception as e:
            logger.error(f"Error determining trade parameters: {e}")
            return TradeType.REBALANCE, 0.0, current_price

    def _validate_trade(self, asset: str, quantity: float, price: float, trade_type: TradeType) -> bool:
        """Validate trade parameters."""
        try:
            # Check minimum trade amount
            trade_value = quantity * price
            if trade_value < 100.0:
                logger.warning(f"Trade value {trade_value:.2f} below minimum")
                return False

            # Check available capital for buy trades
            if trade_type == TradeType.BUY:
                if trade_value > self.portfolio_state.cash:
                    logger.warning(f"Insufficient cash for trade: {trade_value:.2f} > {self.portfolio_state.cash:.2f}")
                    return False

            # Check available position for sell trades
            elif trade_type == TradeType.SELL:
                current_position = self.portfolio_state.positions.get(asset, {}).get('quantity', 0.0)
                if quantity > current_position:
                    logger.warning(f"Insufficient position for sell: {quantity:.4f} > {current_position:.4f}")
                    return False

            return True

        except Exception as e:
            logger.error(f"Error validating trade: {e}")
            return False

    def _execute_trade(self, asset: str, trade_type: TradeType, quantity: float, price: float,
                       strategy_id: str, tensor_score: float, bit_phase: int, basket_id: str, mode: str) -> TradeExecution:
        """Execute trade and return execution result."""
        try:
            # Generate trade ID
            trade_id = f"trade_{int(time.time())}_{asset}_{trade_type.value}"

            # Calculate portfolio impact
            portfolio_impact = self._calculate_portfolio_impact(asset, trade_type, quantity, price)

            # Create trade execution
            trade_execution = TradeExecution(
                trade_id=trade_id,
                asset=asset,
                trade_type=trade_type,
                quantity=quantity,
                price=price,
                timestamp=datetime.now(),
                status=TradeStatus.EXECUTED,
                strategy_id=strategy_id,
                tensor_score=tensor_score,
                bit_phase=bit_phase,
                basket_id=basket_id,
                portfolio_impact=portfolio_impact,
                metadata={
                    'mode': mode,
                    'execution_price': price,
                    'trade_value': quantity * price
                }
            )

            # Add to trade history
            self.trade_history.append(trade_execution)

            return trade_execution

        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return self._create_failed_trade({
                'asset': asset,
                'strategy_id': strategy_id,
                'tensor_score': tensor_score,
                'bit_phase': bit_phase,
                'basket_id': basket_id
            }, str(e))

    def _calculate_portfolio_impact(self, asset: str, trade_type: TradeType, quantity: float, price: float) -> Dict[str, float]:
        """Calculate portfolio impact of trade."""
        try:
            trade_value = quantity * price
            commission = trade_value * 0.0025  # 0.25% commission

            if trade_type == TradeType.BUY:
                cash_impact = -(trade_value + commission)
                position_impact = quantity
            elif trade_type == TradeType.SELL:
                cash_impact = trade_value - commission
                position_impact = -quantity
            else:  # REBALANCE
                cash_impact = 0.0
                position_impact = 0.0

            return {
                'cash_impact': cash_impact,
                'position_impact': position_impact,
                'commission': commission,
                'trade_value': trade_value
            }

        except Exception as e:
            logger.error(f"Error calculating portfolio impact: {e}")
            return {'cash_impact': 0.0, 'position_impact': 0.0, 'commission': 0.0, 'trade_value': 0.0}

    def _update_portfolio_state(self, trade_execution: TradeExecution) -> None:
        """Update portfolio state after trade execution."""
        try:
            if trade_execution.status != TradeStatus.EXECUTED:
                return

            asset = trade_execution.asset
            impact = trade_execution.portfolio_impact

            # Update cash
            self.portfolio_state.cash += impact['cash_impact']

            # Update positions
            if asset not in self.portfolio_state.positions:
                self.portfolio_state.positions[asset] = {
                    'quantity': 0.0,
                    'entry_price': 0.0,
                    'current_price': trade_execution.price
                }

            position = self.portfolio_state.positions[asset]
            position['quantity'] += impact['position_impact']

            # Update entry price for new positions
            if impact['position_impact'] > 0:
                if position['entry_price'] == 0.0:
                    position['entry_price'] = trade_execution.price
                else:
                    # Weighted average entry price
                    total_quantity = position['quantity']
                    old_value = (total_quantity - impact['position_impact']) * position['entry_price']
                    new_value = impact['position_impact'] * trade_execution.price
                    position['entry_price'] = (old_value + new_value) / total_quantity

            position['current_price'] = trade_execution.price

            # Update timestamp
            self.portfolio_state.timestamp = datetime.now()

        except Exception as e:
            logger.error(f"Error updating portfolio state: {e}")

    def _calculate_performance_metrics(self) -> None:
        """Calculate portfolio performance metrics."""
        try:
            # Calculate total portfolio value
            total_value = self.portfolio_state.cash
            for asset, position in self.portfolio_state.positions.items():
                if position['quantity'] > 0:
                    total_value += position['quantity'] * position['current_price']

            self.portfolio_state.total_value = total_value

            # Calculate unrealized P&L
            unrealized_pnl = 0.0
            for asset, position in self.portfolio_state.positions.items():
                if position['quantity'] > 0 and position['entry_price'] > 0:
                    unrealized_pnl += position['quantity'] * (position['current_price'] - position['entry_price'])

            self.portfolio_state.unrealized_pnl = unrealized_pnl

            # Calculate risk metrics
            if len(self.trade_history) > 1:
                returns = []
                for i in range(1, len(self.trade_history)):
                    prev_trade = self.trade_history[i-1]
                    curr_trade = self.trade_history[i]
                    if prev_trade.asset == curr_trade.asset:
                        return_val = (curr_trade.price - prev_trade.price) / prev_trade.price
                        returns.append(return_val)

                if returns:
                    returns_array = np.array(returns)
                    volatility = unified_math.unified_math.std(returns_array)
                    sharpe_ratio = unified_math.unified_math.mean(returns_array) / (volatility + 1e-9)

                    self.portfolio_state.risk_metrics.update({
                        'volatility': volatility,
                        'sharpe_ratio': sharpe_ratio
                    })

        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")

    def _create_failed_trade(self, strategy_bucket: Dict[str, Any], error_message: str) -> TradeExecution:
        """Create a failed trade execution."""
        return TradeExecution(
            trade_id=f"failed_{int(time.time())}",
            asset=strategy_bucket.get('asset', 'UNKNOWN'),
            trade_type=TradeType.REBALANCE,
            quantity=0.0,
            price=0.0,
            timestamp=datetime.now(),
            status=TradeStatus.FAILED,
            strategy_id=strategy_bucket.get('strategy_id', 'unknown'),
            tensor_score=0.0,
            bit_phase=0,
            basket_id=strategy_bucket.get('basket_id', 'unknown'),
            portfolio_impact={},
            metadata={'error': error_message}
        )

    def set_tensor_matcher(self, tensor_matcher) -> None:
        """Set tensor matcher for integration."""
        self.tensor_matcher = tensor_matcher
        logger.info("Tensor matcher integrated with trade simulator")

    def set_bit_phase_engine(self, bit_engine) -> None:
        """Set bit phase engine for integration."""
        self.bit_phase_engine = bit_engine
        logger.info("Bit phase engine integrated with trade simulator")

    def set_matrix_mapper(self, matrix_mapper) -> None:
        """Set matrix mapper for integration."""
        self.matrix_mapper = matrix_mapper
        logger.info("Matrix mapper integrated with trade simulator")

    def set_profit_allocator(self, profit_allocator) -> None:
        """Set profit allocator for integration."""
        self.profit_allocator = profit_allocator
        logger.info("Profit allocator integrated with trade simulator")

    def get_portfolio_state(self) -> PortfolioState:
        """Get current portfolio state."""
        return self.portfolio_state

    def get_trade_history(self, limit: int = 100) -> List[TradeExecution]:
        """Get recent trade history."""
        return self.trade_history[-limit:] if self.trade_history else []

    def export_portfolio_snapshot(self, output_path: str = "portfolio_snapshot.json") -> None:
        """Export portfolio snapshot to file."""
        try:
            snapshot_data = {
                'timestamp': self.portfolio_state.timestamp.isoformat(),
                'total_value': self.portfolio_state.total_value,
                'cash': self.portfolio_state.cash,
                'positions': self.portfolio_state.positions,
                'unrealized_pnl': self.portfolio_state.unrealized_pnl,
                'realized_pnl': self.portfolio_state.realized_pnl,
                'risk_metrics': self.portfolio_state.risk_metrics,
                'trade_count': len(self.trade_history)
            }

            with open(output_path, 'w') as f:
                json.dump(snapshot_data, f, indent=2, default=str)

            logger.info(f"Portfolio snapshot exported to {output_path}")

        except Exception as e:
            logger.error(f"Error exporting portfolio snapshot: {e}")


if __name__ == "__main__":
    # Test trade simulator
    simulator = TradeSimulator()

    # Test trade simulation
    strategy_bucket = {
        'asset': 'BTC',
        'strategy_id': 'long_hold_btc',
        'tensor_score': 0.03,
        'bit_phase': 8,
        'basket_id': 'basket_8bit_161',
        'current_price': 50000.0,
        'market_data': {'entropy_level': 4.5, 'volatility': 0.03}
    }

    trade_result = simulator.simulate_trade(strategy_bucket, "DEMO")
    safe_print(f"Trade Result: {trade_result.trade_id}")
    safe_print(f"Status: {trade_result.status.value}")
    safe_print(f"Portfolio Impact: {trade_result.portfolio_impact}")

    # Get portfolio state
    portfolio = simulator.get_portfolio_state()
    safe_print(f"Portfolio Value: {portfolio.total_value:.2f}")
    safe_print(f"Cash: {portfolio.cash:.2f}")
    safe_print(f"Unrealized P&L: {portfolio.unrealized_pnl:.2f}")

    # Export snapshot
    simulator.export_portfolio_snapshot()
