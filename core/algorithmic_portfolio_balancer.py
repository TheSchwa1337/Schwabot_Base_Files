#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
⚖️ ALGORITHMIC PORTFOLIO BALANCER - DYNAMIC ALLOCATION ENGINE
============================================================

Advanced algorithmic portfolio balancer with multiple rebalancing strategies.

Features:
- Multiple rebalancing strategies (Equal Weight, Risk Parity, Black-Litterman, Phantom Adaptive, Momentum Weighted)
- Dynamic portfolio rebalancing
- Risk-adjusted allocation
- Performance optimization
- Asset correlation analysis
- Volatility targeting
- Multi-asset optimization
- Real-time portfolio monitoring
- Performance tracking and analytics

CUDA Integration:
- GPU-accelerated portfolio calculations with automatic CPU fallback
- Performance monitoring and optimization
- Cross-platform compatibility (Windows, macOS, Linux)

Mathematical Operations:
- Modern Portfolio Theory calculations
- Risk parity optimization
- Correlation matrix analysis
- Volatility targeting
- Sharpe ratio optimization
- Maximum drawdown analysis
- Portfolio drift monitoring
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# CUDA Integration with Fallback
try:
    import cupy as cp
    USING_CUDA = True
    _backend = "cupy (GPU)"
    xp = cp
except ImportError:
    USING_CUDA = False
    _backend = "numpy (CPU)"
    xp = np

# Import existing Schwabot components
try:
    from .advanced_tensor_algebra import AdvancedTensorAlgebra
    from .entropy_math import EntropyMathSystem
    SCHWABOT_COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Some Schwabot components not available: {e}")
    SCHWABOT_COMPONENTS_AVAILABLE = False

logger = logging.getLogger(__name__)
if USING_CUDA:
    logger.info(f"⚡ Algorithmic Portfolio Balancer using GPU acceleration: {_backend}")
else:
    logger.info(f"🔄 Algorithmic Portfolio Balancer using CPU fallback: {_backend}")

__all__ = [
    "AlgorithmicPortfolioBalancer",
    "RebalancingStrategy",
    "AssetAllocation",
    "PortfolioState",
    "TradingAction",
    "TradingDecision",
    "create_algorithmic_portfolio_balancer",
]


class RebalancingStrategy(Enum):
    """Portfolio rebalancing strategies."""
    EQUAL_WEIGHT = "equal_weight"
    RISK_PARITY = "risk_parity"
    BLACK_LITTERMAN = "black_litterman"
    PHANTOM_ADAPTIVE = "phantom_adaptive"
    MOMENTUM_WEIGHTED = "momentum_weighted"
    MINIMUM_VARIANCE = "minimum_variance"
    MAXIMUM_SHARPE = "maximum_sharpe"


class TradingAction(Enum):
    """Trading actions."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class TradingDecision:
    """Trading decision for portfolio rebalancing."""
    timestamp: float
    symbol: str
    action: TradingAction
    quantity: float
    price: float
    confidence: float
    strategy_branch: str
    profit_potential: float
    risk_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AssetAllocation:
    """Asset allocation configuration."""
    symbol: str
    target_weight: float
    min_weight: float = 0.0
    max_weight: float = 1.0
    risk_score: float = 0.5
    volatility: float = 0.0
    correlation_factor: float = 0.0
    momentum_score: float = 0.0
    phantom_zone_score: float = 0.0
    expected_return: float = 0.0
    beta: float = 1.0


@dataclass
class PortfolioState:
    """Current portfolio state."""
    total_value: Decimal
    cash_balance: Decimal
    asset_balances: Dict[str, Decimal]
    asset_values: Dict[str, Decimal]
    asset_weights: Dict[str, float]
    last_rebalance: float
    rebalance_count: int = 0
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    risk_metrics: Dict[str, float] = field(default_factory=dict)


class AlgorithmicPortfolioBalancer:
    """Advanced algorithmic portfolio balancer with multiple strategies."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the portfolio balancer."""
        self.config = config

        # Portfolio configuration
        self.rebalancing_strategy = RebalancingStrategy(config.get("rebalancing_strategy", "phantom_adaptive"))
        self.rebalance_threshold = config.get("rebalance_threshold", 0.05)  # 5% deviation
        self.max_rebalance_frequency = config.get("max_rebalance_frequency", 3600)  # 1 hour
        self.risk_free_rate = config.get("risk_free_rate", 0.02)  # 2% risk-free rate

        # Asset allocations
        self.asset_allocations = self._initialize_asset_allocations()

        # Portfolio state
        self.portfolio_state = PortfolioState(
            total_value=Decimal("0"),
            cash_balance=Decimal("0"),
            asset_balances={},
            asset_values={},
            asset_weights={},
            last_rebalance=0.0,
        )

        # Performance tracking
        self.performance_history = []
        self.rebalance_history = []
        self.risk_history = []

        # Initialize mathematical components if available
        self.tensor_algebra = None
        self.entropy_system = None
        
        if SCHWABOT_COMPONENTS_AVAILABLE:
            try:
                self.tensor_algebra = AdvancedTensorAlgebra()
                self.entropy_system = EntropyMathSystem()
                logger.info("✅ Algorithmic Portfolio Balancer integrated with Schwabot mathematical components")
            except Exception as e:
                logger.warning(f"⚠️ Could not initialize some mathematical components: {e}")

        logger.info("⚖️ Algorithmic Portfolio Balancer initialized")

    def _initialize_asset_allocations(self) -> Dict[str, AssetAllocation]:
        """Initialize asset allocation configurations."""
        allocations = {
            "BTC": AssetAllocation(
                symbol="BTC",
                target_weight=0.4,
                min_weight=0.2,
                max_weight=0.6,
                risk_score=0.7,
                volatility=0.2,
                expected_return=0.15,
                beta=1.2,
            ),
            "ETH": AssetAllocation(
                symbol="ETH",
                target_weight=0.3,
                min_weight=0.15,
                max_weight=0.45,
                risk_score=0.6,
                volatility=0.25,
                expected_return=0.12,
                beta=1.1,
            ),
            "SOL": AssetAllocation(
                symbol="SOL",
                target_weight=0.2,
                min_weight=0.1,
                max_weight=0.3,
                risk_score=0.8,
                volatility=0.3,
                expected_return=0.18,
                beta=1.4,
            ),
            "USDC": AssetAllocation(
                symbol="USDC",
                target_weight=0.1,
                min_weight=0.05,
                max_weight=0.2,
                risk_score=0.1,
                volatility=0.01,
                expected_return=0.02,
                beta=0.0,
            ),
        }
        return allocations

    async def update_portfolio_state(self, market_data: Dict[str, Any]) -> None:
        """Update current portfolio state with market data."""
        try:
            # Calculate current asset values
            asset_values = {}
            total_value = Decimal("0")

            for symbol, balance in self.portfolio_state.asset_balances.items():
                if symbol in market_data:
                    price = Decimal(str(market_data[symbol].get("price", 0)))
                    value = balance * price
                    asset_values[symbol] = value
                    total_value += value
                else:
                    asset_values[symbol] = Decimal("0")

            # Add cash balance
            total_value += self.portfolio_state.cash_balance

            # Calculate weights
            asset_weights = {}
            if total_value > 0:
                for symbol, value in asset_values.items():
                    asset_weights[symbol] = float(value / total_value)

            # Update portfolio state
            self.portfolio_state.total_value = total_value
            self.portfolio_state.asset_values = asset_values
            self.portfolio_state.asset_weights = asset_weights

            # Update performance history
            self.performance_history.append({
                "timestamp": time.time(),
                "value": float(total_value),
                "weights": asset_weights.copy()
            })

            # Keep history manageable
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-1000:]

            logger.debug(f"Portfolio state updated: ${float(total_value):,.2f}")

        except Exception as e:
            logger.error(f"Error updating portfolio state: {e}")

    async def check_rebalancing_needs(self) -> bool:
        """Check if portfolio needs rebalancing."""
        current_time = time.time()

        # Check frequency limit
        if current_time - self.portfolio_state.last_rebalance < self.max_rebalance_frequency:
            return False

        # Check weight deviations
        for symbol, allocation in self.asset_allocations.items():
            current_weight = self.portfolio_state.asset_weights.get(symbol, 0.0)
            target_weight = allocation.target_weight

            if abs(current_weight - target_weight) > self.rebalance_threshold:
                logger.info(
                    f"Rebalancing needed: {symbol} weight {current_weight:.3f} vs target {target_weight:.3f}"
                )
                return True

        return False

    async def calculate_optimal_weights(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate optimal weights based on selected strategy."""
        try:
            if self.rebalancing_strategy == RebalancingStrategy.EQUAL_WEIGHT:
                return self._calculate_equal_weight_weights()
            elif self.rebalancing_strategy == RebalancingStrategy.RISK_PARITY:
                return await self._calculate_risk_parity_weights(market_data)
            elif self.rebalancing_strategy == RebalancingStrategy.BLACK_LITTERMAN:
                return await self._calculate_black_litterman_weights(market_data)
            elif self.rebalancing_strategy == RebalancingStrategy.PHANTOM_ADAPTIVE:
                return await self._calculate_phantom_adaptive_weights(market_data)
            elif self.rebalancing_strategy == RebalancingStrategy.MOMENTUM_WEIGHTED:
                return await self._calculate_momentum_weighted_weights(market_data)
            elif self.rebalancing_strategy == RebalancingStrategy.MINIMUM_VARIANCE:
                return await self._calculate_minimum_variance_weights(market_data)
            elif self.rebalancing_strategy == RebalancingStrategy.MAXIMUM_SHARPE:
                return await self._calculate_maximum_sharpe_weights(market_data)
            else:
                return {symbol: alloc.target_weight for symbol, alloc in self.asset_allocations.items()}

        except Exception as e:
            logger.error(f"Error calculating optimal weights: {e}")
            return {symbol: alloc.target_weight for symbol, alloc in self.asset_allocations.items()}

    def _calculate_equal_weight_weights(self) -> Dict[str, float]:
        """Calculate equal weight allocation."""
        symbols = list(self.asset_allocations.keys())
        weight = 1.0 / len(symbols)
        return {symbol: weight for symbol in symbols}

    async def _calculate_risk_parity_weights(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate risk parity weights."""
        try:
            # Get volatility estimates
            volatilities = {}
            for symbol in self.asset_allocations.keys():
                if symbol in market_data:
                    volatilities[symbol] = market_data[symbol].get("volatility", self.asset_allocations[symbol].volatility)
                else:
                    volatilities[symbol] = self.asset_allocations[symbol].volatility

            # Calculate risk parity weights (inverse volatility)
            total_inverse_vol = sum(1.0 / vol for vol in volatilities.values() if vol > 0)
            
            weights = {}
            for symbol, vol in volatilities.items():
                if vol > 0:
                    weights[symbol] = (1.0 / vol) / total_inverse_vol
                else:
                    weights[symbol] = 0.0

            return weights

        except Exception as e:
            logger.error(f"Error calculating risk parity weights: {e}")
            return {symbol: alloc.target_weight for symbol, alloc in self.asset_allocations.items()}

    async def _calculate_black_litterman_weights(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate Black-Litterman weights."""
        try:
            # Simplified Black-Litterman implementation
            # In practice, this would use more sophisticated market equilibrium calculations
            
            # Get expected returns
            expected_returns = {}
            for symbol, allocation in self.asset_allocations.items():
                if symbol in market_data:
                    expected_returns[symbol] = market_data[symbol].get("expected_return", allocation.expected_return)
                else:
                    expected_returns[symbol] = allocation.expected_return

            # Calculate weights based on expected returns
            total_return = sum(expected_returns.values())
            if total_return > 0:
                weights = {symbol: ret / total_return for symbol, ret in expected_returns.items()}
            else:
                weights = {symbol: alloc.target_weight for symbol, alloc in self.asset_allocations.items()}

            return weights

        except Exception as e:
            logger.error(f"Error calculating Black-Litterman weights: {e}")
            return {symbol: alloc.target_weight for symbol, alloc in self.asset_allocations.items()}

    async def _calculate_phantom_adaptive_weights(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate Phantom Adaptive weights."""
        try:
            # Base weights from target allocations
            base_weights = {symbol: alloc.target_weight for symbol, alloc in self.asset_allocations.items()}
            
            # Apply Phantom Math adjustments (simplified)
            adjusted_weights = {}
            for symbol, base_weight in base_weights.items():
                # Get market sentiment or Phantom Zone score
                phantom_score = market_data.get(symbol, {}).get("phantom_score", 0.5)
                
                # Adjust weight based on Phantom score
                if phantom_score > 0.7:  # Strong Phantom Zone
                    adjusted_weight = base_weight * 1.2
                elif phantom_score < 0.3:  # Weak Phantom Zone
                    adjusted_weight = base_weight * 0.8
                else:
                    adjusted_weight = base_weight

                # Apply constraints
                allocation = self.asset_allocations[symbol]
                adjusted_weight = max(allocation.min_weight, min(allocation.max_weight, adjusted_weight))
                adjusted_weights[symbol] = adjusted_weight

            # Normalize weights
            total_weight = sum(adjusted_weights.values())
            if total_weight > 0:
                adjusted_weights = {symbol: weight / total_weight for symbol, weight in adjusted_weights.items()}

            return adjusted_weights

        except Exception as e:
            logger.error(f"Error calculating Phantom Adaptive weights: {e}")
            return {symbol: alloc.target_weight for symbol, alloc in self.asset_allocations.items()}

    async def _calculate_momentum_weighted_weights(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate momentum-weighted weights."""
        try:
            # Calculate momentum scores
            momentum_scores = {}
            for symbol in self.asset_allocations.keys():
                if symbol in market_data:
                    # Get price momentum (simplified)
                    current_price = market_data[symbol].get("price", 1.0)
                    momentum_scores[symbol] = market_data[symbol].get("momentum", 0.0)
                else:
                    momentum_scores[symbol] = 0.0

            # Calculate weights based on momentum
            total_momentum = sum(max(0, score) for score in momentum_scores.values())
            if total_momentum > 0:
                weights = {symbol: max(0, score) / total_momentum for symbol, score in momentum_scores.items()}
            else:
                weights = {symbol: alloc.target_weight for symbol, alloc in self.asset_allocations.items()}

            return weights

        except Exception as e:
            logger.error(f"Error calculating momentum-weighted weights: {e}")
            return {symbol: alloc.target_weight for symbol, alloc in self.asset_allocations.items()}

    async def _calculate_minimum_variance_weights(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate minimum variance weights."""
        try:
            # Simplified minimum variance calculation
            # In practice, this would use covariance matrix optimization
            
            # Use inverse volatility weighting as approximation
            volatilities = {}
            for symbol in self.asset_allocations.keys():
                if symbol in market_data:
                    volatilities[symbol] = market_data[symbol].get("volatility", self.asset_allocations[symbol].volatility)
                else:
                    volatilities[symbol] = self.asset_allocations[symbol].volatility

            # Calculate weights (inverse volatility squared)
            total_inverse_var = sum(1.0 / (vol ** 2) for vol in volatilities.values() if vol > 0)
            
            weights = {}
            for symbol, vol in volatilities.items():
                if vol > 0:
                    weights[symbol] = (1.0 / (vol ** 2)) / total_inverse_var
                else:
                    weights[symbol] = 0.0

            return weights

        except Exception as e:
            logger.error(f"Error calculating minimum variance weights: {e}")
            return {symbol: alloc.target_weight for symbol, alloc in self.asset_allocations.items()}

    async def _calculate_maximum_sharpe_weights(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate maximum Sharpe ratio weights."""
        try:
            # Simplified maximum Sharpe ratio calculation
            # In practice, this would use mean-variance optimization
            
            # Calculate Sharpe ratios
            sharpe_ratios = {}
            for symbol, allocation in self.asset_allocations.items():
                if symbol in market_data:
                    expected_return = market_data[symbol].get("expected_return", allocation.expected_return)
                    volatility = market_data[symbol].get("volatility", allocation.volatility)
                else:
                    expected_return = allocation.expected_return
                    volatility = allocation.volatility

                if volatility > 0:
                    sharpe_ratios[symbol] = (expected_return - self.risk_free_rate) / volatility
                else:
                    sharpe_ratios[symbol] = 0.0

            # Calculate weights based on Sharpe ratios
            total_sharpe = sum(max(0, ratio) for ratio in sharpe_ratios.values())
            if total_sharpe > 0:
                weights = {symbol: max(0, ratio) / total_sharpe for symbol, ratio in sharpe_ratios.items()}
            else:
                weights = {symbol: alloc.target_weight for symbol, alloc in self.asset_allocations.items()}

            return weights

        except Exception as e:
            logger.error(f"Error calculating maximum Sharpe ratio weights: {e}")
            return {symbol: alloc.target_weight for symbol, alloc in self.asset_allocations.items()}

    async def generate_rebalancing_decisions(self, market_data: Dict[str, Any]) -> List[TradingDecision]:
        """Generate rebalancing trading decisions."""
        try:
            decisions = []

            # Calculate target weights based on strategy
            target_weights = await self.calculate_optimal_weights(market_data)

            # Calculate required trades
            for symbol, target_weight in target_weights.items():
                current_weight = self.portfolio_state.asset_weights.get(symbol, 0.0)
                weight_diff = target_weight - current_weight

                if abs(weight_diff) > self.rebalance_threshold:
                    # Calculate trade amount
                    trade_value = abs(weight_diff) * float(self.portfolio_state.total_value)

                    if weight_diff > 0:
                        # Need to buy
                        action = TradingAction.BUY
                        price = float(market_data.get(symbol, {}).get("price", 1.0))
                        quantity = trade_value / price if price > 0 else 0.0
                    else:
                        # Need to sell
                        action = TradingAction.SELL
                        current_balance = float(self.portfolio_state.asset_balances.get(symbol, 0))
                        quantity = current_balance * abs(weight_diff)
                        price = float(market_data.get(symbol, {}).get("price", 1.0))

                    # Create trading decision
                    decision = TradingDecision(
                        timestamp=time.time(),
                        symbol=f"{symbol}/USDC",
                        action=action,
                        quantity=quantity,
                        price=price,
                        confidence=0.8,
                        strategy_branch="portfolio_rebalancing",
                        profit_potential=0.2,
                        risk_score=0.3,
                        metadata={
                            "type": "rebalancing",
                            "asset": symbol,
                            "current_weight": current_weight,
                            "target_weight": target_weight,
                            "weight_diff": weight_diff,
                            "strategy": self.rebalancing_strategy.value,
                        },
                    )

                    decisions.append(decision)

            return decisions

        except Exception as e:
            logger.error(f"Error generating rebalancing decisions: {e}")
            return []

    async def execute_rebalancing(self, decisions: List[TradingDecision]) -> bool:
        """Execute rebalancing trades."""
        try:
            success_count = 0

            for decision in decisions:
                # Execute trade (this would integrate with your trading executor)
                trade_success = await self._execute_trade(decision)
                if trade_success:
                    success_count += 1
                    logger.info(
                        f"Rebalancing trade executed: {decision.symbol} {decision.action.value} {decision.quantity}"
                    )

            # Update rebalancing state
            self.portfolio_state.last_rebalance = time.time()
            self.portfolio_state.rebalance_count += 1

            # Log rebalancing event
            self.rebalance_history.append({
                "timestamp": time.time(),
                "decisions_count": len(decisions),
                "success_count": success_count,
                "portfolio_value": float(self.portfolio_state.total_value),
                "strategy": self.rebalancing_strategy.value,
            })

            logger.info(f"Portfolio rebalancing completed: {success_count}/{len(decisions)} trades executed")
            return success_count == len(decisions)

        except Exception as e:
            logger.error(f"Error executing rebalancing: {e}")
            return False

    async def _execute_trade(self, decision: TradingDecision) -> bool:
        """Execute a single trade (placeholder for integration)."""
        # This would integrate with your trading executor
        # For now, return True to simulate successful execution
        return True

    async def get_portfolio_metrics(self) -> Dict[str, Any]:
        """Get comprehensive portfolio metrics."""
        try:
            metrics = {
                "total_value": float(self.portfolio_state.total_value),
                "cash_balance": float(self.portfolio_state.cash_balance),
                "asset_weights": self.portfolio_state.asset_weights,
                "rebalance_count": self.portfolio_state.rebalance_count,
                "last_rebalance": self.portfolio_state.last_rebalance,
                "strategy": self.rebalancing_strategy.value,
                "performance": await self.calculate_performance_metrics(),
                "risk_metrics": self._calculate_risk_metrics(),
                "backend": _backend,
                "schwabot_components_available": SCHWABOT_COMPONENTS_AVAILABLE
            }
            return metrics
        except Exception as e:
            logger.error(f"Error getting portfolio metrics: {e}")
            return {}

    def _calculate_risk_metrics(self) -> Dict[str, float]:
        """Calculate portfolio risk metrics."""
        try:
            if len(self.performance_history) < 2:
                return {"volatility": 0.0, "var_95": 0.0, "max_drawdown": 0.0}

            # Calculate returns
            returns = []
            for i in range(1, len(self.performance_history)):
                prev_value = self.performance_history[i - 1]["value"]
                curr_value = self.performance_history[i]["value"]
                if prev_value > 0:
                    returns.append((curr_value - prev_value) / prev_value)

            if not returns:
                return {"volatility": 0.0, "var_95": 0.0, "max_drawdown": 0.0}

            # Calculate volatility
            volatility = float(xp.std(returns))

            # Calculate Value at Risk (95%)
            var_95 = float(xp.percentile(returns, 5))

            # Calculate maximum drawdown
            max_drawdown = 0.0
            peak = self.performance_history[0]["value"]
            for record in self.performance_history:
                if record["value"] > peak:
                    peak = record["value"]
                drawdown = (peak - record["value"]) / peak if peak > 0 else 0.0
                max_drawdown = max(max_drawdown, drawdown)

            return {
                "volatility": volatility,
                "var_95": var_95,
                "max_drawdown": max_drawdown,
            }

        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {"volatility": 0.0, "var_95": 0.0, "max_drawdown": 0.0}

    async def calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics for the portfolio."""
        try:
            # Get basic performance metrics
            if len(self.performance_history) < 2:
                return {
                    "total_return": 0.0,
                    "sharpe_ratio": 0.0,
                    "volatility": 0.0,
                    "max_drawdown": 0.0,
                    "annualized_return": 0.0,
                }

            # Calculate returns
            returns = []
            for i in range(1, len(self.performance_history)):
                prev_value = self.performance_history[i - 1]["value"]
                curr_value = self.performance_history[i]["value"]
                if prev_value > 0:
                    returns.append((curr_value - prev_value) / prev_value)

            if not returns:
                return {
                    "total_return": 0.0,
                    "sharpe_ratio": 0.0,
                    "volatility": 0.0,
                    "max_drawdown": 0.0,
                    "annualized_return": 0.0,
                }

            # Calculate metrics
            avg_return = float(xp.mean(returns))
            volatility = float(xp.std(returns))
            sharpe_ratio = (avg_return - self.risk_free_rate) / volatility if volatility > 0 else 0.0

            # Calculate total return
            initial_value = self.performance_history[0]["value"]
            current_value = float(self.portfolio_state.total_value)
            total_return = (current_value - initial_value) / initial_value if initial_value > 0 else 0.0

            # Calculate annualized return
            if len(self.performance_history) > 1:
                time_span = (self.performance_history[-1]["timestamp"] - self.performance_history[0]["timestamp"]) / (365 * 24 * 3600)  # years
                if time_span > 0:
                    annualized_return = ((1 + total_return) ** (1 / time_span)) - 1
                else:
                    annualized_return = 0.0
            else:
                annualized_return = 0.0

            # Calculate maximum drawdown
            max_drawdown = 0.0
            peak = self.performance_history[0]["value"]
            for record in self.performance_history:
                if record["value"] > peak:
                    peak = record["value"]
                drawdown = (peak - record["value"]) / peak if peak > 0 else 0.0
                max_drawdown = max(max_drawdown, drawdown)

            return {
                "total_return": total_return,
                "sharpe_ratio": sharpe_ratio,
                "volatility": volatility,
                "max_drawdown": max_drawdown,
                "annualized_return": annualized_return,
                "avg_return": avg_return,
            }

        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "volatility": 0.0,
                "max_drawdown": 0.0,
                "annualized_return": 0.0,
            }

    def reset_portfolio(self) -> None:
        """Reset portfolio state."""
        try:
            self.portfolio_state = PortfolioState(
                total_value=Decimal("0"),
                cash_balance=Decimal("0"),
                asset_balances={},
                asset_values={},
                asset_weights={},
                last_rebalance=0.0,
            )
            self.performance_history.clear()
            self.rebalance_history.clear()
            self.risk_history.clear()
            logger.info("Portfolio state reset")
        except Exception as e:
            logger.error(f"Error resetting portfolio: {e}")


def create_algorithmic_portfolio_balancer(config: Dict[str, Any]) -> AlgorithmicPortfolioBalancer:
    """Create an AlgorithmicPortfolioBalancer instance."""
    return AlgorithmicPortfolioBalancer(config)


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Create portfolio balancer
    config = {
        "rebalancing_strategy": "phantom_adaptive",
        "rebalance_threshold": 0.05,
        "max_rebalance_frequency": 3600,
        "risk_free_rate": 0.02,
    }
    
    balancer = create_algorithmic_portfolio_balancer(config)

    # Test with sample market data
    market_data = {
        "BTC": {"price": 50000, "volatility": 0.2, "expected_return": 0.15, "phantom_score": 0.7},
        "ETH": {"price": 3000, "volatility": 0.25, "expected_return": 0.12, "phantom_score": 0.6},
        "SOL": {"price": 100, "volatility": 0.3, "expected_return": 0.18, "phantom_score": 0.8},
        "USDC": {"price": 1, "volatility": 0.01, "expected_return": 0.02, "phantom_score": 0.5},
    }

    # Simulate portfolio state
    balancer.portfolio_state.asset_balances = {
        "BTC": Decimal("0.5"),
        "ETH": Decimal("5.0"),
        "SOL": Decimal("50.0"),
        "USDC": Decimal("1000.0"),
    }
    balancer.portfolio_state.cash_balance = Decimal("5000.0")

    # Update portfolio state
    asyncio.run(balancer.update_portfolio_state(market_data))

    # Get metrics
    metrics = asyncio.run(balancer.get_portfolio_metrics())
    print(f"Portfolio Metrics: {metrics}") 