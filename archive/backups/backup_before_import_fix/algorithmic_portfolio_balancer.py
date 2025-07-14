#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Algorithmic Portfolio Balancer - Mathematical Implementation
============================================================

Advanced portfolio balancing system that uses mathematical algorithms
to optimize asset allocation and rebalancing strategies.
"""

import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List

import numpy as np

from .clean_trading_pipeline import TradingAction, TradingDecision
from .phantom_detector import PhantomZone
from .phantom_logger import PhantomLogger
from .phantom_registry import PhantomRegistry

# Mathematical portfolio balancing algorithms
# Implementation follows modern portfolio theory with enhanced mathematical models

logger = logging.getLogger(__name__)


class RebalancingStrategy(Enum):
    """Portfolio rebalancing strategies."""

    EQUAL_WEIGHT = "equal_weight"
    RISK_PARITY = "risk_parity"
    BLACK_LITTERMAN = "black_litterman"
    PHANTOM_ADAPTIVE = "phantom_adaptive"
    MOMENTUM_WEIGHTED = "momentum_weighted"


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


class AlgorithmicPortfolioBalancer:
    """Advanced algorithmic portfolio balancer."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.phantom_registry = PhantomRegistry()
        self.phantom_logger = PhantomLogger()

        # Portfolio configuration
        self.rebalancing_strategy = RebalancingStrategy(config.get("rebalancing_strategy", "phantom_adaptive"))
        self.rebalance_threshold = config.get("max_rebalance_frequency", 3600)  # 1 hour

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

        logger.info("Algorithmic Portfolio Balancer initialized")

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
            ),
            "ETH": AssetAllocation(
                symbol="ETH",
                target_weight=0.3,
                min_weight=0.15,
                max_weight=0.45,
                risk_score=0.6,
                volatility=0.25,
            ),
            "SOL": AssetAllocation(
                symbol="SOL",
                target_weight=0.2,
                min_weight=0.1,
                max_weight=0.3,
                risk_score=0.8,
                volatility=0.3,
            ),
            "USDC": AssetAllocation(
                symbol="USDC",
                target_weight=0.1,
                min_weight=0.5,
                max_weight=0.2,
                risk_score=0.1,
                volatility=0.01,
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

            logger.debug("Portfolio state updated: ${0:,.2f}".format(total_value))

        except Exception as e:
            logger.error("Error updating portfolio state: {0}".format(e))

    async def check_rebalancing_needs(self) -> bool:
        """Check if portfolio needs rebalancing."""
        current_time = time.time()

        # Check frequency limit
        if current_time - self.portfolio_state.last_rebalance < self.rebalance_threshold:
            return False

        # Check weight deviations
        for symbol, allocation in self.asset_allocations.items():
            current_weight = self.portfolio_state.asset_weights.get(symbol, 0.0)
            target_weight = allocation.target_weight

            if abs(current_weight - target_weight) > self.rebalance_threshold:
                logger.info(
                    "Rebalancing needed: {0} weight {1:.3f} vs target {2:.3f}".format(
                        symbol, current_weight, target_weight
                    )
                )
                return True

        return False

    async def calculate_phantom_adjusted_weights(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate weights adjusted by Phantom Math detection."""
        try:
            # Get Phantom Zone data
            phantom_zones = await self._get_phantom_zones(market_data)

            # Calculate adjusted weights
            adjusted_weights = {}
            base_weights = {symbol: alloc.target_weight for symbol, alloc in self.asset_allocations.items()}

            for symbol, base_weight in base_weights.items():
                # Get Phantom Zone score for this asset
                phantom_score = self._get_asset_phantom_score(symbol, phantom_zones)

                # Adjust weight based on Phantom Math
                if phantom_score > 0.7:  # Strong Phantom Zone
                    adjusted_weight = base_weight * 1.2  # Increase allocation
                elif phantom_score < 0.3:  # Weak Phantom Zone
                    adjusted_weight = base_weight * 0.8  # Decrease allocation
                else:
                    adjusted_weight = base_weight

                # Apply constraints
                allocation = self.asset_allocations[symbol]
                adjusted_weight = max(allocation.min_weight, min(allocation.max_weight, adjusted_weight))

                adjusted_weights[symbol] = adjusted_weight

            # Normalize weights
            total_weight = sum(adjusted_weights.values())
            if total_weight > 0:
                adjusted_weights = {k: v / total_weight for k, v in adjusted_weights.items()}

            return adjusted_weights

        except Exception as e:
            logger.error("Error calculating Phantom-adjusted weights: {0}".format(e))
            return {}

    async def generate_rebalancing_decisions(self, market_data: Dict[str, Any]) -> List[TradingDecision]:
        """Generate rebalancing trading decisions."""
        try:
            decisions = []

            # Calculate target weights based on strategy
            if self.rebalancing_strategy == RebalancingStrategy.PHANTOM_ADAPTIVE:
                target_weights = await self.calculate_phantom_adjusted_weights(market_data)
            else:
                target_weights = {symbol: alloc.target_weight for symbol, alloc in self.asset_allocations.items()}

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
                        quantity = trade_value / float(market_data[symbol].get("price", 1))
                    else:
                        # Need to sell
                        quantity = abs(float(self.portfolio_state.asset_balances.get(symbol, 0)) * weight_diff)

                    # Create trading decision
                    decision = TradingDecision(
                        timestamp=time.time(),
                        symbol="{0}/USDC".format(symbol),
                        action=action,
                        quantity=quantity,
                        price=float(market_data[symbol].get("price", 0)),
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
            logger.error("Error generating rebalancing decisions: {0}".format(e))
            return []

    async def execute_rebalancing(self, decisions: List[TradingDecision]) -> bool:
        """Execute rebalancing trades."""
        try:
            success_count = 0

            for decision in decisions:
                # Log the rebalancing decision
                await self.phantom_logger.log_phantom_zone(
                    PhantomZone(
                        timestamp=decision.timestamp,
                        symbol=decision.symbol,
                        entropy_score=0.5,
                        flatness_score=0.5,
                        similarity_score=0.5,
                        potential_score=0.5,
                        zone_type="rebalancing",
                        metadata=decision.metadata,
                    )
                )

                # Execute trade (this would integrate with your trading executor)
                # For now, we'll simulate execution
                trade_success = await self._execute_trade(decision)
                if trade_success:
                    success_count += 1
                    logger.info(
                        "Rebalancing trade executed: {0} {1} {2}".format(
                            decision.symbol, decision.action.value, decision.quantity
                        )
                    )

            # Update rebalancing state
            self.portfolio_state.last_rebalance = time.time()
            self.portfolio_state.rebalance_count += 1

            # Log rebalancing event
            self.rebalance_history.append(
                {
                    "timestamp": time.time(),
                    "decisions_count": len(decisions),
                    "success_count": success_count,
                    "portfolio_value": float(self.portfolio_state.total_value),
                }
            )

            logger.info(
                "Portfolio rebalancing completed: {0}/{1} trades executed".format(success_count, len(decisions))
            )
            return success_count == len(decisions)

        except Exception as e:
            logger.error("Error executing rebalancing: {0}".format(e))
            return False

    async def _execute_trade(self, decision: TradingDecision) -> bool:
        """Execute a single trade (placeholder for integration)."""
        # This would integrate with your CCXT trading executor
        # For now, return True to simulate successful execution
        return True

    async def _get_phantom_zones(self, market_data: Dict[str, Any]) -> List[PhantomZone]:
        """Get Phantom Zone data for assets."""
        try:
            zones = []
            for symbol in self.asset_allocations.keys():
                if symbol in market_data:
                    # Get recent Phantom Zones for this asset
                    recent_zones = await self.phantom_registry.get_recent_zones(symbol, hours=24)
                    zones.extend(recent_zones)
            return zones
        except Exception as e:
            logger.error("Error getting Phantom Zones: {0}".format(e))
            return []

    def _get_asset_phantom_score(self, symbol: str, phantom_zones: List[PhantomZone]) -> float:
        """Calculate Phantom Zone score for an asset."""
        try:
            asset_zones = [zone for zone in phantom_zones if zone.symbol == symbol]
            if not asset_zones:
                return 0.5  # Neutral score

            # Calculate average potential score
            avg_potential = sum(zone.potential_score for zone in asset_zones) / len(asset_zones)
            return avg_potential

        except Exception as e:
            logger.error("Error calculating Phantom score for {0}: {1}".format(symbol, e))
            return 0.5

    async def get_portfolio_metrics(self) -> Dict[str, Any]:
        """Get comprehensive portfolio metrics."""
        try:
            metrics = {
                "total_value": float(self.portfolio_state.total_value),
                "cash_balance": float(self.portfolio_state.cash_balance),
                "asset_weights": self.portfolio_state.asset_weights,
                "rebalance_count": self.portfolio_state.rebalance_count,
                "last_rebalance": self.portfolio_state.last_rebalance,
                "performance": self._calculate_performance_metrics(),
            }
            return metrics
        except Exception as e:
            logger.error("Error getting portfolio metrics: {0}".format(e))

    def _calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate portfolio performance metrics."""
        try:
            if len(self.performance_history) < 2:
                return {"return": 0.0, "volatility": 0.0, "sharpe_ratio": 0.0}

            # Calculate returns
            returns = []
            for i in range(1, len(self.performance_history)):
                prev_value = self.performance_history[i - 1]["value"]
                curr_value = self.performance_history[i]["value"]
                if prev_value > 0:
                    returns.append((curr_value - prev_value) / prev_value)

            if not returns:
                return {"return": 0.0, "volatility": 0.0, "sharpe_ratio": 0.0}

            avg_return = np.mean(returns)
            volatility = np.std(returns)
            sharpe_ratio = avg_return / volatility if volatility > 0 else 0.0

            return {
                "return": avg_return,
                "volatility": volatility,
                "sharpe_ratio": sharpe_ratio,
            }

        except Exception as e:
            logger.error("Error calculating performance metrics: {0}".format(e))
            return {"return": 0.0, "volatility": 0.0, "sharpe_ratio": 0.0}

    async def calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics for the portfolio."""
        try:
            # Get basic performance metrics
            basic_metrics = self._calculate_performance_metrics()

            # Calculate additional metrics
            current_time = time.time()

            # Calculate total return
            if len(self.performance_history) >= 2:
                initial_value = self.performance_history[0]["value"]
                current_value = float(self.portfolio_state.total_value)
                total_return = (current_value - initial_value) / initial_value if initial_value > 0 else 0.0
            else:
                total_return = 0.0

            # Calculate max drawdown
            max_drawdown = 0.0
            if len(self.performance_history) > 1:
                peak = self.performance_history[0]["value"]
                for record in self.performance_history:
                    if record["value"] > peak:
                        peak = record["value"]
                    drawdown = (peak - record["value"]) / peak if peak > 0 else 0.0
                    max_drawdown = max(max_drawdown, drawdown)

            # Calculate drift score
            drift_score = 0.0
            if self.portfolio_state.asset_weights:
                for symbol, allocation in self.asset_allocations.items():
                    current_weight = self.portfolio_state.asset_weights.get(symbol, 0.0)
                    target_weight = allocation.target_weight
                    drift_score += abs(current_weight - target_weight)
                drift_score /= len(self.asset_allocations)

            # Count rebalances today
            today_start = current_time - (current_time % 86400)  # Start of today
            rebalances_today = sum(1 for record in self.rebalance_history if record["timestamp"] >= today_start)

            # Format last rebalance time
            if self.portfolio_state.last_rebalance > 0:
                last_rebalance = time.strftime(
                    "%Y-%m-%d %H:%M:%S",
                    time.localtime(self.portfolio_state.last_rebalance),
                )
            else:
                last_rebalance = "Never"

            return {
                "total_return": total_return,
                "sharpe_ratio": basic_metrics.get("sharpe_ratio", 0.0),
                "volatility": basic_metrics.get("volatility", 0.0),
                "max_drawdown": max_drawdown,
                "last_rebalance": last_rebalance,
                "rebalances_today": rebalances_today,
                "drift_score": drift_score,
                "total_value": float(self.portfolio_state.total_value),
                "cash_balance": float(self.portfolio_state.cash_balance),
                "asset_weights": self.portfolio_state.asset_weights,
            }

        except Exception as e:
            logger.error("Error calculating performance metrics: {0}".format(e))
            return {
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "volatility": 0.0,
                "max_drawdown": 0.0,
                "last_rebalance": "Never",
                "rebalances_today": 0,
                "drift_score": 0.0,
                "total_value": 0.0,
                "cash_balance": 0.0,
                "asset_weights": {},
            }


# Factory function for easy integration
def create_portfolio_balancer(config: Dict[str, Any]) -> AlgorithmicPortfolioBalancer:
    """Create a portfolio balancer instance."""
    return AlgorithmicPortfolioBalancer(config)
