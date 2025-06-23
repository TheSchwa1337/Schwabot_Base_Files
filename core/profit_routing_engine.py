#!/usr/bin/env python3
"""
Profit Routing Engine - Schwabot UROS v1.0
==========================================

Implements intelligent profit routing and management for trading operations.
Critical for optimizing profit distribution and risk management.
"""

import numpy as np
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from core.type_defs import BitLevel, MatrixPhase, MatrixControllerType

logger = logging.getLogger(__name__)


class RoutingStrategy(Enum):
    """Profit routing strategies."""
    EQUAL_WEIGHT = "equal_weight"
    RISK_ADJUSTED = "risk_adjusted"
    PERFORMANCE_BASED = "performance_based"
    ADAPTIVE = "adaptive"
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"


@dataclass
class ProfitRoute:
    """Represents a profit routing configuration."""
    route_id: str
    strategy: RoutingStrategy
    target_allocation: float  # 0.0 to 1.0
    risk_factor: float  # 0.0 to 1.0
    performance_weight: float  # 0.0 to 1.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProfitTransaction:
    """Represents a profit transaction."""
    transaction_id: str
    amount: float
    source_strategy: str
    target_route: str
    routing_strategy: RoutingStrategy
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = False
    error_message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetric:
    """Represents performance metrics for routing decisions."""
    strategy_id: str
    total_profit: float
    total_loss: float
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    risk_adjusted_return: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ProfitRoutingEngine:
    """
    Implements intelligent profit routing and management for trading operations.
    Optimizes profit distribution based on performance and risk metrics.
    """
    
    def __init__(self):
        """Initialize the profit routing engine."""
        self.profit_routes: Dict[str, ProfitRoute] = {}
        self.profit_transactions: List[ProfitTransaction] = []
        self.performance_metrics: Dict[str, PerformanceMetric] = {}
        self.routing_history: List[Dict[str, Any]] = []
        
        # Routing parameters
        self.default_routing_strategy = RoutingStrategy.RISK_ADJUSTED
        self.min_allocation = 0.05  # 5% minimum allocation
        self.max_allocation = 0.40  # 40% maximum allocation
        self.risk_free_rate = 0.02  # 2% risk-free rate
        self.rebalancing_threshold = 0.1  # 10% threshold for rebalancing
        
        # Performance tracking
        self.total_profit_routed = 0.0
        self.total_loss_routed = 0.0
        self.routing_success_rate = 1.0
        self.optimization_enabled = True
        
        logger.info("Profit Routing Engine initialized")
    
    def create_profit_route(
        self,
        route_id: str,
        strategy: RoutingStrategy,
        target_allocation: float,
        risk_factor: float = 0.5,
        performance_weight: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ProfitRoute:
        """Create a new profit routing configuration."""
        # Validate parameters
        target_allocation = np.clip(target_allocation, self.min_allocation, self.max_allocation)
        risk_factor = np.clip(risk_factor, 0.0, 1.0)
        performance_weight = np.clip(performance_weight, 0.0, 1.0)
        
        route = ProfitRoute(
            route_id=route_id,
            strategy=strategy,
            target_allocation=target_allocation,
            risk_factor=risk_factor,
            performance_weight=performance_weight,
            metadata=metadata or {}
        )
        
        self.profit_routes[route_id] = route
        
        logger.info(f"Created profit route: {route_id} ({strategy.value})")
        return route
    
    def route_profit(
        self,
        amount: float,
        source_strategy: str,
        target_route: str,
        routing_strategy: Optional[RoutingStrategy] = None
    ) -> ProfitTransaction:
        """Route profit to specified target."""
        if target_route not in self.profit_routes:
            logger.error(f"Target route not found: {target_route}")
            return None
        
        route = self.profit_routes[target_route]
        strategy = routing_strategy or route.strategy
        
        # Create transaction
        transaction = ProfitTransaction(
            transaction_id=f"profit_tx_{int(time.time() * 1000)}",
            amount=amount,
            source_strategy=source_strategy,
            target_route=target_route,
            routing_strategy=strategy
        )
        
        try:
            # Apply routing strategy
            adjusted_amount = self._apply_routing_strategy(amount, route, strategy)
            
            # Execute routing
            success = self._execute_routing(adjusted_amount, target_route)
            
            transaction.success = success
            if success:
                self.total_profit_routed += adjusted_amount
                logger.info(f"Successfully routed {adjusted_amount:.2f} profit to {target_route}")
            else:
                transaction.error_message = "Routing execution failed"
                logger.error(f"Failed to route profit to {target_route}")
        
        except Exception as e:
            transaction.success = False
            transaction.error_message = str(e)
            logger.error(f"Profit routing error: {e}")
        
        self.profit_transactions.append(transaction)
        
        # Update routing history
        self.routing_history.append({
            "timestamp": transaction.timestamp,
            "amount": transaction.amount,
            "target_route": target_route,
            "strategy": strategy.value,
            "success": transaction.success
        })
        
        return transaction
    
    def _apply_routing_strategy(
        self, amount: float, route: ProfitRoute, strategy: RoutingStrategy
    ) -> float:
        """Apply routing strategy to adjust amount."""
        if strategy == RoutingStrategy.EQUAL_WEIGHT:
            return amount * route.target_allocation
        
        elif strategy == RoutingStrategy.RISK_ADJUSTED:
            # Adjust based on risk factor
            risk_adjustment = 1.0 - route.risk_factor
            return amount * route.target_allocation * risk_adjustment
        
        elif strategy == RoutingStrategy.PERFORMANCE_BASED:
            # Adjust based on performance weight
            performance_adjustment = route.performance_weight
            return amount * route.target_allocation * performance_adjustment
        
        elif strategy == RoutingStrategy.ADAPTIVE:
            # Adaptive adjustment based on multiple factors
            risk_adjustment = 1.0 - route.risk_factor
            performance_adjustment = route.performance_weight
            adaptive_factor = (risk_adjustment + performance_adjustment) / 2.0
            return amount * route.target_allocation * adaptive_factor
        
        elif strategy == RoutingStrategy.CONSERVATIVE:
            # Conservative routing with reduced allocation
            conservative_factor = 0.7
            return amount * route.target_allocation * conservative_factor
        
        elif strategy == RoutingStrategy.AGGRESSIVE:
            # Aggressive routing with increased allocation
            aggressive_factor = 1.3
            return amount * route.target_allocation * aggressive_factor
        
        else:
            # Default to equal weight
            return amount * route.target_allocation
    
    def _execute_routing(self, amount: float, target_route: str) -> bool:
        """Execute the actual profit routing."""
        try:
            # Simulate routing execution
            # In a real implementation, this would interface with actual systems
            time.sleep(0.001)  # Simulate processing time
            
            # Update route metadata
            route = self.profit_routes[target_route]
            if "total_routed" not in route.metadata:
                route.metadata["total_routed"] = 0.0
            route.metadata["total_routed"] += amount
            route.metadata["last_routing"] = datetime.now()
            
            return True
        
        except Exception as e:
            logger.error(f"Routing execution failed: {e}")
            return False
    
    def update_performance_metrics(
        self,
        strategy_id: str,
        total_profit: float,
        total_loss: float,
        win_rate: float,
        sharpe_ratio: float,
        max_drawdown: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> PerformanceMetric:
        """Update performance metrics for a strategy."""
        # Calculate risk-adjusted return
        total_return = total_profit - total_loss
        risk_adjusted_return = sharpe_ratio * total_return if total_return > 0 else 0.0
        
        metric = PerformanceMetric(
            strategy_id=strategy_id,
            total_profit=total_profit,
            total_loss=total_loss,
            win_rate=win_rate,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            risk_adjusted_return=risk_adjusted_return,
            metadata=metadata or {}
        )
        
        self.performance_metrics[strategy_id] = metric
        
        logger.debug(f"Updated performance metrics for {strategy_id}")
        return metric
    
    def optimize_routing_allocation(self) -> Dict[str, float]:
        """Optimize routing allocation based on performance metrics."""
        if not self.optimization_enabled or not self.performance_metrics:
            return {}
        
        # Calculate performance scores for each strategy
        performance_scores = {}
        total_score = 0.0
        
        for strategy_id, metric in self.performance_metrics.items():
            # Multi-factor performance score
            profit_score = min(1.0, metric.total_profit / 10000.0)  # Normalize
            win_rate_score = metric.win_rate
            sharpe_score = min(1.0, max(0.0, metric.sharpe_ratio / 2.0))  # Normalize
            drawdown_score = max(0.0, 1.0 - metric.max_drawdown)  # Invert drawdown
            
            # Weighted score
            score = (
                0.3 * profit_score +
                0.3 * win_rate_score +
                0.2 * sharpe_score +
                0.2 * drawdown_score
            )
            
            performance_scores[strategy_id] = score
            total_score += score
        
        # Calculate optimal allocations
        optimal_allocations = {}
        
        if total_score > 0:
            for strategy_id, score in performance_scores.items():
                # Proportional allocation based on performance
                allocation = score / total_score
                
                # Apply constraints
                allocation = np.clip(allocation, self.min_allocation, self.max_allocation)
                optimal_allocations[strategy_id] = allocation
        
        # Update route allocations
        for route_id, route in self.profit_routes.items():
            if route_id in optimal_allocations:
                old_allocation = route.target_allocation
                new_allocation = optimal_allocations[route_id]
                
                # Only update if change exceeds threshold
                if abs(new_allocation - old_allocation) > self.rebalancing_threshold:
                    route.target_allocation = new_allocation
                    route.metadata["last_optimization"] = datetime.now()
                    route.metadata["allocation_change"] = new_allocation - old_allocation
                    
                    logger.info(f"Optimized allocation for {route_id}: {old_allocation:.3f} -> {new_allocation:.3f}")
        
        return optimal_allocations
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive routing statistics."""
        total_routes = len(self.profit_routes)
        total_transactions = len(self.profit_transactions)
        successful_transactions = sum(1 for tx in self.profit_transactions if tx.success)
        
        # Strategy distribution
        strategy_counts = {}
        for route in self.profit_routes.values():
            strategy = route.strategy.value
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        # Performance metrics summary
        avg_performance_metrics = {}
        if self.performance_metrics:
            avg_profit = sum(m.total_profit for m in self.performance_metrics.values()) / len(self.performance_metrics)
            avg_loss = sum(m.total_loss for m in self.performance_metrics.values()) / len(self.performance_metrics)
            avg_win_rate = sum(m.win_rate for m in self.performance_metrics.values()) / len(self.performance_metrics)
            avg_sharpe = sum(m.sharpe_ratio for m in self.performance_metrics.values()) / len(self.performance_metrics)
            
            avg_performance_metrics = {
                "average_profit": avg_profit,
                "average_loss": avg_loss,
                "average_win_rate": avg_win_rate,
                "average_sharpe_ratio": avg_sharpe
            }
        
        # Routing efficiency
        routing_efficiency = successful_transactions / max(1, total_transactions)
        
        return {
            "total_routes": total_routes,
            "total_transactions": total_transactions,
            "successful_transactions": successful_transactions,
            "routing_efficiency": routing_efficiency,
            "total_profit_routed": self.total_profit_routed,
            "total_loss_routed": self.total_loss_routed,
            "strategy_distribution": strategy_counts,
            "average_performance_metrics": avg_performance_metrics,
            "optimization_enabled": self.optimization_enabled
        }
    
    def get_routing_recommendations(self) -> List[str]:
        """Get routing recommendations based on analysis."""
        recommendations = []
        stats = self.get_routing_statistics()
        
        # Check routing efficiency
        if stats["routing_efficiency"] < 0.95:
            recommendations.append("Low routing efficiency detected. Review routing configurations.")
        
        # Check profit distribution
        if stats["total_profit_routed"] < stats["total_loss_routed"]:
            recommendations.append("Net loss detected. Consider adjusting risk parameters.")
        
        # Check strategy distribution
        strategy_dist = stats["strategy_distribution"]
        if len(strategy_dist) < 2:
            recommendations.append("Limited strategy diversity. Consider adding more routing strategies.")
        
        # Check performance metrics
        avg_metrics = stats["average_performance_metrics"]
        if avg_metrics.get("average_win_rate", 0) < 0.5:
            recommendations.append("Low win rate detected. Review strategy performance.")
        
        if avg_metrics.get("average_sharpe_ratio", 0) < 1.0:
            recommendations.append("Low Sharpe ratio detected. Consider risk-adjusted optimization.")
        
        return recommendations
    
    def get_trading_signals(self) -> List[Dict[str, Any]]:
        """Generate trading signals based on routing analysis."""
        signals = []
        
        # Generate signals from performance metrics
        for strategy_id, metric in self.performance_metrics.items():
            # High performance signal
            if metric.sharpe_ratio > 2.0 and metric.win_rate > 0.6:
                signal = {
                    "type": "high_performance",
                    "strategy_id": strategy_id,
                    "sharpe_ratio": metric.sharpe_ratio,
                    "win_rate": metric.win_rate,
                    "confidence": min(1.0, metric.sharpe_ratio / 3.0),
                    "strength": min(1.0, metric.win_rate),
                    "timestamp": metric.timestamp,
                    "metadata": metric.metadata
                }
                signals.append(signal)
            
            # Risk warning signal
            if metric.max_drawdown > 0.2:  # 20% drawdown
                signal = {
                    "type": "risk_warning",
                    "strategy_id": strategy_id,
                    "max_drawdown": metric.max_drawdown,
                    "confidence": min(1.0, metric.max_drawdown),
                    "strength": min(1.0, metric.max_drawdown),
                    "timestamp": metric.timestamp,
                    "metadata": metric.metadata
                }
                signals.append(signal)
        
        # Generate signals from routing statistics
        stats = self.get_routing_statistics()
        
        # High efficiency signal
        if stats["routing_efficiency"] > 0.98:
            signal = {
                "type": "high_routing_efficiency",
                "efficiency": stats["routing_efficiency"],
                "confidence": stats["routing_efficiency"],
                "strength": stats["routing_efficiency"],
                "timestamp": datetime.now(),
                "metadata": {"total_transactions": stats["total_transactions"]}
            }
            signals.append(signal)
        
        return signals


def main() -> None:
    """Main function for testing the profit routing engine."""
    # Initialize engine
    engine = ProfitRoutingEngine()
    
    # Create profit routes
    route1 = engine.create_profit_route("conservative", RoutingStrategy.CONSERVATIVE, 0.3, 0.2)
    route2 = engine.create_profit_route("balanced", RoutingStrategy.RISK_ADJUSTED, 0.4, 0.5)
    route3 = engine.create_profit_route("aggressive", RoutingStrategy.AGGRESSIVE, 0.3, 0.8)
    
    # Update performance metrics
    engine.update_performance_metrics("strategy_1", 5000.0, 1000.0, 0.65, 1.8, 0.15)
    engine.update_performance_metrics("strategy_2", 3000.0, 800.0, 0.55, 1.2, 0.12)
    engine.update_performance_metrics("strategy_3", 8000.0, 2000.0, 0.75, 2.5, 0.18)
    
    # Route profits
    engine.route_profit(1000.0, "strategy_1", "conservative")
    engine.route_profit(1500.0, "strategy_2", "balanced")
    engine.route_profit(2000.0, "strategy_3", "aggressive")
    
    # Optimize allocations
    optimal_allocations = engine.optimize_routing_allocation()
    print(f"Optimal allocations: {optimal_allocations}")
    
    # Get statistics
    stats = engine.get_routing_statistics()
    print(f"Routing statistics: {stats}")
    
    # Get recommendations
    recommendations = engine.get_routing_recommendations()
    print(f"Recommendations: {recommendations}")
    
    # Get trading signals
    signals = engine.get_trading_signals()
    print(f"Generated {len(signals)} trading signals")


if __name__ == "__main__":
    main() 