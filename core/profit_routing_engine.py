# Import safe print for Windows compatibility
try:
    from .utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug
except ImportError:
    try:
        from core.utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug
    except ImportError:
        def safe_print(message): print(message)
        def info(message): print(f"[INFO] {message}")
        def warn(message): print(f"[WARN] {message}")
        def error(message): print(f"[ERROR] {message}")
        def success(message): print(f"[SUCCESS] {message}")
        def debug(message): print(f"[DEBUG] {message}")
from core.unified_math_system import unified_math
import numpy as np
#!/usr/bin/env python3
"""
Profit Routing Engine - Schwabot UROS v1.0
=========================================

Central logic handler for profit destination allocation among BTC, USDC, ETH, XRP.
Features:
- Profit Delta Vector: P = (P_BTC, P_USDC, P_XRP, ...)
- Weighted Strategy Entropy Matrix: M_ij = unified_math.log(P_i / P_j + ε)
- Dynamic Rebalancing Logic using eigenvector centrality
- Markov transition probabilities for zone rotation
- Integration with strategy_mapper.py and profit_cycle_allocator.py
"""

from core.unified_math_system import unified_math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)


class RoutingStrategy(Enum):
    """Routing strategy types."""
    CONSERVATIVE = "conservative"
    RISK_ADJUSTED = "risk_adjusted"
    AGGRESSIVE = "aggressive"
    BALANCED = "balanced"


@dataclass
class ProfitRoute:
    """Represents a profit routing configuration."""
    route_id: str
    strategy: RoutingStrategy
    allocation_weights: Dict[str, float]
    risk_tolerance: float
    performance_metrics: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)
    hash_signature: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyPerformance:
    """Represents performance metrics for a trading strategy."""
    strategy_name: str
    total_profit: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    profit_factor: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ProfitRoutingEngine:
    """
    Implements profit routing logic for multi-asset allocation.
    Handles dynamic rebalancing and strategy optimization.
    """
    
    def __init__(self) -> None:
        """Initialize the profit routing engine."""
        self.routes: Dict[str, ProfitRoute] = {}
        self.strategy_performances: Dict[str, StrategyPerformance] = {}
        self.profit_history: List[Dict[str, float]] = []
        self.transition_matrix: Optional[np.ndarray[Any, Any]] = None
        
        # Routing parameters
        self.max_routes = 50
        self.rebalancing_threshold = 0.1
        self.entropy_threshold = 0.5
        self.markov_memory_size = 100
        
        # Asset configuration
        self.supported_assets = ["BTC", "USDC", "ETH", "XRP"]
        self.default_allocation = {
            "BTC": 0.4,
            "USDC": 0.3,
            "ETH": 0.2,
            "XRP": 0.1
        }
        
        # Performance tracking
        self.total_profit_routed = 0.0
        self.routing_efficiency = 0.0
        self.rebalancing_count = 0
        
        logger.info("Profit Routing Engine initialized")
    
    def create_profit_route(
        self,
        route_name: str,
        strategy: RoutingStrategy,
        risk_tolerance: float,
        performance_threshold: float
    ) -> ProfitRoute:
        """Create a new profit routing configuration."""
        route_id = f"route_{int(datetime.now().timestamp())}"
        
        # Calculate allocation weights based on strategy and risk
        allocation_weights = self._calculate_allocation_weights(strategy, risk_tolerance)
        
        # Initialize performance metrics
        performance_metrics = {
            "total_profit": 0.0,
            "routing_efficiency": 0.0,
            "risk_adjusted_return": 0.0,
            "rebalancing_frequency": 0.0
        }
        
        # Generate hash signature
        hash_input = f"{route_name}_{strategy.value}_{risk_tolerance}_{performance_threshold}"
        hash_signature = hashlib.sha256(hash_input.encode()).hexdigest()[:16]
        
        route = ProfitRoute(
            route_id=route_id,
            strategy=strategy,
            allocation_weights=allocation_weights,
            risk_tolerance=risk_tolerance,
            performance_metrics=performance_metrics,
            hash_signature=hash_signature,
            metadata={"route_name": route_name}
        )
        
        self.routes[route_id] = route
        
        # Maintain route limit
        if len(self.routes) > self.max_routes:
            # Remove oldest route
            oldest_route_id = unified_math.min(self.routes.keys(), key=lambda k: self.routes[k].timestamp)
            del self.routes[oldest_route_id]
        
        logger.info(f"Created profit route: {route_name} ({strategy.value})")
        return route
    
    def _calculate_allocation_weights(
        self, strategy: RoutingStrategy, risk_tolerance: float
    ) -> Dict[str, float]:
        """Calculate allocation weights based on strategy and risk tolerance."""
        base_weights = self.default_allocation.copy()
        
        if strategy == RoutingStrategy.CONSERVATIVE:
            # Favor stable assets (USDC)
            base_weights["USDC"] *= (1.0 + risk_tolerance)
            base_weights["BTC"] *= (1.0 - 0.5 * risk_tolerance)
            base_weights["ETH"] *= (1.0 - 0.3 * risk_tolerance)
            base_weights["XRP"] *= (1.0 - 0.2 * risk_tolerance)
        
        elif strategy == RoutingStrategy.AGGRESSIVE:
            # Favor volatile assets (BTC, ETH)
            base_weights["BTC"] *= (1.0 + risk_tolerance)
            base_weights["ETH"] *= (1.0 + 0.5 * risk_tolerance)
            base_weights["USDC"] *= (1.0 - 0.5 * risk_tolerance)
            base_weights["XRP"] *= (1.0 + 0.3 * risk_tolerance)
        
        elif strategy == RoutingStrategy.RISK_ADJUSTED:
            # Balanced approach with risk adjustment
            base_weights["BTC"] *= (1.0 + 0.3 * risk_tolerance)
            base_weights["ETH"] *= (1.0 + 0.2 * risk_tolerance)
            base_weights["USDC"] *= (1.0 - 0.2 * risk_tolerance)
            base_weights["XRP"] *= (1.0 + 0.1 * risk_tolerance)
        
        # Normalize weights
        total_weight = sum(base_weights.values())
        normalized_weights = {asset: weight / total_weight for asset, weight in base_weights.items()}
        
        return normalized_weights
    
    def update_performance_metrics(
        self,
        strategy_name: str,
        total_profit: float,
        max_drawdown: float,
        sharpe_ratio: float,
        win_rate: float,
        profit_factor: float
    ) -> None:
        """Update performance metrics for a trading strategy."""
        performance = StrategyPerformance(
            strategy_name=strategy_name,
            total_profit=total_profit,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor
        )
        
        self.strategy_performances[strategy_name] = performance
        logger.debug(f"Updated performance for strategy: {strategy_name}")
    
    def route_profit(
        self, profit_amount: float, strategy_name: str, route_name: str
    ) -> Dict[str, float]:
        """Route profit according to specified strategy and route."""
        if not self.routes:
            logger.warning("No routes available for profit routing")
            return {}
        
        # Find the specified route
        target_route = None
        for route in self.routes.values():
            if route.metadata.get("route_name") == route_name:
                target_route = route
                break
        
        if not target_route:
            logger.warning(f"Route '{route_name}' not found, using default allocation")
            target_route = list(self.routes.values())[0]
        
        # Calculate profit allocation using profit delta vector
        profit_allocation = {}
        for asset, weight in target_route.allocation_weights.items():
            profit_allocation[asset] = profit_amount * weight
        
        # Update route performance
        target_route.performance_metrics["total_profit"] += profit_amount
        target_route.performance_metrics["routing_efficiency"] = self._calculate_routing_efficiency(
            target_route, profit_allocation
        )
        
        # Update global metrics
        self.total_profit_routed += profit_amount
        self.routing_efficiency = self._calculate_global_routing_efficiency()
        
        # Store profit history
        self.profit_history.append({
            "timestamp": datetime.now(),
            "strategy": strategy_name,
            "route": route_name,
            "amount": profit_amount,
            "allocation": profit_allocation
        })
        
        # Maintain history size
        if len(self.profit_history) > self.markov_memory_size:
            self.profit_history = self.profit_history[-self.markov_memory_size:]
        
        logger.info(f"Routed {profit_amount:.2f} profit using {route_name}")
        return profit_allocation
    
    def _calculate_routing_efficiency(
        self, route: ProfitRoute, allocation: Dict[str, float]
    ) -> float:
        """Calculate routing efficiency for a specific route."""
        if not allocation:
            return 0.0
        
        # Calculate entropy of allocation
        total_allocation = sum(allocation.values())
        if total_allocation == 0:
            return 0.0
        
        probabilities = [amount / total_allocation for amount in allocation.values()]
        entropy = -sum(p * np.log2(p + 1e-10) for p in probabilities)
        
        # Normalize entropy to [0, 1]
        max_entropy = np.log2(len(allocation))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return float(normalized_entropy)
    
    def _calculate_global_routing_efficiency(self) -> float:
        """Calculate global routing efficiency across all routes."""
        if not self.routes:
            return 0.0
        
        efficiencies = [
            route.performance_metrics["routing_efficiency"]
            for route in self.routes.values()
        ]
        
        return float(unified_math.unified_math.mean(efficiencies))
    
    def optimize_routing_allocation(self) -> Dict[str, float]:
        """Optimize routing allocation using eigenvector centrality and Markov chains."""
        if not self.profit_history:
            return self.default_allocation
        
        # Build profit delta vector
        profit_delta_vector = self._build_profit_delta_vector()
        
        # Calculate weighted strategy entropy matrix
        entropy_matrix = self._calculate_entropy_matrix(profit_delta_vector)
        
        # Apply eigenvector centrality
        centrality_weights = self._calculate_eigenvector_centrality(entropy_matrix)
        
        # Update Markov transition matrix
        self._update_transition_matrix()
        
        # Combine centrality with transition probabilities
        optimized_allocation = self._combine_centrality_and_transitions(
            centrality_weights
        )
        
        logger.info("Optimized routing allocation using eigenvector centrality")
        return optimized_allocation
    
    def _build_profit_delta_vector(self) -> Dict[str, float]:
        """Build profit delta vector from historical data."""
        if not self.profit_history:
            return {asset: 0.0 for asset in self.supported_assets}
        
        # Calculate cumulative profits per asset
        asset_profits = {asset: 0.0 for asset in self.supported_assets}
        
        for entry in self.profit_history:
            allocation = entry.get("allocation", {})
            for asset, amount in allocation.items():
                if asset in asset_profits:
                    asset_profits[asset] += amount
        
        # Calculate deltas (relative to mean)
        total_profit = sum(asset_profits.values())
        mean_profit = total_profit / len(asset_profits) if asset_profits else 0.0
        
        profit_deltas = {}
        for asset, profit in asset_profits.items():
            profit_deltas[asset] = profit - mean_profit
        
        return profit_deltas
    
    def _calculate_entropy_matrix(
        self, profit_deltas: Dict[str, float]
    ) -> np.ndarray[Any, Any]:
        """Calculate weighted strategy entropy matrix."""
        n_assets = len(self.supported_assets)
        entropy_matrix = np.zeros((n_assets, n_assets))
        
        for i, asset_i in enumerate(self.supported_assets):
            for j, asset_j in enumerate(self.supported_assets):
                if i == j:
                    entropy_matrix[i, j] = 0.0
                else:
                    p_i = profit_deltas.get(asset_i, 0.0)
                    p_j = profit_deltas.get(asset_j, 0.0)
                    
                    # M_ij = unified_math.log(P_i / P_j + ε)
                    epsilon = 1e-10
                    if unified_math.abs(p_j) < epsilon:
                        entropy_matrix[i, j] = 0.0
                    else:
                        entropy_matrix[i, j] = unified_math.unified_math.log(unified_math.abs(p_i / p_j) + epsilon)
        
        return entropy_matrix
    
    def _calculate_eigenvector_centrality(self, matrix: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Calculate eigenvector centrality of the matrix."""
        try:
            # Ensure matrix is symmetric
            symmetric_matrix = (matrix + matrix.T) / 2
            
            # Calculate eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eigh(symmetric_matrix)
            
            # Find the eigenvector corresponding to the largest eigenvalue
            max_eigenvalue_idx = np.argmax(eigenvalues)
            centrality_vector = eigenvectors[:, max_eigenvalue_idx]
            
            # Normalize to positive values
            centrality_vector = unified_math.unified_math.abs(centrality_vector)
            centrality_vector = centrality_vector / np.sum(centrality_vector)
            
            return centrality_vector
        
        except Exception as e:
            logger.warning(f"Eigenvector centrality calculation failed: {e}")
            # Return uniform distribution as fallback
            n_assets = len(self.supported_assets)
            return np.ones(n_assets) / n_assets
    
    def _update_transition_matrix(self) -> None:
        """Update Markov transition matrix from profit history."""
        if len(self.profit_history) < 2:
            self.transition_matrix = None
            return
        
        n_assets = len(self.supported_assets)
        transition_matrix = np.zeros((n_assets, n_assets))
        
        # Count transitions between assets
        for i in range(len(self.profit_history) - 1):
            current_allocation = self.profit_history[i].get("allocation", {})
            next_allocation = self.profit_history[i + 1].get("allocation", {})
            
            # Find dominant assets (highest allocation)
            current_dominant = unified_math.max(current_allocation.items(), key=lambda x: x[1])[0] if current_allocation else None
            next_dominant = unified_math.max(next_allocation.items(), key=lambda x: x[1])[0] if next_allocation else None
            
            if current_dominant and next_dominant:
                try:
                    current_idx = self.supported_assets.index(current_dominant)
                    next_idx = self.supported_assets.index(next_dominant)
                    transition_matrix[current_idx, next_idx] += 1
                except ValueError:
                    continue
        
        # Normalize transition matrix
        row_sums = transition_matrix.sum(axis=1)
        for i in range(n_assets):
            if row_sums[i] > 0:
                transition_matrix[i, :] /= row_sums[i]
            else:
                # Uniform distribution if no transitions
                transition_matrix[i, :] = 1.0 / n_assets
        
        self.transition_matrix = transition_matrix
    
    def _combine_centrality_and_transitions(
        self, centrality_weights: np.ndarray[Any, Any]
    ) -> Dict[str, float]:
        """Combine eigenvector centrality with Markov transition probabilities."""
        if self.transition_matrix is None:
            # Use only centrality weights
            allocation = {}
            for i, asset in enumerate(self.supported_assets):
                allocation[asset] = float(centrality_weights[i])
            return allocation
        
        # Combine centrality with transition probabilities
        n_assets = len(self.supported_assets)
        combined_weights = np.zeros(n_assets)
        
        for i in range(n_assets):
            # Weighted combination: 70% centrality, 30% transition probability
            centrality_component = 0.7 * centrality_weights[i]
            transition_component = 0.3 * unified_math.unified_math.mean(self.transition_matrix[:, i])
            combined_weights[i] = centrality_component + transition_component
        
        # Normalize
        combined_weights = unified_math.unified_math.abs(combined_weights)
        combined_weights = combined_weights / np.sum(combined_weights)
        
        # Convert to dictionary
        allocation = {}
        for i, asset in enumerate(self.supported_assets):
            allocation[asset] = float(combined_weights[i])
        
        return allocation
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive routing statistics."""
        total_routes = len(self.routes)
        active_routes = sum(
            1 for route in self.routes.values()
            if route.performance_metrics["total_profit"] > 0
        )
        
        # Calculate average performance metrics
        avg_efficiency = unified_math.mean([
            route.performance_metrics["routing_efficiency"]
            for route in self.routes.values()
        ]) if self.routes else 0.0
        
        avg_profit = unified_math.mean([
            route.performance_metrics["total_profit"]
            for route in self.routes.values()
        ]) if self.routes else 0.0
        
        return {
            "total_routes": total_routes,
            "active_routes": active_routes,
            "total_profit_routed": self.total_profit_routed,
            "routing_efficiency": self.routing_efficiency,
            "average_efficiency": float(avg_efficiency),
            "average_profit_per_route": float(avg_profit),
            "rebalancing_count": self.rebalancing_count,
            "profit_history_size": len(self.profit_history)
        }
    
    def get_routing_recommendations(self) -> List[Dict[str, Any]]:
        """Get routing recommendations based on current performance."""
        recommendations = []
        
        if not self.routes:
            return recommendations
        
        # Find best performing route
        best_route = max(
            self.routes.values(),
            key=lambda r: r.performance_metrics["total_profit"]
        )
        
        recommendations.append({
            "type": "best_performing_route",
            "route_name": best_route.metadata.get("route_name", "Unknown"),
            "strategy": best_route.strategy.value,
            "total_profit": best_route.performance_metrics["total_profit"],
            "efficiency": best_route.performance_metrics["routing_efficiency"]
        })
        
        # Check for rebalancing opportunities
        if self.routing_efficiency < self.entropy_threshold:
            recommendations.append({
                "type": "rebalancing_recommended",
                "current_efficiency": self.routing_efficiency,
                "threshold": self.entropy_threshold,
                "suggestion": "Consider rebalancing allocation weights"
            })
        
        # Strategy performance recommendations
        if self.strategy_performances:
            best_strategy = max(
                self.strategy_performances.values(),
                key=lambda s: s.sharpe_ratio
            )
            
            recommendations.append({
                "type": "best_strategy",
                "strategy_name": best_strategy.strategy_name,
                "sharpe_ratio": best_strategy.sharpe_ratio,
                "win_rate": best_strategy.win_rate
            })
        
        return recommendations
    
    def get_trading_signals(self) -> List[Dict[str, Any]]:
        """Generate trading signals based on routing analysis."""
        signals = []
        
        if not self.routes:
            return signals
        
        # High efficiency signal
        if self.routing_efficiency > 0.8:
            signals.append({
                "type": "high_routing_efficiency",
                "efficiency": self.routing_efficiency,
                "timestamp": datetime.now(),
                "metadata": {
                    "total_routes": len(self.routes),
                    "total_profit_routed": self.total_profit_routed
                }
            })
        
        # Rebalancing signal
        if self.routing_efficiency < self.entropy_threshold:
            signals.append({
                "type": "rebalancing_needed",
                "current_efficiency": self.routing_efficiency,
                "threshold": self.entropy_threshold,
                "timestamp": datetime.now(),
                "metadata": {
                    "suggestion": "Optimize allocation weights"
                }
            })
        
        # Strategy performance signals
        for strategy_name, performance in self.strategy_performances.items():
            if performance.sharpe_ratio > 1.5:
                signals.append({
                    "type": "high_performing_strategy",
                    "strategy_name": strategy_name,
                    "sharpe_ratio": performance.sharpe_ratio,
                    "timestamp": datetime.now(),
                    "metadata": {
                        "win_rate": performance.win_rate,
                        "profit_factor": performance.profit_factor
                    }
                })
        
        return signals


def main() -> None:
    """Main function for testing the profit routing engine."""
    logging.basicConfig(level=logging.INFO)
    
    # Initialize engine
    engine = ProfitRoutingEngine()
    
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
    safe_print(f"Optimal allocations: {optimal_allocations}")
    
    # Get statistics
    stats = engine.get_routing_statistics()
    safe_print(f"Routing statistics: {stats}")
    
    # Get recommendations
    recommendations = engine.get_routing_recommendations()
    safe_print(f"Routing recommendations: {len(recommendations)}")
    
    # Get trading signals
    signals = engine.get_trading_signals()
    safe_print(f"Generated {len(signals)} trading signals") 