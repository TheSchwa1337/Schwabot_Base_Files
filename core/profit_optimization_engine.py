"""Module for Schwabot trading system."""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Profit Optimization Engine 🚀

Advanced portfolio optimization with tensor math integration:
• Multi-method portfolio optimization (gradient descent, genetic algorithm)
• Sharpe ratio and risk-adjusted return maximization
• Real-time portfolio rebalancing and weight optimization
• GPU/CPU tensor operations for fast calculations
• Integration with risk management and exchange systems

Features:
- GPU-accelerated optimization with automatic CPU fallback
- Multiple optimization algorithms and methods
- Real-time portfolio rebalancing
- Risk-adjusted return optimization
- Integration with Schwabot's tensor math chain
"""

from enum import Enum
from dataclasses import dataclass, field
import time
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import cupy as cp
    import numpy as np

    USING_CUDA = True
    xp = cp
    _backend = "cupy (GPU)"
except ImportError:
    try:
        import numpy as np

        USING_CUDA = False
        xp = np
        _backend = "numpy (CPU)"
    except ImportError:
        xp = None
        _backend = "none"

logger = logging.getLogger(__name__)
if xp is None:
    logger.warning("❌ NumPy not available for optimization calculations")
else:
    logger.info(f"⚡ ProfitOptimizationEngine using {_backend} for tensor operations")


class OptimizationMethod(Enum):
    """Available optimization methods."""

    GRADIENT_DESCENT = "gradient_descent"
    GENETIC_ALGORITHM = "genetic_algorithm"
    PARTICLE_SWARM = "particle_swarm"
    BLACK_LITTERMAN = "black_litterman"
    MEAN_VARIANCE = "mean_variance"


class OptimizationStatus(Enum):
    """Optimization status."""

    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class OptimizationResult:
    """Result of portfolio optimization."""
    
    method: OptimizationMethod
    weights: Any  # xp.ndarray
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    convergence_iterations: int
    optimization_time: float
    status: OptimizationStatus
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class PortfolioConstraints:
    """Portfolio optimization constraints."""
    
    min_weight: float = 0.0
    max_weight: float = 1.0
    target_return: Optional[float] = None
    max_volatility: Optional[float] = None
    risk_free_rate: float = 0.02
    rebalance_threshold: float = 0.05  # 5% threshold for rebalancing


class ProfitOptimizationEngine:
    """
    Advanced profit optimization engine with tensor math integration.
    Handles portfolio optimization, rebalancing, and risk-adjusted return maximization.
    """

    def __init__(self, constraints: Optional[PortfolioConstraints] = None) -> None:
        self.constraints = constraints or PortfolioConstraints()
        self.optimization_history: List[OptimizationResult] = []
        self.current_weights: Optional[Any] = None  # xp.ndarray
        self.processing_mode = "gpu" if USING_CUDA else "cpu"
        self.is_optimizing = False

    def optimize_portfolio(
        self,
        returns: Any,
        method: OptimizationMethod = OptimizationMethod.GRADIENT_DESCENT,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
    ) -> OptimizationResult:
        """Optimize portfolio weights for maximum Sharpe ratio."""
        try:
            if xp is None:
                raise ValueError("Tensor operations not available")

            self.is_optimizing = True
            start_time = time.time()

            logger.info(f"🚀 Starting portfolio optimization with {method.value}")

            if method == OptimizationMethod.GRADIENT_DESCENT:
                result = self._gradient_descent_optimization(
                    returns, max_iterations, tolerance
                )
            elif method == OptimizationMethod.GENETIC_ALGORITHM:
                result = self._genetic_algorithm_optimization(returns, max_iterations)
            elif method == OptimizationMethod.PARTICLE_SWARM:
                result = self._particle_swarm_optimization(returns, max_iterations)
            else:
                raise ValueError(f"Unsupported optimization method: {method}")

            result.method = method
            result.optimization_time = time.time() - start_time
            result.status = OptimizationStatus.COMPLETED

            self.optimization_history.append(result)
            self.current_weights = result.weights

            logger.info(
                f"✅ Optimization completed: Sharpe ratio = {result.sharpe_ratio:.4f}"
            )
            return result

        except Exception as e:
            logger.error(f"❌ Optimization failed: {e}")
            return OptimizationResult(
                method=method,
                weights=xp.array([]) if xp is not None else [],
                expected_return=0.0,
                expected_volatility=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                max_drawdown=0.0,
                convergence_iterations=0,
                optimization_time=0.0,
                status=OptimizationStatus.FAILED,
            )
        finally:
            self.is_optimizing = False

    def _gradient_descent_optimization(
        self, returns: Any, max_iterations: int, tolerance: float
    ) -> OptimizationResult:
        """Gradient descent optimization for maximum Sharpe ratio."""
        try:
            n_assets = returns.shape[1]
            weights = xp.ones(n_assets) / n_assets
            mean_returns = xp.mean(returns, axis=0)
            cov_matrix = xp.cov(returns.T)
            learning_rate = 0.01
            best_sharpe = -xp.inf
            best_weights = weights.copy()
            for iteration in range(max_iterations):
                portfolio_return = xp.dot(weights, mean_returns)
                portfolio_volatility = xp.sqrt(
                    xp.dot(weights, xp.dot(cov_matrix, weights))
                )
                if portfolio_volatility > 0:
                    sharpe_ratio = (
                        portfolio_return - self.constraints.risk_free_rate
                    ) / portfolio_volatility
                else:
                    sharpe_ratio = 0.0
                if sharpe_ratio > best_sharpe:
                    best_sharpe = sharpe_ratio
                    best_weights = weights.copy()
                if portfolio_volatility > 0:
                    gradient = (
                        mean_returns * portfolio_volatility
                        - (portfolio_return - self.constraints.risk_free_rate)
                        * xp.dot(cov_matrix, weights)
                        / portfolio_volatility
                    ) / (portfolio_volatility**2)
                else:
                    gradient = mean_returns
                weights_new = weights + learning_rate * gradient
                weights_new = xp.clip(
                    weights_new,
                    self.constraints.min_weight,
                    self.constraints.max_weight,
                )
                weights_new = weights_new / xp.sum(weights_new)
                if xp.linalg.norm(weights_new - weights) < tolerance:
                    break
                weights = weights_new
            final_return = xp.dot(best_weights, mean_returns)
            final_volatility = xp.sqrt(
                xp.dot(best_weights, xp.dot(cov_matrix, best_weights))
            )
            final_sharpe = (
                (final_return - self.constraints.risk_free_rate) / final_volatility
                if final_volatility > 0
                else 0.0
            )
            return OptimizationResult(
                method=OptimizationMethod.GRADIENT_DESCENT,
                weights=best_weights,
                expected_return=float(final_return),
                expected_volatility=float(final_volatility),
                sharpe_ratio=float(final_sharpe),
                sortino_ratio=self._calculate_sortino_ratio(best_weights, returns),
                max_drawdown=self._calculate_max_drawdown(best_weights, returns),
                convergence_iterations=iteration + 1,
                optimization_time=0.0,  # Will be set by caller
                status=OptimizationStatus.COMPLETED,
            )
        except Exception as e:
            logger.error(f"❌ Gradient descent optimization failed: {e}")
            raise

    def _genetic_algorithm_optimization(
        self, returns: Any, max_iterations: int
    ) -> OptimizationResult:
        """Genetic algorithm optimization for portfolio weights."""
        try:
            n_assets = returns.shape[1]
            population_size = 50
            mutation_rate = 0.1
            population = []
            for _ in range(population_size):
                weights = xp.random.random(n_assets)
                weights = weights / xp.sum(weights)
                population.append(weights)
            best_sharpe = -xp.inf
            best_weights = None
            for generation in range(max_iterations):
                fitness_scores = []
                for weights in population:
                    sharpe = self._calculate_sharpe_ratio(weights, returns)
                    fitness_scores.append(sharpe)
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_weights = weights.copy()
                new_population = []
                for _ in range(population_size):
                    idx1, idx2 = xp.random.choice(len(population), 2, replace=False)
                    parent1 = (
                        population[idx1]
                        if fitness_scores[idx1] > fitness_scores[idx2]
                        else population[idx2]
                    )
                    parent2 = population[xp.random.choice(len(population))]
                    crossover_point = xp.random.randint(1, n_assets)
                    child = xp.concatenate(
                        [parent1[:crossover_point], parent2[crossover_point:]]
                    )
                    if xp.random.random() < mutation_rate:
                        mutation_idx = xp.random.randint(0, n_assets)
                        child[mutation_idx] = xp.random.random()
                    child = child / xp.sum(child)
                    new_population.append(child)
                population = new_population
            if best_weights is None:
                best_weights = xp.ones(n_assets) / n_assets
            return OptimizationResult(
                method=OptimizationMethod.GENETIC_ALGORITHM,
                weights=best_weights,
                expected_return=float(xp.dot(best_weights, xp.mean(returns, axis=0))),
                expected_volatility=float(
                    xp.sqrt(
                        xp.dot(best_weights, xp.dot(xp.cov(returns.T), best_weights))
                    )
                ),
                sharpe_ratio=float(best_sharpe),
                sortino_ratio=self._calculate_sortino_ratio(best_weights, returns),
                max_drawdown=self._calculate_max_drawdown(best_weights, returns),
                convergence_iterations=max_iterations,
                optimization_time=0.0,
                status=OptimizationStatus.COMPLETED,
            )
        except Exception as e:
            logger.error(f"❌ Genetic algorithm optimization failed: {e}")
            raise

    def _particle_swarm_optimization(
        self, returns: Any, max_iterations: int
    ) -> OptimizationResult:
        """Particle swarm optimization for portfolio weights."""
        try:
            n_assets = returns.shape[1]
            n_particles = 30
            particles = []
            velocities = []
            personal_best = []
            personal_best_fitness = []
            for _ in range(n_particles):
                weights = xp.random.random(n_assets)
                weights = weights / xp.sum(weights)
                particles.append(weights)
                velocities.append(xp.random.random(n_assets) * 0.1)
                personal_best.append(weights.copy())
                personal_best_fitness.append(-xp.inf)
            global_best = particles[0].copy()
            global_best_fitness = -xp.inf
            for iteration in range(max_iterations):
                for i in range(n_particles):
                    fitness = self._calculate_sharpe_ratio(particles[i], returns)
                    if fitness > personal_best_fitness[i]:
                        personal_best_fitness[i] = fitness
                        personal_best[i] = particles[i].copy()
                    if fitness > global_best_fitness:
                        global_best_fitness = fitness
                        global_best = particles[i].copy()
                for i in range(n_particles):
                    w = 0.7  # inertia
                    c1 = 1.5  # cognitive parameter
                    c2 = 1.5  # social parameter
                    velocities[i] = (
                        w * velocities[i]
                        + c1 * xp.random.random() * (personal_best[i] - particles[i])
                        + c2 * xp.random.random() * (global_best - particles[i])
                    )
                    particles[i] += velocities[i]
                    particles[i] = xp.clip(
                        particles[i],
                        self.constraints.min_weight,
                        self.constraints.max_weight,
                    )
                    particles[i] = particles[i] / xp.sum(particles[i])
            return OptimizationResult(
                method=OptimizationMethod.PARTICLE_SWARM,
                weights=global_best,
                expected_return=float(xp.dot(global_best, xp.mean(returns, axis=0))),
                expected_volatility=float(
                    xp.sqrt(xp.dot(global_best, xp.dot(xp.cov(returns.T), global_best)))
                ),
                sharpe_ratio=float(global_best_fitness),
                sortino_ratio=self._calculate_sortino_ratio(global_best, returns),
                max_drawdown=self._calculate_max_drawdown(global_best, returns),
                convergence_iterations=max_iterations,
                optimization_time=0.0,
                status=OptimizationStatus.COMPLETED,
            )
        except Exception as e:
            logger.error(f"❌ Particle swarm optimization failed: {e}")
            raise

    def _calculate_sharpe_ratio(self, weights: Any, returns: Any) -> float:
        """Calculate Sharpe ratio for given weights."""
        try:
            portfolio_returns = xp.dot(returns, weights)
            mean_return = xp.mean(portfolio_returns)
            volatility = xp.std(portfolio_returns)
            if volatility > 0:
                return (mean_return - self.constraints.risk_free_rate) / volatility
            else:
                return 0.0
        except Exception:
            return 0.0

    def _calculate_sortino_ratio(self, weights: Any, returns: Any) -> float:
        """Calculate Sortino ratio for given weights."""
        try:
            portfolio_returns = xp.dot(returns, weights)
            mean_return = xp.mean(portfolio_returns)
            downside_returns = portfolio_returns[portfolio_returns < mean_return]
            if len(downside_returns) > 0:
                downside_deviation = xp.std(downside_returns)
                if downside_deviation > 0:
                    return (
                        mean_return - self.constraints.risk_free_rate
                    ) / downside_deviation
            return 0.0
        except Exception:
            return 0.0

    def _calculate_max_drawdown(self, weights: Any, returns: Any) -> float:
        """Calculate maximum drawdown for given weights."""
        try:
            portfolio_returns = xp.dot(returns, weights)
            cumulative_returns = xp.cumprod(1 + portfolio_returns)
            running_max = xp.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - running_max) / running_max
            return float(xp.min(drawdowns))
        except Exception:
            return 0.0

    def should_rebalance(self, current_weights: Any, target_weights: Any) -> bool:
        """Check if portfolio needs rebalancing."""
        if current_weights is None or target_weights is None:
            return True
        weight_diff = xp.abs(current_weights - target_weights)
        max_diff = xp.max(weight_diff)
        return max_diff > self.constraints.rebalance_threshold

    def optimize_profit(
        self, weights: List[float], returns: List[float], risk_aversion: float = 0.5
    ) -> float:
        """
        Optimize profit using the formula: P = Σ w_i * r_i - λ * Σ w_i²
        Args:
        weights: Portfolio weights
        returns: Asset returns
        risk_aversion: Risk aversion parameter (lambda)
        Returns:
        Optimized profit value
        """
        try:
            if xp is None:
                raise ValueError("Tensor operations not available")
            w = xp.array(weights)
            r = xp.array(returns)
            w = w / xp.sum(w)
            expected_return = xp.sum(w * r)
            risk_penalty = risk_aversion * xp.sum(w**2)
            optimized_profit = expected_return - risk_penalty
            return float(optimized_profit)
        except Exception as e:
            logger.error(f"Profit optimization failed: {e}")
            return 0.0

    def calculate_risk_adjusted_return(
        self, returns: List[float], risk_free_rate: float = 0.02
    ) -> float:
        """
        Calculate risk-adjusted return.
        Args:
        returns: Asset returns
        risk_free_rate: Risk-free rate
        Returns:
        Risk-adjusted return
        """
        try:
            if xp is None:
                raise ValueError("Tensor operations not available")
            returns_array = xp.array(returns)
            mean_return = xp.mean(returns_array)
            volatility = xp.std(returns_array)
            if volatility > 0:
                risk_adjusted_return = (mean_return - risk_free_rate) / volatility
            else:
                risk_adjusted_return = 0.0
            return float(risk_adjusted_return)
        except Exception as e:
            logger.error(f"Risk-adjusted return calculation failed: {e}")
            return 0.0

    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization history."""
        try:
            if not self.optimization_history:
                return {"error": "No optimization history"}
            latest = self.optimization_history[-1]
            return {
                "total_optimizations": len(self.optimization_history),
                "latest_method": latest.method.value,
                "latest_sharpe_ratio": latest.sharpe_ratio,
                "latest_expected_return": latest.expected_return,
                "latest_expected_volatility": latest.expected_volatility,
                "optimization_time": latest.optimization_time,
                "status": latest.status.value,
                "current_weights": latest.weights.tolist()
                if latest.weights is not None
                else [],
            }
        except Exception as e:
            logger.error(f"❌ Failed to get optimization summary: {e}")
            return {"error": str(e)}


# Singleton instance for global use
profit_optimization_engine = ProfitOptimizationEngine()
