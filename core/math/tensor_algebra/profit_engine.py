from __future__ import annotations

#!/usr/bin/env python3
"""
Profit Engine - Advanced Profit Surface Calculations
===================================================

Provides advanced mathematical functions for profit surface calculations,
long-hold position optimization, and profit gradient analysis for the
Schwabot trading system.

Core Functions:
- compute_profit_surface: Calculate multi-dimensional profit surfaces
- optimize_long_hold_positions: Optimize long-term holding strategies
- calculate_profit_gradient: Analyze profit gradients and optimal paths
- estimate_profit_curves: Estimate profit curves over time
- analyze_profit_distribution: Analyze profit distribution patterns
"""


import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple, Optional, Union, Dict, Any
import logging
from scipy.optimize import minimize
from scipy.interpolate import griddata
from scipy.stats import norm, skew, kurtosis
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)

logger = logging.getLogger(__name__)


class ProfitEngine:
    """
    Advanced Profit Engine for Schwabot Trading System.

    This engine provides comprehensive profit analysis and optimization
    capabilities for multi-dimensional trading strategies.
    """

    def __init__(self):
        """Initialize the profit engine."""
        self.epsilon = 1e-8  # Small value to prevent division by zero
        self.max_iterations = 1000  # Maximum optimization iterations
        self.convergence_tolerance = 1e-6  # Optimization convergence tolerance

        logger.info("Profit Engine initialized")

    def compute_profit_surface(self, price_map: NDArray, hold_map: NDArray) -> NDArray:
        """
        Compute multi-dimensional profit surface from price and hold maps.

        Args:
            price_map: Price data matrix
            hold_map: Holding period matrix

        Returns:
            Profit surface matrix
        """
        try:
            if price_map.shape != hold_map.shape:
                raise ValueError("Price and hold maps must have the same shape")

            # Calculate price changes
            price_changes = np.gradient(price_map, axis=0)

            # Calculate holding period effects
            hold_effects = np.exp(-hold_map / 100.0)  # Exponential decay

            # Compute profit surface
            profit_surface = price_changes * hold_effects

            # Apply smoothing
            profit_surface = self._smooth_surface(profit_surface)

            return profit_surface

        except Exception as e:
            logger.error(f"Profit surface computation failed: {e}")
            return np.zeros_like(price_map)

    def optimize_long_hold_positions(self, price_series: NDArray) -> NDArray:
        """
        Optimize long-hold positions based on price series analysis.

        Args:
            price_series: Historical price series

        Returns:
            Optimized holding periods
        """
        try:
            if len(price_series) < 10:
                return np.array([1.0] * len(price_series))

            # Calculate volatility
            returns = np.diff(np.log(price_series))
            volatility = np.std(returns)

            # Calculate trend strength
            trend_strength = self._calculate_trend_strength(price_series)

            # Optimize holding periods
            optimal_holds = []

            for i in range(len(price_series)):
                # Base holding period
                base_hold = 1.0

                # Adjust for volatility
                vol_adjustment = 1.0 / (1.0 + volatility * 10)

                # Adjust for trend
                trend_adjustment = 1.0 + trend_strength * 0.5

                # Calculate optimal hold
                optimal_hold = base_hold * vol_adjustment * trend_adjustment
                optimal_holds.append(max(0.1, min(5.0, optimal_hold)))

            return np.array(optimal_holds)

        except Exception as e:
            logger.error(f"Long-hold optimization failed: {e}")
            return np.array([1.0] * len(price_series))

    def calculate_profit_gradient(self, profit_surface: NDArray) -> Tuple[NDArray, NDArray]:
        """
        Calculate profit gradient and optimal paths.

        Args:
            profit_surface: Profit surface matrix

        Returns:
            Tuple of (gradient_x, gradient_y)
        """
        try:
            # Calculate gradients
            gradient_x = np.gradient(profit_surface, axis=1)
            gradient_y = np.gradient(profit_surface, axis=0)

            # Normalize gradients
            gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

            # Avoid division by zero
            gradient_magnitude = np.where(gradient_magnitude > self.epsilon,
                                        gradient_magnitude, self.epsilon)

            gradient_x = gradient_x / gradient_magnitude
            gradient_y = gradient_y / gradient_magnitude

            return gradient_x, gradient_y

        except Exception as e:
            logger.error(f"Profit gradient calculation failed: {e}")
            return np.zeros_like(profit_surface), np.zeros_like(profit_surface)

    def estimate_profit_curves(self, price_data: NDArray,
                             time_horizons: List[float]) -> Dict[str, NDArray]:
        """
        Estimate profit curves over different time horizons.

        Args:
            price_data: Historical price data
            time_horizons: List of time horizons in days

        Returns:
            Dictionary of profit curves for each horizon
        """
        try:
            profit_curves = {}

            for horizon in time_horizons:
                # Calculate returns for this horizon
                horizon_returns = []

                for i in range(len(price_data) - int(horizon)):
                    start_price = price_data[i]
                    end_price = price_data[i + int(horizon)]
                    return_rate = (end_price - start_price) / start_price
                    horizon_returns.append(return_rate)

                if horizon_returns:
                    profit_curves[f"horizon_{horizon}"] = np.array(horizon_returns)
                else:
                    profit_curves[f"horizon_{horizon}"] = np.array([])

            return profit_curves

        except Exception as e:
            logger.error(f"Profit curve estimation failed: {e}")
            return {f"horizon_{h}": np.array([]) for h in time_horizons}

    def analyze_profit_distribution(self, profit_data: NDArray) -> Dict[str, float]:
        """
        Analyze profit distribution patterns.

        Args:
            profit_data: Profit data array

        Returns:
            Dictionary of distribution statistics
        """
        try:
            if len(profit_data) == 0:
                return {
                    'mean': 0.0,
                    'std': 0.0,
                    'skewness': 0.0,
                    'kurtosis': 0.0,
                    'var_95': 0.0,
                    'var_99': 0.0,
                    'max_profit': 0.0,
                    'max_loss': 0.0
                }

            # Basic statistics
            mean_profit = float(np.mean(profit_data))
            std_profit = float(np.std(profit_data))

            # Higher moments
            skewness = float(skew(profit_data)) if len(profit_data) > 2 else 0.0
            kurtosis_val = float(kurtosis(profit_data)) if len(profit_data) > 3 else 0.0

            # Value at Risk
            var_95 = float(np.percentile(profit_data, 5))
            var_99 = float(np.percentile(profit_data, 1))

            # Extremes
            max_profit = float(np.max(profit_data))
            max_loss = float(np.min(profit_data))

            return {
                'mean': mean_profit,
                'std': std_profit,
                'skewness': skewness,
                'kurtosis': kurtosis_val,
                'var_95': var_95,
                'var_99': var_99,
                'max_profit': max_profit,
                'max_loss': max_loss
            }

        except Exception as e:
            logger.error(f"Profit distribution analysis failed: {e}")
            return {
                'mean': 0.0,
                'std': 0.0,
                'skewness': 0.0,
                'kurtosis': 0.0,
                'var_95': 0.0,
                'var_99': 0.0,
                'max_profit': 0.0,
                'max_loss': 0.0
            }

    def optimize_portfolio_allocation(self, profit_curves: Dict[str, NDArray],
                                    risk_tolerance: float = 0.5) -> Dict[str, float]:
        """
        Optimize portfolio allocation based on profit curves.

        Args:
            profit_curves: Dictionary of profit curves
            risk_tolerance: Risk tolerance (0.0 to 1.0)

        Returns:
            Dictionary of optimal allocations
        """
        try:
            allocations = {}
            total_weight = 0.0

            for horizon, curve in profit_curves.items():
                if len(curve) == 0:
                    allocations[horizon] = 0.0
                    continue

                # Calculate Sharpe ratio
                mean_return = np.mean(curve)
                std_return = np.std(curve)

                if std_return > self.epsilon:
                    sharpe_ratio = mean_return / std_return
                else:
                    sharpe_ratio = 0.0

                # Calculate allocation weight
                # Higher Sharpe ratio and lower risk tolerance = higher allocation
                weight = sharpe_ratio * (1.0 - risk_tolerance)
                weight = max(0.0, weight)  # Ensure non-negative

                allocations[horizon] = weight
                total_weight += weight

            # Normalize allocations
            if total_weight > 0:
                for horizon in allocations:
                    allocations[horizon] /= total_weight
            else:
                # Equal allocation if no positive weights
                n_horizons = len(allocations)
                for horizon in allocations:
                    allocations[horizon] = 1.0 / n_horizons

            return allocations

        except Exception as e:
            logger.error(f"Portfolio allocation optimization failed: {e}")
            # Return equal allocation as fallback
            n_horizons = len(profit_curves)
            return {horizon: 1.0 / n_horizons for horizon in profit_curves.keys()}

    def calculate_optimal_entry_timing(self, price_series: NDArray,
                                     profit_threshold: float = 0.02) -> List[int]:
        """
        Calculate optimal entry timing based on profit potential.

        Args:
            price_series: Historical price series
            profit_threshold: Minimum profit threshold

        Returns:
            List of optimal entry indices
        """
        try:
            optimal_entries = []

            # Calculate rolling profit potential
            window_size = min(20, len(price_series) // 4)

            for i in range(window_size, len(price_series)):
                # Calculate potential profit from this point
                current_price = price_series[i]
                future_prices = price_series[i:i+window_size]

                if len(future_prices) > 0:
                    max_future_price = np.max(future_prices)
                    potential_profit = (max_future_price - current_price) / current_price

                    if potential_profit >= profit_threshold:
                        optimal_entries.append(i)

            return optimal_entries

        except Exception as e:
            logger.error(f"Optimal entry timing calculation failed: {e}")
            return []

    def _smooth_surface(self, surface: NDArray, sigma: float = 1.0) -> NDArray:
        """Apply Gaussian smoothing to surface."""
        try:
            from scipy.ndimage import gaussian_filter
            return gaussian_filter(surface, sigma=sigma)
        except ImportError:
            # Fallback to simple smoothing
            return surface
        except Exception:
            return surface

    def _calculate_trend_strength(self, price_series: NDArray) -> float:
        """Calculate trend strength of price series."""
        try:
            if len(price_series) < 2:
                return 0.0

            # Calculate linear trend
            x = np.arange(len(price_series))
            slope, _ = np.polyfit(x, price_series, 1)

            # Normalize by price range
            price_range = np.max(price_series) - np.min(price_series)
            if price_range > 0:
                trend_strength = abs(slope) / price_range
            else:
                trend_strength = 0.0

            return float(trend_strength)

        except Exception:
            return 0.0


# Global instance for convenience
profit_engine = ProfitEngine()

# Convenience functions
def compute_profit_surface(price_map: NDArray, hold_map: NDArray) -> NDArray:
    """Convenience function for profit surface computation."""
    return profit_engine.compute_profit_surface(price_map, hold_map)


def optimize_long_hold_positions(price_series: NDArray) -> NDArray:
    """Convenience function for long-hold position optimization."""
    return profit_engine.optimize_long_hold_positions(price_series)


def calculate_profit_gradient(profit_surface: NDArray) -> Tuple[NDArray, NDArray]:
    """Convenience function for profit gradient calculation."""
    return profit_engine.calculate_profit_gradient(profit_surface)


def estimate_profit_curves(price_data: NDArray,
                         time_horizons: List[float]) -> Dict[str, NDArray]:
    """Convenience function for profit curve estimation."""
    return profit_engine.estimate_profit_curves(price_data, time_horizons)


def analyze_profit_distribution(profit_data: NDArray) -> Dict[str, float]:
    """Convenience function for profit distribution analysis."""
    return profit_engine.analyze_profit_distribution(profit_data)


if __name__ == "__main__":
    # Test the profit engine
    import numpy as np

    # Import safe print for Windows compatibility
    try:
        from ...utils.windows_cli_compatibility import safe_print
    except ImportError:
        try:
#             from core.utils.windows_cli_compatibility import safe_print  # F811: duplicate import
        except ImportError:
            def safe_print(message):
                print(message)

    def main():
        """Main function to test profit engine and ensure proper initialization."""
        try:
            safe_print("💰 Testing Profit Engine")
            safe_print("=" * 40)

            # Create test data
            price_series = np.array([100, 102, 98, 105, 103, 107, 110, 108, 112, 115])
            price_map = np.random.rand(10, 10) * 100
            hold_map = np.random.rand(10, 10) * 10

            safe_print(f"Price Series: {price_series}")
            safe_print(f"Price Map Shape: {price_map.shape}")
            safe_print(f"Hold Map Shape: {hold_map.shape}")

            # Test profit surface computation
            safe_print("\n📊 Testing Profit Surface Computation:")
            profit_surface = compute_profit_surface(price_map, hold_map)
            safe_print(f"✅ Profit Surface Shape: {profit_surface.shape}")
            safe_print(f"✅ Profit Surface Range: [{np.min(profit_surface):.4f}, {np.max(profit_surface):.4f}]")

            # Test long-hold optimization
            safe_print("\n⏱️ Testing Long-Hold Optimization:")
            optimal_holds = optimize_long_hold_positions(price_series)
            safe_print(f"✅ Optimal Holds: {optimal_holds}")
            safe_print(f"✅ Hold Range: [{np.min(optimal_holds):.4f}, {np.max(optimal_holds):.4f}]")

            # Test profit gradient
            safe_print("\n📈 Testing Profit Gradient:")
            gradient_x, gradient_y = calculate_profit_gradient(profit_surface)
            safe_print(f"✅ Gradient X Range: [{np.min(gradient_x):.4f}, {np.max(gradient_x):.4f}]")
            safe_print(f"✅ Gradient Y Range: [{np.min(gradient_y):.4f}, {np.max(gradient_y):.4f}]")

            # Test profit curves
            safe_print("\n📉 Testing Profit Curves:")
            time_horizons = [1, 3, 7]
            profit_curves = estimate_profit_curves(price_series, time_horizons)
            for horizon, curve in profit_curves.items():
                safe_print(f"✅ {horizon}: {len(curve)} data points")

            # Test profit distribution analysis
            safe_print("\n📊 Testing Profit Distribution Analysis:")
            if len(profit_curves['horizon_1']) > 0:
                distribution = analyze_profit_distribution(profit_curves['horizon_1'])
                safe_print("✅ Distribution Statistics:")
                for key, value in distribution.items():
                    safe_print(f"   {key}: {value:.4f}")

            # Test portfolio allocation optimization
            safe_print("\n🎯 Testing Portfolio Allocation:")
            allocations = profit_engine.optimize_portfolio_allocation(profit_curves, risk_tolerance=0.5)
            safe_print("✅ Portfolio Allocations:")
            for horizon, allocation in allocations.items():
                safe_print(f"   {horizon}: {allocation:.4f}")

            # Test optimal entry timing
            safe_print("\n⏰ Testing Optimal Entry Timing:")
            optimal_entries = profit_engine.calculate_optimal_entry_timing(price_series, profit_threshold=0.02)
            safe_print(f"✅ Optimal Entry Points: {optimal_entries}")

            # Test advanced profit engine features
            safe_print("\n🔬 Testing Advanced Features:")

            # Test trend strength calculation
            trend_strength = profit_engine._calculate_trend_strength(price_series)
            safe_print(f"✅ Trend Strength: {trend_strength:.4f}")

            # Test surface smoothing
            smoothed_surface = profit_engine._smooth_surface(profit_surface, sigma=1.0)
            safe_print(f"✅ Smoothed Surface Shape: {smoothed_surface.shape}")

            safe_print("\n🎉 Profit Engine tests completed successfully!")
            return True

        except Exception as e:
            safe_print(f"❌ Profit Engine test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    # Run main function
    success = main()
    import sys
    sys.exit(0 if success else 1)
