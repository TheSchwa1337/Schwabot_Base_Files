import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List

from core.backend_math import backend_info, get_backend

xp = get_backend()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean Unified Math System - Advanced Mathematical Operations

Provides a comprehensive, unified mathematical system for the Schwabot trading
platform. This module integrates various mathematical operations into a single
cohesive interface with GPU/CPU acceleration support.

Key Features:
- Unified mathematical operations with GPU acceleration
- Advanced statistical calculations
- Risk management metrics
- Portfolio optimization
- Performance tracking and analysis
"""

# Log backend status
logger = logging.getLogger(__name__)
backend_status = backend_info()
if backend_status["accelerated"]:
    logger.info("⚡ Clean Unified Math using GPU acceleration: CuPy (GPU)")
else:
    logger.info("🔄 Clean Unified Math using CPU fallback: NumPy (CPU)")


@dataclass
class MathResult:
    """Result container for mathematical operations."""

    value: Any
    operation: str
    timestamp: float
    metadata: Dict[str, Any]


class CleanUnifiedMathSystem:
    """Clean unified mathematical framework for trading calculations."""

    def __init__(self):
        """Initialize the unified math system."""
        self.calculation_history: List[MathResult] = []
        self.operation_cache: Dict[str, float] = {}

    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers with caching."""
        cache_key = f"multiply_{a}_{b}"
        if cache_key in self.operation_cache:
            return self.operation_cache[cache_key]

        result = a * b
        self._log_calculation("multiply", result, {"a": a, "b": b})
        return result

    def add(self, a: float, b: float) -> float:
        """Add two numbers with caching."""
        cache_key = f"add_{a}_{b}"
        if cache_key in self.operation_cache:
            return self.operation_cache[cache_key]

        result = a + b
        self._log_calculation("add", result, {"a": a, "b": b})
        return result

    def subtract(self, a: float, b: float) -> float:
        """Subtract two numbers with caching."""
        cache_key = f"subtract_{a}_{b}"
        if cache_key in self.operation_cache:
            return self.operation_cache[cache_key]

        result = a - b
        self._log_calculation("subtract", result, {"a": a, "b": b})
        return result

    def divide(self, a: float, b: float) -> float:
        """Divide two numbers with caching and error handling."""
        if b == 0:
            logger.error("Division by zero attempted: {0} / {1}".format(a, b))
            return 0.0

        cache_key = f"divide_{a}_{b}"
        if cache_key in self.operation_cache:
            return self.operation_cache[cache_key]

        result = a / b
        self._log_calculation("divide", result, {"a": a, "b": b})
        return result

    def power(self, base: float, exponent: float) -> float:
        """Calculate power with caching."""
        cache_key = f"power_{base}_{exponent}"
        if cache_key in self.operation_cache:
            return self.operation_cache[cache_key]

        result = xp.power(base, exponent)
        self._log_calculation("power", result, {"base": base, "exponent": exponent})
        return result

    def sqrt(self, value: float) -> float:
        """Calculate square root with caching and validation."""
        if value < 0:
            logger.warning("Negative value for sqrt: {0}".format(value))
            return 0.0

        cache_key = f"sqrt_{value}"
        if cache_key in self.operation_cache:
            return self.operation_cache[cache_key]

        result = xp.sqrt(value)
        self._log_calculation("sqrt", result, {"value": value})
        return result

    def exp(self, value: float) -> float:
        """Calculate exponential with caching."""
        cache_key = f"exp_{value}"
        if cache_key in self.operation_cache:
            return self.operation_cache[cache_key]

        result = xp.exp(value)
        self._log_calculation("exp", result, {"value": value})
        return result

    def sin(self, value: float) -> float:
        """Calculate sine with caching."""
        cache_key = f"sin_{value}"
        if cache_key in self.operation_cache:
            return self.operation_cache[cache_key]

        result = xp.sin(value)
        self._log_calculation("sin", result, {"value": value})
        return result

    def cos(self, value: float) -> float:
        """Calculate cosine with caching."""
        cache_key = f"cos_{value}"
        if cache_key in self.operation_cache:
            return self.operation_cache[cache_key]

        result = xp.cos(value)
        self._log_calculation("cos", result, {"value": value})
        return result

    def log(self, value: float, base: float = math.e) -> float:
        """Calculate logarithm with caching and validation."""
        if value <= 0:
            logger.warning("Non-positive value for log: {0}".format(value))
            return 0.0

        cache_key = f"log_{value}_{base}"
        if cache_key in self.operation_cache:
            return self.operation_cache[cache_key]

        result = xp.log(value) / xp.log(base)
        self._log_calculation("log", result, {"value": value, "base": base})
        return result

    def abs(self, value: float) -> float:
        """Calculate absolute value with caching."""
        cache_key = f"abs_{value}"
        if cache_key in self.operation_cache:
            return self.operation_cache[cache_key]

        result = xp.abs(value)
        self._log_calculation("abs", result, {"value": value})
        return result

    def min(self, values: List[float]) -> float:
        """Find minimum value with caching."""
        if not values:
            logger.warning("Empty list for min operation")
            return 0.0

        cache_key = f"min_{hash(tuple(values))}"
        if cache_key in self.operation_cache:
            return self.operation_cache[cache_key]

        result = xp.min(values)
        self._log_calculation("min", result, {"values": values})
        return result

    def max(self, values: List[float]) -> float:
        """Find maximum value with caching."""
        if not values:
            logger.warning("Empty list for max operation")
            return 0.0

        cache_key = f"max_{hash(tuple(values))}"
        if cache_key in self.operation_cache:
            return self.operation_cache[cache_key]

        result = xp.max(values)
        self._log_calculation("max", result, {"values": values})
        return result

    def mean(self, values: List[float]) -> float:
        """Calculate mean with caching."""
        if not values:
            logger.warning("Empty list for mean operation")
            return 0.0

        cache_key = f"mean_{hash(tuple(values))}"
        if cache_key in self.operation_cache:
            return self.operation_cache[cache_key]

        result = xp.mean(values)
        self._log_calculation("mean", result, {"values": values})
        return result

    def optimize_profit(self, base_profit: float, enhancement_factor: float, confidence: float) -> float:
        """Optimize profit using mathematical enhancement."""
        try:
            # Apply enhancement factor with confidence weighting
            enhanced_profit = base_profit * (1.0 + enhancement_factor * confidence)

            # Apply mathematical optimization
            optimized_profit = xp.tanh(enhanced_profit) * xp.abs(enhanced_profit)

            self._log_calculation(
                "optimize_profit",
                optimized_profit,
                {
                    "base_profit": base_profit,
                    "enhancement_factor": enhancement_factor,
                    "confidence": confidence,
                },
            )

            return optimized_profit
        except Exception as e:
            logger.error("Error in profit optimization: {0}".format(e))
            return base_profit

    def calculate_risk_adjustment(self, profit: float, volatility: float, confidence: float) -> float:
        """Calculate risk-adjusted profit."""
        try:
            # Risk adjustment formula
            risk_factor = 1.0 - (volatility * (1.0 - confidence))
            adjusted_profit = profit * xp.clip(risk_factor, 0.1, 2.0)

            self._log_calculation(
                "risk_adjustment",
                adjusted_profit,
                {"profit": profit, "volatility": volatility, "confidence": confidence},
            )

            return adjusted_profit
        except Exception as e:
            logger.error("Error in risk adjustment: {0}".format(e))
            return profit

    def calculate_portfolio_weight(self, confidence: float, max_risk: float) -> float:
        """Calculate portfolio weight based on confidence and risk."""
        try:
            # Weight calculation using sigmoid function
            weight = 1.0 / (1.0 + xp.exp(-10 * (confidence - 0.5)))
            risk_adjusted_weight = weight * (1.0 - max_risk)

            self._log_calculation(
                "portfolio_weight",
                risk_adjusted_weight,
                {"confidence": confidence, "max_risk": max_risk},
            )

            return xp.clip(risk_adjusted_weight, 0.0, 1.0)
        except Exception as e:
            logger.error("Error in portfolio weight calculation: {0}".format(e))
            return 0.5

    def calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.2) -> float:
        """Calculate Sharpe ratio for a series of returns."""
        try:
            if len(returns) < 2:
                logger.warning("Insufficient data for Sharpe ratio calculation")
                return 0.0

            returns_array = xp.array(returns)
            mean_return = xp.mean(returns_array)
            std_dev = xp.std(returns_array)

            if std_dev == 0:
                logger.warning("Zero standard deviation for Sharpe ratio")
                return 0.0

            sharpe_ratio = (mean_return - risk_free_rate) / std_dev

            self._log_calculation(
                "sharpe_ratio",
                sharpe_ratio,
                {
                    "returns": returns,
                    "risk_free_rate": risk_free_rate,
                    "mean_return": mean_return,
                    "std_dev": std_dev,
                },
            )

            return sharpe_ratio
        except Exception as e:
            logger.error("Error calculating Sharpe ratio: {0}".format(e))
            return 0.0

    def integrate_all_systems(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate all mathematical systems for comprehensive analysis."""
        try:
            results = {}

            # Extract key parameters
            base_profit = input_data.get("base_profit", 0.0)
            enhancement_factor = input_data.get("enhancement_factor", 1.0)
            confidence = input_data.get("confidence", 0.5)
            volatility = input_data.get("volatility", 0.1)
            max_risk = input_data.get("max_risk", 0.2)
            returns = input_data.get("returns", [])

            # Calculate integrated metrics
            optimized_profit = self.optimize_profit(base_profit, enhancement_factor, confidence)
            risk_adjusted_profit = self.calculate_risk_adjustment(optimized_profit, volatility, confidence)
            portfolio_weight = self.calculate_portfolio_weight(confidence, max_risk)
            sharpe_ratio = self.calculate_sharpe_ratio(returns)

            # Compile results
            results = {
                "optimized_profit": optimized_profit,
                "risk_adjusted_profit": risk_adjusted_profit,
                "portfolio_weight": portfolio_weight,
                "sharpe_ratio": sharpe_ratio,
                "confidence_score": confidence,
                "risk_score": volatility,
                "enhancement_applied": enhancement_factor,
            }

            self._log_calculation("system_integration", results, input_data)
            return results

        except Exception as e:
            logger.error("Error in system integration: {0}".format(e))
            return {"error": str(e)}

    def _log_calculation(self, operation: str, result: float, metadata: Dict[str, Any]) -> None:
        """Log a calculation for debugging and analysis."""
        math_result = MathResult(
            value=result,
            operation=operation,
            timestamp=time.time(),
            metadata=metadata,
        )
        self.calculation_history.append(math_result)
        # Cache the result for potential reuse
        cache_key = "{0}_{1}".format(operation, hash(str(metadata)))
        self.operation_cache[cache_key] = result

    def get_calculation_history(self) -> List[MathResult]:
        """Get calculation history."""
        return self.calculation_history.copy()

    def clear_cache(self) -> None:
        """Clear the operation cache."""
        self.operation_cache.clear()
        logger.info("Operation cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the operation cache."""
        return {
            "cache_size": len(self.operation_cache),
            "history_size": len(self.calculation_history),
            "cache_keys": list(self.operation_cache.keys()),
        }

    def get_calculation_summary(self) -> Dict[str, Any]:
        """Get summary of recent calculations."""
        try:
            # Count operations by type
            operation_counts = {}
            for calc in self.calculation_history:
                op = calc.operation
                operation_counts[op] = operation_counts.get(op, 0) + 1

            # Get recent calculations
            recent = self.calculation_history[-10:] if self.calculation_history else []
            return {
                "total_calculations": len(self.calculation_history),
                "operation_counts": operation_counts,
                "recent_operations": [calc.operation for calc in recent],
                "last_calculation_time": (self.calculation_history[-1].timestamp if self.calculation_history else 0),
            }
        except Exception as e:
            logger.error("Calculation summary error: {0}".format(e))
            return {"error": str(e)}


def optimize_brain_profit(price: float, volume: float, confidence: float, enhancement_factor: float) -> float:
    """Optimize brain profit using unified math system."""
    try:
        math_system = CleanUnifiedMathSystem()

        # Calculate base profit
        base_profit = price * volume * 0.01  # 1% base profit

        # Apply optimization
        optimized_profit = math_system.optimize_profit(base_profit, enhancement_factor, confidence)

        return optimized_profit
    except Exception as e:
        logger.error("Error in brain profit optimization: {0}".format(e))
        return 0.0


def calculate_position_size(confidence: float, portfolio_value: float, max_risk_percent: float) -> float:
    """Calculate position size based on confidence and risk parameters."""
    try:
        math_system = CleanUnifiedMathSystem()

        # Convert percentage to decimal
        max_risk = max_risk_percent / 100.0

        # Calculate weight
        weight = math_system.calculate_portfolio_weight(confidence, max_risk)

        # Calculate position size
        position_size = portfolio_value * weight

        return position_size
    except Exception as e:
        logger.error("Error in position size calculation: {0}".format(e))
        return 0.0


def test_clean_unified_math_system():
    """Test the clean unified math system."""
    print("=== Testing Clean Unified Math System ===")

    math_system = CleanUnifiedMathSystem()

    # Test basic operations
    print("Testing basic operations...")
    assert math_system.add(2, 3) == 5
    assert math_system.multiply(4, 5) == 20
    assert math_system.subtract(10, 3) == 7
    assert math_system.divide(15, 3) == 5

    # Test advanced operations
    print("Testing advanced operations...")
    assert math_system.power(2, 3) == 8
    assert abs(math_system.sqrt(16) - 4) < 0.001
    assert abs(math_system.exp(1) - 2.718) < 0.1

    # Test profit optimization
    print("Testing profit optimization...")
    optimized = math_system.optimize_profit(100.0, 0.5, 0.8)
    print("Optimized profit: {0}".format(optimized))

    # Test system integration
    print("Testing system integration...")
    input_data = {
        "base_profit": 100.0,
        "enhancement_factor": 0.5,
        "confidence": 0.8,
        "volatility": 0.1,
        "max_risk": 0.2,
        "returns": [0.01, 0.02, -0.01, 0.03, 0.01],
    }

    results = math_system.integrate_all_systems(input_data)
    print("Integration results: {0}".format(results))

    # Get summary
    summary = math_system.get_calculation_summary()
    print("\nCalculation Summary:")
    print("  Total calculations: {0}".format(summary['total_calculations']))
    print("Operation counts: {0}".format(summary.get('operation_counts', {})))
    print(" Clean Unified Math System test completed")


if __name__ == "__main__":
    test_clean_unified_math_system()

# Create a global instance for easy access
clean_unified_math = CleanUnifiedMathSystem()


# Export the function for backward compatibility
def clean_unified_math_function():
    """Return the global clean unified math instance."""
    return clean_unified_math
