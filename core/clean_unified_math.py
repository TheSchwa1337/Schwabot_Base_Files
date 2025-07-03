"""Clean Unified Mathematics System for Schwabot."""

import logging
import math
import time
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

# -*- coding: utf-8 -*-

# Clean Unified Mathematics System for Schwabot
# ============================================
#
# Clean mathematical framework that integrates with the brain trading system.
# Provides mathematical operations, optimization algorithms, and integration bridges.

# Configure logging
logger = logging.getLogger(__name__)


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
        self.operation_cache: Dict[str, Any] = {}
        self.calculation_history: List[MathResult] = []

    def multiply(self, a: Union[float, int], b: Union[float, int]) -> float:
        """Multiply two numbers."""
        try:
            result = float(a) * float(b)
            self._log_calculation("multiply", result, {"a": a, "b": b})
            return result
        except Exception as e:
            logger.error(f"Multiplication error: {e}")
            return 0.0

    def add(self, a: Union[float, int], b: Union[float, int]) -> float:
        """Add two numbers."""
        try:
            result = float(a) + float(b)
            self._log_calculation("add", result, {"a": a, "b": b})
            return result
        except Exception as e:
            logger.error(f"Addition error: {e}")
            return 0.0

    def subtract(self, a: Union[float, int], b: Union[float, int]) -> float:
        """Subtract two numbers."""
        try:
            result = float(a) - float(b)
            self._log_calculation("subtract", result, {"a": a, "b": b})
            return result
        except Exception as e:
            logger.error(f"Subtraction error: {e}")
            return 0.0

    def divide(self, a: Union[float, int], b: Union[float, int]) -> float:
        """Divide two numbers."""
        try:
            if b == 0:
                logger.warning("Division by zero, returning 0")
                return 0.0
            result = float(a) / float(b)
            self._log_calculation("divide", result, {"a": a, "b": b})
            return result
        except Exception as e:
            logger.error(f"Division error: {e}")
            return 0.0

    def power(self, base: Union[float, int], exponent: Union[float, int]) -> float:
        """Raise base to the power of exponent."""
        try:
            result = float(base) ** float(exponent)
            self._log_calculation("power", result, {"base": base, "exponent": exponent})
            return result
        except Exception as e:
            logger.error(f"Power calculation error: {e}")
            return 0.0

    def sqrt(self, value: Union[float, int]) -> float:
        """Calculate square root."""
        try:
            if value < 0:
                logger.warning("Square root of negative number, returning 0")
                return 0.0
            result = math.sqrt(float(value))
            self._log_calculation("sqrt", result, {"value": value})
            return result
        except Exception as e:
            logger.error(f"Square root error: {e}")
            return 0.0

    def exp(self, value: Union[float, int]) -> float:
        """Calculate exponential (e^x)."""
        try:
            result = math.exp(float(value))
            self._log_calculation("exp", result, {"value": value})
            return result
        except Exception as e:
            logger.error(f"Exponential error: {e}")
            return 1.0

    def sin(self, value: Union[float, int]) -> float:
        """Calculate sine."""
        try:
            result = math.sin(float(value))
            self._log_calculation("sin", result, {"value": value})
            return result
        except Exception as e:
            logger.error(f"Sine error: {e}")
            return 0.0

    def cos(self, value: Union[float, int]) -> float:
        """Calculate cosine."""
        try:
            result = math.cos(float(value))
            self._log_calculation("cos", result, {"value": value})
            return result
        except Exception as e:
            logger.error(f"Cosine error: {e}")
            return 1.0

    def log(self, value: Union[float, int], base: Union[float, int] = math.e) -> float:
        """Calculate logarithm."""
        try:
            if value <= 0:
                logger.warning("Logarithm of non-positive number, returning 0")
                return 0.0
            result = math.log(float(value), float(base))
            self._log_calculation("log", result, {"value": value, "base": base})
            return result
        except Exception as e:
            logger.error(f"Logarithm error: {e}")
            return 0.0

    def abs(self, value: Union[float, int]) -> float:
        """Calculate absolute value."""
        try:
            result = abs(float(value))
            self._log_calculation("abs", result, {"value": value})
            return result
        except Exception as e:
            logger.error(f"Absolute value error: {e}")
            return 0.0

    def min(self, *values) -> float:
        """Find minimum value."""
        try:
            if not values:
                return 0.0
            result = min(float(v) for v in values)
            self._log_calculation("min", result, {"values": values})
            return result
        except Exception as e:
            logger.error(f"Minimum calculation error: {e}")
            return 0.0

    def max(self, *values) -> float:
        """Find maximum value."""
        try:
            if not values:
                return 0.0
            result = max(float(v) for v in values)
            self._log_calculation("max", result, {"values": values})
            return result
        except Exception as e:
            logger.error(f"Maximum calculation error: {e}")
            return 0.0

    def mean(self, values: List[Union[float, int]]) -> float:
        """Calculate arithmetic mean."""
        try:
            if not values:
                return 0.0
            result = sum(float(v) for v in values) / len(values)
            self._log_calculation("mean", result, {"values": values})
            return result
        except Exception as e:
            logger.error(f"Mean calculation error: {e}")
            return 0.0

    def optimize_profit(
        self, base_profit: float, enhancement_factor: float, confidence: float
    ) -> float:
        """Optimize profit calculation using mathematical enhancement."""
        try:
            # Mathematical optimization using multiple factors
            confidence_boost = self.power(confidence, 1.5)  # Exponential confidence scaling
            enhancement_effect = self.multiply(enhancement_factor, 1.2)  # 20% enhancement bonus

            # Combined optimization
            optimized = self.multiply(
                base_profit, self.multiply(confidence_boost, enhancement_effect)
            )

            # Apply mathematical smoothing
            smoothed = self.multiply(optimized, 0.95)  # 5% smoothing factor

            self._log_calculation(
                "optimize_profit",
                smoothed,
                {
                    "base_profit": base_profit,
                    "enhancement_factor": enhancement_factor,
                    "confidence": confidence,
                },
            )

            return smoothed

        except Exception as e:
            logger.error(f"Profit optimization error: {e}")
            return base_profit

    def calculate_risk_adjustment(
        self, profit: float, volatility: float, confidence: float
    ) -> float:
        """Calculate risk-adjusted profit score."""
        try:
            # Risk adjustment based on volatility and confidence
            risk_factor = self.subtract(1.0, self.multiply(volatility, 0.5))
            confidence_factor = self.add(confidence, 0.1)  # Minimum confidence boost

            # Apply risk adjustment
            adjusted_profit = self.multiply(
                profit, self.multiply(risk_factor, confidence_factor)
            )

            self._log_calculation(
                "risk_adjustment",
                adjusted_profit,
                {"profit": profit, "volatility": volatility, "confidence": confidence},
            )

            return adjusted_profit

        except Exception as e:
            logger.error(f"Risk adjustment error: {e}")
            return profit

    def calculate_portfolio_weight(self, confidence: float, max_weight: float = 0.1) -> float:
        """Calculate portfolio weight based on confidence."""
        try:
            # Weight calculation using confidence scaling
            base_weight = self.multiply(confidence, max_weight)

            # Apply mathematical curve for better distribution
            curved_weight = self.multiply(
                base_weight, self.power(confidence, 0.5)
            )  # Square root curve

            self._log_calculation(
                "portfolio_weight",
                curved_weight,
                {"confidence": confidence, "max_weight": max_weight},
            )

            return curved_weight

        except Exception as e:
            logger.error(f"Portfolio weight calculation error: {e}")
            return 0.0

    def calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio for risk-adjusted performance."""
        try:
            if not returns or len(returns) < 2:
                return 0.0

            # Calculate excess returns
            excess_returns = [self.subtract(r, risk_free_rate) for r in returns]

            # Calculate mean and standard deviation
            mean_excess = self.mean(excess_returns)

            # Calculate standard deviation manually
            variance_sum = sum(self.power(self.subtract(r, mean_excess), 2) for r in excess_returns)
            variance = self.divide(variance_sum, len(excess_returns) - 1)
            std_dev = self.sqrt(variance)

            # Calculate Sharpe ratio
            if std_dev == 0:
                return 0.0

            sharpe = self.divide(mean_excess, std_dev)

            self._log_calculation(
                "sharpe_ratio",
                sharpe,
                {
                    "returns_count": len(returns),
                    "mean_excess": mean_excess,
                    "std_dev": std_dev,
                },
            )

            return sharpe

        except Exception as e:
            logger.error(f"Sharpe ratio calculation error: {e}")
            return 0.0

    def integrate_all_systems(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Main integration function for all mathematical systems."""
        try:
            results = {}

            # Extract input data
            tensor_data = input_data.get("tensor", [[50000, 1000]])
            metadata = input_data.get("metadata", {})

            # Perform mathematical calculations
            if tensor_data:
                # Simple processing of tensor data
                if isinstance(tensor_data, list) and tensor_data:
                    first_row = tensor_data[0] if tensor_data[0] else [0, 0]
                    if len(first_row) >= 2:
                        price, volume = first_row[0], first_row[1]

                        # Calculate basic metrics
                        momentum = self.multiply(price, 0.0001)  # Simple momentum
                        volume_factor = self.sqrt(volume)
                        combined_score = self.add(momentum, volume_factor)

                        results["momentum"] = momentum
                        results["volume_factor"] = volume_factor
                        results["combined_score"] = combined_score

            # Add system metadata
            results["timestamp"] = time.time()
            results["input_metadata"] = metadata
            results["calculation_count"] = len(self.calculation_history)

            return results

        except Exception as e:
            logger.error(f"System integration error: {e}")
            import time
            return {"error": str(e), "timestamp": time.time()}

    def _log_calculation(self, operation: str, result: Any, metadata: Dict[str, Any]) -> None:
        """Log a calculation for debugging and analysis."""
        import time
        math_result = MathResult(
            value=result,
            operation=operation,
            timestamp=time.time(),
            metadata=metadata,
        )
        self.calculation_history.append(math_result)

        # Cache the result for potential reuse
        cache_key = f"{operation}_{hash(str(metadata))}"
        self.operation_cache[cache_key] = result

        logger.debug(f"Math operation '{operation}' completed: {result}")

    def get_calculation_history(self) -> List[MathResult]:
        """Get the history of all calculations performed."""
        return self.calculation_history.copy()

    def clear_cache(self) -> None:
        """Clear the operation cache."""
        self.operation_cache.clear()
        logger.info("Math operation cache cleared")

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
            if not self.calculation_history:
                return {"total_calculations": 0}

            # Count operations
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
                "last_calculation_time": (
                    self.calculation_history[-1].timestamp if self.calculation_history else 0
                ),
            }

        except Exception as e:
            logger.error(f"Calculation summary error: {e}")
            return {"error": str(e)}


# Global instance for easy access
clean_unified_math = CleanUnifiedMathSystem()


def optimize_brain_profit(
    price: float, volume: float, confidence: float, enhancement_factor: float = 1.0
) -> float:
    """Optimized profit calculation for brain trading signals.

    Args:
        price: Asset price
        volume: Trading volume
        confidence: Signal confidence (0-1)
        enhancement_factor: Brain enhancement factor

    Returns:
        Optimized profit score
    """
    try:
        # Base profit calculation
        base_profit = clean_unified_math.multiply(price, volume) * 0.001  # 0.1% base

        # Apply brain optimization
        optimized_profit = clean_unified_math.optimize_profit(
            base_profit, enhancement_factor, confidence
        )

        # Apply risk adjustment based on volatility estimation
        volatility = clean_unified_math.min(
            0.5, clean_unified_math.divide(abs(price - 50000), 50000)
        )
        risk_adjusted = clean_unified_math.calculate_risk_adjustment(
            optimized_profit, volatility, confidence
        )

        return risk_adjusted

    except Exception as e:
        logger.error(f"Brain profit optimization error: {e}")
        return 0.0


def calculate_position_size(
    confidence: float, portfolio_value: float, max_risk_percent: float = 0.1
) -> float:
    """Calculate position size based on confidence and risk management.

    Args:
        confidence: Signal confidence (0-1)
        portfolio_value: Total portfolio value
        max_risk_percent: Maximum risk percentage (0-1)

    Returns:
        Position size in dollars
    """
    try:
        # Calculate maximum position based on risk
        max_position = clean_unified_math.multiply(portfolio_value, max_risk_percent)

        # Calculate confidence-based weight
        weight = clean_unified_math.calculate_portfolio_weight(confidence, max_risk_percent)

        # Calculate final position size
        position_size = clean_unified_math.multiply(portfolio_value, weight)

        # Ensure within maximum risk bounds
        final_size = clean_unified_math.min(position_size, max_position)

        return final_size

    except Exception as e:
        logger.error(f"Position size calculation error: {e}")
        return 0.0


def test_clean_unified_math_system():
    """Test the clean unified math system functionality."""
    print("🧮 Testing Clean Unified Math System")
    print("=" * 40)

    # Test basic operations
    print("Basic Operations:")
    print(f"5 + 3 = {clean_unified_math.add(5, 3)}")
    print(f"  10 * 2.5 = {clean_unified_math.multiply(10, 2.5)}")
    print(f"  100 / 4 = {clean_unified_math.divide(100, 4)}")
    print(f"  sqrt(25) = {clean_unified_math.sqrt(25)}")

    # Test optimization functions
    print("\nOptimization Functions:")
    optimized = clean_unified_math.optimize_profit(1000, 1.5, 0.8)
    print(f"  Optimized profit: {optimized:.2f}")

    risk_adjusted = clean_unified_math.calculate_risk_adjustment(1000, 0.2, 0.7)
    print(f"  Risk adjusted: {risk_adjusted:.2f}")

    # Test brain profit optimization
    print("\nBrain Trading Integration:")
    brain_profit = optimize_brain_profit(50000, 1000, 0.75, 1.2)
    print(f"  Brain optimized profit: {brain_profit:.2f}")

    position_size = calculate_position_size(0.8, 100000, 0.1)
    print(f"  Position size: ${position_size:.2f}")

    # Test performance metrics
    returns = [0.05, 0.02, -0.01, 0.03, 0.01]
    sharpe = clean_unified_math.calculate_sharpe_ratio(returns)
    print(f"  Sharpe ratio: {sharpe:.3f}")

    # Test integration function
    input_data = {"tensor": [[50000, 1200], [49500, 1100]], "metadata": {"source": "test"}}
    integration_result = clean_unified_math.integrate_all_systems(input_data)
    print("\nIntegration Result:")
    print(f"Combined Score: {integration_result.get('combined_score', 0):.2f}")

    # Show calculation summary
    summary = clean_unified_math.get_calculation_summary()
    print("\nCalculation Summary:")
    print(f"  Total calculations: {summary['total_calculations']}")
    print(f"Operation counts: {summary.get('operation_counts', {})}")

    print("✅ Clean Unified Math System test completed")


if __name__ == "__main__":
    test_clean_unified_math_system()
