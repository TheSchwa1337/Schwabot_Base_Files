# -*- coding: utf-8 -*-
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from utils.safe_print import safe_print

from core.unified_math_system import unified_math

# from dual_unicore_handler import DualUnicoreHandler

# Initialize Unicode handler
# unicore = DualUnicoreHandler()

logger = logging.getLogger(__name__)


class LineRenderEngine:
    def __init__(self, max_points=1000):
        self.max_points = max_points

    def render_price_line(
        self, prices: List[float], timestamps: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """Renders a price line from a list of prices."""
        try:
            if not prices:
                return {"points": [], "error": "No price data"}

            # Limit points for performance
            if len(prices) > self.max_points:
                step = len(prices) // self.max_points
                prices = prices[::step]
                if timestamps:
                    timestamps = timestamps[::step]

            # Create line points
            if timestamps is None:
                timestamps = list(range(len(prices)))

            points = [(float(t), float(p)) for t, p in zip(timestamps, prices)]

            return {
                "points": points,
                "count": len(points),
                "min_price": unified_math.min(prices),
                "max_price": unified_math.max(prices),
                "type": "price_line",
            }
        except Exception as e:
            logger.error(f"Price line rendering failed: {e}")
            return {"points": [], "error": str(e)}

    def render_mathematical_function(
        self, func_values: List[float], x_range: Optional[Tuple[float, float]] = None
    ) -> Dict[str, Any]:
        """Renders a line from a list of mathematical function values."""
        try:
            if not func_values:
                return {"points": [], "error": "No function data"}

            # Create x values
            if x_range is None:
                x_values = np.linspace(0, 1, len(func_values))
            else:
                x_values = np.linspace(x_range[0], x_range[1], len(func_values))

            # Create line points
            points = [(float(x), float(y)) for x, y in zip(x_values, func_values)]

            return {
                "points": points,
                "count": len(points),
                "min_value": unified_math.min(func_values),
                "max_value": unified_math.max(func_values),
                "type": "function_line",
            }
        except Exception as e:
            logger.error(f"Function line rendering failed: {e}")
            return {"points": [], "error": str(e)}

    def render_tensor_visualization(self, tensor_data: np.ndarray) -> Dict[str, Any]:
        """Renders a line visualization of a tensor."""
        try:
            if tensor_data.size == 0:
                return {"lines": [], "error": "No tensor data"}

            # Flatten tensor for line representation
            flat_data = tensor_data.flatten()

            # Create line representation
            x_values = np.arange(len(flat_data))
            points = [(float(x), float(y)) for x, y in zip(x_values, flat_data)]

            return {
                "points": points,
                "count": len(points),
                "shape": tensor_data.shape,
                "min_value": float(unified_math.min(tensor_data.flatten())),
                "max_value": float(unified_math.max(tensor_data.flatten())),
                "type": "tensor_line",
            }
        except Exception as e:
            logger.error(f"Tensor visualization failed: {e}")
            return {"points": [], "error": str(e)}


# Global instance for easy import
line_renderer = LineRenderEngine()


# Convenience functions for main pipeline
def render_price_line(
    prices: List[float], timestamps: Optional[List[float]] = None
) -> Dict[str, Any]:
    """Convenience function to render a price line."""
    return line_renderer.render_price_line(prices, timestamps)


def render_mathematical_function(
    func_values: List[float], x_range: Optional[Tuple[float, float]] = None
) -> Dict[str, Any]:
    """Convenience function to render a mathematical function line."""
    return line_renderer.render_mathematical_function(func_values, x_range)


def main():
    """Main function for testing the LineRenderEngine."""
    safe_print("Line Render Engine - Mathematical Visualization System")

    # Test price line rendering
    test_prices = [100.0, 101.5, 99.8, 102.3, 98.7]
    price_result = render_price_line(test_prices)
    safe_print(f"Price line rendered: {price_result.get('count', 0)} points")

    # Test mathematical function rendering
    test_function = [unified_math.sin(x / 10) for x in range(100)]
    func_result = render_mathematical_function(test_function)
    safe_print(f"Function line rendered: {func_result.get('count', 0)} points")

    safe_print("Line Render Engine test completed successfully")


if __name__ == "__main__":
    main()
